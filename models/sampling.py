"""Fixed-batch autoregressive token sampling."""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

from models import topk


class SamplingState(NamedTuple):
    position: int
    key: jax.Array
    tokens: jax.Array
    kv: list
    done: jax.Array
    ended: jax.Array       # the row emitted EOS (as opposed to hitting a cap)
    generated: jax.Array   # tokens drawn per row
    token_logprobs: jax.Array
    vocab_logprobs: jax.Array | None
    head: topk.Carry | None
    tail: topk.TailCarry | None


class Resume(NamedTuple):
    """What a later stage of a staged decode needs from an earlier one: the
    KV cache of every position decoded so far, the random key where the
    draws stopped, and the rows that emitted EOS. Positions before the
    smallest resume position keep the cache the weights of their own stage
    produced, as inference engines with in-flight weight updates do."""
    kv: object
    key: jax.Array
    ended: jax.Array


class Generation(NamedTuple):
    """One filled token batch and the sampler distributions recorded with it.
    Payload fields are None unless requested; rows past an early stop hold
    zeros or stale values, and consumers mask them. `generated` counts the
    tokens drawn per row and `ended` says whether the row emitted EOS; token
    ids cannot tell these apart when the pad and EOS ids coincide."""
    tokens: jax.Array
    token_logprobs: jax.Array
    vocab_logprobs: jax.Array | None
    head_logprobs: jax.Array | None
    head_ids: jax.Array | None
    tail_ids: jax.Array | None
    tail_logprobs: jax.Array | None
    generated: jax.Array
    ended: jax.Array


def _step(state, forward, weights, prompt_lengths, stop_ids, max_new_tokens):
    # The draw at a position depends on the key and the position only, so a
    # decode cut into stages draws the same tokens as one uncut decode.
    sample_key, tail_key = jax.random.split(jax.random.fold_in(state.key, state.position))
    logits, kv = forward(
        state.tokens[:, state.position, None],
        weights,
        state.kv,
        reshard(
            jnp.full((state.tokens.shape[0], 1), state.position),
            P("data", None),
        ),
        # Finished slots decode padding: mask them out of the MoE dispatch so
        # they cost no expert flops or gather traffic.
        live=~state.done[:, None],
    )
    logits = logits[:, 0].astype(jnp.float32)
    logprobs = jax.nn.log_softmax(logits)
    sampled = jax.random.categorical(sample_key, logits)
    vocab_logprobs = state.vocab_logprobs
    if vocab_logprobs is not None:
        vocab_logprobs = vocab_logprobs.at[:, state.position].set(
            logprobs.astype(vocab_logprobs.dtype))
    head = state.head
    if head is not None:
        head = topk.observe(head, state.position, logprobs)
    tail = state.tail
    if tail is not None:
        tail = topk.observe_tail(tail, state.position, tail_key, logits, logprobs)

    # On a mesh whose data axis has size 1 (tp over every chip) the gather
    # drops the data sharding, and the select below rejects a P(None) case
    # next to the P("data") draw.
    buffered_token = reshard(state.tokens[:, state.position + 1], P("data"))
    sampling = (~state.done) & (state.position + 1 >= prompt_lengths)
    next_token = jnp.where(sampling, sampled, buffered_token)
    tokens = state.tokens.at[:, state.position + 1].set(next_token)
    targets = reshard(
        jax.nn.one_hot(next_token, logits.shape[-1], dtype=jnp.float32),
        jax.typeof(logprobs).sharding.spec,
    )
    token_logprobs = state.token_logprobs.at[:, state.position].set(
        jnp.where(state.done, 0.0, (targets * logprobs).sum(-1)))
    ended = state.ended | (sampling & (sampled[:, None] == stop_ids).any(-1))
    generated = state.generated + sampling.astype(jnp.int32)
    done = state.done | ended
    if max_new_tokens is not None:
        done = done | (sampling & (generated >= max_new_tokens))
    return SamplingState(
        state.position + 1, state.key, tokens, kv, done, ended, generated,
        token_logprobs, vocab_logprobs, head, tail)


@partial(jax.jit, static_argnames=(
    "forward", "init_kv", "vocab_size", "logprobs_dtype", "logprobs_topk",
    "tail_samples", "max_new_tokens", "skip_full_rows"))
def _generate(key, forward, init_kv, weights, tokens,
              prompt_lengths, stop_ids, vocab_size, logprobs_dtype,
              logprobs_topk, tail_samples, max_new_tokens, skip_full_rows,
              resume):
    batch_size, seq_len = tokens.shape
    if resume is None:
        kv, position = init_kv(batch_size, seq_len), 0
        ended = jnp.zeros(batch_size, bool, out_sharding=P("data"))
    else:
        # The cache holds every position below each row's resume position
        # minus one; the loop restarts at the smallest of those and
        # teacher-forces the other rows' already drawn tokens up to theirs.
        kv, key, ended = resume
        position = jnp.maximum(prompt_lengths.min() - 1, 0)
    state = SamplingState(
        position=position,
        key=key,
        tokens=reshard(tokens, P("data", None)),
        kv=kv,
        # A row whose prompt fills the buffer has nothing to draw; it still
        # gets scored (teacher forcing) unless the caller opts out, in which
        # case it must not keep the loop alive for the other rows.
        done=ended | (prompt_lengths >= seq_len if skip_full_rows
                      else jnp.zeros(batch_size, bool, out_sharding=P("data"))),
        ended=ended,
        generated=jnp.zeros(batch_size, jnp.int32, out_sharding=P("data")),
        token_logprobs=jnp.zeros(
            (batch_size, seq_len), jnp.float32,
            out_sharding=P("data", None)),
        vocab_logprobs=(
            None if logprobs_dtype is None else jnp.zeros(
                (batch_size, seq_len, vocab_size),
                logprobs_dtype,
                out_sharding=P("data", None, "model"),
            )
        ),
        head=(
            topk.init(batch_size, seq_len, vocab_size, logprobs_topk)
            if logprobs_topk and not tail_samples else None
        ),
        tail=(
            topk.init_tail(batch_size, seq_len, logprobs_topk, tail_samples)
            if tail_samples else None
        ),
    )
    state = jax.lax.while_loop(
        lambda state: (state.position + 1 < seq_len) & (~state.done).any(),
        lambda state: _step(
            state, forward, weights, prompt_lengths, stop_ids, max_new_tokens),
        state,
    )
    head_logprobs = head_ids = tail_ids = tail_logprobs = None
    if state.head is not None:
        head_logprobs, head_ids = topk.finalize(
            state.head, state.position, seq_len)
    if state.tail is not None:
        head_logprobs, head_ids, tail_ids, tail_logprobs = state.tail
    return Generation(
        state.tokens, state.token_logprobs, state.vocab_logprobs,
        head_logprobs, head_ids, tail_ids, tail_logprobs,
        state.generated, state.ended), Resume(state.kv, state.key, state.ended)


def generate(key, model, tokens, prompt_lengths, *, weights=None, forward=None,
             init_kv=None, eos_id=None, logprobs_dtype=None, logprobs_topk=0,
             tail_samples=0, max_new_tokens=None, skip_full_rows=False,
             resume=None, return_resume=False):
    """Fill one fixed token batch and record the distributions used.

    logprobs_dtype keeps the full [B, T, V] distribution; logprobs_topk
    keeps only the per-step top-k head (ids + logprobs) instead, which is the
    information score centering consumes, at about k/V of the memory. tail_samples
    additionally draws that many tokens per step from q with the head masked
    out, giving the head/tail payload for stratified estimators.
    resume (a Resume from an earlier stage, with return_resume=True) carries
    the KV cache, the random key and the ended rows into a further stage
    whose prompt_lengths are the rows' resume positions; the cache makes a
    stage cost its new tokens rather than the whole prefix.

    max_new_tokens stops every row after that many generated tokens, which
    lets a rollout be continued later under other weights. skip_full_rows
    starts rows whose prompt fills the buffer as done, so a continuation
    batch ends when its live rows end; by default such rows are still scored.
    eos_id is one token id or a sequence of them, any of which ends a row;
    by default the model's stop ids (GLM lists several in its config).
    """
    if tail_samples and not logprobs_topk:
        raise ValueError("tail_samples requires a top-k head (logprobs_topk > 0)")
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    weights = model.weights if weights is None else weights
    forward = model.forward if forward is None else forward
    init_kv = model.init_kv if init_kv is None else init_kv
    stop_ids = model.stop_ids if eos_id is None else eos_id
    stop_ids = reshard(jnp.atleast_1d(jnp.asarray(stop_ids, jnp.int32)), P())
    # A resumed stage continues the earlier stage's draws from its key.
    key = reshard(resume.key if resume is not None else key, P())
    prompt_lengths = reshard(jnp.asarray(prompt_lengths, jnp.int32), P("data"))
    generation, next_resume = _generate(
        key, forward, init_kv, weights, tokens,
        prompt_lengths, stop_ids, model.config["vocab_size"], logprobs_dtype,
        logprobs_topk, tail_samples, max_new_tokens, skip_full_rows, resume)
    return (generation, next_resume) if return_resume else generation
