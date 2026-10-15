"""Exact distributed top-k collection of per-step sampling distributions.

Records every decode step's exact top-k head without materializing the full
[batch, seq, vocab] history. Steps accumulate in a [batch, CHUNK, vocab] ring
buffer, then one batched selection per chunk compresses them into the head
buffers. Total ring memory is CHUNK/seq of the full array.

Each vocabulary shard selects its exact local top-k; a second exact top-k over
the concatenated shard candidates gives the exact global result. The
transported payload remains [batch, seq, k].

Carry-based API for jitted decode loops:

    carry = init(batch_size, seq_len, vocab_size, k)
    carry = observe(carry, position, logprobs)     # every step
    head_logprobs, head_ids = finalize(carry, position, seq_len)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

CHUNK = 32    # steps buffered between batched compressions


def take_vocab(vocab, ids):
    """take_along_axis on [..., V] whose vocab axis may be TP-sharded: the
    gather output drops that axis, so its sharding must be explicit."""
    spec = jax.typeof(vocab).sharding.spec
    if spec[-1] is None:
        return jnp.take_along_axis(vocab, ids, -1)
    leading = tuple(i[..., None] for i in jnp.indices(ids.shape[:-1]))
    return vocab.at[(*leading, ids)].get(out_sharding=P(*spec[:-1], None))


class Carry(NamedTuple):
    ring: jax.Array
    head_logprobs: jax.Array
    head_ids: jax.Array


def init(batch_size, seq_len, vocab_size, k):
    rows = P(("data", "model"), None, None)
    # Head buffers are padded to a chunk multiple so the final partial-chunk
    # compress can never be clamped (and misaligned) by dynamic_update_slice.
    padded = -(-seq_len // CHUNK) * CHUNK
    return Carry(
        ring=jnp.zeros(
            (batch_size, CHUNK, vocab_size), jnp.float32,
            out_sharding=P("data", None, "model")),
        head_logprobs=jnp.zeros(
            (batch_size, padded, k), jnp.float32, out_sharding=rows),
        head_ids=jnp.zeros(
            (batch_size, padded, k), jnp.int32, out_sharding=rows),
    )


def observe(carry, position, logprobs):
    ring = carry.ring.at[:, position % CHUNK].set(logprobs)
    head_logprobs, head_ids = jax.lax.cond(
        (position + 1) % CHUNK == 0,
        lambda: _compress(
            ring, carry.head_logprobs, carry.head_ids, position + 1 - CHUNK),
        lambda: (carry.head_logprobs, carry.head_ids),
    )
    return Carry(ring, head_logprobs, head_ids)


def finalize(carry, position, seq_len):
    """Compress the final partial chunk and trim the padding. Rows past an
    early stop hold stale values; consumers mask them."""
    head_logprobs, head_ids = _compress(
        carry.ring, carry.head_logprobs, carry.head_ids,
        position // CHUNK * CHUNK)
    return head_logprobs[:, :seq_len], head_ids[:, :seq_len]


def select(block, k):
    """Exact global top-k of [B, n, V] over a possibly TP-sharded vocab axis,
    returned as [B, n, k] rows spread over every device."""
    def local_topk(x):
        values, ids = jax.lax.top_k(x, k, is_stable=False)
        return values, ids + jax.lax.axis_index("model") * x.shape[-1]

    values, ids = jax.shard_map(
        local_topk, in_specs=P("data", None, "model"),
        out_specs=(P("data", None, "model"), P("data", None, "model")))(block)
    rows = P(("data", "model"), None, None)
    values, ids = reshard(values, rows), reshard(ids, rows)
    top_values, idx = jax.lax.top_k(values, k, is_stable=False)
    return top_values, jnp.take_along_axis(ids, idx, axis=-1)


def _compress(ring, head_logprobs, head_ids, start):
    top_logprobs, top_ids = select(ring, head_logprobs.shape[-1])
    return (
        jax.lax.dynamic_update_slice(head_logprobs, top_logprobs, (0, start, 0)),
        jax.lax.dynamic_update_slice(head_ids, top_ids, (0, start, 0)),
    )


class TailCarry(NamedTuple):
    """Head/tail payload collected per decode step: the exact top-H head and
    M tokens drawn from q with the head masked out, all with their logprobs
    under the full sampling distribution."""
    head_logprobs: jax.Array
    head_ids: jax.Array
    tail_ids: jax.Array
    tail_logprobs: jax.Array


def init_tail(batch_size, seq_len, head_k, num_tail):
    rows = P(("data", "model"), None, None)
    buffer = lambda k, dtype: jnp.zeros(
        (batch_size, seq_len, k), dtype, out_sharding=rows)
    return TailCarry(
        head_logprobs=buffer(head_k, jnp.float32),
        head_ids=buffer(head_k, jnp.int32),
        tail_ids=buffer(num_tail, jnp.int32),
        tail_logprobs=buffer(num_tail, jnp.float32),
    )


def observe_tail(carry, position, key, logits, logprobs):
    """Record this step's exact head and draw the tail. The head is selected
    per step (no ring) because the draws need it as a mask right away."""
    head_logprobs, head_ids = select(logprobs[:, None, :], carry.head_ids.shape[-1])
    head_logprobs, head_ids = head_logprobs[:, 0], head_ids[:, 0]
    # Mask the head by id, not by a logprob threshold: bf16 logits tie across
    # the vocabulary, and a threshold would also remove tied non-head tokens.
    row_index = jnp.arange(logits.shape[0], out_sharding=P("data"))[:, None]
    masked = logits.at[row_index, reshard(head_ids, P("data", None))].set(
        -jnp.inf, out_sharding=jax.typeof(logits).sharding.spec)
    num_tail = carry.tail_ids.shape[-1]
    draws = jnp.stack([
        jax.random.categorical(draw_key, masked)
        for draw_key in jax.random.split(key, num_tail)], axis=-1)
    tail_logprobs = take_vocab(logprobs, draws)
    rows = P(("data", "model"), None)
    return TailCarry(
        head_logprobs=carry.head_logprobs.at[:, position].set(head_logprobs),
        head_ids=carry.head_ids.at[:, position].set(head_ids),
        tail_ids=carry.tail_ids.at[:, position].set(reshard(draws, rows)),
        tail_logprobs=carry.tail_logprobs.at[:, position].set(
            reshard(tail_logprobs, rows)),
    )
