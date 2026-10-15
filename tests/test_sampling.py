"""Fixed-batch sampling preserves prompts, replays, stops at EOS, and ships
exact top-k heads with optional conditional tail draws."""
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

import models
from models import sampling


def main():
    checkpoint_dir = "~/.cache/postax/weights"
    model = models.load(
        "Qwen/Qwen3-0.6B-Base",
        checkpoint_dir,
        init="random",
        hidden_size=128,
        num_hidden_layers=2,
        key=jax.random.key(0),
    )
    sample_tokens = jnp.zeros((8, 2), jnp.int32)
    assert jax.eval_shape(
        partial(model.forward, dtype=jnp.bfloat16),
        sample_tokens,
        model.weights,
    ).dtype == jnp.bfloat16
    assert jax.eval_shape(
        partial(model.forward, dtype=jnp.float32),
        sample_tokens,
        model.weights,
    ).dtype == jnp.float32
    assert model.init_kv(8, 2, dtype=jnp.float32)[0].dtype == jnp.float32

    rng = np.random.default_rng(0)
    prompt_lengths = np.asarray([3, 7, 12, 5, 9, 4, 15, 6])
    tokens = np.full(
        (len(prompt_lengths), 48),
        model.tokenizer.pad_token_id,
        np.int32,
    )
    for row, length in enumerate(prompt_lengths):
        tokens[row, :length] = rng.integers(10, 5000, length)
    if model.tokenizer.eos_token_id is not None:
        tokens[0, 1] = model.tokenizer.eos_token_id

    first = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1,
        logprobs_dtype=jnp.float32)
    replay = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1,
        logprobs_dtype=jnp.float32)
    first_tokens, replay_tokens = np.asarray(first.tokens), np.asarray(replay.tokens)
    assert np.array_equal(first_tokens, replay_tokens)
    assert first.vocab_logprobs.shape == (
        *tokens.shape, model.config["vocab_size"])
    assert first.vocab_logprobs.dtype == jnp.float32
    recorded = np.take_along_axis(
        np.asarray(first.vocab_logprobs[:, :-1]),
        first_tokens[:, 1:, None],
        axis=-1,
    )[..., 0]
    assert np.array_equal(recorded, np.asarray(first.token_logprobs[:, :-1]))
    assert np.array_equal(
        np.asarray(first.token_logprobs),
        np.asarray(replay.token_logprobs),
    )
    assert np.array_equal(
        np.asarray(first.vocab_logprobs[:, :1]),
        np.asarray(replay.vocab_logprobs[:, :1]),
    )
    for row, length in enumerate(prompt_lengths):
        assert np.array_equal(first_tokens[row, :length], tokens[row, :length])
    assert first.tail_ids is None and first.tail_logprobs is None

    # In-loop head collection must ship exact values at its ids and preserve
    # exact top-k mass through k=1024. Deterministic replay makes the full and
    # transported distributions directly comparable. Single device: the
    # whole vocabulary is one local block, the highest-collision deployment.
    head = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1,
        logprobs_topk=1024)
    full = np.asarray(first.vocab_logprobs)
    live = np.asarray(first.token_logprobs) != 0.0
    gathered = np.take_along_axis(full, np.asarray(head.head_ids), axis=-1)
    assert np.allclose(gathered[live], np.asarray(head.head_logprobs)[live])
    for k in (128, 1024):
        exact = np.partition(full, -k, axis=-1)[..., -k:]
        exact_mass = np.exp(exact).sum(-1)
        transported_mass = np.exp(np.asarray(head.head_logprobs)[..., :k]).sum(-1)
        assert np.allclose(
            transported_mass[live], exact_mass[live], rtol=1e-6, atol=1e-7
        ), f"k={k} transported head measurably loses probability mass"

    # Head/tail payload: an exact top-H head per step plus M tail tokens drawn
    # from q with the head masked out, each with its full-distribution logprob.
    tail = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1,
        logprobs_topk=4, tail_samples=3)
    assert np.array_equal(np.asarray(tail.tokens), first_tokens)
    assert tail.tail_ids.shape == (*tokens.shape, 3)
    assert tail.tail_logprobs.shape == (*tokens.shape, 3)
    head_ids = np.asarray(tail.head_ids)
    tail_ids = np.asarray(tail.tail_ids)
    # bf16 logits tie often across 150k tokens, so compare head values, not
    # ids, against the exact top-4 of the full distribution.
    exact_head = np.partition(full, -4, axis=-1)[..., -4:]
    assert np.allclose(
        np.sort(np.asarray(tail.head_logprobs)[live], -1),
        np.sort(exact_head[live], -1))
    assert np.allclose(
        np.take_along_axis(full, head_ids, -1)[live],
        np.asarray(tail.head_logprobs)[live])
    assert not np.any(
        (tail_ids[live][:, :, None] == head_ids[live][:, None, :]))
    assert np.allclose(
        np.take_along_axis(full, tail_ids, -1)[live],
        np.asarray(tail.tail_logprobs)[live])

    # The draws follow q conditioned outside the head: on a batch of identical
    # prompts the first generated position shares one distribution, and the
    # mean sampler logprob of the drawn tail tokens matches its conditional
    # expectation to within four standard errors.
    same = np.tile(tokens[1:2], (256, 1))
    same_lengths = np.full(256, prompt_lengths[1])
    same_gen = sampling.generate(
        jax.random.key(2), model, same, same_lengths, eos_id=-1,
        logprobs_dtype=jnp.float32, logprobs_topk=4, tail_samples=3)
    position = prompt_lengths[1] - 1
    logq = np.asarray(same_gen.vocab_logprobs)[0, position]
    q = np.exp(logq)
    is_tail = np.ones_like(q, bool)
    is_tail[np.asarray(same_gen.head_ids)[0, position]] = False
    tail_cond = q[is_tail] / q[is_tail].sum()
    expected_mean = (tail_cond * logq[is_tail]).sum()
    expected_std = np.sqrt((tail_cond * (logq[is_tail] - expected_mean) ** 2).sum())
    drawn = np.asarray(same_gen.tail_logprobs)[:, position].reshape(-1)
    standard_error = expected_std / np.sqrt(drawn.size)
    assert abs(drawn.mean() - expected_mean) < 4 * standard_error, (
        drawn.mean(), expected_mean, standard_error)

    # A per-row cap on new tokens: every row stops after exactly that many
    # generated tokens (or at EOS), and the rest of the row stays padding.
    capped = np.asarray(sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1,
        max_new_tokens=3).tokens)
    capped_gen = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1,
        max_new_tokens=3)
    for row, length in enumerate(prompt_lengths):
        assert np.all(capped[row, length:length + 3] != model.tokenizer.pad_token_id)
        assert np.all(capped[row, length + 3:] == model.tokenizer.pad_token_id)
        # Same key, same first three tokens as the uncapped generation.
        assert np.array_equal(capped[row, :length + 3], first_tokens[row, :length + 3])
    # The sampler reports how many tokens it drew per row and whether the row
    # emitted EOS, which token ids cannot tell apart when pad and EOS share an id.
    assert np.asarray(capped_gen.generated).tolist() == [3] * len(prompt_lengths)
    assert not np.asarray(capped_gen.ended).any()
    assert np.asarray(first.generated).tolist() == (48 - prompt_lengths).tolist()

    prompt_eos = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths)
    prompt_eos_tokens = np.asarray(prompt_eos.tokens)
    assert np.any(
        prompt_eos_tokens[0, prompt_lengths[0]:]
        != model.tokenizer.pad_token_id
    ), "an EOS inside the teacher-forced prompt must not stop generation"

    # KV fake-quant: teacher-force one fixed sequence and score it under
    # increasingly coarse cache writes; the mismatch must be monotone in bits.
    forced = np.full(len(prompt_lengths), tokens.shape[1])
    scores = {}
    for bits in (0, 8, 3):
        forced_gen = sampling.generate(
            jax.random.key(1), model, first_tokens, forced, eos_id=-1,
            forward=partial(model.forward, dtype=jnp.bfloat16,
                            kv_quant_bits=bits))
        scores[bits] = np.asarray(forced_gen.token_logprobs)
    absdiff = {b: np.abs(scores[b] - scores[0]).mean() for b in (8, 3)}
    assert 0 < absdiff[8] < absdiff[3]

    # Rows whose prompt fills the buffer are scored by default (the KV check
    # above relies on it) and skipped only on request, so that a continuation
    # batch with one finished row ends when its live rows end.
    idle = sampling.generate(
        jax.random.key(1), model, first_tokens, forced, eos_id=-1, skip_full_rows=True)
    assert np.asarray(idle.generated).tolist() == [0] * len(prompt_lengths)
    assert not np.asarray(idle.token_logprobs).any()

    eos_id = int(first_tokens[0, prompt_lengths[0] + 2])
    stopped_gen = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=eos_id)
    stopped = np.asarray(stopped_gen.tokens)
    assert bool(np.asarray(stopped_gen.ended)[0])
    assert int(np.asarray(stopped_gen.generated)[0]) == 3
    eos_position = next(
        position
        for position in range(prompt_lengths[0], stopped.shape[1])
        if stopped[0, position] == eos_id
    )
    assert np.all(
        stopped[0, eos_position + 1:] == model.tokenizer.pad_token_id)
    print("ok: one fixed batch, exact logprobs and top-k, head/tail draws, "
          "deterministic replay, KV fake-quant, EOS stop")


if __name__ == "__main__":
    main()
