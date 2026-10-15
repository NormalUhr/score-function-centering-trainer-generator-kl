"""The staged decode carries its KV cache: under constant weights a decode
cut into stages reproduces the single-stage decode's tokens and
log-probabilities, and only the positions from the smallest resume position
on are recomputed. Runs on the CPU with a tiny random model."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np

import models
import train_rl
from models import sampling

WEIGHTS_DIR = os.path.expanduser("~/.cache/postax/weights")


def main():
    model = models.load("Qwen/Qwen3-0.6B-Base", WEIGHTS_DIR, init="random",
                        hidden_size=128, num_hidden_layers=2, key=jax.random.key(0))
    rng = np.random.default_rng(0)
    prompt_lengths = np.asarray([3, 7, 12, 5])
    seq_len = 40
    tokens = np.full((len(prompt_lengths), seq_len), model.tokenizer.pad_token_id, np.int32)
    for row, length in enumerate(prompt_lengths):
        tokens[row, :length] = rng.integers(10, 5000, length)
    for payload in (dict(logprobs_topk=4), dict(logprobs_topk=4, tail_samples=1)):
        staged_matches_single(model, tokens, prompt_lengths, payload)
    print("ok: staged decode with a carried KV cache reproduces the single-stage decode")


def staged_matches_single(model, tokens, prompt_lengths, payload):
    full = sampling.generate(jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1, **payload)

    # Three stages of 9 tokens and a last one to the end, each resumed from
    # the KV cache and the random key the previous stage left behind.
    generation, resume = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1, max_new_tokens=9,
        return_resume=True, **payload)
    for max_new in (9, 9, None):
        generation, _, resume = train_rl.continue_rollouts(
            model, generation, model.weights, prompt_lengths, None, payload,
            max_new_tokens=max_new, eos_id=-1, resume=resume, return_resume=True)
    assert np.array_equal(np.asarray(generation.tokens), np.asarray(full.tokens))
    assert np.allclose(np.asarray(generation.token_logprobs), np.asarray(full.token_logprobs), atol=1e-5)
    # The last buffer slot predicts nothing and holds whatever the top-k ring
    # buffer had there; every scored position agrees.
    assert np.array_equal(np.asarray(generation.head_ids)[:, :-1], np.asarray(full.head_ids)[:, :-1])
    assert np.allclose(np.asarray(generation.head_logprobs)[:, :-1], np.asarray(full.head_logprobs)[:, :-1], atol=1e-5)
    assert np.array_equal(np.asarray(generation.generated), np.asarray(full.generated))
    if payload.get("tail_samples"):
        # The tail draws are keyed by position, so staging does not change them.
        assert np.array_equal(np.asarray(generation.tail_ids)[:, :-1], np.asarray(full.tail_ids)[:, :-1])
        assert np.allclose(np.asarray(generation.tail_logprobs)[:, :-1], np.asarray(full.tail_logprobs)[:, :-1], atol=1e-5)

    # Without the carried cache the continuation re-decodes from position 0
    # under the same weights and still agrees: the cache changes the cost,
    # not the result, when the weights do not change.
    first, _ = sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1, max_new_tokens=9,
        return_resume=True, **payload)
    redone, _ = train_rl.continue_rollouts(
        model, first, model.weights, prompt_lengths, jax.random.key(1), payload,
        max_new_tokens=None, eos_id=-1)
    assert np.array_equal(np.asarray(redone.tokens)[:, :prompt_lengths.min() + 9],
                          np.asarray(full.tokens)[:, :prompt_lengths.min() + 9])


if __name__ == "__main__":
    main()
