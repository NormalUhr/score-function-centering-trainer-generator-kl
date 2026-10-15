"""Tiny end-to-end checks for the synchronous RL loop."""
import ast
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_training(
        steps, seq_len=160, staleness=0, sampler_dtype="bfloat16",
        score_center=False, is_level=None, is_stat="ratio", is_low=0.0,
        is_high="inf", is_outside="drop", is_sign="both", kv_quant_bits=0,
        eval_enabled=False, vocab_logprobs=16, tail_samples=0,
        sampler_kl_coeff=0.0):
    weights_dir = os.path.expanduser("~/.cache/postax/weights")
    command = [
        sys.executable,
        str(ROOT / "train_rl.py"),
        "model.source=Qwen/Qwen3-0.6B-Base",
        "model.init=random",
        "model.hidden_size=128",
        "model.num_hidden_layers=2",
        f"model.weights_dir={weights_dir}",
        f"dtypes.sampler={sampler_dtype}",
        "dtypes.trainer=float32",
        "dtypes.master=float32",
        f"sampler.staleness={staleness}",
        f"sampler.kv_quant_bits={kv_quant_bits}",
        "rl.num_prompts=2",
        "rl.group_size=2",
        f"rl.seq_len={seq_len}",
        f"rl.sc={str(score_center).lower()}",
        ("sampler.vocab_logprobs="
         f"{vocab_logprobs if vocab_logprobs is not None else 'null'}"),
        f"sampler.tail_samples={tail_samples}",
        f"rl.sampler_kl.coeff={sampler_kl_coeff}",
        f"rl.is.level={is_level or 'null'}",
        f"rl.is.stat={is_stat}",
        f"rl.is.low={is_low}",
        f"rl.is.high={is_high}",
        f"rl.is.outside={is_outside}",
        f"rl.is.sign={is_sign}",
        f"stop.steps={steps}",
        f"eval.enabled={str(eval_enabled).lower()}",
        "eval.num_prompts=2",
        "eval.every_steps=2",
        "log.wandb_mode=disabled",
        "log.every_steps=1",
        "env.args.num_operands=3",
        "env.args.min_number=1",
        "env.args.max_number=30",
        "env.args.max_target=100",
        "env.args.dataset_size=8",
        "env.args.num_eval_examples=2",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "WANDB_MODE": "disabled"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout)
    metrics = [
        ast.literal_eval(match.group(1))
        for match in re.finditer(r"Step \d+ (\{[^\n]+\})", completed.stdout)
    ]
    assert len(metrics) == steps, completed.stdout
    return metrics


def tail_and_kl():
    """Head/tail payload and the sampler-KL term run end to end, log their
    diagnostics, and follow the age schedule; invalid combinations are
    rejected before training starts."""
    stratified = run_training(
        steps=3, staleness=2, score_center=True, vocab_logprobs=4,
        tail_samples=3, sampler_kl_coeff=0.1)
    assert all(math.isfinite(row["loss"]) for row in stratified)
    assert all(math.isfinite(row["kl_tail"]) for row in stratified)
    assert all(math.isfinite(row["kl_sampler"]) for row in stratified)
    assert all(0 <= row["frac_tail_clipped"] <= 1 for row in stratified)
    assert all(row["tail_coeff_max"] >= 0 for row in stratified)
    # Sampler ages 0, 1, 2 optimizer steps: zero at age 0, then 0.1 / sqrt(age).
    assert [round(row["kl_coeff"], 6) for row in stratified] == [0.0, 0.1, round(0.1 / math.sqrt(2), 6)]
    elapsed = [row["elapsed_s"] for row in stratified]
    assert elapsed[0] > 0 and elapsed[-1] > elapsed[0]

    modeled = run_training(
        steps=1, kv_quant_bits=3, score_center=True, is_level="token",
        is_high=2.0, is_outside="clamp", sampler_kl_coeff=0.1)[0]
    assert math.isfinite(modeled["loss"])
    assert math.isfinite(modeled["kl_sampler"])
    assert "tail_coeff_max" not in modeled and "kl_tail" not in modeled

    for kwargs, message in (
            (dict(tail_samples=2, vocab_logprobs="full"), "sampler.tail_samples"),
            (dict(sampler_kl_coeff=0.1, vocab_logprobs=None), "rl.sampler_kl.coeff")):
        try:
            run_training(steps=1, **kwargs)
        except RuntimeError as error:
            assert message in str(error), (kwargs, str(error)[-500:])
        else:
            raise AssertionError(f"accepted invalid config {kwargs}")
    print("ok: head/tail payload, sampler-KL term, age schedule, validation")


def continued_rollout_boundary():
    """continue_rollouts keeps, at every position, the payload of the weights
    that produced the token there: the first stage's for predictions below
    resume - 1, the second stage's from there on; rows that ended in the first
    stage keep its payload everywhere and generate nothing more."""
    import jax
    import numpy as np
    sys.path.insert(0, str(ROOT))
    import models
    from models import sampling
    import train_rl

    model = models.load("Qwen/Qwen3-0.6B-Base", os.path.expanduser("~/.cache/postax/weights"),
                        init="random", hidden_size=128, num_hidden_layers=2, key=jax.random.key(0))
    fresh = jax.tree.map(lambda w: w + 0.05 * jax.random.normal(jax.random.key(9), w.shape, w.dtype)
                         if w.ndim > 1 else w, model.weights)
    rng = np.random.default_rng(0)
    prompt_lengths = np.asarray([3, 7, 12, 5, 9, 4, 30, 6])
    seq_len = 32
    tokens = np.full((len(prompt_lengths), seq_len), model.tokenizer.pad_token_id, np.int32)
    for row, length in enumerate(prompt_lengths):
        tokens[row, :length] = rng.integers(10, 5000, length)
    payload = dict(logprobs_topk=4, tail_samples=2)
    first = sampling.generate(jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1,
                              max_new_tokens=6, **payload)
    # Row 0 is forced to look ended after stage one.
    ended = np.zeros(len(prompt_lengths), bool)
    ended[0] = True
    first = first._replace(ended=jax.numpy.asarray(ended))
    merged, from_stale = train_rl.continue_rollouts(
        model, first, fresh, prompt_lengths, jax.random.key(2), payload)

    generated = np.asarray(first.generated)
    resume = np.where(ended | (generated == 0), seq_len, prompt_lengths + generated)
    expected = np.arange(seq_len)[None, :] < (resume - 1)[:, None]
    assert np.array_equal(from_stale, expected)
    # Row 6 has two free positions: stage one fills both, so it is full and
    # stage two skips it.
    assert generated[6] == 2 and resume[6] == seq_len
    merged_generated = np.asarray(merged.generated)
    assert merged_generated[6] == 2
    # Row 0 (ended) generates nothing more; a continued row does.
    assert merged_generated[0] == generated[0]
    assert merged_generated[1] > generated[1]
    # Stage-one payload survives on stale positions, stage-two elsewhere.
    first_head = np.asarray(first.head_ids)
    merged_head = np.asarray(merged.head_ids)
    for row in range(len(prompt_lengths)):
        stale_positions = np.flatnonzero(from_stale[row])
        assert np.array_equal(merged_head[row, stale_positions], first_head[row, stale_positions])
    fresh_positions = np.flatnonzero(~from_stale[1])
    assert len(fresh_positions) > 0
    # The prefix produced by stage one is kept verbatim.
    merged_tokens = np.asarray(merged.tokens)
    first_tokens = np.asarray(first.tokens)
    for row in range(len(prompt_lengths)):
        assert np.array_equal(merged_tokens[row, :resume[row]], first_tokens[row, :resume[row]])
    # Fresh positions of a continued row differ from stage one's payload
    # somewhere (different weights), which shows the second stage was used.
    assert not np.array_equal(merged_head[1, fresh_positions], first_head[1, fresh_positions])
    # Independent oracle for the boundary: score the merged tokens under each
    # snapshot by teacher forcing (prompt length = buffer length). Below
    # resume - 1 the merged log-probabilities equal the stale snapshot's, from
    # resume - 1 to the last generated position the fresh snapshot's.
    full = np.full(len(prompt_lengths), seq_len)
    stale_ref = np.asarray(sampling.generate(
        jax.random.key(3), model, merged.tokens, full, eos_id=-1, **payload).token_logprobs)
    fresh_ref = np.asarray(sampling.generate(
        jax.random.key(3), model, merged.tokens, full, eos_id=-1, weights=fresh,
        **payload).token_logprobs)
    merged_logp = np.asarray(merged.token_logprobs)
    for row in range(len(prompt_lengths)):
        # Tokens L .. L+g-1 are predicted at positions L-1 .. L+g-2.
        last = prompt_lengths[row] + merged_generated[row] - 2
        stale_positions = np.arange(prompt_lengths[row] - 1, min(resume[row] - 1, last + 1))
        fresh_positions_row = np.arange(resume[row] - 1, last + 1)
        assert np.allclose(merged_logp[row, stale_positions], stale_ref[row, stale_positions]), row
        if len(fresh_positions_row):
            assert np.allclose(merged_logp[row, fresh_positions_row], fresh_ref[row, fresh_positions_row]), row
            assert not np.allclose(merged_logp[row, fresh_positions_row], stale_ref[row, fresh_positions_row]), row
    print("ok: continued-rollout boundary and payload merge")


def main():
    # eval on: the eval path re-enters sample() and must stay in sync with
    # its return shape.
    metrics = run_training(
        steps=3, staleness=1, score_center=True, vocab_logprobs="full",
        is_level="token", is_high=2.0, is_outside="clamp",
        eval_enabled=True)
    assert all(math.isfinite(row["loss"]) for row in metrics)
    assert all(0 <= row["frac_truncated"] <= 1 for row in metrics)
    assert all(row["pol_entropy"] > 0 for row in metrics)
    assert all(row["logp_absdiff_samp_train"] >= 0 for row in metrics)
    assert [row["sampler_age"] for row in metrics] == [0, 1, 0]

    approx = run_training(
        steps=1, staleness=1, score_center=True, is_level="token",
        is_high=2.0, is_outside="clamp")[0]
    assert math.isfinite(approx["loss"])

    # Exact full-vocabulary MIS+SC: this is a distinct loss path from the
    # default modeled top-k tail and needs an end-to-end multi-step check.
    mis = run_training(
        steps=2, kv_quant_bits=3, score_center=True,
        is_level="token", is_low=0.5, is_high=5.0,
        vocab_logprobs="full")
    assert all(math.isfinite(row["loss"]) for row in mis)
    assert all(0 <= row["frac_masked"] <= 1 for row in mis)

    # PPO with a stale sampler and a quantized KV cache: the surrogate sees
    # ratios away from 1.
    ppo = run_training(
        steps=2, staleness=1, kv_quant_bits=3, is_level="token",
        is_low=0.8, is_high=1.2, is_sign="ppo")
    assert all(math.isfinite(row["loss"]) for row in ppo)
    assert all(0 <= row["frac_masked"] <= 1 for row in ppo)

    # DPPO (binary-TV band at the paper's defaults) without SC.
    dppo = run_training(
        steps=1, kv_quant_bits=3, is_level="token",
        is_stat="tv", is_high=0.2, is_sign="ppo")[0]
    assert math.isfinite(dppo["loss"])
    assert 0 <= dppo["frac_masked"] <= 1

    # seq_mean level (GSPO-shaped band, wide enough to keep most sequences)
    # composes with topk: the weight scales the centered score.
    seq = run_training(
        steps=1, kv_quant_bits=3, score_center=True,
        is_level="seq_mean", is_low=0.5, is_high=2.0)[0]
    assert math.isfinite(seq["loss"])
    assert 0 <= seq["frac_masked"] <= 1

    # TOPR's seq_sum level uses one full-trajectory ratio.
    topr = run_training(
        steps=1, kv_quant_bits=3, is_level="seq_sum", is_high=1.0,
        is_outside="clamp", is_sign="neg")[0]
    assert math.isfinite(topr["loss"])
    assert 0 <= topr["frac_masked"] <= 1

    ppo_sc = run_training(
        steps=1, kv_quant_bits=3, is_level="token", is_low=0.8, is_high=1.2,
        is_sign="ppo", score_center=True, vocab_logprobs="full")[0]
    assert math.isfinite(ppo_sc["loss"])
    assert 0 <= ppo_sc["frac_masked"] <= 1

    no_completion = run_training(steps=1, seq_len=96, sampler_dtype="int8")[0]
    assert no_completion["mean_completion_len"] == 0
    # kl_head/head_mass are logged even without score centering.
    assert "kl_head" in no_completion and "head_mass" in no_completion

    no_vocab = run_training(steps=1, vocab_logprobs=None)[0]
    assert "kl_head" not in no_vocab and "head_mass" not in no_vocab
    try:
        run_training(
            steps=1, score_center=True, vocab_logprobs=None)
    except RuntimeError as error:
        assert "rl.sc=true requires sampler.vocab_logprobs" in str(error)
    else:
        raise AssertionError("score centering accepted no sampler distribution")

    # The default top-k collection feeds both the centering and the
    # always-on head diagnostics.
    kv_quant = run_training(
        steps=1, kv_quant_bits=3, score_center=True)[0]
    assert math.isfinite(kv_quant["loss"])
    assert kv_quant["logp_absdiff_samp_train"] > 0
    assert math.isfinite(kv_quant["kl_head"])
    assert 0 < kv_quant["head_mass"] <= 1

    tail_and_kl()
    continued_rollout_boundary()
    print(
        "ok: synchronous loop, stale sampler, the IS grid (TIS, MIS, PPO, "
        "DPPO, seq_mean and seq_sum levels), 8-bit sampler, KV fake-quant, "
        "overlong prompts, and score centering (full and top-k)")


if __name__ == "__main__":
    main()
