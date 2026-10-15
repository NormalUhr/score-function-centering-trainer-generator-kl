"""The RL loss is ordinary token-level REINFORCE."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

import losses
from losses import ISCorrection

TIS = ISCorrection(outside="clamp", high=2.0)


def rollout_loss(
        logits, sampled_token, sampling_vocab_logprobs,
        centered, is_cfg=None, topk=False, sampling_token_logprob=None,
        advantage=1.0):
    stack = lambda row: jnp.stack([row, jnp.zeros_like(row)])[None]
    sampled_logprob = (
        sampling_vocab_logprobs[sampled_token]
        if sampling_token_logprob is None else sampling_token_logprob)
    batch = {
        "tokens": jnp.asarray([[0, sampled_token]]),
        "mask": jnp.asarray([[False, True]]),
        "advantages": jnp.asarray([advantage]),
        "sampling_token_logprobs": jnp.asarray([[sampled_logprob, 0.0]]),
    }
    if centered:
        batch["sampling_vocab_logprobs"] = stack(sampling_vocab_logprobs)
    if topk:
        head_logprobs, head_ids = jax.lax.top_k(sampling_vocab_logprobs, 2)
        batch["center_ids"] = stack(head_ids)
        batch["center_logprobs"] = stack(head_logprobs)
    loss, _ = losses.loss_fn_rl(
        jnp.broadcast_to(logits, (1, 2, logits.size)),
        lambda x: x,
        batch,
        is_cfg,
        score_center=centered or topk,
    )
    return loss


def main():
    logits = jnp.asarray([[
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [3.0, 1.0, 0.0],
    ]])
    batch = {
        "tokens": jnp.asarray([[0, 2, 1]]),
        "mask": jnp.asarray([[False, True, True]]),
        "advantages": jnp.asarray([2.0]),
        "sampling_token_logprobs": jnp.zeros((1, 3)),
    }
    actual, metrics = losses.loss_fn_rl(logits, lambda x: x, batch)
    expected = -2 * np.mean([
        float(jax.nn.log_softmax(logits[0, 0])[2]),
        float(jax.nn.log_softmax(logits[0, 1])[1]),
    ])
    assert np.allclose(actual, expected)
    assert set(metrics) == {
        "pol_entropy", "logp_absdiff_samp_train", "kl_samp_train",
        "frac_masked"}
    assert metrics["frac_masked"] == 0
    chunked, chunked_metrics = losses.loss_fn_rl(logits, lambda x: x, batch, chunk=1)
    assert np.allclose(chunked, actual)
    assert all(np.allclose(chunked_metrics[k], metrics[k]) for k in metrics)

    # mask band (MIS): in-band tokens get the exact importance weight,
    # out-of-band tokens are dropped. Both ratios here are exp(logprob) < 0.5.
    token_logprobs = np.asarray([
        float(jax.nn.log_softmax(logits[0, 0])[2]),
        float(jax.nn.log_softmax(logits[0, 1])[1]),
    ])
    dropped, dropped_metrics = losses.loss_fn_rl(
        logits, lambda x: x, batch, ISCorrection(low=0.5, high=5.0))
    assert dropped == 0
    assert dropped_metrics["frac_masked"] == 1
    weighted, weighted_metrics = losses.loss_fn_rl(
        logits, lambda x: x, batch, ISCorrection(low=0.2, high=5.0))
    assert np.allclose(
        weighted, -2 * np.mean(np.exp(token_logprobs) * token_logprobs))
    assert weighted_metrics["frac_masked"] == 0
    # The band's action is logged per side: both ratios here sit below the
    # low bound, so the drop above acted on every token from the low side.
    assert dropped_metrics["frac_ratio_clipped_low"] == 1
    assert dropped_metrics["frac_ratio_clipped_high"] == 0
    assert weighted_metrics["frac_ratio_clipped_low"] == 0
    assert weighted_metrics["frac_ratio_clipped_high"] == 0
    # TIS (clamp at high = 2, open low side): a token whose ratio exceeds the
    # clip is counted on the high side; one below 1 with no low bound is not.
    tis = ISCorrection(high=2.0, outside="clamp")
    tis_batch = dict(batch, sampling_token_logprobs=jnp.asarray(
        [[-5.0, 0.0, 0.0]], jnp.float32))  # first target: r = e**5 * p > 2
    tis_loss, tis_metrics = losses.loss_fn_rl(
        logits, lambda x: x, tis_batch, tis)
    assert np.allclose(tis_loss, -2 * np.mean(
        [2.0 * token_logprobs[0], np.exp(token_logprobs[1]) * token_logprobs[1]]))
    assert tis_metrics["frac_ratio_clipped_high"] == 0.5
    assert tis_metrics["frac_ratio_clipped_low"] == 0
    assert tis_metrics["frac_masked"] == 0
    # A one-sided correction (sign=neg) leaves positive advantages alone, so
    # the clip acted on none of this positive sequence's tokens.
    _, one_sided = losses.loss_fn_rl(
        logits, lambda x: x, tis_batch, ISCorrection(high=2.0, outside="clamp", sign="neg"))
    assert one_sided["frac_ratio_clipped_high"] == 0
    # The uncorrected loss logs no band metrics (asserted by the key set above).

    # TOPR uses one full-trajectory ratio: positives are untouched (w = 1);
    # negatives get min(product_t r_t, 1). This differs from both token-level
    # correction and the geometric-mean sequence ratio below.
    topr = ISCorrection(
        level="seq_sum", high=1.0, outside="clamp", sign="neg")
    pos_loss, _ = losses.loss_fn_rl(logits, lambda x: x, batch, topr)
    assert np.allclose(pos_loss, actual)
    neg_batch = dict(batch, advantages=jnp.asarray([-2.0]))
    neg_loss, _ = losses.loss_fn_rl(logits, lambda x: x, neg_batch, topr)
    trajectory_ratio = np.exp(token_logprobs.sum())  # sampler logprobs are 0
    assert np.allclose(
        neg_loss, trajectory_ratio * 2 * np.mean(token_logprobs))

    # stat=tv (DPPO): the band tests the binary TV divergence |q - p|; the
    # sign=ppo mask blocks only moves away from the sampler. Here q = 1 and
    # p < 1, so D > 0.2 but r < 1: positives are kept (with weight r),
    # negatives are dropped, and the symmetric gate drops both.
    dppo = ISCorrection(stat="tv", high=0.2, sign="ppo")
    kept_pos, kept_pos_metrics = losses.loss_fn_rl(logits, lambda x: x, batch, dppo)
    assert np.allclose(kept_pos, weighted)
    assert kept_pos_metrics["frac_masked"] == 0
    dropped_neg, dropped_neg_metrics = losses.loss_fn_rl(
        logits, lambda x: x, neg_batch, dppo)
    assert dropped_neg == 0
    assert dropped_neg_metrics["frac_masked"] == 1
    sym_gate, _ = losses.loss_fn_rl(
        logits, lambda x: x, batch, ISCorrection(stat="tv", high=0.2))
    assert sym_gate == 0
    # At q = p the divergence is 0: everything in band with weight r = 1.
    dppo_on_policy, _ = losses.loss_fn_rl(
        logits, lambda x: x,
        dict(batch, sampling_token_logprobs=jnp.asarray(
            [[*token_logprobs, 0.0]], jnp.float32)),
        dppo)
    assert np.allclose(dppo_on_policy, actual)

    # seq_mean level: one weight on the geometric-mean ratio scales the whole
    # sequence; drop zeroes it.
    seq_stat = np.exp(token_logprobs.mean())  # sampler logprobs are 0
    seq_weighted, seq_weighted_metrics = losses.loss_fn_rl(
        logits, lambda x: x, batch,
        ISCorrection(level="seq_mean", low=0.2, high=2.0))
    assert np.allclose(seq_weighted, seq_stat * actual)
    assert seq_weighted_metrics["frac_masked"] == 0
    seq_dropped, seq_dropped_metrics = losses.loss_fn_rl(
        logits, lambda x: x, batch,
        ISCorrection(level="seq_mean", low=0.5, high=2.0))
    assert seq_dropped == 0
    assert seq_dropped_metrics["frac_masked"] == 1
    # seq + stat=tv gates on the mean token TV (the D_bar of DPPO's Eq. 8):
    # mean |1 - p| is ~0.71 here, so delta 0.8 keeps (weight = seq ratio)
    # and delta 0.5 drops.
    seq_tv, _ = losses.loss_fn_rl(
        logits, lambda x: x, batch,
        ISCorrection(level="seq_mean", stat="tv", high=0.8))
    assert np.allclose(seq_tv, seq_stat * actual)
    assert losses.loss_fn_rl(
        logits, lambda x: x, batch,
        ISCorrection(level="seq_mean", stat="tv", high=0.5))[0] == 0

    # PPO band: at q = p the ratio is 1 (in-band), the loss is the constant
    # -mean(advantage), and the surrogate's gradient equals REINFORCE's.
    ppo = ISCorrection(low=0.8, high=1.2, sign="ppo")
    ppo_batch = dict(batch, sampling_token_logprobs=jnp.asarray(
        [[*token_logprobs, 0.0]], jnp.float32))
    ppo_grad = jax.grad(lambda lg: losses.loss_fn_rl(
        lg, lambda x: x, ppo_batch, ppo)[0])(logits)
    pg_grad = jax.grad(lambda lg: losses.loss_fn_rl(
        lg, lambda x: x, batch)[0])(logits)
    assert np.allclose(ppo_grad, pg_grad, atol=1e-6)
    ppo_loss, ppo_metrics = losses.loss_fn_rl(
        logits, lambda x: x, ppo_batch, ppo)
    assert np.allclose(ppo_loss, actual)  # detached band: w = 1 at q = p
    assert ppo_metrics["frac_masked"] == 0
    # Tokens already moved past `high` in the advantage direction are dropped
    # (clipped branch is constant): zero gradient, counted in frac_masked.
    stale = dict(batch, sampling_token_logprobs=jnp.asarray(
        [[*(token_logprobs - np.log(2.0)), 0.0]], jnp.float32))
    stale_grad = jax.grad(lambda lg: losses.loss_fn_rl(
        lg, lambda x: x, stale, ppo)[0])(logits)
    assert jnp.abs(stale_grad).max() < 1e-6
    assert losses.loss_fn_rl(
        logits, lambda x: x, stale, ppo)[1]["frac_masked"] == 1
    # A < 0 with ratio above the dual clip (3.0): also constant, zero gradient.
    neg = dict(stale, advantages=jnp.asarray([-2.0]),
               sampling_token_logprobs=jnp.asarray(
                   [[*(token_logprobs - np.log(4.0)), 0.0]], jnp.float32))
    neg_grad = jax.grad(lambda lg: losses.loss_fn_rl(
        lg, lambda x: x, neg, ppo)[0])(logits)
    assert jnp.abs(neg_grad).max() < 1e-6
    assert losses.loss_fn_rl(
        logits, lambda x: x, neg, ppo)[1]["frac_masked"] == 1
    # A < 0 with ratio inside (low, 3.0): kept, gradient flows.
    neg_kept = dict(neg, sampling_token_logprobs=jnp.asarray(
        [[*(token_logprobs - np.log(2.0)), 0.0]], jnp.float32))
    kept_grad = jax.grad(lambda lg: losses.loss_fn_rl(
        lg, lambda x: x, neg_kept, ppo)[0])(logits)
    assert jnp.abs(kept_grad).max() > 1e-3
    assert losses.loss_fn_rl(
        logits, lambda x: x, neg_kept, ppo)[1]["frac_masked"] == 0

    # Group-centered advantages: +1 for a solved completion and -1 otherwise,
    # minus the mean of its group.
    assert np.array_equal(
        losses.compute_advantages(np.asarray([1.0, 0.0, -2.0, 3.0]), 2),
        [1.0, -1.0, -1.0, 1.0])
    assert np.array_equal(
        losses.compute_advantages(np.asarray([1.0, 1.0, 0.0, 0.0]), 2),
        np.zeros(4))
    assert np.allclose(
        losses.compute_advantages(np.asarray([1.0, 0.0, 0.0, 0.0]), 4),
        [1.5, -0.5, -0.5, -0.5])

    trainer_logits = jnp.asarray([0.5, -0.2, 1.1])
    sampling_vocab_logprobs = jax.nn.log_softmax(
        trainer_logits + jnp.asarray([0.1, 0.3, -0.2]))
    expected_gradients = {}
    # The (0.8, 1.1) band masks some of the three tokens but not all.
    for name, centered, is_cfg in (
            ("none", False, None),
            ("centered", True, None),
            ("centered_tis", True, TIS),
            ("centered_mis", True, ISCorrection(low=0.8, high=1.1)),
            ("centered_ppo", True,
             ISCorrection(low=0.8, high=1.2, sign="ppo")),
            ("centered_neg_clamp", True,
             ISCorrection(high=1.0, outside="clamp", sign="neg"))):
        gradient = jax.grad(
            lambda logits, token: rollout_loss(
                logits, token, sampling_vocab_logprobs, centered, is_cfg))
        gradients = jax.vmap(gradient, in_axes=(None, 0))(
            trainer_logits, jnp.arange(trainer_logits.size))
        expected_gradients[name] = (
            jnp.exp(sampling_vocab_logprobs)[:, None] * gradients
        ).sum(0)
    assert jnp.abs(expected_gradients["none"]).max() > 1e-3
    for name, drift in expected_gradients.items():
        if name != "none":
            assert jnp.abs(drift).max() < 1e-6, name

    # Subsampled centering: with V=3 and k=2 the tail is one token, so
    # topk reproduces the exact correction and the expected gradient
    # under constant reward must still vanish.
    q = jnp.exp(sampling_vocab_logprobs)
    for is_cfg in (
            None,
            TIS,
            ISCorrection(low=0.8, high=1.1),
            ISCorrection(low=0.8, high=1.2, sign="ppo"),
            ISCorrection(high=1.0, outside="clamp", sign="neg")):
        gradient = jax.grad(
            lambda logits, token: rollout_loss(
                logits, token, sampling_vocab_logprobs, False, is_cfg,
                topk=True,
            )
        )
        gradients = jax.vmap(gradient, in_axes=(None, 0))(
            trainer_logits, jnp.arange(trainer_logits.size))
        drift = (q[:, None] * gradients).sum(0)
        assert jnp.abs(drift).max() < 1e-6, is_cfg

    # On a multi-token tail, the fast path must equal the generic full-vocab
    # calculation under the explicitly reconstructed q_hat. This pins the
    # optimization independently of how accurately q_hat models the true q.
    trainer_logits5 = jnp.asarray([0.7, -0.4, 1.2, 0.1, -1.0])
    sampling_logprobs5 = jax.nn.log_softmax(
        trainer_logits5 + jnp.asarray([-0.3, 0.5, 0.1, -0.2, 0.4]))
    p5 = jax.nn.softmax(trainer_logits5)
    head_q5, head_ids5 = jax.lax.top_k(sampling_logprobs5, 2)
    rho5 = ((1.0 - jnp.exp(head_q5).sum())
            / (1.0 - p5[head_ids5].sum()))
    qhat5 = rho5 * p5
    qhat5 = qhat5.at[head_ids5].set(jnp.exp(head_q5))
    qhat_logprobs5 = jnp.log(qhat5)
    ratio_corrections = (
        None,
        ISCorrection(),
        TIS,
        ISCorrection(low=0.5, high=2.0, outside="clamp"),
        ISCorrection(low=0.8, high=1.1),
        ISCorrection(low=0.8, high=1.2, sign="ppo"),
        ISCorrection(high=1.0, outside="clamp", sign="neg"),
    )
    for is_cfg in ratio_corrections:
        for advantage in (-1.0, 1.0):
            for token in range(trainer_logits5.size):
                sample_q = sampling_logprobs5[token]
                fast = jax.value_and_grad(lambda logits: rollout_loss(
                    logits, token, sampling_logprobs5, False, is_cfg,
                    topk=True, sampling_token_logprob=sample_q,
                    advantage=advantage))(trainer_logits5)
                generic = jax.value_and_grad(lambda logits: rollout_loss(
                    logits, token, qhat_logprobs5, True, is_cfg,
                    sampling_token_logprob=sample_q,
                    advantage=advantage))(trainer_logits5)
                case = (is_cfg, advantage, token)
                assert np.allclose(fast[0], generic[0], atol=1e-6), case
                assert np.allclose(fast[1], generic[1], atol=1e-6), case

    try:
        rollout_loss(
            trainer_logits, 0, sampling_vocab_logprobs, True,
            ISCorrection(stat="tv", high=0.05))
    except ValueError as error:
        assert "stat=ratio" in str(error)
    else:
        raise AssertionError("TV correction composed with score centering")
    print(
        "ok: REINFORCE, the IS-correction grid (TIS, MIS, PPO, DPPO, TOPR, "
        "seq_mean and seq_sum levels), centered variants, exact sampler "
        "logprobs, topk centering drift, and group baselines")


if __name__ == "__main__":
    main()
