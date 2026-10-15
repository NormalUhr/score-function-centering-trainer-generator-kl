"""Head/tail payload estimators: unbiased centering, the sampler-KL term with
its sparse difference gradient, coefficient clipping, and the age schedule of
the coefficient."""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

import losses
from losses import ISCorrection, SamplerKL

TIS = ISCorrection(outside="clamp", high=2.0)
MIS = ISCorrection(low=0.5, high=5.0)


def payload(q_logprobs, head_k):
    """Exact top-k head of q and the conditional tail distribution."""
    head_logprobs, head_ids = jax.lax.top_k(q_logprobs, head_k)
    q = np.exp(np.asarray(q_logprobs))
    tail_ids = np.setdiff1d(np.arange(q.size), np.asarray(head_ids))
    tail_cond = q[tail_ids] / q[tail_ids].sum()
    return head_logprobs, head_ids, tail_ids, tail_cond


def token_loss(logits, sampled_token, q_logprobs, head_k, tail_draws,
               is_cfg=None, score_center=True, sampler_kl=None,
               kl_coeff=0.0, advantage=1.0):
    """One generated token (position 0 predicts position 1) with a head/tail
    payload; tail_draws are the M sampled tail ids, or None for the top-k
    payload alone (modeled tail)."""
    stack = lambda row: jnp.stack([row, jnp.zeros_like(row)])[None]
    head_logprobs, head_ids, _, _ = payload(q_logprobs, head_k)
    batch = {
        "tokens": jnp.asarray([[0, sampled_token]]),
        "mask": jnp.asarray([[False, True]]),
        "advantages": jnp.asarray([advantage]),
        "sampling_token_logprobs": jnp.asarray([[q_logprobs[sampled_token], 0.0]]),
        "center_ids": stack(head_ids),
        "center_logprobs": stack(head_logprobs),
    }
    if tail_draws is not None:
        draws = jnp.asarray(tail_draws)
        batch["tail_ids"] = stack(draws)
        batch["tail_logprobs"] = stack(q_logprobs[draws])
    if sampler_kl is not None:
        batch["kl_coeff"] = jnp.asarray(kl_coeff, jnp.float32)
    loss, metrics = losses.loss_fn_rl(
        jnp.broadcast_to(logits, (1, 2, logits.size)),
        lambda x: x, batch, is_cfg,
        score_center=score_center, sampler_kl=sampler_kl)
    return loss, metrics


def expected_gradient(fn, q_logprobs, head_k, num_tail, sampled_tokens=None):
    """E over a ~ q (or a fixed token) and M IID tail draws t ~ q(.|tail) of
    grad_logits fn(a, draws)."""
    _, _, tail_ids, tail_cond = payload(q_logprobs, head_k)
    q = np.exp(np.asarray(q_logprobs))
    tokens = range(q.size) if sampled_tokens is None else sampled_tokens
    total = 0.0
    for token in tokens:
        token_weight = q[token] if sampled_tokens is None else 1.0
        for draws in itertools.product(range(tail_ids.size), repeat=num_tail):
            weight = token_weight * np.prod([tail_cond[d] for d in draws])
            total = total + weight * np.asarray(fn(token, tail_ids[list(draws)]))
    return total


def main():
    logits = jnp.asarray([0.7, -0.4, 1.2, 0.1, -1.0])
    q_logprobs = jax.nn.log_softmax(
        logits + jnp.asarray([-0.3, 0.5, 0.1, -0.2, 0.4]))
    p = np.asarray(jax.nn.softmax(logits))
    q = np.exp(np.asarray(q_logprobs))
    head_k = 2

    # 1. Head/tail centering is unbiased: under a constant advantage the
    # expected gradient over a ~ q and IID conditional tail draws vanishes,
    # alone and composed with token-level ratio corrections.
    for num_tail in (1, 2):
        for is_cfg in (None, TIS, MIS):
            drift = expected_gradient(
                lambda token, draws: jax.grad(lambda lg: token_loss(
                    lg, token, q_logprobs, head_k, draws, is_cfg)[0])(logits),
                q_logprobs, head_k, num_tail)
            assert np.abs(drift).max() < 1e-6, (num_tail, is_cfg, drift)
    # The head/tail path is a different estimator from the modeled tail: for
    # a single draw the two losses differ, so the branch really runs.
    _, _, tail_ids, _ = payload(q_logprobs, head_k)
    sampled, _ = token_loss(logits, 2, q_logprobs, head_k, tail_ids[:1])
    modeled, _ = token_loss(logits, 2, q_logprobs, head_k, None)
    assert not np.allclose(sampled, modeled)

    # 2. Sampler-KL term with the top-k payload: the gradient is exactly
    # coeff * (p - q_hat) with the tail of q modeled as rho * p.
    kl = SamplerKL(coeff=1.0, clip=5.0)
    head_logprobs, head_ids, _, _ = payload(q_logprobs, head_k)
    rho = (1 - np.exp(head_logprobs).sum()) / (1 - p[np.asarray(head_ids)].sum())
    q_hat = rho * p
    q_hat[np.asarray(head_ids)] = np.exp(head_logprobs)
    for coeff in (0.3, 2.0):
        kl_grad = jax.grad(lambda lg: token_loss(
            lg, 2, q_logprobs, head_k, None, score_center=False,
            sampler_kl=kl, kl_coeff=coeff, advantage=0.0)[0])(logits)
        assert np.allclose(kl_grad, coeff * (p - q_hat), atol=1e-6), coeff
    _, metrics = token_loss(
        logits, 2, q_logprobs, head_k, None, score_center=False,
        sampler_kl=kl, kl_coeff=1.0)
    assert np.allclose(
        metrics["kl_sampler"], (q_hat * (np.log(q_hat) - np.log(p))).sum(),
        atol=1e-6)

    # 3. Sampler-KL term with the head/tail payload: exact head coordinates
    # plus the sparse difference estimator on the sampled tail coordinates.
    # In expectation over the tail draws the gradient is coeff * (p - q) and
    # the value estimate is KL(q || p).
    for num_tail in (1, 2):
        kl_grad = expected_gradient(
            lambda token, draws: jax.grad(lambda lg: token_loss(
                lg, token, q_logprobs, head_k, draws, score_center=False,
                sampler_kl=kl, kl_coeff=0.7, advantage=0.0)[0])(logits),
            q_logprobs, head_k, num_tail, sampled_tokens=[2])
        assert np.allclose(kl_grad, 0.7 * (p - q), atol=1e-6), num_tail
        kl_value = expected_gradient(
            lambda token, draws: token_loss(
                logits, token, q_logprobs, head_k, draws, score_center=False,
                sampler_kl=kl, kl_coeff=1.0)[1]["kl_sampler"],
            q_logprobs, head_k, num_tail, sampled_tokens=[2])
        assert np.allclose(kl_value, (q * (np.log(q) - np.log(p))).sum(),
                           atol=1e-6), num_tail
    # A single draw puts gradient only on the head and on that coordinate.
    one_draw = jax.grad(lambda lg: token_loss(
        lg, 2, q_logprobs, head_k, tail_ids[:1], score_center=False,
        sampler_kl=kl, kl_coeff=1.0, advantage=0.0)[0])(logits)
    touched = np.flatnonzero(np.abs(np.asarray(one_draw)) > 1e-9)
    assert set(touched) <= set(np.asarray(head_ids)) | {tail_ids[0]}, touched

    # 4. Clipping bounds the tail coefficient p/q - 1 from above and the
    # diagnostic reports the unclipped maximum.
    spiky = jax.nn.log_softmax(jnp.asarray([3.0, 2.0, 1.0, 0.0, -6.0]))
    spiky_p = np.asarray(jax.nn.softmax(logits))
    _, _, spiky_tail, _ = payload(spiky, head_k)
    rare = int(np.asarray(spiky_tail)[np.argmin(np.exp(spiky)[spiky_tail])])
    ratio = spiky_p[rare] / np.exp(spiky[rare]) - 1
    assert ratio > 5.0
    tight = SamplerKL(coeff=1.0, clip=5.0)
    loose = SamplerKL(coeff=1.0, clip=1e9)
    tail_mass = 1 - np.exp(jax.lax.top_k(spiky, head_k)[0]).sum()
    for cfg, expected_coeff in ((tight, 5.0), (loose, ratio)):
        grad = jax.grad(lambda lg: token_loss(
            lg, 0, spiky, head_k, [rare], score_center=False,
            sampler_kl=cfg, kl_coeff=1.0, advantage=0.0)[0])(logits)
        assert np.allclose(grad[rare], tail_mass * expected_coeff, rtol=1e-5)
    _, spiky_metrics = token_loss(
        logits, 0, spiky, head_k, [rare], score_center=False,
        sampler_kl=tight, kl_coeff=1.0)
    assert np.allclose(spiky_metrics["tail_coeff_max"], ratio, rtol=1e-5)
    assert spiky_metrics["frac_tail_clipped"] == 1.0

    # 5. A zero coefficient leaves the loss unchanged.
    plain, _ = token_loss(logits, 2, q_logprobs, head_k, tail_ids[:1])
    zero, _ = token_loss(logits, 2, q_logprobs, head_k, tail_ids[:1],
                         sampler_kl=kl, kl_coeff=0.0)
    assert np.allclose(plain, zero)

    # 6. The coefficient follows the age e of the token's sampler in optimizer
    # steps: coeff / sqrt(e), and 0 at age 0, where the token's sampler is the
    # trainer and there is nothing to correct beyond numerics.
    cfg = SamplerKL(coeff=1.0, clip=5.0)
    assert np.allclose([float(cfg.coefficient(e)) for e in (0, 1, 4, 100)], [0.0, 1.0, 0.5, 0.1])
    # Array input keeps the age-0 rule elementwise.
    assert np.allclose(cfg.coefficient(np.array([0, 1, 4])), [0.0, 1.0, 0.5])
    assert np.allclose(SamplerKL(coeff=0.3, clip=5.0).coefficient(16), 0.075)

    print("ok: head/tail centering drift, sampler-KL gradients (modeled and "
          "sampled tails), clipping, the age schedule")


if __name__ == "__main__":
    main()
