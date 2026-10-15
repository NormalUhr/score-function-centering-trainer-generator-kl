import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from models.topk import take_vocab


def get_batch_targets(tokens, mask):
    y = jnp.roll(tokens, -1, axis=1)
    mask = jnp.roll(mask, -1, axis=1).at[:, -1].set(False)
    return tokens, y, mask


def compute_advantages(rewards, group_size):
    """+1 for a solved completion and -1 otherwise, minus the mean of its
    group of `group_size` completions of the same prompt."""
    solved = np.asarray(rewards) > 0
    groups = np.where(solved, 1.0, -1.0).astype(np.float32).reshape(-1, group_size)
    return (groups - groups.mean(axis=1, keepdims=True)).reshape(-1)


IS_LEVELS = ("token", "seq_mean", "seq_sum")
IS_STATS = ("ratio", "tv")
IS_OUTSIDE = ("clamp", "drop")
IS_SIGNS = ("both", "neg", "ppo")

DUAL_CLIP = 3.0  # PPO band top for A < 0, the dual clip of arXiv:1912.09729


class ISCorrection(NamedTuple):
    """One point on the IS-correction grid. TIS, MIS, PPO, DAPO, GSPO, DPPO
    and TOPR are spellings of it (the table in README.md). Weights are always
    detached REINFORCE coefficients."""
    level: str = "token"   # token | seq_mean (mean log r) | seq_sum (sum log r)
    stat: str = "ratio"    # band statistic: the ratio r = p/q, or the binary
                           # TV divergence |q - p| (DPPO, arXiv:2602.04879)
    low: float = 0.0       # trust band; 0 / inf = open side (tv: high = delta)
    high: float = math.inf
    outside: str = "drop"  # clamp = pull weight to the nearest bound | drop = 0
    sign: str = "both"     # both | neg = which advantages are corrected
                           # (the others keep w=1) | ppo = block only moves
                           # away from the sampler:
                           # ratio: pos (0, high], neg [low, DUAL_CLIP];
                           # tv: pos r>1, neg r<1, when tv > high (no dual clip)
    def in_band(self, r, positive, tv):
        if self.stat == "tv":
            if self.sign == "ppo":
                return jnp.where(positive, (r <= 1.0) | (tv <= self.high),
                                 (r >= 1.0) | (tv <= self.high))
            return tv <= self.high
        if self.sign == "ppo":
            return jnp.where(
                positive, r <= self.high, (r >= self.low) & (r <= DUAL_CLIP))
        return (r >= self.low) & (r <= self.high)


def is_weights(log_ratio, positive, c, tv=None):
    """Detached IS weight on r = exp(log_ratio), clamped to e**+-20. For
    level=seq_mean uses per-sequence means for log_ratio (and tv), while
    level=seq_sum uses the per-sequence sum (the full trajectory ratio).
    stat=tv applies the same weight on the tv statistic instead."""
    r = jnp.exp(jnp.clip(log_ratio, -20.0, 20.0))
    w = jnp.where(
        c.in_band(r, positive, tv),
        r,
        jnp.clip(r, c.low, c.high) if c.outside == "clamp" else 0.0)
    if c.sign == "neg":
        w = jnp.where(positive, 1.0, w)
    return jax.lax.stop_gradient(w)


def band_acted(log_ratio, positive, c, tv=None):
    """Tokens whose weight the band changed: outside the band and, for a
    one-sided correction, on the corrected side. Same arguments as is_weights."""
    r = jnp.exp(jnp.clip(log_ratio, -20.0, 20.0))
    acted = ~c.in_band(r, positive, tv)
    if c.sign == "neg":
        acted = acted & ~positive
    return acted


def centering_weights(vocab_logprobs, sampling_logprobs, positive, c):
    """Effective centering mass q*w for a pointwise ratio correction.

    The calculation stays in log space, so the common in-band case
    q * (p/q) = p is exact and cannot overflow. The same helper operates on
    the full vocabulary, the exact top-k head, and one representative token
    for the modeled top-k tail.
    """
    lo = (sampling_logprobs + math.log(c.low)
          if c.low > 0 else -math.inf)
    hi = (sampling_logprobs + math.log(c.high)
          if not math.isinf(c.high) else math.inf)
    if c.sign == "ppo":
        in_band = jnp.where(
            positive[..., None], vocab_logprobs <= hi,
            (vocab_logprobs >= lo)
            & (vocab_logprobs <= sampling_logprobs + math.log(DUAL_CLIP)))
    else:
        in_band = (vocab_logprobs >= lo) & (vocab_logprobs <= hi)
    w = jnp.exp(jnp.where(
        in_band,
        vocab_logprobs,
        jnp.clip(vocab_logprobs, lo, hi) if c.outside == "clamp" else -jnp.inf))
    if c.sign == "neg":
        w = jnp.where(positive[..., None], jnp.exp(sampling_logprobs), w)
    return w


class SamplerKL(NamedTuple):
    """KL(q || p) from the sampler to the trainer as a per-token penalty.

    Its logit gradient is p - q. The transported head contributes exact
    coordinates; a sampled tail contributes the sparse difference estimator
    (p/q - 1 on each drawn coordinate, clipped at `clip`), a modeled tail the
    trainer's own tail mass. The coefficient decays with the age of the
    token's sampler, counted in optimizer steps since its weights were
    published: coeff / sqrt(age), and 0 at age 0, where the sampler is the
    trainer itself and any difference between them is numerical."""
    coeff: float = 0.0
    clip: float = 5.0  # upper bound on the tail coefficient p/q - 1

    def coefficient(self, stale_steps):
        steps = jnp.asarray(stale_steps, jnp.float32)
        return jnp.where(steps >= 1.0, self.coeff * jax.lax.rsqrt(jnp.maximum(steps, 1.0)), 0.0)


def loss_fn_rl(hidden, head, batch, is_cfg=None, chunk=128,
               score_center=False, sampler_kl=None):
    """REINFORCE and diagnostics using logprobs recorded during sampling.

    is_cfg (ISCorrection) reweights REINFORCE with detached IS weights on the
    p/q ratio: per token (level=token; the ppo shape is gradient-identical to
    the clipped surrogate), or one weight per sequence, applied after the
    scan. level=seq_mean uses the geometric-mean ratio; level=seq_sum uses the
    full trajectory ratio (the product of token ratios). Token-level ratio
    corrections compose with score centering by integrating under the
    effective mass q*w; TV corrections deliberately do not compose with score
    centering.

    head maps pre-head activations [B, C, D] to logits [B, C, V]. It runs
    inside the loss in rematerialized sequence chunks, so the
    full-vocabulary float32 intermediates (log-softmax, centering weights and
    their cotangents) only ever cover `chunk` positions per device at a time.

    When score_center is enabled, the batch determines the estimator. Full
    sampler logprobs give exact centering; otherwise the transported top-k
    head is exact and q's tail is either sampled (tail_ids / tail_logprobs:
    M draws from q conditioned outside the head, weighted by the residual
    mass) or modeled as proportional to the trainer's p. Every ratio
    correction is constant over a modeled tail, so its effective mass is
    alpha*p and the zero-expected-score identity leaves only top-k gathers in
    the centering backward pass.

    sampler_kl (SamplerKL) adds batch["kl_coeff"] * KL(q || p) per token,
    estimated on the same payload; the coefficient is a scalar or [B, T].
    """
    token_c = is_cfg if is_cfg and is_cfg.level == "token" else None
    seq_c = (is_cfg if is_cfg and is_cfg.level in ("seq_mean", "seq_sum")
             else None)
    if is_cfg and is_cfg.level == "seq_sum" and is_cfg.stat != "ratio":
        raise ValueError("level=seq_sum requires stat=ratio")
    if score_center and is_cfg and is_cfg.stat != "ratio":
        raise ValueError("score centering composes only with stat=ratio")
    full_center = score_center and "sampling_vocab_logprobs" in batch
    topk_center = (
        score_center and not full_center and "center_ids" in batch)
    if score_center and not (full_center or topk_center):
        raise ValueError(
            "score centering requires full or top-k sampler vocab logprobs")
    sampled_tail = "tail_ids" in batch
    if sampled_tail and "center_ids" not in batch:
        raise ValueError("a sampled tail requires the top-k head payload")
    if sampler_kl is not None:
        if "center_ids" not in batch:
            raise ValueError("sampler KL requires the top-k sampler payload")
        if "kl_coeff" not in batch:
            raise ValueError("sampler KL requires batch['kl_coeff']")
    kl_coeff = batch.get("kl_coeff")
    per_token_coeff = kl_coeff is not None and jnp.ndim(kl_coeff) == 2
    _, target_tokens, target_mask = get_batch_targets(batch["tokens"], batch["mask"])
    seq_len = target_tokens.shape[1]
    chunk = next(c for c in range(min(chunk, seq_len), 0, -1) if seq_len % c == 0)

    def chunk_sums(sums, start):
        slc = lambda x: jax.lax.dynamic_slice_in_dim(x, start, chunk, axis=1)
        sg = jax.lax.stop_gradient
        logits = head(slc(hidden))
        vocab_logprobs = jax.nn.log_softmax(logits.astype(jnp.float32))
        token_logprobs = take_vocab(
            vocab_logprobs, slc(target_tokens)[..., None])[..., 0]
        log_ratio = token_logprobs - slc(batch["sampling_token_logprobs"])
        positive = batch["advantages"][:, None] >= 0
        tv = None
        if is_cfg and is_cfg.stat == "tv":
            tv = jnp.abs(jnp.exp(slc(batch["sampling_token_logprobs"]))
                         - jnp.exp(token_logprobs))
        importance_weights = 1.0
        if token_c:
            importance_weights = is_weights(log_ratio, positive, token_c, tv)
        score = importance_weights * token_logprobs
        # Negative entropy of the trainer distribution: a metric, and the
        # full-vocab factor of the modeled tail. Detached, because its logit
        # gradient is identically zero (E_p[grad log p] = 0), so backward never
        # touches the full vocabulary.
        plogp = sg((sg(jnp.exp(vocab_logprobs)) * vocab_logprobs).sum(-1))

        head_q = head_p = head_ids = None
        if "center_ids" in batch:
            head_ids = slc(batch["center_ids"])
            head_q = slc(batch["center_logprobs"])
            head_p = take_vocab(vocab_logprobs, head_ids)
            q_head = jnp.exp(head_q)
            p_head = jnp.exp(head_p)
            tail_q_mass = jnp.maximum(1.0 - q_head.sum(-1), 0.0)
            # q_tail modeled as rho*p_tail, matching its transported mass.
            rho = tail_q_mass / jnp.maximum(1.0 - p_head.sum(-1), 1e-6)
        tail_q = tail_p = tail_ids = None
        if sampled_tail:
            tail_ids = slc(batch["tail_ids"])
            tail_q = slc(batch["tail_logprobs"])
            tail_p = take_vocab(vocab_logprobs, tail_ids)

        center = None
        if full_center:
            sampling_vocab_logprobs = slc(batch["sampling_vocab_logprobs"]).astype(jnp.float32)
            cw = (centering_weights(
                      vocab_logprobs, sampling_vocab_logprobs, positive, token_c)
                  if token_c else jnp.exp(sampling_vocab_logprobs))
            center = (sg(cw) * vocab_logprobs).sum(-1)
        if topk_center:
            # Apply the same correction to the exact transported head as the
            # full path applies to every vocabulary token.
            head_cw = (centering_weights(head_p, head_q, positive, token_c)
                       if token_c else q_head)
            center = (sg(head_cw) * head_p).sum(-1)
            if sampled_tail:
                # Stratified tail: the draws are IID from q outside the head,
                # so their mean, scaled by the residual mass, is unbiased for
                # the tail's share of E_q[w * grad log p].
                tail_w = (is_weights(tail_p - tail_q, positive[..., None], token_c)
                          if token_c else 1.0)
                center += sg(tail_q_mass) * (sg(tail_w) * tail_p).mean(-1)
            else:
                # Since p/q = 1/rho is constant on the modeled tail, every
                # ratio correction produces effective tail mass alpha*p.
                # Evaluate alpha with one synthetic p=1, q=rho token so
                # band/sign semantics remain centralized in centering_weights
                # rather than being reimplemented here.
                alpha = rho
                if token_c:
                    alpha = centering_weights(
                        jnp.zeros_like(rho)[..., None],
                        jnp.log(rho)[..., None], positive, token_c)[..., 0]
                # The head gather stays in the graph: only the full sum is
                # zero-gradient, a partial sum over the head is not.
                center += sg(alpha) * (
                    plogp - (sg(p_head) * head_p).sum(-1))
        if center is not None:
            score = score - center

        mask = slc(target_mask)
        sums = dict(sums)
        if head_q is not None:
            # Head-based KL(q||p) with the tail modeled as q = rho*p (exact
            # when q's tail is proportional to p's): tail contribution is
            # tail mass * log rho.
            kl_head_exact = (q_head * (head_q - head_p)).sum(-1)
            kl_head = (kl_head_exact
                       + tail_q_mass * jnp.log(jnp.maximum(rho, 1e-9)))
            sums["kl_head"] += (kl_head * mask).sum()
            sums["head_mass"] += (q_head.sum(-1) * mask).sum()
        if sampled_tail:
            # Same KL with the tail term estimated from the draws instead.
            kl_tail = kl_head_exact + tail_q_mass * (tail_q - tail_p).mean(-1)
            sums["kl_tail"] += (kl_tail * mask).sum()
        if sampler_kl is not None:
            # Surrogate whose logit gradient is p - q_hat: detached
            # coefficients on raw logits, so the sparse tail term touches only
            # the drawn coordinates (a log-softmax gather would leak a dense
            # -(sum of coefficients) * p term).
            logits32 = logits.astype(jnp.float32)
            head_z = take_vocab(logits32, head_ids)
            kl_surrogate = (sg(p_head - q_head) * head_z).sum(-1)
            if sampled_tail:
                excess = jnp.exp(tail_p - tail_q) - 1.0  # p/q - 1 per draw
                tail_z = take_vocab(logits32, tail_ids)
                kl_surrogate += sg(tail_q_mass) * (
                    sg(jnp.minimum(excess, sampler_kl.clip)) * tail_z).mean(-1)
                kl_value = kl_tail
                live = mask[..., None]
                sums["tail_coeff_max"] = jnp.maximum(
                    sums["tail_coeff_max"],
                    jnp.where(live, jnp.abs(excess), 0.0).max())
                sums["tail_clipped"] += (
                    live & (excess > sampler_kl.clip)).sum()
            else:
                # Modeled tail: (1 - rho) * sum_tail p_v grad z_v, written
                # through the log-normalizer whose gradient is the full p.
                kl_surrogate += sg(1.0 - rho) * (
                    jax.nn.logsumexp(logits32, -1)
                    - (sg(p_head) * head_z).sum(-1))
                kl_value = kl_head
            coeff = slc(kl_coeff) if per_token_coeff else kl_coeff
            sums["kl_reg"] += (coeff * kl_surrogate * mask).sum()
            sums["kl_sampler"] += (kl_value * mask).sum()
        sums["score"] += (score * mask).sum(-1)
        sums["entropy"] += (-plogp * mask).sum()
        sums["absdiff"] += (
            jnp.abs(slc(batch["sampling_token_logprobs"]) - token_logprobs)
            * mask
        ).sum()
        # Mean of log q - log p over the sampled tokens: an estimate of KL(q || p).
        sums["kl_samp_train"] += (
            (slc(batch["sampling_token_logprobs"]) - token_logprobs) * mask
        ).sum()
        if seq_c:
            sums["log_ratio"] += (sg(log_ratio) * mask).sum(-1)
            if tv is not None:
                sums["tv"] += (sg(tv) * mask).sum(-1)
        if token_c:
            acted = band_acted(log_ratio, positive, token_c, tv) & mask
            sums["clipped_high"] += (acted & (log_ratio > 0)).sum()
            sums["clipped_low"] += (acted & (log_ratio < 0)).sum()
            if token_c.outside == "drop":
                sums["masked"] += ((importance_weights == 0.0) * mask).sum()
        return sums, None

    scalar = jnp.zeros((), jnp.float32)
    # Per-sequence accumulators inherit the advantages' [B] sharding so the
    # scan carry type is stable.
    per_seq = jnp.zeros_like(batch["advantages"], dtype=jnp.float32)
    init = {"score": per_seq, "entropy": scalar, "absdiff": scalar,
            "kl_samp_train": scalar, "masked": scalar}
    if token_c:
        init.update(clipped_high=scalar, clipped_low=scalar)
    if seq_c:
        init["log_ratio"] = per_seq
        if seq_c.stat == "tv":
            init["tv"] = per_seq
    if "center_ids" in batch:
        init.update(kl_head=scalar, head_mass=scalar)
    if sampled_tail:
        init["kl_tail"] = scalar
    if sampler_kl is not None:
        init.update(kl_reg=scalar, kl_sampler=scalar)
        # The clipping diagnostics exist only where the sparse tail estimator runs.
        if sampled_tail:
            init.update(tail_coeff_max=scalar, tail_clipped=scalar)
    sums, _ = jax.lax.scan(
        jax.checkpoint(chunk_sums), init, jnp.arange(0, seq_len, chunk))
    count = jnp.maximum(target_mask.sum(), 1)
    weight = batch["advantages"]
    frac_masked = sums["masked"] / count
    if seq_c:
        seq_count = jnp.maximum(target_mask.sum(1), 1)
        seq_log_ratio = sums["log_ratio"]
        if seq_c.level == "seq_mean":
            seq_log_ratio = seq_log_ratio / seq_count
        w_seq = is_weights(
            seq_log_ratio, batch["advantages"] >= 0, seq_c,
            sums["tv"] / seq_count if seq_c.stat == "tv" else None)
        weight = weight * w_seq
        if seq_c.outside == "drop":
            frac_masked = (w_seq == 0.0).mean()  # fraction of sequences
    loss = -(weight * sums["score"]).sum() / count
    metrics = {
        "pol_entropy": sums["entropy"] / count,
        "logp_absdiff_samp_train": sums["absdiff"] / count,
        "kl_samp_train": sums["kl_samp_train"] / count,
        "frac_masked": frac_masked,
    }
    # Fraction of tokens (or, at sequence level, of sequences) whose ratio the
    # band clipped or dropped, per side; separate from truncation at the
    # length budget, which the rollout stage reports.
    if token_c:
        metrics["frac_ratio_clipped_high"] = sums["clipped_high"] / count
        metrics["frac_ratio_clipped_low"] = sums["clipped_low"] / count
    if seq_c:
        acted = band_acted(
            seq_log_ratio, batch["advantages"] >= 0, seq_c,
            sums["tv"] / seq_count if seq_c.stat == "tv" else None)
        metrics["frac_ratio_clipped_high"] = (acted & (seq_log_ratio > 0)).mean()
        metrics["frac_ratio_clipped_low"] = (acted & (seq_log_ratio < 0)).mean()
    if "center_ids" in batch:
        metrics["kl_head"] = sums["kl_head"] / count
        metrics["head_mass"] = sums["head_mass"] / count
    if sampled_tail:
        metrics["kl_tail"] = sums["kl_tail"] / count
    if sampler_kl is not None:
        loss = loss + sums["kl_reg"] / count
        metrics["kl_sampler"] = sums["kl_sampler"] / count
        if sampled_tail:
            metrics["tail_coeff_max"] = sums["tail_coeff_max"]
            metrics["frac_tail_clipped"] = (
                sums["tail_clipped"] / (count * batch["tail_ids"].shape[-1]))
    return loss, metrics
