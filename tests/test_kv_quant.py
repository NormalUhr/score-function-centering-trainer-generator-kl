"""kv_fake_quantize: grid exactness and scaling for int-N / fp8 / fp6 / fp4."""
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from models.quant import kv_fake_quantize

X = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 1, 8, 128), jnp.bfloat16) * 3.0


def rel_err(spec):
    y = np.asarray(kv_fake_quantize(X, spec), np.float32)
    x = np.asarray(X, np.float32)
    return np.sqrt(((y - x) ** 2).mean() / (x ** 2).mean()), y


def test_float_grids_are_exact_after_unscaling():
    for spec, mdt in (("fp6", ml_dtypes.float6_e3m2fn), ("fp4", jnp.float4_e2m1fn)):
        y = np.asarray(kv_fake_quantize(X, spec), np.float32)
        gmax = float(ml_dtypes.finfo(mdt).max)
        scale = np.abs(np.asarray(X, np.float32)).max(-1, keepdims=True) / gmax
        z = y / np.maximum(scale, 1e-9)
        # bf16 container rounds the rescaled product; snap back before checking
        z16 = z.astype(np.float32)
        grid = z16.astype(mdt).astype(np.float32)
        np.testing.assert_allclose(z16, grid, rtol=2e-2, atol=1e-2)


def test_intn_is_the_per_token_head_absmax_grid():
    for bits in (8, 6, 4):
        y = np.asarray(kv_fake_quantize(X, bits), np.float32)
        u = np.asarray(X, np.float32)
        qmax = 2 ** (bits - 1) - 1
        scale = np.abs(u).max(-1, keepdims=True) / qmax
        ref = (np.round(u / np.maximum(scale, 1e-9)) * scale).astype(ml_dtypes.bfloat16)
        np.testing.assert_array_equal(y, ref.astype(np.float32))


def test_severity_ordering():
    errs = {spec: rel_err(spec)[0] for spec in (8, "fp8", 6, "fp6", 4, "fp4")}
    assert errs[8] < errs[6] < errs[4]           # int ladder
    assert errs["fp8"] < errs["fp6"] < errs["fp4"]  # float ladder
    assert errs["fp6"] > errs[6]  # 2 mantissa bits < 5 effective int bits
    for spec, e in errs.items():
        assert 0 < e < 1, (spec, e)


def test_not_identity_under_jit():
    snap = jax.jit(lambda v: kv_fake_quantize(v, "fp6"))
    assert not np.array_equal(np.asarray(snap(X)), np.asarray(X))
