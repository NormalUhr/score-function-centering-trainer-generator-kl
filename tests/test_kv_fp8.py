"""fp8 (e4m3) KV-cache snap: not a jit no-op, exactly e4m3-representable."""
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from models import quant


def test_fp8_kv_snap_is_e4m3_grid_not_identity():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 4, 1, 8, 128), jnp.bfloat16) * 3.0

    snap = jax.jit(lambda v: quant._snap_float(
        v.astype(jnp.float32), jnp.float8_e4m3fn).astype(v.dtype))
    y = np.asarray(snap(x), np.float32)

    assert not np.array_equal(y, np.asarray(x, np.float32))  # no silent no-op
    roundtrip = y.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    np.testing.assert_array_equal(y, roundtrip)  # every value on the e4m3 grid


def test_fp8_kv_snap_saturates_at_448():
    x = jnp.array([1000.0, -1000.0, 447.9, 0.0], jnp.float32)
    y = np.asarray(quant._snap_float(x, jnp.float8_e4m3fn))
    assert y[0] == 448.0 and y[1] == -448.0 and y[3] == 0.0
