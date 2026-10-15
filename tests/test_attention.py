"""Sharded GQA agrees with unsharded attention, including its gradients.

Run with XLA_FLAGS=--xla_force_host_platform_device_count=4 to also exercise
tensor and data parallelism on CPU.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

from models.model import attention


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("kv_heads", [2, 8])
def test_attention_matches_unsharded_values_and_gradients(decode, kv_heads):
    devices = jax.device_count()
    tp = 2 if devices % 2 == 0 else 1
    mesh = jax.make_mesh((devices // tp, tp), ("data", "model"),
                         axis_types=(AxisType.Explicit, AxisType.Explicit))
    batch, seq, heads, width = max(4, devices), 6, 8, 8
    query_len = 1 if decode else seq
    rng = np.random.default_rng(7)
    arrays = [rng.normal(size=shape).astype(np.float32) for shape in (
        (batch, query_len, heads, width),
        (batch, seq, kv_heads, width),
        (batch, seq, kv_heads, width))]
    # Decode batches have different live KV lengths, as in the actual sampler.
    mask = (np.arange(seq)[None, None, None, :]
            <= (np.arange(batch) % seq)[:, None, None, None]) if decode else None

    with jax.set_mesh(mesh):
        replicated = [jax.device_put(a, NamedSharding(mesh, P())) for a in arrays]
        ref_mask = jax.device_put(mask, NamedSharding(mesh, P())) if decode else None

        def reference(q, k, v):
            return jax.nn.dot_product_attention(q, k, v, mask=ref_mask, is_causal=not decode)

        expected = jax.jit(reference)(*replicated)
        expected_grads = jax.jit(jax.grad(
            lambda q, k, v: jnp.square(reference(q, k, v)).sum(),
            argnums=(0, 1, 2)))(*replicated)

        sharded = [jax.device_put(a, NamedSharding(mesh, P("data", None, "model", None)))
                   for a in arrays]
        shard_mask = jax.device_put(mask, NamedSharding(mesh, P("data", None, None, None))) if decode else None

        def actual(q, k, v):
            return attention(q, k, v, mask=shard_mask, is_causal=not decode)

        got = jax.jit(actual)(*sharded)
        got_grads = jax.jit(jax.grad(
            lambda q, k, v: jnp.square(actual(q, k, v)).sum(),
            argnums=(0, 1, 2)))(*sharded)

    np.testing.assert_allclose(got, expected, atol=2e-6, rtol=2e-5)
    for got_grad, expected_grad in zip(got_grads, expected_grads):
        np.testing.assert_allclose(got_grad, expected_grad, atol=2e-6, rtol=2e-5)
