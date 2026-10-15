"""Correctness checks for the MoE ragged-dot backward."""

import jax
import jax.numpy as jnp
import numpy as np

from models import quant


def test_empty_expert_weight_gradients_are_zero():
    """An expert with no routed rows has an exactly zero weight gradient."""
    sizes = jnp.array([3, 0, 2, 0], dtype=jnp.int32)
    x = jax.random.normal(jax.random.key(0), (5, 7))
    w = jax.random.normal(jax.random.key(1), (4, 7, 6))
    dout = jax.random.normal(jax.random.key(2), (5, 6))
    expert_ids = jnp.array([0, 0, 0, 2, 2], dtype=jnp.int32)

    actual = jax.value_and_grad(
        lambda lhs, rhs: jnp.vdot(
            quant.qragged_dot(lhs, rhs, sizes, expert_ids), dout),
        argnums=(0, 1))(x, w)
    reference = jax.value_and_grad(
        lambda lhs, rhs: jnp.vdot(
            jax.lax.ragged_dot(lhs, rhs, sizes), dout),
        argnums=(0, 1))(x, w)
    actual_loss, (actual_dx, actual_dw) = actual
    reference_loss, (reference_dx, reference_dw) = reference

    assert bool(jnp.all(jnp.isfinite(actual_dx)))
    assert bool(jnp.all(jnp.isfinite(actual_dw)))
    np.testing.assert_array_equal(
        np.asarray(actual_dw[jnp.array([1, 3])]), 0)
    # Triton and XLA choose different TF32 tilings for the active groups.
    np.testing.assert_allclose(
        np.asarray(actual_loss), np.asarray(reference_loss), rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(
        np.asarray(actual_dx), np.asarray(reference_dx), rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(
        np.asarray(actual_dw), np.asarray(reference_dw), rtol=2e-2, atol=2e-3)
