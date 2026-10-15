"""Unit checks for the 8-bit sampler weights (models/quant.py)."""
import jax
import jax.numpy as jnp
import numpy as np

from models import quant

jax.config.update("jax_platform_name", "cpu")


def _rand(shape, seed=0):
    return jax.random.normal(jax.random.key(seed), shape, dtype=jnp.float32)


def _mesh():
    return jax.make_mesh((1, 1), ("data", "model"),
                         axis_types=(jax.sharding.AxisType.Explicit,) * 2)


def test_quantize_tree_skip_rules_and_scales():
    weights = {
        "layers.0.q_proj.weight": _rand((64, 32), 1),
        "layers.0.o_proj.weight": _rand((4, 16, 32), 2),  # contracts trailing 2 axes
        "layers.0.mlp.experts.weight": _rand((4, 32, 16), 3),  # contracts middle axis
        "layers.0.norm.weight": _rand((32,), 4),
        "layers.0.q_proj.bias": _rand((64,), 5),
        "layers.0.mlp.gate.weight": _rand((32, 4), 6),
    }
    with jax.set_mesh(_mesh()):
        out = quant.quantize_tree(weights, jnp.int8)
    for k, w in weights.items():
        untouched = w.ndim <= 1 or "norm" in k or "bias" in k or k.endswith("mlp.gate.weight")
        if untouched:
            np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(w))
        else:
            assert isinstance(out[k], quant.QArray) and out[k].q.shape == w.shape
            assert out[k].q.dtype == jnp.int8
    assert out["layers.0.q_proj.weight"].scale.shape == (64,)
    assert out["layers.0.o_proj.weight"].scale.shape == (4,)
    assert out["layers.0.mlp.experts.weight"].scale.shape == (4, 16)
    # Absmax rounding: every dequantized entry is within half a step of the original.
    q = out["layers.0.q_proj.weight"]
    dequantized = np.asarray(q.q, np.float32) * np.asarray(q.scale)[:, None]
    assert (np.abs(dequantized - np.asarray(weights["layers.0.q_proj.weight"]))
            <= np.asarray(q.scale)[:, None] / 2 + 1e-6).all()


def test_shared_expert_weights_quantize_as_dense_matrices():
    # GLM-style shared experts are ordinary (out, in) matrices named
    # mlp.shared_experts.*; only the stacked mlp.experts.* tensors contract
    # their middle axis.
    w = _rand((48, 32), 7)
    with jax.set_mesh(_mesh()):
        q = quant.quantize_tree({"layers.0.mlp.shared_experts.gate_proj.weight": w}, jnp.int8)
    # One scale per output row, as for any dense matrix; the swapped expert
    # layout would give one per input column.
    assert q["layers.0.mlp.shared_experts.gate_proj.weight"].scale.shape == (w.shape[0],)


def test_qeinsum_matches_the_dequantized_product():
    x = _rand((2, 3, 32), 8)
    w = _rand((16, 32), 9)
    with jax.set_mesh(_mesh()):
        qw = quant.quantize_tree({"layers.0.q_proj.weight": w}, jnp.float8_e4m3fn)["layers.0.q_proj.weight"]
        y = quant.qeinsum("btd,fd->btf", x, qw, preferred_element_type=jnp.float32)
    exact = jnp.einsum("btd,fd->btf", x, w)
    assert y.shape == exact.shape
    assert float(jnp.abs(y - exact).max()) < 0.1 * float(jnp.abs(exact).max())
    assert float(jnp.abs(y - exact).max()) > 0
