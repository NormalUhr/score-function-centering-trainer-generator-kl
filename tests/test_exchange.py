"""Tests of the exchange format for published weights: one
weights-<part>.safetensors file per trainer process holding the shards on
that process's devices (with their slices in the file's metadata), and a
reader that reassembles every tensor from all parts."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType
from safetensors import safe_open

from models.model import _get_sharding, exchange_files, load_exchange, make_mesh, place, save_exchange


def tree():
    key = jax.random.key(0)
    return {
        "model.layers.mlp.up_proj.weight": jax.random.normal(key, (6, 40), jnp.float32).astype(jnp.bfloat16),
        "model.layers.mlp.down_proj.weight": jax.random.normal(key, (2, 3, 40), jnp.float32).astype(jnp.bfloat16),
        "model.layers.input_layernorm.weight": jnp.linspace(0.5, 1.5, 40, dtype=jnp.float32),
        "model.norm.weight": jnp.linspace(1.5, 0.5, 40, dtype=jnp.float32),
        "model.embed_tokens.weight": jnp.arange(2 * 40, dtype=jnp.float32).reshape(2, 40).astype(jnp.bfloat16),
    }


def single_device_mesh():
    return jax.make_mesh((1, 1), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))


def test_single_process_round_trip_keeps_values_and_dtypes(tmp_path):
    saved = tree()
    with jax.set_mesh(single_device_mesh()):
        save_exchange(saved, tmp_path)
        assert [path.name for path in exchange_files(tmp_path)] == ["weights-0.safetensors"]
        assert not [path for path in tmp_path.iterdir() if path.name.startswith("tmp")]
        with safe_open(str(tmp_path / "weights-0.safetensors"), framework="numpy") as f:
            assert "layout" in f.metadata() and all("@" in key for key in f.keys())
        loaded = load_exchange(tmp_path, dp_shard=False)
    assert set(loaded) == set(saved)
    for name, value in saved.items():
        assert loaded[name].dtype == value.dtype and loaded[name].shape == value.shape, name
        np.testing.assert_array_equal(np.asarray(loaded[name]), np.asarray(value))


def test_float_dtype_casts_matrices_only(tmp_path):
    saved = {"model.layers.mlp.up_proj.weight": jnp.ones((4, 8), jnp.float32),
             "model.layers.input_layernorm.weight": jnp.ones((8,), jnp.float32)}
    with jax.set_mesh(single_device_mesh()):
        save_exchange(saved, tmp_path, float_dtype=jnp.bfloat16)
        loaded = load_exchange(tmp_path, dp_shard=False)
    assert loaded["model.layers.mlp.up_proj.weight"].dtype == jnp.bfloat16
    assert loaded["model.layers.input_layernorm.weight"].dtype == jnp.float32


@pytest.mark.skipif(jax.device_count() < 8, reason="needs eight host devices (XLA_FLAGS)")
def test_two_device_groups_write_their_shards_and_the_reader_reassembles(tmp_path):
    """Two simulated processes, each owning four of eight devices, write the
    shards of a tensor that is split across them; the reader joins the parts."""
    mesh = make_mesh(4, 2)
    names = ["model.layers.mlp.up_proj.weight", "model.layers.self_attn.q_proj.weight",
             "model.layers.input_layernorm.weight", "model.embed_tokens.weight"]
    host = {names[0]: np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4).astype(jnp.bfloat16),
            names[1]: np.arange(2 * 4 * 2 * 8, dtype=np.float32).reshape(2, 4, 2, 8).astype(jnp.bfloat16),
            names[2]: np.linspace(0.5, 1.5, 2 * 4, dtype=np.float32).reshape(2, 4),
            names[3]: np.arange(6 * 8, dtype=np.float32).reshape(6, 8).astype(jnp.bfloat16)}
    with jax.set_mesh(mesh):
        saved = {name: place(value, _get_sharding(name, dp_shard=True)) for name, value in host.items()}
        devices = list(mesh.devices.flat)
        save_exchange(saved, tmp_path, part=0, devices=devices[:4])
        save_exchange(saved, tmp_path, part=1, devices=devices[4:])
        assert [path.name for path in exchange_files(tmp_path)] == ["weights-0.safetensors", "weights-1.safetensors"]
        loaded = load_exchange(tmp_path, dp_shard=False)
    for name, value in host.items():
        np.testing.assert_array_equal(np.asarray(loaded[name]).astype(np.float32), np.asarray(value).astype(np.float32))
        assert loaded[name].dtype == value.dtype


@pytest.mark.skipif(jax.device_count() < 8, reason="needs eight host devices (XLA_FLAGS)")
def test_missing_part_is_refused(tmp_path):
    mesh = make_mesh(8, 1)
    with jax.set_mesh(mesh):
        saved = {"model.layers.mlp.up_proj.weight": place(np.ones((2, 8, 16), np.float32), _get_sharding("model.layers.mlp.up_proj.weight", dp_shard=True))}
        save_exchange(saved, tmp_path, part=0, devices=list(mesh.devices.flat)[:4])
        with pytest.raises(ValueError, match="cover"):
            load_exchange(tmp_path, dp_shard=False)


def test_load_exchange_without_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_exchange(tmp_path, dp_shard=False)
