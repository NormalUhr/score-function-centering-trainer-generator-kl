"""models.model.place puts a host array onto a sharding through per-device
pieces; it must agree with jax.device_put on values and sharding for sharded
and replicated specs, on a mesh of several devices. Run with
XLA_FLAGS=--xla_force_host_platform_device_count=8 (the tests skip otherwise,
so that the other suites keep their single device)."""
import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from models.model import make_mesh, place

jax.config.update("jax_platform_name", "cpu")
pytestmark = pytest.mark.skipif(jax.device_count() < 8, reason="needs eight host devices (XLA_FLAGS)")


def test_place_matches_device_put_for_sharded_and_replicated_specs():
    mesh = make_mesh(4, 2)
    x = np.arange(8 * 6 * 4, dtype=np.float32).reshape(8, 6, 4)
    for spec in (P(("data", "model"), None, None), P("model", None, "data"), P(None, "model"), P()):
        sharding = NamedSharding(mesh, spec)
        placed = place(x, sharding)
        reference = jax.device_put(x, sharding)
        assert placed.sharding.is_equivalent_to(reference.sharding, x.ndim)
        np.testing.assert_array_equal(np.asarray(placed), x)
        assert placed.dtype == reference.dtype and placed.shape == x.shape


def test_place_keeps_a_non_contiguous_view():
    mesh = make_mesh(8, 1)
    x = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 8, 6).swapaxes(1, 2)  # (4, 6, 8) strided view
    placed = place(x, NamedSharding(mesh, P(None, None, "data")))
    np.testing.assert_array_equal(np.asarray(placed), np.ascontiguousarray(x))


def test_place_accepts_a_partition_spec_under_the_current_mesh():
    mesh = make_mesh(4, 2)
    x = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    with jax.set_mesh(mesh):
        placed = place(x, P(("data", "model"), None))
    assert placed.sharding.is_equivalent_to(NamedSharding(mesh, P(("data", "model"), None)), x.ndim)
    np.testing.assert_array_equal(np.asarray(placed), x)
