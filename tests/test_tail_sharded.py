"""Head/tail collection on a 4 x 2 (data x model) mesh of host devices: exact
head under a TP-sharded vocabulary, masking by id rather than by value so ties
at the head boundary stay in the tail, and tail logprobs from the full q."""
import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

from models import topk


def main():
    assert jax.device_count() == 8, jax.devices()
    mesh = jax.make_mesh((4, 2), ("data", "model"),
                         axis_types=(AxisType.Explicit, AxisType.Explicit))
    batch, vocab, head_k, num_tail, seq_len = 8, 64, 4, 3, 2
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(batch, vocab)).astype(np.float32)
    # Force a three-way tie at the head boundary in every row: two tokens from
    # far down the ranking take the fourth-largest logit, so exactly one of
    # the three tied tokens is head and the other two are tail.
    order = np.argsort(-logits, axis=-1)
    tied = order[:, [3, 20, 40]]
    for row in range(batch):
        logits[row, tied[row]] = logits[row, tied[row, 0]]
    with jax.set_mesh(mesh):
        logits = jax.device_put(logits, NamedSharding(mesh, P("data", "model")))
        logprobs = jax.nn.log_softmax(logits)
        carry = topk.init_tail(batch, seq_len, head_k, num_tail)
        observe = jax.jit(topk.observe_tail, static_argnames="position")
        carry = observe(carry, 1, jax.random.key(0), logits, logprobs)

    full = np.asarray(logprobs)
    head_ids = np.asarray(carry.head_ids)[:, 1]
    head_logprobs = np.asarray(carry.head_logprobs)[:, 1]
    tail_ids = np.asarray(carry.tail_ids)[:, 1]
    tail_logprobs = np.asarray(carry.tail_logprobs)[:, 1]
    assert np.asarray(carry.head_ids)[:, 0].max() == 0  # untouched position
    # Exact head values (ids may pick either tied token).
    assert np.allclose(np.sort(head_logprobs, -1),
                       np.sort(np.partition(full, -head_k, -1)[:, -head_k:], -1))
    assert np.allclose(np.take_along_axis(full, head_ids, -1), head_logprobs)
    assert len(set(head_ids[0])) == head_k
    # Tail draws avoid the head ids and carry full-q logprobs.
    assert not np.any(tail_ids[:, :, None] == head_ids[:, None, :])
    assert np.allclose(np.take_along_axis(full, tail_ids, -1), tail_logprobs)
    # The tied token that was NOT selected into the head must remain a
    # possible tail draw: over many keys it appears with its q probability.
    with jax.set_mesh(mesh):
        draws = np.concatenate([
            np.asarray(observe(carry, 1, jax.random.key(seed), logits,
                               logprobs).tail_ids)[:, 1]
            for seed in range(400)], axis=-1)
    for row in range(batch):
        in_head = [token for token in tied[row] if token in head_ids[row]]
        assert len(in_head) == 1, (row, tied[row], head_ids[row])
        q = np.exp(full[row])
        tail_mass = 1.0 - q[head_ids[row]].sum()
        for tied_out in set(tied[row]) - set(in_head):
            expected = q[tied_out] / tail_mass
            observed = np.mean(draws[row] == tied_out)
            standard_error = np.sqrt(expected * (1 - expected) / draws.shape[1])
            assert abs(observed - expected) < 5 * standard_error, (
                row, tied_out, observed, expected, standard_error)
    print("ok: sharded head/tail collection, exact head, id-exact masking "
          "with boundary ties, tail draws from q outside the head")


if __name__ == "__main__":
    main()
