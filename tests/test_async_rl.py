"""Tests of the asynchronous pipeline: the exchange helpers (weights mailbox
and rollout queue), the staged decode with per-position sampler versions
against a teacher-forced oracle, and the two roles end to end on CPU with a
tiny random model."""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType

import async_rl
import train_rl
import losses
import models
from models import sampling

WEIGHTS_DIR = os.path.expanduser("~/.cache/postax/weights")


def exchange():
    """The mailbox keeps the newest versions and restores dtypes; the queue is
    first in, first out, atomic, and empty when drained; the KL coefficient
    takes an array of ages."""
    mesh = jax.make_mesh((1, 1), ("data", "model"),
                         axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    with tempfile.TemporaryDirectory() as root:
        box = async_rl.Mailbox(Path(root) / "weights", keep_versions=2)
        assert box.latest() is None
        weights = {"model.embed_tokens.weight": jnp.ones((4, 8), jnp.bfloat16),
                   "model.norm.weight": jnp.arange(8, dtype=jnp.float32)}
        for version in (0, 2, 4):
            box.publish(version, weights)
        assert box.latest() == 4
        kept = sorted(p.name for p in (Path(root) / "weights").iterdir() if p.is_dir())
        assert kept == ["v000002", "v000004"], kept
        loaded = box.load(4, dp_shard=False)
        assert loaded["model.norm.weight"].dtype == jnp.float32
        assert np.array_equal(np.asarray(loaded["model.norm.weight"]), np.arange(8))
        assert loaded["model.embed_tokens.weight"].dtype == jnp.bfloat16

        queue = async_rl.RolloutQueue(Path(root) / "queue")
        assert queue.size() == 0 and queue.pop_oldest(timeout=0.2) is None
        written = queue.push({"a": np.arange(3)}, generator_index=1, batch_index=0)
        files = [p for p in (Path(root) / "queue").iterdir() if not p.name.startswith("tmp")]
        assert written == files[0].stat().st_size > 0, (written, files)  # the bytes of the npz, for the timing table
        queue.push({"a": np.arange(5)}, generator_index=0, batch_index=0)
        assert queue.size() == 2
        first = queue.pop_oldest(timeout=1.0)
        assert first["a"].shape == (3,), first
        assert queue.last_read_s >= 0.0, "a read records its time for the timing table"
        assert queue.size() == 1
        assert not [p for p in (Path(root) / "queue").iterdir() if p.name.startswith("tmp")]
    coefficient = losses.SamplerKL(coeff=0.1).coefficient(np.array([0, 1, 4]))
    assert np.allclose(np.asarray(coefficient), [0.0, 0.1, 0.05]), coefficient
    print("ok: mailbox, queue, coefficient over an array of ages")


def versions(rows=0):
    """Every prediction position carries the version of the weights that
    produced it: the staged decode swaps weights after two stages, and the
    logged log-probabilities of the positions decoded under the first version
    match teacher forcing under it. With `rows` the batch is decoded in row
    slices of that size, each through all its stages before the next, so the
    later slices see only the second version."""
    model = models.load("Qwen/Qwen3-0.6B-Base", WEIGHTS_DIR, init="random",
                        hidden_size=128, num_hidden_layers=2, key=jax.random.key(0))
    other = jax.tree.map(
        lambda w: w + 0.05 * jax.random.normal(jax.random.key(9), w.shape, w.dtype)
        if w.ndim > 1 else w, model.weights)
    rng = np.random.default_rng(0)
    prompt_lengths = np.asarray([3, 7, 12, 5])
    seq_len = 32
    tokens = np.full((len(prompt_lengths), seq_len), model.tokenizer.pad_token_id, np.int32)
    for row, length in enumerate(prompt_lengths):
        tokens[row, :length] = rng.integers(10, 5000, length)
    calls = []

    def current_weights():
        calls.append(1)
        return (0, model.weights) if len(calls) <= 2 else (1, other)

    payload = dict(logprobs_topk=4)
    generation, version_at, used, row_stages = async_rl.decode_batch(
        model, tokens, prompt_lengths, jax.random.key(1), payload, chunk_tokens=5,
        current_weights=current_weights, eos_id=-1, rows=rows)
    generated = np.asarray(generation.generated)
    assert np.array_equal(prompt_lengths + generated, np.full(4, seq_len)), generated
    # Stages per slice: the slice's rows all reach the buffer end, and the
    # slice runs until its longest-prompted row has done so.
    starts = list(range(0, len(prompt_lengths), rows or len(prompt_lengths)))
    stages = [math.ceil((seq_len - prompt_lengths[s:s + (rows or len(prompt_lengths))].min()) / 5)
              for s in starts]
    assert used == [0, 0] + [1] * (sum(stages) - 2), (used, stages)
    # The active row-stages decoded: at least one row per stage, at most the slice's rows.
    assert len(used) <= row_stages <= len(used) * (rows or len(prompt_lengths)), (row_stages, len(used))
    full = np.full(len(prompt_lengths), seq_len)
    reference = np.asarray(sampling.generate(
        jax.random.key(3), model, generation.tokens, full, eos_id=-1, weights=model.weights,
        **payload).token_logprobs)
    logp = np.asarray(generation.token_logprobs)
    offset = 0
    for index, start in enumerate(starts):
        for row in range(start, min(start + (rows or len(prompt_lengths)), len(prompt_lengths))):
            length = prompt_lengths[row]
            last = length + generated[row] - 2  # last position that predicted a token
            assert (version_at[row, :length - 1] == -1).all(), version_at[row]
            assert (version_at[row, last + 1:] == -1).all(), version_at[row]
            # Stage k of this slice covers this row's tokens k*5 .. k*5+4.
            for stage in range(stages[index]):
                lo, hi = length - 1 + 5 * stage, min(length - 1 + 5 * (stage + 1), last + 1)
                if lo <= last:
                    assert (version_at[row, lo:hi] == used[offset + stage]).all(), (row, stage, version_at[row])
            # Positions decoded under version 0 (a consistent cache) match
            # teacher forcing under version 0; later positions mix the two
            # versions through the carried cache and are not re-derivable.
            for position in range(length - 1, last + 1):
                if version_at[row, position] == 0:
                    assert np.isclose(logp[row, position], reference[row, position], atol=1e-4), (row, position)
        offset += stages[index]
    print(f"ok: staged decode carries the producing version at every position (rows={rows})")


def split_batch():
    """A generator batch is cut into trainer slices along the row axis on
    prompt-group boundaries; per-batch scalars are shared by every slice."""
    data = {"tokens": np.arange(12).reshape(4, 3), "rewards": np.array([1., 0., 1., 1.]),
            "prompt_lengths": np.array([5, 6, 7, 8]),
            "tail_ids": np.arange(12).reshape(4, 3, 1), "tail_logprobs": -np.ones((4, 3, 1)),
            "frac_parsed": np.float32(0.5), "versions_used": np.array([3, 4])}
    slices = async_rl.split_batch(data, rows=2)
    assert len(slices) == 2 and slices[1]["tokens"].tolist() == [[6, 7, 8], [9, 10, 11]], slices
    # The sampled tail is a per-row payload like the head, so it is cut with the rows.
    assert slices[1]["tail_ids"].shape == (2, 3, 1) and slices[1]["tail_ids"][0, 0, 0] == 6, slices[1]
    assert slices[0]["tail_logprobs"].shape == (2, 3, 1)
    assert slices[0]["rewards"].tolist() == [1., 0.] and slices[1]["frac_parsed"] == np.float32(0.5)
    assert slices[0]["versions_used"].tolist() == [3, 4]
    assert slices[1]["prompt_lengths"].tolist() == [7, 8], slices[1]
    # A slice must split evenly over the trainer's data-parallel devices; the
    # check is shared with the synchronous loop and runs before any model loads.
    train_rl.check_slice_rows(128, 8)
    for rows, devices in ((6, 4), (64, 6)):
        try:
            train_rl.check_slice_rows(rows, devices)
            raise AssertionError((rows, devices))
        except ValueError as error:
            assert "data parallelism" in str(error), error
    print("ok: a generator batch splits into trainer slices")


def end_to_end(extra=(), steps=3, keep_run_dir=None):
    """One generator and one trainer on CPU: the trainer consumes the queue,
    publishes a version per step, logs the age distribution, and both exit;
    later batches carry ages above zero. With `async.trainer_prompts` below
    the generator's batch, every slice of a batch is its own step and version.
    With `keep_run_dir`, the run directory is copied there and returned."""
    with tempfile.TemporaryDirectory() as root:
        common = [
            sys.executable, str(ROOT / "async_rl.py"),
            "model.source=Qwen/Qwen3-0.6B-Base", "model.init=random",
            "model.hidden_size=128", "model.num_hidden_layers=2",
            f"model.weights_dir={WEIGHTS_DIR}", "dtypes.trainer=float32",
            "dtypes.master=float32", "rl.num_prompts=2", "rl.group_size=2",
            "rl.seq_len=160", "rl.sc=true", "sampler.vocab_logprobs=8",
            "rl.sampler_kl.coeff=0.1",
            "rl.is.level=token", "rl.is.high=2.0", "rl.is.outside=clamp",
            f"stop.steps={steps}", "eval.enabled=true", "eval.every_steps=2", "eval.num_prompts=2",
            "env.args.num_eval_examples=2",  # countdown's own evaluation split, kept small
            "log.wandb_mode=disabled",
            f"log.dir={root}/out", "log.project=async-test", "log.run_name=e2e",
            f"async.exchange_dir={root}/exchange", "async.chunk_tokens=8",
            "async.queue_depth=1", "async.generators=1", "run.seed=1", *extra]
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "WANDB_MODE": "disabled"}
        generator_log = open(f"{root}/generator.log", "w")
        generator = subprocess.Popen(
            common + ["role=generator", "async.generator_index=0"], env=env,
            stdout=generator_log, stderr=subprocess.STDOUT)
        trainer = subprocess.run(common + ["role=trainer"], env=env, capture_output=True,
                                 text=True, timeout=1500)
        try:
            generator.wait(timeout=300)
        finally:
            generator_log.close()
        generator_text = Path(f"{root}/generator.log").read_text()
        assert trainer.returncode == 0, trainer.stdout[-3000:] + trainer.stderr[-3000:]
        assert generator.returncode == 0, generator_text[-3000:]
        rows = [json.loads(line) for line in open(f"{root}/out/async-test/e2e/logs.jsonl")]
        assert [row["step"] for row in rows] == list(range(steps)), rows
        assert rows[-1]["version"] == steps, rows[-1]
        for row in rows:
            assert 0 <= row["frac_age0"] <= 1 and row["age_max"] >= row["age_mean"] >= 0, row
            assert row["versions_in_batch"] >= 1 and math.isfinite(row["loss"]), row
        assert any(row["age_max"] > 0 for row in rows), rows
        # Evaluation runs at steps 0, 2 (every 2) and the last step.
        assert all(("eval_reward" in row) == (row["step"] % 2 == 0 or row["step"] == steps - 1) for row in rows), rows
        if "async.trainer_prompts=1" in extra:
            # Two prompts per generator batch, one per trainer step: steps 0 and 1
            # are the two slices of the first batch, with one version between them
            # and no second wait for the queue.
            assert rows[0]["batch_versions_used"] == rows[1]["batch_versions_used"], rows[:2]
            assert rows[1]["version"] == rows[0]["version"] + 1 and rows[1]["wait_s"] == 0.0, rows[:2]
            # Batch 0 was decoded entirely under version 0; its second slice is
            # consumed at version 1, so every token there is exactly one step old.
            assert rows[1]["frac_age0"] == 0.0 and rows[1]["age_max"] == 1, rows[1]
            assert rows[2]["batch_versions_used"] != rows[1]["batch_versions_used"] or rows[2]["wait_s"] > 0, rows[1:3]
        exchange = Path(root) / "exchange"
        assert (exchange / "DONE").exists()
        # At most one batch may be left: the one the generator finished while
        # the trainer ran its final evaluation and wrote DONE.
        assert len(list((exchange / "queue").glob("*.npz"))) <= 1
        assert "generator 0 done after" in generator_text, generator_text[-2000:]
        if keep_run_dir is not None:
            shutil.copytree(f"{root}/out/async-test/e2e", keep_run_dir)
    print(f"ok: generator and trainer end to end on CPU ({steps} steps{', sliced' if extra else ''})")
    return keep_run_dir


def rejections():
    """Settings the asynchronous roles cannot honour are refused before any
    model loads: a set staleness, a trainer slice that does not divide the
    batch, a full-vocabulary payload, a trainer over several hosts without a
    coordinator, and a missing exchange directory."""
    base = [sys.executable, str(ROOT / "async_rl.py"), "model.source=Qwen/Qwen3-0.6B-Base",
            "model.init=random", "model.hidden_size=128", "model.num_hidden_layers=2",
            f"model.weights_dir={WEIGHTS_DIR}", "rl.sc=true", "sampler.vocab_logprobs=8",
            "log.wandb_mode=disabled", "role=trainer"]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "WANDB_MODE": "disabled"}
    with tempfile.TemporaryDirectory() as root:
        exchange = f"async.exchange_dir={root}/x"
        for extra, message in (
                ([exchange, "sampler.staleness=1"], "sampler.staleness"),
                ([exchange, "rl.num_prompts=4", "async.trainer_prompts=3"], "async.trainer_prompts"),
                ([exchange, "sampler.vocab_logprobs=full"], "full vocabulary"),
                ([exchange, "async.trainer_hosts=2"], "async.coordinator"),
                ([], "async.exchange_dir")):
            result = subprocess.run(base + extra, env=env, capture_output=True, text=True,
                                    timeout=300)
            assert result.returncode != 0 and message in result.stderr, (extra, result.stderr[-800:])
    print("ok: unsupported asynchronous settings are refused")


if __name__ == "__main__":
    exchange()
    versions()
    versions(rows=3)  # uneven slices: 3 + 1 rows
    rejections()
    split_batch()
    end_to_end()
    end_to_end(["async.trainer_prompts=1", "async.decode_rows=1"], steps=4)
    # The head-and-tail payload: one draw from the sampler's tail per position
    # travels through the staged decode and the queue to the trainer's loss.
    run_dir = end_to_end(["sampler.tail_samples=1"], steps=3, keep_run_dir=tempfile.mkdtemp() + "/tail")
    rows = [json.loads(line) for line in open(f"{run_dir}/logs.jsonl")]
    assert all("frac_tail_clipped" in row and 0 <= row["frac_tail_clipped"] <= 1 for row in rows), rows[0]
    shutil.rmtree(Path(run_dir).parent)


def test_publication_mesh_follows_sampler_shard():
    """The trainer publishes in the synchronous sampler's layout: tensor
    parallelism from sampler.shard.tp (all devices by default) and data
    parallelism over the rest, so a model whose key-value heads do not split
    over every device still publishes (Qwen3-30B-A3B has 4, fewer than 8
    devices)."""
    code = (
        "import os; os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'; "
        "os.environ['JAX_PLATFORMS'] = 'cpu'; import sys; sys.path.insert(0, %r); "
        "from omegaconf import OmegaConf; import async_rl; "
        "print(dict(async_rl.publication_mesh(OmegaConf.create({'tp': 2, 'dp': None})).shape)); "
        "print(dict(async_rl.publication_mesh(OmegaConf.create({'tp': None, 'dp': None})).shape))"
        % str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.splitlines() == ["{'data': 2, 'model': 2}", "{'data': 1, 'model': 4}"]


def test_load_exchange_returns_the_saved_tree(tmp_path):
    """`load_exchange` gives back every tensor `save_exchange` wrote, with its
    dtype and shape, including bf16 matrices and a float32 vector."""
    from models.model import load_exchange, save_exchange
    key = jax.random.key(0)
    saved = {
        "model.layers.mlp.up_proj.weight": jax.random.normal(key, (6, 40), jnp.float32).astype(jnp.bfloat16),
        "model.layers.input_layernorm.weight": jnp.linspace(0.5, 1.5, 40, dtype=jnp.float32),
        "model.embed_tokens.weight": jnp.arange(2 * 40, dtype=jnp.float32).reshape(2, 40).astype(jnp.bfloat16),
    }
    save_exchange(saved, tmp_path)
    mesh = jax.make_mesh((1, 1), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    with jax.set_mesh(mesh):
        loaded = load_exchange(tmp_path, dp_shard=False)
    assert set(loaded) == set(saved)
    for name, value in saved.items():
        assert loaded[name].dtype == value.dtype, name
        assert loaded[name].shape == value.shape, name
        np.testing.assert_array_equal(np.asarray(loaded[name]), np.asarray(value))


def test_read_safetensors_in_chunks(tmp_path):
    """The exchange reader splits every tensor into chunks read by several
    threads, and the chunks join into the saved bytes."""
    from safetensors.numpy import save_file
    from models.model import _read_safetensors
    saved = {"model.embed_tokens.weight": jnp.arange(3 * 1000, dtype=jnp.float32).reshape(3, 1000).astype(jnp.bfloat16),
             "model.norm.weight": jnp.arange(1000, dtype=jnp.float32)}
    save_file({name: np.asarray(value) for name, value in saved.items()}, str(tmp_path / "weights.safetensors"))
    loaded = dict(_read_safetensors(str(tmp_path / "weights.safetensors"), workers=3, chunk_bytes=1000))
    for name, value in saved.items():
        assert loaded[name].dtype == value.dtype
        np.testing.assert_array_equal(loaded[name], np.asarray(value))


def test_queue_oldest_read_leaves_the_entry_until_removed(tmp_path):
    """The processes of a multi-host trainer all read the same oldest batch
    before one of them removes it: `oldest` returns the entry's name and
    arrays without deleting anything, and `remove` deletes by name."""
    queue = async_rl.RolloutQueue(tmp_path)
    assert queue.oldest(timeout=0.2) is None
    queue.push({"a": np.arange(3)}, generator_index=1, batch_index=0)
    queue.push({"a": np.arange(5)}, generator_index=0, batch_index=0)
    name, arrays = queue.oldest(timeout=1.0)
    assert arrays["a"].shape == (3,) and queue.size() == 2
    again, _ = queue.oldest(timeout=1.0)
    assert again == name
    queue.remove(name)
    assert queue.size() == 1
    assert queue.oldest(timeout=1.0)[1]["a"].shape == (5,)
    assert queue.pop_oldest(timeout=1.0)["a"].shape == (5,) and queue.size() == 0


def test_wait_for_parts_returns_once_every_part_is_present(tmp_path):
    """Process 0 renames a staging directory only when every process's part
    file is there; a missing part raises after the timeout."""
    (tmp_path / "weights-0.safetensors").write_bytes(b"")
    try:
        async_rl.wait_for_parts(tmp_path, parts=2, timeout=0.3, poll=0.05)
        raise AssertionError("returned with a part missing")
    except TimeoutError as error:
        assert "weights-1.safetensors" in str(error)
    (tmp_path / "weights-1.safetensors").write_bytes(b"")
    async_rl.wait_for_parts(tmp_path, parts=2, timeout=0.3, poll=0.05)


def test_split_eval_holds_out_the_head_of_the_training_set():
    """The first `num_prompts` rows of the shuffled training set are the
    evaluation rows, and the rest trains."""
    from datasets import Dataset
    train = Dataset.from_list([{"prompt": f"t{index}", "answer": str(index)} for index in range(6)])
    eval_rows, remaining = async_rl.split_eval(train, num_prompts=2)
    assert [row["prompt"] for row in eval_rows] == ["t0", "t1"]
    assert [row["prompt"] for row in remaining] == [f"t{index}" for index in range(2, 6)]


def test_eval_chunks_pad_the_last_chunk_to_the_batch():
    """Evaluation rows are cut into batches of `num_prompts`; a short last
    batch repeats its own rows to keep one compiled shape, and the count of
    real rows says how many results to keep."""
    rows = [f"r{index}" for index in range(5)]
    chunks = list(async_rl.eval_chunks(rows, num_prompts=2))
    assert chunks == [(["r0", "r1"], 2), (["r2", "r3"], 2), (["r4", "r4"], 1)], chunks


def test_env_args_reach_the_configuration_from_the_dotlist(monkeypatch):
    """Environment arguments in the manifests' dotlist form land in cfg.env.args
    with YAML typing: a bare comma list stays a string, a bracketed list
    becomes a list, and a number becomes an int."""
    from omegaconf import OmegaConf
    monkeypatch.setattr(sys, "argv", [
        "async_rl.py", "log.wandb_mode=disabled", "env.id=countdown",
        "env.args.names=a,b", "env.args.dataset_size=100", "env.args.sets=[a,b]"])
    import utils
    cfg = utils.parse_cfg(train_rl.DEFAULT_CFG)
    assert OmegaConf.to_container(cfg.env.args) == {
        "names": "a,b", "dataset_size": 100, "sets": ["a", "b"]}


def test_without_data_axis_replicates_the_published_copy_over_data():
    """A trainer over several processes publishes a copy that is sharded over
    the model axis only: the data axis of its specs, which runs across the
    processes, is dropped, also inside tuple axes (expert parallelism)."""
    from jax.sharding import PartitionSpec as P
    assert async_rl.without_data_axis(P("model", "data")) == P("model", None)
    assert async_rl.without_data_axis(P(None, "model", "data")) == P(None, "model", None)
    assert async_rl.without_data_axis(P("data", "model", None)) == P(None, "model", None)
    assert async_rl.without_data_axis(P()) == P()
    assert async_rl.without_data_axis(P(("data", "model"), None, None)) == P("model", None, None)
    assert async_rl.without_data_axis(P(("model", "data"), None)) == P("model", None)
    mesh = jax.make_mesh((1, 1), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    assert async_rl.publication_spec(P("model", "data"), mesh) == P("model", "data")  # one process: unchanged


def test_validate_trainer_hosts():
    async_rl.validate_trainer_hosts(1, 0, None)
    async_rl.validate_trainer_hosts(2, 1, "127.0.0.1:29400")
    for hosts, index, coordinator, message in ((0, 0, None, "trainer_hosts"),
                                               (2, 2, "a:1", "trainer_host_index"),
                                               (2, 0, None, "async.coordinator"),
                                               (2, 0, "nohostport", "async.coordinator")):
        try:
            async_rl.validate_trainer_hosts(hosts, index, coordinator)
            raise AssertionError((hosts, index, coordinator))
        except ValueError as error:
            assert message in str(error), error


def test_generate_in_slices_returns_host_arrays(monkeypatch):
    """Every field of the generation comes back as a host (numpy) array, with
    or without slicing: a two-host trainer's evaluation generates arrays that
    span both hosts' devices, and converting such an array with np.asarray
    raises; the gather to the host happens inside the slice concatenation."""
    import jax.numpy as jnp
    from models import sampling
    fields = sampling.Generation._fields
    def fake_generate(key, model, tokens, prompt_lengths, **kwargs):
        n = tokens.shape[0]
        return sampling.Generation(*[jnp.zeros((n, 3), jnp.int32) if name == "tokens" else None for name in fields])
    monkeypatch.setattr(sampling, "generate", fake_generate)
    tokens = np.zeros((4, 3), np.int32); lengths = np.ones(4, np.int32)
    from models.model import make_mesh
    with jax.set_mesh(make_mesh(1, 1)):  # the trainer calls it under the sampler mesh
        for rows in (0, 2):
            generation = async_rl.generate_in_slices(jax.random.key(0), None, tokens, lengths, rows)
            assert isinstance(generation.tokens, np.ndarray), (rows, type(generation.tokens))
            assert generation.tokens.shape == (4, 3)


def test_eval_slice_rows_match_a_generators_cache_per_device():
    # The evaluation matches a generator's per-device KV cache spread over the
    # whole sampler mesh: 128 rows over 8 devices, times 16.
    assert async_rl.eval_slice_rows(128, 8, 16) == 256
