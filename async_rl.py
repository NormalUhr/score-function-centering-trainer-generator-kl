"""Asynchronous RL with per-token sampler provenance.

A generator process decodes rollouts continuously under the latest published
weights and swaps weights between decode stages, so one sequence can carry
several sampler versions; a trainer process consumes finished rollouts from a
queue, trains, and publishes its weights after every optimizer step. The two
roles share one exchange directory, on a local disk or on a shared filesystem:

    <exchange>/weights/v000123/  published weights (stacked layout, dtypes kept),
                                 one weights-<part>.safetensors per trainer process
    <exchange>/weights/LATEST    the newest version number
    <exchange>/queue/*.npz       finished rollout batches, oldest first
    <exchange>/DONE              written by the trainer when it stops

    python async_rl.py role=trainer   async.exchange_dir=/tmp/x <overrides>
    python async_rl.py role=generator async.exchange_dir=/tmp/x \\
        async.generator_index=0 <the same overrides>

The trainer can span several hosts as one multi-process JAX program
(async.trainer_hosts, async.trainer_host_index, async.coordinator): every
process consumes the same rollout batch, writes its share of each
publication, and process 0 alone logs, renames a finished publication and
writes DONE.

Staleness is measured: a token's age is the number of optimizer steps
between the version that produced it and the trainer's version when the
batch is consumed, and the per-token KL coefficient follows that age.
The rest of the loss is the synchronous trainer's (`train_rl.make_optimize`).
"""
import json
import multiprocessing
import os
import shutil
import threading
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
from omegaconf import OmegaConf

import losses
import models
import prompting
import timing
import train_rl
import utils
from models import sampling
from models.model import load_exchange, make_mesh, save_exchange
from tasks import rollout

ROLES = ("trainer", "generator")
AGE_BINS = 256  # the logged age histogram: counts for ages 0..254 and one overflow bin


def wait_for_parts(directory, parts, timeout, poll=0.2):
    """Block until `directory` holds the part file of every one of `parts`
    trainer processes (`save_exchange` renames a part into place only once
    it is complete); raise TimeoutError naming the missing parts otherwise."""
    directory = Path(directory)
    deadline = time.monotonic() + timeout
    while True:
        missing = [f"weights-{part}.safetensors" for part in range(parts)
                   if not (directory / f"weights-{part}.safetensors").exists()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{directory} still lacks {missing} after {timeout} s; a trainer "
                               "process has probably died")
        time.sleep(poll)


class Mailbox:
    """Published weights: one directory per version and a LATEST file, each
    written to a temporary name and renamed, so a reader never sees a partial
    version. Only the newest `keep_versions` directories are kept."""

    def __init__(self, directory, keep_versions=3, part_timeout=1800):
        self.directory = Path(directory)
        self.keep_versions = keep_versions
        self.part_timeout = part_timeout
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, version):
        return self.directory / f"v{version:06d}"

    def latest(self):
        try:
            return int((self.directory / "LATEST").read_text())
        except (FileNotFoundError, ValueError):
            return None

    def publish(self, version, weights):
        """Write this process's part of `weights` into the version's staging
        directory; process 0 then waits for every process's part and renames
        the directory. The wait watches the files rather than running a
        device barrier: publishing happens in a background thread, and a
        collective issued there while the main thread trains can deadlock
        against the training step's collectives on the same devices."""
        staging = self.directory / f"tmp-v{version:06d}"
        save_exchange(weights, staging)
        if jax.process_index() != 0:
            return
        wait_for_parts(staging, jax.process_count(), self.part_timeout)
        os.replace(staging, self.path(version))
        latest = self.directory / "LATEST.tmp"
        latest.write_text(str(version))
        os.replace(latest, self.directory / "LATEST")
        versions = sorted(int(p.name[1:]) for p in self.directory.iterdir()
                          if p.is_dir() and p.name.startswith("v"))
        for old in versions[:-self.keep_versions]:
            shutil.rmtree(self.path(old), ignore_errors=True)

    def load(self, version, dp_shard=False):
        return load_exchange(self.path(version), dp_shard)


class RolloutQueue:
    """Finished rollout batches as .npz files named by completion time, so
    the trainer consumes them oldest first; a file is written under a
    temporary name and renamed once complete."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last_read_s = 0.0  # seconds the last `oldest` spent loading its file

    def _files(self):
        return sorted(p for p in self.directory.glob("*.npz") if not p.name.startswith("tmp"))

    def size(self):
        return len(self._files())

    def push(self, arrays, generator_index, batch_index):
        name = f"{time.time_ns():020d}-g{generator_index}-b{batch_index:06d}.npz"
        staging = self.directory / f"tmp-{name}"
        with open(staging, "wb") as f:
            np.savez(f, **arrays)
        written = staging.stat().st_size
        os.replace(staging, self.directory / name)
        return written

    def oldest(self, timeout=None, poll=0.2, stop=None):
        """The oldest batch as (name, dict of arrays), left in the queue so
        that every process of a multi-host trainer can read the same entry
        before one removes it; None when `timeout` seconds pass or `stop()`
        turns true with the queue empty."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            files = self._files()
            if files:
                started = time.perf_counter()
                with np.load(files[0], allow_pickle=False) as data:
                    arrays = {name: data[name] for name in data.files}
                self.last_read_s = time.perf_counter() - started
                return files[0].name, arrays
            if deadline is not None and time.monotonic() >= deadline:
                return None
            if stop is not None and stop():
                return None
            time.sleep(poll)

    def remove(self, name):
        (self.directory / name).unlink()

    def pop_oldest(self, timeout=None, poll=0.2, stop=None):
        """The oldest batch as a dict of arrays, removed from the queue; None
        as for `oldest`."""
        entry = self.oldest(timeout, poll, stop)
        if entry is None:
            return None
        name, arrays = entry
        self.remove(name)
        return arrays


def host_array(value):
    """A generation's array on the host. On a multi-host trainer the array
    spans every host's devices, and numpy cannot read a device it does not
    own, so the rows are gathered to every host first."""
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        value = multihost_utils.process_allgather(value, tiled=True)
    return np.asarray(value)


def concat_rows(parts):
    return sampling.Generation(*[
        None if fields[0] is None else np.concatenate([host_array(f) for f in fields])
        for fields in zip(*parts)])


def eval_slice_rows(decode_rows, generator_tp, mesh_size):
    """Rows of one evaluation generation slice: the same KV cache per device
    as a generator's decode slices (a generator spreads decode_rows over its
    tensor-parallel devices, the evaluation spreads its rows over the whole
    sampler mesh)."""
    return decode_rows // generator_tp * mesh_size


def generate_in_slices(key, model, tokens, prompt_lengths, rows, **kwargs):
    """sampling.generate over the batch in row slices of `rows` (0 = all at
    once), so that a long sequence budget does not put every row's KV cache
    on the devices at the same time. The first slice keeps the key."""
    batch_size = tokens.shape[0]
    if not rows or rows >= batch_size:
        return concat_rows([sampling.generate(key, model, tokens, prompt_lengths, **kwargs)])
    # Slice on the host: the buffer may live on another mesh than the sampler's;
    # the key is moved to the current mesh before it is folded.
    tokens, prompt_lengths = np.asarray(tokens), np.asarray(prompt_lengths)
    key = jax.sharding.reshard(key, jax.sharding.PartitionSpec())
    parts = []
    for index, start in enumerate(range(0, batch_size, rows)):
        part = slice(start, min(start + rows, batch_size))
        part_key = key if index == 0 else jax.random.fold_in(key, index)
        parts.append(sampling.generate(
            part_key, model, tokens[part], prompt_lengths[part], **kwargs))
    return concat_rows(parts)


def decode_batch(model, tokens, prompt_lengths, key, payload, chunk_tokens, current_weights,
                 eos_id=None, rows=0):
    """Decode one batch in stages of `chunk_tokens` tokens. Before every stage
    `current_weights()` returns (version, weights) and the stage runs under
    them, so a version published mid-batch takes over the unfinished rows;
    the KV cache is carried from stage to stage, as in an inference engine
    with in-flight weight updates. `rows` > 0 decodes the batch in row slices
    of that size, each slice through all its stages before the next starts,
    which bounds the sampler's memory (one slice's KV cache); 0 decodes every
    row at once. Returns the merged Generation, the [B, T] version of the
    weights that produced the prediction at each position (-1 where nothing
    was predicted), the list of versions the stages used, in order, and the
    number of active row-stages (rows still generating, summed over the
    stages), the unit a stage's decode time scales with."""
    prompt_lengths = np.asarray(prompt_lengths, np.int32)
    batch_size, seq_len = tokens.shape
    version_at = np.full((batch_size, seq_len), -1, np.int32)
    used, parts, row_stages = [], [], 0
    if (prompt_lengths >= seq_len).all():
        raise ValueError("every prompt fills the sequence buffer; nothing to generate "
                         "(raise rl.seq_len)")
    slices = [slice(start, min(start + (rows or batch_size), batch_size))
              for start in range(0, batch_size, rows or batch_size)]
    for index, part in enumerate(slices):
        # The first slice keeps the batch key, so a batch decoded in one
        # slice draws with the batch key itself.
        part_key = key if index == 0 else jax.random.fold_in(key, index)
        lengths = prompt_lengths[part]
        generation, resume = None, None
        generated = np.zeros(len(lengths), np.int32)
        finished = lengths >= seq_len
        while not finished.all():
            version, weights = current_weights()
            used.append(version)
            row_stages += int((~finished).sum())
            before = lengths + generated
            if generation is None:
                generation, resume = sampling.generate(
                    part_key, model, tokens[part], lengths, weights=weights, eos_id=eos_id,
                    max_new_tokens=chunk_tokens, return_resume=True, **payload)
            else:
                generation, _, resume = train_rl.continue_rollouts(
                    model, generation, weights, lengths, None, payload,
                    max_new_tokens=chunk_tokens, eos_id=eos_id, resume=resume,
                    return_resume=True)
            generated = np.asarray(generation.generated)
            after = lengths + generated
            for row in np.flatnonzero(~finished):
                # Tokens before[row] .. after[row]-1 were predicted at positions
                # before[row]-1 .. after[row]-2 under this stage's weights.
                version_at[part.start + row, before[row] - 1:after[row] - 1] = version
            finished = np.asarray(generation.ended) | (after >= seq_len) | (generated == 0)
        parts.append(generation)
    return (parts[0] if len(parts) == 1 else concat_rows(parts)), version_at, used, row_stages


ROW_ARRAYS = ("tokens", "mask", "version", "rewards", "sampling_token_logprobs",
              "prompt_lengths", "completion_lengths", "truncated", "center_ids", "center_logprobs",
              "tail_ids", "tail_logprobs")


def split_batch(data, rows):
    """Cut a generator batch into trainer slices of `rows` sequences along the
    row axis; `rows` is a multiple of the group size, so prompt groups stay
    whole. Per-batch scalars and the list of versions used are shared."""
    total = data["tokens"].shape[0]
    per_row = [name for name in ROW_ARRAYS if name in data]
    shared = {name: value for name, value in data.items() if name not in per_row}
    return [{**shared, **{name: data[name][start:start + rows] for name in per_row}}
            for start in range(0, total, rows)]


def age_summary(ages, target_mask, version_at):
    """Per-token age statistics over the predicted positions of a batch."""
    values = ages[target_mask]
    if values.size == 0:
        values = np.zeros(1, np.int64)
    versions = np.where(target_mask, version_at, -1)
    per_row = [np.unique(v[v >= 0]).size for v in versions]
    hist = np.bincount(np.minimum(values, AGE_BINS - 1), minlength=AGE_BINS)
    return {
        "age_mean": float(values.mean()),
        "age_max": int(values.max()),
        "age_p50": float(np.percentile(values, 50)),
        "age_p90": float(np.percentile(values, 90)),
        "frac_age0": float((values == 0).mean()),
        "versions_in_batch": int(np.unique(values).size),
        "frac_mixed_sequences": float(np.mean([n > 1 for n in per_row])),
        "age_hist": hist.tolist(),
    }


def _exchange(cfg):
    a = cfg["async"]
    exchange = Path(a.exchange_dir).expanduser()
    return (exchange, Mailbox(exchange / "weights", a.keep_versions, part_timeout=a.wait_timeout),
            RolloutQueue(exchange / "queue"))


def _run_dir(cfg):
    return Path(cfg.log.dir).expanduser() / cfg.log.project / cfg.log.run_name


def split_eval(dataset, num_prompts):
    """The evaluation rows and the training set: the first `num_prompts` rows
    of the shuffled `dataset` are held out, and the rest trains."""
    return [dataset[index] for index in range(num_prompts)], dataset.select(range(num_prompts, len(dataset)))


def eval_chunks(rows, num_prompts):
    """Yield (rows of one evaluation batch, number of real rows). A short
    last batch repeats its own rows up to `num_prompts`, so every batch has
    the compiled shape; the caller keeps the results of the real rows only."""
    for start in range(0, len(rows), num_prompts):
        chunk = rows[start:start + num_prompts]
        yield [chunk[index % len(chunk)] for index in range(num_prompts)], len(chunk)


def _environment(cfg, model):
    """The environment, the shuffled training set and the evaluation rows
    (`split_eval`): the same split in both roles."""
    env_args = OmegaConf.to_container(cfg.env.args) if cfg.env.args else {}
    environment = rollout.load(cfg.env.id, **env_args)
    if getattr(environment, "max_turns", 1) != 1 or environment.provides_advantages:
        raise ValueError("async RL supports single-turn environments with scalar rewards")
    dataset = environment.get_dataset().shuffle(seed=cfg.run.seed)
    eval_rows = []
    if cfg.eval.enabled:
        eval_rows, dataset = split_eval(dataset, cfg.eval.num_prompts)
    prompt_format = prompting.resolve_prompt_format(cfg.env.prompt_format, cfg.model.source)
    chat_template_kwargs = OmegaConf.to_container(cfg.env.chat_template_kwargs)
    prompting.validate_prompt_contract(model.tokenizer, prompt_format, chat_template_kwargs,
                                       stop_ids=model.stop_ids)
    return environment, dataset, eval_rows, prompt_format, chat_template_kwargs


def _sampler_dtype_tree(weights, float_dtype):
    """The synchronous sampler's dtype rule: matrices in the sampler dtype,
    norms and other vectors as they are."""
    return {key: weight.astype(float_dtype) if weight.ndim > 1 and "norm" not in key else weight
            for key, weight in weights.items()}


def without_data_axis(spec):
    """`spec` with the data axis replaced by replication, also inside tuple
    axes: expert weights sharded over ("data", "model") become sharded over
    "model" alone, so that every process holds every expert of its copy."""
    def drop(axis):
        if isinstance(axis, tuple):
            kept = tuple(name for name in axis if name != "data")
            return None if not kept else kept[0] if len(kept) == 1 else kept
        return None if axis == "data" else axis
    return P(*(drop(axis) for axis in spec))


def publication_spec(spec, mesh):
    """The published copy's layout on `mesh`: the trainer's own spec when the
    mesh is on one process, and the spec without its data axis when the mesh
    spans several, since that axis runs across the processes and every
    process must hold whole tensors to write its part of a publication."""
    if len({device.process_index for device in mesh.devices.flat}) == 1:
        return spec
    return without_data_axis(spec)


def publication_mesh(shard):
    """The synchronous sampler's layout on the trainer's devices: tensor
    parallelism from `shard` (every device by default) and data parallelism
    over the rest. A model whose key-value heads do not split over every
    device needs the smaller tensor-parallel size here as in the sampler; a
    trainer over several hosts needs tensor parallelism within one host, so
    that every process holds whole tensors to publish."""
    data_parallel, tensor_parallel = train_rl.resolve_shard(shard, jax.device_count(), "sampler.shard")
    return make_mesh(data_parallel, tensor_parallel)


def run_generator(cfg):
    a = cfg["async"]
    exchange, mailbox, queue = _exchange(cfg)
    done = exchange / "DONE"
    key = jax.random.fold_in(jax.random.key(cfg.run.seed), 1000 + a.generator_index)
    with timing.context("load model"):
        model = models.load(
            cfg.model.source, cfg.model.weights_dir, tp_size=jax.device_count(),
            dp_shard=False, init=cfg.model.init, hidden_size=cfg.model.hidden_size,
            num_hidden_layers=cfg.model.num_hidden_layers, key=jax.random.fold_in(key, 1))
    environment, dataset, _, prompt_format, chat_template_kwargs = _environment(cfg, model)
    sampler_dtype = jnp.dtype(cfg.dtypes.sampler)
    forward = partial(model.forward, dtype=sampler_dtype)
    init_kv = partial(model.init_kv, dtype=sampler_dtype)
    topk = cfg.sampler.vocab_logprobs if isinstance(cfg.sampler.vocab_logprobs, int) else 0
    payload = dict(forward=forward, init_kv=init_kv, logprobs_topk=topk,
                   tail_samples=cfg.sampler.tail_samples)
    num_prompts, group_size, seq_len = cfg.rl.num_prompts, cfg.rl.group_size, cfg.rl.seq_len
    log_path = _run_dir(cfg) / f"generator-{a.generator_index}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"version": None, "weights": None, "loads": 0, "load_s": 0.0}

    def current_weights():
        latest = mailbox.latest()
        if latest is not None and latest != state["version"]:
            started = time.perf_counter()
            state["weights"] = jax.block_until_ready(mailbox.load(latest, dp_shard=False))
            # The published weights replace the loaded checkpoint, so the
            # generator holds one copy.
            model.weights = state["weights"]
            state["version"], state["loads"] = latest, state["loads"] + 1
            state["load_s"] += time.perf_counter() - started
        return state["version"], state["weights"]

    # Every batch starts from published weights: the trainer publishes
    # version 0 before it consumes anything.
    while mailbox.latest() is None and not done.exists():
        time.sleep(a.poll_seconds)
    batch_index = 0
    run_start = time.perf_counter()
    while not done.exists():
        while queue.size() >= a.queue_depth and not done.exists():
            time.sleep(a.poll_seconds)
        if done.exists():
            break
        global_index = batch_index * a.generators + a.generator_index
        rows = [dataset[(global_index * num_prompts + offset) % len(dataset)]
                for offset in range(num_prompts)]
        inputs = [{**row, "example_id": example_id}
                  for example_id, row in enumerate(rows) for _ in range(group_size)]
        token_buffer, prompt_lengths = rollout.prompt_buffer(
            inputs, model.tokenizer, seq_len, prompt_format=prompt_format,
            chat_template_kwargs=chat_template_kwargs)
        key, batch_key = jax.random.split(key)
        started = time.perf_counter()
        generation, version_at, used, row_stages = decode_batch(
            model, token_buffer, prompt_lengths, batch_key, payload, a.chunk_tokens,
            current_weights, rows=a.decode_rows)
        decode_seconds = time.perf_counter() - started
        if done.exists():
            break  # the trainer stopped while this batch decoded: nobody will consume it
        host_tokens = np.asarray(generation.tokens)
        mask, completion_lengths, rewards, frac_parsed, truncated = train_rl.finish_rollouts(
            host_tokens, prompt_lengths, inputs, environment, model.tokenizer, seq_len,
            group_size, stop_ids=model.stop_ids)
        arrays = {
            "tokens": host_tokens, "mask": mask, "prompt_lengths": prompt_lengths,
            "sampling_token_logprobs": np.asarray(generation.token_logprobs, np.float32),
            "version": version_at, "rewards": rewards,
            "completion_lengths": completion_lengths, "truncated": truncated,
            "frac_parsed": np.float32(frac_parsed), "versions_used": np.asarray(used),
            "generator_index": np.int32(a.generator_index), "batch_index": np.int32(batch_index),
            "decode_seconds": np.float32(decode_seconds),
        }
        if generation.head_ids is not None:
            arrays["center_ids"] = np.asarray(generation.head_ids)
            arrays["center_logprobs"] = np.asarray(generation.head_logprobs, np.float32)
        if generation.tail_ids is not None:
            arrays["tail_ids"] = np.asarray(generation.tail_ids)
            arrays["tail_logprobs"] = np.asarray(generation.tail_logprobs, np.float32)
        if done.exists():
            break  # the trainer stopped while this batch was scored: nobody will consume it
        pushing = time.perf_counter()
        rollout_bytes = queue.push(arrays, a.generator_index, batch_index)
        push_seconds = time.perf_counter() - pushing
        record = {"batch": batch_index, "versions_used": used, "stages": len(used),
                  "row_stages": row_stages,
                  "reward": float(rewards.mean()), "decode_s": decode_seconds,
                  "score_s": time.perf_counter() - started - decode_seconds - push_seconds,
                  "rollout_mb": rollout_bytes / 2 ** 20, "push_s": push_seconds,
                  "peak_mem_gb": utils.peak_memory_gb(),
                  "queue_len": queue.size(), "mean_completion_len": float(completion_lengths.mean()),
                  "weight_loads": state["loads"], "load_s_total": state["load_s"],
                  "elapsed_s": time.perf_counter() - run_start}
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"generator {a.generator_index} batch {batch_index}", record)
        batch_index += 1
    print(f"generator {a.generator_index} done after {batch_index} batches")


def run_trainer(cfg):
    a = cfg["async"]
    exchange, mailbox, queue = _exchange(cfg)
    done = exchange / "DONE"
    primary = jax.process_index() == 0  # owns the exchange, the logs and DONE
    if primary:
        done.unlink(missing_ok=True)
        if any(mailbox.directory.iterdir()) or queue.size():
            raise RuntimeError(f"exchange directory {exchange} is not empty: another run's weights "
                               "or rollouts would be mixed into this one")
    key = jax.random.key(cfg.run.seed)
    _, trainer_tp = train_rl.resolve_shard(cfg.train.shard, 1, "train.shard")
    with timing.context("load model"):
        model = models.load(
            cfg.model.source, cfg.model.weights_dir, tp_size=trainer_tp,
            dp_shard=cfg.train.shard.fsdp, init=cfg.model.init,
            hidden_size=cfg.model.hidden_size, num_hidden_layers=cfg.model.num_hidden_layers,
            key=jax.random.fold_in(key, 1))
    environment, _, eval_rows, prompt_format, chat_template_kwargs = _environment(cfg, model)
    model.weights = jax.tree.map(
        lambda weight: weight.astype(jnp.dtype(cfg.dtypes.master)), model.weights)
    trainer_forward = partial(model.forward, dtype=jnp.dtype(cfg.dtypes.trainer))
    trainer_mesh = jax.tree.leaves(model.weights)[0].sharding.mesh
    jax.set_mesh(trainer_mesh)
    sampler_dtype = jnp.dtype(cfg.dtypes.sampler)
    # Publication and evaluation use the synchronous sampler's layout: tensor
    # parallelism from sampler.shard over the trainer's devices, matrices in bf16.
    sampler_mesh = publication_mesh(cfg.sampler.shard)
    sampler_shardings = jax.tree.map(
        lambda weight: NamedSharding(sampler_mesh, publication_spec(weight.sharding.spec, sampler_mesh)),
        model.weights)
    copy_sampler_weights = jax.jit(
        partial(_sampler_dtype_tree, float_dtype=sampler_dtype), out_shardings=sampler_shardings)
    num_prompts, group_size, seq_len = cfg.rl.num_prompts, cfg.rl.group_size, cfg.rl.seq_len
    token_sharding = NamedSharding(trainer_mesh, P("data", None))
    rollout_sharding = NamedSharding(trainer_mesh, P("data"))
    center_sharding = NamedSharding(trainer_mesh, P("data", None, None))
    total_steps = cfg.stop.steps
    optimizer, optimizer_state = utils.make_optimizer(cfg.opt, total_steps, model.weights)
    run_dir = utils.start_run(cfg)
    is_cfg = (losses.ISCorrection(**OmegaConf.to_container(cfg.rl["is"]))
              if cfg.rl["is"].level is not None else None)
    sampler_kl = (losses.SamplerKL(**OmegaConf.to_container(cfg.rl.sampler_kl))
                  if cfg.rl.sampler_kl.coeff > 0 else None)
    optimize = train_rl.make_optimize(cfg, model, optimizer, trainer_forward, is_cfg, sampler_kl)

    publishing = {"thread": None, "seconds": 0.0, "error": None}

    def finish_publish():
        if publishing["thread"] is not None:
            publishing["thread"].join()
            publishing["thread"] = None
        if publishing["error"] is not None:
            raise RuntimeError("publishing the weights failed; the generators would keep "
                               "decoding under stale weights") from publishing["error"]

    # The published snapshot keeps the trainer's own sharding: each process
    # writes the shards its devices hold, so no process needs a whole copy of
    # the model (which at 100B parameters does not fit beside the masters).
    snapshot_weights = jax.jit(partial(_sampler_dtype_tree, float_dtype=sampler_dtype))

    def publish(version):
        """Cast the weights to the sampler dtype on device, then transfer and
        write them in a background thread, so the trainer's next step overlaps
        the transfer; the returned time is what the trainer waited for. A
        failed write surfaces at the next publish or at the end."""
        started = time.perf_counter()
        finish_publish()
        copy = jax.block_until_ready(snapshot_weights(jax.block_until_ready(model.weights)))

        def write():
            began = time.perf_counter()
            try:
                mailbox.publish(version, copy)
            except Exception as error:  # noqa: BLE001 - re-raised on the main thread
                publishing["error"] = error
            publishing["seconds"] = time.perf_counter() - began

        publishing["thread"] = threading.Thread(target=write, daemon=True)
        publishing["thread"].start()
        return time.perf_counter() - started

    eval_key = jax.random.fold_in(key, 3)

    def evaluate(step):
        # A publication's cast copy lives until its background write ends. When
        # an evaluation step is also a publication step, the evaluation's own
        # cast copy would sit beside it, and two copies of a 100B model beside
        # the masters exceed the device memory. Waiting for the write keeps one
        # copy.
        finish_publish()
        eval_weights = copy_sampler_weights(jax.block_until_ready(model.weights))
        payload = dict(forward=partial(model.forward, dtype=jnp.bfloat16),
                       init_kv=partial(model.init_kv, dtype=jnp.bfloat16))
        rewards, lengths, parsed, samples = [], [], [], []
        for chunk_index, (rows, real_rows) in enumerate(eval_chunks(eval_rows, num_prompts)):
            inputs = [{**row, "example_id": example_id}
                      for example_id, row in enumerate(rows) for _ in range(group_size)]
            token_buffer, prompt_lengths = rollout.prompt_buffer(
                inputs, model.tokenizer, seq_len, prompt_format=prompt_format,
                chat_template_kwargs=chat_template_kwargs)
            # The key is derived under the trainer mesh; generate() reshards it
            # to the sampler mesh (deriving it inside the sampler-mesh context
            # fails on an explicit-mesh mismatch).
            chunk_key = jax.random.fold_in(eval_key, chunk_index * num_prompts)
            with jax.set_mesh(sampler_mesh):
                generation = generate_in_slices(
                    chunk_key, model, token_buffer, prompt_lengths,
                    eval_slice_rows(a.decode_rows, cfg.sampler.shard.tp or 1, sampler_mesh.size),
                    weights=eval_weights, **payload)
            # generate_in_slices returns host arrays, gathered from every host
            # of a multi-host trainer, so every host scores the same rows.
            _, completion_lengths, chunk_rewards, frac_parsed, _ = train_rl.finish_rollouts(
                generation.tokens, prompt_lengths, inputs, environment, model.tokenizer, seq_len,
                group_size, stop_ids=model.stop_ids)
            keep = real_rows * group_size
            rewards += chunk_rewards[:keep].tolist()
            lengths += completion_lengths[:keep].tolist()
            parsed += [frac_parsed] * keep  # the chunk's mean, over its repeated rows too when padded
            missing = cfg.eval.save_samples - len(samples)
            if missing > 0:
                _, _, texts = rollout.completions(
                    generation.tokens[:min(keep, missing)], prompt_lengths[:min(keep, missing)],
                    model.tokenizer, model.stop_ids)
                samples += rollout.sample_records(
                    step, inputs, texts, chunk_rewards, completion_lengths, limit=len(texts))
        if samples and jax.process_index() == 0:
            with open(Path(run_dir) / "eval_samples.jsonl", "a") as f:
                for record in samples:
                    f.write(json.dumps(record) + "\n")
        return {"eval_reward": float(np.mean(rewards)),
                "eval_mean_completion_len": float(np.mean(lengths)),
                "eval_frac_parsed": float(np.mean(parsed))}

    def publish_failed():
        return publishing["error"] is not None

    def next_batch():
        """The oldest rollout batch, removed from the queue. Every process of
        a multi-host trainer reads the same entry: a barrier after the reads
        keeps the entry in place until every process has it, and a second
        one after the removal keeps any process from reading the queue again
        before the entry is gone."""
        entry = queue.oldest(timeout=a.wait_timeout, poll=a.poll_seconds, stop=publish_failed)
        if entry is None:
            # Generators decode only under published weights, so a failed
            # publication is the likelier cause of an empty queue.
            finish_publish()
            raise RuntimeError(f"no rollout batch arrived for {a.wait_timeout} s; "
                               "the generators have probably died")
        name, data = entry
        if jax.process_count() == 1:
            queue.remove(name)
            return data
        multihost_utils.sync_global_devices("rollout batch read")
        if primary:
            queue.remove(name)
        multihost_utils.sync_global_devices("rollout batch removed")
        return data

    version = 0  # optimizer steps applied
    step_rows = (a.trainer_prompts or num_prompts) * group_size
    train_rl.check_slice_rows(step_rows, trainer_mesh.shape["data"])
    pending = []  # slices of the generator batch in hand, oldest first
    run_start = time.perf_counter()
    publish_seconds = publish(version)
    try:
        for step in range(total_steps):
            should_log = step % cfg.log.every_steps == 0 or step == total_steps - 1
            eval_metrics = {}
            if should_log and eval_rows and (step % cfg.eval.every_steps == 0
                                             or step == total_steps - 1):
                eval_metrics = evaluate(step)  # before the step, as the synchronous loop does
            wait_seconds = 0.0
            if not pending:
                waited = time.perf_counter()
                data = next_batch()
                wait_seconds = time.perf_counter() - waited
                pending = split_batch(data, step_rows)
            data = pending.pop(0)
            rewards = data["rewards"]
            mask = data["mask"]
            advantages = losses.compute_advantages(rewards, group_size)
            target_mask = np.roll(mask, -1, axis=1)
            target_mask[:, -1] = False
            version_at = data["version"]
            assert (version_at[target_mask] >= 0).all(), "a predicted position without a version"
            ages = np.maximum(np.where(target_mask, version - version_at, 0), 0).astype(np.int64)
            summary = age_summary(ages, target_mask, version_at)
            batch = {
                "tokens": jax.device_put(data["tokens"], token_sharding),
                "mask": jax.device_put(mask, token_sharding),
                "advantages": jax.device_put(advantages, rollout_sharding),
                "sampling_token_logprobs": jax.device_put(
                    data["sampling_token_logprobs"], token_sharding),
            }
            if "center_ids" in data:
                batch["center_ids"] = jax.device_put(data["center_ids"], center_sharding)
                batch["center_logprobs"] = jax.device_put(data["center_logprobs"], center_sharding)
            if "tail_ids" in data:
                batch["tail_ids"] = jax.device_put(data["tail_ids"], center_sharding)
                batch["tail_logprobs"] = jax.device_put(data["tail_logprobs"], center_sharding)
            kl_coeff = None
            if sampler_kl is not None:
                kl_coeff = np.asarray(sampler_kl.coefficient(ages), np.float32)
                batch["kl_coeff"] = jax.device_put(kl_coeff, token_sharding)
            reward = float(rewards.mean())
            key, optimizer_key = jax.random.split(key)
            model.weights, optimizer_state, optimizer_metrics = optimize(
                model.weights, optimizer_state, batch, optimizer_key)
            del batch
            version += 1
            if (step + 1) % a.publish_every == 0 or step == total_steps - 1:
                publish_seconds = publish(version)
            if should_log:
                optimizer_metrics = {name: np.asarray(value) for name, value in optimizer_metrics.items()}
                metrics = {
                    "step": step,
                    "reward": reward,
                    "version": version,
                    "mean_completion_len": float(data["completion_lengths"].mean()),
                    "frac_nonzero_advantage": float((advantages != 0).mean()),
                    "frac_parsed": float(data["frac_parsed"]),
                    "frac_truncated": float(data["truncated"].mean()),
                    **summary,
                    "batch_versions_used": data["versions_used"].tolist(),
                    "generator_index": int(data["generator_index"]),
                    "decode_s": float(data["decode_seconds"]),
                    "queue_len": queue.size(),
                    "wait_s": wait_seconds,
                    "rollout_read_s": queue.last_read_s,
                    "peak_mem_gb": utils.peak_memory_gb(),
                    "publish_s": publish_seconds,
                    "publish_write_s": publishing["seconds"],
                    **({"kl_coeff": float(kl_coeff[target_mask].mean()) if target_mask.any() else 0.0}
                       if kl_coeff is not None else {}),
                    "elapsed_s": time.perf_counter() - run_start,
                    "weight_norm": utils.weight_norm(model.weights),
                    **optimizer_metrics,
                    **eval_metrics,
                }
                utils.log_metrics(step, metrics, run_dir)
    finally:
        try:
            finish_publish()
        finally:
            if primary:
                done.write_text("done\n")
    if cfg.log.save_model:
        utils.save_model(model, run_dir, "final_model")


def validate_trainer_hosts(hosts, index, coordinator):
    if hosts < 1 or not 0 <= index < hosts:
        raise ValueError("async.trainer_hosts must be positive and contain async.trainer_host_index")
    if hosts > 1 and not (isinstance(coordinator, str) and ":" in coordinator):
        raise ValueError("async.coordinator (host:port of trainer host 0) is required with "
                         "async.trainer_hosts > 1")


def initialize_trainer_hosts(overrides):
    """Join this process to the other trainer processes when the command line
    asks for several hosts. This reads the raw overrides because parsing the
    full configuration (`utils.parse_cfg`) already asks JAX for the process
    index, which starts the backend, after which jax.distributed.initialize
    refuses to run."""
    hosts = OmegaConf.select(overrides, "async.trainer_hosts", default=1)
    if OmegaConf.select(overrides, "role") != "trainer" or hosts == 1:
        return
    index = OmegaConf.select(overrides, "async.trainer_host_index", default=0)
    coordinator = OmegaConf.select(overrides, "async.coordinator")
    validate_trainer_hosts(hosts, index, coordinator)
    jax.distributed.initialize(coordinator_address=coordinator, num_processes=hosts, process_id=index)


def validate_async(cfg):
    a = cfg["async"]
    if cfg.role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if not a.exchange_dir:
        raise ValueError("async.exchange_dir is required")
    if a.generators < 1 or not 0 <= a.generator_index < a.generators:
        raise ValueError("async.generators must be positive and contain async.generator_index")
    validate_trainer_hosts(a.trainer_hosts, a.trainer_host_index, a.coordinator)
    for name in ("chunk_tokens", "queue_depth", "publish_every", "keep_versions"):
        if a[name] < 1:
            raise ValueError(f"async.{name} must be positive")
    if a.poll_seconds <= 0:
        raise ValueError("async.poll_seconds must be positive")
    if a.decode_rows < 0:
        raise ValueError("async.decode_rows must be zero (every row at once) or positive")
    if cfg.sampler.staleness:
        raise ValueError("async RL measures staleness; sampler.staleness must be 0")
    if cfg.sampler.kv_quant_bits:
        raise ValueError("async RL transports the payload of an exact bf16 sampler: "
                         "KV-cache quantization is not supported")
    if cfg.sampler.tail_samples and not isinstance(cfg.sampler.vocab_logprobs, int):
        raise ValueError("sampler.tail_samples needs an integer sampler.vocab_logprobs (the head)")
    if cfg.sampler.vocab_logprobs == "full":
        raise ValueError("async RL does not transport the full vocabulary")
    if cfg["async"].wait_timeout <= 0:
        raise ValueError("async.wait_timeout must be positive")
    if a.trainer_prompts is not None:
        if a.trainer_prompts < 1 or cfg.rl.num_prompts % a.trainer_prompts:
            raise ValueError("async.trainer_prompts must be positive and divide rl.num_prompts")


def main():
    # Spawn, never fork: a scorer worker forked from this threaded CUDA
    # process inherits its state and can fail silently.
    multiprocessing.set_start_method("spawn", force=True)
    initialize_trainer_hosts(OmegaConf.from_cli())
    cfg = utils.parse_cfg(train_rl.DEFAULT_CFG)
    train_rl.validate(cfg)
    validate_async(cfg)
    if cfg.role == "trainer":
        run_trainer(cfg)
    else:
        run_generator(cfg)


if __name__ == "__main__":
    main()
