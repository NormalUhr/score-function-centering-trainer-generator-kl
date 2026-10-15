import multiprocessing
import os
import time
from functools import partial

os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", "0.85")
# No BFC preallocation: at 30B the fp32 masters pin the middle of the pool
# during sampling and fragment it below the trainer step's contiguous needs.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P, reshard
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import losses
import models
import prompting
import timing
import utils
from models import quant, sampling
from tasks import rollout


DEFAULT_CFG = """
role: null  # async_rl.py only: trainer | generator
model:
  source: Qwen/Qwen3-0.6B
  weights_dir: ~/.cache/postax/weights
  init: pretrained
  hidden_size: null
  num_hidden_layers: null
train:
  shard:
    dp: null
    tp: 1
    fsdp: true
dtypes:
  sampler: bfloat16
  trainer: bfloat16
  master: float32
sampler:
  shard:
    dp: null
    tp: null
  staleness: 0  # synchronous loop: optimizer steps between sampler refreshes
  kv_quant_bits: 0  # fake quant of sampler KV-cache writes: 0 (off) | 2..16 (int-N) | fp8 (e4m3, unit scale) | fp6/fp4 (scaled mini-float)
  vocab_logprobs: 128  # null | positive integer (top-k) | full
  tail_samples: 0  # tokens drawn per step from q outside the top-k head
                   # (head/tail payload); needs an integer vocab_logprobs
async:  # async_rl.py: generator processes feed a trainer process through an exchange directory
  exchange_dir: null  # shared by the trainer and its generators (weights, queue, DONE)
  generators: 1  # generator processes feeding one trainer
  generator_index: 0  # this generator's index (role=generator)
  chunk_tokens: 128  # tokens decoded per stage before the mailbox is read again
  decode_rows: 0  # rows a generator decodes at once within a stage; 0 = its whole batch.
                  # Slices bound the sampler's KV cache at long sequence budgets.
  queue_depth: 4  # a generator starts a batch only while fewer than this many
                  # unconsumed batches wait, so at most depth + generators - 1 wait
  publish_every: 1  # trainer steps between weight publications
  trainer_prompts: null  # prompts per optimizer step; null = the generator's whole batch.
                         # A smaller slice makes several steps per rollout batch, each
                         # with its own ages and publication: more updates per rollout,
                         # so a larger measured staleness at the same generator layout.
  keep_versions: 3  # published versions kept on disk
  poll_seconds: 0.5  # wait between checks of the queue and the mailbox
  wait_timeout: 1800  # seconds the trainer waits for a batch (or for another trainer host's
                      # part of a publication) before it gives up
  trainer_hosts: 1  # processes of a trainer that spans several hosts (one multi-process JAX
                    # program over their GPUs); every process publishes its share of the weights
  trainer_host_index: 0  # this process's index among the trainer hosts (role=trainer)
  coordinator: null  # host:port of trainer host 0, required when trainer_hosts > 1
rl:
  num_prompts: 16
  group_size: 8
  seq_len: 1024
  sc: false  # uses sampler.vocab_logprobs: exact if full, modeled tail if
             # top-k, stratified if tail_samples > 0
  sampler_kl:  # coeff / sqrt(age) * KL(q || p) per token (0 at age 0), estimated on the same payload
    coeff: 0.0
    clip: 5.0  # upper bound on the sampled-tail coefficient p/q - 1
  is:  # IS correction on the trainer/sampler ratio p/q; TIS, MIS, PPO, DAPO,
       # GSPO, DPPO and TOPR are points in this grid (the table in README.md)
    level: null    # null | token | seq_mean (geometric mean) | seq_sum (full product)
    stat: ratio    # ratio | tv: what the band tests, the ratio r = p/q or the
                   # binary TV divergence |q - p| (DPPO; high = delta, e.g. 0.2)
    low: 0.0       # trust band on the ratio; 0 / inf leave a side open
    high: .inf     # .inf is YAML for float inf; CLI overrides accept plain inf
    outside: drop  # clamp = pull weight to the nearest bound | drop = zero it
    sign: both     # both | neg = which advantages get corrected (the others keep w=1)
                   # | ppo = PPO band: pos (0, high], neg [low, 3.0] (dual clip)
env:
  id: countdown
  args: null
  # auto is only a naming-convention convenience. Experiment sweeps should
  # explicitly select text for Base weights and chat for post-trained weights.
  prompt_format: auto
  chat_template_kwargs:
    enable_thinking: false
opt:
  optimizer: adamw
  lr: 1e-6
  b2: 0.99
  lr_warmup: 0.0
stop:
  steps: 100
eval:
  enabled: true
  num_prompts: 128  # multiple of rl.num_prompts; held out from the head of the shuffled training set
  every_steps: 50
  save_samples: 0  # completions written to the run's eval_samples.jsonl at each evaluation (0 = none)
log:
  project: postax-rl
  wandb_mode: online
  dir: ./outputs
  run_name: null
  save_model: false
  every_steps: 1
run:
  seed: 0
"""


def resolve_shard(shard, default_tp, name):
    tensor_parallel = default_tp if shard.tp is None else shard.tp
    if jax.device_count() % tensor_parallel:
        raise ValueError(
            f"{name}.tp={tensor_parallel} does not divide "
            f"device_count={jax.device_count()}")
    data_parallel = jax.device_count() // tensor_parallel
    if shard.dp is not None and shard.dp != data_parallel:
        raise ValueError(
            f"{name}.dp={shard.dp}, but chips/tp={data_parallel}")
    return data_parallel, tensor_parallel


def continue_rollouts(model, first, fresh_weights, prompt_lengths, key, payload,
                      max_new_tokens=None, eos_id=None, resume=None, return_resume=False):
    """Next stage of a multi-version rollout: rows that have not ended
    continue under `fresh_weights`, and the payload of every position keeps
    the snapshot that actually produced its token. Returns the merged
    Generation and a [B, T] mask of the positions whose prediction came from
    the earlier stages (and, with return_resume, the sampling.Resume for a
    further stage). `max_new_tokens` caps this stage so that a further stage
    can follow. With `resume` (the earlier stage's sampling.Resume) the
    stage reuses the KV cache instead of re-decoding the prefix, so its cost
    is its new tokens; `key` is then unused."""
    seq_len = first.tokens.shape[1]
    generated = np.asarray(first.generated)
    ended = np.asarray(first.ended) | (generated == 0)
    positions = np.where(ended, seq_len, prompt_lengths + generated).astype(np.int32)
    second, next_resume = sampling.generate(
        key, model, first.tokens, positions, weights=fresh_weights,
        skip_full_rows=True, max_new_tokens=max_new_tokens, eos_id=eos_id,
        resume=resume, return_resume=True, **payload)
    # Predictions at positions below positions - 1 produced the stale tokens.
    from_stale = np.arange(seq_len)[None, :] < (positions - 1)[:, None]

    def merge(a, b):
        if a is None:
            return None
        mask = from_stale if a.ndim == 2 else from_stale[..., None]
        return np.where(mask, np.asarray(a), np.asarray(b))
    merged = sampling.Generation(
        second.tokens, merge(first.token_logprobs, second.token_logprobs), None,
        merge(first.head_logprobs, second.head_logprobs),
        merge(first.head_ids, second.head_ids),
        merge(first.tail_ids, second.tail_ids),
        merge(first.tail_logprobs, second.tail_logprobs),
        generated + np.asarray(second.generated),
        np.asarray(second.ended) | ended)
    return (merged, from_stale, next_resume) if return_resume else (merged, from_stale)


def check_slice_rows(rows, data_parallel):
    """One optimizer slice is sharded over the trainer's data axis, so its row
    count must divide evenly; shared by the synchronous and asynchronous loops."""
    if rows % data_parallel:
        raise ValueError(
            f"one optimizer slice of {rows} sequences (prompts per step times rl.group_size) "
            f"must be divisible by data parallelism ({data_parallel})")


def make_optimize(cfg, model, optimizer, trainer_forward, is_cfg, sampler_kl):
    """One optimizer step on a rollout batch, jitted with donated weights and
    optimizer state. Shared by the synchronous loop and the asynchronous
    trainer."""

    @partial(jax.jit, donate_argnames=("weights", "optimizer_state"))
    def optimize(weights, optimizer_state, batch, key):
        def objective(weights):
            # Padding rows are dead weight in the MoE dispatch: mask them out.
            hidden = trainer_forward(
                batch["tokens"], weights, logits=False,
                live=batch["tokens"] != model.tokenizer.pad_token_id)
            return losses.loss_fn_rl(
                hidden,
                lambda x: model.head(x, weights, dtype=jnp.dtype(cfg.dtypes.trainer)),
                batch,
                is_cfg,
                score_center=cfg.rl.sc,
                sampler_kl=sampler_kl,
            )

        (loss, rl_metrics), gradients = jax.value_and_grad(
            objective, has_aux=True)(weights)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, weights)
        weights = utils.apply_updates(weights, updates, key)
        return weights, optimizer_state, {
            "loss": loss,
            "grad_norm": optax.global_norm(gradients),
            "update_norm": optax.global_norm(updates),
            **rl_metrics,
        }
    return optimize


def finish_rollouts(host_tokens, prompt_lengths, inputs, environment, tokenizer, seq_len,
                    group_size, stop_ids=None):
    """Host side of a rollout batch: completion masks and lengths, rewards
    from the environment's rubric, the answer-extraction rate and the
    truncation flags. Shared by the synchronous sampler and the asynchronous
    generator. `stop_ids` are the model's turn-ending ids (Model.stop_ids);
    the tokenizer's eos by default."""
    stop_ids = (tokenizer.eos_token_id,) if stop_ids is None else tuple(stop_ids)
    mask, completion_lengths, texts = rollout.completions(
        host_tokens, prompt_lengths, tokenizer, stop_ids)
    rewards = rollout.score(environment.rubric, inputs, texts, group_size)
    # Answer-extraction rate, which separates changes in the policy's answer
    # format from scorer failures. The extractor is environment.parser's,
    # not rubric.parser's.
    extract = getattr(environment.parser, "extract_fn", str)
    frac_parsed = np.mean([
        bool(answer := extract(text)) and answer != text for text in texts])
    truncated = (
        (prompt_lengths + completion_lengths >= seq_len)
        & ~np.isin(host_tokens[:, -1], stop_ids)
    )
    return mask, completion_lengths, rewards, frac_parsed, truncated


def train(cfg):
    key = jax.random.key(cfg.run.seed)
    sampler_dtype = jnp.dtype(cfg.dtypes.sampler)
    quantized = sampler_dtype.itemsize == 1
    sampler_float_dtype = jnp.bfloat16 if quantized else sampler_dtype
    _, trainer_tp = resolve_shard(cfg.train.shard, 1, "train.shard")
    with timing.context("load model"):
        model = models.load(
            cfg.model.source,
            cfg.model.weights_dir,
            tp_size=trainer_tp,
            dp_shard=cfg.train.shard.fsdp,
            init=cfg.model.init,
            hidden_size=cfg.model.hidden_size,
            num_hidden_layers=cfg.model.num_hidden_layers,
            key=jax.random.fold_in(key, 1),
        )
    chat_template_kwargs = OmegaConf.to_container(cfg.env.chat_template_kwargs)
    prompt_format = prompting.resolve_prompt_format(
        cfg.env.prompt_format, cfg.model.source)
    prompting.validate_prompt_contract(
        model.tokenizer, prompt_format, chat_template_kwargs, stop_ids=model.stop_ids)
    # Persist the resolved protocol in W&B/run metadata rather than the
    # convenience spelling "auto".
    cfg.env.prompt_format = prompt_format
    model.weights = jax.tree.map(
        lambda weight: weight.astype(jnp.dtype(cfg.dtypes.master)),
        model.weights,
    )
    trainer_forward = partial(model.forward, dtype=jnp.dtype(cfg.dtypes.trainer))
    trainer_mesh = jax.tree.leaves(model.weights)[0].sharding.mesh
    jax.set_mesh(trainer_mesh)

    sampler_dp, sampler_tp = resolve_shard(cfg.sampler.shard, jax.device_count(), "sampler.shard")
    sampler_mesh = jax.make_mesh(
        (sampler_dp, sampler_tp),
        ("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    sampler_shardings = jax.tree.map(
        lambda weight: NamedSharding(sampler_mesh, weight.sharding.spec),
        model.weights,
    )
    copy_sampler_weights = jax.jit(
        lambda weights: {
            key: weight.astype(sampler_float_dtype)
            if weight.ndim > 1 and "norm" not in key else weight
            for key, weight in weights.items()
        },
        out_shardings=sampler_shardings,
    )

    quantize_sampler_weights = jax.jit(partial(quant.quantize_tree, dtype=sampler_dtype))

    def sampler_snapshot(weights):
        weights = copy_sampler_weights(jax.block_until_ready(weights))
        if not quantized:
            return weights
        with jax.set_mesh(sampler_mesh):
            return quantize_sampler_weights(weights)

    sampler_forward = partial(
        model.forward, dtype=sampler_float_dtype,
        kv_quant_bits=cfg.sampler.kv_quant_bits)
    sampler_init_kv = partial(model.init_kv, dtype=sampler_float_dtype)
    # Eval always samples clean bf16 (no weight or KV quantization): it
    # measures what the policy learned, not how the sampler executes it.
    eval_forward = partial(model.forward, dtype=jnp.bfloat16)

    total_steps = cfg.stop.steps
    optimizer, optimizer_state = utils.make_optimizer(cfg.opt, total_steps, model.weights)
    run_dir = utils.start_run(cfg)
    print(f"n_params={sum(w.size for w in jax.tree.leaves(model.weights)):_}")

    is_cfg = (losses.ISCorrection(**OmegaConf.to_container(cfg.rl["is"]))
              if cfg.rl["is"].level is not None else None)
    sampler_kl = (losses.SamplerKL(**OmegaConf.to_container(cfg.rl.sampler_kl))
                  if cfg.rl.sampler_kl.coeff > 0 else None)

    optimize = make_optimize(cfg, model, optimizer, trainer_forward, is_cfg, sampler_kl)

    env_args = OmegaConf.to_container(cfg.env.args) if cfg.env.args else {}
    environment = rollout.load(cfg.env.id, **env_args)
    if getattr(environment, "max_turns", 1) != 1:
        raise ValueError("synchronous RL supports single-turn environments")
    if environment.provides_advantages:
        raise ValueError(
            "synchronous RL expects environments to return scalar rewards")
    # Hub datasets are ordered by source; the sequential scan below would
    # otherwise turn source clusters into a difficulty curriculum.
    dataset = environment.get_dataset().shuffle(seed=cfg.run.seed)
    seq_len = cfg.rl.seq_len
    if not len(dataset):
        raise ValueError("environment training dataset is empty")
    preview_tokens = prompting.tokenize_prompt(
        dataset[0]["prompt"], model.tokenizer, prompt_format,
        chat_template_kwargs)[:seq_len]
    if jax.process_index() == 0:
        print(prompting.prompt_diagnostic(
            model.tokenizer, prompt_format, preview_tokens, stop_ids=model.stop_ids))
    num_prompts = cfg.rl.num_prompts
    group_size = cfg.rl.group_size
    token_sharding = NamedSharding(trainer_mesh, P("data", None))
    rollout_sharding = NamedSharding(trainer_mesh, P("data"))
    vocab_sharding = NamedSharding(trainer_mesh, P("data", None, "model"))
    center_sharding = NamedSharding(trainer_mesh, P("data", None, None))

    def sample(rows, weights, key, vocab_logprobs_spec, forward=None,
               tail_samples=0):
        inputs = [
            {**row, "example_id": example_id}
            for example_id, row in enumerate(rows)
            for _ in range(group_size)
        ]
        token_buffer, prompt_lengths = rollout.prompt_buffer(
            inputs, model.tokenizer, seq_len,
            prompt_format=prompt_format,
            chat_template_kwargs=chat_template_kwargs)
        payload = dict(
            forward=forward or sampler_forward, init_kv=sampler_init_kv,
            logprobs_dtype=(jnp.float32 if vocab_logprobs_spec == "full" else None),
            logprobs_topk=(vocab_logprobs_spec
                           if isinstance(vocab_logprobs_spec, int) else 0),
            tail_samples=tail_samples)
        centering = {}
        with jax.set_mesh(sampler_mesh):
            generation = sampling.generate(
                key, model, token_buffer, prompt_lengths, weights=weights, **payload)
            tokens = generation.tokens
            sampling_token_logprobs = generation.token_logprobs
            sampling_vocab_logprobs = generation.vocab_logprobs
            if generation.head_ids is not None:
                # The transported head feeds both kl_head diagnostics and SC.
                centering = {"center_ids": generation.head_ids,
                             "center_logprobs": generation.head_logprobs}
            if generation.tail_ids is not None:
                centering.update(tail_ids=generation.tail_ids,
                                 tail_logprobs=generation.tail_logprobs)
            host_tokens = np.asarray(tokens)
        tokens = reshard(tokens, token_sharding)
        sampling_token_logprobs = jax.device_put(
            sampling_token_logprobs, token_sharding)
        centering = {name: jax.device_put(value, center_sharding)
                     for name, value in centering.items()}
        mask, completion_lengths, rewards, frac_parsed, truncated = finish_rollouts(
            host_tokens, prompt_lengths, inputs, environment, model.tokenizer, seq_len,
            group_size, stop_ids=model.stop_ids)
        return (
            tokens, sampling_token_logprobs, sampling_vocab_logprobs,
            mask, rewards, completion_lengths, truncated, centering, frac_parsed)

    eval_rows = []
    if cfg.eval.enabled:
        # Hold out the head of the shuffled training set rather than the
        # environment's own evaluation split, which can follow a different
        # distribution.
        eval_dataset = dataset.select(range(cfg.eval.num_prompts))
        dataset = dataset.select(range(cfg.eval.num_prompts, len(dataset)))
        eval_rows = [
            eval_dataset[index]
            for index in range(len(eval_dataset))
        ]
        if len(eval_rows) % num_prompts:
            raise ValueError(
                "eval.num_prompts must be a multiple of rl.num_prompts "
                "(eval runs in train-shaped chunks)")
    eval_key = jax.random.fold_in(key, 3)

    data_parallel = trainer_mesh.shape["data"]
    check_slice_rows(num_prompts * group_size, data_parallel)
    if eval_rows and len(eval_rows) * group_size % data_parallel:
        raise ValueError(
            "eval.num_prompts * rl.group_size must be divisible by "
            f"data parallelism ({data_parallel})")
    refresh_period = cfg.sampler.staleness + 1
    run_start = time.perf_counter()

    for step in tqdm(range(total_steps), desc="RL", disable=jax.process_index() != 0):
        if step % refresh_period == 0:
            sampler_weights = sampler_snapshot(model.weights)
        should_log = step % cfg.log.every_steps == 0 or step == total_steps - 1
        eval_metrics = {}
        if should_log and eval_rows and (
                step % cfg.eval.every_steps == 0
                or step == total_steps - 1):
            eval_weights = copy_sampler_weights(jax.block_until_ready(model.weights))
            # Eval never feeds a trainer step, so no distribution payload is
            # consumed regardless of the training sampler's transport spec.
            chunks = [
                sample(eval_rows[start:start + num_prompts], eval_weights,
                       jax.random.fold_in(eval_key, start),
                       vocab_logprobs_spec=None,
                       forward=eval_forward)
                for start in range(0, len(eval_rows), num_prompts)
            ]
            del eval_weights
            eval_metrics = {
                "eval_reward": np.mean([c[4].mean() for c in chunks]),
                "eval_mean_completion_len": np.mean([c[5].mean() for c in chunks]),
                "eval_frac_parsed": np.mean([c[8] for c in chunks]),
            }
        rows = [
            dataset[(step * num_prompts + offset) % len(dataset)]
            for offset in range(num_prompts)
        ]
        key, sample_key = jax.random.split(key)
        sampler_age = step % refresh_period  # optimizer steps since the refresh
        (tokens, sampling_token_logprobs, sampling_vocab_logprobs, mask, rewards,
         completion_lengths, truncated, centering, frac_parsed) = sample(
             rows, sampler_weights, sample_key, cfg.sampler.vocab_logprobs,
             tail_samples=cfg.sampler.tail_samples)
        advantages = losses.compute_advantages(rewards, group_size)
        kl_coeff = None
        if sampler_kl is not None:
            kl_coeff = float(sampler_kl.coefficient(sampler_age))
        reward = rewards.mean()
        batch = {
            "tokens": tokens,
            "mask": jax.device_put(mask, token_sharding),
            "advantages": jax.device_put(advantages, rollout_sharding),
            "sampling_token_logprobs": sampling_token_logprobs,
            **{name: reshard(value, center_sharding)
               for name, value in centering.items()},
        }
        if kl_coeff is not None:
            batch["kl_coeff"] = jax.device_put(np.float32(kl_coeff), NamedSharding(trainer_mesh, P()))
        if refresh_period == 1:
            del sampler_weights
        if sampling_vocab_logprobs is not None:
            # Reshard to the trainer mesh only once the sampler weights can be
            # dropped: the source, the destination and the collective's scratch
            # space add up to about three times the full-vocabulary array,
            # which sets the memory peak between steps.
            batch["sampling_vocab_logprobs"] = reshard(
                sampling_vocab_logprobs, vocab_sharding)
            del sampling_vocab_logprobs
        key, optimizer_key = jax.random.split(key)
        model.weights, optimizer_state, optimizer_metrics = optimize(
            model.weights, optimizer_state, batch, optimizer_key)
        del batch

        if should_log:
            optimizer_metrics = {name: np.asarray(value) for name, value in optimizer_metrics.items()}
            metrics = {
                "step": step,
                "reward": reward,
                "mean_completion_len": completion_lengths.mean(),
                "frac_nonzero_advantage": (advantages != 0).mean(),
                "frac_parsed": frac_parsed,
                "frac_truncated": truncated.mean(),
                "sampler_age": sampler_age,
                **({"kl_coeff": kl_coeff} if kl_coeff is not None else {}),
                "elapsed_s": time.perf_counter() - run_start,
                "weight_norm": utils.weight_norm(model.weights),
                **optimizer_metrics,
                **eval_metrics,
            }
            utils.log_metrics(step, metrics, run_dir)

    if cfg.log.save_model:
        utils.save_model(model, run_dir, "final_model")
    if jax.process_index() == 0:
        wandb.finish()


def validate(cfg):
    """Reject configurations the loop cannot run. Shared with async_rl.py."""
    if cfg.sampler.staleness < 0:
        raise ValueError("sampler.staleness must be non-negative")
    if cfg.sampler.kv_quant_bits not in (0, "fp8", "fp6", "fp4", *range(2, 17)):
        raise ValueError(
            "sampler.kv_quant_bits must be 0, in [2, 16], or 'fp8'/'fp6'/'fp4'")
    vocab_logprobs = cfg.sampler.vocab_logprobs
    if not (
            vocab_logprobs is None
            or vocab_logprobs == "full"
            or (isinstance(vocab_logprobs, int)
                and not isinstance(vocab_logprobs, bool)
                and vocab_logprobs > 0)):
        raise ValueError(
            "sampler.vocab_logprobs must be null, a positive integer, or 'full'")
    if not isinstance(cfg.rl.sc, bool):
        raise ValueError("rl.sc must be true or false")
    if cfg.rl.sc and vocab_logprobs is None:
        raise ValueError("rl.sc=true requires sampler.vocab_logprobs")
    topk_payload = isinstance(vocab_logprobs, int) and not isinstance(vocab_logprobs, bool)
    if (not isinstance(cfg.sampler.tail_samples, int)
            or isinstance(cfg.sampler.tail_samples, bool)
            or cfg.sampler.tail_samples < 0):
        raise ValueError("sampler.tail_samples must be a non-negative integer")
    if cfg.sampler.tail_samples and not topk_payload:
        raise ValueError(
            "sampler.tail_samples > 0 requires an integer sampler.vocab_logprobs "
            "(the top-k head the tail is drawn outside of)")
    skl = cfg.rl.sampler_kl
    if skl.coeff < 0:
        raise ValueError("rl.sampler_kl.coeff must be non-negative")
    if skl.coeff > 0:
        if not topk_payload:
            raise ValueError(
                "rl.sampler_kl.coeff > 0 requires an integer sampler.vocab_logprobs")
        if not skl.clip > 0:
            raise ValueError("rl.sampler_kl.clip must be positive")
    isc = cfg.rl["is"]
    if isinstance(isc.high, str):
        # CLI yaml reads bare 'inf' as a string ('.inf' is the yaml float).
        isc.high = float(isc.high)
    if isc.level is not None:
        if isc.level not in losses.IS_LEVELS:
            raise ValueError(f"rl.is.level must be null or one of {losses.IS_LEVELS}")
        if isc.outside not in losses.IS_OUTSIDE:
            raise ValueError(f"rl.is.outside must be one of {losses.IS_OUTSIDE}")
        if isc.stat not in losses.IS_STATS:
            raise ValueError(f"rl.is.stat must be one of {losses.IS_STATS}")
        if isc.sign not in losses.IS_SIGNS:
            raise ValueError(f"rl.is.sign must be one of {losses.IS_SIGNS}")
        if not 0 <= isc.low <= isc.high:
            raise ValueError("rl.is bounds must satisfy 0 <= low <= high")
        if isc.stat == "tv":
            if isc.level == "seq_sum":
                raise ValueError("rl.is.level=seq_sum requires rl.is.stat=ratio")
            if isc.outside != "drop" or isc.low != 0:
                raise ValueError(
                    "rl.is.stat=tv switches on the divergence: outside=drop, low=0")
            if not 0 < isc.high <= 1:
                raise ValueError(
                    "rl.is.high is the TV threshold delta, in (0, 1]")
        elif isc.outside == "drop" and not isc.low <= 1 <= isc.high:
            raise ValueError(
                "rl.is drop band must contain the on-policy ratio 1")
        if isc.sign == "ppo" and isc.outside != "drop":
            raise ValueError("rl.is.sign=ppo is the PPO surrogate: outside=drop")
    if cfg.rl.sc:
        if isc.level is not None and isc.stat != "ratio":
            raise ValueError(
                "score centering composes only with rl.is.stat=ratio")
    if cfg.rl.group_size < 1 or cfg.rl.num_prompts < 1:
        raise ValueError("rl.group_size and rl.num_prompts must be positive")
    if cfg.stop.steps < 1:
        raise ValueError("stop.steps must be positive")
    if cfg.eval.save_samples < 0:
        raise ValueError("eval.save_samples must be non-negative")


def main():
    # Spawn, never fork: a scorer worker forked from this threaded CUDA
    # process inherits its state and can fail silently.
    multiprocessing.set_start_method("spawn", force=True)
    cfg = utils.parse_cfg(DEFAULT_CFG)
    validate(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
