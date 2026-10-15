import json
import operator as op
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import wandb
from omegaconf import OmegaConf


def parse_cfg(default_config):
    cfg = OmegaConf.create(default_config)
    OmegaConf.set_struct(cfg, True)
    if "env" in cfg and cfg.env.args is not None:
        # Free-form load_environment kwargs; the keyset varies by environment.
        OmegaConf.set_struct(cfg.env.args, False)
    overrides = OmegaConf.from_cli()
    primary = jax.process_index() == 0
    if primary:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project=OmegaConf.select(overrides, "log.project", default=cfg.log.project),
            mode=OmegaConf.select(overrides, "log.wandb_mode", default=cfg.log.wandb_mode),
            name=OmegaConf.select(overrides, "log.run_name", default=cfg.log.run_name),
        )
        print(f"{wandb.run.id=}")
    cfg = OmegaConf.merge(cfg, overrides)
    if primary:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
    return cfg


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def make_optimizer(opt, total_steps, weights):
    warmup_steps = int(opt.lr_warmup * total_steps)
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(0, opt.lr, warmup_steps),
            optax.constant_schedule(opt.lr),
        ],
        [warmup_steps],
    )
    if opt.optimizer == "sgd":
        optimizer = optax.sgd(lr_schedule)
    elif opt.optimizer == "adamw":
        optimizer = optax.adamw(lr_schedule, 0.9, opt.b2, weight_decay=0.02)
    else:
        raise ValueError(f"Unknown optimizer: {opt.optimizer}")
    return optimizer, optimizer.init(weights)


def start_run(train_config):
    train_config = OmegaConf.to_container(train_config, resolve=True)
    primary = jax.process_index() == 0
    resolved_run_name = train_config["log"]["run_name"] or (wandb.run.id if primary else "run")
    run_dir = Path(train_config["log"]["dir"]).expanduser() / train_config["log"]["project"] / resolved_run_name
    if primary:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "config.json", train_config)
        print(f"run_dir={run_dir}")
    return run_dir


def save_model(model, run_dir, name="final_model"):
    if jax.process_index() == 0:
        output_dir = Path(run_dir) / name
        model.save(model, output_dir)
        print(f"saved_model_dir={output_dir}")


def peak_memory_gb():
    """Peak device memory in use by this process, in GiB, over its local
    devices; 0.0 where the backend reports no memory statistics (CPU)."""
    peak = 0
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        peak = max(peak, stats.get("peak_bytes_in_use", 0))
    return peak / 2 ** 30


def weight_norm(weights):
    # f32 accumulation: a bf16 result is too coarse to show the norm drifting.
    return jax.tree.reduce_associative(
        op.add, jax.tree.map(lambda w: (w.astype(jnp.float32) ** 2).sum(), weights)) ** 0.5


def _stochastic_round_bf16(x, key):
    """f32 -> bf16 rounded up/down with probability given by the remainder.

    Stays entirely in 32-bit types: 16-bit random bits / uint16 bitcasts need
    width-changing reshapes that the TPU compiler lowers through its
    convolution emitter, whose fusion cost model overflows the stack on
    fused graphs the size of an opt_step. Zeroing the low mantissa bits makes
    the final f32 -> bf16 convert exact.
    """
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    rnd = jax.random.bits(key, x.shape, jnp.uint32, out_sharding=jax.typeof(x).sharding.spec)
    bits = (bits + (rnd & jnp.uint32(0xFFFF))) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(jnp.bfloat16)


def apply_updates(params, updates, key):
    """optax.apply_updates, with stochastic rounding for bfloat16 params.

    Round-to-nearest freezes bf16 masters whenever the per-element step is
    below half an ULP (about 2e-3 relative), which RL learning rates always
    are, and without momentum the rounded-away updates never accumulate.
    Stochastic rounding keeps the expected update instead.
    """
    leaves, treedef = jax.tree.flatten(params)
    if not any(l.dtype == jnp.bfloat16 for l in leaves):
        return optax.apply_updates(params, updates)
    keys = jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))

    def apply(p, u, k):
        if p.dtype != jnp.bfloat16:
            return (p + u).astype(p.dtype)
        return _stochastic_round_bf16(p.astype(jnp.float32) + u.astype(jnp.float32), k)

    return jax.tree.map(apply, params, updates, keys)


def log_metrics(step, metrics, run_dir=None):
    metrics = jax.tree.map(lambda x: x.item() if hasattr(x, "item") else x, metrics)
    if jax.process_index() == 0:
        print(f"Step {step}", metrics)
        wandb.log(metrics, step=step, commit=True)
        if run_dir is not None:
            with open(Path(run_dir) / "logs.jsonl", "a") as f:
                f.write(json.dumps({"step": step, **metrics}, default=str) + "\n")
