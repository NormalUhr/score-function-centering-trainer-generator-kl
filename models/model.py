import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh, PartitionSpec as P, reshard
from safetensors import safe_open
from safetensors.numpy import save_file
from transformers import AutoTokenizer

from models import quant
from models.quant import qeinsum


@dataclass
class Model:
    source: str
    weights: dict[str, Any]
    forward: Callable
    head: Callable
    init_kv: Callable
    tokenizer: Any
    config: dict[str, Any]
    save: Callable
    stop_ids: tuple[int, ...]  # token ids that end a generated turn


def make_mesh(data_parallel, tensor_parallel):
    """A (data, model) mesh over every device of the program. Built from
    mesh_utils rather than jax.make_mesh, which refuses devices of several
    hosts; the GPUs are ordered by id, so a host's devices are contiguous
    along the model axis whenever tensor parallelism stays within a host."""
    devices = mesh_utils.create_device_mesh((data_parallel, tensor_parallel))
    return Mesh(devices, ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))


def place(value, sharding):
    """Put a host array onto `sharding` (a NamedSharding, or a PartitionSpec
    under the current mesh) from per-device pieces. jax.device_put of a whole
    host array onto a sharding that spans several processes takes seconds per
    gigabyte; placing per-device pieces takes milliseconds."""
    if isinstance(sharding, P):
        sharding = jax.sharding.NamedSharding(jax.sharding.get_mesh(), sharding)
    pieces = [jax.device_put(value[index], device)
              for device, index in sharding.addressable_devices_indices_map(value.shape).items()]
    return jax.make_array_from_single_device_arrays(value.shape, sharding, pieces)


def apply_rope(x, cfg, pos=0):
    B, T, N, H = x.shape
    # GLM rotates only the first partial_rotary_factor of every head; the
    # remaining dims carry no position and pass through unchanged.
    R = int(H * cfg.get("partial_rotary_factor", 1.0))
    positions = pos + jnp.broadcast_to(jnp.arange(T)[None, :], [B, T])
    freq = 1.0 / (cfg["rope_theta"] ** (jnp.arange(0, R, 2, dtype=jnp.float32) / R))
    inp = jnp.einsum("bt,h->bth", positions, freq, precision=jax.lax.Precision.HIGHEST)
    sin, cos = jnp.sin(inp).astype(x.dtype), jnp.cos(inp).astype(x.dtype)
    x1, x2 = x[:, :, :, :R // 2], x[:, :, :, R // 2:R]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[:, :, :, R:]], axis=-1)


def rms_norm(x, gamma, eps, axis=-1):
    rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=axis, keepdims=True) + eps)
    return (gamma * x / rms).astype(x.dtype)


def forward_mlp(x, w, prefix="mlp."):
    """SwiGLU MLP with HF weight names under `prefix` (the dense layers' mlp.
    and the MoE layers' mlp.shared_experts.)."""
    gate = jax.nn.silu(qeinsum("btd,fd->btf", x, w[prefix + "gate_proj.weight"], preferred_element_type=jnp.float32, out_sharding=P("data", None, "model")))
    up = qeinsum("btd,fd->btf", x, w[prefix + "up_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model"))
    return qeinsum("btf,df->btd", gate * up, w[prefix + "down_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, None))


def route(cfg, x, w):
    """Expert weights and indices [B, T, K] of every token. Qwen scores with a
    softmax over the experts; GLM (as DeepSeek V3) scores with a sigmoid and
    picks the experts by score plus a per-expert correction bias, a
    load-balancing term that does not enter the weights."""
    K = cfg["num_experts_per_tok"]
    logits = jnp.einsum("btd,ed->bte", x, w["mlp.gate.weight"], preferred_element_type=jnp.float32, out_sharding=P("data", None, None))
    if "mlp.gate.e_score_correction_bias" in w:
        scores = jax.nn.sigmoid(logits)
        _, idx = jax.lax.top_k(scores + w["mlp.gate.e_score_correction_bias"], K)
        probs = jnp.take_along_axis(scores, idx, axis=-1)
        if cfg["norm_topk_prob"]:
            probs = probs / (probs.sum(-1, keepdims=True) + 1e-20)
        probs = probs * cfg["routed_scaling_factor"]
    else:
        probs, idx = jax.lax.top_k(jax.nn.softmax(logits, -1), K)
        probs = probs / probs.sum(-1, keepdims=True)
    return probs.astype(x.dtype), idx


def forward_moe(cfg, x, w, live=None):
    E, K = _num_experts(cfg), cfg["num_experts_per_tok"]
    if live is None:  # callers without a mask dispatch every row
        live = jnp.ones(x.shape[:2], dtype=bool, out_sharding=P("data", None))
    probs, idx = route(cfg, x, w)

    is_q = lambda v: isinstance(v, quant.QArray)

    def experts(x, probs, idx, live, wg, wu, wd):
        shape = x.shape  # this chip's batch shard; tokens are exchanged, expert weights stay put
        x, probs, idx, live = (jax.lax.all_gather(v, "data", axis=0, tiled=True) for v in (x, probs, idx, live))
        S, D = x.shape[0] * x.shape[1], x.shape[-1]
        El = (wg.q if is_q(wg) else wg).shape[0]  # experts on this chip
        eid = idx.reshape(-1) - jax.lax.axis_index(("data", "model")) * El
        # Dead (padding) rows never claim a slot in the gathered-row buffer.
        local = (eid >= 0) & (eid < El) & jnp.repeat(live.reshape(-1), K)
        order = jnp.argsort(jnp.where(local, eid, El))  # rows for remote experts sort last
        # Row buffer 2x the mean chip load: keeps every live local row unless a
        # chip is >2x overloaded, and drops most of the remote rows' gather
        # traffic. The surviving remote rows run through the last local expert
        # (rather than falling outside group_sizes, whose gradients are
        # undefined) and their outputs are masked below.
        order = order[:min(S * K, max(2 * S * K * El // E, 128))]
        sel = jnp.where(local, eid, El)[order]
        rows, gid = x.reshape(S, D)[order // K], jnp.minimum(sel, El - 1)
        sizes = jnp.bincount(gid, length=El).astype(jnp.int32)
        h = jax.nn.silu(quant.qragged_dot(rows, wg, sizes, gid)) * quant.qragged_dot(rows, wu, sizes, gid)
        y = jnp.where((sel < El)[:, None], quant.qragged_dot(h, wd, sizes, gid), 0)
        y = jnp.zeros((S, D), x.dtype).at[order // K].add(y * probs.reshape(-1)[order][:, None])
        y = jax.lax.psum_scatter(y, "data", scatter_dimension=0, tiled=True)
        return jax.lax.psum(y, "model").reshape(shape)

    s = P(("data", "model"), None, None)  # every chip holds E / n_chips full-width experts
    wg, wu, wd = (w[f"mlp.experts.{p}_proj.weight"] for p in ("gate", "up", "down"))
    wg, wu, wd = (quant.QArray(reshard(v.q, s), reshard(v.scale, P(("data", "model"), None))) if is_q(v)
                  else reshard(v, s) for v in (wg, wu, wd))
    spec = lambda v: quant.QArray(s, P(("data", "model"), None)) if is_q(v) else s
    # Ties the expert weight grads into the backward's critical path; otherwise
    # the scheduler sinks all layers' weight-grad dots to the end of the program
    # (their only consumer), keeping every layer's dispatch buffers live.
    x, wg, wu, wd = jax.lax.optimization_barrier((x, wg, wu, wd))
    # check_vma=False: tokamax.ragged_dot's custom VJP does not annotate
    # manual axes and trips the varying-mesh-axes validation.
    y = jax.shard_map(experts, in_specs=(P("data", None, None),) * 3 + (P("data", None), spec(wg), spec(wu), spec(wd)),
                      out_specs=P("data", None, None), check_vma=False)(x, probs, idx, live, wg, wu, wd)
    if "mlp.shared_experts.gate_proj.weight" in w:  # GLM: one dense expert serves every token
        y += forward_mlp(x, w, "mlp.shared_experts.")
    return y


def attention(q, k, v, mask=None, *, is_causal=False):
    # Attention is independent across batch rows and KV-head groups. Run it
    # locally so JAX's internal GQA reshape need not infer explicit sharding.
    # jax.nn.dot_product_attention defaults to the XLA kernel, which
    # materializes the [B, N, T, T] f32 logits (26 GB per device at T = 8192
    # with 96 heads); the causal path takes cuDNN flash attention on GPUs,
    # which needs 16-bit inputs and no mask array.
    flash = (is_causal and mask is None and jax.default_backend() == "gpu"
             and q.dtype in (jnp.bfloat16, jnp.float16))
    heads = P("data", None, "model", None)
    return jax.shard_map(
        lambda q, k, v, mask: jax.nn.dot_product_attention(
            q, k, v, mask=mask, is_causal=is_causal,
            implementation="cudnn" if flash else None),
        in_specs=(heads, heads, heads, P("data", None, None, None) if mask is not None else None),
        out_specs=heads,
    )(q, k, v, mask)


def forward_layer(cfg, x, w, kv=None, pos=0, live=None, kv_quant_bits=0):
    x_norm = rms_norm(x, w["input_layernorm.weight"], cfg["rms_norm_eps"])

    q = qeinsum("btd,nhd->btnh", x_norm, w["self_attn.q_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))
    k = qeinsum("bsd,khd->bskh", x_norm, w["self_attn.k_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))
    v = qeinsum("bsd,khd->bskh", x_norm, w["self_attn.v_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))

    if "self_attn.q_proj.bias" in w:
        q += w["self_attn.q_proj.bias"]
        k += w["self_attn.k_proj.bias"]
        v += w["self_attn.v_proj.bias"]

    if "self_attn.q_norm.weight" in w:
        q = rms_norm(q, w["self_attn.q_norm.weight"], cfg["rms_norm_eps"])
        k = rms_norm(k, w["self_attn.k_norm.weight"], cfg["rms_norm_eps"])

    q = apply_rope(q, cfg, pos)
    k = apply_rope(k, cfg, pos)

    if kv is not None:  # decode: one token per row, pos is per-slot [B, 1]
        upd = jnp.stack([k, v])
        if kv_quant_bits:  # int-N / fp8 / fp6 / fp4, see quant.kv_fake_quantize
            upd = quant.kv_fake_quantize(upd, kv_quant_bits)
        kv = jax.vmap(lambda c, u, p: jax.lax.dynamic_update_slice(c, u, (0, p, 0, 0)),
                      in_axes=(1, 1, 0), out_axes=1)(kv, upd, pos[:, 0])
        k, v = kv
        attn_mask = (jnp.arange(k.shape[1])[None, None, :] <= pos[:, :, None])[:, None]
        attn_out = attention(q, k, v, mask=attn_mask)
    else:  # training: causal flash attention, no mask array
        attn_out = attention(q, k, v, is_causal=True)
    x += qeinsum("btnh,dnh->btd", attn_out, w["self_attn.o_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, None))

    x_norm = rms_norm(x, w["post_attention_layernorm.weight"], cfg["rms_norm_eps"])
    if "mlp.gate.weight" in w:
        x += forward_moe(cfg, x_norm, w, live)
    else:
        x += forward_mlp(x_norm, w)
    return x, kv


def head(cfg, x, weights, dtype=jnp.bfloat16):
    is_q = lambda w: isinstance(w, quant.QArray)
    out_embed = weights["model.embed_tokens.weight"] if cfg["tie_word_embeddings"] else weights["lm_head.weight"]
    if not is_q(out_embed): out_embed = out_embed.astype(dtype)
    return qeinsum("btd,vd->btv", x, out_embed, preferred_element_type=x.dtype, out_sharding=P("data", None, "model"))


def forward(cfg, x, weights, kv=None, pos=0, dtype=jnp.bfloat16, logits=True, live=None, kv_quant_bits=0):
    is_q = lambda w: isinstance(w, quant.QArray)
    # Compute cast per layer, inside the remat region: casting the whole
    # tree up front keeps every layer's compute copy live into the backward and
    # blocks optimizer-backward fusion (fp32 masters: 33.7G -> 13.9G temps).
    cast = lambda w: w.astype(dtype) if not is_q(w) and w.ndim > 1 else w
    x = reshard(x, P("data", None))
    x = quant.take(cast(weights["model.embed_tokens.weight"]), x,
                   out_sharding=P("data", None, None)).astype(dtype)

    # Layer weights are stacked [L, ...]; scanning keeps the compiled program
    # one layer deep, so each layer's remat/backward buffers are reused
    # instead of packed side by side (unrolled: 62 GiB of GPU temps at B128).
    # The dense layers ahead of a MoE stack (GLM's first layer) hold other
    # weights than the rest, so they stay unstacked and run first.
    prefix, layers = _split_layers(weights)

    def layer_fn(x, layer):
        lw, kv_i = layer
        return forward_layer(cfg, x, jax.tree.map(cast, lw, is_leaf=is_q), kv_i, pos, live, kv_quant_bits)

    return_kv = kv is not None
    if kv is None:
        for lw in prefix:
            x, _ = jax.remat(layer_fn)(x, (lw, None))
        x, kv = jax.lax.scan(jax.remat(layer_fn), x, (layers, kv))
    else:
        # Decode: unrolled layers, each updating its own cache entry in
        # place. The scan path would rebuild a stacked cache (ys) every
        # token and nest a while loop inside the decode loop.
        kv = list(kv)
        for i in range(len(kv)):
            lw = prefix[i] if i < len(prefix) else jax.tree.map(lambda w: w[i - len(prefix)], layers)
            x, kv[i] = layer_fn(x, (lw, kv[i]))

    x = rms_norm(x, weights["model.norm.weight"], cfg["rms_norm_eps"])
    if logits:
        x = head(cfg, x, weights, dtype)
    return (x, kv) if return_kv else x


def init_kv(L, K, H, B, T, dtype=jnp.bfloat16):
    # One cache array per layer: a stacked cache would be rewritten whole on
    # every decoded token instead of aliased in the decode loop carry.
    return [
        jnp.zeros(
            (2, B, T, K, H), dtype=dtype,
            out_sharding=P(None, "data", None, "model", None))
        for _ in range(L)
    ]


def _head_dim(cfg):
    return cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])


def _num_experts(cfg):
    """Routed expert count, None for a dense model (Qwen: num_experts, GLM: n_routed_experts)."""
    return cfg.get("num_experts", cfg.get("n_routed_experts"))


def _dense_prefix(cfg):
    """Layers ahead of the MoE stack that keep a dense MLP (GLM's first_k_dense_replace)."""
    return cfg.get("first_k_dense_replace", 0)


def _layer_index(key):
    """Layer index of an unstacked per-layer key model.layers.<i>.<name>, else None."""
    index = key.removeprefix("model.layers.").partition(".")[0]
    return int(index) if key.startswith("model.layers.") and index.isdigit() else None


def _split_layers(weights):
    """Per-layer weights of a tree, without the model.layers. prefix: the
    unstacked prefix layers as one dict per layer in order, and the stacked
    layers as one dict of [L, ...] arrays."""
    prefix, stacked = defaultdict(dict), {}
    for key, value in weights.items():
        if not key.startswith("model.layers."):
            continue
        index, name = _layer_index(key), key.removeprefix("model.layers.")
        if index is None:
            stacked[name] = value
        else:
            prefix[index][name.partition(".")[2]] = value
    return [prefix[i] for i in range(len(prefix))], stacked


def stop_token_ids(cfg, tokenizer):
    """Token ids that end a generated turn. A config may list several (GLM
    also closes a turn with <|user|> or <|observation|>); with a single id
    the tokenizer's eos applies, which post-trained checkpoints may set
    apart from the config's."""
    ids = cfg.get("eos_token_id")
    if isinstance(ids, list):
        return tuple(int(i) for i in ids)
    return () if tokenizer.eos_token_id is None else (tokenizer.eos_token_id,)


def _make_model(source, cfg, weights, tokenizer):
    return Model(
        source=str(source),
        weights=weights,
        forward=partial(forward, cfg),
        head=partial(head, cfg),
        init_kv=partial(init_kv, cfg["num_hidden_layers"], cfg["num_key_value_heads"], _head_dim(cfg)),
        tokenizer=tokenizer,
        config=cfg,
        save=save,
        stop_ids=stop_token_ids(cfg, tokenizer),
    )


def _load_template(model_id, hf_ckpt_dir, allow_patterns=None):
    source = Path(model_id).expanduser()
    if not source.exists():
        source = Path(hf_ckpt_dir).expanduser() / model_id
        snapshot_download(repo_id=model_id, local_dir=source, allow_patterns=allow_patterns)
    cfg = json.loads((source / "config.json").read_text())
    if cfg.get("model_type") not in ("qwen3", "qwen3_moe", "glm4_moe"):
        raise ValueError(f"Only Qwen3 and GLM-4.5 checkpoints are supported, got model_type={cfg.get('model_type')}")
    if cfg.get("n_group", 1) != 1:
        raise ValueError(f"grouped expert routing is not implemented, got n_group={cfg['n_group']}")
    if cfg.get("use_sliding_window"):
        raise ValueError("sliding-window attention is not implemented (attention here is always full)")
    if cfg.get("rope_scaling"):
        raise ValueError(f"rope_scaling is not implemented, got {cfg['rope_scaling']}")
    tokenizer = AutoTokenizer.from_pretrained(source)
    return source, tokenizer, cfg


def _make_random_cfg(template_cfg, hidden_size=None, num_hidden_layers=None):
    hidden_size = template_cfg["hidden_size"] if hidden_size is None else int(hidden_size)
    num_hidden_layers = template_cfg["num_hidden_layers"] if num_hidden_layers is None else int(num_hidden_layers)
    if hidden_size < 128 or hidden_size % 128 != 0:
        raise ValueError(f"hidden_size must be a positive multiple of 128, got {hidden_size}")
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

    cfg = dict(template_cfg)
    cfg["hidden_size"] = hidden_size
    cfg["num_hidden_layers"] = num_hidden_layers
    cfg["num_attention_heads"] = hidden_size // 128
    cfg["num_key_value_heads"] = math.gcd(cfg["num_attention_heads"], 8)
    cfg["intermediate_size"] = 3 * hidden_size
    if "max_window_layers" in cfg: cfg["max_window_layers"] = min(cfg["max_window_layers"], num_hidden_layers)
    if "layer_types" in cfg: cfg["layer_types"] = ["full_attention"] * num_hidden_layers
    return cfg


def _get_sharding(key, dp_shard=False):
    """Per-layer weights are stacked [L, ...]: their specs gain a leading None.
    The unstacked prefix layers (model.layers.<i>.<name>) keep the plain spec."""
    if "norm" in key or "bias" in key or key.endswith("mlp.gate.weight"): return P()
    if "mlp.experts." in key:  # stacked (E, D, F) gate/up and (E, F, D) down: always fully expert-parallel
        spec = P(("data", "model"), None, None)
    elif any(k in key for k in ("q_proj", "k_proj", "v_proj")):
        spec = P("model", None, "data") if dp_shard else P("model", None, None)
    elif "o_proj" in key:
        spec = P("data", "model", None) if dp_shard else P(None, "model", None)
    elif any(k in key for k in ("gate_proj", "up_proj", "embed_tokens", "lm_head")):
        spec = P("model", "data") if dp_shard else P("model")
    elif "down_proj" in key:
        spec = P("data", "model") if dp_shard else P(None, "model")
    else:
        raise ValueError(f"Unrecognized key: {key}")
    return P(None, *spec) if key.startswith("model.layers.") and _layer_index(key) is None else spec


def _load_weights(model_ckpt_dir, cfg, dp_shard=False):
    N, K, D, H = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"], _head_dim(cfg)
    L, prefix, E = cfg["num_hidden_layers"], _dense_prefix(cfg), _num_experts(cfg)
    weights, experts, layered = {}, defaultdict(dict), defaultdict(dict)

    def add(key, value):  # model.layers.{i}.{name} past the dense prefix collects into stacked [L - prefix, ...]
        index = _layer_index(key)
        # Every tensor goes to the devices as soon as it is read, each layer of
        # a stack sharded like its slice; the stack is then joined on the
        # devices. Joining on the host would hold the whole model there
        # (about 400 GB for GLM-4.5-Air with the stacking copies).
        value = place(value, _get_sharding(key, dp_shard))
        if index is not None and index >= prefix:
            name = key.removeprefix("model.layers.").partition(".")[2]
            layered[f"model.layers.{name}"][index - prefix] = value
        else:
            weights[key] = value

    # The chunked reader streams each file with eight threads, which reads a
    # large checkpoint from a network filesystem far faster than the mmap page
    # faults of safe_open.
    files = sorted(model_ckpt_dir.glob("*.safetensors"))
    started = time.perf_counter()
    for number, file in enumerate(files, 1):
        if jax.process_index() == 0:  # a large checkpoint takes minutes to load; the log shows where the time goes
            print(f"[load] file {number}/{len(files)} {file.name} at {time.perf_counter() - started:.0f}s", flush=True)
        for key, value in _read_safetensors(file):
            index = _layer_index(key)
            if index is not None and index >= L:  # GLM's multi-token-prediction layer is not part of the model
                continue
            if ".mlp.experts." in key:  # mlp.experts.{i}.{proj}.weight: one layer's experts stack as soon as all are read
                layer, _, rest = key.partition(".mlp.experts.")
                group = f"{layer}.mlp.experts.{rest.partition('.')[2]}"
                experts[group][int(rest.partition(".")[0])] = value
                if len(experts[group]) == E:  # (E, out, in) -> (E, in, out): the expert dot contracts the middle axis
                    parts = experts.pop(group)
                    add(group, np.stack([parts[e] for e in range(E)]).swapaxes(1, 2))
                continue
            if "q_proj.bias" in key: value = value.reshape([N, H])
            if "k_proj.bias" in key or "v_proj.bias" in key: value = value.reshape([K, H])
            if "q_proj.weight" in key: value = value.reshape([N, H, D])
            if "k_proj.weight" in key or "v_proj.weight" in key: value = value.reshape([K, H, D])
            if "o_proj.weight" in key: value = value.reshape([D, N, H])
            add(key, value)
    if experts:
        raise ValueError(f"incomplete expert groups in {model_ckpt_dir}: {sorted(experts)[:3]}")
    for key in list(layered):
        parts = layered.pop(key)
        weights[key] = reshard(jnp.stack([parts[i] for i in range(L - prefix)]), _get_sharding(key, dp_shard))
        if jax.process_index() == 0:
            print(f"[load] stacked {key} at {time.perf_counter() - started:.0f}s", flush=True)
    if not weights:
        raise ValueError(f"No safetensors files found in {model_ckpt_dir}")
    return weights


def _random_weights(cfg, key, dp_shard=False):
    std = cfg["initializer_range"]
    D, F, N, K, H, V = cfg["hidden_size"], cfg.get("intermediate_size"), cfg["num_attention_heads"], cfg["num_key_value_heads"], _head_dim(cfg), cfg["vocab_size"]
    L, prefix, E = cfg["num_hidden_layers"], _dense_prefix(cfg), _num_experts(cfg)
    weights = {}

    def add(name, shape, ones=False, dtype=jnp.bfloat16):
        nonlocal key
        if ones:
            value = jnp.ones(shape, dtype=dtype)
        else:
            key, subkey = jax.random.split(key)
            value = (jax.random.normal(subkey, shape, dtype=jnp.float32) * std).astype(dtype)
        weights[name] = jax.device_put(value, _get_sharding(name, dp_shard))

    add("model.embed_tokens.weight", (V, D))
    add("model.norm.weight", (D,), ones=True)
    if not cfg["tie_word_embeddings"]: add("lm_head.weight", (V, D))

    attention = [
        ("input_layernorm.weight", (D,), True),
        ("post_attention_layernorm.weight", (D,), True),
        ("self_attn.q_proj.weight", (N, H, D), False),
        ("self_attn.k_proj.weight", (K, H, D), False),
        ("self_attn.v_proj.weight", (K, H, D), False),
        ("self_attn.o_proj.weight", (D, N, H), False),
    ]
    if cfg.get("attention_bias"):
        attention += [("self_attn.q_proj.bias", (N, H), False), ("self_attn.k_proj.bias", (K, H), False),
                      ("self_attn.v_proj.bias", (K, H), False)]
    if cfg["model_type"] in ("qwen3", "qwen3_moe") or cfg.get("use_qk_norm"):
        attention += [("self_attn.q_norm.weight", (H,), True), ("self_attn.k_norm.weight", (H,), True)]
    dense = [
        ("mlp.gate_proj.weight", (F, D), False),
        ("mlp.up_proj.weight", (F, D), False),
        ("mlp.down_proj.weight", (D, F), False),
    ]
    if E is not None:
        Fm = cfg["moe_intermediate_size"]
        mlp = [
            ("mlp.gate.weight", (E, D), False),
            ("mlp.experts.gate_proj.weight", (E, D, Fm), False),
            ("mlp.experts.up_proj.weight", (E, D, Fm), False),
            ("mlp.experts.down_proj.weight", (E, Fm, D), False),
        ]
        if cfg["model_type"] == "glm4_moe":  # a shared expert next to the routed ones
            Fs = Fm * cfg["n_shared_experts"]
            mlp += [("mlp.shared_experts.gate_proj.weight", (Fs, D), False),
                    ("mlp.shared_experts.up_proj.weight", (Fs, D), False),
                    ("mlp.shared_experts.down_proj.weight", (D, Fs), False)]
    else:
        mlp = dense
    for i in range(prefix):  # dense layers ahead of the stack keep their own HF keys
        for name, shape, ones in attention + dense:
            add(f"model.layers.{i}.{name}", shape, ones)
    for name, shape, ones in attention + mlp:  # the remaining layers are stacked [L - prefix, ...]
        add("model.layers." + name, (L - prefix, *shape), ones)
    if E is not None and cfg["model_type"] == "glm4_moe":  # the router's correction bias stays float32, as in the checkpoints
        add("model.layers.mlp.gate.e_score_correction_bias", (L - prefix, E), dtype=jnp.float32)
    return weights


def load(model_id="Qwen/Qwen3-0.6B-Base", hf_ckpt_dir="~/.cache/postax/weights", tp_size=1, dp_shard=False, init="pretrained", hidden_size=None, num_hidden_layers=None, key=None):
    if jax.device_count() % tp_size != 0:
        raise ValueError(f"tp_size={tp_size} does not divide device_count={jax.device_count()}")
    mesh = make_mesh(jax.device_count() // tp_size, tp_size)
    jax.set_mesh(mesh)
    has_architecture_override = hidden_size is not None or num_hidden_layers is not None

    if init == "pretrained":
        if has_architecture_override:
            raise ValueError("hidden_size and num_hidden_layers overrides are only supported when init='random'")
        model_ckpt_dir, tokenizer, cfg = _load_template(model_id, hf_ckpt_dir)
        return _make_model(model_id, cfg, _load_weights(model_ckpt_dir, cfg, dp_shard), tokenizer)

    if init == "random":
        if key is None:
            raise ValueError("Random initialization requires a JAX PRNG key")
        _, tokenizer, template_cfg = _load_template(model_id, hf_ckpt_dir, ["*.json", "*.txt", "*.jinja"])
        cfg = _make_random_cfg(template_cfg, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        return _make_model(model_id, cfg, _random_weights(cfg, key, dp_shard), tokenizer)

    raise ValueError(f"Unknown init mode: {init}")


def save(model, output_dir):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _dense_prefix(model.config)
    hf = {}
    for k, value in model.weights.items():
        # The router's correction bias picks experts by small score
        # differences; the reference keeps it float32 among bf16 weights.
        dtype = jnp.float32 if k.endswith("e_score_correction_bias") else jnp.bfloat16
        value = np.asarray(jax.device_get(jnp.asarray(value, dtype=dtype)))
        stacked = k.startswith("model.layers.") and _layer_index(k) is None
        name = k.removeprefix("model.layers.")
        for i, v in enumerate(value if stacked else [value]):
            k = f"model.layers.{prefix + i}.{name}" if stacked else k
            if ".mlp.experts." in k:  # unstack (E, in, out) back to per-expert HF (out, in)
                layer, _, proj = k.partition(".mlp.experts.")
                for e in range(v.shape[0]):
                    # safetensors writes the raw buffer, so a transposed view
                    # must be made contiguous first.
                    hf[f"{layer}.mlp.experts.{e}.{proj}"] = np.ascontiguousarray(v[e].T)
                continue
            if "q_proj" in k or "k_proj" in k or "v_proj" in k: v = v.reshape(-1, v.shape[-1]) if v.ndim == 3 else v.reshape(-1)
            if "o_proj.weight" in k: v = v.reshape(v.shape[0], -1)
            hf[k] = v

    save_file(hf, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(json.dumps(model.config, indent=2) + "\n")
    model.tokenizer.save_pretrained(output_dir)


def _shard_blocks(array, devices):
    """One (index, block) per distinct shard of `array` that lives on `devices`;
    replicas of a shard on several of those devices are taken once."""
    blocks = {}
    for shard in array.addressable_shards:
        if shard.device in devices and shard.index not in blocks:
            blocks[shard.index] = shard.data
    return blocks


def save_exchange(weights, output_dir, part=None, devices=None, float_dtype=None):
    """Write the shards of `weights` that live on `devices` (this process's
    devices by default) to weights-<part>.safetensors, one tensor per shard
    with its slices recorded in the file's metadata, so that a trainer whose
    tensors are split over several processes publishes without ever holding a
    whole copy: each process writes what its devices hold and `load_exchange`
    joins the parts. `float_dtype` casts matrices (norm gains and other
    vectors keep their dtype) on their device before the host transfer. The
    file appears under its final name only once complete."""
    part = jax.process_index() if part is None else part
    devices = set(jax.local_devices() if devices is None else devices)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors, layout = {}, {}
    for name, array in weights.items():
        cast = float_dtype is not None and array.ndim > 1 and "norm" not in name
        entry = layout[name] = {"shape": list(array.shape), "shards": {}}
        for number, (index, block) in enumerate(_shard_blocks(array, devices).items()):
            key = f"{name}@{part}.{number}"
            tensors[key] = np.asarray(block.astype(float_dtype) if cast else block)
            entry["shards"][key] = [[axis.start or 0, size if axis.stop is None else axis.stop]
                                    for axis, size in zip(index, array.shape)]
    staging = output_dir / f"tmp-weights-{part}.safetensors"
    save_file(tensors, str(staging), metadata={"layout": json.dumps(layout)})
    os.replace(staging, output_dir / f"weights-{part}.safetensors")


def exchange_files(directory):
    """The safetensors files of a published version, in part order: one
    weights-<part>.safetensors per trainer process."""
    return sorted(Path(directory).expanduser().glob("weights-*.safetensors"))


SAFETENSORS_DTYPES = {"BF16": jnp.bfloat16, "F16": np.float16, "F32": np.float32, "F64": np.float64,
                      "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
                      "U8": np.uint8, "BOOL": np.bool_}


class _SafetensorsFile:
    """A safetensors file read tensor by tensor in `chunk_bytes` pieces by
    `workers` threads (os.preadv releases the interpreter lock): the stacked
    layout keeps a model's largest weights in a few tensors of many GB, and
    one stream reads a network filesystem at a fraction of the bandwidth
    that eight streams reach."""

    def __init__(self, path, workers=8, chunk_bytes=256 << 20):
        self.path = str(path)
        with open(self.path, "rb") as f:
            header_length = int.from_bytes(f.read(8), "little")
            self.header = json.loads(f.read(header_length))
        self.base = 8 + header_length
        self.metadata = self.header.pop("__metadata__", None) or {}
        self.fd = os.open(self.path, os.O_RDONLY)
        self.pool = ThreadPoolExecutor(workers)
        self.chunk_bytes = chunk_bytes

    def _read_chunk(self, view, file_offset, start, end):
        while start < end:
            count = os.preadv(self.fd, [view[start:end]], file_offset + start)
            if count <= 0:
                raise IOError(f"short read in {self.path}")
            start += count

    def read(self, name):
        meta = self.header[name]
        start, end = meta["data_offsets"]
        buffer = np.empty(end - start, dtype=np.uint8)
        view = memoryview(buffer)
        bounds = [(a, min(a + self.chunk_bytes, end - start)) for a in range(0, end - start, self.chunk_bytes)]
        list(self.pool.map(lambda b: self._read_chunk(view, self.base + start, *b), bounds))
        return buffer.view(SAFETENSORS_DTYPES[meta["dtype"]]).reshape(meta["shape"])

    def close(self):
        self.pool.shutdown()
        os.close(self.fd)


def _read_safetensors(path, workers=8, chunk_bytes=256 << 20):
    """Yield (name, array) for every tensor of a safetensors file."""
    file = _SafetensorsFile(path, workers, chunk_bytes)
    try:
        for name in file.header:
            yield name, file.read(name)
    finally:
        file.close()


def load_exchange(input_dir, dp_shard=False):
    """Read a version written by `save_exchange` (every part of it) onto the
    current mesh with the sharding rule of `_load_weights`: each tensor is
    joined from its shards across the parts, one tensor at a time."""
    paths = exchange_files(input_dir)
    if not paths:
        raise FileNotFoundError(f"no weights-*.safetensors in {input_dir}")
    files = [_SafetensorsFile(path) for path in paths]
    try:
        weights, shards = {}, {}
        for file in files:
            for name, entry in json.loads(file.metadata["layout"]).items():
                target = shards.setdefault(name, {"shape": tuple(entry["shape"]), "blocks": []})
                target["blocks"] += [(file, key, slices) for key, slices in entry["shards"].items()]
        for name, entry in shards.items():
            full, covered, seen = None, 0, set()
            for file, key, slices in entry["blocks"]:
                window = tuple(slice(a, b) for a, b in slices)
                if window in seen:  # a tensor replicated over processes is written by each of them
                    continue
                seen.add(window)
                block = file.read(key)
                if full is None:
                    full = np.empty(entry["shape"], block.dtype)
                full[window] = block
                covered += block.size
            if covered != math.prod(entry["shape"]):
                raise ValueError(f"the parts in {input_dir} cover {covered} of {math.prod(entry['shape'])} "
                                 f"elements of {name}; a trainer process's part is missing")
            weights[name] = place(full, _get_sharding(name, dp_shard))
        return weights
    finally:
        for file in files:
            file.close()
