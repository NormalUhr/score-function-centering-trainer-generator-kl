"""GLM-4.5 (glm4_moe) support on CPU with a tiny random model: partial rotary,
the sigmoid router with its correction bias plus the shared expert, one dense
layer ahead of the stacked MoE layers, several stop ids, and the GLM chat
template under the prompt contract. The tiny qwen3_moe logits recorded in
tests/data must not move."""
import json
import os
import sys
import tempfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, PartitionSpec as P, reshard
from safetensors import safe_open
from safetensors.numpy import save_file

import prompting
import train_rl
from models import model as model_lib
from models import sampling
from tasks import rollout

WEIGHTS_DIR = os.path.expanduser("~/.cache/postax/weights")
GLM_ID = "zai-org/GLM-4.5-Air"
TOKENIZER_FILES = ["*.json", "*.txt", "*.jinja"]
GOLDEN_QWEN3_MOE = ROOT / "tests" / "data" / "qwen3_moe_tiny_logits.npy"

TINY_GLM = dict(
    model_type="glm4_moe", hidden_size=256, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, num_hidden_layers=3, first_k_dense_replace=1, intermediate_size=384,
    n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=128, n_shared_experts=1,
    norm_topk_prob=True, routed_scaling_factor=1.0, n_group=1, topk_group=1,
    partial_rotary_factor=0.5, rope_theta=1e6, rope_scaling=None, attention_bias=True,
    use_qk_norm=False, rms_norm_eps=1e-5, vocab_size=512, tie_word_embeddings=False,
    initializer_range=0.02, eos_token_id=[3, 5], pad_token_id=3)

TINY_QWEN3_MOE = dict(
    model_type="qwen3_moe", hidden_size=256, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, num_hidden_layers=2, num_experts=8, num_experts_per_tok=2,
    moe_intermediate_size=128, norm_topk_prob=True, rope_theta=1e6, rope_scaling=None,
    rms_norm_eps=1e-6, vocab_size=512, tie_word_embeddings=False, initializer_range=0.02,
    eos_token_id=151643)

# Token ids stay clear of the tiny model's stop ids 3 and 5.
TOKENS = np.random.default_rng(0).integers(8, 512, (2, 12)).astype(np.int32)


def set_mesh():
    mesh = jax.make_mesh((1, 1), ("data", "model"),
                         axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)


@pytest.fixture(autouse=True)
def restore_mesh():
    """The model helpers set a global mesh; put the previous one back, so the
    tests of other files do not run under it."""
    previous = jax.sharding.get_mesh()
    yield
    jax.set_mesh(previous)


def glm_tokenizer():
    _, tokenizer, cfg = model_lib._load_template(GLM_ID, WEIGHTS_DIR, TOKENIZER_FILES)
    return tokenizer, cfg


def tiny_glm(key=0):
    set_mesh()
    tokenizer, _ = glm_tokenizer()
    weights = model_lib._random_weights(TINY_GLM, jax.random.key(key))
    return model_lib._make_model(GLM_ID, TINY_GLM, weights, tokenizer)


def tiny_qwen3_moe_logits():
    set_mesh()
    weights = model_lib._random_weights(TINY_QWEN3_MOE, jax.random.key(1))
    return np.asarray(model_lib.forward(
        TINY_QWEN3_MOE, jnp.asarray(TOKENS), weights, dtype=jnp.float32))


def decode(model, tokens, dtype=jnp.float32):
    """Logits of every position through the per-layer kv cache, one token at a
    time. The step is jitted as in the sampler; run eagerly, the cache update
    loses its sharding on a mesh of several devices."""
    batch, length = tokens.shape
    kv = model.init_kv(batch, length, dtype=dtype)
    step = jax.jit(partial(model.forward, dtype=dtype))
    logits = []
    for position in range(length):
        pos = reshard(jnp.full((batch, 1), position), P("data", None))
        out, kv = step(jnp.asarray(tokens[:, position:position + 1]), model.weights, kv, pos)
        logits.append(np.asarray(out[:, 0]))
    return np.stack(logits, axis=1)


def test_weight_layout():
    """The dense prefix layer keeps its own HF keys; the MoE layers stack."""
    model = tiny_glm()
    shapes = {k: tuple(v.shape) for k, v in model.weights.items()}
    assert shapes["model.layers.0.mlp.gate_proj.weight"] == (384, 256)
    assert shapes["model.layers.0.self_attn.q_proj.bias"] == (4, 64)
    assert shapes["model.layers.0.self_attn.k_proj.bias"] == (2, 64)
    assert "model.layers.0.mlp.gate.weight" not in shapes
    assert shapes["model.layers.mlp.gate.weight"] == (2, 8, 256)
    assert shapes["model.layers.mlp.gate.e_score_correction_bias"] == (2, 8)
    assert model.weights["model.layers.mlp.gate.e_score_correction_bias"].dtype == jnp.float32
    assert shapes["model.layers.mlp.experts.gate_proj.weight"] == (2, 8, 256, 128)
    assert shapes["model.layers.mlp.experts.down_proj.weight"] == (2, 8, 128, 256)
    assert shapes["model.layers.mlp.shared_experts.gate_proj.weight"] == (2, 128, 256)
    assert shapes["model.layers.mlp.shared_experts.down_proj.weight"] == (2, 256, 128)
    assert shapes["model.layers.self_attn.q_proj.weight"] == (2, 4, 64, 256)
    assert "model.layers.mlp.gate_proj.weight" not in shapes
    assert len(model.init_kv(1, 4)) == 3
    assert model.stop_ids == (3, 5)


def test_partial_rotary_leaves_the_pass_through_dims_alone():
    x = jax.random.normal(jax.random.key(0), (1, 5, 1, 64))
    rotated = model_lib.apply_rope(x, TINY_GLM, pos=7)
    np.testing.assert_array_equal(np.asarray(rotated[..., 32:]), np.asarray(x[..., 32:]))
    assert not np.allclose(np.asarray(rotated[..., :32]), np.asarray(x[..., :32]))
    full = model_lib.apply_rope(x, {**TINY_GLM, "partial_rotary_factor": 1.0}, pos=7)
    assert not np.allclose(np.asarray(full[..., 32:]), np.asarray(x[..., 32:]))


def test_forward_and_decode_agree():
    model = tiny_glm()
    prefill = np.asarray(model.forward(jnp.asarray(TOKENS), model.weights, dtype=jnp.float32))
    assert prefill.shape == (2, 12, 512) and np.isfinite(prefill).all()
    decoded = decode(model, TOKENS)
    np.testing.assert_allclose(decoded[:, -1], prefill[:, -1], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(decoded, prefill, rtol=1e-4, atol=1e-4)


def test_save_load_round_trip():
    model = tiny_glm()
    with tempfile.TemporaryDirectory() as root:
        model_lib.save(model, root)
        with safe_open(Path(root) / "model.safetensors", framework="numpy") as f:
            keys = set(f.keys())
            bias = f.get_tensor("model.layers.1.mlp.gate.e_score_correction_bias")
        assert bias.dtype == np.float32 and bias.shape == (8,)
        assert "model.layers.0.mlp.gate_proj.weight" in keys
        assert "model.layers.2.mlp.experts.7.down_proj.weight" in keys
        assert "model.layers.2.mlp.shared_experts.up_proj.weight" in keys
        assert "model.layers.0.self_attn.q_proj.bias" in keys
        assert not any(k.startswith("model.layers.mlp") or k.startswith("model.layers.self_attn") for k in keys)
        # A trailing layer beyond num_hidden_layers (GLM's multi-token-prediction
        # layer) is skipped, whatever file it sits in.
        save_file({"model.layers.3.mlp.gate.weight": np.zeros((8, 256), np.float32),
                   "model.layers.3.mlp.experts.0.gate_proj.weight": np.zeros((128, 256), np.float32)},
                  str(Path(root) / "extra.safetensors"))
        loaded = model_lib.load(root, init="pretrained")
    assert set(loaded.weights) == set(model.weights)
    for name, value in model.weights.items():
        assert loaded.weights[name].dtype == value.dtype, name
        np.testing.assert_array_equal(np.asarray(loaded.weights[name]).astype(np.float32),
                                      np.asarray(value).astype(np.float32), err_msg=name)
    assert loaded.stop_ids == (3, 5)
    before = np.asarray(model.forward(jnp.asarray(TOKENS), model.weights, dtype=jnp.float32))
    after = np.asarray(loaded.forward(jnp.asarray(TOKENS), loaded.weights, dtype=jnp.float32))
    np.testing.assert_array_equal(before, after)


def test_qwen3_moe_logits_unchanged():
    np.testing.assert_allclose(tiny_qwen3_moe_logits(), np.load(GOLDEN_QWEN3_MOE), rtol=1e-6, atol=1e-6)


def test_real_config_is_accepted():
    tokenizer, cfg = glm_tokenizer()
    assert cfg["model_type"] == "glm4_moe"
    assert model_lib.stop_token_ids(cfg, tokenizer) == (151329, 151336, 151338)
    _, qwen_tokenizer, qwen_cfg = model_lib._load_template("Qwen/Qwen3-0.6B-Base", WEIGHTS_DIR, TOKENIZER_FILES)
    assert model_lib.stop_token_ids(qwen_cfg, qwen_tokenizer) == (qwen_tokenizer.eos_token_id,)
    with tempfile.TemporaryDirectory() as root:
        (Path(root) / "config.json").write_text(json.dumps({**cfg, "n_group": 2}))
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (Path(root) / name).write_bytes((Path(WEIGHTS_DIR) / GLM_ID / name).read_bytes())
        with pytest.raises(ValueError, match="n_group"):
            model_lib._load_template(root, WEIGHTS_DIR)


def test_generation_stops_at_any_stop_id():
    model = tiny_glm()
    pad = TINY_GLM["pad_token_id"]
    prompt_lengths = np.asarray([4, 6], np.int32)
    tokens = np.full((2, 24), pad, np.int32)
    for row, length in enumerate(prompt_lengths):
        tokens[row, :length] = TOKENS[row, :length]
    payload = dict(forward=partial(model.forward, dtype=jnp.float32),
                   init_kv=partial(model.init_kv, dtype=jnp.float32))
    free = np.asarray(sampling.generate(
        jax.random.key(1), model, tokens, prompt_lengths, eos_id=-1, **payload).tokens)
    ends = (prompt_lengths[0] + 2, prompt_lengths[1] + 3)
    stops = (int(free[0, ends[0]]), int(free[1, ends[1]]))
    assert stops[0] != stops[1] and pad not in stops
    stopped = sampling.generate(jax.random.key(1), model, tokens, prompt_lengths, eos_id=stops, **payload)
    stopped_tokens = np.asarray(stopped.tokens)
    for row, end in enumerate(ends):
        np.testing.assert_array_equal(stopped_tokens[row, :end + 1], free[row, :end + 1])
        assert (stopped_tokens[row, end + 1:] == pad).all()
    assert np.asarray(stopped.ended).tolist() == [True, True]
    assert np.asarray(stopped.generated).tolist() == [3, 4]


class FakeRubric:
    has_group_rewards = False

    async def score_rollout(self, state):
        state["reward"] = float(len(state["completion"][0].content))

    async def cleanup(self, state):
        pass


def test_completions_and_truncation_with_several_stop_ids():
    tokenizer, cfg = glm_tokenizer()
    stop_ids = model_lib.stop_token_ids(cfg, tokenizer)
    pad, end_of_text, user = tokenizer.pad_token_id, 151329, 151336
    assert pad == end_of_text
    seq_len = 8
    tokens = np.full((4, seq_len), pad, np.int32)
    tokens[:, :2] = [10, 11]
    tokens[0, 2:5] = [20, 21, user]
    tokens[1, 2:] = [20, 21, 22, 23, 24, 25]
    tokens[2, 2:5] = [20, 21, end_of_text]
    tokens[3, 2:] = [20, 21, 22, 23, 24, user]
    prompt_lengths = np.full(4, 2, np.int32)

    mask, lengths, texts = rollout.completions(tokens, prompt_lengths, tokenizer, stop_ids=stop_ids)
    assert lengths.tolist() == [3, 6, 3, 6]
    assert mask.sum(1).tolist() == [3, 6, 3, 6] and mask[0, 2:5].all() and not mask[0, :2].any()
    assert texts[0] == texts[2] == tokenizer.decode([20, 21], skip_special_tokens=True)
    assert texts[1] == tokenizer.decode([20, 21, 22, 23, 24, 25], skip_special_tokens=True)
    # With one stop id (the default) <|user|> is ordinary text and the pad
    # token after it is taken for the end-of-text stop.
    _, lengths, _ = rollout.completions(tokens, prompt_lengths, tokenizer)
    assert lengths.tolist() == [4, 6, 3, 6]

    environment = SimpleNamespace(rubric=FakeRubric(), parser=SimpleNamespace(extract_fn=lambda text: "x"))
    mask, lengths, rewards, frac_parsed, truncated = train_rl.finish_rollouts(
        tokens, prompt_lengths, [{"prompt": []}] * 4, environment, tokenizer, seq_len, 2,
        stop_ids=stop_ids)
    assert lengths.tolist() == [3, 6, 3, 6]
    assert truncated.tolist() == [False, True, False, False]
    assert frac_parsed == 1.0 and rewards.shape == (4,)


def test_glm_chat_template_under_the_prompt_contract():
    tokenizer, cfg = glm_tokenizer()
    stop_ids = model_lib.stop_token_ids(cfg, tokenizer)
    assert prompting.resolve_prompt_format("auto", GLM_ID) == "chat"
    prompting.validate_prompt_contract(tokenizer, "chat", {"enable_thinking": False}, stop_ids=stop_ids)
    prompting.validate_prompt_contract(tokenizer, "chat", {}, stop_ids=stop_ids)
    # The template never emits the tokenizer's end-of-text token in a
    # conversation: GLM closes a turn with <|user|>, so the contract needs the
    # model's full stop list.
    with pytest.raises(ValueError, match="chat template does not emit"):
        prompting.validate_prompt_contract(tokenizer, "chat", {"enable_thinking": False})
    messages = [{"role": "user", "content": "What is 2 + 3?"}]
    no_think = prompting.tokenize_prompt(messages, tokenizer, "chat", {"enable_thinking": False})
    assert tokenizer.decode(no_think) == "[gMASK]<sop><|user|>\nWhat is 2 + 3?/nothink<|assistant|>\n<think></think>"
    thinking = prompting.tokenize_prompt(messages, tokenizer, "chat", {})
    assert tokenizer.decode(thinking) == "[gMASK]<sop><|user|>\nWhat is 2 + 3?<|assistant|>"
    diagnostic = prompting.prompt_diagnostic(tokenizer, "chat", no_think, stop_ids=stop_ids)
    assert "stop_ids=(151329, 151336, 151338)" in diagnostic and "<|assistant|>" in diagnostic


def test_qwen_chat_template_still_passes_the_contract():
    _, tokenizer, _ = model_lib._load_template("Qwen/Qwen3-0.6B-Base", WEIGHTS_DIR, TOKENIZER_FILES)
    # Qwen Base ships the chat template of the post-trained model: the
    # template ends turns with <|im_end|>, which is not the Base tokenizer's
    # end-of-text token, so the contract rejects chat on Base weights.
    with pytest.raises(ValueError, match="chat template does not emit"):
        prompting.validate_prompt_contract(tokenizer, "chat", {"enable_thinking": False})
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    prompting.validate_prompt_contract(tokenizer, "chat", {"enable_thinking": False}, stop_ids=(im_end,))
    prompting.validate_prompt_contract(tokenizer, "chat", {}, stop_ids=(im_end,))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("ok:", name)
