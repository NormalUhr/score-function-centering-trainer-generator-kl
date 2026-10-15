#!/usr/bin/env bash
# Build the Python environment for the experiments on a Linux machine with
# NVIDIA GPUs, check that JAX sees them, and download the Qwen3-0.6B weights.
# Run from the repository root; safe to re-run. The larger models (Qwen3-4B,
# Qwen3-30B-A3B, Qwen3-30B-A3B-Base, GLM-4.5-Air) are downloaded by the first
# run that needs them into the same weights directory.
#
#   bash scripts/setup_env.sh            # environment in ./.venv, weights in ./weights
#   SC_ASYNC_ROOT=/path/to/root bash scripts/setup_env.sh   # weights and outputs under another root
set -euo pipefail

ROOT=${SC_ASYNC_ROOT:-$(pwd)}
mkdir -p "$ROOT"/{weights,outputs,logs,launched}

if ! command -v uv >/dev/null 2>&1; then
  pip install --quiet uv
fi
if ! [ -x .venv/bin/python ]; then
  uv venv --python 3.13 .venv
fi
PY=$(pwd)/.venv/bin/python
echo "== installing requirements into $PY"
uv pip install --python "$PY" --quiet -r requirements.txt "jax[cuda12]==0.11.1"
# The INTELLECT-2 math environment of the quantized-sampler study (Environments Hub).
uv pip install --python "$PY" --quiet "intellect-math==0.1.7" \
  --extra-index-url https://hub.primeintellect.ai/primeintellect/simple/

echo "== environment"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -8
"$PY" - <<'EOF'
import jax, sys
print("python", sys.executable)
print("jax", jax.__version__, "devices", jax.device_count(), jax.devices()[0].platform)
import wandb, omegaconf, verifiers, intellect_math  # noqa: F401  (the import chain of train_rl.py)
print("imports ok")
EOF

echo "== weights"
"$PY" - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-0.6B", local_dir="$ROOT/weights/Qwen/Qwen3-0.6B")
print("ready Qwen/Qwen3-0.6B")
EOF
echo "== setup done (root $ROOT)"
