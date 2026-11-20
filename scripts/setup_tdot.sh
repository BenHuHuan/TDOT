#!/usr/bin/env bash
set -euo pipefail

# One-shot installer for TDOT (Web/MuJoCo 4.0 env + NablaMimic/NablaEngine GPU env)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[TDOT Setup] 1/2 Creating main Python venv (.venv) with Flask + MuJoCo + PyTorch..."
if command -v uv >/dev/null 2>&1; then
  uv venv .venv --python 3.12
  uv pip install --python .venv/bin/python -e ".[dev]"
else
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -e ".[dev]"
fi

echo "[TDOT Setup] 2/2 Checking NablaMimic + NablaEngine GPU environment..."
if [[ -x "${HOME}/NablaMimic/.venv/bin/python" ]]; then
  echo "  -> Found existing NablaMimic venv at ${HOME}/NablaMimic/.venv"
elif [[ -x "${ROOT_DIR}/third_party/NablaMimic/.venv/bin/python" ]]; then
  echo "  -> Found existing NablaMimic venv at ${ROOT_DIR}/third_party/NablaMimic/.venv"
else
  mkdir -p "${ROOT_DIR}/third_party"
  git clone --recursive https://github.com/sailab-ai/NablaMimic.git "${ROOT_DIR}/third_party/NablaMimic"
  (
    cd "${ROOT_DIR}/third_party/NablaMimic"
    if command -v uv >/dev/null 2>&1; then
      uv sync
    else
      python3 -m venv .venv
      .venv/bin/pip install -e .
    fi
  )
fi

echo "[TDOT Setup] Done! Run: bash scripts/launch_tdot.sh"
