#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH

# Resolve host Python (Flask + MuJoCo 4.0 + PyTorch)
if [[ -n "${TDOT_PYTHON:-}" && -x "${TDOT_PYTHON}" ]]; then
  PY="${TDOT_PYTHON}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PY="${ROOT_DIR}/.venv/bin/python"
elif [[ -x "${HOME}/mjlab/.venv-mj4/bin/python" ]]; then
  PY="${HOME}/mjlab/.venv-mj4/bin/python"
else
  PY="$(command -v python3)"
fi

# Resolve NablaEngine Python
if [[ -z "${NABLA_PYTHON:-}" ]]; then
  if [[ -x "${HOME}/NablaMimic/.venv/bin/python" ]]; then
    export NABLA_PYTHON="${HOME}/NablaMimic/.venv/bin/python"
  elif [[ -x "${ROOT_DIR}/third_party/NablaMimic/.venv/bin/python" ]]; then
    export NABLA_PYTHON="${ROOT_DIR}/third_party/NablaMimic/.venv/bin/python"
  fi
fi

# Ensure nvcc is visible for JAX/XLA GPU compilation if installed
if [[ -x "/usr/local/cuda/bin/nvcc" && ":${PATH}:" != *":/usr/local/cuda/bin:"* ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
fi

export PYTHONPATH="${ROOT_DIR}"
exec "${PY}" -m tdot.cli serve "$@"
