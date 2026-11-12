#!/usr/bin/env bash
# Creates .venv, installs the torch build that supports this GPU, then requirements.txt. Safe to re-run.
#   PYTHON=python3.12 bash setup.sh       interpreter used to create .venv (default: python3)
#   TORCH_INDEX=<url> bash setup.sh       force a PyTorch wheel index instead of choosing by GPU
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
TORCH_VERSION=2.14.1

if [ ! -x .venv/bin/python ]; then
    "$PYTHON" -m venv .venv
fi

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)
if [ -z "${TORCH_INDEX:-}" ]; then
    if [ -z "$cap" ]; then
        echo "error: nvidia-smi reports no GPU; set TORCH_INDEX to choose a PyTorch wheel index explicitly" >&2
        exit 1
    fi
    # cu130 wheels start at sm_75; Volta (sm_70, e.g. V100) needs the cu126 build.
    if awk -v c="$cap" 'BEGIN { exit !(c >= 7.5) }'; then
        TORCH_INDEX=https://download.pytorch.org/whl/cu130
    elif awk -v c="$cap" 'BEGIN { exit !(c >= 7.0) }'; then
        TORCH_INDEX=https://download.pytorch.org/whl/cu126
    else
        echo "error: compute capability $cap is older than any torch $TORCH_VERSION CUDA build supports" >&2
        exit 1
    fi
fi
echo "GPU compute capability: ${cap:-unknown}; torch index: $TORCH_INDEX"

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python - "$cap" <<'PY'
import sys
import torch

cap = sys.argv[1]
arch = torch.cuda.get_arch_list()
print(f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}, compiled for: {' '.join(arch) or 'n/a'}")
if torch.cuda.is_available():
    print(f"bf16 supported natively: {torch.cuda.is_bf16_supported(including_emulation=False)}, "
          f"with emulation: {torch.cuda.is_bf16_supported(including_emulation=True)}")
else:
    print("bf16 support: not checked (CUDA is not visible to PyTorch in this shell)")
if not cap:
    print("no GPU detected: architecture check skipped")
    sys.exit(0)
want = "sm_" + cap.replace(".", "")
if want not in arch:
    print(f"ERROR: this torch build has no kernels for {want} (compute capability {cap})")
    sys.exit(1)
print(f"OK: {want} is supported. Next: .venv/bin/python scripts/check_gpu.py")
PY
