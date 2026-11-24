# Load with:  source <clone>/analysis/env.sh   (bash, from any directory)
# Activates the venv, exports the data paths and cd's into analysis/. Per-machine overrides of
# VENV, W8, D10 or VAL10 go in env.local.sh next to this file (gitignored). It is the only override:
# values inherited from the shell are ignored, so re-sourcing, or sourcing another clone, is consistent.
[ -n "${BASH_VERSION:-}" ] || { echo "env.sh: needs bash" >&2; return 1; }
ANALYSIS=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return 1

VENV=$ANALYSIS/.venv
W8=$ANALYSIS/weights                     # model_stepNNNNNN.safetensors
D10=$ANALYSIS/fineweb10B                 # FineWeb-10B shards
VAL10=
if [ -f "$ANALYSIS/env.local.sh" ]; then source "$ANALYSIS/env.local.sh"; fi
VAL10=${VAL10:-$D10/fineweb_val_000000.bin}
export W8 D10 VAL10 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
elif [ "$VENV" != "$ANALYSIS/.venv" ]; then
    echo "env.sh: VENV=$VENV (from env.local.sh) has no bin/activate" >&2
    return 1
else
    echo "env.sh: no venv at $VENV, using $(command -v python || echo 'no python on PATH')" >&2
fi
cd "$ANALYSIS"
