#!/usr/bin/env bash
# Starts train_feasibility.py detached from the terminal, so it survives closing the terminal or logging out.
#   bash scripts/run_detached.sh <train_feasibility.py arguments>
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f run.pid ] && kill -0 "$(cat run.pid)" 2>/dev/null; then
    echo "error: a run is already active (pid $(cat run.pid)); stop it first with: bash scripts/stop_run.sh" >&2
    exit 1
fi

mkdir -p logs
log="logs/run_$(date +%Y%m%d-%H%M%S).log"
# Unbuffered so `tail -f` shows every step as soon as it completes.
PYTHONUNBUFFERED=1 nohup .venv/bin/python train_feasibility.py "$@" > "$log" 2>&1 &
pid=$!
echo "$pid" > run.pid
ln -sfn "$(basename "$log")" logs/latest.log

echo "started pid $pid"
echo "follow: tail -f $log"
echo "stop:   bash scripts/stop_run.sh   (one call = one Ctrl+C; call it again to abort)"
