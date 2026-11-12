#!/usr/bin/env bash
# Sends one SIGINT (the same as one Ctrl+C) to the run started by run_detached.sh and waits for it to exit.
# A second invocation sends a second SIGINT, which aborts the run.
#   WAIT_SECONDS=300 bash scripts/stop_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f run.pid ]; then
    echo "no run.pid: nothing to stop" >&2
    exit 1
fi
pid=$(cat run.pid)
if ! kill -0 "$pid" 2>/dev/null; then
    echo "process $pid is not running"
    rm -f run.pid
    exit 0
fi

kill -INT "$pid"
wait_s=${WAIT_SECONDS:-180}
echo "sent SIGINT to $pid; waiting up to ${wait_s} s (it finishes the current micro-batch, runs a short calibration and writes the reports)"
for _ in $(seq "$wait_s"); do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "process $pid exited; end of its log:"
        rm -f run.pid
        [ -e logs/latest.log ] && tail -n 25 logs/latest.log
        exit 0
    fi
    sleep 1
done
echo "still running after ${wait_s} s (a compilation in progress delays the stop)."
echo "Check with: tail -f logs/latest.log. Run this script again to send a second SIGINT, which aborts."
