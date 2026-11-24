#!/bin/bash
# lambda_max of the lm_head Gauss-Newton block at all 11 checkpoints, one after another.
# Launch with:  setsid nohup bash scripts/run_all.sh > logs/run_all.txt 2>&1 < /dev/null &

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1        # venv, paths ($W8, $VAL10), cd into analysis/
S=4096; W=128

# wait until no other power-method run is using the GPU
while pgrep -f "python run_power.py" > /dev/null; do sleep 60; done

prev=""
for step in 0 153 306 459 612 765 918 1071 1224 1377 1530; do
  s6=$(printf "%06d" $step)
  echo "===== step $step  $(date) ====="
  python run_power.py --ckpt $W8/model_step${s6}.safetensors --step $step \
    --shard $VAL10 --S $S --W $W --iters 300 --tol 1e-3 --min_iters 15 \
    --params lm_head --exp checkpoints $prev || break
  last=$(ls -t results/checkpoints/step${s6}_*_eigvec.npy | head -n 1)
  # step 0 (lm_head all zeros, uniform predictions) is a poor starting direction,
  # so step 153 also starts from a random vector; warm starts from step 306 on
  if [ "$step" -ge 153 ]; then prev="--init $last"; fi
done

python compare.py --exp checkpoints --x step
echo "===== done $(date) ====="
