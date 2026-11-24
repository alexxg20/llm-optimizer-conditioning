#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" || exit 1
for step in 1530 765; do
  s6=$(printf "%06d" $step)
  echo "===== step $step random start $(date) ====="
  python run_power.py --ckpt $W8/model_step${s6}.safetensors --step $step --shard $VAL10 --set 0 \
    --S 4096 --W 128 --iters 300 --tol 1e-3 --min_iters 15 --params lm_head --seed 1 --exp warm_check
done
echo "===== done $(date) ====="
