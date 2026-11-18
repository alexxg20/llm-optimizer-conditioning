#!/bin/bash
source ~/ML_project/analysis/env.sh
S=4096; W=128
while pgrep -f "python run_power.py" > /dev/null; do sleep 60; done
prev="--init $(ls -t results/checkpoints/step000153_*_eigvec.npy | head -n 1)"
for step in 306 459 612 765 918 1071 1224 1377 1530; do
  s6=$(printf "%06d" $step)
  echo "===== step $step  $(date) ====="
  python run_power.py --ckpt $W8/model_step${s6}.safetensors --step $step \
    --shard $VAL10 --S $S --W $W --iters 300 --tol 1e-3 --min_iters 15 \
    --params lm_head --exp checkpoints $prev || break
  prev="--init $(ls -t results/checkpoints/step${s6}_*_eigvec.npy | head -n 1)"
done
python compare.py --exp checkpoints --x step
echo "===== done $(date) ====="
