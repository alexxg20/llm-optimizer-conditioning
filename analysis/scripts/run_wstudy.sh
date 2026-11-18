#!/bin/bash
source ~/ML_project/analysis/env.sh
prev=""
for W in 8 16 32 64 128; do
  python run_power.py --ckpt $W8/model_step000918.safetensors --step 918 \
    --shard $VAL10 --S 4096 --W $W --iters 300 --params lm_head --exp W_study_918 $prev || break
  prev="--init $(ls -t results/W_study_918/*_eigvec.npy | head -n 1)"
done
python compare.py --exp W_study_918 --x W
