# llm-optimizer-conditioning

Studying optimizer selection in LLM pretraining through local quadratic approximations and
norm-dependent condition numbers.

At checkpoints along one training run, the loss is approximated by a convex local quadratic model
(following A. Meterez et al., ["A Defense of the Quadratic Model"](https://arxiv.org/abs/2607.21716),
2026). Its curvature (the largest eigenvalue and the condition number of the Gauss–Newton matrix, in
the geometry of each optimizer) is compared across gradient descent, Adam and Muon.

## Contents

- [`pretraining/`](pretraining/README.md): the training run the analysis uses. One complete run of
  [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) record #14 on a single GPU, with a
  checkpoint every 10 % of training and the exact tokens consumed between checkpoints. Results in
  [`pretraining/RESULTS.md`](pretraining/RESULTS.md).
- [`analysis/`](analysis/README.md): curvature of the local quadratic model at those checkpoints.
  So far the largest Gauss–Newton eigenvalue (the smoothness constant of gradient descent) of the
  `lm_head` block, by power iteration with matrix-free Gauss–Newton–vector products, at all 11
  checkpoints, with checks on the number of windows and on warm starts.

## Checkpoints and data

Published on Hugging Face at
[`epfl-optimizer-conditioning/modded-nanogpt-1gpu`](https://huggingface.co/epfl-optimizer-conditioning/modded-nanogpt-1gpu)
(private, organization members):

- `weights/`: model weights at steps 0, 153, …, 1530 (safetensors, 1.40 GiB each).
- `full/`: the same checkpoints with the state of all optimizers, e.g. Adam's second moments (3.87 GiB each).
- `data/`: every token window consumed between consecutive checkpoints, in training order.
