# Results: full single-GPU run (run `20261115-232515-94a608`)

One complete run of the pinned record #14 on an RTX 5070 Ti Laptop GPU (12 GB), started 2026-11-15:

```bash
.venv/bin/python train_feasibility.py --micro-batch-tokens 4096 --no-ddp --accept-deviations --checkpoint-every-fraction 0.1
```

No further training runs are planned; the checkpoints of this run are the deliverable.

## Outcome

- Completed all 1530 steps (802,160,640 tokens): 6.94 h of training, 7.6 h wall clock including 14
  validations (≈ 88 s each), compilation, checkpoints and 426 cooling pauses (14.2 min). 96x slower than 8xH100.
- **Final val loss 3.3193**, versus **3.2777 ± 0.004** for the 40 original 8xH100 runs of the same code
  (`records/120424_ValueEmbed/*.txt` at commit 9730304; range 3.271–3.290). Same curve shape, offset
  +0.12 at step 125 and +0.04–0.05 from step 250 on. No NaN or divergence.
- Thermals: GPU peak 84 °C, clocks limited by the 92 W power cap only (thermal slowdown 0%); step time
  drifted +5.2% over the run.

| step | val loss | original mean | gap |
|---:|---:|---:|---:|
| 125 | 4.5249 | 4.4042 | +0.121 |
| 500 | 3.7532 | 3.7035 | +0.050 |
| 1000 | 3.5099 | 3.4649 | +0.045 |
| 1530 | 3.3193 | 3.2777 | +0.042 |

## Interpreting the gap

Estimated split: **≈ 0.018 measurement, ≈ 0.024 training** (extrapolated, not measured directly).

- Measurement: validation in 4,096-token chunks cuts attention context at every chunk start.
  `scripts/eval_checkpoint.py` re-evaluates the final checkpoint at shorter chunks (4,096 reproduces
  3.3193 exactly):

  | chunk tokens | 512 | 1,024 | 2,048 | 4,096 |
  |---|---:|---:|---:|---:|
  | val loss | 3.4399 | 3.3746 | 3.3373 | 3.3193 |

  From 1,024 to 4,096 the excess halves with each doubling (fit `a + b/T`, residuals ≤ 3e-4), which
  extrapolates to ≈ 3.301 without chunk cuts (≈ 3.302 against the original's own 65,536-token chunks).
- Training: the remaining ≈ 0.024 comes from training with 4,096-token micro-batches (context cuts) and
  summing bf16 gradients (embeddings, scalars) over 128 micro-batches instead of 8. The gap is largest at
  step 125, when the attention window is only 192 tokens and context cuts barely matter, which points at
  the bf16 accumulation early in training.

Chunks longer than 4,096 cannot be used to measure this: the pinned `Rotary` computes position angles in
bf16 and this model never trained on positions >= 4,096 (16k/32k/65k give 3.78/4.48/5.02).
Results: `reports/20261115-232515-94a608/eval_chunks_001530{,_short}.json`.

## Artifacts (not in git)

- `checkpoints/20261115-232515-94a608/`: 11 checkpoints (steps 0, 153, …, 1530) with optimizer state,
  plus the consumed-token records, 42 GiB.
- `checkpoints/20261115-232515-94a608-hf/`: sharing copy with a README, safetensors weights, logs and `SHA256SUMS`.
- `reports/20261115-232515-94a608/`: run report and per-step log.
