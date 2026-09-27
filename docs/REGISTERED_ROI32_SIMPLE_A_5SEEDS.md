# Reduced Three-Phase Simple Stage A: Five Seeds

Study output: `runs/registered_roi32_simple_a_5seeds_20260920`.
Seeds: 42, 43, 44, 45, 46, run sequentially on the single RTX 5090.

Every seed trains a fresh representation and a fresh Flow model. Its Flow
copies its own best online representation; no earlier generator weights are
reused. All seeds use the same existing reduced MRI/latent cache, patient
split, production architecture, training budgets and validation selection.

Stage A has five active losses: reconstruction 1, JEPA 1, future 0.25,
variance 0.1 and covariance 0.01. pCR, latent delta, anatomy stability and
dedicated separation have zero weight. Other previously disabled tasks and
Stage C remain disabled. The unchanged runtime still computes diagnostic
values for zero-weight terms; these values do not contribute to optimization.

BF16, batch 256 for representation and 80 for Flow, accumulation 1. Each seed
has 625/6000 updates and 160000/480000 pair presentations, respectively. LR
and EMA retain the existing sample-based conversion. Selection uses 102 fixed
patient/transition-spread pairs, with MC4 Heun20 for Flow. The training seed
also determines validation noises under the existing implementation.
Keep all five results; do not select the best seed as the study result.

The controller writes resolved per-seed JSON configurations before training,
locks its queue, checks source/config/data file identities and launches the
existing single-run controller. It waits for both stages and validates the
best/last checkpoint metadata before starting the next seed. Failures stop
the queue without advancing. SIGTERM/SIGINT are forwarded to the active
trainer for its normal checkpoint handling. Resume skips completed seeds.

Run from `/path/to/research/MAM/symm-fm-world-v2`:

```bash
python -B scripts/run_registered_roi32_multiseed.py \
  --config configs/ablations/registered_roi32_5090_simple_a.yaml \
  --manifest data/registered_roi32/manifest.json \
  --output runs/registered_roi32_simple_a_5seeds_20260920 \
  --seeds 42 43 44 45 46 --detach
```

After an interruption, run the same command with `--resume`. The original
20260919 experiment remains separate. Do not modify bound runtime sources or
seed configs while the queue is active. Each seed retains only best/last
checkpoints per stage; allow approximately 23 GiB for five completed seeds,
plus smoke artifacts and temporary checkpoint writes. Each seed launch
requires at least 15 GiB free disk space.

Monitor the study `progress.json` / `controller.log`, or
`seed_42/representation/metrics.jsonl` and `seed_42/flow/metrics.jsonl` for
individual optimizer updates. A study `COMPLETE.json` is written only after
all five seeds pass completion checks. Profiling and disposable GPU smoke
artifacts are under `preflight/` and are never reused as formal weights.
