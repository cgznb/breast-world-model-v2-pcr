# Registered ROI32 on RTX 5090

This standalone extraction of `symm-fm-world-v2.zip` trains fresh V2 weights from
the existing `MeWM-ISPY2/runs/registered_three_phase_roi32_v1` cache. Original
datasets, codec weights and earlier generator runs are read-only inputs.

## Data

- MRI: precontrast / first-postcontrast / metadata-selected late, CZYX
  `3 x 32 x 128 x 128`, on each patient's original fixed T0 crop.
- Raw continuous VQ latents: `24 x 8 x 32 x 32`, float16 storage. No extra resize.
- Patient split: 764 training / 102 validation; 2657 / 382 visits;
  3462 / 538 forward longitudinal pairs. There is no independent test split.
- All 3039 arrays, original cache identities, phase order, ROI geometry and
  4000 date intervals are checked. The 24 training-only latent moments replay
  the original statistics. Clinical vocabularies use training patients only.
- Observed pCR labels are auxiliary supervision, never inference conditions.
  The locked cohort CSV matches all 866 patients. No measured enhancement
  sidecars, follow-up tumor masks, biomarker targets or external teacher are
  imported. Those unavailable auxiliary losses are explicitly disabled.
- T0 registration provenance and same-grid checks are retained; they do not
  establish perfect anatomical registration. Rollout stage C is disabled
  because source-known segment-level treatment plans are unavailable.

## Runtime

`configs/registered_roi32_5090.yaml` retains the production model widths and the
real MONAI velocity U-Net. BF16 autocast uses the existing PyTorch/CUDA/MONAI
installation. No GPU wheel or existing environment is replaced.

The phase mixer chunks independent spatial-position sequences below CUDA's
SDPA batch limit. This does not partition its three-phase attention sequence.
Activation checkpointing is disabled to improve speed on the reduced tensors.
Raw latents are cached in CPU RAM. GPU memory is used by real training tensors.

Reference batch is 8. The original 20000 representation and 60000 flow reference
updates mean exactly 160000 and 480000 sampled pairs. Physical batches 256 / 80
produce 625 / 6000 optimizer updates, with partial final batches supported.
LR warmup/decay and EMA follow processed samples. Auxiliary warmup is in reference
updates; the original one-in-four auxiliary batch fraction is preserved.
Validation/checkpoint intervals round up to a complete physical optimizer update.
Larger batches change optimization and within-batch moment losses; this is not
an identical-gradient reproduction of the original microbatch configuration.

Each selection evaluation covers a fixed 102-pair subset, one pair per validation
patient, spread over available transitions. Flow uses four samples and Heun20.
These are development selection results, not a full 538-pair final evaluation.
Only `best.pt` and `last.pt` are retained per stage, including optimizer, EMA,
scheduler, RNG and configuration. The seed is fixed, but strict CUDA determinism
is disabled because adaptive 3D pooling backward has no deterministic CUDA
implementation in the installed PyTorch build.

New artifacts use file paths, sizes and modification times, and full structured
configuration comparison. No checksum values are recorded. These identities
detect normal accidental changes; they are not cryptographic content guarantees.

## Commands

Run from `/path/to/research/MAM/symm-fm-world-v2`:

```bash
python -B scripts/run_registered_roi32.py \
  --config configs/registered_roi32_5090.yaml \
  --manifest data/registered_roi32/manifest.json \
  --output runs/registered_roi32_20260919 --detach
```

After an interruption, use the same command with `--resume`. The controller lock
rejects duplicate launches into the same output. Runtime sources, input manifest,
resolved configuration and latent identities must remain fixed during training.

```bash
tail -f runs/registered_roi32_20260919/controller.log
```

`launch.json` records the controller PID. Root `progress.json` records controller
state; each stage's `progress.json` and `metrics.jsonl` record actual optimizer
steps, reference progress, sampled pairs, finite losses and CUDA memory peaks.
SIGTERM to this controller requests a checkpoint after the current update.
The controller automatically starts flow after representation completes.

Measured batch trials are under `runs/registered_roi32_20260919/preflight/`.
They use disposable models, including both representation branches and active
flow auxiliary losses. The GPU smoke run is separate from formal training.

## Launch Verification

On 2026-09-19, representation batch 256 passed at 29094 MiB peak allocated;
batch 288 exceeded the profiling memory limit. Flow batch 80 passed its active
auxiliary graph at 28205 MiB and 89.56 pairs/s. Batch 88 reached 30592 MiB but
slowed to 71.10 pairs/s; batch 96 exceeded the limit. Batch 80 leaves practical
headroom and is faster. The formal process used about 29875 / 32607 MiB in
`nvidia-smi`, with 100% GPU utilization at the observed training instant.

All 67 tests passed. A separate real-data GPU run completed three optimizer
updates per stage, validation and checkpoint readback at the selected full
batches. Finite weights, optimizer state, pCR supervision and flow EMA were
verified. Formal training is a fresh initialization, not a continuation of the
disposable profiling or smoke weights. The first formal 102-patient validation
and best checkpoint succeeded. No generation-quality conclusion follows from
these launch checks.

`data/registered_roi32/codec.pt` is a weights-only export of the original frozen
ROI32 codec. Strict state loading, weight roundtrip and decoded three-phase
`[1,3,32,128,128]` output matched the original implementation exactly on a real
cached visit (maximum absolute difference zero). The original codec was not
changed. Pass this exported checkpoint to `world.py sample --codec` when image
output is needed. Verification is in `codec.verification.json` beside the export.
