"""Freeze an interim EMA model and render six prespecified CPU MRI forecasts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import html
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO.parent / "MeWM-ISPY2"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ORIGINAL))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from mewm_ispy2.registered_three_phase_data import ThreePhaseCrops, load_config
from symm_world.codec import load_codec
from symm_world.data import DatasetStore
from symm_world.training import load_inference
from symm_world.utils import file_identity, read_json, save_checkpoint, write_json

PHASES = ("Precontrast", "First-post", "Late")
METHODS = (("source", "Real source"), ("target", "Real follow-up"),
           ("sample", "Generated sample 1"), ("mean", "Mean of 4 images"),
           ("vq_target", "Target VQ (diagnostic)"))
SHAPE = (3, 32, 128, 128)


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def check_runtime(run):
    binding = read_json(run / "runtime_binding.json")
    identities = [binding["config"], binding["manifest"], *binding["sources"]]
    for identity in identities:
        if file_identity(identity["path"]) != identity:
            raise ValueError(f"Active runtime binding changed: {identity['path']}")
    return len(identities)


def snapshot(run, output):
    path = run / "flow/best.pt"
    # The trainer atomically replaces best.pt; one open descriptor pins one version.
    with path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        identity = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        value = torch.load(handle, map_location="cpu", weights_only=True)
    if value["stage"] != "flow" or not value.get("teacher"):
        raise ValueError("Expected a Flow checkpoint with evaluated EMA weights")
    payload = {key: value[key] for key in ("schema", "stage", "step", "best", "metadata")}
    payload.update(model=value["teacher"], teacher=None)
    save_checkpoint(output / "inference_ema.pt", payload)
    receipt = {"source_checkpoint": identity, "step": value["step"],
               "best_validation_latent_mae": value["best"], "weights": "EMA teacher",
               "snapshot": file_identity(output / "inference_ema.pt")}
    write_json(output / "checkpoint.json", receipt)
    return receipt


def select_cases(store, seed):
    rng, selected, used = random.Random(seed), [], set()
    for earlier, later in itertools.combinations(range(4), 2):
        stages = (f"T{earlier}", f"T{later}")
        pool = sorted((p for p in store.records("val")
                       if (store.views[p["source"]]["visit"], store.views[p["target"]]["visit"]) == stages
                       and p["patient_id"] not in used), key=lambda p: p["id"])
        pair = rng.choice(pool)
        used.add(pair["patient_id"])
        selected.append({"case": f"Case{len(selected) + 1:02d}", "pair_id": pair["id"],
                         "patient_id": pair["patient_id"], "source": pair["source"], "target": pair["target"],
                         "transition": " -> ".join(stages), "days": pair["conditions"]["delta_days"],
                         "noise_seeds": [seed + 1000 * len(selected) + k for k in range(4)]})
    return selected


class SourceOnlyStore(DatasetStore):
    allowed_source = None

    def read_raw(self, view_id):
        if self.allowed_source is not None and view_id != self.allowed_source:
            raise ValueError("Target latent access during forecasting")
        return super().read_raw(view_id)

    def read_aux(self, *args, **kwargs):
        raise ValueError("Outcome and auxiliary labels are forbidden in this review")


def infer(args):
    output = args.output
    check_runtime(args.run)
    if args.resume:
        protocol = read_json(output / "protocol.json")
        checkpoint = protocol["checkpoint"]
        if (file_identity(output / "inference_ema.pt") != checkpoint["snapshot"]
                or protocol["selection_seed"] != args.seed
                or protocol["cpu_threads"] != args.threads
                or protocol["codec"] != file_identity(REPO / "data/registered_roi32/codec.pt")):
            raise ValueError("Review resume protocol differs from the frozen run")
    else:
        output.mkdir(parents=True, exist_ok=False)
        checkpoint = snapshot(args.run, output)
    log(f"Frozen EMA step {checkpoint['step']}; CPU threads={args.threads}")
    gc.collect()
    model, meta = load_inference(output / "inference_ema.pt", "cpu")
    codec = load_codec(REPO / "data/registered_roi32/codec.pt", "cpu")
    store = SourceOnlyStore(REPO / "data/registered_roi32/manifest.json")
    store.set_statistics(meta["statistics"])
    cases = select_cases(store, args.seed)
    if args.resume:
        for selected, saved in zip(cases, protocol["cases"], strict=True):
            if any(selected[key] != saved[key] for key in selected):
                raise ValueError("Prespecified case selection changed")
        cases = protocol["cases"]
    else:
        protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": checkpoint,
                "selection_seed": args.seed, "selection": "One distinct validation patient per transition, seeded random; no score selection",
                "device": "cpu", "precision": "float32", "cpu_threads": args.threads,
                "sampler": "heun", "steps": 20, "samples": 4, "sample_batch": 1,
                "mean_policy": "Arithmetic mean of four individually decoded MRI images",
                "source_only_inference": True, "independent_test": False,
                "slice_policy": "Maximum-area slice of source fixed-T0 predicted mask",
                "window_policy": "Shared source/target foreground 1st to 99.5th percentile; display only",
                "codec": file_identity(REPO / "data/registered_roi32/codec.pt"),
                "cases": cases}
    write_json(output / "protocol.json", protocol)
    original_config, baseline = load_config(ORIGINAL / "configs/registered_three_phase_roi32_v1.yaml")
    inventory = read_json(Path(original_config["output_dir"]) / "inventory.json")
    visits = {v["visit_id"]: v for v in inventory["visits"]}
    pairs = {p["id"]: p for p in store.pairs}
    means = torch.tensor(meta["statistics"]["mean"]).reshape(1, 24, 1, 1, 1)
    stds = torch.tensor(meta["statistics"]["std"]).reshape(1, 24, 1, 1, 1)
    completed = 0
    for case in cases:
        if (output / f"{case['case']}.npz").exists():
            case_arrays(output, case)
            log(f"{case['case']} saved volumes verified; reusing")
            continue
        if args.max_new_cases is not None and completed >= args.max_new_cases:
            break
        start = time.monotonic()
        store.allowed_source = case["source"]
        observed, conditions = store.source_batch(pairs[case["pair_id"]], "cpu")
        generations, latents = [], []
        with torch.inference_mode():
            for k, seed in enumerate(case["noise_seeds"]):
                generator = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn(observed.shape, generator=generator)
                forecast = model.sample(observed, conditions, noise=noise, steps=20, method="heun")
                latent = forecast * stds + means
                decoded = codec.decode(latent)[0].numpy().copy()
                if decoded.shape != SHAPE or not np.isfinite(decoded).all():
                    raise ValueError("Invalid generated MRI")
                generations.append(decoded)
                latents.append(latent[0].numpy().copy())
                log(f"{case['case']} {case['transition']}: sample {k + 1}/4 decoded, {time.monotonic() - start:.1f}s")
        # Target images and latents are first opened after all four forecasts exist.
        store.allowed_source = None
        dataset = ThreePhaseCrops(baseline, inventory, records=[visits[case["source"]], visits[case["target"]]])
        source, target = dataset[0], dataset[1]
        with torch.inference_mode():
            vq_target = codec.decode(store.read_raw(case["target"])[None])[0].numpy().copy()
        samples = np.stack(generations)
        arrays = {"source": source["image"].numpy(), "target": target["image"].numpy(),
                  "samples": samples, "mean": samples.mean(axis=0), "vq_target": vq_target,
                  "raw_latents": np.stack(latents), "mask": source["mask"][0].numpy().astype(bool),
                  "source_valid": source["valid"].numpy(), "target_valid": target["valid"].numpy(),
                  "source_coverage": source["coverage"].numpy(), "target_coverage": target["coverage"].numpy()}
        if not np.array_equal(source["mask"].numpy(), target["mask"].numpy()):
            raise ValueError("Source and target do not share the fixed T0 mask")
        np.savez_compressed(output / f"{case['case']}.npz", **arrays)
        case.update(seconds=time.monotonic() - start, source_largest_clipped=bool(source["largest_clipped"]),
                    target_largest_clipped=bool(target["largest_clipped"]))
        write_json(output / "protocol.json", protocol)
        render_case(output, case, checkpoint["step"])
        log(f"{case['case']} comparison saved; total {case['seconds']:.1f}s")
        completed += 1
    check_runtime(args.run)


def case_arrays(output, case):
    with np.load(output / f"{case['case']}.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key in ("source", "target", "mean", "vq_target"):
        if arrays[key].shape != SHAPE or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid cached {key}")
    if arrays["samples"].shape != (4, *SHAPE) or not np.isfinite(arrays["samples"]).all():
        raise ValueError("Invalid sample archive")
    if not np.array_equal(arrays["samples"].mean(0), arrays["mean"]):
        raise ValueError("Saved image mean does not replay")
    arrays["sample"] = arrays["samples"][0]
    mask = arrays["mask"]
    if mask.shape != SHAPE[1:] or not mask.any():
        raise ValueError("Empty or invalid source mask")
    z = int(mask.sum(axis=(1, 2)).argmax())
    foreground = np.concatenate([arrays[name][arrays[name + "_valid"].astype(bool)] for name in ("source", "target")])
    low, high = np.percentile(foreground, (1, 99.5)).tolist()
    if high <= low:
        raise ValueError("Degenerate real-MRI display window")
    return arrays, z, low, high


def panel(ax, data, mask, low, high, cmap="gray"):
    picture = ax.imshow(data, cmap=cmap, vmin=low, vmax=high, origin="lower", interpolation="nearest")
    if mask.any() and not mask.all():
        ax.contour(mask, levels=[0.5], colors=["#27df9d"], linewidths=0.65)
    ax.set_axis_off()
    return picture


def render_case(output, case, step):
    a, z, low, high = case_arrays(output, case)
    title = f"{case['case']} | {case['transition']} | {case['days']:g} days | EMA step {step} | axial z={z}"
    fig, axes = plt.subplots(3, 5, figsize=(14, 8.6), constrained_layout=True)
    fig.suptitle(title, fontsize=13)
    for row, phase in enumerate(PHASES):
        for col, (key, label) in enumerate(METHODS):
            pic = panel(axes[row, col], a[key][row, z], a["mask"][z], low, high)
            if row == 0:
                axes[row, col].set_title(label, fontsize=10)
        axes[row, 0].text(-0.05, 0.5, phase, transform=axes[row, 0].transAxes,
                          rotation=90, ha="right", va="center", fontsize=11)
    fig.colorbar(pic, ax=list(axes.flat), shrink=0.65, label="Shared MRI intensity window")
    fig.supxlabel("Green: fixed T0 predicted mask, not follow-up ground truth. Target VQ uses the real target.", fontsize=9)
    fig.savefig(output / f"{case['case']}_phases.png", dpi=140)
    plt.close(fig)
    enhancement = {key: np.stack((a[key][1] - a[key][0], a[key][2] - a[key][0])) for key, _ in METHODS}
    limit = max(float(np.percentile(np.abs(np.stack((enhancement["source"], enhancement["target"]))), 99.5)), 0.1)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(title + " | enhancement", fontsize=12)
    for row, phase in enumerate(("First-post minus pre", "Late minus pre")):
        for col, (key, label) in enumerate(METHODS):
            pic = panel(axes[row, col], enhancement[key][row, z], a["mask"][z], -limit, limit, "coolwarm")
            if row == 0:
                axes[row, col].set_title(label, fontsize=10)
        axes[row, 0].text(-0.05, 0.5, phase, transform=axes[row, 0].transAxes,
                          rotation=90, ha="right", va="center", fontsize=10)
    fig.colorbar(pic, ax=list(axes.flat), shrink=0.65, label="Shared enhancement window")
    fig.supxlabel("Registered phase differences; fixed T0 contour. Target VQ is a target-informed diagnostic.", fontsize=9)
    fig.savefig(output / f"{case['case']}_enhancement.png", dpi=140)
    plt.close(fig)
    result = {"case": case["case"], "transition": case["transition"], "z": z,
              "window": [low, high], "enhancement_window": [-limit, limit], "phase_metrics": []}
    for p, phase in enumerate(PHASES):
        common = a["source_coverage"][p].astype(bool) & a["target_coverage"][p].astype(bool) & a["target_valid"][p].astype(bool)
        roi = a["mask"] & common
        if not common.any() or not roi.any():
            raise ValueError("Empty evaluation support")
        for key, _ in METHODS:
            if key != "target":
                error = np.abs(a[key][p] - a["target"][p])
                result["phase_metrics"].append({"phase": phase, "method": key,
                    "common_support_target_foreground_mae": float(error[common].mean()),
                    "fixed_T0_roi_mae": float(error[roi].mean()),
                    "common_foreground_voxels": int(common.sum()), "roi_voxels": int(roi.sum())})
    write_json(output / f"{case['case']}_metrics.json", result)
    return result


def overview(output, cases, step):
    fig, axes = plt.subplots(len(cases), 5, figsize=(13, 2.6 * len(cases)), constrained_layout=True)
    fig.suptitle(f"Symm-FM World V2 | six fixed validation cases | first-post | EMA step {step}", fontsize=13)
    for row, case in enumerate(cases):
        a, z, low, high = case_arrays(output, case)
        for col, (key, label) in enumerate(METHODS):
            panel(axes[row, col], a[key][1, z], a["mask"][z], low, high)
            if row == 0:
                axes[row, col].set_title(label, fontsize=10)
        axes[row, 0].text(-0.05, 0.5, f"{case['case']}\n{case['transition']}\n{case['days']:g} days\nz={z}",
                          transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.supxlabel("Shared real-MRI window within each row; windows differ across cases. Green: fixed T0 predicted mask.", fontsize=9)
    fig.savefig(output / "overview.png", dpi=150)
    plt.close(fig)


def training_curves(run, output, step):
    rows = []
    with (run / "flow/metrics.jsonl").open() as handle:
        for line in handle:
            if line.endswith("\n"):
                rows.append(json.loads(line))
    validations = []
    for path in sorted((run / "flow").glob("validation_*.json")):
        result = read_json(path)
        validations.append({"step": int(path.stem.rsplit("_", 1)[1]), "pairs": result["pairs"],
                            "patients": result["patients"],
                            **{key: result["metrics"][key]["patient_macro_mean"] for key in
                               ("latent_mean_mae", "sample_latent_mae", "copy_source_latent_mae")}})
    losses = np.array([row["loss"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot([r["step"] for r in rows], losses, color="#91b6af", alpha=0.5, linewidth=0.5, label="Training loss")
    window = min(50, len(rows))
    axes[0].plot([r["step"] for r in rows[window - 1:]], np.convolve(losses, np.ones(window) / window, mode="valid"),
                 color="#1f6b62", linewidth=1.6, label=f"{window}-update moving mean")
    axes[0].set_ylabel("Flow training objective")
    for key, label, color in (("latent_mean_mae", "Mean of 4 latents", "#1f6b62"),
                              ("sample_latent_mae", "Individual samples", "#b24761"),
                              ("copy_source_latent_mae", "Source copy", "#707070")):
        axes[1].plot([r["step"] for r in validations], [r[key] for r in validations], marker=".", color=color, label=label)
    axes[1].set_ylabel("Validation standardized latent MAE")
    for ax in axes:
        ax.axvline(step, color="#426b97", linestyle="--", linewidth=1, label="Visualized checkpoint")
        ax.set_xlabel("Flow optimizer update")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    fig.suptitle("Interim training | validation: 102 patients, one selected pair each | MC4, Heun20", fontsize=11)
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)
    record = {"latest_training_step": rows[-1]["step"], "training_rows": len(rows),
              "training_loss_and_gradient_finite": all(np.isfinite(r["loss"]) and np.isfinite(r["gradient_norm"]) for r in rows),
              "validation_history": validations}
    write_json(output / "training_snapshot.json", record)
    return record


def gallery(output, protocol):
    step = protocol["checkpoint"]["step"]
    blocks = []
    for case in protocol["cases"]:
        name = case["case"]
        blocks.append(f'<section id="{name}"><h2>{name} &middot; {html.escape(case["transition"])} &middot; {case["days"]:g} days</h2>'
                      f'<a href="{name}_phases.png"><img loading="lazy" src="{name}_phases.png" alt="{name}, three MRI phases"></a>'
                      f'<a href="{name}_enhancement.png"><img loading="lazy" src="{name}_enhancement.png" alt="{name}, contrast enhancement"></a></section>')
    page = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>World V2 - interim three-phase MRI</title>
<style>*{{box-sizing:border-box}}body{{margin:0;font:15px/1.5 system-ui,sans-serif;color:#202722;background:#fff;letter-spacing:0}}
main{{max-width:1380px;margin:auto;padding:24px 16px}}h1{{font-size:26px;margin:0 0 8px}}h2{{font-size:20px}}
a{{color:#246b62}}nav{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}img{{display:block;width:100%;height:auto}}
section{{border-top:1px solid #ddd;padding:22px 0}}p{{max-width:1000px}}.meta{{color:#55605b}}
</style><main><h1>World V2 / Three-phase MRI</h1>
<p class="meta">Interim EMA checkpoint: {step} / 6000 updates. Six fixed validation patients. CPU float32, Heun20, four samples per case.</p>
<p>Columns: real source, real follow-up, generated sample 1, mean of four decoded images, and target VQ reconstruction.
The VQ column uses the real follow-up and is a diagnostic, not a forecast. Green contours mark the fixed T0 predicted region, not follow-up tumor ground truth.</p>
<p>Cases and sample 1 were fixed without looking at prediction quality. These six illustrations do not estimate full-cohort performance.
MRI windows are shared within each case and derived from the two real scans. Averaging generated images can remove detail.</p>
<nav><a href="overview.png">Overview</a><a href="training_curves.png">Training curves</a><a href="report.md">Protocol and metrics</a>
{''.join(f'<a href="#{c["case"]}">{c["case"]}</a>' for c in protocol['cases'])}</nav>
<section><h2>First-post overview</h2><a href="overview.png"><img src="overview.png" alt="Six fixed cases, first-post phase"></a></section>
<section><h2>Training and validation</h2><a href="training_curves.png"><img src="training_curves.png" alt="Training objective and validation latent MAE"></a></section>
{''.join(blocks)}</main></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def finish(args):
    output = args.output
    protocol = read_json(output / "protocol.json")
    if file_identity(output / "inference_ema.pt") != protocol["checkpoint"]["snapshot"]:
        raise ValueError("Frozen checkpoint changed")
    cases, step = protocol["cases"], protocol["checkpoint"]["step"]
    metrics = [render_case(output, case, step) for case in cases]
    overview(output, cases, step)
    training = training_curves(args.run, output, step)
    gallery(output, protocol)
    lines = ["# Interim World V2 MRI review", "", f"Frozen EMA step {step}; the original training continues.", "",
             f"Six distinct validation patients, one per forward transition, selected using seed {protocol['selection_seed']} before forecasting. "
             "Four fixed independent CPU float32 draws per patient, sequential batch 1, Heun20. "
             "This does not numerically replay GPU BF16 validation. Target images and target latents are opened only after each case's four forecasts.", "",
             "The mean is computed after decoding all four samples. Sample 1 is not chosen by quality. "
             "Target VQ reconstruction uses the true target and is not a forecast. All three phases share the original training-DCE0 normalization.", "",
             "Slice: maximal-area source fixed-T0 predicted mask. Contour: that same mask, not follow-up tumor ground truth. "
             "Display: one source/target foreground 1st-to-99.5th percentile window per case, shared across methods and phases; "
             "this target-informed display operation does not enter inference. Axial XY spacing is 0.7032 mm in both directions. "
             "The full volumes have shape 3x32x128x128 and Z spacing 2 mm.", "",
             "Metrics below average the six illustrative cases equally, over unclipped full-volume voxels within both acquisition supports "
             "and target foreground. They are descriptive and do not replace full-cohort evaluation. The validation cohort is used "
             "for checkpoint selection and is not an independent test.", "",
             "| Phase | Method | Common foreground MAE | Fixed T0 region MAE |", "|---|---|---:|---:|"]
    aggregate = []
    for phase in PHASES:
        for method, label in METHODS:
            if method == "target":
                continue
            rows = [r for m in metrics for r in m["phase_metrics"] if r["phase"] == phase and r["method"] == method]
            values = {key: float(np.mean([r[key] for r in rows])) for key in
                      ("common_support_target_foreground_mae", "fixed_T0_roi_mae")}
            aggregate.append({"phase": phase, "method": method, "cases": len(rows), **values})
            lines.append(f"| {phase} | {label} | {values['common_support_target_foreground_mae']:.6f} | {values['fixed_T0_roi_mae']:.6f} |")
    lines += ["", "[Gallery](index.html) | [Overview](overview.png) | [Curves](training_curves.png)", "",
              "The NPZ files retain source/target images, four decoded samples, the image mean, raw generated latents, "
              "target VQ reconstruction and masks. checkpoint.json identifies the pinned source version by path, size and modification time. "
              "protocol.json records case selection and noise seeds. training_snapshot.json contains the plotted validation history.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "case_aggregate.json", aggregate)
    image_checks = []
    for path in sorted(output.glob("*.png")):
        with Image.open(path) as im:
            pixels = np.asarray(im.convert("RGB"))
            if float(pixels.std()) < 5:
                raise ValueError(f"Blank figure: {path}")
            image_checks.append({"file": path.name, "size": list(im.size), "pixel_std": float(pixels.std())})
    verification = {"status": "passed", "cases": len(cases), "unique_patients": len({c["patient_id"] for c in cases}),
                    "raw_shape": list(SHAPE), "sample_mean_exact": True, "finite_images": True,
                    "source_only_sampling_guard": True, "active_runtime_files_unchanged": check_runtime(args.run),
                    "latest_training_step_when_plotted": training["latest_training_step"], "png_checks": image_checks}
    write_json(output / "verification.json", verification)
    log(f"Review complete: {output / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=REPO / "runs/registered_roi32_20260919")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-cases", type=int)
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not args.render_only:
        infer(args)
    protocol = read_json(args.output / "protocol.json")
    if all((args.output / f"{case['case']}.npz").is_file() for case in protocol["cases"]):
        finish(args)
    else:
        log("Partial review saved; continue with --resume using the same output and settings")


if __name__ == "__main__":
    main()
