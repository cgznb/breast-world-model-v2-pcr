"""Render frozen ROI32 forecasts as figures and an offline volume browser."""
from __future__ import annotations

import base64
import io
import itertools
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runs/registered_roi32_20260919/analysis_30_pcr_v1v4_20260919"
WEB = ROOT / "review/web"
TEMPLATE = REPO / "scripts/world_v2_review_web"
PHASES = ("Precontrast", "First-post", "Late")
PANELS = (("source", "Observed source"), ("target", "True follow-up"),
          ("sample1", "World V2: draw 1"), ("mean", "World V2: mean of 4"),
          ("target_vq", "True follow-up VQ"))


def read(path):
    return json.loads(path.read_text())


def identity(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False) + "\n")


def packed_volume(volume):
    value = np.asarray(volume, dtype=np.float32)
    assert value.shape == (3, 32, 128, 128) and np.isfinite(value).all()
    lower, upper = float(value.min()), float(value.max())
    scale = max((upper - lower) / 65535, 1e-10)
    packed = np.rint((value.astype(np.float64) - lower) / scale).clip(0, 65535).astype(np.uint16)
    error = float(np.max(np.abs(packed.astype(np.float64) * scale + lower - value)))
    assert error <= scale / 2 + 1e-7
    raster = packed.reshape(4096, 384)
    rgb = np.zeros((4096, 384, 3), dtype=np.uint8)
    rgb[:, :, 0], rgb[:, :, 1] = raster >> 8, raster & 255
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", compress_level=1)
    return {"png": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "lower": lower, "scale": scale, "max_error": error}


def packed_mask(mask):
    buffer = io.BytesIO()
    Image.fromarray((mask.reshape(4096, 128) * 255).astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def panel(axis, value, window, mask=None):
    axis.imshow(value, cmap="gray", vmin=window[0], vmax=window[1], interpolation="nearest")
    if mask is not None and mask.any() and not mask.all():
        axis.contour(mask, levels=[.5], colors=["#38d49e"], linewidths=.45)
    axis.set_xticks([])
    axis.set_yticks([])


def figure_case(case, volumes, mask, z, window, destination):
    fig, axes = plt.subplots(3, 5, figsize=(12.5, 8), layout="constrained")
    for phase, row in enumerate(axes):
        for column, ((key, title), ax) in enumerate(zip(PANELS, row)):
            panel(ax, volumes[key][phase, z], window, mask[z])
            if phase == 0:
                ax.set_title(title, fontsize=10)
            if column == 0:
                ax.set_ylabel(PHASES[phase], fontsize=11)
    fig.suptitle(f"{case['case']} | {case['transition']} | {case['days']:g} days | z={z} | EMA 4000 / Heun25", fontsize=12)
    fig.savefig(destination / "comparison.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.6), layout="constrained")
    real_deltas = np.concatenate([(volumes[name][phase] - volumes[name][0]).ravel()
                                 for name in ("source", "target") for phase in (1, 2)])
    limit = max(float(np.percentile(np.abs(real_deltas), 99)), 1e-5)
    for phase, row in zip((1, 2), axes):
        for column, ((key, title), ax) in enumerate(zip(PANELS, row)):
            ax.imshow(volumes[key][phase, z] - volumes[key][0, z], cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_xticks([])
            ax.set_yticks([])
            if phase == 1:
                ax.set_title(title, fontsize=10)
            if column == 0:
                ax.set_ylabel(PHASES[phase] + " - Pre", fontsize=10)
    fig.suptitle(f"{case['case']} | phase intensity differences | common range +/- {limit:.3f}", fontsize=12)
    fig.savefig(destination / "enhancement.png", dpi=140)
    plt.close(fig)
    return limit


def case_metrics(case, arrays, volumes):
    rows = []
    for phase, name in enumerate(PHASES):
        common = arrays["source_valid"][phase].astype(bool) & arrays["target_valid"][phase].astype(bool)
        for region, selected in (("common_foreground", common), ("fixed_T0_region", common & arrays["mask"])):
            if not selected.any():
                raise ValueError("Empty metric support")
            target = arrays["target"][phase][selected].astype(np.float64)
            source = arrays["source"][phase][selected].astype(np.float64)
            candidates = {k: v[phase][selected].astype(np.float64) for k, v in volumes.items() if k != "target"}
            for method, values in candidates.items():
                rows.append({"case": case["case"], "transition": case["transition"], "phase": name, "region": region,
                             "method": method, "voxels": int(selected.sum()),
                             "mae": float(np.abs(values - target).mean()),
                             "rmse": float(np.sqrt(np.square(values - target).mean())),
                             "change_from_source_mae": float(np.abs(values - source).mean()),
                             "observed_change_mae": float(np.abs(target - source).mean())})
            individual = [r for r in rows if r["phase"] == name and r["region"] == region and r["method"].startswith("sample")]
            rows.append({**individual[0], "method": "individual_mean",
                         **{k: float(np.mean([r[k] for r in individual])) for k in ("mae", "rmse", "change_from_source_mae")}})
    return rows


def overview(cases, entries, name, title):
    fig, axes = plt.subplots(len(cases), 5, figsize=(12.5, 2.25 * len(cases)), layout="constrained", squeeze=False)
    for case, row in zip(cases, axes):
        with np.load(ROOT / "review" / case["case"] / "volumes.npz", allow_pickle=False) as a:
            data = {name: a[name] for name in ("source", "target", "mean", "target_vq")}
            data["sample1"] = a["samples"][0]
            info = entries[case["case"]]
            for (key, label), ax in zip(PANELS, row):
                panel(ax, data[key][1, info["slice"]], info["window"], a["mask"][info["slice"]])
                if row is axes[0]:
                    ax.set_title(label, fontsize=10)
            row[0].set_ylabel(f"{case['case']}\n{case['transition']}\n{case['days']:g} days", fontsize=9)
    for (_, label), ax in zip(PANELS, axes[0]):
        ax.set_title(label, fontsize=10)
    fig.suptitle(title + " | first-post | fixed T0 contour", fontsize=12)
    fig.savefig(WEB / name, dpi=140)
    plt.close(fig)


def main():
    protocol = read(ROOT / "protocol.json")
    assert read(ROOT / "review/INFERENCE_COMPLETE.json")["patients"] == 30
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / "cases").mkdir(exist_ok=True)
    entries, metric_rows, max_pack_error = {}, [], 0.0
    for case in protocol["cases"]:
        folder = ROOT / "review" / case["case"]
        receipt = read(folder / "complete.json")
        assert receipt["source_only_audit_passed"] and receipt["target_loaded_after_forecast"]
        assert receipt["protocol"] == identity(ROOT / "protocol.json")
        assert all(identity(Path(p["path"])) == p for p in receipt["artifacts"])
        with np.load(folder / "volumes.npz", allow_pickle=False) as a:
            arrays = dict(a)
        assert np.array_equal(arrays["samples"].mean(0), arrays["mean"])
        mask = arrays["mask"].astype(bool)
        z = int(mask.sum(axis=(1, 2)).argmax())
        values = np.concatenate([arrays[k][arrays[k + "_valid"].astype(bool)] for k in ("source", "target")])
        window = np.percentile(values, [.5, 99.5]).astype(float).tolist()
        assert window[1] > window[0]
        volumes = {k: arrays[k] for k in ("source", "target", "source_vq", "target_vq", "mean")}
        volumes.update({f"sample{i + 1}": sample for i, sample in enumerate(arrays["samples"])})
        delta_window = figure_case(case, volumes, mask, z, window, folder)
        metric_rows.extend(case_metrics(case, arrays, volumes))
        payload = {"volumes": {name: packed_volume(value) for name, value in volumes.items()}, "mask": packed_mask(mask)}
        max_pack_error = max(max_pack_error, max(v["max_error"] for v in payload["volumes"].values()))
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        (WEB / "cases" / (case["case"] + ".js")).write_text(f"window.registerWorldCase({json.dumps(case['case'])},{encoded});\n")
        entries[case["case"]] = {k: v for k, v in case.items() if k not in ("patient_id", "source", "target", "pair_id")}
        entries[case["case"]].update(slice=z, window=window, delta_window=delta_window,
                                      clipped=receipt["source_clipped"] or receipt["target_clipped"])
        print(json.dumps({"rendered": case["case"], "slice": z}), flush=True)
    frame = pd.DataFrame(metric_rows)
    frame.to_csv(WEB / "case_metrics.csv", index=False)
    summary = frame.groupby(["phase", "region", "method"], as_index=False).agg(
        patients=("case", "nunique"), mae=("mae", "mean"), patient_sd=("mae", "std"), rmse=("rmse", "mean"))
    summary.to_csv(WEB / "metric_summary.csv", index=False)
    for transition in sorted({c["transition"] for c in protocol["cases"]}):
        cases = [c for c in protocol["cases"] if c["transition"] == transition]
        overview(cases, entries, transition.replace(" -> ", "_") + ".png", "World V2 | " + transition)
    overview(protocol["cases"][:6], entries, "overview.png", "Six time transitions | EMA step 4000 / Heun25")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, region in zip(axes, ("common_foreground", "fixed_T0_region")):
        selected = summary[summary.region == region]
        for i, (method, label, color) in enumerate((("source", "Copy source", "#767676"),
                ("individual_mean", "Individual draw", "#b04968"), ("mean", "Mean of 4", "#168675"),
                ("target_vq", "True follow-up VQ", "#4778aa"))):
            ys = selected[selected.method == method].set_index("phase").loc[list(PHASES), "mae"]
            ax.bar(np.arange(3) + (i - 1.5) * .18, ys, .18, label=label, color=color)
        ax.set_xticks(range(3), PHASES)
        ax.set_title(region.replace("_", " "), fontsize=12)
        ax.set_ylabel("MRI MAE (shared training normalization)")
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle("30 distinct patients | equal patient weight | lower is better", fontsize=13)
    fig.savefig(WEB / "metrics.png", dpi=150)
    plt.close(fig)
    (WEB / "manifest.js").write_text("window.WORLD_CASES=" + json.dumps(list(entries.values()), ensure_ascii=True) + ";\n")
    for name in ("index.html", "viewer.css", "viewer.js"):
        shutil.copyfile(TEMPLATE / name, WEB / name)
    lucide = Path("/path/to/research/.codex/euler20_review_browser/offline_test_20260914/Local Results/ispy2_mu_results_offline_20260914/lucide.min.js")
    shutil.copyfile(lucide, WEB / "lucide.min.js")
    write_json(WEB / "render_receipt.json", {"patients": 30, "phases": 3, "draws": 4, "checkpoint_step": 4000,
               "target_only_diagnostic": "target_vq", "slice_policy": "Maximum fixed T0 mask area; no generated-image selection",
               "display_window": "Shared 0.5/99.5 percentiles of source and target foreground, display only",
               "metrics": "Unclipped float32 MRI; each patient equally weighted; common foreground and fixed T0 region",
               "max_display_packing_error": max_pack_error, "protocol": identity(ROOT / "protocol.json"),
               "source_only_receipts_checked": 30})
    print(json.dumps({"status": "complete", "web": str(WEB), "metric_rows": len(frame)}), flush=True)


if __name__ == "__main__":
    main()
