"""Independent readback checks for the World V2 review and pCR inputs."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "runs/registered_roi32_20260919/analysis_30_pcr_v1v4_20260919"
PHASES = ("Precontrast", "First-post", "Late")


def read(path):
    return json.loads(path.read_text())


def identity(path):
    path = Path(path).resolve()
    st = path.stat()
    return {"path": str(path), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def verify_receipt(path, protocol):
    receipt = read(path)
    assert receipt["protocol"] == identity(protocol)
    assert all(identity(item["path"]) == item for item in receipt["artifacts"])
    return receipt


def review():
    web = ROOT / "review/web"
    protocol = read(ROOT / "protocol.json")
    cases = protocol["cases"]
    assert len(cases) == len({c["patient_id"] for c in cases}) == 30
    assert set(Counter(c["transition"] for c in cases).values()) == {5}
    metrics = pd.read_csv(web / "case_metrics.csv", float_precision="round_trip")
    manifest_text = (web / "manifest.js").read_text()
    manifest = json.loads(manifest_text.removeprefix("window.WORLD_CASES=").removesuffix(";\n"))
    info = {c["case"]: c for c in manifest}
    errors, pack_errors, diversity, enhancement = [], [], [], []
    for case in cases:
        folder = ROOT / "review" / case["case"]
        receipt = verify_receipt(folder / "complete.json", ROOT / "protocol.json")
        assert receipt["source_only_audit_passed"] and receipt["target_loaded_after_forecast"]
        with np.load(folder / "volumes.npz", allow_pickle=False) as archive:
            data = dict(archive)
        assert data["samples"].shape == (4, 3, 32, 128, 128)
        assert data["normalized_latents"].shape == (4, 24, 8, 32, 32)
        assert all(np.isfinite(a).all() for a in data.values())
        assert np.array_equal(data["samples"].mean(axis=0), data["mean"])
        mask = data["mask"].astype(bool)
        assert int(mask.sum((1, 2)).argmax()) == info[case["case"]]["slice"]
        real = np.concatenate([data[name][data[name + "_valid"].astype(bool)] for name in ("source", "target")])
        assert np.array_equal(np.percentile(real, [.5, 99.5]), info[case["case"]]["window"])
        volumes = {name: data[name] for name in ("source", "target", "source_vq", "target_vq", "mean")}
        volumes.update({f"sample{i+1}": v for i, v in enumerate(data["samples"])})
        packed_text = (web / "cases" / (case["case"] + ".js")).read_text()
        key, packed = json.loads("[" + packed_text.removeprefix("window.registerWorldCase(").removesuffix(");\n") + "]")
        assert key == case["case"] and set(packed["volumes"]) == set(volumes)
        for name, item in packed["volumes"].items():
            rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(item["png"].split(",", 1)[1]))), dtype=np.uint16)
            decoded = ((rgb[:, :, 0] * 256 + rgb[:, :, 1]).astype(np.float64) * item["scale"] + item["lower"]).reshape(3, 32, 128, 128)
            error = float(np.max(np.abs(decoded - volumes[name])))
            assert error <= item["scale"] / 2 + 1e-7
            pack_errors.append(error)
        decoded_mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(packed["mask"].split(",", 1)[1])))).reshape(32, 128, 128) > 0
        assert np.array_equal(decoded_mask, mask)
        rows = metrics[metrics.case == case["case"]]
        for row in rows.itertuples():
            phase = PHASES.index(row.phase)
            support = data["source_valid"][phase].astype(bool) & data["target_valid"][phase].astype(bool)
            if row.region == "fixed_T0_region":
                support &= mask
            assert support.sum() == row.voxels
            methods = [f"sample{i+1}" for i in range(4)] if row.method == "individual_mean" else [row.method]
            residuals = np.stack([volumes[k][phase][support].astype(np.float64) for k in methods]) - data["target"][phase][support].astype(np.float64)
            expected_mae = np.abs(residuals).mean(1).mean()
            expected_rmse = np.sqrt(np.square(residuals).mean(1)).mean()
            errors.extend([abs(row.mae - expected_mae), abs(row.rmse - expected_rmse)])
        for phase, name in enumerate(PHASES):
            support = mask & data["source_valid"][phase].astype(bool) & data["target_valid"][phase].astype(bool)
            samples = data["samples"][:, phase, support].astype(np.float64)
            pairs = [float(np.abs(samples[a] - samples[b]).mean()) for a, b in itertools.combinations(range(4), 2)]
            diversity.append({"case": case["case"], "phase": name, "fixed_T0_pairwise_mae": float(np.mean(pairs))})
        for phase in (1, 2):
            support = mask.copy()
            for stage in ("source", "target"):
                support &= data[stage + "_valid"][0].astype(bool) & data[stage + "_valid"][phase].astype(bool)
            true = (data["target"][phase].astype(np.float64) - data["target"][0].astype(np.float64))[support]
            for name in ("source", "sample1", "sample2", "sample3", "sample4", "mean", "target_vq"):
                prediction = (volumes[name][phase].astype(np.float64) - volumes[name][0].astype(np.float64))[support]
                enhancement.append({"case": case["case"], "difference": PHASES[phase] + "-Precontrast", "method": name,
                                    "fixed_T0_mae": float(np.abs(prediction - true).mean())})
        print(json.dumps({"review_case_verified": case["case"]}), flush=True)
    assert max(errors) < 1e-12 and len(metrics) == 1620
    assert min(r["fixed_T0_pairwise_mae"] for r in diversity) > 0
    pd.DataFrame(diversity).to_csv(web / "sample_diversity.csv", index=False)
    pd.DataFrame(enhancement).to_csv(web / "phase_difference_metrics.csv", index=False)
    return {"passed": True, "patients": 30, "raw_volume_metric_rows": len(metrics), "metric_replays": len(errors),
            "max_metric_error": max(errors), "packed_volumes_verified": len(pack_errors),
            "max_packing_error": max(pack_errors), "masks_windows_slices_means_verified": True,
            "all_case_phase_draw_diversities_positive": True}


def pcr():
    protocol = read(ROOT / "protocol.json")
    assert read(ROOT / "pcr/FEATURES_COMPLETE.json")["cases"] == 628
    tasks = [(mode, r) for r in protocol["routes"] for mode in (("direct",) if r["timepoint"] == 1 else ("direct", "rollout", "previous_real"))]
    replay_errors, archives = [], []
    for mode, route in tasks:
        tp, pid = route["timepoint"], route["patient_id"]
        folder = ROOT / "pcr" / mode / route["key"]
        receipt = verify_receipt(folder / "complete.json", ROOT / "protocol.json")
        assert receipt["source_read_audit_passed"] and receipt["target_image_or_latent_reads"] == 0
        allowed = {f"{pid}:T0"}
        if mode == "previous_real":
            allowed.add(f"{pid}:T{tp-1}")
        assert set(receipt["declared_real_latents"]) == allowed
        assert f"{pid}:T{tp}" not in allowed
        if mode == "rollout":
            canonical = "direct" if tp == 2 else "rollout"
            parent = ROOT / "pcr" / canonical / f"{pid}_T{tp-1}" / "volumes.npz"
            assert receipt["parent_generated_latent"] == identity(parent)
        else:
            assert receipt["parent_generated_latent"] is None
        replay_errors.append(receipt["real_T0_pillar_replay_max_error"])
        archives.append(folder / "volumes.npz")
    count = 0
    for mode in ("direct", "rollout", "previous_real"):
        for route in protocol["routes"]:
            for draw in range(4):
                path = ROOT / "pcr/embeddings" / mode / f"draw_{draw}" / route["patient_id"] / (route["key"] + ".pt")
                feature = torch.load(path, map_location="cpu", weights_only=True)
                assert feature.shape == (1152,) and torch.isfinite(feature).all() and feature.abs().sum() > 0
                if route["timepoint"] == 1 and mode != "direct":
                    original = ROOT / "pcr/embeddings/direct" / f"draw_{draw}" / route["patient_id"] / (route["key"] + ".pt")
                    assert torch.equal(feature, torch.load(original, map_location="cpu", weights_only=True))
                count += 1
    def check_archive(path):
        with np.load(path, allow_pickle=False) as archive:
            for key, shape in (("samples", (4, 3, 32, 128, 128)), ("normalized_latents", (4, 24, 8, 32, 32)),
                               ("source_foreground", (3, 32, 128, 128))):
                values = archive[key]
                assert values.shape == shape and np.isfinite(values).all()
        return 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        archive_count = sum(pool.map(check_archive, archives))
    assert count == 3312 and archive_count == 628
    return {"passed": True, "source_only_receipts": len(tasks), "finite_volume_archives": archive_count,
            "generated_volumes": 4 * archive_count, "finite_feature_vectors": count,
            "real_T0_pillar_replay_max_error": max(replay_errors), "shared_T1_exact_equality": True,
            "rollout_parent_identities_verified": True, "future_image_and_latent_reads": 0}


def replay():
    sys.path.insert(0, str(REPO / "scripts"))
    import evaluate_world_v2 as inference
    inference.configure()
    context = inference.ForecastContext()
    records = read(ROOT / "protocol.json")["routes"]
    tasks = [("single_worker", "direct", records[0])]
    parallel_receipts = sorted((ROOT / "pcr/direct").glob("*/complete.json"))
    for path in parallel_receipts:
        receipt = read(path)
        if receipt["completed_utc"] >= "2026-09-19T15:02:40":
            tasks.append(("parallel_workers", "direct", receipt["route"]))
            break
    assert len(tasks) == 2
    results = []
    for policy, mode, route in tasks:
        pair = context.pairs[route["direct_pair"]]
        with context.source_guard([], [pair["source"]]):
            latent, samples = context.forecast(context.source(pair["source"]), pair, route["noise_seed"])
        with np.load(ROOT / "pcr" / mode / route["key"] / "volumes.npz", allow_pickle=False) as old:
            latent_error = float(np.max(np.abs(latent.cpu().numpy() - old["normalized_latents"])))
            image_error = float(np.max(np.abs(samples - old["samples"])))
        results.append({"execution_policy": policy, "latent_max_error": latent_error, "decoded_MRI_max_error": image_error})
        print(json.dumps(results[-1]), flush=True)
    return {"latent_bitwise_replay_passed": all(r["latent_max_error"] == 0 for r in results),
            "decoded_MRI_bitwise_replay_passed": all(r["decoded_MRI_max_error"] == 0 for r in results),
            "interpretation": "Sampler latents and FP32 decoder outputs have separate reproducibility checks; no image tolerance is silently substituted for bitwise equality",
            "sampler_replay": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("review", "pcr", "replay"), required=True)
    args = parser.parse_args()
    if args.mode != "replay":
        torch.set_num_threads(1)
    result = {"review": review, "pcr": pcr, "replay": replay}[args.mode]()
    path = ROOT / (args.mode + "_independent_verification.json")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
