"""Import the registered, reduced three-phase cache without resampling it."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from .conditioning import validate_condition
from .data import DatasetStore, PHASES, SCHEMA, _legacy_conditions
from .utils import file_identity, read_json, write_json


def verify_identity(identity):
    if file_identity(identity["path"]) != identity:
        raise ValueError(f"Source file changed: {identity['path']}")


def convert_registered_roi32(root, output, labels_csv=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    inventory = read_json(root / "inventory.json")
    cache = read_json(root / "latents/cache_manifest.json")
    previous_stats = read_json(root / "latents/statistics.json")
    complete = read_json(root / "latents/COMPLETE.json")
    if (inventory.get("schema") != "registered_three_phase_roi32_v1"
            or inventory.get("phase_order") != PHASES or inventory["missing_sources"]
            or complete["status"] != "passed"):
        raise ValueError("Expected a completed, full-cohort registered three-phase ROI32 cache")
    verify_identity(cache["codec"])
    for identity in cache["files"]:
        verify_identity(identity)
    by_filename = {Path(row["path"]).name: row for row in cache["files"]}
    if len(by_filename) != len(cache["files"]):
        raise ValueError("Duplicate latent filenames")
    labels = {}
    if labels_csv:
        with Path(labels_csv).open() as handle:
            for row in csv.DictReader(handle):
                if row["pid"] in labels:
                    raise ValueError("Duplicate pCR patient ID")
                value = float(row["pCR"])
                if value not in (0, 1):
                    raise ValueError("pCR must be an observed binary target")
                labels[row["pid"]] = int(value)
    visits = {v["visit_id"]: v for v in inventory["visits"]}
    reports, metadata, views = {}, {}, []
    registration_counts = Counter()
    for visit in inventory["visits"]:
        vid, pid = visit["visit_id"], visit["patient_id"]
        if len(visit["phase_indices"]) != 3 or visit["phase_indices"][:2] != [0, 1] or visit["phase_indices"][2] <= 1:
            raise ValueError("Incorrect pre / first-post / metadata-late ordering")
        crop_path = visit["crop_report"]["path"]
        if crop_path not in reports:
            verify_identity(visit["crop_report"])
            reports[crop_path] = read_json(crop_path)
        crop = reports[crop_path]
        if crop["shape_zyx"] != [32, 128, 128] or crop["spacing_zyx_mm"] != [2.0, 0.7032, 0.7032]:
            raise ValueError("This adapter accepts only the existing reduced ROI32 geometry")
        verify_identity(visit["metadata_source"])
        meta = read_json(visit["metadata_source"]["path"])
        if meta["registered_to_visit"] != "T0" or meta["intra_visit_registration"]["status"] != "completed":
            raise ValueError("Unverified T0 or intravisit registration")
        if meta["registration_status"] not in {"fixed_reference", "deformable", "rigid_fallback"}:
            raise ValueError("Unexpected longitudinal registration status")
        metadata[vid] = meta
        registration_counts[meta["registration_status"]] += 1
        latent = by_filename[vid.replace(":", "_") + ".npy"]
        array = np.load(latent["path"], allow_pickle=False)
        if array.shape != (24, 8, 32, 32) or array.dtype != np.float16 or not np.isfinite(array).all():
            raise ValueError("ROI32 cache must contain finite float16 [24,8,32,32] raw latents")
        view = {"id": vid, "patient_id": pid, "visit": visit["visit"], "split": visit["fold"],
                "latent": latent["path"], "source_available_grid": True,
                "grid_id": pid + ":fixed_T0_ROI32", "kinetics_kind": "unavailable",
                "geometry": {k: crop[k] for k in ("shape_zyx", "spacing_zyx_mm", "crop_affine_ras")}}
        if labels_csv:
            view["labels"] = {"pcr": labels[pid]}
        views.append(view)
    if len(views) != len(by_filename):
        raise ValueError("Latent cache and inventory do not cover exactly the same visits")
    pairs = []
    known_dates = {"tcia_metadata_study_uid", "tcia_study_uid_study_date", "local_dicom_study_date"}
    for pair in inventory["pairs"]:
        source, target = (visits[pair[k]] for k in ("earlier_visit_id", "later_visit_id"))
        if any(v["visit_date_source"] not in known_dates for v in (source, target)):
            raise ValueError("Clinical interval lacks a recognized original date source")
        delta = (datetime.fromisoformat(target["visit_date"])-datetime.fromisoformat(source["visit_date"])).days
        if pair["interval_missing"] or delta <= 0 or delta != pair["delta_days"]:
            raise ValueError("Longitudinal interval does not replay from original visit dates")
        if metadata[source["visit_id"]]["target_geometry"] != metadata[target["visit_id"]]["target_geometry"]:
            raise ValueError("Longitudinal target geometry mismatch")
        conditions = _legacy_conditions(pair)
        conditions.update(delta_days=delta, interval_verified=True)
        pairs.append({"id": pair["pair_id"], "patient_id": pair["patient_id"], "split": pair["split"],
                      "source": pair["earlier_visit_id"], "target": pair["later_visit_id"],
                      "conditions": validate_condition(conditions), "registered": True,
                      "anatomy_comparable": True, "qc_weight": 1.0,
                      "treatment_verified": False, "plan_known_at_initial_source": False,
                      "scenario_id": None})
    expected_training = sorted(v["visit_id"] for v in visits.values() if v["fold"] == "train")
    if (previous_stats["fit_split"] != "train" or previous_stats["fit_visit_ids"] != expected_training
            or previous_stats["phase_order"] != PHASES):
        raise ValueError("Previous normalization was not fitted on the retained training visits")
    normalization = {k: inventory["image_normalization"][k] for k in ("mean", "std", "fit_split", "scope")}
    manifest = {"schema": SCHEMA, "phase_order": PHASES, "latent_channels": 24,
                "views": views, "pairs": pairs, "image_normalization": normalization,
                "provenance": {"adapter": "registered_three_phase_roi32", "source_inventory": file_identity(root/"inventory.json"),
                               "source_cache": file_identity(root/"latents/cache_manifest.json"),
                               "codec": cache["codec"], "registration_status": dict(registration_counts),
                               "pcr_labels": file_identity(labels_csv) if labels_csv else None,
                               "pcr_policy": "supervision only; never an inference condition",
                               "end_to_end_test_isolation_verified": False,
                               "rollout_policy": "disabled: no verified source-known segment-level treatment plan",
                               "auxiliary_policy": "no measured kinetics, follow-up segmentation or external teacher imported"}}
    write_json(output, manifest)
    store = DatasetStore(output)
    stats = store.fit_statistics()
    for name in ("mean", "std"):
        if not np.allclose(stats[name], previous_stats[name], rtol=0, atol=1e-12):
            raise ValueError("Recomputed training-only statistics differ from original ROI32 statistics")
    report = store.audit(scan_arrays=False)
    report.update(latent_shapes={"(24, 8, 32, 32)": len(views)}, raw_arrays_checked=len(views),
                  image_shape_czyx=[3, 32, 128, 128], verified_intervals=len(pairs),
                  training_statistics_replayed=True, pcr_patients=len({v["patient_id"] for v in views if "labels" in v}))
    write_json(output.with_suffix(".audit.json"), report)
    return report
