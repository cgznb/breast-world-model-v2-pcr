"""Audited legacy-cache adapter, source-only reads, triplets and patient sampling."""
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import copy
import hashlib
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from .conditioning import ALLOWED, validate_condition
from .utils import read_json, write_json, file_digest, file_identity

PHASES = ["pre_aqc0", "first_post_aqc1", "metadata_late"]
SCHEMA = "symm_world_manifest_v2"


def stage_index(stage):
    if not isinstance(stage, str) or not stage.startswith("T") or not stage[1:].isdigit():
        raise ValueError(f"Invalid longitudinal stage: {stage!r}")
    return int(stage[1:])


def geometry_digest(value):
    # Keep a literal geometry key: this workspace does not record checksums.
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _legacy_conditions(pair):
    clinical = {**(pair.get("baseline_clinical") or {}), **(pair.get("treatment") or {})}
    forbidden = [k for k in clinical if any(x in k.lower() for x in ("pcr", "survival", "recurrence", "pathology", "outcome"))]
    if forbidden:
        raise ValueError(f"Legacy manifest contains forbidden input fields: {forbidden}")
    result = {k: v for k, v in clinical.items() if k in ALLOWED}
    result.update(stage_i=pair["earlier_stage"], stage_j=pair["later_stage"], delta_days=pair.get("delta_days"),
                  interval_verified=(pair.get("interval_source") == "verified_relative_dicom_study_date"
                                     and not pair.get("interval_missing", True)))
    return validate_condition(result)


def convert_legacy(root, output, qc_path=None, auxiliary_root=None):
    """No training or patient images are downloaded, copied or modified."""
    root = Path(root).resolve()
    inv = read_json(root / "admitted_inventory.json")
    if inv.get("phase_order") != PHASES:
        raise ValueError("Legacy phase order differs from the audited three-phase workflow")
    quality = read_json(qc_path) if qc_path else {}
    visits = {v["visit_id"]: v for v in inv["visits"]}
    views = []
    for v in inv["views"]:
        visit = visits[v["visit_id"]]
        q = quality.get("views", {}).get(v["view_id"], {})
        row = {"id": v["view_id"], "patient_id": v["patient_id"], "visit": visit["visit"], "split": v["split"],
               "latent": str((root / v["latent_file"]).resolve()), "geometry": v.get("geometry", {}),
               "grid_id": geometry_digest(v.get("geometry", {})),
               "source_available_grid": q.get("source_available_grid", False),
               "kinetics_kind": q.get("kinetics_kind", "unavailable")}
        if v.get("reference_file"):
            row["images"] = str((root / v["reference_file"]).resolve())
        if auxiliary_root:
            p = Path(auxiliary_root).resolve() / (v["view_id"] + ".npz")
            if p.exists():
                row["auxiliary"] = str(p)
        if "labels" in q:
            row["labels"] = q["labels"]
        views.append(row)
    pairs = []
    for p in inv["pairs"]:
        pid = str(p["pair_id"])
        q = quality.get("pairs", {}).get(pid, {})
        conditions = _legacy_conditions(p)
        if q.get("conditions"):
            conditions.update(q["conditions"])
            conditions = validate_condition(conditions)
        pairs.append({"id": pid, "patient_id": p["patient_id"], "split": p["split"],
                      "source": p["source_view"], "target": p["target_view"], "conditions": conditions,
                      "registered": q.get("registered", False), "qc_weight": q.get("qc_weight", 1.0),
                      "treatment_verified": q.get("treatment_verified", False),
                      "anatomy_comparable": q.get("anatomy_comparable", False),
                      "scenario_id": q.get("scenario_id"),
                      "plan_known_at_initial_source": q.get("plan_known_at_initial_source", False)})
    result = {"schema": SCHEMA, "phase_order": PHASES, "latent_channels": 24,
              "views": views, "pairs": pairs, "provenance": {"adapter": "cgznb/symm-fm native_three_phase",
              "upstream_inventory": file_identity(root / "admitted_inventory.json"),
              "geometry_policy": "retained; NOT assumed registered or source-available"},
              "image_normalization": inv.get("image_normalization", {})}
    write_json(output, result)
    return DatasetStore(output).audit(scan_arrays=False)


def _merged_segments(segments):
    """Canonicalize a regimen timeline; splitting an identical segment is neutral."""
    by_drug = defaultdict(list)
    for e in segments:
        by_drug[(e["drug"], round(float(e["dose"]), 6))].append((float(e["start"]), float(e["end"])))
    result = []
    for (drug, dose), spans in sorted(by_drug.items()):
        spans.sort()
        merged = []
        for start, end in spans:
            if merged and start <= merged[-1][1] + 1e-5:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        result.extend((drug, dose, round(a, 5), round(b, 5)) for a, b in merged)
    return result


class DatasetStore:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.manifest = read_json(self.path)
        m = self.manifest
        if m.get("schema") != SCHEMA or m.get("phase_order") != PHASES or m.get("latent_channels") != 24:
            raise ValueError("Unsupported manifest / phase / latent contract")
        self.digest = file_digest(self.path)
        self.identity = file_identity(self.path)
        self.raw_cache = {}
        self.views = {v["id"]: v for v in m["views"]}
        self.pairs = list(m["pairs"])
        if len(self.views) != len(m["views"]) or len({p["id"] for p in self.pairs}) != len(self.pairs):
            raise ValueError("Duplicate view/pair IDs")
        patients = {}
        for view in self.views.values():
            split = view["split"]
            if split not in {"train", "val", "test"}:
                raise ValueError("Split must be train/val/test")
            if patients.setdefault(view["patient_id"], split) != split:
                raise ValueError("Patient overlaps training/validation/test")
            stage_index(view["visit"])
        for p in self.pairs:
            s, t = self.views[p["source"]], self.views[p["target"]]
            if s["patient_id"] != t["patient_id"] or s["patient_id"] != p["patient_id"]:
                raise ValueError("Cross-patient longitudinal pair")
            if s["split"] != t["split"] or s["split"] != p["split"]:
                raise ValueError("Cross-split longitudinal pair")
            if stage_index(s["visit"]) >= stage_index(t["visit"]):
                raise ValueError("Pair chronology is not strictly forward")
            p["conditions"] = validate_condition(p["conditions"])
            if p["conditions"].get("stage_i") != s["visit"] or p["conditions"].get("stage_j") != t["visit"]:
                raise ValueError("Condition stages disagree with observations")
            q = float(p.get("qc_weight", 1))
            if not 0 < q <= 1:
                raise ValueError("qc_weight must be in (0,1]; excluded data must be removed explicitly")
        self.patient_splits = patients
        self.statistics = None

    def resolve(self, path):
        return (self.path.parent / path).resolve() if not Path(path).is_absolute() else Path(path)

    def read_raw(self, view_id):
        if view_id in self.raw_cache:
            return self.raw_cache[view_id].clone()
        v = self.views[view_id]
        value = np.load(self.resolve(v["latent"]), allow_pickle=False)
        if isinstance(value, np.lib.npyio.NpzFile):
            with value as archive:
                value = np.asarray(archive["latent"], np.float32)
        else:
            value = np.asarray(value, np.float32)
        if value.ndim != 4 or value.shape[0] != 24 or not np.isfinite(value).all():
            raise ValueError(f"Invalid three-phase latent in {view_id}")
        return torch.from_numpy(value.copy())

    def preload(self):
        for key in self.views:
            if key not in self.raw_cache:
                self.raw_cache[key] = self.read_raw(key)

    def fit_statistics(self):
        # Deduplicate exactly by retained view, matching upstream cache semantics.
        total, square, count = np.zeros(24), np.zeros(24), 0
        n = 0
        for key, view in self.views.items():
            if view["split"] != "train":
                continue
            x = self.read_raw(key).numpy().astype(np.float64).reshape(24, -1)
            total += x.sum(1)
            square += np.square(x).sum(1)
            count += x.shape[1]
            n += 1
        if not count:
            raise ValueError("Cannot fit normalization without training observations")
        mean = total / count
        std = np.sqrt(np.maximum(square/count - mean**2, 0))
        if not np.isfinite(std).all() or (std < 1e-8).any():
            raise ValueError("Degenerate training latent channels")
        self.statistics = {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train",
                           "views": n, "manifest_identity": self.identity}
        return self.statistics

    def set_statistics(self, statistics):
        if statistics.get("fit_split") != "train":
            raise ValueError("Normalization must be fitted on training data")
        if statistics.get("manifest_identity") != self.identity:
            raise ValueError("Manifest changed since normalization was fitted")
        self.statistics = statistics

    def normalize(self, z):
        if self.statistics is None:
            raise ValueError("Fit/load statistics first")
        mean = z.new_tensor(self.statistics["mean"]).view(24, 1, 1, 1)
        std = z.new_tensor(self.statistics["std"]).view(24, 1, 1, 1)
        return (z-mean)/std

    def read_aux(self, view_id, grid):
        view = self.views[view_id]
        out = {}
        if view.get("auxiliary"):
            with np.load(self.resolve(view["auxiliary"]), allow_pickle=False) as arrays:
                for key in ("kinetics", "kinetics_mask", "segmentation", "segmentation_mask", "external_tokens"):
                    if key in arrays:
                        x = np.asarray(arrays[key], np.float32)
                        if not np.isfinite(x).all():
                            raise ValueError(f"Nonfinite auxiliary {key}")
                        out[key] = torch.from_numpy(x.copy())
        for key in ("kinetics", "kinetics_mask", "segmentation", "segmentation_mask"):
            if key in out:
                if out[key].ndim == 3:
                    out[key] = out[key][None]
                if out[key].ndim != 4:
                    raise ValueError(f"{key} must be [C,D,H,W]")
                if key.endswith("mask") and ((out[key] < 0) | (out[key] > 1)).any():
                    raise ValueError(f"{key} must be in [0,1]")
                out[key] = F.adaptive_avg_pool3d(out[key][None], grid)[0]
        labels = view.get("labels", {})
        if "biomarkers" in labels:
            vals = labels["biomarkers"]
            if len(vals) != 4:
                raise ValueError("Exactly four declared biomarker targets are required")
            if any(v is not None and not math.isfinite(float(v)) for v in vals):
                raise ValueError("Nonfinite biomarker label")
            out["biomarkers"] = torch.tensor([0 if v is None else v for v in vals], dtype=torch.float32)
            out["biomarker_mask"] = torch.tensor([v is not None for v in vals], dtype=torch.float32)
        if "pcr" in labels:
            if labels["pcr"] not in (0, 1, None):
                raise ValueError("pCR labels must be 0, 1 or missing")
            if labels["pcr"] is not None:
                out["pcr"] = torch.tensor(float(labels["pcr"]))
        return out

    def pair_batch(self, pairs, grid, device="cpu"):
        source, target = [], []
        for p in pairs:
            a, b = self.read_raw(p["source"]), self.read_raw(p["target"])
            if a.shape != b.shape:
                raise ValueError("Source/target shapes differ; do not silently interpolate longitudinal data")
            source.append(self.normalize(a)); target.append(self.normalize(b))
        return {"source": torch.stack(source).to(device), "target": torch.stack(target).to(device),
                "conditions": [p["conditions"] for p in pairs], "records": pairs,
                "source_aux": [self.read_aux(p["source"], grid) for p in pairs],
                "target_aux": [self.read_aux(p["target"], grid) for p in pairs]}

    def source_batch(self, pair, device="cpu"):
        # Intentionally no call to read_aux() and no target path resolution.
        return self.normalize(self.read_raw(pair["source"]))[None].to(device), [pair["conditions"]]

    def records(self, split):
        return [p for p in self.pairs if p["split"] == split]

    def triplets(self, cfg, split="train"):
        pairs = self.records(split)
        # A transition to another view of the same visit is NOT automatically
        # composable: require common verified geometry across all four endpoints.
        by_patient = defaultdict(list)
        for p in pairs:
            by_patient[p["patient_id"]].append(p)
        accepted, reasons = [], Counter()
        for patient_pairs in by_patient.values():
            for ab in patient_pairs:
                a, b = self.views[ab["source"]], self.views[ab["target"]]
                for bc in patient_pairs:
                    b2, c = self.views[bc["source"]], self.views[bc["target"]]
                    if b2["visit"] != b["visit"]:
                        continue
                    for ac in patient_pairs:
                        if self.views[ac["source"]]["visit"] != a["visit"] or self.views[ac["target"]]["visit"] != c["visit"]:
                            continue
                        endpoints = [a, b, b2, c, self.views[ac["source"]], self.views[ac["target"]]]
                        error = None
                        grids = [v.get("grid_id") for v in endpoints]
                        geometries = [geometry_digest(v.get("geometry", {})) for v in endpoints]
                        if None in grids or len(set(grids)) != 1 or len(set(geometries)) != 1:
                            error = "incompatible_coordinate_grids"
                        elif not all(v.get("source_available_grid", False) for v in endpoints):
                            error = "future_dependent_or_unverified_output_grid"
                        elif cfg.require_registered and not all(p.get("registered", False) for p in (ab, bc, ac)):
                            error = "registration_unverified"
                        conds = [p["conditions"] for p in (ab, bc, ac)]
                        if error is None and cfg.require_verified_intervals:
                            if not all(x.get("interval_verified") for x in conds):
                                error = "clinical_intervals_unverified"
                            elif abs(conds[0]["delta_days"] + conds[1]["delta_days"] - conds[2]["delta_days"]) > 1e-3:
                                error = "nonadditive_clinical_intervals"
                        if error is None:
                            static_keys = ("age", "hr_status", "her2_status", "mammaprint")
                            if any(len({json.dumps(x.get(k), sort_keys=True) for x in conds}) != 1 for k in static_keys):
                                error = "inconsistent_baseline_information"
                        if error is None and cfg.require_verified_treatment:
                            if not ab.get("scenario_id") or len({p.get("scenario_id") for p in (ab,bc,ac)}) != 1 or not all(p.get("plan_known_at_initial_source", False) for p in (ab,bc,ac)):
                                error = "scenario_not_declared_at_initial_source"
                            elif not all(p.get("treatment_verified", False) for p in (ab, bc, ac)):
                                error = "treatment_plan_unverified"
                            elif _merged_segments(conds[0].get("action_segments", []) + conds[1].get("action_segments", [])) != _merged_segments(conds[2].get("action_segments", [])):
                                error = "incompatible_treatment_schedules"
                            elif not conds[2].get("action_segments"):
                                error = "missing_explicit_treatment_schedule"
                        if error:
                            reasons[error] += 1
                        else:
                            accepted.append((ab, bc, ac))
        return accepted, dict(reasons)

    def audit(self, scan_arrays=True, rollout_cfg=None):
        shapes = Counter()
        if scan_arrays:
            for key in self.views:
                shapes[str(tuple(self.read_raw(key).shape))] += 1
        report = {"manifest_identity": self.identity, "patients": dict(Counter(self.patient_splits.values())),
                  "pairs": dict(Counter(p["split"] for p in self.pairs)), "latent_shapes": dict(shapes),
                  "views": len(self.views), "patient_split_overlap": False,
                  "kinetics_kind": dict(Counter(v.get("kinetics_kind", "unavailable") for v in self.views.values())),
                  "source_available_grid_views": sum(v.get("source_available_grid", False) for v in self.views.values()),
                  "warning": "Equal tensor sizes do not establish longitudinal registration or source-only localization."}
        if rollout_cfg:
            triples, rejected = self.triplets(rollout_cfg)
            report["eligible_training_triplets"] = len(triples)
            report["triplet_rejections"] = rejected
        return report


class PatientBalancedSampler:
    """Sample patient -> transition -> observation pair, with a counter RNG.

    This avoids over-weighting patients with many all-pairs entries. Sampling is
    exactly replayable after resume, independent of DataLoader worker states.
    """
    def __init__(self, records, seed):
        self.seed = int(seed)
        self.groups = defaultdict(lambda: defaultdict(list))
        for p in records:
            self.groups[p["patient_id"]][(p["conditions"]["stage_i"], p["conditions"]["stage_j"])].append(p)
        self.patients = sorted(self.groups)
        if not self.patients:
            raise ValueError("No records to sample")
    def batch(self, counter, size):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(counter)]))
        out = []
        for _ in range(size):
            patient = self.patients[int(rng.integers(len(self.patients)))]
            transitions = sorted(self.groups[patient])
            candidates = self.groups[patient][transitions[int(rng.integers(len(transitions)))]]
            out.append(candidates[int(rng.integers(len(candidates)))])
        return out


class PatientBalancedTripletSampler:
    """Counter-based patient -> triplet sampling for sparse follow-up cohorts."""
    def __init__(self, triplets, seed):
        self.seed=int(seed)
        self.groups=defaultdict(list)
        for triple in triplets:self.groups[triple[0]["patient_id"]].append(triple)
        self.patients=sorted(self.groups)
        if not self.patients:raise ValueError("No eligible triplets")
    def sample(self,counter):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,9173,int(counter)]))
        patient=self.patients[int(rng.integers(len(self.patients)))]
        candidates=self.groups[patient]
        return candidates[int(rng.integers(len(candidates)))]
