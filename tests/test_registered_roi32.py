import csv
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from symm_world.data import DatasetStore, PHASES
from symm_world.evaluation import balanced_records
from symm_world.registered_roi32 import convert_registered_roi32
from symm_world.training import initialize_metadata, stage_budget, train_stage
from symm_world.utils import file_identity, read_json, write_json


@pytest.fixture
def roi_cache(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    latent_root = root / "latents/raw"
    latent_root.mkdir(parents=True)
    codec = root / "codec.pt"
    torch.save({"test_fixture": True}, codec)
    visits, pairs, files, training = [], [], [], []
    rng = np.random.default_rng(14)
    for i, split in enumerate(("train", "train", "val")):
        patient = f"fixture_{i}"
        crop_path = root / f"{patient}.json"
        write_json(crop_path, {"shape_zyx": [32, 128, 128], "spacing_zyx_mm": [2.0, .7032, .7032],
                              "crop_affine_ras": np.eye(4).tolist()})
        for t in (0, 1):
            vid = f"{patient}:T{t}"
            meta = root / f"{patient}_T{t}.json"
            write_json(meta, {"registered_to_visit": "T0", "registration_status": "rigid_fallback",
                              "intra_visit_registration": {"status": "completed"}, "target_geometry": {"grid": i}})
            raw = rng.normal(size=(24, 8, 32, 32)).astype(np.float16)
            latent = latent_root / f"{patient}_T{t}.npy"
            np.save(latent, raw)
            files.append(file_identity(latent))
            visits.append({"visit_id": vid, "patient_id": patient, "visit": f"T{t}", "fold": split,
                           "phase_indices": [0, 1, 3], "crop_report": file_identity(crop_path),
                           "metadata_source": file_identity(meta), "visit_date": f"2020-01-{1+t:02d}",
                           "visit_date_source": "local_dicom_study_date"})
            if split == "train":
                training.append(raw.astype(np.float64).reshape(24, -1))
        pairs.append({"pair_id": patient+":T0->T1", "patient_id": patient, "split": split,
                      "earlier_stage": "T0", "later_stage": "T1", "earlier_visit_id": patient+":T0",
                      "later_visit_id": patient+":T1", "interval_missing": False, "interval_source": "source_bundle:local_dicom_study_date",
                      "delta_days": 1, "baseline_clinical": {"age": 50}, "treatment": {"treatment_arm": "assigned"}})
    joined = np.concatenate(training, axis=1)
    write_json(root/"inventory.json", {"schema": "registered_three_phase_roi32_v1", "phase_order": PHASES,
                                      "missing_sources": [], "visits": visits, "pairs": pairs,
                                      "image_normalization": {"mean": 2, "std": 3, "fit_split": "train", "scope": "training"}})
    write_json(root/"latents/cache_manifest.json", {"files": files, "codec": file_identity(codec)})
    write_json(root/"latents/COMPLETE.json", {"status": "passed"})
    write_json(root/"latents/statistics.json", {"mean": joined.mean(1).tolist(), "std": joined.std(1).tolist(),
               "fit_split": "train", "phase_order": PHASES,
               "fit_visit_ids": sorted(v["visit_id"] for v in visits if v["fold"] == "train")})
    labels = root/"labels.csv"
    with labels.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pid", "pCR"])
        writer.writerows((f"fixture_{i}", i % 2) for i in range(3))
    return root, labels


def test_roi_cache_exact_latents_split_labels_and_statistics(roi_cache, tmp_path):
    root, labels = roi_cache
    output = tmp_path/"manifest.json"
    original = read_json(root/"latents/cache_manifest.json")["files"]
    report = convert_registered_roi32(root, output, labels)
    assert report["patients"] == {"train": 2, "val": 1}
    assert report["verified_intervals"] == 3
    store = DatasetStore(output)
    for pair in store.pairs:
        assert "pcr" not in pair["conditions"]
        assert pair["conditions"]["delta_days"] == 1
        assert not pair["treatment_verified"]
    before = store.read_raw(next(iter(store.views)))
    store.preload()
    after = store.read_raw(next(iter(store.views)))
    assert torch.equal(before, after)
    after.zero_()
    assert torch.equal(before, store.read_raw(next(iter(store.views))))
    assert original == [file_identity(v["path"]) for v in original]
    assert "sha256" not in output.read_text()


def test_roi_rejects_wrong_resolution(roi_cache, tmp_path):
    root, labels = roi_cache
    inventory = read_json(root/"inventory.json")
    path = Path(inventory["visits"][0]["crop_report"]["path"])
    crop = read_json(path)
    crop["shape_zyx"] = [128, 128, 128]
    write_json(path, crop)
    for visit in inventory["visits"]:
        if visit["crop_report"]["path"] == str(path):
            visit["crop_report"] = file_identity(path)
    write_json(root/"inventory.json", inventory)
    with pytest.raises(ValueError, match="reduced ROI32"):
        convert_registered_roi32(root, tmp_path/"bad.json", labels)


def test_roi_rejects_incorrect_interval(roi_cache, tmp_path):
    root, labels = roi_cache
    inventory = read_json(root/"inventory.json")
    inventory["pairs"][0]["delta_days"] = 9
    write_json(root/"inventory.json", inventory)
    with pytest.raises(ValueError, match="interval"):
        convert_registered_roi32(root, tmp_path/"bad.json", labels)


def test_resume_rejects_changed_assets_and_config(cfg, store, tmp_path):
    root = tmp_path/"run"
    initialize_metadata(cfg, store, root)
    changed = copy.deepcopy(cfg)
    changed.training.batch_size += 1
    with pytest.raises(ValueError, match="config/manifest"):
        initialize_metadata(changed, store, root)
    view = next(iter(store.views.values()))
    path = store.resolve(view["latent"])
    np.save(path, np.load(path) + 1)
    with pytest.raises(ValueError, match="files changed"):
        initialize_metadata(cfg, store, root)


def test_validation_subset_spreads_across_patients():
    records = [{"id": f"{p}:{i}", "patient_id": p} for p in ("a", "b", "c") for i in range(3)]
    assert {p["patient_id"] for p in balanced_records(records, 3)} == {"a", "b", "c"}
    assert {p["id"].split(":")[1] for p in balanced_records(records, 3)} == {"0", "1", "2"}


def test_larger_batch_retains_original_pair_budget(cfg, store, tmp_path):
    cfg.training.reference_batch_size = 2
    cfg.training.representation_steps = 5
    cfg.training.batch_size = 3
    cfg.training.accumulation = 2
    budget = stage_budget(cfg, "representation")
    assert budget["steps"] == 2 and budget["samples"] == 10
    assert budget["auxiliary_every"] == cfg.training.auxiliary_every
    result = train_stage("representation", cfg, store, tmp_path/"budget")
    assert result["complete"] and result["sampled_pairs"] == 10
    import json
    rows = [json.loads(line) for line in (tmp_path/"budget/representation/metrics.jsonl").read_text().splitlines()]
    assert [row["effective_batch"] for row in rows] == [6, 4]
    assert rows[-1]["reference_step"] == 5
