"""Frozen World V2 forecasts for a 30-case review and existing pCR classifiers."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
MEWM = REPO / "vendor/mewm-ispy2"
PCR = REPO / "pcr"
RUN = REPO / "runs/registered_roi32_20260919"
OUTPUT = RUN / "analysis_30_pcr_v1v4_20260919"
PCR_BASE = PCR / "results/registered_three_phase_roi32_pcr_v1_20260917"
sys.path[:0] = [str(REPO / "src"), str(PCR), str(MEWM)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import SimpleITK as sitk
import torch

from mewm_ispy2.registered_three_phase_data import ThreePhaseCrops, load_config
from mewm_ispy2.registered_roi32_data import visit_filename
from src.registered_three_phase_generated_pcr import source_read_guard
from src.registered_three_phase_pcr import build_volume
from scripts.run_first_post_pcr import load_frozen_pillar, pillar_forward
from symm_world.codec import load_codec
from symm_world.data import DatasetStore
from symm_world.training import load_inference
from symm_world.utils import file_identity, read_json, write_json

SEED, DRAWS, STEPS = 20260919, 4, 25
SHAPE = (3, 32, 128, 128)
PHASES = ("Precontrast", "First-post", "Late")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def progress(output, mode, stage, **fields):
    record = dict(updated_utc=now(), pid=os.getpid(), mode=mode, stage=stage, **fields)
    write_json(output / f"{mode}_progress.json", record)
    print(json.dumps(record), flush=True)


def configure():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def verify_runtime():
    binding = read_json(RUN / "runtime_binding.json")
    identities = [binding["config"], binding["manifest"], *binding["sources"]]
    for identity in identities:
        if file_identity(identity["path"]) != identity:
            raise ValueError(f"World runtime changed: {identity['path']}")
    if read_json(RUN / "progress.json")["status"] != "complete":
        raise ValueError("World training must be complete")
    return len(identities)


def frozen_json(path, data):
    if path.exists():
        if read_json(path) != data:
            raise ValueError(f"Frozen analysis protocol differs: {path}")
    else:
        write_json(path, data)


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temp, **arrays)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def save_tensor(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value.detach().cpu(), temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def select_review(store):
    groups = {}
    for a, b in itertools.combinations(range(4), 2):
        key = (f"T{a}", f"T{b}")
        groups[key] = sorted([p for p in store.records("val") if
            (p["conditions"]["stage_i"], p["conditions"]["stage_j"]) == key], key=lambda p: p["id"])
    rng, used, cases = random.Random(SEED), set(), []
    for key in sorted(groups):
        rng.shuffle(groups[key])
    for repetition in range(5):
        for key in sorted(groups):
            pair = next(p for p in groups[key] if p["patient_id"] not in used)
            used.add(pair["patient_id"])
            cases.append({"case": f"Case{len(cases) + 1:02d}", "pair_id": pair["id"],
                          "patient_id": pair["patient_id"], "source": pair["source"], "target": pair["target"],
                          "transition": " -> ".join(key), "days": pair["conditions"]["delta_days"],
                          "noise_seed": SEED + 10000 + len(cases)})
    if len(used) != 30 or set(Counter(c["transition"] for c in cases).values()) != {5}:
        raise ValueError("Invalid 30-patient transition balance")
    return cases


def pcr_routes(store):
    cohort = read_json(PCR_BASE / "cohort.json")
    patients = sorted(cohort["split"]["val"])
    if set(patients) != {v["patient_id"] for v in store.views.values() if v["split"] == "val"}:
        raise ValueError("World validation and historical pCR populations differ")
    available = {pid: {v["timepoint"] for v in cohort["visits"] if v["patient_id"] == pid} for pid in patients}
    pairs = {(p["patient_id"], store.views[p["source"]]["visit"], store.views[p["target"]]["visit"]): p
             for p in store.records("val")}
    routes, counts = [], [0] * 4
    for pid in patients:
        length = 0
        while length < 4 and length in available[pid]:
            counts[length] += 1
            length += 1
        if length < 1:
            raise ValueError("Missing real T0")
        for tp in range(1, length):
            routes.append({"patient_id": pid, "timepoint": tp, "key": f"{pid}_T{tp}",
                           "direct_pair": pairs[pid, "T0", f"T{tp}"]["id"],
                           "adjacent_pair": pairs[pid, f"T{tp - 1}", f"T{tp}"]["id"],
                           "noise_seed": SEED + 40000 + len(routes)})
    if (len(patients), len(routes), counts) != (102, 276, [102, 100, 92, 84]):
        raise ValueError("Historical contiguous-prefix population changed")
    return routes, counts


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    verify_runtime()
    store = DatasetStore(REPO / "data/registered_roi32/manifest.json")
    cases = select_review(store)
    routes, counts = pcr_routes(store)
    protocol = {"schema": "world_v2_30case_pcr_review_v1", "checkpoint": file_identity(RUN / "flow/best.pt"),
                "codec": file_identity(REPO / "data/registered_roi32/codec.pt"),
                "manifest": file_identity(store.path), "generator_weights": "EMA step 4000",
                "selection_seed": SEED, "sampler": "heun", "steps": STEPS, "draws": DRAWS,
                "generation_precision": "bf16", "solver_state_precision": "float32", "sample_batch": DRAWS,
                "decoder_precision": "float32", "pillar_precision": "float32; TF32 disabled; batch 1",
                "solver_policy": "Heun25 matches the previous pCR generation protocol; World checkpoint selection used Heun20",
                "review_selection": "30 distinct validation patients, five per forward transition, no quality selection",
                "pcr_real_feature_root": str(PCR_BASE / "embeddings/real"),
                "pcr_population": "Historical 102-patient test cohort also used for World validation/selection",
                "independent_test": False, "pcr_optimizer_updates": 0,
                "pcr_prefix_counts": counts, "pcr_future_visits": len(routes),
                "pcr_support": "Fixed real T0 per-phase foreground for every generated input policy",
                "pcr_ensemble": "Mean of four complete-sequence pCR probabilities within each fold model",
                "pcr_time_policy": "Retrospective recorded stages and intervals; no future images in direct/rollout inference",
                "rollout_policy": "Inference composition only; World Stage C was disabled, not trained",
                "cases": cases, "routes": routes}
    frozen_json(output / "protocol.json", protocol)
    return protocol


class SourceStore(DatasetStore):
    allowed = None

    def read_raw(self, view_id):
        if self.allowed is not None and view_id not in self.allowed:
            raise ValueError("Undeclared real latent read during forecasting")
        return super().read_raw(view_id)

    def read_aux(self, *args, **kwargs):
        raise ValueError("pCR outcomes and target auxiliary labels are forbidden during generation")


class ForecastContext:
    def __init__(self):
        self.model, self.metadata = load_inference(RUN / "flow/best.pt", "cuda:0")
        self.codec = load_codec(REPO / "data/registered_roi32/codec.pt", "cuda:0")
        self.store = SourceStore(REPO / "data/registered_roi32/manifest.json")
        self.store.set_statistics(self.metadata["statistics"])
        self.pairs = {p["id"]: p for p in self.store.pairs}
        config, baseline = load_config(MEWM / "configs/registered_three_phase_roi32_v1.yaml")
        self.inventory = read_json(Path(config["output_dir"]) / "inventory.json")
        self.crops = ThreePhaseCrops(baseline, self.inventory, split="val")
        self.crop_indices = {r["visit_id"]: i for i, r in enumerate(self.crops.records)}
        self.visits = {r["visit_id"]: r for r in self.inventory["visits"]}
        self.data_root = Path(baseline["output_dir"]) / "data"
        self.denied = {s["path"] for row in self.inventory["visits"] for s in row["phase_sources"]}
        self.denied |= {r["model_mask_path"] for r in self.inventory["visits"] if r["model_mask_path"]}
        self.prefixes = [Path(config["output_dir"]) / "latents/raw", self.data_root / "images",
                         self.data_root / "patients", Path("/path/to/research/Datasets"),
                         Path(config["output_dir"]) / "source_supplement"]
        stats = self.metadata["statistics"]
        self.mean = torch.tensor(stats["mean"], device="cuda:0").reshape(1, 24, 1, 1, 1)
        self.std = torch.tensor(stats["std"], device="cuda:0").reshape(1, 24, 1, 1, 1)

    @contextmanager
    def source_guard(self, image_ids, latent_ids):
        allowed = []
        for view_id in image_ids:
            v = self.visits[view_id]
            allowed += [self.data_root / "images" / visit_filename(view_id),
                        self.data_root / "patients" / (v["patient_id"] + ".npz"),
                        v["metadata_source"]["path"], v["crop_report"]["path"],
                        *[s["path"] for s in v["phase_sources"]]]
        for view_id in latent_ids:
            path = self.store.resolve(self.store.views[view_id]["latent"])
            expected = self.metadata["asset_signatures"][f"{view_id}/latent"]
            stat = path.stat()
            if expected != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
                raise ValueError("Cached real latent changed since training")
            allowed.append(path)
        self.store.allowed = set(latent_ids)
        try:
            with source_read_guard(allowed, self.denied, self.prefixes) as audit:
                yield audit
        finally:
            self.store.allowed = None

    def real(self, view_id):
        return self.crops[self.crop_indices[view_id]]

    def source(self, view_id):
        return self.store.normalize(self.store.read_raw(view_id))[None].to("cuda:0")

    @torch.inference_mode()
    def forecast(self, source, pair, seed):
        if len(source) == 1:
            source = source.repeat(DRAWS, 1, 1, 1, 1)
        if source.shape != (DRAWS, 24, 8, 32, 32):
            raise ValueError("Expected four paired noise trajectories")
        generator = torch.Generator(device="cuda:0").manual_seed(seed)
        noise = torch.cat([torch.randn(source[:1].shape, device="cuda:0", generator=generator) for _ in range(DRAWS)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = self.model.sample(source, [pair["conditions"]] * DRAWS, noise=noise, steps=STEPS, method="heun")
        samples = self.decode(latent)
        return latent.float(), samples

    @torch.inference_mode()
    def decode(self, normalized):
        images = [self.codec.decode(z * self.std + self.mean)[0].float().cpu().numpy() for z in normalized.split(1)]
        result = np.stack(images)
        if result.shape[1:] != SHAPE or not np.isfinite(result).all():
            raise ValueError("Invalid generated three-phase MRI")
        return result


def completed(path, protocol_path):
    if not path.exists():
        return False
    record = read_json(path)
    if record["protocol"] != file_identity(protocol_path):
        raise ValueError("Cached case belongs to a different protocol")
    for artifact in record["artifacts"]:
        if file_identity(artifact["path"]) != artifact:
            raise ValueError("Cached case artifact changed")
    return record


def review_inference(output, protocol, limit):
    root = output / "review"
    pending = [case for case in protocol["cases"] if not completed(root / case["case"] / "complete.json", output / "protocol.json")]
    if not pending:
        return
    context = ForecastContext()
    started = time.monotonic()
    for index, case in enumerate(pending[:limit] if limit else pending):
        folder = root / case["case"]
        pair = context.pairs[case["pair_id"]]
        with context.source_guard([case["source"]], [case["source"]]) as audit:
            source = context.source(case["source"])
            real_source = context.real(case["source"])
            latent, samples = context.forecast(source, pair, case["noise_seed"])
        target = context.real(case["target"])
        target_latent = context.source(case["target"])
        target_vq = context.decode(target_latent)[0]
        source_vq = context.decode(source)[0]
        arrays = {"source": real_source["image"].numpy(), "target": target["image"].numpy(),
                  "samples": samples, "mean": samples.mean(0), "source_vq": source_vq, "target_vq": target_vq,
                  "normalized_latents": latent.cpu().numpy(), "mask": real_source["mask"][0].numpy().astype(bool),
                  "source_valid": real_source["valid"].numpy(), "target_valid": target["valid"].numpy(),
                  "source_coverage": real_source["coverage"].numpy(), "target_coverage": target["coverage"].numpy()}
        if not torch.equal(real_source["mask"], target["mask"]):
            raise ValueError("Longitudinal fixed mask differs")
        save_npz(folder / "volumes.npz", **arrays)
        write_json(folder / "complete.json", {"protocol": file_identity(output / "protocol.json"), "case": case,
                   "artifacts": [file_identity(folder / "volumes.npz")], "source_only_audit_passed": True,
                   "source_files_opened": len(audit["opened"]), "target_loaded_after_forecast": True,
                   "source_clipped": bool(real_source["largest_clipped"]), "target_clipped": bool(target["largest_clipped"]),
                   "completed_utc": now()})
        progress(output, "review", "generating", completed=30 - len(pending) + index + 1, total=30,
                 elapsed_seconds=time.monotonic() - started)
    del context
    gc.collect()
    torch.cuda.empty_cache()
    if all(completed(root / c["case"] / "complete.json", output / "protocol.json") for c in protocol["cases"]):
        write_json(root / "INFERENCE_COMPLETE.json", {"patients": 30, "samples": 120, "completed_utc": now()})


def canonical(mode, tp):
    return "direct" if tp == 1 else mode


def pcr_folder(output, mode, route):
    return output / "pcr" / canonical(mode, route["timepoint"]) / route["key"]


def feature_path(output, mode, route, draw):
    return output / "pcr/embeddings" / mode / f"draw_{draw}" / route["patient_id"] / (route["key"] + ".pt")


def encode_samples(pillar, samples, foreground, normalization):
    result = []
    with ThreadPoolExecutor(max_workers=min(4, len(samples))) as pool:
        volumes = pool.map(lambda image: build_volume(image, foreground, normalization)[0], samples)
        for volume in volumes:
            with torch.inference_mode():
                result.append(pillar_forward(pillar, volume[None].to("cuda:0"))[0])
            del volume
    return result


def pcr_tasks(protocol):
    return [(mode, r) for r in protocol["routes"] for mode in (("direct",) if r["timepoint"] == 1 else ("direct", "rollout", "previous_real"))]


def finish_pcr(output, protocol):
    tasks = pcr_tasks(protocol)
    if all(completed(pcr_folder(output, m, r) / "complete.json", output / "protocol.json") for m, r in tasks):
        write_json(output / "pcr/FEATURES_COMPLETE.json", {"cases": len(tasks), "unique_generated_volumes": 4 * len(tasks),
                   "feature_vectors_including_shared_T1": 3 * len(protocol["routes"]) * 4,
                   "target_image_or_latent_reads": 0, "optimizer_updates": 0, "completed_utc": now()})


def pcr_inference(output, protocol, limit, worker=0, workers=1):
    tasks = pcr_tasks(protocol)
    patients = sorted({r["patient_id"] for r in protocol["routes"]})
    assigned = set(patients[worker::workers])
    tasks = [(m, r) for m, r in tasks if r["patient_id"] in assigned]
    progress_mode = "pcr" if workers == 1 else f"pcr_worker_{worker}"
    pending = [(mode, r) for mode, r in tasks if not completed(pcr_folder(output, mode, r) / "complete.json", output / "protocol.json")]
    if not pending:
        if workers == 1:
            finish_pcr(output, protocol)
        progress(output, progress_mode, "worker_complete", completed=len(tasks), total=len(tasks))
        return
    context, pillar = ForecastContext(), load_frozen_pillar()
    started, done, current_pid = time.monotonic(), len(tasks) - len(pending), None
    t0, foreground, replay_error = None, None, None
    for index, (mode, route) in enumerate(pending[:limit] if limit else pending):
        pid, tp = route["patient_id"], route["timepoint"]
        t0_id = f"{pid}:T0"
        pair = context.pairs[route["direct_pair"] if mode == "direct" else route["adjacent_pair"]]
        allowed_latents = [t0_id] + ([pair["source"]] if mode == "previous_real" else [])
        with context.source_guard([t0_id], allowed_latents) as audit:
            if pid != current_pid:
                t0 = context.source(t0_id)
                item = context.real(t0_id)
                foreground = item["valid"].numpy()
                replay = encode_samples(pillar, item["image"].numpy()[None], foreground, context.crops.baseline.normalization)[0]
                expected = torch.load(PCR_BASE / "embeddings/real" / pid / f"{pid}_T0.pt", map_location="cpu", weights_only=True)
                replay_error = float((replay - expected).abs().max())
                if not torch.allclose(replay, expected, atol=1e-6, rtol=1e-6):
                    raise ValueError(f"Original real T0 Pillar features do not replay: {replay_error}")
                current_pid = pid
            parent = None
            if mode == "direct" or tp == 1:
                source = t0
            elif mode == "previous_real":
                source = context.source(pair["source"])
            else:
                previous = next(r for r in protocol["routes"] if r["patient_id"] == pid and r["timepoint"] == tp - 1)
                parent = pcr_folder(output, "rollout", previous) / "volumes.npz"
                if not completed(parent.with_name("complete.json"), output / "protocol.json"):
                    raise ValueError("Missing matching-draw rollout predecessor")
                with np.load(parent, allow_pickle=False) as cached:
                    source = torch.from_numpy(cached["normalized_latents"].copy()).to("cuda:0")
            latent, samples = context.forecast(source, pair, route["noise_seed"])
            features = encode_samples(pillar, samples, foreground, context.crops.baseline.normalization)
        folder = pcr_folder(output, mode, route)
        save_npz(folder / "volumes.npz", samples=samples, normalized_latents=latent.cpu().numpy(), source_foreground=foreground)
        artifacts = [file_identity(folder / "volumes.npz")]
        for save_mode in (("direct", "rollout", "previous_real") if tp == 1 else (mode,)):
            for draw, feature in enumerate(features):
                path = feature_path(output, save_mode, route, draw)
                save_tensor(path, feature)
                artifacts.append(file_identity(path))
        write_json(folder / "complete.json", {"protocol": file_identity(output / "protocol.json"), "route": route,
                   "mode": mode, "artifacts": artifacts, "real_T0_pillar_replay_max_error": replay_error,
                   "source_read_audit_passed": True, "target_image_or_latent_reads": 0,
                   "declared_real_latents": sorted(set(allowed_latents)), "source_files_opened": len(audit["opened"]),
                   "parent_generated_latent": file_identity(parent) if parent else None, "completed_utc": now()})
        done += 1
        elapsed = time.monotonic() - started
        progress(output, progress_mode, "generating_and_encoding", completed=done, total=len(tasks),
                 mode_current=mode, elapsed_seconds=elapsed,
                 estimated_remaining_seconds=elapsed / (index + 1) * (len(tasks) - done))
        del source, latent, samples, features
    del context, pillar
    gc.collect()
    torch.cuda.empty_cache()
    progress(output, progress_mode, "worker_complete", completed=done, total=len(tasks))
    if workers == 1:
        finish_pcr(output, protocol)


def pcr_worker(output, protocol, limit, worker, workers):
    configure()
    pcr_inference(output, protocol, limit, worker, workers)


def parallel_pcr(output, protocol, limit, workers):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=pcr_worker, args=(output, protocol, limit, i, workers)) for i in range(workers)]
    tasks = pcr_tasks(protocol)
    initial = sum(bool(completed(pcr_folder(output, m, r) / "complete.json", output / "protocol.json")) for m, r in tasks)
    started = time.monotonic()
    try:
        for process in processes:
            process.start()
        write_json(output / "pcr_workers.json", {"controller_pid": os.getpid(), "worker_pids": [p.pid for p in processes],
                   "workers": workers, "partition": "Whole patients in sorted round-robin order; trajectories stay inside one worker",
                   "per_worker_sample_batch": DRAWS, "pillar_batch": 1, "started_utc": now()})
        while any(p.is_alive() for p in processes):
            for process in processes:
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"pCR worker exited with {process.exitcode}")
            done = sum((pcr_folder(output, m, r) / "complete.json").is_file() for m, r in tasks)
            elapsed = time.monotonic() - started
            progress(output, "pcr", "parallel_generation_and_encoding", completed=done, total=len(tasks),
                     workers=workers, elapsed_seconds=elapsed,
                     estimated_remaining_seconds=elapsed / max(done - initial, 1) * (len(tasks) - done))
            time.sleep(15)
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"pCR worker exited with {process.exitcode}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join()
    finish_pcr(output, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mode", choices=("prepare", "review", "pcr"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.mode != "pcr" and args.workers != 1:
        parser.error("Patient workers only apply to pCR generation")
    protocol = prepare(args.output)
    if args.mode == "prepare":
        print(json.dumps({"output": str(args.output), "review_patients": 30, "pcr_routes": 276}), flush=True)
        return
    if args.detach:
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--mode", args.mode, "--output", str(args.output)]
        if args.limit:
            command += ["--limit", str(args.limit)]
        command += ["--workers", str(args.workers)]
        with (args.output / f"{args.mode}.log").open("a") as log:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
        write_json(args.output / f"{args.mode}_launch.json", {"pid": process.pid, "started_utc": now(), "command": command})
        print(json.dumps({"pid": process.pid, "mode": args.mode, "log": str(args.output / f"{args.mode}.log")}), flush=True)
        return
    configure()
    with (args.output / "gpu_inference.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.mode == "pcr" and args.workers > 1:
                parallel_pcr(args.output, protocol, args.limit, args.workers)
            else:
                (review_inference if args.mode == "review" else pcr_inference)(args.output, protocol, args.limit)
            verify_runtime()
            progress(args.output, args.mode, "finished_batch")
        except BaseException as exc:
            progress(args.output, args.mode, "failed", error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == "__main__":
    main()
