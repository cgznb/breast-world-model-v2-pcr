"""Sequential, resumable seeds using the unchanged registered ROI32 trainer."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_registered_roi32.py"
sys.path.insert(0, str(REPO / "src"))
from symm_world.config import from_dict, load_config
from symm_world.training import stage_budget
from symm_world.utils import file_identity, read_json, write_json

STAGES = ("representation", "flow")
ACTIVE = {"reconstruction": 1.0, "jepa": 1.0, "future": .25,
          "variance": .1, "covariance": .01}
DISABLED = ("pcr", "latent_delta", "anatomy", "separation", "kinetics",
            "segmentation", "biomarker", "external_teacher", "decoded_kinetics")


def now():
    return datetime.now(timezone.utc).isoformat()


def acquire_lock(root, inherited=None):
    root.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(inherited, "a") if inherited is not None else (root / "queue.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("This study already has a running queue") from None
    return lock


def prepare(config_path, manifest, root, seeds):
    cfg = load_config(config_path)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("Seeds must be distinct integers in [0, 2**32)")
    if cfg.rollout.enabled or any(getattr(cfg.loss, k) != v for k, v in ACTIVE.items()):
        raise ValueError("Expected the agreed five-loss Stage A and no rollout stage")
    if any(getattr(cfg.loss, k) != 0 for k in DISABLED):
        raise ValueError("The removed auxiliary objectives must have zero weight")
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json"
    if not protocol_path.exists() and any((root / f"seed_{seed}").exists() for seed in seeds):
        raise ValueError("Existing seed outputs have no study protocol; use a new study directory")
    configs = []
    for seed in seeds:
        seeded = copy.deepcopy(cfg)
        seeded.training.seed = seed
        path = root / "configs" / f"seed_{seed}.json"
        if path.exists():
            if load_config(path).to_dict() != seeded.to_dict():
                raise ValueError(f"Seed configuration changed: {path}")
        elif protocol_path.exists():
            raise ValueError(f"Bound seed configuration is missing: {path}")
        else:
            write_json(path, seeded.to_dict())
        configs.append(file_identity(path))
    sources = sorted((REPO / "src/symm_world").glob("*.py")) + [RUNNER, Path(__file__).resolve()]
    protocol = {
        "schema": "registered_roi32_simple_a_multiseed_v1",
        "base_config": file_identity(config_path), "configuration": cfg.to_dict(),
        "manifest": file_identity(manifest), "seeds": list(seeds), "stages": list(STAGES),
        "seed_configs": configs, "runtime_sources": [file_identity(p) for p in sources],
        "budgets": {stage: stage_budget(cfg, stage) for stage in STAGES},
        "ordering": "one GPU; each seed runs representation then flow before the next seed",
        "initialization": "fresh per seed; each flow loads its own best representation",
        "checkpoint_selection": "existing validation protocol; report all seeds, no best-seed selection",
        "disabled_loss_policy": "zero contribution; existing runtime still computes diagnostic values",
    }
    protocol = json.loads(json.dumps(protocol))
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError("Study configuration, data or runtime changed; use a new output directory")
    else:
        write_json(protocol_path, protocol)
    return protocol


def verify_bindings(protocol):
    for identity in [protocol["base_config"], protocol["manifest"],
                     *protocol["seed_configs"], *protocol["runtime_sources"]]:
        if file_identity(identity["path"]) != identity:
            raise ValueError(f"Study input changed: {identity['path']}")


def completion(folder, cfg):
    progress_path = folder / "progress.json"
    if not progress_path.exists() or read_json(progress_path).get("status") != "complete":
        return None
    import torch
    result = {}
    for stage in STAGES:
        status = read_json(folder / stage / "status.json")
        budget = stage_budget(cfg, stage)
        if (not status.get("complete") or status["optimizer_steps"] != budget["steps"]
                or status["sampled_pairs"] != budget["samples"] or status["trained_tasks"].get("pcr")):
            raise ValueError(f"Incomplete or inconsistent {stage} result in {folder}")
        checkpoints = {}
        for kind in ("best", "last"):
            path = folder / stage / f"{kind}.pt"
            value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if (value["stage"] != stage
                    or from_dict(value["metadata"]["config"]).to_dict() != cfg.to_dict()
                    or value["metadata"]["trained_tasks"].get("pcr")
                    or not 0 < value["step"] <= budget["steps"]
                    or (kind == "last" and value["step"] != budget["steps"])):
                raise ValueError(f"Checkpoint does not match completed seed: {path}")
            checkpoints[kind] = {"path": str(path), "step": value["step"], "best_score": value["best"]}
            del value
        result[stage] = {"optimizer_steps": status["optimizer_steps"],
                         "sampled_pairs": status["sampled_pairs"], "checkpoints": checkpoints}
    return result


def snapshot(root, protocol, states, status, active_seed=None, child_pid=None, error=None):
    active = {}
    if active_seed is not None:
        for stage in STAGES:
            path = root / f"seed_{active_seed}" / stage / "progress.json"
            if path.exists():
                value = read_json(path)
                active[stage] = {k: value[k] for k in ("step", "loss", "sampled_pairs", "target_pairs",
                                                       "peak_allocated_mib") if k in value}
    value = {"status": status, "pid": os.getpid(), "updated_utc": now(),
             "seeds": states, "completed_seeds": sum(s["status"] == "complete" for s in states),
             "total_seeds": len(states), "active_seed": active_seed, "child_pid": child_pid,
             "active_progress": active, "budgets": protocol["budgets"]}
    if error is not None:
        value["error"] = error
    write_json(root / "progress.json", value)
    return value


def run_queue(root, protocol, resume=False):
    if (root / "launch.json").exists() and not resume:
        raise FileExistsError("This study was already started; pass --resume")
    states = [{"seed": seed, "status": "pending", "output": str(root / f"seed_{seed}")}
              for seed in protocol["seeds"]]
    stopped = {"requested": False}
    child = None
    current = None

    def request_stop(signum, _frame):
        stopped["requested"] = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        write_json(root / "launch.json", {"pid": os.getpid(), "started_utc": now(), "resume": resume,
                                           "seeds": protocol["seeds"], "python": sys.executable})
        for current, identity in zip(states, protocol["seed_configs"], strict=True):
            verify_bindings(protocol)
            cfg = load_config(identity["path"])
            folder = Path(current["output"])
            done = completion(folder, cfg)
            if done is not None:
                current.update(status="complete", stages=done)
                snapshot(root, protocol, states, "running")
                continue
            if stopped["requested"]:
                break
            if shutil.disk_usage(root).free < 15 * 1024**3:
                raise RuntimeError("Less than 15 GiB remains; queue paused before the next seed")
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-B", str(RUNNER), "--config", identity["path"],
                       "--manifest", protocol["manifest"]["path"], "--output", str(folder), "--stage", "all"]
            if resume:
                command.append("--resume")
            env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
                   "PYTORCH_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": "0"}
            with (folder / "controller.log").open("a") as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                if stopped["requested"] and child.poll() is None:
                    child.terminate()
                current.update(status="running", started_utc=now(), child_pid=child.pid)
                print(json.dumps({"event": "seed_started", **current}), flush=True)
                while True:
                    snapshot(root, protocol, states, "stopping" if stopped["requested"] else "running",
                             current["seed"], child.pid)
                    try:
                        code = child.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            child = None
            done = completion(folder, cfg) if code == 0 else None
            if done is not None:
                current.update(status="complete", finished_utc=now(), stages=done)
                print(json.dumps({"event": "seed_complete", "seed": current["seed"]}), flush=True)
            elif stopped["requested"]:
                current.update(status="stopped", exit_code=code)
                break
            else:
                current.update(status="failed", exit_code=code)
                raise RuntimeError(f"Seed {current['seed']} did not complete; inspect {folder / 'controller.log'}")
        status = "complete" if all(s["status"] == "complete" for s in states) else "stopped"
        result = snapshot(root, protocol, states, status)
        if status == "complete":
            write_json(root / "COMPLETE.json", result)
        return result
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        if current is not None and current["status"] != "complete":
            current["status"] = "failed"
        snapshot(root, protocol, states, "failed", error=str(error))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    with acquire_lock(root, args.lock_fd) as lock:
        protocol = prepare(args.config, args.manifest, root, args.seeds)
        if args.prepare_only:
            print(json.dumps({"prepared": True, "seeds": args.seeds, "protocol": str(root / "protocol.json")}), flush=True)
            return
        if (root / "launch.json").exists() and not args.resume:
            raise FileExistsError("This study was already started; pass --resume")
        if args.detach:
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                       "--config", str(Path(args.config).resolve()), "--manifest", str(Path(args.manifest).resolve()),
                       "--output", str(root), "--seeds", *map(str, args.seeds), "--lock-fd", str(lock.fileno())]
            if args.resume:
                command.append("--resume")
            with (root / "controller.log").open("a") as log:
                process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(lock.fileno(),))
            print(json.dumps({"pid": process.pid, "output": str(root), "seeds": args.seeds}), flush=True)
            return
        result = run_queue(root, protocol, resume=args.resume)
        print(json.dumps({"event": result["status"], "completed_seeds": result["completed_seeds"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
