"""Locked foreground or detached execution of the registered ROI32 experiment."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from symm_world.utils import file_identity, read_json, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["all", "representation", "flow"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--lock-fd", type=int)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(args.lock_fd, "a") if args.lock_fd is not None else (root/"run.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This output already has a running controller")
    if args.detach:
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                   "--config", str(Path(args.config).resolve()), "--manifest", str(Path(args.manifest).resolve()),
                   "--output", str(root), "--stage", args.stage, "--lock-fd", str(lock.fileno())]
        if args.resume:
            command.append("--resume")
        if args.stop_after:
            command.extend(["--stop-after", str(args.stop_after)])
        env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
               "PYTORCH_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": "0"}
        with (root/"controller.log").open("a") as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(lock.fileno(),))
        print(json.dumps({"pid": process.pid, "output": str(root), "log": str(root/"controller.log")}))
        return
    from symm_world.cli import train_all
    from symm_world.config import load_config
    from symm_world.data import DatasetStore
    from symm_world.training import train_stage
    import torch
    cfg = load_config(args.config)
    if shutil.disk_usage(root).free < 15*1024**3:
        raise RuntimeError("Less than 15 GiB remains for checkpoints")
    binding = {"config": file_identity(args.config), "manifest": file_identity(args.manifest),
               "sources": [file_identity(p) for p in sorted((REPO/"src/symm_world").glob("*.py"))]
                           + [file_identity(__file__)]}
    binding_path = root/"runtime_binding.json"
    if binding_path.exists() and read_json(binding_path) != binding:
        raise ValueError("Runtime sources/config/manifest changed; use an isolated new run")
    write_json(binding_path, binding)
    launch = {"pid": os.getpid(), "started_utc": now(), "config": str(Path(args.config).resolve()),
              "manifest": str(Path(args.manifest).resolve()), "output": str(root), "resume": args.resume,
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    write_json(root/"launch.json", launch)
    write_json(root/"progress.json", {**launch, "status": "running", "updated_utc": now()})
    print(json.dumps({"event": "started", **launch}), flush=True)
    try:
        store = DatasetStore(args.manifest)
        if args.stage == "all":
            result = train_all(cfg, store, root, resume=args.resume, stop_after=args.stop_after)
        else:
            result = [train_stage(args.stage, cfg, store, root, resume=args.resume, stop_after=args.stop_after)]
        status = "complete" if all(r["complete"] for r in result) else "stopped"
        write_json(root/"progress.json", {"pid": os.getpid(), "status": status, "stages": result, "updated_utc": now()})
        print(json.dumps({"event": status, "stages": result}), flush=True)
    except BaseException as exc:
        write_json(root/"progress.json", {"pid": os.getpid(), "status": "failed", "error": str(exc), "updated_utc": now()})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
