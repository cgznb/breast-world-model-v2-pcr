"""Disposable real-data optimizer steps, including the largest auxiliary graph."""
from pathlib import Path
import argparse
import copy
import gc
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from symm_world.config import load_config
from symm_world.conditioning import ConditionSchema
from symm_world.data import DatasetStore, PatientBalancedSampler
from symm_world.losses import representation_loss, pair_flow_loss
from symm_world.models import RepresentationSystem, WorldModel
from symm_world.utils import seed_all, autocast_context, update_ema, write_json


def measure(cfg, store, stage, batch_size, updates):
    seed_all(cfg.training.seed, cfg.training.cpu_threads, cfg.training.strict_determinism)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rep = RepresentationSystem(cfg, ConditionSchema.fit(store.records("train")), store.statistics).cuda()
    model = rep if stage == "representation" else WorldModel(cfg, rep).cuda()
    if stage != "representation":
        del rep
    teacher = copy.deepcopy(model).eval().requires_grad_(False) if stage != "representation" else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    sampler = PatientBalancedSampler(store.records("train"), cfg.training.seed)
    elapsed = []
    for step in range(updates):
        model.train()
        opt.zero_grad(set_to_none=True)
        batch = store.pair_batch(sampler.batch(step, batch_size), cfg.encoder.token_grid, "cuda")
        # Seeds 0/1 force both representation branches and both flow directions.
        torch.manual_seed(step % 2)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with autocast_context("cuda", cfg.training.precision):
            if stage == "representation":
                loss, parts, summary = representation_loss(model, batch)
            else:
                loss, parts = pair_flow_loss(model, teacher, batch, cfg.training.auxiliary_warmup)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite profiling loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, cfg.training.grad_clip, error_if_nonfinite=True)
        opt.step()
        if stage == "representation":
            model.update_target(cfg.training.ema_decay)
            model.enqueue(summary)
        else:
            update_ema(teacher, model, cfg.training.ema_decay)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
        del loss, batch
    return {"stage": stage, "batch_size": batch_size, "passed": True,
            "seconds_per_update": sum(elapsed[2:])/len(elapsed[2:]),
            "pairs_per_second": batch_size/(sum(elapsed[2:])/len(elapsed[2:])),
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
            "gradient_norm": float(norm), "updates": updates,
            "encoder_checkpointing": cfg.encoder.checkpoint_blocks,
            "velocity_checkpointing": cfg.velocity.checkpoint_blocks,
            "flow_auxiliary_active": stage == "flow", "disposable_weights": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stages", nargs="+", default=["representation", "flow"])
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 8, 16, 24, 32, 48, 64])
    parser.add_argument("--updates", type=int, default=6)
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.cuda.set_per_process_memory_fraction(0.96)
    store = DatasetStore(args.manifest)
    store.fit_statistics()
    rows = []
    for stage in args.stages:
        for batch in args.batches:
            try:
                row = measure(cfg, store, stage, batch, args.updates)
            except torch.cuda.OutOfMemoryError:
                row = {"stage": stage, "batch_size": batch, "passed": False, "reason": "CUDA out of memory"}
            rows.append(row)
            print(json.dumps(row), flush=True)
            write_json(args.output, {"gpu": torch.cuda.get_device_name(), "results": rows})
            gc.collect()
            torch.cuda.empty_cache()
            if not row["passed"]:
                break


if __name__ == "__main__":
    main()
