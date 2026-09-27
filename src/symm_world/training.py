"""Three explicit optimization stages with atomic, deterministic resume."""
from __future__ import annotations
import copy
import json
import math
import signal
from pathlib import Path
import time
import torch
from .config import from_dict
from .conditioning import ConditionSchema
from .data import DatasetStore, PatientBalancedSampler, PatientBalancedTripletSampler
from .models import RepresentationSystem, WorldModel
from .losses import representation_loss, pair_flow_loss, rollout_loss
from .evaluation import evaluate_model
from .utils import (read_json,write_json,seed_all,autocast_context,save_checkpoint,load_checkpoint,
                    update_ema,file_digest)


def assets_signature(store):
    result = {}
    for key,v in store.views.items():
        for kind in ("latent","auxiliary"):
            if v.get(kind):
                p = store.resolve(v[kind]); s = p.stat()
                result[f"{key}/{kind}"] = {"size":s.st_size,"mtime_ns":s.st_mtime_ns}
                declared = v.get(kind+"_sha256")
                if declared and file_digest(p) != declared:
                    raise ValueError(f"Asset checksum mismatch: {key}/{kind}")
    return result


def task_support(store, cfg):
    source_views = {p["source"] for p in store.records("train")}
    visit_views = source_views | {p["target"] for p in store.records("train")}
    counts = {"pcr":0,"kinetics":0,"segmentation":0,"biomarker":0,"external_teacher":0}
    for view in visit_views:
        aux = store.read_aux(view,cfg.encoder.token_grid)
        for loss,key in (("kinetics","kinetics"),("segmentation","segmentation"),
                         ("biomarker","biomarkers"),("external_teacher","external_tokens")):
            counts[loss] += int(key in aux)
        counts["pcr"] += int(view in source_views and "pcr" in aux)
    return counts


def initialize_metadata(cfg, store, root):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    path = root/"metadata.json"
    signatures = assets_signature(store)
    if path.exists():
        meta = read_json(path)
        if from_dict(meta["config"]).to_dict() != cfg.to_dict() or meta["manifest_identity"] != store.identity:
            raise ValueError("Run config/manifest differs from its saved contract; use a new output directory")
        if meta["asset_signatures"] != signatures:
            raise ValueError("Cached latent/auxiliary files changed since run creation")
        store.set_statistics(meta["statistics"])
        return meta
    stats = store.fit_statistics()
    schema = ConditionSchema.fit(store.records("train"))
    support = task_support(store,cfg)
    if cfg.loss.external_teacher and not support["external_teacher"]:
        raise ValueError("external_teacher loss enabled without exported training teacher features")
    meta = {"schema":"symm_world_run_v2","config":cfg.to_dict(),
            "manifest_identity":store.identity,"statistics":stats,"condition_schema":schema.to_dict(),
            "asset_signatures":signatures,"task_support":support,
            "trained_tasks":{k:False for k in support},
            "supervised_updates":{k:0 for k in support},
            "audit":store.audit(scan_arrays=False,rollout_cfg=cfg.rollout)}
    write_json(path,meta)
    return meta


def make_representation(cfg,meta,device):
    return RepresentationSystem(cfg,ConditionSchema(**meta["condition_schema"]),meta["statistics"]).to(device)


def representation_checkpoint(root):
    root = Path(root)/"representation"
    path = root/"best.pt"
    return path if path.exists() else root/"last.pt"


def build_stage(stage,cfg,meta,root,device):
    rep = make_representation(cfg,meta,device)
    if stage == "representation":
        return rep,None
    p = representation_checkpoint(root)
    if not p.exists():
        raise FileNotFoundError("Train representation stage first; no pretrained patient-state encoder is silently fabricated")
    value = load_checkpoint(p)
    rep.load_state_dict(value["model"],strict=True)
    meta["trained_tasks"] = copy.deepcopy(value["metadata"]["trained_tasks"])
    meta["supervised_updates"] = copy.deepcopy(value["metadata"].get("supervised_updates", {}))
    model = WorldModel(cfg,rep).to(device)
    del rep
    if stage == "rollout":
        p = Path(root)/"flow"/"best.pt"
        if not p.exists():
            p = Path(root)/"flow"/"last.pt"
        value = load_checkpoint(p)
        model.load_state_dict(value.get("teacher") or value["model"],strict=True)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    return model,teacher


def rng_state():
    return {"cpu":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    torch.set_rng_state(value["cpu"])
    if value["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["cuda"])


def representation_validation(model,store,cfg):
    from .evaluation import balanced_records
    records = balanced_records(store.records("val"), cfg.training.validation_pairs)
    if not records:
        raise ValueError("A validation split is required for checkpoint selection")
    device = next(model.parameters()).device
    old = model.training; model.eval()
    devices = [device.index or 0] if device.type=="cuda" else []
    values = []
    with torch.random.fork_rng(devices=devices),torch.no_grad():
        torch.manual_seed(cfg.training.seed+12345)
        for p in records:
            batch = store.pair_batch([p],cfg.encoder.token_grid,device)
            with autocast_context(device, cfg.training.precision):
                _,parts,_ = representation_loss(model,batch)
            values.append(parts["reconstruction"]+parts["future"]+parts["jepa"])
    model.train(old)
    return {"selection_score":sum(values)/len(values),"selection_metric":"representation_reconstruction_future_jepa",
            "split":"val","pairs":len(values),"independent_test":False}


def learning_rate(step,total,warmup):
    warmup = min(warmup,max(1,total//5))
    if step < warmup:
        return (step+1)/max(warmup,1)
    progress = (step-warmup)/max(1,total-warmup)
    return .01+.99*.5*(1+math.cos(math.pi*min(progress,1)))


def stage_budget(cfg, stage):
    training = cfg.training
    batch = training.stage_batches.get(stage, training.batch_size)
    effective = batch * training.accumulation
    reference_batch = training.reference_batch_size or effective
    reference_steps = getattr(training, stage + "_steps")
    samples = reference_steps * reference_batch
    return {"batch": batch, "effective": effective, "reference_batch": reference_batch,
            "reference_steps": reference_steps, "samples": samples,
            "steps": math.ceil(samples/effective),
            "validate_every": max(1, math.ceil(training.validate_every*reference_batch/effective)),
            "checkpoint_every": max(1, math.ceil(training.checkpoint_every*reference_batch/effective)),
            # Preserve the fraction of batches receiving auxiliary supervision.
            "auxiliary_every": training.auxiliary_every}


def train_stage(stage,cfg,store,root,*,resume=False,stop_after=None,codec=None):
    if stage not in {"representation","flow","rollout"}:
        raise ValueError("Unknown training stage")
    seed_all(cfg.training.seed,cfg.training.cpu_threads,cfg.training.strict_determinism)
    device = torch.device(cfg.training.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use configs/smoke.yaml for CPU verification")
    meta = initialize_metadata(cfg,store,root)
    if cfg.training.preload_latents:
        store.preload()
    triples,rejected = store.triplets(cfg.rollout)
    if stage=="rollout" and (not cfg.rollout.enabled or not triples):
        raise ValueError(f"Rollout training is disabled or no eligible triplets exist. Rejections: {rejected}")
    if cfg.loss.decoded_kinetics and codec is None:
        raise ValueError("decoded_kinetics is enabled, but no matching codec was supplied")
    budget = stage_budget(cfg, stage)
    total = budget["steps"]
    if total < 1:
        raise ValueError(f"No optimizer steps configured for {stage}")
    folder = Path(root)/stage; folder.mkdir(parents=True,exist_ok=True)
    last = folder/"last.pt"
    if last.exists() and not resume:
        raise FileExistsError("Existing run: use --resume or a new output directory")
    model,teacher = build_stage(stage,cfg,meta,root,device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    lr = cfg.training.representation_lr if stage=="representation" else cfg.training.lr
    optimizer = torch.optim.AdamW(parameters,lr=lr,weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:learning_rate(
        s*budget["effective"]/budget["reference_batch"], budget["reference_steps"], cfg.training.warmup_steps))
    step,best = 0,None
    if resume and last.exists():
        saved = load_checkpoint(last)
        if (saved["stage"] != stage or from_dict(saved["metadata"]["config"]).to_dict() != cfg.to_dict()
                or saved["manifest_identity"] != store.identity):
            raise ValueError("Resume checkpoint contract mismatch")
        model.load_state_dict(saved["model"],strict=True)
        if teacher is not None:
            teacher.load_state_dict(saved["teacher"],strict=True)
        optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        step,best = saved["step"],saved["best"]
        meta["trained_tasks"] = copy.deepcopy(saved["metadata"]["trained_tasks"])
        meta["supervised_updates"] = copy.deepcopy(saved["metadata"].get("supervised_updates", {}))
        restore_rng(saved["rng"])
    sampler = PatientBalancedSampler(store.records("train"),cfg.training.seed)
    triplet_sampler = PatientBalancedTripletSampler(triples,cfg.training.seed) if stage=="rollout" else None
    limit = min(total,step+stop_after) if stop_after is not None else total
    if stop_after is not None and stop_after < 1:
        raise ValueError("--stop-after must be positive")
    stopping = {"requested":False}
    handlers = {}
    for sig in (signal.SIGTERM,signal.SIGINT):
        handlers[sig] = signal.signal(sig,lambda *_:stopping.update(requested=True))
    history = folder/"metrics.jsonl"
    batch_size = budget["batch"]
    def save(path):
        save_checkpoint(path,{"schema":"symm_world_checkpoint_v2","stage":stage,"step":step,"best":best,
                              "manifest_identity":store.identity,"metadata":meta,
                              "model":model.state_dict(),"teacher":teacher.state_dict() if teacher is not None else None,
                              "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"rng":rng_state()})
    try:
        while step < limit and not stopping["requested"]:
            begun = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            aggregate, summaries = {},[]
            update_samples = min(budget["effective"], budget["samples"] - step*budget["effective"])
            for micro in range(cfg.training.accumulation):
                micro_samples = min(batch_size, update_samples - micro*batch_size)
                if micro_samples <= 0:
                    break
                counter = step*cfg.training.accumulation+micro
                pairs = sampler.batch(counter,micro_samples)
                batch = store.pair_batch(pairs,cfg.encoder.token_grid,device)
                with autocast_context(device,cfg.training.precision):
                    if stage=="representation":
                        loss,parts,summary = representation_loss(model,batch)
                        summaries.append(summary)
                    else:
                        # Rollout fine-tuning retains velocity supervision.
                        absolute_step = step*budget["effective"]/budget["reference_batch"]
                        absolute_step += cfg.training.flow_steps if stage=="rollout" else 0
                        active = absolute_step >= cfg.training.auxiliary_warmup and step%budget["auxiliary_every"]==0
                        loss,parts = pair_flow_loss(model,teacher,batch,absolute_step,codec=codec,auxiliary_active=active)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                weight = micro_samples/update_samples
                (loss*weight).backward()
                parts["loss"] = float(loss.detach())
                for k,v in parts.items():
                    aggregate[k] = aggregate.get(k,0)+v*weight
                del loss,batch
            if stage=="rollout" and step%cfg.rollout.every==0:
                # Deterministic triplet draw. Independent MC noises are not paired
                # across direct/composed distributions.
                triplet = triplet_sampler.sample(step // cfg.rollout.every)
                with autocast_context(device,cfg.training.precision):
                    reg,parts = rollout_loss(model,teacher,store,triplet,device=device)
                if not torch.isfinite(reg):
                    raise FloatingPointError("Nonfinite rollout loss")
                reg.backward()
                aggregate.update(parts)
                aggregate["loss"] += float(reg.detach())
            norm = torch.nn.utils.clip_grad_norm_(parameters,cfg.training.grad_clip,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            decay = cfg.training.ema_decay**(update_samples/budget["reference_batch"])
            if stage=="representation":
                model.update_target(decay)
                model.enqueue(torch.cat(summaries))
                for task in meta["trained_tasks"]:
                    if aggregate.get("label_count/"+task, 0) > 0 and getattr(cfg.loss, task) > 0:
                        meta["supervised_updates"][task] = meta["supervised_updates"].get(task,0)+1
                        meta["trained_tasks"][task] = True
            else:
                update_ema(teacher,model,decay)
            aggregate.update(stage=stage,step=step,gradient_norm=float(norm),
                             lr=optimizer.param_groups[0]["lr"],seconds=time.perf_counter()-begun,
                             batch_size=batch_size, effective_batch=update_samples,
                             sampled_pairs=min(step*budget["effective"], budget["samples"]),
                             target_pairs=budget["samples"],
                             reference_step=min(step*budget["effective"], budget["samples"])/budget["reference_batch"])
            if device.type == "cuda":
                aggregate.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                                 peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2)
            with history.open("a",encoding="utf-8") as h:
                h.write(json.dumps(aggregate,allow_nan=False)+"\n")
            if step <= 2 or step%cfg.training.log_every==0:
                print(json.dumps(aggregate,allow_nan=False),flush=True)
                write_json(folder/"progress.json",aggregate)
            # Avoid extra validation on a mere partial stop: exact resume should
            # match uninterrupted execution's optimizer and RNG trajectory.
            if step%budget["validate_every"]==0 or step==total:
                if stage=="representation":
                    result = representation_validation(model,store,cfg)
                    score = result["selection_score"]
                else:
                    result = evaluate_model(teacher,store,"val",meta,limit=cfg.training.validation_pairs,seed=cfg.training.seed)
                    score = result["metrics"]["latent_mean_mae"]["patient_macro_mean"]
                write_json(folder/f"validation_{step:07d}.json",result)
                if best is None or score < best:
                    best = score
                    # Save optimizer-compatible student as best checkpoint, and
                    # the evaluated EMA teacher in the same file. Inference uses
                    # teacher, representation initialization uses student.
                    save(folder/"best.pt")
            if step==1 or step%budget["checkpoint_every"]==0 or step==limit or stopping["requested"]:
                save(last)
        if not last.exists() or step==limit:
            save(last)
    finally:
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    result = {"stage":stage,"optimizer_steps":step,"configured_steps":total,"complete":step==total,
              "reference_steps":budget["reference_steps"], "target_pairs":budget["samples"],
              "sampled_pairs":min(step*budget["effective"], budget["samples"]),
              "best_validation_score":best,"checkpoint":str(last),"trained_tasks":meta["trained_tasks"]}
    write_json(folder/"status.json",result)
    return result


def load_inference(path,device="cpu"):
    value = load_checkpoint(path)
    if value.get("schema") != "symm_world_checkpoint_v2" or value["stage"]=="representation":
        raise ValueError("Inference requires a flow or rollout checkpoint")
    meta = value["metadata"]; cfg = from_dict(meta["config"])
    rep = make_representation(cfg,meta,"cpu")
    model = WorldModel(cfg,rep)
    state = value.get("teacher") or value["model"]
    model.load_state_dict(state,strict=True)
    model.eval().requires_grad_(False).to(device)
    return model,meta
