"""Command line interface. Sampling has no manifest or target-input argument."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
import json
import platform
import numpy as np
import torch
from .config import load_config
from .data import DatasetStore, convert_legacy
from .synthetic import create_synthetic
from .training import train_stage, load_inference
from .evaluation import evaluate_model
from .conditioning import validate_condition
from .auxiliary import build_auxiliary
from .codec import load_codec
from .utils import read_json, write_json, file_identity, seed_all


def load_source(path):
    path = Path(path).resolve()
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as a:
            if "latent" not in a:
                raise ValueError("Source NPZ must contain 'latent', not future images or unspecified arrays")
            array = a["latent"].copy()
    else:
        raise ValueError("Source must be an explicit .npy or .npz VQ latent")
    array = np.asarray(array, np.float32)
    if array.ndim != 4 or array.shape[0] != 24 or not np.isfinite(array).all():
        raise ValueError("Source must be a finite [24,D,H,W] array")
    return torch.from_numpy(array)[None]


@torch.inference_mode()
def sample_checkpoint(checkpoint, source_path, conditions, output, *, device="cpu", normalization="raw",
                      samples=None, steps=None, method=None, seed=1729, direction=1, codec=None):
    """No target files, labels, metadata inventories or patient records are read."""
    output = Path(output).resolve()
    if output.suffix != ".npz":
        raise ValueError("Output extension must be .npz")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Refusing to overwrite an existing prediction")
    condition = validate_condition(conditions)
    model, meta = load_inference(checkpoint, device)
    seed_all(seed, model.cfg.training.cpu_threads, model.cfg.training.strict_determinism)
    source = load_source(source_path).to(device)
    mean = source.new_tensor(meta["statistics"]["mean"]).view(1,24,1,1,1)
    std = source.new_tensor(meta["statistics"]["std"]).view(1,24,1,1,1)
    if normalization == "raw":
        source = (source-mean)/std
    elif normalization != "standardized":
        raise ValueError("normalization must explicitly be raw or standardized")
    k = samples if samples is not None else model.cfg.sampling.samples
    n = steps if steps is not None else model.cfg.sampling.steps
    solver = method or model.cfg.sampling.method
    if k < 1 or n < 1 or direction not in (-1,1):
        raise ValueError("Invalid samples, solver steps or direction")
    prediction = model.sample_many(source,[condition],k,steps=n,method=solver,direction=direction)
    raw = prediction*std[:,None]+mean[:,None]
    arrays = {"latent":raw[0].cpu().float().numpy(),
              "standardized_latent":prediction[0].cpu().float().numpy()}
    if codec is not None:
        arrays["images"] = torch.stack([codec.decode(raw[:,i]) for i in range(k)],1)[0].float().cpu().numpy()
    result = {"schema":"symm_world_prediction_v2","source_identity":file_identity(source_path),
              "checkpoint_identity":file_identity(checkpoint),"conditions":condition,"seed":seed,
              "direction":"forward" if direction==1 else "retrodiction",
              "solver":solver,"steps":n,"samples":k,"latent_shape":list(raw[0].shape),
              "source_only":True,"target_reads":0,"phase_order":["pre_aqc0","first_post_aqc1","metadata_late"],
              "output_grid":"array grid matching the observed latent shape; physical mapping requires independently audited source geometry",
              "warning":"Research scenario forecast, not a causal treatment recommendation; reverse is not biological recovery."}
    if direction==1 and meta.get("trained_tasks",{}).get("pcr",False):
        result["source_landmark_pcr_probability"] = float(model.pcr_probability(source,[condition])[0])
        result["pcr_warning"] = "Source-state auxiliary classifier; not a generated-future risk head. Calibration must be independently evaluated."
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    write_json(output.with_suffix(".json"),result)
    return result


def train_all(cfg, store, root, *, resume=False, stop_after=None, codec=None):
    # Reject impossible full experiments before spending stage A/B compute.
    if cfg.rollout.enabled and not store.triplets(cfg.rollout)[0]:
        raise ValueError(f"Full rollout experiment has no eligible triplets: {store.triplets(cfg.rollout)[1]}. Use cache_safe.yaml for the legacy unregistered cache.")
    results=[]
    for stage in ("representation","flow","rollout") if cfg.rollout.enabled else ("representation","flow"):
        result=train_stage(stage,cfg,store,root,resume=resume,stop_after=stop_after,codec=codec)
        results.append(result)
        if not result["complete"]:
            break
    return results


def smoke(root, config):
    root=Path(root).resolve()
    cfg=load_config(config)
    if cfg.training.device!="cpu" or cfg.velocity.backend!="native":
        raise ValueError("smoke requires an explicit CPU native-backend configuration")
    manifest=create_synthetic(root/"data")
    store=DatasetStore(manifest)
    stages=train_all(cfg,store,root/"run")
    checkpoint=root/"run"/"rollout"/"last.pt"
    if not checkpoint.exists(): checkpoint=root/"run"/"flow"/"last.pt"
    model,meta=load_inference(checkpoint,"cpu"); store.set_statistics(meta["statistics"])
    evaluation=evaluate_model(model,store,"test",meta,seed=cfg.training.seed)
    pair=store.records("test")[0]
    prediction=sample_checkpoint(checkpoint,store.resolve(store.views[pair["source"]]["latent"]),
                                 pair["conditions"],root/"sample.npz",device="cpu",seed=cfg.training.seed)
    report={"passed":True,"synthetic_only":True,"patient_data_tested":False,"gpu_tested":False,
            "torch":str(torch.__version__),"python":platform.python_version(),"stages":stages,
            "architecture_parameters":sum(p.numel() for p in model.parameters()),
            "trainable_world_parameters":sum(p.numel() for p in model.velocity.parameters()) if hasattr(model,"velocity") else None,
            "test_evaluation":evaluation,"sample":prediction}
    write_json(root/"report.json",report)
    return report


def parser():
    p=argparse.ArgumentParser(description="Predictive three-phase DCE state + stochastic symmetric world dynamics")
    sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("make-synthetic"); s.add_argument("--output",required=True);s.add_argument("--patients",type=int,default=7)
    s=sub.add_parser("convert-legacy");s.add_argument("--root",required=True);s.add_argument("--output",required=True);s.add_argument("--qc");s.add_argument("--auxiliary-root")
    s=sub.add_parser("audit");s.add_argument("--manifest",required=True);s.add_argument("--config",required=True);s.add_argument("--output",required=True);s.add_argument("--scan-arrays",action="store_true")
    s=sub.add_parser("cache-aux");s.add_argument("--manifest",required=True);s.add_argument("--output-manifest",required=True);s.add_argument("--output-dir",required=True)
    s.add_argument("--kind",choices=("measured","codec_proxy"),default="measured");s.add_argument("--grid",nargs=3,type=int,default=(4,8,8));s.add_argument("--codec");s.add_argument("--device",default="cpu");s.add_argument("--trusted-legacy-codec",action="store_true")
    s=sub.add_parser("cache-teacher");s.add_argument("--manifest",required=True);s.add_argument("--output-manifest",required=True);s.add_argument("--output-dir",required=True)
    s.add_argument("--teacher",required=True,help="Locally verified TorchScript teacher with Tensor[B,C,D,H,W] output")
    s.add_argument("--config",required=True);s.add_argument("--device",default="cpu")
    s=sub.add_parser("train");s.add_argument("--config",required=True);s.add_argument("--manifest",required=True);s.add_argument("--output",required=True)
    s.add_argument("--stage",choices=("all","representation","flow","rollout"),default="all");s.add_argument("--resume",action="store_true");s.add_argument("--stop-after",type=int)
    s.add_argument("--codec");s.add_argument("--trusted-legacy-codec",action="store_true")
    s=sub.add_parser("evaluate");s.add_argument("--checkpoint",required=True);s.add_argument("--manifest",required=True);s.add_argument("--output",required=True)
    s.add_argument("--split",choices=("val","test"),default="test");s.add_argument("--device",default="cpu");s.add_argument("--limit",type=int)
    s=sub.add_parser("sample");s.add_argument("--checkpoint",required=True);s.add_argument("--source",required=True);s.add_argument("--conditions",required=True);s.add_argument("--output",required=True)
    s.add_argument("--device",default="cpu");s.add_argument("--normalization",choices=("raw","standardized"),default="raw");s.add_argument("--direction",choices=("forward","reverse"),default="forward")
    s.add_argument("--samples",type=int);s.add_argument("--steps",type=int);s.add_argument("--method",choices=("euler","heun"));s.add_argument("--seed",type=int,default=1729)
    s.add_argument("--codec");s.add_argument("--trusted-legacy-codec",action="store_true")
    s=sub.add_parser("smoke");s.add_argument("--output",required=True);s.add_argument("--config",default=str(Path(__file__).resolve().parents[2]/"configs"/"smoke.yaml"))
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if a.command=="make-synthetic":
        result={"manifest":str(create_synthetic(a.output,a.patients))}
    elif a.command=="convert-legacy":
        result=convert_legacy(a.root,a.output,a.qc,a.auxiliary_root)
    elif a.command=="audit":
        cfg=load_config(a.config);result=DatasetStore(a.manifest).audit(a.scan_arrays,cfg.rollout);write_json(a.output,result)
    elif a.command=="cache-aux":
        codec=load_codec(a.codec,a.device,trusted_legacy=a.trusted_legacy_codec) if a.codec else None
        result=build_auxiliary(DatasetStore(a.manifest),a.output_manifest,a.output_dir,grid=tuple(a.grid),codec=codec,device=a.device,kind=a.kind)
    elif a.command=="cache-teacher":
        from .teachers import export_teacher_targets
        result=export_teacher_targets(DatasetStore(a.manifest),a.teacher,a.output_manifest,a.output_dir,load_config(a.config),a.device)
    elif a.command=="train":
        cfg=load_config(a.config);store=DatasetStore(a.manifest)
        codec=load_codec(a.codec,cfg.training.device,trusted_legacy=a.trusted_legacy_codec) if a.codec else None
        if a.stage=="all":
            result=train_all(cfg,store,a.output,resume=a.resume,stop_after=a.stop_after,codec=codec)
        else:
            result=train_stage(a.stage,cfg,store,a.output,resume=a.resume,stop_after=a.stop_after,codec=codec)
    elif a.command=="evaluate":
        model,meta=load_inference(a.checkpoint,a.device);store=DatasetStore(a.manifest);store.set_statistics(meta["statistics"])
        seed_all(model.cfg.training.seed,model.cfg.training.cpu_threads,model.cfg.training.strict_determinism)
        result=evaluate_model(model,store,a.split,meta,limit=a.limit);write_json(a.output,result)
    elif a.command=="sample":
        codec=load_codec(a.codec,a.device,trusted_legacy=a.trusted_legacy_codec) if a.codec else None
        result=sample_checkpoint(a.checkpoint,a.source,read_json(a.conditions),a.output,device=a.device,normalization=a.normalization,
                                 samples=a.samples,steps=a.steps,method=a.method,seed=a.seed,direction=1 if a.direction=="forward" else -1,codec=codec)
    elif a.command=="smoke":
        result=smoke(a.output,a.config)
    else:
        raise ValueError("Unknown command")
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))

if __name__=="__main__":
    main()
