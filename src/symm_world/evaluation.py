"""Source-only inference metrics; no best-of-K selection against future labels."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import numpy as np
import torch
from .losses import energy_score
from .utils import autocast_context


def balanced_records(records, limit):
    if not limit:
        return records
    groups = defaultdict(list)
    for pair in sorted(records, key=lambda p: (p["patient_id"], p["id"])):
        groups[pair["patient_id"]].append(pair)
    # Spread the first patient round across available longitudinal transitions.
    for index, (patient, values) in enumerate(groups.items()):
        shift = index % len(values)
        groups[patient] = values[shift:] + values[:shift]
    selected = []
    while len(selected) < min(limit, len(records)):
        for values in groups.values():
            if values and len(selected) < limit:
                selected.append(values.pop(0))
    return selected


def auroc(labels, probabilities):
    y = np.asarray(labels, dtype=int); p = np.asarray(probabilities, dtype=float)
    if not len(y) or len(set(y)) != 2:
        return None
    pos, neg = p[y==1], p[y==0]
    return float(((pos[:,None]>neg[None]).astype(float)+.5*(pos[:,None]==neg[None])).mean())


def summarize_rows(rows, seed=1729):
    if not rows:
        raise ValueError("Cannot evaluate an empty split")
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    keys = sorted({k for r in rows for k,v in r.items() if isinstance(v, (float,int)) and not isinstance(v,bool)})
    result = {}
    rng = np.random.default_rng(seed)
    for key in keys:
        if key in {"pcr_label","pcr_probability"}:
            continue
        patient_values = np.asarray([np.mean([r[key] for r in rs if key in r]) for rs in by_patient.values() if any(key in r for r in rs)])
        boots = np.asarray([rng.choice(patient_values,len(patient_values),replace=True).mean() for _ in range(500)])
        result[key] = {"patient_macro_mean":float(patient_values.mean()),
                       "patient_bootstrap_95ci":[float(x) for x in np.quantile(boots,[.025,.975])]}
    return result


@torch.inference_mode()
def evaluate_model(model, store, split, metadata, *, limit=None, seed=1729):
    cfg = model.cfg
    records = store.records(split)
    if not records:
        raise ValueError(f"No {split} records")
    records = balanced_records(records, limit)
    old_mode = model.training
    model.eval()
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    rows = []
    pcr_active = metadata.get("trained_tasks",{}).get("pcr",False)
    with torch.random.fork_rng(devices=devices), autocast_context(device, cfg.training.precision):
        for pair in records:
            pair_seed = seed+int(hashlib.sha256(pair["id"].encode()).hexdigest()[:8],16)
            torch.manual_seed(pair_seed)
            source, conditions = store.source_batch(pair, device)
            samples = model.sample_many(source, conditions, cfg.sampling.samples,
                                          steps=cfg.sampling.steps, method=cfg.sampling.method)
            # Only AFTER source-only generation do we read the reference target.
            target = store.normalize(store.read_raw(pair["target"]))[None].to(device)
            mean = samples.mean(1)
            feature = torch.stack([model.encoder(samples[:,k]).disease.flatten(1) for k in range(samples.shape[1])],1)
            target_feature = model.encoder(target).disease.flatten(1)
            row = {"pair_id":pair["id"], "patient_id":pair["patient_id"],
                   "transition":conditions[0]["stage_i"]+"->"+conditions[0]["stage_j"],
                   "latent_mean_mae":float((mean-target).abs().mean()),
                   "latent_mean_rmse":float((mean-target).square().mean().sqrt()),
                   "copy_source_latent_mae":float((source-target).abs().mean()),
                   "sample_latent_mae":float((samples-target[:,None]).abs().mean()),
                   "sample_diversity":float(samples.std(1,unbiased=False).mean())}
            if samples.shape[1] >= 2:
                row["semantic_energy_score"] = float(energy_score(feature,target_feature))
            for i,name in enumerate(("pre","early","late")):
                row[f"{name}_latent_mae"] = float((mean[:,i*8:(i+1)*8]-target[:,i*8:(i+1)*8]).abs().mean())
            if pcr_active:
                probability = float(model.pcr_probability(source,conditions)[0])
                aux = store.read_aux(pair["source"],cfg.encoder.token_grid)
                if "pcr" in aux:
                    row["pcr_probability"], row["pcr_label"] = probability,float(aux["pcr"])
                    row["pcr_brier"] = (probability-float(aux["pcr"]))**2
            rows.append(row)
    model.train(old_mode)
    # pCR AUROC is reported at the declared transition/landmark, not pooled across
    # correlated forecasts from different horizons or patient stages.
    pcr = {}
    for transition in sorted({r["transition"] for r in rows}):
        selected = [r for r in rows if r["transition"] == transition and "pcr_label" in r]
        if selected:
            pcr[transition] = {"n":len(selected),"auroc":auroc([r["pcr_label"] for r in selected],[r["pcr_probability"] for r in selected])}
    return {"split":split,"independent_test":split=="test" and bool(store.manifest.get("provenance",{}).get("end_to_end_test_isolation_verified",False)),"pairs":len(rows),
            "patients":len({r["patient_id"] for r in rows}),
            "patient_disjoint_within_manifest":True,
            "end_to_end_test_isolation_verified":bool(store.manifest.get("provenance",{}).get("end_to_end_test_isolation_verified",False)),
            "synthetic":bool(store.manifest.get("provenance",{}).get("synthetic",False)),"metrics":summarize_rows(rows,seed),
            "pcr_by_transition":pcr,"records":rows,"solver":cfg.sampling.method,
            "solver_steps":cfg.sampling.steps,"samples":cfg.sampling.samples,
            "selection_policy":"ensemble mean / proper score; never best-of-K",
            "metric_space":"standardized VQ latent and learned semantic features, not MRI PSNR or true tumor volume",
            "warning":"Validation patients used for checkpoint selection are not an independent test set."}
