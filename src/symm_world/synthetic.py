"""Fully synthetic deterministic fixtures. These are NOT medical validation data."""
from pathlib import Path
import math
import numpy as np
from .data import PHASES,SCHEMA,geometry_digest
from .utils import write_json


def create_synthetic(root,patients=7,shape=(4,8,8),seed=123):
    root = Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists():
        raise FileExistsError("Synthetic output already exists")
    if patients < 5:
        raise ValueError("Use >=5 synthetic patients to cover all three splits")
    rng = np.random.default_rng(seed)
    coords = np.stack(np.meshgrid(*[np.linspace(-1,1,n) for n in shape],indexing="ij"))
    spatial = np.exp(-np.square(coords).sum(0)*5).astype(np.float32)
    geometry = {"shape_zyx":list(shape),"synthetic":True,"canonical_source_grid":True}
    views,pairs = [],[]
    for i in range(patients):
        split = "train" if i < patients-3 else "val" if i < patients-1 else "test"
        patient = f"synthetic_{i:03d}"
        arm = "drug_a" if i%2==0 else "drug_b"
        identity = rng.normal(0,.3,(24,*shape)).astype(np.float32)
        for stage in range(3):
            view_id = f"{patient}_T{stage}"
            response = 1-stage*(.2 if arm=="drug_a" else .1)
            z = identity + response*spatial[None]*rng.uniform(.6,1.2,(24,1,1,1)).astype(np.float32)
            z += rng.normal(0,.05,z.shape).astype(np.float32)
            np.save(root/(view_id+".npy"),z)
            early,late = response*spatial,.7*response*spatial
            kin = np.stack((early,late,late-early)).astype(np.float32)
            mask = (spatial>.4).astype(np.float32)[None]
            np.savez_compressed(root/(view_id+".npz"),kinetics=kin,kinetics_mask=np.ones((1,*shape),np.float32),
                                 segmentation=mask,segmentation_mask=np.ones_like(mask))
            views.append({"id":view_id,"patient_id":patient,"visit":f"T{stage}","split":split,
                          "latent":view_id+".npy","auxiliary":view_id+".npz","geometry":geometry,
                          "grid_id":geometry_digest(geometry),"source_available_grid":True,
                          "kinetics_kind":"synthetic_fixture",
                          "labels":{"pcr":i%2,"biomarkers":[float(np.log1p(mask.sum())),float(early.mean()),float(late.mean()),float(early.std())]}})
        for a,b in ((0,1),(0,2),(1,2)):
            conditions = {"stage_i":f"T{a}","stage_j":f"T{b}","age":45+i,"treatment_arm":arm,
                          "hr_status":"positive" if i%2 else "negative","her2_status":"negative",
                          "mammaprint":"high","interval_verified":True,"delta_days":30*(b-a),
                          "action_segments":[{"drug":arm,"start":30*a,"end":30*b,"dose":1.0,"known_at_source":True}]}
            pairs.append({"id":f"{patient}_{a}_{b}","patient_id":patient,"split":split,
                          "source":f"{patient}_T{a}","target":f"{patient}_T{b}","conditions":conditions,
                          "registered":True,"treatment_verified":True,"scenario_id":patient+"_assigned_plan",
                          "plan_known_at_initial_source":True,"anatomy_comparable":True,"qc_weight":1.0})
    value = {"schema":SCHEMA,"phase_order":PHASES,"latent_channels":24,"views":views,"pairs":pairs,
             "provenance":{"synthetic":True,"seed":seed,"NOT_CLINICAL_EVIDENCE":True,"end_to_end_test_isolation_verified":True}}
    write_json(root/"manifest.json",value)
    return root/"manifest.json"
