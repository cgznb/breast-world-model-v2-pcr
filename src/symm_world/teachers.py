"""Optional offline semantic teacher. Never initializes a random 'foundation model'.

A supplied, trusted TorchScript encoder must return a dense 3D feature tensor.
The same teacher processes each same-visit DCE phase, then features are averaged.
This is a teacher interface, NOT bundled pretrained DINO/VoCo/V-JEPA weights.
"""
from pathlib import Path
import copy
import torch
import torch.nn.functional as F
import numpy as np
from .auxiliary import read_images
from .utils import file_identity,write_json


@torch.inference_mode()
def export_teacher_targets(store,teacher_path,output_manifest,output_dir,cfg,device="cpu"):
    path=Path(teacher_path).resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    # TorchScript is executable code: users must trust this local artifact.
    teacher=torch.jit.load(str(path),map_location=device).eval()
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=copy.deepcopy(store.manifest);count=missing=0
    for view in manifest["views"]:
        if not view.get("images"):
            missing+=1;continue
        image,support,_=read_images(store.resolve(view["images"]))
        phases=torch.from_numpy(image[:,None]).to(device)
        feature=teacher(phases)
        if not isinstance(feature,torch.Tensor) or feature.ndim!=5 or feature.shape[0]!=3:
            raise ValueError("Teacher must return Tensor[3,C,D,H,W] for three independent phase volumes")
        if feature.shape[1]!=cfg.encoder.dim or not torch.isfinite(feature).all():
            raise ValueError("Teacher feature dimension must match encoder.dim, with finite outputs; export the intended layer explicitly")
        pooled=F.adaptive_avg_pool3d(feature.float(),cfg.encoder.token_grid).mean(0).flatten(1).T
        tokens=F.layer_norm(pooled,(pooled.shape[-1],)).cpu().numpy()
        arrays={}
        if view.get("auxiliary"):
            with np.load(store.resolve(view["auxiliary"]),allow_pickle=False) as a:
                arrays.update({k:a[k].copy() for k in a.files})
        arrays["external_tokens"]=tokens
        output=out/(view["id"]+".npz")
        if output.exists():raise FileExistsError(output)
        np.savez_compressed(output,**arrays)
        view["auxiliary"]=str(output);view["auxiliary_identity"]=file_identity(output);count+=1
    if count==0: raise ValueError("No image sidecars available; no teacher features exported")
    for view in manifest["views"]:
        for k in ("latent","images","auxiliary"):
            if view.get(k): view[k]=str(store.resolve(view[k]))
    result={"teacher_identity":file_identity(path),"teacher_format":"trusted TorchScript dense 3D encoder",
            "exported_views":count,"missing_image_views":missing,"pretrained_status":"user-supplied; not inferred or certified by this program",
            "projection":"phase average, adaptive pooling, channel LayerNorm","parent_manifest_identity":store.identity}
    manifest["external_teacher_provenance"]=result
    write_json(output_manifest,manifest)
    return result
