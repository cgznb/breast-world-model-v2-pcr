"""Measured DCE/segmentation sidecars and optional frozen-codec proxies."""
from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn.functional as F
from .losses import enhancement_differences
from .utils import write_json, file_identity


def _unpack_mask(array,shape):
    if tuple(array.shape) == tuple(shape):
        return array.astype(bool)
    return np.unpackbits(array, count=int(np.prod(shape))).reshape(shape).astype(bool)


def read_images(path):
    with np.load(path,allow_pickle=False) as a:
        key = "images" if "images" in a else "image"
        image = np.asarray(a[key],np.float32)
        if image.ndim != 4 or image.shape[0] != 3 or not np.isfinite(image).all():
            raise ValueError("Image sidecar must have finite [3,D,H,W] images")
        if "support" not in a:
            raise ValueError("DCE kinetic supervision needs explicit per-phase support, not a fabricated full mask")
        support = _unpack_mask(a["support"],image.shape)
        segmentation = None
        if "tumor_mask" in a:
            segmentation = np.asarray(a["tumor_mask"],np.float32)
            if segmentation.ndim == 3:
                segmentation = segmentation[None]
            if segmentation.shape != (1,*image.shape[1:]):
                raise ValueError("tumor_mask and images must share the same visit grid")
    return image,support,segmentation


def build_auxiliary(store,output_manifest,output_dir,*,grid=(4,8,8),codec=None,device="cpu",kind="measured"):
    if kind not in {"measured","codec_proxy"}:
        raise ValueError("Auxiliary kind must be measured or codec_proxy")
    if kind == "codec_proxy" and codec is None:
        raise ValueError("A codec checkpoint is required for a codec proxy")
    out = Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    manifest = copy.deepcopy(store.manifest)
    if kind == "measured":
        normalization = manifest.get("image_normalization", {})
        mean = np.asarray(normalization.get("mean", []), dtype=float)
        std = np.asarray(normalization.get("std", []), dtype=float)
        if mean.size != 1 or std.size != 1 or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Measured DCE differences require documented shared scalar image_normalization mean/std across all three phases")
    counts = {"measured":0,"codec_proxy":0,"missing":0}
    for view in manifest["views"]:
        if kind == "measured":
            if not view.get("images"):
                counts["missing"] += 1
                continue
            images,support,seg = read_images(store.resolve(view["images"]))
            value = torch.from_numpy(images)[None]
            valid = torch.from_numpy(support.all(0)[None,None].astype(np.float32))
        else:
            raw = store.read_raw(view["id"])[None].to(device)
            with torch.inference_mode():
                value = codec.decode(raw).cpu()
            # The entire decoder tensor is valid for the explicitly named
            # reconstruction proxy; this is not a measured acquisition mask.
            valid = torch.ones_like(value[:,:1])
            seg = None
        kinetics = enhancement_differences(value)
        weights = F.adaptive_avg_pool3d(valid,grid)
        # Pool weighted differences; unsupported voxels cannot bias the mean.
        kinetic = F.adaptive_avg_pool3d(kinetics*valid,grid)/weights.clamp_min(1e-6)
        arrays = {}
        if view.get("auxiliary"):
            with np.load(store.resolve(view["auxiliary"]), allow_pickle=False) as previous:
                arrays.update({k: previous[k].copy() for k in previous.files})
        arrays.update(kinetics=kinetic[0].numpy(),kinetics_mask=(weights[0]>.5).float().numpy())
        if seg is not None:
            arrays["segmentation"] = F.adaptive_avg_pool3d(torch.from_numpy(seg)[None],grid)[0].numpy()
            arrays["segmentation_mask"] = (weights[0]>.5).float().numpy()
        path = out/(view["id"]+".npz")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite auxiliary targets: {path}")
        np.savez_compressed(path,**arrays)
        view["auxiliary"] = str(path)
        view["kinetics_kind"] = kind
        view["auxiliary_identity"] = file_identity(path)
        counts[kind] += 1
    # Keep all original absolute/relative paths valid under the new manifest.
    for view in manifest["views"]:
        for key in ("latent","images"):
            if view.get(key):
                view[key] = str(store.resolve(view[key]))
        if view.get("auxiliary") and not Path(view["auxiliary"]).is_absolute():
            view["auxiliary"] = str(store.resolve(view["auxiliary"]))
    manifest["auxiliary_provenance"] = {"kind":kind,"parent_manifest_identity":store.identity,"counts":counts,
                                       "targets":"difference signals in shared image-normalization units; NOT PE/SER/Tofts parameters"}
    write_json(output_manifest,manifest)
    return counts
