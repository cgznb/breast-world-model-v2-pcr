#!/usr/bin/env python3
"""Encode an ALREADY prepared, shared-normalized three-phase source visit.

This does not register images or convert DICOM. No target visit is accepted.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import numpy as np
import torch
from symm_world.codec import load_codec
from symm_world.auxiliary import read_images
from symm_world.utils import write_json,file_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',required=True,help='NPZ images[3,D,H,W] and support; already prepared')
    p.add_argument('--codec',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--confirm-shared-training-normalization',action='store_true',required=True)
    p.add_argument('--trusted-legacy-codec',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve()
    if out.suffix!='.npz':raise ValueError('Output must be .npz')
    if out.exists() or out.with_suffix('.json').exists():raise FileExistsError(out)
    if a.precision=='bf16' and not a.device.startswith('cuda'):raise ValueError('bf16 export requires CUDA')
    image,_,_=read_images(a.images)
    codec=load_codec(a.codec,a.device,trusted_legacy=a.trusted_legacy_codec)
    with torch.inference_mode(),torch.autocast(torch.device(a.device).type,dtype=torch.bfloat16,enabled=a.precision=='bf16'):
        latent=codec.encode(torch.from_numpy(image)[None].to(a.device)).float()[0].cpu().numpy()
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,latent=latent)
    write_json(out.with_suffix('.json'),{'source_images_sha256':file_digest(a.images),'codec_sha256':file_digest(a.codec),
               'precision':a.precision,'shape':list(latent.shape),'normalization':'user-confirmed original shared training image normalization',
               'phase_order':['pre_aqc0','first_post_aqc1','metadata_late'],'source_only':True})
    print(out)

if __name__=='__main__':main()
