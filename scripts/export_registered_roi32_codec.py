"""Export local ROI32 codec weights and compare decoding to the original class."""
from pathlib import Path
from dataclasses import asdict
import argparse
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
import numpy as np
import torch
from symm_world.codec import load_codec
from symm_world.utils import file_identity, read_json, write_json, save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--original-repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = read_json(args.manifest)
    identity = manifest["provenance"]["codec"]
    if file_identity(identity["path"]) != identity:
        raise ValueError("The cache's original codec changed")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    codec = load_codec(identity["path"], "cpu", trusted_legacy=True)
    sys.path.insert(0, str(Path(args.original_repo).resolve()))
    from mewm_ispy2.vqgan import MRILevelVQGAN, VQGANConfig
    original = MRILevelVQGAN(VQGANConfig(**asdict(codec.codec.config))).eval().requires_grad_(False)
    original.load_state_dict(codec.codec.state_dict(), strict=True)
    raw = torch.from_numpy(np.load(manifest["views"][0]["latent"], allow_pickle=False).astype(np.float32))[None]
    with torch.no_grad():
        decoded = codec.decode(raw)
        references = []
        for phase in raw.split(8, 1):
            quantized, _ = original.quantizer(phase)
            references.append(original.decode(quantized))
        reference = torch.cat(references, 1)
    error = float((decoded-reference).abs().max())
    if decoded.shape != (1, 3, 32, 128, 128) or not torch.isfinite(decoded).all() or error > 1e-5:
        raise ValueError(f"ROI32 codec replay failed: shape={tuple(decoded.shape)}, max_error={error}")
    save_checkpoint(output, {"schema": "symm_world_codec_v2", "model_config": asdict(codec.codec.config),
                             "codec_state": codec.codec.state_dict(), "source_identity": identity,
                             "numeric_contract": "registered_ROI32_shared_training_DCE0_normalization",
                             "image_normalization": manifest["image_normalization"]})
    restored = load_codec(output)
    if any(not torch.equal(value, restored.codec.state_dict()[key]) for key, value in codec.codec.state_dict().items()):
        raise ValueError("Safe codec weight roundtrip failed")
    report = {"passed": True, "source_identity": identity, "output_identity": file_identity(output),
              "decoded_shape": list(decoded.shape), "original_decoder_max_absolute_error": error,
              "weight_roundtrip_exact": True, "original_checkpoint_unchanged": file_identity(identity["path"]) == identity}
    write_json(output.with_suffix(".verification.json"), report)
    print(report)


if __name__ == "__main__":
    main()
