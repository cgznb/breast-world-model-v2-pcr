"""Record executed architecture shapes from the frozen World V2 checkpoint."""
from __future__ import annotations
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch
from symm_world.training import load_inference
from symm_world.codec import load_codec

ROOT = REPO / "runs/registered_roi32_20260919"
OUT = ROOT / "analysis_30_pcr_v1v4_20260919"


def shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [shape(v) for v in value]
    return type(value).__name__


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260919)
    model, metadata = load_inference(ROOT / "flow/best.pt", "cpu")
    codec = load_codec(REPO / "data/registered_roi32/codec.pt", "cpu")
    manifest = json.loads((REPO / "data/registered_roi32/manifest.json").read_text())
    pair = next(p for p in manifest["pairs"] if p["split"] == "val")
    names = ["encoder.stem", "encoder.phase_mixer", "encoder.fusion", "encoder.difference_stem",
             "encoder.stages.0.1", "encoder.stages.1.1", "encoder.stages.2.3", "encoder.multiscale",
             "encoder.anatomy_pool", "encoder.disease_pool", "clinical", "future_predictor", "context_adapter",
             "velocity.network.conv_in", "velocity.network.down_blocks.0", "velocity.network.down_blocks.1",
             "velocity.network.down_blocks.2", "velocity.network.middle_block", "velocity.network.up_blocks.0",
             "velocity.network.up_blocks.1", "velocity.network.up_blocks.2", "velocity.network.out"]
    shapes, hooks = {}, []
    for name in names:
        hooks.append(model.get_submodule(name).register_forward_hook(
            lambda module, inputs, output, name=name: shapes.__setitem__(name, {"input": shape(inputs), "output": shape(output)})))
    latent = torch.zeros(1, 24, 8, 32, 32)
    with torch.inference_mode():
        context = model.context(latent, [pair["conditions"]])
        velocity = model.velocity(torch.cat((latent, latent), 1), torch.tensor([.5]), context)
    assert context.shape == (1, 30, 192) and velocity.shape == (1, 48, 8, 32, 32)
    assert torch.isfinite(velocity).all()
    for hook in hooks:
        hook.remove()
    modules = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
    config = metadata["config"]
    checkpoint = torch.load(ROOT / "flow/best.pt", map_location="cpu", weights_only=True, mmap=True)
    result = {"checkpoint_step": checkpoint["step"], "checkpoint_weights": "EMA", "input": list(latent.shape),
              "total_parameters": sum(p.numel() for p in model.parameters()), "module_parameters": modules,
              "flow_trainable_parameters": modules["velocity"] + modules["context_adapter"],
              "codec_parameters": sum(p.numel() for p in codec.parameters()),
              "codec_configuration": vars(codec.codec.config), "shapes": shapes,
              "active_backend": type(model.velocity).__name__, "config": {k: config[k] for k in ("encoder", "velocity", "loss", "rollout")},
              "reverse_training_probability": config["training"]["reverse_probability"],
              "auxiliary_tasks_trained": metadata["trained_tasks"]}
    (OUT / "architecture_facts.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("shapes", "config")}))


if __name__ == "__main__":
    main()
