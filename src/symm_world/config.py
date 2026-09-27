"""Strict, versioned experiment configuration. Unknown fields are errors."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import yaml

@dataclass
class EncoderConfig:
    phase_channels: int = 8
    stem_width: int = 48
    widths: tuple[int, ...] = (96, 192, 384)
    depths: tuple[int, ...] = (2, 2, 4)
    heads: tuple[int, ...] = (3, 6, 12)
    window: tuple[int, int, int] = (2, 4, 4)
    token_grid: tuple[int, int, int] = (2, 4, 4)
    dim: int = 192
    query_heads: int = 6
    anatomy_tokens: int = 4
    disease_tokens: int = 8
    predictor_depth: int = 4
    phase_mixer_depth: int = 2
    drop_path: float = 0.0
    checkpoint_blocks: bool = True
    mask_ratio: float = 0.5
    phase_drop_probability: float = 0.5

@dataclass
class VelocityConfig:
    # 'monai' calls the actual MONAI DiffusionModelUNet, not a local imitation.
    # 'native' is the separately tested full residual/cross-attention 3D U-Net.
    backend: str = "monai"
    channels: tuple[int, ...] = (128, 256, 384)
    num_res_blocks: int = 2
    attention_heads: int = 8
    checkpoint_blocks: bool = True
    source_tokens: bool = True
    use_predictive_prior: bool = True

@dataclass
class LossConfig:
    reconstruction: float = 1.0
    jepa: float = 1.0
    future: float = 0.25
    variance: float = 0.1
    covariance: float = 0.01
    anatomy: float = 0.02
    separation: float = 0.01
    latent_delta: float = 0.1
    kinetics: float = 0.1
    segmentation: float = 0.1
    biomarker: float = 0.1
    pcr: float = 0.05
    external_teacher: float = 0.0
    velocity: float = 1.0
    semantic_endpoint: float = 0.05
    local_map: float = 0.02
    composition: float = 0.1
    rollout_anchor: float = 0.1
    decoded_kinetics: float = 0.0

@dataclass
class TrainingConfig:
    seed: int = 20260919
    device: str = "cuda"
    precision: str = "bf16"
    batch_size: int = 1
    accumulation: int = 8
    representation_steps: int = 20000
    flow_steps: int = 60000
    rollout_steps: int = 10000
    lr: float = 1e-4
    representation_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    log_every: int = 25
    checkpoint_every: int = 1000
    validate_every: int = 1000
    validation_pairs: int = 16
    auxiliary_every: int = 4
    auxiliary_warmup: int = 2000
    reverse_probability: float = 0.5
    cpu_threads: int = 2
    strict_determinism: bool = True
    stage_batches: dict[str, int] = field(default_factory=dict)
    preload_latents: bool = False
    reference_batch_size: int = 0

@dataclass
class RolloutConfig:
    enabled: bool = False
    samples: int = 2
    steps: int = 4
    every: int = 4
    require_registered: bool = True
    require_verified_intervals: bool = True
    require_verified_treatment: bool = True
    local_map_delta: float = 0.05
    mmd_bandwidths: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

@dataclass
class SamplingConfig:
    steps: int = 20
    method: str = "heun"
    samples: int = 4

@dataclass
class Config:
    schema: str = "symm_world_v2"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def validate(self) -> "Config":
        e, v, t, r = self.encoder, self.velocity, self.training, self.rollout
        if self.schema != "symm_world_v2":
            raise ValueError("Unsupported configuration schema")
        if len(e.widths) != len(e.depths) or len(e.widths) != len(e.heads) or not e.widths:
            raise ValueError("Encoder widths/depths/heads must have equal nonzero length")
        if e.query_heads < 1 or v.attention_heads < 1 or any(h < 1 for h in e.heads):
            raise ValueError("Head counts must be positive")
        if any(w % h for w, h in zip(e.widths, e.heads)) or e.dim % e.query_heads:
            raise ValueError("Attention dimensions must be divisible by head counts")
        if any(x < 1 for x in (*e.window, *e.token_grid, *e.depths, *v.channels)):
            raise ValueError("Network sizes must be positive")
        if e.stem_width % e.query_heads:
            raise ValueError("stem_width must be divisible by query_heads")
        if len(e.window)!=3 or len(e.token_grid)!=3 or min(e.anatomy_tokens,e.disease_tokens,e.predictor_depth,e.phase_mixer_depth,v.num_res_blocks)<1:
            raise ValueError("Invalid spatial dimensions or network depths")
        if min(t.lr,t.representation_lr,t.grad_clip)<=0 or not all(math.isfinite(x) for x in (t.lr,t.representation_lr,t.grad_clip,t.weight_decay)) or t.weight_decay<0:
            raise ValueError("Invalid optimizer hyperparameters")
        if min(t.representation_steps,t.flow_steps,t.rollout_steps,t.warmup_steps,t.auxiliary_warmup)<0:
            raise ValueError("Training steps must be nonnegative")
        if not r.mmd_bandwidths or any(not math.isfinite(x) or x<=0 for x in r.mmd_bandwidths):
            raise ValueError("MMD bandwidths must be finite and positive")
        if e.phase_channels != 8:
            raise ValueError("This release preserves the audited three-phase 3x8 VQ contract")
        if not 0 < e.mask_ratio < 1 or not 0 <= e.phase_drop_probability <= 1:
            raise ValueError("Invalid masking probabilities")
        if v.backend not in {"monai", "native"}:
            raise ValueError("velocity.backend must be monai or native")
        if any(w % v.attention_heads for w in v.channels):
            raise ValueError("Velocity channels must be divisible by attention_heads")
        if t.precision not in {"fp32", "bf16"}:
            raise ValueError("Only fp32 and bf16 are supported; no silent FP16 substitution")
        if t.device == "cpu" and t.precision != "fp32":
            raise ValueError("CPU tests use explicit fp32")
        for name in ("batch_size", "accumulation", "log_every", "checkpoint_every",
                     "validate_every", "validation_pairs", "auxiliary_every", "cpu_threads"):
            if getattr(t, name) < 1:
                raise ValueError(f"training.{name} must be positive")
        if not 0 <= t.reverse_probability <= 1 or not 0 <= t.ema_decay < 1:
            raise ValueError("Invalid reverse probability / EMA decay")
        if set(t.stage_batches) - {"representation", "flow", "rollout"}:
            raise ValueError("Unknown stage in training.stage_batches")
        if any(not isinstance(v, int) or v < 1 for v in t.stage_batches.values()):
            raise ValueError("Stage batch sizes must be positive integers")
        if not isinstance(t.reference_batch_size, int) or t.reference_batch_size < 0:
            raise ValueError("reference_batch_size must be a nonnegative integer")
        if r.samples < 2 or r.steps < 1 or r.every < 1 or not 0 < r.local_map_delta < 1:
            raise ValueError("Distributional rollout needs >=2 independent samples")
        if self.sampling.steps < 1 or self.sampling.samples < 1 or self.sampling.method not in {"euler", "heun"}:
            raise ValueError("Invalid sampling configuration")
        if any(not math.isfinite(x) or x < 0 for x in asdict(self.loss).values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def _construct(cls, value):
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    allowed = {f.name for f in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    obj = cls(**value)
    # Normalize lists from YAML for checkpoint round trips.
    for name in ("widths", "depths", "heads", "window", "token_grid", "channels", "mmd_bandwidths"):
        if hasattr(obj, name):
            setattr(obj, name, tuple(getattr(obj, name)))
    return obj


def from_dict(value: dict[str, Any]) -> Config:
    top = {"schema", "encoder", "velocity", "loss", "training", "rollout", "sampling"}
    if set(value) - top:
        raise ValueError(f"Unknown top-level config keys: {sorted(set(value) - top)}")
    classes = {"encoder": EncoderConfig, "velocity": VelocityConfig, "loss": LossConfig,
               "training": TrainingConfig, "rollout": RolloutConfig, "sampling": SamplingConfig}
    parts = {k: _construct(cls, value.get(k, {})) for k, cls in classes.items()}
    return Config(schema=value.get("schema", "symm_world_v2"), **parts).validate()


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    def read(p: Path, seen: set[Path]) -> dict:
        if p in seen:
            raise ValueError("Cyclic configuration inheritance")
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("Configuration must be a YAML mapping")
        parent = d.pop("extends", None)
        if parent:
            base = read((p.parent / parent).resolve(), seen | {p})
            for key, val in d.items():
                if isinstance(val, dict) and isinstance(base.get(key), dict):
                    base[key] = {**base[key], **val}
                else:
                    base[key] = val
            return base
        return d
    return from_dict(read(path, set()))
