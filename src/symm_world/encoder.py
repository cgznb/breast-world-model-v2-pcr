"""A trainable semantic encoder on top of unchanged, continuous VQ coordinates.

The codec latent is never relabelled as anatomy/disease channels. Those states
are learned attention-pooled projections, and the image decoder still receives
its original 24-channel representation.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .config import EncoderConfig
from .layers import (ConvNeXt3D, SwinBlock3D, PatchMerging3D, QueryPool,
                     TokenPredictor, position_3d, maybe_checkpoint)

@dataclass
class PatientState:
    dense: torch.Tensor
    anatomy: torch.Tensor
    disease: torch.Tensor
    grid: tuple[int, int, int]

    @property
    def summary(self):
        return torch.cat((self.anatomy.mean(1), self.disease.mean(1)), -1)

    def detached(self):
        return PatientState(self.dense.detach(), self.anatomy.detach(), self.disease.detach(), self.grid)


class PhaseMixer(nn.Module):
    """Cross-phase attention at corresponding positions inside one MRI visit."""
    def __init__(self, dim, heads, depth):
        super().__init__()
        self.phase_embedding = nn.Parameter(torch.randn(1, 3, dim) * .02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim*4, dropout=0,
                                            activation="gelu", batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False)
    def forward(self, x):
        b, p, c, d, h, w = x.shape
        seq = x.permute(0, 3, 4, 5, 1, 2).reshape(-1, p, c)
        # Each spatial position is independent. CUDA SDPA limits its batch
        # dimension, which large ROI32 batches can exceed after flattening.
        seq = torch.cat([self.layers(part + self.phase_embedding)
                         for part in seq.split(32768, dim=0)], dim=0)
        return seq.reshape(b, d, h, w, p*c).movedim(-1, 1)


class PhaseStateEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        c, s = cfg.phase_channels, cfg.stem_width
        self.register_buffer("latent_mean", torch.zeros(1, 24, 1, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, 24, 1, 1, 1))
        self.stem = nn.Sequential(nn.Conv3d(c, s, 3, stride=2, padding=1),
                                   ConvNeXt3D(s), ConvNeXt3D(s))
        self.phase_mixer = PhaseMixer(s, cfg.query_heads, cfg.phase_mixer_depth)
        self.fusion = nn.Conv3d(s*3, cfg.widths[0], 1)
        # These are feature differences, NOT signal-intensity enhancement ratios.
        self.difference_stem = nn.Sequential(nn.Conv3d(24, cfg.widths[0], 3, stride=2, padding=1),
                                             ConvNeXt3D(cfg.widths[0]))
        self.merges, self.stages, self.projections = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for level, (width, depth, heads) in enumerate(zip(cfg.widths, cfg.depths, cfg.heads)):
            self.merges.append(nn.Identity() if level == 0 else PatchMerging3D(cfg.widths[level-1], width))
            self.stages.append(nn.ModuleList([SwinBlock3D(width, heads, cfg.window, shifted=bool(j%2),
                                                            drop_path=cfg.drop_path) for j in range(depth)]))
            self.projections.append(nn.Conv3d(width, cfg.dim, 1))
        self.multiscale = nn.Sequential(nn.Linear(cfg.dim*len(cfg.widths), cfg.dim*2), nn.GELU(),
                                         nn.Linear(cfg.dim*2, cfg.dim), nn.LayerNorm(cfg.dim))
        self.anatomy_pool = QueryPool(cfg.dim, cfg.query_heads, cfg.anatomy_tokens)
        self.disease_pool = QueryPool(cfg.dim, cfg.query_heads, cfg.disease_tokens)

    @torch.no_grad()
    def set_normalization(self, mean, std):
        mean = torch.as_tensor(mean).reshape_as(self.latent_mean)
        std = torch.as_tensor(std).reshape_as(self.latent_std)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid latent normalization")
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)

    def raw_differences(self, z):
        raw = z * self.latent_std + self.latent_mean
        p, e, l = raw.split(8, 1)
        # A common per-channel scale preserves phase-difference meaning.
        scale = self.latent_std.reshape(1, 3, 8, 1, 1, 1).mean(1).clamp_min(1e-5)
        return torch.cat(((e-p)/scale, (l-p)/scale, (l-e)/scale), 1)

    def forward(self, z):
        if z.ndim != 5 or z.shape[1] != 24:
            raise ValueError("Expected [B,24,D,H,W] continuous standardized VQ latent")
        b, _, d, h, w = z.shape
        phase = z.reshape(b*3, 8, d, h, w)
        phase = maybe_checkpoint(self.stem, phase, enabled=self.cfg.checkpoint_blocks)
        phase = phase.reshape(b, 3, *phase.shape[1:])
        mixed = maybe_checkpoint(self.phase_mixer, phase, enabled=self.cfg.checkpoint_blocks)
        x = self.fusion(mixed) + self.difference_stem(self.raw_differences(z))
        levels = []
        for merge, stage, project in zip(self.merges, self.stages, self.projections):
            x = merge(x)
            for block in stage:
                x = maybe_checkpoint(block, x, enabled=self.cfg.checkpoint_blocks)
            levels.append(F.adaptive_avg_pool3d(project(x), self.cfg.token_grid).flatten(2).transpose(1, 2))
        dense = self.multiscale(torch.cat(levels, -1))
        dense = dense + position_3d(self.cfg.token_grid, self.cfg.dim, dense.device, dense.dtype)
        return PatientState(dense, self.anatomy_pool(dense), self.disease_pool(dense), self.cfg.token_grid)


class StateHeads(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        def dense_head(out):
            return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, out))
        self.reconstruction = dense_head(24)
        self.latent_delta = dense_head(24)
        self.kinetics = dense_head(3)
        self.segmentation = dense_head(1)
        self.biomarker = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, 4))
        self.external_projection = dense_head(d)
        self.pcr = nn.Sequential(nn.LayerNorm(d*3), nn.Linear(d*3, d*2), nn.GELU(),
                                 nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, 1))

    def volume(self, kind, state):
        value = getattr(self, kind)(state.dense).transpose(1, 2)
        return value.reshape(len(value), -1, *state.grid)

    def risk(self, state, condition_tokens):
        return self.pcr(torch.cat((state.summary, condition_tokens.mean(1)), -1)).squeeze(-1)


class MaskedStatePredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.dim) * .02)
        self.predictor = TokenPredictor(cfg.dim, cfg.query_heads, cfg.predictor_depth)
    def forward(self, context):
        q = self.mask_token.expand(len(context.dense), math.prod(context.grid), -1)
        q = q + position_3d(context.grid, self.cfg.dim, q.device, q.dtype)
        return self.predictor(q, context.dense)


class ResidualStatePredictor(nn.Module):
    """Treatment-conditioned predictive representation, not a deterministic image decoder."""
    def __init__(self, cfg):
        super().__init__()
        self.predictor = TokenPredictor(cfg.dim, cfg.query_heads, cfg.predictor_depth)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
    def forward(self, state, conditions):
        memory = torch.cat((state.anatomy, state.disease, conditions), 1)
        delta = self.predictor(state.disease, memory)
        return state.disease + self.residual_scale * delta


def corrupt_latent(z, cfg):
    """Mask BEFORE the new convolution/phase mixer; no direct target-value shortcut."""
    b = len(z)
    n = math.prod(cfg.token_grid)
    count = max(1, min(n-1, round(n*cfg.mask_ratio))) if n > 1 else 1
    mask = torch.zeros((b, n), device=z.device, dtype=torch.bool)
    # Random coarse blocks become contiguous 3D regions at latent resolution.
    for row in mask:
        row[torch.randperm(n, device=z.device)[:count]] = True
    coarse = mask.reshape(b, 1, *cfg.token_grid)
    high = F.interpolate(coarse.float(), size=z.shape[2:], mode="nearest").bool()
    out = z.masked_fill(high, 0)
    if cfg.phase_drop_probability:
        out = out.clone().reshape(b, 3, 8, *z.shape[2:])
        for i in range(b):
            if torch.rand((), device=z.device) < cfg.phase_drop_probability:
                p = int(torch.randint(3, (), device=z.device))
                out[i, p] = 0
        out = out.reshape_as(z)
    return out, mask
