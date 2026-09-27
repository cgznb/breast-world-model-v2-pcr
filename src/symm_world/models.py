from __future__ import annotations
import copy
import torch
from torch import nn
from .encoder import PhaseStateEncoder, StateHeads, MaskedStatePredictor, ResidualStatePredictor
from .conditioning import ClinicalEncoder, ConditionSchema
from .velocity import build_velocity
from .flow import integrate
from .utils import update_ema


class RepresentationSystem(nn.Module):
    def __init__(self, cfg, schema: ConditionSchema, statistics):
        super().__init__()
        self.cfg = cfg
        self.encoder = PhaseStateEncoder(cfg.encoder)
        self.encoder.set_normalization(statistics["mean"], statistics["std"])
        self.target_encoder = copy.deepcopy(self.encoder).eval().requires_grad_(False)
        self.clinical = ClinicalEncoder(schema, cfg.encoder.dim, cfg.encoder.query_heads)
        self.masked_predictor = MaskedStatePredictor(cfg.encoder)
        self.future_predictor = ResidualStatePredictor(cfg.encoder)
        self.heads = StateHeads(cfg.encoder)
        # A detached history augments patient-level moments with microbatch=1.
        self.register_buffer("moment_queue", torch.zeros(64, cfg.encoder.dim*2))
        self.register_buffer("queue_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, decay):
        update_ema(self.target_encoder, self.encoder, decay)

    @torch.no_grad()
    def enqueue(self, summaries):
        for s in summaries.detach():
            p = int(self.queue_pointer)
            self.moment_queue[p].copy_(s)
            self.queue_pointer.fill_((p+1) % len(self.moment_queue))
            self.queue_count.fill_(min(len(self.moment_queue), int(self.queue_count)+1))


class WorldModel(nn.Module):
    def __init__(self, cfg, representation: RepresentationSystem):
        super().__init__()
        self.cfg = cfg
        self.encoder = copy.deepcopy(representation.encoder).eval().requires_grad_(False)
        self.heads = copy.deepcopy(representation.heads).eval().requires_grad_(False)
        self.clinical = copy.deepcopy(representation.clinical).eval().requires_grad_(False)
        self.future_predictor = copy.deepcopy(representation.future_predictor).eval().requires_grad_(False)
        d = cfg.encoder.dim
        layer = nn.TransformerEncoderLayer(d, cfg.encoder.query_heads, 4*d, dropout=0,
                                            activation="gelu", batch_first=True, norm_first=True)
        self.context_adapter = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.velocity = build_velocity(cfg.velocity, d)

    def train(self, mode=True):
        super().train(mode)
        # Frozen parameters do NOT mean no_grad: gradients through synthetic
        # states must still reach velocity parameters during rollout training.
        for module in (self.encoder, self.heads, self.clinical, self.future_predictor):
            module.eval()
        return self

    def context(self, observed, conditions, direction=1):
        clinical = self.clinical(conditions, direction=direction)
        if not self.cfg.velocity.source_tokens:
            return self.context_adapter(clinical)
        state = self.encoder(observed)
        components = [clinical, state.anatomy, state.disease]
        if self.cfg.velocity.use_predictive_prior:
            prior = self.future_predictor(state, clinical) if direction == 1 else state.disease
            components.append(prior)
        return self.context_adapter(torch.cat(components, 1))

    def sample(self, observed, conditions, noise=None, *, steps=20, method="heun", direction=1):
        """Only observed endpoint + available covariates; no target is accepted."""
        if observed.ndim != 5 or observed.shape[1] != 24 or not torch.isfinite(observed).all():
            raise ValueError("Observed latent must be finite [B,24,D,H,W]")
        if len(conditions) != len(observed):
            raise ValueError("Condition batch size differs from source batch")
        if noise is None:
            noise = torch.randn_like(observed)
        if noise.shape != observed.shape:
            raise ValueError("Sampling noise shape differs from observed latent")
        context = self.context(observed, conditions, direction)
        joint = torch.cat((noise, observed), 1) if direction == 1 else torch.cat((observed, noise), 1)
        z = integrate(self.velocity, joint, context, steps=steps, method=method, direction=direction)
        return z[:, :24] if direction == 1 else z[:, 24:]

    def sample_many(self, observed, conditions, samples, *, steps, method="heun", direction=1):
        if samples < 1:
            raise ValueError("At least one sample is required")
        # Sequential calls avoid a K-fold batch, but retained differentiable
        # graphs still scale with K. This does NOT guarantee low rollout VRAM.
        return torch.stack([self.sample(observed, conditions, steps=steps, method=method, direction=direction)
                            for _ in range(samples)], 1)

    def pcr_probability(self, observed, conditions):
        state = self.encoder(observed)
        clinical = self.clinical(conditions, direction=1)
        return self.heads.risk(state, clinical).sigmoid()
