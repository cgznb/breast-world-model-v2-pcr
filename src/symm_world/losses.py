"""Representation and generative objectives, with explicit stochastic semantics."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .encoder import corrupt_latent
from .flow import make_path, estimate_endpoints, local_teacher_map, expand_time
from .utils import zero_loss


def cosine_distance(x, y):
    return (1-(F.normalize(x.float(), dim=-1)*F.normalize(y.float(), dim=-1)).sum(-1)).mean()


def variance_covariance(x):
    """VICReg-style moments; samples are rows. No unbiased variance at N=1."""
    x = x.float().reshape(-1, x.shape[-1])
    if len(x) < 2:
        return zero_loss(x), zero_loss(x)
    centered = x-x.mean(0)
    std = torch.sqrt(centered.square().mean(0) + 1e-4)
    variance = F.relu(1-std).mean()
    cov = centered.T@centered/(len(x)-1)
    off = cov - torch.diag_embed(cov.diag())
    return variance, off.square().sum()/x.shape[1]


def cross_covariance(summary):
    if len(summary) < 2:
        return zero_loss(summary)
    a, d = summary.float().chunk(2, -1)
    a, d = a-a.mean(0), d-d.mean(0)
    return ((a.T@d)/(len(a)-1)).square().mean()


def weighted_huber(pred, target, mask):
    mask = mask.to(pred).expand_as(pred)
    return (F.smooth_l1_loss(pred.float(), target.to(pred).float(), reduction="none")*mask).sum()/mask.sum().clamp_min(1)


def auxiliary_losses(heads, state, auxiliaries):
    """Missing labels mean NO loss, never a negative/pCR=0 pseudo-label."""
    ref = state.dense
    losses = {k: zero_loss(ref) for k in ("kinetics", "segmentation", "biomarker", "external_teacher")}
    counts = {k: 0 for k in losses}
    kin, seg = heads.volume("kinetics", state), heads.volume("segmentation", state)
    bio = heads.biomarker(state.disease.mean(1))
    ext = heads.external_projection(state.dense)
    for i, aux in enumerate(auxiliaries):
        if "kinetics" in aux:
            if "kinetics_mask" not in aux:
                raise ValueError("Kinetic targets require an explicit validity/support mask")
            losses["kinetics"] = losses["kinetics"] + weighted_huber(kin[i], aux["kinetics"], aux["kinetics_mask"])
            counts["kinetics"] += int(aux["kinetics_mask"].sum() > 0)
        if "segmentation" in aux:
            if "segmentation_mask" not in aux:
                raise ValueError("Segmentation targets require a validity mask; all-zero tumor masks are allowed")
            truth = aux["segmentation"].to(seg)
            mask = aux["segmentation_mask"].to(seg).expand_as(truth)
            if ((truth < 0)|(truth > 1)).any():
                raise ValueError("Soft segmentation targets must be in [0,1]")
            pred = seg[i]
            bce = (F.binary_cross_entropy_with_logits(pred, truth, reduction="none")*mask).sum()/mask.sum().clamp_min(1)
            probability = pred.sigmoid()
            dice = 1-(2*(probability*truth*mask).sum()+1)/((probability*mask).sum()+(truth*mask).sum()+1)
            losses["segmentation"] = losses["segmentation"]+bce+dice
            counts["segmentation"] += int(mask.sum() > 0)
        if "biomarkers" in aux:
            losses["biomarker"] = losses["biomarker"]+weighted_huber(bio[i], aux["biomarkers"], aux["biomarker_mask"])
            counts["biomarker"] += int(aux["biomarker_mask"].sum() > 0)
        if "external_tokens" in aux:
            teacher = aux["external_tokens"].to(ext)
            if teacher.shape != ext[i].shape:
                raise ValueError("External teacher grid/dimension mismatch: re-export the declared teacher features")
            losses["external_teacher"] = losses["external_teacher"]+cosine_distance(ext[i], teacher.detach())
            counts["external_teacher"] += 1
    return {k: v/max(1, counts[k]) for k, v in losses.items()}, counts


def representation_loss(system, batch):
    cfg = system.cfg
    # Include terminal visits in self-supervision without using them as inputs
    # to the separate future-prediction branch.
    use_target = bool(torch.rand(()) < .5)
    observed = batch["target"] if use_target else batch["source"]
    aux = batch["target_aux"] if use_target else batch["source_aux"]
    state = system.encoder(observed)
    source_state = system.encoder(batch["source"]) if use_target else state
    with torch.no_grad():
        target_state = system.target_encoder(batch["target"])
        teacher_observed = target_state if use_target else system.target_encoder(batch["source"])
    corrupted, mask = corrupt_latent(observed, cfg.encoder)
    context = system.encoder(corrupted)
    predicted = system.masked_predictor(context)
    jepa = F.smooth_l1_loss(predicted.float(), teacher_observed.dense.float(), reduction="none").mean(-1)
    jepa = (jepa*mask).sum()/mask.sum().clamp_min(1)
    clinical = system.clinical(batch["conditions"])
    future = system.future_predictor(source_state, clinical)
    losses = {
        "reconstruction": F.smooth_l1_loss(system.heads.volume("reconstruction", state), F.adaptive_avg_pool3d(observed, state.grid)),
        "latent_delta": F.smooth_l1_loss(system.heads.volume("latent_delta", state), F.adaptive_avg_pool3d(system.encoder.raw_differences(observed), state.grid)),
        "jepa": jepa,
        "future": F.smooth_l1_loss(future.float(), target_state.disease.float()),
    }
    queued = system.moment_queue[:int(system.queue_count)].detach().clone()
    moment_batch = torch.cat((source_state.summary.float(), queued), 0)
    var_global, cov_global = variance_covariance(moment_batch)
    var_dense, cov_dense = variance_covariance(state.dense)
    losses["variance"], losses["covariance"] = var_global+var_dense, cov_global+cov_dense
    losses["separation"] = cross_covariance(moment_batch)
    losses["anatomy"] = zero_loss(state.dense)
    n = 0
    for i, rec in enumerate(batch["records"]):
        if rec.get("anatomy_comparable", False):
            losses["anatomy"] = losses["anatomy"]+F.smooth_l1_loss(source_state.anatomy[i].mean(0), target_state.anatomy[i].mean(0))
            n += 1
    losses["anatomy"] = losses["anatomy"]/max(n, 1)
    labels, counts = auxiliary_losses(system.heads, state, aux)
    losses.update(labels)
    risk = system.heads.risk(source_state, clinical)
    losses["pcr"], npcr = zero_loss(risk), 0
    for i, a in enumerate(batch["source_aux"]):
        if "pcr" in a:
            losses["pcr"] = losses["pcr"]+F.binary_cross_entropy_with_logits(risk[i], a["pcr"].to(risk))
            npcr += 1
    losses["pcr"] = losses["pcr"]/max(1, npcr)
    total = sum(getattr(cfg.loss, k)*v for k, v in losses.items())
    metrics = {k: float(v.detach()) for k, v in losses.items()}
    metrics.update({f"label_count/{k}": v for k, v in counts.items()})
    metrics["label_count/pcr"] = npcr
    return total, metrics, source_state.summary.detach()


def pair_flow_loss(model, teacher, batch, step, *, codec=None, auxiliary_active=None):
    cfg = model.cfg
    path = make_path(batch["source"], batch["target"])
    reverse = bool(torch.rand(()) < cfg.training.reverse_probability)
    direction = -1 if reverse else 1
    observed = batch["target"] if reverse else batch["source"]
    context = model.context(observed, batch["conditions"], direction)
    velocity = model.velocity(path.joint, path.tau, context)
    diff = (velocity.float()-path.velocity.float()).square()
    branch_x = diff[:, :24].flatten(1).mean(1)
    branch_y = diff[:, 24:].flatten(1).mean(1)
    quality = diff.new_tensor([r.get("qc_weight", 1) for r in batch["records"]])
    losses = {"velocity": ((branch_x+branch_y)*quality).mean()}
    active = step >= cfg.training.auxiliary_warmup and step % cfg.training.auxiliary_every == 0
    if auxiliary_active is not None:
        active = auxiliary_active
    losses["semantic_endpoint"] = zero_loss(velocity)
    losses["local_map"] = zero_loss(velocity)
    losses["decoded_kinetics"] = zero_loss(velocity)
    counts = {}
    if active:
        earlier, later = estimate_endpoints(path.joint, velocity, path.tau)
        prediction = earlier if reverse else later
        real = batch["source"] if reverse else batch["target"]
        real_aux = batch["source_aux"] if reverse else batch["target_aux"]
        if cfg.loss.semantic_endpoint or cfg.loss.kinetics or cfg.loss.segmentation or cfg.loss.biomarker:
            pred_state = model.encoder(prediction)
            with torch.no_grad():
                true_state = model.encoder(real)
            losses["semantic_endpoint"] = cosine_distance(pred_state.dense, true_state.dense)
            extra, counts = auxiliary_losses(model.heads, pred_state, real_aux)
            # External teacher distillation trains the encoder in stage A only;
            # the frozen external projection is not an arbitrary new critic.
            extra.pop("external_teacher")
            losses.update(extra)
        if cfg.loss.local_map:
            with torch.no_grad():
                teacher_context = teacher.context(observed, batch["conditions"], direction)
                mapped, dt = local_teacher_map(teacher.velocity, path.joint, path.tau, teacher_context, cfg.rollout.local_map_delta)
            dtb = expand_time(dt, path.joint)
            error = ((path.joint+dtb*velocity-mapped)/dtb.clamp_min(1e-3)).float().square().flatten(1).mean(1)
            mask = dt >= 1e-3
            losses["local_map"] = (error*mask).sum()/mask.sum().clamp_min(1)
        if cfg.loss.decoded_kinetics:
            if codec is None:
                raise ValueError("decoded_kinetics requires the user's matching frozen VQ checkpoint")
            losses["decoded_kinetics"] = decoded_kinetic_loss(codec, prediction, real, model.encoder)
    total = sum(getattr(cfg.loss, k)*v for k, v in losses.items())
    metrics = {k: float(v.detach()) for k, v in losses.items()}
    metrics.update(velocity_x=float(branch_x.mean().detach()), velocity_y=float(branch_y.mean().detach()),
                   direction=direction, auxiliary_active=int(active))
    metrics.update({f"label_count/{k}": v for k, v in counts.items()})
    return total, metrics


def distribution_mmd(x, y, bandwidths=(.25, .5, 1, 2)):
    """Biased RBF MMD per patient; sample axes may NEVER mix patients."""
    if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError("MMD inputs must be [patients,samples,features]")
    if x.shape[1] < 2 or y.shape[1] < 2:
        raise ValueError("A stochastic consistency batch needs at least 2 samples")
    x, y = x.float(), y.float()
    dim = x.shape[-1]
    def kernel(a,b):
        sq = torch.cdist(a,b).square()/dim
        return sum(torch.exp(-sq/(2*s*s)) for s in bandwidths)/len(bandwidths)
    return (kernel(x,x).mean((1,2))+kernel(y,y).mean((1,2))-2*kernel(x,y).mean((1,2))).mean()


def energy_score(samples, target):
    """Unbiased ensemble energy score, a proper distributional scoring objective.

    It does not tell every stochastic sample to equal the one observed target.
    Finite-sample unbiased estimates can occasionally be slightly negative.
    """
    if samples.ndim != 3 or target.shape != (samples.shape[0], samples.shape[-1]) or samples.shape[1] < 2:
        raise ValueError("Energy score needs samples [B,K>=2,F], targets [B,F]")
    x, y = samples.float(), target.float()
    k, f = x.shape[1], x.shape[-1]
    distance = torch.linalg.vector_norm(x-y[:, None], dim=-1).mean(1)/math.sqrt(f)
    pairwise = torch.cdist(x,x).sum((1,2))/(k*(k-1)*math.sqrt(f))
    return (distance-.5*pairwise).mean()


def rollout_loss(model, teacher, store, triplet, *, device):
    """Source-only stochastic two-hop predictions vs independent direct samples."""
    ab, bc, ac = triplet
    cfg = model.cfg
    source, c01 = store.source_batch(ab, device)
    c12, c02 = [bc["conditions"]], [ac["conditions"]]
    # True intermediate MRI is NOT read or fed back into the student rollout.
    composed_features, direct_features = [], []
    for _ in range(cfg.rollout.samples):
        z1 = model.sample(source, c01, steps=cfg.rollout.steps, method="heun")
        z2 = model.sample(z1, c12, steps=cfg.rollout.steps, method="heun")
        composed_features.append(model.encoder(z2).disease.flatten(1))
        with torch.no_grad():
            direct = teacher.sample(source, c02, steps=cfg.rollout.steps, method="heun")
            direct_features.append(teacher.encoder(direct).disease.flatten(1))
    composed = torch.stack(composed_features, 1)
    direct = torch.stack(direct_features, 1)
    # A real terminal state is a supervision target, not an inference input.
    with torch.no_grad():
        actual = store.normalize(store.read_raw(ac["target"]))[None].to(device)
        actual_feature = teacher.encoder(actual).disease.flatten(1)
    comp = distribution_mmd(composed, direct, cfg.rollout.mmd_bandwidths)
    anchor = energy_score(composed, actual_feature)
    total = cfg.loss.composition*comp+cfg.loss.rollout_anchor*anchor
    return total, {"composition": float(comp.detach()), "rollout_anchor": float(anchor.detach()),
                   "rollout_samples": cfg.rollout.samples, "rollout_solver_steps": cfg.rollout.steps}


def enhancement_differences(images):
    if images.ndim != 5 or images.shape[1] != 3:
        raise ValueError("Kinetic differences require joint [B,3,D,H,W] images")
    p, e, l = images.split(1, 1)
    return torch.cat((e-p, l-p, l-e), 1)


def decoded_kinetic_loss(codec, predicted, target, encoder):
    # Same patch in predicted and target decoder coordinates; never claim this
    # proxy supervises true MRI physiology without a measured image target.
    sizes = [min(n, lim) for n, lim in zip(predicted.shape[2:], (8,16,16))]
    starts = [int(torch.randint(max(1, n-s+1), ())) for n,s in zip(predicted.shape[2:], sizes)]
    sl = (slice(None), slice(None), *[slice(a,a+s) for a,s in zip(starts,sizes)])
    p = predicted[sl]*encoder.latent_std+encoder.latent_mean
    t = target[sl]*encoder.latent_std+encoder.latent_mean
    pi = codec.decode(p)
    with torch.no_grad():
        ti = codec.decode(t)
    # Suppress decoder padding artifacts by omitting the outer decoded voxels.
    crop = (slice(None), slice(None), *[slice(4,-4) if n > 8 else slice(None) for n in pi.shape[2:]])
    return F.smooth_l1_loss(enhancement_differences(pi[crop]).float(), enhancement_differences(ti[crop]).float())
