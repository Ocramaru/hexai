"""What a run is scored on: the baselines, and one pass over the val set.

Author: Marco Cassar (@Ocramaru)
"""

import torch
import torch.nn.functional as F

from hexai.data import split_context_target, target_view
from hexai.models.vvae.visuals import row_corr
from hexai.utilities import answerable_mask


def masked_mean(total, n, fraction):
    """A masked error summed over a set, put back on the same scale as an unmasked one."""
    return total / max(n * fraction, 1e-6)


def masked_error(guess, truth, mask):
    """Squared error over answerable pixels, summed across the batch."""
    return (((guess - truth) ** 2) * mask.unsqueeze(1)).flatten(1).sum(1).sum().item()


@torch.no_grad()
def baselines(model, loader, device, target):
    """What trivial answers score, so recon has a zero point. Returns (whole_frame, answerable,
    copy_anchor, graded_fraction), the last three measured only over answerable pixels."""
    truths = torch.cat([split_context_target(f, p, 1, target)[2] for f, p in loader])
    mean_image = truths.mean(0, keepdim=True)
    whole = F.mse_loss(mean_image.expand_as(truths), truths,
                       reduction="none").flatten(1).sum(1).mean().item()

    mean_seen, copy_seen, n, graded = 0.0, 0.0, 0, 0.0
    for f, p in loader:
        context, context_poses, truth, query = split_context_target(f, p, 1, target)
        context, context_poses = context.to(device), context_poses.to(device)
        truth, query = truth.to(device), query.to(device)
        carved = model.encoder.carve(context, context_poses)  # geometry, not weights
        mask = answerable_mask(carved, truth, query, model.config.bounds, *model.camera)
        anchor = int(context_poses[0, :, :3].norm(dim=1).argmin())  # its pose is identity
        mean_seen += masked_error(mean_image.to(truth), truth, mask)
        copy_seen += masked_error(target_view(context[:, :, anchor], target), truth, mask)
        graded += mask.mean().item() * len(f)
        n += len(f)
    fraction = graded / n
    return whole, masked_mean(mean_seen, n, fraction), masked_mean(copy_seen, n, fraction), fraction


def unknown_scores(pred, was_seen):
    """(mae, gap) for the observedness channel. The gap is mean prediction where the carving saw
    minus where it did not, which a constant cannot fake and mae alone cannot see."""
    hit = was_seen > 0.5
    gap = ((pred * hit).sum() / hit.sum().clamp(min=1)
           - (pred * ~hit).sum() / (~hit).sum().clamp(min=1))
    return (pred - was_seen).abs().mean().item(), gap.item()


def lift_fidelity(model, context, context_poses):
    """How much a learned lift's volume resembles the one exact geometry would have produced, or
    None when the lift is that geometry. The encoder's own lift stays geometric whichever is in
    use, so this is the only thing that says whether the learned one places mass or routes
    around the problem. Not a target to maximise: a lift that only reproduces the hard one has
    bought nothing, and the point of learning it is to survive depth the hard one cannot."""
    if model.encoder.learned_lift is None: return None
    learned = model.encoder.learned_lift(context, context_poses)[:, -1].flatten(1)
    exact = model.encoder.lift(context, context_poses)[:, -1].flatten(1)
    return row_corr(learned, exact).mean().item()


def volume_scores(model, mu, carved, context, context_poses):
    """(lift agreement, invented density, decoded volumes). Agreement is over voxels a ray reached,
    so it is blind to invention by construction, which is why the second number exists."""
    decoded = model.decoder.volume(mu)[:, model.config.density_index].flatten(1)
    lifted = model.encoder.lift(context, context_poses)[:, -1].flatten(1)
    seen = lifted > 1e-3  # only where rays landed; most of the box is unobserved, not empty
    never = carved[:, 1].flatten(1) < 0.5
    agree = row_corr(decoded * seen, lifted * seen).mean().item()
    invented = (decoded * never).sum().item() / never.sum().clamp(min=1).item()
    return agree, invented, decoded


def scene_similarity(decoded):
    """How alike different scenes in a batch decode. Near 1 means the decoder ignores its input."""
    if len(decoded) < 2: return None
    c = torch.corrcoef(decoded).fill_diagonal_(0)
    return (c.sum() / (len(decoded) * (len(decoded) - 1))).item()


@torch.no_grad()
def validate(model, loader, device, target, beta, floors, loss_fn):
    """One pass over the val set -> the val/ metrics, ready to log. floors comes from baselines()."""
    whole_floor, seen_floor, copy_floor, _ = floors
    model.eval()
    totals, n, graded, seen_px = [0.0, 0.0, 0.0], 0, 0.0, 0.0
    corr, agree, mae, invent, unknown, gaps, fidelity, occluded = [], [], [], [], [], [], [], []
    for frames, poses in loader:
        context, context_poses, truth, query = split_context_target(frames, poses, 1, target)
        context, context_poses = context.to(device), context_poses.to(device)
        truth, query = truth.to(device), query.to(device)
        carved = model.encoder.carve(context, context_poses)  # gates the occlusion charge below
        out = model(context, context_poses, query, gate=carved[:, 1:2])
        losses = loss_fn(out, truth, beta)  # always mse and unpenalised, so runs compare
        if out.occluded is not None: occluded.append(out.occluded.mean().item())
        totals = [t + v.item() * len(frames) for t, v in zip(totals, losses, strict=True)]
        n += len(frames)
        if target != "rgb":  # depth in metres, the one number with physical meaning
            mae.append(((out.x_hat[:, -1] - truth[:, -1]).abs().mean() * model.config.depth_max).item())

        mask = answerable_mask(carved, truth, query, model.config.bounds, *model.camera)
        graded += masked_error(out.x_hat, truth, mask)
        seen_px += mask.mean().item() * len(frames)

        agreement, invented, decoded = volume_scores(model, out.mu, carved, context, context_poses)
        agree.append(agreement)
        invent.append(invented)
        if (similarity := scene_similarity(decoded)) is not None: corr.append(similarity)
        if (fidel := lift_fidelity(model, context, context_poses)) is not None:
            fidelity.append(fidel)
        if model.config.predict_unknown:
            err, gap = unknown_scores(model.decoder.volume(out.mu)[:, -1].flatten(1),
                                      carved[:, 1].flatten(1))
            unknown.append(err)
            gaps.append(gap)

    loss, recon, kl = [t / n for t in totals]
    answerable = masked_mean(graded, n, seen_px / n)
    mean = lambda xs: sum(xs) / max(len(xs), 1)  # noqa: E731
    metrics = {"val/loss": loss, "val/recon": recon, "val/kl": kl,
               "val/skill": 1 - recon / whole_floor,
               "val/recon_answerable": answerable,
               "val/skill_answerable": 1 - answerable / seen_floor,
               "val/skill_vs_copy": 1 - answerable / copy_floor,
               "val/scene_corr": mean(corr),
               "val/lift_agreement": mean(agree),
               "val/invented_density": mean(invent),
               "val/occluded_density": mean(occluded)}
    if mae: metrics["val/depth_mae_m"] = mean(mae)
    if unknown: metrics["val/unknown_mae"], metrics["val/unknown_gap"] = mean(unknown), mean(gaps)
    if fidelity: metrics["val/lift_fidelity"] = mean(fidelity)
    return metrics


def summarise(epoch, metrics):
    """The one line an epoch prints. Every number here has a zero point or a unit."""
    unknown = f", unknown gap {metrics['val/unknown_gap']:+.3f}" if "val/unknown_gap" in metrics else ""
    fidelity = f", lift fid {metrics['val/lift_fidelity']:.3f}" if "val/lift_fidelity" in metrics else ""
    return (f"epoch {epoch}: val loss {metrics['val/loss']:.4g} (recon {metrics['val/recon']:.4g}, "
            f"kl {metrics['val/kl']:.4g}), scene corr {metrics['val/scene_corr']:.3f}, "
            f"skill {metrics['val/skill']:+.3f}/{metrics['val/skill_answerable']:+.3f}, "
            f"vs copy {metrics['val/skill_vs_copy']:+.3f}, "
            f"lift agree {metrics['val/lift_agreement']:.3f}, "
            f"answerable {metrics['val/recon_answerable']:.1f}, "
            f"invented {metrics['val/invented_density']:.3f}, "
            f"occluded {metrics['val/occluded_density']:.3f}{unknown}{fidelity}")
