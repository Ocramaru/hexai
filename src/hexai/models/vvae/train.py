"""Training loop for the geometric VVAE. Runs detached, so it survives a closed laptop.

One run:

    uv run python -m hexai.models.vvae.train --lr 0.003

A sweep, defined by sweep.json alongside this file:

    wandb sweep src/hexai/models/vvae/sweep.json
    tmux new -s geo
    wandb agent <entity>/<project>/<sweep_id>

Checkpoints go to runs/vvae/, named after the flags that vary. Findings live in
.agents/learned.md.

Author: Marco Cassar (@Ocramaru)
"""

import os

os.environ["MPLBACKEND"] = "Agg"  # no display on a detached session

import argparse
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_angle_axis, quat_mul
from tqdm import tqdm as _tqdm

from hexai.models.utilities import AdaptiveRestarts
from hexai.models.vvae import VVAEGeometric, VVAEGeometricConfig
from hexai.models.vvae.data import (
    TartanGroundWindowDataset,
    add_pose_noise,
    find_trajectory_dirs,
    make_dataloader,
    split_context_target,
    target_view,
)
from hexai.models.vvae.geometry import pixel_to_camera_ray

ROOT = Path(__file__).resolve().parents[4]
HOUSE = Path(os.environ.get("TARTAN_ROOT", ROOT / "data/tartanground")) / "House/Data_omni"
CHECKPOINTS = Path(__file__).parent / ".runs"  # gitignored, beside the code it belongs to

# a bar redraws over itself on a terminal but floods a redirected log, so drop it when piped
tqdm = partial(_tqdm, file=sys.stdout, ncols=100, mininterval=2.0, disable=not sys.stdout.isatty())

DEPTH_CMAP = 'magma'  # every depth panel, so one edit changes them all

JITTER_FOVY = 49.0  # the hexapod's lens; --jitter re-aims the head at this fov and masks

METRIC_NOTES = """\
wandb has no per-metric description, so the glossary lives here.

skill             0 is no better than predicting the val set's mean image, 1 is perfect.
                  The scale for recon, which has none on its own.
recon_answerable  recon over pixels a context frame actually saw, per pixel. The fair
                  comparison when some runs train with the mask and some do not.
lift_agreement    does the decoded volume match the geometry that went in, over observed
                  voxels only. Blind to unobserved space by construction.
invented_density  density the decoder puts where no camera reached. Lower is better, but an
                  empty volume wins it outright, so read it beside lift_agreement.
scene_corr        how alike different scenes decode. Near 1 means the decoder is ignoring
                  its input. A model emitting noise also scores well, so never read alone.
unknown_mae       how well the observedness channel matches the carving. Only with unknown=1.
depth_mae_m       depth error in metres, the one number with a physical unit.

The *_best variants are running bests, because the summary a sweep reads keeps the last
logged value and these swing between epochs.
"""


def vvae_loss(out, x, beta=1.0, observed=None, mask=None, kl_floor=0.1):
    """Recon + KL, both summed per sample then meaned over the batch.

    mask drops target pixels, [B, H, W]. observed is [B, *latent_grid] in [0, 1] and scales the KL
    per latent voxel: kl_floor where seen, full beta where not."""
    err = F.mse_loss(out.x_hat, x, reduction='none')
    recon = (err if mask is None else err * mask.unsqueeze(1)).flatten(1).sum(1)
    kl = -0.5 * (1 + out.logvar - out.mu.pow(2) - out.logvar.exp())
    if observed is not None:
        weight = observed.mul(kl_floor - 1).add(1.0)  # kl_floor where seen, 1 where not
        kl = kl * weight.unsqueeze(1)
    return (recon + beta * kl.flatten(1).sum(1)).mean(), recon.mean(), kl.flatten(1).sum(1).mean()


SCHEDULERS = ("none", "onecycle", "adaptive")


METRIC_DRIVEN = (AdaptiveRestarts,)


def make_scheduler(name, optimizer, lr, total_steps, epochs=40):
    """onecycle warms up over the first 30% then anneals to nothing. adaptive warms up the same
    way but holds rate in reserve, and steps on the validation metric rather than per batch."""
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
    if name == "adaptive":
        # decay is sized to the run: with no restarts the rate lands near 1% of peak on the last
        # epoch. A fixed decay bottoms out early on a long run and wastes its tail
        return AdaptiveRestarts(optimizer, lr, horizon=epochs, warmup=max(1, round(0.3 * epochs)))
    return None


def out_vol(model, z):
    """The decoded grid. decoder.volume is the raw grid; forward also returns it, so this exists
    only for the places that decode a latent on their own."""
    return model.decoder.volume(z)


@torch.no_grad()
def answerable_mask(model, carved, truth, query):
    """[B, H, W] in [0, 1]: for each target pixel, how much the context frames observed the point
    it sits on. Read from the carving, not the model's own coverage, which would be gameable."""
    cfg = model.config
    h, w = cfg.render_hw
    k_inv = pixel_to_camera_ray((h, w), cfg.source_fovy_deg).to(truth)
    v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    pix = torch.stack([u.flatten(), v.flatten(), torch.ones(h * w)]).to(truth)
    rays = (k_inv @ pix)[None]                                   # [1, 3, H*W]
    metres = (truth[:, -1] * cfg.depth_max).flatten(1)[:, None]  # [B, 1, H*W]
    local = rays * metres
    rot = torch.stack([matrix_from_quat(q[3:]) for q in query])
    pts = rot @ local + query[:, :3, None]                       # canonical frame

    lower = cfg.bounds[0].to(truth)[None, :, None]
    span = (cfg.bounds[1] - cfg.bounds[0]).to(truth)[None, :, None]
    unit = ((pts - lower) / span * 2 - 1).flip(1)                # grid_sample reads xyz reversed
    grid = unit.permute(0, 2, 1)[:, None, None]                  # [B, 1, 1, H*W, 3]
    seen = F.grid_sample(carved[:, 1:], grid, align_corners=False, padding_mode='zeros')
    return seen.view(-1, h, w).clamp(0, 1)


@torch.no_grad()
def observed_weight(model, context, context_poses, kl_mode, carved=None):
    """[B, *latent_grid] of how much of each latent voxel's region was seen, or None for a flat KL.
    Pooled down from the carving by the same factor the encoder downsamples by, so one weight lands
    on one latent voxel."""
    if kl_mode != "observed":
        return None
    seen = (model.encoder.carve(context, context_poses) if carved is None else carved)[:, 1:]
    shrink = model.config.downsample_scale ** model.config.n_downsamples
    return F.avg_pool3d(seen, shrink)[:, 0]


def _yaw_view(frame, fovy_deg, yaw_deg):
    """[4, H, W] re-rendered as the same camera turned about its own optical centre. No parallax in
    a pure rotation, so the homography is exact; depth divides by the rotated ray's z because it
    measures along the optical axis and that axis moved. Edges that leave the source go black."""
    _, h, w = frame.shape
    k_inv = pixel_to_camera_ray((h, w), fovy_deg).to(frame)
    v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    pix = torch.stack([u.flatten(), v.flatten(), torch.ones(h * w)]).to(frame)
    a = torch.deg2rad(torch.tensor(float(yaw_deg)))
    rot = torch.tensor([[a.cos(), 0, a.sin()], [0, 1, 0], [-a.sin(), 0, a.cos()]]).to(frame)
    rays = rot @ (k_inv @ pix)
    src = torch.linalg.inv(k_inv) @ rays
    xy = (src[:2] / src[2]) / torch.tensor([[w - 1], [h - 1]]).to(frame) * 2 - 1
    out = F.grid_sample(frame[None], xy.T.view(1, h, w, 2), align_corners=True)[0]
    return torch.cat([out[:3], out[3:] / rays[2].view(1, h, w).clamp(min=1e-6)]).clamp(0, 1)


def _row_corr(a, b):
    """Pearson correlation between matching rows of two [N, D] tensors."""
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-8)


def _show(ax, img, title):
    """[C, H, W] in [0, 1]. Three channels are rgb, anything else is colour-mapped depth, which is
    always the last channel. Depth is pinned to 0-1 so one colour is one distance in every panel."""
    a = img.permute(1, 2, 0).numpy()
    if a.shape[2] == 3:
        ax.imshow(a)
    else:
        ax.imshow(a[..., -1], cmap=DEPTH_CMAP, vmin=0.0, vmax=1.0)
    ax.set_title(title)


def _floor_plan(ax, density, pose, config, title):
    """Density summed down the vertical axis, so it reads as a floor plan rather than a view. The
    box is the window's middle camera, and the arrow is where the query camera sits inside it and
    which way it looks."""
    ax.imshow(density.sum(1).T.numpy(), origin="lower", cmap="magma", aspect="auto")
    lower, size = config.bounds[0], config.bounds[1] - config.bounds[0]
    grid = torch.tensor(config.grid_resolution, dtype=torch.float32)
    x, _, z = (pose[:3].cpu() - lower) / size * grid
    heading = matrix_from_quat(pose[3:].cpu()) @ torch.tensor([0.0, 0.0, 1.0])
    ax.arrow(float(x), float(z), float(heading[0]) * 5, float(heading[2]) * 5,
             head_width=1.4, color="cyan", length_includes_head=True)
    ax.set_title(title)


@torch.no_grad()
def log_novel_view(model, batch, step=None, key='novel', target=1, mode='rgb'):
    """Truth against the render, and the lifted volume against the decoded one. The image pair alone
    cannot tell undertrained from broken -- a saturated density field also renders flat grey -- so
    the volumes go beside it, and the density fractions go up as scalars."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    frames, poses = batch
    context, context_poses, truth, query = split_context_target(frames[:1], poses[:1], target, mode)
    context, context_poses = context.to(device), context_poses.to(device)
    out = model(context, context_poses, query.to(device))
    lifted = model.encoder.lift(context, context_poses)[0].cpu()
    decoded = out_vol(model, out.mu)[0].cpu()
    if was_training: model.train()

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    _show(axes[0], truth[0], "truth, held out")
    _show(axes[1], out.x_hat[0].cpu().clamp(0, 1), "rendered from its pose")
    _floor_plan(axes[2], lifted[3], query[0], model.config, "lifted volume, top-down")
    _floor_plan(axes[3], decoded[model.config.density_index], query[0], model.config,
                "decoded volume, top-down")
    for ax in axes: ax.set(xticks=[], yticks=[])
    fig.tight_layout()

    if wandb.run is not None:
        wandb.log({key: wandb.Image(fig),
                   "volume/lifted_occupied": (lifted[3] > 1e-3).float().mean().item(),
                   "volume/decoded_solid": (decoded[model.config.density_index] > 0.5).float().mean().item(),
                   "volume/decoded_mean": decoded[model.config.density_index].mean().item()}, step=step)
    plt.close(fig)


@torch.no_grad()
def log_pose_sweep(model, batch, step=None, key='sweep', target=1, mode='rgb', yaws=(-24, -12, 12, 24)):
    """Truth over render, angle by angle. The latent is encoded once and rendered from each yaw, so
    a row that stays the same across the sweep is a latent that has stopped depending on viewpoint,
    which no single-pose panel can show. Truth is the target frame turned about its optical centre,
    which is exact for a rotation."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    frames, poses = batch
    context, context_poses, _, query = split_context_target(frames[:1], poses[:1], target, mode)
    mu, _ = model.encoder(context.to(device), context_poses.to(device))

    up = torch.tensor([0.0, 1.0, 0.0], device=device)  # camera y is down, so yaw turns about it
    # a yaw past a third of the fov aims the truth panel off the source frame
    limit = model.config.source_fovy_deg / 3
    offsets = sorted({round(max(-limit, min(limit, y))) for y in (*yaws, 0.0)})
    source = frames[0, :, target]  # the full rgb+depth frame, before the target mode trims it
    truths, renders = [], []
    for deg in offsets:
        turn = quat_from_angle_axis(torch.deg2rad(torch.tensor([float(deg)], device=device)), up)
        q = query.to(device).clone()
        q[:, 3:] = quat_mul(q[:, 3:], turn)
        renders.append(model.decoder(mu, q)[0][0].cpu().clamp(0, 1))  # (image, volume)
        truths.append(target_view(_yaw_view(source, model.config.source_fovy_deg, deg)[None], mode)[0])
    if was_training: model.train()

    fig, axes = plt.subplots(2, len(offsets), figsize=(2.6 * len(offsets), 5.6))
    for col, deg in enumerate(offsets):
        label = "its pose" if deg == 0 else f"{deg:+.0f} deg"
        _show(axes[0, col], truths[col], f"truth {label}")
        _show(axes[1, col], renders[col], f"render {label}")
    for ax in axes.ravel(): ax.set(xticks=[], yticks=[])
    fig.tight_layout()

    if wandb.run is not None:
        flat = torch.stack([r.flatten() for r in renders])
        spread = _row_corr(flat[:-1], flat[1:]).mean().item()  # 1.0 means the pose changes nothing
        wandb.log({key: wandb.Image(fig), f"volume/viewpoint_sameness_{key[-1]}": spread}, step=step)
    plt.close(fig)
    return fig


def train_geometric(model, train_loader, test_loader, name=None, epochs=20, lr=0.001, beta=1.0,
                    device='cuda', n_epochs=5, pose_noise=0.02, project="vvae",
                    scheduler="none", target="rgb", scenes=3, kl_mode="uniform", mask_unknown=False, unknown_weight=1.0,
                    resume=None, taper=False):
    dataset = train_loader.dataset.dataset  # Subset -> the dataset it indexes into
    assert dataset.camera == model.camera, f"dataset {dataset.camera} != model {model.camera}"

    config = {  # names match sweep.json's parameters, so an agent's values land on the same keys
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "lr": lr,
        "beta": beta,
        "kl_mode": kl_mode,
        "mask_unknown": mask_unknown,
        "predict_unknown": model.config.predict_unknown,
        "scheduler": scheduler,
        "target": target,
        "pose_noise": pose_noise,
        "steps_per_epoch": len(train_loader),
        "camera_fovy": dataset.camera[1],
        "num_frames": dataset.num_frames,
        "head_jitter": dataset.include_head_jitter,
        "grid": model.config.grid_resolution,
        "latent": model.config.latent_shape,
        "base_channels": model.config.base_channels,
        "groups": model.config.groups,
        "enc_params": sum(p.numel() for p in model.encoder.parameters()),
        "camera": model.camera,
        "model": model.__class__.__name__,
        "optimizer": "Adam",
    }

    CHECKPOINTS.mkdir(exist_ok=True)
    run = wandb.init(project=project, name=name, config=config, notes=METRIC_NOTES)
    try:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        sched = make_scheduler(scheduler, optimizer, lr, epochs * len(train_loader), epochs)
        per_batch_lr = sched is not None and not isinstance(sched, METRIC_DRIVEN)

        # epoch as its own axis; runs with different steps_per_epoch only align on it
        run.define_metric("epoch")
        for section in ("val/*", "volume/*", "train/recon_epoch", "train/lr"):
            run.define_metric(section, step_metric="epoch")
        # wandb keeps the running best itself, under a suffixed key: val/lift_agreement.max and
        # val/scene_corr.min. A sweep should point at those rather than a hand-tracked _best
        run.define_metric("val/lift_agreement", step_metric="epoch", summary="max")
        run.define_metric("val/scene_corr", step_metric="epoch", summary="min")
        run.define_metric("val/skill", step_metric="epoch", summary="max")
        run.define_metric("val/invented_density", step_metric="epoch", summary="min")

        # fixed indices across the held-out trajectory, so every run logs the same scenes
        picks = torch.linspace(0, len(test_loader.dataset) - 1, scenes).round().long().tolist()
        samples = [tuple(x.unsqueeze(0) for x in test_loader.dataset[i]) for i in picks]
        print(f'logging scenes {picks} of {len(test_loader.dataset)} val windows', flush=True)
        global_step, best_corr, best_agree, first_epoch = 1, float("inf"), -1.0, 0
        if resume:  # weights alone are not enough: a schedule mid-slope needs its own state back
            ckpt = torch.load(resume, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if sched and ckpt.get("scheduler"): sched.load_state_dict(ckpt["scheduler"])
            global_step, best_corr = ckpt["global_step"], ckpt["best_corr"]
            best_agree = ckpt.get("best_agree", -1.0)
            first_epoch = ckpt["epoch"]
            print(f'resumed {resume} at epoch {first_epoch}, lr {optimizer.param_groups[0]["lr"]:.2e}',
                  flush=True)
        if taper and isinstance(sched, AdaptiveRestarts):
            sched.taper(over=epochs)  # stop holding rate in reserve, anneal out from here
            print(f'tapering to {sched.floor:.0e} over {epochs} epochs', flush=True)

        # what predicting the val set's mean image scores, the zero point for val/skill
        with torch.no_grad():
            truths = torch.cat([split_context_target(f, p, 1, target)[2] for f, p in test_loader])
            floor = F.mse_loss(truths.mean(0, keepdim=True).expand_as(truths), truths,
                               reduction="none").flatten(1).sum(1).mean().item()
        print(f'mean-image baseline: recon {floor:.1f}', flush=True)

        for epoch in range(first_epoch, first_epoch + epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{first_epoch + epochs}')
            train_recon, train_batches = 0.0, 0

            for i, (frames, poses) in enumerate(pbar, 1):
                poses = add_pose_noise(poses, pose_noise)  # the pose quality deployment will have
                context, context_poses, truth, query = split_context_target(frames, poses, mode=target)
                optimizer.zero_grad()

                context, context_poses = context.to(device), context_poses.to(device)
                truth, query = truth.to(device), query.to(device)
                out = model(context, context_poses, query)
                needs_carve = mask_unknown or kl_mode == "observed" or model.config.predict_unknown
                carved = model.encoder.carve(context, context_poses) if needs_carve else None
                mask = answerable_mask(model, carved, truth, query) if mask_unknown else None
                t_loss, t_recon, t_kl = vvae_loss(
                    out, truth, beta,
                    observed_weight(model, context, context_poses, kl_mode, carved), mask)
                if model.config.predict_unknown and carved is not None:
                    # the unknown head, supervised everywhere against the carving
                    t_loss = t_loss + unknown_weight * F.mse_loss(
                        out.volume[:, -1], carved[:, 1], reduction="none").flatten(1).sum(1).mean()
                t_loss.backward()
                optimizer.step()
                if sched and not isinstance(sched, METRIC_DRIVEN): sched.step()

                train_recon, train_batches = train_recon + t_recon.item(), train_batches + 1
                pbar.set_postfix(loss=f'{t_loss.item():.3g}', recon=f'{t_recon.item():.3g}', kl=f'{t_kl.item():.3g}')
                step_log = {"train/loss": t_loss.item(), "train/recon": t_recon.item(),
                            "train/kl": t_kl.item(), "epoch": epoch + i / len(train_loader)}
                if per_batch_lr:  # a metric-driven schedule only moves once an epoch
                    step_log["train/lr"] = optimizer.param_groups[0]["lr"]
                run.log(step_log, step=global_step)
                global_step += 1

            model.eval()
            totals, n, scene_corr, agree, mae = [0.0, 0.0, 0.0], 0, [], [], []
            answerable, seen_px, invent, unknown_err = 0.0, 0.0, [], []
            with torch.no_grad():
                for frames, poses in test_loader:
                    context, context_poses, truth, query = split_context_target(frames, poses, 1, target)
                    context, context_poses = context.to(device), context_poses.to(device)
                    truth, query = truth.to(device), query.to(device)
                    out = model(context, context_poses, query)
                    losses = vvae_loss(out, truth, beta)  # always mse, so runs compare
                    totals = [t + v.item() * len(frames) for t, v in zip(totals, losses, strict=True)]
                    n += len(frames)
                    if target != "rgb":  # depth in metres, the one number with physical meaning
                        mae.append(((out.x_hat[:, -1] - truth[:, -1]).abs().mean()
                                    * model.config.depth_max).item())

                    # both maskings every run, so masked and unmasked cells stay comparable
                    carved_v = model.encoder.carve(context, context_poses)
                    vmask = answerable_mask(model, carved_v, truth, query)
                    sq = (out.x_hat - truth) ** 2
                    answerable += (sq * vmask.unsqueeze(1)).flatten(1).sum(1).sum().item()
                    seen_px += vmask.mean().item() * len(frames)

                    decoded = out_vol(model, out.mu)[:, model.config.density_index].flatten(1)
                    lifted = model.encoder.lift(context, context_poses)[:, -1].flatten(1)
                    # only where rays landed; most of the box is unobserved, not empty
                    seen = lifted > 1e-3
                    agree.append(_row_corr(decoded * seen, lifted * seen).mean().item())

                    # density put where no camera reached; lift_agreement cannot see this
                    never = carved_v[:, 1].flatten(1) < 0.5
                    invent.append((decoded * never).sum().item() / never.sum().clamp(min=1).item())
                    if model.config.predict_unknown:
                        pred = out_vol(model, out.mu)[:, -1].flatten(1)
                        unknown_err.append((pred - carved_v[:, 1].flatten(1)).abs().mean().item())
                    if len(frames) > 1:  # how alike the decoded volumes of different scenes are.
                        c = torch.corrcoef(decoded).fill_diagonal_(0)  # near 1 means the decoder
                        scene_corr.append((c.sum() / (len(decoded) * (len(decoded) - 1))).item())

            v_loss, v_recon, v_kl = [t / n for t in totals]
            corr = sum(scene_corr) / max(len(scene_corr), 1)
            skill = 1 - v_recon / floor  # 0 is no better than the mean image, 1 is perfect
            v_answerable = answerable / n / max(seen_px / n, 1e-6)  # per answerable pixel
            print(f'epoch {epoch+1}: val loss {v_loss:.4g} (recon {v_recon:.4g}, kl {v_kl:.4g}), '
                  f'scene corr {corr:.3f}, skill {skill:+.3f}, lift agree '
                  f'{sum(agree)/len(agree):.3f}, answerable {v_answerable:.1f}, '
                  f'invented {sum(invent)/len(invent):.3f}', flush=True)
            # running bests, because a sweep reads the summary and summary keeps the last value
            best_corr = min(best_corr, corr)
            best_agree = max(best_agree, sum(agree) / len(agree))
            metrics = {"val/loss": v_loss, "val/recon": v_recon, "val/kl": v_kl,
                       "val/scene_corr": corr, "val/scene_corr_best": best_corr,
                       "val/skill": skill, "val/recon_answerable": v_answerable,
                       "val/lift_agreement": sum(agree) / len(agree),
                       "val/invented_density": sum(invent) / len(invent),
                       "val/lift_agreement_best": best_agree,
                       "train/recon_epoch": train_recon / max(train_batches, 1),
                       "train/lr": optimizer.param_groups[0]["lr"],
                       "epoch": epoch + 1}
            if mae: metrics["val/depth_mae_m"] = sum(mae) / len(mae)
            if unknown_err: metrics["val/unknown_mae"] = sum(unknown_err) / len(unknown_err)
            run.log(metrics, step=global_step)
            if isinstance(sched, METRIC_DRIVEN): sched.step(v_recon)  # these step on the metric
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": sched.state_dict() if sched else None,
                        "epoch": epoch + 1, "global_step": global_step, "best_corr": best_corr,
                        "best_agree": best_agree,
                        "config": config}, CHECKPOINTS / f'{run.name}.pt')

            if epoch % n_epochs == 0:
                for s, sample in enumerate(samples):
                    log_novel_view(model, sample, step=global_step, mode=target, key=f'novel/scene{s}')
                    log_pose_sweep(model, sample, step=global_step, mode=target, key=f'sweep/scene{s}')
    finally:
        run.finish()

    return model


def main():
    # underscores, not hyphens: these have to match sweep.json's parameter names exactly
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--unknown", type=int, default=0)  # observedness channel + KL weighted by it
    parser.add_argument("--scheduler", default="none", choices=SCHEDULERS)
    parser.add_argument("--target", default="depth", choices=("rgb", "gray_depth", "depth"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)  # 1 overlaps windows, 5 does not
    parser.add_argument("--num_frames", type=int, default=7)
    parser.add_argument("--jitter", type=int, default=1)  # see JITTER_FOVY, and implies masking
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")  # a checkpoint to carry on from
    parser.add_argument("--taper", type=int, default=0)  # anneal out instead of holding rate back
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hw = (args.image_size, args.image_size)
    dataset = TartanGroundWindowDataset(
        trajectory_paths=find_trajectory_dirs(HOUSE), num_frames=args.num_frames, image_hw=hw,
        include_pose=True, stride=args.stride,
        head_fovy_deg=JITTER_FOVY if args.jitter else None, include_head_jitter=bool(args.jitter),
    )
    splits = [dataset.get_trajectory(paths)
              for paths in (dataset.trajectory_paths[1:], dataset.trajectory_paths[:1])]
    # val shuffles too: at stride 1 an unshuffled batch is near-identical windows, which
    # scene_corr would read as the decoder collapsing
    loaders = [make_dataloader(s, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers) for s in splits]
    print(f"{len(dataset)} windows, {len(splits[0])} train / {len(splits[1])} val", flush=True)

    channels, groups = args.base_channels, args.groups
    model = VVAEGeometric(VVAEGeometricConfig(render_hw=hw, n_downsamples=2, downsample_scale=2,
                                              kernel_size=3, base_channels=channels, groups=groups,
                                              target=args.target, predict_unknown=bool(args.unknown),
                                              source_fovy_deg=dataset.camera[1]))
    train_geometric(model, *loaders, scheduler=args.scheduler,
                    target=args.target, kl_mode="observed" if args.unknown else "uniform", mask_unknown=bool(args.jitter),
                    resume=args.resume, taper=bool(args.taper),
                    name=f'geo-c{channels}g{groups}-{args.target}-{args.scheduler}'
                         f'-b{args.beta:.2g}-lr{args.lr:.2g}'
                         f'{"-jit" if args.jitter else ""}'
                         f'{"-unk" if args.unknown else ""}',
                    epochs=args.epochs, lr=args.lr, beta=args.beta)


if __name__ == "__main__":
    main()
