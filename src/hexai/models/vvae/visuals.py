"""Figures for a training run. Nothing here touches the model, it only reads it.

Author: Marco Cassar (@Ocramaru)
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # importable from a detached session on its own

from math import gcd

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_angle_axis, quat_mul

from hexai.data import split_context_target, target_view
from hexai.utilities import answerable_mask
from hexai.utilities.camera import pixel_to_camera_ray

DEPTH_CMAP = 'magma'  # every depth panel, so one edit changes them all


def _yaw_view(frame, fovy_deg, yaw_deg):
    """[4, H, W] re-rendered as the same camera turned about its own optical centre. No parallax in
    a pure rotation, so the homography is exact; depth divides by the rotated ray's z because it
    measures along the optical axis and that axis moved. Returns the view and a mask of the pixels
    that stayed inside the source, since zero depth outside it would read as a surface at the lens."""
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
    seen = (xy.abs() <= 1).all(0).view(h, w)  # outside the source grid_sample pads zeros, and a
    turned = torch.cat([out[:3], out[3:] / rays[2].view(1, h, w).clamp(min=1e-6)]).clamp(0, 1)
    return turned, seen  # depth of zero reads as a surface at the lens rather than as no data

def row_corr(a, b):
    """Pearson correlation between matching rows of two [N, D] tensors."""
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-8)

def _graded(ax, render, truth, mask, title):
    """Squared error over the answerable mask, what the loss reads. Grey is never graded."""
    err = (((render - truth) ** 2).mean(0) * mask).numpy()
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("0.4")
    ax.imshow(np.ma.masked_where((mask < 0.5).numpy(), err), cmap=cmap, vmin=0.0,
              vmax=max(float(err.max()), 1e-6))
    ax.set_title(f"{title} — grey is ungraded, {float(mask.mean()):.0%} is not")


def _show(ax, img, title, missing=None):
    """[C, H, W] in [0, 1]. Three channels are rgb, anything else is colour-mapped depth, which is
    always the last channel. Depth is pinned to 0-1 so one colour is one distance in every panel."""
    a = img.permute(1, 2, 0).numpy()
    if a.shape[2] == 3:
        ax.imshow(a)
    else:
        cmap = plt.get_cmap(DEPTH_CMAP).copy()
        cmap.set_bad("0.4")  # grey, so no-data cannot be read as a surface at zero depth
        depth = a[..., -1] if missing is None else np.ma.masked_where(missing.numpy(), a[..., -1])
        ax.imshow(depth, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)

def _arrow(ax, pose, config, colour, length=5.0):
    """One camera on a floor plan, at its voxel position and pointing where it looks."""
    lower, size = config.bounds[0], config.bounds[1] - config.bounds[0]
    grid = torch.tensor(config.grid_resolution, dtype=torch.float32)
    x, _, z = (pose[:3].cpu() - lower) / size * grid
    heading = matrix_from_quat(pose[3:].cpu()) @ torch.tensor([0.0, 0.0, 1.0])
    ax.arrow(float(x), float(z), float(heading[0]) * length, float(heading[2]) * length,
             head_width=1.4, color=colour, length_includes_head=True)

HEXAPOD_HEIGHT = 0.242  # head top standing, from hexapod.xml: base at 0.15 plus the pan/tilt stack


def _side_view(ax, density, config, title):
    """Density summed along x, which keeps the vertical axis a floor plan sums away -- and that is
    the axis a gap under a table lives in. Height runs up the panel, forward runs across it.

    Do not add origin="lower" here. Camera y points down, so grid index 0 is the top of the box and
    imshow's default already draws it at the top; flipping it puts the ceiling underfoot."""
    ax.imshow(density.sum(0).numpy(), cmap="magma", aspect="auto")
    voxel = (config.bounds[1][1] - config.bounds[0][1]) / config.grid_resolution[1]
    floor = config.grid_resolution[1] - 1  # the box's lowest row, which is as near a floor as it
    ax.plot([1.2, 1.2], [floor, floor - HEXAPOD_HEIGHT / float(voxel)], color="cyan", lw=3,
            solid_capstyle="butt")  # gets: sim flies the camera at 1.75 m and the box is 1.5 m deep
    ax.set_title(title)


def _floor_plan(ax, density, pose, config, title, context_poses=None):
    """Density summed down the vertical axis, so it reads as a floor plan rather than a view. The
    box is the window's middle camera, and the arrow is where the query camera sits inside it and
    which way it looks. context_poses draws the frames it was given, dimmer and behind."""
    ax.imshow(density.sum(1).T.numpy(), origin="lower", cmap="magma", aspect="auto")
    if context_poses is not None:
        for p in context_poses: _arrow(ax, p, config, "deepskyblue", length=3.5)
    _arrow(ax, pose, config, "cyan")
    nx, _, nz = config.grid_resolution  # an arrow reaching past the box would otherwise stretch the
    ax.set_xlim(-0.5, nx - 0.5)         # axes and pad the panel with white, which reads as empty box
    ax.set_ylim(-0.5, nz - 0.5)
    ax.set_title(title)

@torch.no_grad()
def log_novel_view(model, batch, step=None, key='media/novel', target=1, mode='rgb', note='',
                   keep=False):
    """Truth against the render, and the lifted volume against the decoded one. The image pair alone
    cannot tell undertrained from broken -- a saturated density field also renders flat grey -- so
    the volumes go beside it, and the density fractions go up as scalars."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    frames, poses = batch
    context, context_poses, truth, query = split_context_target(frames[:1], poses[:1], target, mode)
    context, context_poses = context.to(device), context_poses.to(device)
    truth, query = truth.to(device), query.to(device)
    out = model(context, context_poses, query)
    lifted = model.encoder.lift(context, context_poses)[0].cpu()
    decoded = model.decoder.volume(out.mu)[0].cpu()
    carved = model.encoder.carve(context, context_poses)
    mask = answerable_mask(carved, truth, query, model.config.bounds, *model.camera)[0].cpu()
    if was_training: model.train()

    render, truth, query = out.x_hat[0].cpu().clamp(0, 1), truth[0].cpu(), query.cpu()
    density = decoded[model.config.density_index]
    fig, axes = plt.subplots(1, 7, figsize=(24.5, 3.8))
    _show(axes[0], truth, "truth, held out")
    _show(axes[1], render, "rendered from its pose")
    _graded(axes[2], render, truth, mask, "graded error")
    _floor_plan(axes[3], lifted[-1], query[0], model.config, "lifted, top-down")
    _floor_plan(axes[4], density, query[0], model.config, "decoded, top-down")
    _side_view(axes[5], lifted[-1], model.config, "lifted, from the side")
    _side_view(axes[6], density, model.config, "decoded, from the side")
    for ax in axes: ax.set(xticks=[], yticks=[])
    if note: fig.suptitle(note, fontsize=9)
    fig.tight_layout()

    if wandb.run is not None:
        wandb.log({key: wandb.Image(fig),
                   "val/lifted_occupied": (lifted[-1] > 1e-3).float().mean().item(),
                   "val/decoded_solid": (decoded[model.config.density_index] > 0.5).float().mean().item(),
                   "val/decoded_mean": decoded[model.config.density_index].mean().item()}, step=step)
    if keep: return fig  # a closed figure will not render in a notebook
    plt.close(fig)

def _head_label(pose):
    """Where a camera points and how far it sits from the window's centre, for a panel title."""
    heading = matrix_from_quat(pose[3:].cpu()) @ torch.tensor([0.0, 0.0, 1.0])
    yaw = float(torch.rad2deg(torch.atan2(heading[0], heading[2])))
    pitch = float(torch.rad2deg(torch.asin(heading[1].clamp(-1, 1).neg())))
    return f"{yaw:+.0f}° {pitch:+.0f}°  {float(pose[:3].norm()):.2f} m"

@torch.no_grad()
def log_context_strip(model, batch, step=None, key='media/context', target=1, mode='rgb', note='',
                      keep=False):
    """Every frame the model was given, where those cameras were, and what it made of them. The
    other figures start after the encode, so this is the only one showing the input."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    frames, poses = batch
    context, context_poses, truth, query = split_context_target(frames[:1], poses[:1], target, mode)
    context, context_poses = context.to(device), context_poses.to(device)
    truth, query = truth.to(device), query.to(device)
    out = model(context, context_poses, query)
    decoded = model.decoder.volume(out.mu)[0].cpu()
    carved = model.encoder.carve(context, context_poses)
    mask = answerable_mask(carved, truth, query, model.config.bounds, *model.camera)[0].cpu()
    if was_training: model.train()

    seen, given = context[0].cpu(), context_poses[0].cpu()
    render, truth = out.x_hat[0].cpu().clamp(0, 1), truth[0].cpu()
    n = seen.shape[1]
    cols = n * 4 // gcd(n, 4)  # so n frames and 4 panels both divide the row evenly
    fig = plt.figure(figsize=(2.1 * n, 5.6))
    gs = fig.add_gridspec(2, cols, height_ratios=[1, 1.5])
    for i in range(n):
        ax = fig.add_subplot(gs[0, i * cols // n:(i + 1) * cols // n])
        _show(ax, seen[:, i], _head_label(given[i]))
        ax.title.set_fontsize(8)
        ax.set(xticks=[], yticks=[])

    wide = [fig.add_subplot(gs[1, i * cols // 4:(i + 1) * cols // 4]) for i in range(4)]
    _show(wide[0], truth, "asked for, held out")
    _show(wide[1], render, "what it produced")
    _graded(wide[2], render, truth, mask, "graded error")
    _floor_plan(wide[3], decoded[model.config.density_index], query[0].cpu(), model.config,
                "cameras over the decoded volume", context_poses=given)
    for ax in wide: ax.set(xticks=[], yticks=[])
    fig.suptitle(f"{n} frames in, one held out{f'  --  {note}' if note else ''}", fontsize=9)
    fig.tight_layout()

    if wandb.run is not None: wandb.log({key: wandb.Image(fig)}, step=step)
    if keep: return fig
    plt.close(fig)

@torch.no_grad()
def log_pose_sweep(model, batch, step=None, key='media/sweep', target=1, mode='rgb', note='',
                   yaws=(-24, -12, 12, 24), keep=False):
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
    truths, renders, missing = [], [], []
    for deg in offsets:
        turn = quat_from_angle_axis(torch.deg2rad(torch.tensor([float(deg)], device=device)), up)
        q = query.to(device).clone()
        q[:, 3:] = quat_mul(q[:, 3:], turn)
        renders.append(model.decoder(mu, q)[0][0].cpu().clamp(0, 1))  # (image, volume)
        turned, seen = _yaw_view(source, model.config.source_fovy_deg, deg)
        truths.append(target_view(turned[None], mode)[0])
        missing.append(~seen)
    if was_training: model.train()

    fig, axes = plt.subplots(2, len(offsets), figsize=(2.6 * len(offsets), 5.6))
    for col, deg in enumerate(offsets):
        label = "its pose" if deg == 0 else f"{deg:+.0f} deg"
        _show(axes[0, col], truths[col], f"truth {label}", missing[col])
        _show(axes[1, col], renders[col], f"render {label}")
    for ax in axes.ravel(): ax.set(xticks=[], yticks=[])
    if note: fig.suptitle(note, fontsize=9)
    fig.tight_layout()

    if wandb.run is not None:
        flat = torch.stack([r.flatten() for r in renders])
        spread = row_corr(flat[:-1], flat[1:]).mean().item()  # 1.0 means the pose changes nothing
        wandb.log({key: wandb.Image(fig), f"val/viewpoint_sameness/{key.split(chr(47))[-1]}": spread}, step=step)
    if keep: return fig
    plt.close(fig)


def _latent_plan(ax, plane, title, limit):
    """One latent channel from above, its own grid rather than the volume's.

    Diverging and centred on zero, because a latent is signed and a sequential map gives zero no
    position -- above and below the prior read the same. `limit` is shared across the channels, or
    each panel autoscales to its own range and a quiet channel looks like a loud one."""
    ax.imshow(plane.T.numpy(), origin="lower", cmap="RdBu_r", aspect="auto",
              interpolation="nearest", vmin=-limit, vmax=limit)
    ax.set_title(title, fontsize=8)


@torch.no_grad()
def log_latent(model, batch, step=None, key='media/latent', target=1, mode='rgb', note='',
               keep=False):
    """The latent seen from above, one panel per channel, beside the volume it decodes to. This
    architecture's whole claim is that the latent's axes are space; a latent that reads the same
    in every cell has stopped being a map however well the render scores, and no other figure
    would show it."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    frames, poses = batch
    context, context_poses, _, query = split_context_target(frames[:1], poses[:1], target, mode)
    context, context_poses = context.to(device), context_poses.to(device)
    out = model(context, context_poses, query.to(device))
    mu = out.mu[0].cpu()
    decoded = model.decoder.volume(out.mu)[0, model.config.density_index].cpu()
    if was_training: model.train()

    channels = mu.shape[0]
    planes = [mu[c].sum(1) for c in range(channels)]  # sum the vertical axis
    limit = max(float(p.abs().max()) for p in planes) or 1.0  # one scale, so channels compare
    fig, axes = plt.subplots(1, channels + 1, figsize=(2.3 * (channels + 1), 2.8))
    for c in range(channels):
        _latent_plan(axes[c], planes[c], f"latent {c}", limit)
    _floor_plan(axes[-1], decoded, query[0], model.config, "decodes to")
    for ax in axes: ax.set(xticks=[], yticks=[])
    fig.suptitle(f"latent {tuple(mu.shape)}{f'  --  {note}' if note else ''}", fontsize=9)
    fig.tight_layout()

    if wandb.run is not None:
        wandb.log({key: wandb.Image(fig), "val/latent_alignment": latent_alignment(model, mu, decoded)},
                  step=step)
    if keep: return fig
    plt.close(fig)


def latent_alignment(model, mu, density):
    """How much the latent's layout matches the volume it produces, after pooling the volume down to
    the latent's grid. Near zero means the encoder stored the scene somewhere other than in place,
    which renders fine and maps badly. Read as the best channel, since one is enough to carry it."""
    shrink = model.config.downsample_scale ** model.config.n_downsamples
    pooled = F.avg_pool3d(density[None, None], shrink)[0, 0].flatten()
    return max(row_corr(mu[c].flatten()[None], pooled[None]).abs().item() for c in range(len(mu)))


@torch.no_grad()
def save_latent(model, batch, path, target=1, mode='rgb'):
    """Encode one window and write the latent, the volume it decodes to, the geometry that went in
    and the poses, as an npz. A figure flattens one axis away; this keeps all three so the latent
    can be turned in a 3D view, and it is small enough to keep one per run."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    frames, poses = batch
    context, context_poses, truth, query = split_context_target(frames[:1], poses[:1], target, mode)
    context, context_poses = context.to(device), context_poses.to(device)
    out = model(context, context_poses, query.to(device))
    record = {
      "latent": out.mu[0].cpu().numpy(),
      "decoded": model.decoder.volume(out.mu)[0].cpu().numpy(),
      "lifted": model.encoder.lift(context, context_poses)[0].cpu().numpy(),
      "carved": model.encoder.carve(context, context_poses)[0].cpu().numpy(),
      "context_poses": context_poses[0].cpu().numpy(),
      "query_pose": query[0].cpu().numpy(),
      "truth": truth[0].cpu().numpy(),
      "render": out.x_hat[0].cpu().numpy(),
      "bounds": model.config.bounds.numpy(),
      "grid_resolution": np.array(model.config.grid_resolution),
      "density_index": np.array(model.config.density_index),
    }
    if was_training: model.train()
    np.savez_compressed(str(path), **record)
    return path


def latent_figure(record, channel=0, threshold=0.6):
    """A plotly view of a record from save_latent: the latent's cells as a cloud, sized by value,
    over the voxels the decoded volume calls solid. Needs plotly, which is a dev dependency, so it
    is imported here rather than at module scope."""
    import plotly.graph_objects as go  # noqa: PLC0415

    latent, decoded = record["latent"][channel], record["decoded"][record["density_index"]]
    lx, ly, lz = np.meshgrid(*[np.arange(n) for n in latent.shape], indexing="ij")
    shrink = np.array(decoded.shape) / np.array(latent.shape)
    solid = np.argwhere(decoded > threshold) / shrink  # onto the latent's grid, so they overlay

    weight = np.abs(latent).ravel()
    return go.Figure([
      go.Scatter3d(x=solid[:, 0], y=solid[:, 2], z=-solid[:, 1], mode="markers", name="decoded solid",
                   marker={"size": 2, "color": "orange", "opacity": 0.25}),
      go.Scatter3d(x=lx.ravel(), y=lz.ravel(), z=-ly.ravel(), mode="markers",
                   name=f"latent {channel}", marker={"size": 4 + 12 * weight / (weight.max() or 1),
                                                     "color": latent.ravel(), "colorscale": "Viridis",
                                                     "opacity": 0.85}),
    ], layout={"scene": {"xaxis_title": "x", "yaxis_title": "z (forward)", "zaxis_title": "up"},
               "title": f"latent channel {channel} over the volume it decodes"})
