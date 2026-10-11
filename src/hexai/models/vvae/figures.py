"""Regenerates everything in figures/ from one checkpoint, so the images in the READMEs are all
the same model rather than whichever run happened to be open when each was saved.

    uv run python -m hexai.models.vvae.figures --checkpoint .runs/<name>-best.pt

The two geometry figures need no checkpoint at all -- they are exact arithmetic on a window, and
they are the ones worth trusting when a trained model looks wrong.

Author: Marco Cassar (@Ocramaru)
"""

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from hexai.data import TartanGroundWindowDataset, find_trajectory_dirs
from hexai.models.vvae.model import VVAEGeometric, VVAEGeometricConfig
from hexai.models.vvae.visuals import (DEPTH_CMAP, log_latent, log_novel_view,
                                       log_pose_sweep)
from hexai.utilities import carve_window
from hexai.utilities.camera import pixel_to_camera_ray

HERE = Path(__file__).parent
ROOT = HERE.parents[3]
TARTAN = Path(os.environ.get("TARTAN_ROOT", ROOT / "data/tartanground"))


def rebuild(checkpoint: Path):
    """The model exactly as it was trained. Every shape comes off the checkpoint's own config, so
    a run at a different grid or latent width regenerates without editing this file."""
    state = torch.load(checkpoint, weights_only=False)
    cfg = state["config"]
    hw, fovy, depth_max = cfg["camera"]
    # read the lift off the weights, not the config: runs from before it was a config key have one
    learned = any(k.startswith("encoder.learned_lift.") for k in state["model"])
    model = VVAEGeometric(VVAEGeometricConfig(
        render_hw=tuple(hw), n_downsamples=2, downsample_scale=2, kernel_size=3,
        grid_resolution=tuple(cfg["grid"]), latent_channels=cfg["latent"][0],
        base_channels=cfg["base_channels"], groups=cfg["groups"], target=cfg["target"],
        predict_unknown=cfg["predict_unknown"], learned_lift=learned,
        source_fovy_deg=fovy, depth_max=depth_max))

    # the lift used to deposit rgb beside density; it deposits density alone now, so anything
    # trained before that change wants a 4-channel volume this code cannot build
    lifted = state["model"]["encoder.conv.weight"].shape[1]
    if lifted != model.config.in_channels:
        raise SystemExit(
            f"{checkpoint.name} was trained when the lift deposited {lifted} channels and this "
            f"code deposits {model.config.in_channels}. Restoring appearance in lift_window is "
            f"what it would take to load it; pick a checkpoint from after the change instead.")
    model.load_state_dict(state["model"])
    model.eval()
    return model, state, cfg


def window(cfg, env: str, index: int):
    """One val window, loaded the way the run's own validation loaded it -- same fov, same jitter,
    same frame count. A figure drawn from a differently built window is not showing this model."""
    hw, fovy, _ = cfg["camera"]
    data = TartanGroundWindowDataset(
        find_trajectory_dirs(TARTAN / env / "Data_omni"), num_frames=cfg["num_frames"],
        image_hw=tuple(hw), include_pose=True,
        head_fovy_deg=fovy if cfg["head_jitter"] else None,
        include_head_jitter=cfg["head_jitter"])
    frames, poses = data[index % len(data)]
    return tuple(x.unsqueeze(0) for x in (frames, poses))


def three_state_carving(batch, config, out: Path):
    """Horizontal slices of the carving. The one figure that shows why unknown is its own state:
    dark is never seen, and calling it empty is the mistake the whole design is built to avoid."""
    frames, poses = (x[0] for x in batch)
    carved = carve_window(frames, poses, config.bounds, config.grid_resolution,
                          config.source_fovy_deg, config.depth_max)
    state = torch.zeros_like(carved[0])
    state[carved[1] > 0.5] = 1.0
    state[carved[0] > 0.5] = 2.0

    # which pixels made each slice: unproject the anchor frame and keep the ones landing in the slab.
    # The anchor's pose is identity in the box, so its rays need no transform
    anchor = frames.shape[1] // 2
    rgb, depth = frames[:3, anchor], frames[3, anchor]
    h, w = depth.shape
    rays = pixel_to_camera_ray((h, w), config.source_fovy_deg)
    pixels = torch.stack(torch.meshgrid(torch.arange(w).float(), torch.arange(h).float(),
                                        indexing="xy")[:2] + (torch.ones(h, w),)).reshape(3, -1)
    heights_m = ((rays @ pixels) * (depth.reshape(1, -1) * config.depth_max))[1].reshape(h, w)

    ny = config.grid_resolution[1]
    span = float(config.bounds[1][1] - config.bounds[0][1])
    band = span / ny * 2  # a single 5 cm layer catches too few pixels to see, so show a slab
    rows = [int(ny * f) for f in (0.2, 0.4, 0.6, 0.8)]
    fig, axes = plt.subplots(2, len(rows), figsize=(13, 6.4))
    for col, y in enumerate(rows):
        metres = float(config.bounds[0][1]) + y * span / ny
        lit = (heights_m - metres).abs() < band
        faded = rgb.permute(1, 2, 0).numpy() * 0.25
        faded[lit.numpy()] = rgb.permute(1, 2, 0).numpy()[lit.numpy()]
        axes[0, col].imshow(faded)
        # camera y points down, so height above the lens is -y and the label has to negate it
        axes[0, col].set_title(f"pixels {-metres:+.2f} m above the camera", fontsize=9)
        # viridis here is three states, not depth, and its dark/teal/yellow is what the notes name
        axes[1, col].imshow(state[:, y].T.numpy(), cmap="viridis", origin="lower", vmin=0, vmax=2)
        axes[1, col].set_title(f"the slice they carve, {-metres:+.2f} m", fontsize=9)
    for ax in axes.ravel(): ax.set(xticks=[], yticks=[])
    fig.suptitle("dark unknown, teal free, yellow occupied — camera at bottom centre", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    fractions = {n: float((state == v).float().mean()) for n, v in
                 (("unknown", 0), ("free", 1), ("occupied", 2))}
    return fractions


def slice_correspondence(batch, config, out: Path):
    """One row of the depth image unprojected onto the floor plan it produced. Ties a slice back to
    a picture, which is the only way to tell a real wall from a transform bug."""
    frames, poses = (x[0] for x in batch)
    h, w = frames.shape[-2:]
    anchor = frames.shape[1] // 2  # the middle frame: the box is anchored on it, so its pose is
    depth, row = frames[3, anchor], h // 2  # identity and its rays need no transform to land right

    rays = pixel_to_camera_ray((h, w), config.source_fovy_deg)
    pixels = torch.stack([torch.arange(w).float(), torch.full((w,), float(row)), torch.ones(w)])
    points = (rays @ pixels) * (depth[row] * config.depth_max)

    lo, hi = config.bounds[0], config.bounds[1]
    res = torch.tensor(config.grid_resolution)
    grid_xy = (points[[0, 2]] - lo[[0, 2], None]) / (hi - lo)[[0, 2], None] * res[[0, 2], None]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(frames[:3, anchor].permute(1, 2, 0).numpy())
    axes[0].set_title("what the anchor camera saw")
    axes[1].imshow(depth.numpy(), cmap=DEPTH_CMAP, vmin=0, vmax=1)  # every depth panel, one map
    axes[1].set_title("its depth, and the row we take")
    for ax in axes[:2]: ax.axhline(row, color="lime", lw=2)
    carved = carve_window(frames, poses, config.bounds, config.grid_resolution,
                          config.source_fovy_deg, config.depth_max)
    axes[2].imshow(carved[0].sum(1).T.numpy(), cmap=DEPTH_CMAP, origin="lower")
    axes[2].scatter(grid_xy[0], grid_xy[1], s=4, c="lime")
    axes[2].set_xlim(0, config.grid_resolution[0])
    axes[2].set_ylim(0, config.grid_resolution[2])
    axes[2].set_title("those same pixels, placed in the box")
    for ax in axes: ax.set(xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(HERE / ".runs" /
                        "geo-c32g4-depth-onecycle-b0.0001-lr0.003-jit-unk-lift-g32x60x32c4.pt"))
    parser.add_argument("--out", default=str(HERE / "figures"))
    parser.add_argument("--env", default="Office")  # the held-out building, so nothing is memorised
    parser.add_argument("--index", type=int, default=660)  # a wall and boxes on the floor, no ceiling in it
    parser.add_argument("--seed", type=int, default=42)  # head jitter is drawn per load, so pin it
    args = parser.parse_args()
    matplotlib.use("Agg")  # saving, not showing. In main so that importing this file from a
                           # notebook does not force the backend on it
    np.random.seed(args.seed)  # or the same index aims the head differently every run
    torch.manual_seed(args.seed)  # and the decoder renders a drawn latent, not mu, so seed torch

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    model, state, cfg = rebuild(Path(args.checkpoint))
    batch = window(cfg, args.env, args.index)
    print(f"{Path(args.checkpoint).name}: epoch {state['epoch']}, "
          f"grid {cfg['grid']}, {len(cfg['environments'])} buildings, holding out {cfg['val_env']}")

    fractions = three_state_carving(batch, model.config, out / "three-state-carving.png")
    print("  three-state-carving.png   " + "  ".join(f"{k} {v:.1%}" for k, v in fractions.items()))
    slice_correspondence(batch, model.config, out / "slice-correspondence.png")
    print("  slice-correspondence.png")

    mode = cfg["target"]
    for name, fn in (("pose-sweep", log_pose_sweep), ("novel-view", log_novel_view),
                     ("latent-top-down", log_latent)):
        fig = fn(model, batch, mode=mode, keep=True)
        fig.savefig(out / f"{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}.png")


if __name__ == "__main__":
    main()
