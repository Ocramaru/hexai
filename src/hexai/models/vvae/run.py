"""Entry point for a geometric VVAE run, started from one of config.py's presets.

    uv run python -m hexai.models.vvae.run baseline
    uv run python -m hexai.models.vvae.run baseline --epochs 20 --resume .runs/<name>.pt

Anything on the command line beats the preset, and --help lists every option with its reason.

Author: Marco Cassar (@Ocramaru)
"""

import os
from pathlib import Path

import numpy as np
import torch
import tyro

from hexai.data import TartanGroundConfig
from hexai.models.vvae import VVAEGeometric, VVAEGeometricConfig
from hexai.models.vvae.config import PRESETS
from hexai.models.vvae.train import train_geometric

ROOT = Path(__file__).resolve().parents[4]
TARTAN = Path(os.environ.get("TARTAN_ROOT", ROOT / "data/tartanground"))

JITTER_FOVY = 49.0  # the hexapod's lens; jitter re-aims the head at this fov and masks


def main():
  run = tyro.extras.overridable_config_cli(PRESETS, description=__doc__, use_underscores=True,
                                           config=(tyro.conf.FlagConversionOff,))
  torch.manual_seed(run.seed)
  np.random.seed(run.seed)

  image_hw = (run.image_size, run.image_size)
  data = TartanGroundConfig(
    root=TARTAN, envs=run.envs, val_env=run.val_env, num_frames=run.num_frames, stride=run.stride,
    image_hw=image_hw, head_fovy_deg=JITTER_FOVY if run.jitter else None,
    include_head_jitter=run.jitter, batch_size=run.batch_size, num_workers=run.num_workers,
    envs_per_epoch=run.envs_per_epoch or None, samples_per_epoch=run.samples_per_epoch or None)
  train_loader, test_loader, summary = data.loaders()
  print(summary, flush=True)

  model = VVAEGeometric(VVAEGeometricConfig(
    render_hw=image_hw, n_downsamples=2, downsample_scale=2, grid_resolution=run.grid,
    latent_channels=run.latent_channels, kernel_size=3, base_channels=run.base_channels,
    groups=run.groups, target=run.target, predict_unknown=run.unknown,
    learned_lift=run.learned_lift, source_fovy_deg=data.camera[1]))
  train_geometric(
    model, train_loader, test_loader, name=run.name, project=run.project, epochs=run.epochs,
    lr=run.lr, beta=run.beta, scheduler=run.scheduler, shape=run.shape, period=run.period,
    warmup=run.warmup, target=run.target, kl_mode="observed" if run.unknown else "uniform",
    mask_unknown=run.jitter, occlusion=run.occlusion, residual=run.loss,
    depth_space=run.depth_space, depth_noise=run.depth_noise, resume=run.resume, taper=run.taper,
    revive=run.revive, snapshot_every=run.snapshot_every)


if __name__ == "__main__":
  main()
