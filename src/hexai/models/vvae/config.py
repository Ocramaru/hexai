"""What a geometric VVAE run is: one dataclass, and the named presets run.py starts from.

A preset states only what differs from the defaults, so the type checker reads every one of them.

Author: Marco Cassar (@Ocramaru)
"""

import dataclasses
import os
from pathlib import Path
from typing import Literal

from hexai.models.vvae.train import DepthSpace, Residual, Scheduler
from hexai.utilities import Decay


@dataclasses.dataclass(frozen=True)
class Run:
  """One run. Names are underscored, since a wandb sweep's parameters have to match them exactly."""

  base_channels: int = 32
  groups: int = 4
  learned_lift: bool = False  # convs, not unprojecting
  latent_channels: int = 4
  grid: tuple[int, int, int] = (32, 24, 32)  # voxels per axis; y is the robot's up
  epochs: int = 60
  lr: float = 1e-3
  beta: float = 0.01
  loss: Residual = "mse"  # what an uncertain ray is charged
  depth_space: DepthSpace = "linear"
  occlusion: float = 0.0  # charge for density the render cannot see
  unknown: bool = False  # observedness channel + KL weighted by it
  scheduler: Scheduler = "none"
  shape: Decay = "cosine"  # periodic's decay
  period: int = 50  # periodic's cycle, in epochs
  warmup: int = 0  # epochs per cycle; 0 leaves it to make_scheduler, which climbs over 5
  target: Literal["rgb", "gray_depth", "depth"] = "depth"
  batch_size: int = 4
  image_size: int = 64
  stride: int = 1  # 1 overlaps windows, 5 does not
  envs: tuple[str, ...] | None = None  # all of TARTAN_ROOT if unset
  envs_per_epoch: int = 0  # 0 uses them all
  samples_per_epoch: int = 0  # 0 scales with envs_per_epoch
  val_env: str | None = None  # which environment to hold out
  num_frames: int = 7
  jitter: bool = True  # aims the head at the hexapod's fov each frame, and implies masking
  depth_noise: float = 0.0  # a monocular estimate's scale error
  num_workers: int = 6
  seed: int = 42
  project: str = os.environ.get("WANDB_PROJECT", "vvae")  # WANDB_PROJECT, or vvae-test to throw away
  name: str | None = None  # default is wandb's own, which is unique per run
  resume: Path | None = None  # a checkpoint to carry on from
  revive: bool = True  # on resume, put the ceiling back up
  snapshot_every: int = 50  # keep an epoch every N, 0 off
  taper: bool = False  # anneal out instead of holding rate back


PRESETS = {
  "baseline": ("the learned lift at 32x60x32 on the periodic schedule, 200 epochs", Run(
    envs_per_epoch=5, samples_per_epoch=2500, val_env="Office",
    unknown=True, grid=(32, 60, 32), learned_lift=True,
    scheduler="periodic", warmup=10, lr=0.003, beta=0.0001, epochs=200)),
}
