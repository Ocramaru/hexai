"""
Loads downloaded TartanGround RGB+depth trajectories into VVAE-shaped windows.

Author: Marco Cassar (@Ocramaru)
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset

from hexai.utilities.camera import pixel_to_camera_ray

Sample = torch.Tensor | tuple[torch.Tensor, torch.Tensor]  # frames, or (frames, poses)

# OpenCV's own internal threading can deadlock against multiple simultaneous
cv2.setNumThreads(0)


# TartanGround's per-file depth is stored as an 8-bit RGBA PNG whose 4 raw
# bytes per pixel ARE an IEEE-754 float32 (little-endian)
def _read_depth_meters(
  path: Path, image_hw: tuple[int, int] | None = None
) -> np.ndarray:
  depth_rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
  if depth_rgba is None:
    raise FileNotFoundError(f"could not read depth file: {path}")
  depth = depth_rgba.view("<f4").squeeze(-1)
  if image_hw is not None:
    depth = cv2.resize(depth, image_hw[::-1], interpolation=cv2.INTER_NEAREST)
  return depth


_INV_255 = np.float32(1.0 / 255.0)


def _read_rgb(path: Path, image_hw: tuple[int, int] | None = None) -> np.ndarray:
  """RGB in [0, 1], cast+scaled in one fused pass rather than
  `.astype(float32) / 255.0` (two full-array passes)."""
  bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if bgr is None:
    raise FileNotFoundError(f"could not read image file: {path}")
  if image_hw is not None:
    bgr = cv2.resize(bgr, image_hw[::-1], interpolation=cv2.INTER_AREA)
  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
  return np.multiply(rgb, _INV_255, dtype=np.float32)


def _read_poses(trajectory_path: Path, camera_name: str) -> np.ndarray:
  """[N, 7] world poses from the `meta` modality, one row per frame: tx ty tz qx qy qz qw."""
  pose_path = trajectory_path / f"pose_{camera_name}.txt"
  if not pose_path.exists():
    raise FileNotFoundError(f"{pose_path} missing -- download the 'meta' modality")
  return np.loadtxt(pose_path, dtype=np.float32)


def _rotate_view(
  rgb: np.ndarray,
  depth: np.ndarray,
  source_k_inv: np.ndarray,
  out_k_inv: np.ndarray,
  rotation: np.ndarray,
  out_hw: tuple[int, int],
):
  """Re-renders a frame as a camera rotated about its own optical centre, which is what the
  pan/tilt head does. Rotating about the centre has no parallax so a homography is exact, but
  depth measures along the optical axis and that axis moved, so it rescales by the ray's z."""
  h, w = out_hw
  v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
  in_source = rotation @ out_k_inv @ np.stack([u.ravel(), v.ravel(), np.ones(h * w)])
  pixels = np.linalg.inv(source_k_inv) @ in_source
  map_x = (pixels[0] / pixels[2]).reshape(h, w).astype(np.float32)
  map_y = (pixels[1] / pixels[2]).reshape(h, w).astype(np.float32)
  depth = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST) / in_source[2].reshape(h, w)
  return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR), depth.astype(np.float32)


def _aim_poses(poses: np.ndarray, heads: Rotation) -> np.ndarray:
  """Folds per-frame head rotations into a window's relative poses. The middle frame's aimed camera
  is the canonical frame, so it stays identity and the rest are relative to it."""
  anchor = len(poses) // 2
  base = Rotation.from_quat(poses[:, [4, 5, 6, 3]])  # stored wxyz, scipy wants xyzw
  aimed = (heads[anchor].inv() * base * heads).as_quat()
  return np.concatenate(
    [heads[anchor].inv().apply(poses[:, :3]), aimed[:, [3, 0, 1, 2]]], axis=1
  ).astype(np.float32)


_NED2CAM = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])  # NED axes -> camera axes


def _relative_poses(poses: np.ndarray) -> np.ndarray:
  """World poses -> poses relative to the window's middle row, so a window is scene-centric. Also
  converts out of TartanGround's conventions: NED axes to camera axes, xyzw quaternions to the wxyz
  mjlab's math helpers expect.

  The middle rather than the first because the robot turns as it walks: anchored on frame 0 the
  camera drifts to a z spread of 0.214 m and the canonical box hangs off one end of the window,
  against 0.085 m and 2% more points retained when it is anchored in the middle."""
  anchor = len(poses) // 2
  rot = Rotation.from_quat(poses[:, 3:])
  anchor_inv = rot[anchor].inv()
  basis = Rotation.from_matrix(_NED2CAM)

  translation = anchor_inv.apply(poses[:, :3] - poses[anchor, :3]) @ _NED2CAM.T
  rotation = (basis * (anchor_inv * rot) * basis.inv()).as_quat()
  return np.concatenate([translation, rotation[:, [3, 0, 1, 2]]], axis=1).astype(
    np.float32
  )


def split_context_target(
  frames: torch.Tensor,
  poses: torch.Tensor,
  target: int | None = None,
  mode: str = "rgb",
):
  """Holds one frame out of a [B, 4, T, H, W] window so the decoder has to render a view the encoder
  never saw, which is what stops the latent storing a view instead of a scene. Returns the context
  frames and poses, then the target's rgb and its pose.

  The target is never the middle frame, because that is the anchor and its query pose is identity:
  rendering the volume from its own origin never looks at it from an angle, so geometry that is
  wrong off-axis costs nothing and the pose input goes unused. Every other frame is a real
  viewpoint, covered 90% at the ends and 99.5% beside the anchor. Pass a target to keep validation
  comparable across epochs."""
  count = frames.shape[2]
  anchor = count // 2
  choices = [t for t in range(count) if t != anchor]
  target = choices[int(torch.randint(len(choices), (1,)))] if target is None else target
  context = [t for t in range(count) if t != target]
  return (
    frames[:, :, context],
    poses[:, context],
    target_view(frames[:, :, target], mode),
    poses[:, target],
  )


_LUMA = (0.299, 0.587, 0.114)


def target_view(frame: torch.Tensor, mode: str = "rgb") -> torch.Tensor:
  """[B, 4, H, W] rgb+depth -> the channels the decoder is asked to render. depth is geometry
  alone, which is what the robot actually needs; rgb makes the latent pay to remember colour."""
  if mode == "rgb":
    return frame[:, :3]
  if mode == "depth":
    return frame[:, 3:]
  luma = torch.tensor(_LUMA, device=frame.device, dtype=frame.dtype).view(1, 3, 1, 1)
  return torch.cat([(frame[:, :3] * luma).sum(1, keepdim=True), frame[:, 3:]], dim=1)


def add_pose_noise(
  poses: torch.Tensor, translation_frac: float = 0.02, rotation_deg: float = 0.5
) -> torch.Tensor:
  """[B, T, 7] relative poses -> the same with visual-odometry-like error. Call it from
  the training loop, not the dataset, so validation still sees the true poses.

  Each frame-to-frame increment is perturbed and the window re-integrated, so error grows
  along T the way VO's does rather than landing independently per frame. Defaults are a
  frame-to-frame VO budget: 2% of the step translated, half a degree rotated.
  """
  out = poses.clone()
  for b in range(poses.shape[0]):
    p = poses[b].numpy()
    rot = Rotation.from_quat(p[:, 3:])
    step_rot = rot[:-1].inv() * rot[1:]
    step_t = rot[:-1].inv().apply(p[1:, :3] - p[:-1, :3])

    n = len(step_t)
    step_t = step_t + (
      torch.randn(n, 3).numpy()
      * translation_frac
      * np.linalg.norm(step_t, axis=1, keepdims=True)
    )
    step_rot = step_rot * Rotation.from_rotvec(
      torch.randn(n, 3).numpy() * np.radians(rotation_deg)
    )

    walk_t, walk_rot = [np.zeros(3)], [Rotation.identity()]
    for i in range(n):
      walk_t.append(walk_t[-1] + walk_rot[-1].apply(step_t[i]))
      walk_rot.append(walk_rot[-1] * step_rot[i])
    out[b, :, :3] = torch.from_numpy(np.stack(walk_t).astype(np.float32))
    out[b, :, 3:] = torch.from_numpy(
      np.stack([r.as_quat() for r in walk_rot]).astype(np.float32)
    )
  return out


def _frame_pairs(trajectory_path: Path, camera_name: str) -> list[tuple[Path, Path]]:
  """(image, depth) path pairs for one camera in one trajectory, in frame order."""
  image_dir = trajectory_path / f"image_{camera_name}"
  depth_dir = trajectory_path / f"depth_{camera_name}"
  pairs = []
  for image_path in sorted(image_dir.glob(f"*_{camera_name}.png")):
    depth_path = depth_dir / image_path.name.replace(
      f"_{camera_name}.png", f"_{camera_name}_depth.png"
    )
    if not depth_path.exists():
      raise FileNotFoundError(f"no matching depth file for {image_path}")
    pairs.append((image_path, depth_path))
  return pairs


@dataclass
class TartanGroundWindowDataset(Dataset[Sample]):
  """[num_frames]-length RGB+depth windows from one or more TartanGround trajectory
  directories. Each sample is `[4, T, H, W]` float32: channels 0-2 RGB in [0, 1],
  channel 3 depth scaled to [0, 1] -- matching `VVAEConfig(in_channels=4)`. Each
  camera contributes its own independent windows rather than being fused, so the
  cameras are just more training data through one shared encoder. Leftover frames
  that don't fill a full window are dropped."""

  trajectory_paths: list[Path]
  num_frames: int = 5
  stride: int | None = None
  """Frames between window starts. None steps by a full window, so each frame is used once.
  A smaller stride overlaps them, which is not redundant for a posed model: every window is
  anchored on its own middle frame, so shifting the start moves the canonical box and changes
  the geometry of every sample."""
  camera_names: tuple[str, ...] = ("lcam_front",)
  depth_max: float = 4.0
  """Meters. Values above this are clipped."""
  image_hw: tuple[int, int] | None = None  # if None will use default size
  include_debug_image: bool = False  # plot the first frame on construction
  include_pose: bool = (
    False  # also return [T, 7] poses relative to the window's first frame
  )
  source_fovy_deg: float = (
    90.0  # cube faces are 90 by definition, confirmed by warp consistency
  )
  head_fovy_deg: float | None = (
    None  # re-render at this fovy, None keeps the source camera
  )
  include_head_jitter: bool = (
    False  # aim each frame randomly rather than straight ahead
  )

  _image_hw: tuple[int, int] = field(init=False)
  _source_hw: tuple[int, int] = field(init=False)
  _head_range_deg: tuple[float, float] = field(init=False)
  _source_k_inv: np.ndarray = field(init=False)
  _out_k_inv: np.ndarray = field(init=False)

  def __post_init__(self) -> None:
    self._windows: list[list[tuple[Path, Path]]] = []
    self._window_trajectory: list[
      Path
    ] = []  # which trajectory_path each window came from
    self._window_camera: list[str] = []  # and which camera
    self._window_poses: list[np.ndarray] = []
    skipped: list[str] = []
    for trajectory_path in self.trajectory_paths:
      for camera_name in self.camera_names:
        # a download in progress leaves environments half-written, so step over what is not
        # there yet rather than failing the whole dataset
        try:
          pairs = _frame_pairs(trajectory_path, camera_name)
          poses = (
            _read_poses(trajectory_path, camera_name) if self.include_pose else None
          )
        except FileNotFoundError:
          pairs = []
          poses = None
        # an interrupted download leaves the directory made and empty as often as absent
        if not pairs:
          skipped.append(f"{trajectory_path.parent.parent.name}/{trajectory_path.name}")
          continue
        if poses is not None and len(poses) != len(pairs):
          raise ValueError(
            f"{len(poses)} poses for {len(pairs)} frames in {trajectory_path}"
          )

        for start in range(
          0, len(pairs) - self.num_frames + 1, self.stride or self.num_frames
        ):
          self._windows.append(pairs[start : start + self.num_frames])
          self._window_trajectory.append(trajectory_path)
          self._window_camera.append(camera_name)
          if poses is not None:
            self._window_poses.append(
              _relative_poses(poses[start : start + self.num_frames])
            )

    if skipped:
      print(
        f"skipping {len(skipped)} incomplete trajectories: {', '.join(sorted(set(skipped))[:4])}"
        f"{' ...' if len(skipped) > 4 else ''}"
      )
    if not self._windows:
      raise FileNotFoundError(
        f"no complete trajectories under {self.trajectory_paths[:1]}"
      )
    first_image = cv2.imread(str(self._windows[0][0][0]), cv2.IMREAD_COLOR)
    if first_image is None:
      raise FileNotFoundError(self._windows[0][0][0])
    self._source_hw = first_image.shape[:2]
    self._image_hw = self.image_hw if self.image_hw is not None else self._source_hw

    # how far the re-rendered cone can be aimed before it leaves the source cone
    if self.head_fovy_deg is None:
      self._head_range_deg = (0.0, 0.0)
    else:
      aspect = np.array([self._image_hw[1] / self._image_hw[0], 1.0])
      half = lambda fovy: np.arctan(aspect * np.tan(np.radians(fovy) / 2))  # noqa: E731
      self._head_range_deg = tuple(
        np.degrees(
          np.maximum(half(self.source_fovy_deg) - half(self.head_fovy_deg), 0.0)
        )
      )

    self._source_k_inv = pixel_to_camera_ray(
      self._source_hw, self.source_fovy_deg
    ).numpy()
    self._out_k_inv = pixel_to_camera_ray(
      self._image_hw, self.head_fovy_deg or self.source_fovy_deg
    ).numpy()

    if self.include_debug_image:
      self._show_debug_image()

  @property
  def camera(self) -> tuple[tuple[int, int], float, float]:
    """The (image_hw, fovy, depth_max) a geometric model has to be built with to match this dataset.
    Only image_hw mismatches loudly; the other two agree on shape and quietly place every point at
    the wrong angle or the wrong distance."""
    return self._image_hw, self.head_fovy_deg or self.source_fovy_deg, self.depth_max

  def _show_debug_image(self) -> None:
    """First frame of the first window, to eyeball that loading and resizing landed."""
    import matplotlib.pyplot as plt

    sample = self[0]
    if isinstance(sample, tuple):
      sample = sample[0]
    fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(8, 4))
    ax_rgb.imshow(sample[:3, 0].permute(1, 2, 0).numpy())
    ax_rgb.set_title(f"rgb {tuple(sample.shape[2:])}")
    depth = ax_depth.imshow(sample[3, 0].numpy(), cmap="viridis", vmin=0, vmax=1)
    ax_depth.set_title(f"depth, clipped at {self.depth_max} m")
    fig.colorbar(depth, ax=ax_depth)
    for ax in (ax_rgb, ax_depth):
      ax.axis("off")
    fig.suptitle(f"{self._window_trajectory[0].name} {self._window_camera[0]}")

  def __len__(self) -> int:
    return len(self._windows)

  def get_trajectory(
    self,
    trajectory_paths: Path | Sequence[Path] | None = None,
    camera_names: str | Sequence[str] | None = None,
  ) -> "TartanGroundSubset":
    """Windows from one or more trajectories, as a `Subset` -- indexes back into
    this dataset, so `__getitem__` still does the actual loading. Either argument
    takes one value or many, and `None` means all of them. Windows come out in
    this dataset's order, not the order asked for."""
    trajectory_paths = (
      self.trajectory_paths if trajectory_paths is None else trajectory_paths
    )
    paths = (
      {trajectory_paths}
      if isinstance(trajectory_paths, Path)
      else set(trajectory_paths)
    )
    camera_names = self.camera_names if camera_names is None else camera_names
    cameras = {camera_names} if isinstance(camera_names, str) else set(camera_names)

    if missing := paths - set(self._window_trajectory):
      raise ValueError(f"no windows for trajectories: {sorted(missing)}")
    if missing := cameras - set(self.camera_names):
      raise ValueError(f"dataset has no cameras: {sorted(missing)}")
    indices = [
      i
      for i, (p, c) in enumerate(
        zip(self._window_trajectory, self._window_camera, strict=True)
      )
      if p in paths and c in cameras
    ]
    if not indices:
      raise ValueError(f"no windows for {sorted(paths)} x {sorted(cameras)}")
    return TartanGroundSubset(self, indices)

  @property
  def loaded_trajectories(self) -> list[Path]:
    """The trajectories that actually produced windows, which is not `trajectory_paths` while a
    download is still filling environments in."""
    present = set(self._window_trajectory)
    return [p for p in self.trajectory_paths if p in present]

  def window_trajectory(self, index: int) -> Path:
    """Which trajectory a window came from, and so which environment."""
    return self._window_trajectory[index]

  def __getitem__(self, index: int) -> Sample:
    window = self._windows[index]
    heads = self._sample_head_rotations(len(window))

    # Write each frame directly into its slice of the final [4, T, H, W]
    out = np.empty((4, len(window), *self._image_hw), dtype=np.float32)
    for t, (image_path, depth_path) in enumerate(window):
      read_hw = (
        None if self.head_fovy_deg else self._image_hw
      )  # re-rendering wants the source
      rgb = _read_rgb(image_path, read_hw)
      depth = _read_depth_meters(depth_path, read_hw)  # [H, W], meters
      if self.head_fovy_deg:
        rgb, depth = _rotate_view(
          rgb,
          depth,
          self._source_k_inv,
          self._out_k_inv,
          heads[t].as_matrix(),
          self._image_hw,
        )
      out[:3, t] = rgb.transpose(2, 0, 1)
      np.clip(
        depth, 0.0, self.depth_max, out=out[3, t]
      )  # write clipped depth straight into out
      out[3, t] /= self.depth_max  # in-place scale, no extra allocation

    if not self.include_pose:
      return torch.from_numpy(out)
    poses = self._window_poses[index]
    if self.head_fovy_deg:
      poses = _aim_poses(poses, heads)
    return torch.from_numpy(out), torch.from_numpy(poses)

  def _sample_head_rotations(self, count: int) -> Rotation:
    """One rotation per frame, straight ahead unless jitter is on."""
    if not self.include_head_jitter:
      return Rotation.identity(count)
    angles = np.random.uniform(-1, 1, (count, 2)) * np.array(self._head_range_deg)
    return Rotation.from_euler("yx", angles, degrees=True)


class TartanGroundSubset(Subset[Sample]):
  """A `Subset` that still answers for itself: which trajectories and cameras it
  actually holds, and narrowing again from there rather than from the full set."""

  dataset: TartanGroundWindowDataset  # pyright: ignore[reportIncompatibleVariableOverride]

  @property
  def trajectory_paths(self) -> list[Path]:
    present = {self.dataset.window_trajectory(i) for i in self.indices}
    return [p for p in self.dataset.trajectory_paths if p in present]

  @property
  def camera_names(self) -> tuple[str, ...]:
    present = {self.dataset._window_camera[i] for i in self.indices}
    return tuple(c for c in self.dataset.camera_names if c in present)

  def get_trajectory(
    self,
    trajectory_paths: Path | Sequence[Path] | None = None,
    camera_names: str | Sequence[str] | None = None,
  ) -> "TartanGroundSubset":
    """Same as the dataset's, but intersected with what this subset already holds,
    so a split can't leak back rows it excluded."""
    keep = set(self.indices)
    narrowed = self.dataset.get_trajectory(trajectory_paths, camera_names)
    if not (indices := [i for i in narrowed.indices if i in keep]):
      raise ValueError(
        f"nothing in this subset for {trajectory_paths} x {camera_names}"
      )
    return TartanGroundSubset(self.dataset, indices)


def find_trajectory_dirs(root: Path, version: str = "Data_omni") -> list[Path]:
  """Every trajectory directory under `root`, which may be one environment or many.

  `data/tartanground/House/Data_omni` gives that house's P0000/P0001/P0002. `data/tartanground`
  gives the same for every environment downloaded under it."""
  if root.name.startswith("Data_"):
    return sorted(p for p in root.iterdir() if p.is_dir())
  if (root / version).is_dir():
    return sorted(p for p in (root / version).iterdir() if p.is_dir())
  found: list[Path] = []
  for env in sorted(root.iterdir()):
    versioned = env / version
    if versioned.is_dir():
      found.extend(p for p in sorted(versioned.iterdir()) if p.is_dir())
  return found


def environment_of(trajectory: Path) -> str:
  """Which TartanGround environment a trajectory belongs to. Splitting on this rather than on
  trajectories holds out a whole building, where a second walk through the same one is a much
  weaker test."""
  return trajectory.parent.parent.name


def make_dataloader(
  dataset: Dataset,
  batch_size: int,
  num_workers: int = 4,
  persistent_workers: bool = False,
  **kwargs,
) -> DataLoader:
  """DataLoader with prefetch_factor set per current PyTorch guidance (2-4
  workers/GPU). Takes anything dataset-shaped, so a `get_trajectory` Subset or a
  ConcatDataset works as-is. pin_memory is omitted; pass pin_memory=True yourself
  once batches are moved to a GPU.

  persistent_workers defaults off. Turning it on saves respawning workers each
  epoch, but the workers then outlive the iterator, and a loader rebuilt in a
  live session (or one interrupted mid-epoch) orphans them -- pass it from a
  training loop that runs to completion, not from interactive code."""
  return DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers and num_workers > 0,
    prefetch_factor=2 if num_workers > 0 else None,
    **kwargs,
  )


class EnvironmentSampler(Sampler[int]):
  """Draws each epoch from a random handful of environments rather than all of them.

  An epoch then costs the same however many are downloaded, which is what makes a large catalogue
  usable: 18 environments at once is 88 minutes an epoch, five is a quarter of that, and over
  enough epochs every environment still gets seen. Grouping by environment rather than sampling
  windows at random keeps a batch coherent -- the model sees one building's statistics at a time,
  the way it will on the robot.
  """

  def __init__(
    self,
    subset: TartanGroundSubset,
    envs_per_epoch: int,
    samples_per_epoch: int | None = None,
    seed: int = 0,
  ):
    self.by_env: dict[str, list[int]] = {}
    for position, index in enumerate(subset.indices):
      env = environment_of(subset.dataset.window_trajectory(index))
      self.by_env.setdefault(env, []).append(position)
    self.envs_per_epoch = min(envs_per_epoch, len(self.by_env))
    self.generator = torch.Generator().manual_seed(seed)
    # a fixed count regardless of which environments come up, so every epoch is the same size
    self.samples = samples_per_epoch or round(
      len(subset) * self.envs_per_epoch / len(self.by_env)
    )

  def __len__(self) -> int:
    return self.samples

  def __iter__(self):
    names = list(self.by_env)
    picked = torch.randperm(len(names), generator=self.generator)[: self.envs_per_epoch]
    pool = [i for p in picked for i in self.by_env[names[int(p)]]]
    draw = (
      torch.randint(len(pool), (self.samples,), generator=self.generator)
      if self.samples > len(pool)
      else torch.randperm(len(pool), generator=self.generator)[: self.samples]
    )
    return (pool[int(i)] for i in draw)


@dataclass(frozen=True)
class TartanGroundConfig:
  """Everything a run needs from the data, in one object: which environments, how windows are cut,
  what the camera is, and how the split falls.

  `build()` returns the train and validation subsets, `loaders()` wraps them. The split holds out
  a whole environment when more than one is present, since a second walk through the same building
  shares its geometry and would measure memorisation rather than transfer.
  """

  root: Path
  envs: tuple[str, ...] | None = None  # None takes every environment under root
  val_env: str | None = None  # None holds out the last alphabetically
  version: str = "Data_omni"

  num_frames: int = 7
  stride: int | None = 1
  image_hw: tuple[int, int] = (64, 64)
  camera_names: tuple[str, ...] = ("lcam_front",)
  depth_max: float = 4.0

  source_fovy_deg: float = 90.0  # the cube faces the data was rendered with
  head_fovy_deg: float | None = None  # re-render at the robot's lens instead
  include_head_jitter: bool = False  # aim each frame, rather than straight ahead

  batch_size: int = 4
  num_workers: int = 6
  envs_per_epoch: int | None = None  # None uses every environment each epoch
  samples_per_epoch: int | None = None  # None scales the epoch with envs_per_epoch

  def __post_init__(self) -> None:
    # a config read from json gives root as a str, and every use of it here is a Path
    object.__setattr__(self, "root", Path(self.root))

  @property
  def camera(self) -> tuple[tuple[int, int], float, float]:
    """What a model must be built with to match this data. Only image_hw mismatches loudly; the
    other two agree on shape and quietly place every point wrong."""
    return self.image_hw, self.head_fovy_deg or self.source_fovy_deg, self.depth_max

  def trajectories(self) -> list[Path]:
    if self.envs:
      return [
        t for e in self.envs for t in find_trajectory_dirs(self.root / e, self.version)
      ]
    return find_trajectory_dirs(self.root, self.version)

  def dataset(self) -> "TartanGroundWindowDataset":
    return TartanGroundWindowDataset(
      trajectory_paths=self.trajectories(),
      num_frames=self.num_frames,
      stride=self.stride,
      camera_names=self.camera_names,
      depth_max=self.depth_max,
      image_hw=self.image_hw,
      include_pose=True,
      source_fovy_deg=self.source_fovy_deg,
      head_fovy_deg=self.head_fovy_deg,
      include_head_jitter=self.include_head_jitter,
    )

  def build(self) -> tuple[TartanGroundSubset, TartanGroundSubset, str]:
    """(train, val, what was held out)."""
    ds = self.dataset()
    have = (
      ds.loaded_trajectories
    )  # not what was asked for, if a download is still running
    envs = sorted({environment_of(t) for t in have})
    if len(envs) > 1:
      held = self.val_env or envs[-1]
      if held not in envs:
        raise ValueError(
          f"val_env {held!r} is not among the environments present: {envs}"
        )
      val = [t for t in have if environment_of(t) == held]
      train = [t for t in have if environment_of(t) != held]
    else:
      held = f"{envs[0]} by trajectory, only one environment present"
      val, train = have[:1], have[1:]
    return ds.get_trajectory(train), ds.get_trajectory(val), held

  def loaders(self) -> tuple[DataLoader, DataLoader, str]:
    """(train, val, a line describing the split). Both shuffled: at stride 1 an unshuffled batch
    is near-identical windows, which any across-batch metric reads as the model collapsing."""
    train, val, held = self.build()
    make = lambda s, **kw: make_dataloader(  # noqa: E731
      s, batch_size=self.batch_size, num_workers=self.num_workers, **kw
    )
    if self.envs_per_epoch:
      sampler = EnvironmentSampler(train, self.envs_per_epoch, self.samples_per_epoch)
      train_loader = make(train, sampler=sampler)
      note = f", {sampler.samples} of them an epoch from {sampler.envs_per_epoch} environments"
    else:
      train_loader, note = make(train, shuffle=True), ""
    return (
      train_loader,
      make(val, shuffle=True),
      f"{len(train)} train / {len(val)} val windows, holding out {held}{note}",
    )


def combine(*configs: TartanGroundConfig) -> tuple[DataLoader, DataLoader, str]:
  """Train on several configs at once, for when one source wants a different recipe from another.
  A list of environments belongs in one config; this is for differing window length or jitter.

  Validation comes from the first config, so the held-out set stays one clean thing. All must
  agree on `camera`, since a model is built for exactly one.
  """
  if not configs:
    raise ValueError("combine() needs at least one config")
  cameras = {c.camera for c in configs}
  if len(cameras) > 1:
    raise ValueError(f"configs disagree on the camera: {sorted(cameras)}")

  trains, val, held = [], None, ""
  for i, c in enumerate(configs):
    tr, va, h = c.build()
    trains.append(tr)
    if i == 0:
      val, held = va, h
  first = configs[0]
  train = trains[0] if len(trains) == 1 else ConcatDataset(trains)
  make = lambda s: make_dataloader(  # noqa: E731
    s, batch_size=first.batch_size, shuffle=True, num_workers=first.num_workers
  )
  assert val is not None
  return (
    make(train),
    make(val),
    f"{len(train)} train / {len(val)} val windows, holding out {held}",
  )
