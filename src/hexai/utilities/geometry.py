"""
Geometry for lifting posed RGB+depth frames into a voxel grid, and rendering one back.

Part of the encoder's forward pass rather than preprocessing: gradients flow through it,
and it has to be inside whatever gets exported to the robot.

Author: Marco Cassar (@Ocramaru)
"""

from typing import Literal, overload

import torch
import torch.nn.functional as F
from mjlab.utils.lab_api.math import matrix_from_quat

from hexai.utilities.camera import pixel_to_camera_ray


def lift_window(frames: torch.Tensor, poses: torch.Tensor, bounds: torch.Tensor,
                resolution: tuple[int, int, int], fovy_deg: float, depth_max: float = 4.0,
                include_splatting: bool = True):
  """Unprojects a [4, T, H, W] rgb+depth window into one [1, *resolution] canonical grid of
  occupancy. Depth places the point rather than being carried, so what the grid holds is confidence
  about a piece of space; colour informs where the mass goes upstream of here and is not deposited,
  since the decoder renders geometry alone."""
  h, w = frames.shape[-2:]
  rays = pixel_to_camera_ray((h, w), fovy_deg).to(frames)
  v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
  pixels = torch.stack([u.flatten(), v.flatten(), torch.ones(h * w)]).to(frames)
  mass = frames.new_zeros(0, h * w)  # no values to sum: the hit count alone is the confidence
  counts = torch.zeros(*resolution).to(frames)
  for t in range(frames.shape[1]):
    points = (rays @ pixels) * (frames[3, t] * depth_max).flatten()
    points = matrix_from_quat(poses[t, 3:]) @ points + poses[t, :3].unsqueeze(1)
    counts = counts + scatter_to_grid(points, mass, bounds, resolution, include_splatting)[1]
  density = -torch.expm1(-counts)  # saturating, so splatting's soft edges survive as soft density
  return density.unsqueeze(0)


@overload
def render_view(volume: torch.Tensor, pose: torch.Tensor, bounds: torch.Tensor,
                image_hw: tuple[int, int], fovy_deg: float, depth_max: float = ...,
                include_depth: bool = ..., occlusion: Literal[False] = ...,
                gate: torch.Tensor | None = ...) -> torch.Tensor: ...
@overload
def render_view(volume: torch.Tensor, pose: torch.Tensor, bounds: torch.Tensor,
                image_hw: tuple[int, int], fovy_deg: float, depth_max: float = ...,
                include_depth: bool = ..., *, occlusion: Literal[True],
                gate: torch.Tensor | None = ...) -> tuple[torch.Tensor, torch.Tensor]: ...
def render_view(volume: torch.Tensor, pose: torch.Tensor, bounds: torch.Tensor,
                image_hw: tuple[int, int], fovy_deg: float, depth_max: float = 4.0,
                include_depth: bool = False, occlusion: bool = False,
                gate: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
  """Renders a [C, X, Y, Z] canonical volume from a query pose into an image, marching each pixel's
  ray and compositing front to back with the last channel as density. Trilinear throughout, so
  gradients reach the volume.

  Four samples per smallest voxel. Fewer and oblique rays slip between voxels, which bands flat
  surfaces and worsens as resolution rises.

  include_depth appends expected depth, read off the same weights, so a pure density volume still
  renders something to supervise against.

  occlusion also returns the density where transmittance has already collapsed: the complement of
  the compositing weights, which is mass the image cannot see and so never constrains.

  gate is a [1, X, Y, Z] observedness volume that narrows that to mass no context frame saw either.
  Hidden from one query pose is not the same as never observed -- most of what a query occludes,
  the other frames did see -- so ungated this charges mostly real geometry."""
  h, w = image_hw
  voxel = ((bounds[1] - bounds[0]) / torch.tensor(volume.shape[1:]).to(bounds)).min()
  samples = int(4 * depth_max / voxel)
  v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
  pixels = torch.stack([u.flatten(), v.flatten(), torch.ones(h * w)]).to(volume)
  rays = matrix_from_quat(pose[3:]).to(volume) @ (pixel_to_camera_ray(image_hw, fovy_deg).to(volume) @ pixels)
  depths = torch.linspace(0.05, depth_max, samples).to(volume)
  points = pose[:3].to(volume)[:, None, None] + rays[:, :, None] * depths  # [3, H*W, S]

  unit = (points - bounds[0, :, None, None]) / (bounds[1] - bounds[0])[:, None, None] * 2 - 1
  grid = unit.flip(0).permute(1, 2, 0)[None, None]  # grid_sample reads xyz reversed
  sampled = F.grid_sample(volume[None], grid, align_corners=False, padding_mode="zeros")[0, :, 0]
  colour, density = sampled[:-1], sampled[-1].clamp(0, 1)
  behind = torch.cumprod(1 - density, dim=-1)[..., :-1]  # shifted, not divided: dividing the
  ones = torch.ones_like(behind[..., :1])                # inclusive product underflows in float32
  transmittance = torch.cat([ones, behind], -1)
  weights = density * transmittance

  parts = [(colour * weights).sum(-1)] if len(colour) else []
  if include_depth:
    parts.append(((weights * depths).sum(-1) / depth_max).unsqueeze(0))
  image = torch.cat(parts).view(-1, h, w)
  if not occlusion:
    return image
  # detached, so this charges the hidden mass and not the surface doing the hiding: attached, a
  # denser front surface raises the bill on everything behind it and transparency is the way out
  charge = density * (1 - transmittance.detach())
  if gate is not None:
    seen = F.grid_sample(gate[None], grid, align_corners=False, padding_mode="zeros")[0, 0, 0]
    charge = charge * (1 - seen.detach().clamp(0, 1))
  return image, charge.mean()


def carve_window(frames: torch.Tensor, poses: torch.Tensor, bounds: torch.Tensor,
                 resolution: tuple[int, int, int], fovy_deg: float, depth_max: float = 4.0,
                 margin: float = 0.15) -> torch.Tensor:
  """[4, T, H, W] rgb+depth and [T, 7] poses -> [2, *resolution] of (occupied, observed), both in
  [0, 1]. Splits the box into the three states a camera can actually report, where the density the
  lift produces only has two:

    free      the voxel sits in front of the measured surface, a ray passed through it
    occupied  the voxel straddles the surface, within margin of it
    unknown   behind the surface, or outside the camera's view -- never seen, which is not the
              same as empty, and is the distinction that lets a model say it does not know

  Projective rather than ray-marched: each voxel centre is thrown into every camera and compared
  against that pixel's depth, so it costs voxels x frames rather than rays x samples."""
  res = torch.tensor(resolution, device=frames.device)
  idx = torch.stack(torch.meshgrid(*[torch.arange(int(n), device=frames.device) for n in res],
                                   indexing="ij"), dim=-1)
  size = (bounds[1] - bounds[0]) / res
  centres = (bounds[0] + (idx + 0.5) * size).reshape(-1, 3).T  # [3, V]

  h, w = frames.shape[-2:]
  k = torch.linalg.inv(pixel_to_camera_ray((h, w), fovy_deg).to(frames))
  free = torch.zeros(centres.shape[1], device=frames.device, dtype=frames.dtype)
  occupied = torch.zeros_like(free)

  for t in range(frames.shape[1]):
    rot = matrix_from_quat(poses[t, 3:])
    local = rot.T @ (centres - poses[t, :3].unsqueeze(1))  # canonical -> this camera
    pixels = k @ local
    depth_along_axis = local[2]
    u, v = pixels[0] / pixels[2], pixels[1] / pixels[2]
    inside = (depth_along_axis > 0) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)

    grid = torch.stack([u / (w - 1) * 2 - 1, v / (h - 1) * 2 - 1], dim=-1)[None, None]
    measured = F.grid_sample(frames[3, t][None, None], grid, mode="nearest",
                             align_corners=True)[0, 0, 0] * depth_max
    delta = measured - depth_along_axis
    free += (inside & (delta > margin)).to(frames)
    occupied += (inside & (delta.abs() <= margin)).to(frames)

  seen = free + occupied
  return torch.stack([(occupied / seen.clamp(min=1)).view(*resolution),
                      -torch.expm1(-seen).view(*resolution)])


_CORNERS = [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]


def scatter_to_grid(points: torch.Tensor, values: torch.Tensor, bounds: torch.Tensor,
                    resolution: tuple[int, int, int], include_splatting: bool = True
                    ) -> tuple[torch.Tensor, torch.Tensor]:
  """Bins [3, N] points into a voxel grid, summing their [C, N] values and their weight, and
  dropping whatever falls outside bounds ([2, 3], lower then upper). Sums rather than assigns
  because many points share a voxel. Differentiates through the values but not the indices.

  Splatting spreads each point trilinearly over the eight voxels around it, which is what keeps
  a surface continuous; hard binning drops it in one voxel and perforates surfaces once the grid
  is fine enough that neighbouring voxels go empty."""
  res = torch.tensor(resolution, device=points.device).unsqueeze(1)
  coord = (points - bounds[0, :, None]) / (bounds[1] - bounds[0])[:, None] * res
  coord = coord - 0.5 if include_splatting else coord
  base = coord.floor()
  frac = coord - base if include_splatting else torch.zeros_like(coord)  # frac() keeps the sign

  voxels = resolution[0] * resolution[1] * resolution[2]
  grid = torch.zeros(len(values), voxels, device=points.device, dtype=values.dtype)
  weights = torch.zeros(voxels, device=points.device, dtype=values.dtype)
  for corner in _CORNERS if include_splatting else _CORNERS[:1]:
    offset = torch.tensor(corner, device=points.device).unsqueeze(1)
    idx = (base + offset).long()
    inside = ((idx >= 0) & (idx < res)).all(0)
    weight = (offset * frac + (1 - offset) * (1 - frac)).prod(0)[inside].to(values)
    flat = (idx[0] * resolution[1] + idx[1])[inside] * resolution[2] + idx[2][inside]
    grid.index_add_(1, flat, values[:, inside] * weight)
    weights.index_add_(0, flat, weight)
  return grid.view(-1, *resolution), weights.view(*resolution)


@torch.no_grad()
def answerable_mask(carved: torch.Tensor, truth: torch.Tensor, query: torch.Tensor,
                    bounds: torch.Tensor, image_hw: tuple[int, int], fovy_deg: float,
                    depth_max: float = 4.0):
  """[B, H, W] in [0, 1]: for each target pixel, how much the context frames observed the point it
  sits on. Read from the carving, not from any model's own coverage, which would be gameable."""
  h, w = image_hw
  k_inv = pixel_to_camera_ray((h, w), fovy_deg).to(truth)
  v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
  pix = torch.stack([u.flatten(), v.flatten(), torch.ones(h * w)]).to(truth)
  rays = (k_inv @ pix)[None]                               # [1, 3, H*W]
  metres = (truth[:, -1] * depth_max).flatten(1)[:, None]  # [B, 1, H*W]
  rot = torch.stack([matrix_from_quat(q[3:]) for q in query])
  pts = rot @ (rays * metres) + query[:, :3, None]         # canonical frame

  lower = bounds[0].to(truth)[None, :, None]
  span = (bounds[1] - bounds[0]).to(truth)[None, :, None]
  unit = ((pts - lower) / span * 2 - 1).flip(1)            # grid_sample reads xyz reversed
  grid = unit.permute(0, 2, 1)[:, None, None]              # [B, 1, 1, H*W, 3]
  seen = F.grid_sample(carved[:, 1:], grid, align_corners=False, padding_mode="zeros")
  return seen.view(-1, h, w).clamp(0, 1)


def lift_volume(camera_volume: torch.Tensor, poses: torch.Tensor, bounds: torch.Tensor,
                resolution: tuple[int, int, int], fovy_deg: float,
                depth_max: float = 4.0) -> torch.Tensor:
  """[N, C, D, H, W] volumes in each camera's own frame and [N, 7] poses -> [N, C, *resolution] in
  the canonical one, by asking where each canonical voxel falls in each camera and sampling there.
  The inverse of how carve_window projects, and only a pose can do it, so it is not a convolution.

  Every frame shares the voxel centres and differs only by pose, so a batch is one grid_sample.
  Voxels behind a camera go out of range and pad to zero rather than aliasing onto the mirrored
  ray in front."""
  res = torch.tensor(resolution, device=camera_volume.device)
  idx = torch.stack(torch.meshgrid(*[torch.arange(int(n), device=camera_volume.device) for n in res],
                                   indexing="ij"), dim=-1)
  centres = (bounds[0] + (idx + 0.5) * (bounds[1] - bounds[0]) / res).reshape(-1, 3).T  # [3, V]

  n, _, _, h, w = camera_volume.shape
  k = torch.linalg.inv(pixel_to_camera_ray((h, w), fovy_deg).to(camera_volume))
  local = matrix_from_quat(poses[:, 3:]).transpose(1, 2) @ (centres - poses[:, :3, None])
  along = local[:, 2]
  pixels = k @ local
  u, v = pixels[:, 0] / along.clamp(min=1e-6), pixels[:, 1] / along.clamp(min=1e-6)

  grid = torch.stack([u / (w - 1) * 2 - 1, v / (h - 1) * 2 - 1, along / depth_max * 2 - 1], dim=-1)
  grid = torch.where((along > 0).unsqueeze(-1), grid, torch.full_like(grid, 2.0))
  return F.grid_sample(camera_volume, grid.view(n, *resolution, 3), align_corners=False,
                       padding_mode="zeros")
