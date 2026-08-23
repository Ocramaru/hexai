"""
Geometry for lifting posed RGB+depth frames into a voxel grid, and rendering one back.

Part of the encoder's forward pass rather than preprocessing: gradients flow through it,
and it has to be inside whatever gets exported to the robot.

Author: Marco Cassar (@Ocramaru)
"""

import torch
import torch.nn.functional as F
from mjlab.utils.lab_api.math import matrix_from_quat

from hexai.utilities.camera import pixel_to_camera_ray


def lift_window(frames: torch.Tensor, poses: torch.Tensor, bounds: torch.Tensor,
                resolution: tuple[int, int, int], fovy_deg: float, depth_max: float = 4.0,
                include_splatting: bool = True):
  """Unprojects a [4, T, H, W] rgb+depth window into one [4, *resolution] canonical grid: mean rgb
  over every ray that hit each voxel, plus occupancy. Depth places the point rather than being
  carried, so what the grid holds is appearance and confidence about a piece of space."""
  h, w = frames.shape[-2:]
  rays = pixel_to_camera_ray((h, w), fovy_deg).to(frames)
  v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
  pixels = torch.stack([u.flatten(), v.flatten(), torch.ones(h * w)]).to(frames)
  totals = torch.zeros(3, *resolution).to(frames)
  counts = torch.zeros(*resolution).to(frames)
  for t in range(frames.shape[1]):
    points = (rays @ pixels) * (frames[3, t] * depth_max).flatten()
    points = matrix_from_quat(poses[t, 3:]) @ points + poses[t, :3].unsqueeze(1)
    values = frames[:3, t].flatten(1)
    summed, hit = scatter_to_grid(points, values, bounds, resolution, include_splatting)
    totals, counts = totals + summed, counts + hit
  density = -torch.expm1(-counts)  # saturating, so splatting's soft edges survive as soft density
  return torch.cat([totals / counts.clamp(min=1), density.unsqueeze(0)])


def render_view(volume: torch.Tensor, pose: torch.Tensor, bounds: torch.Tensor,
                image_hw: tuple[int, int], fovy_deg: float, depth_max: float = 4.0,
                include_depth: bool = False) -> torch.Tensor:
  """Renders a [C, X, Y, Z] canonical volume from a query pose into an image, marching each pixel's
  ray and compositing front to back with the last channel as density. Trilinear throughout, so
  gradients reach the volume.

  Four samples per smallest voxel. Fewer and oblique rays slip between voxels, which bands flat
  surfaces and worsens as resolution rises.

  include_depth appends expected depth, read off the same weights, so a pure density volume still
  renders something to supervise against."""
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
  weights = density * torch.cat([ones, behind], -1)

  parts = [(colour * weights).sum(-1)] if len(colour) else []
  if include_depth:
    parts.append(((weights * depths).sum(-1) / depth_max).unsqueeze(0))
  return torch.cat(parts).view(-1, h, w)


def carve_window(frames: torch.Tensor, poses: torch.Tensor, bounds: torch.Tensor,
                 resolution: tuple[int, int, int], fovy_deg: float, depth_max: float = 4.0,
                 margin: float = 0.15) -> torch.Tensor:
  """[4, T, H, W] rgb+depth and [T, 7] poses -> [2, *resolution] of (occupied, observed), both in
  [0, 1]. Splits the box into the three states a camera can actually report, where the density the
  lift produces only has two:

    free      the voxel sits in front of the measured surface, a ray passed through it
    occupied  the voxel straddles the surface, within margin of it
    unknown   behind the surface, or outside the frustum -- never seen, which is not the same as
              empty, and is the distinction that lets a model say it does not know

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
