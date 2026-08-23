"""Camera intrinsics. Not tied to any model, and used by both the loaders and the lift.

Author: Marco Cassar (@Ocramaru)
"""

import torch


def pixel_to_camera_ray(image_hw: tuple[int, int], fovy_deg: float) -> torch.Tensor:
  """[3, 3] mapping homogeneous pixels to camera-frame rays, the inverse of the pinhole intrinsics.
  A ray times a metric depth is the point that pixel saw.

  Focal length comes from height and vertical fov, since square pixels give one focal length but
  two fovs, and both mjlab and the robot's camera are specified by fovy. The principal point is the
  image centre, so an off-centre crop would need its own."""
  h, w = image_hw
  f = (h / 2) / torch.tan(torch.tensor(fovy_deg).deg2rad() / 2)
  return torch.linalg.inv(
    torch.tensor([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])
  )
