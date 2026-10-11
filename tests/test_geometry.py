"""The geometry conventions, each of which has cost a bug at least once.

Index order, which direction a pose points, grid_sample's reversed coordinates and splat weights
at the edges of the box are all invisible when wrong: nothing raises, the model just fails to
converge. These pin them to numbers worked out by hand.

Author: Marco Cassar (@Ocramaru)
"""

import pytest
import torch

from hexai.data import add_depth_noise
from hexai.utilities import (
  answerable_mask,
  carve_window,
  lift_volume,
  lift_window,
  render_view,
  scatter_to_grid,
)
from hexai.utilities.camera import pixel_to_camera_ray

BOUNDS = torch.tensor([[-2.0, -2.0, 0.0], [2.0, 2.0, 4.0]])  # 4 m cube, camera at the near face
RES = (16, 16, 16)
IDENTITY = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])  # at the origin, looking down +z


@pytest.fixture
def flat_wall():
  """One frame whose every pixel reports a surface at 2 m, half of the 4 m range."""
  frames = torch.zeros(4, 1, 32, 32)
  frames[:3, 0], frames[3, 0] = 0.7, 0.5
  return frames


def test_centre_pixel_looks_down_the_optical_axis():
  ray = pixel_to_camera_ray((64, 64), 90.0) @ torch.tensor([32.0, 32.0, 1.0])
  assert torch.allclose(ray, torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)


def test_focal_length_comes_from_height_not_width():
  """Square pixels give one focal length but two fovs, and fovy is what both mjlab and the robot
  are specified by. A focal read off the width puts every unprojected point at the wrong angle."""
  k_inv = pixel_to_camera_ray((32, 64), 90.0)  # wide, so height and width disagree
  assert 1 / k_inv[0, 0] == pytest.approx(16.0, abs=1e-3)  # (h / 2) / tan(45 deg)


def test_a_point_lands_in_the_voxel_that_contains_it():
  """Index order is (x, y, z) throughout, and flattening is z fastest."""
  bounds = torch.tensor([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
  point = torch.tensor([[0.5], [1.5], [2.5]])  # inside voxel (0, 1, 2) at resolution 4
  grid, _ = scatter_to_grid(point, torch.ones(1, 1), bounds, (4, 4, 4), include_splatting=False)
  assert grid[0].nonzero().tolist() == [[0, 1, 2]]


def test_splat_weights_are_a_partition_of_unity():
  """Trilinear splatting spreads one point over eight voxels. Weights that do not sum to one make
  density depend on where in a voxel a surface happens to fall."""
  points = torch.rand(3, 64) * 3.0 + 0.5  # comfortably inside a 0..4 box
  bounds = torch.tensor([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
  _, weights = scatter_to_grid(points, torch.ones(1, 64), bounds, (8, 8, 8))
  assert weights.sum() == pytest.approx(64.0, abs=1e-3)
  assert weights.min() >= 0.0


def test_splat_weights_stay_non_negative_at_the_edges():
  """A point outside the box has corners on both sides of the boundary. The ones that fall outside
  are dropped, so what is left must still be a non-negative fraction rather than the remainder."""
  bounds = torch.tensor([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
  outside = torch.tensor([[-0.3, 4.3, 2.0], [2.0, 2.0, -1.0], [2.0, 2.0, 5.0]]).T
  _, weights = scatter_to_grid(outside, torch.ones(1, 3), bounds, (8, 8, 8))
  assert weights.min() >= 0.0
  assert weights.sum() <= 3.0 + 1e-4  # never more than one unit of weight per point


def test_lift_puts_the_surface_where_the_depth_says(flat_wall):
  """A wall at 2 m of a 4 m box, 16 voxels deep, belongs either side of index 8. The lift carries
  occupancy alone now, so the grid is one channel."""
  occupancy = lift_window(flat_wall, IDENTITY, BOUNDS, RES, 90.0, 4.0)[-1]
  heaviest = int(occupancy.sum(dim=(0, 1)).argmax())
  assert heaviest in (7, 8)


def test_carve_reports_free_in_front_and_nothing_behind(flat_wall):
  """The three states a camera can report. Behind the surface is unknown, not empty, which is the
  distinction the whole unknown channel exists to carry."""
  occupied, observed = carve_window(flat_wall, IDENTITY, BOUNDS, RES, 90.0, 4.0)
  assert observed[8, 8, 4] > 0.5   # in front, a ray passed through it
  assert observed[8, 8, 8] > 0.5   # at the surface
  assert observed[8, 8, 14] == 0.0  # behind, never seen
  assert occupied[8, 8, 8] == pytest.approx(1.0)


def test_render_reads_back_the_depth_of_a_slab():
  """grid_sample takes its coordinates reversed, so a volume rendered without the flip comes back
  transposed and this depth lands somewhere else entirely."""
  volume = torch.zeros(1, 16, 16, 16)
  volume[0, :, :, 8] = 1.0  # solid sheet two metres down the z axis
  pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  rendered = render_view(volume, pose, BOUNDS, (8, 8), 90.0, depth_max=4.0, include_depth=True)
  assert rendered.shape == (1, 8, 8)
  assert float(rendered[0, 4, 4]) * 4.0 == pytest.approx(2.1, abs=0.25)


def test_occlusion_charges_for_mass_the_render_cannot_see():
  """Two identical slabs, one visible and one hidden behind it. The render is blind to the second,
  which is the whole problem -- so the penalty has to see it, and must not charge for the first."""
  pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  front = torch.zeros(1, 16, 16, 16)
  front[0, :, :, 4] = 1.0  # a metre out, in full view
  both = front.clone()
  both[0, :, :, 12] = 1.0  # three metres out, entirely behind the first

  empty = render_view(torch.zeros(1, 16, 16, 16), pose, BOUNDS, (8, 8), 90.0, 4.0,
                      include_depth=True, occlusion=True)[1]
  visible, hidden = (render_view(v, pose, BOUNDS, (8, 8), 90.0, 4.0, include_depth=True,
                                 occlusion=True)[1] for v in (front, both))
  assert float(empty) == pytest.approx(0.0)  # an empty volume wins this outright, which is the trap
  assert float(hidden) > float(visible) * 1.4  # the hidden slab costs, though it renders identically
  # a surface still pays for its own far side, since transmittance collapses inside the first
  # sample of it. That floor is real, and small beside a volume that fills the room
  assert float(visible) < float(hidden) < 0.2


def test_the_gate_spares_geometry_a_context_frame_saw():
  """Hidden from the query is not the same as never observed. Most of what one pose occludes, the
  other frames did see, so ungated the charge lands mostly on real geometry."""
  pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  volume = torch.zeros(1, 16, 16, 16)
  volume[0, :, :, 4] = 1.0   # the surface the query renders
  volume[0, :, :, 12] = 1.0  # and mass behind it, which the query cannot see

  seen_none = torch.zeros(1, 16, 16, 16)  # nothing was observed, so everything hidden is invented
  seen_all = torch.ones(1, 16, 16, 16)    # every frame saw it all, so nothing is invented
  def charge(g):
    return float(render_view(volume, pose, BOUNDS, (8, 8), 90.0, 4.0, include_depth=True,
                             occlusion=True, gate=g)[1])
  assert charge(seen_all) == pytest.approx(0.0, abs=1e-6)  # fully observed costs nothing
  assert charge(seen_none) > 0.01  # unobserved and hidden is the whole point
  assert charge(None) == pytest.approx(charge(seen_none), rel=1e-5)  # no gate is the ungated charge


def test_occlusion_is_off_unless_asked_for():
  """Every other caller wants an image, not a tuple, so the flag has to be opt-in."""
  volume = torch.zeros(1, 16, 16, 16)
  volume[0, :, :, 8] = 1.0
  pose = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  assert isinstance(render_view(volume, pose, BOUNDS, (8, 8), 90.0, 4.0, True), torch.Tensor)


def test_answerable_mask_is_high_where_the_camera_looked(flat_wall):
  """The mask is read from the carving rather than the model's own coverage, so a target pixel
  whose point a context frame saw scores near one."""
  carved = carve_window(flat_wall, IDENTITY, BOUNDS, RES, 90.0, 4.0)[None]
  truth = flat_wall[3:, 0][None]  # the depth channel as the render target
  mask = answerable_mask(carved, truth, IDENTITY[0][None], BOUNDS, (32, 32), 90.0, 4.0)
  assert mask.shape == (1, 32, 32)
  assert float(mask[0, 16, 16]) > 0.5


def test_depth_noise_leaves_rgb_alone_and_disagrees_per_frame():
  """The estimate is made one frame at a time, so the frames of a window must not share a scale.
  A single window-wide error would displace the volume; per-frame error smears it."""
  frames = torch.rand(2, 4, 7, 16, 16) * 0.5 + 0.25
  noisy = add_depth_noise(frames, scale=0.1, shift=0.0, relative=0.0, bleed=0.0)
  assert torch.equal(noisy[:, :3], frames[:, :3])
  ratio = (noisy[0, 3] / frames[0, 3]).mean(dim=(-2, -1))  # one scale per frame
  assert ratio.std() > 0.01


def test_depth_noise_stays_in_range():
  """Depth is carried normalised over depth_max, and the lift reads it as metres directly."""
  frames = torch.rand(2, 4, 7, 16, 16)
  noisy = add_depth_noise(frames, scale=0.5, shift=0.2, relative=0.3)
  assert noisy[:, 3].min() >= 0.0
  assert noisy[:, 3].max() <= 1.0


def test_lift_volume_lands_a_depth_bin_at_its_canonical_depth():
  """A camera volume is in the camera's frame, so only a pose can move it into the box. Getting the
  depth axis or the pose direction backwards puts the geometry somewhere else entirely."""
  camera_volume = torch.zeros(2, 1, 32, 32, 32)
  camera_volume[:, 0, 16] = 1.0  # solid at depth bin 16 of 32 over 4 m, so a shell at 2 m
  poses = torch.tensor([[0.0, 0, 0, 1, 0, 0, 0], [0.0, 0, 1, 1, 0, 0, 0]])
  out = lift_volume(camera_volume, poses, BOUNDS, RES, 90.0, 4.0)
  assert int(out[0, 0].sum(dim=(0, 1)).argmax()) == 8   # 2 m of a 4 m box, 16 deep
  assert int(out[1, 0].sum(dim=(0, 1)).argmax()) == 12  # camera moved +1 m, so the shell does too


def test_lift_volume_drops_what_is_behind_the_camera():
  """Voxels behind the camera project onto the mirrored ray in front, so they must be pushed out
  of range rather than sampled: otherwise the box fills with a reflection of itself."""
  camera_volume = torch.ones(1, 1, 16, 16, 16)
  behind = torch.tensor([[0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 0.0]])  # past the far face, looking away
  out = lift_volume(camera_volume, behind, BOUNDS, RES, 90.0, 4.0)
  assert float(out.sum()) == pytest.approx(0.0)
