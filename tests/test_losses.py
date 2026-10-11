"""What each recon residual asks of an ambiguous ray, which is what decides whether the decoder
commits to a surface or renders the average of two.

Author: Marco Cassar (@Ocramaru)
"""

import pytest
import torch

from hexai.models.vvae.train import DEPTH_SPACES, RESIDUALS


def best_guess(residual, targets, space="linear"):
  """The single depth that minimises this residual against a set of targets -- what a ray sees when
  the same geometry is ambiguous across the batch. Solved by search, so no closed form is assumed."""
  warp = DEPTH_SPACES[space]
  grid = torch.linspace(0.05, 1.0, 400)
  cost = torch.stack([RESIDUALS[residual](warp(g.expand_as(targets)) - warp(targets)).sum()
                      for g in grid])
  return float(grid[int(cost.argmin())])


def test_mse_hedges_between_two_surfaces_and_l1_picks_one():
  """A wall at 1 m in half the samples and open to 3 m in the other half, over a 4 m box. mse is
  minimised at the mean, which is a surface that was never there; l1 at the median, which is one of
  the two. This is the blur, and it is the objective asking for it rather than the model failing."""
  near, far = 0.25, 0.75  # 1 m and 3 m, normalised over depth_max
  targets = torch.tensor([near] * 8 + [far] * 8)

  assert best_guess("mse", targets) == pytest.approx(0.5, abs=0.01)  # dead between, never there
  picked = best_guess("l1", targets)
  assert min(abs(picked - near), abs(picked - far)) < 0.01  # sits on one of the two surfaces


def test_an_outlying_ray_drags_mse_further_than_l1():
  """Depth from a monocular estimate carries gross errors, not just noise. One bad target moves the
  mse optimum a long way and the l1 optimum barely at all."""
  clean = torch.tensor([0.25] * 15)
  fouled = torch.cat([clean, torch.tensor([1.0])])
  assert abs(best_guess("mse", fouled) - 0.25) > 3 * abs(best_guess("l1", fouled) - 0.25)


def test_log_and_inverse_charge_near_surfaces_more_than_far_ones():
  """Linear charges the same metres anywhere, so a 0.5 m error at 4 m costs as much as at 1 m --
  yet only the near one decides whether the robot can step. Relative spaces fix the weighting."""
  err = 0.125  # half a metre of a four metre box
  for space in ("log", "inverse"):
    warp = DEPTH_SPACES[space]
    near = abs(float(warp(torch.tensor(0.25 + err)) - warp(torch.tensor(0.25))))
    far = abs(float(warp(torch.tensor(0.75 + err)) - warp(torch.tensor(0.75))))
    assert near > far, f"{space} should charge the near error more"
  warp = DEPTH_SPACES["linear"]
  near = abs(float(warp(torch.tensor(0.25 + err)) - warp(torch.tensor(0.25))))
  far = abs(float(warp(torch.tensor(0.75 + err)) - warp(torch.tensor(0.75))))
  assert near == far  # which is the thing the other two are there to change


def test_every_space_stays_finite_at_zero_depth():
  """Depth arrives normalised and can be zero, and a run that dies on a log at epoch 40 is worse
  than one that never started."""
  for space, warp in DEPTH_SPACES.items():
    out = warp(torch.zeros(4))
    assert torch.isfinite(out).all(), space
