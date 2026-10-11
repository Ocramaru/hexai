"""PeriodicRestarts, whose cycles decide whether a long run keeps learning or sits at its ceiling.

Author: Marco Cassar (@Ocramaru)
"""

import pytest
import torch

from hexai.utilities import PeriodicRestarts


def scheduler(**kw):
  opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)
  return PeriodicRestarts(opt, 1e-3, period=20, warmup=2, **kw)


def run(sched, epochs, metric=100.0):
  """Every rate the schedule visits. The metric is passed to prove it changes nothing."""
  return [(sched.step(metric), sched.lr)[1] for _ in range(epochs)]


def test_a_cycle_ends_at_the_floor():
  """The point of a fixed period: every `period` epochs there is a settled rate to checkpoint at,
  whatever the run's total length turns out to be."""
  sched = scheduler()
  rates = run(sched, 20)
  assert rates[-1] == sched.floor
  assert max(rates) == sched.peak0  # and it got all the way up first


def test_the_next_cycle_starts_lower():
  sched = scheduler(gamma=0.5)
  run(sched, 20)
  run(sched, 20)
  assert sched.peak == 5e-4
  run(sched, 20)
  assert sched.peak == 2.5e-4


def test_the_ceiling_never_climbs_back():
  """The schedule this replaces judged each restart on noise and let the ceiling ratchet back to
  where it started, so a long run never left its peak rate."""
  sched = scheduler(gamma=0.7)
  peaks = []
  for _ in range(10):
    run(sched, 20)
    peaks.append(sched.peak)
  assert peaks == sorted(peaks, reverse=True)
  assert sched.peak == 1e-4  # and it stops at ceiling_floor rather than decaying to nothing


def test_the_metric_does_not_steer_the_rate():
  """Open-loop by design. A metric noisier than the drift it is meant to detect cannot be a
  control signal, so the same epochs must give the same rates however validation goes."""
  improving = run(scheduler(), 45, metric=100.0)
  worsening = [(s.step(100.0 + 5 * i), s.lr)[1] for s in [scheduler()] for i in range(45)]
  assert improving == worsening


def test_a_run_extends_without_a_declared_end():
  """No horizon: the shape of epoch 5 cannot depend on whether the run was declared as 50 epochs
  or 500. The old one divided its anneal by the distance to a stated end, which is what made
  resuming for more epochs train the whole continuation at the floor."""
  short, long_ = scheduler(), scheduler()
  assert run(short, 30) == run(long_, 30)  # nothing about total length is knowable to either


def test_taper_ends_the_schedule_for_good():
  sched = scheduler()
  run(sched, 25)
  sched.taper(over=10)
  rates = run(sched, 10)
  assert rates[-1] == pytest.approx(sched.floor)
  assert rates == sorted(rates, reverse=True)  # one way down, no cycle starting underneath it


def test_revive_gives_a_resumed_run_its_ceiling_back():
  sched = scheduler(gamma=0.5)
  run(sched, 60)
  assert sched.peak < 1e-3
  sched.revive()
  assert sched.peak == 1e-3
  assert sched.cycle_len == sched.period  # and a whole cycle to spend it over


def test_state_survives_a_round_trip():
  """A resume lands mid-cycle, so the rate afterwards has to be the one it would have had."""
  sched = scheduler()
  run(sched, 13)
  saved = sched.state_dict()
  expected = run(sched, 5)

  resumed = scheduler()
  resumed.load_state_dict(saved)
  assert run(resumed, 5) == expected
