"""Learning-rate decay shapes. Model agnostic, and used by schedulers.py.

Every shape takes progress through a segment in [0, 1] and returns a multiplier that starts at 1
and ends at or near 0, so they are interchangeable and comparable. What separates them is where
they spend their time: measured over a 56-epoch segment, the fraction of it spent above a tenth of
the peak rate is 91% linear, 82% cosine, 71% quadratic, 50% exponential, 11% inverse.

That matters most when nothing interrupts the anneal. Under a scheduler that restarts often, the
segments are short and every shape behaves much the same.

Author: Marco Cassar (@Ocramaru)
"""

import math
from collections.abc import Callable
from typing import Literal

Decay = Literal["cosine", "linear", "quadratic", "sqrt", "exponential", "inverse", "step", "constant"]

DECAYS: dict[Decay, Callable[[float], float]] = {  # the type checker holds these keys to Decay
  "cosine": lambda f: (1 + math.cos(math.pi * f)) / 2,  # the usual default, what onecycle anneals
  "linear": lambda f: 1 - f,  # holds high longest of the shapes that reach zero
  "quadratic": lambda f: (1 - f) ** 2,  # linear with an earlier bend
  "sqrt": lambda f: (1 - f) ** 0.5,  # linear with a later one, nearly all of it near peak
  "exponential": lambda f: 0.01**f,  # fixed fraction per step, so it front-loads the decay
  "inverse": lambda f: 1 / (1 + 99 * f),  # the classic SGD rate, almost all of it near the floor
  "step": lambda f: 0.1 ** min(int(f * 3), 3),  # three cliffs, what ResNet-era training used
  "constant": lambda f: 1.0,  # no anneal at all
}


def decay_at(shape: Decay, progress: float, peak: float, floor: float = 0.0) -> float:
  """The rate at `progress` through a segment annealing `peak` down toward `floor`. Progress is
  clamped, so overrunning a segment holds the final rate rather than turning it negative."""
  return floor + (peak - floor) * DECAYS[shape](min(max(progress, 0.0), 1.0))
