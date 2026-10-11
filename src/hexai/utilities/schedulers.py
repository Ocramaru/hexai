"""Learning-rate schedulers. Model agnostic, and built on the shapes in decay.py.

Author: Marco Cassar (@Ocramaru)
"""

from hexai.utilities.decay import Decay, decay_at


class PeriodicRestarts:
  """Cycles that anneal to the floor and start again from a lower ceiling.

  A cycle is a fixed span of epochs rather than a fraction of the run, so extending a run adds
  cycles instead of reshaping the ones already spent, and nothing here reads a metric. What a
  metric-driven version cost is in the VVAE's .agents/learned.md."""

  def __init__(self, optimizer, lr, period=50,
               shape: Decay = "cosine",  # any of decay.py's; linear holds near the peak, inverse near floor
               t_mult=1.0, gamma=0.7, ceiling_floor=0.1,
               warmup=5,  # each cycle re-warms: a peak that trains well warmed blows up cold
               div=25.0, floor=1e-6):
    self.opt, self.peak, self.shape = optimizer, lr, shape
    self.peak0, self.gamma, self.ceiling_floor = lr, gamma, ceiling_floor
    self.period, self.t_mult, self.cycle_len = period, t_mult, period
    self.warmup, self.div, self.floor = warmup, div, floor
    self.epoch, self.cycle, self.cycle_start = 0, 0, 0
    self.tapering, self.taper_decay = False, 0.7
    self.current = lr / div
    self._set(self.current)

  @property
  def lr(self):
    return self.opt.param_groups[0]["lr"]

  def _set(self, value):
    for g in self.opt.param_groups:
      g["lr"] = max(value, self.floor)

  def _rate(self):
    """Where this epoch sits in its own cycle: a short climb to the ceiling, then the decay shape
    down to the floor. Both are measured against `cycle_len`, never against the end of the run."""
    into = self.epoch - self.cycle_start  # 1 .. cycle_len
    warm = min(self.warmup, self.cycle_len // 2)  # never spend more than half a cycle climbing
    if into <= warm:
      return self.peak * (1 / self.div + (into / max(warm, 1)) * (1 - 1 / self.div))
    return decay_at(self.shape, (into - warm) / max(self.cycle_len - warm, 1), self.peak, self.floor)

  def state_dict(self):
    """Everything needed to carry on mid-cycle: how far down the ceiling has been stepped, which
    cycle is running and how long it is, and where in it we are."""
    return {k: v for k, v in self.__dict__.items() if k != "opt"}

  def load_state_dict(self, state):
    self.__dict__.update(state)
    self._set(self.current)

  def revive(self):
    """Put the ceiling back where it started and begin a fresh cycle. A run is resumed because
    there is more to try, and inheriting a ceiling the last one walked down guarantees a crawl."""
    self.peak, self.cycle_len = self.peak0, self.period
    self.cycle, self.cycle_start = 0, self.epoch

  def taper(self, over=None):
    """Stop restarting and run the rate down to the floor. One way, and the only thing that ends
    the schedule: until it is called the cycles deliberately continue, which is what lets a run be
    extended without picking an endpoint up front."""
    self.tapering = True
    if over:
      self.taper_decay = (self.floor / max(self.current, self.floor)) ** (1 / max(over, 1))

  def step(self, metric=None):
    """Once per epoch. A metric is accepted so the call site need not care which scheduler it
    holds, and ignored: it selects checkpoints, it no longer steers the rate."""
    self.epoch += 1
    if self.tapering:
      self.current *= self.taper_decay
      return self._set(self.current)

    if self.epoch - self.cycle_start > self.cycle_len:  # this cycle has run out, so open the next
      self.cycle, self.cycle_start = self.cycle + 1, self.epoch - 1
      self.cycle_len = max(1, round(self.cycle_len * self.t_mult))
      self.peak = max(self.peak * self.gamma, self.peak0 * self.ceiling_floor)

    self.current = self._rate()
    self._set(self.current)
