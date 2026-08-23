"""Learning-rate schedulers. Model agnostic, and built on the shapes in decay.py.

Author: Marco Cassar (@Ocramaru)
"""

from hexai.models.utilities.decay import decay_at


class AdaptiveRestarts:
  """Anneal while the metric improves, push the rate back up when it stalls, lower the ceiling when
  a push does not pay off. Four things keep it from thrashing: an average on the metric so a noisy
  epoch cannot trigger a restart, an average on the rate so a push ramps rather than snaps, a
  threshold so a trivial gain does not count as progress, and a probe window so a push is judged on
  its best few epochs rather than its first.

  Unlike a fixed schedule it never finishes on its own, holding rate in reserve so a run can be
  extended. Call `taper()` to anneal out. Stepped on a validation metric, once per epoch, and
  assumes lower is better.
  """

  def __init__(self, optimizer, lr, horizon, shape="cosine", patience=2, ceiling_decay=0.6,
               smoothing=0.35, metric_ema=0.5, threshold=0.005, probe=3, warmup=3, div=25.0,
               floor=1e-6):
    self.opt, self.peak, self.horizon, self.shape = optimizer, lr, horizon, shape
    self.patience, self.ceiling_decay, self.floor = patience, ceiling_decay, floor
    self.smoothing, self.metric_ema = smoothing, metric_ema
    self.threshold, self.probe = threshold, probe
    self.warmup, self.div, self.epoch, self.segment_start = warmup, div, 0, 0
    self.tapering, self.taper_decay = False, 0.7
    self.target = self.current = lr / div
    self.best, self.stale, self.smoothed = float("inf"), 0, None
    self.probe_left, self.probe_best, self.pre_restart = 0, float("inf"), float("inf")
    self._set(self.current)

  @property
  def lr(self):
    return self.opt.param_groups[0]["lr"]

  def _set(self, value):
    for g in self.opt.param_groups:
      g["lr"] = max(value, self.floor)

  def _anneal(self):
    """Where this epoch sits in the current segment, a segment running from the last restart to the
    end of training. Later restarts therefore anneal faster, having less run left to spend."""
    span = max(self.horizon - self.segment_start, 1)
    return decay_at(self.shape, (self.epoch - self.segment_start) / span, self.peak, self.floor)

  def state_dict(self):
    """Everything needed to carry on mid-slope: where the ceiling has been driven to, how stale the
    metric is, both averages, and how far into warmup or a probe it got."""
    return {k: v for k, v in self.__dict__.items() if k != "opt"}

  def load_state_dict(self, state):
    self.__dict__.update(state)
    self._set(self.current)

  def taper(self, over=None):
    """Stop pushing the rate back up and run it down to the floor. Until this is called the
    schedule deliberately never finishes, so a run can be extended."""
    self.tapering = True
    self.target = self.current  # they lag each other, so run down from where we actually are
    if over:
      self.taper_decay = (self.floor / max(self.current, self.floor)) ** (1 / max(over, 1))

  def step(self, metric):
    self.epoch += 1
    self.smoothed = metric if self.smoothed is None else (
      (1 - self.metric_ema) * self.smoothed + self.metric_ema * metric)
    m = self.smoothed

    if self.tapering:
      self.target *= self.taper_decay
      self.current += self.smoothing * (self.target - self.current)
      return self._set(self.current)

    if self.epoch <= self.warmup:
      # linear climb to the peak; the adaptive logic only starts after
      frac = self.epoch / self.warmup
      self.best = min(self.best, m)
      self.target = self.current = self.peak * (1 / self.div + frac * (1 - 1 / self.div))
      self.segment_start = self.epoch  # the anneal starts where warmup ends
      return self._set(self.current)

    if self.probe_left:
      # judge a push on its best few epochs, not its first. Raising the rate costs before it pays,
      # so a one-epoch verdict condemns every restart and walks the ceiling down to nothing
      self.probe_best = min(self.probe_best, m)
      self.probe_left -= 1
      if not self.probe_left and self.probe_best >= self.pre_restart:
        self.peak *= self.ceiling_decay
      self.target = self._anneal()
      self.current += self.smoothing * (self.target - self.current)
      return self._set(self.current)

    if m < self.best * (1 - self.threshold):  # a gain has to be worth something to count
      self.best, self.stale = m, 0
    else:
      self.best, self.stale = min(self.best, m), self.stale + 1

    if self.stale > self.patience:
      self.stale, self.pre_restart = 0, m
      self.probe_left, self.probe_best = self.probe, float("inf")
      self.segment_start, self.target = self.epoch, self.peak
    else:
      self.target = self._anneal()

    self.current += self.smoothing * (self.target - self.current)  # ease, do not snap
    self._set(self.current)
