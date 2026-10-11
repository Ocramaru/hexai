---
name: open
description: Open
kind: learned
---

# Open

- No genuinely good novel-view render yet — every result so far is about not failing rather than succeeding.

  ![pose sweep](../figures/pose-sweep.png)

  Truth over render across a yaw sweep. The render tracks the pan, so the latent is not viewpoint-collapsed, but it holds none of the structure.

- Whether the unknown-space work helps, tested *with* jitter, where masking reaches 42% of pixels rather than 23%.
- Jitter is capped at ±20° by reading a single cube face. The hexapod's actual pan/tilt range is not written down anywhere — measure it before deciding whether reading adjacent faces is worth it.
- Whether the learned lift earns its place. It has never been compared against the hard one on a schedule that works, and the run that introduced it also changed the grid and the sample count, so nothing about it is attributable yet.
