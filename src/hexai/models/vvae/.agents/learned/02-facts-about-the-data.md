---
name: facts-about-the-data
description: Facts about the data
kind: learned
---

# Facts about the data

- A window's query viewpoint varies by only **3.7°** without jitter. "Render the held-out view" is nearly "render the view you were given", and a model emitting one canonical frustum is right most of the time. Now priced: copying the given view scores 42.5 there, against a mean-image 172.9.
- **The answerable fraction is ~50% without jitter and ~47% with**, measured as the mean of the soft mask over held-out Office. The 77% and 58% recorded below came from a different count and do not agree; quote which measure with the number.
- **Most of the box is unobserved, not empty** — a single view reports first surfaces, and what is behind them is occluded rather than free. Carving leaves ~65% unknown; the lift, which marks only surfaces, leaves ~88%. Scene-dependent, so quote the measure with the number.

  ![three-state carving](../figures/three-state-carving.png)

  Horizontal slices, camera at bottom centre. Dark unknown, blue free, yellow occupied. Each slice shows only what exists at that height, which is why a staircase looks different in every one — to relate a slice back to an image, unproject one image row onto it ([example](../figures/slice-correspondence.png)).

- Consecutive frames sit **0.21 m** apart (measured off the pose file, not window drift). Stride 1 overlaps windows and multiplies them by roughly `num_frames` — legitimately, since every window is anchored on its own middle frame, so the canonical box moves with it.
