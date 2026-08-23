# What we learned

Running log for the geometric VVAE — what experiments actually showed. The design rationale is in [context.md](context.md), open tasks in [todo.md](todo.md). Newest at the top, entries a line or two.

## Settled

- **depth beats rgb** as the render target — 0.386 mean scene-corr against 0.454. Depth needs no appearance channel at all, so the volume drops from 4 channels to 1 and the latent stops paying to remember colour.
- **mse beats an edge-aware loss** (0.277 against 0.388) even on depth, where the gradient term should have helped. The edge loss has been removed.
- **onecycle beats the alternatives** — it led on skill in all four of its runs against `adaptive` and against a fixed cosine-with-restarts schedule since deleted. Best rate ~3e-3, well above the 1.3e-3 we assumed for a while.
- **Learning rate has a narrow window.** 2e-4 collapses, 1e-3 works, 3e-3 collapses as a constant — but survives as a onecycle *peak*, because warmup is what stops the early blowup.
- **A single global beta collapses the latent.** 0.05 takes scene-corr from 0.277 to 0.750 at the same rate (same reference run as the loss comparison). Below ~1e-3 is safe.
- **Jitter always implies masking.** Aiming the head lifts query viewpoint spread from 3.7° to 21.4°, but drops the answerable fraction from 77% to 58% — without the mask, two fifths of the loss would score pixels no context frame saw.
- **The robot's fov is 49°, not a hyperparameter.** `pixel_to_camera_ray` derives focal length from it, so a wrong fovy puts every unprojected point at the wrong angle. Seven frames restores the coverage a 90° camera got from four.

## Facts about the data

- A window's query viewpoint varies by only **3.7°** without jitter. "Render the held-out view" is nearly "render the view you were given", and a model emitting one canonical frustum is right most of the time. This is still the largest unexplained gap.
- **Most of the box is unobserved, not empty** — a single view reports first surfaces, and what is behind them is occluded rather than free. Carving leaves ~65% unknown; the lift, which marks only surfaces, leaves ~88%. Scene-dependent, so quote the measure with the number.

  ![three-state carving](../figures/three-state-carving.png)

  Horizontal slices, camera at bottom centre. Dark unknown, blue free, yellow occupied. Each slice shows only what exists at that height, which is why a staircase looks different in every one — to relate a slice back to an image, unproject one image row onto it ([example](../figures/slice-correspondence.png)).

- Consecutive frames sit **0.21 m** apart (measured off the pose file, not window drift). Stride 1 overlaps windows and multiplies them by roughly `num_frames` — legitimately, since every window is anchored on its own middle frame, so the canonical box moves with it.

## Diagnoses that held up

- **The collapse was optimization, not architecture.** A single batch overfits to recon 5.1 with distinct volumes, so the renderer, density parameterisation and capacity are all fine.
- **It was never overfitting.** Train and val failed identically; more data would not have helped a model that could not fit what it had.

## Measurement traps, each of which cost a run

- **Hyperband kills the runs that win.** This model reads 0.645 at 6 epochs and 0.277 at 15. Never prune before the signal exists.
- **Bayes confounds what it sweeps together.** lr and beta moved along one diagonal for 18 runs, and the apparent "high lr is worse" was entirely beta.
- **Do not unshuffle validation.** At stride 1 consecutive windows share all but one frame, so scene-corr stopped measuring the decoder and started measuring the overlap.
- **Run names must carry what varies**, or a grid's cells overwrite each other's checkpoints. Eight cells, one file.
- **Mask training and validation the same way**, or a masked run is scored on the 24% of pixels it was deliberately never taught — which carry 82% of its reported error.
- **Check the metric can see the hypothesis.** `lift_agreement` is measured only over observed voxels, so it was structurally blind to a change that only affects unobserved ones. `invented_density` was added for exactly this.
- Summary keeps the *last* logged value and these metrics swing ±0.3 between epochs. Log a running best.

## Open

- No genuinely good novel-view render yet — every result so far is about not failing rather than succeeding.

  ![pose sweep](../figures/pose-sweep.png)

  Truth over render across a yaw sweep. The render tracks the pan, so the latent is not viewpoint-collapsed, but it holds none of the structure.

- Whether the unknown-space work helps, tested *with* jitter, where masking reaches 42% of pixels rather than 23%.
- Jitter is capped at ±20° by reading a single cube face. The hexapod's actual pan/tilt range is not written down anywhere — measure it before deciding whether reading adjacent faces is worth it.
- `tests/test_geometry.py` still does not exist. Every convention it would pin has already caused a bug.
