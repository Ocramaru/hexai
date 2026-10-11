---
name: measurement-traps-each-of-which
description: Measurement traps, each of which cost a run
kind: learned
---

# Measurement traps, each of which cost a run

- **A figure that redraws differently every time it is regenerated.** `figures.py` seeded numpy but not torch, and `log_novel_view` renders `out.x_hat`, which comes off the sampled latent rather than `mu` — so `novel-view.png` changed on every run while the other four came out byte-identical, because their panels all read `mu`. Regenerating the set to compare two checkpoints would have shown a difference that was only the draw. Both seeds are set now, and byte-identical output across two runs is the check that it holds.
- **Best-on-one-epoch picks the luckiest validation, not the best model.** A single `recon_answerable` swings ±31, about as far as a whole run improves, so both 200-epoch runs put `-best.pt` at epoch 11, roughly 0.13 skill above where the model actually settled. Selection runs on an EMA now, logged as `val/selection_score` so the chart shows what did the choosing.
- **A metric-driven schedule needs a window wider than its noise.** `AdaptiveRestarts` restarted after 3 stale epochs measured against an all-time best that only ratcheted down, so once a run stopped improving the condition was permanently true: it fired every 6 epochs for 200 and held 131 of them above 2.5e-3. Its verdict on a restart compared the best of 3 epochs against a single one, which noise alone wins 3 times in 4, so the ceiling climbed back as often as it decayed. A 3-epoch window cannot resolve anything smaller than ~0.21 skill. Replaced by `PeriodicRestarts`, which is open-loop.
- **The checkpoint on disk is the last epoch, not the best.** It is written unconditionally after every validation, so a run that regresses overwrites its own good state. Nothing from the epoch-61-to-80 window of the three-building run survived.
- ~~**Warmup scales with the run.**~~ and ~~**`horizon` comes back with the scheduler state.**~~ Both were `AdaptiveRestarts`, and both are gone with it: warmup is per cycle now, and a cycle is a fixed span, so there is no horizon to restore stale.

- **Hyperband kills the runs that win.** This model reads 0.645 at 6 epochs and 0.277 at 15. Never prune before the signal exists.
- **Bayes confounds what it sweeps together.** lr and beta moved along one diagonal for 18 runs, and the apparent "high lr is worse" was entirely beta.
- **Do not unshuffle validation.** At stride 1 consecutive windows share all but one frame, so scene-corr stopped measuring the decoder and started measuring the overlap.
- **Run names must be unique per run**, or a grid's cells overwrite each other's checkpoints — eight cells, one file. Listing the flags in the name does not achieve that: it holds only until a parameter is added and nobody updates the format string, which is how `depth_noise` nearly put a fresh run on a finished one's checkpoints. Runs take wandb's own name now, which is never reused, and `.runs/<name>.json` says what the run was. Offline runs have no name, only an id — that one is worth remembering, since it would put every offline run on `None.pt`.
- **Mask training and validation the same way**, or a masked run is scored on the 24% of pixels it was deliberately never taught — which carry 82% of its reported error.
- **Check the metric can see the hypothesis.** `lift_agreement` is measured only over observed voxels, so it was structurally blind to a change that only affects unobserved ones. `invented_density` was added for exactly this.
- Summary keeps the *last* logged value and these metrics swing ±0.3 between epochs. Log a running best.
