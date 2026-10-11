# hexai

<!-- map: generated. Edit the note, not this; after a manual edit run `agent remap` -->

| section | line |
|---|---|
| What constrains everything | 17 |
| Layout | 26 |

<!-- /map -->


Perception and learning for the Freenove Big Hexapod: a small twelve-servo walker with one camera on a pan/tilt head and a Raspberry Pi 4B for a brain.

Simulation, robot assets and math helpers come from [mjlab](../mjlab) as a published release, not an editable install of the fork — so anything hexai imports has to exist in the version pyproject pins, and what only exists in the fork has to come across rather than be imported. Everything here is the robot's own stack, and none of it needs to live inside a fork of someone else's repo.

## What constrains everything

| | |
|---|---|
| Pi 4B | ~5 GFLOP at 1 Hz for anything that ships. Conv2d hits XNNPACK on ARM, Conv3d does not |
| one camera | 49 degree fov, no stereo, no lidar. Depth is inferred, and only relatively |
| pan/tilt head | rotation is free and gives **zero** parallax; only body translation buys depth |
| SG90 servos | 3 degrees of pointing error, so commanded angles are not pose |

## Layout

`hexai/utilities` depends on nothing, `hexai/data` on utilities, `hexai/models` on both. Strictly one direction — nothing leans back up.

Where a new file goes: `utilities` if a second model would plausibly want it, `models` if it is specific to one. That test is why geometry, camera intrinsics and schedulers sit in `utilities`.

Importing `hexai` loads the repo's `.env`, and so does every script in `scripts/`; `.env.example` lists what it holds.

Open work, including what is still to migrate out of the fork, is in [[todo]].
