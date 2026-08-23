# hexai

Perception and learning for the Freenove Big Hexapod: a small twelve-servo walker with one camera
on a pan/tilt head and a Raspberry Pi 4B for a brain.

Simulation, robot assets and math helpers come from [mjlab](../mjlab), installed editable.
Everything here is the robot's own stack, and none of it needs to live inside a fork of someone
else's repo.

## What constrains everything

| | |
|---|---|
| Pi 4B | ~5 GFLOP at 1 Hz for anything that ships. Conv2d hits XNNPACK on ARM, Conv3d does not |
| one camera | 49 degree fov, no stereo, no lidar. Depth is inferred, and only relatively |
| pan/tilt head | rotation is free and gives **zero** parallax; only body translation buys depth |
| SG90 servos | 3 degrees of pointing error, so commanded angles are not pose |

## Layout

`hexai/utilities` depends on nothing, `hexai/data` on utilities, `hexai/models` on both. Strictly
one direction — nothing leans back up.

Open work, including what is still to migrate out of the fork, is in [todo.md](todo.md).
