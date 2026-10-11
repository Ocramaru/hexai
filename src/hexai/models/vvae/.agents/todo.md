# TODO

<!-- map: generated. Edit the note, not this; after a manual edit run `agent remap` -->

| section | line |
|---|---|
| Next, in order | 15 |
| Known limitations worth a decision | 25 |

<!-- /map -->


Open tasks for the VVAE. Finished ones move to [[todo.archive]], which the hook does not load. Findings go in [[learned]], design rationale in [[context]].

## Next, in order

Each of these is one run against the `--occlusion 0` leg already on disk, unless it says otherwise.

- [ ] **Point density.** The grid is under-sampled and it confounds everything else: 7 frames of 64x64 into 32x60x32 gives ~12 points per occupied voxel on context.md's 4% basis, under the ~16 where surfaces perforate, and 3.9 on the measured occupancy. The grid went 32,24,32 to 32,60,32 and `image_size` never moved. Split `image_size` into an input size and a render size, feed 128x128 and keep rendering 64x64: 4x the points for almost no extra ray marching. Re-baseline after, since every number to date is measured under-sampled.
- [ ] **Latent capacity.** `latent_channels` 4 against 8 and 16. 3840 numbers for 61,440 voxels has never been tested as the binding constraint, and it is nearly free to move -- it is the last 1x1 conv, so the Pi budget barely notices.
- [ ] **The loss.** mse's optimum for an uncertain ray is the mean of the modes, which is why the decoder emits a smooth field. Cheapest first: l1 or huber, whose optimum is the median and so commits to a mode; then the residual in inverse depth, which matches what a monocular estimate is defined up to and stops far surfaces dominating; then supervising the ray's weight distribution directly, which `render_view` already computes and throws away.
- [ ] **The learned lift.** `--learned_lift False` on an otherwise identical run. Known 35% faster, quality never measured.
- [ ] Pose head: visual odometry from image pairs, servo angles as a rotation prior, for deployment.

## Known limitations worth a decision

- [ ] Head jitter is capped at ±20° because a 90° cube face rendered out at the robot's 49° fov leaves only (90-49)/2 of yaw before the window runs off the source. hexapod.xml has the real range from CAD: pan ±90, tilt -40/+90, so the cap is 22% of what the head does. House and Office now have all five faces; widening means picking the face per target angle and handling the seams.
- [ ] The occlusion penalty is built, gated and measured, and at weight 1000 it empties the volume — see [[learned]]. Decide between a low-weight sweep (50 and 150, against the `--occlusion 0` leg already on disk) and dropping the lever. The response is steep, so the useful band is narrow: the target is `decoded_solid` falling from 0.167 toward the lifted 0.119, not past it.
- [ ] `PeriodicRestarts` has no run behind it. It is verified open-loop and its cycles trace correctly, but nothing has trained on it, so the `baseline` preset in `config.py` points at an untested schedule until one has.
- [ ] Unproven flags accumulate and multiply the config space: `--learned_lift` and `--occlusion` are both optional because neither is known good, not because either is a real choice. Each one should end as a default or a deletion once its run lands. `--jitter` and `--unknown` are the older pair with the same problem, and they bundle two changes each.
