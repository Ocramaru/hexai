# VVAE — 3D latent plan

Why the architecture is what it is. What the experiments actually showed is in [learned.md](learned.md), open tasks in [todo.md](todo.md).

The latent is a viewpoint-independent belief about the space around the robot, not a compressed view. Only the encoder ships to the Pi; the decoder is training scaffolding.

## Constraints

| | |
|---|---|
| Pi encoder budget | **~5 GFLOP** @ 1 Hz (3–7 range) |
| Conv2d vs Conv3d | Conv2d hits XNNPACK on ARM, Conv3d does not |
| parallax @2 m | 5 cm baseline → 4.4 px; 20 cm → 17.8 px |
| SG90 pointing error | 3° = **9.9 px** — swamps parallax below ~20 cm baseline |
| rotation | head pan/tilt gives **zero** parallax; only body translation buys depth |
| relative-depth drift | ±10% per-frame scale smears a surface over 1.72x the voxels, so depth cannot be a hard index |
| depth consistency vs accuracy | ±5% per-frame scale wobble costs as much as being globally wrong by 25% (1.89x vs 1.94x smear) |
| global depth scale | recoverable: grid occupancy is minimised at the true scale, so multi-frame agreement resolves what a monocular model cannot know |
| grid resolution | bounded by point density, not by what looks precise. 5 frames of 160x160 give ~131 points per occupied voxel at 32x24x32 but only ~16 at 96x72x96, and surfaces perforate. Resolution and image size move together |
| ray samples | must scale with voxel size, four per smallest voxel. Too few and oblique rays slip between voxels, which bands flat surfaces and worsens as resolution rises |

Frames must be ~20 cm apart in travel. Commanded servo angles cannot be trusted as pose.

## Data

`pose_<camera>.txt` from the `meta` modality: one line per frame, `tx ty tz qx qy qz qw`, unit quaternions, count matching the images exactly. Sim pose is also the supervision for a deployment pose head (commanded → refined).

Trajectories are continuous 10 Hz walks at ~2 m/s, so consecutive frames sit **0.21 m** apart with no stride needed. The source is 90° FOV at 640x640, re-rendered to the robot's 49° when the head is aimed. Pose convention is `R_cam = R_ned @ NED2CAM.T`; warping under it cuts photometric error 3.5x at six frames apart, while the raw-quaternion reading is worse than not warping — the data source is validated for this.

Anchoring a window on its middle frame rather than its first keeps 86.8% of points inside the canonical box against 84.6%, and cuts the query camera's z spread from 0.214 m to 0.085 m — the robot turns as it walks, so a first-frame anchor lets the box hang off one end of the window.

Without head jitter a window's query viewpoint varies by only 3.7°, so "render the held-out view" is nearly "render the view you were given". Aiming the head lifts that to 21.4° and is free, the robot's servos already do it — but it drops the answerable fraction from 77% to 58%, so it always comes with the loss mask.

On the robot the same 0.21 m baseline takes ~2 s, so frames must be spaced by distance travelled, not time. Sim camera height is 1.75 m against the hexapod's ~0.2 m, a viewpoint gap the encoder has to cross.

## Structure

Encoder, per frame independently. What is built today:

    unproject by depth + pose -> splat into the canonical grid    exact geometry, not learned
    sum over frames                                               order-invariant
    enc3d -> latent

Step 5 replaces that first line with a learned lift, keeping the rest:

    enc2d(rgb, depth) -> C*D channels -> reshape [C, D, H, W]     learned, 2D convs only

Decoder, query-pose conditioned, rendering a **held-out** view:

    V_hat = dec3d(z);  image = render(V_hat, query_pose)

The render target is depth, not rgb — pure geometry needs no appearance channel, so the volume is density alone and the latent stops paying to remember colour.

Three choices carry the design:

**The frustum will be a channel reshape**, not a convolved depth axis — free, and it keeps all per-pixel work in Conv2d. Not yet built; the current lift is fixed geometry, and by wall clock it is 45% of the encoder.

**Depth is spent on position, not carried as a value.** A voxel's z index is its depth, so scattering a depth channel would store distance-from-whichever-camera-saw-it inside a representation meant to be viewpoint independent. Uncertain depth belongs as mass spread across neighbouring voxels, which the soft lift expresses and a scalar cannot. What the scatter sums is appearance, and the hit count is the confidence.

**Pose is a coordinate transform, not a conditioning vector.** A conv writes to a fixed position, while lifting and warping both need the destination to depend on the content (depth). Convolution cannot express that routing.

**The decoder renders a view that was never encoded.** Reconstructing the encoded frames is maximally satisfiable by a viewpoint-dependent latent; a held-out query pose is the only thing that forces a scene. Split each window into context and target, resampled per batch, poses relative to the window's middle frame. The target is never that middle frame: its query pose is identity, so the render never looks at the volume from an angle, geometry that is wrong off-axis costs nothing, and the pose input can go unused. Every other frame is a real viewpoint.

Per-frame-then-sum rather than convolving the stack: the same room observed in a different order is the same room, and it admits any frame count.

Ruled out: plane-sweep cost volume (needs pose accuracy the 3° error denies); pose as FiLM conditioning (cannot express the warp).

## Build order

1. ~~Loader returns per-frame pose, relative to the window's middle frame.~~ Done, with `add_pose_noise` for the training loop to apply so validation still sees truth.
2. ~~`resample` + accumulate, measured depth and a hard lift.~~ Done, splatted rather than hard-binned, verified by eye against a top-down floor plan.
3. ~~`render` + novel-view loss, masked to what a context frame actually saw.~~ Done.
4. ~~Three-state carving: free, occupied, unknown, so the model can hold "I never saw this" instead of calling it empty or solid.~~ Done, `predict_unknown` and `kl_mode observed`.
5. Swap the hard lift for the learned `C*D` channel reshape.
6. Pose head (VO from image pairs, servo angles as a rotation prior) for deployment.

Steps 2–3 are verifiable before any training, and a bug in the transform is invisible once a network is learning around it.

Canonical box is x [-2, 2], y [-1.5, 1.5], z [0, 4] metres in the middle frame's camera, at 32 x 24 x 32 for 12.5 cm cubic voxels. Fitting the box to the camera cone's far plane is worse than keeping it tight, since a cuboid matching a cone at 4 m is mostly empty at 1 m.

## Open questions

- `D`, and whether bins are linear or inverse-depth.
- Encoder cost is dominated by the convs, not the lift: `base_channels` 32 -> 16 with `groups=2` takes it from 9.7 to **1.3 GFLOP**, inside the budget, for 0.30M params. FLOPs undercount the lift though — by wall clock on four cores it is 45% of the encoder, and being fixed geometry it shrinks only with fewer or smaller frames.
- A single view reports first surfaces, so most of the box is occluded rather than empty. The lift cannot say which, and that is what the carving adds.
