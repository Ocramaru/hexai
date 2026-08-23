# TODO

Open tasks for the VVAE. Cross off as they land. Findings go in [learned.md](learned.md), design rationale in [context.md](context.md).

## Still unwritten

- [ ] `tests/test_geometry.py`. Every convention it would pin — index order, pose direction, `grid_sample`'s reversed coordinates, splat weights staying non-negative at box edges — has already caused a bug. The checks only exist in scratch files.
- [ ] Export `VVAEGeometric` and friends from `hexai/__init__.py`; only the frame-stack `VVAE` is exported.

## Plan steps not started

- [ ] Swap the hard lift for the learned `C*D` channel reshape (step 5 in context.md). The lift is fixed geometry and 45% of encoder wall clock.
- [ ] Pose head: visual odometry from image pairs, servo angles as a rotation prior, for deployment.

## Known limitations worth a decision

- [ ] Density behind the first surface is invisible to a depth render, so it is unconstrained: measured 0.375 mean density in hidden samples against 0.051 in visible ones, 7.3x. If the `unknown` arm of the sweep does not fix it, penalise density where transmittance has already collapsed — the renderer has transmittance in hand.
- [ ] Head jitter is capped at ±20° because the loader reads a single cube face. The hexapod's real pan/tilt range is not written down anywhere; measure it before deciding whether reading adjacent faces is worth it.
- [ ] Notebooks account for every ruff error in the repo (~100, mostly `F811` and `I001` from normal cell-by-cell style), so `make check` cannot pass. Either exclude `*.ipynb` in pyproject or accept the noise.
