# Done

Finished tasks moved out of [[todo]] so it stays a list of what is open. Not loaded by
the .agents hook, which reads context.md, learned.md and todo.md only — findings worth carrying
belong in [[learned]], not here.

- [x] `tests/test_geometry.py`. Nine tests pinning the conventions that have each cost a bug: focal from height rather than width, index order, the trilinear partition of unity and what it does at the box edges, `grid_sample`'s reversed coordinates, and carve's three states. Checked against mutants — removing the coordinate flip and transposing the flat index both fail it.
- [x] Export the geometric model from `hexai/__init__.py`; it exported nothing at all, not just the frame-stack `VVAE`.
- [x] Notebooks no longer block the linter. `pyproject.toml` excludes `*.ipynb`, and the isort rules that caused most of the noise are no longer selected.
- [x] Swap the hard lift for the learned `C*D` channel reshape. `FrustumLift` widens to `in_channels * depth_bins` with 2D convs, reshapes to a frustum, and `warp_frustum` moves it into the canonical box; `--learned_lift 1`. The encoder's own `lift` stays geometric, so `lift_agreement` and the figures keep meaning something, and `lift_fidelity` measures how close the two get.
