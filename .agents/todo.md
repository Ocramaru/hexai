# TODO

Repo-wide tasks. Per-package notes live in that package's own `.agents/`.

The plan for migrating the rest of the stack out of the fork lives in `../mjlab/.agents/todo.md`,
where the work actually happens.

## Next

- [ ] More environments. 73 are available, one is downloaded, and the model has only ever seen
      one building. Indoor first, since the robot is indoor: `Office`, `Hospital`, `Restaurant`,
      `Supermarket`, `RetroOffice`, `CountryHouse`, `ArchVizTinyHouseDay`. House stays the held-out
      one so today's numbers remain comparable.
- [ ] Extend the winning run rather than restarting it. It is on `adaptive`, which holds rate in
      reserve, so `--resume` continues the schedule mid-slope and `--taper 1` ends it.
- [ ] Ablate the two bundles. `--jitter` flips fov, aiming and masking together; `--unknown` flips
      the observedness channel and the KL weighting together. We know each pair works, not which
      half earns its place, and that decides what ships to the Pi.

## Owed

- [ ] Three tests in `tests/test_hexapod_geometry.py` skip until the training scripts move: they
      cross-check `HEXAPOD_HEAD_ENVELOPE_MAX_M` and `BODY_GEOMS` against `train_hexapod*.py`.

## Housekeeping


- [ ] `data/` is a symlink to `../mjlab/data`. Fine locally; set `TARTAN_ROOT` elsewhere.
- [ ] No tests in this repo yet. `tests/test_geometry.py` is owed from the VVAE work — see
      `src/hexai/models/vvae/.agents/todo.md`.
