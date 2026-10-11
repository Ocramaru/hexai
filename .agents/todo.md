# TODO

<!-- map: generated. Edit the note, not this; after a manual edit run `agent remap` -->

| section | line |
|---|---|
| Next | 18 |
| Owed | 24 |
| Housekeeping | 28 |

<!-- /map -->


Repo-wide tasks. Per-package notes live in that package's own `.agents/`. Finished ones move to [[todo.archive]].

The plan for migrating the rest of the stack out of the fork lives in `../mjlab/.agents/todo.md`, where the work actually happens.

## Next

- [ ] More environments. 12 are on disk and Office is held out. The indoor ones still missing are `RetroOffice`, `CountryHouse`, `ArchVizTinyHouseDay`, `ArchVizTinyHouseNight`, `OldBrickHouseDay` and `AmericanDiner`; `scripts/download_tartanground.py --indoor` fetches only what is not already there.
- [ ] Extend the winning run rather than restarting it. `periodic` runs in cycles, so `--resume` continues at the next one and `--taper True` still ends it. Its ceiling only ever falls within a run, and `revive` on resume is what puts it back.
- [ ] Ablate the two bundles. `--jitter` flips fov, aiming and masking together; `--unknown` flips the observedness channel and the KL weighting together. We know each pair works, not which half earns its place, and that decides what ships to the Pi.

## Owed

- [ ] Three tests in `tests/test_hexapod_geometry.py` skip until the training scripts move: they cross-check `HEXAPOD_HEAD_ENVELOPE_MAX_M` and `BODY_GEOMS` against `train_hexapod*.py`.

## Housekeeping

- [ ] The dataset is lazy -- it indexes paths and reads images per sample -- but the index is 2.1 KB per window, mostly `Path` objects, 14 per window. That is 4.5 MB for one environment and ~330 MB for all 73, and every dataloader worker forks a copy. If that bites, store a trajectory id plus frame offset and rebuild paths on read.
- [ ] `mujoco.MjSpec` has no stub, so pyright reports it in `assets/hexapod`. Downgraded to a warning; mjlab solves it by generating stubs (`typings/generate_mujoco_stubs.sh`) if it ever becomes worth doing here.
- [ ] ROS puts its own Python on `PYTHONPATH`, which shadows the venv. `env -u PYTHONPATH` before pytest, or drop it from the shell profile.
- [ ] `data/` is a symlink to `../mjlab/data`. Fine locally; set `TARTAN_ROOT` in `.env` elsewhere.
