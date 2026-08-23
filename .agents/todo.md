# TODO

Repo-wide tasks. Per-package notes live in that package's own `.agents/`.

The plan for migrating the rest of the stack out of the fork lives in `../mjlab/.agents/todo.md`,
where the work actually happens.

## Owed

- [ ] Three tests in `tests/test_hexapod_geometry.py` skip until the training scripts move: they
      cross-check `HEXAPOD_HEAD_ENVELOPE_MAX_M` and `BODY_GEOMS` against `train_hexapod*.py`.

## Housekeeping


- [ ] `data/` is a symlink to `../mjlab/data`. Fine locally; set `TARTAN_ROOT` elsewhere.
- [ ] No tests in this repo yet. `tests/test_geometry.py` is owed from the VVAE work — see
      `src/hexai/models/vvae/.agents/todo.md`.
