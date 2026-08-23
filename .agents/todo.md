# TODO

Repo-wide tasks. Per-package notes live in that package's own `.agents/`.

## Migration from the mjlab fork

~49k lines of the hexapod stack are still untracked inside `../mjlab`, a fork of someone else's
repo. The VVAE came across first; the rest follows one module at a time, refined before it moves.

**The rule while split:** `hexai` has mjlab installed editable, so hexai may import mjlab. The
reverse cannot work without making them circular, so a module only moves once nothing left in
mjlab imports it.

### Do early

- [ ] Empty or delete `mjlab/src/mjlab/models/__init__.py`. It imports twelve of the modules below
      plus perception and rl, so it breaks the moment any one of them moves.
- [ ] Delete `mjlab/src/mjlab/models/vvae/` and `models/utilities/` once the running sweep finishes
      — its `sweep.json` and `.runs/` still point there.

### Leaves, nothing else of ours depends on them

Movable in any order, once refined.

- [ ] `models/adapters` · `models/aol` · `models/motor_head` · `models/osd` · `models/probes` · `models/rfnl` · `models/task_handler`
- [ ] `perception/floorplan` · `perception/hardware_realism`
- [ ] `rl/config` · `rl/spatial_softmax`

### The perception spine, in this order

`visual_mapping` is the keystone: seven modules depend on it and it depends on nothing, so moving
it early unblocks the rest.

- [ ] `perception/visual_mapping`
- [ ] `perception/splat`
- [ ] `perception/evidence_map_batch`, then `perception/evidence_map`
- [ ] `perception/candidates`
- [ ] `perception/viz`, `models/splat_refiner`, `models/selector`
- [ ] `models/deploy`, `models/hub`
- [ ] `perception/explorer_runtime`

### Tied to mjlab's simulation, decide rather than sequence

These import `mjlab.envs` / `tasks` / `sim` directly. Whether they move at all is a real question:
they may belong here importing mjlab, or may be better left as a thin layer in the fork.

- [ ] `rl/stage2_env` · `rl/hub_obs_wrapper` · `rl/vecenv_wrapper` · `rl/exporter_utils`
- [ ] `tasks/velocity/mdp/explore_command.py` and the modified `rewards.py` — mjlab may discover
      these by import path, so they might have to stay or be registered from outside.

### Then

- [ ] The 25 scripts and 21 tests, which follow whatever they import.
- [ ] Decide what the fork keeps: the hexapod asset and the cartpole tutorial are genuinely
      mjlab-shaped and are plausible upstream PRs.

## Housekeeping

- [ ] `data/` is a symlink to `../mjlab/data`. Fine locally; set `TARTAN_ROOT` elsewhere.
- [ ] No tests in this repo yet. `tests/test_geometry.py` is owed from the VVAE work — see
      `src/hexai/models/vvae/.agents/todo.md`.
