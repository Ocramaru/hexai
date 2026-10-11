# hexai

Perception and learning for the **Freenove Big Hexapod** — a twelve-servo walker with one camera on
a pan/tilt head and a Raspberry Pi 4B for a brain.

The hard part: a camera reports surfaces, one flat image at a time, while a robot needs to know
where things *are* — including that it does not know what is behind the wall it can see.

## Install

Requires [mjlab](https://github.com/mujocolab/mjlab) for the simulator and its math helpers. It
comes from PyPI, so nothing needs checking out alongside.

```bash
uv sync
uv run python -m hexai.models.vvae.run --help
```

Training data is [TartanGround](https://tartanair.org/). Point `TARTAN_ROOT` at it, or place it at
`./data/tartanground`. The toolbox that fetches it pins its own dependencies, so it lives in a
separate checkout:

```bash
TARTANAIR_PYTHON=../tartanairpy/.venv/bin/python uv run python scripts/download_tartanground.py Office
```

Training spans every environment under that root and holds one out for validation, so a run is
scored on a building it has never seen.

## Tracking

Runs log to [wandb](https://wandb.ai). Set `WANDB_MODE=offline` to keep them local, or
`WANDB_MODE=disabled` to skip it.

Every run carries a glossary of its metrics in the run notes, since wandb has no per-metric
description. Each also logs a few fixed scenes as images — truth against render across a yaw pan,
and the volume top-down — chosen by index so they are the same windows in every run and comparable
across a sweep.

Sweeps are defined beside the model they train, for example `src/hexai/models/vvae/sweep.json`:

```bash
uv run wandb sweep src/hexai/models/vvae/sweep.json
wandb agent <entity>/<project>/<sweep_id>
```

Checkpoints land in a gitignored `.runs/`, named after the run — wandb's own `fanciful-sunset-12`
rather than a string of the flags, which only stays unique until someone adds a parameter and
forgets to list it. What the run *was* is written beside it as `<name>.json`, so
`grep -l '"occlusion": 1000' .runs/*.json` finds it. `--name` sets one by hand.

Checkpoints carry optimizer and scheduler state, so `--resume` continues a schedule mid-slope
rather than restarting it, and a continuation keeps the name of the run it continues so its
snapshots stay together.

## Layout

| | |
|---|---|
| `hexai/assets/hexapod/` | the robot: constants measured from CAD and calibrated on hardware, and the MJCF mjlab simulates |
| `hexai/models/vvae/` | posed camera frames in, a 3D scene latent out — see its [README](src/hexai/models/vvae/README.md) |
| `hexai/data/` | datasets and loaders |
| `hexai/utilities/` | geometry, camera intrinsics, decay shapes, schedulers |

Dependencies run one way: `utilities` needs nothing, `data` needs `utilities`, `models` needs both.

## Agent notes

Any directory may hold `.agents/` with three files by convention:

| | |
|---|---|
| `context.md` | why the design is what it is, and what was ruled out |
| `learned.md` | what experiments showed, and traps that already cost a run |
| `todo.md` | open tasks |

A `PreToolUse` hook feeds these to Claude when it edits a file, walking that file's directory up to
the repo root so nested notes compose, outermost first. Each is sent once per session; a
`PostCompact` hook clears that memo so they survive a compaction. `/agents <path>` loads them on
demand. Hooks live in `.claude/hooks/`.

## Status

Early. The VVAE trains and its geometry is verified, but it has yet to produce a good novel view —
see [its notes](src/hexai/models/vvae/.agents/learned.md). Most of the stack still lives in a fork
of mjlab and is [migrating across](.agents/todo.md) a module at a time.

Progress is written up in [devlog/](devlog/), newest first.

## License

Apache-2.0. Marco Cassar — [@Ocramaru](https://github.com/Ocramaru), <dev@marcocassar.info>.
