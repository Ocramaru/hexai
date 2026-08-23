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
uv run python -m hexai.models.vvae.train --help
```

Training data is [TartanGround](https://tartanair.org/). Point `TARTAN_ROOT` at it, or place it at
`./data/tartanground`.

## Layout

| | |
|---|---|
| `hexai/assets/hexapod/` | the robot: constants measured from CAD and calibrated on hardware, and the MJCF mjlab simulates |
| `hexai/models/vvae/` | posed camera frames in, a 3D scene latent out — see its [README](src/hexai/models/vvae/README.md) |
| `hexai/data/` | datasets and loaders |
| `hexai/utilities/` | tied to no model: geometry, camera intrinsics, decay shapes, schedulers |

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

## License

Apache-2.0. Author: Marco Cassar ([@Ocramaru](https://github.com/Ocramaru)).
