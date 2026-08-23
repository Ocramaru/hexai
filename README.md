# hexai

Perception and learning for the Freenove Big Hexapod. Simulation, assets and math helpers come from
[mjlab](../mjlab), installed editable; everything here is the robot's own stack.

```bash
uv sync                                    # installs hexai and mjlab editable
uv run python -m hexai.models.vvae.train --help
```

Set `TARTAN_ROOT` if the TartanGround data is not at `./data/tartanground` (it symlinks to mjlab's by default).

## Layout

| | |
|---|---|
| `src/hexai/models/vvae/` | the geometric VVAE: posed frames in, a 3D scene latent out. See its [README](src/hexai/models/vvae/README.md) |
| `src/hexai/models/utilities/` | model-agnostic pieces: decay shapes, schedulers |

## Agent notes

Any directory may hold `.agents/` with `context.md` (why the design is what it is), `learned.md`
(what experiments showed, and traps that already cost a run) and `todo.md` (open tasks).

A `PreToolUse` hook feeds these in when Claude edits a file, walking that file's directory up to the
repo root so nested notes compose. Each is sent once per session; a `PostCompact` hook clears that
memo so they survive a compaction. Run `/agents <path>` to load them on demand.

Author: Marco Cassar (@Ocramaru)
