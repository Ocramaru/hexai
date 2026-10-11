# Working in this repo

Machine-wide standards and the `.agents` convention are installed globally and load themselves — see `~/.agents/AGENTS.md`. Only what is specific to hexai belongs here.

Run pytest with `env -u PYTHONPATH`, or ROS's site-packages shadows collection.

```sh
uv run ruff check --fix # lint
uv run ty check # fast type check
env -u PYTHONPATH uv run pytest tests/
```

Layering is `utilities` → `data` → `models`, one direction only. Where a new file goes: `utilities` if a second model would plausibly want it, `models` if it is specific to one.
