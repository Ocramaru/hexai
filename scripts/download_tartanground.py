"""Download TartanGround environments.

The toolbox is heavy and pins its own dependencies, so it lives in a separate checkout with its
own interpreter rather than in this project. Point TARTANAIR_PYTHON at it:

    TARTANAIR_PYTHON=../tartanairpy/.venv/bin/python uv run python scripts/download_tartanground.py Downtown

With no environment named it lists what is available, and it skips anything already on disk.

Author: Marco Cassar (@Ocramaru)
"""

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARTAN = Path(os.environ.get("TARTAN_ROOT", ROOT / "data/tartanground"))
CAMERAS = ("lcam_front",)
MODALITIES = ("image", "depth", "meta")


def run_in_toolbox(body: str) -> int:
  python = os.environ.get("TARTANAIR_PYTHON")
  if not python or not Path(python).exists():
    sys.exit("set TARTANAIR_PYTHON to the interpreter of a checkout that has tartanair installed")
  return subprocess.call([python, "-c", textwrap.dedent(body)])


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("envs", nargs="*", help="environment names; omit to list what is available")
  p.add_argument("--version", default="omni")
  args = p.parse_args()

  TARTAN.mkdir(parents=True, exist_ok=True)
  have = sorted(d.name for d in TARTAN.iterdir() if d.is_dir() and not d.name.startswith("."))
  print(f"already at {TARTAN}: {have or 'nothing'}")

  if not args.envs:
    raise SystemExit(run_in_toolbox(f"""
      import tartanair as ta
      ta.init({str(TARTAN)!r})
      print("available:", ta.list_envs())
    """))

  wanted = [e for e in args.envs if e not in have]
  if not wanted:
    print("nothing to do, all requested environments are already on disk")
    return
  print(f"downloading {wanted}")
  raise SystemExit(run_in_toolbox(f"""
    import os
    os.environ["MPLBACKEND"] = "Agg"          # the toolbox draws on import
    import tartanair as ta
    ta.init({str(TARTAN)!r})
    ta.download_ground(env={wanted!r}, version=[{args.version!r}], traj=[],
                       modality={list(MODALITIES)!r}, camera_name={list(CAMERAS)!r}, unzip=True)
  """))


if __name__ == "__main__":
  main()
