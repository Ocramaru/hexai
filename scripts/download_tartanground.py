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

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # before TARTAN_ROOT and TARTANAIR_PYTHON are read, here and below
TARTAN = Path(os.environ.get("TARTAN_ROOT", ROOT / "data/tartanground"))
CAMERAS = ("lcam_front",)
MODALITIES = ("image", "depth", "meta")

# rooms and corridors at something like a house's scale, which is what the robot will meet; the rest
# of the catalogue is outdoors, fantasy or city-sized, real scenes but not this robot's
INDOOR = (
  "House", "Office", "Hospital", "Restaurant", "Supermarket", "RetroOffice", "CountryHouse",
  "ArchVizTinyHouseDay", "ArchVizTinyHouseNight", "OldBrickHouseDay", "OldBrickHouseNight",
  "AmericanDiner", "HQWesternSaloon", "Prison", "AbandonedSchool", "CarWelding",
  "IndustrialHangar", "Sewerage",
)


def run_in_toolbox(body: str, capture: bool = False) -> subprocess.CompletedProcess:
  """Runs `body` under the tartanairpy checkout's interpreter, headless, since the toolbox draws on
  import. A captured run raises on failure, because its output is about to be parsed."""
  python = os.environ.get("TARTANAIR_PYTHON")
  if not python or not Path(python).exists():
    sys.exit("set TARTANAIR_PYTHON to the interpreter of a checkout that has tartanair installed")
  return subprocess.run([python, "-c", textwrap.dedent(body)], env={**os.environ, "MPLBACKEND": "Agg"},
                        capture_output=capture, text=True, check=capture)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("envs", nargs="*", help="environment names; omit to list what is available")
  parser.add_argument("--indoor", action="store_true", help=f"adds the {len(INDOOR)} indoor environments")
  parser.add_argument("--all", action="store_true", help="everything in the catalogue, ~400 GB")
  parser.add_argument("--version", default="omni")
  parser.add_argument("--cameras", default=",".join(CAMERAS),
                      help="comma separated cube faces; the side and top faces are what a pan/tilt "
                           "head wider than +/-20 deg needs, since one face is only 90 deg across")
  args = parser.parse_args()
  cameras = tuple(args.cameras.split(","))

  TARTAN.mkdir(parents=True, exist_ok=True)
  # per camera, not per environment: asking for a new cube face of an environment already on disk
  # is the normal case once the head range widens, and a name-only check would skip all of them
  have = sorted(directory.name for directory in TARTAN.iterdir()
                if directory.is_dir() and not directory.name.startswith(".")
                if all(any(directory.glob(f"*/*/image_{camera}")) for camera in cameras))
  print(f"already at {TARTAN} with {list(cameras)}: {have or 'nothing'}")

  # added to whatever was named, not instead of it, and in order without repeats
  if args.indoor: args.envs = list(dict.fromkeys([*args.envs, *INDOOR]))
  if args.all:
    listing = run_in_toolbox(f"""
      import tartanair as ta
      ta.init({str(TARTAN)!r})
      print(",".join(ta.list_envs()["remote"]))
    """, capture=True)
    args.envs = list(dict.fromkeys([*args.envs, *listing.stdout.strip().splitlines()[-1].split(",")]))

  if not args.envs:
    raise SystemExit(run_in_toolbox(f"""
      import tartanair as ta
      ta.init({str(TARTAN)!r})
      print("available:", ta.list_envs())
    """).returncode)

  wanted = [environment for environment in args.envs if environment not in have]
  if not wanted:
    print("nothing to do, all requested environments are already on disk")
    return
  print(f"downloading {len(wanted)} x {len(cameras)} faces: {wanted}")
  print(f"roughly {5.4 * len(wanted) * len(cameras):.0f} GB at House's size, and environments vary")
  raise SystemExit(run_in_toolbox(f"""
    import tartanair as ta
    ta.init({str(TARTAN)!r})
    ta.download_ground(env={wanted!r}, version=[{args.version!r}], traj=[],
                       modality={list(MODALITIES)!r}, camera_name={list(cameras)!r}, unzip=True)
  """).returncode)


if __name__ == "__main__":
  main()
