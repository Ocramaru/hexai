"""Notebooks commit as source. An output left in one stays in the history for good.

Author: Marco Cassar (@Ocramaru)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_carry_no_output():
  """Figures worth keeping go in figures/ as png, which github renders; plotly output it does not."""
  dirty = [path.relative_to(ROOT) for path in (ROOT / "src").rglob("*.ipynb")
           if any(cell.get("outputs") for cell in json.loads(path.read_text())["cells"])]
  assert not dirty, f"clear them with `uv run jupyter nbconvert --clear-output --inplace`: {dirty}"
