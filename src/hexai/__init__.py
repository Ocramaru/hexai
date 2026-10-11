"""Perception and learning for the Freenove Big Hexapod.

Depends on mjlab for simulation and math helpers; everything here is the robot's own stack.

Author: Marco Cassar (@Ocramaru)
"""

from pathlib import Path

from dotenv import load_dotenv

from hexai.models.vvae import (
  VVAE,
  VVAEConfig,
  VVAEGeometric,
  VVAEGeometricConfig,
  VVAEOutput,
)

# Importing hexai at all loads the repo's .env, so a notebook kernel and pytest get the wandb key
# and project the same way `uv run --env-file` would. An exported variable still wins, and this
# sits below the imports only because nothing above reads the environment -- move it up if that
# stops being true. The path is resolved from this file, since a notebook runs from its own folder.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

__all__ = [
  "VVAE",
  "VVAEConfig",
  "VVAEGeometric",
  "VVAEGeometricConfig",
  "VVAEOutput",
]
