"""The Freenove Big Hexapod: twelve servos, a pan/tilt head, one camera.

Constants were measured from the manufacturer's CAD and calibrated on hardware; see
`hexapod_constants.py` and `tests/test_hexapod_geometry.py` for provenance.

Author: Marco Cassar (@Ocramaru)
"""

from hexai.assets.hexapod.hexapod_constants import (
  HEXAPOD_ARTICULATION,
  get_hexapod_robot_cfg,
  get_spec,
)

__all__ = [
  "HEXAPOD_ARTICULATION",
  "get_hexapod_robot_cfg",
  "get_spec",
]
