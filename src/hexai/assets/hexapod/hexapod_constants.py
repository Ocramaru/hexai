# pyright: reportAttributeAccessIssue=false
# mujoco's C bindings ship no stubs, so MjSpec and its members are invisible to pyright.
# Scoped to this file rather than disabled repo-wide, where the rule catches real typos.
"""Freenove Big Hexapod Kit constants.

Geometry is measured from the manufacturer's STEP assembly; see the header of
``xmls/hexapod.xml`` and ``tests/test_hexapod_geometry.py`` for provenance.

Actuators are MG995 hobby servos (18 of them, three per leg).  Unlike the
BLDC actuators on the Unitree robots, these are closed-loop position servos
with an internal controller we cannot see, so the stiffness and damping below
are a model of the servo's behaviour rather than a physical property.
"""

import math
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF.
##

HEXAPOD_XML: Path = Path(__file__).parent / "xmls" / "hexapod.xml"
assert HEXAPOD_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(HEXAPOD_XML))


##
# Actuator config.
##

# MG995 published specification at 6 V.
#   stall torque 10 kgf.cm -> 0.98 N.m
#   no-load speed 0.16 s / 60 deg -> 6.54 rad/s
# Clone MG995s commonly fall well short of the published figure, so the
# effort limit is derated to 70%.  Prefer measuring a real servo over
# trusting this number: an over-strong actuator in sim produces gaits the
# hardware cannot execute.
MG995_STALL_TORQUE = 0.98
MG995_DERATE = 0.70
MG995_EFFORT_LIMIT = MG995_STALL_TORQUE * MG995_DERATE
MG995_VELOCITY_LIMIT = 6.54

# Recorded for reference; BuiltinPositionActuatorCfg has no velocity_limit
# field, so the servo's speed ceiling is enforced by stiffness/effort, not
# by a hard clamp.

# Reflected inertia of the gear train, seen at the output shaft.  The MG995
# does not publish rotor inertia or gear ratio, so this is an estimate; it
# mainly damps high-frequency chatter and is worth revisiting if the learned
# gait looks buzzy.
MG995_ARMATURE = 1.0e-4

# A hobby servo's gains are set by its internal controller, not by us, so
# these model its behaviour rather than a physical property.  Calibrated
# against the PHYSICAL robot (2026-08-01): commanded to the standing stance
# over ssh, the real belly plate measured 27.5 mm off the ground against an
# ideal 39.4 mm; kp=3.5 reproduces that sag in sim to within a millimetre
# (28.3 mm), stable with no oscillation.  The effective stiffness folds in
# gear backlash and controller deadband, which we do not model separately.
# Erring soft is the safe side for transfer: a gait learned on mushier
# servos still runs on stiffer ones.
MG995_STIFFNESS = 3.5
MG995_DAMPING = 0.13

# One config per joint type: the regex selects all six legs at once.
HEXAPOD_COXA_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_coxa_joint",),
  stiffness=MG995_STIFFNESS,
  damping=MG995_DAMPING,
  effort_limit=MG995_EFFORT_LIMIT,
  armature=MG995_ARMATURE,
)
HEXAPOD_FEMUR_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_femur_joint",),
  stiffness=MG995_STIFFNESS,
  damping=MG995_DAMPING,
  effort_limit=MG995_EFFORT_LIMIT,
  armature=MG995_ARMATURE,
)
HEXAPOD_TIBIA_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_tibia_joint",),
  stiffness=MG995_STIFFNESS,
  damping=MG995_DAMPING,
  effort_limit=MG995_EFFORT_LIMIT,
  armature=MG995_ARMATURE,
)

# Pan/tilt head SG90s.  Published stall 1.8 kgf.cm at 4.8 V -> 0.176 N.m,
# derated like the MG995s.  Gains are modelled, not calibrated: the head
# carries ~30 g, so tracking is easy and these numbers are uncritical.
SG90_EFFORT_LIMIT = 0.176 * MG995_DERATE
HEXAPOD_HEAD_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("head_pan_joint", "head_tilt_joint"),
  stiffness=1.0,
  damping=0.03,
  effort_limit=SG90_EFFORT_LIMIT,
  armature=1.0e-5,
)

##
# Hardware mapping.
##

# PCA9685 channels, from Code/Server/control.py set_leg_angles() -- with
# Freenove's leg numbering mapped onto ours.  Their body frame points the
# opposite way (their Leg 1 is the front-RIGHT leg), verified on hardware by
# wiggling individual channels: ch12 moved the mid-right leg, ch19 the
# mid-left, ch15 the front-right.  Their legs run 1=FR 2=MR 3=RR 4=RL 5=ML
# 6=FL against our 1=RL 2=ML 3=FL 4=FR 5=MR 6=RR.
SERVO_CHANNELS: dict[str, int] = {
  "leg1_coxa_joint": 22,
  "leg1_femur_joint": 23,
  "leg1_tibia_joint": 27,
  "leg2_coxa_joint": 19,
  "leg2_femur_joint": 20,
  "leg2_tibia_joint": 21,
  "leg3_coxa_joint": 16,
  "leg3_femur_joint": 17,
  "leg3_tibia_joint": 18,
  "leg4_coxa_joint": 15,
  "leg4_femur_joint": 14,
  "leg4_tibia_joint": 13,
  "leg5_coxa_joint": 12,
  "leg5_femur_joint": 11,
  "leg5_tibia_joint": 10,
  "leg6_coxa_joint": 9,
  "leg6_femur_joint": 8,
  "leg6_tibia_joint": 31,
  "head_tilt_joint": 0,
  "head_pan_joint": 1,
}

# Legs 1-3 are the left side, 4-6 the right.  The MJCF keeps both sides
# symmetric so the policy sees one leg; the mirroring is applied here instead.
LEFT_LEGS = frozenset({1, 2, 3})


def _leg_number(joint_name: str) -> int | None:
  """The leg a joint belongs to, or None for the head."""
  return int(joint_name[3]) if joint_name.startswith("leg") else None


# Per-servo trim, degrees, added after the sign is applied.
#
# Servo horns mount on a splined shaft in discrete steps, so no two servos
# share a true zero.  Freenove's calibration stores the result on the robot
# in Code/Server/point.txt; copy that file over and run
# thirdparty/hexapod-cad/tools/freenove_pipeline.compute_trims() to fill
# this in.  Leaving it at zero costs a degree or two per joint.
SERVO_TRIM_DEG: dict[str, float] = dict.fromkeys(SERVO_CHANNELS, 0.0)

SERVO_MIN_DEG = 0.0
SERVO_MAX_DEG = 180.0

# Command = centre + sign * degrees(qpos), derived from Freenove's pipeline
# (ported and verified in thirdparty/hexapod-cad/tools/freenove_pipeline.py)
# and anchored to hardware by the channel wiggle tests:
#
#   coxa    centre 90, sign +1 on ALL legs.  The coxa servos are mounted
#           rotationally symmetric around the chassis, not mirrored, and a
#           +10 command on the mid-right leg swung its tip toward the front
#           (counter-clockwise, our +qpos).
#   femur   centre 90, +1 right / -1 left (their "90 - b" branch is the
#           right side).
#   tibia   centre 0 sign -1 right, centre 180 sign +1 left: the horn is
#           clocked at the straight leg, and the command is the flexion
#           angle.  Cross-check: their installation script drives right
#           tibias to 10 and left tibias to 170 -- both are qpos = -10 deg
#           through this map, the same slightly-bent install pose.
# Head servos: centre 90 assumed for both, signs UNVERIFIED on hardware --
# check with one small command per joint before trusting, like the legs.
_SERVO_MAP: dict[str, tuple[float, float]] = {}
for _n in SERVO_CHANNELS:
  _leg = _leg_number(_n)
  if _leg is None:
    _SERVO_MAP[_n] = (90.0, 1.0)
    continue
  _left = _leg in LEFT_LEGS
  if "tibia" in _n:
    _SERVO_MAP[_n] = (180.0, 1.0) if _left else (0.0, -1.0)
  elif "femur" in _n:
    _SERVO_MAP[_n] = (90.0, -1.0) if _left else (90.0, 1.0)
  else:
    _SERVO_MAP[_n] = (90.0, 1.0)


def servo_degrees(joint_name: str, qpos_rad: float) -> float:
  """Convert a MuJoCo joint angle to an MG995 command in [0, 180] degrees.

  This is the deploy-time half of the symmetric-MJCF choice documented in
  ``xmls/hexapod.xml``: the model keeps both sides identical and the tibia
  zeroed at the straight leg, and the hardware's centres and mirrors are
  applied here.

  Raises:
    KeyError: if ``joint_name`` is not a known hexapod joint.
  """
  centre, sign = _SERVO_MAP[joint_name]
  degrees = centre + sign * math.degrees(qpos_rad) + SERVO_TRIM_DEG[joint_name]
  return min(SERVO_MAX_DEG, max(SERVO_MIN_DEG, degrees))


def servo_command(joint_name: str, qpos_rad: float) -> tuple[int, float]:
  """Return the ``(channel, degrees)`` pair to write to the PCA9685."""
  return SERVO_CHANNELS[joint_name], servo_degrees(joint_name, qpos_rad)


##
# Collision config.
##

_FOOT_REGEX = "^leg[1-6]_foot_col$"

# Feet only: fastest, and sufficient while the gait is being learned.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_REGEX,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
)

# Everything collides.  Feet get real friction; the rest is frictionless so
# a leg brushing the chassis does not silently become a foothold.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_col$",),
  solref=(0.01, 1),
  condim={_FOOT_REGEX: 6, ".*_col$": 1},
  priority={_FOOT_REGEX: 1},
  friction={_FOOT_REGEX: (0.8, 5e-3, 5e-4)},
)

##
# Initial state.
##

# Feet 140 mm out from each hip with the body 25 mm above them, matching the
# home stance in Code/Server/control.py.  The joint angles below still need
# solving against our measured geometry rather than the vendor's planar IK.
# Standing stance: the manufacturer's home pose, feet 140 mm out from each
# hip and 110 mm below it, solved against our measured link geometry by
# thirdparty/hexapod-cad/tools/solve_stance.py.  The hip sits 31.8 mm above
# the body origin, so the origin stands 78.2 mm up.
#
# This needs the tibia at -101.4 deg, which an earlier +/-90 range made
# unreachable and forced a wider 165 mm stance.  Freenove's own code resolved
# it: their installation script clocks the tibia horn at the straight-leg
# position, so the tibia's travel is [-160, 0], and the home pose sits
# comfortably inside it.
STANCE_FEMUR_RAD = 0.4189  # +24.0 deg
STANCE_TIBIA_RAD = -1.7698  # -101.4 deg

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0782),
  joint_pos={
    ".*_coxa_joint": 0.0,
    ".*_femur_joint": STANCE_FEMUR_RAD,
    ".*_tibia_joint": STANCE_TIBIA_RAD,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

HEXAPOD_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    HEXAPOD_COXA_ACTUATOR_CFG,
    HEXAPOD_FEMUR_ACTUATOR_CFG,
    HEXAPOD_TIBIA_ACTUATOR_CFG,
    HEXAPOD_HEAD_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_hexapod_robot_cfg() -> EntityCfg:
  """Get a fresh hexapod robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=HEXAPOD_ARTICULATION,
  )


# Per-actuator action scale, following the Go1 convention: a unit action maps
# to a quarter of the deflection at which the PD controller saturates.  Leg
# actuators only -- this feeds locomotion policies, and the head servos are
# not part of a gait's action space (they hold their default pose).
HEXAPOD_ACTION_SCALE: dict[str, float] = {}
for _actuator in (
  HEXAPOD_COXA_ACTUATOR_CFG,
  HEXAPOD_FEMUR_ACTUATOR_CFG,
  HEXAPOD_TIBIA_ACTUATOR_CFG,
):
  assert _actuator.effort_limit is not None
  for _name in _actuator.target_names_expr:
    HEXAPOD_ACTION_SCALE[_name] = 0.25 * _actuator.effort_limit / _actuator.stiffness


if __name__ == "__main__":
  from mjlab.entity.entity import Entity

  robot = Entity(get_hexapod_robot_cfg())
  model = robot.spec.compile()
  print(f"nq={model.nq} nv={model.nv} nu={model.nu} mass={model.body_mass.sum():.4f}")
