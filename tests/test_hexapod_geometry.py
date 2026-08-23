# pyright: reportAttributeAccessIssue=false
# mujoco's C bindings ship no stubs, so MjSpec and its members are invisible to pyright.
# Scoped to this file rather than disabled repo-wide, where the rule catches real typos.
"""Geometry regression tests for the Freenove hexapod MJCF.

Every expected value here was measured from the manufacturer's STEP assembly
(the manufacturer's STEP assembly) by locating the servo
horns, which sit on the servo output shafts and therefore mark the joint axes.
Offsets were solved across all six legs so that fixed linkage structure could
be separated from the arbitrary pose the assembly was saved in.

These tests exist so that adding the remaining legs cannot silently perturb
the geometry we verified for leg 2.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from hexai.assets.hexapod.hexapod_constants import get_spec

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
needs_scripts = pytest.mark.skipif(
  not (SCRIPTS / "train_hexapod.py").exists(),
  reason="cross-checks a constant against train_hexapod*.py, still in the mjlab fork",
)

HEXAPOD_XML = (
  Path(__file__).resolve().parents[1] / "src/hexai/assets/hexapod/xmls/hexapod.xml"
)

# Joint-axis positions relative to the base body, at the zero pose, in mm.
#
# Two frame changes are applied to the raw CAD measurements:
#   1. Yaw:  forward is -90 deg in CAD (the ultrasonic sensor and camera tower
#            sit there), so robot yaw = CAD yaw + 90, i.e. (x, y) -> (-y, x).
#   2. Height: the body origin is the chassis box centre, which is the CAD
#            origin shifted down 33.8 mm, so z_robot = z_cad + 33.8.
CAD_JOINT_AXES_MM = {
  "leg2_coxa": (0.0, 85.0, 31.8),  # CAD (85.0, 0.0, -2.0)
  "leg2_femur": (-11.8, 118.3, -7.5),  # CAD (118.3, 11.8, -41.3)
  "leg2_tibia": (-11.8, 208.3, -7.5),  # CAD (208.3, 11.8, -41.3)
}

# Height of the coxa axes above the body origin, mm.
HIP_HEIGHT_MM = 31.8

# Link lengths in mm, constant across all six legs in the CAD.
FEMUR_LENGTH_MM = 90.0
TIBIA_LENGTH_MM = 110.0

TOLERANCE_MM = 0.5


@pytest.fixture
def model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(HEXAPOD_XML))


@pytest.fixture
def data(model: mujoco.MjModel) -> mujoco.MjData:
  d = mujoco.MjData(model)
  d.qpos[:] = 0.0
  d.qpos[3] = 1.0  # identity quaternion for the freejoint
  mujoco.mj_forward(model, d)
  return d


def body_pos_mm(model, data, name: str) -> np.ndarray:
  """Body origin relative to the base body, in millimetres."""
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
  base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
  return (data.xpos[bid] - data.xpos[base]) * 1000.0


@pytest.mark.parametrize("name,expected", CAD_JOINT_AXES_MM.items())
def test_joint_axis_matches_cad(model, data, name, expected):
  measured = body_pos_mm(model, data, name)
  error = np.linalg.norm(measured - np.array(expected))
  assert error < TOLERANCE_MM, f"{name}: {measured.round(2)} vs CAD {expected}"


def test_link_lengths(model, data):
  femur = body_pos_mm(model, data, "leg2_femur")
  tibia = body_pos_mm(model, data, "leg2_tibia")
  assert np.linalg.norm(tibia - femur) == pytest.approx(
    FEMUR_LENGTH_MM, abs=TOLERANCE_MM
  )

  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "leg2_foot")
  base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
  foot = (data.site_xpos[sid] - data.xpos[base]) * 1000.0
  assert np.linalg.norm(foot - tibia) == pytest.approx(
    TIBIA_LENGTH_MM, abs=TOLERANCE_MM
  )


def set_joint(model, data, name: str, degrees: float) -> None:
  jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  data.qpos[model.jnt_qposadr[jid]] = np.radians(degrees)
  mujoco.mj_forward(model, data)


def foot_mm(model, data) -> np.ndarray:
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "leg2_foot")
  base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
  return (data.site_xpos[sid] - data.xpos[base]) * 1000.0


def test_femur_pitch_matches_cad_pose(model, data):
  """The CAD assembly has leg 2's femur pitched up 5.04 deg.

  Driving our model to that angle should reproduce the measured tibia axis,
  which checks the chain end-to-end rather than link by link.
  """
  set_joint(model, data, "leg2_femur_joint", 5.04)
  measured = body_pos_mm(model, data, "leg2_tibia")
  cad_posed = np.array([-11.8, 207.6, 0.4])  # CAD (207.6, 11.8, -33.4)
  assert np.linalg.norm(measured - cad_posed) < TOLERANCE_MM


LEG_LAYOUT = {  # leg: (hip radius mm, robot yaw deg, side sign)
  1: (93.8, 144.0, +1),
  2: (85.0, 90.0, +1),
  3: (93.8, 36.0, +1),
  4: (93.8, -36.0, -1),
  5: (85.0, -90.0, -1),
  6: (93.8, -144.0, -1),
}


@pytest.mark.parametrize("leg,expected", LEG_LAYOUT.items())
def test_hip_placement(model, data, leg, expected):
  radius, yaw, _ = expected
  hip = body_pos_mm(model, data, f"leg{leg}_coxa")
  assert np.hypot(hip[0], hip[1]) == pytest.approx(radius, abs=TOLERANCE_MM)
  assert np.degrees(np.arctan2(hip[1], hip[0])) == pytest.approx(yaw, abs=0.1)
  assert hip[2] == pytest.approx(HIP_HEIGHT_MM, abs=TOLERANCE_MM)


@pytest.mark.parametrize("leg", LEG_LAYOUT)
def test_every_leg_has_identical_links(model, data, leg):
  """All six legs are the same physical assembly; only placement differs."""
  femur = body_pos_mm(model, data, f"leg{leg}_femur")
  tibia = body_pos_mm(model, data, f"leg{leg}_tibia")
  coxa = body_pos_mm(model, data, f"leg{leg}_coxa")
  assert np.linalg.norm(femur - coxa) == pytest.approx(52.85, abs=TOLERANCE_MM)
  assert np.linalg.norm(tibia - femur) == pytest.approx(90.0, abs=TOLERANCE_MM)


def test_left_and_right_legs_are_mirrored(model, data):
  """Mirror pairs 1<->6, 2<->5, 3<->4 must reflect through the xz plane."""
  for left, right in ((1, 6), (2, 5), (3, 4)):
    lf = body_pos_mm(model, data, f"leg{left}_femur")
    rf = body_pos_mm(model, data, f"leg{right}_femur")
    assert lf[0] == pytest.approx(rf[0], abs=TOLERANCE_MM)
    assert lf[1] == pytest.approx(-rf[1], abs=TOLERANCE_MM)
    assert lf[2] == pytest.approx(rf[2], abs=TOLERANCE_MM)


@pytest.mark.parametrize("leg", LEG_LAYOUT)
def test_positive_femur_lifts_every_leg(model, data, leg):
  """Option A: one sign convention across all six legs, including mirrored ones."""
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"leg{leg}_foot")
  base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
  before = (data.site_xpos[sid] - data.xpos[base])[2]
  set_joint(model, data, f"leg{leg}_femur_joint", 30.0)
  assert (data.site_xpos[sid] - data.xpos[base])[2] > before


def test_total_mass_matches_cad(model):
  """Mass and inertia are solved from the CAD, calibrated against the scale.

  Measured on the real robot: MG995 servo 54.2 g with its cable, matching
  the published 55 g to within scale precision; and the acrylic plates at
  1.005 g/cm3 effective (nominal 1.18, less sheet-thickness tolerance).  Steel is 6.73 g/cm3 effective (M4 nut 0.74 g over its CAD volume), brass
  standoffs nominal 8.5.  Screws are point masses at their CAD positions,
  and every servo mount is topped up to the measured 8.72 g of hardware.
  Four 48 g cells sit at the battery-holder positions, and an 80 g wiring
  harness reconciles the chassis against the assembled robot: 1900 g without
  batteries on the scale, so 2091 g with them.  The legs are pinned
  independently -- the tibia body is 71.6 g against 71.43 g measured.
  """
  assert model.body_mass.sum() == pytest.approx(2.091, abs=2e-3)


# --- v9: head collision geom -------------------------------------------------
#
# The head/camera mount had only a VISUAL geom (head_tilt_vis), so the
# body-contact damage term (train_hexapod_goal.BODY_GEOMS /
# mdp.body_contact_count) could not sense impacts on the part carrying the
# camera the whole project depends on. head_tilt_col is its collision twin.

STANCE_FEMUR_DEG = np.degrees(0.4189)
STANCE_TIBIA_DEG = np.degrees(-1.7698)


def _standing_data(model) -> mujoco.MjData:
  """Robot at the trained standing stance (hexapod_constants.INIT_STATE)."""
  data = mujoco.MjData(model)
  data.qpos[:] = 0.0
  data.qpos[3] = 1.0  # Identity freejoint quaternion.
  data.qpos[2] = 0.0782
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name is None:
      continue
    if name.endswith("_femur_joint"):
      data.qpos[model.jnt_qposadr[i]] = np.radians(STANCE_FEMUR_DEG)
    elif name.endswith("_tibia_joint"):
      data.qpos[model.jnt_qposadr[i]] = np.radians(STANCE_TIBIA_DEG)
  mujoco.mj_forward(model, data)
  return data


def test_head_collision_geom_matches_the_visual_pose_and_extent(model):
  """Follows the ``*_col`` convention: a collision twin of the visual geom,
  not an independently-eyeballed box."""
  vis = model.geom("head_tilt_vis")
  col = model.geom("head_tilt_col")
  assert col.type == vis.type
  np.testing.assert_allclose(col.size, vis.size)
  np.testing.assert_allclose(col.pos, vis.pos)
  assert model.geom_bodyid[col.id] == model.geom_bodyid[vis.id]


def test_head_collision_geom_is_massless():
  """Mass lives in the body's <inertial>, like every other ``*_col`` geom.

  Checked on the pre-compile ``MjSpec`` -- compiled ``MjModel`` has no
  per-geom mass array, only per-body, and ``head_tilt``'s explicit
  ``<inertial>`` would make a body-mass comparison indirect."""
  col = get_spec().geom("head_tilt_col")
  assert col.mass == pytest.approx(0.0)


def test_head_collision_geom_collides_by_default(model):
  """No explicit contype/conaffinity in the XML: FULL_COLLISION
  (hexapod_constants.py, pattern ``.*_col$``) sets them programmatically, and
  the raw XML defaults (1, 1) already match what that config would apply."""
  col = model.geom("head_tilt_col")
  assert model.geom_contype[col.id] == 1
  assert model.geom_conaffinity[col.id] == 1


def test_total_mass_unchanged_by_the_head_collision_geom(model):
  """Same assertion ``test_total_mass_matches_cad`` makes -- a massless geom
  addition must not move this number at all."""
  assert model.body_mass.sum() == pytest.approx(2.091, abs=2e-3)


def test_no_self_contact_at_the_standing_stance(model):
  data = _standing_data(model)
  assert data.ncon == 0


def test_no_self_contact_across_the_full_pan_tilt_sweep(model):
  """The head_tilt_col<->base_col near-miss (~1 mm at the closest pose,
  before the ``base``/``head_tilt`` <exclude> pair) checked exhaustively, not
  just at one pose -- the standing-stance check alone would have missed it,
  since it only occurs off-centre in pan/tilt.
  """
  data = _standing_data(model)
  pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_pan_joint")
  tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_tilt_joint")
  pan_adr = model.jnt_qposadr[pan_id]
  tilt_adr = model.jnt_qposadr[tilt_id]
  max_ncon = 0
  for pan_deg in np.linspace(-90.0, 90.0, 13):
    for tilt_deg in np.linspace(-40.0, 90.0, 14):
      data.qpos[pan_adr] = np.radians(pan_deg)
      data.qpos[tilt_adr] = np.radians(tilt_deg)
      mujoco.mj_forward(model, data)
      max_ncon = max(max_ncon, data.ncon)
  assert max_ncon == 0


def test_head_body_pair_is_excluded():
  """The grandparent-grandchild artifact base_col/head_tilt_col needed the
  same fix as the pre-existing base/femur pairs -- verified on the spec
  rather than assumed, since a silently-missing exclude would only show up
  as an occasional spurious contact deep in the pan/tilt sweep."""
  spec = get_spec()
  pairs = {(e.bodyname1, e.bodyname2) for e in spec.excludes}
  assert ("base", "head_tilt") in pairs or ("head_tilt", "base") in pairs


@needs_scripts
def test_body_contact_geom_set_picks_up_the_head_automatically():
  """The consumer-side half of this task: train_hexapod_goal.BODY_GEOMS is
  derived from the compiled spec's ``*_col`` geoms (minus the feet), so the
  new geom must join it without any hand-edit there."""
  import importlib.util
  import sys
  from pathlib import Path

  scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
  name = "train_hexapod_goal"
  if name in sys.modules:
    module = sys.modules[name]
  else:
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    inserted = str(scripts_dir) not in sys.path
    if inserted:
      sys.path.insert(0, str(scripts_dir))
    try:
      sys.modules[spec.name] = module
      spec.loader.exec_module(module)
    finally:
      if inserted:
        sys.path.remove(str(scripts_dir))
  assert "head_tilt_col" in module.BODY_GEOMS


def _load_train_hexapod():
  import importlib.util
  import sys
  from pathlib import Path

  scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
  name = "train_hexapod"
  if name in sys.modules:
    return sys.modules[name]
  spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  inserted = str(scripts_dir) not in sys.path
  if inserted:
    sys.path.insert(0, str(scripts_dir))
  try:
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
  finally:
    if inserted:
      sys.path.remove(str(scripts_dir))
  return module


# --- v10: robot head-height envelope (overhang clearance) -------------------


@needs_scripts
def test_head_envelope_matches_the_measured_max(model):
  """Recomputes the worst-case head height directly from the compiled MJCF
  and checks it against ``train_hexapod.HEXAPOD_HEAD_ENVELOPE_MAX_M`` -- the
  constant that ``WALKER_OVERHANG_CLEARANCE`` is sized against. A regression
  here (a geometry edit that quietly raises the head) would otherwise only
  show up as the walker hitting overhangs again.
  """
  train_hexapod = _load_train_hexapod()
  data = _standing_data(model)
  pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_pan_joint")
  tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_tilt_joint")
  pan_adr = model.jnt_qposadr[pan_id]
  tilt_adr = model.jnt_qposadr[tilt_id]
  col = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head_tilt_col")
  size = model.geom_size[col]
  corners = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
  )

  max_top = -np.inf
  for pan_deg in np.linspace(-90.0, 90.0, 13):
    for tilt_deg in np.linspace(-40.0, 90.0, 14):
      data.qpos[pan_adr] = np.radians(pan_deg)
      data.qpos[tilt_adr] = np.radians(tilt_deg)
      mujoco.mj_forward(model, data)
      xmat = data.geom_xmat[col].reshape(3, 3)
      tops = data.geom_xpos[col][2] + (corners * size) @ xmat.T[:, 2]
      max_top = max(max_top, float(tops.max()))

  assert max_top == pytest.approx(train_hexapod.HEXAPOD_HEAD_ENVELOPE_MAX_M, abs=2e-3)
  # The constant must not UNDER-state the true envelope -- that would leave
  # WALKER_OVERHANG_CLEARANCE's margin smaller than documented.
  assert train_hexapod.HEXAPOD_HEAD_ENVELOPE_MAX_M >= max_top - 1e-6


@needs_scripts
def test_walker_overhang_clearance_clears_the_head_envelope_with_margin():
  train_hexapod = _load_train_hexapod()
  floor, _ceiling = train_hexapod.WALKER_OVERHANG_CLEARANCE
  margin = floor - train_hexapod.HEXAPOD_HEAD_ENVELOPE_MAX_M
  assert margin >= 0.05, f"only {margin * 1000:.0f} mm of clearance margin"
