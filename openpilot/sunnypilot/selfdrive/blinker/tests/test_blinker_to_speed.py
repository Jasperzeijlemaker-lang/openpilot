"""Whole chain in the picture simulator: camera picture -> Blinker Watch -> note -> Merge Assist -> cruise speed."""
import numpy as np

from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.blinker import beacon
from openpilot.sunnypilot.selfdrive.blinker.blinker_core import BlinkerWatcher
from openpilot.sunnypilot.selfdrive.blinker.sim import Scene, RPY, CAM_HEIGHT
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.merge_controller import SmartCruiseControlMerge, MergeState, TRIGGER_BLINKER
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.tests.test_merge_controller import make_radar, make_model, make_car_state

V_EGO = 31.


def run(seconds=6., seed=1, signal=0, y_left=-3.6, closing=6., dist=45., **scene_kw):
  """A vehicle `closing` m/s slower than us, y_left metres to our left (negative = right lane). Returns (first time the
  cruise target dropped or None, lowest target, controller)."""
  params = Params()
  for key, val in (("SmartCruiseControlMerge", True), ("SmartCruiseControlMergeTestOnly", False), ("SmartCruiseControlMergeBothSides", True),
                   ("BlinkerWatch", True)):
    params.put_bool(key, val, block=True)
  params.put("SmartCruiseControlMergeMaxDrop", 30, block=True)
  params.remove(beacon.KEY)
  clock = [500.]
  scc = SmartCruiseControlMerge()
  scc.clock = lambda: clock[0]
  watcher = BlinkerWatcher()
  scene = Scene(np.random.default_rng(seed), dist=dist, y_left=y_left, v_ego=V_EGO, v_rel=-closing, signal=signal, **scene_kw)
  cs = make_car_state(False, V_EGO)
  first, lowest = None, V_CRUISE_UNSET
  for i in range(int(seconds / DT_MDL)):
    clock[0] += DT_MDL
    y, uv, tracks = scene.frame(DT_MDL)
    detections = watcher.process(y, uv, tracks, RPY, CAM_HEIGHT, V_EGO)
    if detections and i % 4 == 0:
      params.put(beacon.KEY, beacon.encode(detections, clock[0]))
    radar = make_radar([(slot, d, -yl, V_EGO + vr, V_EGO) for slot, d, yl, vr in tracks])
    scc.update({'modelV2': make_model([]), 'carState': cs, 'carStateSP': radar}, True, False, V_EGO, 0., V_EGO)
    if scc.output_v_target != V_CRUISE_UNSET and scc.state == MergeState.slowing:
      assert scc.candidate is None or scc.candidate.trigger == TRIGGER_BLINKER
      lowest = min(lowest, scc.output_v_target)
      if first is None:
        first = (i + 1) * DT_MDL
  return first, lowest, scc


def test_right_lane_car_signalling_left_brings_the_cruise_down():
  hits = 0
  for seed in range(10):
    first, lowest, _ = run(seed=seed, signal=-1)
    if first is not None:
      hits += 1
      assert first < 4.5
      np.testing.assert_allclose(lowest, V_EGO - 6. + 5. * CV.MPH_TO_MS, atol=0.05)
  assert hits >= 9


def test_left_lane_car_signalling_right_brings_the_cruise_down():
  first, lowest, _ = run(seed=3, signal=1, y_left=3.6)
  assert first is not None and lowest < V_EGO - 3.


def test_no_signal_brake_lights_hazards_and_shadows_change_nothing():
  for k, kw in enumerate(({}, {"brake": True}, {"hazards": True}, {"brake": True, "brake_pump_hz": 1.5}, {"flicker_hz": 1.5}, {"glints": 0.15})):
    for seed in range(4):
      first, _, scc = run(seed=10 * k + seed, **kw)
      assert first is None and not scc.fault, kw


def test_signalling_away_changes_nothing():
  first, _, _ = run(seed=2, signal=1)       # right-lane car signalling right
  assert first is None
