"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import itertools
import math
import random
from typing import Any

import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.merge_controller import SmartCruiseControlMerge, \
  MIN_V_HIGHWAY, MIN_V_TARGET, _HOLD_TIME, _LOST_TIME, _MAX_EVENT_TIME, _COOLDOWN_TIME, _REARM_TIME, \
  _TIMEOUT_LOCKOUT_TIME, _FAULT_ALERT_TIME, _MAX_FRAME_GAP, TRIGGER_MEASURED, TRIGGER_PREDICTED, TRIGGER_ENCROACHING, \
  TRIGGER_LANE_ENDING, TRIGGER_RADAR_CONVERGING, TRIGGER_RADAR_SLOW, TRIGGER_RADAR_FOLLOW, TRIGGER_RADAR_EARLY, _RADAR_PASS_DELTA, _RADAR_FOLLOW_TIME
from openpilot.common.test import OpenpilotTestCase

MergeState = custom.LongitudinalPlanSP.SmartCruiseControl.MergeState

V_EGO = 31.  # m/s, ~70 mph
V_CRUISE = 31.
LEAD_T = np.array(ModelConstants.LEAD_T_IDXS)
_frame_ids = itertools.count(1)


def make_lead(dist: float, y: float, v: float, lat_speed: float = 0., prob: float = 0.9, v_ego: float = V_EGO):
  """A lead vehicle `dist` ahead and `y` to the right, drifting left (toward us) at `lat_speed`."""
  lead = log.ModelDataV2.LeadDataV3.new_message()
  lead.prob = prob
  lead.t = [float(t) for t in LEAD_T]
  lead.x = [float(dist + (v - v_ego) * t) for t in LEAD_T]
  lead.y = [float(y - lat_speed * t) for t in LEAD_T]
  lead.v = [float(v) for _ in LEAD_T]
  lead.a = [0. for _ in LEAD_T]
  return lead


def make_model(leads, curvature: float = 0., frame_id: int | None = None, model_v_ego: float | None = None):
  model = messaging.new_message('modelV2')
  model.modelV2.frameId = next(_frame_ids) if frame_id is None else frame_id
  if model_v_ego is not None:
    velocity = log.XYZTData.new_message()
    velocity.x = [float(model_v_ego)]
    model.modelV2.velocity = velocity
  xs = V_EGO * np.array(ModelConstants.T_IDXS)
  position = log.XYZTData.new_message()
  position.x = [float(x) for x in xs]
  position.y = [float(0.5 * curvature * x ** 2) for x in xs]
  model.modelV2.position = position
  model.modelV2.leadsV3 = leads
  return model.modelV2


def add_lanes(md, right_lane_ends: bool = False, left_lane_ends: bool = False, right_lane: bool = True, left_lane: bool = True,
              edge_std: float = 0.3, line_prob: float = 0.9):
  """Straight road. Our lane is 3.6 m wide. Optionally a lane on each side that either continues or tapers away by 90 m."""
  xs = [float(x) for x in ModelConstants.X_IDXS]

  def xy(ys):
    m = log.XYZTData.new_message()
    m.x = xs
    m.y = [float(y) for y in ys]
    return m

  def edge(sign, has_lane, ends):
    full = 1.8 + (3.6 if has_lane else 0.) + 0.5
    if not (has_lane and ends):
      return [sign * full for _ in xs]
    return [sign * (full - 3.6 * min(max((x - 20.) / 70., 0.), 1.)) for x in xs]

  md.laneLines = [xy([-5.4] * len(xs)), xy([-1.8] * len(xs)), xy([1.8] * len(xs)), xy([5.4] * len(xs))]
  md.laneLineProbs = [line_prob] * 4
  md.roadEdges = [xy(edge(-1., left_lane, left_lane_ends)), xy(edge(1., right_lane, right_lane_ends))]
  md.roadEdgeStds = [edge_std, edge_std]
  return md


def make_radar(points):
  """points: (slot, metres ahead, metres to the RIGHT of straight ahead, speed over the ground in m/s, our speed)."""
  msg = custom.CarStateSP.new_message()
  tracks = msg.init('mergeRadarTracks', len(points))
  for track, (slot, dist, right, v_obj, v_ego) in zip(tracks, points, strict=True):
    track.slot = slot
    track.dRel = dist
    track.yRel = -right  # the radar counts left as positive
    track.vRel = v_obj - v_ego
  return msg


def make_car_state(blinker: bool = False, v_ego: float = V_EGO):
  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = v_ego
  car_state.carState.rightBlinker = blinker
  return car_state.carState


class TestSmartCruiseControlMerge(OpenpilotTestCase):

  def setup_method(self):
    self.params = Params()
    self.params.put_bool("SmartCruiseControlMerge", True, block=True)
    self.params.put_bool("SmartCruiseControlMergeTestOnly", False, block=True)
    self.params.put("SmartCruiseControlMergeMaxDrop", 20, block=True)
    self.params.put_bool("SmartCruiseControlMergeBothSides", False, block=True)
    self.scc = SmartCruiseControlMerge()
    self.car_state = make_car_state()
    self.blinker = False
    self.lanes: dict = {}

  def run_scene(self, seconds: float, dist: float = 40., y0: float = 3.6, v: float = 27., lat_speed: float = 0.,
                prob: float = 0.9, v_ego: float = V_EGO, override: bool = False, curvature: float = 0., side: float = 1.,
                model_predicts: bool = False, stop_y: float = 0., v_cruise: float = V_CRUISE, frame_id: int | None = None,
                long_enabled: bool = True):
    """Drive for `seconds` with one vehicle drifting toward our path at `lat_speed` until it reaches `stop_y`."""
    y = y0
    self.car_state = make_car_state(self.blinker, v_ego)
    for _ in range(int(seconds / DT_MDL)):
      path_y = 0.5 * curvature * dist ** 2
      lead = make_lead(dist, side * y + path_y, v, side * lat_speed if model_predicts else 0., prob, v_ego)
      md = make_model([lead], curvature, frame_id)
      if self.lanes:
        add_lanes(md, **self.lanes)
      sm: Any = {'modelV2': md, 'carState': self.car_state}
      self.scc.update(sm, long_enabled, override, v_ego, 0., v_cruise)
      self.check_output(v_cruise)
      y = max(stop_y, y - lat_speed * DT_MDL)
    self.last_y = y

  def check_output(self, v_cruise: float = V_CRUISE):
    """Limits that must hold after every single frame, whatever went in."""
    out = self.scc.output_v_target
    assert math.isfinite(out)
    assert math.isfinite(self.scc.output_a_target)
    if out != V_CRUISE_UNSET:
      assert (self.scc.is_active or self.scc.ready) and not self.scc.test_only and not self.scc.fault
      assert MIN_V_TARGET <= out <= v_cruise
      assert out >= self.scc.v_ref - self.scc.max_drop - 1e-6
    if self.scc.ready:
      # getting ready may never ask for less than our own speed
      assert out != V_CRUISE_UNSET and out >= self.scc.v_ego and self.scc.state == MergeState.enabled
    if self.scc.brake_warning:
      assert self.scc.state in (MergeState.slowing, MergeState.holding)

  def test_initial_state(self):
    assert self.scc.state == MergeState.disabled
    assert not self.scc.is_active
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_system_disabled(self):
    self.params.put_bool("SmartCruiseControlMerge", False, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_scene(3., lat_speed=0.8)
    assert self.scc.state == MergeState.disabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_vehicle_staying_in_next_lane_is_ignored(self):
    # Passing a slower car that holds its own lane must not slow us down.
    self.run_scene(8., lat_speed=0.)
    assert self.scc.state == MergeState.enabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_merging_vehicle_slows_us(self):
    self.run_scene(2., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.slowing
    assert self.scc.is_active
    # slower merging car, 40 m ahead: aim below its speed, but never more than 20 mph under the set speed
    assert V_CRUISE - 20 * CV.MPH_TO_MS <= self.scc.output_v_target < 27.

  def test_reduction_is_capped(self):
    # Merging car doing 40 mph in front of us at 70 mph: the drop stops at the configured maximum.
    self.run_scene(2., v=18., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.slowing
    np.testing.assert_allclose(self.scc.output_v_target, V_CRUISE - 20 * CV.MPH_TO_MS, atol=1e-3)

  def test_max_drop_param(self):
    self.params.put("SmartCruiseControlMergeMaxDrop", 10, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_scene(2., v=18., lat_speed=0.8, stop_y=1.2)
    np.testing.assert_allclose(self.scc.output_v_target, V_CRUISE - 10 * CV.MPH_TO_MS, atol=1e-3)

  def test_faster_vehicle_is_ignored(self):
    # A car merging ahead that is pulling away needs no room from us.
    self.run_scene(3., v=V_EGO + 5., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.enabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_far_vehicle_is_ignored(self):
    self.run_scene(3., dist=200., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.enabled

  def test_left_side_is_ignored(self):
    self.run_scene(3., lat_speed=0.8, stop_y=1.2, side=-1.)
    assert self.scc.state == MergeState.enabled

  def test_left_side_with_both_sides_setting(self):
    self.params.put_bool("SmartCruiseControlMergeBothSides", True, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_scene(2., lat_speed=0.8, stop_y=1.2, side=-1.)
    assert self.scc.state == MergeState.slowing
    assert self.scc.candidate.offset < 0.
    assert V_CRUISE - 20 * CV.MPH_TO_MS <= self.scc.output_v_target < 27.

  def test_right_side_still_works_with_both_sides_setting(self):
    self.params.put_bool("SmartCruiseControlMergeBothSides", True, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_scene(2., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.slowing
    assert self.scc.candidate.offset > 0.

  def test_left_side_faster_passing_car_is_ignored(self):
    # The common case on the left: a car passes us and pulls in ahead while pulling away. No slowdown.
    self.params.put_bool("SmartCruiseControlMergeBothSides", True, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_scene(3., v=V_EGO + 4., lat_speed=0.8, stop_y=1.2, side=-1.)
    assert self.scc.state == MergeState.enabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_left_side_car_moving_away_is_ignored(self):
    # A car in the left lane drifting further left is not coming into our lane.
    self.params.put_bool("SmartCruiseControlMergeBothSides", True, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_scene(3., y0=1.5, v=V_EGO, lat_speed=-0.6, stop_y=-10., side=-1.)
    assert self.scc.state == MergeState.enabled

  def test_low_model_confidence_is_ignored(self):
    self.run_scene(3., lat_speed=0.8, stop_y=1.2, prob=0.2)
    assert self.scc.state == MergeState.enabled

  def test_not_on_highway(self):
    self.run_scene(3., dist=20., v=10., lat_speed=0.8, stop_y=1.2, v_ego=MIN_V_HIGHWAY - 1.)
    assert self.scc.state == MergeState.enabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_own_lane_change_is_ignored(self):
    self.blinker = True
    self.run_scene(3., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.enabled

  def test_works_in_a_curve(self):
    # Same merge, on a bend. Offsets are measured from our planned path, so this must still trigger...
    self.run_scene(2., lat_speed=0.8, stop_y=1.2, curvature=0.002)
    assert self.scc.state == MergeState.slowing

  def test_curve_alone_does_not_trigger(self):
    # ...and a car holding the next lane through the bend must not.
    self.run_scene(8., lat_speed=0., curvature=0.002)
    assert self.scc.state == MergeState.enabled

  def test_model_prediction_trigger(self):
    # Barely moving yet, but the model predicts it will be in our lane within 2 s.
    self.run_scene(0.2, y0=3.0, lat_speed=0.)
    assert self.scc.state == MergeState.enabled
    for _ in range(int(1. / DT_MDL)):
      lead = make_lead(40., 3.0, 27., lat_speed=1.2)
      sm: Any = {'modelV2': make_model([lead]), 'carState': self.car_state}
      self.scc.update(sm, True, False, V_EGO, 0., V_CRUISE)
    assert self.scc.state == MergeState.slowing
    assert self.scc.candidate.trigger == TRIGGER_PREDICTED

  def test_test_only_mode_never_changes_speed(self):
    self.params.put_bool("SmartCruiseControlMergeTestOnly", True, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_scene(2., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.slowing
    assert self.scc.detected
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_override(self):
    self.run_scene(2., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.slowing
    self.run_scene(0.1, lat_speed=0.8, stop_y=1.2, override=True)
    assert self.scc.state == MergeState.overriding
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_target_only_goes_down_during_event(self):
    self.run_scene(2., v=24., lat_speed=0.8, stop_y=1.2)
    low = self.scc.output_v_target
    # the merging car speeds up while still merging: keep the lower target until the event ends
    self.run_scene(0.3, y0=1.2, v=29., lat_speed=0.)
    assert self.scc.is_active
    assert self.scc.output_v_target == low

  def test_full_cycle_releases_speed(self):
    # merge in: 3.6 m -> 0 m (fully in our lane, now an ordinary lead)
    self.run_scene(5., lat_speed=0.8, stop_y=0.)
    assert self.scc.state in (MergeState.slowing, MergeState.holding)

    self.run_scene(_LOST_TIME + _HOLD_TIME + 0.5, y0=0., lat_speed=0.)
    assert self.scc.state == MergeState.enabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_stuck_detection_times_out(self):
    # A never-ending "merge" (bad detection) must let go on its own and then stay quiet for the cooldown.
    for _ in range(int((_MAX_EVENT_TIME + 1.) / DT_MDL)):
      lead = make_lead(40., 2.4, 27., lat_speed=0.9)
      sm: Any = {'modelV2': make_model([lead]), 'carState': self.car_state}
      self.scc.update(sm, True, False, V_EGO, 0., V_CRUISE)
      if self.scc.frame * DT_MDL > _MAX_EVENT_TIME + 0.6:
        assert self.scc.state == MergeState.enabled
        assert self.scc.output_v_target == V_CRUISE_UNSET
    assert self.scc.cooldown_frames > 0
    assert _COOLDOWN_TIME > 0

  def test_no_leads(self):
    for _ in range(int(2. / DT_MDL)):
      sm: Any = {'modelV2': make_model([]), 'carState': self.car_state}
      self.scc.update(sm, True, False, V_EGO, 0., V_CRUISE)
    assert self.scc.state == MergeState.enabled

  # ---------- failsafes ----------

  def test_bad_numbers_never_trigger_or_crash(self):
    for bad in (float('nan'), float('inf'), -float('inf'), 1e30, -1e30):
      for field in ('prob', 'x', 'y', 'v'):
        self.scc = SmartCruiseControlMerge()
        for i in range(int(2. / DT_MDL)):
          lead = make_lead(40., 3.6 - 0.8 * i * DT_MDL, 27.)
          if field == 'prob':
            lead.prob = bad
          else:
            setattr(lead, field, [bad for _ in LEAD_T])
          sm: Any = {'modelV2': make_model([lead]), 'carState': self.car_state}
          self.scc.update(sm, True, False, V_EGO, 0., V_CRUISE)
          self.check_output()
        assert not self.scc.fault, (bad, field)
        assert self.scc.state == MergeState.enabled, (bad, field)

  def test_bad_ego_inputs_never_trigger_or_crash(self):
    for bad in (float('nan'), float('inf'), -1e30):
      for which in ('v_ego', 'a_ego', 'v_cruise'):
        self.scc = SmartCruiseControlMerge()
        y = 3.6
        for _ in range(int(2. / DT_MDL)):
          v_ego = bad if which == 'v_ego' else V_EGO
          cs = make_car_state(False, v_ego)
          sm: Any = {'modelV2': make_model([make_lead(40., y, 27.)]), 'carState': cs}
          self.scc.update(sm, True, False, v_ego, bad if which == 'a_ego' else 0., bad if which == 'v_cruise' else V_CRUISE)
          assert abs(self.scc.output_a_target) <= 5.
          if which != 'a_ego':
            assert self.scc.output_v_target == V_CRUISE_UNSET
          y = max(1.2, y - 0.8 * DT_MDL)
        assert not self.scc.fault

  def test_bad_path_is_ignored(self):
    for _ in range(int(2. / DT_MDL)):
      md = make_model([make_lead(40., 2.4, 27., lat_speed=0.9)])
      md.position.y = [float('nan') for _ in md.position.y]
      self.scc.update({'modelV2': md, 'carState': self.car_state}, True, False, V_EGO, 0., V_CRUISE)
      self.check_output()
    assert self.scc.state == MergeState.enabled
    assert not self.scc.fault

  def test_unexpected_error_latches_off(self):
    self.run_scene(2., lat_speed=0.8, stop_y=1.2)
    assert self.scc.is_active
    # carState missing: something is badly wrong upstream
    self.scc.update({'modelV2': make_model([])}, True, False, V_EGO, 0., V_CRUISE)
    assert self.scc.fault
    assert self.scc.fault_alert
    assert self.scc.state == MergeState.disabled
    assert not self.scc.is_active
    assert self.scc.output_v_target == V_CRUISE_UNSET
    # good data again: it stays off until the process restarts
    self.run_scene(_FAULT_ALERT_TIME + 1., lat_speed=0.8, stop_y=1.2)
    assert self.scc.fault
    assert not self.scc.fault_alert
    assert self.scc.state == MergeState.disabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_garbage_arguments_latch_off(self):
    self.scc.update(None, True, False, "fast", None, object())  # type: ignore[arg-type]
    assert self.scc.fault
    assert self.scc.output_v_target == V_CRUISE_UNSET
    assert self.scc.output_a_target == 0.

  def test_repeated_model_frame_is_ignored(self):
    # Model stopped publishing: the same frame over and over must never count as a merge.
    self.run_scene(5., y0=2.4, lat_speed=0.9, model_predicts=True, stop_y=2.4, frame_id=7)
    assert self.scc.state == MergeState.enabled

  def test_model_frame_gap_resets_tracking(self):
    self.run_scene(0.3, lat_speed=0.8, stop_y=1.2)
    for _ in range(_MAX_FRAME_GAP + 2):
      next(_frame_ids)
    self.run_scene(DT_MDL, y0=3.3, lat_speed=0.)
    assert self.scc.candidate_frames == 0
    assert all(t.closing_speed == 0. for t in self.scc.tracks)

  def test_timeout_locks_out(self):
    def stuck(seconds):
      for _ in range(int(seconds / DT_MDL)):
        sm: Any = {'modelV2': make_model([make_lead(40., 2.4, 27., lat_speed=0.9)]), 'carState': self.car_state}
        self.scc.update(sm, True, False, V_EGO, 0., V_CRUISE)
        self.check_output()

    stuck(_MAX_EVENT_TIME + 1.)
    assert self.scc.state == MergeState.enabled
    # still "detecting" for most of the lockout: nothing may happen
    stuck(_TIMEOUT_LOCKOUT_TIME - 2.)
    assert self.scc.state == MergeState.enabled
    # and even after the lockout, the stuck detection must clear first (re-arm)
    stuck(5.)
    assert self.scc.state == MergeState.enabled
    self.run_scene(_REARM_TIME + 0.2, lat_speed=0.)
    stuck(1.)
    assert self.scc.state == MergeState.slowing

  def test_cooldown_between_events(self):
    self.run_scene(2., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.slowing
    self.run_scene(0.1, lat_speed=0., override=True)
    self.run_scene(0.1, lat_speed=0.)
    assert self.scc.state == MergeState.enabled
    # a normal end only needs the short cooldown; the stricter "road must look clear" rule is for cut-off events
    assert self.scc.cooldown_frames > 0 and self.scc.armed
    self.run_scene(0.5, lat_speed=0.8, y0=3.6, stop_y=1.2)
    assert self.scc.state == MergeState.enabled

  def test_never_below_absolute_floor(self):
    self.params.put("SmartCruiseControlMergeMaxDrop", 30, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_scene(2., dist=25., v=5., lat_speed=0.8, stop_y=1.2, v_ego=21., v_cruise=22.)
    assert self.scc.state == MergeState.slowing
    np.testing.assert_allclose(self.scc.output_v_target, MIN_V_TARGET)

  def test_drop_is_measured_from_actual_speed(self):
    # Set to 70 mph but only doing 56 mph in traffic. The 20 mph limit counts from 56, not from 70.
    self.run_scene(2., dist=30., v=10., lat_speed=0.8, stop_y=1.2, v_ego=25.)
    assert self.scc.state == MergeState.slowing
    np.testing.assert_allclose(self.scc.output_v_target, 25. - 20 * CV.MPH_TO_MS, atol=1e-3)

  def test_cruise_speed_not_set(self):
    self.run_scene(3., lat_speed=0.8, stop_y=1.2, v_cruise=V_CRUISE_UNSET * CV.KPH_TO_MS)
    assert self.scc.state == MergeState.enabled

  def test_set_speed_lowered_during_event(self):
    self.run_scene(2., v=26., lat_speed=0.8, stop_y=1.2)
    assert self.scc.output_v_target < V_CRUISE
    # driver dials the set speed far down: we must never ask for more than the set speed
    self.run_scene(0.5, y0=1.2, v=26., lat_speed=0., v_cruise=18.)

  def test_cruise_off_cancels(self):
    self.run_scene(2., lat_speed=0.8, stop_y=1.2)
    assert self.scc.is_active
    self.run_scene(DT_MDL, y0=1.2, lat_speed=0., long_enabled=False)
    assert self.scc.state == MergeState.disabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_lead_speed_uses_car_speed_scale(self):
    # Model thinks we do 33 m/s and the other car 29: that is 4 m/s slower than us, i.e. 27 on the car's scale.
    y = 3.6
    for _ in range(int(2. / DT_MDL)):
      md = make_model([make_lead(40., y, 29.)], model_v_ego=33.)
      self.scc.update({'modelV2': md, 'carState': self.car_state}, True, False, V_EGO, 0., V_CRUISE)
      y = max(1.2, y - 0.8 * DT_MDL)
    assert self.scc.state == MergeState.slowing
    np.testing.assert_allclose(self.scc.candidate.v_lead, 27., atol=1e-3)

  def test_fuzz(self):
    """Throw 40,000 frames of random good and bad data at it. Nothing may escape, every limit must hold."""
    rng = random.Random(1234)
    specials = [float('nan'), float('inf'), -float('inf'), 0., -1., 1e9, -1e9]

    def num(lo, hi, p_bad=0.03):
      return rng.choice(specials) if rng.random() < p_bad else rng.uniform(lo, hi)

    active_run = 0
    longest = 0
    events = 0
    was_active = False
    for scenario in range(400):
      long_enabled = rng.random() < 0.9
      override = rng.random() < 0.05
      v_ego = num(0., 45.)
      v_cruise = num(10., 45.) if rng.random() < 0.9 else V_CRUISE_UNSET * CV.KPH_TO_MS
      blinker = rng.random() < 0.05
      n_leads = rng.choice([0, 1, 2, 3])
      seeds = [(num(0., 120.), num(-6., 6.), num(0., 45.), num(-2., 2.), num(0., 1.)) for _ in range(n_leads)]
      if scenario % 3 == 0:
        # a believable merge, so the active states get exercised too (bad values are still mixed in below)
        v_ego = rng.uniform(22., 36.)
        v_cruise = v_ego + rng.uniform(-2., 6.)
        seeds = [(rng.uniform(15., 70.), rng.uniform(2., 4.), v_ego - rng.uniform(-1., 15.), rng.uniform(0.4, 1.5), 0.9)]
      if rng.random() < 0.3:
        self.params.put_bool("SmartCruiseControlMergeTestOnly", rng.random() < 0.5, block=True)
        self.params.put_bool("SmartCruiseControlMergeBothSides", rng.random() < 0.5, block=True)
        self.params.put("SmartCruiseControlMergeMaxDrop", rng.choice([5, 20, 30, -4, 999]), block=True)
      for i in range(100):
        leads = []
        for dist, y, v, lat, prob in seeds:
          lead = make_lead(dist, y - lat * i * DT_MDL, v, lat, 0.9, 30.)
          lead.prob = prob if rng.random() < 0.9 else num(0., 1., 0.5)
          if rng.random() < 0.02:
            lead.x = [num(0., 100., 0.5) for _ in range(rng.choice([0, 1, 6]))]
          leads.append(lead)
        md = make_model(leads, num(-0.003, 0.003, 0.), None if rng.random() < 0.97 else rng.choice([0, 5, 10 ** 9]))
        if scenario % 2 == 0:
          add_lanes(md, right_lane_ends=scenario % 4 == 0, left_lane_ends=scenario % 8 == 0, edge_std=num(0., 2., 0.05),
                    line_prob=num(0., 1., 0.05))
          if rng.random() < 0.02:
            md.roadEdges[1].y = [num(-5., 9., 0.3) for _ in md.roadEdges[1].y]
          if rng.random() < 0.01:
            md.laneLines[2].x = []
        if rng.random() < 0.01:
          md.position.x = []
        cs = make_car_state(blinker, v_ego)
        sm_fuzz: Any = {'modelV2': md, 'carState': cs}
        if scenario % 5 != 0:
          pts = [(rng.randrange(32), num(0., 180.), num(-10., 10.), num(-5., 45.), num(0., 45.)) for _ in range(rng.choice([0, 3, 12, 32]))]
          if scenario % 3 == 1:
            pts.append((1, max(5., 120. - i * 0.6), 6.6, 18., 30.))
          sm_fuzz['carStateSP'] = make_radar(pts)
        self.scc.update(sm_fuzz, long_enabled, override, v_ego, num(-3., 3.), v_cruise)

        out = self.scc.output_v_target
        assert not self.scc.fault
        assert math.isfinite(out) and math.isfinite(self.scc.output_a_target)
        if not long_enabled or override:
          assert not self.scc.is_active
          assert out == V_CRUISE_UNSET
        if out != V_CRUISE_UNSET:
          assert self.scc.is_active and not self.scc.test_only
          assert MIN_V_TARGET <= out <= v_cruise
          assert out >= self.scc.v_ref - self.scc.max_drop - 1e-6
          assert out >= self.scc.v_ref - 40 * CV.MPH_TO_MS - 1e-6
        active_run = active_run + 1 if self.scc.is_active else 0
        longest = max(longest, active_run)
        events += int(self.scc.is_active and not was_active)
        was_active = self.scc.is_active

    assert longest <= int(_MAX_EVENT_TIME / DT_MDL) + 1
    assert events > 15  # the fuzz really did reach the active states
    assert _COOLDOWN_TIME > 0 and _HOLD_TIME > 0 and _LOST_TIME > 0

  # ---------- acting early ----------

  def test_measured_trigger_is_reported(self):
    self.run_scene(2., lat_speed=0.8, stop_y=1.2)
    assert self.scc.candidate.trigger == TRIGGER_MEASURED

  def test_slow_car_hugging_our_line_triggers_without_moving(self):
    # 9 mph slower than us and sitting 2.4 m off our path (on the line), not moving sideways at all.
    self.run_scene(1., y0=2.4, v=27., lat_speed=0.)
    assert self.scc.state == MergeState.slowing
    assert self.scc.candidate.trigger == TRIGGER_ENCROACHING

  def test_slow_car_centered_in_its_lane_is_ignored(self):
    # Passing a much slower truck that stays in the middle of the next lane, and that lane continues.
    self.lanes = {}
    self.run_scene(6., y0=3.6, v=20., lat_speed=0.)
    assert self.scc.state == MergeState.enabled
    self.lanes = {'right_lane_ends': False}
    self.run_scene(6., y0=3.6, v=20., lat_speed=0.)
    assert self.scc.state == MergeState.enabled
    assert not self.scc.lane_ending[1.]

  def test_on_ramp_slow_car_triggers_before_it_moves(self):
    # The situation from the road: we do 65 mph, a car on the on-ramp does 40 mph, still centered in its own lane.
    self.params.put("SmartCruiseControlMergeMaxDrop", 30, block=True)
    self.scc = SmartCruiseControlMerge()
    self.lanes = {'right_lane_ends': True}
    v_ego, v_other = 65 * CV.MPH_TO_MS, 40 * CV.MPH_TO_MS
    self.run_scene(1.5, dist=120., y0=3.6, v=v_other, lat_speed=0., v_ego=v_ego, v_cruise=v_ego)
    assert self.scc.lane_ending[1.]
    assert self.scc.state == MergeState.slowing
    assert self.scc.candidate.trigger == TRIGGER_LANE_ENDING
    # speed is brought down to about theirs (never under the 35 mph floor), not just trimmed a little
    assert MIN_V_TARGET <= self.scc.output_v_target <= v_other

  def test_lane_ending_with_car_at_our_speed_is_ignored(self):
    self.lanes = {'right_lane_ends': True}
    self.run_scene(3., y0=3.6, v=V_EGO, lat_speed=0.)
    assert self.scc.lane_ending[1.]
    assert self.scc.state == MergeState.enabled

  def test_lane_ending_needs_confident_road_edge(self):
    self.lanes = {'right_lane_ends': True, 'edge_std': 2.5}
    self.run_scene(3., y0=3.6, v=20., lat_speed=0.)
    assert not self.scc.lane_ending[1.]
    assert self.scc.state == MergeState.enabled

  def test_left_lane_ending_only_with_both_sides(self):
    self.lanes = {'left_lane_ends': True}
    self.run_scene(3., y0=3.6, v=20., lat_speed=0., side=-1.)
    assert self.scc.lane_ending[-1.]
    assert self.scc.state == MergeState.enabled
    self.params.put_bool("SmartCruiseControlMergeBothSides", True, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_scene(3., y0=3.6, v=20., lat_speed=0., side=-1.)
    assert self.scc.state == MergeState.slowing
    assert self.scc.candidate.trigger == TRIGGER_LANE_ENDING

  def test_shoulder_is_not_a_lane(self):
    # No lane on the right at all, just a narrow strip to the road edge.
    self.lanes = {'right_lane': False}
    self.run_scene(3., y0=3.6, v=20., lat_speed=0.)
    assert not self.scc.lane_ending[1.]

  def test_reacts_further_ahead(self):
    # 4.5 s of travel ahead used to be out of range.
    self.run_scene(2., dist=V_EGO * 4.5, v=22., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.slowing

  def test_matches_slower_speed_and_opens_gap(self):
    self.params.put("SmartCruiseControlMergeMaxDrop", 40, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_scene(2., dist=30., v=24., lat_speed=0.8, stop_y=1.2)
    # 30 m ahead is about 1 s: aim well under their 24 m/s so the gap grows
    assert self.scc.output_v_target < 24. - 3.

  def test_debug_slots_are_logged(self):
    self.run_scene(0.5, y0=3.6, v=27., lat_speed=0., prob=0.1)
    assert self.scc.state == MergeState.enabled
    np.testing.assert_allclose(self.scc.debug_prob[0], 0.1, atol=1e-3)
    np.testing.assert_allclose(self.scc.debug_offset[0], 3.6, atol=1e-3)
    np.testing.assert_allclose(self.scc.debug_speed[0], 27., atol=1e-3)

  # ---------- the car's own radar ----------

  def run_radar(self, seconds: float, objects, v_ego: float = V_EGO, v_cruise: float = V_CRUISE, curvature: float = 0., noise: float = 0.,
                seed: int = 0):
    """objects: list of dicts(slot, dist, right, v, lat_speed [m/s toward us], stop). Positions advance with time."""
    rng = random.Random(seed)
    objs = [dict(o) for o in objects]
    cs = make_car_state(self.blinker, v_ego)
    for _ in range(int(seconds / DT_MDL)):
      points = []
      for o in objs:
        path_y = 0.5 * curvature * o['dist'] ** 2
        points.append((o['slot'], o['dist'], o['right'] + path_y + rng.uniform(-noise, noise), o['v'], v_ego))
        o['dist'] = max(1., o['dist'] + (o['v'] - v_ego) * DT_MDL)
        if 'lat_speed' in o:
          step = o['lat_speed'] * DT_MDL
          o['right'] = max(o.get('stop', 0.), o['right'] - step) if o['right'] > 0 else min(-o.get('stop', 0.), o['right'] + step)
      sm: Any = {'modelV2': make_model([], curvature), 'carState': cs, 'carStateSP': make_radar(points)}
      self.scc.update(sm, True, False, v_ego, 0., v_cruise)
      self.check_output(v_cruise)

  def test_radar_slow_queue_on_the_right(self):
    # The recorded on-ramp: we do 69 mph, a line of cars two lanes to the right does 41 mph, 110 m ahead, not moving over yet.
    self.params.put("SmartCruiseControlMergeMaxDrop", 30, block=True)
    self.scc = SmartCruiseControlMerge()
    v_ego, v_q = 69 * CV.MPH_TO_MS, 41 * CV.MPH_TO_MS
    self.run_radar(1., [{'slot': 1, 'dist': 110., 'right': 6.6, 'v': v_q}, {'slot': 10, 'dist': 135., 'right': 6.8, 'v': v_q}], v_ego=v_ego, v_cruise=v_ego)
    assert self.scc.state == MergeState.slowing
    assert self.scc.candidate.trigger == TRIGGER_RADAR_SLOW
    assert self.scc.radar_objects == 2
    # defensive: pass them at most 15 mph faster
    np.testing.assert_allclose(self.scc.output_v_target, v_q + _RADAR_PASS_DELTA, atol=0.05)

  def test_radar_passing_a_slightly_slower_truck_is_ignored(self):
    # 10 mph slower in the next lane and holding its lane: ordinary passing, no slowdown.
    self.run_radar(6., [{'slot': 3, 'dist': 90., 'right': 3.6, 'v': V_EGO - 10 * CV.MPH_TO_MS}])
    assert self.scc.state == MergeState.enabled
    assert self.scc.radar_objects == 1

  def test_radar_vehicle_moving_into_our_lane(self):
    self.run_radar(3., [{'slot': 5, 'dist': 45., 'right': 3.6, 'v': V_EGO - 2., 'lat_speed': 0.7, 'stop': 1.5}])
    assert self.scc.state == MergeState.slowing
    assert self.scc.candidate.trigger == TRIGGER_RADAR_CONVERGING
    assert self.scc.output_v_target < V_EGO - 2.

  def test_radar_faster_vehicle_moving_in_is_ignored(self):
    self.params.put_bool("SmartCruiseControlMergeBothSides", True, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_radar(4., [{'slot': 5, 'dist': 20., 'right': -3.6, 'v': V_EGO + 3., 'lat_speed': 0.7, 'stop': 1.5}])
    assert self.scc.state == MergeState.enabled

  def test_radar_noisy_sideways_reading_does_not_trigger(self):
    # A car holding the next lane, with the kind of jitter a radar has. Several seeds, 20 s each.
    for seed in range(5):
      self.scc = SmartCruiseControlMerge()
      self.run_radar(20., [{'slot': 4, 'dist': 60., 'right': 3.6, 'v': V_EGO - 1.}], noise=0.3, seed=seed)
      assert self.scc.state == MergeState.enabled, seed

  def test_radar_standing_objects_are_ignored(self):
    # Signs and barriers: not moving over the ground.
    self.run_radar(3., [{'slot': 2, 'dist': 120., 'right': 5., 'v': 0.}, {'slot': 6, 'dist': 80., 'right': 3., 'v': 1.}])
    assert self.scc.state == MergeState.enabled
    assert self.scc.radar_objects == 0

  def test_radar_vehicle_in_our_own_lane_is_left_to_stock_cruise(self):
    self.run_radar(2., [{'slot': 2, 'dist': 90., 'right': 0.3, 'v': 15.}])
    assert self.scc.state == MergeState.enabled

  def test_radar_left_side_needs_both_sides(self):
    queue = [{'slot': 1, 'dist': 100., 'right': -6.6, 'v': 17.}]
    self.run_radar(1.5, queue)
    assert self.scc.state == MergeState.enabled
    self.params.put_bool("SmartCruiseControlMergeBothSides", True, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_radar(1.5, queue)
    assert self.scc.state == MergeState.slowing

  def test_radar_far_side_lanes_are_ignored(self):
    # Three lanes over.
    self.run_radar(3., [{'slot': 1, 'dist': 100., 'right': 11., 'v': 15.}])
    assert self.scc.state == MergeState.enabled

  def test_radar_works_in_a_curve(self):
    # A car holding the next lane through a bend sits off to the side of "straight ahead" but not of our path.
    self.run_radar(8., [{'slot': 4, 'dist': 70., 'right': 3.6, 'v': V_EGO - 0.5}], curvature=0.002)
    assert self.scc.state == MergeState.enabled

  def test_radar_slot_reuse_is_not_motion(self):
    # The radar hands slot 9 to a different vehicle every few frames. That must never look like a sideways move.
    cs = make_car_state(False, V_EGO)
    for i in range(int(6. / DT_MDL)):
      right = 3.6 if (i // 4) % 2 == 0 else 2.2
      dist = 50. if (i // 4) % 2 == 0 else 80.
      sm: Any = {'modelV2': make_model([]), 'carState': cs, 'carStateSP': make_radar([(9, dist, right, V_EGO - 1., V_EGO)])}
      self.scc.update(sm, True, False, V_EGO, 0., V_CRUISE)
    assert self.scc.state == MergeState.enabled

  def test_radar_bad_numbers(self):
    cs = make_car_state(False, V_EGO)
    for bad in (float('nan'), float('inf'), -1e30, 1e30):
      self.scc = SmartCruiseControlMerge()
      for _ in range(40):
        msg = make_radar([(1, 100., 6.6, 17., V_EGO), (2, 60., 3.6, V_EGO, V_EGO)])
        msg.mergeRadarTracks[0].dRel = bad
        msg.mergeRadarTracks[1].yRel = bad
        self.scc.update({'modelV2': make_model([]), 'carState': cs, 'carStateSP': msg}, True, False, V_EGO, 0., V_CRUISE)
        self.check_output()
      assert not self.scc.fault and self.scc.state == MergeState.enabled

  def test_radar_test_only(self):
    self.params.put_bool("SmartCruiseControlMergeTestOnly", True, block=True)
    self.scc = SmartCruiseControlMerge()
    self.run_radar(1.5, [{'slot': 1, 'dist': 100., 'right': 6.6, 'v': 17.}])
    assert self.scc.state == MergeState.slowing
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_long_queue_is_not_cut_off_early(self):
    # Passing a long slow queue takes a while. The slowdown must hold for its whole length.
    queue = [{'slot': k, 'dist': 60. + 25. * k, 'right': 6.6, 'v': 15.} for k in range(20)]
    for second in range(20):
      self.run_radar(1., [dict(q, dist=q['dist'] - 16. * second) for q in queue if q['dist'] - 16. * second > 5.])
      if second >= 1:
        assert self.scc.is_active, second

  # ---------- follow-through: one merge, one slowdown ----------

  def run_phases(self, phases, dist=38., right=4.3, v_obj=None, v_ego=None, slot=5):
    """phases: (seconds, sideways speed toward us in m/s). Returns the state after every frame."""
    v_ego = 56 * CV.MPH_TO_MS if v_ego is None else v_ego
    v_obj = 50 * CV.MPH_TO_MS if v_obj is None else v_obj
    cs = make_car_state(False, v_ego)
    states, triggers, targets = [], [], []
    for seconds, lat_speed in phases:
      for _ in range(int(seconds / DT_MDL)):
        sm: Any = {'modelV2': make_model([]), 'carState': cs, 'carStateSP': make_radar([(slot, dist, right, v_obj, v_ego)])}
        self.scc.update(sm, True, False, v_ego, 0., 70 * CV.MPH_TO_MS)
        self.check_output(70 * CV.MPH_TO_MS)
        states.append(self.scc.state)
        triggers.append(self.scc.candidate.trigger if self.scc.candidate is not None else 0)
        targets.append(self.scc.output_v_target)
        dist = max(5., dist + (v_obj - v_ego) * DT_MDL)
        right -= lat_speed * DT_MDL
    return states, triggers, targets

  @staticmethod
  def count_events(states):
    active = [s in (MergeState.slowing, MergeState.holding) for s in states]
    return sum(1 for a, b in zip([False] + active, active, strict=False) if b and not a)

  def test_recorded_hesitant_merge_is_one_slowdown(self):
    # Recorded Nov 24: the car moved over, paused next to our line for 3 s, then came in.
    # The old code let go during the pause, sped up, then slowed again.
    states, triggers, targets = self.run_phases([(2., 0.), (2., 0.6), (3., 0.), (2., 1.), (4., 0.)])
    assert self.count_events(states) == 1
    assert TRIGGER_RADAR_CONVERGING in triggers and TRIGGER_RADAR_FOLLOW in triggers
    # it never let go while the car was still beside us and coming (the pause is 4 s to 7 s)
    pause = states[int(4.5 / DT_MDL):int(7. / DT_MDL)]
    assert all(s == MergeState.slowing for s in pause)

  def test_target_is_close_to_the_merging_cars_speed(self):
    # 50 mph car, 1.5 s ahead of us: aim a few mph under it, not 9 under as before.
    states, triggers, targets = self.run_phases([(2., 0.), (2., 0.6)])
    low = min(t for t in targets if t != V_CRUISE_UNSET)
    assert 43 * CV.MPH_TO_MS < low < 49 * CV.MPH_TO_MS

  def test_follow_lets_go_of_a_car_that_stays_beside_us(self):
    # It moved toward us once, then just drives along in its lane, matching our speed. Do not hold the speed down for ever.
    v = 56 * CV.MPH_TO_MS
    states, triggers, targets = self.run_phases([(2., 0.), (1.5, 0.6), (_RADAR_FOLLOW_TIME + _HOLD_TIME + 4., 0.)], v_obj=v, v_ego=v)
    assert self.count_events(states) == 1
    assert states[-1] == MergeState.enabled

  def test_follow_lets_go_when_the_car_backs_out(self):
    states, triggers, targets = self.run_phases([(2., 0.), (1.5, 0.6), (1., 0.), (2., -0.8), (_HOLD_TIME + 1., 0.)])
    assert self.count_events(states) == 1
    assert states[-1] == MergeState.enabled
    # following stopped soon after it started moving away
    back_out = int(4.5 / DT_MDL)
    assert TRIGGER_RADAR_FOLLOW not in triggers[back_out + int(1.5 / DT_MDL):]

  def test_follow_never_starts_an_event(self):
    # A car driving beside us with no sideways move must never be "followed".
    states, triggers, targets = self.run_phases([(15., 0.)], right=3.4)
    assert self.count_events(states) == 0 and TRIGGER_RADAR_FOLLOW not in triggers

  def test_follow_does_not_stick_to_a_reused_radar_slot(self):
    # The merging car leaves the radar and the slot is reused by another car far away in the same lane.
    v_ego, v_obj = 56 * CV.MPH_TO_MS, 50 * CV.MPH_TO_MS
    cs = make_car_state(False, v_ego)
    triggers = []
    right = 4.3
    for i in range(int(8. / DT_MDL)):
      t = i * DT_MDL
      if 2. <= t < 4.5:
        right -= 0.6 * DT_MDL
      point = (5, 38., right, v_obj, v_ego) if t < 4.5 else (5, 70., 3.6, v_obj, v_ego)
      self.scc.update({'modelV2': make_model([]), 'carState': cs, 'carStateSP': make_radar([point])}, True, False, v_ego, 0., 70 * CV.MPH_TO_MS)
      self.check_output(70 * CV.MPH_TO_MS)
      triggers.append(self.scc.candidate.trigger if self.scc.candidate is not None else 0)
    assert TRIGGER_RADAR_CONVERGING in triggers
    assert TRIGGER_RADAR_FOLLOW not in triggers[int(4.6 / DT_MDL):]

  def test_target_eases_to_the_cars_speed_once_it_is_in(self):
    # While it moves in we aim under its speed to open a gap. Once it is in our lane we aim at its speed, not below.
    v_obj = 50 * CV.MPH_TO_MS
    states, triggers, targets = self.run_phases([(2., 0.), (5., 0.7), (2., 0.)], v_obj=v_obj)
    assert self.count_events(states) == 1
    slowing = [t for s, t in zip(states, targets, strict=True) if s == MergeState.slowing]
    holding = [t for s, t in zip(states, targets, strict=True) if s == MergeState.holding]
    assert slowing and holding
    assert min(slowing) < v_obj - 1.
    np.testing.assert_allclose(holding, v_obj - 0.5, atol=0.3)

  # ---------- early: a vehicle coming across fast from further out ----------

  def run_tracking(self, phases, right, dist=49., v_obj=None, v_ego=None):
    """Like run_phases, but also returns the vehicle's offset at every frame."""
    v_ego = 58 * CV.MPH_TO_MS if v_ego is None else v_ego
    v_obj = 47 * CV.MPH_TO_MS if v_obj is None else v_obj
    cs = make_car_state(False, v_ego)
    out = []
    for seconds, lat_speed in phases:
      for _ in range(int(seconds / DT_MDL)):
        sm: Any = {'modelV2': make_model([]), 'carState': cs, 'carStateSP': make_radar([(5, dist, right, v_obj, v_ego)])}
        self.scc.update(sm, True, False, v_ego, 0., 70 * CV.MPH_TO_MS)
        self.check_output(70 * CV.MPH_TO_MS)
        out.append((self.scc.state, self.scc.candidate.trigger if self.scc.candidate is not None else 0, self.scc.output_v_target, right))
        dist = max(5., dist + (v_obj - v_ego) * DT_MDL)
        right -= lat_speed * DT_MDL
    return out, v_obj

  def test_recorded_fast_merge_is_caught_early(self):
    # Recorded Nov 24: first seen 8.6 m to the right, coming across at 1.6 m/s. The old code waited until it was 3.8 m away.
    out, v_obj = self.run_tracking([(3.5, 1.6)], right=8.6)
    first = next(o for o in out if o[0] == MergeState.slowing)
    assert first[1] == TRIGGER_RADAR_EARLY
    assert first[3] > 5.5  # still more than a lane and a half away when we start
    # gentle at first: not below its speed while it is still far out
    early = [o[2] for o in out if o[1] == TRIGGER_RADAR_EARLY and o[2] != V_CRUISE_UNSET]
    assert early and min(early) >= v_obj - 0.01
    # then the normal response takes over, in the same event
    assert any(o[1] == TRIGGER_RADAR_CONVERGING for o in out)
    states = [o[0] for o in out]
    assert self.count_events(states) == 1

  def test_ordinary_lane_change_two_lanes_over_is_not_early(self):
    # A car going from the third lane to the second at a normal pace. It is not coming for our lane.
    out, _ = self.run_tracking([(1., 0.), (3.6, 1.0), (3., 0.)], right=7.2)
    assert all(o[1] != TRIGGER_RADAR_EARLY for o in out)

  def test_slow_drift_far_out_is_not_early(self):
    out, _ = self.run_tracking([(8., 0.3)], right=8.4)
    assert all(o[1] != TRIGGER_RADAR_EARLY for o in out)

  def test_fast_mover_that_is_faster_than_us_is_not_early(self):
    v = 58 * CV.MPH_TO_MS
    out, _ = self.run_tracking([(3., 1.6)], right=8.6, v_obj=v + 3., v_ego=v)
    assert all(o[0] not in (MergeState.slowing, MergeState.holding) for o in out)

  def test_early_only_event_lets_go(self):
    # It came across fast but stopped in the lane next to ours. A gentle slowdown, then back to normal.
    out, _ = self.run_tracking([(2., 1.6), (_HOLD_TIME + 3., 0.)], right=8.6)
    assert any(o[1] == TRIGGER_RADAR_EARLY for o in out)
    assert out[-1][0] == MergeState.enabled

  def test_brief_fast_move_far_out_is_not_early(self):
    # A quick sideways move that would still need more than 4 s to reach our lane.
    out, _ = self.run_tracking([(1.1, 1.4), (4., 0.)], right=8.45)
    assert all(o[1] != TRIGGER_RADAR_EARLY for o in out)

  def test_far_away_fast_mover_is_not_early(self):
    # Beyond 60 m the radar's sideways reading jumps around too much to trust a one-second move.
    v = 58 * CV.MPH_TO_MS
    out, _ = self.run_tracking([(2.5, 1.6)], right=8.6, dist=85., v_obj=v, v_ego=v)
    assert all(o[1] != TRIGGER_RADAR_EARLY for o in out)

  # ---------- getting ready, and the brake warning ----------

  def test_ready_brings_set_speed_to_our_speed_without_slowing(self):
    # Recorded Nov 25: in traffic at 58 mph with the cruise set to 70. A car far to the right drifts our way slowly.
    out, _ = self.run_tracking([(1.5, 0.), (3., 0.5)], right=8.4, dist=60., v_obj=54 * CV.MPH_TO_MS)
    v_ego = 58 * CV.MPH_TO_MS
    ready = [o for o in out if o[0] == MergeState.enabled and o[2] != V_CRUISE_UNSET]
    assert ready, "expected the set speed to be prepared before any slowdown"
    assert all(v_ego < o[2] <= v_ego + 1.1 for o in ready)
    first_slow = next((i for i, o in enumerate(out) if o[0] == MergeState.slowing), len(out))
    first_ready = next(i for i, o in enumerate(out) if o[2] != V_CRUISE_UNSET)
    assert first_ready < first_slow

  def test_ready_is_silent_in_test_only(self):
    self.params.put_bool("SmartCruiseControlMergeTestOnly", True, block=True)
    self.scc = SmartCruiseControlMerge()
    out, _ = self.run_tracking([(1.5, 0.), (3., 0.5)], right=8.4, dist=60., v_obj=54 * CV.MPH_TO_MS)
    assert all(o[2] == V_CRUISE_UNSET for o in out)

  def test_ready_not_for_a_car_holding_its_lane_or_a_faster_one(self):
    out, _ = self.run_tracking([(6., 0.)], right=3.6, dist=40., v_obj=54 * CV.MPH_TO_MS)
    assert all(o[2] == V_CRUISE_UNSET for o in out)
    v = 58 * CV.MPH_TO_MS
    out, _ = self.run_tracking([(1.5, 0.), (3., 0.5)], right=8.4, dist=60., v_obj=v + 4., v_ego=v)
    assert all(o[2] == V_CRUISE_UNSET for o in out)

  def test_ready_ends_when_the_car_stops_coming(self):
    out, _ = self.run_tracking([(1.5, 0.), (1.5, 0.5), (8., 0.)], right=8.4, dist=60., v_obj=58 * CV.MPH_TO_MS)
    assert any(o[2] != V_CRUISE_UNSET for o in out)
    assert out[-1][2] == V_CRUISE_UNSET

  def brake_flags(self, **kw):
    flags = []
    orig = self.scc.update

    def wrapped(*a, **k):
      orig(*a, **k)
      flags.append(self.scc.brake_warning)
    self.scc.update = wrapped
    try:
      self.run_tracking(**kw)
    finally:
      self.scc.update = orig
    return flags

  def test_brake_warning_for_close_slow_cut_in(self):
    # Recorded Nov 25: a 51 mph car pulls out of a slow line 29 m ahead while we do 66.
    flags = self.brake_flags(phases=[(1.5, 0.), (2., 1.5)], right=3.6, dist=40., v_obj=51 * CV.MPH_TO_MS, v_ego=66 * CV.MPH_TO_MS)
    assert any(flags)

  def test_no_brake_warning_for_a_comfortable_merge(self):
    # Same sideways move, but 60 m ahead and only 6 mph slower: the cruise can deal with that.
    flags = self.brake_flags(phases=[(1.5, 0.), (3., 1.0)], right=4.2, dist=60., v_obj=56 * CV.MPH_TO_MS, v_ego=62 * CV.MPH_TO_MS)
    assert not any(flags)
