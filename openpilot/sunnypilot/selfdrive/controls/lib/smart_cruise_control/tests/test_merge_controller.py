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
  _TIMEOUT_LOCKOUT_TIME, _FAULT_ALERT_TIME, _MAX_FRAME_GAP
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
    self.scc = SmartCruiseControlMerge()
    self.car_state = make_car_state()
    self.blinker = False

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
      sm: Any = {'modelV2': make_model([lead], curvature, frame_id), 'carState': self.car_state}
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
      assert self.scc.is_active and not self.scc.test_only and not self.scc.fault
      assert MIN_V_TARGET <= out <= v_cruise
      assert out >= self.scc.v_ref - self.scc.max_drop - 1e-6

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
    self.run_scene(3., dist=150., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.enabled

  def test_left_side_is_ignored(self):
    self.run_scene(3., lat_speed=0.8, stop_y=1.2, side=-1.)
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
    self.run_scene(0.2, y0=2.4, lat_speed=0.)
    assert self.scc.state == MergeState.enabled
    for _ in range(int(1. / DT_MDL)):
      lead = make_lead(40., 2.4, 27., lat_speed=0.9)
      sm: Any = {'modelV2': make_model([lead]), 'carState': self.car_state}
      self.scc.update(sm, True, False, V_EGO, 0., V_CRUISE)
    assert self.scc.state == MergeState.slowing

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

  def test_rearm_needed_between_events(self):
    self.run_scene(2., lat_speed=0.8, stop_y=1.2)
    assert self.scc.state == MergeState.slowing
    self.run_scene(0.1, lat_speed=0., override=True)
    self.run_scene(0.1, lat_speed=0.)
    assert self.scc.state == MergeState.enabled
    assert self.scc.cooldown_frames > 0 and not self.scc.armed

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
        if rng.random() < 0.01:
          md.position.x = []
        cs = make_car_state(blinker, v_ego)
        self.scc.update({'modelV2': md, 'carState': cs}, long_enabled, override, v_ego, num(-3., 3.), v_cruise)

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
          assert out >= self.scc.v_ref - 30 * CV.MPH_TO_MS - 1e-6
        active_run = active_run + 1 if self.scc.is_active else 0
        longest = max(longest, active_run)
        events += int(self.scc.is_active and not was_active)
        was_active = self.scc.is_active

    assert longest <= int(_MAX_EVENT_TIME / DT_MDL) + 1
    assert events > 15  # the fuzz really did reach the active states
    assert _COOLDOWN_TIME > 0 and _HOLD_TIME > 0 and _LOST_TIME > 0
