"""Merge Assist acting on turn signals seen by Blinker Watch."""
import types

import numpy as np

from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.blinker import beacon
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import merge_controller as mc
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.merge_controller import SmartCruiseControlMerge, MergeState, TRIGGER_BLINKER, \
  TRIGGER_RADAR_SLOW, TRIGGER_RADAR_CONVERGING
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.tests.test_merge_controller import make_radar, make_model, make_car_state

V_EGO = 31.          # ~70 mph
LEFT, RIGHT = -1, 1


class TestMergeBlinker:
  def setup_method(self):
    self.params = Params()
    self.params.put_bool("SmartCruiseControlMerge", True, block=True)
    self.params.put_bool("SmartCruiseControlMergeTestOnly", False, block=True)
    self.params.put("SmartCruiseControlMergeMaxDrop", 30, block=True)
    self.params.put_bool("SmartCruiseControlMergeBothSides", True, block=True)
    self.params.put_bool("BlinkerWatch", True, block=True)
    self.params.remove(beacon.KEY)
    self.now = 100.
    self.scc = SmartCruiseControlMerge()
    self.scc.clock = lambda: self.now
    self.lowest = V_CRUISE_UNSET
    self.triggers: set[int] = set()

  def drive(self, seconds, dist=40., right=3.6, v_obj=V_EGO - 6., signal=0, note_from=0., note_until=1e9, note_dist_error=0., note_slot=5,
            raw_note=None, lat_speed=0., v_ego=V_EGO, v_cruise=None):
    """One vehicle (radar slot 5) beside us. `signal` is what Blinker Watch reports for it between note_from and note_until."""
    cs = make_car_state(False, v_ego)
    for i in range(int(seconds / DT_MDL)):
      t = i * DT_MDL
      self.now += DT_MDL
      if raw_note is not None:
        self.params.put(beacon.KEY, raw_note)
      elif signal and note_from <= t < note_until and i % 4 == 0:
        det = types.SimpleNamespace(slot=note_slot, side=signal, dist=dist + note_dist_error)
        self.params.put(beacon.KEY, beacon.encode([det], self.now))
      sm = {'modelV2': make_model([]), 'carState': cs, 'carStateSP': make_radar([(5, dist, right, v_obj, v_ego)])}
      self.scc.update(sm, True, False, v_ego, 0., v_ego if v_cruise is None else v_cruise)
      out = self.scc.output_v_target
      assert out == V_CRUISE_UNSET or (np.isfinite(out) and mc.MIN_V_TARGET <= out <= max(v_ego, v_cruise or 0.) + 1e-6)
      if self.scc.state == MergeState.slowing and self.scc.candidate is not None:
        self.triggers.add(self.scc.candidate.trigger)
        self.lowest = min(self.lowest, out)
      dist = max(5., dist + (v_obj - v_ego) * DT_MDL)
      right = right - lat_speed * DT_MDL if right > 0 else right + lat_speed * DT_MDL
    return dist

  def test_signal_toward_us_from_the_right_slows_to_its_speed_plus_5mph(self):
    self.drive(3., right=3.6, signal=LEFT)
    assert self.triggers == {TRIGGER_BLINKER}
    np.testing.assert_allclose(self.lowest, V_EGO - 6. + 5. * CV.MPH_TO_MS, atol=0.05)

  def test_signal_toward_us_from_the_left(self):
    self.drive(3., right=-3.6, signal=RIGHT)
    assert self.triggers == {TRIGGER_BLINKER}

  def test_left_side_is_ignored_when_only_the_right_is_watched(self):
    self.params.put_bool("SmartCruiseControlMergeBothSides", False, block=True)
    self.scc = SmartCruiseControlMerge()
    self.scc.clock = lambda: self.now
    self.drive(3., right=-3.6, signal=RIGHT)
    assert not self.triggers

  def test_no_signal_no_slowdown(self):
    self.drive(3., signal=0)
    assert not self.triggers and self.scc.state == MergeState.enabled

  def test_signal_away_from_us_does_nothing(self):
    self.drive(3., right=3.6, signal=RIGHT)
    assert not self.triggers

  def test_switched_off_means_signals_are_ignored(self):
    self.params.put_bool("BlinkerWatch", False, block=True)
    self.scc = SmartCruiseControlMerge()
    self.scc.clock = lambda: self.now
    self.drive(3., signal=LEFT)
    assert not self.triggers

  def test_test_only_mode_changes_no_speed(self):
    self.params.put_bool("SmartCruiseControlMergeTestOnly", True, block=True)
    self.scc = SmartCruiseControlMerge()
    self.scc.clock = lambda: self.now
    cs = make_car_state(False, V_EGO)
    for i in range(60):
      self.now += DT_MDL
      self.params.put(beacon.KEY, beacon.encode([types.SimpleNamespace(slot=5, side=LEFT, dist=40.)], self.now))
      self.scc.update({'modelV2': make_model([]), 'carState': cs, 'carStateSP': make_radar([(5, 40., 3.6, V_EGO - 6., V_EGO)])}, True, False, V_EGO, 0., V_EGO)
      assert self.scc.output_v_target == V_CRUISE_UNSET
    assert self.scc.state == MergeState.slowing and self.scc.candidate.trigger == TRIGGER_BLINKER

  def test_vehicle_as_fast_as_us_or_faster_is_left_alone(self):
    self.drive(3., v_obj=V_EGO + 1., signal=LEFT)
    assert not self.triggers

  def test_vehicle_two_lanes_over_or_far_ahead_is_left_alone(self):
    self.drive(3., right=7.2, signal=LEFT)
    assert not self.triggers
    self.setup_method()
    self.drive(1.5, dist=75., v_obj=V_EGO - 1., signal=LEFT)
    assert not self.triggers

  def test_note_for_another_vehicle_or_another_place_is_ignored(self):
    self.drive(3., signal=LEFT, note_slot=6)
    assert not self.triggers
    self.setup_method()
    self.drive(3., signal=LEFT, note_dist_error=12.)   # the radar slot was reused for a different vehicle
    assert not self.triggers

  def test_signal_that_stops_is_forgotten(self):
    self.drive(2.6, signal=LEFT, note_until=1.)
    assert TRIGGER_BLINKER in self.triggers
    assert not self.scc.blinkers                      # 1.6 s after the last note
    self.drive(1.5, signal=0, dist=34., v_obj=V_EGO - 1.)
    self.drive(8., signal=0, dist=34., v_obj=V_EGO - 1.)
    assert self.scc.state == MergeState.enabled and self.scc.output_v_target == V_CRUISE_UNSET

  def test_unreadable_notes_never_cause_a_fault_or_a_slowdown(self):
    for junk in ("", "|", "5:-1:40.0", "5:-1:40.0|abc", "5:0:40.0|100.0", "5:-1:nan|100.0", "x" * 500, "5:-1:40.0|nan", "5:-1:40.0|999999", b"\xff\xfe", 12):
      self.setup_method()
      self.drive(1., raw_note=junk)
      assert not self.triggers and not self.scc.fault, junk

  def test_broken_store_never_causes_a_fault(self):
    class Broken:
      def get(self, key):
        raise OSError("no such store")
    self.scc.mem_params = Broken()
    self.drive(2., signal=LEFT)
    assert not self.triggers and not self.scc.fault

  def test_slow_vehicle_signalling_away_is_not_hung_back_for(self):
    # we do 70, a car in the right lane does 40: without a signal the 'slow traffic beside us' rule holds us to its speed + 20 mph
    v_slow = 40 * CV.MPH_TO_MS
    self.drive(1.5, dist=90., right=3.6, v_obj=v_slow, signal=0)
    assert self.triggers == {TRIGGER_RADAR_SLOW}
    self.setup_method()
    self.drive(1.5, dist=90., right=3.6, v_obj=v_slow, signal=RIGHT)   # signalling right from the right lane: it is leaving
    assert not self.triggers and self.scc.output_v_target == V_CRUISE_UNSET

  def test_a_real_move_over_still_wins_over_an_away_signal(self):
    # the camera says "signalling away" but the radar sees it coming into our lane: the radar is believed
    self.drive(3., dist=45., right=3.6, v_obj=V_EGO - 2., signal=RIGHT, lat_speed=0.7)
    assert TRIGGER_RADAR_CONVERGING in self.triggers

  def test_signal_then_move_over_hands_on_to_the_normal_merge_rule(self):
    self.drive(1.5, dist=45., right=3.6, v_obj=V_EGO - 5., signal=LEFT)
    first = self.lowest
    assert self.triggers == {TRIGGER_BLINKER}
    self.drive(2.5, dist=36., right=3.6, v_obj=V_EGO - 5., signal=LEFT, lat_speed=0.8)
    assert TRIGGER_RADAR_CONVERGING in self.triggers and self.lowest < first      # now it makes real room


class TestBeacon:
  def test_round_trip(self):
    dets = [types.SimpleNamespace(slot=5, side=-1, dist=40.26), types.SimpleNamespace(slot=17, side=1, dist=12.)]
    assert beacon.decode(beacon.encode(dets, 50.), 50.5, 1.2) == {5: (-1, 40.3), 17: (1, 12.0)}

  def test_old_or_future_notes_are_dropped(self):
    text = beacon.encode([types.SimpleNamespace(slot=5, side=-1, dist=40.)], 50.)
    assert beacon.decode(text, 51.3, 1.2) == {} and beacon.decode(text, 49., 1.2) == {}

  def test_at_most_eight_vehicles(self):
    dets = [types.SimpleNamespace(slot=i, side=1, dist=20.) for i in range(20)]
    assert len(beacon.decode(beacon.encode(dets, 1.), 1., 1.)) == 8
