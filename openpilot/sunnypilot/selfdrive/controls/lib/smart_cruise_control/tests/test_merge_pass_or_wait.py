"""Merge Assist deciding between three answers for a vehicle that wants into our lane:
pass it (keep speed), let it in ahead with the room that is already there, or make room."""
import numpy as np

from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.merge_controller import MergeState, TRIGGER_BLINKER, TRIGGER_LANE_ENDING, \
  TRIGGER_RADAR_CONVERGING, TRIGGER_RADAR_EARLY, TRIGGER_RADAR_FOLLOW
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.tests.test_merge_blinker import TestMergeBlinker as Base, V_EGO, LEFT

MOVING_IN = {TRIGGER_RADAR_CONVERGING, TRIGGER_RADAR_EARLY, TRIGGER_RADAR_FOLLOW}


class TestPassOrWait(Base):
  def lane_ends(self, side=1.):
    """Pretend the camera sees the lane on that side ending."""
    self.scc._update_lane_ending = lambda md: None
    self.scc.lane_ending[side] = True

  # ----- a vehicle that signals -----

  def test_signal_from_a_car_we_are_about_to_go_by_is_passed(self):
    # we come up on it 23 mph faster; it starts signalling when we are 9 m behind it: we are past before it would move
    dist = self.drive(2.0, dist=30., right=3.6, v_obj=V_EGO - 10.5)
    assert 8. < dist < 10. and not self.triggers
    self.drive(0.4, dist=dist, right=3.6, v_obj=V_EGO - 10.5, signal=LEFT)
    assert not self.triggers and self.scc.passes > 0 and self.scc.output_v_target == V_CRUISE_UNSET

  def test_signal_from_a_car_well_ahead_is_waited_for(self):
    self.drive(3., dist=45., right=3.6, v_obj=V_EGO - 6., signal=LEFT)
    assert self.triggers == {TRIGGER_BLINKER} and self.scc.passes == 0

  def test_signal_from_a_wide_car_we_catch_slowly_is_waited_for(self):
    # also a little wide of the lane line, but only 9 mph slower and 45 m ahead: easing off works, so we do
    self.drive(3., dist=45., right=4.2, v_obj=V_EGO - 4., signal=LEFT)
    assert self.triggers == {TRIGGER_BLINKER} and self.scc.passes == 0

  def test_car_in_the_next_lane_is_never_waved_off_because_we_are_too_fast(self):
    # 20 m ahead, 22 mph slower, signalling in: the cruise cannot fix this, but it slows as much as it can
    self.drive(0.8, dist=20., right=3.6, v_obj=V_EGO - 10., signal=LEFT)
    assert self.triggers and self.scc.passes == 0

  def test_car_moving_in_from_the_next_lane_is_never_waved_off_because_we_are_too_fast(self):
    self.drive(2.5, dist=40., right=4.4, v_obj=V_EGO - 9., lat_speed=0.7)
    assert self.triggers & MOVING_IN and self.scc.passes == 0

  # ----- a vehicle that is already moving over -----

  def test_car_moving_in_ahead_with_room_at_our_speed_is_left_to_the_cars_own_cruise(self):
    # recorded 2026-11-26 08:11: 41 m ahead, 1 mph slower. We dropped 4 mph for it. Now: nothing.
    for dist, slower in ((41., 0.5), (60., 0.), (72., 0.4)):
      self.setup_method()
      self.drive(4., dist=dist, right=3.8, v_obj=V_EGO - slower, lat_speed=0.7, v_cruise=V_EGO + 6.)
      assert not self.triggers and self.scc.output_v_target == V_CRUISE_UNSET and not self.scc.ready, dist

  def test_car_moving_in_ahead_with_room_but_slower_gets_matched(self):
    self.drive(3., dist=60., right=3.8, v_obj=V_EGO - 3., lat_speed=0.7)
    assert self.triggers & MOVING_IN
    np.testing.assert_allclose(self.lowest, V_EGO - 3., atol=0.3)   # its speed, no lower

  def test_radar_ghosts_beside_the_lane_do_nothing(self):
    # recorded 2026-11-26 16:02: a wall the radar reports as objects doing 2 to 20 mph, 6 to 16 m ahead, drifting our way
    for v_ghost in (1., 4.5, 7., 9.):
      self.setup_method()
      self.drive(3., dist=14., right=8.1, v_obj=v_ghost, lat_speed=0.5, v_ego=29., v_cruise=31.3)
      assert not self.triggers and not self.scc.ready and self.scc.output_v_target == V_CRUISE_UNSET, v_ghost

  def test_getting_ready_leaves_room_to_speed_up(self):
    # a slower car beside us drifts our way: the set speed comes down, but stays 2.5 m/s above our speed
    dist, right, asked = 28., 8.0, []
    for _ in range(30):
      dist = self.drive(0.1, dist=dist, right=right, v_obj=V_EGO - 3., lat_speed=0.4, v_cruise=V_EGO + 6.)
      right -= 0.04
      if self.scc.ready:
        asked.append(self.scc.output_v_target)
    assert asked and all(abs(a - (V_EGO + 2.5)) < 1e-6 for a in asked)

  def test_no_getting_ready_for_a_car_at_our_speed(self):
    # recorded 2026-11-26 08:26: a car 16 m ahead on the right at our speed held the set speed down
    dist, right = 16., 7.6
    for _ in range(25):
      dist = self.drive(0.1, dist=dist, right=right, v_obj=V_EGO, lat_speed=0.5, v_cruise=V_EGO + 6.)
      right -= 0.05
      assert not self.scc.ready and self.scc.output_v_target == V_CRUISE_UNSET

  def test_car_moving_in_close_at_our_speed_still_gets_room(self):
    self.drive(4., dist=35., right=3.8, v_obj=V_EGO, lat_speed=0.7)   # 1.1 s ahead of us: too tight to call it room
    assert self.triggers & MOVING_IN
    assert self.lowest < V_EGO - 0.4

  def test_car_moving_in_close_and_slower_gets_room(self):
    self.drive(4., dist=30., right=3.8, v_obj=V_EGO - 4., lat_speed=0.7)
    assert self.triggers & MOVING_IN
    assert self.lowest < V_EGO - 4. - 0.5        # below its speed, so the gap opens

  def test_slow_car_drifting_in_from_two_lanes_over_while_we_fly_by_is_passed(self):
    # the on-ramp car of 2026-11-25 34:51: 24 m ahead, 21 mph slower, 7.8 m to the right and coming across fast
    self.drive(1.4, dist=32., right=9.0, v_obj=V_EGO - 9.4, lat_speed=2.5)   # it is still more than a lane away
    assert not self.triggers and self.scc.passes > 0

  def test_same_car_but_only_a_little_slower_is_waited_for(self):
    self.drive(1.4, dist=32., right=9.0, v_obj=V_EGO - 2., lat_speed=2.5)
    assert self.triggers & MOVING_IN and self.scc.passes == 0

  # ----- its lane is ending -----

  def test_lane_ending_and_we_cannot_get_by_in_time_lets_it_in(self):
    self.lane_ends()
    self.drive(3., dist=40., right=3.6, v_obj=V_EGO - 4.)
    assert self.triggers == {TRIGGER_LANE_ENDING}
    np.testing.assert_allclose(self.lowest, V_EGO - 4., atol=0.05)   # its speed: it comes in ahead of us

  def test_lane_ending_but_we_are_past_it_first_keeps_our_speed(self):
    self.lane_ends()
    self.drive(1.0, dist=12., right=3.6, v_obj=V_EGO - 9.)            # clear in under 3 s, its lane lasts longer than that
    assert not self.triggers and self.scc.passes > 0

  def test_no_lane_end_no_signal_no_move_means_nothing_happens(self):
    self.drive(3., dist=40., right=3.6, v_obj=V_EGO - 4.)
    assert not self.triggers and self.scc.passes == 0 and self.scc.state == MergeState.enabled

  def test_lane_ending_on_the_other_side_does_not_count(self):
    self.lane_ends(side=-1.)
    self.drive(3., dist=40., right=3.6, v_obj=V_EGO - 4.)
    assert not self.triggers

  def test_lane_ending_with_a_car_as_fast_as_us_does_nothing(self):
    self.lane_ends()
    self.drive(3., dist=40., right=3.6, v_obj=V_EGO)
    assert not self.triggers

  # ----- the sums -----

  def test_time_to_pass(self):
    assert self.scc._time_to_pass(20., 10.) == 3.0                    # 20 m to it + 10 m to be clear, at 10 m/s
    assert self.scc._time_to_pass(15., 1.) == float('inf')            # barely faster: that is not a pass
    assert self.scc._time_to_pass(15., -3.) == float('inf')

  def test_lane_end_time(self):
    self.scc.v_ego = 30.
    assert self.scc._lane_end_time(40., 25., 1.) == float('inf')
    self.scc.lane_ending[1.] = True
    np.testing.assert_allclose(self.scc._lane_end_time(40., 25., 1.), (110. - 40.) / 25.)   # camera looks 4 s ahead, capped at 110 m
    assert self.scc._lane_end_time(200., 25., 1.) == 0.
