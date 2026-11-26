import math
import random

from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.curve_speed import button_cruise_curve_speed, BUTTON_A_LAT, VISION_A_LAT

MPH = CV.MPH_TO_MS


class TestButtonCruiseCurveSpeed:
  def test_inactive_passes_through(self):
    assert button_cruise_curve_speed(False, V_CRUISE_UNSET, 20.) == V_CRUISE_UNSET
    assert button_cruise_curve_speed(True, V_CRUISE_UNSET, 20.) == V_CRUISE_UNSET

  def test_recorded_ramp_entry(self):
    # Recorded Nov 25, 20:11: entering a 185 m radius ramp. The vision controller asked for 34 mph
    # (its curve speed of 43 minus a 4 s braking allowance). The driver took the ramp at 46 to 50 mph.
    v_target = math.sqrt(VISION_A_LAT * 185.)
    out = button_cruise_curve_speed(True, 34 * MPH, v_target)
    assert 46 * MPH < out < 50 * MPH

  def test_matches_the_sideways_force_we_aim_for(self):
    for radius in (120., 185., 300., 600.):
      out = button_cruise_curve_speed(True, 5., math.sqrt(VISION_A_LAT * radius))
      assert abs(out ** 2 / radius - BUTTON_A_LAT) < 1e-6

  def test_never_below_what_vision_asks(self):
    # leaving a curve: vision already allows more than the plain curve speed
    assert button_cruise_curve_speed(True, 30., 20.) == 30.

  def test_never_below_min_speed(self):
    assert button_cruise_curve_speed(True, 1., 0.5) >= MIN_V

  def test_bad_numbers(self):
    for bad in (float('nan'), float('inf'), -1., 0.):
      assert button_cruise_curve_speed(True, 15., bad) == 15.
    assert button_cruise_curve_speed(True, float('nan'), 20.) == V_CRUISE_UNSET
    assert button_cruise_curve_speed(True, None, 20.) == V_CRUISE_UNSET

  def test_fuzz_only_ever_raises_the_target(self):
    rng = random.Random(1)
    for _ in range(5000):
      out_v, vt = rng.uniform(0, 60), rng.uniform(-5, 60)
      out = button_cruise_curve_speed(True, out_v, vt)
      assert math.isfinite(out) and out >= out_v
      assert out <= max(out_v, max(vt, MIN_V) * 1.2)
