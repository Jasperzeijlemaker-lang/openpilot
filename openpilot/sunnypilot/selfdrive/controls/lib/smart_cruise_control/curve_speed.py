"""
Curve speed for cars whose cruise is driven through the set-speed buttons.

The vision curve controller was written for cars where openpilot commands the acceleration itself. There it hands
out a speed that sits well below the real curve speed for a few seconds (target speed plus a deceleration times a
4 s horizon), and the planner turns that into a smooth braking profile. A button-driven cruise takes that number
literally as its set speed, so the car slows far more than the curve needs.

For those cars the set speed should simply be the speed that fits the curve.
"""
import math

from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V

VISION_A_LAT = 2.0  # m/s2. What the vision controller's own target speed is built on
BUTTON_A_LAT = 2.5  # m/s2 sideways force to aim for. Owner drove this ramp by hand at 2.1 to 2.7
_SCALE = math.sqrt(BUTTON_A_LAT / VISION_A_LAT)


def button_cruise_curve_speed(vision_active: bool, vision_output: float, vision_v_target: float) -> float:
  """Set speed for the curve ahead, or V_CRUISE_UNSET when the vision controller is not acting.
  Never lower than what the vision controller itself asks for, and never lower than its plain curve speed."""
  try:
    if not vision_active or not math.isfinite(vision_output) or vision_output >= V_CRUISE_UNSET:
      return vision_output if (isinstance(vision_output, (int, float)) and math.isfinite(vision_output)) else V_CRUISE_UNSET
    if not (math.isfinite(vision_v_target) and vision_v_target > 0.):
      return float(vision_output)
    return float(max(vision_output, max(vision_v_target, MIN_V) * _SCALE))
  except Exception:
    return V_CRUISE_UNSET
