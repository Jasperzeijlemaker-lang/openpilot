"""
Per-car scale for the proportional gain of the torque steering controller (version 0).

Ioniq 6: on gentle banked curves the steering rack sticks, then jumps. The controller answers the jump, the rack
sticks at the other side, and so on: a wobble of 1 to 5 degrees about once every 0.8 s, lasting several seconds
(recorded 2026-11-25 17:10 and 27:15, 2026-11-26 07:20 and 17:57). In each cycle the steering command swings about
+-0.2 around a holding value of only about 0.07, so it changes sign every time. How hard the command swings for a
given miss is the loop gain. Lowering it is the standard cure for a stick-slip cycle.

Per m/s2 of miss, the command changed by about 0.90 on Nov 25 (strength 2.5) and 1.00 on Nov 26 (strength 3.0 plus
friction 0.05, which adds gain near the centre). With strength 3.0, no friction term and this scale it is about 0.53.
The feed-forward for curves is untouched, so the car still turns in as before; only corrections are gentler.
"""

KP_SCALE = {
  "HYUNDAI_IONIQ_6": 0.6,
}
_MIN_SCALE, _MAX_SCALE = 0.4, 1.0


def kp_scale(car_fingerprint) -> float:
  """Scale for this car, 1.0 for every other car. Never raises and never leaves 0.4 to 1.0."""
  try:
    scale = float(KP_SCALE.get(str(car_fingerprint), 1.0))
    if not _MIN_SCALE <= scale <= _MAX_SCALE:   # also catches NaN
      return 1.0
    return scale
  except Exception:
    return 1.0
