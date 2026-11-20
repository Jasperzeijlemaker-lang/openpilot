"""Synthetic unit tests for the offline light-pulse classifier."""
import unittest

import cv2
import numpy as np

from blinker_watch import PulseDetector, lamp_score_bgr

ROI = (8, 8, 24, 24)


def frame(color: str, on: bool, full_brightness: int = 0) -> np.ndarray:
  x = np.full((32, 32, 3), full_brightness, np.uint8)
  if on:
    bgr = (0, 155, 255) if color == "amber" else (0, 0, 255)
    cv2.rectangle(x, (11, 11), (21, 21), bgr, -1)
  return x


def series(color: str, f, duration: float = 6., fps: float = 12.) -> list:
  detector = PulseDetector(color)
  return [detector.update(i/fps, lamp_score_bgr(frame(color, f(i/fps)), ROI, color))
          for i in range(int(duration * fps))]


class BlinkWatchTests(unittest.TestCase):
  def test_amber_repeated_blinks_detected(self):
    results = series("amber", lambda t: (t % 1.) < 0.4)
    self.assertTrue(any(x.confirmed for x in results))

  def test_red_repeated_blinks_detected(self):
    results = series("red", lambda t: (t % 0.8) < 0.36)
    self.assertTrue(any(x.confirmed for x in results))

  def test_constant_red_brake_light_does_not_trigger(self):
    results = series("red", lambda t: True)
    self.assertFalse(any(x.candidate for x in results))

  def test_constant_amber_reflection_does_not_trigger(self):
    results = series("amber", lambda t: True)
    self.assertFalse(any(x.candidate for x in results))

  def test_single_flash_is_not_enough(self):
    results = series("amber", lambda t: 1.0 <= t < 1.4)
    self.assertFalse(any(x.candidate for x in results))

  def test_strobe_too_fast_is_rejected(self):
    results = series("red", lambda t: (t % 0.24) < 0.12)
    self.assertFalse(any(x.confirmed for x in results))

  def test_invalid_input_rejected(self):
    detector = PulseDetector("red")
    detector.update(0., 0.2)
    with self.assertRaises(ValueError):
      detector.update(0., 0.4)
    with self.assertRaises(ValueError):
      detector.update(1., float("nan"))

  def test_roi_out_of_bounds_rejected(self):
    with self.assertRaises(ValueError):
      lamp_score_bgr(frame("red", True), (0,0,100,100), "red")


if __name__ == "__main__":
  unittest.main()
