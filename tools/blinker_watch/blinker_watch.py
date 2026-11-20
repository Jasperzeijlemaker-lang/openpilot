"""Offline blink-pattern detector for manually selected road-camera lamp ROIs.

Research only. NO connection to cruise, steering, braking or openpilot controls.
A detection is NOT proof of merge intent.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass(frozen=True)
class BlinkResult:
  confirmed: bool
  candidate: bool
  color: str
  score: float
  modulation: float
  periods: tuple[float, ...]
  transition_count: int


class PulseDetector:
  """Find repeated ON/OFF transitions in a time series, not single flashes.

  Timestamp must be monotonic. A fresh confirmation needs at least two pulses,
  with 0.45-1.6 seconds between leading edges, and an OFF period in between.
  """
  def __init__(self, color: str, history_seconds: float = 4.5,
               min_modulation: float = 0.24):
    if color not in ("amber", "red"):
      raise ValueError("color must be amber or red")
    self.color = color
    self.history_seconds = history_seconds
    self.min_modulation = min_modulation
    self.history: Deque[tuple[float, float]] = deque()

  def update(self, timestamp: float, score: float) -> BlinkResult:
    if not (np.isfinite(timestamp) and np.isfinite(score) and 0 <= score <= 1.0):
      raise ValueError("invalid blink sample")
    if self.history and timestamp <= self.history[-1][0]:
      raise ValueError("non-increasing timestamp")
    self.history.append((float(timestamp), float(score)))
    while self.history and timestamp - self.history[0][0] > self.history_seconds:
      self.history.popleft()
    times, values = zip(*self.history)
    if len(values) < 8:
      return BlinkResult(False, False, self.color, float(score), 0., (), 0)
    v = np.asarray(values)
    lo, hi = np.quantile(v, [0.20, 0.80])
    modulation = float(hi - lo)
    if modulation < self.min_modulation or hi < 0.42:
      return BlinkResult(False, False, self.color, float(score), modulation, (), 0)

    # Suppress one-frame sensor spikes; use a threshold halfway between OFF and ON.
    smooth = np.array([np.median(v[max(0, i-1):min(len(v), i+2)]) for i in range(len(v))])
    states = smooth > (lo + 0.5 * modulation)
    rises = []
    falls = []
    for i in range(1, len(states)):
      if not states[i-1] and states[i]:
        rises.append(float(times[i]))
      if states[i-1] and not states[i]:
        falls.append(float(times[i]))
    periods = tuple(round(b-a, 3) for a, b in zip(rises, rises[1:]))
    last_period_ok = bool(periods and 0.45 <= periods[-1] <= 1.60)
    # A fall between the last two rises prevents two successive brightness spikes
    # from being interpreted as a blinker.
    candidate = bool(last_period_ok and len(falls) >= 1 and len(rises) >= 2 and
                     any(rises[-2] < f < rises[-1] for f in falls))
    # Confirmation at 2 repeated periods, with bounded variability.
    confirmed = bool(candidate and len(periods) >= 2 and 0.45 <= periods[-2] <= 1.60 and
                     abs(periods[-1] - periods[-2]) <= 0.30 and
                     timestamp - rises[-1] <= 0.85)
    return BlinkResult(confirmed, candidate, self.color, float(score), modulation,
                       periods, len(rises) + len(falls))


def lamp_score_bgr(frame_bgr: np.ndarray, roi: tuple[int, int, int, int], color: str) -> float:
  """Return 0..1 warm-light activity for a *tight* lamp bounding box.

  Requires OpenCV but not any AI model. Assumes the ROI tracks the SAME lamp.
  This module does NOT detect vehicles, locate lamps or track moving objects.
  """
  import cv2
  x1, y1, x2, y2 = roi
  if min(x1, y1) < 0 or x2 > frame_bgr.shape[1] or y2 > frame_bgr.shape[0] or x2 <= x1 or y2 <= y1:
    raise ValueError("ROI is out of frame bounds")
  hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  if color == "amber":
    mask = (h >= 5) & (h <= 30) & (s >= 85) & (v >= 75)
  elif color == "red":
    mask = ((h <= 11) | (h >= 170)) & (s >= 85) & (v >= 75)
  else:
    raise ValueError("color must be amber or red")
  strength = np.where(mask, v.astype(np.float32) / 255., 0.).reshape(-1)
  # Top 10% reduces sensitivity to how tightly a moving light fills the crop,
  # but also makes it easier for a specular reflection to cause a false pulse.
  top_k = max(3, int(0.10 * len(strength)))
  return float(np.mean(np.partition(strength, -top_k)[-top_k:]))
