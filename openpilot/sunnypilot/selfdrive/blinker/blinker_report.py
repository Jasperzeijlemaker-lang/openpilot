"""Pieces of Blinker Watch that need neither the camera nor messaging, so they can be tested off the device."""
import time

import numpy as np

from openpilot.common.swaglog import cloudlog

REPORT_EVERY = 1.0     # s between log lines for a vehicle that keeps flashing


def planes_from_buf(buf) -> tuple[np.ndarray, np.ndarray]:
  """Brightness plane and colour plane of an NV12 camera buffer, as views where possible (no big copies)."""
  data = np.frombuffer(buf.data, dtype=np.uint8)
  stride, height, width, uv_offset = int(buf.stride), int(buf.height), int(buf.width), int(buf.uv_offset)
  y = data[:stride * height].reshape(height, stride)[:, :width]
  uv = data[uv_offset:uv_offset + stride * (height // 2)].reshape(height // 2, stride)[:, :width]
  return y, uv


class Reporter:
  """Turns per-frame detections into a few log lines: one when a vehicle starts flashing, then one a second."""

  def __init__(self, log=None, clock=time.monotonic):
    # error=True is not a fault here: it is what routes a log line into the small logs (qlog),
    # so the results can be read without uploading the big ones.
    self.log = log or (lambda **kw: cloudlog.event("blinker_watch", error=True, **kw))
    self.clock = clock
    self.active: dict[int, tuple[int, float]] = {}   # slot -> (side, last report time)
    self.count = 0

  def update(self, detections) -> None:
    now = self.clock()
    current = set()
    for d in detections:
      current.add(d.slot)
      prev = self.active.get(d.slot)
      if prev is None or prev[0] != d.side or now - prev[1] >= REPORT_EVERY:
        self.active[d.slot] = (d.side, now)
        self.count += prev is None or prev[0] != d.side
        self.log(what="flashing", slot=int(d.slot), side="left" if d.side < 0 else "right", confidence=round(float(d.confidence), 2),
                 period=round(float(d.period), 2), dist=round(float(d.dist), 1), y_left=round(float(d.y_left), 1))
    for slot in [s for s in self.active if s not in current]:
      del self.active[slot]
