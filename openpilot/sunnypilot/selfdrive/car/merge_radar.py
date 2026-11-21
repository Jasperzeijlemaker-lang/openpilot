"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Front radar object list for Hyundai CAN FD cars that put it on the ADAS bus (seen on a 2025 Ioniq 6).

The radar sends 32 messages, 0x3A5 to 0x3C4, 24 bytes each, 20 times a second. Each one describes one object.
The layout below was worked out from recorded drives by matching fields against the stock cruise's lead
distance and against how each object's distance changes over time:

  bytes 0-1   CRC16 (the usual Hyundai CAN FD checksum)
  byte  2     counter
  bit   64    12 bits  distance ahead, 0.1 m          (0 = empty slot)
  bit   76    12 bits  sideways position, signed, 0.05 m, positive = left
  bit   88    14 bits  speed relative to us, signed, 0.01 m/s

This file only reads. It never sends anything to the car, and nothing here feeds the normal lead tracking.
The objects are published for Smart Cruise Control - Merge and for the drive log.
"""
import math
from typing import NamedTuple

from opendbc.car.hyundai.hyundaicanfd import hkg_can_fd_checksum

RADAR_BUS = 0
RADAR_START_ADDR = 0x3A5
RADAR_MSG_COUNT = 32
RADAR_MSG_LEN = 24
STALE_NS = int(0.3e9)  # an object not refreshed for this long is dropped

MAX_DIST = 250.  # m
MAX_LATERAL = 60.  # m
MAX_REL_SPEED = 90.  # m/s


class RadarObject(NamedTuple):
  slot: int
  d_rel: float  # m ahead
  y_rel: float  # m, positive = left
  v_rel: float  # m/s, negative = we are catching up


def decode_object(slot: int, dat: bytes) -> RadarObject | None:
  """One radar message -> one object. None for an empty slot, a bad checksum or an implausible value."""
  if len(dat) != RADAR_MSG_LEN:
    return None
  if hkg_can_fd_checksum(RADAR_START_ADDR + slot, None, bytearray(dat)) != int.from_bytes(dat[:2], "little"):
    return None

  raw = int.from_bytes(dat, "little")
  dist = ((raw >> 64) & 0xFFF) * 0.1
  if dist <= 0.:
    return None
  lat = (raw >> 76) & 0xFFF
  if lat >= 0x800:
    lat -= 0x1000
  spd = (raw >> 88) & 0x3FFF
  if spd >= 0x2000:
    spd -= 0x4000
  y_rel, v_rel = lat * 0.05, spd * 0.01

  if not (math.isfinite(dist) and dist < MAX_DIST and abs(y_rel) < MAX_LATERAL and abs(v_rel) < MAX_REL_SPEED):
    return None
  return RadarObject(slot, dist, y_rel, v_rel)


class MergeRadar:
  """Keeps the latest object in each of the 32 radar slots. Any error switches it off for good."""

  def __init__(self):
    self.objects: dict[int, tuple[int, RadarObject]] = {}  # slot -> (time seen in ns, object)
    self.failed = False
    self.seen = False

  def update(self, can_list) -> None:
    if self.failed:
      return
    try:
      now = 0
      for nanos, msgs in can_list:
        now = max(now, nanos)
        for msg in msgs:
          address = msg[0]
          if RADAR_START_ADDR <= address < RADAR_START_ADDR + RADAR_MSG_COUNT and msg[2] == RADAR_BUS:
            self.seen = True
            slot = address - RADAR_START_ADDR
            obj = decode_object(slot, msg[1])
            if obj is None:
              self.objects.pop(slot, None)
            else:
              self.objects[slot] = (nanos, obj)
      if now and self.objects:
        for slot in [s for s, (seen_ns, _) in self.objects.items() if now - seen_ns > STALE_NS]:
          del self.objects[slot]
    except Exception:
      self.failed = True
      self.objects = {}

  def fill(self, car_state_sp) -> None:
    """Write the current objects into a carStateSP message that is about to be sent."""
    if self.failed or not self.objects:
      return
    try:
      objs = [obj for _, obj in sorted(self.objects.values(), key=lambda item: item[1].slot)]
      tracks = car_state_sp.init('mergeRadarTracks', len(objs))
      for track, obj in zip(tracks, objs, strict=True):
        track.slot = obj.slot
        track.dRel = obj.d_rel
        track.yRel = obj.y_rel
        track.vRel = obj.v_rel
    except Exception:
      self.failed = True
      self.objects = {}
