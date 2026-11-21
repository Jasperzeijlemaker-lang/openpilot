"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.cereal import custom
from openpilot.sunnypilot.selfdrive.car.merge_radar import MergeRadar, decode_object, RADAR_START_ADDR, RADAR_BUS, STALE_NS

# Real messages recorded on a 2025 Ioniq 6. Each is the radar object that the stock cruise was following at that moment,
# with the stock cruise's own distance and relative speed for it (SCC_CONTROL), which were not used to build the decoder's output.
REAL = [
  # slot, message, stock cruise distance (m), stock cruise relative speed (m/s), decoded sideways position (m, + left)
  (7, 'eb3b815261ff3086d66101868004c00100000000d0020000', 46.8, 1.2, 1.1),
  (7, '7d3ec7d261ff300612b3021980f81f0000000000d0020000', 78.5, 0.2, 2.15),
  (7, 'b030dd9261ff30099ec1ff3a80f81f0100000000d0020000', 41.3, 0.5, -0.2),
]
EMPTY = (0, '8d9455c000000000000000000000000000000000d0020000')


def can_frame(nanos, msgs):
  return [(nanos, [(RADAR_START_ADDR + slot, bytes.fromhex(dat), bus) for slot, dat, bus in msgs])]


class TestDecode:
  def test_matches_stock_cruise(self):
    for slot, dat, dist, rel_speed, lateral in REAL:
      obj = decode_object(slot, bytes.fromhex(dat))
      assert obj is not None and obj.slot == slot
      assert abs(obj.d_rel - dist) < 0.5
      assert abs(obj.v_rel - rel_speed) < 0.3
      np.testing.assert_allclose(obj.y_rel, lateral, atol=1e-6)

  def test_empty_slot(self):
    assert decode_object(EMPTY[0], bytes.fromhex(EMPTY[1])) is None

  def test_bad_checksum_is_rejected(self):
    slot, dat = REAL[0][:2]
    raw = bytearray(bytes.fromhex(dat))
    raw[9] ^= 0x10  # one flipped bit in the distance
    assert decode_object(slot, bytes(raw)) is None

  def test_wrong_slot_is_rejected(self):
    # The checksum covers the message address, so a message read as another slot must fail.
    slot, dat = REAL[0][:2]
    assert decode_object(slot + 1, bytes.fromhex(dat)) is None

  def test_wrong_length(self):
    assert decode_object(0, b'\x00' * 8) is None
    assert decode_object(0, b'') is None


class TestMergeRadar:
  def test_collects_objects_and_fills_message(self):
    radar = MergeRadar()
    radar.update(can_frame(1_000_000_000, [(REAL[0][0], REAL[0][1], RADAR_BUS), EMPTY + (RADAR_BUS,)]))
    assert radar.seen and not radar.failed
    assert list(radar.objects) == [7]

    msg = custom.CarStateSP.new_message()
    radar.fill(msg)
    assert len(msg.mergeRadarTracks) == 1
    track = msg.mergeRadarTracks[0]
    assert track.slot == 7
    np.testing.assert_allclose([track.dRel, track.yRel, track.vRel], [47.0, 1.1, 1.34], atol=1e-3)

  def test_other_bus_is_ignored(self):
    radar = MergeRadar()
    radar.update(can_frame(1, [(REAL[0][0], REAL[0][1], RADAR_BUS + 1)]))
    assert not radar.objects and not radar.seen

  def test_object_that_goes_empty_is_dropped(self):
    radar = MergeRadar()
    radar.update(can_frame(1, [(7, REAL[0][1], RADAR_BUS)]))
    radar.update(can_frame(2, [(7, '00' * 24, RADAR_BUS)]))
    assert not radar.objects

  def test_stale_object_is_dropped(self):
    radar = MergeRadar()
    radar.update(can_frame(1, [(7, REAL[0][1], RADAR_BUS)]))
    radar.update([(1 + STALE_NS + 1, [(0x100, b'\x00' * 8, 1)])])
    assert not radar.objects

  def test_garbage_never_raises(self):
    radar = MergeRadar()
    radar.update(None)
    assert radar.failed and not radar.objects
    radar.update(can_frame(1, [(7, REAL[0][1], RADAR_BUS)]))  # stays off
    assert not radar.objects
    msg = custom.CarStateSP.new_message()
    radar.fill(msg)
    assert len(msg.mergeRadarTracks) == 0

  def test_no_objects_leaves_message_alone(self):
    msg = custom.CarStateSP.new_message()
    MergeRadar().fill(msg)
    assert len(msg.mergeRadarTracks) == 0
