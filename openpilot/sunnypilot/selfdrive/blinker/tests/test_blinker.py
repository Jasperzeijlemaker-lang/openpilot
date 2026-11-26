import numpy as np

from openpilot.sunnypilot.selfdrive.blinker.blinker_core import BlinkerWatcher, SideSignal, warm_energy, box_for, CAM_W, CAM_H
from openpilot.sunnypilot.selfdrive.blinker.blinker_report import planes_from_buf, Reporter
from openpilot.sunnypilot.selfdrive.blinker.sim import Scene, run_scene, RED_LIT, RPY, CAM_HEIGHT


def rng(seed):
  return np.random.default_rng(seed)


def result(seed, **kw):
  sides, first = run_scene(BlinkerWatcher(), Scene(rng(seed), **kw))
  return sides, first


class TestSideSignal:
  def test_square_wave_on_one_side(self):
    for side in (-1, 1):
      s = SideSignal()
      for i in range(60):
        on = 40. if (i // 7) % 2 else 0.
        r = s.update(on if side < 0 else 0., on if side > 0 else 0.)
      assert r == side and 0.6 <= s.period <= 0.8

  def test_both_sides_together_is_not_a_signal(self):
    s = SideSignal()
    for i in range(60):
      on = 40. if (i // 7) % 2 else 0.
      assert s.update(on, on) == 0

  def test_both_sides_in_step_but_uneven_is_not_a_signal(self):
    # hazard lights with one lamp half hidden or dirtier than the other
    s = SideSignal()
    for i in range(80):
      on = (i // 7) % 2
      assert s.update(45. * on, 12. * on + (3. if i % 5 == 0 else 0.)) == 0

  def test_steady_or_single_blip_is_not_a_signal(self):
    s = SideSignal()
    for _i in range(60):
      assert s.update(30., 0.) == 0
    s = SideSignal()
    for i in range(60):
      assert s.update(50. if 20 <= i < 26 else 0., 0.) == 0

  def test_too_slow_or_too_fast_is_not_a_signal(self):
    for half in (2, 12, 16):   # 5 Hz, 0.8 Hz and 0.6 Hz
      s = SideSignal()
      for i in range(60):
        assert s.update(40. if (i // half) % 2 else 0., 0.) == 0

  def test_one_side_lighting_up_once_is_not_a_signal(self):
    # the sun reaching one corner of a car, or one lamp switching on and staying on
    for start in (15, 30, 45):
      s = SideSignal()
      for i in range(80):
        assert s.update(40. if i >= start else 0., 0.) == 0, (start, i)

  def test_bad_numbers_reset(self):
    s = SideSignal()
    for i in range(60):
      s.update(40. if (i // 7) % 2 else 0., 0.)
    assert s.update(float('nan'), 0.) == 0 and len(s.left) == 0


class TestWarmEnergy:
  def test_shapes_and_empties(self):
    assert warm_energy(np.zeros((0, 4), np.uint8), np.zeros((0, 4), np.uint8)) == 0.
    assert warm_energy(np.zeros((4, 4), np.uint8), np.zeros((4, 5), np.uint8)) == 0.

  def test_lit_lamp_beats_dark_paint(self):
    y = np.full((20, 20), 50, np.uint8); cr = np.full((20, 20), 139, np.uint8)
    dark = warm_energy(y, cr)
    y[8:12, 8:12] = 220; cr[8:12, 8:12] = 172
    assert dark == 0. and warm_energy(y, cr) > 20.


class TestBox:
  def test_far_is_smaller_and_left_is_left(self):
    near, far = box_for(15., 0., RPY, CAM_HEIGHT), box_for(45., 0., RPY, CAM_HEIGHT)
    assert near[2] - near[0] > 2.5 * (far[2] - far[0])
    assert box_for(25., 3.5, RPY, CAM_HEIGHT)[1] < box_for(25., 0., RPY, CAM_HEIGHT)[1] < box_for(25., -3.5, RPY, CAM_HEIGHT)[1]

  def test_unusable_positions(self):
    assert box_for(1., 0., RPY, CAM_HEIGHT) is None          # too close
    assert box_for(200., 0., RPY, CAM_HEIGHT) is None        # too small to see a lamp
    assert box_for(8., 9., RPY, CAM_HEIGHT) is None          # outside the picture


class TestSimulatedScenes:
  def test_left_and_right_signal_are_found_quickly(self):
    for seed, side in enumerate((-1, 1, -1, 1)):
      sides, first = result(seed, dist=22., y_left=-3.5 * side, signal=side)
      assert first is not None and first <= 2.6, (side, first)
      assert set(sides) <= {0, side}

  def test_red_turn_signal_while_braking(self):
    # many US cars flash the brake lamp as the turn signal; the other side stays lit
    sides, first = result(11, dist=20., signal=1, signal_color=RED_LIT, brake=True)
    assert first is not None and set(sides) <= {0, 1}

  def test_detection_rate_by_distance(self):
    for dist, need in ((12., 0.95), (25., 0.95), (40., 0.85)):
      hit = 0
      for seed in range(40):
        side = -1 if seed % 2 else 1
        sides, first = result(100 + seed, dist=dist, y_left=float(rng(seed).uniform(-4, 4)), signal=side, v_rel=float(rng(seed).uniform(-3, 1)),
                              rate_hz=float(rng(seed + 1).uniform(1.0, 2.0)))
        hit += first is not None and set(sides) <= {0, side}
      assert hit / 40 >= need, (dist, hit)

  def test_no_false_alarm_scenes(self):
    scenes = [
      dict(),                                             # nothing lit
      dict(brake=True),                                   # steady brake lights
      dict(brake=True, brake_pump_hz=1.5),                # driver tapping the brakes: both sides flash
      dict(hazards=True),                                 # hazard lights: both sides flash
      dict(flicker_hz=1.5),                               # shadows of posts / bridge joints at signal rate
      dict(glints=0.15),                                  # sun glints on chrome, one frame at a time
      dict(backlit=True, brake=True, flicker_hz=1.2),
      dict(drop_frames=0.1, brake=True),
    ]
    for k, kw in enumerate(scenes):
      for seed in range(12):
        r = rng(1000 + 17 * k + seed)
        sides, first = result(2000 + 31 * k + seed, dist=float(r.uniform(9, 45)), y_left=float(r.uniform(-5, 5)), **kw)
        assert first is None, (kw, seed, first)

  def test_signal_survives_dropped_frames_and_wobble(self):
    hit = 0
    for seed in range(20):
      _, first = result(300 + seed, dist=20., signal=1, drop_frames=0.03, jitter=0.10)
      hit += first is not None
    assert hit >= 16

  def test_reused_radar_slot_starts_fresh(self):
    w = BlinkerWatcher()
    run_scene(w, Scene(rng(5), dist=20., signal=1), seconds=3.)
    assert 7 in w.signals and len(w.signals[7].left) > 20
    sc = Scene(rng(6), dist=45.)                     # the same slot now reports a different car far away
    y, uv, tracks = sc.frame()
    assert not w.process(y, uv, tracks, RPY, CAM_HEIGHT, 28.)
    assert len(w.signals[7].left) == 1

  def test_vehicle_hidden_behind_a_nearer_one_is_not_blamed(self):
    w = BlinkerWatcher(); sc = Scene(rng(9), dist=18., signal=1, v_rel=0.); slots = set()
    for _ in range(100):
      y, uv, tracks = sc.frame()
      tracks = [(3, 40., -1.3, 0.)] + tracks         # a second radar object further up the road, seen right behind the flashing lamp
      slots |= {d.slot for d in w.process(y, uv, tracks, RPY, CAM_HEIGHT, 28.)}
    assert slots == {7}

  def test_garbage_inputs_never_raise(self):
    w = BlinkerWatcher()
    y = np.zeros((CAM_H, CAM_W), np.uint8); uv = np.full((CAM_H // 2, CAM_W), 128, np.uint8)
    bad = float('nan')
    for tracks in ([(1, bad, 0., 0.)], [(1, 20., bad, 0.)], [(1, 20., 0., bad)], [(1, -5., 0., 0.)], [(1, 1e9, 0., 0.)], []):
      assert w.process(y, uv, tracks, RPY, CAM_HEIGHT, 28.) == []
    assert w.process(y, uv, [(1, 20., 0., 0.)], (bad, 0., 0.), CAM_HEIGHT, 28.) == []
    assert w.process(y, uv, [(1, 20., 0., 0.)], RPY, bad, 28.) == []
    assert w.process(y, uv, [(1, 20., 0., 0.)], (0., 0.), CAM_HEIGHT, 28.) == []
    assert w.process(y, uv, [(1, 20., 0., 0.)], RPY, CAM_HEIGHT, bad) == []


class FakeBuf:
  """Same layout as the device's camera buffer: padded rows, colour plane after an aligned brightness plane."""

  def __init__(self, y, uv, stride=1408):
    h, w = y.shape
    self.width, self.height, self.stride = w, h, stride
    y_rows = (h + 15) // 16 * 16
    self.uv_offset = stride * y_rows
    data = np.zeros(self.uv_offset + stride * ((h // 2 + 15) // 16 * 16), np.uint8)
    data[:stride * h].reshape(h, stride)[:, :w] = y
    data[self.uv_offset:self.uv_offset + stride * (h // 2)].reshape(h // 2, stride)[:, :w] = uv
    self.data = memoryview(data)


class TestDeviceGlue:
  def test_planes_from_buf_round_trip(self):
    r = rng(3)
    y = r.integers(0, 255, (CAM_H, CAM_W), dtype=np.uint8); uv = r.integers(0, 255, (CAM_H // 2, CAM_W), dtype=np.uint8)
    y2, uv2 = planes_from_buf(FakeBuf(y, uv))
    assert np.array_equal(y, y2) and np.array_equal(uv, uv2)

  def test_whole_chain_through_a_device_style_buffer(self):
    w = BlinkerWatcher(); sc = Scene(rng(8), dist=20., y_left=-3.4, signal=-1); found = 0
    for _ in range(100):
      y, uv, tracks = sc.frame()
      yb, uvb = planes_from_buf(FakeBuf(y, uv))
      found += bool(w.process(yb, uvb, tracks, RPY, CAM_HEIGHT, 28.))
    assert found > 10

  def test_reporter_logs_start_then_once_a_second(self):
    lines = []; now = [0.]
    rep = Reporter(log=lambda **kw: lines.append(kw), clock=lambda: now[0])
    det = type("D", (), dict(slot=3, side=1, confidence=0.8, period=0.7, dist=20., y_left=-3.))()
    for i in range(50):                  # 2.5 s of continuous flashing at 20 Hz
      now[0] = i * 0.05
      rep.update([det])
    assert len(lines) == 3 and lines[0]["side"] == "right" and rep.count == 1
    rep.update([])
    assert not rep.active
