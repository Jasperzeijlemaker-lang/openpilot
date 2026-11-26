"""Runs the device loop of Blinker Watch against stand-ins for the camera, the messages and the settings store."""
import sys
import types

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.blinker import beacon, blinkerd, blinker_report
from openpilot.sunnypilot.selfdrive.blinker.sim import Scene, RPY, CAM_HEIGHT
from openpilot.sunnypilot.selfdrive.blinker.tests.test_blinker import FakeBuf


class Done(BaseException):
  pass


class FakeLog:
  def __init__(self):
    self.events, self.errors, self.notes = [], [], []

  def event(self, name, **kw):
    self.events.append(kw)

  def exception(self, msg):
    self.errors.append(msg)


def install(monkeypatch, frames, enabled=True, connect_fails=0, recv_errors=(), frame_ids=None, bad_calibration=False):
  """frames: list of (y, uv, tracks). Returns the fake log."""
  log = FakeLog()
  state = dict(i=-1, connects=0)

  class Client:
    def __init__(self, name, stream, conflate):
      assert name == "camerad" and stream == "narrow" and conflate is True
      self.connected = False
      self.frame_id = 0

    def is_connected(self):
      return self.connected

    def connect(self, blocking):
      assert blocking is False
      state['connects'] += 1
      self.connected = state['connects'] > connect_fails
      return self.connected

    def recv(self):
      state['i'] += 1
      i = state['i']
      if i >= len(frames):
        raise Done()
      if i in recv_errors:
        raise RuntimeError("camera hiccup")
      if frames[i] is None:
        return None
      self.frame_id = frame_ids[i] if frame_ids else i + 1
      return FakeBuf(frames[i][0], frames[i][1])

  class SubMaster:
    def __init__(self, services):
      assert services == ['carState', 'carStateSP', 'extrinsicsCalibration']
      self.seen = dict.fromkeys(services, False)
      self.data = {}

    def update(self, timeout):
      f = frames[state['i']]
      tr = [types.SimpleNamespace(slot=s, dRel=d, yRel=y, vRel=v) for s, d, y, v in f[2]]
      self.data = dict(carState=types.SimpleNamespace(vEgo=28.), carStateSP=types.SimpleNamespace(mergeRadarTracks=tr),
                       extrinsicsCalibration=types.SimpleNamespace(rpyCalib=[] if bad_calibration else list(RPY), height=[CAM_HEIGHT]))
      self.seen = dict.fromkeys(self.seen, True)

    def __getitem__(self, k):
      return self.data[k]

  class Params:
    def __init__(self, path=""):
      assert path in ("", "/dev/shm/params")
      self.memory_only = bool(path)

    def get_bool(self, key):
      assert key == "BlinkerWatch" and not self.memory_only
      return enabled

    def put(self, key, value):
      assert self.memory_only and key == "BlinkerWatchSeen"      # never written to the device's storage
      log.notes.append(value)

  mods = {
    "msgq": types.ModuleType("msgq"),
    "msgq.visionipc": types.SimpleNamespace(VisionIpcClient=Client),
    "openpilot.cereal.visionipc": types.SimpleNamespace(VisionStreamType=types.SimpleNamespace(VISION_STREAM_NARROW_ROAD="narrow")),
    "openpilot.cereal.messaging": types.SimpleNamespace(SubMaster=SubMaster),
    "openpilot.common.params": types.SimpleNamespace(Params=Params),
  }
  for k, v in mods.items():
    monkeypatch.setitem(sys.modules, k, v)
  import openpilot.cereal as cereal_pkg
  monkeypatch.setattr(cereal_pkg, "messaging", mods["openpilot.cereal.messaging"], raising=False)
  monkeypatch.setattr(cereal_pkg, "visionipc", mods["openpilot.cereal.visionipc"], raising=False)
  monkeypatch.setattr(blinkerd, "cloudlog", log)
  monkeypatch.setattr(blinker_report, "cloudlog", log)
  monkeypatch.setattr(blinkerd.time, "sleep", lambda s: None)
  return log


def scene_frames(n=110, **kw):
  sc = Scene(np.random.default_rng(4), dist=20., y_left=-3.4, **kw)
  out = []
  for _ in range(n):
    y, uv, tracks = sc.frame()
    out.append((y.copy(), uv.copy(), tracks))
  return out


def flashing(log):
  return [e for e in log.events if e.get("what") == "flashing"]


def test_loop_reports_a_flashing_vehicle(monkeypatch):
  log = install(monkeypatch, scene_frames(signal=1))
  with pytest.raises(Done):
    blinkerd.run()
  hits = flashing(log)
  assert log.events[0] == {"what": "started", "error": True} and all(e["error"] is True for e in log.events) and 2 <= len(hits) <= 6
  assert all(h["side"] == "right" and h["slot"] == 7 and 13 < h["dist"] < 21 for h in hits) and not log.errors
  # the notes for Merge Assist: about 5 a second while it flashes, each readable and pointing at the same vehicle
  assert 10 <= len(log.notes) <= 25
  for note in log.notes:
    stamp = float(note.split("|")[1])
    seen = beacon.decode(note, stamp, 1.)
    assert list(seen) == [7] and seen[7][0] == 1 and 13 < seen[7][1] < 21


def test_loop_is_quiet_without_a_signal(monkeypatch):
  log = install(monkeypatch, scene_frames(brake=True))
  with pytest.raises(Done):
    blinkerd.run()
  assert not flashing(log) and not log.errors and not log.notes


def test_switched_off_means_no_looking(monkeypatch):
  log = install(monkeypatch, scene_frames(signal=1), enabled=False)
  with pytest.raises(Done):
    blinkerd.run()
  assert not flashing(log)


def test_camera_errors_empty_frames_and_late_connection_do_not_stop_it(monkeypatch):
  frames = scene_frames(signal=1, n=160)
  frames[3] = None
  log = install(monkeypatch, frames, connect_fails=3, recv_errors={10, 11})
  with pytest.raises(Done):
    blinkerd.run()
  assert flashing(log) and len(log.errors) == 2


def test_missing_frames_throw_the_history_away(monkeypatch):
  frames = scene_frames(signal=1, n=110)
  ids = [1 + i + (i // 25) * 10 for i in range(110)]     # a hole of 10 frames every 1.25 s: never enough clean history
  log = install(monkeypatch, frames, frame_ids=ids)
  with pytest.raises(Done):
    blinkerd.run()
  assert not flashing(log)


def test_no_calibration_means_no_guessing(monkeypatch):
  log = install(monkeypatch, scene_frames(signal=1), bad_calibration=True)
  with pytest.raises(Done):
    blinkerd.run()
  assert not flashing(log) and not log.errors


def test_main_idles_instead_of_crashing_when_it_cannot_start(monkeypatch):
  log = FakeLog()
  monkeypatch.setattr(blinkerd, "cloudlog", log)
  monkeypatch.setattr(blinkerd, "run", lambda: (_ for _ in ()).throw(ImportError("no camera library")))
  naps = []

  def nap(s):
    naps.append(s)
    if len(naps) >= 3:
      raise Done()
  monkeypatch.setattr(blinkerd.time, "sleep", nap)
  with pytest.raises(Done):
    blinkerd.main()
  assert log.errors == ["blinker_watch could not start"] and naps == [60., 60., 60.]
