import random

from openpilot.sunnypilot.system.remote_update import RemoteUpdate, REQUEST_PARAM, STATUS_PARAM, \
  NO_RESPONSE_TIME, MAX_UPDATE_TIME, SETTLE_TIME


class FakeParams:
  def __init__(self, **kw):
    self.d = dict(kw)
    self.reboots = 0
    self.fail_put = False

  def get_bool(self, k):
    return bool(self.d.get(k, False))

  def get(self, k):
    return self.d.get(k)

  def put(self, k, v):
    if self.fail_put:
      raise OSError("disk full")
    self.d[k] = v

  def put_bool(self, k, v):
    if self.fail_put:
      raise OSError("disk full")
    if k == "DoReboot" and v:
      # the single most important rule: never ask for a restart with the car on, or with the request still set
      assert self.d.get("IsOffroad"), "restart requested while the car is on"
      assert not self.d.get(REQUEST_PARAM), "restart requested with the request still set"
      self.reboots += 1
    self.d[k] = v


class Rig:
  def __init__(self, updater_found=True, **params):
    self.p = FakeParams(**params)
    self.t = 0.
    self.signals = 0
    self.updater_found = updater_found
    self.ru = RemoteUpdate(self.p, signal_fn=self._signal, clock=lambda: self.t)

  def _signal(self):
    self.signals += 1
    return self.updater_found

  def run(self, seconds, dt=2.):
    for _ in range(int(seconds / dt)):
      self.ru.step()
      self.t += dt

  @property
  def requested(self):
    return self.p.get_bool(REQUEST_PARAM)

  @property
  def status(self):
    return self.p.get(STATUS_PARAM) or ""


class TestRemoteUpdate:
  def test_nothing_happens_without_request(self):
    r = Rig(IsOffroad=True, UpdateAvailable=True)
    r.run(600)
    assert r.signals == 0 and r.p.reboots == 0 and r.status == ""

  def test_car_on_is_refused(self):
    r = Rig(IsOffroad=False, UpdateAvailable=True, **{REQUEST_PARAM: True})
    r.run(60)
    assert r.p.reboots == 0 and r.signals == 0 and not r.requested
    assert "Car is on" in r.status

  def test_downloaded_update_installs_once(self):
    r = Rig(IsOffroad=True, UpdateAvailable=True, **{REQUEST_PARAM: True})
    r.run(600)
    assert r.p.reboots == 1 and not r.requested and r.signals == 0

  def test_full_download_then_restart(self):
    r = Rig(IsOffroad=True, UpdaterState="idle", **{REQUEST_PARAM: True})
    r.run(4)
    assert r.signals == 1 and r.requested and r.p.reboots == 0
    r.p.d["UpdaterState"] = "downloading..."
    r.run(120)
    assert r.p.reboots == 0 and "downloading" in r.status
    r.p.d["UpdaterState"] = "idle"
    r.run(4)
    r.p.d["UpdateAvailable"] = True
    r.run(10)
    assert r.p.reboots == 1 and not r.requested and r.signals == 1

  def test_up_to_date(self):
    r = Rig(IsOffroad=True, UpdaterState="idle", UpdateFailedCount=0, **{REQUEST_PARAM: True})
    r.run(4)
    r.p.d["UpdaterState"] = "checking..."
    r.run(10)
    r.p.d["UpdaterState"] = "idle"
    r.run(SETTLE_TIME + 10)
    assert r.p.reboots == 0 and not r.requested and "newest" in r.status

  def test_fast_fetch_seen_through_fetch_time(self):
    r = Rig(IsOffroad=True, UpdaterState="idle", UpdaterLastFetchTime="a", **{REQUEST_PARAM: True})
    r.run(4)
    r.p.d["UpdaterLastFetchTime"] = "b"
    r.run(SETTLE_TIME + 10)
    assert not r.requested and "newest" in r.status and r.p.reboots == 0

  def test_download_failure(self):
    r = Rig(IsOffroad=True, UpdaterState="idle", **{REQUEST_PARAM: True})
    r.run(4)
    r.p.d["UpdaterState"] = "downloading..."
    r.run(10)
    r.p.d.update(UpdaterState="idle", UpdateFailedCount=2)
    r.run(SETTLE_TIME + 10)
    assert r.p.reboots == 0 and not r.requested and "failed" in r.status

  def test_car_started_during_download(self):
    r = Rig(IsOffroad=True, UpdaterState="idle", **{REQUEST_PARAM: True})
    r.run(4)
    r.p.d["UpdaterState"] = "downloading..."
    r.run(20)
    r.p.d.update(IsOffroad=False, UpdateAvailable=True)
    r.run(60)
    assert r.p.reboots == 0 and not r.requested

  def test_updates_disabled(self):
    r = Rig(IsOffroad=True, DisableUpdates=True, **{REQUEST_PARAM: True})
    r.run(20)
    assert r.signals == 0 and r.p.reboots == 0 and not r.requested and "Disable Updates" in r.status

  def test_updater_missing(self):
    r = Rig(updater_found=False, IsOffroad=True, **{REQUEST_PARAM: True})
    r.run(20)
    assert r.signals == 1 and not r.requested and r.p.reboots == 0

  def test_updater_never_answers(self):
    r = Rig(IsOffroad=True, UpdaterState="idle", **{REQUEST_PARAM: True})
    r.run(NO_RESPONSE_TIME + 20)
    assert not r.requested and r.p.reboots == 0 and r.signals == 1

  def test_download_times_out(self):
    r = Rig(IsOffroad=True, UpdaterState="downloading...", **{REQUEST_PARAM: True})
    r.run(MAX_UPDATE_TIME + 20)
    assert not r.requested and r.p.reboots == 0 and "too long" in r.status

  def test_user_cancels(self):
    r = Rig(IsOffroad=True, UpdaterState="idle", **{REQUEST_PARAM: True})
    r.run(4)
    r.p.d[REQUEST_PARAM] = False
    r.run(4)
    r.p.d["UpdateAvailable"] = True
    r.run(60)
    assert r.p.reboots == 0

  def test_storage_errors_do_not_crash_or_restart(self):
    r = Rig(IsOffroad=True, UpdateAvailable=True, **{REQUEST_PARAM: True})
    r.p.fail_put = True
    r.run(60)
    assert r.p.reboots == 0

  def test_fuzz_never_restarts_with_car_on_or_twice(self):
    # FakeParams asserts on every restart request that the car is off and the request was cleared
    rng = random.Random(7)
    states = ["idle", "checking...", "downloading...", "finalizing update...", "", None]
    for _ in range(300):
      r = Rig(updater_found=rng.random() > 0.1, IsOffroad=rng.random() > 0.3)
      for _ in range(200):
        if rng.random() < 0.05:
          r.p.d[REQUEST_PARAM] = rng.random() > 0.3
        if rng.random() < 0.05:
          r.p.d["IsOffroad"] = rng.random() > 0.3
        if rng.random() < 0.05:
          r.p.d["UpdateAvailable"] = rng.random() > 0.5
        if rng.random() < 0.1:
          r.p.d["UpdaterState"] = rng.choice(states)
        if rng.random() < 0.02:
          r.p.d["UpdaterLastFetchTime"] = rng.random()
        if rng.random() < 0.02:
          r.p.d["DisableUpdates"] = rng.random() > 0.5
        r.p.fail_put = rng.random() < 0.02
        before = r.p.reboots
        r.ru.step()
        r.t += rng.choice([0.5, 2., 30., 400.])
        if r.p.reboots > before:
          # a real device restarts here; the request must already be off
          assert not r.requested
          break
