#!/usr/bin/env python3
"""
Blinker Watch.

Looks at the road camera, at the spots where the car's radar reports other vehicles, and notes when one of them is
flashing a turn signal. It writes what it sees to the log, and leaves a short note for Merge Assist (see beacon.py). Merge Assist decides
what to do with it, inside its own limits. This process itself never touches speed or steering.

It runs as its own process and is listed as optional, so if it stops for any reason the rest of the system and
the ability to engage are not affected.
"""
import time

from openpilot.common.swaglog import cloudlog

STATS_EVERY = 60.0     # s between "still running" lines
MAX_FRAME_GAP = 2      # frames; more than this missing and the history is thrown away
IDLE_SLEEP = 0.05
BEACON_EVERY = 4       # frames between notes to the merge controller (5 a second)


def run() -> None:
  # imported here so that a missing piece is caught by main() below, not a crash that restarts forever
  import openpilot.cereal.messaging as messaging
  from openpilot.cereal.visionipc import VisionStreamType
  from msgq.visionipc import VisionIpcClient
  from openpilot.common.params import Params
  from openpilot.sunnypilot.selfdrive.blinker.blinker_core import BlinkerWatcher
  from openpilot.sunnypilot.selfdrive.blinker.blinker_report import Reporter, planes_from_buf
  from openpilot.sunnypilot.selfdrive.blinker import beacon

  params = Params()
  try:
    mem_params = Params(beacon.MEM_PARAMS_PATH)   # memory only: nothing is written to the device's storage
  except Exception:
    mem_params = None                             # keep watching and logging even if the note cannot be left
    cloudlog.exception("blinker_watch: no memory store")
  watcher = BlinkerWatcher()
  reporter = Reporter()
  sm = messaging.SubMaster(['carState', 'carStateSP', 'extrinsicsCalibration'])
  client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD, True)
  cloudlog.event("blinker_watch", error=True, what="started")   # error=True: see blinker_report.py

  last_frame_id = None
  frames = 0
  spent = 0.
  errors = 0
  last_stats = time.monotonic()
  enabled = params.get_bool("BlinkerWatch")

  while True:
    try:
      if not client.is_connected():
        if not client.connect(False):
          time.sleep(0.2)
          continue
        watcher.reset()
      buf = client.recv()
      if buf is None:
        time.sleep(IDLE_SLEEP)
        continue

      t0 = time.monotonic()
      sm.update(0)
      frame_id = int(client.frame_id)
      if last_frame_id is not None and not 0 < frame_id - last_frame_id <= MAX_FRAME_GAP:
        watcher.reset()
      last_frame_id = frame_id

      if enabled and sm.seen['carStateSP'] and sm.seen['extrinsicsCalibration'] and sm.seen['carState']:
        cal = sm['extrinsicsCalibration']
        rpy = list(cal.rpyCalib)
        height = float(cal.height[0]) if len(cal.height) else 1.22
        if len(rpy) == 3:
          tracks = [(t.slot, t.dRel, t.yRel, t.vRel) for t in sm['carStateSP'].mergeRadarTracks]
          y, uv = planes_from_buf(buf)
          detections = watcher.process(y, uv, tracks, rpy, height, float(sm['carState'].vEgo))
          reporter.update(detections)
          # tell the merge controller, a few times a second. It forgets by itself when the lines stop coming.
          if detections and mem_params is not None and frames % BEACON_EVERY == 0:
            mem_params.put(beacon.KEY, beacon.encode(detections, time.monotonic()))
      frames += 1
      spent += time.monotonic() - t0

      now = time.monotonic()
      if now - last_stats > STATS_EVERY:
        enabled = params.get_bool("BlinkerWatch")
        cloudlog.event("blinker_watch", error=True, what="stats", frames=frames, ms_per_frame=round(1000. * spent / max(frames, 1), 2),
                       vehicles=watcher.tracked, flashing_seen=reporter.count, errors=errors, enabled=enabled)
        frames, spent, last_stats = 0, 0., now
    except Exception:
      # Never let this helper die in a loop of restarts. Note it, wait, carry on.
      errors += 1
      if errors <= 5:
        cloudlog.exception("blinker_watch error")
      watcher.reset()
      time.sleep(1.0)


def main() -> None:
  try:
    run()
  except Exception:
    cloudlog.exception("blinker_watch could not start")
  # Stay alive but idle, so the manager does not keep restarting a helper that cannot work.
  while True:
    time.sleep(60.)


if __name__ == "__main__":
  main()
