"""
Exit Watch (watch only).

Notices the moment when our lane splits (an exit or a fork), the vehicle ahead takes the other branch, and the
driving model's planned path starts to drift after it.

This class only observes. It has no output that any controller reads: it sets flags for the log and for an alert.
It cannot change steering or speed. Any error switches it off until restart.
"""
from collections import deque

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog

MIN_V = 20.  # m/s. Highway only
_SANE_SPEED = 70.
_NEAR_X = 8.  # m ahead where "our lane as it is now" is measured
_MIN_WIDTH, _MAX_WIDTH = 2.7, 4.3  # m. A normal lane right in front of us
_STAY_LINE_PROB = 0.5  # the lane line on the side we stay on must be seen clearly
_SPLIT_WIDEN = 0.8  # m. The lane is this much wider than it was a moment ago: it is splitting
_WIDTH_TAU = 3.  # s. How slowly the remembered normal lane width follows the measured one
_WIDTH_STEP = 0.3  # m. Only widths this close to the remembered one are learned
_WIDTH_LEARN_PROB = 0.7  # both lines must be this clear for the width to be learned
_WIDTH_MIN_TIME = 1.  # s of learning before the remembered width is trusted
_DRIFT_WINDOW = 2.  # s. The stay-side line is compared with where it was this long ago
_SPLIT_LINE_PROB = 0.3  # ...or the line on the leaving side has faded away
_OBJ_MIN_DIST, _OBJ_MAX_DIST = 12., 90.  # m
_OBJ_MIN_SPEED = 4.  # m/s over the ground
_IN_LANE = (0.2, 0.8)  # fraction of the lane width, measured from the line we stay next to
_LEFT_LANE = 0.95  # fraction: the vehicle is now out over where our far line would be
_OUT_MAX = 2.2  # fraction: further than this is not "just left our lane"
_IN_LANE_MEMORY = 4.  # s. It must have been in our lane this recently
_TRACK_JUMP_DIST = 4.  # m between frames: the radar reused the slot
_DRIFT_FOLLOW = 0.4  # m. Our lane, as the model draws it, has shifted this far toward the leaving vehicle
_ENTER_TIME = 0.3  # s
_HOLD_TIME = 2.  # s
_BLINKER_TIME = 4.  # s. With a blinker on, we are the one leaving


def _frames(seconds: float) -> int:
  return max(1, int(round(seconds / DT_MDL)))


def _finite(*values) -> bool:
  return all(np.isfinite(v) for v in values)


class _Track:
  def __init__(self):
    self.dist = 0.
    self.history: deque[tuple[bool, bool]] = deque(maxlen=_frames(_IN_LANE_MEMORY))
    self.updated = False


class ExitWatch:
  def __init__(self):
    self.fault = False
    self.split = False  # lane splits ahead and the vehicle ahead takes the other branch
    self.following = False  # ...and the lane the model draws for us shifts after it
    self.side = 0  # +1 the other branch goes right, -1 left, 0 none
    self.drift = 0.  # m, how far our lane has shifted toward that side
    self.lead_dist = 0.
    self.lane_widening = 0.  # m
    self._tracks: dict[int, _Track] = {}
    self._width_ref = 0.  # m, the normal width of our lane, learned while both lines are clear and steady
    self._width_frames = 0
    self._stay_history: deque[tuple[float, float]] = deque(maxlen=_frames(_DRIFT_WINDOW))  # (left, right) line at _NEAR_X
    self._enter = 0
    self._hold = 0
    self._blinker = 0
    self._last_frame_id: int | None = None

  def _clear(self) -> None:
    self.split = self.following = False
    self.side = 0
    self.drift = self.lead_dist = self.lane_widening = 0.

  def update(self, sm, v_ego: float) -> None:
    if self.fault:
      return
    try:
      self._update(sm, v_ego)
    except Exception:
      cloudlog.exception("exit_watch: unexpected error, switching off")
      self.fault = True
      self._clear()
      self._tracks = {}

  def _update(self, sm, v_ego: float) -> None:
    md = sm['modelV2']
    frame_id = int(md.frameId)
    if frame_id == self._last_frame_id:
      return  # nothing new
    gap = None if self._last_frame_id is None else frame_id - self._last_frame_id
    self._last_frame_id = frame_id
    if gap is None or gap <= 0 or gap > 5:
      self._tracks = {}
      self._stay_history.clear()
      self._width_frames = 0
      self._enter = 0

    CS = sm['carState']
    self._blinker = _frames(_BLINKER_TIME) if (CS.leftBlinker or CS.rightBlinker) else max(0, self._blinker - 1)
    self._hold = max(0, self._hold - 1)

    found = self._look(sm, md, v_ego) if (_finite(v_ego) and MIN_V <= v_ego < _SANE_SPEED and self._blinker == 0) else None
    if found is None:
      self._enter = 0
      if self._hold == 0:
        self._clear()
      return

    self._enter += 1
    if self._enter >= _frames(_ENTER_TIME):
      side, dist, widening, drift = found
      self.split = True
      self.side = side
      self.lead_dist = dist
      self.lane_widening = widening
      self.drift = max(self.drift, drift) if self.following else drift
      self.following = self.following or drift > _DRIFT_FOLLOW
      self._hold = _frames(_HOLD_TIME)

  def _look(self, sm, md, v_ego: float):
    lines, probs = md.laneLines, md.laneLineProbs
    if len(lines) < 4 or len(probs) < 4:
      return None
    lx = np.asarray(lines[1].x, dtype=float)
    left_y = np.asarray(lines[1].y, dtype=float)
    right_y = np.asarray(lines[2].y, dtype=float)
    path_x = np.asarray(md.position.x, dtype=float)
    path_y = np.asarray(md.position.y, dtype=float)
    if not (len(lx) > 1 and len(lx) == len(left_y) == len(right_y) and len(path_x) > 1 and len(path_x) == len(path_y)):
      return None
    if not (bool(np.all(np.isfinite(lx))) and bool(np.all(np.isfinite(left_y))) and bool(np.all(np.isfinite(right_y))) and
            bool(np.all(np.isfinite(path_x))) and bool(np.all(np.isfinite(path_y))) and
            bool(np.all(np.diff(lx) > 0.)) and bool(np.all(np.diff(path_x) > 0.))):
      return None
    left_prob, right_prob = float(probs[1]), float(probs[2])
    if not _finite(left_prob, right_prob):
      return None

    left_near, right_near = float(np.interp(_NEAR_X, lx, left_y)), float(np.interp(_NEAR_X, lx, right_y))
    width_now = right_near - left_near
    # Learn the normal lane width only while the lane is clear and steady. During a split it stays as it was.
    if _MIN_WIDTH < width_now < _MAX_WIDTH and min(left_prob, right_prob) > _WIDTH_LEARN_PROB:
      if self._width_frames == 0:
        self._width_ref = width_now
        self._width_frames = 1
      elif abs(width_now - self._width_ref) < _WIDTH_STEP:
        self._width_ref += (DT_MDL / (_WIDTH_TAU + DT_MDL)) * (width_now - self._width_ref)
        self._width_frames += 1
    past = self._stay_history[0] if len(self._stay_history) == self._stay_history.maxlen else None
    self._stay_history.append((left_near, right_near))
    if self._width_frames < _frames(_WIDTH_MIN_TIME) or not 0. < width_now < 12.:
      return None
    width = self._width_ref

    try:
      points = sm['carStateSP'].mergeRadarTracks
    except (KeyError, AttributeError):
      return None

    for track in self._tracks.values():
      track.updated = False

    best = None
    for point in points:
      slot, dist, y_right, v_rel = int(point.slot), float(point.dRel), -float(point.yRel), float(point.vRel)
      if not (_finite(dist, y_right, v_rel) and _OBJ_MIN_DIST < dist < _OBJ_MAX_DIST and abs(y_right) < 20. and
              _OBJ_MIN_SPEED < v_ego + v_rel < _SANE_SPEED):
        continue
      track = self._tracks.setdefault(slot, _Track())
      if track.history and abs(dist - track.dist) > _TRACK_JUMP_DIST:
        track.history.clear()
      track.dist = dist
      track.updated = True

      l_at = float(np.interp(dist, lx, left_y))
      r_at = float(np.interp(dist, lx, right_y))
      # Position as a fraction of our lane width, counted from the line we would stay next to.
      from_left = (y_right - l_at) / width   # for a branch going right we stay next to the left line
      from_right = (r_at - y_right) / width  # for a branch going left we stay next to the right line
      was_in = (any(h[0] for h in track.history), any(h[1] for h in track.history))
      track.history.append((_IN_LANE[0] < from_left < _IN_LANE[1], _IN_LANE[0] < from_right < _IN_LANE[1]))

      for i, (side, frac, stay_prob, leave_prob) in enumerate(((1, from_left, left_prob, right_prob),
                                                                (-1, from_right, right_prob, left_prob))):
        if not (was_in[i] and _LEFT_LANE < frac < _OUT_MAX and stay_prob > _STAY_LINE_PROB):
          continue
        widening = max(float(r_at - l_at), width_now) - width
        if not (widening > _SPLIT_WIDEN or leave_prob < _SPLIT_LINE_PROB):
          continue  # an ordinary lane change by the vehicle ahead: our lane is not splitting
        # Has the line we should stay next to moved toward the leaving vehicle? Then the model is redrawing
        # our lane over there, or the car is already drifting that way.
        stay_now = left_near if side == 1 else right_near
        stay_past = stay_now if past is None else (past[0] if side == 1 else past[1])
        drift = side * (stay_now - stay_past)
        if best is None or dist < best[1]:
          best = (side, dist, widening, drift)

    self._tracks = {slot: track for slot, track in self._tracks.items() if track.updated}
    return best
