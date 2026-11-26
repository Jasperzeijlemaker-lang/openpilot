"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Smart Cruise Control - Merge (SCC-Merge)

Watches the driving model's lead outputs for a vehicle that sits in the lane next to ours
and is about to come into our path (an on-ramp merge or a cut-in). Four things can trigger it:
measured sideways motion, the model predicting lane entry, a slower vehicle hugging our lane line,
or a slower vehicle in a lane that ends ahead. By default only the
right side is watched; the "both sides" setting adds the left. While that is true,
it lowers the cruise target so the merging vehicle has room to slot in ahead of us.

It only ever lowers the target. Like the other Smart Cruise Control controllers, the result is
picked up either by openpilot longitudinal control, or by ICBM, which presses the car's own
cruise buttons (the only option on cars such as the Hyundai Ioniq 6, where the stock radar
cruise stays in charge of gas and brake).

All thresholds below are first estimates. Use "test only" mode and the logged
longitudinalPlanSP.smartCruiseControl.merge fields to tune them before trusting them.

Failsafes, in the order they apply:
  1. Any exception inside this controller is caught. The controller then switches itself off
     until the next restart (fault latch) and the rest of the planner keeps running.
  2. Every input is checked for NaN/inf and for a plausible range before it is used.
  3. A repeated or late model frame is never treated as new information.
  4. The output is either "no request" or a finite speed that is at or below the set speed,
     at most the configured drop below the speed we had when the event began, and never
     below an absolute floor.
  5. One event lasts at most _MAX_EVENT_TIME. An event that had to be cut off this way locks
     the feature out for _TIMEOUT_LOCKOUT_TIME.
  6. After any event the road must be clear of candidates for _REARM_TIME before a new one may start.
  7. Gas pedal, cruise off, our own turn signal, low speed, or the toggle going off all cancel at once.
"""
import math
from collections import deque

import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

MergeState = custom.LongitudinalPlanSP.SmartCruiseControl.MergeState

ACTIVE_STATES = (MergeState.slowing, MergeState.holding)
ENABLED_STATES = (MergeState.enabled, MergeState.overriding, *ACTIVE_STATES)

# +1: watch the right side (on-ramps in right-hand traffic). -1: watch the left side.
# Lateral values follow the model frame: positive y is to the right.
# The "both sides" setting watches the other side as well.
MERGE_SIDE = 1.

MIN_V_HIGHWAY = 20.  # m/s (~45 mph). Below this we are not on a highway; do nothing.
MAX_V_CRUISE = 50.  # m/s (~112 mph). A set speed above this means "cruise speed not set"; do nothing.
MIN_V_TARGET = 15.6  # m/s (~35 mph). Never ask for less than this, whatever the settings say.

# --- Which model leads count as "a vehicle merging into our lane" ---
_LEAD_PROB_TH = 0.3  # model confidence that this vehicle is or will become our lead
_IN_PATH_OFFSET = 1.2  # m. Closer to our path center than this: already in our lane, normal cruise handles it
_MAX_OFFSET = 4.6  # m. Farther than this: more than one lane away, ignore
_MIN_DIST = 2.  # m
_MAX_TIME_GAP = 5.  # s. Ignore vehicles farther ahead than this many seconds of travel
_MIN_MAX_DIST = 30.  # m. Floor for the distance limit above
_MAX_FASTER = 2.  # m/s. A vehicle pulling away from us faster than this needs no room

_CLOSING_SPEED_TH = 0.2  # m/s. Measured sideways speed toward our path that counts as merging
_MAX_CLOSING_SPEED = 3.  # m/s. No real vehicle changes lanes faster than this; treat as a glitch
_PRED_HORIZON_IDX = 1  # index into lead trajectory: the model's prediction 2 s ahead
_PRED_ENTRY_MIN_MOVE = 0.4  # m. Predicted sideways move needed for the "model predicts lane entry" trigger

# Trigger 3: a clearly slower vehicle that is already close to our lane line. No sideways motion needed.
_ENCROACH_OFFSET = 2.7  # m from our path center. A vehicle centered in the next lane sits at about 3.6 m
_ENCROACH_SLOWER = 3.  # m/s. It must be at least this much slower than us

# Trigger 4: the lane next to us ends ahead (on-ramp, lane drop). Whoever is in it has to come over.
_LANE_END_SLOWER = 1.  # m/s. The vehicle in the ending lane must be at least this much slower than us
_LANE_END_NEAR_X = 8.  # m ahead. Where we measure the next lane's width "here"
_LANE_END_FAR_T = 4.  # s of travel ahead. Where we measure it "ahead"
_LANE_END_FAR_X_LIMITS = (50., 110.)  # m
_LANE_END_NEAR_WIDTH = (2.5, 6.)  # m. Here, there must be about one lane between our lane line and the road edge
_LANE_END_FAR_WIDTH = 1.6  # m. Ahead, less than this is left of it
_LANE_END_MIN_SHRINK = 1.5  # m
_LANE_END_LINE_PROB = 0.3  # model confidence in our own lane line on that side
_LANE_END_EDGE_STD = 1.  # m. Model uncertainty of the road edge on that side must be below this
_LANE_END_TIME = 0.5  # s. The lane must look like it ends for this long

TRIGGER_NONE, TRIGGER_MEASURED, TRIGGER_PREDICTED, TRIGGER_ENCROACHING, TRIGGER_LANE_ENDING = range(5)
TRIGGER_RADAR_CONVERGING, TRIGGER_RADAR_SLOW, TRIGGER_RADAR_FOLLOW, TRIGGER_RADAR_EARLY = 5, 6, 7, 8

# --- The car's own radar (when it reports its object list, see sunnypilot/selfdrive/car/merge_radar.py) ---
# Unlike the camera model, the radar reports vehicles in the lanes beside us. Two things are acted on:
#   converging: a vehicle beside us is moving sideways toward our path           -> make room, match its speed
#   slow:       vehicles beside us are much slower than us (a queue, an on-ramp) -> defensive: limit how fast we pass them
_RADAR_MIN_OBJ_SPEED = 4.  # m/s. Slower than this over the ground: a sign, a barrier, a parked car. Ignore
_RADAR_MIN_DIST = 4.  # m
_RADAR_MAX_DIST = 160.  # m
_RADAR_IN_PATH_OFFSET = 1.3  # m. Closer to our path than this: in our lane, the stock cruise handles it
_RADAR_MAX_OFFSET = 8.5  # m. About two lanes to the side. An on-ramp usually starts that far out
_RADAR_CONVERGE_MAX_OFFSET = 4.5  # m. A sideways move only counts for a vehicle in the lane next to ours
_RADAR_CONVERGE_MAX_DIST = 90.  # m. Beyond this the radar's sideways reading is too noisy to trust a move
_RADAR_CONVERGE_SPEED = (0.25, 3.)  # m/s toward our path, right now
_RADAR_CONVERGE_WINDOW = 1.5  # s. ...and it must have come this much closer over this long:
_RADAR_CONVERGE_MOVE = 0.75  # m
_RADAR_CONVERGE_AVG = 5  # readings averaged at each end of the window, so one jittery reading cannot fake a move
_RADAR_CONVERGE_MAX_FASTER = 0.5  # m/s. A vehicle faster than us needs no room
_RADAR_CONFIRM_TIME = 0.4  # s. A radar object must be tracked this long before it counts
_RADAR_PASS_DELTA = 15. * CV.MPH_TO_MS  # Defensive mode: never pass vehicles beside us more than this much faster
_RADAR_SLOW_TTC = 10.  # s. Defensive mode starts when we would reach the slow vehicle within this time
_RADAR_SLOW_MIN_DELTA = 2.  # m/s over the pass limit before defensive mode reacts at all
_RADAR_TRACK_JUMP_DIST = 4.  # m between frames: the radar reused the slot for another object
# Early: a vehicle further out that is coming across quickly and would reach our lane within a few seconds.
# The evidence is weaker than for a vehicle right next to us, so the response is gentler.
_RADAR_EARLY_WINDOW = 1.  # s
_RADAR_EARLY_MOVE = 1.2  # m toward our path within that window. An ordinary lane change is slower than this
_RADAR_EARLY_TIME_TO_LANE = 4.  # s until it reaches our lane at its current sideways speed
_RADAR_EARLY_DELTA = 0.  # match its speed. The cruise slows so gently that a softer first step only wastes time
# Get ready: this car can only slow down by pressing the cruise 'speed down' button, and the set speed often sits
# well above the speed we are really driving (traffic). Pressing it down to our speed takes seconds and slows nothing.
# So when a vehicle beside us starts drifting our way, do that part in advance. It does not slow the car.
_READY_MOVE = 0.35  # m toward our path within the early window
_READY_MAX_DIST = 80.  # m
_READY_MAX_FASTER = 2.  # m/s
_READY_HOLD_TIME = 2.  # s
_READY_MARGIN = 1.  # m/s above our speed: never below it, so getting ready cannot slow us
# Brake warning: the cruise cannot brake. Tell the driver when the gap is closing faster than it can handle.
_BRAKE_WARN_MIN_CLOSING = 2.  # m/s
_BRAKE_WARN_HEADWAY = 0.8  # s of following distance we want to keep to the vehicle coming in
_BRAKE_WARN_DECEL = 1.0  # m/s2 needed to keep that distance. The cruise manages about half of this
_BRAKE_WARN_MAX_OFFSET = 4.  # m. It must be next to our lane or entering it
_BRAKE_WARN_HOLD_TIME = 1.5  # s
_RADAR_EARLY_MAX_DIST = 60.  # m. Further out the radar's sideways reading is too jumpy to act on a short move
# Follow-through: once a vehicle made us slow down, keep the lower speed until it is in our lane or clearly not coming.
_RADAR_FOLLOW_TIME = 8.  # s after the last real trigger. A vehicle that just drives next to us is let go after this
_RADAR_FOLLOW_MAX_OFFSET = 5.  # m
_RADAR_FOLLOW_MAX_FASTER = 2.  # m/s. It is pulling away; no room needed
_RADAR_FOLLOW_MAX_AWAY = 0.3  # m/s sideways away from our path: it changed its mind
_DEBUG_SLOTS = 3

_CLOSING_FILTER_TAU = 0.4  # s. Low-pass on the measured sideways speed
_TRACK_JUMP_DIST = 6.  # m. Jump in distance between frames: the model switched to another vehicle
_TRACK_JUMP_OFFSET = 1.  # m. Same, sideways
_MAX_FRAME_GAP = 3  # model frames. A longer gap than this: forget what we were tracking

# --- Sanity limits on raw inputs. Anything outside is ignored. ---
_SANE_DIST = 250.  # m
_SANE_OFFSET = 30.  # m
_SANE_SPEED = 70.  # m/s
_SANE_ACCEL = 5.  # m/s^2

# --- Timing ---
_ENTER_TIME = 0.25  # s. A candidate must persist this long before we slow down
_LOST_TIME = 0.5  # s. Candidate gone for this long: stop tracking it, start the hold timer
_HOLD_TIME = 3.  # s. Keep the reduced speed this long after the vehicle is gone or has merged
_MAX_EVENT_TIME = 40.  # s. Hard cap on one slowdown, guards against a stuck false detection
_COOLDOWN_TIME = 3.  # s. Quiet time after an event before a new one may start
_REARM_TIME = 1.  # s. After an event, no candidate may be seen for this long before a new event
_TIMEOUT_LOCKOUT_TIME = 30.  # s. Quiet time after an event that hit _MAX_EVENT_TIME
_BLINKER_SUPPRESS_TIME = 3.  # s. Ignore everything this long after our own turn signal (we are changing lanes)
_FAULT_ALERT_TIME = 5.  # s. How long the fault alert is requested

# --- How much to slow ---
_SPEED_MARGIN = 0.5  # m/s. Aim this far below the merging vehicle's speed
_GAP_EXTRA_V = [3., 1.5, 0.]  # m/s. Extra slowdown to open a gap while the vehicle is close
_GAP_EXTRA_BP = [0.8, 1.5, 2.5]  # s. Time gap breakpoints for the above
_DEFAULT_MAX_DROP_MPH = 30
_MAX_DROP_LIMITS_MPH = (5, 40)


def _FRAMES(t: float) -> int:
  return int(round(t / DT_MDL))


def _finite(*values: float) -> bool:
  return all(math.isfinite(v) for v in values)


class MergeCandidate:
  def __init__(self, dist: float, offset: float, closing_speed: float, v_lead: float, trigger: int = TRIGGER_NONE, slot: int = -1):
    self.slot = slot  # radar slot of the vehicle, -1 for camera candidates
    self.dist = dist
    self.offset = offset
    self.closing_speed = closing_speed
    self.v_lead = v_lead
    self.trigger = trigger

  def wanted_speed(self, v_ego: float) -> float:
    if self.trigger == TRIGGER_RADAR_SLOW:
      # Defensive: keep passing, but not much faster than the vehicles beside us.
      return self.v_lead + _RADAR_PASS_DELTA
    if self.trigger == TRIGGER_RADAR_EARLY:
      # Proactive: take most of the speed difference out before it arrives.
      return self.v_lead + _RADAR_EARLY_DELTA
    # Make room: a little under its speed, lower still while it is close so the gap grows.
    time_gap = self.dist / max(v_ego, 0.1)
    return self.v_lead - _SPEED_MARGIN - float(np.interp(time_gap, _GAP_EXTRA_BP, _GAP_EXTRA_V))


class LeadTrack:
  """Follows one model lead slot and measures how fast it moves sideways toward our path."""

  def __init__(self):
    self.reset()

  def reset(self) -> None:
    self.valid = False
    self.dist = 0.
    self.offset = 0.
    self.closing_speed = 0.

  def update(self, dist: float, offset: float, dt: float) -> None:
    # The model may switch this slot to a different vehicle. Do not treat that as motion.
    if not self.valid or dt <= 0. or abs(dist - self.dist) > _TRACK_JUMP_DIST or abs(offset - self.offset) > _TRACK_JUMP_OFFSET:
      self.valid = True
      self.closing_speed = 0.
    else:
      # Positive when the vehicle moves toward our path center.
      raw = (abs(self.offset) - abs(offset)) / dt
      if abs(raw) > _MAX_CLOSING_SPEED * 4.:
        # One-frame spike. Start over instead of letting it leak into the filter.
        self.closing_speed = 0.
      else:
        alpha = dt / (_CLOSING_FILTER_TAU + dt)
        self.closing_speed += alpha * (raw - self.closing_speed)

    self.dist = dist
    self.offset = offset


class RadarTrack:
  """One radar slot: how long we have seen the same object and how fast it moves sideways toward our path."""

  def __init__(self):
    self.frames = 0
    self.dist = 0.
    self.offset = 0.
    self.closing_speed = 0.
    self.updated = False
    self.history: deque[float] = deque(maxlen=_FRAMES(_RADAR_CONVERGE_WINDOW) + 1)  # |offset|, one per model frame

  def update(self, dist: float, offset: float, dt: float) -> None:
    same = self.frames > 0 and dt > 0. and abs(dist - self.dist) < _RADAR_TRACK_JUMP_DIST and abs(offset - self.offset) < _TRACK_JUMP_OFFSET
    if not same:
      self.frames = 1
      self.closing_speed = 0.
      self.history.clear()
    else:
      self.frames += 1
      raw = (abs(self.offset) - abs(offset)) / dt
      alpha = dt / (_CLOSING_FILTER_TAU + dt)
      self.closing_speed += alpha * (raw - self.closing_speed)
    self.dist = dist
    self.offset = offset
    self.history.append(abs(offset))
    self.updated = True

  @property
  def moved_closer(self) -> float:
    """How much closer to our path it is now than a full window ago. 0 until we have watched it that long."""
    if len(self.history) < self.history.maxlen:
      return 0.
    values = list(self.history)
    n = _RADAR_CONVERGE_AVG
    return sum(values[:n]) / n - sum(values[-n:]) / n

  @property
  def moved_closer_early(self) -> float:
    """The same over the shorter early window. 0 until we have watched it that long."""
    count = _FRAMES(_RADAR_EARLY_WINDOW) + 1
    if len(self.history) < count:
      return 0.
    values = list(self.history)[-count:]
    n = min(_RADAR_CONVERGE_AVG, 3)
    return sum(values[:n]) / n - sum(values[-n:]) / n


class SmartCruiseControlMerge:
  v_target: float = V_CRUISE_UNSET
  a_target: float = 0.
  v_ego: float = 0.
  a_ego: float = 0.
  output_v_target: float = V_CRUISE_UNSET
  output_a_target: float = 0.

  def __init__(self):
    self.params = Params()
    self.frame = -1
    self.long_enabled = False
    self.long_override = False
    self.is_enabled = False
    self.is_active = False
    self.v_cruise_setpoint = 0.
    self.v_ref = 0.

    self.enabled = False
    self.test_only = True
    self.both_sides = False
    self.max_drop = _DEFAULT_MAX_DROP_MPH * CV.MPH_TO_MS

    self.state = MergeState.disabled
    self.tracks: list[LeadTrack] = []
    self.candidate: MergeCandidate | None = None
    self.detected = False
    self.fault = False
    self.fault_frames = 0

    # Kept only for the log, so a drive can be studied afterwards. Never used for control.
    self.debug_prob = [0.] * _DEBUG_SLOTS
    self.debug_dist = [0.] * _DEBUG_SLOTS
    self.debug_offset = [0.] * _DEBUG_SLOTS
    self.debug_speed = [0.] * _DEBUG_SLOTS
    self.lane_ending = {1.: False, -1.: False}  # +1 right, -1 left
    self.radar_tracks: dict[int, RadarTrack] = {}
    self.radar_objects = 0
    self.follow_slot = -1
    self.follow_side = 1.
    self.follow_frames = 0
    self.ready_frames = 0
    self.ready = False  # the set speed is being brought down to our own speed, nothing more
    self.brake_warning_frames = 0
    self.brake_warning = False
    self.event_v_lead = 0.
    self.event_make_room = False
    self.lane_ending_frames = {1.: 0, -1.: 0}

    self.last_model_frame_id: int | None = None
    self.candidate_frames = 0
    self.lost_frames = 0
    self.clear_frames = 0
    self.armed = True
    self.hold_frames = 0
    self.event_frames = 0
    self.cooldown_frames = 0
    self.blinker_frames = 0

    try:
      self._read_params()
    except Exception:
      self._set_fault()

  # ----- settings -----

  def _read_params(self) -> None:
    self.enabled = self.params.get_bool("SmartCruiseControlMerge")
    self.test_only = self.params.get_bool("SmartCruiseControlMergeTestOnly")
    self.both_sides = self.params.get_bool("SmartCruiseControlMergeBothSides")
    try:
      max_drop_mph = float(self.params.get("SmartCruiseControlMergeMaxDrop", return_default=True))
    except (TypeError, ValueError):
      max_drop_mph = float(_DEFAULT_MAX_DROP_MPH)
    if not math.isfinite(max_drop_mph):
      max_drop_mph = float(_DEFAULT_MAX_DROP_MPH)
    self.max_drop = float(np.clip(max_drop_mph, *_MAX_DROP_LIMITS_MPH)) * CV.MPH_TO_MS

  def _update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self._read_params()

  # ----- detection -----

  def _reset_tracking(self) -> None:
    for track in self.tracks:
      track.reset()
    self.candidate = None
    self.candidate_frames = 0
    self.radar_tracks = {}
    self.radar_objects = 0
    self.follow_slot = -1
    self.follow_frames = 0
    self.ready_frames = 0
    self.brake_warning_frames = 0
    for side in self.lane_ending:
      self.lane_ending[side] = False
      self.lane_ending_frames[side] = 0

  def _update_lane_ending(self, md) -> None:
    """Does the lane next to ours run out ahead? Compare its width here with its width further on."""
    far_x = min(max(self.v_ego * _LANE_END_FAR_T, _LANE_END_FAR_X_LIMITS[0]), _LANE_END_FAR_X_LIMITS[1])
    for side, line_idx, edge_idx in ((1., 2, 1), (-1., 1, 0)):
      ending = False
      if (len(md.laneLines) > line_idx and len(md.roadEdges) > edge_idx and len(md.laneLineProbs) > line_idx and
          len(md.roadEdgeStds) > edge_idx):
        line, edge = md.laneLines[line_idx], md.roadEdges[edge_idx]
        lx, ly = np.asarray(line.x, dtype=float), np.asarray(line.y, dtype=float)
        ex, ey = np.asarray(edge.x, dtype=float), np.asarray(edge.y, dtype=float)
        line_prob, edge_std = float(md.laneLineProbs[line_idx]), float(md.roadEdgeStds[edge_idx])
        usable = (len(lx) > 1 and len(lx) == len(ly) and len(ex) > 1 and len(ex) == len(ey) and
                  bool(np.all(np.isfinite(lx))) and bool(np.all(np.isfinite(ly))) and bool(np.all(np.isfinite(ex))) and
                  bool(np.all(np.isfinite(ey))) and bool(np.all(np.diff(lx) > 0.)) and bool(np.all(np.diff(ex) > 0.)) and
                  lx[-1] >= far_x and ex[-1] >= far_x and _finite(line_prob, edge_std) and
                  line_prob >= _LANE_END_LINE_PROB and 0. <= edge_std < _LANE_END_EDGE_STD)
        if usable:
          near = side * float(np.interp(_LANE_END_NEAR_X, ex, ey) - np.interp(_LANE_END_NEAR_X, lx, ly))
          far = side * float(np.interp(far_x, ex, ey) - np.interp(far_x, lx, ly))
          ending = (_LANE_END_NEAR_WIDTH[0] < near < _LANE_END_NEAR_WIDTH[1] and far < _LANE_END_FAR_WIDTH and
                    near - far > _LANE_END_MIN_SHRINK)
      self.lane_ending_frames[side] = self.lane_ending_frames[side] + 1 if ending else 0
      self.lane_ending[side] = self.lane_ending_frames[side] >= _FRAMES(_LANE_END_TIME)

  def _model_dt(self, md) -> float:
    """Time since the last model frame we used. 0 means this frame is not new: use nothing from it."""
    frame_id = int(md.frameId)
    last = self.last_model_frame_id
    self.last_model_frame_id = frame_id
    if last is None:
      return 0.
    delta = frame_id - last
    if delta <= 0:
      return 0.
    if delta > _MAX_FRAME_GAP:
      self._reset_tracking()
      return 0.
    return delta * DT_MDL

  def _find_candidate(self, md, dt: float) -> MergeCandidate | None:
    leads = md.leadsV3
    path_x = np.asarray(md.position.x, dtype=float)
    path_y = np.asarray(md.position.y, dtype=float)
    has_path = (len(path_x) > 1 and len(path_x) == len(path_y) and bool(np.all(np.isfinite(path_x))) and
                bool(np.all(np.isfinite(path_y))) and bool(np.all(np.diff(path_x) > 0.)))
    if not has_path:
      self._reset_tracking()
      return None

    # The model reports lead speed on its own speed scale. Shift it onto the car's measured speed, like radard does.
    model_v_ego = float(md.velocity.x[0]) if len(md.velocity.x) > 0 else self.v_ego
    if not _finite(model_v_ego) or abs(model_v_ego - self.v_ego) > 5.:
      model_v_ego = self.v_ego

    while len(self.tracks) < len(leads):
      self.tracks.append(LeadTrack())

    max_dist = max(self.v_ego * _MAX_TIME_GAP, _MIN_MAX_DIST)
    best: MergeCandidate | None = None
    self._update_lane_ending(md)
    for i in range(_DEBUG_SLOTS):
      self.debug_prob[i] = self.debug_dist[i] = self.debug_offset[i] = self.debug_speed[i] = 0.

    for slot, (track, lead) in enumerate(zip(self.tracks, leads, strict=False)):
      if len(lead.x) <= _PRED_HORIZON_IDX or len(lead.y) <= _PRED_HORIZON_IDX or len(lead.v) < 1:
        track.reset()
        continue

      prob = float(lead.prob)
      dist = float(lead.x[0])
      lead_y = float(lead.y[0])
      pred_dist = float(lead.x[_PRED_HORIZON_IDX])
      pred_y = float(lead.y[_PRED_HORIZON_IDX])
      v_lead = self.v_ego + (float(lead.v[0]) - model_v_ego)
      sane = (_finite(prob, dist, lead_y, pred_dist, pred_y, v_lead) and 0. <= dist < _SANE_DIST and
              abs(lead_y) < _SANE_OFFSET and abs(pred_y) < _SANE_OFFSET and abs(pred_dist) < _SANE_DIST and
              -5. < v_lead < _SANE_SPEED)
      if not sane or not 0. <= prob <= 1.:
        track.reset()
        continue

      # Sideways distance from where *we* plan to be when we reach the vehicle's position.
      # Measuring against our path, not straight ahead, keeps this correct in curves.
      offset = lead_y - float(np.interp(dist, path_x, path_y))
      track.update(dist, offset, dt)
      if slot < _DEBUG_SLOTS:
        self.debug_prob[slot], self.debug_dist[slot], self.debug_offset[slot], self.debug_speed[slot] = prob, dist, offset, v_lead

      on_watched_side = self.both_sides or offset * MERGE_SIDE > 0.
      in_next_lane = _IN_PATH_OFFSET < abs(offset) < _MAX_OFFSET
      in_range = _MIN_DIST < dist < max_dist
      needs_room = v_lead < self.v_ego + _MAX_FASTER
      if not (prob >= _LEAD_PROB_TH and on_watched_side and in_next_lane and in_range and needs_room):
        continue

      # Trigger 1: we measure it moving sideways toward us, at a speed a real vehicle could have.
      measured = _CLOSING_SPEED_TH < track.closing_speed < _MAX_CLOSING_SPEED

      # Trigger 2: the model itself predicts it will be in our lane 2 s from now.
      pred_offset = pred_y - float(np.interp(pred_dist, path_x, path_y))
      predicted = abs(pred_offset) < _IN_PATH_OFFSET and (abs(offset) - abs(pred_offset)) > _PRED_ENTRY_MIN_MOVE

      # Trigger 3: clearly slower than us and already close to our lane line.
      encroaching = abs(offset) < _ENCROACH_OFFSET and v_lead < self.v_ego - _ENCROACH_SLOWER

      # Trigger 4: its lane ends ahead, so it has to come over, and it is slower than us.
      lane_ending = self.lane_ending[1. if offset > 0. else -1.] and v_lead < self.v_ego - _LANE_END_SLOWER

      trigger = (TRIGGER_MEASURED if measured else TRIGGER_PREDICTED if predicted else
                 TRIGGER_ENCROACHING if encroaching else TRIGGER_LANE_ENDING if lane_ending else TRIGGER_NONE)
      if trigger != TRIGGER_NONE and (best is None or dist < best.dist):
        best = MergeCandidate(dist, offset, track.closing_speed, v_lead, trigger)

    return best

  def _find_radar_candidate(self, sm: messaging.SubMaster, md, dt: float) -> MergeCandidate | None:
    """Vehicles beside our lane, from the car's own radar. Returns the one that asks for the lowest speed."""
    try:
      points = sm['carStateSP'].mergeRadarTracks
    except (KeyError, AttributeError):
      return None

    path_x = np.asarray(md.position.x, dtype=float)
    path_y = np.asarray(md.position.y, dtype=float)
    if not (len(path_x) > 1 and len(path_x) == len(path_y) and bool(np.all(np.isfinite(path_x))) and
            bool(np.all(np.isfinite(path_y))) and bool(np.all(np.diff(path_x) > 0.))):
      self.radar_tracks = {}
      self.radar_objects = 0
      return None

    for track in self.radar_tracks.values():
      track.updated = False

    max_dist = min(max(self.v_ego * _MAX_TIME_GAP, _MIN_MAX_DIST), _RADAR_MAX_DIST)
    best: MergeCandidate | None = None
    best_wanted = float('inf')
    beside = 0

    for point in points:
      slot, dist, y_left, v_rel = int(point.slot), float(point.dRel), float(point.yRel), float(point.vRel)
      v_obj = self.v_ego + v_rel
      if not (_finite(dist, y_left, v_rel) and _RADAR_MIN_DIST < dist < _SANE_DIST and abs(y_left) < _SANE_OFFSET and
              _RADAR_MIN_OBJ_SPEED < v_obj < _SANE_SPEED):
        continue

      # The radar measures straight ahead of the car. Our path may curve, so measure from the path. Positive = right.
      offset = -y_left - float(np.interp(dist, path_x, path_y))
      track = self.radar_tracks.setdefault(slot, RadarTrack())
      track.update(dist, offset, dt)

      if not _RADAR_IN_PATH_OFFSET < abs(offset) < _RADAR_MAX_OFFSET:
        continue
      if not (self.both_sides or offset * MERGE_SIDE > 0.):
        continue
      if track.frames < _FRAMES(_RADAR_CONFIRM_TIME):
        continue
      beside += 1
      if dist > max_dist:
        continue

      closing = self.v_ego - v_obj  # how fast we catch up with it
      converging = (abs(offset) < _RADAR_CONVERGE_MAX_OFFSET and dist < _RADAR_CONVERGE_MAX_DIST and
                    v_obj < self.v_ego + _RADAR_CONVERGE_MAX_FASTER and
                    _RADAR_CONVERGE_SPEED[0] < track.closing_speed < _RADAR_CONVERGE_SPEED[1] and
                    track.moved_closer > _RADAR_CONVERGE_MOVE)
      slow = closing > _RADAR_PASS_DELTA + _RADAR_SLOW_MIN_DELTA and dist / closing < _RADAR_SLOW_TTC
      early = (not converging and dist < _RADAR_EARLY_MAX_DIST and v_obj < self.v_ego + _RADAR_CONVERGE_MAX_FASTER and
               track.moved_closer_early > _RADAR_EARLY_MOVE and track.closing_speed > _RADAR_CONVERGE_SPEED[0] and
               (abs(offset) - _RADAR_IN_PATH_OFFSET) / track.closing_speed < _RADAR_EARLY_TIME_TO_LANE)
      # The vehicle we are already slowing for: stay with it while it is still next to us and not backing out.
      follow = (not converging and slot == self.follow_slot and self.follow_frames > 0 and self.state in ACTIVE_STATES and
                offset * self.follow_side > 0. and abs(offset) < _RADAR_FOLLOW_MAX_OFFSET and dist < _RADAR_CONVERGE_MAX_DIST and
                v_obj < self.v_ego + _RADAR_FOLLOW_MAX_FASTER and track.closing_speed > -_RADAR_FOLLOW_MAX_AWAY)

      if (dist < _READY_MAX_DIST and v_obj < self.v_ego + _READY_MAX_FASTER and track.moved_closer_early > _READY_MOVE and
          track.closing_speed > _RADAR_CONVERGE_SPEED[0]):
        self.ready_frames = _FRAMES(_READY_HOLD_TIME)

      if converging or slow or follow or early:
        trigger = (TRIGGER_RADAR_CONVERGING if converging else TRIGGER_RADAR_FOLLOW if follow else
                   TRIGGER_RADAR_EARLY if early else TRIGGER_RADAR_SLOW)
        candidate = MergeCandidate(dist, offset, track.closing_speed, v_obj, trigger, slot)
        wanted = candidate.wanted_speed(self.v_ego)
        if wanted < best_wanted:
          best, best_wanted = candidate, wanted

    self.radar_tracks = {slot: track for slot, track in self.radar_tracks.items() if track.updated}
    self.radar_objects = min(beside, 255)

    # Remember which vehicle is moving in. Only a real sideways move starts or renews the follow-through.
    self.follow_frames = max(0, self.follow_frames - 1)
    if best is not None and best.trigger == TRIGGER_RADAR_CONVERGING:
      self.follow_slot = best.slot
      self.follow_side = 1. if best.offset > 0. else -1.
      self.follow_frames = _FRAMES(_RADAR_FOLLOW_TIME)
    track = self.radar_tracks.get(self.follow_slot)
    if self.follow_frames == 0 or track is None or track.frames < _FRAMES(_RADAR_CONFIRM_TIME):
      # Time is up, the radar lost it, or the slot now holds a different object.
      self.follow_slot = -1
      self.follow_frames = 0
    return best

  def _update_calculations(self, sm: messaging.SubMaster) -> None:
    self.cooldown_frames = max(0, self.cooldown_frames - 1)
    self.blinker_frames = max(0, self.blinker_frames - 1)

    CS = sm['carState']
    if CS.leftBlinker or CS.rightBlinker:
      self.blinker_frames = _FRAMES(_BLINKER_SUPPRESS_TIME)

    # Use the car's measured speed. Fall back to the planner's value only if it is unusable.
    v_ego_measured = float(CS.vEgo)
    if _finite(v_ego_measured):
      self.v_ego = v_ego_measured

    md = sm['modelV2']
    dt = self._model_dt(md)

    inputs_ok = _finite(self.v_ego, self.a_ego, self.v_cruise_setpoint)
    cruise_set = inputs_ok and MIN_V_TARGET < self.v_cruise_setpoint < MAX_V_CRUISE
    watching = (self.long_enabled and not self.long_override and self.enabled and cruise_set and
                MIN_V_HIGHWAY <= self.v_ego < _SANE_SPEED and self.blinker_frames == 0)

    if watching and dt > 0.:
      camera = self._find_candidate(md, dt)
      radar = self._find_radar_candidate(sm, md, dt)
      # Whichever asks for the lower speed wins.
      if camera is not None and radar is not None:
        self.candidate = camera if camera.wanted_speed(self.v_ego) <= radar.wanted_speed(self.v_ego) else radar
      else:
        self.candidate = camera if camera is not None else radar
    else:
      if not watching:
        self._reset_tracking()
      for i in range(_DEBUG_SLOTS):
        self.debug_prob[i] = self.debug_dist[i] = self.debug_offset[i] = self.debug_speed[i] = 0.
      # Not watching, or no new model frame: we have learned nothing new, so there is no candidate.
      self.candidate = None

    if self.candidate is not None:
      self.candidate_frames += 1
      self.lost_frames = 0
      self.clear_frames = 0
    else:
      self.candidate_frames = 0
      self.lost_frames += 1
      self.clear_frames += 1

    if self.clear_frames >= _FRAMES(_REARM_TIME):
      self.armed = True

    self.detected = self.candidate_frames >= _FRAMES(_ENTER_TIME)

  # ----- state machine -----

  def _end_event(self, timed_out: bool = False) -> None:
    was_active = self.state in ACTIVE_STATES
    self.state = MergeState.enabled
    self.event_frames = 0
    self.hold_frames = 0
    self.event_make_room = False
    self.v_target = V_CRUISE_UNSET
    if was_active:
      # Only an event that had to be cut off needs the road to look clear again before the next one.
      if timed_out:
        self.armed = False
        self.clear_frames = 0
      self.cooldown_frames = max(self.cooldown_frames, _FRAMES(_TIMEOUT_LOCKOUT_TIME if timed_out else _COOLDOWN_TIME))

  def _update_state_machine(self) -> tuple[bool, bool]:
    # ENABLED, SLOWING, HOLDING, OVERRIDING
    if self.state != MergeState.disabled:
      # longitudinal and feature disable always have priority in a non-disabled state
      if not self.long_enabled or not self.enabled:
        self._end_event()
        self.state = MergeState.disabled
      elif self.long_override:
        # Driver is on the gas. Their call; drop the event.
        self._end_event()
        self.state = MergeState.overriding

      else:
        if self.state in ACTIVE_STATES:
          self.event_frames += 1

        # ENABLED
        if self.state == MergeState.enabled:
          if self.detected and self.armed and self.cooldown_frames == 0:
            self.state = MergeState.slowing
            self.event_frames = 0
            self.v_target = V_CRUISE_UNSET
            # Everything in this event is measured from the speed we have right now.
            self.v_ref = min(self.v_cruise_setpoint, self.v_ego)

        # OVERRIDING
        elif self.state == MergeState.overriding:
          if not self.long_override:
            self.state = MergeState.enabled

        # SLOWING
        elif self.state == MergeState.slowing:
          if self.event_frames >= _FRAMES(_MAX_EVENT_TIME):
            self._end_event(timed_out=True)
          elif self.lost_frames >= _FRAMES(_LOST_TIME):
            # Vehicle is gone, or it finished merging and is now a normal lead.
            self.state = MergeState.holding
            self.hold_frames = _FRAMES(_HOLD_TIME)
            # The gap is made. Stop aiming below its speed, or we fall back and then have to speed up again.
            ease = self.event_v_lead - _SPEED_MARGIN
            if self.event_make_room and _finite(ease, self.v_target) and self.v_target < ease:
              self.v_target = min(ease, self.v_cruise_setpoint) if _finite(self.v_cruise_setpoint) else self.v_target

        # HOLDING
        elif self.state == MergeState.holding:
          self.hold_frames -= 1
          if self.event_frames >= _FRAMES(_MAX_EVENT_TIME):
            self._end_event(timed_out=True)
          elif self.hold_frames <= 0:
            self._end_event()
          elif self.detected:
            self.state = MergeState.slowing

    # DISABLED
    elif self.state == MergeState.disabled:
      if self.long_enabled and self.enabled:
        if self.long_override:
          self.state = MergeState.overriding
        else:
          self.state = MergeState.enabled

    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES

    return enabled, active

  # ----- output -----

  def _update_solution(self) -> None:
    # The planner works out the deceleration itself from the lower speed target.
    self.a_target = min(max(self.a_ego, -_SANE_ACCEL), _SANE_ACCEL) if _finite(self.a_ego) else 0.

    if self.state not in ACTIVE_STATES:
      self.v_target = V_CRUISE_UNSET
      return

    if self.state == MergeState.slowing and self.candidate is not None:
      wanted = self.candidate.wanted_speed(self.v_ego)
      self.event_v_lead = self.candidate.v_lead
      self.event_make_room = self.candidate.trigger not in (TRIGGER_RADAR_SLOW, TRIGGER_RADAR_EARLY)
      # Lowest speed we may ask for: the configured drop below where we started, and never under the absolute floor.
      floor = max(self.v_ref - self.max_drop, MIN_V_TARGET)
      ceiling = self.v_cruise_setpoint
      if _finite(wanted, floor, ceiling) and floor <= ceiling:
        wanted = min(max(wanted, floor), ceiling)
        # Within one event the target only goes down. It is released in one step when the event ends.
        self.v_target = min(self.v_target, wanted)

  def get_v_target_from_control(self) -> float:
    if self.fault or not self.is_active or self.test_only:
      return V_CRUISE_UNSET

    v = self.v_target
    # Last line of defence. Whatever happened above, only a sane number may leave this controller.
    if not _finite(v, self.v_cruise_setpoint) or not MIN_V_TARGET < self.v_cruise_setpoint < MAX_V_CRUISE:
      return V_CRUISE_UNSET
    # The drop limit can be changed in settings while an event is running. Always honour the current one.
    if _finite(self.v_ref, self.max_drop):
      v = max(v, self.v_ref - self.max_drop)
    if v < MIN_V_TARGET or v > self.v_cruise_setpoint:
      return V_CRUISE_UNSET

    return float(v)

  def get_a_target_from_control(self) -> float:
    return self.a_target

  # ----- fault handling -----

  @property
  def fault_alert(self) -> bool:
    return self.fault and self.fault_frames < _FRAMES(_FAULT_ALERT_TIME)

  def _set_fault(self) -> None:
    """Something unexpected happened. Switch off and stay off until the next restart."""
    if not self.fault:
      try:
        cloudlog.exception("SmartCruiseControlMerge fault, controller disabled until restart")
      except Exception:
        pass
    self.fault = True
    self.state = MergeState.disabled
    self.is_enabled = False
    self.is_active = False
    self.detected = False
    self.candidate = None
    self.v_target = V_CRUISE_UNSET
    self.output_v_target = V_CRUISE_UNSET
    ok = isinstance(self.a_ego, (int, float)) and math.isfinite(self.a_ego)
    self.output_a_target = min(max(float(self.a_ego), -_SANE_ACCEL), _SANE_ACCEL) if ok else 0.

  def _update(self, sm: messaging.SubMaster) -> None:
    self._update_params()
    self._update_calculations(sm)

    self.is_enabled, self.is_active = self._update_state_machine()
    self._update_solution()

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()
    self._update_ready()
    self._update_brake_warning()

  def _update_ready(self) -> None:
    """Bring the set speed down to just above our own speed while a vehicle beside us drifts our way."""
    self.ready_frames = max(0, self.ready_frames - 1)
    self.ready = False
    if (self.ready_frames == 0 or self.state != MergeState.enabled or self.test_only or self.fault or
        self.output_v_target != V_CRUISE_UNSET):
      return
    cruise = self.v_cruise_setpoint
    if not (_finite(cruise, self.v_ego, self.max_drop) and MIN_V_TARGET < cruise < MAX_V_CRUISE):
      return
    v = max(self.v_ego + _READY_MARGIN, cruise - self.max_drop, MIN_V_TARGET)
    if v < cruise - 0.5:
      self.v_ref = cruise
      self.output_v_target = float(v)
      self.ready = True

  def _update_brake_warning(self) -> None:
    self.brake_warning_frames = max(0, self.brake_warning_frames - 1)
    c = self.candidate
    if (self.state == MergeState.slowing and c is not None and c.trigger != TRIGGER_RADAR_SLOW and
        _finite(c.dist, c.v_lead, c.offset, self.v_ego) and abs(c.offset) < _BRAKE_WARN_MAX_OFFSET):
      closing = self.v_ego - c.v_lead
      room = max(c.dist - _BRAKE_WARN_HEADWAY * self.v_ego, 2.)
      if closing > _BRAKE_WARN_MIN_CLOSING and closing ** 2 / (2. * room) > _BRAKE_WARN_DECEL:
        self.brake_warning_frames = _FRAMES(_BRAKE_WARN_HOLD_TIME)
    self.brake_warning = self.brake_warning_frames > 0 and self.state in ACTIVE_STATES and not self.fault

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float,
             v_cruise_setpoint: float) -> None:
    self.frame += 1
    self.a_ego = a_ego

    if self.fault:
      self.fault_frames += 1
      self._set_fault()
      return

    try:
      self.long_enabled = bool(long_enabled)
      self.long_override = bool(long_override)
      self.v_ego = float(v_ego)
      self.a_ego = float(a_ego)
      self.v_cruise_setpoint = float(v_cruise_setpoint)
      self._update(sm)
    except Exception:
      self._set_fault()
