"""
Blinker (turn signal) detection core. Plain numpy. No camera, car or messaging code in here, so it can be tested
anywhere, including with synthetic pictures.

Idea: the radar says where a vehicle is. That gives a small box in the camera picture. A turn signal is a
warm-coloured light on ONE side of that box that switches on and off about 1 to 2 times a second. Brake lights and
changes in exposure or shadow light up or darken both sides together, so a flash only counts when one side
changes much more than the other.

Sides are named as seen from behind the vehicle: LEFT = -1, RIGHT = +1.
"""
from collections import deque

import numpy as np

FPS = 20.
WINDOW = 3.0             # s of history looked at
MIN_PERIOD, MAX_PERIOD = 0.45, 1.0   # s per flash (1.0 to 2.2 flashes a second; the legal range is 1 to 2)
MIN_CYCLES = 2.5         # flashes that must be seen before it counts
WARM_CR = 142            # red-difference chroma above this counts as red or orange light (128 is grey)
MIN_LUMA = 60            # ...and it has to be bright enough to be a lamp, not dark paint
MIN_ABS_SWING = 6.0      # the flashing side must change at least this much (warm-light units)
SIDE_RATIO = 2.0         # ...and at least this many times more than the other side
IN_STEP_MIN_SHARE = 0.2  # other side changing at least this share as much...
IN_STEP_MAX_CORR = 0.5   # ...and this much in step with the flashing side: both sides are flashing together
MIN_CORR = 0.55          # how well the signal repeats after one period (1 = perfectly)
MAX_HALF_CORR = -0.25    # ...and it must be the opposite after half a period
MAX_EDGE_RATIO = 1.5     # switches seen, compared with what a clean flash of that rate gives
HOLD_FRAMES = 6          # frames in a row the answer must stay the same before it is reported (0.3 s)
DUTY = (0.25, 0.75)      # share of the time the lamp is on

# Camera of the comma 4 (os04c10): picture size and focal length in pixels.
CAM_W, CAM_H, CAM_F = 1344, 760, 1141.5
RADAR_TO_CAMERA = 1.52   # m, the radar sits this far ahead of the camera
LAMP_HEIGHT = 0.75       # m above the road, middle of the box we look at
BOX_WIDTH = 2.0          # m
BOX_UP, BOX_DOWN = 0.40, 0.30   # box height above / below its middle, as a share of its width
MIN_BOX_PX = 14          # smaller than this and a lamp is under 2 pixels: do not even try
MAX_DIST = 70.           # m
MIN_OBJ_SPEED = 4.       # m/s over the ground: ignore signs and parked things
MAX_SIDE = 9.            # m to the side


def warm_energy(y: np.ndarray, cr: np.ndarray) -> float:
  """How much red/orange light is in a patch. y and cr are uint8 arrays of the same shape."""
  if y.size == 0 or y.shape != cr.shape:
    return 0.
  w = cr.astype(np.float32) - WARM_CR
  np.clip(w, 0., None, out=w)
  w *= (y >= MIN_LUMA)
  # a lamp is small: the brightest part of the patch matters, not its area
  k = max(4, w.size // 12)
  top = np.partition(w.ravel(), -k)[-k:]
  return float(top.sum()) / float(np.sqrt(k))


class SideSignal:
  """Flash detector for one vehicle. Feed it the warm-light energy of the left and right half of its box."""

  def __init__(self, fps: float = FPS):
    self.fps = fps
    n = int(WINDOW * fps)
    self.left: deque[float] = deque(maxlen=n)
    self.right: deque[float] = deque(maxlen=n)
    self.result = 0
    self.confidence = 0.
    self.period = 0.

  def reset(self) -> None:
    self.left.clear()
    self.right.clear()
    self.result, self.confidence, self.period = 0, 0., 0.

  def update(self, left: float, right: float) -> int:
    if not (np.isfinite(left) and np.isfinite(right)):
      self.reset()
      return 0
    self.left.append(float(left))
    self.right.append(float(right))
    self.result, self.confidence, self.period = self._analyse()
    return self.result

  def _analyse(self) -> tuple[int, float, float]:
    n = len(self.left)
    min_lag, max_lag = int(MIN_PERIOD * self.fps), int(MAX_PERIOD * self.fps)
    if n < int(MIN_CYCLES * min_lag) + 2:
      return 0, 0., 0.
    L = np.asarray(self.left, dtype=np.float64)
    R = np.asarray(self.right, dtype=np.float64)
    best = (0, 0., 0.)
    for side, a, b in ((-1, L, R), (1, R, L)):
      lo, hi = np.percentile(a, 10), np.percentile(a, 90)
      swing = hi - lo
      other = np.percentile(b, 90) - np.percentile(b, 10)
      if swing < MIN_ABS_SWING or swing < SIDE_RATIO * other:
        continue
      # hazard lights, tapped brakes and passing shadows flash BOTH sides in step. If the other side moves in step
      # with this one at all noticeably, it is not a turn signal.
      if other > IN_STEP_MIN_SHARE * swing:
        ac, bc = a - a.mean(), b - b.mean()
        denom = float(np.sqrt(np.dot(ac, ac) * np.dot(bc, bc)))
        if denom > 1e-9 and float(np.dot(ac, bc)) / denom > IN_STEP_MAX_CORR:
          continue
      on = a > (hi + lo) / 2.
      duty = float(on.mean())
      edges = int(np.count_nonzero(on[1:] != on[:-1]))   # times the lamp switched on or off
      if not DUTY[0] < duty < DUTY[1]:
        continue
      # this side, minus what the other side does at the same moment (brakes, exposure, shadows)
      d = a - b
      d = d - d.mean()
      var = float(np.dot(d, d))
      if var <= 1e-9:
        continue
      for lag in range(min_lag, max_lag + 1):
        if n < MIN_CYCLES * lag:
          break
        half_lag = max(lag // 2, 1)
        full = float(np.dot(d[lag:], d[:-lag])) / var * n / (n - lag)
        half = float(np.dot(d[half_lag:], d[:-half_lag])) / var * n / (n - half_lag)
        # a lamp flashing once per `lag` switches twice per `lag`. Much more than that is flicker, not a signal.
        if full > MIN_CORR and half < MAX_HALF_CORR and edges <= MAX_EDGE_RATIO * 2. * n / lag + 1:
          conf = min(1., (full - MIN_CORR) / (1. - MIN_CORR) * 0.6 + min(-half, 1.) * 0.4)
          if conf > best[1]:
            best = (side, conf, lag / self.fps)
    return best


def _rotation(rpy) -> np.ndarray:
  r, p, y = (float(v) for v in rpy)
  cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
  return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                   [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                   [-sp, sr * cp, cr * cp]])


def box_for(d_rel: float, y_left: float, rpy, cam_height: float, width: int = CAM_W, height: int = CAM_H,
            focal: float = CAM_F) -> tuple[int, int, int, int, int] | None:
  """Pixel box (x0, x_mid, x1, y0, y1) around the back of a vehicle the radar reports, or None if it is not usable."""
  dist = d_rel + RADAR_TO_CAMERA
  if dist < 3.:
    return None
  p = _rotation(rpy).T @ np.array([dist, y_left, LAMP_HEIGHT - cam_height])
  xv, yv, zv = -p[1], -p[2], p[0]
  if zv < 3.:
    return None
  u, v = focal * xv / zv + width / 2., focal * yv / zv + height / 2.
  w = focal * BOX_WIDTH / zv
  x0, x1, y0, y1 = int(u - w / 2.), int(u + w / 2.), int(v - w * BOX_UP), int(v + w * BOX_DOWN)
  if x1 - x0 < MIN_BOX_PX or x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
    return None
  return x0, int(u), x1, y0, y1


class Detection:
  __slots__ = ("slot", "side", "confidence", "period", "dist", "y_left")

  def __init__(self, slot, side, confidence, period, dist, y_left):
    self.slot, self.side, self.confidence, self.period, self.dist, self.y_left = slot, side, confidence, period, dist, y_left


class BlinkerWatcher:
  """Follows every vehicle the radar reports and says which of them is flashing a turn signal.
  Call process() once per camera frame with the frame's two planes (NV12 layout)."""

  def __init__(self, fps: float = FPS):
    self.fps = fps
    self.signals: dict[int, SideSignal] = {}
    self.last_dist: dict[int, float] = {}
    self.held: dict[int, tuple[int, int]] = {}   # slot -> (side, frames in a row)
    self.tracked = 0

  def reset(self) -> None:
    self.signals.clear()
    self.last_dist.clear()
    self.held.clear()

  def process(self, y_plane: np.ndarray, uv_plane: np.ndarray, tracks, rpy, cam_height: float, v_ego: float) -> list[Detection]:
    """y_plane: height x width, uint8. uv_plane: height/2 x width, uint8, bytes alternate Cb, Cr.
    tracks: iterable of (slot, d_rel, y_left, v_rel)."""
    h, w = y_plane.shape[:2]
    out: list[Detection] = []
    seen = set()
    if not (np.isfinite(v_ego) and np.isfinite(cam_height) and 0.5 < cam_height < 2.5 and len(rpy) == 3 and
            all(np.isfinite(float(a)) and abs(float(a)) < 0.3 for a in rpy)):
      self.reset()
      return out
    focal = CAM_F * w / CAM_W
    nearer: list[tuple[int, int, int, int]] = []
    # nearest first: a vehicle hidden behind a nearer one is skipped, or it would be blamed for the nearer one's lamp
    for slot, d_rel, y_left, v_rel in sorted(tracks, key=lambda t: t[1] if np.isfinite(t[1]) else 1e9):
      slot = int(slot)
      if not (np.isfinite(d_rel) and np.isfinite(y_left) and np.isfinite(v_rel)):
        continue
      if not (3. < d_rel < MAX_DIST and abs(y_left) < MAX_SIDE and v_ego + v_rel > MIN_OBJ_SPEED):
        continue
      box = box_for(d_rel, y_left, rpy, cam_height, w, h, focal)
      if box is None:
        continue
      x0, xm, x1, y0, y1 = box
      ym = (y0 + y1) // 2
      if any(a <= xm <= b and c <= ym <= d for a, b, c, d in nearer):
        continue
      nearer.append((x0, x1, y0, y1))
      # the radar reuses slots: a jump in distance means it is a different vehicle now
      if slot in self.last_dist and abs(self.last_dist[slot] - d_rel) > 4.:
        self.signals.pop(slot, None)
        self.held.pop(slot, None)
      self.last_dist[slot] = d_rel
      sig = self.signals.get(slot)
      if sig is None:
        sig = self.signals[slot] = SideSignal(self.fps)
      # colour is stored at half resolution: take every second brightness pixel to match it
      ya, yb = y0 // 2 * 2, y1 // 2 * 2
      xa, xmid, xb = x0 // 2 * 2, xm // 2 * 2, x1 // 2 * 2
      y_l = y_plane[ya:yb:2, xa:xmid:2]
      y_r = y_plane[ya:yb:2, xmid:xb:2]
      cr_l = uv_plane[ya // 2:yb // 2, xa + 1:xmid:2]
      cr_r = uv_plane[ya // 2:yb // 2, xmid + 1:xb:2]
      side = sig.update(warm_energy(y_l, cr_l), warm_energy(y_r, cr_r))
      seen.add(slot)
      # a real signal keeps flashing: it has to hold for a short while before it is reported
      run = self.held.get(slot, (0, 0))
      self.held[slot] = (side, run[1] + 1 if side and side == run[0] else int(bool(side)))
      if side and self.held[slot][1] >= HOLD_FRAMES:
        out.append(Detection(slot, side, sig.confidence, sig.period, d_rel, y_left))
    for slot in [s for s in self.signals if s not in seen]:
      del self.signals[slot]
      self.last_dist.pop(slot, None)
      self.held.pop(slot, None)
    self.tracked = len(seen)
    return out
