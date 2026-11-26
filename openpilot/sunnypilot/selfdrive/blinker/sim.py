"""
A small picture simulator for the blinker detector: draws the back of a vehicle with tail lamps into a camera frame
(NV12 layout, as the device delivers it), where the radar says the vehicle is, and lets lamps flash, brake or stay dark.
Used by the tests. It is a drawing, not a driving simulator: it checks the detector's logic, not real-world optics.
"""
import numpy as np

from openpilot.sunnypilot.selfdrive.blinker.blinker_core import CAM_W, CAM_H, box_for

RPY = (0.0, -0.02, -0.02)
CAM_HEIGHT = 1.2

AMBER = (225, 70, 172)      # (brightness, Cb, Cr) of a lit orange signal
RED_LIT = (205, 105, 196)   # lit brake light / red turn signal
RED_DARK = (55, 122, 139)   # unlit red lens in shade
BODY = (48, 128, 128)


class Scene:
  def __init__(self, rng, dist=25., y_left=0., v_ego=28., v_rel=-1., signal=0, signal_color=AMBER, rate_hz=1.5, duty=0.5,
               brake=False, brake_pump_hz=0., hazards=False, flicker_hz=0., glints=0., jitter=0.06, noise=4., stride_pad=64,
               drop_frames=0., backlit=False):
    self.rng = rng
    self.__dict__.update(dist=dist, y_left=y_left, v_ego=v_ego, v_rel=v_rel, signal=signal, signal_color=signal_color,
                         rate_hz=rate_hz, duty=duty, brake=brake, brake_pump_hz=brake_pump_hz, hazards=hazards,
                         flicker_hz=flicker_hz, glints=glints, jitter=jitter, noise=noise, stride_pad=stride_pad,
                         drop_frames=drop_frames, backlit=backlit)
    self.phase = rng.uniform(0, 1)
    self.t = 0.
    self._y = self._uv = self._last_region = None

  def _lamp_on(self, hz, duty=0.5):
    return ((self.t * hz + self.phase) % 1.0) < duty

  def frame(self, dt=0.05):
    """Returns (y_plane, uv_plane, tracks) for the next camera frame."""
    rng = self.rng
    self.t += dt
    self.dist = max(6., self.dist + self.v_rel * dt)
    stride = CAM_W + self.stride_pad
    base = 95. if not self.backlit else 70.
    gain = 1.0 + (0.35 * np.sign(np.sin(2 * np.pi * self.flicker_hz * self.t)) if self.flicker_hz else 0.)
    if self._y is None:
      # the far background is drawn once; only the area around the vehicle is redrawn every frame (for speed)
      self._y = np.clip(rng.normal(base, self.noise, (CAM_H, stride)), 0, 255).astype(np.uint8)
      self._uv = np.clip(rng.normal(128., 1.5, (CAM_H // 2, stride)), 0, 255).astype(np.uint8)
    y, uv = self._y, self._uv
    tracks = [(7, self.dist + rng.normal(0, 0.15), self.y_left + rng.normal(0, 0.08), self.v_rel)]
    box = box_for(self.dist, self.y_left, RPY, CAM_HEIGHT)
    regions = [self._last_region]
    if box is not None:
      pad = (box[2] - box[0]) // 2 + 8
      self._last_region = (max(box[0] - pad, 0) // 2 * 2, min(box[2] + pad, CAM_W) // 2 * 2,
                           max(box[3] - pad, 0) // 2 * 2, min(box[4] + pad, CAM_H) // 2 * 2)
      regions.append(self._last_region)
    for reg in regions:
      if reg is not None:
        ra, rb, rc, rd = reg
        y[rc:rd, ra:rb] = np.clip(rng.normal(base * gain, self.noise, (rd - rc, rb - ra)), 0, 255)
        uv[rc // 2:rd // 2, ra:rb] = np.clip(rng.normal(128., 1.5, (rd // 2 - rc // 2, rb - ra)), 0, 255)
    if box is None:
      return y[:, :CAM_W], uv[:, :CAM_W], tracks
    x0, xm, x1, y0, y1 = box
    w = x1 - x0
    # the radar and the calibration are never exact: the real car sits a little off the box
    off = int(rng.normal(0, self.jitter) * w)
    cx0, cx1 = x0 + off + int(0.05 * w), x1 + off - int(0.05 * w)   # car is 1.8 m of the 2.0 m box
    cy0, cy1 = y0 + int(0.15 * (y1 - y0)), y1 - int(0.05 * (y1 - y0))

    def paint(xa, xb, ya, yb, col):
      xa, xb, ya, yb = max(xa, 0), min(xb, CAM_W), max(ya, 0), min(yb, CAM_H)
      if xb <= xa or yb <= ya:
        return
      y[ya:yb, xa:xb] = np.clip(rng.normal(col[0] * (gain if col is BODY or col is RED_DARK else 1.), self.noise, (yb - ya, xb - xa)), 0, 255)
      ua, ub = xa // 2 * 2, max(xb // 2 * 2, xa // 2 * 2 + 2)
      uv[ya // 2:max(yb // 2, ya // 2 + 1), ua:ub:2] = col[1]
      uv[ya // 2:max(yb // 2, ya // 2 + 1), ua + 1:ub:2] = col[2]

    paint(cx0, cx1, cy0, cy1, BODY)
    lamp_w = max(2, int(0.17 * (cx1 - cx0)))
    lamp_h = max(2, int(0.13 * (cy1 - cy0)))
    ly0 = cy0 + int(0.45 * (cy1 - cy0))
    brake_on = self.brake and (not self.brake_pump_hz or self._lamp_on(self.brake_pump_hz))
    for side, xa in ((-1, cx0), (1, cx1 - lamp_w)):
      col = RED_LIT if brake_on else RED_DARK
      flashing = self.hazards or self.signal == side
      if flashing:
        col = self.signal_color if self._lamp_on(self.rate_hz, self.duty) else (RED_DARK if self.signal_color is AMBER or not brake_on else RED_DARK)
      paint(xa, xa + lamp_w, ly0, ly0 + lamp_h, col)
    if self.glints and rng.random() < self.glints:
      gx = rng.integers(cx0, max(cx1 - 4, cx0 + 1))
      paint(gx, gx + 4, cy0 + 2, cy0 + 6, RED_LIT)
    return y[:, :CAM_W], uv[:, :CAM_W], tracks


def run_scene(watcher, scene, seconds=6., fps=20.):
  """Runs one scene. Returns (sides reported per frame, time of the first report or None)."""
  sides, first = [], None
  for i in range(int(seconds * fps)):
    y, uv, tracks = scene.frame(1. / fps)
    if scene.drop_frames and scene.rng.random() < scene.drop_frames:
      continue
    dets = watcher.process(y, uv, tracks, RPY, CAM_HEIGHT, scene.v_ego)
    side = dets[0].side if dets else 0
    sides.append(side)
    if side and first is None:
      first = (i + 1) / fps
  return sides, first
