import random
from types import SimpleNamespace as NS

import numpy as np

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.exit_watch import ExitWatch, _HOLD_TIME
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.tests.test_merge_controller import make_model, make_radar, make_car_state

V = 29.  # m/s, 65 mph
XS = [float(x) for x in ModelConstants.X_IDXS]


def scene(lead_right: float, lead_dist: float = 40., split: int = 0, path_drift: float = 0., blinker: bool = False, v_ego: float = V,
          leave_prob: float = 0.9, stay_prob: float = 0.9, lead_v: float | None = None, width: float = 3.6):
  """split: +1 the right line peels away to the right from 20 m on, -1 the left line peels away, 0 no split.
  path_drift: metres the model has shifted our whole lane toward the split side."""
  md = make_model([])

  def xy(ys):
    m = log.XYZTData.new_message()
    m.x = XS
    m.y = [float(y) for y in ys]
    return m

  peel = [3.6 * min(max((x - 20.) / 40., 0.), 1.) for x in XS]
  half = width / 2.
  shift = (split or 1) * path_drift
  left = [-half + shift - (p if split == -1 else 0.) for p in peel]
  right = [half + shift + (p if split == 1 else 0.) for p in peel]
  md.laneLines = [xy([-half - 3.6] * len(XS)), xy(left), xy(right), xy([half + 3.6] * len(XS))]
  probs = [0.9, 0.9, 0.9, 0.9]
  if split == 1:
    probs[1], probs[2] = stay_prob, leave_prob
  elif split == -1:
    probs[2], probs[1] = stay_prob, leave_prob
  md.laneLineProbs = probs
  px = [float(x) for x in md.position.x]
  position = log.XYZTData.new_message()
  position.x = px
  position.y = [0. for _ in px]
  md.position = position
  radar = make_radar([(3, lead_dist, lead_right, V if lead_v is None else lead_v, v_ego)])
  return {'modelV2': md, 'carState': make_car_state(blinker, v_ego), 'carStateSP': radar}


def run(watch, seconds, lead_from, lead_to, **kw):
  n = int(seconds / DT_MDL)
  seen = False
  for i in range(n):
    y = lead_from + (lead_to - lead_from) * i / max(n - 1, 1)
    watch.update(scene(y, **kw), kw.get('v_ego', V))
    assert not watch.fault
    seen = seen or watch.split
  return seen


class TestExitWatch:
  def test_lead_takes_right_exit_path_holds(self):
    w = ExitWatch()
    assert not run(w, 2., 0., 0., split=1)
    assert run(w, 4., 0., 4.5, split=1)
    assert w.side == 1 and not w.following

  def test_lead_takes_right_exit_and_our_path_follows(self):
    w = ExitWatch()
    run(w, 2., 0., 0., split=1)
    assert run(w, 4., 0., 4.5, split=1)
    assert not w.following
    run(w, 1., 4.5, 4.5, split=1, path_drift=1.)
    assert w.side == 1 and w.following and w.drift > 0.4

  def test_left_fork(self):
    w = ExitWatch()
    run(w, 2., 0., 0., split=-1)
    assert run(w, 4., 0., -4.5, split=-1)
    run(w, 1., -4.5, -4.5, split=-1, path_drift=1.)
    assert w.side == -1 and w.following

  def test_faded_line_counts_as_split(self):
    w = ExitWatch()
    run(w, 2., 0., 0.)
    assert run(w, 4., 0., 4.5, split=0, leave_prob=0.1) is False  # probs only apply when split != 0 in this helper
    w = ExitWatch()
    run(w, 2., 0., 0., split=1)
    assert run(w, 4., 0., 4.5, split=1, leave_prob=0.1)

  def test_ordinary_lane_change_by_lead_is_ignored(self):
    w = ExitWatch()
    run(w, 2., 0., 0.)
    assert not run(w, 5., 0., 3.6)

  def test_split_but_lead_stays_is_ignored(self):
    w = ExitWatch()
    assert not run(w, 8., 0., 0.3, split=1)

  def test_car_that_was_always_in_the_next_lane_is_ignored(self):
    w = ExitWatch()
    assert not run(w, 8., 3.6, 4.6, split=1)

  def test_blinker_means_we_are_leaving(self):
    w = ExitWatch()
    run(w, 2., 0., 0., split=1, blinker=True)
    assert not run(w, 4., 0., 4.5, split=1, blinker=True)

  def test_low_speed_is_ignored(self):
    w = ExitWatch()
    run(w, 2., 0., 0., split=1, v_ego=15., lead_v=15.)
    assert not run(w, 4., 0., 4.5, split=1, v_ego=15., lead_v=15.)

  def test_unclear_stay_line_is_ignored(self):
    w = ExitWatch()
    run(w, 2., 0., 0., split=1)  # lane width learned while both lines are clear
    assert not run(w, 4., 0., 4.5, split=1, stay_prob=0.2)

  def test_strange_lane_width_is_ignored(self):
    w = ExitWatch()
    run(w, 2., 0., 0., split=1, width=5.5)
    assert not run(w, 4., 0., 4.5, split=1, width=5.5)

  def test_flag_clears_after_hold(self):
    w = ExitWatch()
    run(w, 2., 0., 0., split=1)
    run(w, 4., 0., 4.5, split=1)
    assert w.split
    run(w, _HOLD_TIME + 1., 0., 0.)
    assert not w.split and not w.following and w.side == 0

  def test_repeated_frame_is_not_counted_twice(self):
    w = ExitWatch()
    sm = scene(4.5, split=1)
    for _ in range(100):
      w.update(sm, V)
    assert not w.split

  def test_error_latches_off(self):
    w = ExitWatch()
    w.update(None, V)
    assert w.fault and not w.split
    run_ok = scene(0., split=1)
    w.update(run_ok, V)
    assert w.fault and not w.split

  def test_missing_radar_and_bad_numbers(self):
    w = ExitWatch()
    sm = scene(0., split=1)
    del sm['carStateSP']
    w.update(sm, V)
    assert not w.fault
    for bad in (float('nan'), float('inf'), -1e30, 1e30):
      w = ExitWatch()
      for _ in range(30):
        sm = scene(4.5, split=1)
        sm['carStateSP'].mergeRadarTracks[0].dRel = bad
        w.update(sm, V)
        sm = scene(4.5, split=1)
        sm['carStateSP'].mergeRadarTracks[0].yRel = bad
        w.update(sm, bad if bad != 1e30 else V)
        sm = scene(bad if np.isfinite(bad) else 0., split=1)
        w.update(sm, V)
      assert not w.fault and not w.split

  def test_fuzz_never_faults(self):
    rng = random.Random(3)
    w = ExitWatch()
    for _ in range(4000):
      sm = scene(rng.uniform(-8, 8), lead_dist=rng.uniform(0, 150), split=rng.choice([-1, 0, 1]), path_drift=rng.uniform(-3, 3),
                 blinker=rng.random() < 0.05, v_ego=rng.uniform(0, 45), leave_prob=rng.random(), stay_prob=rng.random(),
                 width=rng.uniform(1.5, 6.))
      if rng.random() < 0.05:
        sm['modelV2'] = NS(frameId=rng.randint(0, 10 ** 6), laneLines=[], laneLineProbs=[], position=NS(x=[], y=[]))
      w.update(sm, rng.uniform(0, 45))
      assert not w.fault
      assert w.side in (-1, 0, 1) and np.isfinite(w.drift)

  def test_recorded_exit_lane_opening(self):
    # Recorded Nov 19, 15:16 into the drive: a lane opens on the right and the car ahead moves into it.
    # The model's lane lines spread apart near the car too (3.3 m to 4.7 m), so the width must be remembered from before.
    w = ExitWatch()
    run(w, 3., 0.4, 0.4, lead_dist=22., width=3.3)
    assert not w.split
    seen = run(w, 1.5, 1.4, 2.8, lead_dist=22., width=4.6, split=1, leave_prob=0.6, stay_prob=0.95)
    assert seen and w.side == 1
