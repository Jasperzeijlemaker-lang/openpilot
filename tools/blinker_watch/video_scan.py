"""Command-line, offline fcamera.hevc scanner for the Blink Watch prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from blinker_watch import PulseDetector, lamp_score_bgr


def roi_type(text: str) -> tuple[int, int, int, int]:
  try:
    p = tuple(int(x) for x in text.split(","))
  except ValueError as e:
    raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2") from e
  if len(p) != 4 or p[0] < 0 or p[1] < 0 or p[2] <= p[0] or p[3] <= p[1]:
    raise argparse.ArgumentTypeError("invalid ROI coordinates")
  return p


def scan(video: Path, roi: tuple[int, int, int, int], color: str,
         sample_fps: float = 10.0, reference_roi: tuple[int, int, int, int] | None = None,
         source_fps: float | None = None) -> dict:
  cap = cv2.VideoCapture(str(video))
  if not cap.isOpened():
    raise ValueError(f"Cannot decode video: {video}")
  fps = source_fps or cap.get(cv2.CAP_PROP_FPS)
  if fps <= 0:
    cap.release()
    raise ValueError("Missing video FPS")
  detector = PulseDetector(color)
  next_sample_time = 0.
  frame_idx = 0
  sampled = 0
  candidate_times = []
  confirmed_times = []
  last_candidate = False
  last_confirmed = False
  maximum_modulation = 0.
  try:
    while True:
      ok, frame = cap.read()
      if not ok:
        break
      timestamp = frame_idx / fps
      frame_idx += 1
      if timestamp + 0.00001 < next_sample_time:
        continue
      next_sample_time += 1. / sample_fps
      score = lamp_score_bgr(frame, roi, color)
      if reference_roi is not None:
        score = max(0., score - lamp_score_bgr(frame, reference_roi, color))
      result = detector.update(timestamp, score)
      maximum_modulation = max(maximum_modulation, result.modulation)
      sampled += 1
      if result.candidate and not last_candidate:
        candidate_times.append(round(timestamp, 2))
      if result.confirmed and not last_confirmed:
        confirmed_times.append(round(timestamp, 2))
      last_candidate = result.candidate
      last_confirmed = result.confirmed
  finally:
    cap.release()
  return {"input": str(video), "color": color, "roi": list(roi), "reference_roi": list(reference_roi) if reference_roi else None,
          "frames_decoded": frame_idx, "frames_sampled": sampled, "source_fps": fps,
          "duration_seconds": round(frame_idx/fps,2),
          "candidate_times_s": candidate_times, "confirmed_times_s": confirmed_times,
          "max_modulation": round(maximum_modulation, 3),
          "note": "Offline ROI research only; a blink does not establish merge intent and cannot change car controls."}


def main() -> None:
  ap = argparse.ArgumentParser(description="Offline, TEST-ONLY amber/red blinker probe; no vehicle control")
  ap.add_argument("video", type=Path, help="Local front-road camera recording, e.g. fcamera.hevc")
  ap.add_argument("--roi", type=roi_type, required=True, help="Tight light box x1,y1,x2,y2 (pixels)")
  ap.add_argument("--color", choices=("amber", "red"), required=True)
  ap.add_argument("--reference-roi", type=roi_type, help="Optional non-blinking similar-colored patch to compensate global exposure")
  ap.add_argument("--sample-fps", type=float, default=10.)
  ap.add_argument("--source-fps", type=float, help="Override FPS: raw HEVC may be misreported as 25 FPS")
  ap.add_argument("--json", type=Path, help="Save a small summary, no frames or location data")
  args = ap.parse_args()
  if args.source_fps is not None and not 1 <= args.source_fps <= 120:
    ap.error("--source-fps must be between 1 and 120")
  if not 4 <= args.sample_fps <= 30:
    ap.error("--sample-fps must be between 4 and 30")
  report = scan(args.video, args.roi, args.color, args.sample_fps, args.reference_roi, args.source_fps)
  result_text = json.dumps(report, indent=2)
  print(result_text)
  if args.json:
    args.json.write_text(result_text + "\n")


if __name__ == "__main__":
  main()
