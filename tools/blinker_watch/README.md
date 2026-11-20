# Blinker Watch — offline proof of concept (TEST ONLY)

A **standalone desktop research tool**, not yet part of on-road sunnypilot.
It scans a known turn-signal lamp region in a local road-facing camera recording
and tests whether amber or red light pulses repeat at plausible intervals.

## Safety boundary

- No imports or calls to driving controllers, cruise, steering, CAN, or buttons.
- It cannot alter vehicle behavior. A blink alone is **not** a safe reason to slow down.
- It does **not** find vehicles, track a moving lamp, identify the left/right side of a car,
  establish merge intent, or measure whether a following vehicle can slow safely.
- Do not deploy this as a live driver-assistance feature. Keep `SmartCruiseControlMergeTestOnly` enabled for existing Merge Assist validation.
- Use road-facing recordings only; no cabin video, route metadata, or keys are needed.

## Install dependencies (desktop)

Python 3.11+, `numpy`, and `opencv-python`.

## Example

```bash
cd tools/blinker_watch
python -m unittest -v test_blinker_watch.py
python video_scan.py /path/to/fcamera.hevc --color amber --roi 500,320,520,338 --source-fps 20 --json scan.json
python video_scan.py /path/to/fcamera.hevc --color red --roi 500,320,520,338
```

On some raw `.hevc` files, OpenCV guesses 25 FPS despite 20 FPS capture;
use `--source-fps 20` when the capture cadence is known from the recording.

ROIs are **pixel coordinates for a tightly cropped lamp** that must remain on the
same tracked lamp for the entire clip. Fixed ROIs are inadequate for a moving car.
A future vehicle/lamp tracker must supply frame-by-frame ROIs. Use `--reference-roi`
to subtract activity in a similar non-blinking patch, reducing exposure artifacts.

`candidate_times_s` = an ON/OFF/ON sequence. `confirmed_times_s` = at least
three ON transitions with two stable pulse periods. Both are research labels,
not driving decisions.

## Research plan

1. Validate existing real recordings as negative controls.
2. Identify **synchronized** road-facing video and log segments with visibly signaling cars.
3. Annotate per-frame vehicle and lamp ROIs, including red, amber, brakes, hazards,
   daytime glare and moving reflections.
4. Measure miss rate, false alerts/hour, first-detection delay and latency on a desktop.
5. Add tracking and replay-only integration, with no connection to speed control.

No claim about real-world performance is made from synthetic tests.
