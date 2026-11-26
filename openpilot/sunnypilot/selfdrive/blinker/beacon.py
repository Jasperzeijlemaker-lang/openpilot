"""
How Blinker Watch tells the merge controller what it sees: one short line of text in the memory-only settings store.
"vehicles | time". Each vehicle is "radar slot:side:distance". Side: -1 left signal, +1 right signal.
Plain Python, no other imports, so both sides can use it safely.
"""
KEY = "BlinkerWatchSeen"
MEM_PARAMS_PATH = "/dev/shm/params"
MAX_VEHICLES = 8


def encode(detections, now: float) -> str:
  """detections: objects with slot, side, dist."""
  parts = [f"{int(d.slot)}:{-1 if d.side < 0 else 1}:{float(d.dist):.1f}" for d in list(detections)[:MAX_VEHICLES]]
  return ",".join(parts) + f"|{float(now):.2f}"


def decode(text, now: float, max_age: float) -> dict[int, tuple[int, float]]:
  """Returns {radar slot: (side, distance)}. Anything old, odd or unreadable gives an empty result."""
  try:
    if isinstance(text, bytes):
      text = text.decode()
    if not text or len(text) > 400:
      return {}
    body, stamp = text.split("|")
    age = now - float(stamp)
    if not -0.1 <= age <= max_age:     # also rejects NaN. The small allowance is for the rounding of the time stamp
      return {}
    out: dict[int, tuple[int, float]] = {}
    for part in body.split(",")[:MAX_VEHICLES]:
      slot, side, dist = part.split(":")
      slot, side, dist = int(slot), int(side), float(dist)
      if side not in (-1, 1) or not 0. < dist < 300. or not 0 <= slot < 256:
        return {}
      out[slot] = (side, dist)
    return out
  except Exception:
    return {}
