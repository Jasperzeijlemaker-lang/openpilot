"""
Faster "speed down" button for the Ioniq 6 (a road test, with an off switch).

This car has no openpilot gas/brake control. sunnypilot lowers the cruise speed the way a thumb would: it sends short
"speed down" button presses. Stock, it waits 0.2 s between presses, and on this car the set speed falls about
3 mph a second (measured 2026-11-25). The car's own cruise slows harder the further the set speed is below the
driven speed (more regen), so a set speed that falls faster makes Merge Assist bite sooner.

Here the wait between "speed down" presses is 0.1 s. Nothing else changes: the same button message, the same number
of copies, the same counter handling, and "speed up" presses keep the stock 0.2 s. Whether the car accepts presses
this fast can only be seen on the road: the set speed in the drive log shows it.

The press code lives in sunnypilot's opendbc, which this fork only borrows, so the stock method is swapped at start-up
instead of edited. The swap only happens if that method is exactly the version this was written against.
"""
import hashlib
import inspect

from openpilot.common.swaglog import cloudlog

PARAM = "FastSpeedDown"
CARS = ("HYUNDAI_IONIQ_6",)
STOCK_INTERVAL = 0.2  # s between presses, as in sunnypilot
FAST_INTERVAL = 0.1  # s between "speed down" presses
STOCK_SOURCE_SHA = "6ea146c54066ec2c"  # first 16 hex digits of sha256 of the stock method's source
_COPIES = 20
_COUNTER_OFFSETS = (1, 1, 0, None)


def make_method(DT_CTRL, HyundaiFlags, Buttons, create_buttons):
  """The stock method, with one difference: the wait before the next "speed down" press."""
  def create_canfd_mock_button_messages(self, packer, CS, CAN, send_button):
    can_sends = []
    if self.CP.flags & HyundaiFlags.CANFD_ALT_BUTTONS:
      pass
    else:
      interval = FAST_INTERVAL if send_button == Buttons.SET_DECEL else STOCK_INTERVAL
      if (self.frame - self.last_button_frame) * DT_CTRL > interval:
        self.button_frame += 1
        button_counter_offset = _COUNTER_OFFSETS[self.button_frame % 4]
        if button_counter_offset is not None:
          for _ in range(_COPIES):
            can_sends.append(create_buttons(packer, self.CP, CAN, (CS.buttons_counter + button_counter_offset) % 0xF, send_button))
          self.last_button_frame = self.frame
    return can_sends
  return create_canfd_mock_button_messages


def install(CP, params) -> bool:
  """Swap in the faster method. True if it was swapped. Any doubt at all: leave the stock method alone."""
  try:
    if str(CP.carFingerprint) not in CARS or params is None or not params.get_bool(PARAM):
      return False
    from opendbc.car import DT_CTRL
    from opendbc.car.hyundai import hyundaicanfd
    from opendbc.car.hyundai.values import HyundaiFlags, Buttons
    from opendbc.sunnypilot.car.hyundai.icbm import IntelligentCruiseButtonManagementInterface as Stock

    stock = Stock.__dict__.get("create_canfd_mock_button_messages")
    if stock is None or getattr(stock, "_fast_speed_down", False):
      return bool(stock is not None)
    sha = hashlib.sha256(inspect.getsource(stock).encode()).hexdigest()[:16]
    if sha != STOCK_SOURCE_SHA:
      cloudlog.event("fast_speed_down", error=True, what="not installed: stock press code has changed", sha=sha)
      return False
    method = make_method(DT_CTRL, HyundaiFlags, Buttons, hyundaicanfd.create_buttons)
    method._fast_speed_down = True
    method._stock = stock
    Stock.create_canfd_mock_button_messages = method
    cloudlog.event("fast_speed_down", error=True, what="installed", interval=FAST_INTERVAL)
    return True
  except Exception:
    cloudlog.exception("fast_speed_down not installed")
    return False
