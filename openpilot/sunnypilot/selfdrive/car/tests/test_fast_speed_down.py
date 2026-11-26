import hashlib
import inspect
import types

import pytest

from opendbc.car import DT_CTRL
from opendbc.car.hyundai import hyundaicanfd
from opendbc.car.hyundai.values import HyundaiFlags, Buttons
from opendbc.sunnypilot.car.hyundai.icbm import IntelligentCruiseButtonManagementInterface as Stock
from openpilot.sunnypilot.selfdrive.car import fast_speed_down as fsd


class FakeParams:
  def __init__(self, on=True):
    self.on = on

  def get_bool(self, key):
    assert key == "FastSpeedDown"
    return self.on


def cp(car="HYUNDAI_IONIQ_6"):
  return types.SimpleNamespace(carFingerprint=car, flags=int(HyundaiFlags.CANFD))


@pytest.fixture(autouse=True)
def stock_method(monkeypatch):
  """Every test starts from, and ends with, the stock method. Button messages become plain tuples."""
  original = Stock.__dict__["create_canfd_mock_button_messages"]
  original = getattr(original, "_stock", original)
  monkeypatch.setattr(Stock, "create_canfd_mock_button_messages", original)
  monkeypatch.setattr(hyundaicanfd, "create_buttons", lambda packer, CP, CAN, cnt, btn: ("button", cnt, btn))
  return original


def presses(method, button, seconds=3., flags=int(HyundaiFlags.CANFD)):
  """Run a press method the way the car controller does. Returns the list of (time, messages) for every burst sent."""
  me = types.SimpleNamespace(CP=types.SimpleNamespace(flags=flags), frame=0, last_button_frame=0, button_frame=0)
  out = []
  for frame in range(int(seconds / DT_CTRL)):
    me.frame = frame
    cs = types.SimpleNamespace(buttons_counter=(frame // 2) % 16)   # the car's own button message counts up at 50 Hz
    sent = method(me, None, cs, None, button)
    if sent:
      out.append((round(frame * DT_CTRL, 2), sent))
  return out


def fast_method():
  return fsd.make_method(DT_CTRL, HyundaiFlags, Buttons, hyundaicanfd.create_buttons)


class TestFastSpeedDown:
  def test_written_against_the_press_code_this_fork_ships(self, stock_method):
    assert hashlib.sha256(inspect.getsource(stock_method).encode()).hexdigest()[:16] == fsd.STOCK_SOURCE_SHA

  def test_speed_up_presses_are_exactly_as_stock(self, stock_method):
    assert presses(fast_method(), Buttons.RES_ACCEL) == presses(stock_method, Buttons.RES_ACCEL)

  def test_with_the_stock_wait_speed_down_presses_are_exactly_as_stock(self, stock_method, monkeypatch):
    monkeypatch.setattr(fsd, "FAST_INTERVAL", fsd.STOCK_INTERVAL)
    assert presses(fast_method(), Buttons.SET_DECEL) == presses(stock_method, Buttons.SET_DECEL)

  def test_speed_down_presses_come_about_twice_as_often(self, stock_method):
    stock, fast = presses(stock_method, Buttons.SET_DECEL), presses(fast_method(), Buttons.SET_DECEL)
    assert 1.8 < len(fast) / len(stock) < 2.2
    assert 4.3 < len(stock) / 3. < 5.0 and 8.3 < len(fast) / 3. < 9.0        # presses a second
    gaps = [b[0] - a[0] for a, b in zip(fast, fast[1:], strict=False)]
    assert min(gaps) >= 0.1                                                   # never closer together than 0.1 s

  def test_each_press_is_the_same_message_as_stock(self, stock_method):
    stock, fast = presses(stock_method, Buttons.SET_DECEL), presses(fast_method(), Buttons.SET_DECEL)
    for _, burst in fast:
      assert len(burst) == 20 and len(set(burst)) == 1 and burst[0][2] == Buttons.SET_DECEL and 0 <= burst[0][1] < 16
    assert {len(b) for _, b in stock} == {20}

  def test_cars_with_the_other_button_layout_get_nothing_as_stock(self, stock_method):
    flags = int(HyundaiFlags.CANFD | HyundaiFlags.CANFD_ALT_BUTTONS)
    assert presses(fast_method(), Buttons.SET_DECEL, flags=flags) == presses(stock_method, Buttons.SET_DECEL, flags=flags) == []

  def test_install_swaps_the_method_once(self, stock_method):
    assert fsd.install(cp(), FakeParams())
    swapped = Stock.__dict__["create_canfd_mock_button_messages"]
    assert swapped is not stock_method and swapped._stock is stock_method
    assert fsd.install(cp(), FakeParams())                                     # asked again: still the same swap
    assert Stock.__dict__["create_canfd_mock_button_messages"] is swapped
    assert 8.3 < len(presses(swapped, Buttons.SET_DECEL)) / 3. < 9.0

  def test_not_installed_when_switched_off_or_for_another_car(self, stock_method):
    for car, params in ((cp(), FakeParams(False)), (cp("HYUNDAI_IONIQ_5"), FakeParams()), (cp(), None)):
      assert not fsd.install(car, params)
      assert Stock.__dict__["create_canfd_mock_button_messages"] is stock_method

  def test_not_installed_when_the_stock_code_has_changed(self, stock_method, monkeypatch):
    monkeypatch.setattr(fsd, "STOCK_SOURCE_SHA", "0" * 16)
    assert not fsd.install(cp(), FakeParams())
    assert Stock.__dict__["create_canfd_mock_button_messages"] is stock_method

  def test_install_never_raises(self, stock_method):
    class Broken:
      def get_bool(self, key):
        raise OSError("no store")
    assert not fsd.install(cp(), Broken())
    assert not fsd.install(None, FakeParams())
    assert Stock.__dict__["create_canfd_mock_button_messages"] is stock_method

  def test_the_real_car_controller_picks_up_the_swap(self, stock_method):
    from opendbc.car.hyundai.carcontroller import CarController
    assert "create_canfd_mock_button_messages" not in CarController.__dict__    # it uses the shared method, so the swap reaches it
    assert fsd.install(cp(), FakeParams())
    assert CarController.create_canfd_mock_button_messages is Stock.__dict__["create_canfd_mock_button_messages"]
