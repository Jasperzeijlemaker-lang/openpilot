from types import SimpleNamespace as NS

from opendbc.car import structs
from openpilot.sunnypilot.selfdrive.car.interfaces import _apply_measured_torque_tune, MEASURED_LAT_ACCEL_FACTOR


def make_cp(fingerprint, factor=2.5, kind='torque', friction=0.005):
  torque = NS(latAccelFactor=factor, friction=friction, latAccelOffset=0.)
  return NS(carFingerprint=fingerprint, lateralTuning=NS(which=lambda: kind, torque=torque))


class TestMeasuredTorqueTune:
  def test_ioniq_6_placeholder_is_replaced(self):
    cp = make_cp("HYUNDAI_IONIQ_6")
    assert _apply_measured_torque_tune(cp)
    assert cp.lateralTuning.torque.latAccelFactor == 3.0
    assert cp.lateralTuning.torque.friction == 0.005  # nothing else changes

  def test_real_car_params_object(self):
    cp = structs.CarParams()
    cp.carFingerprint = "HYUNDAI_IONIQ_6"
    cp.lateralTuning.init('torque')
    cp.lateralTuning.torque.latAccelFactor = 2.5
    cp.lateralTuning.torque.friction = 0.005
    assert _apply_measured_torque_tune(cp)
    assert abs(cp.lateralTuning.torque.latAccelFactor - 3.0) < 1e-6
    assert abs(cp.lateralTuning.torque.friction - 0.005) < 1e-6
    assert not _apply_measured_torque_tune(cp)  # second call changes nothing more

  def test_other_cars_are_untouched(self):
    for car in ("HYUNDAI_IONIQ_5", "TOYOTA_COROLLA_TSS2", "", "MOCK"):
      cp = make_cp(car)
      assert not _apply_measured_torque_tune(cp)
      assert cp.lateralTuning.torque.latAccelFactor == 2.5

  def test_a_value_someone_else_set_is_kept(self):
    # upstream gets real numbers one day, or the value was already changed: leave it alone
    for factor in (2.7, 3.2, 2.0):
      cp = make_cp("HYUNDAI_IONIQ_6", factor=factor)
      assert not _apply_measured_torque_tune(cp)
      assert cp.lateralTuning.torque.latAccelFactor == factor

  def test_only_torque_steering(self):
    cp = make_cp("HYUNDAI_IONIQ_6", kind='pid')
    assert not _apply_measured_torque_tune(cp)
    assert cp.lateralTuning.torque.latAccelFactor == 2.5

  def test_bad_input_never_raises(self):
    assert not _apply_measured_torque_tune(None)
    assert not _apply_measured_torque_tune(NS(carFingerprint="HYUNDAI_IONIQ_6", lateralTuning=None))
    assert not _apply_measured_torque_tune(make_cp("HYUNDAI_IONIQ_6", factor=float('nan')))

  def test_values_are_sane(self):
    for placeholder, measured in MEASURED_LAT_ACCEL_FACTOR.values():
      assert 1.0 < placeholder < 5.0 and 1.0 < measured < 5.0
      assert abs(measured / placeholder - 1) < 0.35  # never a drastic change
