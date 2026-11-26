"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from typing import Any

from opendbc.car import structs
from opendbc.car.interfaces import CarInterfaceBase
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import get_nn_model_path
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.helpers import set_speed_limit_assist_availability

from openpilot.sunnypilot.sunnylink.statsd import STATSLOGSP


def _enforce_torque_lateral_control(CP: structs.CarParams, params: Params | None = None, enabled: bool = False) -> bool:
  if params is None:
    params = Params()

  if CP.steerControlType != structs.CarParams.SteerControlType.angle:
    enabled = params.get_bool("EnforceTorqueControl")

  return enabled


def _initialize_neural_network_lateral_control(CP: structs.CarParams, CP_SP: structs.CarParamsSP,
                                               params: Params | None = None, enabled: bool = False) -> bool:
  if params is None:
    params = Params()

  nnlc_model_path, nnlc_model_name, exact_match = get_nn_model_path(CP)

  if nnlc_model_name == "MOCK":
    cloudlog.error({"nnlc event": "car doesn't match any Neural Network model"})

  if nnlc_model_name != "MOCK" and CP.steerControlType != structs.CarParams.SteerControlType.angle:
    enabled = params.get_bool("NeuralNetworkLateralControl")

  CP_SP.neuralNetworkLateralControl.model.path = nnlc_model_path
  CP_SP.neuralNetworkLateralControl.model.name = nnlc_model_name
  CP_SP.neuralNetworkLateralControl.fuzzyFingerprint = not exact_match

  return enabled


def _initialize_intelligent_cruise_button_management(CP: structs.CarParams, CP_SP: structs.CarParamsSP, params: Params | None = None) -> None:
  if params is None:
    params = Params()

  icbm_enabled = params.get_bool("IntelligentCruiseButtonManagement")
  if icbm_enabled and CP_SP.intelligentCruiseButtonManagementAvailable and not CP.openpilotLongitudinalControl:
    CP_SP.pcmCruiseSpeed = False


def _initialize_torque_lateral_control(CI: CarInterfaceBase, CP: structs.CarParams, enforce_torque: bool, nnlc_enabled: bool) -> None:
  if nnlc_enabled or enforce_torque:
    CI.configure_torque_tune(CP.carFingerprint, CP.lateralTuning)


def _cleanup_unsupported_params(CP: structs.CarParams, CP_SP: structs.CarParamsSP, params: Params | None = None) -> None:
  if params is None:
    params = Params()

  if params.get_bool("LateralJerkTorqueController") and params.get_bool("NeuralNetworkLateralControl"):
    cloudlog.warning("LateralJerkTorqueController and NeuralNetworkLateralControl both enabled, disabling both")
    params.put_bool("LateralJerkTorqueController", False, block=True)
    params.put_bool("NeuralNetworkLateralControl", False, block=True)

  if CP.steerControlType == structs.CarParams.SteerControlType.angle:
    cloudlog.warning("SteerControlType is angle, cleaning up params")
    params.remove("NeuralNetworkLateralControl")
    params.remove("EnforceTorqueControl")
    params.remove("LateralJerkTorqueController")

  if not CP_SP.intelligentCruiseButtonManagementAvailable or CP.openpilotLongitudinalControl:
    cloudlog.warning("ICBM not available or openpilot Longitudinal Control enabled, cleaning up params")
    params.remove("IntelligentCruiseButtonManagement")

  if not CP.openpilotLongitudinalControl and CP_SP.pcmCruiseSpeed:
    cloudlog.warning("openpilot Longitudinal Control and ICBM not available, cleaning up params")
    params.remove("CustomAccIncrementsEnabled")
    params.remove("SmartCruiseControlVision")
    params.remove("SmartCruiseControlMap")
    params.remove("SmartCruiseControlMerge")

  set_speed_limit_assist_availability(CP, CP_SP, params)


# Steering strength measured on the owner's car, used in place of a generic placeholder.
# The number is the sideways acceleration (m/s^2) the car reaches at full steering command.
# Ioniq 6: sunnypilot has no measured data and uses 2.5. The device's own estimator measured 3.04 on this car
# (6,166 points, drive of 2026-11-25). With 2.5 every steering correction is about 20% too strong, and the wheel
# swings back and forth for several seconds after a bump.
MEASURED_LAT_ACCEL_FACTOR = {
  "HYUNDAI_IONIQ_6": (2.5, 3.0),  # (placeholder that gets replaced, measured value)
}

# Steering friction is left at sunnypilot's value on purpose. A measured 0.05 was tried on 2026-11-26 (commit
# 886b42b) and the wobble was no better: friction acts like extra gain around the centre, and this car's wobble is
# a gain problem (see latcontrol_torque_car_gain.py). No entries: nothing is replaced.
MEASURED_FRICTION: dict[str, tuple[float, float]] = {}


def _apply_measured_torque_tune(CP: structs.CarParams) -> bool:
  """Replace the placeholder steering strength and friction with the measured ones.
  Only for the listed cars, only with torque steering, and each value only while its placeholder is still in place.
  Custom Tuning and Self-Tune in the settings are applied later by the controller and still win."""
  changed = False
  try:
    if CP.lateralTuning.which() != 'torque':
      return False
    car = str(CP.carFingerprint)
    entry = MEASURED_LAT_ACCEL_FACTOR.get(car)
    if entry is not None and abs(float(CP.lateralTuning.torque.latAccelFactor) - entry[0]) <= 1e-3:
      CP.lateralTuning.torque.latAccelFactor = entry[1]
      changed = True
    entry = MEASURED_FRICTION.get(car)
    if entry is not None and abs(float(CP.lateralTuning.torque.friction) - entry[0]) <= 1e-4:
      CP.lateralTuning.torque.friction = entry[1]
      changed = True
  except Exception:
    cloudlog.exception("measured torque tune not applied")
  return changed


def setup_interfaces(CI: CarInterfaceBase, params: Params | None = None) -> None:
  enforce_torque = _enforce_torque_lateral_control(CI.CP, params)
  nnlc_enabled = _initialize_neural_network_lateral_control(CI.CP, CI.CP_SP, params)
  _initialize_intelligent_cruise_button_management(CI.CP, CI.CP_SP, params)
  _initialize_torque_lateral_control(CI, CI.CP, enforce_torque, nnlc_enabled)
  _apply_measured_torque_tune(CI.CP)
  _cleanup_unsupported_params(CI.CP, CI.CP_SP)

  try:
    STATSLOGSP.raw('sunnypilot.car_params', CI.CP.to_dict())
  except RuntimeError:
    pass  # to_dict fails on macOS due to library issues.
  # STATSLOGSP.raw('sunnypilot_params.car_params_sp', CP_SP.to_dict()) # https://github.com/sunnypilot/opendbc/pull/361


def initialize_params(params) -> list[dict[str, Any]]:
  keys: list = []

  # hyundai
  keys.extend([
    "HyundaiLongitudinalTuning",
  ])

  # subaru
  keys.extend([
    "SubaruStopAndGo",
    "SubaruStopAndGoManualParkingBrake",
  ])

  # tesla
  keys.extend([
    "TeslaCoopSteering",
    "TeslaMadsScreenButton",
  ])

  # toyota
  keys.extend([
    "ToyotaEnforceStockLongitudinal",
    "ToyotaStopAndGoHack",
    "ToyotaTSS2Smooth",
  ])

  return [{k: params.get(k, return_default=True)} for k in keys]
