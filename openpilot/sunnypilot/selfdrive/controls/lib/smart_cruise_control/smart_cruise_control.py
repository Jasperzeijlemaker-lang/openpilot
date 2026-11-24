"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import openpilot.cereal.messaging as messaging
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.vision_controller import SmartCruiseControlVision
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.map_controller import SmartCruiseControlMap
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.merge_controller import SmartCruiseControlMerge
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.exit_watch import ExitWatch


class SmartCruiseControl:
  def __init__(self):
    self.vision = SmartCruiseControlVision()
    self.map = SmartCruiseControlMap()
    self.merge = SmartCruiseControlMerge()
    self.exit_watch = ExitWatch()  # watch only: nothing reads it but the log and an alert

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float, v_cruise: float) -> None:
    self.map.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)
    self.vision.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)
    self.merge.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)
    if self.merge.enabled:
      self.exit_watch.update(sm, v_ego)
