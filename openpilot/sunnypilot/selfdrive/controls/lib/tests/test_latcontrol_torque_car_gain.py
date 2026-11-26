from openpilot.sunnypilot.selfdrive.controls.lib import latcontrol_torque_car_gain as g


class TestKpScale:
  def test_ioniq_6_is_gentler_and_every_other_car_is_unchanged(self):
    assert g.kp_scale("HYUNDAI_IONIQ_6") == 0.6
    for car in ("HYUNDAI_IONIQ_5", "TOYOTA_COROLLA_TSS2", "", None, 5):
      assert g.kp_scale(car) == 1.0

  def test_table_values_are_sane(self):
    for scale in g.KP_SCALE.values():
      assert 0.4 <= scale <= 1.0

  def test_a_bad_table_entry_falls_back_to_no_change(self, monkeypatch):
    for bad in (0., 0.1, 3., float('nan'), "x", None):
      monkeypatch.setitem(g.KP_SCALE, "HYUNDAI_IONIQ_6", bad)
      assert g.kp_scale("HYUNDAI_IONIQ_6") == 1.0

  def test_controller_uses_it(self):
    # the version 0 controller builds its gain table through kp_scale
    import pathlib
    src = (pathlib.Path(g.__file__).parent / "latcontrol_torque_v0.py").read_text()
    assert "scale = kp_scale(CP.carFingerprint)" in src and "[kp * scale for kp in KP_INTERP]" in src
