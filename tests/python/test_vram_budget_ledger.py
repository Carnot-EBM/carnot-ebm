"""Tests for VRAMBudgetLedger and VRAMForecast.

Spec: REQ-INFRA-054, REQ-INFRA-055, REQ-INFRA-056,
      SCENARIO-INFRA-062, SCENARIO-INFRA-063, SCENARIO-INFRA-064
"""

from __future__ import annotations

import pytest
import yaml

from carnot.pipeline.vram_budget_ledger import VRAMBudgetLedger, VRAMForecast


# ---------------------------------------------------------------------------
# VRAMForecast tests
# ---------------------------------------------------------------------------


class TestVRAMForecast:
    def test_headroom_gb_positive_when_feasible(self):
        # SCENARIO-INFRA-062: conductor=9, model=9, total=24 → avail=15, headroom=6
        forecast = VRAMForecast(
            exp_id="exp502",
            is_feasible=True,
            required_gb=9.0,
            available_gb=15.0,
            blocking_experiment=None,
        )
        assert forecast.headroom_gb == pytest.approx(6.0)

    def test_headroom_gb_negative_when_not_feasible(self):
        # SCENARIO-INFRA-063: conductor=9, model=16, total=24 → avail=15, headroom=-1
        forecast = VRAMForecast(
            exp_id="exp502",
            is_feasible=False,
            required_gb=16.0,
            available_gb=15.0,
            blocking_experiment="exp502",
        )
        assert forecast.headroom_gb == pytest.approx(-1.0)

    def test_to_dict_contains_all_fields(self):
        forecast = VRAMForecast(
            exp_id="exp502",
            is_feasible=True,
            required_gb=9.0,
            available_gb=15.0,
            blocking_experiment=None,
        )
        d = forecast.to_dict()
        assert d["exp_id"] == "exp502"
        assert d["is_feasible"] is True
        assert d["required_gb"] == pytest.approx(9.0)
        assert d["available_gb"] == pytest.approx(15.0)
        assert d["headroom_gb"] == pytest.approx(6.0)
        assert d["blocking_experiment"] is None

    def test_to_dict_blocking_experiment_when_not_feasible(self):
        forecast = VRAMForecast(
            exp_id="exp503",
            is_feasible=False,
            required_gb=18.0,
            available_gb=15.0,
            blocking_experiment="exp503",
        )
        d = forecast.to_dict()
        assert d["blocking_experiment"] == "exp503"
        assert d["is_feasible"] is False


# ---------------------------------------------------------------------------
# VRAMBudgetLedger tests
# ---------------------------------------------------------------------------


class TestVRAMBudgetLedger:
    def test_available_gb_default(self):
        # Default: conductor=9, total=24, avail=15
        ledger = VRAMBudgetLedger()
        assert ledger.available_gb == pytest.approx(15.0)

    def test_available_gb_cpu_routing(self):
        # SCENARIO-INFRA-064: conductor=0 (CPU routing), total=24, avail=24
        ledger = VRAMBudgetLedger(conductor_vram_gb=0.0, gpu_total_gb=24.0)
        assert ledger.available_gb == pytest.approx(24.0)

    def test_feasible_when_conductor_9_model_9_total_24(self):
        # SCENARIO-INFRA-062: conductor=9 + model=9 < total=24 → feasible
        ledger = VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)
        ledger.add_experiment("exp507", required_gb=9.0)
        forecast = ledger.check_feasibility("exp507")
        assert forecast.is_feasible is True
        assert forecast.blocking_experiment is None
        assert forecast.required_gb == pytest.approx(9.0)
        assert forecast.available_gb == pytest.approx(15.0)

    def test_not_feasible_when_conductor_9_model_16_total_24(self):
        # SCENARIO-INFRA-063: conductor=9 + model=16 > total=24 → not feasible
        ledger = VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)
        ledger.add_experiment("exp502", required_gb=16.0)
        forecast = ledger.check_feasibility("exp502")
        assert forecast.is_feasible is False
        assert forecast.blocking_experiment == "exp502"
        assert forecast.headroom_gb == pytest.approx(-1.0)

    def test_cpu_routing_more_feasible(self):
        # SCENARIO-INFRA-064: CPU-routed conductor (conductor_vram_gb=0.0) frees full GPU
        ledger_gpu = VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)
        ledger_gpu.add_experiment("exp502", required_gb=18.0)
        ledger_cpu = VRAMBudgetLedger(conductor_vram_gb=0.0, gpu_total_gb=24.0)
        ledger_cpu.add_experiment("exp502", required_gb=18.0)

        forecast_gpu = ledger_gpu.check_feasibility("exp502")
        forecast_cpu = ledger_cpu.check_feasibility("exp502")

        assert forecast_gpu.is_feasible is False  # 18 > 15
        assert forecast_cpu.is_feasible is True   # 18 <= 24

    def test_check_all_returns_all_experiments(self):
        ledger = VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)
        ledger.add_experiment("exp502", required_gb=18.0)
        ledger.add_experiment("exp507", required_gb=9.0)
        ledger.add_experiment("exp511", required_gb=2.0)

        forecasts = ledger.check_all()
        assert len(forecasts) == 3
        ids = [f.exp_id for f in forecasts]
        assert ids == ["exp502", "exp507", "exp511"]

    def test_check_all_mixed_feasibility(self):
        ledger = VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)
        ledger.add_experiment("exp502", required_gb=18.0)  # not feasible (15 avail)
        ledger.add_experiment("exp507", required_gb=9.0)   # feasible
        forecasts = ledger.check_all()
        assert forecasts[0].is_feasible is False
        assert forecasts[1].is_feasible is True

    def test_check_feasibility_key_error_for_unknown(self):
        ledger = VRAMBudgetLedger()
        with pytest.raises(KeyError):
            ledger.check_feasibility("exp_unknown")

    def test_to_yaml_contains_expected_keys(self):
        ledger = VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)
        ledger.add_experiment("exp502", required_gb=18.0)
        yml = ledger.to_yaml()
        data = yaml.safe_load(yml)
        assert data["conductor_vram_gb"] == pytest.approx(9.0)
        assert data["gpu_total_gb"] == pytest.approx(24.0)
        assert data["available_gb"] == pytest.approx(15.0)
        assert data["experiments"]["exp502"] == pytest.approx(18.0)

    def test_to_yaml_cpu_routing(self):
        ledger = VRAMBudgetLedger(conductor_vram_gb=0.0, gpu_total_gb=24.0)
        ledger.add_experiment("exp502", required_gb=18.0)
        yml = ledger.to_yaml()
        data = yaml.safe_load(yml)
        assert data["conductor_vram_gb"] == pytest.approx(0.0)
        assert data["available_gb"] == pytest.approx(24.0)

    def test_exact_boundary_is_feasible(self):
        # required_gb == available_gb exactly → feasible (<=)
        ledger = VRAMBudgetLedger(conductor_vram_gb=9.0, gpu_total_gb=24.0)
        ledger.add_experiment("exp_boundary", required_gb=15.0)
        forecast = ledger.check_feasibility("exp_boundary")
        assert forecast.is_feasible is True
        assert forecast.headroom_gb == pytest.approx(0.0)
        assert forecast.blocking_experiment is None
