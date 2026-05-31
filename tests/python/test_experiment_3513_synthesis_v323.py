"""Tests for scripts/experiment_3513_g_gate_status_synthesis_v323.py.

Spec tracing:
  REQ: ops/north-star.md §2 (stable G1-G4 gate)
  REQ: CLAUDE.md "Depth-Over-Breadth Forcing Function" (relax conditions)
  REQ: CLAUDE.md "Adversarial Artifact Verification" (fabrication gate)
  SCENARIO: flagged_adversarial artifact → skipped, not aggregated in headline
  SCENARIO: P0.1 Route 1 clean → p01_has_clean_verdict=True
  SCENARIO: both routes absent/flagged → p01_has_clean_verdict=False → relax=False
  SCENARIO: G2 external-in-motion + P0.1 clean → depth_forcing_function_can_relax=True
  SCENARIO: random_seed == experiment number → this script avoids the tautology
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    p = ROOT / "scripts" / "experiment_3513_g_gate_status_synthesis_v323.py"
    spec = importlib.util.spec_from_file_location("exp3513", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestLoadArtifact:
    """_load_artifact handles missing, malformed, and flagged files correctly."""

    def test_missing_path_returns_none(self, tmp_path):
        m = _load_module()
        assert m._load_artifact(tmp_path / "nope.json") is None

    def test_valid_artifact_loaded(self, tmp_path):
        m = _load_module()
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"honest_verdict": "complete: ok"}))
        assert m._load_artifact(f) is not None

    def test_flagged_adversarial_true_skipped(self, tmp_path):
        m = _load_module()
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"flagged_adversarial": True, "honest_verdict": "complete: bad"}))
        assert m._load_artifact(f) is None

    def test_flagged_adversarial_false_not_skipped(self, tmp_path):
        m = _load_module()
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"flagged_adversarial": False, "honest_verdict": "complete: ok"}))
        assert m._load_artifact(f) is not None

    def test_invalid_json_returns_none(self, tmp_path):
        m = _load_module()
        f = tmp_path / "exp.json"
        f.write_text("{bad json")
        assert m._load_artifact(f) is None


class TestAvailabilitySummary:
    """_availability_summary reports presence/absence/flag for each depth-block exp."""

    def test_returns_dict_with_all_exp_keys(self):
        m = _load_module()
        summary = m._availability_summary()
        for exp_id in [3505, 3507, 3508, 3509, 3510]:
            assert f"exp{exp_id}" in summary

    def test_missing_reported_as_missing(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "RESULTS", tmp_path)
        summary = m._availability_summary()
        for exp_id in [3505, 3507, 3508, 3509, 3510]:
            assert summary[f"exp{exp_id}"] == "missing"

    def test_flagged_reported_as_skipped(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "RESULTS", tmp_path)
        f = tmp_path / "experiment_3505_something.json"
        f.write_text(json.dumps({"flagged_adversarial": True}))
        summary = m._availability_summary()
        assert summary["exp3505"] == "skipped_flagged_adversarial"

    def test_present_artifact_reported_as_present(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "RESULTS", tmp_path)
        f = tmp_path / "experiment_3505_something.json"
        f.write_text(json.dumps({"honest_verdict": "complete: ok"}))
        summary = m._availability_summary()
        assert summary["exp3505"] == "present"


class TestBuildSynthesisSchema:
    """build_synthesis() emits all required artifact fields with correct types."""

    _REQUIRED_FIELDS = [
        "experiment",
        "honest_verdict",
        "inference_substrate",
        "g1",
        "g2",
        "g3",
        "g4",
        "unmet_gates",
        "p01_route1_sudoku_verdict",
        "p01_route1_solve_rate",
        "p01_route1_exact_baseline_solve_rate",
        "p01_route2_crux_verdict",
        "p01_route2_delta",
        "p01_route2_flip_count",
        "p01_has_clean_verdict",
        "step_to_final_gap_closed_fraction",
        "fr11_beta_law_deployment_validated",
        "g2_package_status",
        "depth_forcing_function_can_relax",
        "gate_status_v323_ready",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ]

    def test_all_required_fields_present(self):
        m = _load_module()
        result = m.build_synthesis()
        for field in self._REQUIRED_FIELDS:
            assert field in result, f"Missing required field: {field}"

    def test_honest_verdict_terminal_prefix(self):
        m = _load_module()
        result = m.build_synthesis()
        v = result["honest_verdict"]
        assert any(
            v.startswith(p)
            for p in ("complete:", "complete_", "success:", "success_", "passed:", "shipped:")
        ), f"honest_verdict does not start with a terminal prefix: {v!r}"

    def test_gate_status_v323_ready_is_true(self):
        m = _load_module()
        result = m.build_synthesis()
        assert result["gate_status_v323_ready"] is True

    def test_inference_substrate_is_aggregation(self):
        m = _load_module()
        result = m.build_synthesis()
        assert result["inference_substrate"] == "aggregation_from_upstream_artifacts"

    def test_unmet_gates_is_list(self):
        m = _load_module()
        result = m.build_synthesis()
        assert isinstance(result["unmet_gates"], list)

    def test_gate_booleans_are_booleans(self):
        m = _load_module()
        result = m.build_synthesis()
        for key in ("g1", "g2", "g3", "g4", "p01_has_clean_verdict",
                    "depth_forcing_function_can_relax"):
            assert isinstance(result[key], bool), f"{key} should be bool, got {type(result[key])}"

    def test_experiment_id_is_3513(self):
        m = _load_module()
        result = m.build_synthesis()
        assert result["experiment"] == 3513

    def test_random_seed_is_not_experiment_number(self):
        """Critical: seed must not equal experiment id (tautology fix for exp3502)."""
        m = _load_module()
        result = m.build_synthesis()
        assert result["random_seed"] != result["experiment"], (
            "random_seed MUST NOT equal experiment id — adversarial_verify flags this as TAUTOLOGY"
        )

    def test_random_seed_is_20260531(self):
        """Fixed seed value is the date constant, not the experiment number."""
        m = _load_module()
        result = m.build_synthesis()
        assert result["random_seed"] == 20260531

    def test_duration_s_is_positive_float(self):
        m = _load_module()
        result = m.build_synthesis()
        assert isinstance(result["duration_s"], float)
        assert result["duration_s"] >= 0.0


class TestP01Route1:
    """P0.1 Route 1 (exp3505 Sudoku) loading and clean-verdict detection."""

    def test_route1_solve_rate_from_artifact(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "experiment_3505_x.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: sudoku_solved",
            "solve_rate": 0.75,
            "exact_baseline_solve_rate": 1.0,
        }))
        monkeypatch.setattr(m, "_EXP3505_PATH", f)
        result = m.build_synthesis()
        assert result["p01_route1_solve_rate"] == pytest.approx(0.75)
        assert result["p01_route1_exact_baseline_solve_rate"] == pytest.approx(1.0)
        assert result["p01_has_clean_verdict"] is True

    def test_route1_none_when_artifact_missing(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "_EXP3505_PATH", tmp_path / "nope.json")
        # Also ensure route 2 is absent so we test route1-only absence
        monkeypatch.setattr(m, "_EXP3507_PATH", tmp_path / "nope2.json")
        result = m.build_synthesis()
        assert result["p01_route1_solve_rate"] is None
        assert result["p01_route1_sudoku_verdict"] is None

    def test_route1_flagged_artifact_treated_as_missing(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "experiment_3505_x.json"
        f.write_text(json.dumps({"flagged_adversarial": True, "solve_rate": 0.99}))
        monkeypatch.setattr(m, "_EXP3505_PATH", f)
        monkeypatch.setattr(m, "_EXP3507_PATH", tmp_path / "nope.json")
        result = m.build_synthesis()
        assert result["p01_route1_solve_rate"] is None
        # With both routes absent/flagged, p01_has_clean_verdict must be False
        assert result["p01_has_clean_verdict"] is False


class TestP01Route2:
    """P0.1 Route 2 (exp3507 in-band) loading — flagged → null."""

    def test_route2_null_when_flagged(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "experiment_3507_x.json"
        f.write_text(json.dumps({
            "flagged_adversarial": True,
            "honest_verdict": "complete: refuted",
            "delta_optimal_vs_self_consistency": 0.5,
        }))
        monkeypatch.setattr(m, "_EXP3507_PATH", f)
        result = m.build_synthesis()
        assert result["p01_route2_crux_verdict"] is None
        assert result["p01_route2_delta"] is None

    def test_route2_values_when_clean(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "experiment_3507_x.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: energy_beats_sc",
            "delta_optimal_vs_self_consistency": 0.08,
            "flip_count_optimal_vs_sc": 12,
        }))
        monkeypatch.setattr(m, "_EXP3507_PATH", f)
        result = m.build_synthesis()
        assert result["p01_route2_crux_verdict"] is not None
        assert result["p01_route2_delta"] == pytest.approx(0.08)
        assert result["p01_route2_flip_count"] == 12


class TestDepthForcingFunctionRelax:
    """depth_forcing_function_can_relax mirrors CLAUDE.md rule.

    Relax := P0.1 clean (at least one route) AND G2 met-or-in-flight.
    """

    def _clean_r1(self, tmp_path):
        """Return a clean, non-flagged Route 1 artifact path in tmp_path."""
        f = tmp_path / "experiment_3505_sudoku.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: sudoku_ok",
            "solve_rate": 0.9,
            "exact_baseline_solve_rate": 1.0,
        }))
        return f

    def _clean_g2(self, tmp_path):
        """Return a clean G2 package artifact path in tmp_path."""
        f = tmp_path / "experiment_3510_g2.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: g2_ready",
            "package_reproduced_auroc": 0.9131,
            "package_auroc_within_ci": True,
            "external_ask_workflow_path": "/some/path/workflow.sh",
        }))
        return f

    def test_relax_true_with_clean_p01_and_g2_in_flight(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "_EXP3505_PATH", self._clean_r1(tmp_path))
        monkeypatch.setattr(m, "_EXP3507_PATH", tmp_path / "no_r2.json")
        monkeypatch.setattr(m, "_EXP3510_PATH", self._clean_g2(tmp_path))
        # Patch other artifact paths to missing so they don't interfere
        for attr in ("_EXP3508_PATH", "_EXP3509_PATH"):
            monkeypatch.setattr(m, attr, tmp_path / "absent.json")
        result = m.build_synthesis()
        assert result["depth_forcing_function_can_relax"] is True

    def test_relax_false_when_p01_missing(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "_EXP3505_PATH", tmp_path / "no_r1.json")
        monkeypatch.setattr(m, "_EXP3507_PATH", tmp_path / "no_r2.json")
        monkeypatch.setattr(m, "_EXP3510_PATH", self._clean_g2(tmp_path))
        for attr in ("_EXP3508_PATH", "_EXP3509_PATH"):
            monkeypatch.setattr(m, attr, tmp_path / "absent.json")
        result = m.build_synthesis()
        assert result["p01_has_clean_verdict"] is False
        assert result["depth_forcing_function_can_relax"] is False

    def test_relax_false_when_p01_clean_but_g2_not_in_flight(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "_EXP3505_PATH", self._clean_r1(tmp_path))
        monkeypatch.setattr(m, "_EXP3507_PATH", tmp_path / "no_r2.json")
        # G2 artifact: auroc_within_ci=False (regression failed) + no external_ask_workflow
        f = tmp_path / "experiment_3510_g2.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: regression_fail",
            "package_reproduced_auroc": 0.85,
            "package_auroc_within_ci": False,
        }))
        monkeypatch.setattr(m, "_EXP3510_PATH", f)
        for attr in ("_EXP3508_PATH", "_EXP3509_PATH"):
            monkeypatch.setattr(m, attr, tmp_path / "absent.json")
        result = m.build_synthesis()
        assert result["p01_has_clean_verdict"] is True
        assert result["depth_forcing_function_can_relax"] is False
