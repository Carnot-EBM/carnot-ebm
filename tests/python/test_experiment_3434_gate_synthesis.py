"""Tests for scripts/experiment_3434_g_gate_status_synthesis_v316.py.

These tests verify the G1-G4 synthesis logic: artifact loading, the
depth-forcing-function relax decision, and required output field schema.

Spec tracing:
  REQ: ops/north-star.md §2 (stable G1-G4 gate)
  REQ: CLAUDE.md "Depth-Over-Breadth Forcing Function" (relax conditions)
  SCENARIO: depth-block artifacts present → correct gate booleans emitted
  SCENARIO: P0.1 artifact missing → depth_forcing_function_can_relax=False
  SCENARIO: flagged_adversarial=True artifact → skipped, not aggregated
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    p = ROOT / "scripts" / "experiment_3434_g_gate_status_synthesis_v316.py"
    spec = importlib.util.spec_from_file_location("exp3434", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    # Do NOT add to sys.modules at the top level — each test loads it fresh
    # to avoid cross-test contamination of patched constants.
    spec.loader.exec_module(m)
    return m


class TestLoadArtifact:
    """_load_artifact correctly handles missing, invalid, and flagged files."""

    def _mod(self):
        return _load_module()

    def test_missing_path_returns_none(self, tmp_path):
        m = self._mod()
        assert m._load_artifact(tmp_path / "nonexistent.json") is None

    def test_valid_artifact_loaded(self, tmp_path):
        m = self._mod()
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"honest_verdict": "complete: ok"}))
        result = m._load_artifact(f)
        assert result is not None
        assert result["honest_verdict"] == "complete: ok"

    def test_flagged_adversarial_true_skipped(self, tmp_path):
        m = self._mod()
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"flagged_adversarial": True, "honest_verdict": "complete: bad"}))
        assert m._load_artifact(f) is None

    def test_flagged_adversarial_false_not_skipped(self, tmp_path):
        m = self._mod()
        f = tmp_path / "exp.json"
        f.write_text(json.dumps({"flagged_adversarial": False, "honest_verdict": "complete: ok"}))
        result = m._load_artifact(f)
        assert result is not None

    def test_invalid_json_returns_none(self, tmp_path):
        m = self._mod()
        f = tmp_path / "exp.json"
        f.write_text("{bad json")
        assert m._load_artifact(f) is None


class TestBuildSynthesisSchema:
    """build_synthesis() emits all required artifact fields."""

    _REQUIRED_FIELDS = [
        "experiment",
        "honest_verdict",
        "g1",
        "g2",
        "g3",
        "g4",
        "unmet_gates",
        "p0_1_v2_verdict",
        "energy_vs_self_consistency_delta",
        "depth_forcing_function_can_relax",
        "gate_status_v316_ready",
        "inference_substrate",
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
            v.startswith(p) for p in ("complete:", "complete_", "success:", "success_", "passed:", "shipped:")
        ), f"honest_verdict does not start with a terminal prefix: {v!r}"

    def test_gate_status_v316_ready_is_true(self):
        m = _load_module()
        result = m.build_synthesis()
        assert result["gate_status_v316_ready"] is True

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
        for key in ("g1", "g2", "g3", "g4", "depth_forcing_function_can_relax"):
            assert isinstance(result[key], bool), f"{key} should be bool, got {type(result[key])}"

    def test_experiment_id_is_3434(self):
        m = _load_module()
        result = m.build_synthesis()
        assert result["experiment"] == 3434


class TestDepthForcingFunctionRelax:
    """depth_forcing_function_can_relax logic mirrors the CLAUDE.md rule.

    Relax := P0.1_clean AND (G2_met OR G2_in_flight).
    """

    def test_relax_false_when_p01_missing(self, tmp_path, monkeypatch):
        """If the P0.1 artifact is missing, relax must be False."""
        m = _load_module()
        # Point P0.1 path to a nonexistent file
        monkeypatch.setattr(m, "_P01_V2_PATH", tmp_path / "no_p01.json")
        result = m.build_synthesis()
        assert result["depth_forcing_function_can_relax"] is False

    def test_relax_false_when_p01_flagged(self, tmp_path, monkeypatch):
        """If the P0.1 artifact is flagged_adversarial=True, relax must be False."""
        m = _load_module()
        f = tmp_path / "exp3426.json"
        f.write_text(json.dumps({"flagged_adversarial": True, "honest_verdict": "complete: x"}))
        monkeypatch.setattr(m, "_P01_V2_PATH", f)
        result = m.build_synthesis()
        assert result["depth_forcing_function_can_relax"] is False

    def test_relax_false_when_p01_clean_but_g2_not_inflight(self, tmp_path, monkeypatch):
        """P0.1 clean but G2 not met and not in-flight → still False."""
        m = _load_module()
        # Provide a clean P0.1 artifact
        f = tmp_path / "exp3426.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: energy_matches",
            "delta_energy_vs_self_consistency": 0.0,
        }))
        monkeypatch.setattr(m, "_P01_V2_PATH", f)
        # Ensure G2 cleanroom shows not-in-flight
        g2f = tmp_path / "exp3430.json"
        g2f.write_text(json.dumps({"g2_status": "ci_gate_failed", "reproduced_in_ci": False}))
        monkeypatch.setattr(m, "_G2_CLEANROOM_PATH", g2f)
        result = m.build_synthesis()
        # G2 is still unmet (publication_gate_state.json has g2_independent_reproducer=false)
        # and the cleanroom g2_in_flight is False
        assert result["depth_forcing_function_can_relax"] is False

    def test_relax_true_when_p01_clean_and_g2_inflight(self, tmp_path, monkeypatch):
        """P0.1 clean AND G2 in-flight (reproduced_in_ci=True) → relax=True."""
        m = _load_module()
        f = tmp_path / "exp3426.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: energy_beats_sc",
            "delta_energy_vs_self_consistency": 0.05,
        }))
        monkeypatch.setattr(m, "_P01_V2_PATH", f)
        g2f = tmp_path / "exp3430.json"
        g2f.write_text(json.dumps({"g2_status": "cleanroom_validated", "reproduced_in_ci": True}))
        monkeypatch.setattr(m, "_G2_CLEANROOM_PATH", g2f)
        result = m.build_synthesis()
        assert result["depth_forcing_function_can_relax"] is True


class TestEnergyVsScDelta:
    """energy_vs_self_consistency_delta comes from exp3426."""

    def test_delta_from_p01_artifact(self, tmp_path, monkeypatch):
        m = _load_module()
        f = tmp_path / "exp3426.json"
        f.write_text(json.dumps({
            "honest_verdict": "complete: ok",
            "delta_energy_vs_self_consistency": 0.123,
        }))
        monkeypatch.setattr(m, "_P01_V2_PATH", f)
        result = m.build_synthesis()
        assert result["energy_vs_self_consistency_delta"] == pytest.approx(0.123)

    def test_delta_none_when_p01_missing(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "_P01_V2_PATH", tmp_path / "nope.json")
        result = m.build_synthesis()
        assert result["energy_vs_self_consistency_delta"] is None


class TestAvailabilitySummary:
    """_availability_summary reports presence/absence of depth-block artifacts."""

    def test_returns_dict_with_all_exp_keys(self):
        m = _load_module()
        summary = m._availability_summary()
        for exp_id in [3426, 3427, 3428, 3429, 3430]:
            assert f"exp{exp_id}" in summary

    def test_missing_reported_as_missing(self, tmp_path, monkeypatch):
        m = _load_module()
        # Point RESULTS to an empty temp dir so nothing is found
        monkeypatch.setattr(m, "RESULTS", tmp_path)
        summary = m._availability_summary()
        for exp_id in [3426, 3427, 3428, 3429, 3430]:
            assert summary[f"exp{exp_id}"] == "missing"

    def test_flagged_reported_as_skipped(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "RESULTS", tmp_path)
        f = tmp_path / "experiment_3426_something.json"
        f.write_text(json.dumps({"flagged_adversarial": True}))
        summary = m._availability_summary()
        assert summary["exp3426"] == "skipped_flagged_adversarial"

    def test_present_artifact_reported_as_present(self, tmp_path, monkeypatch):
        m = _load_module()
        monkeypatch.setattr(m, "RESULTS", tmp_path)
        f = tmp_path / "experiment_3426_something.json"
        f.write_text(json.dumps({"honest_verdict": "complete: ok"}))
        summary = m._availability_summary()
        assert summary["exp3426"] == "present"
