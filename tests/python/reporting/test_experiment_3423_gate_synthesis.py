"""Tests for scripts/experiment_3423_g_gate_status_synthesis_v315.py.

REQ-GATE-001: G1–G4 gate status synthesis must read depth-block artifacts
  and emit a structured record with all required schema fields.
SCENARIO-GATE-001: exp3312 present with validated verdict → p0_1_verdict set,
  depth_forcing_function_can_relax may be True when G2 harness also in-flight.
SCENARIO-GATE-002: Missing depth-block artifacts handled gracefully (return None).
SCENARIO-GATE-003: G2 in-flight detection requires harness path + internal CI
  confirmation + advanced g2_status.
SCENARIO-GATE-004: All required schema fields present in synthesis output.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts/ to path so we can import the module under test
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import experiment_3423_g_gate_status_synthesis_v315 as synth  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────


def _fake_exp3312(*, validated: bool = True) -> dict:
    """Minimal exp3312 artifact."""
    verdict = (
        "complete: energy_descent_beats_ar_premise_validated"
        if validated
        else "complete: energy_descent_premise_not_viable"
    )
    return {
        "experiment": 3312,
        "honest_verdict": verdict,
        "g1_premise_viable": validated,
        "g2_premise_validated": validated,
        "accuracy_delta": 0.09 if validated else -0.02,
        "ar_baseline_accuracy": 0.75,
        "energy_descent_accuracy": 0.84 if validated else 0.73,
        "self_consistency_accuracy": 0.895,
        "n_problems": 200,
        "paired_significance": {
            "p_value": 0.033,
            "bootstrap_delta_ci95": [0.01, 0.165],
        },
    }


def _fake_exp3419(*, in_ci: bool = True, advanced: bool = True) -> dict:
    """Minimal exp3419 artifact."""
    return {
        "experiment": 3419,
        "g2_independent_reproducer": False,
        "g2_status": "advanced_turnkey_harness_internal_confirmation" if advanced else "not_started",
        "harness_path": "scripts/reproduce_fover_headline.py",
        "condition_a_auroc_reproduced": 0.9131,
        "condition_a_in_published_ci": in_ci,
    }


def _fake_gate_result(*, g2: bool = False, g3: bool = True) -> dict:
    """Fake publication_gate.evaluate() result."""
    return {
        "paper_ready": False,
        "gates": {
            "G1": {"pass": True, "detail": "FoVer 0.9131 present", "source": "exp2850.json"},
            "G2": {"pass": g2, "detail": "no independent reproducer" if not g2 else "confirmed"},
            "G3": {"pass": g3, "detail": "no forbidden phrasings", "hits": []},
            "G4": {"pass": True, "detail": "seed+checksum present", "source": "exp2850.json"},
        },
        "unmet_gates": (["G2"] if not g2 else []) + (["G3"] if not g3 else []),
    }


# ── _extract_p0_1_verdict ───────────────────────────────────────────────────


def test_extract_p0_1_verdict_validated():
    # REQ-GATE-001 SCENARIO-GATE-001
    exp = _fake_exp3312(validated=True)
    result = synth._extract_p0_1_verdict(exp)
    assert result.startswith("complete:")
    assert "validated" in result


def test_extract_p0_1_verdict_not_viable():
    # SCENARIO-GATE-001: negative verdict also has a terminal prefix
    exp = _fake_exp3312(validated=False)
    exp["honest_verdict"] = "complete: energy_descent_premise_not_viable"
    result = synth._extract_p0_1_verdict(exp)
    assert result.startswith("complete:")


def test_extract_p0_1_verdict_missing_artifact():
    # SCENARIO-GATE-002: None artifact → "not_run"
    result = synth._extract_p0_1_verdict(None)
    assert result == "not_run"


def test_extract_p0_1_verdict_no_honest_verdict_field_validated():
    # Fallback path: no honest_verdict key but g2_premise_validated=True
    exp = {k: v for k, v in _fake_exp3312().items() if k != "honest_verdict"}
    result = synth._extract_p0_1_verdict(exp)
    assert "validated" in result


def test_extract_p0_1_verdict_no_honest_verdict_field_not_viable():
    # Fallback path: g1_premise_viable=False
    exp = {k: v for k, v in _fake_exp3312(validated=False).items() if k != "honest_verdict"}
    exp["g1_premise_viable"] = False
    exp["g2_premise_validated"] = False
    result = synth._extract_p0_1_verdict(exp)
    assert "not_viable" in result


# ── _g2_in_flight ───────────────────────────────────────────────────────────


def test_g2_in_flight_fully_advanced():
    # REQ-GATE-001 SCENARIO-GATE-003
    exp = _fake_exp3419(in_ci=True, advanced=True)
    assert synth._g2_in_flight(exp) is True


def test_g2_in_flight_not_advanced():
    # advanced=False → not in-flight
    exp = _fake_exp3419(in_ci=True, advanced=False)
    assert synth._g2_in_flight(exp) is False


def test_g2_in_flight_no_ci_confirmation():
    # in_ci=False → not in-flight (internal confirmation required)
    exp = _fake_exp3419(in_ci=False, advanced=True)
    assert synth._g2_in_flight(exp) is False


def test_g2_in_flight_none_artifact():
    # SCENARIO-GATE-002: None → False
    assert synth._g2_in_flight(None) is False


def test_g2_in_flight_no_harness_path():
    # missing harness_path → not in-flight
    exp = _fake_exp3419()
    del exp["harness_path"]
    assert synth._g2_in_flight(exp) is False


# ── compute_synthesis ───────────────────────────────────────────────────────


def _run_synthesis(depth_block_override: dict | None = None) -> dict:
    """Helper: run compute_synthesis with mocked publication_gate."""
    depth_block = {
        "exp3312": _fake_exp3312(),
        "exp3313": {"experiment": 3313, "honest_verdict": "complete: autopsy_done"},
        "exp3417": None,
        "exp3418": None,
        "exp3419": _fake_exp3419(),
    }
    if depth_block_override:
        depth_block.update(depth_block_override)

    mock_pg = MagicMock()
    mock_pg.evaluate.return_value = _fake_gate_result()

    with patch.dict(sys.modules, {"publication_gate": mock_pg}):
        return synth.compute_synthesis(depth_block)


def test_compute_synthesis_required_fields_present():
    # SCENARIO-GATE-004: all required schema fields must be present
    result = _run_synthesis()
    required = {
        "honest_verdict",
        "g1", "g2", "g3", "g4",
        "unmet_gates",
        "p0_1_verdict",
        "depth_forcing_function_can_relax",
        "gate_status_v315_ready",
    }
    missing = required - set(result)
    assert not missing, f"Missing required fields: {missing}"


def test_compute_synthesis_honest_verdict_has_terminal_prefix():
    result = _run_synthesis()
    assert result["honest_verdict"].startswith("complete:")


def test_compute_synthesis_g1_true():
    result = _run_synthesis()
    assert result["g1"] is True


def test_compute_synthesis_g2_false_external_pending():
    # G2 is False because no external reproducer is recorded in gate_state.json
    result = _run_synthesis()
    assert result["g2"] is False


def test_compute_synthesis_unmet_gates_is_list():
    result = _run_synthesis()
    assert isinstance(result["unmet_gates"], list)
    assert "G2" in result["unmet_gates"]


def test_compute_synthesis_gate_status_v315_ready():
    result = _run_synthesis()
    assert result["gate_status_v315_ready"] is True


def test_compute_synthesis_depth_forcing_function_relaxes_when_conditions_met():
    # P0.1 has a verdict AND G2 harness is in-flight → can relax
    result = _run_synthesis()
    assert result["depth_forcing_function_can_relax"] is True


def test_compute_synthesis_depth_function_no_relax_missing_p0_1():
    # Missing exp3312 → p0_1 not run → cannot relax
    result = _run_synthesis({"exp3312": None})
    assert result["depth_forcing_function_can_relax"] is False


def test_compute_synthesis_depth_function_no_relax_missing_g2_harness():
    # Missing exp3419 → no in-flight harness → cannot relax
    result = _run_synthesis({"exp3419": None})
    assert result["depth_forcing_function_can_relax"] is False


def test_compute_synthesis_p0_1_verdict_from_exp3312():
    result = _run_synthesis()
    assert "complete:" in result["p0_1_verdict"]
    assert "validated" in result["p0_1_verdict"]


def test_compute_synthesis_depth_block_presence_reported():
    result = _run_synthesis()
    presence = result["depth_block_artifacts_present"]
    assert presence["exp3312"] is True
    assert presence["exp3419"] is True
    assert presence["exp3417"] is False
    assert presence["exp3418"] is False


# ── _build_p0_1_summary ─────────────────────────────────────────────────────


def test_build_p0_1_summary_present_artifact():
    exp = _fake_exp3312()
    s = synth._build_p0_1_summary(exp)
    assert "exp3312" in s
    assert "0.09" in s or "+0.09" in s or "0.090" in s


def test_build_p0_1_summary_missing_artifact():
    s = synth._build_p0_1_summary(None)
    assert "not_run" in s or "not run" in s


# ── _load_json ──────────────────────────────────────────────────────────────


def test_load_json_none_filename():
    # SCENARIO-GATE-002: None filename → None
    assert synth._load_json(None) is None


def test_load_json_missing_file(tmp_path, monkeypatch):
    # Non-existent file → None
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth._load_json("does_not_exist.json") is None
