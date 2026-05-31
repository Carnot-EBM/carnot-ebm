"""Tests for experiment_3502_g_gate_status_synthesis_v322.py.

REQ-GATE-001: Every capstone must emit g1..g4 booleans + unmet_gates.
REQ-GATE-002: honest_verdict must start with 'complete:'.
REQ-GATE-003: inference_substrate must declare aggregation_from_upstream_artifacts.
REQ-GATE-004: Absent or flagged_adversarial artifacts must contribute null values, not fail.
REQ-GATE-005: depth_forcing_function_can_relax = p01_has_clean_verdict AND G2-external-in-motion.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test.
import importlib.util

_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "experiment_3502_g_gate_status_synthesis_v322.py"
_spec = importlib.util.spec_from_file_location("exp3502", _SCRIPT)
exp3502 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(exp3502)  # type: ignore[union-attr]

load_artifact = exp3502.load_artifact
run_gate = exp3502.run_gate
_is_blocked_verdict = exp3502._is_blocked_verdict
_extract_calibration_diagnosis = exp3502._extract_calibration_diagnosis
_extract_fr11_law = exp3502._extract_fr11_law
main = exp3502.main

REQUIRED_FIELDS = [
    "honest_verdict",
    "inference_substrate",
    "g1",
    "g2",
    "g3",
    "g4",
    "unmet_gates",
    "p01_route1_sudoku_verdict",
    "p01_route1_solve_rate",
    "p01_route2_crux_verdict",
    "p01_route2_delta",
    "p01_route2_flip_count",
    "p01_has_clean_verdict",
    "calibration_diagnosis",
    "fr11_beta_min_lambda_min_law",
    "g2_package_status",
    "depth_forcing_function_can_relax",
    "gate_status_v322_ready",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
]


# ── load_artifact tests (REQ-GATE-004) ────────────────────────────────────────

def test_load_artifact_absent_returns_none(tmp_path: Path) -> None:
    """REQ-GATE-004: absent artifact path returns None, not an exception."""
    result = load_artifact(tmp_path / "nonexistent.json")
    assert result is None


def test_load_artifact_flagged_adversarial_returns_none(tmp_path: Path) -> None:
    """REQ-GATE-004: artifact with flagged_adversarial=true is excluded."""
    p = tmp_path / "flagged.json"
    p.write_text(json.dumps({"flagged_adversarial": True, "honest_verdict": "complete: ok"}))
    assert load_artifact(p) is None


def test_load_artifact_flagged_false_returns_dict(tmp_path: Path) -> None:
    """REQ-GATE-004: artifact with flagged_adversarial=false is loaded normally."""
    payload = {"flagged_adversarial": False, "honest_verdict": "complete: ok", "val": 42}
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(payload))
    result = load_artifact(p)
    assert result is not None
    assert result["val"] == 42


def test_load_artifact_no_flag_field_returns_dict(tmp_path: Path) -> None:
    """REQ-GATE-004: artifact without flagged_adversarial field is loaded normally."""
    payload = {"honest_verdict": "complete: x", "n": 5}
    p = tmp_path / "clean.json"
    p.write_text(json.dumps(payload))
    assert load_artifact(p) == payload


def test_load_artifact_malformed_json_returns_none(tmp_path: Path) -> None:
    """REQ-GATE-004: malformed JSON returns None gracefully."""
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    assert load_artifact(p) is None


# ── _is_blocked_verdict tests ─────────────────────────────────────────────────

def test_is_blocked_verdict_none() -> None:
    """None verdict is treated as blocked (missing = can't tell = not clean)."""
    assert _is_blocked_verdict(None) is True


def test_is_blocked_verdict_complete_blocked() -> None:
    """'complete: blocked_...' is a blocked verdict despite the terminal prefix."""
    assert _is_blocked_verdict("complete: blocked_kona_failure") is True


def test_is_blocked_verdict_clean() -> None:
    """'complete: beta_min_law_established' is a clean (non-blocked) verdict."""
    assert _is_blocked_verdict("complete: beta_min_law_established") is False


def test_is_blocked_verdict_empty_string() -> None:
    """Empty string verdict is treated as blocked."""
    assert _is_blocked_verdict("") is True


# ── _extract_calibration_diagnosis tests ─────────────────────────────────────

def test_extract_calibration_diagnosis_full() -> None:
    """Extracts diagnosis from a full exp3497 payload."""
    d = {
        "honest_verdict": "complete: domain_shift_was_the_cause",
        "step_vs_final_auroc_gap": 0.138,
        "mathaware_recalibrated_correctness_auroc": 0.625,
    }
    diag = _extract_calibration_diagnosis(d)
    assert diag is not None
    assert "domain_shift" in diag
    assert "0.13800" in diag
    assert "0.625" in diag


def test_extract_calibration_diagnosis_minimal() -> None:
    """Works with only honest_verdict present."""
    d = {"honest_verdict": "complete: ok"}
    diag = _extract_calibration_diagnosis(d)
    assert diag == "ok"


def test_extract_calibration_diagnosis_empty() -> None:
    """Returns None-ish string for empty dict (no parts)."""
    diag = _extract_calibration_diagnosis({})
    assert diag is None or diag == ""


# ── _extract_fr11_law tests ───────────────────────────────────────────────────

def test_extract_fr11_law_recommended_rule() -> None:
    """Prefers recommended_phase5_rule string."""
    d = {
        "recommended_phase5_rule": "beta_min = -0.30 + 1.85 * lambda_min",
        "beta_min_lambda_min_fit": {"slope": 1.85, "intercept": -0.30},
    }
    law = _extract_fr11_law(d)
    assert law == "beta_min = -0.30 + 1.85 * lambda_min"


def test_extract_fr11_law_fit_fallback() -> None:
    """Falls back to constructing from beta_min_lambda_min_fit when no rule string."""
    d = {
        "beta_min_lambda_min_fit": {
            "slope": 1.8461,
            "intercept": -0.3001,
            "r_squared": 0.9886,
        }
    }
    law = _extract_fr11_law(d)
    assert law is not None
    assert "lambda_min" in law
    assert "1.8461" in law


def test_extract_fr11_law_absent() -> None:
    """Returns None when neither field is present."""
    assert _extract_fr11_law({}) is None


# ── run_gate smoke test ───────────────────────────────────────────────────────

def test_run_gate_returns_dict_with_gates() -> None:
    """run_gate() returns a dict with 'gates' key containing G1-G4."""
    result = run_gate()
    assert isinstance(result, dict)
    assert "gates" in result
    for key in ("G1", "G2", "G3", "G4"):
        assert key in result["gates"]
        assert "pass" in result["gates"][key]


# ── main() / output artifact tests (REQ-GATE-001..005) ───────────────────────

def test_main_writes_valid_json(tmp_path: Path) -> None:
    """REQ-GATE-001..005: main() writes a valid JSON artifact with all required fields."""
    out = tmp_path / "experiment_3502_g_gate_status_synthesis_v322.json"

    with patch.object(exp3502, "OUTPUT_PATH", out):
        main()

    assert out.exists()
    data = json.loads(out.read_text())
    for field in REQUIRED_FIELDS:
        assert field in data, f"Missing required field: {field}"


def test_honest_verdict_starts_with_complete(tmp_path: Path) -> None:
    """REQ-GATE-002: honest_verdict must start with 'complete:'."""
    out = tmp_path / "experiment_3502_g_gate_status_synthesis_v322.json"
    with patch.object(exp3502, "OUTPUT_PATH", out):
        main()
    data = json.loads(out.read_text())
    assert data["honest_verdict"].startswith("complete:")


def test_inference_substrate_is_aggregation(tmp_path: Path) -> None:
    """REQ-GATE-003: inference_substrate must be aggregation_from_upstream_artifacts."""
    out = tmp_path / "experiment_3502_g_gate_status_synthesis_v322.json"
    with patch.object(exp3502, "OUTPUT_PATH", out):
        main()
    data = json.loads(out.read_text())
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"


def test_gate_status_v322_ready_is_true(tmp_path: Path) -> None:
    """gate_status_v322_ready is always True (terminal completion flag)."""
    out = tmp_path / "experiment_3502_g_gate_status_synthesis_v322.json"
    with patch.object(exp3502, "OUTPUT_PATH", out):
        main()
    data = json.loads(out.read_text())
    assert data["gate_status_v322_ready"] is True


def test_random_seed_is_experiment_number(tmp_path: Path) -> None:
    """random_seed is fixed at 3502 (the experiment number) for determinism."""
    out = tmp_path / "experiment_3502_g_gate_status_synthesis_v322.json"
    with patch.object(exp3502, "OUTPUT_PATH", out):
        main()
    data = json.loads(out.read_text())
    assert data["random_seed"] == 3502


def test_depth_forcing_function_type(tmp_path: Path) -> None:
    """REQ-GATE-005: depth_forcing_function_can_relax is a boolean."""
    out = tmp_path / "experiment_3502_g_gate_status_synthesis_v322.json"
    with patch.object(exp3502, "OUTPUT_PATH", out):
        main()
    data = json.loads(out.read_text())
    assert isinstance(data["depth_forcing_function_can_relax"], bool)


def test_main_skips_absent_artifacts(tmp_path: Path) -> None:
    """REQ-GATE-004: main() completes even when upstream artifacts are absent."""
    out = tmp_path / "experiment_3502_g_gate_status_synthesis_v322.json"
    empty_dir = tmp_path / "results"
    empty_dir.mkdir()

    with (
        patch.object(exp3502, "RESULTS_DIR", empty_dir),
        patch.object(exp3502, "OUTPUT_PATH", out),
    ):
        main()  # must not raise

    data = json.loads(out.read_text())
    # With no upstream artifacts all P0.1 fields should be null.
    assert data["p01_route1_sudoku_verdict"] is None
    assert data["p01_route1_solve_rate"] is None
    assert data["p01_route2_crux_verdict"] is None
    assert data["p01_has_clean_verdict"] is False


def test_main_excludes_flagged_artifact(tmp_path: Path) -> None:
    """REQ-GATE-004: flagged_adversarial artifact is excluded (treated as absent)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Write a flagged exp3494
    flagged = {
        "flagged_adversarial": True,
        "honest_verdict": "complete: ok",
        "solve_rate": 0.999,
    }
    (results_dir / "experiment_3494_p01_sudoku_correctness_first_solve_rate_gate_v1.json").write_text(
        json.dumps(flagged)
    )

    out = tmp_path / "experiment_3502_g_gate_status_synthesis_v322.json"
    with (
        patch.object(exp3502, "RESULTS_DIR", results_dir),
        patch.object(exp3502, "OUTPUT_PATH", out),
    ):
        main()

    data = json.loads(out.read_text())
    # Flagged exp3494 => route1 fields are null.
    assert data["p01_route1_sudoku_verdict"] is None
    assert data["p01_route1_solve_rate"] is None


def test_unmet_gates_is_list(tmp_path: Path) -> None:
    """REQ-GATE-001: unmet_gates is a list (even when all gates pass)."""
    out = tmp_path / "experiment_3502_g_gate_status_synthesis_v322.json"
    with patch.object(exp3502, "OUTPUT_PATH", out):
        main()
    data = json.loads(out.read_text())
    assert isinstance(data["unmet_gates"], list)
