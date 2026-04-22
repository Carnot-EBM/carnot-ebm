"""Tests for Experiment 737 — PSV Domain-Diverse Recovery.

Spec: REQ-PSV-010, REQ-PSV-011, SCENARIO-PSV-010, SCENARIO-PSV-011
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_737_psv_domain_diverse import (  # noqa: E402
    _linear_slope,
    _make_arc_challenge_questions,
    _make_domain_diverse_fns,
    _make_gsm8k_questions,
    _make_math_algebra_questions,
    _run_psv_condition,
    _write_gated_blocked,
    run_experiment,
)


# ---------------------------------------------------------------------------
# REQ-PSV-010: gated-blocked path writes correct artifact when gate fails
# ---------------------------------------------------------------------------


class TestGatedBlockedPath:
    """REQ-PSV-010: when Exp 736 gate is 'fail', write gated_blocked artifact and stop."""

    def test_gated_blocked_writes_correct_status(self, tmp_path: Path) -> None:
        """Gate 'fail' must produce a gated_blocked artifact with the required fields.

        The artifact must satisfy the schema contract so downstream tooling (conductor,
        retrospective scripts) can parse it without special-casing the blocked case.

        REQ-PSV-010
        """
        # Set up a gate file with gate="fail" in the tmp results dir.
        (tmp_path / "results").mkdir()
        gate_data = {
            "gate": "fail",
            "root_cause": "unknown",
            "fix": "unknown",
            "condition_a_slope": 0.0,
            "condition_b_slope": 0.001,
            "condition_c_slope": 0.002,
            "experiment": 736,
        }
        gate_path = tmp_path / "results" / "psv_specialization_gate.json"
        gate_path.write_text(json.dumps(gate_data))

        # Patch the experiment to use our temp repo root so no real files are touched.
        artifact = _write_gated_blocked(tmp_path)

        # Verify required artifact fields.
        assert artifact["experiment"] == 737
        assert artifact["status"] == "gated_blocked"
        assert artifact["honest_verdict"] == "gated_blocked_specialization_not_confirmed"
        assert artifact["gate_source"] == "exp736"
        assert "PSV root cause unknown" in artifact["note"]
        assert artifact["schema"] == "carnot.result.v1"

        # Verify the file was written on disk.
        out_path = tmp_path / "results" / "experiment_737_psv_domain_diverse.json"
        assert out_path.exists()
        written = json.loads(out_path.read_text())
        assert written["status"] == "gated_blocked"

    def test_run_experiment_writes_gated_blocked_when_gate_fails(self, tmp_path: Path) -> None:
        """run_experiment() must detect a failing gate and produce gated_blocked artifact.

        REQ-PSV-010
        """
        (tmp_path / "results").mkdir()
        gate_data = {
            "gate": "fail",
            "root_cause": "unknown",
            "fix": "unknown",
            "condition_a_slope": 0.0,
            "condition_b_slope": 0.01,
            "condition_c_slope": 0.02,
            "experiment": 736,
        }
        (tmp_path / "results" / "psv_specialization_gate.json").write_text(json.dumps(gate_data))

        artifact = run_experiment(repo_root=tmp_path)

        assert artifact["status"] == "gated_blocked"
        assert artifact["honest_verdict"] == "gated_blocked_specialization_not_confirmed"


# ---------------------------------------------------------------------------
# REQ-PSV-011: 30-iteration run produces exactly 30 fp_rate measurements
# ---------------------------------------------------------------------------


class TestThirtyIterations:
    """REQ-PSV-011: the 30-iteration PSV run must produce exactly 30 fp_rate measurements."""

    def test_run_produces_30_fp_rates(self, tmp_path: Path) -> None:
        """Passing gate must trigger a 30-iteration run and record 30 fp_rate values.

        REQ-PSV-011
        """
        (tmp_path / "results").mkdir()
        gate_data = {
            "gate": "pass",
            "root_cause": "constraint_specialization",
            "fix": "domain_diversity",
            "condition_a_slope": 0.0,
            "condition_b_slope": -0.00056391,
            "condition_c_slope": -0.00255639,
            "experiment": 736,
        }
        (tmp_path / "results" / "psv_specialization_gate.json").write_text(json.dumps(gate_data))

        artifact = run_experiment(repo_root=tmp_path)

        assert artifact["status"] == "success"
        assert artifact["iterations_run"] == 30
        assert len(artifact["fp_rates"]) == 30
        # Every entry must be a valid fraction in [0, 1].
        for fp in artifact["fp_rates"]:
            assert 0.0 <= fp <= 1.0

    def test_psv_condition_produces_correct_length(self) -> None:
        """_run_psv_condition must return exactly as many values as questions_per_iter.

        REQ-PSV-011
        """
        gsm8k = _make_gsm8k_questions(0, 10)
        algebra = _make_math_algebra_questions(5)
        arc = _make_arc_challenge_questions(5)
        inf_fn, ver_fn = _make_domain_diverse_fns(gsm8k, algebra, arc)

        # 30 iterations, 5 questions each.
        questions_per_iter = [[*gsm8k[:5]] for _ in range(30)]
        fp_rates = _run_psv_condition(questions_per_iter, inf_fn, ver_fn)

        assert len(fp_rates) == 30

    def test_artifact_fields_present(self, tmp_path: Path) -> None:
        """Artifact must include all required fields for a passing-gate run.

        REQ-PSV-011
        """
        (tmp_path / "results").mkdir()
        gate_data = {
            "gate": "pass",
            "root_cause": "constraint_specialization",
            "fix": "domain_diversity",
            "condition_a_slope": 0.0,
            "condition_b_slope": -0.00056391,
            "condition_c_slope": -0.00255639,
            "experiment": 736,
        }
        (tmp_path / "results" / "psv_specialization_gate.json").write_text(json.dumps(gate_data))

        artifact = run_experiment(repo_root=tmp_path)

        required = {
            "experiment", "status", "honest_verdict", "fp_rate_slope",
            "condition_a_slope", "slope_delta", "iterations_run", "fp_rates",
            "fix_applied", "gate_source",
        }
        for field in required:
            assert field in artifact, f"Missing field: {field}"

        assert artifact["fix_applied"] == "domain_diversity"
        assert artifact["gate_source"] == "exp736"


# ---------------------------------------------------------------------------
# REQ-PSV-010: slope computed correctly from a known series
# ---------------------------------------------------------------------------


class TestSlopeComputation:
    """REQ-PSV-010: _linear_slope must be computed correctly from iteration data."""

    def test_slope_of_constant_series_is_zero(self) -> None:
        """A flat fp_rate series (no change) must produce slope=0.0.

        REQ-PSV-010
        """
        values = [0.65] * 30
        assert _linear_slope(values) == pytest.approx(0.0, abs=1e-10)

    def test_slope_of_increasing_series_is_positive(self) -> None:
        """An increasing fp_rate series must produce a positive slope.

        REQ-PSV-010 — positive slope means PSV is degrading.
        """
        values = [float(i) / 30 for i in range(30)]
        slope = _linear_slope(values)
        assert slope > 0

    def test_slope_of_decreasing_series_is_negative(self) -> None:
        """A decreasing fp_rate series must produce a negative slope.

        REQ-PSV-010 — negative slope means PSV is improving (recovery confirmed).
        """
        values = [1.0 - float(i) / 30 for i in range(30)]
        slope = _linear_slope(values)
        assert slope < 0

    def test_slope_exact_known_value(self) -> None:
        """OLS slope of [0, 1, 2, ..., n-1] / n must equal 1/n.

        This is the closed-form check: y=x/n gives slope=1/n by definition.

        REQ-PSV-010
        """
        n = 30
        values = [i / n for i in range(n)]
        slope = _linear_slope(values)
        # Expected slope for y = x/n is 1/n (since dy/dx = 1/n).
        assert slope == pytest.approx(1.0 / n, rel=1e-6)

    def test_slope_degenerate_input_returns_zero(self) -> None:
        """_linear_slope with fewer than 2 values must return 0.0.

        REQ-PSV-010 — degenerate guard prevents ZeroDivisionError in edge cases.
        """
        assert _linear_slope([]) == 0.0
        assert _linear_slope([0.5]) == 0.0

    def test_slope_from_30_iteration_run_matches_expected(self, tmp_path: Path) -> None:
        """The slope computed by run_experiment matches recomputing from fp_rates in artifact.

        REQ-PSV-010 — slope_delta = fp_rate_slope - condition_a_slope must be consistent.
        """
        (tmp_path / "results").mkdir()
        gate_data = {
            "gate": "pass",
            "root_cause": "constraint_specialization",
            "fix": "domain_diversity",
            "condition_a_slope": 0.0,
            "condition_b_slope": -0.00056391,
            "condition_c_slope": -0.00255639,
            "experiment": 736,
        }
        (tmp_path / "results" / "psv_specialization_gate.json").write_text(json.dumps(gate_data))

        artifact = run_experiment(repo_root=tmp_path)

        recomputed_slope = _linear_slope(artifact["fp_rates"])
        assert recomputed_slope == pytest.approx(artifact["fp_rate_slope"], abs=1e-6)
        expected_delta = artifact["fp_rate_slope"] - artifact["condition_a_slope"]
        assert expected_delta == pytest.approx(artifact["slope_delta"], abs=1e-6)
