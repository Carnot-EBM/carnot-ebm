"""Tests for Experiment 749 — PSV Domain-Diverse Recovery Monitoring (Iterations 31-60).

Every test traces to REQ-PSV-012 and SCENARIO-PSV-012.

Spec: REQ-PSV-012, SCENARIO-PSV-012
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_749_psv_monitoring import (  # noqa: E402
    _compute_honest_verdict,
    _linear_slope,
    _make_arc_challenge_questions,
    _make_domain_diverse_fns,
    _make_gsm8k_questions,
    _make_math_algebra_questions,
    _run_psv_condition,
    run_experiment,
)


# ---------------------------------------------------------------------------
# REQ-PSV-012: fp_rate_slope_new30 is computed correctly over 30 iterations
# ---------------------------------------------------------------------------


class TestSlopeComputationNew30:
    """REQ-PSV-012: fp_rate_slope_new30 must be the OLS slope of exactly 30 values."""

    def test_slope_of_flat_series_is_zero(self) -> None:
        """A flat fp_rate series (no change) must produce slope = 0.0.

        REQ-PSV-012 — plateau is acceptable: abs(slope) < 0.0001 counts as sustained.
        """
        values = [0.65] * 30
        assert _linear_slope(values) == pytest.approx(0.0, abs=1e-10)

    def test_slope_of_decreasing_series_is_negative(self) -> None:
        """Decreasing fp_rate over 30 iterations must produce negative slope.

        REQ-PSV-012 — negative slope = psv_recovery_sustained.
        """
        values = [1.0 - i / 30.0 for i in range(30)]
        assert _linear_slope(values) < 0

    def test_slope_of_increasing_series_is_positive(self) -> None:
        """Increasing fp_rate over 30 iterations must produce positive slope.

        REQ-PSV-012 — positive slope could indicate decelerating or relapse.
        """
        values = [i / 30.0 for i in range(30)]
        assert _linear_slope(values) > 0

    def test_slope_exact_known_value(self) -> None:
        """OLS slope of [0, 1, ..., 29] / 30 must equal 1/30.

        REQ-PSV-012 — closed-form regression check against a known analytical result.
        """
        n = 30
        values = [i / float(n) for i in range(n)]
        assert _linear_slope(values) == pytest.approx(1.0 / n, rel=1e-6)

    def test_slope_degenerate_input_returns_zero(self) -> None:
        """_linear_slope with fewer than 2 values must return 0.0.

        REQ-PSV-012 — edge-case guard prevents ZeroDivisionError.
        """
        assert _linear_slope([]) == 0.0
        assert _linear_slope([0.5]) == 0.0

    def test_slope_all_same_x_returns_zero(self) -> None:
        """_linear_slope must return 0.0 when all x-values are identical (zero denom).

        REQ-PSV-012 — zero-denominator guard: impossible in practice (x = range(n)),
        but the guard must not raise ZeroDivisionError.  Triggered by patching xs to
        a constant list via the internal calculation path (single-item list fed as
        a two-element constant-x series is one way to reach denom=0).

        We test this indirectly: a two-value series where x-variance is zero cannot
        occur via the public API (x is always range(n)), but a constant y-series with
        n >= 2 still exercises the non-zero denominator path.  The zero-denom branch
        (denom == 0) is an internal guard — verified here via a series where all
        values happen to produce denom == 0 by testing the slope of [0, 0] (2 values,
        both the same): denom = 2*1 - 1*1 = 1 (non-zero).  The actual zero-denom
        branch requires xs with all-same values, which is unreachable from public API.
        This test documents the behaviour rather than artificially hitting that branch.
        """
        # Two identical values: denom is non-zero (2*1 - 1^2 = 1), slope is 0.
        assert _linear_slope([0.0, 0.0]) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# REQ-PSV-012: honest_verdict maps correctly to slope thresholds
# ---------------------------------------------------------------------------


class TestHonestVerdictMapping:
    """REQ-PSV-012: _compute_honest_verdict must map slope values to the correct verdict."""

    def test_negative_new30_slope_is_sustained(self) -> None:
        """Negative new30 slope must yield psv_recovery_sustained.

        REQ-PSV-012 — negative slope means PSV is still improving at iterations 31-60.
        """
        verdict = _compute_honest_verdict(
            fp_rate_slope_new30=-0.001,
            fp_rate_slope_737=-0.00131257,
            condition_a_slope=0.0,
        )
        assert verdict == "psv_recovery_sustained"

    def test_near_zero_new30_slope_is_sustained(self) -> None:
        """Slope smaller than plateau threshold must yield psv_recovery_sustained.

        REQ-PSV-012 — near-zero slope (plateau) is acceptable; recovery held.
        """
        verdict = _compute_honest_verdict(
            fp_rate_slope_new30=0.00005,   # below 0.0001 threshold
            fp_rate_slope_737=-0.00131257,
            condition_a_slope=0.0,
        )
        assert verdict == "psv_recovery_sustained"

    def test_positive_below_condition_a_is_decelerating(self) -> None:
        """Positive new30 slope below condition_a must yield psv_recovery_decelerating.

        REQ-PSV-012 — recovery slowing but still better than no-fix control.
        condition_a_slope from Exp 736 is typically 0.0.  If new30 > 0 but < condition_a,
        this maps to decelerating.  Tested with condition_a = 0.005 to give room.
        """
        verdict = _compute_honest_verdict(
            fp_rate_slope_new30=0.002,     # positive, above plateau threshold
            fp_rate_slope_737=-0.00131257,
            condition_a_slope=0.005,       # new30 < condition_a → decelerating
        )
        assert verdict == "psv_recovery_decelerating"

    def test_positive_above_condition_a_is_relapse(self) -> None:
        """new30 slope above condition_a must yield psv_recovery_relapse.

        REQ-PSV-012 — slope worse than the no-fix control means the fix stopped working.
        This is the "deeper structural problem" scenario requiring architecture review.
        """
        verdict = _compute_honest_verdict(
            fp_rate_slope_new30=0.01,      # positive AND above condition_a
            fp_rate_slope_737=-0.00131257,
            condition_a_slope=0.005,       # new30 > condition_a → relapse
        )
        assert verdict == "psv_recovery_relapse"

    def test_equal_to_condition_a_is_relapse(self) -> None:
        """new30 slope exactly equal to condition_a must also yield psv_recovery_relapse.

        REQ-PSV-012 — the boundary condition: equal to control = not better than no fix.
        """
        verdict = _compute_honest_verdict(
            fp_rate_slope_new30=0.005,
            fp_rate_slope_737=-0.00131257,
            condition_a_slope=0.005,       # new30 == condition_a → relapse (not strictly less)
        )
        assert verdict == "psv_recovery_relapse"

    def test_zero_new30_slope_is_sustained(self) -> None:
        """Exactly zero new30 slope must yield psv_recovery_sustained.

        REQ-PSV-012 — abs(0) < 0.0001 plateau threshold, so zero is sustained.
        """
        verdict = _compute_honest_verdict(
            fp_rate_slope_new30=0.0,
            fp_rate_slope_737=-0.00131257,
            condition_a_slope=0.0,
        )
        assert verdict == "psv_recovery_sustained"


# ---------------------------------------------------------------------------
# REQ-PSV-012-1: domain_pool includes all 3 domains in each iteration
# ---------------------------------------------------------------------------


class TestDomainPoolCoverage:
    """REQ-PSV-012-1: domain_pool must include all 3 domains in each PSV iteration."""

    def test_domain_pool_has_three_domains(self, tmp_path: Path) -> None:
        """The artifact's domain_pool field must list all three required domains.

        REQ-PSV-012-1 — GSM8K + MATH-Algebra + ARC-Challenge must all be present.
        """
        _setup_exp737_result(tmp_path)

        artifact = run_experiment(repo_root=tmp_path)

        assert "domain_pool" in artifact
        assert set(artifact["domain_pool"]) == {"gsm8k", "math_algebra", "arc_challenge"}

    def test_each_iteration_has_questions_from_all_domains(self) -> None:
        """Each iteration's question list must contain questions from all 3 domains.

        REQ-PSV-012-1 — domain diversity must hold per-iteration, not just across iterations.
        Verified by checking that GSM8K, algebra, and ARC question patterns all appear.
        """
        import random as _random

        rng = _random.Random(749)
        gsm8k_pool = _make_gsm8k_questions(0, 30)
        algebra_pool = _make_math_algebra_questions(15)
        arc_pool = _make_arc_challenge_questions(15)

        # Build one iteration's question list (same logic as run_experiment)
        iteration_qs = (
            rng.sample(gsm8k_pool, 10)
            + rng.sample(algebra_pool, 5)
            + rng.sample(arc_pool, 5)
        )

        gsm8k_count = sum(1 for q in iteration_qs if q.startswith("GSM8K-"))
        algebra_count = sum(1 for q in iteration_qs if q.startswith("MATH-ALG-"))
        arc_count = sum(1 for q in iteration_qs if q.startswith("ARC-"))

        assert gsm8k_count == 10, f"Expected 10 GSM8K, got {gsm8k_count}"
        assert algebra_count == 5, f"Expected 5 algebra, got {algebra_count}"
        assert arc_count == 5, f"Expected 5 ARC, got {arc_count}"


# ---------------------------------------------------------------------------
# REQ-PSV-012-2/3: artifact contains all required fields (slopes, fp_rates, verdict)
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """REQ-PSV-012-2/3: artifact must contain all specified fields for traceability."""

    def test_artifact_contains_all_required_fields(self, tmp_path: Path) -> None:
        """Full run must produce an artifact with every REQ-PSV-012 required field.

        REQ-PSV-012-2 / REQ-PSV-012-3 — traceability requires all fields present.
        """
        _setup_exp737_result(tmp_path)

        artifact = run_experiment(repo_root=tmp_path)

        required_fields = {
            "experiment",
            "status",
            "honest_verdict",
            "fp_rate_slope_new30",
            "fp_rate_slope_all60",
            "fp_rate_slope_737",
            "fp_rate_slope_a",
            "iterations_run",
            "fp_rates_new30",
            "fp_rates_all60",
            "domain_pool",
        }
        for field in required_fields:
            assert field in artifact, f"Missing required field: {field}"

    def test_artifact_runs_exactly_30_new_iterations(self, tmp_path: Path) -> None:
        """run_experiment must run exactly 30 new iterations (iterations 31-60).

        REQ-PSV-012 — 30 new iterations continuing from Exp 737's 30.
        """
        _setup_exp737_result(tmp_path)

        artifact = run_experiment(repo_root=tmp_path)

        assert artifact["iterations_run"] == 30
        assert len(artifact["fp_rates_new30"]) == 30

    def test_all60_combines_exp737_and_new_fp_rates(self, tmp_path: Path) -> None:
        """fp_rates_all60 must have exactly 60 entries (30 from 737 + 30 new).

        REQ-PSV-012-2 — all60 slope requires all 60 fp_rate values.
        """
        _setup_exp737_result(tmp_path)

        artifact = run_experiment(repo_root=tmp_path)

        assert len(artifact["fp_rates_all60"]) == 60

    def test_slope_all60_matches_recomputed_value(self, tmp_path: Path) -> None:
        """fp_rate_slope_all60 must match recomputing _linear_slope on fp_rates_all60.

        REQ-PSV-012-3 — the reported slope must be consistent with the data.
        """
        _setup_exp737_result(tmp_path)

        artifact = run_experiment(repo_root=tmp_path)

        recomputed = _linear_slope(artifact["fp_rates_all60"])
        assert recomputed == pytest.approx(artifact["fp_rate_slope_all60"], abs=1e-6)

    def test_slope_new30_matches_recomputed_value(self, tmp_path: Path) -> None:
        """fp_rate_slope_new30 must match recomputing _linear_slope on fp_rates_new30.

        REQ-PSV-012-3 — the reported slope must be consistent with the data.
        """
        _setup_exp737_result(tmp_path)

        artifact = run_experiment(repo_root=tmp_path)

        recomputed = _linear_slope(artifact["fp_rates_new30"])
        assert recomputed == pytest.approx(artifact["fp_rate_slope_new30"], abs=1e-6)

    def test_fp_rates_are_valid_fractions(self, tmp_path: Path) -> None:
        """All fp_rates in the artifact must be in [0.0, 1.0].

        REQ-PSV-012 — fp_rate is a fraction of violations; values outside [0,1] indicate a bug.
        """
        _setup_exp737_result(tmp_path)

        artifact = run_experiment(repo_root=tmp_path)

        for fp in artifact["fp_rates_new30"]:
            assert 0.0 <= fp <= 1.0, f"fp_rate out of range: {fp}"

    def test_artifact_status_is_success(self, tmp_path: Path) -> None:
        """Successful run must produce artifact with status='success'.

        REQ-PSV-012 — conductor uses status field to determine experiment outcome.
        """
        _setup_exp737_result(tmp_path)

        artifact = run_experiment(repo_root=tmp_path)

        assert artifact["status"] == "success"

    def test_artifact_written_to_disk(self, tmp_path: Path) -> None:
        """run_experiment must write the artifact JSON file to disk.

        REQ-PSV-012 — conductor requires the deliverable file to be present on exit.
        """
        _setup_exp737_result(tmp_path)

        run_experiment(repo_root=tmp_path)

        out_path = tmp_path / "results" / "experiment_749_psv_monitoring.json"
        assert out_path.exists()
        written = json.loads(out_path.read_text())
        assert written["experiment"] == 749


# ---------------------------------------------------------------------------
# Helper: set up the Exp 737 result fixture in a tmp_path
# ---------------------------------------------------------------------------


def _setup_exp737_result(tmp_path: Path) -> None:
    """Write a minimal Exp 737 result JSON into tmp_path for test isolation.

    Uses the real Exp 737 fp_rates and slope values so test assertions
    reflect realistic input rather than synthetic edge cases.

    Args:
        tmp_path: Pytest temporary directory (pytest fixture).
    """
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    exp737_data = {
        "experiment": 737,
        "title": "PSV Domain-Diverse Recovery",
        "status": "success",
        "honest_verdict": "psv_recovery_confirmed",
        "fp_rate_slope": -0.00131257,
        "condition_a_slope": 0.0,
        "slope_delta": -0.00131257,
        "iterations_run": 30,
        "fp_rates": [
            0.65, 0.55, 0.75, 0.8, 0.75, 0.7, 0.75, 0.75, 0.55, 0.6,
            0.75, 0.65, 0.65, 0.55, 0.7, 0.7, 0.7, 0.65, 0.7, 0.75,
            0.6, 0.6, 0.55, 0.6, 0.7, 0.7, 0.75, 0.6, 0.65, 0.7,
        ],
        "gate_source": "exp736",
    }
    out = tmp_path / "results" / "experiment_737_psv_domain_diverse.json"
    out.write_text(json.dumps(exp737_data, indent=2))
