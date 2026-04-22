"""Tests for Experiment 722 — PSV Pool Exhaustion Controlled Diagnostic.

Covers:
- Condition A and B slopes are both present in the artifact (REQ-PSV-005).
- Gate file is written with the correct schema (SCENARIO-PSV-005).
- honest_verdict is one of the three defined values.
- Synthetic pool and slope utilities behave correctly.

Spec: REQ-PSV-005, SCENARIO-PSV-005
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Imports from the module under test
# ---------------------------------------------------------------------------

from scripts.experiment_722_psv_pool_exhaustion import (
    DELIVERABLE,
    EXPERIMENT_ID,
    _GATE_FILE,
    _linear_slope,
    _make_gsm8k_pool,
    _make_synthetic_fns,
    _run_psv_condition,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal fake repo root
# ---------------------------------------------------------------------------


def _make_repo_root(tmp_path: Path) -> Path:
    """Create a temporary directory tree that mirrors the repo layout.

    Produces:
        <root>/results/                          — artifact output dir
        <root>/results/checkpoints/              — checkpoint dir
        <root>/scripts/conductor_exclusion_manifest.json — minimal manifest

    Why we need this: ExperimentTemplate resolves all paths relative to repo_root,
    and DeliverableGuard validates the output path exists after writing.  A real
    repo root would also work but would pollute the actual results/ directory
    during CI runs.
    """
    root = tmp_path / "carnot"
    (root / "results" / "checkpoints").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    # Minimal exclusion manifest so check_exclusion_manifest() does not raise.
    manifest = {"excluded": []}
    (root / "scripts" / "conductor_exclusion_manifest.json").write_text(
        json.dumps(manifest)
    )
    return root


# ---------------------------------------------------------------------------
# Unit tests for _linear_slope
# ---------------------------------------------------------------------------


def test_linear_slope_flat_series() -> None:
    """A constant series has slope 0.

    REQ-PSV-005: slope must be computable from a flat fp_rate series.
    A flat series (all equal fp_rates) indicates no improvement and no degradation.
    """
    assert _linear_slope([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)


def test_linear_slope_increasing() -> None:
    """A strictly increasing series has a positive slope.

    REQ-PSV-005: a positive slope signals degradation (fp_rate rising over iterations).
    This is the expected Condition A behaviour when pool is exhausted.
    """
    slope = _linear_slope([0.0, 0.1, 0.2, 0.3])
    assert slope > 0


def test_linear_slope_decreasing() -> None:
    """A strictly decreasing series has a negative slope.

    REQ-PSV-005: a negative slope signals improvement (fp_rate falling over iterations).
    This is the expected Condition B behaviour with a rotating pool.
    """
    slope = _linear_slope([0.3, 0.2, 0.1, 0.0])
    assert slope < 0


def test_linear_slope_degenerate_single_value() -> None:
    """A single-element series returns 0.0 (degenerate case, no trend).

    REQ-PSV-005: _linear_slope must not raise for degenerate inputs.
    """
    assert _linear_slope([0.5]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Unit tests for _make_gsm8k_pool
# ---------------------------------------------------------------------------


def test_make_gsm8k_pool_length() -> None:
    """_make_gsm8k_pool(n) returns exactly n questions.

    REQ-PSV-005: Condition B requires a pool of exactly 100 questions.
    """
    for n in [10, 100]:
        pool = _make_gsm8k_pool(n)
        assert len(pool) == n, f"Expected {n} questions, got {len(pool)}"


def test_make_gsm8k_pool_unique() -> None:
    """All questions in the pool are unique strings.

    REQ-PSV-005: duplicate questions would reduce effective pool diversity.
    """
    pool = _make_gsm8k_pool(100)
    assert len(set(pool)) == 100, "All 100 questions must be unique"


def test_make_gsm8k_pool_index_stability() -> None:
    """Question 0 in a 10-question pool matches question 0 in a 100-question pool.

    REQ-PSV-005: Condition A and Condition B must share the same first 10 questions
    so the experiment isolates pool SIZE (not question content) as the variable.
    """
    pool_10 = _make_gsm8k_pool(10)
    pool_100 = _make_gsm8k_pool(100)
    assert pool_10[0] == pool_100[0], "Question 0 must be the same in both pools"
    assert pool_10[9] == pool_100[9], "Question 9 must be the same in both pools"


# ---------------------------------------------------------------------------
# Unit tests for _make_synthetic_fns
# ---------------------------------------------------------------------------


def test_synthetic_fns_correct_on_multiples_of_3() -> None:
    """inference_fn returns CORRECT for question indices that are multiples of 3.

    REQ-PSV-005: the synthetic oracle must be deterministic so slope differences
    are attributable to pool structure, not model randomness.
    """
    pool = _make_gsm8k_pool(10)
    inference_fn, verify_fn = _make_synthetic_fns(pool)
    # Question 0 (index 0, multiple of 3) should be correct
    response = inference_fn(pool[0])
    assert verify_fn(response) is True


def test_synthetic_fns_violation_on_non_multiples() -> None:
    """inference_fn returns VIOLATION for question indices that are not multiples of 3.

    REQ-PSV-005: violation questions must reliably produce negative labels so the
    fp_rate signal is not confounded by false-correct responses.
    """
    pool = _make_gsm8k_pool(10)
    inference_fn, verify_fn = _make_synthetic_fns(pool)
    # Question 1 (index 1, not multiple of 3) should be a violation
    response = inference_fn(pool[1])
    assert verify_fn(response) is False


# ---------------------------------------------------------------------------
# Unit tests for _run_psv_condition
# ---------------------------------------------------------------------------


def test_run_psv_condition_length() -> None:
    """_run_psv_condition returns one fp_rate per iteration.

    REQ-PSV-005: the returned list length must equal the number of iterations
    so _linear_slope receives the correct number of data points.
    """
    pool = _make_gsm8k_pool(10)
    inference_fn, verify_fn = _make_synthetic_fns(pool)
    n_iter = 5
    questions_per_iter = [pool[:5] for _ in range(n_iter)]
    rates = _run_psv_condition(questions_per_iter, inference_fn, verify_fn)
    assert len(rates) == n_iter


def test_run_psv_condition_rates_in_range() -> None:
    """All fp_rate values are in [0.0, 1.0].

    REQ-PSV-005: fp_rate is a probability and must be bounded.
    """
    pool = _make_gsm8k_pool(10)
    inference_fn, verify_fn = _make_synthetic_fns(pool)
    questions_per_iter = [pool for _ in range(3)]
    rates = _run_psv_condition(questions_per_iter, inference_fn, verify_fn)
    for r in rates:
        assert 0.0 <= r <= 1.0, f"fp_rate {r} out of [0, 1]"


# ---------------------------------------------------------------------------
# Integration test: artifact schema and gate file
# ---------------------------------------------------------------------------


def test_artifact_contains_both_slopes(tmp_path: Path) -> None:
    """run_experiment() artifact contains condition_a_slope and condition_b_slope.

    REQ-PSV-005: both slopes must be present so the conductor can verify the
    pool exhaustion hypothesis without re-running the experiment.

    SCENARIO-PSV-005: condition_a_slope and condition_b_slope drive the gate decision.
    """
    root = _make_repo_root(tmp_path)
    artifact = run_experiment(repo_root=root)

    assert "condition_a_slope" in artifact, "artifact must contain condition_a_slope"
    assert "condition_b_slope" in artifact, "artifact must contain condition_b_slope"
    assert isinstance(artifact["condition_a_slope"], float)
    assert isinstance(artifact["condition_b_slope"], float)


def test_artifact_contains_honest_verdict(tmp_path: Path) -> None:
    """run_experiment() artifact contains honest_verdict with a valid value.

    REQ-PSV-005: honest_verdict determines whether REQ-PSV-005 is satisfied and
    guides the Exp 723 gate decision.

    SCENARIO-PSV-005: honest_verdict must be one of three defined strings.
    """
    root = _make_repo_root(tmp_path)
    artifact = run_experiment(repo_root=root)

    valid_verdicts = {
        "pool_exhaustion_confirmed",
        "pool_exhaustion_not_confirmed",
        "pool_exhaustion_ambiguous",
    }
    assert "honest_verdict" in artifact
    assert artifact["honest_verdict"] in valid_verdicts, (
        f"honest_verdict '{artifact['honest_verdict']}' not in {valid_verdicts}"
    )


def test_gate_file_written_with_correct_schema(tmp_path: Path) -> None:
    """run_experiment() writes results/psv_pool_gate.json with the required schema.

    SCENARIO-PSV-005: the gate file must be machine-readable by the conductor
    and contain exactly the fields: gate, diagnosis, condition_a_slope,
    condition_b_slope, experiment.
    """
    root = _make_repo_root(tmp_path)
    run_experiment(repo_root=root)

    gate_path = root / _GATE_FILE
    assert gate_path.exists(), f"Gate file not found at {gate_path}"

    gate_data = json.loads(gate_path.read_text())

    required_keys = {"gate", "diagnosis", "condition_a_slope", "condition_b_slope", "experiment"}
    for key in required_keys:
        assert key in gate_data, f"Gate file missing required key: {key}"

    assert gate_data["gate"] in ("pass", "fail"), (
        f"gate must be 'pass' or 'fail', got '{gate_data['gate']}'"
    )
    assert gate_data["experiment"] == EXPERIMENT_ID


def test_deliverable_json_written(tmp_path: Path) -> None:
    """run_experiment() writes the deliverable JSON file to DELIVERABLE path.

    REQ-PSV-005: assert_deliverable_written() inside run_experiment() guarantees
    the file exists; this test verifies it is parseable JSON with the required schema.
    """
    root = _make_repo_root(tmp_path)
    run_experiment(repo_root=root)

    out_path = root / DELIVERABLE
    assert out_path.exists(), f"Deliverable not found at {out_path}"

    data = json.loads(out_path.read_text())
    assert data["experiment"] == EXPERIMENT_ID
    assert data["status"] == "success"
    assert "honest_verdict" in data
    assert "condition_a_slope" in data
    assert "condition_b_slope" in data


def test_gate_pass_when_pool_exhaustion_confirmed(tmp_path: Path) -> None:
    """Gate is 'pass' when condition_a_slope > 0 AND condition_b_slope < 0.

    SCENARIO-PSV-005: the synthetic fns are designed so that Condition A
    (fixed 10-question pool with deterministic correct/violation split) produces
    a flat or rising fp_rate (slope >= 0) because the same questions are seen
    repeatedly, while Condition B (rotating 100-question pool) introduces
    index-diversity that shifts the slope.

    This test does not assert the exact gate value (that depends on synthetic fn
    behaviour), but it asserts the gate-verdict consistency: gate='pass' iff
    honest_verdict='pool_exhaustion_confirmed'.
    """
    root = _make_repo_root(tmp_path)
    artifact = run_experiment(repo_root=root)

    gate_data = json.loads((root / _GATE_FILE).read_text())

    if artifact["honest_verdict"] == "pool_exhaustion_confirmed":
        assert gate_data["gate"] == "pass"
    else:
        assert gate_data["gate"] == "fail"
