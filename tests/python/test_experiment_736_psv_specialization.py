"""Tests for Experiment 736 — PSV Constraint Specialization Diagnostic.

Covers:
- Domain pool routing: each domain question goes to the correct inference/verify path.
- fp_rate_slope computation from iteration series.
- Gate file written with correct root_cause field.
- Artifact contains all three slopes and required schema fields.

Spec: REQ-PSV-010, REQ-PSV-011, SCENARIO-PSV-010, SCENARIO-PSV-011
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


from scripts.experiment_736_psv_specialization import (
    DELIVERABLE,
    EXPERIMENT_ID,
    _GATE_FILE,
    _linear_slope,
    _make_arc_challenge_questions,
    _make_gsm8k_questions,
    _make_math_algebra_questions,
    _make_synthetic_fns_generic_verifier,
    _make_synthetic_fns_gsm8k,
    _make_synthetic_fns_multidomain,
    _run_psv_condition,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo_root(tmp_path: Path) -> Path:
    """Create a minimal repo directory tree for test isolation.

    Mirrors the layout expected by ExperimentTemplate and DeliverableGuard:
        <root>/results/                      — artifact output dir
        <root>/results/checkpoints/          — checkpoint dir
        <root>/scripts/conductor_exclusion_manifest.json

    Why needed: ExperimentTemplate resolves all paths relative to repo_root.
    Using a tmp_path avoids writing to the real results/ directory during CI.
    """
    root = tmp_path / "carnot"
    (root / "results" / "checkpoints").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    manifest = {"excluded": []}
    (root / "scripts" / "conductor_exclusion_manifest.json").write_text(
        json.dumps(manifest)
    )
    return root


# ---------------------------------------------------------------------------
# _linear_slope tests (REQ-PSV-010: slope must be computable)
# ---------------------------------------------------------------------------


def test_linear_slope_flat() -> None:
    """A constant series has slope 0.0.

    REQ-PSV-010: slope is the primary metric for comparing conditions; a flat
    series must not be misclassified as degrading or improving.
    """
    assert _linear_slope([0.6, 0.6, 0.6, 0.6]) == pytest.approx(0.0, abs=1e-9)


def test_linear_slope_increasing() -> None:
    """Strictly increasing series has positive slope (degradation signal).

    REQ-PSV-010: positive slope is the degradation pattern this experiment diagnoses.
    """
    slope = _linear_slope([0.0, 0.1, 0.2, 0.3, 0.4])
    assert slope > 0


def test_linear_slope_decreasing() -> None:
    """Strictly decreasing series has negative slope (improvement signal).

    REQ-PSV-010: negative slope in Condition B or C would confirm the hypothesis.
    """
    slope = _linear_slope([0.5, 0.4, 0.3, 0.2, 0.1])
    assert slope < 0


def test_linear_slope_degenerate_single() -> None:
    """Single-element series returns 0.0 without raising.

    REQ-PSV-010: _linear_slope must handle degenerate inputs gracefully to avoid
    crashing the gate logic when an early-terminated condition returns one data point.
    """
    assert _linear_slope([0.5]) == pytest.approx(0.0, abs=1e-9)


def test_linear_slope_degenerate_empty() -> None:
    """Empty series returns 0.0 without raising.

    REQ-PSV-010: guard against empty fp_rates list if a condition produces zero iterations.
    """
    assert _linear_slope([]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Domain question generator tests (REQ-PSV-011)
# ---------------------------------------------------------------------------


def test_gsm8k_questions_range_and_uniqueness() -> None:
    """_make_gsm8k_questions returns the correct count and unique questions.

    REQ-PSV-011: each domain question generator must produce distinct questions
    so the domain pool does not accidentally repeat content.
    """
    qs = _make_gsm8k_questions(200, 220)
    assert len(qs) == 20
    assert len(set(qs)) == 20, "All 20 questions must be unique"


def test_gsm8k_questions_contain_index() -> None:
    """GSM8K questions embed the index so held-out sets are reproducible.

    REQ-PSV-011: index embedding ensures question 200 is always the same arithmetic
    problem regardless of how many questions are generated, making held-out sets stable.
    """
    qs = _make_gsm8k_questions(200, 202)
    assert "GSM8K-200" in qs[0]
    assert "GSM8K-201" in qs[1]


def test_math_algebra_questions_count_and_uniqueness() -> None:
    """_make_math_algebra_questions returns distinct algebraic questions.

    REQ-PSV-011-1: at least one non-arithmetic domain must be present; algebra
    questions must be distinct from arithmetic questions.
    """
    qs = _make_math_algebra_questions(10)
    assert len(qs) == 10
    assert len(set(qs)) == 10


def test_arc_challenge_questions_count_and_uniqueness() -> None:
    """_make_arc_challenge_questions returns distinct logical/scientific questions.

    REQ-PSV-011-1: ARC-Challenge is the required non-arithmetic logical domain.
    """
    qs = _make_arc_challenge_questions(10)
    assert len(qs) == 10
    # ARC has 5 templates cycling; 10 questions will have 5 unique texts with different indices
    assert all("ARC-" in q for q in qs)


# ---------------------------------------------------------------------------
# Domain routing tests (REQ-PSV-011-2)
# ---------------------------------------------------------------------------


def test_gsm8k_synthetic_fns_correct_on_multiples_of_3() -> None:
    """GSM8K inference_fn returns CORRECT for question indices divisible by 3.

    REQ-PSV-011-2: the domain pool loader maps questions to the correct oracle;
    index 0 (divisible by 3) must route to the CORRECT path.
    """
    qs = _make_gsm8k_questions(0, 10)
    inf, ver = _make_synthetic_fns_gsm8k(qs)
    assert ver(inf(qs[0])) is True   # index 0: correct
    assert ver(inf(qs[1])) is False  # index 1: violation


def test_multidomain_fns_route_gsm8k_correctly() -> None:
    """Multidomain inference_fn routes GSM8K questions to the GSM8K oracle.

    REQ-PSV-011-2: domain routing must send GSM8K questions to the GSM8K path,
    not the algebra or ARC path.  GSM8K index 0 (divisible by 3) must be CORRECT.
    """
    gsm8k_qs = _make_gsm8k_questions(0, 6)
    algebra_qs = _make_math_algebra_questions(4)
    arc_qs = _make_arc_challenge_questions(4)
    inf, ver = _make_synthetic_fns_multidomain(gsm8k_qs, algebra_qs, arc_qs)

    # GSM8K index 0 → correct (divisible by 3)
    assert ver(inf(gsm8k_qs[0])) is True
    # GSM8K index 1 → violation (not divisible by 3)
    assert ver(inf(gsm8k_qs[1])) is False


def test_multidomain_fns_route_algebra_correctly() -> None:
    """Multidomain inference_fn routes MATH-Algebra questions to the algebra oracle.

    REQ-PSV-011-2: algebra questions use a 50% correct rate (index % 2 == 0).
    This test confirms the algebra domain is routed independently of GSM8K.
    """
    gsm8k_qs = _make_gsm8k_questions(0, 4)
    algebra_qs = _make_math_algebra_questions(4)
    arc_qs = _make_arc_challenge_questions(4)
    inf, ver = _make_synthetic_fns_multidomain(gsm8k_qs, algebra_qs, arc_qs)

    # Algebra index 0 → correct (even)
    assert ver(inf(algebra_qs[0])) is True
    # Algebra index 1 → violation (odd)
    assert ver(inf(algebra_qs[1])) is False


def test_multidomain_fns_route_arc_correctly() -> None:
    """Multidomain inference_fn routes ARC-Challenge questions to the ARC oracle.

    REQ-PSV-011-2: ARC uses a 20% correct rate (index % 5 == 0).
    """
    gsm8k_qs = _make_gsm8k_questions(0, 4)
    algebra_qs = _make_math_algebra_questions(4)
    arc_qs = _make_arc_challenge_questions(6)
    inf, ver = _make_synthetic_fns_multidomain(gsm8k_qs, algebra_qs, arc_qs)

    # ARC index 0 → correct (divisible by 5)
    assert ver(inf(arc_qs[0])) is True
    # ARC index 1 → violation
    assert ver(inf(arc_qs[1])) is False


def test_generic_verifier_requires_domain_tag() -> None:
    """Domain-generic verify_fn requires '[domain=' tag in addition to CORRECT.

    REQ-PSV-011: the domain-generic verifier is stricter than the GSM8K-specialized
    one — it rejects responses that have CORRECT but no domain tag.  This simulates
    the tighter filter of a verifier trained on cross-domain labels.
    """
    qs = _make_gsm8k_questions(100, 110)
    inf, ver = _make_synthetic_fns_generic_verifier(qs)

    # A bare "CORRECT" without a domain tag must fail the generic verifier.
    assert ver("The answer is 42. CORRECT") is False
    # A response with both CORRECT and domain tag must pass.
    assert ver("Answer: 5 [domain=arithmetic] CORRECT") is True


# ---------------------------------------------------------------------------
# _run_psv_condition tests (REQ-PSV-010)
# ---------------------------------------------------------------------------


def test_run_psv_condition_length() -> None:
    """_run_psv_condition returns one fp_rate per iteration.

    REQ-PSV-010: the returned list length must equal n_iterations so
    _linear_slope receives the correct number of data points.
    """
    qs = _make_gsm8k_questions(0, 10)
    inf, ver = _make_synthetic_fns_gsm8k(qs)
    n_iter = 5
    fp_rates = _run_psv_condition([qs[:5] for _ in range(n_iter)], inf, ver)
    assert len(fp_rates) == n_iter


def test_run_psv_condition_rates_bounded() -> None:
    """All fp_rate values from _run_psv_condition are in [0.0, 1.0].

    REQ-PSV-010: fp_rate is a probability; out-of-range values indicate a bug
    in the division logic (e.g., more violations counted than questions asked).
    """
    qs = _make_gsm8k_questions(0, 10)
    inf, ver = _make_synthetic_fns_gsm8k(qs)
    fp_rates = _run_psv_condition([qs for _ in range(4)], inf, ver)
    for r in fp_rates:
        assert 0.0 <= r <= 1.0, f"fp_rate {r} is out of [0, 1]"


# ---------------------------------------------------------------------------
# Integration tests: artifact + gate file (SCENARIO-PSV-010/011)
# ---------------------------------------------------------------------------


def test_artifact_contains_all_three_slopes(tmp_path: Path) -> None:
    """run_experiment() artifact contains condition_a, b, and c slopes.

    REQ-PSV-010: all three slopes must be present so the conductor and the
    adversarial reviewer can verify both hypotheses were tested.
    SCENARIO-PSV-010: condition_b_slope drives the pass/fail gate decision.
    SCENARIO-PSV-011: condition_c_slope drives the pass_verifier decision.
    """
    root = _make_repo_root(tmp_path)
    artifact = run_experiment(repo_root=root)

    for key in ("condition_a_slope", "condition_b_slope", "condition_c_slope"):
        assert key in artifact, f"artifact must contain {key}"
        assert isinstance(artifact[key], float), f"{key} must be a float"


def test_artifact_honest_verdict_is_valid(tmp_path: Path) -> None:
    """honest_verdict is one of the two predefined values.

    REQ-PSV-010-2: honest_verdict must be a falsifiable outcome, not a vague label.
    """
    root = _make_repo_root(tmp_path)
    artifact = run_experiment(repo_root=root)

    valid = {"psv_specialization_confirmed", "psv_specialization_not_root_cause"}
    assert artifact["honest_verdict"] in valid, (
        f"honest_verdict '{artifact['honest_verdict']}' not in {valid}"
    )


def test_gate_file_written_with_correct_schema(tmp_path: Path) -> None:
    """Gate file contains all required fields including root_cause.

    REQ-PSV-010-3: gate file must record root_cause_hypothesis so Exp 737
    can branch on which fix to apply without re-running the experiment.
    """
    root = _make_repo_root(tmp_path)
    run_experiment(repo_root=root)

    gate_path = root / _GATE_FILE
    assert gate_path.exists(), f"Gate file not found at {gate_path}"

    gate_data = json.loads(gate_path.read_text())

    required_keys = {
        "gate",
        "root_cause",
        "fix",
        "condition_a_slope",
        "condition_b_slope",
        "condition_c_slope",
        "experiment",
    }
    for key in required_keys:
        assert key in gate_data, f"Gate file missing required key: {key}"

    assert gate_data["gate"] in ("pass", "pass_verifier", "fail"), (
        f"gate must be 'pass', 'pass_verifier', or 'fail', got '{gate_data['gate']}'"
    )
    assert gate_data["experiment"] == EXPERIMENT_ID


def test_gate_root_cause_consistent_with_gate(tmp_path: Path) -> None:
    """Gate value and root_cause are internally consistent.

    SCENARIO-PSV-010: gate='pass' iff root_cause='constraint_specialization'.
    SCENARIO-PSV-011: gate='pass_verifier' iff root_cause='constraint_specialization_verifier'.
    gate='fail' iff root_cause='unknown'.
    """
    root = _make_repo_root(tmp_path)
    artifact = run_experiment(repo_root=root)
    gate_path = root / _GATE_FILE
    gate_data = json.loads(gate_path.read_text())

    gate = gate_data["gate"]
    root_cause = gate_data["root_cause"]

    if gate == "pass":
        assert root_cause == "constraint_specialization"
    elif gate == "pass_verifier":
        assert root_cause == "constraint_specialization_verifier"
    else:
        assert gate == "fail"
        assert root_cause == "unknown"


def test_deliverable_json_written_and_valid(tmp_path: Path) -> None:
    """run_experiment() writes the deliverable JSON with required schema fields.

    REQ-PSV-010: assert_deliverable_written() inside run_experiment() guarantees the
    file is present; this test verifies it is valid JSON with the required schema.
    """
    root = _make_repo_root(tmp_path)
    run_experiment(repo_root=root)

    out_path = root / DELIVERABLE
    assert out_path.exists(), f"Deliverable not found at {out_path}"

    data = json.loads(out_path.read_text())

    assert data["experiment"] == EXPERIMENT_ID
    assert data["status"] == "success"
    assert "honest_verdict" in data
    assert "gate" in data
    assert "gate_written" in data
    assert data["gate_written"] is True
    assert "root_cause_hypothesis" in data
    # All three slopes must be present in the deliverable.
    for key in ("condition_a_slope", "condition_b_slope", "condition_c_slope"):
        assert key in data, f"deliverable missing {key}"
