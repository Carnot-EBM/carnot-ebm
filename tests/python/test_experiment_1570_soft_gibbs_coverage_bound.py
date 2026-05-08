"""Tests for Exp 1570 Soft-Gibbs Jensen coverage-bound verification.

Spec refs: REQ-SAMPLE-062, SCENARIO-SAMPLE-090.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot.sampling import soft_gibbs_coverage_bound as exp1570


def _write_rows(path: Path, n_rows: int, *, start: int = 0) -> Path:
    rows = []
    for offset in range(n_rows):
        idx = start + offset
        rows.append(
            {
                "question_id": f"cal-{idx}",
                "question": f"How many calibrated tokens are in case {idx}?",
                "response": f"The candidate answer is {idx % 97}.",
                "is_correct": bool(idx % 5 == 0),
                "model": "deterministic-test-generator",
            }
        )
    path.write_text(json.dumps({"pairs": rows}), encoding="utf-8")
    return path


def test_spec_mentions_exp1570_contract() -> None:
    """REQ-SAMPLE-062, SCENARIO-SAMPLE-090: Exp 1570 is spec anchored."""

    spec = (
        exp1570.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-SAMPLE-062" in spec
    assert "SCENARIO-SAMPLE-090" in spec
    assert "experiment_1570_soft_gibbs_coverage_bound_empirical_verification.json" in spec
    assert "alpha_i = P_mu(y notin S_i)" in spec


def test_req_sample_062_corpus_loader_builds_n500_calibration_set(tmp_path: Path) -> None:
    """REQ-SAMPLE-062: calibration corpus construction enforces N>=500."""

    source_a = _write_rows(tmp_path / "corpus_a.json", 300)
    source_b = _write_rows(tmp_path / "corpus_b.json", 260, start=300)

    corpus = exp1570.build_calibration_corpus(
        source_paths=(source_a, source_b),
        n_cases=500,
        seed=1570,
    )

    assert len(corpus) == 500
    assert all(case.question for case in corpus)
    assert all(case.candidate_response for case in corpus)
    assert all(0.0 < case.difficulty < 1.0 for case in corpus)

    with pytest.raises(ValueError, match="N >= 500"):
        exp1570.build_calibration_corpus(source_paths=(source_a, source_b), n_cases=499)

    sparse_source = _write_rows(tmp_path / "sparse.json", 20)
    with pytest.raises(ValueError, match="need at least 500"):
        exp1570.build_calibration_corpus(source_paths=(sparse_source,), n_cases=500)


def test_req_sample_062_alpha_and_jensen_math_on_fixed_pass_matrix() -> None:
    """REQ-SAMPLE-062: alpha_i and Jensen lower bound use verifier-failure rates."""

    pass_matrix = np.asarray(
        [
            [True, True, True, True, True, True],
            [False, True, True, True, True, True],
            [False, False, True, True, True, True],
            [False, False, False, True, True, True],
        ],
        dtype=bool,
    )

    alpha_i = exp1570.measure_alpha_i(pass_matrix)
    predicted = exp1570.jensen_lower_bound(alpha_i, beta=1.0)
    empirical = exp1570.corpus_soft_brs_acceptance_rate(pass_matrix, beta=1.0)
    rows = exp1570.evaluate_beta_grid(pass_matrix, exp1570.BETA_VALUES)
    optimal = exp1570.select_optimal_beta(rows)

    assert alpha_i == pytest.approx((0.75, 0.5, 0.25, 0.0, 0.0, 0.0))
    assert predicted == pytest.approx(math.exp(-1.5))
    assert empirical == pytest.approx(
        (1.0 + math.exp(-1.0) + math.exp(-2.0) + math.exp(-3.0)) / 4.0
    )
    assert empirical >= predicted
    assert [row["beta"] for row in rows] == list(exp1570.BETA_VALUES)
    assert optimal == 0.1

    with pytest.raises(ValueError, match="six verifier"):
        exp1570.measure_alpha_i(np.asarray([[True, False]], dtype=bool))


def test_scenario_sample_090_benchmark_verifies_bound_and_selects_beta(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-090: benchmark writes beta-wise Jensen and empirical curves."""

    source_a = _write_rows(tmp_path / "corpus_a.json", 320)
    source_b = _write_rows(tmp_path / "corpus_b.json", 300, start=320)
    config = exp1570.CoverageBoundConfig(
        n_cases=500,
        source_paths=(source_a, source_b),
        seed=1570,
    )

    artifact = exp1570.run_benchmark(config)

    assert artifact["status"] == "complete"
    assert artifact["metadata"]["spec_refs"] == ["REQ-SAMPLE-062", "SCENARIO-SAMPLE-090"]
    assert artifact["metadata"]["corpus_size"] == 500
    assert artifact["metadata"]["k6_verifier_names"] == list(exp1570.K6_VERIFIER_NAMES)
    assert len(artifact["alpha_i_per_verifier"]) == 6
    assert all(0.0 <= alpha <= 1.0 for alpha in artifact["alpha_i_per_verifier"])
    assert artifact["jensen_bound_holds_for_all_beta"] is True
    assert artifact["optimal_beta_for_deployment"] in exp1570.BETA_VALUES
    assert [row["beta"] for row in artifact["z_beta_jensen_bound"]] == list(exp1570.BETA_VALUES)
    assert [row["beta"] for row in artifact["z_beta_empirical"]] == list(exp1570.BETA_VALUES)
    assert all(
        empirical["empirical_acceptance_rate"] >= predicted["predicted_lower"]
        for predicted, empirical in zip(
            artifact["z_beta_jensen_bound"],
            artifact["z_beta_empirical"],
            strict=True,
        )
    )
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_sample_062_run_experiment_writes_complete_artifact(tmp_path: Path) -> None:
    """REQ-SAMPLE-062: runner writes the terminal Exp 1570 artifact schema."""

    source_a = _write_rows(tmp_path / "corpus_a.json", 260)
    source_b = _write_rows(tmp_path / "corpus_b.json", 260, start=260)
    output_path = tmp_path / "experiment_1570.json"
    config = exp1570.CoverageBoundConfig(
        n_cases=500,
        source_paths=(source_a, source_b),
        seed=1571,
    )

    artifact = exp1570.run_experiment(output_path=output_path, config=config)

    assert exp1570.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["acceptance_gates_passed"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_req_sample_062_validate_artifact_rejects_bad_terminal_values(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-062: artifacts require complete terminal semantics and gates."""

    source_a = _write_rows(tmp_path / "corpus_a.json", 260)
    source_b = _write_rows(tmp_path / "corpus_b.json", 260, start=260)
    valid = exp1570.run_benchmark(
        exp1570.CoverageBoundConfig(n_cases=500, source_paths=(source_a, source_b))
    )

    assert exp1570.validate_artifact(valid) is None

    missing = dict(valid)
    missing.pop("z_beta_empirical")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1570.validate_artifact(missing)

    bad_status = dict(valid, status="blocked")
    with pytest.raises(ValueError, match="status must be complete"):
        exp1570.validate_artifact(bad_status)

    bad_verdict = dict(valid, honest_verdict="bound verified")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1570.validate_artifact(bad_verdict)

    bad_alpha = dict(valid, alpha_i_per_verifier=[0.1])
    with pytest.raises(ValueError, match="six alpha"):
        exp1570.validate_artifact(bad_alpha)

    bad_gate = dict(valid, jensen_bound_holds_for_all_beta=False)
    with pytest.raises(ValueError, match="Jensen"):
        exp1570.validate_artifact(bad_gate)

    bad_beta = dict(valid, optimal_beta_for_deployment=3.0)
    with pytest.raises(ValueError, match="optimal_beta_for_deployment"):
        exp1570.validate_artifact(bad_beta)
