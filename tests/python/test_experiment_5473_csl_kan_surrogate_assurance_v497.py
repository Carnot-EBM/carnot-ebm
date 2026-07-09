"""Tests for Exp5473 CSL KAN surrogate assurance.

Spec refs: REQ-LEARN-5473,
SCENARIO-LEARN-5473-ROLLBACK,
SCENARIO-LEARN-5473-NEGATIVE-TRANSFER,
SCENARIO-LEARN-5473-MONOTONICITY,
SCENARIO-LEARN-5473-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5473_csl_kan_surrogate_assurance_v497 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5473_csl_kan_surrogate_assurance_v497.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5473_csl_kan_surrogate_assurance_v497.py "
    "-m pytest tests/python/test_experiment_5473_csl_kan_surrogate_assurance_v497.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5473_csl_kan_surrogate_assurance_v497.py "
    "--fail-under=100"
)


def _complete_artifact() -> dict[str, object]:
    return mod.build_artifact(root=REPO, tests_run=[TEST_COMMAND, COVERAGE_COMMAND])


def test_req_learn_5473_spec_declares_surrogate_contract() -> None:
    """REQ-LEARN-5473: OpenSpec anchors the V497 surrogate assurance lane."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5473") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5473",
        "SCENARIO-LEARN-5473-ROLLBACK",
        "SCENARIO-LEARN-5473-NEGATIVE-TRANSFER",
        "SCENARIO-LEARN-5473-MONOTONICITY",
        "SCENARIO-LEARN-5473-ARTIFACT",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "context_cost`, `verifier_cost`, `prior_success`, `conflict_risk`, "
        "`memory_age`, and `constraint_violation_history",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    assert " ".join(mod.FEATURE_NAMES) in normalized


def test_req_learn_5473_builds_interpretable_surrogate_artifact() -> None:
    """REQ-LEARN-5473-1/4/6: artifact fields and baseline split are auditable."""

    artifact = _complete_artifact()

    mod.validate_artifact(artifact)
    assert artifact["csl_kan_surrogate_ready"] is True
    assert artifact["surrogate_feature_names"] == list(mod.FEATURE_NAMES)
    assert artifact["surrogate_coefficients_or_basis"]["intercept"] == pytest.approx(
        mod.KanStyleSurrogate().intercept
    )
    assert set(artifact["surrogate_coefficients_or_basis"]["coefficients"]) == set(
        mod.FEATURE_NAMES
    )
    assert artifact["assurance_ratio"] == pytest.approx(1.0)
    assert artifact["threshold_offset"] > 0.0
    assert artifact["accepted_action_rate"] == pytest.approx(1.0)
    assert artifact["constraint_violation_count"] == 0
    assert artifact["rollback_trigger_count"] == 1
    assert artifact["negative_transfer_deflection_rate"] == pytest.approx(1.0)
    assert artifact["no_memory_baseline_score"] == pytest.approx(0.25)
    assert artifact["naive_icl_baseline_score"] == pytest.approx(0.5)
    assert artifact["governed_policy_score"] == pytest.approx(1.0)
    assert artifact["model_weight_mutation"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["honest_verdict"].startswith("complete:")

    local_basis = artifact["surrogate_coefficients_or_basis"]["local_basis"]
    assert local_basis
    assert all(set(row["basis_terms"]) == set(mod.FEATURE_NAMES) for row in local_basis)
    assert all("acceptance_margin" in row for row in artifact["surrogate_rows"])


def test_scenario_learn_5473_rollback_excludes_retired_evidence() -> None:
    """SCENARIO-LEARN-5473-ROLLBACK: rollback rows leave active assurance."""

    active = mod.synthetic_surrogate_row(
        task_id="active-safe",
        features=mod.feature_vector(
            context_cost=40,
            verifier_cost=2,
            prior_success=1.0,
            conflict_risk=0.0,
            memory_age=0.0,
            constraint_violation_history=0.0,
        ),
    )
    rolled_back = mod.synthetic_surrogate_row(
        task_id="rolled-back-safe",
        rollback_required=True,
        features=mod.feature_vector(
            context_cost=40,
            verifier_cost=2,
            prior_success=1.0,
            conflict_risk=0.0,
            memory_age=0.0,
            constraint_violation_history=0.0,
        ),
    )
    scored = mod.score_surrogate_rows([active, rolled_back])
    assurance = mod.compute_assurance(scored)

    assert assurance["rollback_trigger_count"] == 1
    assert assurance["candidate_action_count"] == 1
    assert assurance["accepted_action_count"] == 1
    assert assurance["accepted_action_rate"] == pytest.approx(1.0)
    assert assurance["assurance_ratio"] == pytest.approx(1.0)
    assert scored[1]["active_for_assurance"] is False


def test_scenario_learn_5473_negative_transfer_deflection() -> None:
    """SCENARIO-LEARN-5473-NEGATIVE-TRANSFER: risky memory is rejected."""

    unsafe = mod.synthetic_surrogate_row(
        task_id="poisoned-naive-memory",
        condition="naive_icl",
        accepted_by_final_authority=False,
        negative_transfer_candidate=True,
        negative_transfer_detected=True,
        features=mod.feature_vector(
            context_cost=45,
            verifier_cost=2,
            prior_success=0.95,
            conflict_risk=1.0,
            memory_age=1.0,
            constraint_violation_history=1.0,
        ),
    )
    scored = mod.score_surrogate_rows([unsafe])

    assert scored[0]["surrogate_accept"] is False
    assert scored[0]["threshold_offset"] > 0.0
    assert scored[0]["basis_terms"]["conflict_risk"] < 0.0
    assert mod.negative_transfer_deflection_rate(scored) == pytest.approx(1.0)


def test_scenario_learn_5473_surrogate_monotonicity() -> None:
    """SCENARIO-LEARN-5473-MONOTONICITY: risk lowers score and tightens threshold."""

    surrogate = mod.KanStyleSurrogate()
    low_risk = mod.feature_vector(
        context_cost=60,
        verifier_cost=2,
        prior_success=0.8,
        conflict_risk=0.0,
        memory_age=0.0,
        constraint_violation_history=0.0,
    )
    high_risk = mod.feature_vector(
        context_cost=60,
        verifier_cost=2,
        prior_success=0.8,
        conflict_risk=1.0,
        memory_age=1.0,
        constraint_violation_history=1.0,
    )
    low = surrogate.score(low_risk)
    high = surrogate.score(high_risk)

    assert high["surrogate_score"] < low["surrogate_score"]
    assert high["threshold_offset"] > low["threshold_offset"]
    assert high["acceptance_margin"] < low["acceptance_margin"]
    assert high["basis_terms"]["constraint_violation_history"] < 0.0


def test_scenario_learn_5473_run_writes_stable_json(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5473-ARTIFACT: run() writes the required deliverable."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=REPO,
        result_path=destination,
        tests_run=[TEST_COMMAND, COVERAGE_COMMAND],
        write=True,
    )
    no_write_path = tmp_path / "no-write.json"
    no_write = mod.run(
        root=REPO,
        result_path=no_write_path,
        tests_run=[TEST_COMMAND, COVERAGE_COMMAND],
        write=False,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == artifact
    assert no_write == artifact
    assert not no_write_path.exists()
    assert artifact["csl_kan_surrogate_ready"] is True
    mod.validate_artifact(artifact)


def test_req_learn_5473_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5473-6: checked-in deliverable is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["csl_kan_surrogate_ready"] is True
    assert result["model_weight_mutation"] is False


def test_req_learn_5473_validation_rejects_drift() -> None:
    """REQ-LEARN-5473-6: validator fails closed on artifact drift."""

    artifact = _complete_artifact()
    cases = [
        ("assurance_ratio", None, "assurance_ratio"),
        ("surrogate_feature_names", ["context_cost"], "surrogate_feature_names"),
        ("surrogate_coefficients_or_basis", {}, "surrogate_coefficients_or_basis"),
        ("model_weight_mutation", True, "model_weight_mutation"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("constraint_violation_count", 1, "constraint_violation_count"),
        ("rollback_trigger_count", 0, "rollback_trigger_count"),
        ("negative_transfer_deflection_rate", 0.0, "negative_transfer_deflection_rate"),
        ("csl_kan_surrogate_ready", False, "csl_kan_surrogate_ready"),
        ("honest_verdict", "done", "honest_verdict"),
        ("research_conductor_modified", True, "research_conductor.py"),
    ]
    for field, value, expected in cases:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("threshold_offset")
    with pytest.raises(ValueError, match="threshold_offset"):
        mod.validate_artifact(missing)
