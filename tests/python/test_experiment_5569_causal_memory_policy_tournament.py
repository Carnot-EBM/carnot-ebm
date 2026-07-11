"""Tests for Exp5569 causal memory policy tournament.

Spec refs: REQ-LEARN-5569,
SCENARIO-LEARN-5569-STREAM,
SCENARIO-LEARN-5569-TOURNAMENT,
SCENARIO-LEARN-5569-ROLLBACK,
SCENARIO-LEARN-5569-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5569_causal_memory_policy_tournament as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5569_causal_memory_policy_tournament.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5569_causal_memory_policy_tournament.py "
    "-m pytest tests/python/test_experiment_5569_causal_memory_policy_tournament.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5569_causal_memory_policy_tournament.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def _artifact() -> dict[str, object]:
    return exp.build_artifact(root=REPO, tests_added_or_reused=TESTS_ADDED_OR_REUSED)


def test_req_learn_5569_spec_declares_policy_tournament_contract() -> None:
    """REQ-LEARN-5569: OpenSpec anchors the bounded memory tournament."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5569") :]

    for marker in (
        "REQ-LEARN-5569",
        "SCENARIO-LEARN-5569-STREAM",
        "SCENARIO-LEARN-5569-TOURNAMENT",
        "SCENARIO-LEARN-5569-ROLLBACK",
        "SCENARIO-LEARN-5569-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "future_label_leakage_count=0",
        "policy_ready",
        "confidence interval excluding zero",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert exp.FIELD_PRINCIPLES[field]


def test_scenario_learn_5569_stream_builds_exact_asp_fsm_without_leakage() -> None:
    """SCENARIO-LEARN-5569-STREAM: exact sessions avoid future labels."""

    sessions = exp.build_sessions()
    events = exp.flatten_events(sessions)

    assert len(sessions) >= 5
    assert len(events) == 120
    assert {session["family_kind"] for session in sessions} == {"exact_asp", "exact_fsm"}
    assert exp.future_label_leakage_count(sessions) == 0
    assert all("expected_action" not in event["memory_visible"] for event in events)
    assert all(event["expected_action"] == exp.exact_label(event) for event in events)


def test_scenario_learn_5569_tournament_optimized_beats_static_with_ci() -> None:
    """SCENARIO-LEARN-5569-TOURNAMENT: optimized bounded policy wins."""

    tournament = exp.run_tournament(exp.build_sessions(), exp.DEFAULT_SEEDS)
    static = tournament["arm_summary"][exp.STATIC_CAUSAL_ARM]
    optimized = tournament["arm_summary"][exp.SELF_OPTIMIZED_CAUSAL_ARM]

    assert set(tournament["arms"]) == set(exp.ARM_NAMES)
    assert tournament["seeds"] == list(exp.DEFAULT_SEEDS)
    assert optimized["heldout_exact_success"] > static["heldout_exact_success"]
    assert tournament["optimized_vs_static_ci"]["lower"] > 0.0
    assert tournament["prior_family_regression_max"] <= 0.02
    assert tournament["future_label_leakage_count"] == 0
    assert tournament["policy_search_space"] == list(exp.POLICY_SEARCH_SPACE)
    assert all(
        choice["operation"] in exp.POLICY_SEARCH_SPACE
        and choice["feedback_source"] == "past_exact_energy"
        for choice in tournament["optimization_trace"]
    )


def test_scenario_learn_5569_poisoning_control_rolls_back_to_clean_checkpoint() -> None:
    """SCENARIO-LEARN-5569-ROLLBACK: poisoned memory cannot persist."""

    control = exp.poisoning_control(exp.build_sessions()[0])

    assert control["poisoned_memory_inserted"] is True
    assert control["positive_control_induced_failure"] is True
    assert control["rollback_to_checkpoint"] == "checkpoint-clean-000"
    assert control["poisoned_rows_active_after_rollback"] == 0
    assert control["rollback_burden"] > 0.0
    assert exp.rollback_success(control) is True


def test_scenario_learn_5569_artifact_fields_and_stable_write(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5569-ARTIFACT: receipt exposes gate evidence."""

    destination = tmp_path / exp.RESULT_RELATIVE_PATH.name
    artifact = exp.run(
        root=REPO,
        result_path=destination,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    no_write = exp.run(
        root=REPO,
        result_path=exp.RESULT_RELATIVE_PATH,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=False,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == artifact
    assert no_write["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert exp.validate_artifact(artifact) is True
    assert exp.resolve_path(REPO, exp.RESULT_RELATIVE_PATH) == REPO / exp.RESULT_RELATIVE_PATH
    assert exp.resolve_path(REPO, destination) == destination

    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field]

    assert artifact["continuous_self_learning_target"] is True
    assert len(artifact["sessions"]) >= 5
    assert artifact["n_events"] == 120
    assert artifact["seeds"] == list(exp.DEFAULT_SEEDS)
    assert artifact["arms"] == list(exp.ARM_NAMES)
    assert artifact["future_label_leakage_count"] == 0
    assert artifact["policy_search_space"] == list(exp.POLICY_SEARCH_SPACE)
    assert artifact["weights_mutated"] is False
    assert artifact["forward_transfer_delta"] > 0.0
    assert artifact["backward_retention_delta"] >= -0.02
    assert artifact["action_impact_delta"] > 0.0
    assert 0.0 < artifact["memory_precision"] < 1.0
    assert artifact["retrieval_cost"] > 0.0
    assert artifact["write_amplification"] > 0.0
    assert artifact["poisoning_control"]["positive_control_induced_failure"] is True
    assert artifact["rollback_success"] is True
    assert artifact["promotion_thresholds"]["ci_lower_bound_gt_zero"] is True
    assert artifact["policy_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED


def test_req_learn_5569_policy_gate_fails_closed_on_regressions() -> None:
    """REQ-LEARN-5569-6: promotion gate rejects unsafe policy artifacts."""

    artifact = _artifact()
    assert exp.validate_artifact(artifact) is True

    bad_cases = [
        ("continuous_self_learning_target", False, "continuous_self_learning_target"),
        ("sessions", [], "sessions"),
        ("n_events", 119, "n_events"),
        ("seeds", [1, 2, 3, 4], "seeds"),
        ("arms", list(exp.ARM_NAMES[:-1]), "arms"),
        ("future_label_leakage_count", 1, "future_label_leakage_count"),
        ("policy_search_space", ["write_verified"], "policy_search_space"),
        ("weights_mutated", True, "weights_mutated"),
        ("forward_transfer_delta", 0.0, "forward_transfer_delta"),
        ("backward_retention_delta", -0.03, "backward_retention_delta"),
        ("action_impact_delta", 0.0, "action_impact_delta"),
        ("memory_precision", 0.0, "memory_precision"),
        ("retrieval_cost", 0.0, "retrieval_cost"),
        ("write_amplification", 0.0, "write_amplification"),
        ("poisoning_control", {}, "poisoning_control"),
        ("rollback_success", False, "rollback_success"),
        ("field_principles", {}, "field_principles"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
    ]
    for field, value, expected in bad_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    bad_ci = deepcopy(artifact)
    bad_ci["optimized_vs_static_ci"]["lower"] = 0.0
    bad_ci["policy_ready"] = False
    bad_ci["honest_verdict"] = exp.honest_verdict(bad_ci)
    bad_ci["reproducibility_checksum"] = exp.reproducibility_checksum(bad_ci)
    assert exp.validate_artifact(bad_ci) is True
    assert bad_ci["honest_verdict"].startswith("blocked:")

    invalid_claim = deepcopy(bad_ci)
    invalid_claim["policy_ready"] = True
    invalid_claim["honest_verdict"] = "complete: invalid"
    invalid_claim["reproducibility_checksum"] = exp.reproducibility_checksum(invalid_claim)
    with pytest.raises(ValueError, match="policy_ready"):
        exp.validate_artifact(invalid_claim)

    missing = deepcopy(artifact)
    missing.pop("policy_ready")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    bad_principle_type = deepcopy(artifact)
    bad_principle_type["field_principles"] = "not-a-mapping"
    bad_principle_type["reproducibility_checksum"] = exp.reproducibility_checksum(
        bad_principle_type
    )
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle_type)

    single_seed_ci = exp.confidence_interval([0.125])
    assert single_seed_ci == {"mean": 0.125, "lower": 0.125, "upper": 0.125, "n": 1}
