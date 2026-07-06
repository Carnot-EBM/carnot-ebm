"""Tests for Exp5312 deterministic memory transition verification.

Spec refs: REQ-LEARN-5312, SCENARIO-LEARN-5312, SCENARIO-LEARN-5313.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5312_trustmem_transition_verifier_self_learning_v485 as exp
from carnot.pipeline.memory_transition_verifier import MemoryTransitionVerifier


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5312_spec_declares_transition_verifier_contract() -> None:
    """REQ-LEARN-5312: OpenSpec anchors verifier labels, scores, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5312") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5312",
        "SCENARIO-LEARN-5312",
        "SCENARIO-LEARN-5313",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5302 adaptive memory policy",
        "Exp5303 memory stress",
        "coverage_score",
        "preservation_score",
        "faithfulness_score",
        "reject",
        "persistent state change",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for label in exp.REQUIRED_TRANSITION_LABELS:
        assert f"`{label}`" in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5312_fixture_has_all_transition_labels() -> None:
    """REQ-LEARN-5312-1: fixture covers every required transition label."""

    proposals = exp.build_transition_fixture()
    labels = [proposal.label for proposal in proposals]

    assert labels == list(exp.REQUIRED_TRANSITION_LABELS)
    assert exp.transition_label_counts(proposals) == {
        label: 1 for label in exp.REQUIRED_TRANSITION_LABELS
    }
    assert len({proposal.transition_id for proposal in proposals}) == len(proposals)
    assert all(proposal.source_stress_event_id for proposal in proposals)
    assert any(proposal.safe_expected for proposal in proposals)
    assert any(not proposal.safe_expected for proposal in proposals)


def test_req_learn_5312_verifier_scores_without_model_weight_mutation() -> None:
    """REQ-LEARN-5312-2: deterministic scores use state and evidence only."""

    verifier = MemoryTransitionVerifier()
    useful = exp.transition_by_label("useful_insert")
    corruption = exp.transition_by_label("corruption")

    useful_decision = verifier.verify(useful)
    assert useful_decision.accepted is True
    assert useful_decision.coverage_score == 1.0
    assert useful_decision.preservation_score == 1.0
    assert useful_decision.faithfulness_score == 1.0
    assert useful_decision.model_weights_mutated is False

    prior = deepcopy(corruption.prior_state)
    prior_snapshot = deepcopy(prior)
    corrupt_decision, committed_state = verifier.commit_if_safe(prior, corruption)

    assert corrupt_decision.accepted is False
    assert corrupt_decision.preservation_score == 0.0
    assert prior == prior_snapshot
    assert committed_state == prior_snapshot
    assert corrupt_decision.model_weights_mutated is False


def test_scenario_learn_5312_unsafe_transitions_reject_before_commit() -> None:
    """SCENARIO-LEARN-5312: unsafe writes are rejected with unchanged state."""

    evaluation = exp.evaluate_transition_fixture(exp.build_transition_fixture())
    by_label = {row["label"]: row for row in evaluation["transition_results"]}

    for label in exp.UNSAFE_TRANSITION_LABELS:
        assert by_label[label]["accepted"] is False
        assert by_label[label]["committed_state_changed"] is False
        assert by_label[label]["rejection_reasons"]

    assert by_label["omission"]["coverage_score"] == 0.0
    assert by_label["corruption"]["preservation_score"] == 0.0
    assert by_label["hallucinated_update"]["faithfulness_score"] == 0.5
    assert by_label["stale_retention"]["faithfulness_score"] == 0.0
    assert evaluation["unsafe_transition_rejection_rate"] == 1.0
    assert evaluation["unsafe_transition_rejections"] == 4
    assert evaluation["unsafe_transition_total"] == 4


def test_scenario_learn_5313_safe_transitions_commit_through_verifier() -> None:
    """SCENARIO-LEARN-5313: safe transitions are committed by verifier path."""

    evaluation = exp.evaluate_transition_fixture(exp.build_transition_fixture())
    by_label = {row["label"]: row for row in evaluation["transition_results"]}

    for label in exp.SAFE_TRANSITION_LABELS:
        assert by_label[label]["accepted"] is True
        assert by_label[label]["committed_state_changed"] is True
        assert by_label[label]["coverage_score"] == 1.0
        assert by_label[label]["preservation_score"] == 1.0
        assert by_label[label]["faithfulness_score"] == 1.0

    assert evaluation["safe_transition_commits"] == 4
    assert evaluation["safe_transition_total"] == 4
    assert evaluation["coverage_score"] == 1.0
    assert evaluation["preservation_score"] == 1.0
    assert evaluation["faithfulness_score"] == 1.0
    assert evaluation["memory_transition_verifier_ready"] is True
    assert evaluation["no_model_weight_mutation"] is True


def test_req_learn_5312_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5312-5: run() writes the required Exp5312 artifact."""

    tests_run = [{"command": "unit transition verifier", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "ready_for_exp5313"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning"] is True
    assert artifact["memory_transition_verifier_ready"] is True
    assert artifact["verifier_path"]["value"] == exp.VERIFIER_PATH
    assert artifact["transition_label_counts"]["value"] == {
        label: 1 for label in exp.REQUIRED_TRANSITION_LABELS
    }
    assert artifact["coverage_score"] == 1.0
    assert artifact["preservation_score"] == 1.0
    assert artifact["faithfulness_score"] == 1.0
    assert artifact["unsafe_transition_rejection_rate"] == 1.0
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5312_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5312: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["continuous_self_learning"] is True
    assert result["memory_transition_verifier_ready"] is True
    exp.validate_artifact(result)


def test_req_learn_5312_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5312-5: artifact validation rejects downstream gate drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit transition verifier", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "offline_deterministic_fixture_no_llm"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_learning_gate = deepcopy(artifact)
    bad_learning_gate["continuous_self_learning"] = False
    with pytest.raises(ValueError, match="continuous_self_learning"):
        exp.validate_artifact(bad_learning_gate)

    bad_ready = deepcopy(artifact)
    bad_ready["memory_transition_verifier_ready"] = {"value": True}
    with pytest.raises(ValueError, match="memory_transition_verifier_ready"):
        exp.validate_artifact(bad_ready)

    bad_counts = deepcopy(artifact)
    del bad_counts["transition_label_counts"]["value"]["rollback"]
    with pytest.raises(ValueError, match="transition_label_counts"):
        exp.validate_artifact(bad_counts)

    bad_score = deepcopy(artifact)
    bad_score["coverage_score"] = "1.0"
    with pytest.raises(ValueError, match="coverage_score"):
        exp.validate_artifact(bad_score)

    bad_rate = deepcopy(artifact)
    bad_rate["unsafe_transition_rejection_rate"] = 0.75
    with pytest.raises(ValueError, match="unsafe_transition_rejection_rate"):
        exp.validate_artifact(bad_rate)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = [{"command": "lost principle", "outcome": "passed"}]
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)
