"""Tests for Exp5529 event/topic CSL memory residue stress.

Spec refs: REQ-LEARN-5529,
SCENARIO-LEARN-5529-PROMOTION,
SCENARIO-LEARN-5529-CONTROLS,
SCENARIO-LEARN-5529-ROLLBACK.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5529_csl_event_topic_residue_stress as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5529_csl_event_topic_residue_stress.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5529_csl_event_topic_residue_stress.py "
    "-m pytest tests/python/test_experiment_5529_csl_event_topic_residue_stress.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5529_csl_event_topic_residue_stress.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def test_req_learn_5529_spec_declares_event_topic_residue_contract() -> None:
    """REQ-LEARN-5529: OpenSpec anchors the bounded residue stress artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5529") :]

    for marker in (
        "REQ-LEARN-5529",
        "SCENARIO-LEARN-5529-PROMOTION",
        "SCENARIO-LEARN-5529-CONTROLS",
        "SCENARIO-LEARN-5529-ROLLBACK",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.UPSTREAM_GATE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "semantic-shift or verifier-change",
        "held-out answer label",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_5529_promotion_is_label_independent() -> None:
    """SCENARIO-LEARN-5529-PROMOTION: promotion does not read answer labels."""

    fixture = exp.build_fixture()
    before_event_hash = exp.hash_memory_state(exp.empty_event_memory())
    before_topic_hash = exp.hash_memory_state(exp.empty_topic_memory())

    memory = exp.build_memory_states(fixture)

    assert memory["event_memory_hash_before"] == before_event_hash
    assert memory["topic_memory_hash_before"] == before_topic_hash
    assert memory["event_memory_hash_after"] != before_event_hash
    assert memory["topic_memory_hash_after"] != before_topic_hash
    assert memory["semantic_shift_gate_used"] is True
    assert len(memory["promotion_records"]) == 2
    for record in memory["promotion_records"]:
        assert record["promotion_trigger"] in exp.PROMOTION_TRIGGERS
        assert record["trigger_used_answer_label"] is False
        assert "answer_label" not in record
        assert record["promoted_topic"] in {"database", "access"}


def test_scenario_learn_5529_conditions_and_artifact_fields(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5529-CONTROLS: six conditions quantify safe memory use."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH.name
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field]
    assert artifact["upstream_gate_path"] == exp.UPSTREAM_GATE_PATH.as_posix()
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["semantic_shift_gate_used"] is True
    assert artifact["no_memory_score"] == pytest.approx(0.0)
    assert artifact["event_only_score"] == pytest.approx(2 / 3)
    assert artifact["topic_only_score"] == pytest.approx(2 / 3)
    assert artifact["event_topic_score"] == pytest.approx(1.0)
    assert artifact["stale_memory_score"] == pytest.approx(0.0)
    assert artifact["adversarial_irrelevant_memory_score"] == pytest.approx(0.0)
    assert artifact["heldout_delta"] == pytest.approx(1.0)
    assert artifact["stale_evidence_rejection_rate"] == pytest.approx(1.0)
    assert artifact["negative_transfer_rate"] == pytest.approx(0.0)
    assert artifact["residue_contamination_rate"] == pytest.approx(0.0)
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["csl_residue_stress_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    label_sets = {
        condition: tuple(row["label_id"] for row in rows)
        for condition, rows in artifact["condition_results"].items()
    }
    assert set(label_sets) == set(exp.CONDITIONS)
    assert len(set(label_sets.values())) == 1
    assert artifact["condition_results"]["event_plus_topic"][2]["selected_action"] == (
        "deny-escalation"
    )
    assert artifact["condition_results"]["event_only"][2]["selected_action"] == (
        "grant-escalation"
    )
    assert artifact["control_counts"]["stale_candidates_seen"] == 1
    assert artifact["control_counts"]["stale_candidates_rejected"] == 1
    assert artifact["control_counts"]["negative_transfer_candidates_seen"] == 1
    assert artifact["control_counts"]["negative_transfer_candidates_accepted"] == 0


def test_scenario_learn_5529_rollback_restores_memory_and_validation_rejects_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5529-ROLLBACK: unsafe scratch state can be removed."""

    artifact = exp.run(
        root=REPO,
        result_path=tmp_path / exp.RESULT_RELATIVE_PATH.name,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    relative_artifact = exp.run(
        root=REPO,
        result_path=exp.RESULT_RELATIVE_PATH,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=False,
    )
    assert exp.validate_artifact(artifact) is True
    assert relative_artifact["event_topic_score"] == pytest.approx(1.0)
    rollback = artifact["rollback_evidence"]
    assert rollback["rollback_applied"] is True
    assert rollback["event_hash_restored"] is True
    assert rollback["topic_hash_restored"] is True
    assert rollback["scratch_event_hash"] != rollback["pre_event_hash"]
    assert rollback["scratch_topic_hash"] != rollback["pre_topic_hash"]

    drift_cases = [
        ("upstream_gate_path", "results/wrong.json", "upstream_gate_path"),
        ("semantic_shift_gate_used", False, "semantic_shift_gate_used"),
        ("event_memory_hash_after", artifact["event_memory_hash_before"], "event_memory_hash"),
        ("topic_memory_hash_after", artifact["topic_memory_hash_before"], "topic_memory_hash"),
        ("heldout_delta", 0.5, "heldout_delta"),
        ("stale_evidence_rejection_rate", 0.5, "stale_evidence_rejection_rate"),
        ("negative_transfer_rate", 0.5, "negative_transfer_rate"),
        ("residue_contamination_rate", 0.5, "residue_contamination_rate"),
        ("no_model_weight_mutation", False, "no_model_weight_mutation"),
        ("csl_residue_stress_ready", False, "csl_residue_stress_ready"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
    ]
    for field, value, expected in drift_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("topic_memory_hash_after")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("event_topic_score")
    missing_principle["reproducibility_checksum"] = exp.reproducibility_checksum(
        missing_principle
    )
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(missing_principle)

    bad_rollback = deepcopy(artifact)
    bad_rollback["rollback_evidence"]["event_hash_restored"] = False
    bad_rollback["reproducibility_checksum"] = exp.reproducibility_checksum(bad_rollback)
    with pytest.raises(ValueError, match="rollback_evidence"):
        exp.validate_artifact(bad_rollback)

    no_tests = deepcopy(artifact)
    no_tests["tests_added_or_reused"] = []
    no_tests["reproducibility_checksum"] = exp.reproducibility_checksum(no_tests)
    with pytest.raises(ValueError, match="tests_added_or_reused"):
        exp.validate_artifact(no_tests)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    blocked = deepcopy(artifact)
    blocked["residue_contamination_rate"] = 1.0
    assert exp.honest_verdict(blocked).startswith("blocked:")
