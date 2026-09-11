"""Tests for the bounded-acquisition cold and causal audit.

Spec refs: REQ-CL-7200 and SCENARIO-CL-7200-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7200_v634_acquisition_cold_audit as exp
from carnot import experiment_7199_v634_bounded_acquisition as producer


REPO_ROOT = Path(__file__).resolve().parents[2]


def public(event_id: str, value: int) -> dict[str, object]:
    """Build one event that contains no outcome authority fields."""

    return {
        "event_id": event_id,
        "family_id": "modular_equals",
        "numeric_value": value,
        "public_input": f"family=modular_equals;value={value}",
    }


def learned_state() -> dict[str, object]:
    """Create a controller state where template storage duplicates a singleton."""

    controller = producer.VersionSpaceController()
    state = controller.families["modular_equals"]
    state.hypotheses = {7}
    state.support_ids = ["s1", "s2", "s3"]
    state.validation_ids = [f"v{index}" for index in range(8)]
    state.candidate_parameter = 7
    state.freeze_release_index = 40
    state.committed_template = {
        "family_id": "modular_equals",
        "parameter": 7,
        "predicate": "modular_equals(parameter=7)",
    }
    state.archive = [["s1", "support", 1, 2, "accept", 0]]
    return controller.state_dict()


def complete_fixture() -> dict[str, object]:
    """Build the smallest internally complete terminal artifact for validation tests."""

    artifact = exp.base_artifact(
        checks=[exp.gate_check("fixture", "test", "ready", True, True)],
        upstream={
            "acquisition_run_complete_score": 1,
            "acquisition_value_score": 0,
            "status": "complete",
            "run_date": exp.RUN_DATE,
        },
        source_hashes={},
        run_date=exp.RUN_DATE,
        duration_s=1.0,
        checkpoint_path=Path("results/checkpoints/test.json"),
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": exp.INFERENCE_SUBSTRATE,
            "inference_substrate_class": exp.INFERENCE_SUBSTRATE_CLASS,
            "rows": [
                {
                    "unit_id": "1:4:burst:priority_admission",
                    "arm": "priority_admission",
                    "seed": 1,
                    "metric": "error_rate",
                    "error": 0,
                    "abstention": 0,
                    "passed": True,
                }
            ],
            "reconstruction_rows": [{"passed": True}],
            "metric_recomputation_rows": [{"passed": True}],
            "causal_control_rows": [
                {"control": "delayed_feedback", "passed": True},
                {"control": "shuffled_feedback", "passed": True},
                {"control": "template_only_deletion", "passed": True},
            ],
            "cold_reload_rows": [{"passed": True}],
            "rollback_rows": [{"byte_equal": True, "decision_parity": True, "passed": True}],
            "whole_learning_reset_rows": [
                {"passed": True, "acquisition_dependence_established": True}
            ],
            "mutation_rows": [
                {"mutation_id": name, "passed": True} for name in exp.REQUIRED_MUTATIONS
            ],
            "source_grounding_rows": [{"passed": True}],
            "acquisition_audit_complete_score": 1,
            "memory_promotion_score": 0,
            "verdict_class": "null",
            "honest_verdict": "complete_null: upstream acquisition value was null",
            "runtime_isolation_receipt": {
                "fresh_process": True,
                "no_model_load": True,
                "network_disabled": True,
                "protected_inputs_unchanged": True,
            },
            "scheduler_control_summary": {
                "priority_random_evidence_distinct": True,
                "priority_random_decisions_distinct": True,
                "scheduling_benefit_supported": True,
            },
            "checkpoint_receipt": {"path": "results/checkpoints/test.json"},
        }
    )
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    return artifact


def test_contract_declares_cpu_audit_without_a_model() -> None:
    """REQ-CL-7200: The schema and required field principles are exact."""

    assert exp.EXPERIMENT_ID == 7200
    assert exp.RUN_DATE == "20260911"
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert exp.INFERENCE_SUBSTRATE_CLASS == "cpu_exact_solver_or_simulator"
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.FIELD_PRINCIPLES["field_principles"] == (
        "Echo each declared reason beside the actual evidence contract."
    )


def test_independent_reducer_ignores_producer_outcome_fields() -> None:
    """SCENARIO-CL-7200-RECONSTRUCTION: Raw decisions are independently scored."""

    decisions = [
        {
            "unit_id": "1:4:burst:priority_admission",
            "event_id": "e1",
            "arm": "priority_admission",
            "seed": 1,
            "capacity": 4,
            "delay_schedule": "burst",
            "prediction": "accept",
            "exact_label": "accept",
            "error": 0,
            "abstention": 0,
            "window": "prospective",
        },
        {
            "unit_id": "1:4:burst:priority_admission",
            "event_id": "e2",
            "arm": "priority_admission",
            "seed": 1,
            "capacity": 4,
            "delay_schedule": "burst",
            "prediction": "abstain",
            "exact_label": "accept",
            "error": 0,
            "abstention": 0,
            "window": "prospective",
        },
    ]
    truth = {"e1": {"exact_label": "reject"}, "e2": {"exact_label": "accept"}}
    rows = exp.independent_metric_rows(decisions, truth)
    assert [row["error"] for row in rows] == [1, 1]
    assert [row["false_accept"] for row in rows] == [1, 0]
    assert rows[1]["abstention"] == 1
    assert all(row["authority_read_after_action_seal"] for row in rows)


def test_template_only_and_whole_learning_deletions_are_distinct() -> None:
    """SCENARIO-CL-7200-DELETION: A duplicate template is not the learned state."""

    state = learned_state()
    warmup = producer.VersionSpaceController().state_dict()
    template_state, template_receipt = exp.apply_deletion_intervention(
        state, warmup, "template_only"
    )
    whole_state, whole_receipt = exp.apply_deletion_intervention(state, warmup, "whole_learning")

    restored = exp.controller_from_state(template_state)
    assert restored.families["modular_equals"].hypotheses == {7}
    assert restored.families["modular_equals"].committed_template is None
    assert restored.predict(public("probe", 7))[0] == "accept"
    assert template_receipt["version_spaces_reset"] is False
    assert template_receipt["role_separation_preserved"] is True
    assert template_receipt["candidate_freeze_timestamps_preserved"] is True

    assert whole_state == warmup
    assert whole_receipt["version_spaces_reset"] is True
    assert whole_receipt["archived_future_feedback_reapplied"] is False
    assert whole_receipt["pending_released_labels_reapplied"] is False
    with pytest.raises(ValueError, match="unknown_deletion_intervention"):
        exp.apply_deletion_intervention(state, warmup, "unknown")


def test_rollback_restores_bytes_hash_and_behavior() -> None:
    """SCENARIO-CL-7200-COLD-ROLLBACK: Rejected poison restores exact behavior."""

    state = learned_state()
    probes = [public("accept", 7), public("reject", 8)]
    row = exp.rollback_probe(state, probes)
    assert row["adverse_hash"] != row["checkpoint_hash"]
    assert row["byte_equal"] is True
    assert row["hash_equal"] is True
    assert row["decision_parity"] is True
    assert row["passed"] is True


def test_causal_mutations_are_detected_and_fake_noop_fails() -> None:
    """SCENARIO-CL-7200-ATTACKS: Each prohibited causal claim is fail-closed."""

    baseline = {
        "authority_read_after_action_seal": True,
        "release_index": 8,
        "decision_index": 4,
        "operation": "support_eliminate",
        "state_hash_before": "a",
        "state_hash_after": "b",
        "max_pending": 4,
        "capacity": 4,
        "max_memory_bytes": 10,
        "memory_byte_budget": 20,
    }
    assert exp.causal_evidence_errors([baseline]) == []
    mutations = exp.build_mutation_rows(baseline)
    assert {row["mutation_id"] for row in mutations} == set(exp.REQUIRED_MUTATIONS)
    assert all(row["passed"] for row in mutations)
    fake = deepcopy(baseline)
    fake["state_hash_after"] = "a"
    assert "fake_noop_update" in exp.causal_evidence_errors([fake])


def test_scheduler_credit_requires_distinct_evidence_and_decisions() -> None:
    """SCENARIO-CL-7200-CONTROLS: Identical schedules receive no priority credit."""

    same = exp.scheduler_control_summary(["a", "b"], ["a", "b"], ["x"], ["x"])
    assert same["scheduling_benefit_supported"] is False
    changed = exp.scheduler_control_summary(["a"], ["b"], ["x"], ["y"])
    assert changed["scheduling_benefit_supported"] is True


def test_shuffled_feedback_keeps_the_real_admission_schedule() -> None:
    """SCENARIO-CL-7200-CONTROLS: Shuffle keeps capacity and release slots matched."""

    views = producer.load_upstream_views(REPO_ROOT, producer.DEFAULT_UPSTREAM_ARTIFACT_PATH)
    seed = producer.STREAM_SEEDS[0]
    baseline = exp.replay_control(views, seed=seed, arm="priority_admission")
    shuffled_ids = exp._shuffle_after_warmup(baseline, seed, "priority_admission")
    schedule = [
        row
        for row in baseline["request_records"]
        if int(row["request_index"]) >= producer.WARMUP_COUNT
    ]
    shuffled = exp.replay_control(
        views,
        seed=seed,
        arm="priority_admission",
        feedback_mode="shuffled",
        shuffled_feedback_ids=shuffled_ids,
        matched_request_schedule=schedule,
    )
    assert len(shuffled["request_records"]) == len(baseline["request_records"])
    assert sorted(shuffled_ids) == sorted(row["event_id"] for row in schedule)
    assert shuffled["max_pending"] <= 4


def test_preconditions_parse_real_gates_and_reject_quarantine(tmp_path: Path) -> None:
    """SCENARIO-CL-7200-PRECONDITIONS: Quarantine is checked before stream loading."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, upstream, _hashes = exp.collect_preconditions(
        REPO_ROOT, exp.DEFAULT_PRODUCER_PATH, paths
    )
    assert all(row["passed"] for row in checks)
    assert upstream["acquisition_run_complete_score"] == 1
    assert upstream["acquisition_value_score"] == 0

    quarantined = {
        "status": "complete",
        "run_date": exp.RUN_DATE,
        "milestone": exp.MILESTONE,
        "acquisition_run_complete_score": 1,
        "acquisition_value_score": 0,
        "gate_check_summary": {"passed": True},
        "MODEL_SPECS": [],
        "model_invoked": False,
        "flagged_adversarial": True,
    }
    fixture = tmp_path / "quarantined.json"
    fixture.write_text(json.dumps(quarantined), encoding="utf-8")
    failed, _upstream, _hashes = exp.collect_preconditions(REPO_ROOT, fixture, paths)
    gate = next(row for row in failed if row["check"] == "upstream_not_quarantined")
    assert gate["passed"] is False
    assert gate["observed_value"] is True


def test_missing_upstream_builds_terminal_row_free_block(tmp_path: Path) -> None:
    """SCENARIO-CL-7200-PRECONDITIONS: External absence is blocked, not partial."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, upstream, hashes = exp.collect_preconditions(
        tmp_path / "missing-root", Path("missing.json"), paths
    )
    artifact = exp.build_blocked_artifact(
        checks,
        upstream,
        source_hashes=hashes,
        run_date=exp.RUN_DATE,
        duration_s=0.01,
        checkpoint_path=paths.checkpoint,
    )
    assert artifact["status"] == artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_external:")
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["rows"] == artifact["causal_control_rows"] == []
    assert artifact["gate_check_summary"]["failed_check"]
    assert exp.validate_artifact(artifact) == []


def test_complete_null_finishes_audit_without_promotion() -> None:
    """SCENARIO-CL-7200-TERMINAL: A complete causal audit can retain a null."""

    artifact = complete_fixture()
    assert exp.expected_complete(artifact) == 1
    assert exp.expected_promotion(artifact) == 0
    assert exp.validate_artifact(artifact) == []

    promoted = deepcopy(artifact)
    promoted["upstream_gate_receipt"]["acquisition_value_score"] = 1
    promoted["memory_promotion_score"] = 1
    promoted["verdict_class"] = "positive"
    promoted["honest_verdict"] = "complete_positive: causal acquisition is supported"
    promoted["reproducibility_checksum"] = exp.reproducibility_checksum(promoted)
    assert exp.expected_promotion(promoted) == 1
    assert exp.validate_artifact(promoted) == []


def test_command_refuses_an_invalid_worker_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7200: The command validates before the terminal atomic write."""

    monkeypatch.setattr(exp, "spawn_audit_worker", lambda _args: {"verdict_class": "null"})
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced_error"])
    with pytest.raises(RuntimeError, match="cold audit artifact validation failed:forced_error"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)])
    assert not exp.ExperimentPaths.under(tmp_path).artifact.exists()
