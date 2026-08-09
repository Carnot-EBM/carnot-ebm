"""Tests for Exp6219 two-timescale constraint CSL.

Spec refs: REQ-LEARN-6219, SCENARIO-LEARN-6219-SNAPSHOTS,
SCENARIO-LEARN-6219-TWO-TIMESCALE, SCENARIO-LEARN-6219-ATTACKS,
SCENARIO-LEARN-6219-ROLLBACK.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot import experiment_6219_two_timescale_constraint_csl as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, object]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=write,
    )


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    artifact["continuous_learning_promotion_ready_score"] = mod.ready_score(artifact)
    artifact["status"] = mod.status(artifact)
    artifact["honest_verdict"] = mod.honest_verdict(artifact)
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_6219_spec_declares_contract_fields_and_scenarios() -> None:
    """REQ-LEARN-6219: OpenSpec owns the artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6219") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-6219-1",
        "REQ-LEARN-6219-7",
        "SCENARIO-LEARN-6219-SNAPSHOTS",
        "SCENARIO-LEARN-6219-TWO-TIMESCALE",
        "SCENARIO-LEARN-6219-ATTACKS",
        "SCENARIO-LEARN-6219-ROLLBACK",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        *mod.ARM_NAMES,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6219_artifact_writes_required_temporal_receipts(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6219-SNAPSHOTS: decisions read only pre-event snapshots."""

    artifact = _artifact(tmp_path, write=True)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["train_eval_overlap_count"] == 0
    assert artifact["decision_time_write_count"] == 0
    assert artifact["model_weight_mutation_count"] == 0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    upstream = artifact["upstream_stream_path_hash_and_clean_receipt"]
    assert upstream["exp6145_status"] == "complete_ready"
    assert upstream["exp6145_flagged_adversarial"] is False
    assert upstream["exact_oracle_separated"] is True
    assert upstream["row_count"] == upstream["outcome_count"] == 240

    freeze = artifact["preregistered_chronological_family_blocks_and_future_ids"]
    assert freeze["frozen_before_arm_runs"] is True
    assert freeze["family_block_count"] == 8
    assert freeze["held_future_event_count"] == 120
    assert freeze["memory_record_budget"] == mod.MEMORY_RECORD_BUDGET

    arms = artifact["arm_definitions_and_resource_parity"]
    assert arms["arm_names"] == list(mod.ARM_NAMES)
    assert arms["all_arms_resource_matched"] is True
    assert arms["decision_count_by_arm"] == {arm: 240 for arm in mod.ARM_NAMES}

    snapshots = artifact["immutable_predecision_snapshot_hashes"]
    assert snapshots["snapshot_count"] == 240 * len(mod.ARM_NAMES)
    assert snapshots["read_only"] is True
    assert snapshots["current_outcome_visible_count"] == 0
    assert snapshots["unique_snapshot_hash_count"] > 0

    post = artifact["post_outcome_event_and_verifier_receipts"]
    assert post["all_verifier_receipts_after_outcome"] is True
    assert post["accepted_update_count"] > 0
    assert post["rejected_update_count"] > 0
    assert post["verifier_backend"] == "exp6145_python_z3_exact_sidecar"

    counts = artifact["promoted_quarantined_rejected_and_rolled_back_counts"]
    assert counts["promoted"] == post["accepted_update_count"]
    assert counts["quarantined"] >= 5
    assert counts["rejected"] > 0
    assert counts["rolled_back"] == 1

    assert artifact["rollback_exactness"]["active_store_restored"] is True
    assert artifact["rollback_exactness"]["decision_trace_restored"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert mod.validate_artifact(artifact) is True


def test_scenario_6219_two_timescale_metrics_and_shuffled_control(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6219-TWO-TIMESCALE: arms separate update timing."""

    artifact = _artifact(tmp_path)

    immediate = artifact["immediate_commit_log"]
    block_end = artifact["block_end_consolidation_log"]
    shuffled = artifact["shuffled_memory_receipt"]
    metrics = artifact["accuracy_forward_transfer_retention_and_negative_transfer_by_family_arm"]

    assert immediate["visible_same_event_count"] == 0
    assert immediate["first_visible_event_index"] > immediate["first_commit_event_index"]
    assert block_end["publish_only_after_family_block_end"] is True
    assert block_end["published_block_count"] == 8
    assert shuffled["uses_same_promoted_record_budget"] is True
    assert shuffled["family_alignment_preserved"] is False

    by_arm = metrics["by_arm"]
    assert (
        by_arm["immediate_verified_post_outcome_commit"]["forward_transfer_accuracy"]
        > by_arm["no_memory"]["forward_transfer_accuracy"]
    )
    assert (
        by_arm["slow_block_end_consolidation"]["forward_transfer_accuracy"]
        > by_arm["no_memory"]["forward_transfer_accuracy"]
    )
    assert (
        by_arm["shuffled_memory_control"]["negative_transfer_rate"]
        > by_arm["immediate_verified_post_outcome_commit"]["negative_transfer_rate"]
    )
    assert metrics["aggregate_promotion_allowed"] is True
    assert metrics["protected_gate_failed"] is False

    utility = artifact["update_utility_and_memory_cost"]
    assert utility["memory_record_budget"] == mod.MEMORY_RECORD_BUDGET
    assert utility["by_arm"]["immediate_verified_post_outcome_commit"]["utility_per_record"] > 0
    assert (
        utility["by_arm"]["shuffled_memory_control"]["utility_per_record"]
        < utility["by_arm"]["immediate_verified_post_outcome_commit"]["utility_per_record"]
    )


def test_scenario_6219_store_attacks_restart_and_rollback() -> None:
    """SCENARIO-LEARN-6219-ATTACKS/ROLLBACK: unsafe events fail closed."""

    bundle = mod.load_upstream_stream()
    events = mod.preregister_stream(bundle)["events"]
    baseline_store = mod.ProceduralConstraintStore(max_records=mod.MEMORY_RECORD_BUDGET)
    baseline_store_hash = baseline_store.state_hash()
    baseline_trace_hash = mod.sha256_json([])

    run = mod.run_arm(events[:5], arm_name="immediate_verified_post_outcome_commit")
    store = run["store"]
    attack_receipt = mod.inject_attack_events(store, events[:30])
    restart = mod.replay_store_idempotently(events[:24], max_records=mod.MEMORY_RECORD_BUDGET)
    rollback = store.rollback_to_baseline(
        baseline_store_hash=baseline_store_hash,
        baseline_decision_trace_hash=baseline_trace_hash,
    )

    assert attack_receipt["malformed"]["action"] in {"quarantine", "reject"}
    assert attack_receipt["poisoned"]["reason"] == "poisoned_update"
    assert attack_receipt["duplicate"]["idempotent"] is True
    assert attack_receipt["reordered"]["reason"] == "reordered_event"
    assert attack_receipt["stale"]["reason"] == "stale_ttl"
    assert attack_receipt["poison_propagation_count"] == 0
    assert restart["idempotent"] is True
    assert rollback["active_store_restored"] is True
    assert rollback["decision_trace_restored"] is True


def test_req_6219_validation_rejects_temporal_and_gate_drift(tmp_path: Path) -> None:
    """REQ-LEARN-6219-1/7: schema validation rejects bypasses."""

    artifact = _artifact(tmp_path)

    missing = dict(artifact)
    missing.pop("rollback_exactness")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_learning = deepcopy(artifact)
    bad_learning["continuous_self_learning_task"] = {"value": True}
    bad_learning["reproducibility_checksum"] = mod.reproducibility_checksum(bad_learning)
    with pytest.raises(ValueError, match="continuous_self_learning_task"):
        mod.validate_artifact(bad_learning)

    bad_overlap = deepcopy(artifact)
    bad_overlap["train_eval_overlap_count"] = 1
    _refresh(bad_overlap)
    with pytest.raises(ValueError, match="train_eval_overlap_count"):
        mod.validate_artifact(bad_overlap)

    bad_snapshot = deepcopy(artifact)
    bad_snapshot["decision_time_write_count"] = 1
    _refresh(bad_snapshot)
    assert bad_snapshot["continuous_learning_promotion_ready_score"] == 0.0
    with pytest.raises(ValueError, match="decision_time_write_count"):
        mod.validate_artifact(bad_snapshot)

    bad_weight = deepcopy(artifact)
    bad_weight["model_weight_mutation_count"] = 1
    _refresh(bad_weight)
    with pytest.raises(ValueError, match="model_weight_mutation_count"):
        mod.validate_artifact(bad_weight)

    bad_protected_gate = deepcopy(artifact)
    bad_protected_gate["accuracy_forward_transfer_retention_and_negative_transfer_by_family_arm"][
        "protected_gate_failed"
    ] = True
    _refresh(bad_protected_gate)
    assert bad_protected_gate["status"] == "complete_null"
    assert mod.validate_artifact(bad_protected_gate) is True

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)


def test_req_6219_helper_edges_and_validation_guards(tmp_path: Path) -> None:
    """REQ-LEARN-6219-3/5: helper guard branches stay explicit."""

    assert mod.sha256_file(tmp_path / "missing.bin") is None
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(list_json)

    bundle = mod.load_upstream_stream()
    events = mod.preregister_stream(bundle)["events"]
    assert events[0].to_json()["event_id"] == events[0].event_id

    store = mod.ProceduralConstraintStore(max_records=mod.MEMORY_RECORD_BUDGET)
    decision_time = store.apply_post_outcome(
        events[0],
        visible_from_event_index=events[0].chronological_index,
    )
    assert decision_time["reason"] == "decision_time_visibility"

    disagree = replace(
        events[1],
        event_id="exp6219-disagree",
        verifier_agrees=False,
    )
    disagreement = store.apply_post_outcome(
        disagree,
        visible_from_event_index=disagree.chronological_index + 1,
    )
    assert disagreement["reason"] == "verifier_disagreement"
    assert mod._publish_staged_block(store, []) == []

    timed = mod.run(
        result_path=tmp_path / "timed.json",
        test_exit_codes=_passing_exit_codes(),
    )
    assert timed["duration_s"] >= 0.001

    artifact = _artifact(tmp_path)
    blocked = deepcopy(artifact)
    blocked["upstream_stream_path_hash_and_clean_receipt"]["exact_oracle_separated"] = False
    _refresh(blocked)
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True

    bad_arm = deepcopy(artifact)
    bad_arm["arm_definitions_and_resource_parity"]["arm_names"] = []
    bad_arm["reproducibility_checksum"] = mod.reproducibility_checksum(bad_arm)
    with pytest.raises(ValueError, match="arm mismatch"):
        mod.validate_artifact(bad_arm)

    bad_score = deepcopy(artifact)
    bad_score["continuous_learning_promotion_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_score)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    bad_principles["reproducibility_checksum"] = mod.reproducibility_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)
