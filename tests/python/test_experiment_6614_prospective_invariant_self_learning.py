"""Focused tests for the prospective invariant self-learning contract."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_6614_prospective_invariant_self_learning as exp
from carnot.agentic.arc_invariant_memory import InvariantMemoryStore, sha256_bytes


def _event(
    event_id: str,
    source: str,
    index: int,
    split: str,
    current: list[list[int]],
    prediction: list[list[int]],
    observed: list[list[int]],
    calls: list[str],
) -> exp.ProspectiveEvent:
    current_grid = np.asarray(current, dtype=np.int16)
    predicted_grid = np.asarray(prediction, dtype=np.int16)
    observed_grid = np.asarray(observed, dtype=np.int16)

    def predict() -> np.ndarray:
        calls.append(f"predict:{event_id}")
        return predicted_grid.copy()

    def observe() -> np.ndarray:
        calls.append(f"observe:{event_id}")
        return observed_grid.copy()

    return exp.ProspectiveEvent(
        event_id=event_id,
        source_name=source,
        transition_index=index,
        split=split,
        chronology_index=index,
        current_grid=current_grid,
        action=index % 4,
        action_data=None,
        archive_path=f"data/{source}.npz",
        archive_sha256=exp.sha256_json({"archive": source}),
        source_transition_sha256=exp.sha256_json({"event": event_id}),
        world_model_path=f"results/arc_e3/{source}/world_model.py",
        world_model_sha256=exp.sha256_json({"model": source}),
        predict=predict,
        observe=observe,
    )


def _events() -> tuple[list[exp.ProspectiveEvent], list[str]]:
    calls: list[str] = []
    rows = [
        _event("a0", "adapt-a", 0, "adaptation", [[0, 2], [0, 2]], [[2, 2], [2, 2]], [[1, 1], [1, 1]], calls),
        _event("b0", "adapt-b", 1, "adaptation", [[0, 3], [0, 3]], [[3, 3], [3, 3]], [[1, 2], [1, 2]], calls),
        _event("a1", "adapt-a", 2, "adaptation_future", [[0, 2], [0, 2]], [[2, 2], [2, 2]], [[1, 1], [1, 1]], calls),
        _event("b1", "adapt-b", 3, "adaptation_future", [[0, 3], [0, 3]], [[3, 3], [3, 3]], [[1, 2], [1, 2]], calls),
        _event("r0", "retain-a", 4, "retention", [[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]], calls),
        _event("r1", "retain-b", 5, "retention", [[2, 3], [2, 3]], [[2, 3], [2, 3]], [[2, 3], [2, 3]], calls),
    ]
    return rows, calls


def _static_matrix() -> np.ndarray:
    return np.asarray([[0.55, -0.47], [-0.47, 0.50]], dtype=np.float64)


# REQ-LEARN-6614 and SCENARIO-LEARN-6614-ARTIFACT.
def test_required_contract_and_principles_are_complete() -> None:
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.INFERENCE_SUBSTRATE == (
        "prospective_chronological_live_e3_invariant_side_memory_no_new_llm"
    )
    assert exp.ARMS == (
        "no_learning",
        "static_projector",
        "governed_online_memory",
        "shuffled_admission_control",
    )
    assert exp.CONTINUOUS_SELF_LEARNING_TASK is True
    assert "REQ-LEARN-6614" in (exp.__doc__ or "")


# REQ-LEARN-6614-CHRONOLOGY and REQ-LEARN-6614-DOSE.
def test_freeze_chronology_is_source_disjoint_and_deranged() -> None:
    events, _ = _events()
    first = exp.freeze_chronology(events, seeds=(6614, 16614))
    second = exp.freeze_chronology(events, seeds=(6614, 16614))

    assert first == second
    assert first["source_disjoint"] is True
    assert first["chronology_sha256"].startswith("sha256:")
    assert first["opportunity_count"] == len(events)
    assert first["arm_list"] == list(exp.ARMS)
    assert first["seeds"] == [6614, 16614]
    mapping = first["shuffled_admission_mapping"]
    assert set(mapping) == {row.event_id for row in events if row.split != "retention"}
    assert all(source != target for source, target in mapping.items())


# SCENARIO-LEARN-6614-PREDICT-BEFORE-OBSERVE and MATCHED-ARMS.
def test_run_predicts_all_arms_before_observation_and_matches_rows(tmp_path: Path) -> None:
    events, calls = _events()
    result = exp.run_chronological_comparison(
        events,
        static_matrix=_static_matrix(),
        work_root=tmp_path / "state",
        checkpoint_root=tmp_path / "checkpoints",
        seeds=(6614, 16614),
        capacity=8,
    )

    rows = result["per_unit_rows"]
    assert len(rows) == len(events) * len(exp.ARMS) * 2
    assert len(result["prediction_before_observation_rows"]) == len(rows)
    assert all(row["observation_opened_after_all_predictions"] for row in rows)
    assert all(row["pre_state_hash"] == row["state_hash_at_prediction"] for row in rows)
    assert all(row["row_hash"] == exp.row_hash(row) for row in rows)
    for event in events:
        assert calls.index(f"predict:{event.event_id}") < calls.index(f"observe:{event.event_id}")
    dose = result["arm_and_dose_receipts"]
    assert dose["all_arms_received_every_opportunity"] is True
    assert dose["governed_and_shuffled_candidate_count_matched"] is True
    assert dose["capacity_matched"] is True


# REQ-LEARN-6614-ADMISSION and SCENARIO-LEARN-6614-LIFECYCLE.
@pytest.mark.parametrize(
    ("candidate_error", "candidate_valid", "expected_decision", "expected_active"),
    [
        (0, True, "commit", 1),
        (2, True, "no_op_exact_nonimprovement", 0),
        (0, False, "quarantine_invalid_candidate", 0),
    ],
)
def test_post_event_admission_uses_exact_evidence_only(
    tmp_path: Path,
    candidate_error: int,
    candidate_valid: bool,
    expected_decision: str,
    expected_active: int,
) -> None:
    store = InvariantMemoryStore(tmp_path / "store", total_capacity=4, per_source_capacity=2)
    before = store.canonical_state_bytes()
    result = exp.apply_post_event_update(
        store,
        source_id="feature:1:2:3",
        source_transition_hash=exp.sha256_json({"event": 1}),
        world_model_hash=exp.sha256_json({"model": 1}),
        basis=(1.0, 0.0, 0.0, -1.0),
        threshold=0.0,
        sequence_index=1,
        baseline_error=1,
        candidate_error=candidate_error,
        candidate_valid=candidate_valid,
    )

    assert result["decision"] == expected_decision
    assert result["exact_evidence"]["observed_after_prediction"] is True
    assert len(store.active_records()) == expected_active
    if expected_active:
        assert store.canonical_state_bytes() != before
    assert all(row["row_hash"] == exp.row_hash(row) for row in result["transition_rows"])


# REQ-LEARN-6614-UTILITY and SCENARIO-LEARN-6614-UTILITY.
def test_reducer_uses_later_rows_and_preserves_retention_support() -> None:
    rows = exp.synthetic_gate_rows(
        static_errors=(4, 4),
        governed_errors=(2, 1),
        shuffled_errors=(3, 3),
        retention_error=1,
    )
    summary = exp.recompute_aggregates_from_rows(rows, [], retention_margin=0.0, support_margin=0.0)

    future = summary["held_future_benefit_summary"]
    assert future["governed_benefit_over_static"] > 0.0
    assert future["governed_benefit_over_shuffled"] > 0.0
    assert future["positive_over_both_controls"] is True
    retention = summary["retention_and_support_summary"]
    assert retention["retention_noninferior"] is True
    assert retention["recoverable_support_noninferior"] is True
    assert future["paired_later_event_count"] == 2


# REQ-LEARN-6614-RECOVERY and SCENARIO-LEARN-6614-RECOVERY.
def test_restart_and_rollback_restore_bytes_and_predictions(tmp_path: Path) -> None:
    store = InvariantMemoryStore(tmp_path / "store", total_capacity=4, per_source_capacity=2)
    exp.apply_post_event_update(
        store,
        source_id="feature:1:2:3",
        source_transition_hash=exp.sha256_json({"event": 1}),
        world_model_hash=exp.sha256_json({"model": 1}),
        basis=(1.0, 0.0, 0.0, -1.0),
        threshold=0.0,
        sequence_index=1,
        baseline_error=2,
        candidate_error=1,
        candidate_valid=True,
    )
    probe = np.asarray([[0, 2], [0, 2]], dtype=np.int16)
    prediction = np.asarray([[2, 2], [2, 2]], dtype=np.int16)
    receipt = exp.verify_restart_and_rollback(
        store,
        probe_current=probe,
        probe_prediction=prediction,
        static_matrix=_static_matrix(),
        work_root=tmp_path / "recovery",
    )

    assert receipt["restart_state_byte_equal"] is True
    assert receipt["restart_prediction_equal"] is True
    assert receipt["rollback_state_byte_equal"] is True
    assert receipt["rollback_prediction_equal"] is True
    assert receipt["restart_state_sha256"] == sha256_bytes(store.canonical_state_bytes())


# REQ-LEARN-6614-ATTACKS and SCENARIO-LEARN-6614-ATTACKS.
def test_attack_matrix_contains_every_required_fail_closed_attack(tmp_path: Path) -> None:
    events, _ = _events()
    result = exp.run_chronological_comparison(
        events,
        static_matrix=_static_matrix(),
        work_root=tmp_path / "state",
        checkpoint_root=tmp_path / "checkpoints",
        seeds=(6614,),
        capacity=8,
    )
    attacks = exp.build_attack_rows(result, protected_unchanged=True, frozen_hashes_unchanged=True)

    assert {row["attack_id"] for row in attacks} == set(exp.ATTACK_IDS)
    assert all(row["detected"] and row["failed_closed"] and row["passed"] for row in attacks)
    assert all(row["unsafe_commit_delta"] == 0 for row in attacks)


# REQ-LEARN-6614-ARTIFACT and exact verdict boundaries.
@pytest.mark.parametrize(
    ("benefit", "blocked", "expected_class", "expected_ready"),
    [
        (True, False, "circular_positive", 1.0),
        (False, False, "null", 0.0),
        (False, True, "blocked", 0.0),
    ],
)
def test_verdict_never_upgrades_row_completion(
    benefit: bool,
    blocked: bool,
    expected_class: str,
    expected_ready: float,
) -> None:
    gates = exp.synthetic_acceptance_gate_rows(benefit=benefit, blocked=blocked)
    status, verdict, verdict_class, ready = exp.status_and_verdict(gates)

    assert verdict_class == expected_class
    assert ready == expected_ready
    assert verdict.startswith("complete_") or verdict.startswith("blocked_")
    assert status.startswith("complete_") or status.startswith("blocked_")


# REQ-LEARN-6614-ATOMIC and SCENARIO-LEARN-6614-ARTIFACT.
def test_artifact_validation_and_atomic_write_detect_tamper(tmp_path: Path) -> None:
    events, _ = _events()
    artifact = exp.build_artifact_from_events(
        events,
        static_matrix=_static_matrix(),
        work_root=tmp_path / "work",
        planning_date="20260825",
        seeds=(6614,),
        capacity=8,
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
    )

    assert exp.validate_artifact(artifact) == []
    target = tmp_path / "result.json"
    receipt = exp.atomic_write_artifact(target, artifact)
    assert receipt["atomic_replace"] is True
    assert receipt["written_sha256"] == exp.sha256_file(target)
    assert target.is_file()

    artifact["per_unit_rows"][0]["exact_error"] += 1
    errors = exp.validate_artifact(artifact)
    assert any("row_hash" in error for error in errors)
    assert any("reproducibility_checksum" in error for error in errors)


# REQ-LEARN-6614-PRECONDITIONS and blocked artifact integrity.
def test_failed_upstream_gate_produces_named_block_with_exact_value(tmp_path: Path) -> None:
    artifact = exp.build_blocked_artifact(
        planning_date="20260825",
        gate_name="exp6613_invariant_memory_ready_score",
        expected=1.0,
        observed=0.0,
        tests_run=[],
    )

    assert artifact["status"] == "blocked_upstream"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["continuous_self_learning_ready_score"] == 0.0
    assert artifact["gate_check_summary"]["failed_gate"] == (
        "exp6613_invariant_memory_ready_score"
    )
    assert artifact["gate_check_summary"]["observed"] == 0.0
    assert exp.validate_artifact(artifact) == []
    exp.atomic_write_artifact(tmp_path / "blocked.json", artifact)
    assert (tmp_path / "blocked.json").is_file()


# REQ-LEARN-6614-FROZEN and field provenance.
def test_complete_artifact_declares_frozen_side_state_only_boundary(tmp_path: Path) -> None:
    events, _ = _events()
    artifact = exp.build_artifact_from_events(
        events,
        static_matrix=_static_matrix(),
        work_root=tmp_path / "work",
        planning_date="20260825",
        seeds=(6614,),
        capacity=8,
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
    )

    assert artifact["continuous_self_learning_task"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["frozen_model_policy_receipts"]["all_unchanged"] is True
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["safety_occupancy_and_cost_summary"]["unsafe_commit_count"] == 0
    assert set(artifact["field_provenance"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert "solve_provenance" not in artifact


# REQ-LEARN-6614-CHRONOLOGY and fail-closed input boundaries.
def test_input_and_projection_edges_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events, _ = _events()
    sample = events[0]
    with pytest.raises(ValueError, match="split"):
        replace(sample, split="future")
    with pytest.raises(ValueError, match="indices"):
        replace(sample, chronology_index=-1)
    with pytest.raises(ValueError, match="two-dimensional"):
        replace(sample, current_grid=np.asarray([1, 2]))

    assert exp._grid_receipt(None)["available"] is False
    observed = np.ones((2, 2), dtype=np.int16)
    assert exp._exact_error(None, observed) == observed.size
    assert exp._exact_error(np.ones((1, 1), dtype=np.int16), observed) == observed.size
    projected, diagnostics = exp._project(observed, None, _static_matrix())
    assert projected is None
    assert diagnostics["failure"] == "base_prediction_unavailable"

    def invalid_project(*_args: object, **_kwargs: object) -> None:
        raise ValueError("invalid projection")

    monkeypatch.setattr(exp, "project_prediction", invalid_project)
    projected, diagnostics = exp._project(observed, observed, _static_matrix())
    assert np.array_equal(projected, observed)
    assert diagnostics["failure"].startswith("ValueError")

    with pytest.raises(ValueError, match="static matrix"):
        exp.run_chronological_comparison(
            events,
            static_matrix=np.ones((3, 3)),
            work_root=tmp_path / "bad-state",
            checkpoint_root=tmp_path / "bad-checkpoints",
            seeds=(6614,),
        )


# REQ-LEARN-6614-ADMISSION and exact retrieval filtering.
def test_active_memory_retrieval_and_defensive_receipt_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events, _ = _events()
    event = events[0]
    store = InvariantMemoryStore(tmp_path / "active", total_capacity=4, per_source_capacity=2)
    source_id = exp._source_key(event.current_grid, event.action)
    exp.apply_post_event_update(
        store,
        source_id=source_id,
        source_transition_hash=event.source_transition_sha256,
        world_model_hash=event.world_model_sha256,
        basis=(1.0, 0.0, 0.0, -1.0),
        threshold=0.0,
        sequence_index=0,
        baseline_error=2,
        candidate_error=1,
        candidate_valid=True,
    )
    prediction, retrieved, diagnostics = exp._memory_prediction(
        store, replace(event, chronology_index=1), event.predict(), _static_matrix()
    )
    assert prediction is not None
    assert len(retrieved) == 1
    assert diagnostics["fallback"] is None
    assert exp._wrong_source_key(event, None) != source_id
    assert exp._wrong_source_key(event, event) != source_id

    defensive = InvariantMemoryStore(
        tmp_path / "defensive", total_capacity=4, per_source_capacity=2
    )
    monkeypatch.setattr(
        defensive,
        "admit",
        lambda *_args, **_kwargs: SimpleNamespace(
            post_state=exp.LifecycleState.PROVISIONAL,
            action="quarantine",
            reason="defensive",
        ),
    )
    receipt = exp.apply_post_event_update(
        defensive,
        source_id=source_id,
        source_transition_hash=event.source_transition_sha256,
        world_model_hash=event.world_model_sha256,
        basis=(1.0, 0.0, 0.0, -1.0),
        threshold=0.0,
        sequence_index=0,
        baseline_error=2,
        candidate_error=1,
        candidate_valid=True,
    )
    assert receipt["decision"] == "quarantine_defensive"


# REQ-LEARN-6614-ROWS and recovery edge receipts.
def test_failed_predictions_empty_rows_and_recovery_edges(tmp_path: Path) -> None:
    events, _ = _events()

    def failed_prediction() -> np.ndarray:
        raise RuntimeError("predict failed")

    failed = replace(events[0], predict=failed_prediction)
    invalid = replace(events[1], predict=lambda: np.ones((1, 1), dtype=np.int16))
    result = exp.run_chronological_comparison(
        [invalid, failed],
        static_matrix=_static_matrix(),
        work_root=tmp_path / "failed-state",
        checkpoint_root=tmp_path / "failed-checkpoints",
        seeds=(6614,),
        capacity=4,
    )
    assert all(row["prediction"]["available"] is False for row in result["per_unit_rows"])
    assert any("RuntimeError" in str(row["failure"]) for row in result["per_unit_rows"])

    empty = exp.run_chronological_comparison(
        [],
        static_matrix=_static_matrix(),
        work_root=tmp_path / "empty-state",
        checkpoint_root=tmp_path / "empty-checkpoints",
        seeds=(6614,),
    )
    assert empty["per_unit_rows"] == []
    assert exp._paired_summary([])["sample_size"] == 0
    assert exp._paired_summary([2.0])["lower"] == 2.0
    with pytest.raises(ValueError, match="vectors"):
        exp.synthetic_gate_rows(
            static_errors=(1,),
            governed_errors=(1, 2),
            shuffled_errors=(1,),
            retention_error=0,
        )

    store = InvariantMemoryStore(tmp_path / "empty-store", total_capacity=2, per_source_capacity=1)
    probe = np.zeros((2, 2), dtype=np.int16)
    first = exp.verify_restart_and_rollback(
        store,
        probe_current=probe,
        probe_prediction=probe,
        static_matrix=_static_matrix(),
        work_root=tmp_path / "existing-recovery",
    )
    second = exp.verify_restart_and_rollback(
        store,
        probe_current=probe,
        probe_prediction=probe,
        static_matrix=_static_matrix(),
        work_root=tmp_path / "existing-recovery",
    )
    assert first["rollback_state_byte_equal"] and second["rollback_state_byte_equal"]


# REQ-LEARN-6614-ATOMIC and schema failure diagnostics.
def test_artifact_validator_names_each_contract_failure(tmp_path: Path) -> None:
    events, _ = _events()
    baseline = exp.build_artifact_from_events(
        events,
        static_matrix=_static_matrix(),
        work_root=tmp_path / "work",
        planning_date="20260825",
        seeds=(6614,),
        capacity=8,
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
    )

    missing = dict(baseline)
    missing.pop("status")
    assert exp.validate_artifact(missing) == ["missing required field: status"]

    mutations = (
        ("task", lambda row: row.__setitem__("continuous_self_learning_task", False)),
        ("substrate", lambda row: row.__setitem__("inference_substrate", "wrong")),
        ("oracle", lambda row: row.__setitem__("verifier_is_oracle", False)),
        ("provenance", lambda row: row.__setitem__("field_provenance", {})),
        ("summary", lambda row: row["held_future_benefit_summary"].__setitem__("paired_later_event_count", -1)),
        ("status", lambda row: row.__setitem__("status", "wrong")),
        ("verdict", lambda row: row.__setitem__("honest_verdict", "wrong")),
        ("class", lambda row: row.__setitem__("verdict_class", "wrong")),
        ("ready", lambda row: row.__setitem__("continuous_self_learning_ready_score", 0.5)),
        ("protected", lambda row: row["protected_files_unchanged"].__setitem__("all_unchanged", False)),
        ("frozen", lambda row: row["frozen_model_policy_receipts"].__setitem__("all_unchanged", False)),
    )
    for _name, mutate in mutations:
        changed = copy.deepcopy(baseline)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert exp.validate_artifact(changed)

    blocked = exp.build_blocked_artifact(
        planning_date="20260825",
        gate_name="upstream",
        expected=1.0,
        observed=0.0,
        tests_run=[],
    )
    blocked["continuous_self_learning_ready_score"] = 1.0
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert "blocked artifact cannot open readiness" in exp.validate_artifact(blocked)

    invalid = copy.deepcopy(baseline)
    invalid["continuous_self_learning_task"] = False
    invalid["reproducibility_checksum"] = exp.artifact_checksum(invalid)
    with pytest.raises(ValueError, match="continuous_self_learning_task"):
        exp.atomic_write_artifact(tmp_path / "invalid.json", invalid)


# REQ-LEARN-6614-FROZEN covers the degenerate invariant-basis fallback.
def test_degenerate_candidate_basis_remains_finite() -> None:
    current = np.asarray([[0, 0], [0, 0]], dtype=np.int16)
    observed = np.asarray([[1, 1], [1, 1]], dtype=np.int16)
    before = exp.grid_features(current)
    after = exp.grid_features(observed)
    difference = np.asarray(
        [
            before[0] ** 2 - after[0] ** 2,
            2.0 * (before[0] * before[1] - after[0] * after[1]),
            before[1] ** 2 - after[1] ** 2,
        ]
    )
    static = np.asarray(
        [[difference[0], difference[1]], [difference[1], difference[2]]],
        dtype=np.float64,
    )
    matrix, threshold = exp._candidate_basis(current, observed, static)
    assert matrix.shape == (2, 2)
    assert np.isfinite(matrix).all()
    assert np.isfinite(threshold)
