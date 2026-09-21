"""Tests for the V656 causal Brier-update fixture.

Spec refs: REQ-KAN-7496 and SCENARIO-KAN-7496-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from carnot.experiment_7468_v654_residual_learner import local_design
from carnot.experiment_7496_v656_causal_update_fixture import (
    AUDIT_PROBABILITY,
    COEFFICIENT_CAP,
    LEARNING_RATE_CANDIDATES,
    RESIDUAL_BOUND_CANDIDATES,
    BrierResidualHead,
    _load_object,
    audit_mask,
    build_fixture_artifact,
    build_release_schedule,
    causal_fixture_rows,
    collect_preconditions,
    finite_difference_gradient,
    finalize_artifact,
    independent_reduce,
    main,
    permutation_receipt,
    replay_stream,
    run_analytic_controls,
    select_configuration,
    training_fixture,
    validate_artifact,
)


def _head(*, bound: float = 1.0, learning_rate: float = 0.01, guard_rows=()):
    fixture = training_fixture()
    features = [row["features"] for row in fixture["training"]]
    return BrierResidualHead.from_training_config(
        features,
        learning_rate=learning_rate,
        residual_bound=bound,
        guard_rows=guard_rows,
    )


def test_brier_gradient_matches_finite_difference_and_clipping_boundary() -> None:
    """SCENARIO-KAN-7496-01: interior and clipped gradients match the loss."""

    features = [0.45, 0.55, 0.35, 0.65]
    head = _head()
    analytic, support = head.sparse_gradient(features, 1, frozen_probability=0.4)
    numeric = finite_difference_gradient(head, features, 1, frozen_probability=0.4)
    assert np.max(np.abs(analytic - numeric)) < 1e-8
    assert len(support) <= 16

    design, _ = local_design(features, head.knots)
    head.coefficients[:] = 0.0
    head.bias = head.residual_bound - 1e-4
    near, _ = head.sparse_gradient(features, 0, frozen_probability=0.5)
    near_numeric = finite_difference_gradient(head, features, 0, frozen_probability=0.5)
    assert np.max(np.abs(near - near_numeric)) < 1e-8

    head.bias = head.residual_bound + 0.1
    clipped, _ = head.sparse_gradient(features, 0, frozen_probability=0.5)
    clipped_numeric = finite_difference_gradient(head, features, 0, frozen_probability=0.5)
    assert np.array_equal(clipped, np.zeros_like(design, shape=(design.size + 1,)))
    assert np.array_equal(clipped_numeric, clipped)


def test_feedback_gradient_uses_current_state_not_sealed_probability() -> None:
    """SCENARIO-KAN-7496-02: feedback differentiates the current learner state."""

    head = _head(learning_rate=0.03)
    first = [0.2, 0.3, 0.4, 0.5]
    second = [0.7, 0.6, 0.5, 0.4]
    head.seal_prediction(
        event_id="first",
        source_version="fixture",
        prediction_time=0,
        reveal_time=0,
        features=first,
        frozen_probability=0.5,
    )
    sealed = head.seal_prediction(
        event_id="second",
        source_version="fixture",
        prediction_time=1,
        reveal_time=1,
        features=second,
        frozen_probability=0.5,
    )
    head.apply_feedback("first", label=1, visible_at=0)
    current_probability = head.predict(second, frozen_probability=0.5)["residual_prediction"]
    receipt = head.apply_feedback("second", label=0, visible_at=1)
    assert receipt["pre_update_probability"] == pytest.approx(current_probability)
    assert receipt["pre_update_probability"] != pytest.approx(sealed["residual_prediction"])
    assert receipt["gradient_evaluated_at"] == "current_learner_state"


def test_configuration_is_selected_only_from_training_and_calibration() -> None:
    """SCENARIO-KAN-7496-08: selection freezes the registered bounded grid."""

    fixture = training_fixture()
    selected = select_configuration(fixture["training"], fixture["calibration"])
    assert selected["learning_rate_candidates"] == list(LEARNING_RATE_CANDIDATES)
    assert selected["residual_bound_candidates"] == list(RESIDUAL_BOUND_CANDIDATES)
    assert len(selected["candidate_rows"]) == 6
    assert selected["selection_roles"] == ["training", "calibration"]
    assert selected["heldout_labels_consumed"] is False
    assert selected["actual_coefficient_count"] <= COEFFICIENT_CAP
    assert all(0.0 <= row["gradient_clipping_rate"] <= 1.0 for row in selected["candidate_rows"])
    assert all(0.0 <= row["no_op_rate"] <= 1.0 for row in selected["candidate_rows"])


def test_release_schedule_and_seeded_audit_are_label_independent() -> None:
    """SCENARIO-KAN-7496-03: blocks and masks do not inspect labels."""

    rows = causal_fixture_rows()
    changed = deepcopy(rows)
    for row in changed:
        row["label"] = 1 - row["label"]
    assert audit_mask(rows, seed=749621) == audit_mask(changed, seed=749621)
    schedule = build_release_schedule(rows, seed=749621, delay=8)
    assert AUDIT_PROBABILITY == 0.25
    assert all(item["block_size"] == 8 for item in schedule)
    assert all(item["release_time"] == item["block_end"] + 8 for item in schedule)
    assert all("label" not in item for item in schedule)


@pytest.mark.parametrize(
    ("labels", "reason"),
    [([1], "singleton"), ([0, 0, 0], "identical_labels"), ([], "empty_batch")],
)
def test_degenerate_release_batches_are_explicit_noops(labels: list[int], reason: str) -> None:
    """SCENARIO-KAN-7496-04: degenerate shuffled batches stay named controls."""

    receipt = permutation_receipt([f"event-{i}" for i in range(len(labels))], labels, seed=7)
    assert receipt["no_op"] is True
    assert receipt["no_op_reason"] == reason
    assert receipt["assigned_labels"] == labels


def test_non_degenerate_permutation_is_a_seeded_batch_local_derangement() -> None:
    """SCENARIO-KAN-7496-03/04: a useful shuffle uses only released labels."""

    ids = ["a", "b", "c", "d"]
    labels = [0, 0, 1, 1]
    receipt = permutation_receipt(ids, labels, seed=13)
    assert receipt["no_op"] is False
    assert sorted(receipt["assigned_labels"]) == sorted(labels)
    assert all(index != source for index, source in enumerate(receipt["source_indices"]))
    assert receipt == permutation_receipt(ids, labels, seed=13)
    assert receipt["effective_label_changes"] >= 2


def test_future_label_mutation_cannot_change_pre_release_state() -> None:
    """SCENARIO-KAN-7496-05: unreleased labels cannot affect earlier traces."""

    fixture = training_fixture()
    config = select_configuration(fixture["training"], fixture["calibration"])
    rows = causal_fixture_rows()
    schedule = build_release_schedule(rows, seed=749621, delay=8)
    assert len(schedule) >= 2
    stop = schedule[1]["release_time"]
    mutated = deepcopy(rows)
    future_ids = set(schedule[1]["selected_event_ids"])
    for index, row in enumerate(mutated):
        event_id = f"event-{index}:{row['source_id']}"
        if event_id in future_ids:
            row["label"] = 1 - row["label"]
    left = replay_stream(rows, config, audit_seed=749621, delay=8, stop_before_release=stop)
    right = replay_stream(mutated, config, audit_seed=749621, delay=8, stop_before_release=stop)
    assert left["trace_hash"] == right["trace_hash"]
    assert left["terminal_state_hashes"] == right["terminal_state_hashes"]
    assert left["acknowledgement_hashes"] == right["acknowledgement_hashes"]


def test_stream_covers_gaps_repeated_ids_and_shared_release_batches() -> None:
    """SCENARIO-KAN-7496-03/06: real and shuffled arms share causal batches."""

    fixture = training_fixture()
    config = select_configuration(fixture["training"], fixture["calibration"])
    rows = causal_fixture_rows()
    assert len({row["source_id"] for row in rows}) < len(rows)
    replay = replay_stream(rows, config, audit_seed=749621, delay=0)
    assert replay["no_feedback_event_count"] > 0
    assert len(replay["prediction_event_ids"]) == len(set(replay["prediction_event_ids"]))
    for batch in replay["release_batches"]:
        assert batch["real_event_ids"] == batch["shuffled_event_ids"]
        assert batch["real_release_time"] == batch["shuffled_release_time"]
        assert batch["future_label_origin_count"] == 0


def test_lifecycle_rollback_restart_and_idempotent_replay(tmp_path: Path) -> None:
    """SCENARIO-KAN-7496-06/07: invalid deliveries and restart preserve state."""

    _rows, checks, replay = run_analytic_controls(tmp_path)
    assert checks["lifecycle"]["statuses"] == [
        "not_revealed",
        "reordered",
        "committed",
        "duplicate",
        "identity_conflict",
        "committed",
    ]
    assert checks["rollback"]["passed"] is True
    assert checks["restart"]["passed"] is True
    assert replay["idempotent_replay_passed"] is True
    assert replay["checkpoint_manifest"]["byte_size"] > 0


def test_artifact_reduction_readiness_and_mutations_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-KAN-7496-09/10: raw operands alone determine readiness."""

    artifact = build_fixture_artifact(Path.cwd(), checkpoint_dir=tmp_path)
    assert validate_artifact(artifact, root=Path.cwd()) == []
    reduction = independent_reduce(artifact)
    assert reduction["causal_update_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["small_ebm_training"]["performed"] is True
    assert all(gate["principle"] for gate in artifact["acceptance_gate_results"])
    assert set(artifact) <= set(artifact["field_principles"])

    for field in ("rows", "release_batch_protocol", "state_replay_checks"):
        changed = deepcopy(artifact)
        if isinstance(changed[field], list):
            changed[field] = changed[field][:-1]
        else:
            changed[field]["mutation"] = True
        assert validate_artifact(changed, root=Path.cwd())

    changed = deepcopy(artifact)
    changed["causal_update_ready_score"] = 0
    assert "causal_update_ready_score_mismatch" in validate_artifact(changed, root=Path.cwd())


def test_preconditions_authenticate_original_flags_and_task_requirement() -> None:
    """REQ-KAN-7496: preconditions retain V654 and V655 dispositions."""

    checks, hashes, sidecars = collect_preconditions(Path.cwd())
    assert checks and all(row["passed"] is True for row in checks)
    assert all(row["principle"] for row in checks)
    observed = {row["check"]: row["observed"] for row in checks}
    assert observed["driving_requirement"] == "REQ-KAN-7496"
    assert observed["historical:experiment_7483_v655_continuous_learning:online_benefit_score"] == 0
    assert hashes and sidecars


def test_cli_readers_and_thin_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-KAN-7496-10: fresh reader modes and wrapper delegate exactly."""

    artifact = build_fixture_artifact(Path.cwd(), checkpoint_dir=tmp_path / "state")
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert main(["--date", "20260921", "--root", ".", "--cold-replay", str(path)]) == 0
    assert main(["--date", "20260921", "--root", ".", "--independent-reduce", str(path)]) == 0
    assert _load_object(path) == artifact
    assert _load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    assert _load_object(malformed) == {}

    called: dict[str, object] = {}

    def fake_main(argv=None):
        called["argv"] = argv
        return 0

    monkeypatch.setattr(
        "carnot.experiment_7496_v656_causal_update_fixture.main", fake_main
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(
            "scripts/experiments/experiment_7496_v656_causal_update_fixture.py",
            run_name="__main__",
        )
    assert stopped.value.code == 0
    assert called["argv"] is None


def test_finalize_rejects_false_bare_score() -> None:
    """SCENARIO-KAN-7496-09: a stored score cannot override failed evidence."""

    artifact = build_fixture_artifact(Path.cwd())
    artifact["analytic_checks"]["causal_access"]["passed"] = False
    artifact["causal_update_ready_score"] = 1
    finalize_artifact(artifact)
    assert artifact["causal_update_ready_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
