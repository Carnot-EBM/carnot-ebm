"""Tests for REQ-CL-7496 and SCENARIO-CL-7496-*.

The labels are analytic fixture values. These tests qualify update mechanics
and causal access. They do not measure learning on human data.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7496_v656_causal_update_fixture as exp


def _zero_head(*, bound: float = 1.0, guard_rows: list[dict[str, object]] | None = None):
    """Create a deterministic head so each expected gradient is transparent."""

    head = exp.BrierResidualHead.from_training(
        exp.training_fixture(),
        seed=exp.OPTIMIZER_SEED,
        learning_rate=0.03,
        residual_bound=bound,
        guard_rows=guard_rows or (),
        guard_tolerance=0.0,
    )
    head.coefficients.fill(0.0)
    head.bias = 0.0
    return head


# REQ-CL-7496; SCENARIO-CL-7496-GRADIENT.
def test_brier_gradient_matches_finite_differences_at_clip_cases() -> None:
    head = _zero_head()
    features = exp.training_fixture()[3]
    analytic, support = head.sparse_gradient(features, 1, frozen_probability=0.43)
    numeric = exp.finite_difference_gradient(head, features, 1, frozen_probability=0.43)
    assert np.max(np.abs(analytic - numeric)) <= 1e-8
    assert len(support) <= 4 * 4

    for bias in (1.2, -1.2, 1.0, -1.0):
        head.bias = bias
        analytic, _support = head.sparse_gradient(features, 0, frozen_probability=0.43)
        numeric = exp.finite_difference_gradient(head, features, 0, frozen_probability=0.43)
        assert np.max(np.abs(analytic - numeric)) <= 1e-8
        assert np.array_equal(analytic, np.zeros_like(analytic))

    with pytest.raises(ValueError, match="feedback_label_invalid"):
        head.sparse_gradient(features, 2, frozen_probability=0.43)


# REQ-CL-7496; SCENARIO-CL-7496-CURRENT and SCENARIO-CL-7496-LIFECYCLE.
def test_feedback_uses_current_state_and_invalid_delivery_is_idempotent(tmp_path: Path) -> None:
    head = _zero_head()
    features = exp.training_fixture()[2]
    first = head.seal_prediction(
        event_id="delivery-1",
        source_version="repeated-source",
        prediction_time=0,
        reveal_time=2,
        features=features,
        frozen_probability=0.5,
    )
    second = head.seal_prediction(
        event_id="delivery-2",
        source_version="repeated-source",
        prediction_time=1,
        reveal_time=2,
        features=features,
        frozen_probability=0.5,
    )
    assert first["pre_update_probability"] == first["residual_prediction"]
    assert second["features"] == features.tolist()
    before = head.state_hash
    assert head.apply_feedback("delivery-1", label=1, visible_at=1)["status"] == "not_revealed"
    assert head.state_hash == before
    assert head.apply_feedback("delivery-1", label=1, visible_at=2)["status"] == "committed"
    receipt = head.apply_feedback("delivery-2", label=0, visible_at=2)
    assert receipt["status"] == "committed"
    assert receipt["gradient_probability"] != receipt["pre_update_probability"]
    committed = head.state_hash
    assert head.apply_feedback("delivery-2", label=0, visible_at=3)["status"] == "duplicate"
    assert head.apply_feedback("delivery-2", label=1, visible_at=3)["status"] == "identity_conflict"
    assert head.apply_feedback("missing", label=0, visible_at=3)["status"] == "missing_prediction"
    assert head.state_hash == committed

    checkpoint = tmp_path / "head.json"
    head.save_checkpoint(checkpoint)
    restored = exp.BrierResidualHead.load_checkpoint(checkpoint)
    assert restored.state_hash == head.state_hash
    assert restored.apply_feedback("delivery-2", label=0, visible_at=4)["status"] == "duplicate"
    assert restored.state_hash == head.state_hash

    reordered = _zero_head()
    for index, reveal in ((0, 8), (1, 1)):
        reordered.seal_prediction(
            event_id=f"order-{index}",
            source_version="analytic",
            prediction_time=index,
            reveal_time=reveal,
            features=features,
            frozen_probability=0.5,
        )
    assert reordered.apply_feedback("order-1", label=0, visible_at=1)["status"] == "reordered"


# REQ-CL-7496; SCENARIO-CL-7496-LIFECYCLE.
def test_training_guard_rolls_back_without_retention_labels() -> None:
    features = exp.training_fixture()[4]
    guard = [{"features": features.tolist(), "label": 0, "frozen_probability": 0.5}]
    head = _zero_head(guard_rows=guard)
    head.seal_prediction(
        event_id="guard-conflict",
        source_version="analytic",
        prediction_time=0,
        reveal_time=0,
        features=features,
        frozen_probability=0.5,
    )
    before = head.state_hash
    receipt = head.apply_feedback("guard-conflict", label=1, visible_at=0)
    assert receipt["status"] == "rejected_guard"
    assert receipt["rolled_back"] is True
    assert head.state_hash == before
    assert receipt["retention_labels_used"] == 0


# REQ-CL-7496; SCENARIO-CL-7496-CONTROLS.
def test_release_batch_permutations_name_derangements_and_noops() -> None:
    deranged = exp.permute_released_labels([0, 1, 0, 1], ["a", "b", "c", "d"], seed=7)
    assert deranged["mode"] == "derangement"
    assert all(left != right for left, right in zip([0, 1, 0, 1], deranged["labels"], strict=True))
    singleton = exp.permute_released_labels([1], ["a"], seed=7)
    assert singleton == {"labels": [1], "mode": "singleton_noop", "changed": 0}
    identical = exp.permute_released_labels([0, 0, 0], ["a", "b", "c"], seed=7)
    assert identical["mode"] == "identical_labels_noop"
    mixed = exp.permute_released_labels([0, 0, 1], ["a", "b", "c"], seed=7)
    assert mixed["mode"] == "best_effort_permutation"
    assert mixed["changed"] >= 2
    with pytest.raises(ValueError, match="batch_identity_mismatch"):
        exp.permute_released_labels([0], [], seed=7)


# REQ-CL-7496; SCENARIO-CL-7496-RELEASE.
@pytest.mark.parametrize("delay", [0, 8])
def test_release_batches_are_common_and_future_label_invariant(delay: int) -> None:
    events = exp.fixture_events()
    plan = exp.build_release_plan(events, audit_seed=exp.AUDIT_SEED, delay=delay)
    assert plan
    assert all(row["block_size"] == 8 for row in plan)
    assert all(row["release_time"] == row["block_end"] + delay for row in plan)
    assert all(row["audit_probability"] == 0.25 for row in plan)

    original = exp.run_causal_replay(events, audit_seed=exp.AUDIT_SEED, delay=delay)
    mutated = deepcopy(events)
    later_block = max(int(row["block_id"]) for row in plan)
    for event in mutated:
        if int(event["arrival_index"]) // 8 == later_block:
            event["label"] = 1 - int(event["label"])
    changed = exp.run_causal_replay(mutated, audit_seed=exp.AUDIT_SEED, delay=delay)
    first_block = min(int(row["block_id"]) for row in plan)
    assert original["batch_rows"][first_block] == changed["batch_rows"][first_block]
    assert original["future_access_violations"] == 0
    assert all(row["real_release_time"] == row["shuffled_release_time"] for row in original["batch_rows"])
    assert original["no_feedback_gap_count"] >= 1
    assert original["out_of_order_probe"]["status"] == "batch_not_available"


# REQ-CL-7496; SCENARIO-CL-7496-GRADIENT and SCENARIO-CL-7496-GATES.
def test_analytic_controls_freeze_candidates_and_keep_log_loss_control(tmp_path: Path) -> None:
    evidence = exp.run_analytic_controls(tmp_path)
    assert evidence["gradient_checks"] and all(row["passed"] for row in evidence["gradient_checks"])
    assert evidence["state_replay_checks"]["passed"] is True
    assert evidence["causal_access_checks"]["passed"] is True
    assert {row["arm"] for row in evidence["rows"]} == {"brier", "log_loss"}
    assert evidence["selection"]["learning_rate_candidates"] == [0.001, 0.01, 0.03]
    assert evidence["selection"]["residual_bound_candidates"] == [0.5, 1.0]
    assert evidence["selection"]["selection_roles"] == ["training", "calibration_tuning"]
    assert evidence["selection"]["heldout_labels_consumed"] is False
    assert evidence["clipping_rate"] >= 0.0
    assert evidence["bounded_residual_noop_rate"] >= 0.0
    assert evidence["numeric_elapsed_s"] < 600.0


# REQ-CL-7496; SCENARIO-CL-7496-GATES and SCENARIO-CL-7496-ARTIFACT.
def test_artifact_reduction_and_mutations_fail_closed(tmp_path: Path) -> None:
    artifact = exp.build_fixture_artifact(tmp_path)
    assert exp.validate_artifact(artifact, verify_sources=False) == []
    assert exp.validate_artifact(artifact) == []
    assert artifact["causal_update_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert set(artifact) == set(artifact["field_principles"])
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])
    assert exp.independent_reduce(artifact)["causal_update_ready_score"] == 1

    for field, replacement in (
        ("schema", "wrong"),
        ("run_date", "19000101"),
        ("MODEL_SPECS", ["wrong"]),
        ("model_invoked", True),
        ("invocation_counts", {}),
        ("causal_update_ready_score", 0),
        ("verdict_class", "positive"),
        ("release_batch_protocol", {}),
        ("reproducibility_checksum", "sha256:wrong"),
    ):
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert exp.validate_artifact(changed, verify_sources=False)

    source_changed = deepcopy(artifact)
    source = next(iter(source_changed["source_artifact_hashes"]))
    source_changed["source_artifact_hashes"][source]["path"] = "/tmp/not-present-exp7496"
    assert any(error.startswith("source_hash_invalid:") for error in exp.validate_artifact(source_changed))
    malformed_source = deepcopy(artifact)
    malformed_source["source_artifact_hashes"][source] = "bad"
    assert any(error.startswith("source_hash_row_invalid:") for error in exp.validate_artifact(malformed_source))


# REQ-CL-7496; SCENARIO-CL-7496-ARTIFACT.
def test_preconditions_protocol_and_reader_edges(tmp_path: Path) -> None:
    checks, hashes = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks and all(row["passed"] is True for row in checks)
    assert hashes
    protocol = exp.release_batch_protocol()
    assert protocol["block_size"] == 8
    assert protocol["audit_probability"] == 0.25
    assert protocol["delays"] == [0, 8]
    assert protocol["earlier_v656_inputs"] == []
    assert protocol["importance_penalty"] is False
    assert protocol["four_expert_mixture_changed"] is False
    assert exp.cold_replay(tmp_path / "missing.json", verify_sources=False) == [
        "artifact_unreadable_or_not_object"
    ]
    malformed = tmp_path / "bad.json"
    malformed.write_text("[")
    assert exp._load_object(malformed) == {}
    malformed.write_text("[]")
    assert exp._load_object(malformed) == {}


# REQ-CL-7496; SCENARIO-CL-7496-ARTIFACT.
def test_cli_reader_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact = exp.build_fixture_artifact(tmp_path / "fixture")
    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(path), "--no-source-check"]) == 0
    assert '"errors": []' in capsys.readouterr().out
    assert exp.main(
        ["--date", exp.RUN_DATE, "--independent-reduce", str(path), "--no-source-check"]
    ) == 0
    assert '"causal_update_ready_score": 1' in capsys.readouterr().out

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"schema": "wrong"}))
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(broken), "--no-source-check"]) == 1
    with pytest.raises(SystemExit, match=f"--date must be {exp.RUN_DATE}"):
        exp.run_experiment(exp.REPO_ROOT, "19000101", output_path=tmp_path / "never.json")
