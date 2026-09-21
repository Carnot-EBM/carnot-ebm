"""Tests for REQ-KAN-7482 and SCENARIO-KAN-7482-*.

The fixtures use analytic labels. They verify the anchor implementation and
state lifecycle. They do not claim held-out learning value.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7482_v655_importance_anchor as exp


def _head(*, mode: str = "importance", anchor_lambda: float = 4.0) -> exp.ImportanceAnchorHead:
    head = exp.ImportanceAnchorHead.from_training(
        exp.training_fixture(),
        seed=748200,
        anchor_mode=mode,
        anchor_lambda=anchor_lambda,
        learning_rate=0.04,
    )
    low = exp.training_fixture().min(axis=0)
    head.seal_prediction(
        event_id="prior",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=0,
        features=low,
        frozen_probability=0.5,
    )
    receipt = head.apply_feedback("prior", label=1, visible_at=0)
    assert receipt["status"] == "committed"
    head.consolidate()
    return head


# REQ-KAN-7482; SCENARIO-KAN-7482-01.
def test_loss_gradient_matches_finite_difference() -> None:
    head = _head()
    active = np.flatnonzero(head.importance > 0.0)
    assert active.size > 0
    head.coefficients.reshape(-1)[active[0]] += 0.17
    features = exp.training_fixture()[3]
    _loss, analytic, work = head.loss_and_gradient(features, 0, frozen_probability=0.43)
    numeric = exp.finite_difference_gradient(head, features, 0, frozen_probability=0.43)
    assert np.max(np.abs(analytic - numeric)) <= 1e-8
    assert work["total_anchor_coefficient_count"] == exp.COEFFICIENT_COUNT
    assert work["active_data_gradient_count"] <= exp.COEFFICIENT_COUNT


# REQ-KAN-7482; SCENARIO-KAN-7482-02.
def test_dense_anchor_moves_inactive_changed_coefficient() -> None:
    head = _head(anchor_lambda=6.0)
    low = exp.training_fixture().min(axis=0)
    high = exp.training_fixture().max(axis=0)
    _low_design, low_support = exp.local_design(low, head.knots)
    _high_design, high_support = exp.local_design(high, head.knots)
    inactive = next(
        index for index in low_support if index not in high_support and head.importance[index] > 0.0
    )
    reference = head.reference_coefficients[inactive]
    head.coefficients.reshape(-1)[inactive] = reference + 0.25
    before = abs(head.coefficients.reshape(-1)[inactive] - reference)
    head.seal_prediction(
        event_id="new-high",
        source_version="analytic-v1",
        prediction_time=1,
        reveal_time=1,
        features=high,
        frozen_probability=0.5,
    )
    receipt = head.apply_feedback("new-high", label=0, visible_at=1)
    after = abs(head.coefficients.reshape(-1)[inactive] - reference)
    assert receipt["status"] == "committed"
    assert inactive not in receipt["active_data_support"]
    assert receipt["total_anchor_coefficient_count"] == exp.COEFFICIENT_COUNT
    assert after < before


# REQ-KAN-7482; SCENARIO-KAN-7482-03.
def test_importance_uses_only_admitted_revealed_labels() -> None:
    head = exp.ImportanceAnchorHead.from_training(
        exp.training_fixture(),
        seed=748201,
        anchor_mode="importance",
        anchor_lambda=4.0,
        learning_rate=0.04,
    )
    low, high = exp.training_fixture()[1], exp.training_fixture()[-2]
    first = head.seal_prediction(
        event_id="delayed",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=8,
        features=low,
        frozen_probability=0.4,
    )
    head.seal_prediction(
        event_id="later",
        source_version="analytic-v1",
        prediction_time=1,
        reveal_time=1,
        features=high,
        frozen_probability=0.6,
    )
    assert first["features"] == low.tolist()
    assert first["residual_prediction"] == first["pre_update_probability"]
    assert head.apply_feedback("delayed", label=1, visible_at=7)["status"] == "not_revealed"
    assert head.apply_feedback("later", label=0, visible_at=1)["status"] == "reordered"
    assert head.importance_observations == 0
    admitted = head.apply_feedback("delayed", label=1, visible_at=8)
    assert admitted["importance_observations_before"] == 0
    assert admitted["importance_observations_after"] == 1
    snapshot = head.importance_accumulator.copy()
    assert head.apply_feedback("delayed", label=1, visible_at=9)["status"] == "duplicate"
    assert np.array_equal(head.importance_accumulator, snapshot)
    assert head.apply_feedback("later", label=0, visible_at=9)["status"] == "committed"
    assert head.apply_feedback("missing", label=0, visible_at=9)["status"] == "missing_prediction"
    with pytest.raises(ValueError, match="feedback_label_invalid"):
        head.apply_feedback("later", label=3, visible_at=9)

    empty = exp.ImportanceAnchorHead.from_training(
        exp.training_fixture(),
        seed=748201,
        anchor_mode="importance",
        anchor_lambda=4.0,
        learning_rate=0.04,
    )
    with pytest.raises(ValueError, match="importance_requires_revealed_label"):
        empty.consolidate()


# REQ-KAN-7482; SCENARIO-KAN-7482-06.
def test_checkpoint_restart_replays_acknowledged_state(tmp_path: Path) -> None:
    head = _head()
    high = exp.training_fixture().max(axis=0)
    for index in (1, 2):
        head.seal_prediction(
            event_id=f"event-{index}",
            source_version="analytic-v1",
            prediction_time=index,
            reveal_time=index + 8,
            features=high,
            frozen_probability=0.5,
        )
    assert head.apply_feedback("event-1", label=0, visible_at=9)["status"] == "committed"
    checkpoint = tmp_path / "anchor-checkpoint.json"
    manifest = head.save_checkpoint(checkpoint)
    checkpoint.with_name(f".{checkpoint.name}.tmp-interrupted").write_text("partial")
    left = exp.ImportanceAnchorHead.load_checkpoint(checkpoint)
    right = exp.ImportanceAnchorHead.load_checkpoint(checkpoint)
    left_receipt = left.apply_feedback("event-2", label=0, visible_at=10)
    right_receipt = right.apply_feedback("event-2", label=0, visible_at=10)
    assert left_receipt == right_receipt
    assert left.state_hash == right.state_hash
    assert manifest["state_hash"] != left.state_hash
    damaged = json.loads(checkpoint.read_text())
    damaged["importance_observations"] += 1
    checkpoint.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="checkpoint_hash_invalid"):
        exp.ImportanceAnchorHead.load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="checkpoint_unreadable"):
        exp.ImportanceAnchorHead.load_checkpoint(tmp_path / "missing.json")
    checkpoint.write_text(json.dumps({"schema": "wrong"}))
    with pytest.raises(ValueError, match="checkpoint_schema_invalid"):
        exp.ImportanceAnchorHead.load_checkpoint(checkpoint)


# REQ-KAN-7482; SCENARIO-KAN-7482-04 and SCENARIO-KAN-7482-05.
def test_analytic_trial_has_fair_arms_and_explicit_controls(tmp_path: Path) -> None:
    rows, checks, replay = exp.run_analytic_controls(tmp_path)
    assert len(rows) == len(exp.FIT_SEEDS) * len(exp.ARM_CONFIGS)
    assert checks["gradient_parity"]["passed"] is True
    assert checks["chronology"]["passed"] is True
    assert checks["dense_anchor_work"]["passed"] is True
    assert checks["restart"]["passed"] is True
    assert replay["passed"] is True
    reduced = exp.reduce_trial_rows(rows)
    assert reduced["fair_main_arm_protocol"] is True
    assert reduced["zero_rate_unchanged"] is True
    assert reduced["overlap_and_disjoint_support_present"] is True
    assert reduced["completed_arm_seed_rows"] == len(rows)
    for row in rows:
        assert row["parameter_count"] <= 256
        assert row["replay_buffer_size"] == exp.REPLAY_BUFFER_SIZE
        assert row["learning_rate_candidates"] == list(exp.LEARNING_RATE_CANDIDATES)
        assert row["update_opportunities"] == exp.UPDATE_OPPORTUNITIES


# REQ-KAN-7482; SCENARIO-KAN-7482-07 and SCENARIO-KAN-7482-08.
def test_artifact_reduction_keeps_readiness_separate_from_benefit(tmp_path: Path) -> None:
    artifact = exp.build_fixture_artifact(tmp_path)
    assert exp.validate_artifact(artifact, verify_sources=False) == []
    assert exp.validate_artifact(artifact) == []
    assert artifact["importance_anchor_ready_score"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] in {"circular_positive", "null"}
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])
    no_benefit = deepcopy(artifact)
    for row in no_benefit["rows"]:
        if row["arm"] == "importance_anchor":
            row["retained_support_drift_l2"] = 99.0
    exp.finalize_artifact(no_benefit)
    assert no_benefit["importance_anchor_ready_score"] == 1
    assert no_benefit["verdict_class"] == "null"
    assert exp.validate_artifact(no_benefit, verify_sources=False) == []

    for field, replacement in (
        ("importance_anchor_ready_score", 0),
        ("model_invoked", True),
        ("reproducibility_checksum", "sha256:wrong"),
        ("schema", "wrong"),
        ("run_date", "19000101"),
        ("MODEL_SPECS", ["wrong"]),
        ("invocation_counts", {}),
        ("verifier_is_oracle", False),
        ("anchor_definition", {}),
    ):
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert exp.validate_artifact(changed, verify_sources=False)

    invalid_row = deepcopy(artifact)
    first_source = next(iter(invalid_row["source_artifact_hashes"]))
    invalid_row["source_artifact_hashes"][first_source] = "not-a-row"
    assert any(
        error.startswith("source_hash_row_invalid:") for error in exp.validate_artifact(invalid_row)
    )
    missing_source = deepcopy(artifact)
    missing_source["source_artifact_hashes"][first_source]["path"] = "/tmp/not-present-exp7482"
    assert any(
        error.startswith("source_hash_invalid:") for error in exp.validate_artifact(missing_source)
    )


# REQ-KAN-7482; SCENARIO-KAN-7482-08.
def test_protocol_preconditions_and_reader_edges(tmp_path: Path) -> None:
    protocol = exp.anchor_protocol()
    assert protocol["primary_method"]["arxiv_id"] == "2605.12306v1"
    assert protocol["departures"] == [
        "no_cnn",
        "no_backbone_ewc",
        "no_gradient_mask",
        "prequential_feedback_stream",
    ]
    assert protocol["generator_weights_frozen"] is True
    checks, hashes, sidecars = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks and all(row["passed"] is True for row in checks)
    assert hashes and sidecars
    assert exp.cold_replay(tmp_path / "missing.json", verify_sources=False) == [
        "artifact_unreadable_or_not_object"
    ]
    assert exp._load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[")
    assert exp._load_object(malformed) == {}
    not_object = tmp_path / "list.json"
    not_object.write_text("[]")
    assert exp._load_object(not_object) == {}

    with pytest.raises(ValueError, match="anchor_mode_invalid"):
        exp.ImportanceAnchorHead.from_training(
            exp.training_fixture(),
            seed=1,
            anchor_mode="bad",
            anchor_lambda=1.0,
            learning_rate=0.1,
        )
    with pytest.raises(ValueError, match="anchor_scalar_range_invalid"):
        exp.ImportanceAnchorHead.from_training(
            exp.training_fixture(),
            seed=1,
            anchor_mode="importance",
            anchor_lambda=-1.0,
            learning_rate=0.1,
        )


# REQ-KAN-7482; SCENARIO-KAN-7482-08.
def test_cli_reader_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact = exp.build_fixture_artifact(tmp_path / "fixture")
    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(path), "--no-source-check"]) == 0
    assert '"errors": []' in capsys.readouterr().out
    assert (
        exp.main(["--date", exp.RUN_DATE, "--independent-reduce", str(path), "--no-source-check"])
        == 0
    )
    output = capsys.readouterr().out
    assert '"importance_anchor_ready_score": 1' in output
