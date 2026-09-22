"""Tests for REQ-KAN-7506 and REQ-CL-7506.

These tests use oracle-built fixtures. They qualify mechanics, not benefit on
human labels or a future delayed-feedback evaluation.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7506_v657_causal_prototype as exp


def _head(
    *,
    loss: str = "brier",
    learning_rate: float = 0.03,
    bound: float = 1.0,
) -> exp.BoundedResidualHead:
    """Create a zero-state head so a test can isolate one causal effect."""

    return exp.BoundedResidualHead.from_training(
        exp.training_fixture(),
        learning_rate=learning_rate,
        residual_bound=bound,
        loss=loss,
        basis_kind="local",
    )


# REQ-KAN-7506; SCENARIO-KAN-7506-01; SCENARIO-CL-7506-MATH.
@pytest.mark.parametrize("bound", [0.5, 1.0])
@pytest.mark.parametrize("label", [0, 1])
@pytest.mark.parametrize("raw_scale", [0.0, 0.9, -0.9, 8.0, -8.0])
def test_brier_gradient_matches_finite_difference_every_smooth_branch(
    bound: float, label: int, raw_scale: float
) -> None:
    head = _head(bound=bound)
    head.coefficients[-1] = raw_scale * bound
    features = exp.training_fixture()[3]
    analytic = head.gradient(features, label, base_probability=0.41)[0]
    numeric = exp.finite_difference_gradient(head, features, label, base_probability=0.41)
    assert np.max(np.abs(analytic - numeric)) <= 2e-7
    if raw_scale == 0.0:
        assert np.linalg.norm(analytic) > 0.0
    assert abs(head.predict(features, base_probability=0.41)["residual"]) <= bound


# REQ-KAN-7506; SCENARIO-KAN-7506-01; SCENARIO-CL-7506-MATH.
def test_matched_log_loss_has_same_capacity_and_bound() -> None:
    brier = _head(loss="brier", bound=0.5)
    log_loss = _head(loss="log_loss", bound=0.5)
    assert brier.parameter_count == log_loss.parameter_count
    assert brier.parameter_count <= 256
    assert log_loss.gradient(exp.training_fixture()[1], 1, base_probability=0.37)[0].any()
    with pytest.raises(ValueError, match="label_invalid"):
        brier.gradient(exp.training_fixture()[1], 2, base_probability=0.37)


# REQ-KAN-7506; SCENARIO-KAN-7506-03; SCENARIO-CL-7506-CAUSAL.
def test_release_local_permutation_and_old_future_origin_counterexample() -> None:
    mixed = exp.permute_released_batch(labels=[0, 1, 0, 1], event_ids=["a", "b", "c", "d"], seed=17)
    assert mixed["mode"] == "derangement"
    assert all(left != right for left, right in zip([0, 1, 0, 1], mixed["labels"], strict=True))
    assert set(mixed["label_origins"]) == {"a", "b", "c", "d"}
    assert exp.permute_released_batch([1], ["a"], seed=17)["mode"] == "singleton_noop"
    assert exp.permute_released_batch([0, 0], ["a", "b"], seed=17)["mode"] == (
        "identical_labels_noop"
    )
    with pytest.raises(ValueError, match="origin_not_in_released_batch"):
        exp.validate_release_origins(["block-1"], ["future-block"], release_time=7)


# REQ-KAN-7506; SCENARIO-KAN-7506-02; SCENARIO-CL-7506-LIFECYCLE.
def test_predict_before_update_and_unavailable_label_rejection() -> None:
    machine = exp.CausalEventMachine.create(
        exp.training_fixture(), learning_rate=0.03, residual_bound=1.0, delay=8
    )
    event = exp.fixture_events()[0]
    prediction = machine.predict(event)
    immutable = deepcopy(prediction)
    before = machine.state_hash
    assert machine.deliver_label(event["event_id"], int(event["label"]), visible_at=0) == (
        "not_selected"
    )
    assert machine.state_hash == before
    assert machine.predictions[event["event_id"]] == immutable
    with pytest.raises(ValueError, match="prediction_required_before_label"):
        machine.deliver_label("missing", 0, visible_at=0)


# REQ-KAN-7506; SCENARIO-KAN-7506-02/03; SCENARIO-CL-7506-CAUSAL.
@pytest.mark.parametrize("delay", [0, 8])
def test_delay_replay_withholds_labels_and_future_changes_cannot_flow_back(delay: int) -> None:
    events = exp.fixture_events()
    original = exp.run_causal_fixture(events, delay=delay, stop_after_release_block=0)
    changed_events = deepcopy(events)
    for row in changed_events[8:]:
        row["label"] = 1 - int(row["label"])
    changed = exp.run_causal_fixture(changed_events, delay=delay, stop_after_release_block=0)
    assert original["stable_trace"] == changed["stable_trace"]
    assert original["future_access_violations"] == 0
    assert original["predict_before_update"] is True
    assert all(row["release_time"] == row["block_end"] + delay for row in original["batches"])
    assert all(
        origin in row["event_ids"] for row in original["batches"] for origin in row["label_origins"]
    )


# REQ-KAN-7506; SCENARIO-KAN-7506-01/05; SCENARIO-CL-7506-GATES.
def test_selection_uses_only_training_and_calibration_and_includes_controls() -> None:
    selection = exp.select_fixture_hyperparameters()
    assert selection["learning_rate_candidates"] == [0.001, 0.01, 0.03]
    assert selection["residual_bound_candidates"] == [0.5, 1.0]
    assert selection["selection_roles"] == ["training", "calibration_tuning"]
    assert selection["online_labels_used"] is False
    assert selection["retention_labels_used"] is False
    assert selection["budget_frozen_before_online"] is True
    controls = exp.run_head_controls(selection)
    assert {row["arm"] for row in controls} == {
        "local_brier",
        "local_log_loss",
        "affine_brier",
        "intercept_brier",
        "frozen",
        "zero_step",
    }
    assert next(row for row in controls if row["arm"] == "frozen")["state_changed"] is False
    assert next(row for row in controls if row["arm"] == "zero_step")["state_changed"] is False


# REQ-KAN-7506; SCENARIO-KAN-7506-02; SCENARIO-CL-7506-LIFECYCLE.
def test_update_replays_gradient_against_current_state_without_rewriting_prediction() -> None:
    head = _head()
    features = exp.training_fixture()[2]
    sealed = head.predict(features, base_probability=0.5)
    head.update(features, 1, base_probability=0.5)
    current_before = head.predict(features, base_probability=0.5)["probability"]
    receipt = head.update(features, 0, base_probability=0.5)
    assert receipt["gradient_probability"] == current_before
    assert receipt["gradient_probability"] != sealed["probability"]
    assert sealed == {**sealed}


# REQ-KAN-7506; SCENARIO-KAN-7506-04; SCENARIO-CL-7506-LIFECYCLE.
def test_checkpoint_restart_is_byte_equivalent(tmp_path: Path) -> None:
    events = exp.fixture_events()
    uninterrupted = exp.run_causal_fixture(events, delay=8)
    restarted = exp.run_causal_fixture(events, delay=8, checkpoint_path=tmp_path / "machine.json")
    assert restarted["restart_performed"] is True
    assert restarted["checkpoint_schema"] == exp.CHECKPOINT_SCHEMA
    assert restarted["stable_trace"] == uninterrupted["stable_trace"]
    assert restarted["terminal_checkpoint_bytes"] == uninterrupted["terminal_checkpoint_bytes"]
    payload = json.loads((tmp_path / "machine.json").read_text())
    assert {"models", "pending_queue", "audit_rng_state", "order_cursor"} <= set(payload)


# REQ-KAN-7506; SCENARIO-KAN-7506-05; SCENARIO-CL-7506-GATES.
def test_artifact_reduction_and_mutations_fail_closed(tmp_path: Path) -> None:
    artifact = exp.build_fixture_artifact(tmp_path)
    assert exp.validate_artifact(artifact, verify_sources=False) == []
    assert artifact["causal_update_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert set(artifact) == set(artifact["field_principles"])
    assert exp.independent_reduce(artifact)["causal_update_ready_score"] == 1

    for field, replacement in (
        ("schema", "wrong"),
        ("run_date", "19000101"),
        ("model_invoked", True),
        ("invocation_counts", {}),
        ("causal_update_ready_score", 0),
        ("verdict_class", "positive"),
        ("reproducibility_checksum", "sha256:wrong"),
    ):
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert exp.validate_artifact(changed, verify_sources=False)


# REQ-KAN-7506; SCENARIO-KAN-7506-05; SCENARIO-CL-7506-GATES.
def test_preconditions_and_cli_reader_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checks, hashes = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert hashes
    artifact = exp.build_fixture_artifact(tmp_path / "fixture")
    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)
    assert (
        exp.main(["--date", exp.RUN_DATE, "--cold-replay", str(candidate), "--no-source-check"])
        == 0
    )
    assert '"errors": []' in capsys.readouterr().out
    assert (
        exp.main(
            ["--date", exp.RUN_DATE, "--independent-reduce", str(candidate), "--no-source-check"]
        )
        == 0
    )
    assert '"causal_update_ready_score": 1' in capsys.readouterr().out
    with pytest.raises(SystemExit, match=f"--date must be {exp.RUN_DATE}"):
        exp.run_experiment(exp.REPO_ROOT, "19000101", output_path=tmp_path / "never.json")


# REQ-KAN-7506; SCENARIO-KAN-7506-01/02/04.
def test_numeric_and_machine_inputs_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = exp.training_fixture()
    with pytest.raises(ValueError, match="base_probability_invalid"):
        exp._base_logit(0.0)
    constructor = {
        "knots": None,
        "coefficients": np.zeros(1),
        "learning_rate": 0.01,
        "residual_bound": 1.0,
        "loss": "brier",
        "basis_kind": "intercept",
    }
    for field, replacement, message in (
        ("coefficients", np.asarray([[0.0]]), "coefficients_invalid"),
        ("loss", "wrong", "loss_invalid"),
        ("basis_kind", "wrong", "basis_kind_invalid"),
        ("learning_rate", -1.0, "learning_rate_invalid"),
        ("residual_bound", 0.0, "residual_bound_invalid"),
    ):
        values = {**constructor, field: replacement}
        with pytest.raises(ValueError, match=message):
            exp.BoundedResidualHead(**values)
    with pytest.raises(ValueError, match="training_features_invalid"):
        exp.BoundedResidualHead.from_training(
            [[1.0]], learning_rate=0.01, residual_bound=1.0, loss="brier", basis_kind="local"
        )
    with pytest.raises(ValueError, match="basis_kind_invalid"):
        exp.BoundedResidualHead.from_training(
            fixture, learning_rate=0.01, residual_bound=1.0, loss="brier", basis_kind="wrong"
        )
    head = _head()
    with pytest.raises(ValueError, match="features_invalid"):
        head.predict([1.0], base_probability=0.5)
    no_knots = exp.BoundedResidualHead(
        knots=None,
        coefficients=np.zeros(head.parameter_count),
        learning_rate=0.01,
        residual_bound=1.0,
        loss="brier",
        basis_kind="local",
    )
    with pytest.raises(ValueError, match="local_knots_missing"):
        no_knots.predict(fixture[0], base_probability=0.5)
    log_head = _head(loss="log_loss")
    numeric = exp.finite_difference_gradient(log_head, fixture[0], 1, base_probability=0.4)
    assert (
        np.max(np.abs(numeric - log_head.gradient(fixture[0], 1, base_probability=0.4)[0])) < 2e-7
    )
    head.learning_rate = 1e308
    monkeypatch.setattr(
        head,
        "gradient",
        lambda *_args, **_kwargs: (np.full(head.parameter_count, 1e308), 1.0),
    )
    with pytest.raises(ValueError, match="candidate_state_nonfinite"):
        head.update(fixture[0], 1, base_probability=0.5)

    exp.validate_release_origins(["a"], ["a"], release_time=0)
    with pytest.raises(ValueError, match="release_time_invalid"):
        exp.validate_release_origins(["a"], ["a"], release_time=-1)
    for labels, identities in (
        ([], []),
        ([0], ["a", "b"]),
        ([2], ["a"]),
        ([0, 1], ["a", "a"]),
    ):
        with pytest.raises(ValueError, match="released_batch_invalid"):
            exp.permute_released_batch(labels, identities, seed=1)
    with pytest.raises(ValueError, match="delay_invalid"):
        exp.CausalEventMachine.create(fixture, learning_rate=0.01, residual_bound=1.0, delay=1)

    machine = exp.CausalEventMachine.create(
        fixture, learning_rate=0.01, residual_bound=1.0, delay=8
    )
    event = exp.fixture_events()[0]
    machine.predict(event)
    with pytest.raises(ValueError, match="event_identity_invalid"):
        machine.predict(event)
    bad_order = deepcopy(exp.fixture_events()[1])
    bad_order["arrival_index"] = 3
    with pytest.raises(ValueError, match="arrival_order_invalid"):
        machine.predict(bad_order)
    with pytest.raises(ValueError, match="label_invalid"):
        machine.deliver_label(event["event_id"], 2, visible_at=0)

    for selected_event in exp.fixture_events()[1:7]:
        prediction = machine.predict(selected_event)
    assert prediction["audit_selected"] is True
    assert (
        machine.deliver_label(
            selected_event["event_id"], int(selected_event["label"]), visible_at=6
        )
        == "withheld"
    )
    assert (
        machine.deliver_label(
            selected_event["event_id"], int(selected_event["label"]), visible_at=6
        )
        == "duplicate"
    )
    pending = deepcopy(machine.pending_queue)
    permutation = exp.permute_released_batch(
        [pending[0]["label"]], [pending[0]["event_id"]], seed=1
    )
    with pytest.raises(ValueError, match="label_not_available"):
        machine._apply_batch(pending, permutation, visible_at=0)

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="checkpoint_unreadable"):
        exp.CausalEventMachine.load_checkpoint(missing)
    missing.write_text("{}")
    with pytest.raises(ValueError, match="checkpoint_schema_invalid"):
        exp.CausalEventMachine.load_checkpoint(missing)
    missing.write_text(json.dumps({"schema": exp.CHECKPOINT_SCHEMA}))
    with pytest.raises(ValueError, match="checkpoint_models_invalid"):
        exp.CausalEventMachine.load_checkpoint(missing)


# REQ-KAN-7506; SCENARIO-KAN-7506-05; SCENARIO-CL-7506-GATES.
def test_reducer_validator_and_cli_fail_closed_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert exp.collect_preconditions(tmp_path)[1] == []
    artifact = exp.build_fixture_artifact(tmp_path / "fixture")
    assert exp.validate_artifact(artifact) == []
    assert exp._receipts_pass([{"name": "one", "exit_code": 0, "timed_out": False}], ["one"])
    mutations = []
    for transform in (
        lambda value: value.pop("fixture_rows"),
        lambda value: value["fixture_rows"].update({"gradient_checks": "bad"}),
        lambda value: value["fixture_rows"]["gradient_checks"].clear(),
        lambda value: [
            row.update({"gradient_norm": 0.0}) for row in value["fixture_rows"]["gradient_checks"]
        ],
        lambda value: value["fixture_rows"]["head_controls"].pop(0),
        lambda value: value["fixture_rows"]["head_controls"][0].update({"parameter_count": 257}),
        lambda value: value["fixture_rows"]["head_controls"][-1].update({"state_changed": True}),
        lambda value: value["causality_rows"].pop(),
        lambda value: value["causality_rows"][0].update({"future_access_violations": 1}),
        lambda value: value["causality_rows"][0].update({"predict_before_update": False}),
        lambda value: value["fixture_rows"].update({"restart_check": {"passed": False}}),
        lambda value: value.update({"validation_receipts": [{"exit_code": 1}]}),
    ):
        changed = deepcopy(artifact)
        transform(changed)
        mutations.append(exp.independent_reduce(changed))
    assert all(row["causal_update_ready_score"] == 0 for row in mutations)

    for field, replacement in (
        ("verdict_class", "wrong"),
        ("honest_verdict", "unfinished"),
        ("preconditions_checked", [{"passed": False}]),
        ("acceptance_gate_results", [{"principle": ""}]),
        ("field_principles", {}),
        ("source_artifact_hashes", "bad"),
        ("source_artifact_hashes", ["bad"]),
        ("source_artifact_hashes", [{"path": "missing", "sha256": "sha256:bad"}]),
    ):
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert exp.validate_artifact(changed)
    assert exp.cold_replay(tmp_path / "absent.json") == ["artifact_unreadable_or_not_object"]
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[")
    assert exp._load_object(malformed) == {}
    malformed.write_text("[]")
    assert exp._load_object(malformed) == {}
    with pytest.raises(SystemExit, match=f"--date must be {exp.RUN_DATE}"):
        exp.main(["--date", "19000101"])
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda *_args, **_kwargs: {"causal_update_ready_score": 1, "verdict_class": "null"},
    )
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert '"result"' in capsys.readouterr().out

    class RewritingMachine:
        predictions = {"event": {"different": True}}

        def predict(self, _event: object) -> dict[str, object]:
            return {"different": False}

        def deliver_label(self, *_args: object, **_kwargs: object) -> str:
            return "not_selected"

        def advance(self, _visible_at: int) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(
        exp.CausalEventMachine, "create", lambda *_args, **_kwargs: RewritingMachine()
    )
    with pytest.raises(ValueError, match="prediction_rewritten"):
        exp.run_causal_fixture(
            [
                {
                    "event_id": "event",
                    "arrival_index": 0,
                    "label": 0,
                    "features": [0.0, 0.0, 0.0, 0.0],
                    "base_probability": 0.5,
                }
            ],
            delay=0,
        )
