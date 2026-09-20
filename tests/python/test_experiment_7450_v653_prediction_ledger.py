"""Tests for REQ-AUTO-7450 and SCENARIO-AUTO-7450-*.

The fixtures exercise the causal ledger only. They do not repeat the Exp7440
scientific stream or treat exact synthetic arithmetic as online value.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7450_v653_prediction_ledger as exp


ROOT = Path(__file__).resolve().parents[2]


def _passing_receipts() -> list[dict[str, object]]:
    """Return one passing receipt for every frozen validation command."""

    return [
        {"name": name, "required": True, "passed": True, "exit_code": 0}
        for name in (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def _fixture(tmp_path: Path) -> dict:
    """Build one complete private ledger whose sidecars stay outside the repo."""

    return exp.build_fixture_ledger(tmp_path, relative_dir=Path("ledger"))


def test_req_auto_7450_spec_preconditions_and_frozen_protocol() -> None:
    """REQ-AUTO-7450 authenticates inputs and freezes Exp7440 unchanged."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "### REQ-AUTO-7450:" in text
    for number in range(1, 7):
        assert f"SCENARIO-AUTO-7450-{number:02d}" in text

    checks, hashes, upstream = exp.collect_preconditions(ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert upstream["honest_verdict"] == "complete_null_insufficient_online_benefit"
    assert upstream["flagged_adversarial"] is False
    assert hashes[exp.UPSTREAM_PATH.as_posix()]["original_verdict_class"] == "null"

    protocol = exp.build_replay_protocol()
    assert protocol["orders"] == ["hash_order", "domain_blocked_shift_order"]
    assert protocol["delays"] == [0, 8]
    assert protocol["training_seeds"] == [65201, 65202, 65203, 65204, 65205]
    assert protocol["reveal_schedule"] == {
        "full_block": "uniform_eight_of_32",
        "target_fraction": 0.25,
        "partial_final_block": "floor(n/4)",
    }
    assert protocol["moving_block_lengths"] == [32, 64]
    assert protocol["bootstrap_draws"] == 10_000
    assert len(protocol["arms"]) == 7
    assert len(protocol["primary_comparators"]) == 3
    assert protocol["full_scientific_stream_executed"] is False
    assert len(protocol["protocol_hash"]) == 71


def test_scenario_auto_7450_01_prediction_is_immutable_and_label_free(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7450-01 seals all prediction operands before reveal."""

    fixture = _fixture(tmp_path)
    prediction = fixture["prediction_events"][0]
    required = {
        "event_id",
        "group_id",
        "order",
        "delay",
        "seed",
        "request_order",
        "expert_probabilities",
        "mixture_weights",
        "mixture_prediction",
        "expert_checkpoint_hashes",
        "label_propensity",
        "pre_feedback_state_hash",
        "event_hash",
    }
    assert required <= prediction.keys()
    assert not ({"label", "loss", "per_expert_loss"} & prediction.keys())
    assert exp.prediction_event_hash(prediction) == prediction["event_hash"]
    assert fixture["persisted_before_reveal"] is True

    event = exp.PredictionEvent.from_dict(prediction)
    with pytest.raises((AttributeError, TypeError)):
        event.delay = 99  # type: ignore[misc]
    durable = exp.read_jsonl(tmp_path / fixture["ledger_path"])
    assert durable[0] == prediction
    assert durable[1]["prediction_event_hash"] == prediction["event_hash"]


def test_scenario_auto_7450_02_update_matches_independent_scalar_math(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7450-02 replays saved probabilities, not live experts."""

    fixture = _fixture(tmp_path)
    prediction = fixture["prediction_events"][0]
    feedback = fixture["feedback_events"][0]
    names = exp.EXPERT_NAMES
    losses = {name: -math.log(prediction["expert_probabilities"][name]) for name in names}
    old = feedback["old_log_weights"]
    penalized = {name: old[name] - losses[name] for name in names}
    maximum = max(penalized.values())
    exponentials = {name: math.exp(penalized[name] - maximum) for name in names}
    denominator = math.fsum(exponentials.values())
    posterior = {name: exponentials[name] / denominator for name in names}
    shared = {name: 0.99 * posterior[name] + 0.01 / 4.0 for name in names}
    expected_logs = {name: math.log(shared[name]) for name in names}

    assert feedback["per_expert_loss"] == pytest.approx(losses)
    assert feedback["numeric_update"]["penalized_log_weights"] == pytest.approx(penalized)
    assert feedback["normalizer"]["maximum"] == pytest.approx(maximum)
    assert feedback["normalizer"]["sum_exp"] == pytest.approx(denominator)
    assert feedback["numeric_update"]["posterior_before_share"] == pytest.approx(posterior)
    assert feedback["new_log_weights"] == pytest.approx(expected_logs)
    assert exp.cold_replay(fixture, root=tmp_path)["errors"] == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("changed_expert_prediction", "prediction_event_hash_mismatch"),
        ("future_label_access", "feedback_before_reveal"),
        ("duplicated_feedback", "duplicate_feedback"),
        ("reordered_updates", "feedback_order_invalid"),
    ],
)
def test_scenario_auto_7450_03_causal_mutations_fail_closed(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    """SCENARIO-AUTO-7450-03 rejects each causal record corruption."""

    fixture = _fixture(tmp_path)
    changed = exp.mutate_fixture(fixture, mutation)
    replay = exp.cold_replay(changed, root=tmp_path)
    assert expected in replay["errors"]
    assert replay["valid"] is False


def test_scenario_auto_7450_03_deleted_checkpoint_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7450-03 requires every recorded checkpoint byte hash."""

    fixture = _fixture(tmp_path)
    checkpoint = tmp_path / fixture["initial_state_manifest"]["checkpoints"][0]["path"]
    checkpoint.unlink()
    replay = exp.cold_replay(fixture, root=tmp_path)
    assert "checkpoint_missing" in replay["errors"]


def test_scenario_auto_7450_04_crash_recovery_replays_durable_events(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7450-04 restores a prediction-only crash exactly."""

    fixture = _fixture(tmp_path)
    crash = fixture["crash_recovery"]
    assert crash["prediction_persisted_before_crash"] is True
    assert crash["prediction_hash_preserved"] is True
    assert crash["recovered_final_state_hash"] == crash["uninterrupted_final_state_hash"]
    assert crash["passed"] is True


def test_scenario_auto_7450_05_no_feedback_arm_does_not_move(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7450-05 keeps the no-feedback numeric state fixed."""

    fixture = _fixture(tmp_path)
    control = fixture["no_feedback_control"]
    assert control["prediction_count"] > 0
    assert control["feedback_count"] == 0
    assert control["initial_state_hash"] == control["final_state_hash"]
    assert control["passed"] is True


def test_scenario_auto_7450_06_artifact_reduces_and_rejects_drift(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7450-06 binds readiness to replay and all mutations."""

    artifact = exp.build_fixture_artifact(
        tmp_path,
        validation_receipts=_passing_receipts(),
        relative_dir=Path("artifact-ledger"),
    )
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["prediction_ledger_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert len(artifact["service_stage_receipts"]) == 5
    assert artifact["protocol_hash"] == artifact["replay_protocol"]["protocol_hash"]
    assert exp.independent_reduce(artifact, root=tmp_path) == artifact["independent_reduction"]
    assert exp.validate_artifact(artifact, root=tmp_path) == []

    changed = deepcopy(artifact)
    changed["ledger"]["prediction_events"][0]["mixture_prediction"] = 0.01
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed, root=tmp_path)


def test_req_auto_7450_blocked_and_cli_readers_are_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-AUTO-7450 preserves exact blockers and fresh-process reader modes."""

    failed = {
        "check": "upstream_flag",
        "upstream": "exp7440-v652-mixture-learning",
        "path": exp.UPSTREAM_PATH.as_posix(),
        "field": "flagged_adversarial",
        "expected": False,
        "observed": True,
        "passed": False,
    }
    blocked = exp.build_blocked_artifact(failed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_upstream_flag"
    assert blocked["gate_check_summary"]["observed"] is True

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"ok": True}), encoding="utf-8")
    monkeypatch.setattr(exp, "validate_artifact", lambda value, root: [])
    assert (
        exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path), "--cold-replay", str(candidate)])
        == 0
    )
    monkeypatch.setattr(exp, "independent_reduce", lambda value, root: {"ready": 1})
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--root",
                str(tmp_path),
                "--independent-reduce",
                str(candidate),
            ]
        )
        == 0
    )
    monkeypatch.setattr(exp, "run_experiment", lambda root, run_date, output_path: {"ok": True})
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path)]) == 0
    with pytest.raises(SystemExit, match="--date"):
        exp.main(["--date", "wrong"])
    assert "ready" in capsys.readouterr().out


def test_req_auto_7450_input_and_collector_rejections(tmp_path: Path) -> None:
    """REQ-AUTO-7450 rejects malformed bytes and illegal causal operations."""

    assert exp._load_object(tmp_path / "missing.json") == {}
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    assert exp._load_object(sequence) == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    with pytest.raises(ValueError, match="finite"):
        exp._clip_probability(math.nan)
    with pytest.raises(ValueError, match="binary"):
        exp._loss(2, 0.5)
    bad_rows = tmp_path / "bad.jsonl"
    bad_rows.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        exp.read_jsonl(bad_rows)

    fixture = _fixture(tmp_path)
    event = exp.PredictionEvent.from_dict(fixture["prediction_events"][0])
    assert event.to_dict() == fixture["prediction_events"][0]
    changed = deepcopy(fixture["prediction_events"][0])
    changed["mixture_prediction"] = 0.0
    with pytest.raises(ValueError, match="hash"):
        exp.PredictionEvent.from_dict(changed)

    collector = exp.LedgerCollector(
        tmp_path / "collector-events.jsonl", fixture["initial_state_manifest"]
    )
    probabilities = {name: 0.5 for name in exp.EXPERT_NAMES}
    common = {
        "event_id": "new",
        "group_id": "group-new",
        "arm": exp.LEARNED_ARM,
        "order": "hash_order",
        "delay": 8,
        "seed": exp.TRAINING_SEEDS[0],
        "request_order": 2,
        "expert_probabilities": probabilities,
        "label_propensity": 0.25,
    }
    with pytest.raises(ValueError, match="registered"):
        collector.persist_prediction(**{**common, "arm": "unknown"})
    with pytest.raises(ValueError, match="protocol"):
        collector.persist_prediction(**{**common, "order": "wrong"})
    with pytest.raises(ValueError, match="four named"):
        collector.persist_prediction(**{**common, "expert_probabilities": {}})
    with pytest.raises(ValueError, match="propensity"):
        collector.persist_prediction(**{**common, "label_propensity": 0.0})
    prediction = collector.persist_prediction(**common)
    with pytest.raises(ValueError, match="new"):
        collector.persist_prediction(**common)
    with pytest.raises(ValueError, match="binary"):
        collector.reveal(prediction["event_hash"], label=2, reveal_order=10, label_origin="x")
    with pytest.raises(ValueError, match="persisted"):
        collector.reveal("sha256:unknown", label=1, reveal_order=10, label_origin="x")
    with pytest.raises(ValueError, match="visible"):
        collector.reveal(prediction["event_hash"], label=1, reveal_order=9, label_origin="x")
    collector.reveal(prediction["event_hash"], label=1, reveal_order=10, label_origin="x")
    with pytest.raises(ValueError, match="already"):
        collector.reveal(prediction["event_hash"], label=1, reveal_order=10, label_origin="x")
    with pytest.raises(ValueError, match="unknown mutation"):
        exp.mutate_fixture(fixture, "not-registered")


def test_req_auto_7450_cold_reader_defensive_corruptions(tmp_path: Path) -> None:
    """REQ-AUTO-7450 reports malformed states, rows, hashes, and numeric operands."""

    fixture = _fixture(tmp_path)

    checkpoint = deepcopy(fixture)
    checkpoint["initial_state_manifest"]["checkpoints"][0]["sha256"] = "sha256:wrong"
    assert "checkpoint_hash_mismatch" in exp.cold_replay(checkpoint, root=tmp_path)["errors"]

    initial = deepcopy(fixture)
    del initial["initial_state_manifest"]["arms"][exp.LEARNED_ARM]["initial_log_weights"]
    initial["initial_state_manifest"]["arms"][exp.LEARNED_ARM]["initial_state_hash"] = "wrong"
    errors = exp.cold_replay(initial, root=tmp_path)["errors"]
    assert "initial_state_invalid" in errors
    assert "initial_state_hash_mismatch" in errors

    duplicate = deepcopy(fixture)
    prediction = deepcopy(duplicate["prediction_events"][0])
    duplicate["events"].insert(1, prediction)
    duplicate["prediction_events"].insert(1, deepcopy(prediction))
    assert "duplicate_prediction" in exp.cold_replay(duplicate, root=tmp_path)["errors"]

    bad_arm = deepcopy(fixture)
    target = bad_arm["events"][0]
    target["arm"] = "unknown"
    bad_arm["prediction_events"][0] = deepcopy(target)
    assert "prediction_arm_invalid" in exp.cold_replay(bad_arm, root=tmp_path)["errors"]

    bad_prediction = deepcopy(fixture)
    target = bad_prediction["events"][0]
    del target["expert_probabilities"][exp.EXPERT_NAMES[0]]
    target["label"] = 1
    bad_prediction["prediction_events"][0] = deepcopy(target)
    errors = exp.cold_replay(bad_prediction, root=tmp_path)["errors"]
    assert "expert_probabilities_incomplete" in errors
    assert "prediction_contains_label" in errors
    assert "prediction_numeric_invalid" in errors

    bad_feedback = deepcopy(fixture)
    feedback_index = next(
        index
        for index, row in enumerate(bad_feedback["events"])
        if row["row_type"] == "feedback_event"
    )
    bad_feedback["events"][feedback_index]["label"] = "bad"
    bad_feedback["feedback_events"][0] = deepcopy(bad_feedback["events"][feedback_index])
    assert "feedback_numeric_invalid" in exp.cold_replay(bad_feedback, root=tmp_path)["errors"]

    bad_type = deepcopy(fixture)
    bad_type["events"].append({"row_type": "unknown"})
    assert "event_type_invalid" in exp.cold_replay(bad_type, root=tmp_path)["errors"]
    assert exp.independent_reduce(fixture, root=tmp_path)["prediction_ledger_ready_score"] == 1


def test_req_auto_7450_validator_reports_each_declaration_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-AUTO-7450 makes artifact declaration and reader failures explicit."""

    artifact = exp.build_fixture_artifact(
        tmp_path,
        validation_receipts=_passing_receipts(),
        relative_dir=Path("validator-ledger"),
    )
    mutations = {
        "schema": "wrong",
        "milestone": "wrong",
        "MODEL_SPECS": ["wrong"],
        "invocation_counts": {},
        "inference_substrate_class": "wrong",
        "execution_venue": "wrong",
        "promotion_score": 1,
        "validation_receipts": [],
        "protocol_hash": "sha256:wrong",
    }
    expected = {
        "schema": "artifact_identity_invalid",
        "milestone": "artifact_schedule_invalid",
        "MODEL_SPECS": "current_model_declaration_invalid",
        "invocation_counts": "current_invocation_counts_invalid",
        "inference_substrate_class": "inference_substrate_class_invalid",
        "execution_venue": "execution_venue_invalid",
        "promotion_score": "claim_boundary_invalid",
        "validation_receipts": "required_validation_incomplete",
        "protocol_hash": "protocol_hash_mismatch",
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected[field] in exp.validate_artifact(changed, root=tmp_path)

    monkeypatch.setattr(
        exp, "independent_reduce", lambda value, root: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert any(
        error.startswith("independent_reduction_failed:")
        for error in exp.validate_artifact(artifact, root=tmp_path)
    )
    candidate = tmp_path / "reader-error.json"
    candidate.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--root",
                str(tmp_path),
                "--independent-reduce",
                str(candidate),
            ]
        )
        == 1
    )
    assert "bad" in capsys.readouterr().out
