"""Tests for the bounded risk-sensitive memory controller.

Spec refs: REQ-CL-6854 and SCENARIO-CL-6854-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Callable

import pytest

from carnot.continuous_learning import (
    ACTIONS,
    CheckpointCorruptionError,
    FeedbackError,
    PolicyConfig,
    RiskMatrix,
    RiskSensitiveContextualBandit,
    canonical_json_bytes,
    deterministic_argmin,
    encode_context,
    sha256_json,
)
from carnot import experiment_6854_risk_sensitive_abstention_memory_controller as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def current_fixture() -> dict:
    """Load the frozen opportunity fixture once for all replay tests."""

    return exp.load_fixture(REPO / exp.SOURCE_RELATIVE_PATH)


@pytest.fixture(scope="session")
def current_artifact(current_fixture: dict, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build one full replay without writing to the tracked results directory."""

    return exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        fixture=current_fixture,
        checkpoint_dir=tmp_path_factory.mktemp("exp6854-checkpoints"),
    )


def _context(*, family: str = exp.MANDATED_FAMILIES[0], status: str = "stale_correction_pending") -> dict:
    """Return one valid pre-outcome context for small controller tests."""

    return {
        "relevance": 0.9,
        "uncertainty": 0.2,
        "exact_compatibility": True,
        "age": 1,
        "correction_status": status,
        "family": family,
        "capacity": {"budget": 2, "factual_read_admitted": True},
        "false_positive_risk": 0.3,
        "cost": {"verified_memory": 2, "no_memory": 1, "abstain": 0},
    }


def test_req_cl_6854_artifact_is_complete_and_row_supported(current_artifact: dict) -> None:
    """REQ-CL-6854: a valid replay emits every required field and receipt."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(current_artifact)
    assert current_artifact["risk_sensitive_controller_complete_score"] == 1
    assert current_artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert current_artifact["verifier_is_oracle"] is False
    assert current_artifact["honest_verdict"].startswith("complete_")
    assert len(current_artifact["rows"]) == 765
    assert len(current_artifact["pre_outcome_action_receipts"]) == 765
    assert len(current_artifact["exact_feedback_receipts"]) == 765
    assert len(current_artifact["update_rows"]) == 765


@pytest.mark.parametrize(
    "failure",
    ["readiness", "row_hash", "outcome", "headroom", "dirty_state"],
)
def test_scenario_cl_6854_preconditions_fail_closed(
    current_fixture: dict,
    failure: str,
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6854-PRECONDITIONS: each invalid input writes blocked evidence."""

    fixture = deepcopy(current_fixture)
    controller = RiskSensitiveContextualBandit()
    if failure == "readiness":
        fixture["risk_sensitive_stream_ready_score"] = 0
    elif failure == "row_hash":
        fixture["rows"][0]["row_sha256"] = "sha256:" + "0" * 64
    elif failure == "outcome":
        fixture["rows"][0]["exact_later_outcome"] = None
    elif failure == "headroom":
        for row in fixture["rows"]:
            row["exact_later_outcome"]["signed_direction"] = 0
            row["row_sha256"] = exp.fixture_row_hash(row)
    else:
        controller.freeze_action("dirty", _context(), due_sequence_index=1)

    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        fixture=fixture,
        checkpoint_dir=tmp_path,
        initial_controller=controller,
    )
    assert artifact["risk_sensitive_controller_complete_score"] == 0
    assert artifact["controller_benefit_gate_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] is not None
    assert artifact["gate_check_summary"]["observed"] is not None


def test_scenario_cl_6854_action_freezes_before_delayed_feedback() -> None:
    """SCENARIO-CL-6854-ACTION-FREEZE: receipts exist before exact feedback."""

    controller = RiskSensitiveContextualBandit()
    receipt = controller.freeze_action("decision-1", _context(), due_sequence_index=3)
    assert receipt["serialized_before_feedback"] is True
    assert receipt["policy_state_sha256_before"].startswith("sha256:")
    assert receipt["context_sha256"].startswith("sha256:")
    assert "outcome" not in json.dumps(receipt).lower()
    assert controller.pending_count == 1

    loss = RiskMatrix(3.0).loss(receipt["chosen_action"], direction=1)
    update = controller.apply_bounded_loss(
        "decision-1",
        loss,
        feedback_receipt_sha256="sha256:" + "1" * 64,
        update_sequence_index=3,
    )
    assert update["action_receipt_sha256"] == receipt["receipt_sha256"]
    assert update["bounded_loss"] == loss
    assert controller.pending_count == 0


def test_scenario_cl_6854_delayed_feedback_rejects_unknown_decision() -> None:
    """SCENARIO-CL-6854-DELAYED-FEEDBACK: feedback needs a frozen action."""

    controller = RiskSensitiveContextualBandit()
    with pytest.raises(FeedbackError, match="no pending action"):
        controller.apply_bounded_loss(
            "missing",
            0.0,
            feedback_receipt_sha256="sha256:" + "2" * 64,
            update_sequence_index=1,
        )


@pytest.mark.parametrize("ratio", exp.RISK_SENSITIVITY_RATIOS)
def test_scenario_cl_6854_asymmetric_loss_matrix(ratio: float) -> None:
    """SCENARIO-CL-6854-ASYMMETRIC-LOSS: injection dominates safer misses."""

    matrix = RiskMatrix(ratio)
    harmful = matrix.loss("verified_memory", direction=-1)
    missed = matrix.loss("no_memory", direction=1)
    abstained = matrix.loss("abstain", direction=1)
    assert harmful > missed
    assert harmful > abstained
    assert matrix.as_dict()["risk_ratio"] == ratio


def test_scenario_cl_6854_unseen_context_abstains_without_feature_growth() -> None:
    """SCENARIO-CL-6854-UNSEEN-CONTEXT: unknown categories use fixed safety data."""

    known = encode_context(_context())
    unknown = encode_context(_context(family="unseen-family", status="future-status"))
    assert len(known.features) == len(unknown.features)
    assert unknown.unseen is True
    assert set(unknown.unseen_reasons) == {"family", "correction_status"}

    controller = RiskSensitiveContextualBandit()
    receipt = controller.freeze_action("unseen", _context(family="unseen-family"), 1)
    assert receipt["chosen_action"] == "abstain"
    assert receipt["selection_reason"] == "conservative_unseen_context"


def test_scenario_cl_6854_capacity_bounds_state_and_pending_rows() -> None:
    """SCENARIO-CL-6854-CAPACITY: long replay cannot grow controller state."""

    config = PolicyConfig(max_updates_per_action=8, max_pending=2)
    controller = RiskSensitiveContextualBandit(config)
    matrix = RiskMatrix(config.risk_ratio)
    for index in range(40):
        decision_id = f"decision-{index}"
        receipt = controller.freeze_action(decision_id, _context(), index + 1)
        controller.apply_bounded_loss(
            decision_id,
            matrix.loss(receipt["chosen_action"], direction=index % 3 - 1),
            feedback_receipt_sha256="sha256:" + f"{index:064x}",
            update_sequence_index=index + 1,
        )
    assert max(row["updates"] for row in controller.action_stats.values()) <= 8
    assert controller.pending_count == 0
    assert len(controller.to_bytes()) <= config.max_state_bytes

    controller.freeze_action("pending-1", _context(), 100)
    controller.freeze_action("pending-2", _context(), 100)
    with pytest.raises(FeedbackError, match="pending capacity"):
        controller.freeze_action("pending-3", _context(), 100)


def test_scenario_cl_6854_checkpoint_corruption_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-6854-CHECKPOINT-CORRUPTION: changed bytes never load."""

    controller = RiskSensitiveContextualBandit()
    path = tmp_path / "controller.json"
    manifest = controller.save_checkpoint(path)
    loaded = RiskSensitiveContextualBandit.load_checkpoint(path)
    assert loaded.to_bytes() == controller.to_bytes()
    assert manifest["checkpoint_sha256"].startswith("sha256:")

    corrupted = bytearray(path.read_bytes())
    corrupted[-2] = ord("x")
    with pytest.raises(CheckpointCorruptionError):
        RiskSensitiveContextualBandit.from_bytes(bytes(corrupted))

    envelope = json.loads(controller.to_bytes())
    envelope["payload_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(CheckpointCorruptionError, match="payload hash"):
        RiskSensitiveContextualBandit.from_bytes(
            (json.dumps(envelope, sort_keys=True) + "\n").encode()
        )


def test_scenario_cl_6854_restart_is_byte_identical(current_artifact: dict) -> None:
    """SCENARIO-CL-6854-RESTART-EQUIVALENCE: resumed replay matches clean bytes."""

    result = current_artifact["restart_equivalence_results"]
    assert result["checkpoint_bytes_round_trip"] is True
    assert result["final_bytes_identical"] is True
    assert result["clean_final_state_sha256"] == result["restarted_final_state_sha256"]


def test_scenario_cl_6854_rollback_restores_parent_bytes(current_artifact: dict) -> None:
    """SCENARIO-CL-6854-ROLLBACK: poison is bounded and rollback is exact."""

    result = current_artifact["rollback_results"]
    assert result["poison_loss_was_clamped"] is True
    assert result["mutated_state_sha256"] != result["parent_state_sha256"]
    assert result["restored_state_sha256"] == result["parent_state_sha256"]
    assert result["restored_parent_bytes"] is True
    assert result["model_weight_files_touched"] == []


def test_scenario_cl_6854_tie_breaking_is_deterministic() -> None:
    """SCENARIO-CL-6854-TIE-BREAKING: the safety order resolves equal scores."""

    equal = {action: 0.5 for action in ACTIONS}
    assert deterministic_argmin(equal) == "abstain"
    equal.pop("abstain")
    assert deterministic_argmin(equal) == "no_memory"
    equal.pop("no_memory")
    assert deterministic_argmin(equal) == "verified_memory"
    with pytest.raises(ValueError, match="no action scores"):
        deterministic_argmin({})


def test_scenario_cl_6854_controls_share_every_decision(current_artifact: dict) -> None:
    """SCENARIO-CL-6854-CONTROLS: each row contains all matched arm metrics."""

    expected = set(exp.POLICY_ARMS)
    decision_ids = []
    for row in current_artifact["rows"]:
        assert set(row["arm_metrics"]) == expected
        assert all(metric["decision_id"] == row["decision_id"] for metric in row["arm_metrics"].values())
        decision_ids.append(row["decision_id"])
    assert len(decision_ids) == len(set(decision_ids)) == 765


def test_scenario_cl_6854_completion_and_benefit_gates_are_separate() -> None:
    """SCENARIO-CL-6854-GATES: scientific benefit cannot change execution completeness."""

    gates = exp.compute_terminal_gates(
        planned_count=2,
        rows=[{"decision_id": "a"}, {"decision_id": "b"}],
        action_receipts=[{"decision_id": "a"}, {"decision_id": "b"}],
        feedback_receipts=[{"decision_id": "a"}, {"decision_id": "b"}],
        update_rows=[{"decision_id": "a"}, {"decision_id": "b"}],
        restart_ok=True,
        rollback_ok=True,
        bounded_state=True,
        held_future_effect=0.0,
        false_positive_injection_rate=0.0,
    )
    assert gates["risk_sensitive_controller_complete_score"] == 1
    assert gates["controller_benefit_gate_score"] == 0


def test_req_cl_6854_sensitivity_and_summary_are_row_derived(current_artifact: dict) -> None:
    """REQ-CL-6854: fixed risk ratios and policy summaries remain explicit."""

    assert [row["risk_ratio"] for row in current_artifact["risk_sensitivity_rows"]] == list(
        exp.RISK_SENSITIVITY_RATIOS
    )
    assert set(current_artifact["per_arm_summary"]) == set(exp.POLICY_ARMS)
    learned = current_artifact["per_arm_summary"]["contextual_bandit"]
    assert learned["decision_count"] == len(current_artifact["rows"])
    assert 0.0 <= current_artifact["false_positive_injection_rate"] <= 1.0
    assert 0.0 <= current_artifact["abstention_rate"] <= 1.0


def test_req_cl_6854_checkpoint_manifest_and_latency_are_bounded(current_artifact: dict) -> None:
    """REQ-CL-6854: persisted state, storage, and update timing have receipts."""

    manifest = current_artifact["checkpoint_manifest"]
    assert manifest
    assert all(row["round_trip_bytes_identical"] for row in manifest)
    assert all(row["state_bytes"] <= current_artifact["controller_schema"]["max_state_bytes"] for row in manifest)
    assert current_artifact["state_size_rows"]
    assert current_artifact["latency_rows"]
    assert all(row["update_latency_us"] >= 0.0 for row in current_artifact["latency_rows"])


def test_req_cl_6854_artifact_matches_deterministic_replay(
    current_artifact: dict,
    current_fixture: dict,
) -> None:
    """REQ-CL-6854: the checked-in result has the same stable replay decisions."""

    stored = json.loads((REPO / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert stored["reproducibility_checksum"] == exp.reproducibility_checksum(stored)
    assert stored["rows"] == current_artifact["rows"]
    assert stored["per_arm_summary"] == current_artifact["per_arm_summary"]
    assert stored["source_artifact_hashes"] == exp.source_artifact_hashes(REPO)
    assert current_fixture["risk_sensitive_stream_ready_score"] == 1


def test_req_cl_6854_writer_and_cli_use_requested_paths(
    current_artifact: dict,
    tmp_path: Path,
) -> None:
    """REQ-CL-6854: validated output never needs the tracked result path."""

    invalid = deepcopy(current_artifact)
    invalid["reproducibility_checksum"] = "stale"
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        exp.write_artifact(tmp_path / "invalid.json", invalid)

    direct = tmp_path / "nested" / "direct.json"
    exp.write_artifact(direct, current_artifact)
    assert json.loads(direct.read_text(encoding="utf-8")) == current_artifact

    cli = tmp_path / "cli.json"
    checkpoints = tmp_path / "cli-checkpoints"
    assert exp.main(
        [
            "--date",
            "20260901",
            "--output",
            str(cli),
            "--checkpoint-dir",
            str(checkpoints),
        ]
    ) == 0
    assert json.loads(cli.read_text(encoding="utf-8"))["risk_sensitive_controller_complete_score"] == 1


def test_req_cl_6854_source_and_artifact_validation_are_defensive(
    current_artifact: dict,
    tmp_path: Path,
) -> None:
    """REQ-CL-6854: malformed input and inconsistent output fail visibly."""

    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{invalid", encoding="utf-8")
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp.load_fixture(missing) == {"_load_error": "FileNotFoundError"}
    assert exp.load_fixture(invalid) == {"_load_error": "JSONDecodeError"}
    assert exp.load_fixture(array) == {"_load_error": "not_object"}

    changed = deepcopy(current_artifact)
    changed.pop("rows")
    changed["inference_substrate"] = "wrong"
    changed["verifier_is_oracle"] = True
    changed["honest_verdict"] = "pending"
    errors = exp.validate_artifact(changed)
    assert any("missing required fields" in error for error in errors)
    assert "invalid inference_substrate" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "honest_verdict must start with complete_" in errors
    assert "reproducibility_checksum mismatch" in errors


def test_scenario_cl_6854_invalid_risk_and_policy_bounds_are_rejected() -> None:
    """SCENARIO-CL-6854-ASYMMETRIC-LOSS: invalid safety bounds fail closed."""

    with pytest.raises(ValueError, match="missed_reuse_loss"):
        RiskMatrix(1.0)
    with pytest.raises(ValueError, match="helpful_abstention_loss"):
        RiskMatrix(1.5, missed_reuse_loss=1.0, helpful_abstention_loss=2.0)
    with pytest.raises(ValueError, match="unknown action"):
        RiskMatrix().loss("unknown", 1)
    with pytest.raises(ValueError, match="direction"):
        RiskMatrix().loss("abstain", 2)

    with pytest.raises(ValueError, match="numeric bounds"):
        PolicyConfig(ridge=0.0)
    with pytest.raises(ValueError, match="capacities"):
        PolicyConfig(max_pending=0)
    with pytest.raises(ValueError, match="too small"):
        PolicyConfig(max_state_bytes=1000)


def test_scenario_cl_6854_malformed_context_uses_conservative_fixed_vector() -> None:
    """SCENARIO-CL-6854-UNSEEN-CONTEXT: malformed values cannot extend policy state."""

    malformed = _context()
    malformed.update(
        {
            "relevance": True,
            "uncertainty": 2.0,
            "exact_compatibility": False,
            "age": -1,
            "capacity": "invalid",
            "false_positive_risk": float("inf"),
            "nested": [{"exact_outcome_hash": "leak"}],
        }
    )
    encoded = encode_context(malformed)
    assert set(encoded.unseen_reasons) >= {
        "relevance",
        "uncertainty",
        "exact_compatibility",
        "age",
        "capacity",
        "false_positive_risk",
        "denied_field",
    }

    bad_budget = _context()
    bad_budget["capacity"] = {"budget": True, "factual_read_admitted": False}
    assert "capacity" in encode_context(bad_budget).unseen_reasons
    with pytest.raises(ValueError, match="no known action"):
        deterministic_argmin({"unknown": 0.0})


def _checkpoint_with_change(
    controller: RiskSensitiveContextualBandit,
    change: Callable[[dict], None],
) -> bytes:
    """Return a rehashed checkpoint after a test mutates its payload."""

    envelope = json.loads(controller.to_bytes())
    change(envelope["payload"])
    envelope["payload_sha256"] = sha256_json(envelope["payload"])
    return canonical_json_bytes(envelope)


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda payload: payload.update(schema="wrong"), "schema mismatch"),
        (lambda payload: payload.update(config={"risk_ratio": 0.0}), "config is invalid"),
        (lambda payload: payload.update(action_stats=[]), "state sections"),
        (
            lambda payload: payload["pending"].update(
                {f"pending-{index}": {} for index in range(5)}
            ),
            "pending capacity",
        ),
        (
            lambda payload: payload["action_stats"].update(abstain=[]),
            "action state",
        ),
        (
            lambda payload: payload["action_stats"]["abstain"].update(updates=True),
            "statistics",
        ),
        (
            lambda payload: payload["action_stats"]["abstain"]["precision"].__setitem__(
                0, 0.0
            ),
            "precision",
        ),
        (lambda payload: payload["pending"].update(invalid=[]), "pending rows"),
    ],
)
def test_scenario_cl_6854_checkpoint_structure_rejects_each_corruption(
    change: Callable[[dict], None],
    error: str,
) -> None:
    """SCENARIO-CL-6854-CHECKPOINT-CORRUPTION: every state section validates."""

    controller = RiskSensitiveContextualBandit()
    with pytest.raises(CheckpointCorruptionError, match=error):
        RiskSensitiveContextualBandit.from_bytes(_checkpoint_with_change(controller, change))


def test_scenario_cl_6854_checkpoint_envelope_and_size_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-6854-CHECKPOINT-CORRUPTION: envelope and byte caps are strict."""

    with pytest.raises(CheckpointCorruptionError, match="envelope"):
        RiskSensitiveContextualBandit.from_bytes(b"[]\n")
    with pytest.raises(CheckpointCorruptionError, match="unreadable"):
        RiskSensitiveContextualBandit.load_checkpoint(tmp_path / "missing.json")

    small = RiskSensitiveContextualBandit(PolicyConfig(max_state_bytes=1024))
    small.freeze_action("x" * 2000, _context(), 1)
    with pytest.raises(CheckpointCorruptionError, match="exceeds max_state_bytes"):
        small.to_bytes()

    default = RiskSensitiveContextualBandit()
    oversized = _checkpoint_with_change(
        default,
        lambda payload: payload["config"].update(max_state_bytes=1024),
    )
    with pytest.raises(CheckpointCorruptionError, match="exceeds max_state_bytes"):
        RiskSensitiveContextualBandit.from_bytes(oversized)


def test_scenario_cl_6854_duplicate_freeze_and_nonfinite_feedback_fail() -> None:
    """SCENARIO-CL-6854-DELAYED-FEEDBACK: duplicate and invalid updates are rejected."""

    controller = RiskSensitiveContextualBandit()
    with pytest.raises(FeedbackError, match="blank"):
        controller.freeze_action("", _context(), 1)
    controller.freeze_action("duplicate", _context(), 1)
    with pytest.raises(FeedbackError, match="already pending"):
        controller.freeze_action("duplicate", _context(), 1)
    with pytest.raises(FeedbackError, match="not finite"):
        controller.apply_bounded_loss(
            "duplicate",
            float("nan"),
            feedback_receipt_sha256="sha256:" + "3" * 64,
            update_sequence_index=1,
        )


def test_req_cl_6854_external_paths_temporary_replay_and_complete_validation(
    current_fixture: dict,
    current_artifact: dict,
    tmp_path: Path,
) -> None:
    """REQ-CL-6854: external paths and temporary checkpoints keep strict receipts."""

    assert exp.sha256_file(tmp_path / "missing.json") is None
    external = exp.source_artifact_hashes(tmp_path)
    assert external["exp6853"]["path"].startswith(str(tmp_path))
    assert external["exp6853"]["sha256"] is None

    temporary = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        fixture=current_fixture,
    )
    assert temporary["risk_sensitive_controller_complete_score"] == 1

    bad_receipts = deepcopy(current_artifact)
    bad_receipts["update_rows"] = []
    bad_receipts["gate_check_summary"]["passed"] = False
    bad_receipts["reproducibility_checksum"] = exp.reproducibility_checksum(bad_receipts)
    errors = exp.validate_artifact(bad_receipts)
    assert "complete score contradicts receipt counts" in errors
    assert "complete score contradicts failed execution gates" in errors
