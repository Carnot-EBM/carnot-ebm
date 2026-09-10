"""RED-first tests for the V633 revocable-memory cold audit.

Spec refs: REQ-CL-7185 and SCENARIO-CL-7185-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7185_v633_memory_cold_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER_PATH = REPO_ROOT / "results/experiment_7184_v633_revocable_template_csl.json"
STREAM_PATH = REPO_ROOT / "results/experiment_7183_v633_supersession_stream.json"
SPEC_PATH = REPO_ROOT / "openspec/capabilities/continuous-learning/spec.md"


@pytest.fixture(scope="module")
def snapshot() -> dict[str, object]:
    """Capture immutable producer and sidecar bytes once for parser tests."""

    return exp.capture_input_snapshot(
        repo_root=REPO_ROOT,
        producer_artifact_path=PRODUCER_PATH,
        stream_artifact_path=STREAM_PATH,
    )


@pytest.fixture(scope="module")
def inputs(snapshot: dict[str, object]) -> dict[str, object]:
    """Decode only the bytes already bound by the snapshot hashes."""

    return exp.decode_captured_inputs(snapshot)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Run the isolated worker and nested recurrence reload once."""

    root = tmp_path_factory.mktemp("exp7185")
    args = exp.parse_args(
        [
            "--date",
            exp.RUN_DATE,
            "--artifact-path",
            str(root / "result.json"),
            "--checkpoint-path",
            str(root / "checkpoint.json"),
        ]
    )
    return exp._spawn_worker(args)


def test_req_cl_7185_spec_and_field_contract() -> None:
    """REQ-CL-7185: The spec freezes every independent audit surface."""

    text = SPEC_PATH.read_text(encoding="utf-8").split("## REQ-CL-7185", 1)[1]
    for name in (
        "PRECONDITIONS",
        "RAW-REDUCTION",
        "CAUSALITY",
        "COLD-RETENTION",
        "MUTATIONS",
        "ROLLBACK",
        "CREDIT-CONTROL",
        "DELETION",
        "TERMINAL",
    ):
        assert f"SCENARIO-CL-7185-{name}" in text
    assert exp.EVENT_COUNT == 240
    assert exp.ROW_COUNT == 2880
    assert exp.INFERENCE_SUBSTRATE_CLASS == "no_model_load"
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)


def test_scenario_cl_7185_preconditions_pass_and_gate_failure_blocks(
    snapshot: dict[str, object], inputs: dict[str, object], tmp_path: Path
) -> None:
    """SCENARIO-CL-7185-PRECONDITIONS: The exact Exp7184 gate fails closed."""

    checks, decoded = exp.collect_preconditions(
        snapshot,
        artifact_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    assert decoded is not None
    assert all(row["passed"] for row in checks)
    gate = next(row for row in checks if row["check"] == "same_milestone_gate")
    assert gate == {
        "check": "same_milestone_gate",
        "upstream": "exp7184-revocable-template-csl",
        "field": "memory_run_complete_score",
        "expected_value": 1,
        "observed_value": 1,
        "passed": True,
    }

    changed = deepcopy(inputs["producer"])
    changed["memory_run_complete_score"] = 0
    producer = tmp_path / "changed.json"
    producer.write_text(json.dumps(changed), encoding="utf-8")
    failed_snapshot = exp.capture_input_snapshot(
        repo_root=REPO_ROOT,
        producer_artifact_path=producer,
        stream_artifact_path=STREAM_PATH,
    )
    failed, failed_inputs = exp.collect_preconditions(
        failed_snapshot,
        artifact_path=tmp_path / "blocked.json",
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    blocked = exp.build_blocked_artifact(
        snapshot=failed_snapshot,
        checks=failed,
        inputs=failed_inputs,
        run_date=exp.RUN_DATE,
        duration_s=0.1,
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["rows"] == []
    assert blocked["gate_check_summary"]["failed_check"] == "same_milestone_gate"
    assert blocked["gate_check_summary"]["observed_value"] == 0
    assert exp.validate_artifact(blocked) == []


def test_scenario_cl_7185_raw_reduction_uses_authority_sidecar(
    artifact: dict[str, object], inputs: dict[str, object]
) -> None:
    """SCENARIO-CL-7185-RAW-REDUCTION: Sidecar labels own every row metric."""

    assert len(artifact["rows"]) == exp.ROW_COUNT
    assert all(row["passed"] for row in artifact["rows"])
    assert all(row["producer_metric_parity"] for row in artifact["rows"])
    assert {row["metric"] for row in artifact["metric_recomputation_rows"]} == {
        "future_segment",
        "false_acceptance",
        "transfer",
        "recurrence_retention",
    }
    assert all(row["passed"] for row in artifact["producer_metric_parity_rows"])

    changed = deepcopy(inputs)
    changed["producer"]["rows"][0]["exact_label"] = "accept"
    audited = exp.recompute_rows(changed)
    assert audited[0]["authority_exact_label"] == "reject"
    assert audited[0]["producer_metric_parity"] is False


def test_scenario_cl_7185_causality_counts_commits_not_proposals(
    artifact: dict[str, object]
) -> None:
    """SCENARIO-CL-7185-CAUSALITY: Released evidence and real commits are distinct."""

    assert len(artifact["feedback_causality_rows"]) == exp.ROW_COUNT
    assert all(row["passed"] for row in artifact["feedback_causality_rows"])
    assert artifact["actual_controller_addition_count"] == 36
    assert len(artifact["addition_audit_rows"]) == 36
    assert all(row["passed"] for row in artifact["addition_audit_rows"])
    assert len(artifact["revocation_audit_rows"]) == 24
    assert all(row["passed"] for row in artifact["revocation_audit_rows"])
    assert artifact["actual_controller_addition_count"] < len(
        artifact["upstream_rejection_ledger_rows"]
    )


def test_scenario_cl_7185_cold_retention_reloads_saved_bytes(
    artifact: dict[str, object]
) -> None:
    """SCENARIO-CL-7185-COLD-RETENTION: Recurrence starts in a new process."""

    receipt = artifact["cold_reload_process_receipt"]
    assert receipt["fresh_process"] is True
    assert receipt["checkpoint_hash_verified"] is True
    assert receipt["no_model_load"] is True
    assert len(artifact["cold_reload_rows"]) == 36
    assert all(row["passed"] for row in artifact["cold_reload_rows"])
    assert all(row["loaded_state_hash"] == row["expected_state_hash"] for row in artifact["cold_reload_rows"])


def test_scenario_cl_7185_mutations_fail_only_the_named_assertion(
    artifact: dict[str, object]
) -> None:
    """SCENARIO-CL-7185-MUTATIONS: All six attacks fire their isolating check."""

    expected = {
        "early_feedback_exposure": "causal_feedback_release",
        "stale_source_kept_active": "stale_source_inactive",
        "missing_revocation": "revocation_completeness",
        "poison_accepted": "poison_rejected",
        "instance_id_retained": "template_abstraction",
        "forged_before_after_hash": "transition_hash_integrity",
    }
    assert {row["mutation_id"]: row["failed_assertion"] for row in artifact["mutation_rows"]} == expected
    assert all(row["passed"] for row in artifact["mutation_rows"])
    assert all(row["failed_assertion_count"] == 1 for row in artifact["mutation_rows"])


def test_scenario_cl_7185_rollback_is_byte_identical(artifact: dict[str, object]) -> None:
    """SCENARIO-CL-7185-ROLLBACK: Each seed restores its last valid bytes."""

    assert len(artifact["rollback_rows"]) == len(exp.ORDERING_SEEDS)
    assert all(row["passed"] for row in artifact["rollback_rows"])
    assert all(row["byte_equal"] for row in artifact["rollback_rows"])
    assert all(row["checkpoint_hash"] == row["restored_hash"] for row in artifact["rollback_rows"])
    assert all(row["adverse_hash"] != row["restored_hash"] for row in artifact["rollback_rows"])


def test_scenario_cl_7185_credit_and_deletion_controls_retain_negatives(
    artifact: dict[str, object]
) -> None:
    """SCENARIO-CL-7185-CREDIT-CONTROL: Identical shuffled credit stays null."""

    assert len(artifact["credit_control_rows"]) == exp.EVENT_COUNT * 2
    assert {row["control"] for row in artifact["credit_control_rows"]} == {
        "real_family_credit",
        "seeded_shuffled_credit",
    }
    assert all(row["passed"] for row in artifact["credit_control_rows"])
    assert artifact["credit_control_summary"]["decision_difference_count"] == 0
    assert artifact["credit_control_summary"]["credit_assignment_effective"] is False

    assert artifact["deletion_control_rows"]
    assert artifact["deletion_control_summary"]["previously_improved_count"] > 0
    assert artifact["deletion_control_summary"]["changed_decision_count"] > 0
    assert artifact["deletion_control_summary"]["mechanism_decorative"] is False


def test_scenario_cl_7185_terminal_separates_complete_audit_from_promotion(
    artifact: dict[str, object]
) -> None:
    """SCENARIO-CL-7185-TERMINAL: Safe upstream null memory is not promoted."""

    assert artifact["memory_audit_complete_score"] == 1
    assert artifact["upstream_gate_receipt"]["memory_value_score"] == 0
    assert artifact["memory_promotion_score"] == 0
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["verifier_is_oracle"] is True
    assert exp.validate_artifact(artifact) == []

    forged = deepcopy(artifact)
    forged["memory_promotion_score"] = 1
    forged["verdict_class"] = "positive"
    forged["reproducibility_checksum"] = exp.reproducibility_checksum(forged)
    assert "memory_promotion_score_mismatch" in exp.validate_artifact(forged)
    dropped = deepcopy(artifact)
    dropped["mutation_rows"].pop()
    dropped["reproducibility_checksum"] = exp.reproducibility_checksum(dropped)
    assert "memory_audit_complete_score_mismatch" in exp.validate_artifact(dropped)


def test_req_cl_7185_command_writes_and_validates_terminal_artifact(tmp_path: Path) -> None:
    """REQ-CL-7185: The public command runs file, parser, worker, and gate paths."""

    output = tmp_path / "experiment-7185.json"
    checkpoint = tmp_path / "checkpoint.json"
    assert exp.main(
        [
            "--date",
            exp.RUN_DATE,
            "--artifact-path",
            str(output),
            "--checkpoint-path",
            str(checkpoint),
        ]
    ) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert checkpoint.is_file()
    assert written["memory_audit_complete_score"] == 1
    assert written["memory_promotion_score"] == 0
    assert exp.validate_artifact(written) == []
    assert exp.main(["--validate", "--artifact-path", str(output)]) == 0


def test_req_cl_7185_defensive_parser_and_atomic_write(
    snapshot: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7185: Malformed bytes and failed replacement cannot look complete."""

    malformed = deepcopy(snapshot)
    malformed["files"]["producer_artifact"]["bytes"] = b"[]"
    with pytest.raises(ValueError, match="JSON object"):
        exp.decode_captured_inputs(malformed)
    unreadable = exp.capture_input_snapshot(
        repo_root=REPO_ROOT,
        producer_artifact_path=tmp_path / "missing.json",
        stream_artifact_path=STREAM_PATH,
    )
    checks, decoded = exp.collect_preconditions(
        unreadable,
        artifact_path=tmp_path / "blocked.json",
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    assert decoded is None
    assert checks[0]["passed"] is False

    destination = tmp_path / "atomic.json"
    monkeypatch.setattr(exp.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError, match="stop"):
        exp.write_json_atomic(destination, {"safe": True})
    assert not list(tmp_path.glob(".atomic.json.*"))
