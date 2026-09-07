"""RED-first tests for the fresh-process continual-memory cold audit.

Spec refs: REQ-CL-7107 and SCENARIO-CL-7107-PRECONDITIONS,
SCENARIO-CL-7107-RECONSTRUCTION, SCENARIO-CL-7107-METRICS,
SCENARIO-CL-7107-POISON, SCENARIO-CL-7107-ATOMICITY,
SCENARIO-CL-7107-ROLLBACK, SCENARIO-CL-7107-CAPACITY, and
SCENARIO-CL-7107-VERDICT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7107_v623_continual_memory_cold_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER_PATH = REPO_ROOT / "results/experiment_7106_v623_procedural_memory_csl.json"
STREAM_ARTIFACT_PATH = REPO_ROOT / "results/experiment_7105_v623_exact_constraint_stream.json"
SPEC_PATH = REPO_ROOT / "openspec/capabilities/continuous-learning/spec.md"


@pytest.fixture(scope="module")
def snapshot() -> dict[str, object]:
    """Capture the large immutable inputs once because tests never mutate it."""

    return exp.capture_input_snapshot(
        repo_root=REPO_ROOT,
        producer_artifact_path=PRODUCER_PATH,
        stream_artifact_path=STREAM_ARTIFACT_PATH,
    )


@pytest.fixture(scope="module")
def inputs(snapshot: dict[str, object]) -> dict[str, object]:
    """Decode detached evidence after the fixture has already captured its hashes."""

    return exp.decode_captured_inputs(snapshot)


@pytest.fixture(scope="module")
def artifact(snapshot: dict[str, object]) -> dict[str, object]:
    """Build one deterministic audit result for all read-only assertions."""

    return exp.audit_snapshot(
        snapshot,
        run_date=exp.RUN_DATE,
        duration_s=1.0,
        runtime_receipt=deepcopy(exp.TEST_RUNTIME_RECEIPT),
    )


def _first_row(source: dict[str, object], arm: str, *, committed: bool = False) -> int:
    """Find one row with the requested treatment and optional committed transition."""

    return next(
        index
        for index, row in enumerate(source["rows"])
        if row["arm"] == arm and (not committed or row["commit_record"]["committed"])
    )


def test_req_cl_7107_spec_and_required_field_contract() -> None:
    """REQ-CL-7107: the spec freezes the audit and attack surfaces first."""

    text = SPEC_PATH.read_text(encoding="utf-8").split("## REQ-CL-7107", 1)[1]
    for scenario in (
        "SCENARIO-CL-7107-PRECONDITIONS",
        "SCENARIO-CL-7107-RECONSTRUCTION",
        "SCENARIO-CL-7107-METRICS",
        "SCENARIO-CL-7107-POISON",
        "SCENARIO-CL-7107-ATOMICITY",
        "SCENARIO-CL-7107-ROLLBACK",
        "SCENARIO-CL-7107-CAPACITY",
        "SCENARIO-CL-7107-VERDICT",
    ):
        assert scenario in text
    assert exp.INFERENCE_SUBSTRATE == "fresh-process deterministic memory and transaction replay"
    assert exp.INFERENCE_SUBSTRATE_CLASS == "aggregation"
    assert exp.MODEL_WEIGHTS_CHANGED is False


def test_scenario_cl_7107_preconditions_pass_and_exact_failure_blocks(
    snapshot: dict[str, object], tmp_path: Path
) -> None:
    """SCENARIO-CL-7107-PRECONDITIONS: the producer completion gate runs first."""

    checks, decoded = exp.collect_preconditions(snapshot)
    assert decoded is not None
    assert all(row["passed"] for row in checks)

    changed = deepcopy(decoded["producer"])
    changed["procedural_memory_comparison_complete_score"] = 0
    producer = tmp_path / "producer.json"
    producer.write_text(json.dumps(changed), encoding="utf-8")
    failed_snapshot = exp.capture_input_snapshot(
        repo_root=REPO_ROOT,
        producer_artifact_path=producer,
        stream_artifact_path=STREAM_ARTIFACT_PATH,
    )
    blocked = exp.audit_snapshot(
        failed_snapshot,
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        runtime_receipt=deepcopy(exp.TEST_RUNTIME_RECEIPT),
    )
    summary = blocked["gate_check_summary"]
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["rows"] == []
    assert summary["failed_check"] == "procedural_memory_comparison_complete_score"
    assert summary["expected_value"] == 1
    assert summary["observed_value"] == 0
    assert exp.validate_artifact(blocked) == []


def test_scenario_cl_7107_reconstruction_matches_every_state_and_decision(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7107-RECONSTRUCTION: all empty-state replay rows match."""

    assert artifact["continual_memory_cold_audit_ready_score"] == 1
    assert len(artifact["reconstruction_rows"]) == 720
    assert len(artifact["snapshot_hash_rows"]) == 720
    assert len(artifact["event_replay_rows"]) == 144
    assert all(row["passed"] for row in artifact["reconstruction_rows"])
    assert all(row["passed"] for row in artifact["snapshot_hash_rows"])
    assert all(row["passed"] for row in artifact["event_replay_rows"])
    assert artifact["transaction_log_hash"].startswith("sha256:")
    assert exp.validate_artifact(artifact) == []


def test_scenario_cl_7107_reconstruction_rejects_missing_duplicate_and_reorder(
    inputs: dict[str, object],
) -> None:
    """SCENARIO-CL-7107-RECONSTRUCTION: event coverage and order are exact."""

    missing = deepcopy(inputs)
    missing["producer"]["rows"].pop()
    assert "missing_arm_event" in exp.replay_inputs(missing)["errors"]

    duplicate = deepcopy(inputs)
    duplicate["producer"]["rows"][-1] = deepcopy(duplicate["producer"]["rows"][0])
    assert "duplicate_arm_event" in exp.replay_inputs(duplicate)["errors"]

    reordered = deepcopy(inputs)
    reordered["producer"]["rows"][0], reordered["producer"]["rows"][5] = (
        reordered["producer"]["rows"][5],
        reordered["producer"]["rows"][0],
    )
    assert "event_reorder" in exp.replay_inputs(reordered)["errors"]


def test_scenario_cl_7107_reconstruction_rejects_snapshot_and_hash_substitution(
    inputs: dict[str, object],
) -> None:
    """SCENARIO-CL-7107-RECONSTRUCTION: claimed and active hashes cannot drift."""

    snapshot_attack = deepcopy(inputs)
    snapshot_attack["producer"]["memory_hash_rows"][0]["pre_event_memory_hash"] = (
        "sha256:" + "0" * 64
    )
    evidence = exp.replay_inputs(snapshot_attack)
    assert "memory_hash_projection_mismatch" in evidence["errors"]

    hash_attack = deepcopy(inputs)
    index = _first_row(hash_attack["producer"], "delayed_procedural", committed=True)
    hash_attack["producer"]["rows"][index]["post_commit_memory_hash"] = "sha256:" + "1" * 64
    evidence = exp.replay_inputs(hash_attack)
    assert "post_commit_hash_mismatch" in evidence["errors"]
    assert any(not row["passed"] for row in evidence["snapshot_hash_rows"])


def test_scenario_cl_7107_metrics_recompute_all_strata_and_paired_tests(
    artifact: dict[str, object], inputs: dict[str, object]
) -> None:
    """SCENARIO-CL-7107-METRICS: rows own aggregates and aligned tests."""

    names = {row["metric_table"] for row in artifact["metric_recomputation_rows"]}
    assert names == {
        "overall",
        "group_rows",
        "family_rows",
        "hardness_rows",
        "reuse_decoy_rows",
        "slice_rows",
        "capacity_rows",
        "negative_transfer_rows",
    }
    assert len(artifact["paired_test_rows"]) == 4
    assert all(row["event_count"] == 96 for row in artifact["paired_test_rows"])
    assert all(row["passed"] for row in artifact["producer_auditor_parity_rows"])

    aggregate = deepcopy(inputs)
    aggregate["producer"]["group_rows"][0]["correct_count"] += 1
    attacked = exp.replay_inputs(aggregate)
    assert "producer_aggregate_mismatch" in attacked["errors"]
    assert any(not row["passed"] for row in attacked["producer_auditor_parity_rows"])

    verdict = deepcopy(inputs)
    verdict["producer"]["verdict_class"] = "null"
    assert "producer_verdict_mismatch" in exp.replay_inputs(verdict)["errors"]


def test_scenario_cl_7107_poison_rejects_unsigned_outcome_and_witness(
    inputs: dict[str, object],
) -> None:
    """SCENARIO-CL-7107-POISON: feedback must be exact and verifier-signed."""

    unsigned = deepcopy(inputs)
    index = _first_row(unsigned["producer"], "delayed_procedural", committed=True)
    unsigned["producer"]["rows"][index]["validation_result"]["authority"] = None
    evidence = exp.replay_inputs(unsigned)
    assert "unsigned_feedback" in evidence["errors"]

    poisoned = deepcopy(inputs)
    row = poisoned["producer"]["rows"][index]
    row["exact_post_decision_label"] = next(
        candidate
        for candidate in row["candidate_ids"]
        if candidate != row["exact_post_decision_label"]
    )
    evidence = exp.replay_inputs(poisoned)
    assert "poisoned_outcome" in evidence["errors"]

    witness = deepcopy(inputs)
    witness["producer"]["rows"][index]["witness"]["authority"] = "untrusted"
    evidence = exp.replay_inputs(witness)
    assert "invalid_witness" in evidence["errors"]


def test_scenario_cl_7107_atomicity_attack_matrix_restores_safe_boundaries(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7107-ATOMICITY: partial writes and crashes recover exactly."""

    expected = {
        "partial_prepare",
        "partial_commit",
        "truncated_write",
        "duplicate_commit",
    }
    assert {row["attack_id"] for row in artifact["partial_write_rows"]} == expected
    assert all(row["passed"] for row in artifact["partial_write_rows"])
    assert {row["attack_id"] for row in artifact["crash_recovery_rows"]} == {
        "crash_before_commit",
        "crash_after_commit",
    }
    assert all(row["passed"] for row in artifact["crash_recovery_rows"])
    assert all(
        row["restored_hash"] in {row["parent_hash"], row["child_hash"]}
        for row in artifact["crash_recovery_rows"]
    )


def test_scenario_cl_7107_rollback_and_stale_parent_fail_closed(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7107-ROLLBACK: adverse and drifted inverses preserve exact bytes."""

    assert {row["attack_id"] for row in artifact["stale_parent_rows"]} == {"stale_parent_hash"}
    assert all(row["passed"] for row in artifact["stale_parent_rows"])
    assert {row["attack_id"] for row in artifact["rollback_rows"]} == {
        "adverse_commit_rollback",
        "rollback_drift",
    }
    assert all(row["passed"] for row in artifact["rollback_rows"])
    assert all(row["restored_hash"] == row["parent_hash"] for row in artifact["rollback_rows"])


def test_scenario_cl_7107_capacity_retention_isolation_and_signatures(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7107-CAPACITY: bounds, retention, timing, and signatures pass."""

    assert len(artifact["capacity_rows"]) == 15
    assert len(artifact["eviction_rows"]) == 720
    assert len(artifact["protected_retention_rows"]) == 20
    assert all(row["passed"] for row in artifact["eviction_rows"])
    assert all(row["passed"] for row in artifact["future_label_isolation_rows"])
    assert all(row["passed"] for row in artifact["signature_rows"])
    delayed = [
        row for row in artifact["protected_retention_rows"] if row["arm"] == "delayed_procedural"
    ]
    assert all(row["retention_passed"] for row in delayed)
    capacity_attack = next(
        row for row in artifact["mutation_attack_rows"] if row["attack_id"] == "capacity_overflow"
    )
    assert capacity_attack["passed"] is True
    assert capacity_attack["restored_prior_hash"] is True


def test_scenario_cl_7107_reorder_poison_and_mutation_attacks_are_complete(
    artifact: dict[str, object],
) -> None:
    """REQ-CL-7107: every declared source and store attack is exercised."""

    assert {row["attack_id"] for row in artifact["poison_attack_rows"]} == {
        "unsigned_feedback",
        "poisoned_outcome",
        "invalid_witness",
        "poisoned_witness",
    }
    assert {row["attack_id"] for row in artifact["reorder_attack_rows"]} == {
        "event_reorder",
        "order_sensitivity",
    }
    required = {
        "missing_event",
        "duplicate_event",
        "snapshot_substitution",
        "hash_mutation",
        "capacity_overflow",
        "aggregate_mismatch",
        "verdict_mismatch",
    }
    assert required.issubset({row["attack_id"] for row in artifact["mutation_attack_rows"]})
    for field in (
        "poison_attack_rows",
        "reorder_attack_rows",
        "stale_parent_rows",
        "partial_write_rows",
        "crash_recovery_rows",
        "rollback_rows",
        "mutation_attack_rows",
    ):
        assert all(row["passed"] for row in artifact[field])


def test_scenario_cl_7107_verdict_separates_audit_readiness_from_value(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7107-VERDICT: a safe audit can confirm an upstream null."""

    changed = deepcopy(artifact)
    changed["upstream_gate_receipt"]["procedural_memory_value_ready_score"] = 0
    changed["upstream_gate_receipt"]["upstream_verdict_class"] = "null"
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert changed["continual_memory_cold_audit_ready_score"] == 1
    assert changed["verdict_class"] == "positive"
    assert exp.validate_artifact(changed) == []

    mismatch = deepcopy(artifact)
    mismatch["verdict_class"] = "null"
    assert "verdict_class_mismatch" in exp.validate_artifact(mismatch)
    headline = deepcopy(artifact)
    headline["honest_verdict"] = "complete_null_continual_memory_cold_audit"
    assert "honest_verdict_mismatch" in exp.validate_artifact(headline)


def test_req_cl_7107_artifact_schema_runtime_and_tamper_validation(
    artifact: dict[str, object],
) -> None:
    """REQ-CL-7107: complete provenance and runtime isolation remain auditable."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["model_weights_changed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert all(artifact["runtime_isolation_receipt"].values())

    missing = deepcopy(artifact)
    missing.pop("schema")
    assert exp.validate_artifact(missing)[0].startswith("missing_fields:")
    changed = deepcopy(artifact)
    changed["metric_recomputation_rows"][0]["accuracy"] = -1
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


def test_req_cl_7107_command_writes_only_requested_artifact(tmp_path: Path) -> None:
    """REQ-CL-7107: the public command runs the audit in a fresh process."""

    output = tmp_path / "experiment-7107.json"
    assert exp.main(["--date", exp.RUN_DATE, "--artifact-path", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["continual_memory_cold_audit_ready_score"] == 1
    assert written["runtime_isolation_receipt"]["fresh_process"] is True
    assert written["runtime_isolation_receipt"]["llm_disabled"] is True
    assert exp.validate_artifact(written) == []


def test_req_cl_7107_defensive_decode_worker_and_atomic_cleanup(
    snapshot: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7107: malformed inputs and failed publication cannot look complete."""

    malformed = deepcopy(snapshot)
    malformed["files"]["producer_artifact"]["bytes"] = b"[]"
    with pytest.raises(ValueError, match="JSON object"):
        exp.decode_captured_inputs(malformed)
    malformed_json = deepcopy(snapshot)
    malformed_json["files"]["producer_artifact"]["bytes"] = b"{"
    malformed_checks, malformed_inputs = exp.collect_preconditions(malformed_json)
    assert malformed_inputs is None
    assert malformed_checks[-1]["check"] == "input_json_decoding"
    assert malformed_checks[-1]["observed_value"] == "JSONDecodeError"
    unreadable = exp.capture_input_snapshot(
        repo_root=REPO_ROOT,
        producer_artifact_path=tmp_path / "missing.json",
        stream_artifact_path=STREAM_ARTIFACT_PATH,
    )
    checks, decoded = exp.collect_preconditions(unreadable)
    assert decoded is None
    assert checks[0]["passed"] is False

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "none")
    args = exp.parse_args(["--worker", "--date", exp.RUN_DATE])
    worker = exp.worker_artifact(args)
    assert worker["continual_memory_cold_audit_ready_score"] == 1

    output = tmp_path / "atomic.json"
    monkeypatch.setattr(exp.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError, match="stop"):
        exp.write_artifact(output, worker)
    assert list(tmp_path.glob(".atomic.json.*")) == []


def test_req_cl_7107_defensive_stream_and_replay_gates(
    inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7107: every defensive immutable-row rejection remains executable."""

    assert exp._jsonl_bytes([{"a": 1}]) == b'{"a":1}\n'
    with pytest.raises(ValueError, match="not readable"):
        exp._decode_object({"bytes": None}, "object")
    with pytest.raises(ValueError, match="not readable"):
        exp._decode_jsonl({"bytes": None}, "rows")
    with pytest.raises(ValueError, match="non-object"):
        exp._decode_jsonl({"bytes": b"[]\n"}, "rows")
    assert exp._sign_p(0, 0) == 1.0
    with pytest.raises(ValueError, match="positive"):
        exp._apply_record([], {"memory_key": "x", "memory_id": "x"}, 0)

    attacks: list[tuple[str, str, object]] = []
    missing = deepcopy(inputs)
    missing["stream"].pop()
    attacks.append(("sealed_record_count_mismatch", "count", missing))
    substituted = deepcopy(inputs)
    substituted["stream_artifact"]["event_rows"][0]["event_id"] = "substituted"
    attacks.append(("sealed_stream_artifact_mismatch", "artifact", substituted))
    reordered = deepcopy(inputs)
    reordered["stream"][0], reordered["stream"][1] = reordered["stream"][1], reordered["stream"][0]
    attacks.append(("sealed_stream_reorder", "order", reordered))
    duplicate = deepcopy(inputs)
    duplicate["labels"][-1] = deepcopy(duplicate["labels"][0])
    attacks.append(("duplicate_feedback_event", "duplicate", duplicate))
    decision = deepcopy(inputs)
    decision["decisions"][0]["decision_content_hash"] = "sha256:" + "0" * 64
    attacks.append(("decision_content_hash_mismatch", "decision", decision))
    label = deepcopy(inputs)
    label["labels"][0]["label_content_hash"] = "sha256:" + "0" * 64
    attacks.append(("label_content_hash_mismatch", "label", label))
    content = deepcopy(inputs)
    content["stream"][0]["canonical_content_hash"] = "sha256:" + "0" * 64
    attacks.append(("event_content_hash_mismatch", "content", content))
    malformed = deepcopy(inputs)
    malformed["stream"][0].pop("event_id")
    attacks.append(("invalid_sealed_record", "malformed", malformed))
    for expected, _name, attacked in attacks:
        assert expected in exp._stream_contract_errors(attacked)

    replay = deepcopy(inputs)
    row = replay["producer"]["rows"][0]
    row.update(
        {
            "event_id": "wrong-event",
            "visible_input_hash": "wrong-visible",
            "candidate_set_hash": "wrong-candidates",
            "pre_event_memory_hash": "wrong-pre-event",
            "pre_decision_memory_hash": "wrong-pre-decision",
            "retrieved_items": [],
            "decision": {},
            "decision_seal": "wrong-seal",
            "future_label_accessed": True,
            "proposed_update": {},
            "commit_record": {},
            "evicted_ids": ["wrong-eviction"],
            "eviction_policy": "unsafe",
            "capacity_items": 0,
        }
    )
    row["validation_result"] = {}
    replay["producer"]["model_weights_changed"] = True
    found = set(exp.replay_inputs(replay)["errors"])
    assert {
        "event_identity_mismatch",
        "visible_input_hash_mismatch",
        "candidate_set_hash_mismatch",
        "pre_event_hash_mismatch",
        "pre_decision_hash_mismatch",
        "retrieval_mismatch",
        "decision_replay_mismatch",
        "future_label_isolation_failed",
        "validation_record_mismatch",
        "proposed_update_mismatch",
        "capacity_overflow",
        "transaction_record_mismatch",
        "eviction_mismatch",
        "capacity_or_policy_mismatch",
        "model_weights_changed",
    }.issubset(found)

    missing_decision = deepcopy(inputs)
    missing_decision["decisions"].pop(0)
    assert "missing_decision_event" in exp.replay_inputs(missing_decision)["errors"]

    original_apply = exp._apply_record

    def permit_impossible_overflow(records, record, capacity):
        if record.get("memory_key") == "huge":
            return [], []
        return original_apply(records, record, capacity)

    monkeypatch.setattr(exp, "_apply_record", permit_impossible_overflow)
    assert any(
        row["attack_id"] == "capacity_overflow" and row["passed"] is False
        for row in exp.run_attack_matrix(inputs)["mutation_attack_rows"]
    )


def test_req_cl_7107_null_validation_cli_and_runtime_error_paths(
    snapshot: dict[str, object],
    artifact: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CL-7107: null, validator, guard, and CLI failures stay terminal."""

    bad_runtime = deepcopy(exp.TEST_RUNTIME_RECEIPT)
    bad_runtime["llm_disabled"] = False
    null = exp.audit_snapshot(
        snapshot,
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        runtime_receipt=bad_runtime,
    )
    assert null["verdict_class"] == "null"
    assert exp.validate_artifact(null) == []

    invalid = deepcopy(artifact)
    invalid.update(
        {
            "field_principles": {},
            "schema": "wrong",
            "experiment_id": 0,
            "inference_substrate": "wrong",
            "inference_substrate_class": "wrong",
            "execution_venue": "wrong",
            "model_weights_changed": True,
            "verifier_is_oracle": True,
            "continual_memory_cold_audit_ready_score": 7,
        }
    )
    invalid_errors = set(exp.validate_artifact(invalid))
    assert {
        "field_principles_mismatch",
        "identity_mismatch",
        "inference_substrate_mismatch",
        "inference_substrate_class_mismatch",
        "execution_venue_mismatch",
        "model_weights_changed",
        "verifier_oracle_mismatch",
        "ready_score_not_bare_integer",
        "ready_score_mismatch",
    }.issubset(invalid_errors)

    blocked = deepcopy(artifact)
    blocked["inference_substrate_class"] = "blocked_no_run"
    blocked["continual_memory_cold_audit_ready_score"] = 0
    blocked["verdict_class"] = "blocked"
    blocked["honest_verdict"] = "complete_blocked_invalid"
    blocked["gate_check_summary"]["failed_check"] = None
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    assert "blocked_evidence_mismatch" in exp.validate_artifact(blocked)

    guard = exp.make_runtime_guard([tmp_path / "protected"])
    with pytest.raises(PermissionError, match="network disabled"):
        guard("socket.connect", ())
    guard("open", (object(), "r", 0))
    guard("open", (str(tmp_path / "safe"), "r", 0))
    with pytest.raises(PermissionError, match="read-only"):
        guard("open", (str(tmp_path / "protected"), "w", 0))

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="worker-stop"),
    )
    with pytest.raises(RuntimeError, match="worker-stop"):
        exp._spawn_worker(exp.parse_args([]))
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr="", stdout="[]"),
    )
    with pytest.raises(RuntimeError, match="JSON object"):
        exp._spawn_worker(exp.parse_args([]))

    monkeypatch.setattr(exp, "worker_artifact", lambda _args: {"worker": True})
    assert exp.main(["--worker"]) == 0
    assert json.loads(capsys.readouterr().out) == {"worker": True}
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--validate", "--artifact-path", str(valid_path)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{", encoding="utf-8")
    assert exp.main(["--validate", "--artifact-path", str(malformed_path)]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    monkeypatch.setattr(exp, "_spawn_worker", lambda _args: {})
    with pytest.raises(RuntimeError, match="validation failed"):
        exp.main(["--artifact-path", str(tmp_path / "never.json")])
