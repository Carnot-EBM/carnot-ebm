"""Focused tests for the read-only Qwen3.8 runtime preflight.

Spec refs: REQ-HARNESS-7160 and SCENARIO-HARNESS-7160-*.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from carnot import experiment_7160_v631_qwen38_lease_diagnosis as exp


MODEL_HASH = "sha256:" + "3" * 64
GPU_UUID = "GPU-test-3090"


def _gpu(*, pid: int | None = None, proc_exists: bool = True) -> dict:
    row = {
        "gpu_index": 0,
        "gpu_uuid": GPU_UUID,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "gpu_utilization_pct": 0,
        "gpu_memory_total_mb": 24576,
        "gpu_memory_used_mb": 4 if pid is None else 16000,
        "gpu_memory_free_mb": 24572 if pid is None else 8576,
        "pid": pid,
        "gpu_process_memory_mb": 0 if pid is None else 15996,
        "proc_exists": proc_exists if pid is not None else None,
        "ppid": 101 if pid is not None and proc_exists else None,
        "process_group_id": pid if pid is not None and proc_exists else None,
        "session_id": pid if pid is not None and proc_exists else None,
        "start_time_ticks": 12345 if pid is not None and proc_exists else None,
        "process_start_utc": "2026-09-09T12:00:00Z" if pid is not None else None,
        "age_s": 600.0 if pid is not None else None,
        "command": [
            "/opt/llama-server",
            "--model",
            "/cache/Qwen3.8-27B-Q4_K_M.gguf",
            "--port",
            "8919",
        ]
        if pid is not None and proc_exists
        else [],
        "command_text": (
            "/opt/llama-server --model /cache/Qwen3.8-27B-Q4_K_M.gguf --port 8919"
            if pid is not None and proc_exists
            else ""
        ),
        "command_sha256": "sha256:command" if pid is not None and proc_exists else None,
        "open_port": 8919 if pid is not None and proc_exists else None,
        "model_path": (
            "/cache/Qwen3.8-27B-Q4_K_M.gguf" if pid is not None and proc_exists else None
        ),
        "model_sha256": MODEL_HASH if pid is not None and proc_exists else None,
        "ownership_classification": "pending" if pid is not None else "idle",
        "matching_lease_id": None,
        "ownership_evidence_errors": [],
    }
    return row


def _lease(
    *,
    pid: int = 500,
    task_id: str = exp.TASK_ID,
    fresh: bool = True,
    owner_live: bool = True,
    released: bool = False,
) -> dict:
    return {
        "lease_path": "/tmp/carnot-gpu-leases/device-test.journal.json",
        "readable": True,
        "canonical": True,
        "schema": exp.LEASE_SCHEMA,
        "checksum_valid": True,
        "lease_id": "lease:test",
        "task_id": task_id,
        "device_uuid": GPU_UUID,
        "owner_pid": pid,
        "owner_start_ticks": 12345,
        "owner_executable": "/usr/bin/python3",
        "owner_argv_digest": "sha256:owner-command",
        "expected_model": "/cache/Qwen3.8-27B-Q4_K_M.gguf",
        "port": 8919,
        "model_sha256": MODEL_HASH,
        "phase": "resident",
        "released": released,
        "expires_monotonic_ns": 99_000,
        "fresh": fresh,
        "owner_live": owner_live,
        "signals_sent": [],
        "error": None,
    }


def _cache(*, valid: bool = True) -> list[dict]:
    return [
        {
            "repository": exp.QWEN_MODEL_ID,
            "filename": exp.QWEN_FILENAME if valid else None,
            "path": "/cache/Qwen3.8-27B-Q4_K_M.gguf" if valid else None,
            "real_path": "/cache/blobs/" + "3" * 64 if valid else None,
            "revision": "revision-test" if valid else None,
            "bytes": 16_000_000_000 if valid else None,
            "sha256": MODEL_HASH if valid else None,
            "hash_source": "content_addressed_cache_target" if valid else None,
            "weights_opened": False,
            "valid": valid,
        }
    ]


def _runner(*, valid: bool = True) -> list[dict]:
    return [
        {
            "runner_path": "/opt/llama-server",
            "exists": valid,
            "executable": valid,
            "version": "llama.cpp test",
            "version_check_ok": valid,
            "help_check_ok": valid,
            "cuda_linkage_confirmed": valid,
            "task_owned_process_groups": valid,
            "bounded_token_generation": valid,
            "grammar_or_json_output": valid,
            "owned_teardown": valid,
            "model_argument_present": False,
            "valid": valid,
            "command_receipts": [],
        }
    ]


def _checks() -> list[dict]:
    return [exp.gate_row("diagnostic_dependencies", True, True, True)]


def _ready_artifact() -> dict:
    artifact = exp.base_artifact(exp.RUN_DATE)
    gpu_rows = exp.classify_process_rows([_gpu()], [], current_task_id=exp.TASK_ID)
    return exp.finalize_artifact(
        artifact,
        checks=_checks(),
        gpu_process_rows=gpu_rows,
        lease_ownership_rows=[],
        cache_identity_rows=_cache(),
        runner_capability_rows=_runner(),
        stop_authority_receipt={
            "marker_path": "/home/test/.carnot/stop-authority-armed",
            "marker_present": False,
            "state": "disarmed",
            "signals_sent": [],
            "actions_taken": [],
        },
        duration_s=0.5,
    )


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-IDLE.
def test_idle_gpu_is_a_named_ready_allocation() -> None:
    rows = exp.classify_process_rows([_gpu()], [], current_task_id=exp.TASK_ID)
    decision = exp.readiness_decision(rows, [], _cache(), _runner())

    assert decision["score"] == 1
    assert decision["available_gpu_uuids"] == [GPU_UUID]
    assert decision["conflicting_processes"] == []


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-OWNED.
def test_complete_task_lease_classifies_the_process_as_owned() -> None:
    rows = exp.classify_process_rows([_gpu(pid=500)], [_lease()], current_task_id=exp.TASK_ID)

    assert rows[0]["ownership_classification"] == "owned"
    assert rows[0]["matching_lease_id"] == "lease:test"
    assert rows[0]["ownership_evidence_errors"] == []


# REQ-HARNESS-7160: another current exact owner is adoptable, but is not adopted here.
def test_complete_foreign_lease_classifies_the_process_as_adoptable() -> None:
    rows = exp.classify_process_rows(
        [_gpu(pid=500)], [_lease(task_id="another-live-task")], current_task_id=exp.TASK_ID
    )

    assert rows[0]["ownership_classification"] == "adoptable"
    assert rows[0]["matching_lease_id"] == "lease:test"


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-UNOWNED.
def test_old_unowned_server_is_a_conflict_not_a_name_guess() -> None:
    old = _gpu(pid=233772)
    old["command"] = [
        "/opt/llama-server",
        "--model",
        "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "--port",
        "8919",
    ]
    old["model_path"] = "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    old["model_sha256"] = "sha256:" + "6" * 64

    rows = exp.classify_process_rows([old], [], current_task_id=exp.TASK_ID)
    decision = exp.readiness_decision(rows, [], _cache(), _runner())

    assert rows[0]["ownership_classification"] == "conflicting"
    assert "current_canonical_lease_missing" in rows[0]["ownership_evidence_errors"]
    assert decision["score"] == 0
    assert decision["conflicting_processes"] == [
        {"pid": 233772, "gpu_uuid": GPU_UUID, "memory_mb": 15996}
    ]


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-STALE.
def test_stale_lease_cannot_make_a_live_process_adoptable() -> None:
    rows = exp.classify_process_rows(
        [_gpu(pid=500)], [_lease(fresh=False)], current_task_id=exp.TASK_ID
    )

    assert rows[0]["ownership_classification"] == "conflicting"
    assert "lease_not_fresh" in rows[0]["ownership_evidence_errors"]


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-PROC-RACE.
def test_missing_proc_entry_retains_pid_and_conflict() -> None:
    row = _gpu(pid=500, proc_exists=False)
    rows = exp.classify_process_rows([row], [_lease()], current_task_id=exp.TASK_ID)

    assert rows[0]["pid"] == 500
    assert rows[0]["proc_exists"] is False
    assert rows[0]["ownership_classification"] == "conflicting"
    assert "process_identity_missing" in rows[0]["ownership_evidence_errors"]


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-CACHE.
def test_missing_cache_blocks_readiness_without_process_work() -> None:
    rows = exp.classify_process_rows([_gpu()], [], current_task_id=exp.TASK_ID)
    decision = exp.readiness_decision(rows, [], _cache(valid=False), _runner())

    assert decision["score"] == 0
    assert decision["failed_check"] == "cached_qwen38_q4"


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-OWNED.
def test_deleting_pid_ownership_evidence_changes_readiness() -> None:
    process = _gpu(pid=500)
    lease = _lease()
    owned = exp.classify_process_rows([process], [lease], current_task_id=exp.TASK_ID)
    owned_decision = exp.readiness_decision(owned, [lease], _cache(), _runner())

    mutated_lease = deepcopy(lease)
    del mutated_lease["owner_pid"]
    mutated = exp.classify_process_rows([process], [mutated_lease], current_task_id=exp.TASK_ID)
    mutated_decision = exp.readiness_decision(mutated, [mutated_lease], _cache(), _runner())

    assert owned_decision["score"] == 1
    assert mutated_decision["score"] == 0
    assert mutated[0]["ownership_classification"] == "conflicting"
    assert "lease_owner_pid_mismatch" in mutated[0]["ownership_evidence_errors"]


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-CACHE.
def test_cache_identity_uses_only_exact_content_addressed_target(tmp_path: Path) -> None:
    model_dir = tmp_path / "models--unsloth--Qwen3.8-27B-GGUF"
    blob = model_dir / "blobs" / ("a" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"not model weights")
    snapshot = model_dir / "snapshots" / "revision-38"
    snapshot.mkdir(parents=True)
    model = snapshot / exp.QWEN_FILENAME
    model.symlink_to(Path("../../blobs") / blob.name)

    rows = exp.resolve_cache_identity(
        resolver=lambda hf_id, quant: str(model),
    )

    assert rows[0]["repository"] == exp.QWEN_MODEL_ID
    assert rows[0]["filename"] == exp.QWEN_FILENAME
    assert rows[0]["revision"] == "revision-38"
    assert rows[0]["bytes"] == len(b"not model weights")
    assert rows[0]["sha256"] == "sha256:" + "a" * 64
    assert rows[0]["weights_opened"] is False
    assert rows[0]["valid"] is True


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-ARTIFACT.
def test_terminal_artifact_is_schema_complete_and_cold_valid() -> None:
    result = _ready_artifact()

    assert set(result) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert result["field_principles"] == exp.FIELD_PRINCIPLES
    assert result["status"] == "completed"
    assert result["inference_substrate_class"] == "no_model_load"
    assert result["qwen38_runtime_preflight_ready_score"] == 1
    assert result["honest_verdict"] == "complete_positive_qwen38_runtime_preflight_ready"
    assert exp.validate_artifact(result) == []


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-ARTIFACT.
def test_cold_validator_detects_ownership_and_row_mutations() -> None:
    artifact = exp.base_artifact(exp.RUN_DATE)
    lease = _lease()
    gpu_rows = exp.classify_process_rows([_gpu(pid=500)], [lease], current_task_id=exp.TASK_ID)
    result = exp.finalize_artifact(
        artifact,
        checks=_checks(),
        gpu_process_rows=gpu_rows,
        lease_ownership_rows=[lease],
        cache_identity_rows=_cache(),
        runner_capability_rows=_runner(),
        stop_authority_receipt={
            "marker_path": "/marker",
            "marker_present": False,
            "state": "disarmed",
            "signals_sent": [],
            "actions_taken": [],
        },
        duration_s=0.5,
    )
    result["lease_ownership_rows"][0].pop("owner_pid")
    result["reproducibility_checksum"] = exp.artifact_checksum(result)

    errors = exp.validate_artifact(result)

    assert "process_classification_mismatch" in errors
    assert "readiness_score_mismatch" in errors
    assert "typed_rows_mismatch" in errors


# REQ-HARNESS-7160: missing diagnostic dependencies stop before a model load.
def test_missing_diagnostic_dependency_uses_blocked_no_run() -> None:
    artifact = exp.finalize_artifact(
        exp.base_artifact(exp.RUN_DATE),
        checks=[exp.gate_row("nvidia_smi", True, False, False)],
        gpu_process_rows=[],
        lease_ownership_rows=[],
        cache_identity_rows=[],
        runner_capability_rows=[],
        stop_authority_receipt={},
        duration_s=0.1,
    )

    assert artifact["status"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["check"] == "nvidia_smi"
    assert artifact["honest_verdict"] == "blocked_nvidia_smi"
    assert exp.validate_artifact(artifact) == []


# REQ-HARNESS-7160: secret argument values are never copied to evidence.
def test_command_redaction_keeps_nonsecret_arguments() -> None:
    command = ["llama-server", "--api-key", "secret-value", "--port=8919", "plain"]

    redacted = exp.redact_command(command)

    assert redacted == ["llama-server", "--api-key", "<redacted>", "--port=8919", "plain"]
    assert "secret-value" not in " ".join(redacted)


# REQ-HARNESS-7160: metadata and artifact helpers stay read-only and fail closed.
def test_read_only_helpers_cover_paths_redaction_and_marker_state(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    ready = _ready_artifact()
    exp.write_artifact(target, ready)
    assert exp.validate_artifact(target) == []

    assert exp.redact_command(["tool", "--token=value"])[1] == "--token=<redacted>"
    assert exp.redact_command(["https://user:pass@example.test/path"])[0] == (
        "https://<redacted>@example.test/path"
    )
    assert exp._command_value(["tool", "--model", "one.gguf"], {"--model"}) == "one.gguf"
    assert exp._command_value(["tool", "--port=8919"], {"--port"}) == "8919"
    assert exp._command_value(["tool"], {"--model"}) is None
    assert exp._content_addressed_model_hash(tmp_path / "missing") is None
    ordinary = tmp_path / "ordinary-name"
    ordinary.write_bytes(b"metadata-only fixture")
    assert exp._content_addressed_model_hash(ordinary) is None

    marker = tmp_path / "stop-authority-armed"
    assert exp.stop_authority_receipt(marker)["state"] == "disarmed"
    marker.touch()
    armed = exp.stop_authority_receipt(marker)
    assert armed["state"] == "armed"
    assert armed["signals_sent"] == []


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-ARTIFACT.
def test_cold_validator_reports_schema_and_terminal_corruption(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    unreadable = tmp_path / "unreadable.json"
    nonobject = tmp_path / "nonobject.json"
    unreadable.write_text("{", encoding="utf-8")
    nonobject.write_text("[]", encoding="utf-8")

    assert exp.validate_artifact(missing) == ["artifact_unreadable_or_not_object"]
    assert exp.validate_artifact(unreadable) == ["artifact_unreadable_or_not_object"]
    assert exp.validate_artifact(nonobject) == ["artifact_unreadable_or_not_object"]
    assert exp.validate_artifact(7) == ["artifact_unreadable_or_not_object"]
    assert exp.validate_artifact({"unexpected": True}) == ["artifact_fields_mismatch"]

    corrupt = _ready_artifact()
    corrupt.update(
        {
            "field_principles": {},
            "run_date": "20260908",
            "inference_substrate": "model_load",
            "inference_substrate_class": "blocked_no_run",
            "execution_venue": [],
            "duration_s": -1,
            "gate_check_summary": {},
            "verifier_is_oracle": True,
            "reproducibility_checksum": "wrong",
        }
    )
    corrupt["stop_authority_receipt"]["signals_sent"] = ["SIGTERM"]

    assert {
        "field_principles_mismatch",
        "run_date_mismatch",
        "inference_substrate_mismatch",
        "inference_substrate_class_mismatch",
        "execution_venue_invalid",
        "duration_invalid",
        "gate_check_summary_mismatch",
        "verifier_is_oracle",
        "read_only_contract_violated",
        "reproducibility_checksum_mismatch",
    }.issubset(exp.validate_artifact(corrupt))
