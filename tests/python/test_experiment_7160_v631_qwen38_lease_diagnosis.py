"""Focused tests for the read-only Qwen3.8 runtime preflight.

Spec refs: REQ-HARNESS-7160 and SCENARIO-HARNESS-7160-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7160_v631_qwen38_lease_diagnosis as exp


MODEL_HASH = "sha256:" + "3" * 64
GPU_UUID = "GPU-test-3090"
MODEL_PATH = (
    "/cache/models--unsloth--Qwen3.8-27B-GGUF/snapshots/revision-test/Qwen3.8-27B-Q4_K_M.gguf"
)


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
            MODEL_PATH,
            "--port",
            "8919",
        ]
        if pid is not None and proc_exists
        else [],
        "command_text": (
            f"/opt/llama-server --model {MODEL_PATH} --port 8919"
            if pid is not None and proc_exists
            else ""
        ),
        "command_sha256": "sha256:command" if pid is not None and proc_exists else None,
        "open_port": 8919 if pid is not None and proc_exists else None,
        "model_path": (MODEL_PATH if pid is not None and proc_exists else None),
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
        "expected_model": MODEL_PATH,
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
            "path": MODEL_PATH if valid else None,
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
    runner_path = "/opt/llama-server"
    return [
        {
            "runner_path": runner_path,
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
            "command_receipts": [
                {
                    "command": [runner_path, "--version"],
                    "returncode": 0 if valid else 127,
                    "stdout": "llama.cpp test" if valid else "",
                    "stderr": "",
                    "duration_s": 0.01,
                },
                {
                    "command": [runner_path, "--help"],
                    "returncode": 0 if valid else 127,
                    "stdout": "--n-predict --grammar" if valid else "",
                    "stderr": "",
                    "duration_s": 0.01,
                },
                {
                    "command": ["ldd", runner_path],
                    "returncode": 0 if valid else 127,
                    "stdout": "libggml-cuda.so => /lib\nlibcuda.so => /lib" if valid else "",
                    "stderr": "",
                    "duration_s": 0.01,
                },
            ],
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


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-OWNED.
def test_expected_model_path_mismatch_cannot_grant_ownership() -> None:
    lease = _lease()
    lease["expected_model"] = "/cache/another-model.gguf"

    rows = exp.classify_process_rows([_gpu(pid=500)], [lease], current_task_id=exp.TASK_ID)

    assert rows[0]["ownership_classification"] == "conflicting"
    assert "lease_expected_model_mismatch" in rows[0]["ownership_evidence_errors"]


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


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-ARTIFACT.
def test_cold_validator_recomputes_cache_identity_instead_of_trusting_valid_flag() -> None:
    result = _ready_artifact()
    result["cache_identity_rows"][0]["repository"] = "unsloth/substituted-GGUF"
    result["rows"] = exp.typed_rows(result)
    result["reproducibility_checksum"] = exp.artifact_checksum(result)

    errors = exp.validate_artifact(result)

    assert "cache_identity_mismatch" in errors
    assert "readiness_score_mismatch" in errors


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-ARTIFACT.
def test_cold_validator_recomputes_runner_capabilities_instead_of_trusting_valid_flag() -> None:
    result = _ready_artifact()
    result["runner_capability_rows"][0]["grammar_or_json_output"] = False
    result["rows"] = exp.typed_rows(result)
    result["reproducibility_checksum"] = exp.artifact_checksum(result)

    errors = exp.validate_artifact(result)

    assert "runner_capability_mismatch" in errors
    assert "readiness_score_mismatch" in errors


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-UNOWNED.
def test_blocked_idle_verdict_names_each_conflicting_pid_and_gpu_memory() -> None:
    result = exp.finalize_artifact(
        exp.base_artifact(exp.RUN_DATE),
        checks=_checks(),
        gpu_process_rows=[_gpu(pid=233772)],
        lease_ownership_rows=[],
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

    assert result["status"] == "blocked"
    assert result["honest_verdict"] == "blocked_idle_rtx_3090"
    assert result["gate_check_summary"]["observed_value"]["conflicting_processes"] == [
        {"pid": 233772, "gpu_uuid": GPU_UUID, "memory_mb": 15996}
    ]
    assert exp.validate_artifact(result) == []


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


# REQ-HARNESS-7160: bounded subprocess receipts preserve success and exact failure evidence.
def test_subprocess_boundary_records_success_and_os_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="version output", stderr=""),
    )
    success = exp._run_subprocess(["runner", "--version"], phase=5)

    def raise_os_error(*args: object, **kwargs: object) -> None:
        raise OSError("missing runner")

    monkeypatch.setattr(exp.subprocess, "run", raise_os_error)
    failure = exp._run_subprocess(["missing-runner", "--help"], phase=5)

    assert success["returncode"] == 0
    assert success["stdout"] == "version output"
    assert failure["returncode"] == 127
    assert "OSError: missing runner" in failure["stderr"]
    assert "subprocess_start" in capsys.readouterr().out


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-PROC-RACE.
def test_procfs_helpers_record_ports_identity_and_a_disappeared_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    process_dir = proc_root / "500"
    descriptor_dir = process_dir / "fd"
    descriptor_dir.mkdir(parents=True)
    os.symlink("socket:[42]", descriptor_dir / "9")
    (descriptor_dir / "not-a-link").write_text("fixture", encoding="utf-8")
    net_dir = proc_root / "net"
    net_dir.mkdir()
    tcp_header = "sl local_address rem_address st tx_queue tr tm retrnsmt uid timeout inode"
    tcp_row = "0: 0100007F:22D7 00000000:0000 0A 0 0 0 1000 0 42"
    (net_dir / "tcp").write_text(f"{tcp_header}\n{tcp_row}\n", encoding="utf-8")

    assert exp._listening_ports(500, proc_root) == [8919]
    assert exp._listening_ports(999, proc_root) == []

    blob = tmp_path / ("3" * 64)
    blob.write_bytes(b"metadata-only fixture")
    model = tmp_path / exp.QWEN_FILENAME
    model.symlink_to(blob.name)
    stat_rest = ["S", "101", "500", "500", *(["0"] * 15), "12345"]
    (process_dir / "stat").write_text(
        f"500 (llama server) {' '.join(stat_rest)}\n", encoding="utf-8"
    )
    (process_dir / "cmdline").write_bytes(
        b"/opt/llama-server\x00--model\x00" + str(model).encode() + b"\x00--port\x008919\x00"
    )
    (proc_root / "uptime").write_text("1000.0 0.0\n", encoding="utf-8")
    monkeypatch.setattr(exp.os, "sysconf", lambda name: 100)
    monkeypatch.setattr(exp.time, "time", lambda: 2_000_000_000.0)
    monkeypatch.setattr(exp, "_listening_ports", lambda pid, root: [8919])

    present = exp._read_proc_identity(500, proc_root=proc_root)
    missing = exp._read_proc_identity(999, proc_root=proc_root)

    assert present["proc_exists"] is True
    assert present["ppid"] == 101
    assert present["session_id"] == 500
    assert present["open_port"] == 8919
    assert present["model_sha256"] == MODEL_HASH
    assert missing["proc_exists"] is False
    assert missing["proc_error"].startswith("FileNotFoundError:")


# REQ-HARNESS-7160: every NVIDIA compute PID and every lease journal becomes a row.
def test_gpu_and_lease_collectors_preserve_live_and_malformed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_subprocess(command: list[str], **kwargs: object) -> dict:
        if any("query-gpu" in item for item in command):
            stdout = (
                "0, GPU-test-3090, NVIDIA GeForce RTX 3090, 0, 24576, 16000, 8576\n"
                "1, GPU-idle-3090, NVIDIA GeForce RTX 3090, 0, 24576, 4, 24572\n"
            )
        else:
            stdout = "malformed-row\nGPU-test-3090, 500, /opt/llama-server, 15996\n"
        return {"command": command, "returncode": 0, "stdout": stdout, "stderr": ""}

    identity = {
        key: value
        for key, value in _gpu(pid=500).items()
        if key
        in {
            "proc_exists",
            "ppid",
            "process_group_id",
            "session_id",
            "start_time_ticks",
            "process_start_utc",
            "age_s",
            "command",
            "command_text",
            "command_sha256",
            "open_port",
            "model_path",
            "model_sha256",
        }
    }
    identity["open_ports"] = [8919]
    monkeypatch.setattr(exp, "_run_subprocess", fake_subprocess)
    monkeypatch.setattr(exp, "_read_proc_identity", lambda pid: identity)

    process_rows, receipts = exp.collect_gpu_process_rows()

    assert len(receipts) == 2
    assert process_rows[0]["pid"] == 500
    assert process_rows[1]["pid"] is None
    assert process_rows[1]["ownership_classification"] == "idle"

    lease_dir = tmp_path / "leases"
    lease_dir.mkdir()
    document = {
        "schema": exp.LEASE_SCHEMA,
        "checksum": "checksum",
        "lease_id": "lease:test",
        "task_id": exp.TASK_ID,
        "device_uuid": GPU_UUID,
        "owner": {
            "pid": 500,
            "pid_start_ticks": 12345,
            "executable": "/usr/bin/python3",
            "argv_digest": "sha256:owner",
        },
        "expected_model": MODEL_PATH,
        "port": 8919,
        "model_sha256": MODEL_HASH,
        "phase": "resident",
        "released": False,
        "expires_monotonic_ns": 2_000,
        "recovery": {"signals_sent": []},
    }
    (lease_dir / "device-a.journal.json").write_text(json.dumps(document), encoding="utf-8")
    (lease_dir / "device-b.journal.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(exp.lease_api, "validate_journal_document", lambda *args, **kwargs: [])
    monkeypatch.setattr(exp.lease_api, "journal_checksum", lambda value: "checksum")
    monkeypatch.setattr(exp.time, "monotonic_ns", lambda: 1_000)

    lease_rows = exp.scan_lease_rows(lease_dir, process_rows)

    assert lease_rows[0]["canonical"] is True
    assert lease_rows[0]["owner_live"] is True
    assert lease_rows[0]["port"] == 8919
    assert lease_rows[1]["readable"] is False
    assert lease_rows[1]["error"] == "ValueError: lease_not_object"


# REQ-HARNESS-7160: runner and dependency probes use bounded read-only checks.
def test_runner_and_dependency_collectors_validate_exact_local_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = tmp_path / "llama-server"
    runner.write_text("binary fixture", encoding="utf-8")
    runner.chmod(0o755)

    def fake_runner_command(command: list[str], **kwargs: object) -> dict:
        if command[-1] == "--version":
            stdout = "llama.cpp test"
        elif command[-1] == "--help":
            stdout = "--n-predict --grammar"
        else:
            stdout = "libggml-cuda.so => /lib\nlibcuda.so => /lib"
        return {
            "command": command,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "duration_s": 0.01,
        }

    monkeypatch.setattr(exp, "_run_subprocess", fake_runner_command)
    runner_rows = exp.collect_runner_capabilities(runner)

    assert runner_rows[0]["valid"] is True
    assert exp.runner_capability_errors(runner_rows) == []
    assert runner_rows[0]["model_argument_present"] is False

    root = tmp_path / "repository"
    required = (Path("source.txt"),)
    monkeypatch.setattr(exp, "REQUIRED_SOURCE_PATHS", required)
    for relative in (
        *required,
        Path("tests/python/test_experiment_7160_v631_qwen38_lease_diagnosis.py"),
        Path("scripts/experiments/experiment_7160_v631_qwen38_lease_diagnosis.py"),
        Path("openspec/capabilities/research-harnesses/spec.md"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    result_path = root / exp.RESULT_PATH
    result_path.parent.mkdir(parents=True)
    result_path.write_text("{}", encoding="utf-8")
    lease_dir = tmp_path / "canonical-leases"
    lease_dir.mkdir()
    monkeypatch.setattr(exp.shutil, "which", lambda executable: "/usr/bin/nvidia-smi")

    checks = exp.collect_diagnostic_checks(
        root=root, run_date=exp.RUN_DATE, result_path=result_path, lease_dir=lease_dir
    )

    assert all(row["passed"] for row in checks)
    lease_check = next(row for row in checks if row["check"] == "lease_evidence_paths")
    assert lease_check["observed_value"]["legacy_gpu_memory_state_required"] is False


# REQ-HARNESS-7160 / SCENARIO-HARNESS-7160-ARTIFACT.
def test_run_and_cli_paths_finish_ready_or_blocked_without_process_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_validator = exp.validate_artifact
    monkeypatch.setattr(exp, "collect_diagnostic_checks", lambda **kwargs: _checks())
    monkeypatch.setattr(
        exp,
        "collect_gpu_process_rows",
        lambda: ([_gpu()], [{"returncode": 0}, {"returncode": 0}]),
    )
    monkeypatch.setattr(exp, "scan_lease_rows", lambda lease_dir, rows: [])
    monkeypatch.setattr(exp, "resolve_cache_identity", lambda: _cache())
    monkeypatch.setattr(exp, "resolve_native_llama_server", lambda: Path("/opt/llama-server"))
    monkeypatch.setattr(exp, "collect_runner_capabilities", lambda path: _runner())
    monkeypatch.setattr(
        exp,
        "stop_authority_receipt",
        lambda: {
            "marker_path": "/marker",
            "marker_present": False,
            "state": "disarmed",
            "signals_sent": [],
            "actions_taken": [],
        },
    )
    ready_path = tmp_path / "ready.json"

    ready = exp.run_experiment(
        root=tmp_path, run_date=exp.RUN_DATE, result_path=ready_path, lease_dir=tmp_path
    )

    assert ready["qwen38_runtime_preflight_ready_score"] == 1
    assert exp.validate_artifact(ready_path) == []
    assert exp.main(["--validate", str(ready_path)]) == 0

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("[]", encoding="utf-8")
    assert exp.main(["--validate", str(invalid_path)]) == 1

    monkeypatch.setattr(
        exp,
        "collect_diagnostic_checks",
        lambda **kwargs: [exp.gate_row("procfs", {"readable": True}, {"readable": False}, False)],
    )

    def forbidden_gpu_collection() -> tuple[list[dict], list[dict]]:
        raise AssertionError("GPU collection must not follow a missing diagnostic dependency")

    monkeypatch.setattr(exp, "collect_gpu_process_rows", forbidden_gpu_collection)
    blocked = exp.run_experiment(
        root=tmp_path,
        run_date=exp.RUN_DATE,
        result_path=tmp_path / "blocked.json",
        lease_dir=tmp_path,
    )
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["honest_verdict"] == "blocked_procfs"

    monkeypatch.setattr(exp, "validate_artifact", lambda value: ["forced_validation_error"])
    with pytest.raises(ValueError, match="terminal_artifact_invalid:forced_validation_error"):
        exp.run_experiment(
            root=tmp_path,
            run_date=exp.RUN_DATE,
            result_path=tmp_path / "invalid-terminal.json",
            lease_dir=tmp_path,
        )
    monkeypatch.setattr(exp, "validate_artifact", real_validator)

    captured: dict[str, object] = {}

    def fake_run_experiment(**kwargs: object) -> dict:
        captured.update(kwargs)
        return ready

    monkeypatch.setattr(exp, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(exp, "run_experiment", fake_run_experiment)
    assert exp.main(["--date", exp.RUN_DATE, "--result-path", "cli-result.json"]) == 0
    assert captured["result_path"] == tmp_path / "cli-result.json"
    assert "qwen38_runtime_preflight_ready_score" in capsys.readouterr().out
