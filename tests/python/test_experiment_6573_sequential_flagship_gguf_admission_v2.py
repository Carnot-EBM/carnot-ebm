"""Tests for Exp6573 sequential flagship CUDA admission.

Spec: REQ-REPORT-6573 and SCENARIO-REPORT-6573-SEQUENTIAL through
SCENARIO-REPORT-6573-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6573_sequential_flagship_gguf_admission_v2 as exp


def _identity_rows() -> list[dict]:
    rows = []
    for index, spec in enumerate(exp.MODEL_SPECS):
        digest = str(index + 1) * 64
        rows.append(
            {
                "row_type": "model_metadata_and_hash",
                "sequence_index": index,
                "repository_id": spec["repository_id"],
                "selected_blob_path": f"/cache/blobs/{digest}",
                "trusted_sha256": f"sha256:{digest}",
                "content_metadata": {
                    "architecture": spec["expected_architecture"],
                    "quantization": "Q4_K_M",
                    "tensor_count": 100,
                    "is_language_model": True,
                    "tokenizer_metadata": {
                        "model": "embedded",
                        "token_count": 262_144,
                        "chat_template_present": True,
                    },
                    "shard_metadata": {
                        "split_count": None,
                        "split_no": None,
                    },
                },
                "provenance": {
                    "valid": True,
                    "repository_id": spec["repository_id"],
                    "revision": f"revision-{index}",
                    "snapshot_filename": f"model-{index}.gguf",
                    "resolved_blob_path": f"/cache/blobs/{digest}",
                    "trusted_sha256": f"sha256:{digest}",
                    "trusted_hash_matches_blob_key": True,
                    "symlink_target_matches_blob": True,
                    "ordered_shards": [
                        {
                            "shard_number": 1,
                            "shard_count": 1,
                            "blob_key": digest,
                        }
                    ],
                },
                "rejection_reasons": [],
                "admitted": True,
                "passed": True,
            }
        )
    return rows


def _process_rows(identity_rows: list[dict] | None = None) -> list[dict]:
    identities = identity_rows or _identity_rows()
    rows = []
    for index, identity in enumerate(identities):
        output = f"The lighthouse glows with color number {index}."
        command = [
            "/cache/llama-server",
            "--model",
            identity["selected_blob_path"],
            "--n-gpu-layers",
            "all",
            "--split-mode",
            "none",
        ]
        rows.append(
            {
                "row_type": "sequential_load_generation",
                "repository_id": identity["repository_id"],
                "sequence_index": index,
                "selected_blob_path": identity["selected_blob_path"],
                "pid": 7100 + index,
                "parent_pid": 7000,
                "process_start_ticks": 90_000 + index,
                "os_pid_verified": True,
                "os_parent_pid_verified": True,
                "command_matches_os": True,
                "command": command,
                "command_sha256": exp.sha256_json(command),
                "os_command": command,
                "os_command_sha256": exp.sha256_json(command),
                "cuda_visible_devices": "0",
                "selected_gpu": 0,
                "port": 18100 + index,
                "http_status": 200,
                "server_healthy": True,
                "prompt_sha256": exp.sha256_text(exp.FROZEN_PROMPT),
                "prompt_token_count": 12,
                "output_token_count": 9,
                "raw_output": output,
                "raw_output_sha256": exp.sha256_text(output),
                "response_sha256": exp.sha256_text(f"response-{index}"),
                "stop_reason": "stop",
                "prompt_followed": True,
                "tokenizer_only_output": False,
                "output_reused": False,
                "load_duration_s": 4.0,
                "generation_duration_s": 0.5,
                "duration_s": 5.0,
                "start_monotonic_s": 10.0 + index * 10,
                "end_monotonic_s": 15.0 + index * 10,
                "timed_out": False,
                "shutdown_requested": True,
                "exit_code": 0,
                "normal_shutdown": True,
                "worker_alive_after_exit": False,
                "stdout_sha256": "sha256:" + "a" * 64,
                "stderr_sha256": "sha256:" + "b" * 64,
                "evidence_mode": "measured",
                "failing_stage": "",
                "error": "",
            }
        )
    return rows


def _gpu_receipts(process_rows: list[dict] | None = None) -> list[dict]:
    processes = process_rows or _process_rows()
    receipts = []
    for process in processes:
        pid = process["pid"]
        hf_id = process["repository_id"]
        for stage, used_mb, process_rows_at_sample, task_pids in (
            ("before", 4, [], []),
            (
                "during",
                18_000,
                [
                    {
                        "gpu_uuid": "GPU-0",
                        "pid": pid,
                        "process_name": "llama-server",
                        "used_memory_mb": 17_900,
                    }
                ],
                [pid],
            ),
            ("after", 4, [], []),
        ):
            receipts.append(
                {
                    "repository_id": hf_id,
                    "worker_pid": pid,
                    "stage": stage,
                    "sample_index": len(receipts),
                    "selected_gpu": 0,
                    "device": {
                        "index": 0,
                        "uuid": "GPU-0",
                        "name": "NVIDIA GeForce RTX 3090",
                        "memory_total_mb": 24_576,
                        "memory_used_mb": used_mb,
                        "memory_free_mb": 24_576 - used_mb,
                        "utilization_pct": 70 if stage == "during" else 0,
                    },
                    "all_devices": [
                        {"index": 0, "uuid": "GPU-0"},
                        {"index": 1, "uuid": "GPU-1"},
                    ],
                    "compute_processes": process_rows_at_sample,
                    "task_owned_live_pids": task_pids,
                    "gpu_query_exit_code": 0,
                    "compute_query_exit_code": 0,
                }
            )
    return receipts


def _unload_rows(process_rows: list[dict] | None = None) -> list[dict]:
    processes = process_rows or _process_rows()
    return [
        {
            "row_type": "unload_and_recovery",
            "repository_id": row["repository_id"],
            "sequence_index": row["sequence_index"],
            "worker_pid": row["pid"],
            "shutdown_requested": True,
            "exit_code": 0,
            "normal_shutdown": True,
            "worker_absent_from_proc": True,
            "worker_absent_from_nvidia_smi": True,
            "port": row["port"],
            "port_closed": True,
            "baseline_memory_used_mb": 4,
            "recovered_memory_used_mb": 4,
            "memory_delta_from_baseline_mb": 0,
            "recovery_tolerance_mb": exp.RECOVERY_TOLERANCE_MB,
            "no_task_worker_remains": True,
            "recovery_command_exit_code": 0,
            "recovery_binary_sha256_matches": True,
            "recovery_bounded": True,
            "signals_sent_to_unrelated_pids": [],
            "recovery_complete": True,
            "failing_stage": "",
            "error": "",
        }
        for row in processes
    ]


def _gates(*, ready: bool = True) -> dict:
    return {
        "rows": [
            {
                "upstream": f"exp{6571 + index}",
                "path": str(path),
                "field": field,
                "expected_value": 1.0,
                "observed_value": 1.0 if ready else 0.0,
                "sha256": "sha256:" + str(index + 7) * 64,
                "passed": ready,
            }
            for index, (path, field) in enumerate(exp.UPSTREAM_GATES)
        ],
        "all_structured_gates_passed": ready,
    }


def _preconditions(*, ready: bool = True) -> dict:
    checks = {
        "structured_gates": ready,
        "content_identity": ready,
        "llama_cpp_cuda_build": ready,
        "cuda_telemetry": ready,
        "idle_supported_gpu": ready,
        "one_model_residency": ready,
        "atomic_output_ready": ready,
    }
    return {
        "all_required_preconditions_available": all(checks.values()),
        "checks": checks,
        "failed_preconditions": [name for name, passed in checks.items() if not passed],
        "llama_cpp_build": {"cuda_linked": ready, "binary_sha256": "sha256:" + "c" * 64},
        "selected_gpu": 0 if ready else None,
        "free_vram_arithmetic_used_as_gate": False,
    }


def _protected(*, unchanged: bool = True) -> dict:
    return {
        "all_unchanged": unchanged,
        "research_roadmap_yaml_unchanged": unchanged,
        "research_conductor_py_unchanged": unchanged,
        "rows": [],
    }


def _assemble(**changes: object) -> dict:
    identities = _identity_rows()
    processes = _process_rows(identities)
    values = {
        "upstream_gate_receipts": _gates(),
        "metadata_rows": identities,
        "process_rows": processes,
        "gpu_rows": _gpu_receipts(processes),
        "unload_rows": _unload_rows(processes),
        "preconditions": _preconditions(),
        "protected": _protected(),
        "duration_s": 30.0,
        "tests_run": [{"command": "pytest focused", "exit_code": 0}],
        "run_date": "20260824",
    }
    values.update(changes)
    return exp.assemble_artifact(**values)


# REQ-REPORT-6573-GATES / SCENARIO-REPORT-6573-ATOMIC.
def test_upstream_gates_record_exact_fields_values_hashes_and_fail_closed(tmp_path: Path) -> None:
    for relative, field in exp.UPSTREAM_GATES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({field: 1.0}), encoding="utf-8")

    ready = exp.build_upstream_gate_receipts(tmp_path)
    assert ready["all_structured_gates_passed"] is True
    assert [row["field"] for row in ready["rows"]] == [field for _, field in exp.UPSTREAM_GATES]
    assert all(row["sha256"].startswith("sha256:") for row in ready["rows"])

    (tmp_path / exp.UPSTREAM_GATES[0][0]).write_text("{", encoding="utf-8")
    blocked = exp.build_upstream_gate_receipts(tmp_path)
    assert blocked["all_structured_gates_passed"] is False
    assert blocked["rows"][0]["observed_value"] is None
    assert exp._load_json(tmp_path / "absent.json") == {}


# REQ-REPORT-6573-SEQUENTIAL: GPU choice uses live idleness, never size arithmetic.
def test_idle_gpu_selection_preserves_busy_gpu_and_records_reason() -> None:
    sample = {
        "gpu_query_exit_code": 0,
        "compute_query_exit_code": 0,
        "all_devices": [
            {"index": 0, "uuid": "GPU-0", "name": "NVIDIA RTX", "utilization_pct": 0},
            {"index": 1, "uuid": "GPU-1", "name": "NVIDIA RTX", "utilization_pct": 97},
        ],
        "compute_processes": [
            {"gpu_uuid": "GPU-1", "pid": 999, "process_name": "trainer", "used_memory_mb": 20_000}
        ],
    }

    receipt = exp.choose_idle_gpu(sample)
    assert receipt["selected_gpu"] == 0
    assert receipt["eligible"] is True
    assert receipt["free_vram_arithmetic_used_as_gate"] is False
    assert receipt["preserved_busy_gpu_indices"] == [1]


# REQ-REPORT-6573-CUDA: exact single-device launch flags are frozen.
def test_command_uses_exact_blob_single_cuda_device_and_bounded_policy() -> None:
    command = exp.build_server_command(Path("/bin/llama-server"), Path("/cache/blob"), 18080)

    assert command[command.index("--model") + 1] == "/cache/blob"
    assert command[command.index("--n-gpu-layers") + 1] == "all"
    assert command[command.index("--split-mode") + 1] == "none"
    assert command[command.index("--device") + 1] == "CUDA0"
    assert command[command.index("--fit") + 1] == "off"
    assert command[command.index("--ctx-size") + 1] == str(exp.CONTEXT_SIZE)


# REQ-REPORT-6573-CUDA: a pre-exec parent command cannot satisfy worker identity.
def test_exact_os_command_match_rejects_the_fork_exec_receipt_race() -> None:
    command = exp.build_server_command(Path("/bin/llama-server"), Path("/cache/blob"), 18080)

    assert exp.command_matches_expected(command, command) is True
    assert (
        exp.command_matches_expected(["python", "-m", "carnot.experiment_6573"], command) is False
    )


# REQ-REPORT-6573-CUDA / SCENARIO-REPORT-6573-AUTHENTIC: a stable live
# command must replace a transient launch receipt after the server is healthy.
def test_stable_process_identity_replaces_transient_launch_receipt() -> None:
    command = exp.build_server_command(Path("/bin/llama-server"), Path("/cache/blob"), 18080)
    transient = {
        "pid": 7100,
        "parent_pid": 7000,
        "process_start_ticks": 90_000,
        "command": ["python", "-m", "carnot.experiment_6573"],
        "verified": True,
    }
    stable = {**transient, "command": command}

    receipt = exp.select_process_identity_receipt(transient, stable, command)

    assert receipt == stable
    assert exp.command_matches_expected(receipt["command"], command) is True


# REQ-REPORT-6573-IDENTITY / SCENARIO-REPORT-6573-IDENTITY.
@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda row: row.update(repository_id="wrong/repository"), "repository_identity"),
        (lambda row: row["content_metadata"].update(architecture="wrong"), "architecture"),
        (lambda row: row["content_metadata"].update(quantization=None), "quantization"),
        (
            lambda row: row["content_metadata"]["tokenizer_metadata"].update(token_count=0),
            "embedded_tokenizer",
        ),
        (lambda row: row["provenance"].update(trusted_hash_matches_blob_key=False), "trusted_hash"),
        (lambda row: row["provenance"].update(repository_id="wrong/repository"), "provenance"),
        (lambda row: row["provenance"].update(ordered_shards=[]), "complete_shards"),
    ],
)
def test_content_and_provenance_attacks_fail_closed(mutation, failed_check: str) -> None:
    row = deepcopy(_identity_rows()[0])
    mutation(row)

    checks = exp.identity_checks(row, exp.MODEL_SPECS[0])
    assert checks[failed_check] is False


# REQ-REPORT-6573-GENERATION / SCENARIO-REPORT-6573-GENERATION.
@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("os_pid_verified", False, "os_pid_verified"),
        ("command_matches_os", False, "command_matches_os"),
        ("output_token_count", 0, "generated_tokens"),
        ("raw_output", "", "nonempty_output"),
        ("prompt_followed", False, "prompt_followed"),
        ("tokenizer_only_output", True, "not_tokenizer_only"),
        ("output_reused", True, "output_not_reused"),
        ("timed_out", True, "bounded_generation"),
        ("normal_shutdown", False, "normal_shutdown"),
        ("exit_code", 1, "clean_exit"),
        ("worker_alive_after_exit", True, "worker_absent_after_exit"),
    ],
)
def test_process_and_generation_attacks_fail_closed(
    field: str, value: object, failed_check: str
) -> None:
    row = deepcopy(_process_rows()[0])
    row[field] = value
    if field == "raw_output":
        row["raw_output_sha256"] = exp.sha256_text(str(value))

    assert exp.process_checks(row)[failed_check] is False


# REQ-REPORT-6573-CUDA / SCENARIO-REPORT-6573-AUTHENTIC.
def test_gpu_checks_reject_zero_offload_stale_pid_overlap_and_cross_gpu() -> None:
    rows = _gpu_receipts()[:3]
    assert all(exp.gpu_checks(rows, worker_pid=7100, selected_gpu=0).values())

    zero = deepcopy(rows)
    zero[1]["device"]["memory_used_mb"] = 4
    zero[1]["compute_processes"] = []
    assert exp.gpu_checks(zero, worker_pid=7100, selected_gpu=0)["positive_gpu_residency"] is False

    stale = deepcopy(rows)
    stale[1]["compute_processes"][0]["pid"] = 9999
    assert exp.gpu_checks(stale, worker_pid=7100, selected_gpu=0)["worker_pid_linked"] is False

    overlap = deepcopy(rows)
    overlap[1]["task_owned_live_pids"] = [7100, 7200]
    assert exp.gpu_checks(overlap, worker_pid=7100, selected_gpu=0)["one_model_resident"] is False

    cross_gpu = deepcopy(rows)
    cross_gpu[1]["compute_processes"][0]["gpu_uuid"] = "GPU-1"
    assert exp.gpu_checks(cross_gpu, worker_pid=7100, selected_gpu=0)["selected_gpu_only"] is False


# REQ-REPORT-6573-UNLOAD / SCENARIO-REPORT-6573-UNLOAD.
@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("worker_absent_from_proc", False, "pid_gone"),
        ("port_closed", False, "port_closed"),
        ("memory_delta_from_baseline_mb", exp.RECOVERY_TOLERANCE_MB + 1, "memory_recovered"),
        ("recovery_command_exit_code", 1, "recovery_smoke"),
        ("recovery_binary_sha256_matches", False, "same_binary"),
        ("signals_sent_to_unrelated_pids", [999], "unrelated_processes_preserved"),
    ],
)
def test_unload_and_recovery_attacks_fail_closed(
    field: str, value: object, failed_check: str
) -> None:
    row = deepcopy(_unload_rows()[0])
    row[field] = value
    assert exp.unload_checks(row)[failed_check] is False


# REQ-REPORT-6573-REDUCER / SCENARIO-REPORT-6573-ATOMIC.
def test_complete_artifact_recomputes_every_family_and_required_field() -> None:
    artifact = _assemble()

    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_provenance"])
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["all_mandated_models_loaded_score"] == 1.0
    assert artifact["family_admitted_scores"] == {hf_id: 1.0 for hf_id in exp.MANDATED_HF_IDS}
    assert artifact["verdict_class"] is None
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert len(artifact["per_unit_rows"]) == 15
    assert exp.validate_artifact(artifact) == []


# REQ-REPORT-6573-REDUCER: one missing or failed family cannot hide in aggregate readiness.
def test_partial_blocked_and_reused_output_verdicts_are_honest() -> None:
    processes = _process_rows()
    processes[1]["exit_code"] = 1
    processes[1]["normal_shutdown"] = False
    partial = _assemble(process_rows=processes)

    blocked = _assemble(process_rows=[], gpu_rows=[], unload_rows=[])

    reused = _process_rows()
    reused[1]["raw_output"] = reused[0]["raw_output"]
    reused[1]["raw_output_sha256"] = reused[0]["raw_output_sha256"]
    reused_artifact = _assemble(process_rows=reused)

    assert partial["verdict_class"] == "partial"
    assert partial["family_admitted_scores"][exp.MANDATED_HF_IDS[1]] == 0.0
    assert blocked["verdict_class"] == "blocked"
    assert reused_artifact["all_mandated_models_loaded_score"] == 0.0
    assert (
        "output_not_reused"
        in reused_artifact["aggregate_row_recomputation"]["family_rows"][1]["failed_checks"]
    )


# REQ-REPORT-6573-LEGACY / SCENARIO-REPORT-6573-LEGACY.
def test_legacy_cpu_smoke_never_enters_family_scores() -> None:
    artifact = _assemble()
    artifact["per_unit_rows"].append(
        {"repository_id": "Qwen/Qwen3.5-0.8B", "stage": "legacy_cpu_smoke", "passed": True}
    )

    assert set(artifact["family_admitted_scores"]) == set(exp.MANDATED_HF_IDS)


# REQ-REPORT-6573-ATOMIC: protected mutation and predicted evidence disqualify.
def test_disqualification_and_validation_catch_fabrication_and_tampering() -> None:
    predicted = _process_rows()
    predicted[0]["evidence_mode"] = "predicted"
    artifact = _assemble(process_rows=predicted)
    assert artifact["verdict_class"] == "disqualified"

    protected = _assemble(protected=_protected(unchanged=False))
    assert protected["verdict_class"] == "disqualified"

    valid = _assemble()
    valid["all_mandated_models_loaded_score"] = 0.0
    valid["reproducibility_checksum"] = "sha256:wrong"
    errors = exp.validate_artifact(valid)
    assert "aggregate_score_mismatch" in errors
    assert "reproducibility_checksum_mismatch" in errors

    drifted = _assemble()
    drifted["family_admitted_scores"] = {}
    drifted["aggregate_row_recomputation"]["recomputed_score"] = 0.0
    drifted["inference_substrate"] = "predicted"
    drifted["verifier_is_oracle"] = False
    drifted["field_provenance"] = {}
    drifted["field_principles"] = {}
    drift_errors = exp.validate_artifact(drifted)
    assert "family_scores_mismatch" in drift_errors
    assert "aggregate_recomputation_mismatch" in drift_errors
    assert "inference_substrate_mismatch" in drift_errors
    assert "verifier_is_oracle_mismatch" in drift_errors
    assert "field_provenance_incomplete" in drift_errors
    assert "field_principles_incomplete" in drift_errors


# REQ-REPORT-6573-GATES: an exact structured or CUDA gate block names observed values.
def test_precondition_block_names_failed_checks_and_gate_observations() -> None:
    artifact = _assemble(
        upstream_gate_receipts=_gates(ready=False),
        preconditions=_preconditions(ready=False),
        process_rows=[],
        gpu_rows=[],
        unload_rows=[],
    )

    assert artifact["verdict_class"] == "blocked"
    assert "structured_gates" in artifact["gate_check_summary"]["failed_checks"]
    assert artifact["gate_check_summary"]["blocked_gate_rows"][0]["observed_value"] == 0.0
