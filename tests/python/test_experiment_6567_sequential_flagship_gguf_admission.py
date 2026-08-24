"""Tests for Exp6567 sequential flagship GGUF admission.

Spec: REQ-REPORT-6567 and SCENARIO-REPORT-6567-GATES through
SCENARIO-REPORT-6567-ATOMIC.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_6567_sequential_flagship_gguf_admission as exp


def _valid_upstream() -> dict:
    return {
        "all_structured_gates_passed": True,
        "rows": [
            {
                "upstream": "exp6565-v569-evidence-and-retirement-contract",
                "path": "results/experiment_6565_v569_evidence_and_retirement_contract.json",
                "sha256": "sha256:" + "1" * 64,
                "field": "v569_evidence_contract_ready_score",
                "expected_value": 1.0,
                "observed_value": 1.0,
                "passed": True,
            },
            {
                "upstream": "exp6566-proof-obligation-and-graph-potts-method-contract",
                "path": "results/experiment_6566_proof_obligation_and_graph_potts_method_contract.json",
                "sha256": "sha256:" + "2" * 64,
                "field": "source_method_contract_ready_score",
                "expected_value": 1.0,
                "observed_value": 1.0,
                "passed": True,
            },
        ],
    }


def _valid_model_rows() -> list[dict]:
    rows = []
    for index, hf_id in enumerate(exp.MANDATED_HF_IDS):
        rows.append(
            {
                "hf_id": hf_id,
                "repository_id": hf_id,
                "quantization": "Q4_K_M",
                "absolute_path": f"/cache/revision-{index}/model-{index}-Q4_K_M.gguf",
                "byte_size": 1_000_000 + index,
                "sha256": "sha256:" + str(index + 3) * 64,
                "mtime_ns": 1_700_000_000_000_000_000 + index,
                "revision": f"revision-{index}",
                "is_split_file": False,
                "is_language_model": True,
                "repository_path_matches": True,
                "tokenizer_metadata": {
                    "embedded_tokenizer_ok": True,
                    "loader": "llama.cpp embedded GGUF tokenizer",
                    "tokenizer_model": "gpt2" if index == 0 else "sentencepiece",
                    "chat_template_sha256": "sha256:" + str(index + 6) * 64,
                    "prompt_token_count": 12,
                    "prompt_token_ids_sha256": "sha256:" + str(index + 7) * 64,
                    "autotokenizer_usage_count": 0,
                },
                "resolution_error": "",
            }
        )
    return rows


def _valid_process_rows() -> list[dict]:
    rows = []
    for index, hf_id in enumerate(exp.MANDATED_HF_IDS):
        pid = 4100 + index
        raw_output = f"A luminous lighthouse guards harbor {index}."
        rows.append(
            {
                "hf_id": hf_id,
                "sequence_index": index,
                "loader": "llama.cpp llama-cli external worker",
                "command": ["llama-cli", "-m", f"model-{index}.gguf"],
                "command_sha256": exp.sha256_json(["llama-cli", "-m", f"model-{index}.gguf"]),
                "pid": pid,
                "parent_pid": 4000,
                "process_start_ticks": 100_000 + index,
                "os_pid_verified": True,
                "os_parent_pid_verified": True,
                "command_matches_os": True,
                "start_time_utc": f"2026-08-23T12:00:0{index}Z",
                "end_time_utc": f"2026-08-23T12:00:1{index}Z",
                "start_monotonic_s": 10.0 + index * 20,
                "end_monotonic_s": 20.0 + index * 20,
                "duration_s": 10.0,
                "stdout_sha256": "sha256:" + str(index + 1) * 64,
                "stderr_sha256": "sha256:" + str(index + 2) * 64,
                "raw_output": raw_output,
                "raw_output_sha256": exp.sha256_text(raw_output),
                "prompt_sha256": exp.sha256_text(exp.FROZEN_PROMPT),
                "prompt_token_count": 12,
                "prompt_token_ids_sha256": "sha256:" + str(index + 4) * 64,
                "output_token_count": 7,
                "output_unique_token_count": 6,
                "output_token_ids_sha256": "sha256:" + str(index + 5) * 64,
                "exit_code": 0,
                "terminating_signal": None,
                "timed_out": False,
                "empty_output": False,
                "echo_only_output": False,
                "output_reused": False,
                "worker_alive_after_exit": False,
                "selected_gpu": 0,
                "error": "",
            }
        )
    return rows


def _sample(
    hf_id: str,
    pid: int,
    stage: str,
    used_mb: int,
    process_pids: list[int],
    sample_index: int,
) -> dict:
    return {
        "hf_id": hf_id,
        "worker_pid": pid,
        "stage": stage,
        "sample_index": sample_index,
        "timestamp_utc": f"2026-08-23T12:00:{sample_index:02d}Z",
        "monotonic_s": float(sample_index),
        "selected_gpu": 0,
        "device": {
            "index": 0,
            "uuid": "GPU-0",
            "name": "NVIDIA GeForce RTX 3090",
            "memory_total_mb": 24576,
            "memory_used_mb": used_mb,
            "memory_free_mb": 24576 - used_mb,
            "utilization_pct": 95 if stage == "during" else 0,
            "temperature_c": 60 if stage == "during" else 45,
            "driver_version": "610.43.03",
        },
        "compute_processes": [
            {
                "gpu_uuid": "GPU-0",
                "pid": process_pid,
                "process_name": "llama-cli",
                "used_memory_mb": 16_000,
            }
            for process_pid in process_pids
        ],
        "gpu_query_exit_code": 0,
        "compute_query_exit_code": 0,
        "gpu_query_stdout_sha256": "sha256:" + str(sample_index + 1) * 64,
        "gpu_query_stderr_sha256": "sha256:" + "0" * 64,
        "compute_query_stdout_sha256": "sha256:" + str(sample_index + 2) * 64,
        "compute_query_stderr_sha256": "sha256:" + "0" * 64,
    }


def _valid_gpu_rows() -> list[dict]:
    rows = []
    for index, hf_id in enumerate(exp.MANDATED_HF_IDS):
        pid = 4100 + index
        base = index * 10
        rows.extend(
            [
                _sample(hf_id, pid, "before", 4, [], base),
                _sample(hf_id, pid, "during", 18_000 + index, [pid], base + 1),
                _sample(hf_id, pid, "after", 4, [], base + 2),
            ]
        )
    return rows


def _valid_recovery_rows() -> list[dict]:
    rows = []
    for index, hf_id in enumerate(exp.MANDATED_HF_IDS):
        rows.append(
            {
                "hf_id": hf_id,
                "sequence_index": index,
                "worker_pid": 4100 + index,
                "baseline_memory_used_mb": 4,
                "recovered_memory_used_mb": 4,
                "memory_delta_from_baseline_mb": 0,
                "recovery_tolerance_mb": exp.RECOVERY_TOLERANCE_MB,
                "worker_absent_from_proc": True,
                "worker_absent_from_nvidia_smi": True,
                "no_task_worker_remains": True,
                "recovery_complete": True,
                "recovery_time_utc": f"2026-08-23T12:01:0{index}Z",
                "recovery_monotonic_s": 21.0 + index * 20,
                "recovery_duration_s": 1.0,
                "next_worker_started_after_recovery": True,
                "error": "",
            }
        )
    return rows


def _valid_preconditions() -> dict:
    return {
        "checks": {
            "structured_gates": True,
            "cuda_runtime": True,
            "llama_cpp_cli": True,
            "llama_cpp_python": True,
            "z3": True,
            "all_model_files_resolved": True,
            "all_embedded_tokenizers_valid": True,
            "atomic_output_ready": True,
        },
        "failed_preconditions": [],
        "cpu": {"count": 24, "model": "test CPU"},
        "ram": {"total_kib": 128_000_000, "available_kib": 100_000_000},
        "disk": {"total_bytes": 4_000_000_000_000, "free_bytes": 1_000_000_000_000},
        "cuda": {"available": True, "driver_version": "610.43.03"},
        "llama_cpp_cli": {"available": True, "version": "b9606"},
        "llama_cpp_python": {"available": True, "version": "0.3.33"},
        "z3": {"available": True, "version": "4.16.0"},
        "cached_model_candidates": [],
        "initial_gpu_state": [],
        "protected_file_hashes_before": {},
        "free_vram_arithmetic_used_as_gate": False,
    }


def _valid_protected() -> dict:
    return {
        "all_unchanged": True,
        "changed_paths": [],
        "research_conductor_py_unchanged": True,
        "rows": [],
    }


def _assemble(**changes: object) -> dict:
    values = {
        "upstream_gate_receipts": _valid_upstream(),
        "model_file_rows": _valid_model_rows(),
        "process_rows": _valid_process_rows(),
        "gpu_rows": _valid_gpu_rows(),
        "recovery_rows": _valid_recovery_rows(),
        "preconditions": _valid_preconditions(),
        "protected": _valid_protected(),
        "duration_s": 75.0,
        "tests_run": [{"command": "pytest focused", "exit_code": 0}],
    }
    values.update(changes)
    return exp.assemble_artifact(**values)


# REQ-REPORT-6567 / SCENARIO-REPORT-6567-ATOMIC.
def test_complete_artifact_has_exact_schema_principles_and_score() -> None:
    artifact = _assemble()

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(exp.MANDATED_HF_IDS)
    assert artifact["all_mandated_models_loaded_score"] == 1.0
    assert artifact["aggregate_row_recomputation"]["ready_score_from_rows"] == 1.0
    assert artifact["verdict_class"] is None
    assert artifact["status"].startswith("complete_")
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["random_seed"] == 6567
    assert exp.validate_artifact(artifact) == []


# SCENARIO-REPORT-6567-GATES: exact upstream paths, hashes, and values gate execution.
def test_upstream_gate_receipts_read_expected_contracts(tmp_path: Path) -> None:
    for relative, field in exp.UPSTREAM_GATES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({field: 1.0}), encoding="utf-8")

    receipt = exp.build_upstream_gate_receipts(tmp_path)

    assert receipt["all_structured_gates_passed"] is True
    assert [row["field"] for row in receipt["rows"]] == [
        "v569_evidence_contract_ready_score",
        "source_method_contract_ready_score",
    ]
    assert all(row["passed"] and row["sha256"].startswith("sha256:") for row in receipt["rows"])


# SCENARIO-REPORT-6567-GATES: missing or malformed upstream state blocks honestly.
def test_upstream_gate_receipts_fail_closed_for_missing_and_wrong_values(tmp_path: Path) -> None:
    first_path = tmp_path / exp.UPSTREAM_GATES[0][0]
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_text(
        json.dumps({exp.UPSTREAM_GATES[0][1]: 0.0}),
        encoding="utf-8",
    )

    receipt = exp.build_upstream_gate_receipts(tmp_path)

    assert receipt["all_structured_gates_passed"] is False
    assert receipt["rows"][0]["observed_value"] == 0.0
    assert receipt["rows"][1]["observed_value"] is None
    assert receipt["rows"][1]["sha256"] == "missing"


# SCENARIO-REPORT-6567-GATES: malformed upstream JSON is an absent gate, not a crash.
def test_load_json_rejects_malformed_and_non_object_input(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")

    assert exp._load_json(malformed) == {}
    assert exp._load_json(non_object) == {}


# SCENARIO-REPORT-6567-RESOLVE: cache resolution records concrete file identity.
def test_resolve_model_file_rows_records_files_and_embedded_tokenizers(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for index, hf_id in enumerate(exp.MANDATED_HF_IDS):
        model_dir = tmp_path / hf_id.split("/", 1)[1] / "snapshots" / f"rev-{index}"
        model_dir.mkdir(parents=True)
        path = model_dir / f"family-{index}-Q4_K_M.gguf"
        path.write_bytes(f"model-{index}".encode())
        paths[hf_id] = path

    def resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str:
        assert preferred_quant == "Q4_K_M"
        return str(paths[hf_id])

    def tokenizer_reader(path: str) -> dict:
        assert Path(path).is_file()
        return {
            "embedded_tokenizer_ok": True,
            "loader": "llama.cpp embedded GGUF tokenizer",
            "tokenizer_model": "test",
            "chat_template_sha256": "sha256:" + "8" * 64,
            "prompt_token_count": 4,
            "prompt_token_ids_sha256": "sha256:" + "9" * 64,
            "autotokenizer_usage_count": 0,
        }

    rows = exp.resolve_model_file_rows(resolver=resolver, tokenizer_reader=tokenizer_reader)

    assert [row["hf_id"] for row in rows] == list(exp.MANDATED_HF_IDS)
    assert all(row["absolute_path"].startswith(str(tmp_path)) for row in rows)
    assert all(row["byte_size"] == len(f"model-{index}") for index, row in enumerate(rows))
    assert all(row["sha256"].startswith("sha256:") for row in rows)
    assert all(row["revision"].startswith("rev-") for row in rows)
    assert all(row["repository_path_matches"] for row in rows)
    assert all(row["tokenizer_metadata"]["embedded_tokenizer_ok"] for row in rows)


# SCENARIO-REPORT-6567-RESOLVE: snapshot symlinks keep their auditable cache identity.
def test_resolve_model_file_rows_does_not_dereference_snapshot_symlinks(tmp_path: Path) -> None:
    blob = tmp_path / "blobs" / "content-hash"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"GGUF bytes")
    links: dict[str, Path] = {}
    for index, hf_id in enumerate(exp.MANDATED_HF_IDS):
        snapshot = (
            tmp_path
            / f"models--{hf_id.replace('/', '--')}"
            / "snapshots"
            / f"revision-{index}"
        )
        snapshot.mkdir(parents=True)
        link = snapshot / f"family-{index}-Q4_K_M.gguf"
        link.symlink_to(blob)
        links[hf_id] = link

    rows = exp.resolve_model_file_rows(
        resolver=lambda hf_id, preferred_quant="Q4_K_M": str(links[hf_id]),
        tokenizer_reader=lambda _: {
            "embedded_tokenizer_ok": True,
            "prompt_token_count": 1,
            "autotokenizer_usage_count": 0,
        },
    )

    assert [row["absolute_path"] for row in rows] == [
        str(links[hf_id].absolute()) for hf_id in exp.MANDATED_HF_IDS
    ]
    assert [row["revision"] for row in rows] == [
        "revision-0",
        "revision-1",
        "revision-2",
    ]
    assert all(row["quantization"] == "Q4_K_M" for row in rows)
    assert all(row["is_language_model"] for row in rows)


# SCENARIO-REPORT-6567-ATTACKS: missing weights do not trigger a download.
def test_resolve_model_file_rows_keeps_missing_weights_terminal() -> None:
    calls: list[str] = []

    def resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> None:
        calls.append(hf_id)
        assert preferred_quant == "Q4_K_M"
        return None

    rows = exp.resolve_model_file_rows(resolver=resolver, tokenizer_reader=lambda _: {})

    assert calls == list(exp.MANDATED_HF_IDS)
    assert all(row["resolution_error"] == "model_not_cached" for row in rows)
    assert all(row["absolute_path"] == "" for row in rows)


# SCENARIO-REPORT-6567-RESOLVE: a stale resolver path remains blocked.
def test_resolve_model_file_rows_rejects_stale_resolver_paths(tmp_path: Path) -> None:
    stale_path = tmp_path / "missing.gguf"

    rows = exp.resolve_model_file_rows(
        resolver=lambda _hf_id, preferred_quant="Q4_K_M": str(stale_path),
        tokenizer_reader=lambda _: {},
    )

    assert all(row["resolution_error"] == "resolved_path_missing" for row in rows)


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda row: row.update(repository_id="wrong/repository"), "repository_identity"),
        (lambda row: row.update(repository_path_matches=False), "repository_path"),
        (lambda row: row.update(is_split_file=True), "split_file"),
        (lambda row: row.update(is_language_model=False), "language_model_file"),
        (
            lambda row: row["tokenizer_metadata"].update(embedded_tokenizer_ok=False),
            "embedded_tokenizer",
        ),
        (
            lambda row: row["tokenizer_metadata"].update(autotokenizer_usage_count=1),
            "autotokenizer_forbidden",
        ),
    ],
)
# SCENARIO-REPORT-6567-ATTACKS: aliases, split files, and wrong tokenizers fail closed.
def test_model_receipt_attacks_fail_closed(mutation, expected_reason: str) -> None:
    rows = _valid_model_rows()
    mutation(rows[0])

    checks = exp.model_row_checks(rows[0], exp.MANDATED_HF_IDS[0])

    assert checks[expected_reason] is False


# SCENARIO-REPORT-6567-TELEMETRY: CSV parsing preserves device and process identity.
def test_gpu_and_compute_csv_parsers() -> None:
    gpu_text = (
        "0, GPU-0, NVIDIA GeForce RTX 3090, 24576, 18000, 6576, 95, 60, 610.43.03\n"
        "1, GPU-1, NVIDIA GeForce RTX 3090, 24576, 4, 24120, 0, 45, 610.43.03\n"
    )
    process_text = "GPU-0, 4100, /bin/llama-cli, 17900\n"

    devices = exp.parse_gpu_csv(gpu_text)
    processes = exp.parse_compute_process_csv(process_text)

    assert devices[0]["uuid"] == "GPU-0"
    assert devices[0]["memory_used_mb"] == 18000
    assert devices[1]["memory_free_mb"] == 24120
    assert processes == [
        {
            "gpu_uuid": "GPU-0",
            "pid": 4100,
            "process_name": "/bin/llama-cli",
            "used_memory_mb": 17900,
        }
    ]
    assert exp.parse_gpu_csv("malformed") == []
    assert exp.parse_compute_process_csv("malformed") == []
    assert exp.parse_gpu_csv("x, GPU, name, bad, 1, 2, 3, 4, driver") == []
    assert exp.parse_compute_process_csv("GPU, bad, name, 1") == []


# SCENARIO-REPORT-6567-SEQUENTIAL: process receipts require OS identity and output tokens.
def test_process_row_checks_accept_authentic_external_worker() -> None:
    row = _valid_process_rows()[0]

    checks = exp.process_row_checks(row)

    assert checks and all(checks.values())


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("os_pid_verified", False, "os_pid_verified"),
        ("command_matches_os", False, "command_matches_os"),
        ("empty_output", True, "output_nonempty"),
        ("echo_only_output", True, "output_not_echo_only"),
        ("output_unique_token_count", 1, "output_tokens_nonconstant"),
        ("output_reused", True, "output_not_reused"),
        ("exit_code", 1, "clean_exit"),
        ("timed_out", True, "not_timed_out"),
        ("worker_alive_after_exit", True, "worker_absent_after_exit"),
    ],
)
# SCENARIO-REPORT-6567-ATTACKS: stale, empty, echo, reused, failed, or live workers fail.
def test_process_receipt_attacks_fail_closed(field: str, value: object, failed_check: str) -> None:
    row = _valid_process_rows()[0]
    row[field] = value

    assert exp.process_row_checks(row)[failed_check] is False


# SCENARIO-REPORT-6567-TELEMETRY: during samples bind to PID and show measured load.
def test_gpu_telemetry_checks_accept_pid_linked_changing_samples() -> None:
    rows = _valid_gpu_rows()[:3]

    checks = exp.telemetry_checks(rows, worker_pid=4100, selected_gpu=0)

    assert checks and all(checks.values())


# SCENARIO-REPORT-6567-ATTACKS: constant samples, stale PIDs, and hidden workers fail.
def test_gpu_telemetry_attacks_fail_closed() -> None:
    constant = _valid_gpu_rows()[:3]
    for row in constant:
        row["device"]["memory_used_mb"] = 4
        row["device"]["utilization_pct"] = 0
        row["compute_processes"] = []
    constant_checks = exp.telemetry_checks(constant, worker_pid=4100, selected_gpu=0)

    stale = _valid_gpu_rows()[:3]
    stale[1]["compute_processes"][0]["pid"] = 9999
    stale_checks = exp.telemetry_checks(stale, worker_pid=4100, selected_gpu=0)

    hidden = _valid_gpu_rows()[:3]
    hidden[1]["compute_processes"].append(
        {
            "gpu_uuid": "GPU-0",
            "pid": 7777,
            "process_name": "llama-cli",
            "used_memory_mb": 1000,
        }
    )
    hidden_checks = exp.telemetry_checks(hidden, worker_pid=4100, selected_gpu=0)

    assert constant_checks["samples_nonconstant"] is False
    assert constant_checks["measured_load_delta"] is False
    assert stale_checks["worker_pid_linked_during"] is False
    assert hidden_checks["no_hidden_simultaneous_worker"] is False


# SCENARIO-REPORT-6567-UNLOAD: recovery requires process absence and bounded memory.
def test_recovery_checks_accept_clean_unload_and_reject_failures() -> None:
    row = _valid_recovery_rows()[0]
    assert all(exp.recovery_checks(row).values())

    row["memory_delta_from_baseline_mb"] = exp.RECOVERY_TOLERANCE_MB + 1
    row["worker_absent_from_proc"] = False
    row["next_worker_started_after_recovery"] = False
    checks = exp.recovery_checks(row)

    assert checks["memory_recovered_within_tolerance"] is False
    assert checks["worker_absent_from_proc"] is False
    assert checks["recovery_precedes_next_worker"] is False


# SCENARIO-REPORT-6567-SEQUENTIAL: row construction keeps all families separate.
def test_per_unit_rows_recompute_family_admission() -> None:
    rows = exp.build_per_unit_rows(
        _valid_model_rows(),
        _valid_process_rows(),
        _valid_gpu_rows(),
        _valid_recovery_rows(),
    )

    assert len(rows) == 3
    assert [row["hf_id"] for row in rows] == list(exp.MANDATED_HF_IDS)
    assert all(row["admitted"] for row in rows)
    assert all(row["row_hash"] == exp.row_hash(row) for row in rows)


# SCENARIO-REPORT-6567-ATTACKS: one family cannot stand in for another.
def test_partial_and_blocked_verdicts_name_each_family() -> None:
    process_rows = _valid_process_rows()
    process_rows[1]["exit_code"] = 1
    partial = _assemble(process_rows=process_rows)

    blocked = _assemble(process_rows=[])

    assert partial["all_mandated_models_loaded_score"] == 0.0
    assert partial["verdict_class"] == "partial"
    assert exp.MANDATED_HF_IDS[0] in partial["honest_verdict"]
    assert exp.MANDATED_HF_IDS[1] in partial["honest_verdict"]
    assert blocked["verdict_class"] == "blocked"
    assert blocked["status"].startswith("blocked_")


# SCENARIO-REPORT-6567-ATTACKS: false positive receipts are disqualified.
def test_false_receipt_disqualifies_artifact() -> None:
    process_rows = _valid_process_rows()
    process_rows[0]["os_pid_verified"] = False
    process_rows[0]["receipt_integrity_failure"] = True

    artifact = _assemble(process_rows=process_rows)

    assert artifact["verdict_class"] == "disqualified"
    assert artifact["status"].startswith("disqualified_")
    assert artifact["all_mandated_models_loaded_score"] == 0.0


# SCENARIO-REPORT-6567-ATTACKS: legacy smoke rows never enter readiness.
def test_legacy_smoke_rows_are_excluded_from_reducer() -> None:
    artifact = _assemble()
    smoke = dict(artifact["per_unit_rows"][0])
    smoke.update(
        {
            "hf_id": "Qwen/Qwen3.5-0.8B",
            "condition": "legacy_cpu_smoke",
            "headline_eligible": False,
            "admitted": False,
        }
    )
    smoke["row_hash"] = exp.row_hash(smoke)
    artifact["per_unit_rows"].append(smoke)
    artifact["aggregate_row_recomputation"] = exp.aggregate_row_recomputation(artifact)

    summary = exp.gate_check_summary(artifact)

    assert artifact["aggregate_row_recomputation"]["required_family_row_count"] == 3
    assert artifact["aggregate_row_recomputation"]["ready_score_from_rows"] == 1.0
    assert all("Qwen3.5" not in row["check"] for row in summary["rows"])


# SCENARIO-REPORT-6567-ATOMIC: checksum, rows, and protected state are revalidated.
def test_validation_detects_checksum_row_and_protected_mutation() -> None:
    artifact = _assemble()
    artifact["per_unit_rows"][0]["admitted"] = False
    artifact["protected_files_unchanged"]["all_unchanged"] = False

    errors = exp.validate_artifact(artifact)

    assert "per_unit_rows row_hash mismatch" in errors
    assert "protected files changed" in errors
    assert "aggregate_row_recomputation mismatch" in errors
    assert "reproducibility_checksum mismatch" in errors


# SCENARIO-REPORT-6567-ATOMIC: every closed schema rule rejects tampering.
def test_validation_detects_schema_and_principle_tampering() -> None:
    missing_provenance = _assemble()
    missing_provenance["field_provenance"].pop("status")
    assert "field_provenance must cover required fields" in exp.validate_artifact(
        missing_provenance
    )

    wrong_principle = _assemble()
    wrong_principle["field_provenance"]["status"]["principle"] = "wrong"
    assert "field principle mismatch" in exp.validate_artifact(wrong_principle)

    malformed = _assemble()
    malformed["status"] = "running"
    malformed["honest_verdict"] = "running"
    malformed["verdict_class"] = "positive"
    malformed["inference_substrate"] = "wrong"
    malformed["verifier_is_oracle"] = False
    malformed["MODEL_SPECS"] = []
    malformed["duration_s"] = -1.0
    errors = exp.validate_artifact(malformed)

    assert "status lacks terminal prefix" in errors
    assert "honest_verdict lacks terminal prefix" in errors
    assert "verdict_class must be null, partial, blocked, or disqualified" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "MODEL_SPECS mandated order mismatch" in errors
    assert "duration_s must be nonnegative" in errors


# SCENARIO-REPORT-6567-ATOMIC: atomic write replaces the target and cleans temp files.
def test_atomic_write_json_replaces_target_without_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    exp.atomic_write_json(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


# SCENARIO-REPORT-6567-ATOMIC: validation CLI accepts the terminal artifact.
def test_main_validate_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "artifact.json"
    exp.atomic_write_json(path, _assemble())

    assert exp.main(["--validate", "--result-path", str(path)]) == 0
    assert "validated" in capsys.readouterr().out
    path.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--result-path", str(path)]) == 1
    assert "required field set mismatch" in capsys.readouterr().out
    assert exp.main(["--validate", "--result-path", str(tmp_path / "missing.json")]) == 1
    assert "artifact not found" in capsys.readouterr().out


# REQ-REPORT-6567: helper hashes remain canonical and missing files fail closed.
def test_hash_and_identity_helpers(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"value")

    assert exp.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert exp.sha256_file(path) == exp.sha256_text("value")
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    assert exp.is_split_gguf(Path("model-00001-of-00003.gguf")) is True
    assert exp.is_split_gguf(Path("model-Q4_K_M.gguf")) is False
    assert exp.quantization_from_name("model-UD-Q4_K_M.gguf") == "Q4_K_M"
    assert exp.quantization_from_name("model.gguf") == "unknown"
    assert exp.revision_from_path(Path("/cache/snapshots/revision-x/model.gguf")) == "revision-x"
    assert exp.revision_from_path(Path("/cache/model.gguf")) == "unknown"
