"""Contract tests for the Qwen3.8 bounded runtime cutover receipt.

Spec refs: REQ-VERIFY-7157 and SCENARIO-VERIFY-7157-*.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from carnot import experiment_7157_v630_qwen38_runtime as exp


def _passed_checks() -> list[dict]:
    return [exp.gate_row(name, True, True, True) for name in exp.READINESS_CHECKS]


def _complete_candidate() -> dict:
    artifact = exp.base_artifact(exp.RUN_DATE)
    artifact["source_artifact_hashes"] = deepcopy(exp.REQUIRED_SOURCE_HASHES)
    artifact["MODEL_SPECS"] = [
        {
            "name": "Qwen3.8-27B",
            "hf_id": exp.QWEN_MODEL_ID,
            "gpu": 1,
            "model_path": "/cache/Qwen3.8-27B-Q4_K_M.gguf",
            "selection_role": "current_headline",
            "preferred_quant": "Q4_K_M",
            "remote_allowed": False,
            "resolution_method": "cached_current_model",
            "loaded_path": "/blobs/qwen38",
            "revision": "revision-38",
            "size_bytes": 16_500_000_000,
            "sha256": "sha256:model38",
            "chat_template_source": "embedded_gguf",
            "chat_template_sha256": "sha256:template38",
        }
    ]
    artifact["model_identity_rows"] = [
        {
            "model_id": exp.QWEN_MODEL_ID,
            "repository": exp.QWEN_MODEL_ID,
            "filename": "Qwen3.8-27B-Q4_K_M.gguf",
            "loaded_path": "/blobs/qwen38",
            "revision": "revision-38",
            "size_bytes": 16_500_000_000,
            "sha256": "sha256:model38",
            "template_source": "embedded_gguf",
            "template_sha256": "sha256:template38",
            "template_present": True,
        }
    ]
    server_log = "CUDA0 CUDA : ARCHS"
    artifact["model_load_receipts"] = [
        {
            "model_id": exp.QWEN_MODEL_ID,
            "model_sha256": "sha256:model38",
            "command": ["llama-server", "--n-gpu-layers", "all"],
            "pid": 3800,
            "pid_owned_by_task": True,
            "health_ok": True,
            "startup_duration_s": 11.0,
            "binary_linkage": {
                "cuda_linkage_confirmed": True,
                "libggml_cuda_linked": True,
                "libcuda_linked": True,
            },
            "native_cuda_markers": ["CUDA0", "CUDA : ARCHS"],
            "requested_gpu_layers": "all",
            "task_owned_vram_delta_mb": 16_100,
            "server_log": server_log,
            "server_log_sha256": exp.sha256_text(server_log),
        }
    ]
    prompt = exp.CANARY_PROMPT
    raw_output = "Qwen3.8 emitted a bounded canary."
    raw_response = {"choices": [{"message": {"content": raw_output}}]}
    artifact["generation_receipts"] = [
        {
            "model_id": exp.QWEN_MODEL_ID,
            "prompt": prompt,
            "prompt_sha256": exp.sha256_text(prompt),
            "request": exp.canary_payload(),
            "request_sha256": exp.sha256_text(exp.canonical_json(exp.canary_payload())),
            "raw_output": raw_output,
            "raw_output_sha256": exp.sha256_text(raw_output),
            "parsed_output": exp.parse_canary_output(raw_output),
            "raw_response": raw_response,
            "raw_response_sha256": exp.sha256_text(exp.canonical_json(raw_response)),
            "prompt_tokens": 24,
            "completion_tokens": 9,
            "total_tokens": 33,
            "latency_s": 0.8,
            "gpu_snapshot_phase": "after_generation",
            "server_log_sha256": exp.sha256_text(server_log),
        }
    ]
    artifact["gpu_telemetry_rows"] = [
        {
            "phase": "before",
            "selected_gpu_index": 1,
            "selected_gpu_name": "NVIDIA GeForce RTX 3090",
            "selected_gpu_memory_used_mb": 4,
            "task_pid_memory_mb": 0,
            "task_pid_present": False,
        },
        {
            "phase": "model_loaded",
            "selected_gpu_index": 1,
            "selected_gpu_name": "NVIDIA GeForce RTX 3090",
            "selected_gpu_memory_used_mb": 16_104,
            "task_pid_memory_mb": 16_100,
            "task_pid_present": True,
        },
        {
            "phase": "after_generation",
            "selected_gpu_index": 1,
            "selected_gpu_name": "NVIDIA GeForce RTX 3090",
            "selected_gpu_memory_used_mb": 16_104,
            "task_pid_memory_mb": 16_100,
            "task_pid_present": True,
        },
        {
            "phase": "after_teardown",
            "selected_gpu_index": 1,
            "selected_gpu_name": "NVIDIA GeForce RTX 3090",
            "selected_gpu_memory_used_mb": 4,
            "task_pid_memory_mb": 0,
            "task_pid_present": False,
        },
    ]
    artifact["server_lease_rows"] = [
        {
            "pid": 3800,
            "owned_by_task": True,
            "recorded_identity_present": True,
            "cleanup_action": "terminated",
            "cleanup_bounded": True,
            "cleanup_leak_free": True,
            "signals_sent": [{"target": "process_group", "signal": "SIGTERM"}],
            "signaled_pid": 3800,
            "pid_released": True,
            "vram_released": True,
            "unrelated_process_kill_count_delta": 0,
        }
    ]
    artifact["rows"] = exp.typed_rows(artifact)
    return artifact


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-FIRST-WRITE.
def test_base_artifact_is_schema_complete_and_bounded_no_run() -> None:
    artifact = exp.base_artifact(exp.RUN_DATE)

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert set(exp.FIELD_PRINCIPLES) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "running"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["MODEL_SPECS"] == [exp.HEADLINE_MODEL_SPEC]


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-FIRST-WRITE.
def test_blocked_artifact_is_terminal_and_names_exact_gate(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    running = exp.initialize_artifact(target, exp.RUN_DATE)
    checks = [exp.gate_row("idle_rtx_3090", "one idle RTX 3090", [], False)]

    blocked = exp.finish_blocked(running, target, checks, duration_s=0.25)

    assert blocked["status"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"] == checks[0]
    assert blocked["honest_verdict"] == "blocked_idle_rtx_3090"
    assert exp.validate_artifact(target) == []


# REQ-VERIFY-7157 / SCENARIO-INFER-SOTA-7157-COMPARATORS.
def test_registry_rows_separate_current_mandate_from_comparators() -> None:
    rows = exp.registry_cutover_rows()

    assert rows[0]["hf_id"] == exp.QWEN_MODEL_ID
    assert rows[0]["selection_role"] == "current_headline"
    assert {row["hf_id"] for row in rows[1:]} == set(exp.LEGACY_COMPARATOR_IDS)
    assert all(row["selection_role"] == "legacy_comparator" for row in rows[1:])


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-BOUNDED-GENERATION.
def test_completed_receipt_is_bounded_runtime_only() -> None:
    candidate = _complete_candidate()

    result = exp.finalize_artifact(candidate, _passed_checks(), duration_s=12.0)

    assert result["status"] == "completed"
    assert result["inference_substrate"] == "live_llm_inference"
    assert result["inference_substrate_class"] == "model_bounded_generation"
    assert result["qwen38_runtime_ready_score"] == 1
    assert result["verifier_is_oracle"] is False
    assert "quality" not in result["honest_verdict"]
    assert exp.validate_artifact(result) == []


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-COLD-VALIDATION.
@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("inference_substrate_class", "model_full_generation", "positive_substrate_class"),
        ("qwen38_runtime_ready_score", 0, "positive_readiness_score"),
        ("duration_s", 9.99, "bounded_duration_floor"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ],
)
def test_cold_validation_rejects_claim_mutations(
    field: str, value: object, expected_error: str
) -> None:
    result = exp.finalize_artifact(_complete_candidate(), _passed_checks(), duration_s=12.0)
    result[field] = value
    result["reproducibility_checksum"] = exp.artifact_checksum(result)

    assert expected_error in exp.validate_artifact(result)


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-OWNED-TEARDOWN.
@pytest.mark.parametrize(
    ("container", "field", "value", "expected_error"),
    [
        ("model_load_receipts", "task_owned_vram_delta_mb", 0, "task_owned_vram_missing"),
        ("generation_receipts", "completion_tokens", 0, "completion_tokens_missing"),
        ("server_lease_rows", "pid_released", False, "task_pid_not_released"),
        (
            "server_lease_rows",
            "unrelated_process_kill_count_delta",
            1,
            "unrelated_process_signaled",
        ),
    ],
)
def test_cold_validation_rejects_receipt_mutations(
    container: str, field: str, value: object, expected_error: str
) -> None:
    result = exp.finalize_artifact(_complete_candidate(), _passed_checks(), duration_s=12.0)
    result[container][0][field] = value
    result["rows"] = exp.typed_rows(result)
    result["reproducibility_checksum"] = exp.artifact_checksum(result)

    assert expected_error in exp.validate_artifact(result)


# REQ-VERIFY-7157: the GGUF repository must never use a Transformers tokenizer.
def test_runtime_source_never_calls_transformers_tokenizer() -> None:
    assert exp.__file__ is not None
    source = Path(exp.__file__).read_text(encoding="utf-8")
    forbidden = "Auto" + "Tokenizer"

    assert forbidden not in source


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-COLD-VALIDATION.
def test_positive_evidence_validator_names_every_missing_receipt() -> None:
    candidate = _complete_candidate()
    candidate.update(
        {
            "MODEL_SPECS": [],
            "source_artifact_hashes": {},
            "registry_cutover_rows": [],
            "model_identity_rows": [],
            "model_load_receipts": [],
            "generation_receipts": [],
            "gpu_telemetry_rows": [],
            "server_lease_rows": [],
            "rows": [{"row_type": "unexpected"}],
        }
    )

    errors = exp._positive_evidence_errors(candidate)

    assert {
        "headline_model_spec_mismatch",
        "source_artifact_hashes_mismatch",
        "registry_cutover_rows_mismatch",
        "model_identity_missing",
        "model_load_receipt_missing",
        "generation_receipt_missing",
        "gpu_phase_receipts_missing",
        "server_lease_receipt_missing",
        "typed_rows_mismatch",
    }.issubset(errors)


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-COLD-VALIDATION.
def test_positive_evidence_validator_names_corrupt_receipt_fields() -> None:
    candidate = _complete_candidate()
    spec = candidate["MODEL_SPECS"][0]
    spec.update(
        {
            "model_path": "/cache/wrong.gguf",
            "selection_role": "legacy_comparator",
            "resolution_method": "remote",
            "remote_allowed": True,
            "chat_template_source": "external",
        }
    )
    identity = candidate["model_identity_rows"][0]
    identity.update(
        {
            "repository": "wrong/model",
            "filename": "wrong.gguf",
            "revision": "",
            "size_bytes": 0,
            "sha256": "",
            "template_source": "external",
            "template_present": False,
        }
    )
    load = candidate["model_load_receipts"][0]
    load.update(
        {
            "model_id": "wrong/model",
            "pid_owned_by_task": False,
            "requested_gpu_layers": "0",
            "binary_linkage": {},
            "native_cuda_markers": [],
            "task_owned_vram_delta_mb": 0,
            "server_log_sha256": "wrong-generation-log",
        }
    )
    generation = candidate["generation_receipts"][0]
    generation.update(
        {
            "prompt": "wrong",
            "request": {},
            "raw_output": "",
            "parsed_output": {},
            "raw_response_sha256": "wrong",
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_s": 0,
            "server_log_sha256": "wrong",
        }
    )
    loaded_gpu = next(
        row for row in candidate["gpu_telemetry_rows"] if row["phase"] == "model_loaded"
    )
    loaded_gpu.update(
        {
            "selected_gpu_name": "CPU",
            "task_pid_present": False,
            "task_pid_memory_mb": 0,
        }
    )
    teardown_gpu = next(
        row for row in candidate["gpu_telemetry_rows"] if row["phase"] == "after_teardown"
    )
    teardown_gpu.update({"task_pid_present": True, "task_pid_memory_mb": 1})
    lease = candidate["server_lease_rows"][0]
    lease.update(
        {
            "owned_by_task": False,
            "recorded_identity_present": False,
            "cleanup_bounded": False,
            "cleanup_leak_free": False,
            "pid_released": False,
            "vram_released": False,
            "unrelated_process_kill_count_delta": 1,
        }
    )
    candidate["rows"] = []

    errors = exp._positive_evidence_errors(candidate)

    assert {
        "exact_q4_model_path_mismatch",
        "headline_selection_role_mismatch",
        "canonical_cache_resolution_missing",
        "remote_fallback_enabled",
        "embedded_template_source_missing",
        "model_repository_mismatch",
        "model_filename_mismatch",
        "model_file_identity_incomplete",
        "model_hash_missing",
        "embedded_template_missing",
        "model_load_unconfirmed",
        "task_pid_ownership_missing",
        "all_layer_request_missing",
        "native_cuda_linkage_missing",
        "native_cuda_markers_missing",
        "task_owned_vram_missing",
        "server_log_hash_mismatch",
        "prompt_hash_mismatch",
        "generation_request_mismatch",
        "raw_output_missing_or_changed",
        "parsed_output_mismatch",
        "raw_response_hash_mismatch",
        "completion_tokens_missing",
        "prompt_tokens_missing",
        "generation_latency_missing",
        "generation_log_link_mismatch",
        "selected_gpu_not_rtx_3090",
        "task_owned_gpu_telemetry_missing",
        "task_owned_vram_not_released",
        "task_lease_ownership_missing",
        "task_cleanup_failed",
        "task_pid_not_released",
        "task_vram_not_released",
        "unrelated_process_signaled",
        "typed_rows_mismatch",
    }.issubset(errors)


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-COLD-VALIDATION.
def test_finalize_refuses_incomplete_positive_receipts() -> None:
    with pytest.raises(ValueError, match="positive runtime evidence is incomplete"):
        exp.finalize_artifact(exp.base_artifact(exp.RUN_DATE), _passed_checks(), duration_s=12)


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-COLD-VALIDATION.
def test_validator_rejects_unreadable_nonobject_and_wrong_shape(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    unreadable = tmp_path / "unreadable.json"
    nonobject = tmp_path / "nonobject.json"
    unreadable.write_text("{", encoding="utf-8")
    nonobject.write_text("[]", encoding="utf-8")

    assert exp.validate_artifact(missing) == ["artifact_missing"]
    assert exp.validate_artifact(unreadable) == ["artifact_unreadable"]
    assert exp.validate_artifact(nonobject) == ["artifact_not_object"]
    assert exp.validate_artifact(7) == ["artifact_not_object"]
    assert exp.validate_artifact({"extra": True}) == ["artifact_fields_mismatch"]


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-COLD-VALIDATION.
def test_validator_names_top_level_and_positive_state_corruption() -> None:
    result = exp.finalize_artifact(_complete_candidate(), _passed_checks(), duration_s=12)
    result.update(
        {
            "field_principles": {},
            "run_date": "20260908",
            "execution_venue": "remote",
            "random_seed": 0,
            "status": "running",
            "inference_substrate": "no_inference",
            "honest_verdict": "wrong_quality_claim",
            "gate_check_summary": {},
            "reproducibility_checksum": "wrong",
        }
    )

    errors = exp.validate_artifact(result)

    assert {
        "field_principles_mismatch",
        "run_date_mismatch",
        "execution_venue_mismatch",
        "random_seed_mismatch",
        "honest_verdict_prefix_mismatch",
        "quality_claim_present",
        "gate_check_summary_mismatch",
        "reproducibility_checksum_mismatch",
        "positive_status",
        "positive_inference_substrate",
        "positive_gate_summary",
    }.issubset(errors)


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-COLD-VALIDATION.
def test_validator_rejects_invalid_terminal_class() -> None:
    result = exp.finalize_artifact(_complete_candidate(), _passed_checks(), duration_s=12)
    result["verdict_class"] = "unexpected"
    result["reproducibility_checksum"] = exp.artifact_checksum(result)

    errors = exp.validate_artifact(result)

    assert "verdict_class_invalid" in errors
    assert "terminal_verdict_class_invalid" in errors


# REQ-VERIFY-7157 / SCENARIO-VERIFY-7157-COLD-VALIDATION.
def test_validator_checks_blocked_and_disqualified_terminal_shapes(tmp_path: Path) -> None:
    blocked_path = tmp_path / "blocked.json"
    failed = [exp.gate_row("cache", True, False, False)]
    blocked = exp.finish_blocked(
        exp.base_artifact(exp.RUN_DATE), blocked_path, failed, duration_s=1
    )
    blocked.update(
        {
            "status": "running",
            "inference_substrate": exp.INFERENCE_SUBSTRATE,
            "inference_substrate_class": "model_bounded_generation",
            "qwen38_runtime_ready_score": 1,
            "gate_check_summary": {"passed": True},
        }
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked)
    assert {
        "blocked_status",
        "blocked_inference_substrate",
        "blocked_substrate_class",
        "blocked_readiness_score",
        "blocked_gate_summary",
    }.issubset(blocked_errors)

    disqualified = exp.base_artifact(exp.RUN_DATE)
    disqualified.update(
        {
            "status": "running",
            "preconditions_checked": failed,
            "inference_substrate": "no_inference",
            "inference_substrate_class": "blocked_no_run",
            "qwen38_runtime_ready_score": 1,
            "gate_check_summary": {"passed": True},
            "verdict_class": "disqualified",
            "honest_verdict": "disqualified_runtime_failure",
        }
    )
    disqualified["reproducibility_checksum"] = exp.artifact_checksum(disqualified)
    disqualified_errors = exp.validate_artifact(disqualified)
    assert {
        "disqualified_status",
        "disqualified_inference_substrate",
        "disqualified_substrate_class",
        "disqualified_readiness_score",
        "disqualified_gate_summary",
    }.issubset(disqualified_errors)
