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
