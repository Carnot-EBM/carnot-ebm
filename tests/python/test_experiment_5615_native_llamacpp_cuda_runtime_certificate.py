"""Tests for Exp5615 native llama.cpp CUDA runtime certificate.

Spec refs: REQ-VERIFY-5615, SCENARIO-VERIFY-5615.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5615_native_llamacpp_cuda_runtime_certificate as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5615_native_llamacpp_cuda_runtime_certificate.py")


def _fake_model_specs(tmp_path: Path) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for index, hf_id in enumerate(mod.MANDATED_HEADLINE_IDS):
        stem = hf_id.rsplit("/", 1)[-1].replace("-GGUF", "")
        path = tmp_path / f"{stem}-Q4_K_M.gguf"
        path.write_bytes(b"GGUF" + index.to_bytes(1, "little") + hf_id.encode())
        specs.append(
            {
                "name": stem,
                "hf_id": hf_id,
                "family": mod.model_family(hf_id),
                "role": "dense" if "31B" in hf_id else "moe",
                "gpu": index % 2,
                "model_path": str(path),
                "headline_eligible": True,
            }
        )
    return mod.normalize_model_specs(specs)


def _preconditions(*, ready: bool = True) -> dict[str, object]:
    binary_path = "/opt/llama.cpp/build/bin/llama-cli" if ready else None
    return {
        "native_binary_receipt": {
            "kind": "llama-cli",
            "path": binary_path,
            "executable": ready,
            "sha256": "abc123" if ready else "",
            "version": {"ok": ready, "stdout": "version: 9606 CUDA", "stderr": ""},
            "help": {
                "ok": ready,
                "stdout": "--single-turn\n--gpu-layers\n--json-schema\n",
                "stderr": "",
                "contains_single_turn": ready,
                "contains_gpu_layers": ready,
                "contains_json_schema": ready,
            },
            "dynamic_libraries": {
                "ok": ready,
                "stdout": "libggml-cuda.so\nlibcuda.so\nlibcublas.so",
                "stderr": "",
            },
            "list_devices": {
                "ok": ready,
                "stdout": "CUDA0: NVIDIA GeForce RTX 3090\nCUDA1: NVIDIA GeForce RTX 3090",
                "stderr": "",
            },
        },
        "cuda_build_capability": {
            "cuda_backend_linked": ready,
            "list_devices_reports_cuda": ready,
            "help_reports_gpu_layers": ready,
            "native_cuda_ready": ready,
            "missing_preconditions": [] if ready else ["native_llama_cpp_binary_unavailable"],
        },
        "gpu_device_receipts": {
            "before": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "driver_version": "610.43.03",
                    "memory_free_mb": 24120,
                    "memory_used_mb": 4,
                    "utilization_gpu_pct": 0,
                }
            ],
            "after": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "driver_version": "610.43.03",
                    "memory_free_mb": 24120,
                    "memory_used_mb": 4,
                    "utilization_gpu_pct": 0,
                }
            ],
            "nvidia_smi": {"ok": ready, "stdout": "0, NVIDIA GeForce RTX 3090, 610.43.03"},
        },
        "blocked_preconditions": [] if ready else ["native_llama_cpp_binary_unavailable"],
    }


def _runner(**kwargs: object) -> dict[str, object]:
    control = kwargs["control"]
    model_spec = kwargs["model_spec"]
    command = mod.build_native_cli_command(
        binary_path="/opt/llama.cpp/build/bin/llama-cli",
        model_path=str(model_spec["model_path"]),
        prompt=str(control["prompt"]),
        control_kind=str(control["control_kind"]),
        seed=int(control["seed"]),
    )
    if control["control_kind"] == "positive_control":
        raw = '{"certificate_control":"ok"}'
        stop_reason = "stop_sequence"
        truncation = False
        completion_tokens = 4
    else:
        raw = "{"
        stop_reason = "length"
        truncation = True
        completion_tokens = 1
    return {
        "model_hf_id": model_spec["hf_id"],
        "control_kind": control["control_kind"],
        "prompt": control["prompt"],
        "raw_response": raw,
        "command": command,
        "sampling_parameters": dict(control["sampling_parameters"]),
        "token_counts": {
            "prompt_tokens": len(str(control["prompt"]).split()),
            "completion_tokens": completion_tokens,
            "total_tokens": len(str(control["prompt"]).split()) + completion_tokens,
            "source": "fixture",
        },
        "stop_reason": stop_reason,
        "truncation_flag": truncation,
        "returncode": 0,
        "exit_status": "completed",
        "wall_time_s": 1.25,
        "pid": 12345,
        "port": None,
        "requested_offload_layers": "all",
        "observed_offloaded_layers": 41,
        "observed_total_layers": 41,
        "gpu_memory_before": [{"index": 0, "memory_used_mb": 4}],
        "gpu_memory_during": [[{"index": 0, "memory_used_mb": 4096}]],
        "gpu_memory_after": [{"index": 0, "memory_used_mb": 4}],
        "gpu_memory_delta_mb": 4092,
        "stdout_tail": raw,
        "stderr_tail": "llama_model_load_tensors: offloaded 41/41 layers to GPU\nCUDA0",
    }


def _build_ready_artifact(tmp_path: Path) -> dict[str, object]:
    specs = _fake_model_specs(tmp_path)
    controls = mod.run_native_controls(
        model_specs=specs,
        native_binary_receipt=_preconditions()["native_binary_receipt"],
        native_runner=_runner,
    )
    rows = mod.build_response_envelope_rows(control_results=controls, model_specs=specs)
    return mod.build_artifact(
        model_specs=specs,
        preconditions=_preconditions(),
        control_results=controls,
        evidence_rows=rows,
        response_envelope_path="results/exp5615.responses.jsonl",
        orphan_process_count=0,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )


def test_req_verify_5615_spec_declares_native_certificate_contract() -> None:
    """REQ-VERIFY-5615: OpenSpec anchors native runtime, envelope, and gate fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5615") : spec.index("### REQ-VERIFY-5606")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5615" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "`--single-turn`" in section
    assert "SHALL NOT rerun, summarize, infer, or compare solve-versus-verify task accuracy" in normalized
    assert "denominator three" in section
    for hf_id in mod.MANDATED_HEADLINE_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in normalized


def test_scenario_verify_5615_complete_certificate_preserves_lossless_controls(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5615: all three native CUDA controls certify only with replay."""

    artifact = _build_ready_artifact(tmp_path)

    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["models_certified_count"] == 3
    assert artifact["runtime_certificate_ready_score"] == 1.0
    assert artifact["lossless_replay_rate"] == 1.0
    assert artifact["stop_control_pass_rate"] == 1.0
    assert artifact["semantic_false_accept_count"] == 0
    assert artifact["orphan_process_count"] == 0
    assert artifact["honest_verdict"].startswith("complete:")
    assert list(artifact["offload_layers_by_model"]) == list(mod.MANDATED_HEADLINE_IDS)
    assert all(row["requested"] == "all" for row in artifact["offload_layers_by_model"].values())
    assert all(row["observed"] > 0 for row in artifact["offload_layers_by_model"].values())
    assert all(delta > 0 for delta in artifact["gpu_memory_delta_by_model"].values())
    assert [row["hf_id"] for row in artifact["model_specs"]] == list(mod.MANDATED_HEADLINE_IDS)
    assert len(artifact["response_envelope_rows"]) == 6
    assert all(
        "--single-turn" in row["native_process_receipt"]["command"]
        for row in artifact["response_envelope_rows"]
    )

    replay = mod.replay_response_envelope_rows(artifact["response_envelope_rows"])
    assert replay["lossless_replay_rate"] == 1.0
    assert replay["stop_control_pass_rate"] == 1.0
    assert replay["semantic_false_accept_count"] == 0
    mod.validate_artifact(artifact)


def test_req_verify_5615_blocked_preconditions_do_not_load_models(tmp_path: Path) -> None:
    """REQ-VERIFY-5615: missing native CUDA path emits a terminal blocked artifact."""

    calls: list[str] = []

    def forbidden_runner(**kwargs: object) -> dict[str, object]:
        calls.append(str(kwargs["model_spec"]["hf_id"]))
        raise AssertionError("native runner must not run when preconditions are blocked")

    artifact = mod.run(
        result_path=tmp_path / "blocked.json",
        response_envelope_path=tmp_path / "blocked.jsonl",
        model_specs=_fake_model_specs(tmp_path),
        preconditions=_preconditions(ready=False),
        native_runner=forbidden_runner,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    assert calls == []
    assert artifact["models_certified_count"] == 0
    assert artifact["runtime_certificate_ready_score"] == 0.0
    assert artifact["lossless_replay_rate"] == 0.0
    assert artifact["stop_control_pass_rate"] == 0.0
    assert artifact["honest_verdict"].startswith("blocked_native_preconditions")
    assert artifact["blocked_preconditions"] == ["native_llama_cpp_binary_unavailable"]
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)


def test_req_verify_5615_fail_closed_on_cpu_fallback_and_tampering(tmp_path: Path) -> None:
    """REQ-VERIFY-5615: zero GPU delta, false accepts, bad rows, and orphans block."""

    artifact = _build_ready_artifact(tmp_path)
    controls = copy.deepcopy(artifact["control_results"])
    controls[0]["native_process_receipt"]["gpu_memory_delta_mb"] = 0
    controls[0]["native_process_receipt"]["observed_offloaded_layers"] = 0
    rows = mod.build_response_envelope_rows(
        control_results=controls,
        model_specs=artifact["model_specs"],
    )
    blocked = mod.build_artifact(
        model_specs=artifact["model_specs"],
        preconditions=_preconditions(),
        control_results=controls,
        evidence_rows=rows,
        response_envelope_path="blocked.responses.jsonl",
        orphan_process_count=1,
    )

    assert blocked["models_certified_count"] == 2
    assert blocked["runtime_certificate_ready_score"] < 1.0
    assert blocked["orphan_process_count"] == 1
    assert blocked["honest_verdict"].startswith("blocked_")
    mod.validate_artifact(blocked)

    false_accept_row = None
    for row in copy.deepcopy(artifact["response_envelope_rows"]):
        if row["control_kind"] == "truncated_control":
            raw = b'{"certificate_control":"ok"}'
            row["previous_row_hash"] = ""
            row["raw_response_payload"] = mod.encode_lossless_payload(raw)
            row["payload_hash"] = mod.sha256_bytes(raw)
            row["parsed_object"] = {"certificate_control": "ok"}
            row["exact_control_outcome"] = {
                "validator": "exp5615_certificate_control_v1",
                "accepted": True,
                "expected_control": "truncated_control",
                "observed_control": "unexpected_accept",
                "parser_ok": True,
                "parser_error_type": "",
                "control_passed": False,
            }
            row["exact_validator_outcome"] = dict(row["exact_control_outcome"])
            row["row_hash"] = mod.row_hash(row)
            false_accept_row = row
            break
    assert false_accept_row is not None
    replay = mod.replay_response_envelope_rows([false_accept_row])
    assert replay["semantic_false_accept_count"] == 1

    tampered = copy.deepcopy(artifact["response_envelope_rows"])
    tampered[0]["raw_response_payload"] = mod.encode_lossless_payload(b'{"tampered":true}')
    tampered[0]["row_hash"] = mod.row_hash(tampered[0])
    with pytest.raises(mod.EnvelopeReplayError, match="payload_hash"):
        mod.replay_response_envelope_rows(tampered)


def test_scenario_verify_5615_run_writes_artifact_and_envelope(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5615: injected run writes stable JSON and replayable JSONL."""

    result_path = tmp_path / "experiment_5615.json"
    envelope_path = tmp_path / "experiment_5615.responses.jsonl"
    artifact = mod.run(
        result_path=result_path,
        response_envelope_path=envelope_path,
        model_specs=_fake_model_specs(tmp_path),
        preconditions=_preconditions(),
        native_runner=_runner,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    assert result_path.is_file()
    assert envelope_path.is_file()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert mod.replay_response_envelope_path(envelope_path)["row_count"] == 6
    assert artifact["reproducibility_checksum"] == mod.artifact_checksum(artifact)
