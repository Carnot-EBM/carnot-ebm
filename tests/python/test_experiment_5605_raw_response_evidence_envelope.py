"""Tests for Exp5605 raw response evidence envelopes.

Spec refs: REQ-VERIFY-5605, SCENARIO-VERIFY-5605.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5605_raw_response_evidence_envelope as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5605_raw_response_evidence_envelope.py")


def _fake_model_specs(tmp_path: Path) -> list[dict[str, object]]:
    qwen = tmp_path / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    gemma = tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    qwen.write_bytes(b"qwen fixture path")
    gemma.write_bytes(b"gemma fixture path")
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": mod.QWEN_ID,
            "family": "qwen",
            "role": "moe",
            "gpu": 0,
            "model_path": str(qwen),
            "headline_eligible": True,
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": mod.GEMMA26_ID,
            "family": "gemma",
            "role": "moe",
            "gpu": 1,
            "model_path": str(gemma),
            "headline_eligible": True,
        },
    ]


def _authenticated_receipt(model_specs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "torch_cuda_available": True,
        "torch_device_count": 2,
        "llama_cpp_supports_gpu_offload": True,
        "gpu_offload_authenticated": True,
        "devices": [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090"},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090"},
        ],
        "model_receipts": [
            {
                "model_hf_id": spec["hf_id"],
                "model_path": spec["model_path"],
                "worker_ok": True,
                "llama_cpp_supports_gpu_offload": True,
                "torch_cuda_available": True,
                "torch_device_count": 1,
                "offloaded_layer_count_from_backend_log": 31,
                "gpu_offload_authenticated": True,
                "stderr_tail": "llama.cpp CUDA offloaded 31/31 layers to GPU",
            }
            for spec in model_specs
        ],
    }


def test_req_verify_5605_spec_declares_append_only_raw_payload_contract() -> None:
    """REQ-VERIFY-5605: OpenSpec anchors raw-response envelope fields and gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5605") : spec.index("### REQ-VERIFY-5580")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5605" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert mod.QWEN_ID in section
    assert mod.GEMMA26_ID in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "Payload hashes alone SHALL NOT satisfy the envelope" in section
    assert "raw response must be locally recoverable byte-for-byte" in normalized
    assert "Legacy CPU models MAY appear only as labeled CPU smoke diagnostics" in section
    assert "corrupt one stored payload" in normalized
    assert "fails closed" in normalized
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in normalized


def test_scenario_verify_5605_envelope_replays_losslessly_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5605: stored bytes replay exactly and tampering fails closed."""

    specs = _fake_model_specs(tmp_path)
    artifact = mod.build_artifact(
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    assert artifact["response_rows_written"] == 8
    assert artifact["raw_payloads_preserved"] is True
    assert artifact["lossless_replay_rate"] == 1.0
    assert artifact["truncation_controls_detected"] == 2
    assert artifact["payload_corruption_rejected"] is True
    assert artifact["semantic_false_accept_count"] == 0
    assert artifact["parser_version_replay_passed"] is True
    assert artifact["gpu_offload_authenticated"] is True
    assert artifact["envelope_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert [row["hf_id"] for row in artifact["model_specs"]] == [mod.QWEN_ID, mod.GEMMA26_ID]

    replay = mod.replay_envelope_rows(artifact["response_envelope_rows"])
    assert replay["lossless_replay_rate"] == 1.0
    assert replay["parser_version_replay_passed"] is True
    assert replay["truncation_controls_detected"] == 2
    assert replay["semantic_false_accept_count"] == 0

    for row in artifact["response_envelope_rows"]:
        assert set(mod.REQUIRED_ROW_FIELDS) <= set(row)
        assert row["payload_hash"] == mod.sha256_bytes(
            mod.decode_lossless_payload(row["raw_response_payload"])
        )

    corrupted = mod.corrupt_first_payload(artifact["response_envelope_rows"])
    with pytest.raises(mod.EnvelopeIntegrityError, match="payload_hash"):
        mod.replay_envelope_rows(corrupted)


def test_req_verify_5605_controls_expose_truncation_without_semantic_false_accept(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5605: truncated and malformed controls are visible, not repaired."""

    specs = _fake_model_specs(tmp_path)
    rows = mod.build_envelope_rows(
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
    )
    replay = mod.replay_envelope_rows(rows)
    invalid_rows = [row for row in replay["rows"] if row["control_kind"] != "known_valid"]

    assert len(rows) == 8
    assert {row["model_family"] for row in invalid_rows} == {"qwen", "gemma"}
    assert sum(1 for row in invalid_rows if row["truncation_flag"]) == 2
    assert all(row["exact_validator_outcome"]["accepted"] is False for row in invalid_rows)
    assert all(row["parsed_object"] is None for row in invalid_rows)
    assert replay["semantic_false_accept_count"] == 0


def test_req_verify_5605_reader_fail_closed_branches_are_exercised(tmp_path: Path) -> None:
    """REQ-VERIFY-5605: replay rejects broken chains, hashes, parser drift, and shape."""

    specs = _fake_model_specs(tmp_path)
    rows = mod.build_envelope_rows(
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
    )
    assert mod.normalize_model_specs([{"hf_id": "Qwen/Qwen3.5-0.8B"}]) == []
    assert mod.model_specs_ready("not-a-spec-list") is False
    assert mod.honest_verdict(False, model_ok=False, gpu_ok=False).startswith(
        "blocked_missing_mandated"
    )
    assert mod.honest_verdict(False, model_ok=True, gpu_ok=True).startswith(
        "blocked_raw_response"
    )
    missing_receipt_rows = mod.build_envelope_rows(model_specs=specs, device_receipt={})
    assert missing_receipt_rows[0]["device_offload_receipt"]["receipt_missing"] is True

    previous_hash_bad = copy.deepcopy(rows)
    previous_hash_bad[1]["previous_row_hash"] = "wrong"
    with pytest.raises(mod.EnvelopeIntegrityError, match="previous_row_hash"):
        mod.replay_envelope_rows(previous_hash_bad)

    row_hash_bad = copy.deepcopy(rows)
    row_hash_bad[0]["call_id"] = "tampered-call-id"
    with pytest.raises(mod.EnvelopeIntegrityError, match="row_hash"):
        mod.replay_envelope_rows(row_hash_bad)

    prompt_hash_bad = copy.deepcopy(rows)
    prompt_hash_bad[0]["prompt_hash"] = "0" * 64
    prompt_hash_bad[0]["row_hash"] = mod.row_hash(prompt_hash_bad[0])
    with pytest.raises(mod.EnvelopeIntegrityError, match="prompt_hash"):
        mod.replay_envelope_rows(prompt_hash_bad)

    parser_bad = copy.deepcopy(rows)
    parser_bad[0]["parser_version"] = "old-parser"
    parser_bad[0]["row_hash"] = mod.row_hash(parser_bad[0])
    with pytest.raises(mod.EnvelopeIntegrityError, match="parser_version"):
        mod.replay_envelope_rows(parser_bad)

    parsed_object_bad = copy.deepcopy(rows)
    parsed_object_bad[0]["parsed_object"] = {"label": "invalid"}
    parsed_object_bad[0]["row_hash"] = mod.row_hash(parsed_object_bad[0])
    with pytest.raises(mod.EnvelopeIntegrityError, match="parsed_object"):
        mod.replay_envelope_rows(parsed_object_bad)

    exact_bad = copy.deepcopy(rows)
    exact_bad[0]["exact_validator_outcome"] = dict(exact_bad[0]["exact_validator_outcome"])
    exact_bad[0]["exact_validator_outcome"]["accepted"] = False
    exact_bad[0]["row_hash"] = mod.row_hash(exact_bad[0])
    with pytest.raises(mod.EnvelopeIntegrityError, match="exact_validator_outcome"):
        mod.replay_envelope_rows(exact_bad)

    shape_bad = copy.deepcopy(rows)
    shape_bad[0]["prompt_payload"] = None
    shape_bad[0]["row_hash"] = mod.row_hash(shape_bad[0])
    with pytest.raises(mod.EnvelopeIntegrityError, match="prompt_payload"):
        mod.replay_envelope_rows(shape_bad)


def test_req_verify_5605_artifact_validation_blocks_cpu_fallback_and_overclaims(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5605: readiness fails closed without CUDA/offload and controls."""

    specs = _fake_model_specs(tmp_path)
    cpu_receipt = dict(_authenticated_receipt(specs))
    cpu_receipt["gpu_offload_authenticated"] = False
    cpu_receipt["torch_cuda_available"] = False
    cpu_receipt["llama_cpp_supports_gpu_offload"] = False

    artifact = mod.build_artifact(
        model_specs=specs,
        device_receipt=cpu_receipt,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    assert artifact["gpu_offload_authenticated"] is False
    assert artifact["envelope_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked_")

    overclaim = dict(artifact)
    overclaim["envelope_ready"] = True
    with pytest.raises(ValueError, match="envelope_ready"):
        mod.validate_artifact(overclaim)

    broken_payload_gate = dict(artifact)
    broken_payload_gate["raw_payloads_preserved"] = False
    with pytest.raises(ValueError, match="raw_payloads_preserved"):
        mod.validate_artifact(broken_payload_gate)

    broken_models = dict(artifact)
    broken_models["model_specs"] = [{"hf_id": "Qwen/Qwen3.5-0.8B", "model_path": "legacy"}]
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(broken_models)


def test_scenario_verify_5605_run_writes_deliverable(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5605: runner writes stable JSON at the requested path."""

    specs = _fake_model_specs(tmp_path)
    output = tmp_path / "experiment_5605.json"
    artifact = mod.run(
        result_path=output,
        model_specs=specs,
        device_receipt=_authenticated_receipt(specs),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
