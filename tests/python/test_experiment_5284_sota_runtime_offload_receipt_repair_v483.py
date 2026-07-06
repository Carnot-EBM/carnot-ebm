"""Tests for Exp 5284 SOTA runtime/offload receipt repair.

Spec refs: REQ-VERIFY-5284, SCENARIO-VERIFY-5284.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5284_sota_runtime_offload_receipt_repair_v483 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def _fake_gpu_receipts(*, offload_supported: bool = True, gpu_visible: bool = True) -> dict[str, Any]:
    return {
        "value": {
            "gpu_visible": gpu_visible,
            "nvidia_smi": {"ok": gpu_visible, "stdout": "0, NVIDIA RTX 3090, 24576, 24000"},
            "cuda_runtime": {"ok": gpu_visible, "stdout": "CUDA UMD Version: 13.3"},
            "rocm_smi": {"ok": False, "stderr": "not installed"},
            "torch_cuda": {"import_ok": True, "available": gpu_visible, "device_count": 2},
            "llama_cpp": {
                "import_ok": True,
                "version": "0.3.29",
                "origin": "/venv/llama_cpp/__init__.py",
                "gpu_offload_supported": offload_supported,
            },
            "offload_settings": dict(mod.OFFLOAD_CONFIG),
        },
        "principle": mod.FIELD_PRINCIPLES["gpu_offload_receipts"],
    }


def _cached_pair_provider(*, gpu_indices: tuple[int, int]) -> list[dict[str, Any]]:
    assert gpu_indices == (0, 1)
    return [
        {"hf_id": mod.MANDATED_MODEL_SPECS[0]["hf_id"], "model_path": "/cache/qwen.gguf"},
        {"hf_id": mod.MANDATED_MODEL_SPECS[2]["hf_id"], "model_path": "/cache/gemma26.gguf"},
    ]


def _write_minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def test_req_verify_5284_spec_declares_runtime_offload_gate() -> None:
    """REQ-VERIFY-5284: OpenSpec anchors the v483 offload receipt gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5284") : spec.index("### REQ-VERIFY-5272")]

    for marker in (
        "REQ-VERIFY-5284",
        "SCENARIO-VERIFY-5284",
        str(mod.RESULT_RELATIVE_PATH),
        "live_llm_inference_local_gguf_sota",
        "blocked_preconditions_with_no_quality_claim",
        "sota_offload_ready",
        "smoke_test_not_headline",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_verify_5284_blocks_without_local_mandated_model(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5284: no local mandated GGUF blocks without fake generation."""

    probe_calls: list[str] = []

    def forbidden_generation_probe(**kwargs: Any) -> dict[str, Any]:
        probe_calls.append(str(kwargs["model_spec"]["model_path"]))
        raise AssertionError("generation probe must not run without a local GGUF")

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=lambda _hf_id, _quant: None,
        cached_pair_provider=lambda *, gpu_indices: [],
        gpu_receipts_provider=_fake_gpu_receipts,
        generation_probe=forbidden_generation_probe,
        smoke_tests_provider=lambda: [{"label": "smoke_test_not_headline", "status": "passed"}],
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
        write=True,
    )

    assert probe_calls == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["inference_substrate"]["value"] == mod.BLOCKED_INFERENCE_SUBSTRATE
    assert artifact["sota_offload_ready"] is False
    assert artifact["no_quality_claim"]["value"] is True
    assert artifact["smoke_tests"]["value"][0]["label"] == "smoke_test_not_headline"
    assert (
        artifact["preconditions_checked"]["value"][
            "at_least_one_mandated_model_resolved_without_autotokenizer"
        ]
        is False
    )
    assert artifact["MODEL_SPECS"]["value"]["flagship_moe"]["status"] == "missing_local_gguf"
    mod.validate_artifact(artifact)


def test_scenario_verify_5284_sets_ready_after_live_generation_with_offload(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5284: one mandated live offload receipt opens the gate."""

    gguf = _write_minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    calls: list[str] = []

    def resolver(hf_id: str, _quant: str) -> str | None:
        return str(gguf) if hf_id == mod.MANDATED_MODEL_SPECS[1]["hf_id"] else None

    def generation_probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["model_spec"]["role"])
        assert kwargs["prompt"] == mod.MINIMAL_PROMPT
        assert kwargs["offload_config"]["n_gpu_layers"] == -1
        return {
            "runtime_ready": True,
            "status": "generation_ready",
            "wall_clock_s": 2.25,
            "prompt_checksum": mod.sha16(kwargs["prompt"]),
            "output_checksum": mod.sha16("OK"),
            "output_text_preview": "OK",
            "command": ["llama_cpp.Llama", kwargs["model_spec"]["model_path"]],
            "config": dict(kwargs["offload_config"]),
            "gpu_memory_receipts": {
                "before": [{"index": 0, "memory_used_mb": 4}],
                "after_load": [{"index": 0, "memory_used_mb": 5120}],
                "after_generate": [{"index": 0, "memory_used_mb": 5120}],
                "offload_evidence": True,
                "max_memory_delta_mb": 5116,
            },
        }

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "ready.json",
        model_resolver=resolver,
        cached_pair_provider=_cached_pair_provider,
        gpu_receipts_provider=_fake_gpu_receipts,
        generation_probe=generation_probe,
        tests_run=[{"command": "unit ready", "outcome": "passed"}],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert calls == ["flagship_dense"]
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.LIVE_INFERENCE_SUBSTRATE
    assert artifact["sota_offload_ready"] is True
    assert "flagship_dense" in artifact["sota_offload_ready_principle"]
    dense = artifact["MODEL_SPECS"]["value"]["flagship_dense"]
    assert dense["runtime_status"] == "generation_ready"
    assert dense["live_generation_ready"] is True
    assert dense["file_receipts"]["size_bytes"] == gguf.stat().st_size
    duration = artifact["duration_receipts"]["value"]["per_model"]["flagship_dense"]
    assert duration["wall_clock_s"] == pytest.approx(2.25)
    assert duration["prompt_checksum"] == mod.sha16(mod.MINIMAL_PROMPT)
    assert duration["output_checksum"] == mod.sha16("OK")
    assert artifact["gpu_offload_receipts"]["value"]["per_model"]["flagship_dense"][
        "gpu_memory_receipts"
    ]["offload_evidence"] is True


def test_req_verify_5284_blocks_generation_without_offload_evidence(tmp_path: Path) -> None:
    """REQ-VERIFY-5284: live text alone cannot open the SOTA offload gate."""

    gguf = _write_minimal_gguf(tmp_path / "qwen.gguf")

    def generation_probe(**kwargs: Any) -> dict[str, Any]:
        return {
            "runtime_ready": True,
            "status": "generation_ready",
            "wall_clock_s": 2.0,
            "prompt_checksum": mod.sha16(kwargs["prompt"]),
            "output_checksum": mod.sha16("OK"),
            "gpu_memory_receipts": {
                "before": [{"index": 0, "memory_used_mb": 4}],
                "after_load": [{"index": 0, "memory_used_mb": 4}],
                "offload_evidence": False,
                "max_memory_delta_mb": 0,
            },
        }

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked_no_offload.json",
        model_resolver=lambda hf_id, _quant: (
            str(gguf) if hf_id == mod.MANDATED_MODEL_SPECS[0]["hf_id"] else None
        ),
        cached_pair_provider=_cached_pair_provider,
        gpu_receipts_provider=lambda: _fake_gpu_receipts(offload_supported=False),
        generation_probe=generation_probe,
        tests_run=[],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["sota_offload_ready"] is False
    assert "blocked_no_gpu_offload_evidence" in artifact["sota_offload_ready_principle"]
    assert artifact["MODEL_SPECS"]["value"]["flagship_moe"]["live_generation_ready"] is False


def test_req_verify_5284_schema_and_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5284: schema and GGUF metadata helpers reject bad receipts."""

    gguf = _write_minimal_gguf(tmp_path / "tiny.gguf")
    receipts = mod._file_receipts(gguf)

    assert receipts["checksum_sha256"]
    assert receipts["checksum_head_1m_sha256"]
    assert mod.read_gguf_header(gguf)["magic"] == "GGUF"

    two_chunk = tmp_path / "two_chunk.gguf"
    two_chunk.write_bytes(b"a" * (1024 * 1024 + 1))
    assert mod._file_receipts(two_chunk)["checksum_note"] == "full_sha256_recorded"

    large = tmp_path / "large.gguf"
    with large.open("wb") as handle:
        handle.seek(65 * 1024 * 1024)
        handle.write(b"x")
    assert (
        mod._file_receipts(large)["checksum_note"]
        == "full_sha256_skipped_for_large_file_head_1m_recorded"
    )

    short = tmp_path / "short.gguf"
    short.write_bytes(b"GGUF")
    pointer = tmp_path / "pointer.gguf"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    unsupported = tmp_path / "unsupported.gguf"
    unsupported.write_bytes(b"GGUF" + struct.pack("<IQQ", 99, 0, 0))

    with pytest.raises(ValueError, match="truncated GGUF header"):
        mod.read_gguf_header(short)
    with pytest.raises(ValueError, match="not a GGUF file"):
        mod.read_gguf_header(pointer)
    with pytest.raises(ValueError, match="unsupported GGUF version"):
        mod.read_gguf_header(unsupported)
    assert mod._normalise_gpu_memory_receipts(None)["offload_evidence"] is False

    pointer_artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "pointer.json",
        model_resolver=lambda hf_id, _quant: (
            str(pointer) if hf_id == mod.MANDATED_MODEL_SPECS[0]["hf_id"] else None
        ),
        cached_pair_provider=_cached_pair_provider,
        gpu_receipts_provider=_fake_gpu_receipts,
        generation_probe=lambda **_kwargs: {},
        tests_run=[],
        write=False,
    )
    assert pointer_artifact["MODEL_SPECS"]["value"]["flagship_moe"]["status"] == (
        "blocked_metadata_unreadable"
    )

    valid = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "valid.json",
        model_resolver=lambda _hf_id, _quant: None,
        cached_pair_provider=lambda *, gpu_indices: [],
        gpu_receipts_provider=_fake_gpu_receipts,
        generation_probe=lambda **_kwargs: {},
        tests_run=[],
        write=False,
    )
    assert mod.artifact_schema_errors(valid) == []

    broken = dict(valid)
    broken["sota_offload_ready"] = "false"
    broken["no_quality_claim"] = {
        "value": False,
        "principle": mod.FIELD_PRINCIPLES["no_quality_claim"],
    }
    broken["inference_substrate"] = {
        "value": "live_llm_inference",
        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
    }
    broken["MODEL_SPECS"] = {"value": [], "principle": mod.FIELD_PRINCIPLES["MODEL_SPECS"]}
    broken["tests_run"] = "unit"

    errors = mod.artifact_schema_errors(broken)

    assert "sota_offload_ready must be a bare bool" in errors
    assert "no_quality_claim.value must be true" in errors
    assert f"inference_substrate.value must be {mod.LIVE_INFERENCE_SUBSTRATE} or {mod.BLOCKED_INFERENCE_SUBSTRATE}" in errors
    assert "MODEL_SPECS.value must be an object" in errors
    assert "tests_run must be a list" in errors
    assert "missing required field honest_verdict" in mod.artifact_schema_errors({})
    with pytest.raises(AssertionError, match="sota_offload_ready must be a bare bool"):
        mod.validate_artifact(broken)

    ready_wrong = json.loads(json.dumps(valid))
    ready_wrong["sota_offload_ready"] = True
    assert "ready artifact must use live_llm_inference_local_gguf_sota" in (
        mod.artifact_schema_errors(ready_wrong)
    )

    blocked_wrong = json.loads(json.dumps(valid))
    blocked_wrong["inference_substrate"]["value"] = mod.LIVE_INFERENCE_SUBSTRATE
    assert "blocked artifact must use blocked_preconditions_with_no_quality_claim" in (
        mod.artifact_schema_errors(blocked_wrong)
    )

    missing_role = json.loads(json.dumps(valid))
    missing_role["MODEL_SPECS"]["value"].pop("middle_moe")
    assert "MODEL_SPECS.value missing role middle_moe" in mod.artifact_schema_errors(missing_role)

    bad_model = json.loads(json.dumps(valid))
    bad_model["MODEL_SPECS"]["value"]["flagship_moe"]["hf_id"] = "wrong"
    bad_model["MODEL_SPECS"]["value"]["flagship_dense"]["autotokenizer_used"] = True
    model_errors = mod.artifact_schema_errors(bad_model)
    assert "MODEL_SPECS.value.flagship_moe.hf_id mismatch" in model_errors
    assert "MODEL_SPECS.value.flagship_dense.autotokenizer_used must be false" in model_errors

    bad_duration_shape = json.loads(json.dumps(valid))
    bad_duration_shape["duration_receipts"]["value"]["per_model"] = []
    assert "duration_receipts.value.per_model must be an object" in mod.artifact_schema_errors(
        bad_duration_shape
    )

    bad_duration_values = json.loads(json.dumps(valid))
    bad_duration_values["duration_receipts"]["value"]["per_model"] = {
        "flagship_moe": {
            "runtime_ready": True,
            "wall_clock_s": 0.5,
            "prompt_checksum": "",
            "output_checksum": "",
        }
    }
    duration_errors = mod.artifact_schema_errors(bad_duration_values)
    assert "duration_receipts.value.per_model.flagship_moe is below live duration floor" in (
        duration_errors
    )
    assert "duration_receipts.value.per_model.flagship_moe.prompt_checksum missing" in (
        duration_errors
    )
    assert "duration_receipts.value.per_model.flagship_moe.output_checksum missing" in (
        duration_errors
    )


def test_req_verify_5284_blocks_before_generation_when_gpu_or_llama_missing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5284: Step 0 GPU/llama failures stop live generation attempts."""

    gguf = _write_minimal_gguf(tmp_path / "qwen-Q4_K_M.gguf")
    calls: list[str] = []

    def forbidden_probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["model_spec"]["role"])
        return {"runtime_ready": True}

    def gpu_receipts() -> dict[str, Any]:
        receipts = _fake_gpu_receipts(gpu_visible=False)
        receipts["value"]["llama_cpp"]["import_ok"] = False
        return receipts

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked_step0.json",
        model_resolver=lambda hf_id, _quant: (
            str(gguf) if hf_id == mod.MANDATED_MODEL_SPECS[0]["hf_id"] else None
        ),
        cached_pair_provider=_cached_pair_provider,
        gpu_receipts_provider=gpu_receipts,
        generation_probe=forbidden_probe,
        tests_run=[],
        write=False,
    )

    assert calls == []
    assert artifact["preconditions_checked"]["value"]["blocked_preconditions"] == [
        "gpu_not_visible",
        "llama_cpp_unavailable",
    ]
    qwen = artifact["MODEL_SPECS"]["value"]["flagship_moe"]
    assert qwen["runtime_status"] == "not_attempted_preconditions_failed"
    assert qwen["blocked_preconditions"] == ["gpu_not_visible", "llama_cpp_unavailable"]


def test_req_verify_5284_rejects_too_fast_live_generation(tmp_path: Path) -> None:
    """REQ-VERIFY-5284: live SOTA generation receipts cannot be subsecond."""

    gguf = _write_minimal_gguf(tmp_path / "gemma-4-26B-A4B-it-Q4_K_M.gguf")

    def generation_probe(**kwargs: Any) -> dict[str, Any]:
        return {
            "runtime_ready": True,
            "status": "generation_ready",
            "wall_clock_s": 0.25,
            "prompt_checksum": mod.sha16(kwargs["prompt"]),
            "output_checksum": mod.sha16("too-fast"),
            "gpu_memory_receipts": {"offload_evidence": True, "max_memory_delta_mb": 100},
        }

    with pytest.raises(ValueError, match="sub-second live generation duration"):
        mod.run(
            root=tmp_path,
            artifact_path=tmp_path / "too_fast.json",
            model_resolver=lambda hf_id, _quant: (
                str(gguf) if hf_id == mod.MANDATED_MODEL_SPECS[2]["hf_id"] else None
            ),
            cached_pair_provider=_cached_pair_provider,
            gpu_receipts_provider=_fake_gpu_receipts,
            generation_probe=generation_probe,
            tests_run=[],
            write=False,
        )


def test_req_verify_5284_module_does_not_use_autotokenizer_from_pretrained() -> None:
    """REQ-VERIFY-5284: GGUF repos are never probed through AutoTokenizer."""

    source = Path(mod.__file__).read_text(encoding="utf-8")

    assert "AutoTokenizer.from_pretrained" not in source
    assert "transformers" not in source
