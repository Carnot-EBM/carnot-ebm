"""Tests for Exp 5259 SOTA GGUF GPU-offload runtime preflight.

Spec refs: REQ-VERIFY-5259, SCENARIO-VERIFY-5259.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5259_sota_gguf_gpu_offload_preflight_v481 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def test_req_verify_5259_spec_declares_runtime_gate_contract() -> None:
    """REQ-VERIFY-5259: OpenSpec anchors the runtime preflight contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5259",
        "SCENARIO-VERIFY-5259",
        "experiment_5259_sota_gguf_gpu_offload_preflight_v481.py",
        "results/experiment_5259_sota_gguf_gpu_offload_preflight_v481.json",
        "llama_cpp_runtime_preflight_no_quality_claim",
        "sota_runtime_ready",
        "no_quality_claim",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def _fake_gpu_receipts() -> dict[str, Any]:
    return {
        "value": {
            "gpu_visible": True,
            "nvidia_smi": {"ok": True, "stdout": "0, RTX 3090"},
            "torch_cuda": {"available": True, "device_count": 2},
            "llama_cpp": {"import_ok": True, "version": "0.3.29"},
            "offload_settings": {"n_gpu_layers": -1, "n_ctx": 256, "n_predict": 1},
        },
        "principle": mod.FIELD_PRINCIPLES["gpu_offload_receipts"],
    }


def _cached_pair_provider(*, gpu_indices: tuple[int, int]) -> list[dict[str, Any]]:
    assert gpu_indices == (0, 1)
    return [
        {"hf_id": mod.MANDATED_MODEL_SPECS[0]["hf_id"], "model_path": "/cache/a.gguf"},
        {"hf_id": mod.MANDATED_MODEL_SPECS[2]["hf_id"], "model_path": "/cache/b.gguf"},
    ]


def test_scenario_verify_5259_blocks_pointer_files_without_runtime_probe(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5259: pointer files block before local headline readiness."""

    pointer = tmp_path / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 123456789\n",
        encoding="utf-8",
    )
    runtime_calls: list[str] = []

    def resolver(hf_id: str, preferred_quant: str) -> str | None:
        assert preferred_quant == "Q4_K_M"
        return str(pointer) if hf_id == mod.MANDATED_MODEL_SPECS[0]["hf_id"] else None

    def runtime_probe(**kwargs: Any) -> dict[str, Any]:
        runtime_calls.append(str(kwargs["model_path"]))
        raise AssertionError("runtime probe must not run on metadata-unreadable files")

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=resolver,
        cached_pair_provider=_cached_pair_provider,
        gpu_receipts_provider=_fake_gpu_receipts,
        runtime_probe=runtime_probe,
        tests_run=[{"command": "unit", "outcome": "pass"}],
        write=True,
    )

    assert runtime_calls == []
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["sota_runtime_ready"] is False
    assert artifact["no_quality_claim"]["value"] is True
    receipt = artifact["model_receipts"]["value"]["flagship_moe"]
    assert receipt["status"] == "blocked_metadata_unreadable"
    assert receipt["path"] == str(pointer)
    assert receipt["checksum_sha256"]
    assert receipt["autotokenizer_used"] is False
    assert "not a GGUF file" in receipt["outcome"]
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_scenario_verify_5259_sets_ready_after_one_mandated_runtime_path(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5259: one mandated GGUF load path opens downstream gates."""

    gguf = tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))

    def resolver(hf_id: str, _preferred_quant: str) -> str | None:
        return str(gguf) if hf_id == mod.MANDATED_MODEL_SPECS[1]["hf_id"] else None

    def runtime_probe(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["model_path"] == gguf
        assert kwargs["prompt"] == mod.MINIMAL_PROMPT
        return {
            "runtime_ready": True,
            "status": "runtime_ready",
            "command": "llama_cpp.Llama(... n_gpu_layers=-1 ...)",
            "config": {"n_gpu_layers": -1, "n_ctx": 256, "n_predict": 1},
            "outcome": "vocab tokenization and deterministic dry-run load completed",
            "traceback": None,
        }

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=resolver,
        cached_pair_provider=_cached_pair_provider,
        gpu_receipts_provider=_fake_gpu_receipts,
        runtime_probe=runtime_probe,
        tests_run=[{"command": "unit", "outcome": "pass"}],
        write=False,
    )

    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["sota_runtime_ready"] is True
    assert "ready through flagship_dense" in artifact["sota_runtime_ready_principle"]
    assert artifact["model_receipts"]["value"]["flagship_dense"]["status"] == "runtime_ready"
    assert artifact["model_receipts"]["value"]["flagship_moe"]["status"] == "missing_local_gguf"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5259_schema_errors_fail_closed() -> None:
    """REQ-VERIFY-5259: malformed artifacts are rejected before closeout."""

    artifact = {
        "honest_verdict": {"value": "blocked_sota_runtime_not_ready", "principle": "x"},
        "inference_substrate": {
            "value": "live_llm_inference",
            "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
        },
        "preconditions_checked": {
            "value": {},
            "principle": mod.FIELD_PRINCIPLES["preconditions_checked"],
        },
        "sota_runtime_ready": "false",
        "sota_runtime_ready_principle": "",
        "model_receipts": {"value": {}, "principle": mod.FIELD_PRINCIPLES["model_receipts"]},
        "gpu_offload_receipts": {
            "value": {},
            "principle": mod.FIELD_PRINCIPLES["gpu_offload_receipts"],
        },
        "no_quality_claim": {
            "value": False,
            "principle": mod.FIELD_PRINCIPLES["no_quality_claim"],
        },
        "tests_run": [],
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "inference_substrate.value must be llama_cpp_runtime_preflight_no_quality_claim" in errors
    assert "sota_runtime_ready must be a bare bool" in errors
    assert "no_quality_claim.value must be true" in errors
    assert "model_receipts.value missing role flagship_moe" in errors
    assert "missing required field honest_verdict" in mod.artifact_schema_errors({})

    malformed = dict(artifact)
    malformed["model_receipts"] = {
        "value": [],
        "principle": mod.FIELD_PRINCIPLES["model_receipts"],
    }
    malformed["tests_run"] = "unit"
    malformed["honest_verdict"] = "blocked_sota_runtime_not_ready"
    malformed["gpu_offload_receipts"] = {"value": {}}

    malformed_errors = mod.artifact_schema_errors(malformed)

    assert "honest_verdict must be principle-wrapped" in malformed_errors
    assert "honest_verdict.value must start with complete: or blocked_" in malformed_errors
    assert "model_receipts.value must be an object" in malformed_errors
    assert "tests_run must be a list" in malformed_errors


def test_req_verify_5259_header_and_checksum_edge_cases(tmp_path: Path) -> None:
    """REQ-VERIFY-5259: metadata and checksum helpers fail closed."""

    two_chunk_file = tmp_path / "two_chunk.gguf"
    two_chunk_file.write_bytes(b"a" * (1024 * 1024 + 1))

    receipts = mod._file_receipts(two_chunk_file)

    assert receipts["checksum_sha256"]
    assert receipts["checksum_note"] == "full_sha256_recorded"

    large_file = tmp_path / "large.gguf"
    with large_file.open("wb") as handle:
        handle.seek(65 * 1024 * 1024)
        handle.write(b"x")

    large_receipts = mod._file_receipts(large_file)

    assert large_receipts["checksum_sha256"] is None
    assert large_receipts["checksum_head_1m_sha256"]
    assert large_receipts["checksum_note"] == "full_sha256_skipped_for_large_file_head_1m_recorded"

    short = tmp_path / "short.gguf"
    short.write_bytes(b"GGUF")
    unsupported = tmp_path / "unsupported.gguf"
    unsupported.write_bytes(b"GGUF" + struct.pack("<IQQ", 99, 0, 0))

    with pytest.raises(ValueError, match="truncated GGUF header"):
        mod.read_gguf_header(short)
    with pytest.raises(ValueError, match="unsupported GGUF version"):
        mod.read_gguf_header(unsupported)


def test_req_verify_5259_run_raises_on_schema_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5259: run fails closed if artifact validation regresses."""

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["bad schema"])

    with pytest.raises(ValueError, match="bad schema"):
        mod.run(
            root=tmp_path,
            artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
            model_resolver=lambda _hf_id, _preferred_quant: None,
            cached_pair_provider=_cached_pair_provider,
            gpu_receipts_provider=_fake_gpu_receipts,
            runtime_probe=lambda **_kwargs: {},
            tests_run=[{"command": "unit", "outcome": "pass"}],
            write=False,
        )


def test_req_verify_5259_module_does_not_use_autotokenizer_from_pretrained() -> None:
    """REQ-VERIFY-5259: GGUF repos are never probed through AutoTokenizer."""

    source = Path(mod.__file__).read_text(encoding="utf-8")

    assert "AutoTokenizer.from_pretrained" not in source
    assert "transformers" not in source
