"""Tests for Exp 5297 changed-runtime SOTA GGUF substrate gate.

Spec refs: REQ-VERIFY-5297, SCENARIO-VERIFY-5297.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5297_changed_runtime_sota_substrate_gate_v484 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def _write_minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def _fake_gpu_receipts(*, gpu_visible: bool = True) -> dict[str, Any]:
    return {
        "gpu_visible": gpu_visible,
        "nvidia_smi": {
            "ok": gpu_visible,
            "stdout": "0, NVIDIA RTX 3090, 610.43.02, 24576, 24000, 0",
        },
        "cuda_runtime": {"ok": gpu_visible, "stdout": "CUDA UMD Version: 13.3"},
        "nvcc": {"ok": True, "stdout": "Cuda compilation tools, release 13.3"},
        "torch_cuda": {"import_ok": True, "available": gpu_visible, "device_count": 2},
    }


def _fake_changed_runtime(
    *, changed: bool = True, cuda_backend: bool = True
) -> dict[str, Any]:
    return {
        "backend_kind": "native_llama_cpp_cli" if changed else "llama_cpp_python_legacy",
        "backend_path": "/opt/llama.cpp/build/bin/llama-cli" if changed else None,
        "changed_from_exp5284": changed,
        "changed_from_exp5284_principle": (
            "native llama.cpp CLI with CUDA-linked libraries, not the Exp 5284 "
            "llama-cpp-python import path"
        )
        if changed
        else "legacy Python path repeats Exp 5284 and cannot count",
        "version": {
            "ok": changed,
            "stdout": "version: 9606 built with GNU 16.1.1 for Linux x86_64",
        },
        "list_devices": {
            "ok": cuda_backend,
            "stdout": "CUDA0: NVIDIA GeForce RTX 3090 (24123 MiB, 23858 MiB free)",
        },
        "dynamic_libraries": {
            "ok": cuda_backend,
            "stdout": "libggml-cuda.so.0 => /build/bin/libggml-cuda.so.0\nlibcublas.so.13",
        },
        "cuda_backend_evidence": cuda_backend,
        "old_cpu_only_llama_cpp_python_counted_as_success": False,
    }


def _cached_pair_provider(*, gpu_indices: tuple[int, int]) -> list[dict[str, Any]]:
    assert gpu_indices == (0, 1)
    return [
        {"hf_id": mod.MANDATED_MODEL_SPECS[0]["hf_id"], "model_path": "/cache/qwen.gguf"},
        {"hf_id": mod.MANDATED_MODEL_SPECS[1]["hf_id"], "model_path": "/cache/gemma31.gguf"},
    ]


def test_req_verify_5297_spec_declares_changed_runtime_gate() -> None:
    """REQ-VERIFY-5297: OpenSpec anchors the changed-runtime SOTA gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5297") : spec.index("### REQ-VERIFY-5284")]

    for marker in (
        "REQ-VERIFY-5297",
        "SCENARIO-VERIFY-5297",
        str(mod.RESULT_RELATIVE_PATH),
        mod.LIVE_INFERENCE_SUBSTRATE,
        mod.BLOCKED_INFERENCE_SUBSTRATE,
        "changed_runtime_sota_ready",
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


def test_scenario_verify_5297_blocks_without_changed_runtime_or_model(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5297: blocked preconditions do not fake generation."""

    calls: list[str] = []

    def forbidden_generation_probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(str(kwargs["model_spec"].get("model_path")))
        raise AssertionError("generation probe must not run when Step 0 blocks")

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=lambda _hf_id, _quant: None,
        cached_pair_provider=lambda *, gpu_indices: [],
        gpu_receipts_provider=_fake_gpu_receipts,
        runtime_substrate_provider=lambda: _fake_changed_runtime(changed=False),
        generation_probe=forbidden_generation_probe,
        smoke_tests_provider=lambda: [{"label": "smoke_test_not_headline", "status": "passed"}],
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
        write=True,
    )

    assert calls == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["inference_substrate"]["value"] == mod.BLOCKED_INFERENCE_SUBSTRATE
    assert artifact["changed_runtime_sota_ready"] is False
    assert artifact["no_quality_claim"]["value"] is True
    assert artifact["smoke_tests"]["value"][0]["label"] == "smoke_test_not_headline"
    assert (
        artifact["preconditions_checked"]["value"][
            "at_least_one_mandated_model_resolved_without_autotokenizer"
        ]
        is False
    )
    assert "changed_runtime_substrate_unavailable" in (
        artifact["preconditions_checked"]["value"]["blocked_preconditions"]
    )
    assert artifact["MODEL_SPECS"]["value"]["flagship_moe"]["status"] == "missing_local_gguf"
    mod.validate_artifact(artifact)


def test_scenario_verify_5297_sets_ready_after_native_cli_gpu_offload(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5297: a mandated native-CLI GPU receipt opens the gate."""

    gguf = _write_minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    calls: list[str] = []

    def resolver(hf_id: str, _quant: str) -> str | None:
        return str(gguf) if hf_id == mod.MANDATED_MODEL_SPECS[1]["hf_id"] else None

    def generation_probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["model_spec"]["role"])
        assert kwargs["prompt"] == mod.MINIMAL_PROMPT
        assert kwargs["offload_config"]["n_gpu_layers"] == "all"
        return {
            "runtime_ready": True,
            "status": "generation_ready",
            "wall_clock_s": 2.75,
            "prompt_checksum": mod.sha16(kwargs["prompt"]),
            "output_checksum": mod.sha16("OK"),
            "output_text_preview": "OK",
            "command": [
                "/opt/llama.cpp/build/bin/llama-cli",
                "-m",
                kwargs["model_spec"]["model_path"],
                "-ngl",
                "all",
            ],
            "config": dict(kwargs["offload_config"]),
            "stdout_tail": "OK",
            "stderr_tail": "load_tensors: offloaded 49/49 layers to GPU\nCUDA0",
            "backend_gpu_log_evidence": True,
            "gpu_memory_receipts": {
                "before": [{"index": 0, "memory_used_mb": 4}],
                "during": [{"index": 0, "memory_used_mb": 9216}],
                "after": [{"index": 0, "memory_used_mb": 4}],
                "max_memory_delta_mb": 9212,
                "offload_evidence": True,
            },
        }

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "ready.json",
        model_resolver=resolver,
        cached_pair_provider=_cached_pair_provider,
        gpu_receipts_provider=_fake_gpu_receipts,
        runtime_substrate_provider=_fake_changed_runtime,
        generation_probe=generation_probe,
        tests_run=[{"command": "unit ready", "outcome": "passed"}],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert calls == ["flagship_dense"]
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.LIVE_INFERENCE_SUBSTRATE
    assert artifact["changed_runtime_sota_ready"] is True
    assert "flagship_dense" in artifact["changed_runtime_sota_ready_principle"]
    assert artifact["runtime_substrate_changed"]["value"]["changed_from_exp5284"] is True
    dense = artifact["MODEL_SPECS"]["value"]["flagship_dense"]
    assert dense["runtime_status"] == "generation_ready"
    assert dense["live_generation_ready"] is True
    assert dense["file_receipts"]["size_bytes"] == gguf.stat().st_size
    duration = artifact["duration_receipts"]["value"]["per_model"]["flagship_dense"]
    assert duration["wall_clock_s"] == pytest.approx(2.75)
    assert duration["prompt_checksum"] == mod.sha16(mod.MINIMAL_PROMPT)
    assert duration["output_checksum"] == mod.sha16("OK")
    assert artifact["gpu_offload_receipts"]["value"]["per_model"]["flagship_dense"][
        "gpu_memory_receipts"
    ]["offload_evidence"] is True


def test_req_verify_5297_live_text_without_gpu_offload_stays_blocked(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5297: changed runtime text alone cannot open the SOTA gate."""

    gguf = _write_minimal_gguf(tmp_path / "qwen-Q4_K_M.gguf")

    def generation_probe(**kwargs: Any) -> dict[str, Any]:
        return {
            "runtime_ready": True,
            "status": "generation_ready",
            "wall_clock_s": 2.0,
            "prompt_checksum": mod.sha16(kwargs["prompt"]),
            "output_checksum": mod.sha16("OK"),
            "stdout_tail": "OK",
            "stderr_tail": "llama.cpp CPU path only",
            "backend_gpu_log_evidence": False,
            "gpu_memory_receipts": {
                "before": [{"index": 0, "memory_used_mb": 4}],
                "during": [{"index": 0, "memory_used_mb": 4}],
                "after": [{"index": 0, "memory_used_mb": 4}],
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
        gpu_receipts_provider=_fake_gpu_receipts,
        runtime_substrate_provider=_fake_changed_runtime,
        generation_probe=generation_probe,
        tests_run=[],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["changed_runtime_sota_ready"] is False
    assert "blocked_no_gpu_offload_evidence" in artifact["changed_runtime_sota_ready_principle"]
    assert artifact["MODEL_SPECS"]["value"]["flagship_moe"]["live_generation_ready"] is False


def test_req_verify_5297_schema_and_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5297: schema and GGUF metadata helpers reject bad receipts."""

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
        runtime_substrate_provider=_fake_changed_runtime,
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
        runtime_substrate_provider=_fake_changed_runtime,
        generation_probe=lambda **_kwargs: {},
        tests_run=[],
        write=False,
    )
    assert mod.artifact_schema_errors(valid) == []

    broken = dict(valid)
    broken["changed_runtime_sota_ready"] = "false"
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

    assert "changed_runtime_sota_ready must be a bare bool" in errors
    assert "no_quality_claim.value must be true" in errors
    assert (
        f"inference_substrate.value must be {mod.LIVE_INFERENCE_SUBSTRATE} "
        f"or {mod.BLOCKED_INFERENCE_SUBSTRATE}"
    ) in errors
    assert "MODEL_SPECS.value must be an object" in errors
    assert "tests_run must be a list" in errors
    assert "missing required field honest_verdict" in mod.artifact_schema_errors({})
    with pytest.raises(AssertionError, match="changed_runtime_sota_ready must be a bare bool"):
        mod.validate_artifact(broken)

    ready_wrong = json.loads(json.dumps(valid))
    ready_wrong["changed_runtime_sota_ready"] = True
    assert "ready artifact must use live_llm_inference_changed_local_gguf_sota" in (
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

    bad_runtime = json.loads(json.dumps(valid))
    bad_runtime["runtime_substrate_changed"]["value"][
        "old_cpu_only_llama_cpp_python_counted_as_success"
    ] = True
    assert "runtime_substrate_changed must not count old CPU-only Python path" in (
        mod.artifact_schema_errors(bad_runtime)
    )

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


def test_req_verify_5297_blocks_before_generation_when_gpu_missing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5297: Step 0 GPU failures stop changed-runtime generation."""

    gguf = _write_minimal_gguf(tmp_path / "qwen-Q4_K_M.gguf")
    calls: list[str] = []

    def forbidden_probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["model_spec"]["role"])
        return {"runtime_ready": True}

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked_step0.json",
        model_resolver=lambda hf_id, _quant: (
            str(gguf) if hf_id == mod.MANDATED_MODEL_SPECS[0]["hf_id"] else None
        ),
        cached_pair_provider=_cached_pair_provider,
        gpu_receipts_provider=lambda: _fake_gpu_receipts(gpu_visible=False),
        runtime_substrate_provider=_fake_changed_runtime,
        generation_probe=forbidden_probe,
        tests_run=[],
        write=False,
    )

    assert calls == []
    assert artifact["preconditions_checked"]["value"]["blocked_preconditions"] == [
        "gpu_not_visible"
    ]
    qwen = artifact["MODEL_SPECS"]["value"]["flagship_moe"]
    assert qwen["runtime_status"] == "not_attempted_preconditions_failed"
    assert qwen["blocked_preconditions"] == ["gpu_not_visible"]


def test_req_verify_5297_blocks_before_generation_without_cuda_backend(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5297: a changed backend without CUDA evidence still blocks."""

    gguf = _write_minimal_gguf(tmp_path / "qwen-Q4_K_M.gguf")
    calls: list[str] = []

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked_no_cuda_backend.json",
        model_resolver=lambda hf_id, _quant: (
            str(gguf) if hf_id == mod.MANDATED_MODEL_SPECS[0]["hf_id"] else None
        ),
        cached_pair_provider=_cached_pair_provider,
        gpu_receipts_provider=_fake_gpu_receipts,
        runtime_substrate_provider=lambda: _fake_changed_runtime(cuda_backend=False),
        generation_probe=lambda **kwargs: calls.append(kwargs["model_spec"]["role"]) or {},
        tests_run=[],
        write=False,
    )

    assert calls == []
    assert artifact["preconditions_checked"]["value"]["blocked_preconditions"] == [
        "changed_runtime_gpu_backend_unavailable"
    ]


def test_req_verify_5297_rejects_too_fast_live_generation(tmp_path: Path) -> None:
    """REQ-VERIFY-5297: live SOTA native receipts cannot be subsecond."""

    gguf = _write_minimal_gguf(tmp_path / "gemma-4-26B-A4B-it-Q4_K_M.gguf")

    def generation_probe(**kwargs: Any) -> dict[str, Any]:
        return {
            "runtime_ready": True,
            "status": "generation_ready",
            "wall_clock_s": 0.25,
            "prompt_checksum": mod.sha16(kwargs["prompt"]),
            "output_checksum": mod.sha16("too-fast"),
            "backend_gpu_log_evidence": True,
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
            runtime_substrate_provider=_fake_changed_runtime,
            generation_probe=generation_probe,
            tests_run=[],
            write=False,
        )


def test_req_verify_5297_default_probe_drains_backend_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5297: native CLI logs are drained while GPU memory is polled."""

    popen_receipts: list[dict[str, Any]] = []

    class FakePopen:
        returncode = 0

        def __init__(self, command: list[str], stdout: Any, stderr: Any, text: bool) -> None:
            popen_receipts.append(
                {
                    "command": command,
                    "stdout_is_pipe": stdout is mod.subprocess.PIPE,
                    "stderr_is_pipe": stderr is mod.subprocess.PIPE,
                    "text": text,
                }
            )
            stdout.write(b"OK\n")
            stderr.write(
                b"ggml_cuda_init: CUDA0\nload_tensors: offloaded 49/49 layers to GPU\n"
            )
            stdout.flush()
            stderr.flush()
            self._poll_count = 0

        def poll(self) -> int | None:
            self._poll_count += 1
            return None if self._poll_count == 1 else self.returncode

    snapshots = [
        [{"index": 0, "memory_used_mb": 4}],
        [{"index": 0, "memory_used_mb": 2048}],
        [{"index": 0, "memory_used_mb": 4}],
    ]

    def fake_gpu_snapshot() -> list[dict[str, Any]]:
        return snapshots.pop(0) if snapshots else [{"index": 0, "memory_used_mb": 4}]

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(mod, "_gpu_snapshot", fake_gpu_snapshot)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)

    receipt = mod.default_generation_probe(
        model_spec={
            "role": "flagship_moe",
            "hf_id": mod.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_path": "/cache/qwen.gguf",
        },
        prompt=mod.MINIMAL_PROMPT,
        offload_config=mod.OFFLOAD_CONFIG,
        runtime_substrate={"backend_path": "/opt/llama.cpp/build/bin/llama-cli"},
        timeout_s=5.0,
    )

    assert popen_receipts == [
        {
            "command": receipt["command"],
            "stdout_is_pipe": False,
            "stderr_is_pipe": False,
            "text": False,
        }
    ]
    assert receipt["runtime_ready"] is True
    assert receipt["stdout_tail"] == "OK\n"
    assert "offloaded 49/49 layers to GPU" in receipt["stderr_tail"]
    assert receipt["backend_gpu_log_evidence"] is True
    assert receipt["gpu_memory_receipts"]["offload_evidence"] is True
    assert receipt["gpu_memory_receipts"]["max_memory_delta_mb"] == 2044


def test_req_verify_5297_default_probe_timeout_keeps_offload_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5297: native CLI timeouts keep command and GPU evidence."""

    class FakeTimeoutPopen:
        returncode: int | None = None

        def __init__(self, command: list[str], stdout: Any, stderr: Any, text: bool) -> None:
            self.command = command
            stdout.write(b"")
            stderr.write(b"load_tensors: offloaded 49/49 layers to GPU\nCUDA0\n")
            stdout.flush()
            stderr.flush()

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: int) -> int:
            assert timeout == 10
            return self.returncode or -9

    snapshots = [
        [{"index": 0, "memory_used_mb": 4}],
        [{"index": 0, "memory_used_mb": 4096}],
        [{"index": 0, "memory_used_mb": 4}],
    ]
    clock = iter([0.0, 6.0, 6.1])

    monkeypatch.setattr(mod.subprocess, "Popen", FakeTimeoutPopen)
    monkeypatch.setattr(
        mod,
        "_gpu_snapshot",
        lambda: snapshots.pop(0) if snapshots else [{"index": 0, "memory_used_mb": 4}],
    )
    monkeypatch.setattr(mod.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)

    receipt = mod.default_generation_probe(
        model_spec={
            "role": "flagship_moe",
            "hf_id": mod.MANDATED_MODEL_SPECS[0]["hf_id"],
            "model_path": "/cache/qwen.gguf",
        },
        prompt=mod.MINIMAL_PROMPT,
        offload_config=mod.OFFLOAD_CONFIG,
        runtime_substrate={"backend_path": "/opt/llama.cpp/build/bin/llama-cli"},
        timeout_s=5.0,
    )

    assert receipt["runtime_ready"] is False
    assert receipt["status"] == "blocked_native_cli_timeout"
    assert receipt["command"][0] == "/opt/llama.cpp/build/bin/llama-cli"
    assert receipt["returncode"] == -9
    assert receipt["timeout_s"] == 5.0
    assert "offloaded 49/49 layers to GPU" in receipt["stderr_tail"]
    assert receipt["backend_gpu_log_evidence"] is True
    assert receipt["gpu_memory_receipts"]["offload_evidence"] is True
    assert receipt["gpu_memory_receipts"]["max_memory_delta_mb"] == 4092


def test_req_verify_5297_module_avoids_forbidden_paths() -> None:
    """REQ-VERIFY-5297: no AutoTokenizer, no transformers, no conductor edits."""

    source = Path(mod.__file__).read_text(encoding="utf-8")

    assert "AutoTokenizer.from_pretrained" not in source
    assert "transformers" not in source
    assert "research_conductor.py" not in source
