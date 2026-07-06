"""Tests for Exp 5309 SOTA GGUF runtime timeout root-cause matrix.

Spec refs: REQ-VERIFY-5309, SCENARIO-VERIFY-5309.
"""

from __future__ import annotations

import ast
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5309_sota_runtime_timeout_rootcause_matrix_v485 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def _fake_gpu_backend(*, gpu_visible: bool = True, backend_ready: bool = True) -> dict[str, Any]:
    return {
        "gpu_visible": gpu_visible,
        "vram_before": [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_used_mb": 4},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090", "memory_used_mb": 4},
        ],
        "vram_after": [
            {"index": 0, "name": "NVIDIA GeForce RTX 3090", "memory_used_mb": 4},
            {"index": 1, "name": "NVIDIA GeForce RTX 3090", "memory_used_mb": 4},
        ],
        "nvidia_smi": {
            "ok": gpu_visible,
            "stdout": "0, NVIDIA GeForce RTX 3090, 610.43.02, 24576, 24120, 0",
        },
        "backend_command": "/opt/llama.cpp/build/bin/llama-cli" if backend_ready else None,
        "backend_kind": "native_llama_cpp_cli",
        "backend_version": {"ok": backend_ready, "stderr": "version: 9606 CUDA"},
        "backend_devices": {
            "ok": backend_ready,
            "stdout": "CUDA0: NVIDIA GeForce RTX 3090\nCUDA1: NVIDIA GeForce RTX 3090",
        },
        "backend_dynamic_libraries": {
            "ok": backend_ready,
            "stdout": "libggml-cuda.so.0\nlibcuda.so.1\nlibcublas.so.13",
        },
        "cuda_backend_evidence": backend_ready,
    }


def _resolver_from_paths(paths: dict[str, Path]):
    def resolver(hf_id: str, _quant: str) -> str | None:
        return str(paths[hf_id]) if hf_id in paths else None

    return resolver


def _timeout_probe(**kwargs: Any) -> dict[str, Any]:
    return {
        "status": "timeout_before_first_token",
        "timeout_class": "timeout_before_first_token",
        "completed": False,
        "timed_out": True,
        "timeout_s": 0.25,
        "load_s": 0.11,
        "gpu_offload_evidence_s": 0.12,
        "prompt_ingestion_s": None,
        "first_token_latency_s": None,
        "eight_token_generation_s": None,
        "generated_token_count": 0,
        "stdout_tail": "",
        "stderr_tail": "load_tensors: offloaded 49/49 layers to GPU",
        "gpu_memory_receipts": {
            "before": [{"index": 0, "memory_used_mb": 4}],
            "during": [[{"index": 0, "memory_used_mb": 8192}]],
            "after": [{"index": 0, "memory_used_mb": 4}],
            "max_memory_delta_mb": 8188,
            "offload_evidence": True,
        },
        "command": [kwargs["backend"]["backend_command"], "-m", kwargs["model_spec"]["model_path"]],
        "config": dict(kwargs["runtime_config"]),
    }


def test_req_verify_5309_spec_declares_timeout_matrix_contract() -> None:
    """REQ-VERIFY-5309: OpenSpec anchors the timeout root-cause matrix."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5309") : spec.index("### REQ-VERIFY-5297")]

    for marker in (
        "REQ-VERIFY-5309",
        "SCENARIO-VERIFY-5309",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "sota_runtime_unblocked",
        "no_quality_claim",
        "first-token",
        "8-token generation",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_verify_5309_blocks_when_all_available_models_timeout(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5309: all mandated timeouts keep downstream gate closed."""

    paths = {
        str(spec["hf_id"]): _write_minimal_gguf(tmp_path / f"{spec['role']}.gguf")
        for spec in mod.MANDATED_MODEL_SPECS
    }
    calls: list[str] = []

    def probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["model_spec"]["role"])
        return _timeout_probe(**kwargs)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver_from_paths(paths),
        gpu_backend_provider=_fake_gpu_backend,
        runtime_probe=probe,
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert calls == [spec["role"] for spec in mod.MANDATED_MODEL_SPECS]
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["sota_runtime_unblocked"] is False
    assert artifact["no_quality_claim"] is True
    assert "all_mandated_models_timeout_before_first_token" in artifact["timeout_root_cause"]["value"]
    for role, row in artifact["per_model_runtime_matrix"]["value"].items():
        assert row["hf_id"] in {spec["hf_id"] for spec in mod.MANDATED_MODEL_SPECS}
        assert row["timeout_class"] == "timeout_before_first_token"
        assert row["offload_authenticated"] is True
        assert row["first_token_latency_s"] is None
        assert row["eight_token_generation_s"] is None
        assert row["autotokenizer_used"] is False
        assert row["context_size"] == mod.RUNTIME_CONFIG["context_size"]
        assert row["batch_size"] == mod.RUNTIME_CONFIG["batch_size"]


def test_scenario_verify_5309_unblocks_after_one_model_completes_with_offload(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5309: one complete offloaded model opens runtime gate."""

    paths = {
        str(spec["hf_id"]): _write_minimal_gguf(tmp_path / f"{spec['role']}.gguf")
        for spec in mod.MANDATED_MODEL_SPECS
    }

    def probe(**kwargs: Any) -> dict[str, Any]:
        base = _timeout_probe(**kwargs)
        if kwargs["model_spec"]["role"] == "flagship_dense":
            return base | {
                "status": "completed",
                "timeout_class": "completed_no_timeout",
                "completed": True,
                "timed_out": False,
                "prompt_ingestion_s": 0.04,
                "first_token_latency_s": 0.21,
                "eight_token_generation_s": 0.52,
                "generated_token_count": 8,
                "stdout_tail": "OK OK OK OK",
            }
        return base

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "ready.json",
        model_resolver=_resolver_from_paths(paths),
        gpu_backend_provider=_fake_gpu_backend,
        runtime_probe=probe,
        tests_run=[{"command": "unit ready", "outcome": "passed"}],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["sota_runtime_unblocked"] is True
    assert artifact["timeout_root_cause"]["value"] == "none"
    dense = artifact["per_model_runtime_matrix"]["value"]["flagship_dense"]
    assert dense["timeout_class"] == "completed_no_timeout"
    assert dense["completed_load_first_token_and_8_tokens"] is True
    assert dense["offload_authenticated"] is True
    assert dense["generated_token_count"] == 8
    assert dense["first_token_latency_s"] == pytest.approx(0.21)
    assert dense["eight_token_generation_s"] == pytest.approx(0.52)


def test_req_verify_5309_preconditions_block_before_probe_when_backend_missing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5309: missing runtime backend records preconditions, no fake probe."""

    gguf = _write_minimal_gguf(tmp_path / "qwen.gguf")
    calls: list[str] = []

    def forbidden_probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["model_spec"]["role"])
        raise AssertionError("runtime probe must not run without backend preconditions")

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked.json",
        model_resolver=_resolver_from_paths({mod.MANDATED_MODEL_SPECS[0]["hf_id"]: gguf}),
        gpu_backend_provider=lambda: _fake_gpu_backend(backend_ready=False),
        runtime_probe=forbidden_probe,
        tests_run=[{"command": "unit precondition", "outcome": "passed"}],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert calls == []
    assert artifact["sota_runtime_unblocked"] is False
    assert "backend_command_missing" in artifact["preconditions_checked"]["value"]["blocked_preconditions"]
    assert artifact["per_model_runtime_matrix"]["value"]["flagship_moe"][
        "timeout_class"
    ] == "not_attempted_preconditions_failed"
    assert artifact["per_model_runtime_matrix"]["value"]["flagship_dense"][
        "timeout_class"
    ] == "not_available"


def test_req_verify_5309_module_does_not_import_transformers_tokenizer() -> None:
    """REQ-VERIFY-5309: GGUF repos are loaded via paths, not AutoTokenizer."""

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "transformers":
            forbidden.extend(alias.name for alias in node.names)
        if isinstance(node, ast.Import):
            forbidden.extend(alias.name for alias in node.names if alias.name == "transformers")
        if isinstance(node, ast.Attribute) and node.attr == "AutoTokenizer":
            forbidden.append("AutoTokenizer")
    assert forbidden == []


def test_validate_artifact_rejects_unblocked_without_offload(tmp_path: Path) -> None:
    """REQ-VERIFY-5309: success requires authenticated GPU offload evidence."""

    gguf = _write_minimal_gguf(tmp_path / "gemma.gguf")

    def bad_probe(**kwargs: Any) -> dict[str, Any]:
        receipt = _timeout_probe(**kwargs) | {
            "status": "completed",
            "timeout_class": "completed_no_timeout",
            "completed": True,
            "timed_out": False,
            "prompt_ingestion_s": 0.04,
            "first_token_latency_s": 0.21,
            "eight_token_generation_s": 0.52,
            "generated_token_count": 8,
            "backend_gpu_log_evidence": False,
            "stderr_tail": "",
        }
        receipt["gpu_memory_receipts"] = {"max_memory_delta_mb": 0, "offload_evidence": False}
        return receipt

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-no-offload.json",
        model_resolver=_resolver_from_paths({mod.MANDATED_MODEL_SPECS[1]["hf_id"]: gguf}),
        gpu_backend_provider=_fake_gpu_backend,
        runtime_probe=bad_probe,
        tests_run=[{"command": "unit no offload", "outcome": "passed"}],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["sota_runtime_unblocked"] is False
    dense = artifact["per_model_runtime_matrix"]["value"]["flagship_dense"]
    assert dense["completed_load_first_token_and_8_tokens"] is True
    assert dense["offload_authenticated"] is False
    assert "no_authenticated_gpu_offload" in artifact["timeout_root_cause"]["value"]


def test_req_verify_5309_gguf_header_and_metadata_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5309: malformed GGUF metadata is blocked before runtime."""

    truncated = tmp_path / "truncated.gguf"
    truncated.write_bytes(b"GGUF")
    not_gguf = tmp_path / "not.gguf"
    not_gguf.write_bytes(b"NOPE" + struct.pack("<IQQ", 3, 1, 1))
    bad_version = tmp_path / "bad-version.gguf"
    bad_version.write_bytes(b"GGUF" + struct.pack("<IQQ", 99, 1, 1))

    with pytest.raises(ValueError, match="truncated"):
        mod.read_gguf_header(truncated)
    with pytest.raises(ValueError, match="not a GGUF"):
        mod.read_gguf_header(not_gguf)
    with pytest.raises(ValueError, match="unsupported"):
        mod.read_gguf_header(bad_version)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "bad-metadata.json",
        model_resolver=_resolver_from_paths({mod.MANDATED_MODEL_SPECS[0]["hf_id"]: not_gguf}),
        gpu_backend_provider=_fake_gpu_backend,
        runtime_probe=lambda **_: pytest.fail("probe should not run for bad metadata"),
        tests_run=[{"command": "unit bad metadata", "outcome": "passed"}],
        write=False,
    )

    mod.validate_artifact(artifact)
    qwen = artifact["MODEL_SPECS"]["value"]["flagship_moe"]
    assert qwen["status"] == "blocked_metadata_unreadable"
    assert qwen["blocked_preconditions"][0].startswith("metadata_unreadable")
    assert artifact["per_model_runtime_matrix"]["value"]["flagship_moe"][
        "timeout_class"
    ] == "not_available"


def test_req_verify_5309_timeout_and_precondition_classifiers() -> None:
    """REQ-VERIFY-5309: timeout classes and root causes stay specific."""

    assert (
        mod._classify_probe_timeout(
            completed=True,
            timed_out=False,
            first_token_latency_s=0.1,
            generated_token_count=8,
        )
        == "completed_no_timeout"
    )
    assert (
        mod._classify_probe_timeout(
            completed=False,
            timed_out=True,
            first_token_latency_s=None,
            generated_token_count=0,
        )
        == "timeout_before_first_token"
    )
    assert (
        mod._classify_probe_timeout(
            completed=False,
            timed_out=True,
            first_token_latency_s=0.2,
            generated_token_count=3,
        )
        == "timeout_during_8_token_generation"
    )
    assert (
        mod._classify_probe_timeout(
            completed=False,
            timed_out=False,
            first_token_latency_s=None,
            generated_token_count=0,
        )
        == "no_first_token"
    )
    assert (
        mod._classify_probe_timeout(
            completed=False,
            timed_out=False,
            first_token_latency_s=0.2,
            generated_token_count=3,
        )
        == "generation_incomplete"
    )

    model_specs = {
        spec["role"]: {
            "status": "missing_local_gguf",
            "role": spec["role"],
            "hf_id": spec["hf_id"],
        }
        for spec in mod.MANDATED_MODEL_SPECS
    }
    assert mod._precondition_blockers(_fake_gpu_backend(gpu_visible=False), model_specs) == [
        "gpu_not_visible",
        "no_mandated_sota_gguf_resolved",
    ]
    assert mod._precondition_blockers(_fake_gpu_backend(backend_ready=False), model_specs) == [
        "backend_command_missing",
        "no_mandated_sota_gguf_resolved",
    ]
    backend_no_cuda = _fake_gpu_backend()
    backend_no_cuda["cuda_backend_evidence"] = False
    assert mod._precondition_blockers(backend_no_cuda, model_specs) == [
        "cuda_backend_evidence_missing",
        "no_mandated_sota_gguf_resolved",
    ]

    assert mod._timeout_root_cause(matrix={}, precondition_blockers=[]) == "no_mandated_sota_gguf_available"
    assert (
        mod._timeout_root_cause(
            matrix={
                "flagship_moe": {
                    "model_available": True,
                    "timeout_class": "timeout_during_8_token_generation",
                }
            },
            precondition_blockers=[],
        )
        == "timeout_during_bounded_8_token_generation"
    )
    assert (
        mod._timeout_root_cause(
            matrix={"flagship_moe": {"model_available": True, "timeout_class": "no_first_token"}},
            precondition_blockers=[],
        )
        == "no_first_token_observed"
    )
    assert "generation_incomplete" in mod._timeout_root_cause(
        matrix={
            "flagship_moe": {
                "model_available": True,
                "timeout_class": "generation_incomplete",
            }
        },
        precondition_blockers=[],
    )
    native_abort = mod._timeout_root_cause(
        matrix={
            "flagship_moe": {
                "model_available": True,
                "timeout_class": "generation_incomplete",
                "returncode": -6,
                "stdout_tail": (
                    "--no-conversation is not supported by llama-cli\n"
                    "GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed"
                ),
                "stderr_tail": "",
            }
        },
        precondition_blockers=[],
    )
    assert native_abort.startswith("native_llama_cpp_generation_abort_after_authenticated_offload")
    assert "llama-completion-compatible" in native_abort
    assert (
        mod._native_runtime_abort_cause(
            {"returncode": -6, "stdout_tail": "--no-conversation is not supported by llama-cli"}
        )
        == "llama_cli_no_conversation_unsupported"
    )
    assert (
        mod._native_runtime_abort_cause(
            {"returncode": 134, "stderr_tail": "llama-context.cpp:1712"}
        )
        == "llama_context_batch_assert"
    )
    assert mod._native_runtime_abort_cause({"returncode": -6}) == "native_llama_cpp_abort_signal"


def test_validate_artifact_reports_schema_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-5309: schema validation catches malformed gate artifacts."""

    gguf = _write_minimal_gguf(tmp_path / "gemma31.gguf")

    def ready_probe(**kwargs: Any) -> dict[str, Any]:
        return _timeout_probe(**kwargs) | {
            "status": "completed",
            "timeout_class": "completed_no_timeout",
            "completed": True,
            "timed_out": False,
            "prompt_ingestion_s": 0.04,
            "first_token_latency_s": 0.21,
            "eight_token_generation_s": 0.52,
            "generated_token_count": 8,
            "stdout_tail": "OK OK OK OK",
        }

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "ready.json",
        model_resolver=_resolver_from_paths({mod.MANDATED_MODEL_SPECS[1]["hf_id"]: gguf}),
        gpu_backend_provider=_fake_gpu_backend,
        runtime_probe=ready_probe,
        tests_run=[{"command": "unit schema", "outcome": "passed"}],
        write=False,
    )

    def clone() -> dict[str, Any]:
        return json.loads(json.dumps(artifact))

    malformed_cases = [
        (lambda a: (a.pop("MODEL_SPECS"), a)[1], "missing required fields"),
        (lambda a: (a.__setitem__("milestone", mod.MILESTONE), a)[1], "principle-wrapped"),
        (
            lambda a: (
                a["experiment_id"].__setitem__("value", "wrong"),
                a["milestone"].__setitem__("value", "wrong"),
                a["status"].__setitem__("value", "running"),
                a["honest_verdict"].__setitem__("value", "done"),
                a["inference_substrate"].__setitem__("value", "wrong"),
                a.__setitem__("sota_runtime_unblocked", "yes"),
                a.__setitem__("no_quality_claim", False),
                a["tests_run"].__setitem__("value", "not-list"),
                a,
            )[8],
            "experiment_id mismatch",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"].pop("middle_moe"),
                a["per_model_runtime_matrix"]["value"].pop("middle_moe"),
                a,
            )[2],
            "roles mismatch",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("hf_id", "wrong"),
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__(
                    "autotokenizer_used", True
                ),
                a["per_model_runtime_matrix"]["value"]["flagship_dense"].__setitem__(
                    "autotokenizer_used", True
                ),
                a["per_model_runtime_matrix"]["value"]["flagship_dense"].__setitem__(
                    "context_size", 999
                ),
                a["per_model_runtime_matrix"]["value"]["flagship_dense"].__setitem__(
                    "batch_size", 999
                ),
                a,
            )[5],
            "hf_id mismatch",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"].__setitem__("value", []),
                a["per_model_runtime_matrix"].__setitem__("value", []),
                a,
            )[2],
            "must be objects",
        ),
        (
            lambda a: (a["timeout_root_cause"].__setitem__("value", "still_bad"), a)[1],
            "unblocked artifact must have timeout_root_cause",
        ),
        (
            lambda a: (
                a.__setitem__("sota_runtime_unblocked", False),
                a["timeout_root_cause"].__setitem__("value", ""),
                a,
            )[2],
            "blocked artifact must name timeout_root_cause",
        ),
        (
            lambda a: (
                a["per_model_runtime_matrix"]["value"]["flagship_dense"].__setitem__(
                    "offload_authenticated", False
                ),
                a,
            )[1],
            "unblocked artifact must have at least one",
        ),
        (
            lambda a: (
                a.__setitem__("sota_runtime_unblocked", False),
                a["timeout_root_cause"].__setitem__("value", "blocked"),
                a,
            )[2],
            "blocked artifact cannot contain",
        ),
    ]

    seen_messages: list[str] = []
    for mutate, expected in malformed_cases:
        errors = mod.artifact_schema_errors(mutate(clone()))
        joined = "; ".join(errors)
        seen_messages.append(joined)
        assert expected in joined
    with pytest.raises(AssertionError, match="experiment_id mismatch"):
        bad = clone()
        bad["experiment_id"]["value"] = "wrong"
        mod.validate_artifact(bad)
    assert mod._wrapped_value({"plain": "value"}, "plain") is None
    assert any("no_quality_claim" in message for message in seen_messages)
