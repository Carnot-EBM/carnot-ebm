"""Tests for Exp 5323 native GGUF backend flag bisect.

Spec refs: REQ-VERIFY-5323, SCENARIO-VERIFY-5323.
"""

from __future__ import annotations

import ast
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5323_native_gguf_backend_flag_bisect_v486 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def _resolver_from_paths(paths: dict[str, Path]):
    def resolver(hf_id: str, _quant: str) -> str | None:
        return str(paths[hf_id]) if hf_id in paths else None

    return resolver


def _fake_preconditions(*, gpu_visible: bool = True, backend_ready: bool = True) -> dict[str, Any]:
    cli = "/opt/llama.cpp/build/bin/llama-cli" if backend_ready else None
    completion = "/opt/llama.cpp/build/bin/llama-completion" if backend_ready else None
    server = "/opt/llama.cpp/build/bin/llama-server" if backend_ready else None
    return {
        "gpu_visible": gpu_visible,
        "raw_nvidia_smi": {"ok": gpu_visible, "stdout": "CUDA UMD Version: 13.3"},
        "nvidia_smi": {
            "ok": gpu_visible,
            "stdout": "0, NVIDIA RTX 3090, 610.43.02, 24576, 24000, 0",
        },
        "cuda_driver": {"driver_version": "610.43.02", "cuda_version": "13.3"},
        "vram_before": [
            {"index": 0, "memory_used_mb": 4, "memory_free_mb": 24120},
            {"index": 1, "memory_used_mb": 4, "memory_free_mb": 24120},
        ],
        "free_vram_mb": 48240 if gpu_visible else 0,
        "free_disk": {"path": "/tmp", "free_bytes": 1_000_000_000_000},
        "binary_paths": {
            "llama-cli": cli,
            "llama-completion": completion,
            "llama-server": server,
        },
        "binary_versions": {
            "llama-cli": {"ok": backend_ready, "stderr": "version: 9606 CUDA"},
            "llama-completion": {"ok": backend_ready, "stderr": "version: 9606 CUDA"},
            "llama-server": {"ok": backend_ready, "stderr": "version: 9606 CUDA"},
        },
        "binary_dynamic_libraries": {
            "llama-cli": {"ok": backend_ready, "stdout": "libggml-cuda.so\nlibcuda.so"},
            "llama-completion": {"ok": backend_ready, "stdout": "libggml-cuda.so"},
            "llama-server": {"ok": backend_ready, "stdout": "libggml-cuda.so"},
        },
        "cuda_backend_evidence": backend_ready,
        "blocked_preconditions": [],
    }


def _ready_probe(**kwargs: Any) -> dict[str, Any]:
    variant = kwargs["variant"]
    return {
        "backend_kind": variant["backend_kind"],
        "backend_variant": variant["name"],
        "status": "completed",
        "timeout_class": "completed_no_timeout",
        "completed": True,
        "timed_out": False,
        "timeout_s": kwargs["timeout_s"],
        "wall_clock_s": 22.5,
        "load_s": 12.0,
        "first_token_latency_s": 18.25,
        "eight_token_generation_s": 0.18,
        "generated_token_count": 8,
        "eight_token_completion_status": "completed_8_tokens",
        "stdout_tail": "red blue green yellow orange purple black white",
        "stderr_tail": "load_tensors: offloaded 49/49 layers to GPU\nCUDA0",
        "returncode": 0,
        "backend_gpu_log_evidence": True,
        "command": variant["command"],
        "context": variant["context"],
        "batch": variant["batch"],
        "ubatch": variant["ubatch"],
        "gpu_layers": variant["gpu_layers"],
        "tensor_split": variant["tensor_split"],
        "prompt": variant["prompt"],
        "n_predict": variant["n_predict"],
        "gpu_memory_receipts": {
            "before": [{"index": 0, "memory_used_mb": 4}],
            "during": [[{"index": 0, "memory_used_mb": 9400}]],
            "after": [{"index": 0, "memory_used_mb": 4}],
            "max_memory_delta_mb": 9396,
            "offload_evidence": True,
        },
    }


def _no_offload_probe(**kwargs: Any) -> dict[str, Any]:
    receipt = _ready_probe(**kwargs)
    receipt["backend_gpu_log_evidence"] = False
    receipt["stderr_tail"] = "llama.cpp CPU only"
    receipt["gpu_memory_receipts"] = {
        "before": [{"index": 0, "memory_used_mb": 4}],
        "during": [[{"index": 0, "memory_used_mb": 4}]],
        "after": [{"index": 0, "memory_used_mb": 4}],
        "max_memory_delta_mb": 0,
        "offload_evidence": False,
    }
    return receipt


def test_req_verify_5323_spec_declares_native_backend_bisect_contract() -> None:
    """REQ-VERIFY-5323: OpenSpec anchors the native GGUF backend bisect."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5323") : spec.index("### REQ-VERIFY-5309")]

    for marker in (
        "REQ-VERIFY-5323",
        "SCENARIO-VERIFY-5323",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "llama-cli",
        "llama-completion",
        "llama-server",
        "--no-conversation",
        "GGML_ASSERT(n_tokens_all <= cparams.n_batch)",
        "sota_backend_candidate_ready",
        "runtime_unblocked_min_one_mandated",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5323_command_variants_repair_exp5309_flags() -> None:
    """REQ-VERIFY-5323: command forms avoid Exp 5309 flag and batch blockers."""

    model = {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "model_path": "/cache/gemma.gguf",
        "status": "local_gguf_resolved",
    }
    variants = mod.build_backend_variants(_fake_preconditions(), model)
    by_kind = {variant["backend_kind"]: variant for variant in variants}

    assert {"llama-cli", "llama-completion", "llama-server"} <= set(by_kind)
    cli_command = by_kind["llama-cli"]["command"]
    completion_command = by_kind["llama-completion"]["command"]
    server_command = by_kind["llama-server"]["command"]
    all_flags = cli_command + completion_command + server_command

    assert "--no-conversation" not in all_flags
    assert "-no-cnv" not in all_flags
    assert "-st" in cli_command
    assert "-st" in completion_command
    assert by_kind["llama-cli"]["batch"] >= 128
    assert by_kind["llama-cli"]["ubatch"] >= 64
    assert by_kind["llama-cli"]["context"] >= by_kind["llama-cli"]["batch"]
    assert by_kind["llama-cli"]["n_predict"] == 8


def test_scenario_verify_5323_opens_after_one_mandated_native_cli_receipt(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5323: one offloaded bounded receipt opens the candidate gate."""

    gguf = _write_minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    calls: list[tuple[str, str]] = []

    def probe(**kwargs: Any) -> dict[str, Any]:
        calls.append((kwargs["model_spec"]["role"], kwargs["variant"]["backend_kind"]))
        return _ready_probe(**kwargs)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_resolver_from_paths({mod.MANDATED_MODEL_SPECS[1]["hf_id"]: gguf}),
        preconditions_provider=_fake_preconditions,
        runtime_probe=probe,
        tests_run=[{"command": "unit ready", "outcome": "passed"}],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert calls == [("flagship_dense", "llama-cli")]
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["sota_backend_candidate_ready"] is True
    assert artifact["runtime_unblocked_min_one_mandated"] is True
    assert artifact["no_quality_claim"] is True
    assert artifact["timeout_or_crash_root_cause"]["value"] == "none"
    best = artifact["best_backend_command"]["value"]
    assert best["backend_kind"] == "llama-cli"
    assert best["model_role"] == "flagship_dense"
    assert "--no-conversation" not in best["command"]
    row = artifact["per_model_runtime_matrix"]["value"]["flagship_dense"]
    assert row["attempts"][0]["completed_load_first_token_and_8_tokens"] is True
    assert row["attempts"][0]["offload_authenticated"] is True
    assert row["attempts"][0]["first_token_latency_s"] == pytest.approx(18.25)
    assert row["attempts"][0]["eight_token_completion_status"] == "completed_8_tokens"


def test_req_verify_5323_blocks_when_compute_preconditions_absent(tmp_path: Path) -> None:
    """REQ-VERIFY-5323: missing compute preconditions block before fake probing."""

    calls: list[str] = []

    def forbidden_probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["model_spec"]["role"])
        raise AssertionError("runtime probe must not run when compute preconditions fail")

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked.json",
        model_resolver=lambda _hf_id, _quant: None,
        preconditions_provider=lambda: _fake_preconditions(gpu_visible=False),
        runtime_probe=forbidden_probe,
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert calls == []
    assert artifact["status"]["value"] == "blocked"
    assert artifact["sota_backend_candidate_ready"] is False
    assert artifact["runtime_unblocked_min_one_mandated"] is False
    assert "gpu_not_visible" in artifact["preconditions_checked"]["value"]["blocked_preconditions"]
    assert "no_mandated_sota_gguf_resolved" in artifact["timeout_or_crash_root_cause"]["value"]


def test_req_verify_5323_text_without_offload_stays_blocked(tmp_path: Path) -> None:
    """REQ-VERIFY-5323: 8 generated tokens alone cannot open the native backend gate."""

    gguf = _write_minimal_gguf(tmp_path / "gemma.gguf")
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-no-offload.json",
        model_resolver=_resolver_from_paths({mod.MANDATED_MODEL_SPECS[1]["hf_id"]: gguf}),
        preconditions_provider=_fake_preconditions,
        runtime_probe=_no_offload_probe,
        tests_run=[{"command": "unit no offload", "outcome": "passed"}],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["sota_backend_candidate_ready"] is False
    assert artifact["runtime_unblocked_min_one_mandated"] is False
    assert "no_authenticated_gpu_offload" in artifact["timeout_or_crash_root_cause"]["value"]
    row = artifact["per_model_runtime_matrix"]["value"]["flagship_dense"]
    assert row["attempts"][0]["completed_load_first_token_and_8_tokens"] is True
    assert row["attempts"][0]["offload_authenticated"] is False


def test_req_verify_5323_module_does_not_import_transformers_tokenizer() -> None:
    """REQ-VERIFY-5323: GGUF repos are loaded via paths, not AutoTokenizer."""

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


def test_req_verify_5323_root_cause_classifiers_are_precise() -> None:
    """REQ-VERIFY-5323: timeout and crash classes name the next blocker."""

    assert mod._total_used_mb([{"memory_used_mb": 7}, {"memory_used_mb": 11}]) == 18
    assert mod._free_vram_mb([{"memory_free_mb": 13}, {"memory_free_mb": 17}]) == 30
    assert mod._extract_ms(mod.LOAD_TIME_RE, "load time = 123.0 ms") == pytest.approx(0.123)
    assert mod._extract_ms(mod.LOAD_TIME_RE, "no timing") is None
    assert mod._extract_eval_runs("eval time = 7.0 ms / 8 runs") == 8
    assert mod._extract_eval_runs("no eval runs") is None
    assert mod._attempt_is_ready("bad") is False
    assert (
        mod._wrapped_value(
            {"experiment_id": {"value": mod.EXPERIMENT_ID, "principle": "wrong"}},
            "experiment_id",
        )
        is mod.MISSING_WRAPPED_VALUE
    )

    assert (
        mod.classify_runtime_receipt(
            {
                "completed": False,
                "timed_out": True,
                "first_token_latency_s": None,
                "generated_token_count": 0,
                "returncode": None,
            }
        )
        == "timeout_before_first_token"
    )
    assert (
        mod.classify_runtime_receipt(
            {
                "completed": False,
                "timed_out": False,
                "first_token_latency_s": 0.5,
                "generated_token_count": 0,
                "returncode": -6,
                "stderr_tail": "GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed",
            }
        )
        == "llama_context_batch_assert"
    )
    assert (
        mod.classify_runtime_receipt(
            {
                "completed": False,
                "timed_out": False,
                "first_token_latency_s": 0.5,
                "generated_token_count": 0,
                "returncode": 1,
                "stderr_tail": "--no-conversation unsupported",
            }
        )
        == "llama_cli_no_conversation_unsupported"
    )
    assert (
        mod.classify_runtime_receipt(
            {
                "completed": False,
                "timed_out": True,
                "first_token_latency_s": 0.5,
                "generated_token_count": 2,
                "returncode": None,
            }
        )
        == "timeout_during_8_token_generation"
    )
    assert (
        mod.classify_runtime_receipt(
            {
                "completed": False,
                "timed_out": False,
                "first_token_latency_s": 0.5,
                "generated_token_count": 8,
                "returncode": -6,
            }
        )
        == "native_llama_cpp_abort_signal"
    )
    assert (
        mod.classify_runtime_receipt(
            {
                "completed": False,
                "timed_out": False,
                "first_token_latency_s": None,
                "generated_token_count": 0,
                "returncode": 0,
            }
        )
        == "no_first_token"
    )
    assert (
        mod.classify_runtime_receipt(
            {
                "completed": False,
                "timed_out": False,
                "first_token_latency_s": 0.5,
                "generated_token_count": 2,
                "returncode": 0,
            }
        )
        == "generation_incomplete"
    )
    assert (
        mod.classify_runtime_receipt(
            {
                "completed": False,
                "timed_out": False,
                "first_token_latency_s": 0.5,
                "generated_token_count": 8,
                "returncode": 0,
            }
        )
        == "generation_incomplete"
    )

    malformed = tmp_path = Path("/tmp/nonexistent-exp5323-header.gguf")
    if malformed.exists():
        malformed.unlink()

    root = mod.timeout_or_crash_root_cause(
        {
            "flagship_dense": {
                "model_available": True,
                "attempts": [{"timeout_class": "llama_context_batch_assert"}],
            }
        },
        [],
    )
    assert root == "llama_context_batch_assert_after_native_backend_attempt"
    assert mod.timeout_or_crash_root_cause({"x": {"model_available": False}}, []) == (
        "no_mandated_sota_gguf_resolved"
    )
    assert mod.timeout_or_crash_root_cause({"x": {"model_available": True, "attempts": []}}, []) == (
        "no_native_backend_attempt_executed"
    )
    assert mod.timeout_or_crash_root_cause(
        {"x": {"model_available": True, "attempts": [{"timeout_class": "llama_cli_no_conversation_unsupported"}]}},
        [],
    ) == "llama_cli_no_conversation_unsupported_after_native_backend_attempt"
    assert mod.timeout_or_crash_root_cause(
        {"x": {"model_available": True, "attempts": [{"timeout_class": "timeout_before_first_token"}]}},
        [],
    ) == "all_attempted_native_backends_timeout_before_first_token"
    assert mod.timeout_or_crash_root_cause(
        {
            "x": {
                "model_available": True,
                "attempts": [{"timeout_class": "timeout_during_8_token_generation"}],
            }
        },
        [],
    ) == "timeout_during_bounded_8_token_generation"
    assert mod.timeout_or_crash_root_cause(
        {"x": {"model_available": True, "attempts": [{"timeout_class": "no_first_token"}]}},
        [],
    ) == "no_first_token_observed"
    assert "generation_incomplete" in mod.timeout_or_crash_root_cause(
        {"x": {"model_available": True, "attempts": [{"timeout_class": "generation_incomplete"}]}},
        [],
    )


def test_req_verify_5323_metadata_and_precondition_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-5323: metadata and no-variant edges fail closed."""

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
        model_resolver=_resolver_from_paths({mod.MANDATED_MODEL_SPECS[1]["hf_id"]: not_gguf}),
        preconditions_provider=_fake_preconditions,
        runtime_probe=lambda **_: pytest.fail("probe should not run for bad metadata"),
        tests_run=[],
        write=False,
    )
    mod.validate_artifact(artifact)
    assert artifact["MODEL_SPECS"]["value"]["flagship_dense"]["status"] == (
        "blocked_metadata_unreadable"
    )

    no_binary = _fake_preconditions()
    no_binary["binary_paths"] = {backend: None for backend in mod.NATIVE_BACKENDS}
    assert "no_native_llama_cpp_binary_available" in mod._precondition_blockers(
        no_binary,
        {
            "flagship_dense": {
                "status": "local_gguf_resolved",
                "hf_id": mod.MANDATED_MODEL_SPECS[1]["hf_id"],
            }
        },
    )
    no_cuda = _fake_preconditions()
    no_cuda["cuda_backend_evidence"] = False
    assert "native_llama_cpp_cuda_evidence_missing" in mod._precondition_blockers(
        no_cuda,
        {
            "flagship_dense": {
                "status": "local_gguf_resolved",
                "hf_id": mod.MANDATED_MODEL_SPECS[1]["hf_id"],
            }
        },
    )

    monkeypatch.setattr(mod, "build_backend_variants", lambda *_args, **_kwargs: [])
    gguf = _write_minimal_gguf(tmp_path / "gemma-ok.gguf")
    no_variant = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "no-variant.json",
        model_resolver=_resolver_from_paths({mod.MANDATED_MODEL_SPECS[1]["hf_id"]: gguf}),
        preconditions_provider=_fake_preconditions,
        runtime_probe=lambda **_: pytest.fail("probe should not run without variants"),
        tests_run=[],
        write=False,
    )
    assert no_variant["per_model_runtime_matrix"]["value"]["flagship_dense"][
        "best_attempt_status"
    ] == "not_attempted_no_native_variant"
    assert no_variant["timeout_or_crash_root_cause"]["value"] == (
        "no_native_backend_attempt_executed"
    )


def test_validate_artifact_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5323: schema validation catches malformed bisect artifacts."""

    gguf = _write_minimal_gguf(tmp_path / "gemma.gguf")
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "ready.json",
        model_resolver=_resolver_from_paths({mod.MANDATED_MODEL_SPECS[1]["hf_id"]: gguf}),
        preconditions_provider=_fake_preconditions,
        runtime_probe=_ready_probe,
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
                a.__setitem__("sota_backend_candidate_ready", "yes"),
                a,
            )[2],
            "experiment_id mismatch",
        ),
        (
            lambda a: (
                a["honest_verdict"].__setitem__("value", "done"),
                a["inference_substrate"].__setitem__("value", "wrong"),
                a,
            )[2],
            "honest_verdict",
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
                a["status"].__setitem__("value", "running"),
                a,
            )[1],
            "status",
        ),
        (
            lambda a: (
                a.__setitem__("no_quality_claim", False),
                a,
            )[1],
            "no_quality_claim",
        ),
        (
            lambda a: (
                a["per_model_runtime_matrix"]["value"]["flagship_dense"].__setitem__(
                    "attempts", "bad"
                ),
                a,
            )[1],
            "runtime attempts",
        ),
        (
            lambda a: (
                a["backend_matrix"].__setitem__("value", []),
                a,
            )[1],
            "backend_matrix",
        ),
        (
            lambda a: (
                a["tests_run"].__setitem__("value", "bad"),
                a,
            )[1],
            "tests_run",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("hf_id", "wrong"),
                a["per_model_runtime_matrix"]["value"]["flagship_dense"].__setitem__(
                    "autotokenizer_used", True
                ),
                a,
            )[2],
            "hf_id mismatch",
        ),
        (
            lambda a: (
                a["timeout_or_crash_root_cause"].__setitem__("value", "still_bad"),
                a,
            )[1],
            "ready artifact must have root cause none",
        ),
        (
            lambda a: (
                a.__setitem__("sota_backend_candidate_ready", False),
                a.__setitem__("runtime_unblocked_min_one_mandated", False),
                a["timeout_or_crash_root_cause"].__setitem__("value", ""),
                a,
            )[3],
            "blocked artifact must name root cause",
        ),
        (
            lambda a: (
                a["best_backend_command"].__setitem__("value", None),
                a["per_model_runtime_matrix"]["value"]["flagship_dense"]["attempts"][0].__setitem__(
                    "offload_authenticated", False
                ),
                a,
            )[2],
            "ready artifact must record best_backend_command",
        ),
        (
            lambda a: (
                a["per_model_runtime_matrix"]["value"]["flagship_dense"]["attempts"][0].__setitem__(
                    "offload_authenticated", False
                ),
                a,
            )[1],
            "ready artifact must contain an offloaded bounded attempt",
        ),
        (
            lambda a: (
                a.__setitem__("sota_backend_candidate_ready", False),
                a.__setitem__("runtime_unblocked_min_one_mandated", False),
                a["timeout_or_crash_root_cause"].__setitem__("value", "blocked"),
                a["best_backend_command"].__setitem__("value", {"bad": True}),
                a,
            )[4],
            "blocked artifact cannot contain best_backend_command",
        ),
        (
            lambda a: (
                a["best_backend_command"].__setitem__("value", None),
                a,
            )[1],
            "ready artifact must record best_backend_command",
        ),
    ]

    for mutate, expected in malformed_cases:
        joined = "; ".join(mod.artifact_schema_errors(mutate(clone())))
        assert expected in joined
    with pytest.raises(AssertionError, match="experiment_id mismatch"):
        bad = clone()
        bad["experiment_id"]["value"] = "wrong"
        mod.validate_artifact(bad)
