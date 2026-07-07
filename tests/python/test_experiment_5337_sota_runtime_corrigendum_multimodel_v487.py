"""Tests for Exp 5337 SOTA runtime corrigendum.

Spec refs: REQ-VERIFY-5337, SCENARIO-VERIFY-5337.
"""

from __future__ import annotations

import ast
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5337_sota_runtime_corrigendum_multimodel_v487 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def _resolver_from_paths(paths: dict[str, Path]):
    def resolver(hf_id: str, _quant: str) -> str | None:
        return str(paths[hf_id]) if hf_id in paths else None

    return resolver


def _fake_preconditions(binary: str, *, gpu_visible: bool = True) -> dict[str, Any]:
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
        "binary_paths": {"llama-cli": binary},
        "binary_versions": {"llama-cli": {"ok": True, "stderr": "version: 9606 CUDA"}},
        "binary_dynamic_libraries": {
            "llama-cli": {"ok": True, "stdout": "libggml-cuda.so\nlibcuda.so"}
        },
        "cuda_backend_evidence": True,
        "blocked_preconditions": [],
    }


def _fake_exp5324_artifact(command: list[str], model_path: Path) -> dict[str, Any]:
    return {
        "status": {"value": "complete", "principle": "status"},
        "honest_verdict": {"value": "complete: stable", "principle": "verdict"},
        "sota_runtime_unblocked_stable": True,
        "quality_claim_permitted": False,
        "selected_backend_command": {
            "value": {
                "model_role": "flagship_dense",
                "backend_kind": "llama-cli",
                "backend_variant": "llama-cli-single-turn-batch512",
                "command": command,
                "model_path": str(model_path),
                "context": 512,
                "batch": 512,
                "ubatch": 128,
                "gpu_layers": "all",
                "tensor_split": None,
                "prompt": mod.PROMPT,
                "n_predict": 8,
                "timeout_s": 240.0,
                "gpu_memory_delta_mb": 9000,
            },
            "principle": "selected command",
        },
    }


def _write_prior(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ready_probe(**kwargs: Any) -> dict[str, Any]:
    run_index = kwargs["run_index"]
    variant = kwargs["variant"]
    return {
        "backend_kind": variant["backend_kind"],
        "backend_variant": variant["name"],
        "status": "completed",
        "timeout_class": "completed_no_timeout",
        "completed": True,
        "timed_out": False,
        "timeout_s": kwargs["timeout_s"],
        "wall_clock_s": 20.25 + run_index / 10,
        "load_s": 10.0 + run_index,
        "first_token_latency_s": 18.0 + run_index,
        "eight_token_generation_s": 0.2,
        "generated_token_count": 8,
        "eight_token_completion_status": "completed_8_tokens",
        "stdout_tail": "[Start thinking]\n* count colors\n[Final answer]\nred blue green yellow orange purple black white",
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
            "during": [[{"index": 0, "memory_used_mb": 9400 + run_index}]],
            "after": [{"index": 0, "memory_used_mb": 4}],
            "max_memory_delta_mb": 9396 + run_index,
            "offload_evidence": True,
        },
    }


def _short_probe(**kwargs: Any) -> dict[str, Any]:
    receipt = _ready_probe(**kwargs)
    receipt["wall_clock_s"] = 8.0
    return receipt


def test_req_verify_5337_spec_declares_corrigendum_contract() -> None:
    """REQ-VERIFY-5337: OpenSpec anchors the clean runtime corrigendum."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5337") :]

    for marker in (
        "REQ-VERIFY-5337",
        "SCENARIO-VERIFY-5337",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "live_llm_inference",
        "methodology_duration_s",
        "sota_runtime_clean_receipt_ready",
        "runtime_unblocked_min_one_mandated",
        "quality_claim_permitted=false",
        "no_autotokenizer_used=true",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_verify_5337_clean_dense_repeat_plan_opens_gate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5337: >=60s offloaded dense replay opens the runtime gate."""

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    dense = _write_minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    command = [str(binary), "-m", str(dense), "-p", mod.PROMPT, "-n", "8"]
    prior_path = _write_prior(
        tmp_path / mod.exp5324.RESULT_RELATIVE_PATH,
        _fake_exp5324_artifact(command, dense),
    )
    calls: list[int] = []

    def probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["run_index"])
        return _ready_probe(**kwargs)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        prior_artifact_path=prior_path,
        model_resolver=_resolver_from_paths({"unsloth/gemma-4-31B-it-GGUF": dense}),
        preconditions_provider=lambda: _fake_preconditions(str(binary)),
        runtime_probe=probe,
        tests_run=[{"command": "unit clean", "outcome": "passed"}],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert calls == [1, 2, 3]
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == "live_llm_inference"
    assert artifact["methodology_duration_s"] >= 60
    assert artifact["sota_runtime_clean_receipt_ready"] is True
    assert artifact["runtime_unblocked_min_one_mandated"] is True
    assert artifact["quality_claim_permitted"] is False
    assert artifact["no_autotokenizer_used"] is True
    dense_row = artifact["multi_model_receipt_matrix"]["value"]["flagship_dense"]
    assert dense_row["status"] == "clean_live_receipt_ready"
    assert dense_row["repeat_count"] == 3
    assert dense_row["blocked_reason"] is None
    assert dense_row["answer_text_separable_from_thinking_text"] is True
    assert artifact["runtime_corrigendum_receipt"]["value"]["clean_receipt_ready"] is True
    assert artifact["runtime_corrigendum_receipt"]["value"]["model_role"] == "flagship_dense"
    assert artifact["selected_backend_command"]["value"]["repeat_plan"]["minimum_total_duration_s"] == 60.0
    assert artifact["MODEL_SPECS"]["value"]["flagship_dense"]["autotokenizer_used"] is False


def test_req_verify_5337_duration_floor_blocks_short_live_receipt(tmp_path: Path) -> None:
    """REQ-VERIFY-5337: live_llm_inference receipt must clear the 60s floor."""

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    dense = _write_minimal_gguf(tmp_path / "gemma.gguf")
    command = [str(binary), "-m", str(dense), "-p", mod.PROMPT, "-n", "8"]
    prior_path = _write_prior(
        tmp_path / mod.exp5324.RESULT_RELATIVE_PATH,
        _fake_exp5324_artifact(command, dense),
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-short.json",
        prior_artifact_path=prior_path,
        model_resolver=_resolver_from_paths({"unsloth/gemma-4-31B-it-GGUF": dense}),
        preconditions_provider=lambda: _fake_preconditions(str(binary)),
        runtime_probe=_short_probe,
        tests_run=[{"command": "unit short", "outcome": "passed"}],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["methodology_duration_s"] == pytest.approx(24.0)
    assert artifact["sota_runtime_clean_receipt_ready"] is False
    assert artifact["runtime_unblocked_min_one_mandated"] is False
    assert artifact["runtime_corrigendum_receipt"]["value"]["blocked_reason"] == (
        "methodology_duration_below_60s"
    )


def test_req_verify_5337_blocks_before_probe_when_exp5324_command_missing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5337: absent Exp5324 stable command blocks live replay."""

    prior_path = _write_prior(
        tmp_path / mod.exp5324.RESULT_RELATIVE_PATH,
        {
            "sota_runtime_unblocked_stable": False,
            "selected_backend_command": {"value": None},
        },
    )
    calls: list[int] = []

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked.json",
        prior_artifact_path=prior_path,
        model_resolver=lambda _hf_id, _quant: None,
        preconditions_provider=lambda: _fake_preconditions("/missing/llama-cli"),
        runtime_probe=lambda **kwargs: calls.append(kwargs["run_index"]) or _ready_probe(**kwargs),
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert calls == []
    assert artifact["status"]["value"] == "blocked"
    assert artifact["sota_runtime_clean_receipt_ready"] is False
    assert "exp5324_stable_command_unavailable" in artifact["preconditions_checked"]["value"][
        "blocked_preconditions"
    ]
    assert artifact["multi_model_receipt_matrix"]["value"]["flagship_dense"]["status"] == "blocked"


def test_req_verify_5337_helpers_and_no_transformers_tokenizer(tmp_path: Path) -> None:
    """REQ-VERIFY-5337: helper logic stays precise and avoids AutoTokenizer."""

    assert mod._read_json(Path("/tmp/definitely-missing-exp5337.json")) == {}
    assert mod._prior_stable_command({"sota_runtime_unblocked_stable": False}) is None
    assert (
        mod._prior_stable_command(
            {"sota_runtime_unblocked_stable": True, "selected_backend_command": {"value": "bad"}}
        )
        is None
    )
    assert (
        mod._prior_stable_command(
            {
                "sota_runtime_unblocked_stable": True,
                "selected_backend_command": {"value": {"command": [], "model_role": "flagship_dense"}},
            }
        )
        is None
    )
    assert (
        mod._prior_stable_command(
            {
                "sota_runtime_unblocked_stable": True,
                "selected_backend_command": {
                    "value": {"command": ["/bin/echo"], "model_role": "middle_moe"}
                },
            }
        )
        is None
    )

    assert mod.answer_text_separable_from_thinking_text(
        "[Start thinking]\nplan\n[Final answer]\nred blue"
    )
    assert not mod.answer_text_separable_from_thinking_text("[Start thinking]\nplan only")
    assert mod._extract_final_answer_text("<think>plan</think>answer") == "answer"
    assert mod._extract_final_answer_text("no marker") is None
    assert mod.classify_clean_receipt(
        {
            "completed_load_first_token_and_8_tokens": True,
            "offload_authenticated": True,
            "timeout_class": "completed_no_timeout",
            "timed_out": False,
        }
    ) == "ready"
    assert mod.classify_clean_receipt({"timeout_class": "timeout_before_first_token"}) == "timeout"
    assert mod.classify_clean_receipt(
        {"completed_load_first_token_and_8_tokens": True, "offload_authenticated": False}
    ) == "offload_not_authenticated"
    assert mod.classify_clean_receipt({"timeout_class": "native_llama_cpp_abort_signal"}) == (
        "runtime_crash"
    )
    assert mod.classify_clean_receipt({"timeout_class": "generation_incomplete"}) == (
        "generation_incomplete"
    )
    assert mod.classify_clean_receipt(
        {
            "completed_load_first_token_and_8_tokens": True,
            "offload_authenticated": True,
            "timeout_class": "custom_nonready_class",
        }
    ) == "custom_nonready_class"
    assert mod._receipt_blocked_reason([], []) == "no_dense_receipt_attempted"
    assert mod._receipt_blocked_reason(
        [{"completed_load_first_token_and_8_tokens": False, "timeout_class": "no_first_token"}],
        [],
    ).startswith("receipt_not_clean:")

    assert mod._optional_model_row(
        "flagship_moe",
        {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "status": "missing_local_gguf"},
        attempt_optional_models=False,
    )["blocked_reason"] == "model_file_missing_or_metadata_unreadable"
    assert mod._optional_model_row(
        "flagship_moe",
        {
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "status": "local_gguf_resolved",
            "model_path": "/tmp/qwen.gguf",
        },
        attempt_optional_models=True,
    )["blocked_reason"] == "optional_probe_hook_not_used_in_this_corrigendum_run"
    assert mod._optional_model_row(
        "flagship_moe",
        {
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "status": "local_gguf_resolved",
            "model_path": "/tmp/qwen.gguf",
        },
        attempt_optional_models=False,
    )["blocked_reason"] == "optional_probe_skipped_to_preserve_stable_dense_receipt_after_clean_gate"

    blockers = mod._precondition_blockers(
        preconditions={"gpu_visible": False, "free_vram_mb": 1, "cuda_backend_evidence": False},
        candidate={
            "command": ["/tmp/missing-llama-cli"],
            "model_path": "/tmp/candidate.gguf",
            "gpu_memory_delta_mb": 9999,
        },
        selected_model={"status": "local_gguf_resolved", "model_path": "/tmp/selected.gguf"},
    )
    assert "gpu_not_visible" in blockers
    assert "free_vram_below_exp5324_delta" in blockers
    assert "selected_binary_missing" in blockers
    assert "selected_model_file_missing" in blockers
    assert "native_llama_cpp_cuda_evidence_missing" in blockers

    candidate_model = _write_minimal_gguf(tmp_path / "candidate.gguf")
    selected_model = _write_minimal_gguf(tmp_path / "selected.gguf")
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    drift_blockers = mod._precondition_blockers(
        preconditions={"gpu_visible": True, "free_vram_mb": 1000, "cuda_backend_evidence": True},
        candidate={"command": [str(binary)], "model_path": str(candidate_model)},
        selected_model={"status": "local_gguf_resolved", "model_path": str(selected_model)},
    )
    assert "selected_command_model_path_drift" in drift_blockers

    assert mod._precondition_blockers(
        preconditions={"gpu_visible": True, "free_vram_mb": 0, "cuda_backend_evidence": True},
        candidate=None,
        selected_model=None,
    ) == [
        "exp5324_stable_command_unavailable",
        "free_vram_unavailable",
        "selected_model_file_missing",
    ]
    assert mod._dense_blocked_row(
        {"hf_id": "unsloth/gemma-4-31B-it-GGUF", "model_path": "/tmp/gemma.gguf"},
        [],
        [{"completed_load_first_token_and_8_tokens": False, "timeout_class": "no_first_token"}],
        1.0,
    )["blocked_reason"].startswith("receipt_not_clean:")

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


def test_validate_artifact_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5337: schema validation catches malformed clean receipts."""

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    dense = _write_minimal_gguf(tmp_path / "gemma.gguf")
    command = [str(binary), "-m", str(dense), "-p", mod.PROMPT, "-n", "8"]
    prior_path = _write_prior(
        tmp_path / mod.exp5324.RESULT_RELATIVE_PATH,
        _fake_exp5324_artifact(command, dense),
    )
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "ready.json",
        prior_artifact_path=prior_path,
        model_resolver=_resolver_from_paths({"unsloth/gemma-4-31B-it-GGUF": dense}),
        preconditions_provider=lambda: _fake_preconditions(str(binary)),
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
            lambda a: (a["experiment_id"].__setitem__("value", "wrong"), a)[1],
            "experiment_id mismatch",
        ),
        (
            lambda a: (a["honest_verdict"].__setitem__("value", "done"), a)[1],
            "honest_verdict",
        ),
        (
            lambda a: (a["honest_verdict"].__setitem__("principle", "wrong"), a)[1],
            "honest_verdict must be principle-wrapped",
        ),
        (
            lambda a: (a["inference_substrate"].__setitem__("value", "wrong"), a)[1],
            "inference_substrate mismatch",
        ),
        (
            lambda a: (a["status"].__setitem__("value", "running"), a)[1],
            "status must be complete or blocked",
        ),
        (
            lambda a: (a.__setitem__("methodology_duration_s", "60"), a)[1],
            "methodology_duration_s must be numeric",
        ),
        (
            lambda a: (a.__setitem__("sota_runtime_clean_receipt_ready", "yes"), a)[1],
            "sota_runtime_clean_receipt_ready must be a bare boolean",
        ),
        (
            lambda a: (a.__setitem__("runtime_unblocked_min_one_mandated", "yes"), a)[1],
            "runtime_unblocked_min_one_mandated must be a bare boolean",
        ),
        (
            lambda a: (a.__setitem__("quality_claim_permitted", True), a)[1],
            "quality_claim_permitted must be bare false",
        ),
        (
            lambda a: (a.__setitem__("no_autotokenizer_used", False), a)[1],
            "no_autotokenizer_used must be bare true",
        ),
        (
            lambda a: (a["MODEL_SPECS"]["value"].pop("middle_moe"), a)[1],
            "MODEL_SPECS roles mismatch",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__(
                    "autotokenizer_used", True
                ),
                a,
            )[1],
            "autotokenizer_used",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("hf_id", "wrong"),
                a,
            )[1],
            "hf_id mismatch",
        ),
        (
            lambda a: (a["runtime_corrigendum_receipt"].__setitem__("value", []), a)[1],
            "runtime_corrigendum_receipt must be an object",
        ),
        (
            lambda a: (a["multi_model_receipt_matrix"].__setitem__("value", []), a)[1],
            "multi_model_receipt_matrix must be an object",
        ),
        (
            lambda a: (a["multi_model_receipt_matrix"]["value"].pop("middle_moe"), a)[1],
            "multi_model_receipt_matrix roles mismatch",
        ),
        (
            lambda a: (a["tests_run"].__setitem__("value", "bad"), a)[1],
            "tests_run must be a list",
        ),
        (
            lambda a: (
                a.__setitem__("sota_runtime_clean_receipt_ready", True),
                a.__setitem__("methodology_duration_s", 59.9),
                a,
            )[2],
            "clean receipt cannot be ready below 60s",
        ),
        (
            lambda a: (a["status"].__setitem__("value", "blocked"), a)[1],
            "clean artifact must have complete status",
        ),
        (
            lambda a: (
                a.__setitem__("runtime_unblocked_min_one_mandated", False),
                a,
            )[1],
            "clean artifact must unblock",
        ),
        (
            lambda a: (
                a["multi_model_receipt_matrix"]["value"]["flagship_dense"].__setitem__(
                    "status", "blocked"
                ),
                a,
            )[1],
            "flagship_dense row must be clean",
        ),
        (
            lambda a: (
                a.__setitem__("sota_runtime_clean_receipt_ready", False),
                a,
            )[1],
            "blocked artifact must have blocked status",
        ),
        (
            lambda a: (
                a.__setitem__("sota_runtime_clean_receipt_ready", False),
                a["status"].__setitem__("value", "blocked"),
                a.__setitem__("runtime_unblocked_min_one_mandated", True),
                a,
            )[3],
            "blocked artifact must not unblock runtime",
        ),
    ]

    for mutate, expected in malformed_cases:
        with pytest.raises(AssertionError, match=expected):
            mod.validate_artifact(mutate(clone()))
