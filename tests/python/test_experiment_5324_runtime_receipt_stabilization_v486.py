"""Tests for Exp 5324 runtime receipt stabilization.

Spec refs: REQ-VERIFY-5324, SCENARIO-VERIFY-5324.
"""

from __future__ import annotations

import ast
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5324_runtime_receipt_stabilization_v486 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def _resolver_from_paths(paths: dict[str, Path]):
    def resolver(hf_id: str, _quant: str) -> str | None:
        return str(paths[hf_id]) if hf_id in paths else None

    return resolver


def _fake_prior_artifact(command: list[str], model_path: Path) -> dict[str, Any]:
    return {
        "experiment_id": {"value": mod.exp5323.EXPERIMENT_ID, "principle": "traceability"},
        "status": {"value": "complete", "principle": "status"},
        "honest_verdict": {"value": "complete: native candidate", "principle": "verdict"},
        "sota_backend_candidate_ready": True,
        "runtime_unblocked_min_one_mandated": True,
        "MODEL_SPECS": {
            "value": {
                "flagship_dense": {
                    "role": "flagship_dense",
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "model_path": str(model_path),
                    "status": "local_gguf_resolved",
                    "autotokenizer_used": False,
                }
            },
            "principle": "model specs",
        },
        "best_backend_command": {
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
                "first_token_latency_s": 19.0,
                "eight_token_generation_s": 0.2,
                "gpu_memory_delta_mb": 9000,
            },
            "principle": "best command",
        },
    }


def _write_prior(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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
        "binary_paths": {
            "llama-cli": binary,
            "llama-completion": None,
            "llama-server": None,
        },
        "binary_versions": {
            "llama-cli": {"ok": True, "stderr": "version: 9606 CUDA"},
            "llama-completion": {"ok": False, "stderr": "missing"},
            "llama-server": {"ok": False, "stderr": "missing"},
        },
        "binary_dynamic_libraries": {
            "llama-cli": {"ok": True, "stdout": "libggml-cuda.so\nlibcuda.so"},
            "llama-completion": {"ok": False, "stdout": ""},
            "llama-server": {"ok": False, "stdout": ""},
        },
        "cuda_backend_evidence": True,
        "blocked_preconditions": [],
    }


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
        "wall_clock_s": 21.0 + run_index,
        "load_s": 11.0 + run_index,
        "first_token_latency_s": 18.0 + run_index,
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
            "during": [[{"index": 0, "memory_used_mb": 9400 + run_index}]],
            "after": [{"index": 0, "memory_used_mb": 4}],
            "max_memory_delta_mb": 9396 + run_index,
            "offload_evidence": True,
        },
    }


def _timeout_probe(**kwargs: Any) -> dict[str, Any]:
    receipt = _ready_probe(**kwargs)
    if kwargs["run_index"] == 2:
        receipt.update(
            {
                "status": "timeout",
                "completed": False,
                "timed_out": True,
                "first_token_latency_s": None,
                "generated_token_count": 0,
                "eight_token_completion_status": "incomplete",
                "stderr_tail": "load_tensors: offloaded 49/49 layers to GPU",
                "returncode": None,
            }
        )
    return receipt


def test_req_verify_5324_spec_declares_stability_contract() -> None:
    """REQ-VERIFY-5324: OpenSpec anchors the runtime stability receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5324") : spec.index("### REQ-VERIFY-5323")]

    for marker in (
        "REQ-VERIFY-5324",
        "SCENARIO-VERIFY-5324",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "local_native_llama_cpp_stability_receipts",
        "sota_runtime_unblocked_stable",
        "quality_claim_permitted",
        "command_drift",
        "memory_pressure",
        "model_specific_assertion",
        "timeout",
        "missing_binary",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_verify_5324_three_replays_open_stability_gate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5324: three offloaded bounded replays mark runtime stable."""

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    gguf = _write_minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    command = [str(binary), "-m", str(gguf), "-p", mod.PROMPT, "-n", "8"]
    prior_path = _write_prior(
        tmp_path / mod.exp5323.RESULT_RELATIVE_PATH,
        _fake_prior_artifact(command, gguf),
    )
    calls: list[int] = []

    def probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["run_index"])
        return _ready_probe(**kwargs)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        prior_artifact_path=prior_path,
        model_resolver=_resolver_from_paths({"unsloth/gemma-4-31B-it-GGUF": gguf}),
        preconditions_provider=lambda: _fake_preconditions(str(binary)),
        runtime_probe=probe,
        tests_run=[{"command": "unit stable", "outcome": "passed"}],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert calls == [1, 2, 3]
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["sota_runtime_unblocked_stable"] is True
    assert artifact["quality_claim_permitted"] is False
    assert artifact["stability_failure_class"]["value"] == "none"
    assert artifact["selected_backend_command"]["value"]["command"] == command
    assert artifact["selected_model_spec"]["value"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    receipts = artifact["repeated_receipts"]["value"]
    assert [row["run_index"] for row in receipts] == [1, 2, 3]
    assert all(row["completed_load_first_token_and_8_tokens"] for row in receipts)
    assert all(row["offload_authenticated"] for row in receipts)
    assert receipts[0]["stderr_summary"] == "load_tensors: offloaded 49/49 layers to GPU\nCUDA0"
    assert receipts[1]["load_s"] == pytest.approx(13.0)


def test_req_verify_5324_blocks_before_probe_when_exp5323_candidate_missing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5324: absent Exp5323 candidate command blocks replay."""

    prior_path = _write_prior(
        tmp_path / mod.exp5323.RESULT_RELATIVE_PATH,
        {
            "sota_backend_candidate_ready": False,
            "runtime_unblocked_min_one_mandated": False,
            "best_backend_command": {"value": None},
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
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["sota_runtime_unblocked_stable"] is False
    assert artifact["quality_claim_permitted"] is False
    assert artifact["stability_failure_class"]["value"] == "command_drift"
    assert "exp5323_candidate_unavailable" in artifact["preconditions_checked"]["value"][
        "blocked_preconditions"
    ]


def test_req_verify_5324_timeout_in_any_replay_blocks_stability(tmp_path: Path) -> None:
    """REQ-VERIFY-5324: repeatability requires every bounded replay to finish."""

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    gguf = _write_minimal_gguf(tmp_path / "gemma.gguf")
    command = [str(binary), "-m", str(gguf), "-p", mod.PROMPT, "-n", "8"]
    prior_path = _write_prior(
        tmp_path / mod.exp5323.RESULT_RELATIVE_PATH,
        _fake_prior_artifact(command, gguf),
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-timeout.json",
        prior_artifact_path=prior_path,
        model_resolver=_resolver_from_paths({"unsloth/gemma-4-31B-it-GGUF": gguf}),
        preconditions_provider=lambda: _fake_preconditions(str(binary)),
        runtime_probe=_timeout_probe,
        tests_run=[{"command": "unit timeout", "outcome": "passed"}],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["sota_runtime_unblocked_stable"] is False
    assert artifact["stability_failure_class"]["value"] == "timeout"
    assert artifact["repeated_receipts"]["value"][1]["timeout_class"] == "timeout_before_first_token"
    assert artifact["selected_backend_command"]["value"]["command"] == command


def test_req_verify_5324_failure_classifiers_are_precise() -> None:
    """REQ-VERIFY-5324: failure classes distinguish downstream blockers."""

    assert mod.classify_precondition_failure(["selected_binary_missing"]) == "missing_binary"
    assert mod.classify_precondition_failure(["selected_model_file_missing"]) == "missing_binary"
    assert mod.classify_precondition_failure(["free_vram_unavailable"]) == "memory_pressure"
    assert mod.classify_precondition_failure(["selected_command_model_path_drift"]) == "command_drift"
    assert mod.classify_precondition_failure(["exp5323_candidate_unavailable"]) == "command_drift"
    assert mod.classify_precondition_failure(["native_llama_cpp_cuda_evidence_missing"]) == (
        "command_drift"
    )

    assert (
        mod.classify_stability_failure(
            [{"timeout_class": "llama_context_batch_assert", "stderr_summary": "assert"}],
            [],
        )
        == "model_specific_assertion"
    )
    assert (
        mod.classify_stability_failure(
            [{"timeout_class": "native_llama_cpp_abort_signal", "stderr_summary": "GGML_ASSERT"}],
            [],
        )
        == "model_specific_assertion"
    )
    assert (
        mod.classify_stability_failure(
            [{"timeout_class": "timeout_during_8_token_generation", "stderr_summary": ""}],
            [],
        )
        == "timeout"
    )
    assert (
        mod.classify_stability_failure(
            [{"timeout_class": "completed_no_timeout", "offload_authenticated": False}],
            [],
        )
        == "command_drift"
    )
    assert (
        mod.classify_stability_failure(
            [{"timeout_class": "no_first_token", "stderr_summary": "out of memory"}],
            [],
        )
        == "memory_pressure"
    )


def test_req_verify_5324_defensive_precondition_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5324: malformed prior state and current-resource drift fail closed."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json(tmp_path / "missing.json") == {}
    assert mod._read_json(bad_json) == {}
    assert mod._prior_candidate(
        {
            "sota_backend_candidate_ready": True,
            "runtime_unblocked_min_one_mandated": False,
        }
    ) is None
    assert mod._prior_candidate(
        {
            "sota_backend_candidate_ready": True,
            "runtime_unblocked_min_one_mandated": True,
            "best_backend_command": {"value": "bad"},
        }
    ) is None
    assert mod._prior_candidate(
        {
            "sota_backend_candidate_ready": True,
            "runtime_unblocked_min_one_mandated": True,
            "best_backend_command": {"value": {"command": [], "model_role": "flagship_dense"}},
        }
    ) is None

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    selected_path = _write_minimal_gguf(tmp_path / "selected.gguf")
    drift_path = _write_minimal_gguf(tmp_path / "drift.gguf")
    candidate = {
        "command": [str(binary), "-m", str(drift_path)],
        "model_role": "flagship_dense",
        "model_path": str(drift_path),
        "gpu_memory_delta_mb": 9999,
    }
    selected_model = {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "model_path": str(selected_path),
        "status": "local_gguf_resolved",
    }

    blockers = mod._precondition_blockers(
        preconditions={
            "gpu_visible": False,
            "free_vram_mb": 1,
            "cuda_backend_evidence": False,
        },
        candidate=candidate,
        selected_model=selected_model,
    )

    assert "gpu_not_visible" in blockers
    assert "free_vram_below_exp5323_delta" in blockers
    assert "selected_command_model_path_drift" in blockers
    assert "native_llama_cpp_cuda_evidence_missing" in blockers

    missing_blockers = mod._precondition_blockers(
        preconditions={"gpu_visible": True, "free_vram_mb": 1, "cuda_backend_evidence": True},
        candidate={"command": [str(tmp_path / "missing-binary")], "model_role": "flagship_dense"},
        selected_model={
            "model_path": str(tmp_path / "missing-model.gguf"),
            "status": "local_gguf_resolved",
        },
    )
    assert "selected_binary_missing" in missing_blockers
    assert "selected_model_file_missing" in missing_blockers

    no_vram_blockers = mod._precondition_blockers(
        preconditions={"gpu_visible": True, "free_vram_mb": 0, "cuda_backend_evidence": True},
        candidate=candidate,
        selected_model=selected_model,
    )
    assert "free_vram_unavailable" in no_vram_blockers


def test_req_verify_5324_module_does_not_import_transformers_tokenizer() -> None:
    """REQ-VERIFY-5324: GGUF repos are loaded via paths, not AutoTokenizer."""

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
    """REQ-VERIFY-5324: schema validation catches malformed stability artifacts."""

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    gguf = _write_minimal_gguf(tmp_path / "gemma.gguf")
    command = [str(binary), "-m", str(gguf), "-p", mod.PROMPT, "-n", "8"]
    prior_path = _write_prior(
        tmp_path / mod.exp5323.RESULT_RELATIVE_PATH,
        _fake_prior_artifact(command, gguf),
    )
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "ready.json",
        prior_artifact_path=prior_path,
        model_resolver=_resolver_from_paths({"unsloth/gemma-4-31B-it-GGUF": gguf}),
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
            lambda a: (
                a["experiment_id"].__setitem__("value", "wrong"),
                a,
            )[1],
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
                a.__setitem__("quality_claim_permitted", True),
                a,
            )[1],
            "quality_claim_permitted must be bare false",
        ),
        (
            lambda a: (
                a.__setitem__("sota_runtime_unblocked_stable", "yes"),
                a,
            )[1],
            "sota_runtime_unblocked_stable must be a bare boolean",
        ),
        (
            lambda a: (
                a["repeated_receipts"].__setitem__("value", []),
                a,
            )[1],
            "stable artifact must contain at least three receipts",
        ),
        (
            lambda a: (
                a["repeated_receipts"]["value"][0].__setitem__("offload_authenticated", False),
                a,
            )[1],
            "stable artifact receipts must all be ready",
        ),
        (
            lambda a: (
                a.__setitem__("sota_runtime_unblocked_stable", False),
                a["stability_failure_class"].__setitem__("value", "none"),
                a,
            )[2],
            "blocked artifact must name failure class",
        ),
        (
            lambda a: (
                a["selected_backend_command"].__setitem__("value", None),
                a,
            )[1],
            "selected_backend_command must be an object",
        ),
        (
            lambda a: (
                a["selected_model_spec"].__setitem__("value", []),
                a,
            )[1],
            "selected_model_spec must be an object",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"].pop("middle_moe"),
                a,
            )[1],
            "MODEL_SPECS roles mismatch",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("hf_id", "wrong"),
                a,
            )[1],
            "hf_id mismatch",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("autotokenizer_used", True),
                a,
            )[1],
            "autotokenizer_used",
        ),
        (
            lambda a: (
                a["status"].__setitem__("value", "running"),
                a,
            )[1],
            "status must be complete or blocked",
        ),
        (
            lambda a: (
                a["tests_run"].__setitem__("value", "bad"),
                a,
            )[1],
            "tests_run must be a list",
        ),
        (
            lambda a: (
                a["tests_run"].__setitem__("principle", "wrong"),
                a,
            )[1],
            "tests_run must be principle-wrapped",
        ),
        (
            lambda a: (
                a["repeated_receipts"].__setitem__("value", "bad"),
                a,
            )[1],
            "repeated_receipts must be a list",
        ),
        (
            lambda a: (
                a["status"].__setitem__("value", "blocked"),
                a,
            )[1],
            "stable artifact must have complete status",
        ),
        (
            lambda a: (
                a["stability_failure_class"].__setitem__("value", "timeout"),
                a,
            )[1],
            "stable artifact must have failure class none",
        ),
        (
            lambda a: (
                a.__setitem__("sota_runtime_unblocked_stable", False),
                a["stability_failure_class"].__setitem__("value", "timeout"),
                a["selected_backend_command"].__setitem__("value", {"model_role": "flagship_dense"}),
                a,
            )[3],
            "selected_backend_command must preserve command when present",
        ),
    ]

    for mutate, expected in malformed_cases:
        joined = "; ".join(mod.artifact_schema_errors(mutate(clone())))
        assert expected in joined
    with pytest.raises(AssertionError, match="experiment_id mismatch"):
        bad = clone()
        bad["experiment_id"]["value"] = "wrong"
        mod.validate_artifact(bad)
