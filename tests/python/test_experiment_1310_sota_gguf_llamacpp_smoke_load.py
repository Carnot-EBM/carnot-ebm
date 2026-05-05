"""Tests for Exp 1310 SOTA GGUF llama.cpp smoke-load probe.

Spec: REQ-INFER-SOTA-006,
      SCENARIO-INFER-SOTA-006-001,
      SCENARIO-INFER-SOTA-006-002,
      SCENARIO-INFER-SOTA-006-003
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting.sota_gguf_llamacpp_smoke_load import (
    REQUIRED_ARTIFACT_FIELDS,
    _completion_text,
    _completion_token_count,
    _import_llama_class,
    _probe_gpu_memory_gb,
    _quantization_suffix,
    build_smoke_load_artifact,
    run_experiment,
)


QWEN_SPEC = {
    "name": "Qwen3.6-35B-A3B",
    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "gpu": 0,
    "model_path": "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
}
GEMMA_SPEC = {
    "name": "Gemma4-31B-it",
    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
    "gpu": 1,
    "model_path": "/cache/gemma-4-31B-it-Q4_K_M.gguf",
}


def _write_exp1309(project_root: Path, *, ready: bool) -> Path:
    results = project_root / "results"
    results.mkdir(exist_ok=True)
    path = results / "experiment_1309_sota_gguf_pair_resolver_repair.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "sota_pair_ready": ready,
                "honest_verdict": "ready" if ready else "not_ready",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _cached_pair(*, gpu_indices: tuple[int, int], preferred_quant: str) -> list[dict[str, Any]]:
    assert gpu_indices == (0, 1)
    assert preferred_quant == "Q4_K_M"
    return [dict(QWEN_SPEC), dict(GEMMA_SPEC)]


class _FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def __call__(self) -> float:
        return self._values.pop(0)


class _FakeLlama:
    calls: list[dict[str, Any]] = []
    closed: int = 0

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        self.model_path = kwargs["model_path"]

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert "Carnot smoke-load check" in prompt
        assert kwargs["max_tokens"] == 4
        assert kwargs["temperature"] == 0.0
        return {"choices": [{"text": " ok"}], "usage": {"completion_tokens": 4}}

    def close(self) -> None:
        type(self).closed += 1


def test_exp1310_ready_pair_smoke_loads_and_writes_artifact(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-006 / SCENARIO-INFER-SOTA-006-001: two models smoke-load."""
    _write_exp1309(tmp_path, ready=True)
    _FakeLlama.calls = []
    _FakeLlama.closed = 0
    output_path = tmp_path / "results" / "experiment_1310_sota_gguf_llamacpp_smoke_load.json"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        output_path=output_path,
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (True, _FakeLlama, None),
        gpu_memory_fn=lambda gpu_indices: {str(gpu): 10.0 + gpu for gpu in gpu_indices},
        monotonic=_FakeClock([0.0, 0.5, 1.0, 1.25]),
        max_tokens=4,
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["models_loaded"] == 2
    assert artifact["llama_cpp_import_ok"] is True
    assert artifact["tokens_per_second"] == pytest.approx(10.6667)
    assert artifact["gpu_memory_gb"] == {"0": 10.0, "1": 11.0}
    assert artifact["model_specs_count"] == 2
    assert artifact["models_used"] == [QWEN_SPEC["hf_id"], GEMMA_SPEC["hf_id"]]
    assert artifact["headline_result_possible"] is True
    assert artifact["honest_verdict"] == "sota_pair_llamacpp_smoke_loaded"
    assert artifact["resolved_model_specs"][0]["quantization_suffix"] == "UD-Q4_K_M"
    assert artifact["resolved_model_specs"][1]["quantization_suffix"] == "Q4_K_M"
    assert all(row["generated"] for row in artifact["per_model_results"])
    assert [call["main_gpu"] for call in _FakeLlama.calls] == [0, 1]
    assert _FakeLlama.closed == 2


def test_exp1310_llama_import_failure_blocks_without_generation(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-006-002: import errors are recorded without fake tokens."""
    _write_exp1309(tmp_path, ready=True)

    artifact = build_smoke_load_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (False, None, "ImportError: no module named llama_cpp"),
        gpu_memory_fn=lambda gpu_indices: {"0": 1.0, "1": 1.5},
    )

    assert artifact["status"] == "complete"
    assert artifact["llama_cpp_import_ok"] is False
    assert artifact["llama_cpp_import_error"] == "ImportError: no module named llama_cpp"
    assert artifact["models_loaded"] == 0
    assert artifact["tokens_per_second"] == 0.0
    assert artifact["headline_result_possible"] is False
    assert artifact["honest_verdict"] == "blocked_llama_cpp_import_failed"
    assert all(not row["generated"] for row in artifact["per_model_results"])
    assert all(row["error"] == "llama_cpp_import_failed" for row in artifact["per_model_results"])


def test_exp1310_exp1309_not_ready_blocks_before_runtime_probe(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-006-003: a failed resolver-repair gate stops the probe."""
    _write_exp1309(tmp_path, ready=False)

    def fail_cached_pair(**_: Any) -> list[dict[str, Any]]:
        raise AssertionError("cached_sota_pair must not run when Exp 1309 is blocked")

    artifact = build_smoke_load_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=fail_cached_pair,
        llama_importer=lambda: (_ for _ in ()).throw(AssertionError("llama import attempted")),
    )

    assert artifact["status"] == "complete"
    assert artifact["blocked_reason"] == "exp1309_sota_pair_not_ready"
    assert artifact["models_loaded"] == 0
    assert artifact["llama_cpp_import_attempted"] is False
    assert artifact["llama_cpp_import_ok"] is False
    assert artifact["model_specs_count"] == 0
    assert artifact["models_used"] == []
    assert artifact["headline_result_possible"] is False
    assert artifact["honest_verdict"] == "blocked_exp1309_sota_pair_not_ready"


def test_exp1310_missing_exp1309_artifact_blocks_before_runtime_probe(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-006-003: missing Exp 1309 is treated as not ready."""

    artifact = build_smoke_load_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=lambda **_: pytest.fail("cached_sota_pair must not run"),
        llama_importer=lambda: pytest.fail("llama import must not run"),
    )

    assert artifact["exp1309_gate"]["artifact_found"] is False
    assert artifact["blocked_reason"] == "exp1309_sota_pair_not_ready"
    assert artifact["honest_verdict"] == "blocked_exp1309_sota_pair_not_ready"


def test_exp1310_cached_pair_none_blocks_after_exp1309_ready(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-006: resolver returning no pair is an honest blocker."""
    _write_exp1309(tmp_path, ready=True)

    artifact = build_smoke_load_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        llama_importer=lambda: (_ for _ in ()).throw(AssertionError("llama import attempted")),
    )

    assert artifact["blocked_reason"] == "cached_sota_pair_not_loadable"
    assert artifact["llama_cpp_import_attempted"] is False
    assert artifact["model_specs_count"] == 0
    assert artifact["honest_verdict"] == "blocked_cached_sota_pair_not_loadable"


def test_exp1310_cached_pair_exception_blocks_after_exp1309_ready(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-006: resolver exceptions are terminal and explicit."""
    _write_exp1309(tmp_path, ready=True)

    def bad_cached_pair(**_: Any) -> list[dict[str, Any]]:
        raise RuntimeError("resolver exploded")

    artifact = build_smoke_load_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=bad_cached_pair,
        llama_importer=lambda: pytest.fail("llama import must not run"),
    )

    assert artifact["blocked_reason"] == "cached_sota_pair_exception"
    assert artifact["cached_sota_pair_error"] == "RuntimeError: resolver exploded"
    assert artifact["honest_verdict"] == "blocked_cached_sota_pair_exception"


def test_exp1310_invalid_cached_pair_shape_blocks_after_exp1309_ready(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-006: non-mandated or pathless specs cannot be smoke-loaded."""
    _write_exp1309(tmp_path, ready=True)

    artifact = build_smoke_load_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [
            {**QWEN_SPEC, "hf_id": "legacy/small-model"},
            {**GEMMA_SPEC, "model_path": None},
        ],
        llama_importer=lambda: pytest.fail("llama import must not run"),
    )

    assert artifact["model_specs_count"] == 0
    assert artifact["blocked_reason"] == "cached_sota_pair_not_loadable"


def test_exp1310_model_generation_failure_is_recorded(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-006: per-model exceptions keep the final verdict honest."""
    _write_exp1309(tmp_path, ready=True)

    class FailingLlama:
        def __init__(self, **_: Any) -> None:
            pass

        def __call__(self, *_: Any, **__: Any) -> dict[str, Any]:
            raise RuntimeError("generation failed")

    artifact = build_smoke_load_artifact(
        project_root=tmp_path,
        run_date="20260505",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (True, FailingLlama, None),
        monotonic=_FakeClock([0.0, 0.1, 0.2, 0.3]),
    )

    assert artifact["llama_cpp_import_ok"] is True
    assert artifact["models_loaded"] == 2
    assert artifact["headline_result_possible"] is False
    assert artifact["honest_verdict"] == "sota_pair_smoke_load_failed"
    assert all(row["load_success"] for row in artifact["per_model_results"])
    assert all(row["token_count"] == 0 for row in artifact["per_model_results"])
    assert all("RuntimeError: generation failed" in row["error"] for row in artifact["per_model_results"])


def test_exp1310_import_helper_reports_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-INFER-SOTA-006: the local llama.cpp import helper reports exact state."""

    class DummyLlama:
        pass

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=DummyLlama))
    assert _import_llama_class() == (True, DummyLlama, None)

    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    original_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "llama_cpp":
            raise ImportError("no module named llama_cpp")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert _import_llama_class() == (False, None, "ImportError: no module named llama_cpp")


def test_exp1310_token_helpers_cover_fallbacks() -> None:
    """REQ-INFER-SOTA-006: fallback token accounting remains deterministic."""

    class TokenizingLlama:
        def tokenize(self, text: bytes, *, add_bos: bool) -> list[int]:
            assert text == b"alpha beta"
            assert add_bos is False
            return [1, 2, 3]

    class BrokenTokenizer:
        def tokenize(self, text: bytes, *, add_bos: bool) -> list[int]:
            raise RuntimeError("tokenizer failed")

    assert _completion_text("plain text") == "plain text"
    assert _completion_text({"choices": []}) == ""
    assert _completion_token_count({"choices": []}, "alpha beta", TokenizingLlama()) == 3
    assert _completion_token_count({"choices": []}, "alpha beta", BrokenTokenizer()) == 2
    assert _completion_token_count({"choices": []}, "", object()) == 0
    assert _quantization_suffix(None) is None
    assert _quantization_suffix("/cache/model-f16.gguf") == "unknown"


def test_exp1310_gpu_memory_probe_uses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-INFER-SOTA-006: GPU memory is recorded when nvidia-smi is available."""

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "--query-gpu=index,memory.used" in cmd
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="bad-line\nx, y\n0, 1024\n1, 2048\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _probe_gpu_memory_gb([0, 1]) == {"0": 1.0, "1": 2.0}
    assert _probe_gpu_memory_gb([1]) == {"1": 2.0}


def test_exp1310_gpu_memory_probe_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFER-SOTA-006: missing GPU tooling records an empty memory snapshot."""

    def fake_run(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _probe_gpu_memory_gb([0, 1]) == {}
