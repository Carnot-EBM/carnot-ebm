"""Tests for Exp 1442 live SOTA repair runtime preflight.

Spec: REQ-INFER-SOTA-007,
      SCENARIO-INFER-SOTA-007-001,
      SCENARIO-INFER-SOTA-007-002
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import live_sota_repair_runtime_preflight as mod
from carnot.reporting.live_sota_repair_runtime_preflight import (
    MANDATED_MODEL_SPECS,
    REQUIRED_ARTIFACT_FIELDS,
    _build_probe_command,
    _completion_text,
    _extract_json_from_stdout,
    _summarize_stream,
    build_live_runtime_preflight_artifact,
    probe_gpu_state,
    run_experiment,
    run_live_probe_one,
    run_live_probe_subprocess,
)


QWEN_PATH = "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
GEMMA_PATH = "/cache/gemma-4-31B-it-Q4_K_M.gguf"


def _resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
    assert preferred_quant == "Q4_K_M"
    if hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF":
        return QWEN_PATH
    if hf_id == "unsloth/gemma-4-31B-it-GGUF":
        return GEMMA_PATH
    return None


def _cached_pair(*, gpu_indices: tuple[int, int], preferred_quant: str) -> list[dict[str, Any]]:
    assert gpu_indices == (0, 1)
    assert preferred_quant == "Q4_K_M"
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": QWEN_PATH,
        },
        {
            "name": "Gemma4-31B-it",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "gpu": 1,
            "model_path": GEMMA_PATH,
        },
    ]


def test_exp1442_successful_live_probe_opens_runtime_gate(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-007 / SCENARIO-INFER-SOTA-007-001: live response opens gate."""

    def fake_smoke(model: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "hf_id": model["hf_id"],
            "role": model["role"],
            "model_path": model["model_path"],
            "command": ["python", "-m", "probe", "--model-path", model["model_path"]],
            "runtime_mode": "llama_cpp_subprocess_gpu",
            "returncode": 0,
            "stdout_summary": "{\"usable_response\": true}",
            "stderr_summary": "",
            "elapsed_s": 1.25,
            "truly_live": True,
            "usable_response": True,
            "response_text_preview": "Use repair_action STEP_REWRITE.",
            "blocker": None,
        }

    artifact = build_live_runtime_preflight_artifact(
        project_root=tmp_path,
        run_date="20260506",
        cache_resolver=_resolver,
        cached_pair_fn=_cached_pair,
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
        smoke_probe_fn=fake_smoke,
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["local_sota_runtime_ready"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["models_found_in_cache"] == [
        {
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "role": "flagship_moe_runtime_probe",
            "model_path": QWEN_PATH,
        },
        {
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "role": "flagship_dense_runtime_probe",
            "model_path": GEMMA_PATH,
        },
    ]
    assert artifact["models_missing_from_cache"] == [
        "unsloth/gemma-4-26B-A4B-it-GGUF"
    ]
    assert artifact["cached_sota_pair_preview"][0]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert artifact["smoke_inference_results"][0]["truly_live"] is True
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"] == "live_sota_runtime_ready"


def test_exp1442_missing_cache_blocks_without_smoke_probe(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-007-002: no local mandated cache means no fake inference."""

    def fail_smoke(*_: Any, **__: Any) -> dict[str, Any]:
        raise AssertionError("smoke probe must not run without local models")

    artifact = build_live_runtime_preflight_artifact(
        project_root=tmp_path,
        run_date="20260506",
        cache_resolver=lambda *_args, **_kwargs: None,
        cached_pair_fn=lambda **_kwargs: None,
        gpu_probe_fn=lambda: {"gpu_count": 0, "cuda_available": False},
        smoke_probe_fn=fail_smoke,
    )

    assert artifact["local_sota_runtime_ready"] is False
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["models_found_in_cache"] == []
    assert artifact["models_missing_from_cache"] == [spec["hf_id"] for spec in MANDATED_MODEL_SPECS]
    assert artifact["smoke_inference_results"] == []
    assert "no_mandated_sota_models_found_in_local_cache" in artifact["blockers"]
    assert artifact["honest_verdict"] == "blocked_no_live_sota_runtime"


def test_exp1442_failed_live_probe_keeps_gate_closed(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-007: a cached model is insufficient without usable live output."""

    def fake_smoke(model: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "hf_id": model["hf_id"],
            "role": model["role"],
            "model_path": model["model_path"],
            "command": ["python", "-m", "probe"],
            "runtime_mode": "llama_cpp_subprocess_gpu",
            "returncode": 1,
            "stdout_summary": "",
            "stderr_summary": "RuntimeError: load failed",
            "elapsed_s": 0.5,
            "truly_live": False,
            "usable_response": False,
            "response_text_preview": "",
            "blocker": "RuntimeError: load failed",
        }

    artifact = build_live_runtime_preflight_artifact(
        project_root=tmp_path,
        run_date="20260506",
        cache_resolver=lambda hf_id, **_: QWEN_PATH
        if hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF"
        else None,
        cached_pair_fn=lambda **_kwargs: None,
        gpu_probe_fn=lambda: {"gpu_count": 1, "cuda_available": True},
        smoke_probe_fn=fake_smoke,
    )

    assert artifact["models_found_in_cache"][0]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert artifact["local_sota_runtime_ready"] is False
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["smoke_inference_results"][0]["blocker"] == "RuntimeError: load failed"
    assert "no_mandated_sota_model_completed_live_inference" in artifact["blockers"]


def test_exp1442_cache_resolver_exception_is_recorded(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-007: cache probe errors are blockers, not headline readiness."""

    def bad_resolver(hf_id: str, **_: Any) -> str | None:
        if hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF":
            raise RuntimeError("cache unreadable")
        return None

    artifact = build_live_runtime_preflight_artifact(
        project_root=tmp_path,
        run_date="20260506",
        cache_resolver=bad_resolver,
        cached_pair_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("pair failed")),
        gpu_probe_fn=lambda: {"gpu_count": 0, "cuda_available": False},
        smoke_probe_fn=lambda *_args, **_kwargs: pytest.fail("smoke probe must not run"),
    )

    assert artifact["cache_probe_errors"] == {
        "unsloth/Qwen3.6-35B-A3B-GGUF": "RuntimeError: cache unreadable"
    }
    assert artifact["cached_sota_pair_error"] == "RuntimeError: pair failed"
    assert "cache_probe_errors_present" in artifact["blockers"]


def test_exp1442_run_experiment_writes_in_progress_then_complete(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-007: the runner creates the durable terminal JSON artifact."""
    output_path = tmp_path / "results" / "experiment_1442_live_sota_repair_runtime_preflight.json"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        output_path=output_path,
        cache_resolver=lambda *_args, **_kwargs: None,
        cached_pair_fn=lambda **_kwargs: None,
        gpu_probe_fn=lambda: {"gpu_count": 0, "cuda_available": False},
        smoke_probe_fn=lambda *_args, **_kwargs: pytest.fail("smoke probe must not run"),
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["status"] == "complete"
    assert written["honest_verdict"] == "blocked_no_live_sota_runtime"


def test_exp1442_gpu_probe_records_cuda_and_nvidia_smi() -> None:
    """REQ-INFER-SOTA-007: GPU availability is captured without requiring hardware."""

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "--query-gpu=index,name,memory.total,memory.free,memory.used" in cmd
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="0, RTX 3090, 24576, 12288, 12288\nbad-line\n",
            stderr="",
        )

    probe = probe_gpu_state(
        cuda_available_fn=lambda: True,
        gpu_count_fn=lambda: 1,
        command_runner=fake_run,
    )

    assert probe["cuda_available"] is True
    assert probe["gpu_count"] == 1
    assert probe["nvidia_smi_available"] is True
    assert probe["gpus"] == [
        {
            "index": 0,
            "name": "RTX 3090",
            "memory_total_mb": 24576.0,
            "memory_free_mb": 12288.0,
            "memory_used_mb": 12288.0,
        }
    ]


def test_exp1442_gpu_probe_degrades_when_tooling_missing() -> None:
    """REQ-INFER-SOTA-007: missing GPU tooling is an explicit probe field."""

    def fake_run(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("nvidia-smi")

    probe = probe_gpu_state(
        cuda_available_fn=lambda: False,
        gpu_count_fn=lambda: 0,
        command_runner=fake_run,
    )

    assert probe["cuda_available"] is False
    assert probe["gpu_count"] == 0
    assert probe["nvidia_smi_available"] is False
    assert probe["nvidia_smi_error"] == "FileNotFoundError: nvidia-smi"
    assert probe["gpus"] == []


def test_exp1442_gpu_probe_covers_default_helpers_and_bad_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFER-SOTA-007: default GPU helper imports and parse failures are covered."""
    fake_template = types.ModuleType("scripts.experiment_template")
    fake_template._cuda_is_available = lambda: True
    fake_template._detect_gpu_count_rocm_aware = lambda: 2
    monkeypatch.setitem(sys.modules, "scripts.experiment_template", fake_template)

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="x, RTX, bad, 1, 2\n0, RTX 3090, 24576, 1000, 23576\n",
            stderr="",
        )

    probe = probe_gpu_state(command_runner=fake_run)

    assert probe["cuda_available"] is True
    assert probe["gpu_count"] == 2
    assert probe["gpus"][0]["index"] == 0


def test_exp1442_gpu_probe_records_nvidia_smi_nonzero() -> None:
    """REQ-INFER-SOTA-007: nvidia-smi failures retain stderr summaries."""

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 9, stdout="", stderr="driver failed")

    probe = probe_gpu_state(
        cuda_available_fn=lambda: True,
        gpu_count_fn=lambda: 1,
        command_runner=fake_run,
    )

    assert probe["nvidia_smi_available"] is False
    assert probe["nvidia_smi_error"] == "driver failed"


class _FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def __call__(self) -> float:
        return self._values.pop(0)


class _FakeLlama:
    closed = 0

    def __init__(self, **kwargs: Any) -> None:
        assert kwargs["model_path"] == QWEN_PATH
        assert kwargs["n_gpu_layers"] == -1
        assert kwargs["main_gpu"] == 0
        assert kwargs["n_ctx"] == 512
        assert kwargs["verbose"] is False

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert "Return compact JSON only" in prompt
        assert kwargs["max_tokens"] == 8
        assert kwargs["temperature"] == 0.0
        assert kwargs["echo"] is False
        return {"choices": [{"text": "{\"repair_action\":\"STEP_REWRITE\"}"}]}

    def close(self) -> None:
        type(self).closed += 1


def test_exp1442_probe_one_uses_llama_cpp_and_marks_live() -> None:
    """SCENARIO-INFER-SOTA-007-001: one isolated llama.cpp call is true live inference."""
    _FakeLlama.closed = 0

    result = run_live_probe_one(
        hf_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        role="flagship_moe_runtime_probe",
        model_path=QWEN_PATH,
        gpu=0,
        llama_importer=lambda: (True, _FakeLlama, None),
        monotonic=_FakeClock([10.0, 12.5]),
        max_tokens=8,
    )

    assert result["runtime_mode"] == "llama_cpp_direct_gpu"
    assert result["load_success"] is True
    assert result["truly_live"] is True
    assert result["usable_response"] is True
    assert result["elapsed_s"] == 2.5
    assert result["response_text"] == "{\"repair_action\":\"STEP_REWRITE\"}"
    assert _FakeLlama.closed == 1


def test_exp1442_probe_one_import_failure_is_not_live() -> None:
    """SCENARIO-INFER-SOTA-007-002: llama.cpp import failure cannot count as live."""
    result = run_live_probe_one(
        hf_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        role="flagship_moe_runtime_probe",
        model_path=QWEN_PATH,
        gpu=0,
        llama_importer=lambda: (False, None, "ImportError: no llama_cpp"),
        monotonic=_FakeClock([0.0, 0.1]),
    )

    assert result["load_success"] is False
    assert result["truly_live"] is False
    assert result["usable_response"] is False
    assert result["blocker"] == "ImportError: no llama_cpp"


def test_exp1442_probe_one_generation_exception_is_not_live() -> None:
    """SCENARIO-INFER-SOTA-007-002: generation exceptions are explicit blockers."""

    class ExplodingLlama:
        def __init__(self, **_: Any) -> None:
            pass

        def __call__(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("generation exploded")

    result = run_live_probe_one(
        hf_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        role="flagship_moe_runtime_probe",
        model_path=QWEN_PATH,
        gpu=-1,
        llama_importer=lambda: (True, ExplodingLlama, None),
        monotonic=_FakeClock([0.0, 0.5]),
    )

    assert result["runtime_mode"] == "llama_cpp_direct_gpu"
    assert result["load_success"] is True
    assert result["truly_live"] is False
    assert result["blocker"] == "RuntimeError: generation exploded"


def test_exp1442_subprocess_probe_parses_json_stdout() -> None:
    """REQ-INFER-SOTA-007: subprocess stdout/stderr summaries preserve live evidence."""

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[:3] == ["python3", "-m", "carnot.reporting.live_sota_repair_runtime_preflight"]
        assert kwargs["timeout"] == 30
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='loading...\n{"truly_live": true, "usable_response": true, "response_text": "ok"}\n',
            stderr="warning only",
        )

    model = {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_runtime_probe",
        "model_path": QWEN_PATH,
        "gpu": 0,
    }
    result = run_live_probe_subprocess(
        model,
        command_runner=fake_run,
        monotonic=_FakeClock([1.0, 3.0]),
        python_executable="python3",
        timeout_s=30,
    )

    assert result["command"] == _build_probe_command(model, python_executable="python3")
    assert result["returncode"] == 0
    assert result["stdout_summary"].endswith('"response_text": "ok"}')
    assert result["stderr_summary"] == "warning only"
    assert result["truly_live"] is True
    assert result["usable_response"] is True
    assert result["response_text_preview"] == "ok"
    assert result["elapsed_s"] == 2.0


def test_exp1442_subprocess_failure_is_blocker() -> None:
    """SCENARIO-INFER-SOTA-007-002: nonzero subprocesses are terminal smoke blockers."""

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 2, stdout="not json", stderr="load failed")

    result = run_live_probe_subprocess(
        {
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "role": "flagship_moe_runtime_probe",
            "model_path": QWEN_PATH,
            "gpu": 0,
        },
        command_runner=fake_run,
        monotonic=_FakeClock([0.0, 0.25]),
        python_executable="python3",
    )

    assert result["returncode"] == 2
    assert result["truly_live"] is False
    assert result["usable_response"] is False
    assert result["blocker"] == "probe_subprocess_returncode_2"


def test_exp1442_subprocess_success_without_usable_response_is_blocker() -> None:
    """SCENARIO-INFER-SOTA-007-002: returncode 0 still needs usable live output."""

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout='{"truly_live": false, "usable_response": false, "blocker": "empty"}',
            stderr="",
        )

    result = run_live_probe_subprocess(
        {
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "role": "flagship_moe_runtime_probe",
            "model_path": QWEN_PATH,
            "gpu": 0,
        },
        command_runner=fake_run,
        monotonic=_FakeClock([0.0, 0.25]),
        python_executable="python3",
    )

    assert result["returncode"] == 0
    assert result["blocker"] == "empty"
    assert result["usable_response"] is False


def test_exp1442_output_helpers_cover_edge_cases() -> None:
    """REQ-INFER-SOTA-007: output parsing remains deterministic for logging."""
    assert _summarize_stream("a" * 10, limit=4) == "aaaa..."
    assert _summarize_stream("  \n\t", limit=4) == ""
    assert _extract_json_from_stdout('\nnoise\n{"ok": true}\n') == {"ok": True}
    assert _extract_json_from_stdout('{"ok": true}\n\nnot json') == {"ok": True}
    assert _extract_json_from_stdout("not json") is None
    assert _completion_text({"choices": [{"text": "abc"}]}) == "abc"
    assert _completion_text({"choices": [{"message": {"content": "xyz"}}]}) == "xyz"
    assert _completion_text({"choices": []}) == ""
    assert _completion_text(object()) == ""
    assert _completion_text({"choices": [123]}) == ""
    assert _completion_text({"choices": [{"message": {}}]}) == ""
    assert _completion_text("plain") == "plain"


def test_exp1442_default_helpers_can_be_patched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-INFER-SOTA-007: default helper wrappers stay wired to canonical modules."""
    fake_template = types.ModuleType("scripts.experiment_template")
    fake_template._get_repo_root = lambda: str(tmp_path)
    fake_sota_models = types.ModuleType("carnot.inference.sota_models")
    fake_sota_models.resolve_cached_gguf = lambda hf_id, preferred_quant: "/x"
    fake_sota_models.cached_sota_pair = lambda **kwargs: [{"kwargs": kwargs}]
    monkeypatch.setitem(sys.modules, "scripts.experiment_template", fake_template)
    monkeypatch.setitem(sys.modules, "carnot.inference.sota_models", fake_sota_models)

    assert mod._utc_run_date().isdigit()
    assert mod._repo_root() == tmp_path
    assert mod._default_cache_resolver("model", preferred_quant="Q4_K_M") == "/x"
    assert mod._default_cached_pair(gpu_indices=(0, 1)) == [
        {"kwargs": {"gpu_indices": (0, 1)}}
    ]


def test_exp1442_default_llama_importer_reports_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFER-SOTA-007: the importer wrapper reports exact llama.cpp state."""

    class DummyLlama:
        pass

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=DummyLlama))
    assert mod._default_llama_importer() == (True, DummyLlama, None)

    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    original_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "llama_cpp":
            raise ImportError("no module named llama_cpp")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert mod._default_llama_importer() == (False, None, "ImportError: no module named llama_cpp")


def test_exp1442_cli_probe_one_and_full_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-007: CLI routing covers probe and artifact modes."""
    monkeypatch.setattr(
        mod,
        "run_live_probe_one",
        lambda **_kwargs: {"truly_live": True, "usable_response": True},
    )
    assert mod.main(["--probe-one", "--hf-id", "x", "--role", "r", "--model-path", "p"]) == 0
    assert json.loads(capsys.readouterr().out) == {"truly_live": True, "usable_response": True}

    calls: list[dict[str, Any]] = []

    def fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete"}

    output = tmp_path / "artifact.json"
    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)

    assert mod.main(["--run-date", "20260506", "--output", str(output)]) == 0
    assert calls == [{"run_date": "20260506", "output_path": output}]
