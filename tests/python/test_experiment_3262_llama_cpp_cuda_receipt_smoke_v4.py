"""Tests for Exp 3262 llama.cpp CUDA receipt smoke v4.

Spec refs: REQ-REPORT-3262, SCENARIO-REPORT-3262.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import llama_cpp_cuda_receipt_smoke_3262 as mod


SELECTED_PYTHON = "/repo/.venv/bin/python"
MODEL_ID = "unsloth/Qwen3.5-0.8B-GGUF"

REQUIRED_FIELDS = {
    "llama_cpp_cuda_receipt_smoke_v4_ready",
    "llama_cpp_cuda_receipt_ready",
    "gpu_layers_offloaded",
    "gpu_mem_used_during_call_mib",
    "tokens_generated",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_exp3261(root: Path, *, passed: bool = True) -> Path:
    path = root / mod.EXP3261_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact": "experiment_3261_cuda_recovery_confirmation_smoke_v1",
                "cuda_python_smoke_passed": passed,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_model(cache_root: Path, *, name: str = "Qwen3.5-0.8B-Q4_K_M.gguf") -> Path:
    path = (
        cache_root
        / "models--unsloth--Qwen3.5-0.8B-GGUF"
        / "snapshots"
        / "rev1"
        / name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"gguf-small")
    return path


def _command(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_summary": stdout,
        "stderr_summary": stderr,
    }


def _backend_stdout(*, import_ok: bool = True, supports_gpu: bool = True) -> str:
    return (
        json.dumps(
            {
                "llama_cpp_import_ok": import_ok,
                "llama_cpp_supports_gpu_offload": supports_gpu if import_ok else False,
                "llama_cpp_version": "0.3.23" if import_ok else None,
                "llama_cpp_origin": "/repo/.venv/lib/python/site-packages/llama_cpp/__init__.py",
                "llama_cpp_system_info": "CUDA : ARCHS = 860 | ggml-cuda enabled",
                "backend_error": "" if import_ok else "ModuleNotFoundError",
            },
            sort_keys=True,
        )
        + "\n"
    )


def _worker_stdout(
    *,
    output: str = "CUDA receipt ready.",
    tokens: int = 3,
    layers: int = 17,
    baseline: int = 100,
    during: int = 720,
) -> str:
    payload = {
        "ok": bool(output),
        "output_text": output,
        "tokens_generated": tokens,
        "gpu_layers_offloaded": layers,
        "n_gpu_layers_requested": 24,
        "gpu_mem_baseline_mib": baseline,
        "gpu_mem_used_during_call_mib": during,
        "gpu_mem_delta_during_call_mib": max(0, during - baseline),
        "gpu_memory": {
            "baseline": [{"index": 0, "memory_used_mib": baseline}],
            "during_generate": [{"index": 0, "memory_used_mib": during}],
            "after_generate": [{"index": 0, "memory_used_mib": during}],
        },
        "usage": {"prompt_tokens": 9, "completion_tokens": tokens, "total_tokens": 9 + tokens},
    }
    return json.dumps(payload, sort_keys=True) + "\n"


def _runner(
    *,
    import_ok: bool = True,
    supports_gpu: bool = True,
    worker_returncode: int = 0,
    worker_stdout: str | None = None,
) -> tuple[mod.CommandRunner, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def run(command: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append({"command": command, "kwargs": kwargs})
        joined = "\n".join(command)
        if "exp3262_llama_cpp_backend_probe" in joined:
            return _command(
                command,
                returncode=0 if import_ok else 1,
                stdout=_backend_stdout(import_ok=import_ok, supports_gpu=supports_gpu),
                stderr="" if import_ok else "no module named llama_cpp\n",
            )
        if "--exp3262-cuda-receipt-worker" in command:
            return _command(
                command,
                returncode=worker_returncode,
                stdout=worker_stdout if worker_stdout is not None else _worker_stdout(),
                stderr="llama_model_load: offloaded 17/17 layers to GPU\n"
                if worker_returncode == 0
                else "worker failed\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    return run, calls


def test_req_report_3262_spec_anchor_exists() -> None:
    """REQ-REPORT-3262: OpenSpec declares the CUDA receipt smoke before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3262" in spec
    assert "SCENARIO-REPORT-3262" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "gpu_mem_used_during_call_mib" in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3262_gated_skip_when_exp3261_not_passed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3262: Exp 3261 blocks downstream llama.cpp receipt claims."""

    _write_exp3261(tmp_path, passed=False)
    runner, calls = _runner()

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([1.0, 1.25]).__next__,
    )

    assert json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")) == artifact
    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3262"
    assert artifact["llama_cpp_cuda_receipt_smoke_v4_ready"] is True
    assert artifact["llama_cpp_cuda_receipt_ready"] is False
    assert artifact["blocked_reason"] == "gated_exp3261_cuda_python_smoke_not_passed"
    assert artifact["gpu_layers_offloaded"] == 0
    assert artifact["gpu_mem_used_during_call_mib"] == 0
    assert artifact["tokens_generated"] == 0
    assert artifact["model_specs"] == {}
    assert artifact["honest_verdict"].startswith("complete:")
    assert "llama_cpp_cuda_receipt_ready=false" in artifact["honest_verdict"]
    assert calls == []


def test_req_report_3262_blocks_when_llama_cpp_cuda_or_small_gguf_missing(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3262: missing CUDA-enabled llama.cpp/cache is an honest block."""

    _write_exp3261(tmp_path, passed=True)
    runner, calls = _runner(supports_gpu=False)

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[tmp_path / "empty-cache"],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([2.0, 2.75]).__next__,
    )

    assert artifact["llama_cpp_cuda_receipt_smoke_v4_ready"] is True
    assert artifact["llama_cpp_cuda_receipt_ready"] is False
    assert artifact["blocked_reason"] == "blocked_llama_cpp_cuda_missing"
    assert artifact["llama_cpp_backend"]["llama_cpp_supports_gpu_offload"] is False
    assert artifact["small_gguf_cache"]["selected_model_path"] is None
    assert artifact["gpu_layers_offloaded"] == 0
    assert artifact["tokens_generated"] == 0
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(calls) == 1


def test_scenario_report_3262_successful_cuda_generation_writes_receipt(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3262: CUDA offload, GPU memory growth, and tokens open gate."""

    _write_exp3261(tmp_path, passed=True)
    cache_root = tmp_path / "hf-cache"
    model_path = _write_model(cache_root)
    runner, calls = _runner()

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        env={"EXTRA_ENV_FOR_TEST": "1"},
        command_runner=runner,
        monotonic=iter([10.0, 13.5]).__next__,
        random_seed=3262,
        n_gpu_layers=24,
        max_tokens=8,
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["llama_cpp_cuda_receipt_smoke_v4_ready"] is True
    assert artifact["llama_cpp_cuda_receipt_ready"] is True
    assert artifact["blocked_reason"] == ""
    assert artifact["selected_model_path"] == str(model_path)
    assert artifact["model_specs"]["model_id"] == MODEL_ID
    assert artifact["model_specs"]["size_bytes"] == model_path.stat().st_size
    assert artifact["model_specs"]["n_gpu_layers_requested"] == 24
    assert artifact["gpu_layers_offloaded"] == 17
    assert artifact["gpu_mem_baseline_mib"] == 100
    assert artifact["gpu_mem_used_during_call_mib"] == 720
    assert artifact["gpu_mem_delta_during_call_mib"] == 620
    assert artifact["tokens_generated"] == 3
    assert artifact["generation_output_nonempty"] is True
    assert artifact["random_seed"] == 3262
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "gpu_layers_offloaded=17" in artifact["honest_verdict"]
    assert len(calls) == 2
    assert "exp3262_llama_cpp_backend_probe" in "\n".join(calls[0]["command"])
    assert "--exp3262-cuda-receipt-worker" in calls[1]["command"]
    assert str(model_path) in calls[1]["command"]
    assert calls[1]["kwargs"]["env"]["PYTHONHASHSEED"] == "3262"


def test_req_report_3262_worker_failure_keeps_receipt_gate_closed(tmp_path: Path) -> None:
    """REQ-REPORT-3262: runtime failures do not masquerade as CUDA receipts."""

    _write_exp3261(tmp_path, passed=True)
    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root)
    runner, _calls = _runner(worker_returncode=1, worker_stdout=json.dumps({"ok": False}) + "\n")

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([4.0, 6.0]).__next__,
    )

    assert artifact["llama_cpp_cuda_receipt_smoke_v4_ready"] is True
    assert artifact["llama_cpp_cuda_receipt_ready"] is False
    assert artifact["blocked_reason"] == "llama_cpp_generation_failed"
    assert artifact["worker_attempt"]["returncode"] == 1
    assert artifact["gpu_layers_offloaded"] == 0
    assert artifact["gpu_mem_used_during_call_mib"] == 0
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_3262_incomplete_worker_payload_keeps_receipt_gate_closed(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3262: empty text or no tokens is not a completed CUDA receipt."""

    _write_exp3261(tmp_path, passed=True)
    cache_root = tmp_path / "hf-cache"
    _write_model(cache_root)
    runner, _calls = _runner(worker_stdout=_worker_stdout(output="", tokens=0, layers=0, during=100))

    artifact = mod.build_artifact(
        project_root=tmp_path,
        cache_roots=[cache_root],
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([7.0, 8.0]).__next__,
    )

    assert artifact["llama_cpp_cuda_receipt_ready"] is False
    assert artifact["blocked_reason"] == "llama_cpp_cuda_receipt_incomplete"
    assert artifact["generation_output_nonempty"] is False
    assert artifact["tokens_generated"] == 0


def test_helpers_cover_parsing_selection_and_command_execution(tmp_path: Path) -> None:
    """REQ-REPORT-3262: helper behavior is deterministic and JSON-safe."""

    assert mod._selected_python(tmp_path) == sys.executable
    candidate = tmp_path / ".venv" / "bin" / "python"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# placeholder\n", encoding="utf-8")
    assert mod._selected_python(tmp_path) == str(candidate)

    assert mod._json_from_last_line({"stdout": "noise\n{\"ok\": true}\n", "stderr": ""}) == {
        "ok": True
    }
    assert mod._json_from_last_line({"stdout": "noise\n", "stderr": "bad"}) == {"error": "bad"}
    assert mod._summarize("abcdef", limit=3) == "def"
    assert mod._safe_int("9") == 9
    assert mod._safe_int("bad") is None
    assert mod._model_id_from_path(Path("/cache/models--owner--name/snapshots/rev/model.gguf")) == (
        "owner/name"
    )
    assert mod._model_id_from_path(Path("/plain/model.gguf")) == "local/model"
    assert mod._memory_by_index("not rows") == {}
    assert mod._memory_by_index([{"index": "0", "memory_used_mib": "42"}, object()]) == {
        0: 42
    }
    assert mod._max_memory_used({"during_generate": [{"index": 0, "memory_used_mib": 33}]}) == 33
    assert mod._parse_offloaded_layers("llama_model_load: offloaded 17/17 layers to GPU") == 17
    assert mod._parse_offloaded_layers("llama_model_load: offloading 9 repeating layers to GPU") == 9
    assert mod._parse_offloaded_layers("no gpu layers") == 0
    assert mod._read_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._read_json(bad_json) == {}
    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    assert mod._read_json(scalar_json) == {}
    assert mod._default_cache_roots(tmp_path, {"HUGGINGFACE_HUB_CACHE": "/custom/hub"}) == [
        Path("/custom/hub"),
        tmp_path / "models",
    ]
    assert mod._default_cache_roots(tmp_path, {"HF_HOME": "/custom/hf"}) == [
        Path("/custom/hf/hub"),
        tmp_path / "models",
    ]

    cache_root = tmp_path / "hf-cache"
    small = _write_model(cache_root, name="small-Q4_K_M.gguf")
    large = _write_model(cache_root, name="large-Q4_K_M.gguf")
    large.write_bytes(b"x" * 100)
    mmproj = _write_model(cache_root, name="mmproj-F16.gguf")
    empty = _write_model(cache_root, name="empty.gguf")
    empty.write_bytes(b"")
    selected = mod._select_small_cached_gguf([cache_root])
    assert selected and selected["path"] == str(small)
    assert "mmproj" not in selected["filename"]
    assert mod._select_small_cached_gguf([tmp_path / "missing"]) is None
    parsed_worker, _calls = _runner(worker_stdout=_worker_stdout(layers=0))
    worker = mod._run_receipt_worker(
        selected_python=SELECTED_PYTHON,
        model_path=str(small),
        n_gpu_layers=24,
        max_tokens=4,
        random_seed=3262,
        env={},
        command_runner=parsed_worker,
    )
    assert worker["payload"]["gpu_layers_offloaded"] == 17
    fallback_metrics = mod._worker_metrics(
        {
            "payload": {
                "output_text": "fallback",
                "tokens_generated": 1,
                "gpu_mem_baseline_mib": 10,
                "gpu_mem_used_during_call_mib": 20,
            },
            "stderr_summary": "",
        },
        n_gpu_layers=3,
    )
    assert fallback_metrics["gpu_layers_offloaded"] == 3

    result = mod._run_command([sys.executable, "-c", "print('ok')"], timeout_s=10)
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "ok"


def test_main_prints_artifact(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-REPORT-3262: CLI entrypoint emits the written artifact payload."""

    monkeypatch.setattr(
        mod,
        "run_experiment",
        lambda **_kwargs: {"artifact": mod.ARTIFACT, "llama_cpp_cuda_receipt_ready": False},
    )

    assert mod.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["artifact"] == mod.ARTIFACT
