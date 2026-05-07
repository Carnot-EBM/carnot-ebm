"""Tests for Exp 1463 local SOTA GGUF runtime repair.

Spec: REQ-INFER-SOTA-008,
      SCENARIO-INFER-SOTA-008-001,
      SCENARIO-INFER-SOTA-008-002
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import local_sota_gguf_runtime_repair as mod
from carnot.reporting.local_sota_gguf_runtime_repair import (
    REQUIRED_ARTIFACT_FIELDS,
    _candidate_cuda_library_dirs,
    _extract_ldconfig_libs,
    _has_libcudart_blocker,
    _persistent_blockers_from,
    _prepend_existing_library_dirs,
    _run_command,
    build_runtime_repair_artifact,
    discover_cuda_runtime_state,
    attempt_missing_cache_resolution,
    run_experiment,
)


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
MIDDLE = "unsloth/gemma-4-26B-A4B-it-GGUF"
QWEN_PATH = "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def _prior_libcudart_blocked() -> dict[str, Any]:
    return {
        "status": "complete",
        "gpu_probe": {"gpu_count": 2, "cuda_available": True},
        "models_found_in_cache": [{"hf_id": QWEN, "model_path": QWEN_PATH}],
        "models_missing_from_cache": [MIDDLE],
        "smoke_inference_results": [
            {
                "hf_id": QWEN,
                "truly_live": False,
                "usable_response": False,
                "blocker": "RuntimeError: Failed to load shared library 'libllama.so': libcudart.so.12: cannot open shared object file",
            }
        ],
        "blockers": ["no_mandated_sota_model_completed_live_inference"],
        "local_sota_runtime_ready": False,
        "live_sota_model_inference_used": False,
        "honest_verdict": "blocked_no_live_sota_runtime",
    }


def _repaired_live() -> dict[str, Any]:
    return {
        "status": "complete",
        "gpu_probe": {"gpu_count": 2, "cuda_available": True},
        "models_found_in_cache": [{"hf_id": QWEN, "model_path": QWEN_PATH}],
        "models_missing_from_cache": [MIDDLE],
        "smoke_inference_results": [
            {
                "hf_id": QWEN,
                "model_path": QWEN_PATH,
                "runtime_mode": "llama_cpp_subprocess_gpu",
                "truly_live": True,
                "usable_response": True,
                "response_text_preview": "{\"repair_action\":\"reject\"}",
                "blocker": None,
            }
        ],
        "blockers": [],
        "local_sota_runtime_ready": True,
        "live_sota_model_inference_used": True,
        "honest_verdict": "live_sota_runtime_ready",
    }


def test_exp1463_runtime_path_repair_opens_live_gate(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-008 / SCENARIO-INFER-SOTA-008-001: repaired LD path can go live."""
    observed_env: list[dict[str, str]] = []

    def repaired_probe(*, env: dict[str, str] | None = None, **_: Any) -> dict[str, Any]:
        observed_env.append(dict(env or {}))
        return _repaired_live()

    artifact = build_runtime_repair_artifact(
        project_root=tmp_path,
        run_date="20260507",
        reproduce_probe_fn=lambda **_: _prior_libcudart_blocked(),
        repaired_probe_fn=repaired_probe,
        cuda_discovery_fn=lambda **_: {
            "ldconfig_libs": {"libcuda.so.1": "/usr/lib/libcuda.so.1"},
            "nvidia_smi": {"returncode": 0, "stdout_summary": "RTX 3090"},
            "package_metadata": {"nvidia-cuda-runtime-cu12": "12.6.77"},
            "environment": {"LD_LIBRARY_PATH": ""},
            "candidate_library_dirs": ["/venv/nvidia/cuda_runtime/lib", "/venv/nvidia/cublas/lib"],
            "existing_library_dirs": [
                "/venv/nvidia/cuda_runtime/lib",
                "/venv/nvidia/cuda_runtime/lib",
                "/venv/nvidia/cublas/lib",
            ],
            "libllama_ldd_before": "libcudart.so.12 => not found",
            "libllama_ldd_after": "libcudart.so.12 => /venv/nvidia/cuda_runtime/lib/libcudart.so.12",
        },
        missing_cache_resolution_fn=lambda **_: {
            "attempted": True,
            "hf_id": MIDDLE,
            "status": "not_required_after_live_runtime_success",
            "blocker": None,
        },
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["libcudart_resolution_attempted"]["attempted"] is True
    assert artifact["libcudart_resolution_attempted"]["repair_applied"] is True
    assert observed_env[0]["LD_LIBRARY_PATH"].startswith("/venv/nvidia/cuda_runtime/lib")
    assert artifact["smoke_inference_results"][0]["truly_live"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["local_sota_runtime_ready"] is True
    assert artifact["persistent_blockers"] == []
    assert artifact["honest_verdict"] == "local_sota_runtime_ready"


def test_exp1463_persistent_runtime_blockers_stay_terminal(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-008 / SCENARIO-INFER-SOTA-008-002: failed repair remains honest."""
    failed_after_repair = _prior_libcudart_blocked()
    failed_after_repair["smoke_inference_results"][0]["blocker"] = "OSError: no libcublas.so.12"

    artifact = build_runtime_repair_artifact(
        project_root=tmp_path,
        run_date="20260507",
        reproduce_probe_fn=lambda **_: _prior_libcudart_blocked(),
        repaired_probe_fn=lambda **_: failed_after_repair,
        cuda_discovery_fn=lambda **_: {
            "ldconfig_libs": {},
            "nvidia_smi": {"returncode": 0, "stdout_summary": "RTX 3090"},
            "package_metadata": {},
            "environment": {},
            "candidate_library_dirs": [],
            "existing_library_dirs": [],
            "libllama_ldd_before": "libcudart.so.12 => not found",
            "libllama_ldd_after": "libcudart.so.12 => not found",
        },
        missing_cache_resolution_fn=lambda **_: {
            "attempted": True,
            "hf_id": MIDDLE,
            "status": "blocked_no_download",
            "blocker": "LocalEntryNotFoundError: not in local cache",
        },
    )

    assert artifact["local_sota_runtime_ready"] is False
    assert artifact["live_sota_model_inference_used"] is False
    assert "no_mandated_sota_model_completed_live_inference" in artifact["persistent_blockers"]
    assert "OSError: no libcublas.so.12" in artifact["persistent_blockers"]
    assert "LocalEntryNotFoundError: not in local cache" in artifact["persistent_blockers"]
    assert artifact["honest_verdict"] == "blocked_persistent_local_sota_runtime"


def test_exp1463_run_experiment_writes_in_progress_then_complete(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-008: runner persists the bootstrap artifact before repair work."""
    writes: list[dict[str, Any]] = []

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        output_path=tmp_path / "results" / "experiment_1463_local_sota_gguf_runtime_repair.json",
        reproduce_probe_fn=lambda **_: _prior_libcudart_blocked(),
        repaired_probe_fn=lambda **_: _repaired_live(),
        cuda_discovery_fn=lambda **_: {"existing_library_dirs": [], "candidate_library_dirs": []},
        missing_cache_resolution_fn=lambda **_: {"attempted": False, "status": "skipped"},
        write_json_fn=lambda _path, payload: writes.append(dict(payload)),
    )

    assert writes[0]["status"] == "in_progress"
    assert writes[1] == artifact
    assert writes[1]["status"] == "complete"


def test_exp1463_helpers_keep_path_and_blocker_logic_deterministic(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-008: helper decisions remain deterministic for artifact logging."""
    cuda_dir = tmp_path / "cuda"
    cublas_dir = tmp_path / "cublas"
    cuda_dir.mkdir()

    repaired = _prepend_existing_library_dirs(
        current="/already",
        candidate_dirs=[str(cuda_dir), str(cublas_dir), str(cuda_dir)],
    )

    assert repaired == f"{cuda_dir}:/already"
    assert _has_libcudart_blocker(_prior_libcudart_blocked()) is True
    assert _persistent_blockers_from(
        {
            "blockers": ["top", "top"],
            "smoke_inference_results": [object(), {"blocker": "leaf"}, {"blocker": "leaf"}],
        },
        {"blocker": "cache"},
    ) == ["top", "leaf", "cache"]
    assert _extract_ldconfig_libs("not cuda\nlibcuda.so.1 no arrow\n") == {}


def test_exp1463_runner_persists_json_with_real_writer(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-008-001: complete artifact is JSON-stable on disk."""
    output_path = tmp_path / "results" / "experiment_1463_local_sota_gguf_runtime_repair.json"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        output_path=output_path,
        reproduce_probe_fn=lambda **_: _prior_libcudart_blocked(),
        repaired_probe_fn=lambda **_: _repaired_live(),
        cuda_discovery_fn=lambda **_: {"existing_library_dirs": [], "candidate_library_dirs": []},
        missing_cache_resolution_fn=lambda **_: {"attempted": False, "status": "skipped"},
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_exp1463_cuda_discovery_collects_loader_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-008: CUDA discovery records ldconfig, packages, env, and ldd."""
    site = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    cuda_dir = site / "nvidia" / "cuda_runtime" / "lib"
    cublas_dir = site / "nvidia" / "cublas" / "lib"
    llama_dir = site / "llama_cpp" / "lib"
    python_path = tmp_path / ".venv" / "bin" / "python"
    cuda_dir.mkdir(parents=True)
    cublas_dir.mkdir(parents=True)
    llama_dir.mkdir(parents=True)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    (llama_dir / "libllama.so").write_text("", encoding="utf-8")

    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((cmd, kwargs))
        if cmd[:2] == ["ldconfig", "-p"]:
            stdout = "\tlibcuda.so.1 (libc6,x86-64) => /usr/lib/libcuda.so.1\n"
            stdout += f"\tlibcudart.so.12 (libc6,x86-64) => {cuda_dir / 'libcudart.so.12'}\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        if cmd == ["nvidia-smi"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="RTX 3090", stderr="")
        if "-m" in cmd and "pip" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="Name: llama_cpp_python", stderr="")
        if cmd[0] == "ldd":
            stdout = (
                "libcudart.so.12 => not found"
                if kwargs.get("env") is None
                else f"libcudart.so.12 => {cuda_dir / 'libcudart.so.12'}"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    discovery = discover_cuda_runtime_state(
        project_root=tmp_path,
        env={"LD_LIBRARY_PATH": "/existing", "CUDA_HOME": "/cuda"},
    )

    assert _candidate_cuda_library_dirs(tmp_path) == [str(cuda_dir), str(cublas_dir)]
    assert discovery["existing_library_dirs"] == [str(cuda_dir), str(cublas_dir)]
    assert discovery["ldconfig_libs"]["libcuda.so.1"] == "/usr/lib/libcuda.so.1"
    assert discovery["environment"]["CUDA_HOME"] == "/cuda"
    assert discovery["libllama_ldd_before"] == "libcudart.so.12 => not found"
    assert str(cuda_dir) in discovery["libllama_ldd_after"]
    assert calls[-1][1]["env"]["LD_LIBRARY_PATH"].startswith(str(cuda_dir))


def test_exp1463_command_and_default_wrappers_cover_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-008: command, date, root, and wrapper helpers are explicit."""

    def ok_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 7, stdout="hello", stderr="warn")

    monkeypatch.setattr(subprocess, "run", ok_run)
    assert _run_command(["tool"]) == {
        "command": ["tool"],
        "returncode": 7,
        "stdout_summary": "hello",
        "stderr_summary": "warn",
    }

    def bad_run(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("tool")

    monkeypatch.setattr(subprocess, "run", bad_run)
    assert _run_command(["missing"])["stderr_summary"] == "FileNotFoundError: tool"

    fake_template = types.ModuleType("scripts.experiment_template")
    fake_template._get_repo_root = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "scripts.experiment_template", fake_template)
    assert mod._repo_root() == tmp_path
    assert mod._utc_run_date().isdigit()
    assert mod._venv_python(tmp_path) == sys.executable
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    assert mod._venv_python(tmp_path) == str(venv_python)


def test_exp1463_cuda_discovery_falls_back_to_globbed_libllama(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-008: libllama discovery works across Python minor versions."""
    globbed = tmp_path / ".venv" / "lib" / "python3.11" / "site-packages" / "llama_cpp" / "lib"
    globbed.mkdir(parents=True)
    (globbed / "libllama.so").write_text("", encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        stdout = ""
        if cmd[0] == "ldd":
            stdout = "libcudart.so.12 => not found" if kwargs.get("env") is None else "ok"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    discovery = discover_cuda_runtime_state(project_root=tmp_path, env={})

    assert discovery["libllama_path"].endswith("python3.11/site-packages/llama_cpp/lib/libllama.so")


def test_exp1463_missing_cache_resolution_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFER-SOTA-008: cache-fill attempts report offline and online outcomes."""
    assert attempt_missing_cache_resolution(missing_models=[])["attempted"] is False

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **_: "/cached.gguf"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    assert attempt_missing_cache_resolution(missing_models=[MIDDLE])["status"] == (
        "resolved_from_existing_local_cache"
    )

    def offline_miss(**kwargs: Any) -> str:
        if kwargs["local_files_only"] is True:
            raise RuntimeError("offline miss")
        return "/downloaded.gguf"

    fake_hf.hf_hub_download = offline_miss
    downloaded = attempt_missing_cache_resolution(missing_models=[MIDDLE])
    assert downloaded["status"] == "downloaded_to_local_cache"
    assert downloaded["offline_probe_error"] == "RuntimeError: offline miss"

    blocked = attempt_missing_cache_resolution(missing_models=[MIDDLE], allow_download=False)
    assert blocked["status"] == "blocked_online_download_not_allowed"
    assert blocked["blocker"] == "RuntimeError: offline miss"

    def online_fail(**kwargs: Any) -> str:
        if kwargs["local_files_only"] is True:
            raise RuntimeError("offline miss")
        raise OSError("network down")

    fake_hf.hf_hub_download = online_fail
    failed = attempt_missing_cache_resolution(missing_models=[MIDDLE])
    assert failed["status"] == "blocked_online_download_failed"
    assert failed["blocker"] == "OSError: network down"


def test_exp1463_missing_cache_import_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFER-SOTA-008: absent HuggingFace tooling is an exact cache blocker."""
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    original_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "huggingface_hub":
            raise ImportError("no hub")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    artifact = attempt_missing_cache_resolution(missing_models=[MIDDLE])

    assert artifact["status"] == "blocked_huggingface_hub_unavailable"
    assert artifact["blocker"] == "ImportError: no hub"


def test_exp1463_default_probe_wrappers_delegate_to_exp1442(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-008: wrappers keep Exp 1442 as the live runtime backend."""
    reproduce_calls: list[dict[str, Any]] = []

    def fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        reproduce_calls.append(kwargs)
        return _prior_libcudart_blocked()

    monkeypatch.setattr(mod.preflight, "run_experiment", fake_run_experiment)
    reproduced = mod._reproduce_exp1442_probe(project_root=tmp_path, run_date="20260507")

    assert reproduced["honest_verdict"] == "blocked_no_live_sota_runtime"
    assert reproduce_calls[0]["output_path"] == tmp_path / mod.DEFAULT_REPRODUCED_1442_PATH

    smoke_calls: list[dict[str, Any]] = []

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        smoke = kwargs["smoke_probe_fn"]
        smoke({"hf_id": QWEN, "role": "flagship_moe_runtime_probe", "model_path": QWEN_PATH})
        return _repaired_live()

    def fake_probe(model: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        smoke_calls.append({"model": model, **kwargs})
        completed = kwargs["command_runner"](["probe"], capture_output=True, text=True, timeout=1)
        assert completed.stdout == "ok"
        return {"truly_live": True, "usable_response": True}

    monkeypatch.setattr(mod.preflight, "build_live_runtime_preflight_artifact", fake_build)
    monkeypatch.setattr(mod.preflight, "run_live_probe_subprocess", fake_probe)

    def fake_subprocess_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["env"]["LD_LIBRARY_PATH"] == "/cuda"
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    repaired = mod._run_repaired_preflight(
        project_root=tmp_path,
        run_date="20260507",
        env={"LD_LIBRARY_PATH": "/cuda"},
    )

    assert repaired["local_sota_runtime_ready"] is True
    assert smoke_calls[0]["python_executable"] == sys.executable


def test_exp1463_main_routes_cli_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-INFER-SOTA-008: CLI passes output path and cache policy to the runner."""
    calls: list[dict[str, Any]] = []

    def fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)

    assert mod.main(["--run-date", "20260507", "--output", str(tmp_path / "a.json")]) == 0
    assert calls[0]["run_date"] == "20260507"
    assert calls[0]["output_path"] == tmp_path / "a.json"
    assert calls[0]["missing_cache_resolution_fn"] is mod.attempt_missing_cache_resolution

    assert mod.main(["--no-download"]) == 0
    no_download_fn = calls[1]["missing_cache_resolution_fn"]
    result = no_download_fn(missing_models=[])
    assert result["status"] == "already_resolved_before_cache_attempt"
