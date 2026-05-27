"""Tests for Exp 3207 llama.cpp CUDA rebuild clean subprocess gate.

Spec: REQ-INFER-SOTA-025,
      SCENARIO-INFER-SOTA-025-001,
      SCENARIO-INFER-SOTA-025-002,
      SCENARIO-INFER-SOTA-025-003
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import llama_cpp_cuda_rebuild_clean_subprocess_3207 as mod


SELECTED_PYTHON = "/repo/.venv/bin/python"


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


def _write_ledger(
    root: Path,
    *,
    recommended_next_action: str,
    llama_version: str = "0.3.23",
) -> Path:
    path = root / mod.DEFAULT_ENV_LEDGER_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "carnot.cuda_env_forensics_ledger.v1",
                "experiment_id": "exp3206",
                "recommended_next_action": recommended_next_action,
                "honest_verdict": f"ledger: {recommended_next_action}",
                "pip_show": {
                    "llama-cpp-python": {"metadata": {"Version": llama_version}},
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _torch_payload(cuda_available: bool) -> str:
    return (
        json.dumps(
            {
                "probe": "exp3206_torch_clean_probe",
                "torch_import_ok": True,
                "torch_version": "2.11.0+cu128",
                "torch_cuda_version": "12.8" if cuda_available else None,
                "cuda_available": cuda_available,
                "device_count": 1 if cuda_available else 0,
                "device_names": ["NVIDIA GeForce RTX 3090"] if cuda_available else [],
            },
            sort_keys=True,
        )
        + "\n"
    )


def _llama_payload(*, supports_gpu: bool, system_info: str = "CUDA : ARCHS = 860") -> str:
    return (
        json.dumps(
            {
                "probe": "exp3206_llama_cpp_clean_probe",
                "llama_cpp_import_ok": True,
                "llama_cpp_version": "0.3.23",
                "llama_cpp_origin": "/repo/.venv/lib/python/site-packages/llama_cpp/__init__.py",
                "shared_library_path": "/repo/.venv/lib/python/site-packages/llama_cpp/lib/libllama.so",
                "llama_cpp_supports_gpu_offload": supports_gpu,
                "llama_system_info": system_info,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _runner(
    *,
    torch_cuda: bool,
    llama_supports_before: bool,
    llama_supports_after: bool | None = None,
    rebuild_returncode: int = 0,
) -> tuple[mod.CommandRunner, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    llama_calls = 0

    def fake(command: list[str], **kwargs: Any) -> dict[str, Any]:
        nonlocal llama_calls
        calls.append({"command": command, "kwargs": kwargs})
        joined = "\n".join(command)
        if command[:3] == ["git", "status", "--porcelain"]:
            return _command(command)
        if command[0] == SELECTED_PYTHON and "exp3206_torch_clean_probe" in joined:
            stderr = "CUDA initialization: CUDA unknown error\n" if not torch_cuda else ""
            return _command(command, stdout=_torch_payload(torch_cuda), stderr=stderr)
        if command[0] == SELECTED_PYTHON and "exp3206_llama_cpp_clean_probe" in joined:
            llama_calls += 1
            supports = llama_supports_before
            if llama_calls > 1 and llama_supports_after is not None:
                supports = llama_supports_after
            stderr = (
                "" if supports else "ggml_cuda_init: failed to initialize CUDA: unknown error\n"
            )
            return _command(command, stdout=_llama_payload(supports_gpu=supports), stderr=stderr)
        if command[:3] == [SELECTED_PYTHON, "-m", "pip"]:
            return _command(
                command,
                returncode=rebuild_returncode,
                stdout="building llama-cpp-python with CUDA\n",
                stderr=(
                    "Successfully built llama-cpp-python\n"
                    if rebuild_returncode == 0
                    else "CMake Error: CUDA compiler failed\n"
                ),
            )
        raise AssertionError(f"unexpected command: {command}")

    return fake, calls


def test_req_infer_sota_025_spec_anchor_and_script_exist() -> None:
    """REQ-INFER-SOTA-025: OpenSpec declares the 3207 gate before implementation."""
    spec = (mod.REPO_ROOT / "openspec/capabilities/llm-ebm-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-INFER-SOTA-025" in spec
    assert "SCENARIO-INFER-SOTA-025-001" in spec
    assert "SCENARIO-INFER-SOTA-025-002" in spec
    assert "SCENARIO-INFER-SOTA-025-003" in spec
    assert mod.DEFAULT_ARTIFACT_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_025_001_torch_cuda_blocker_stops_blind_rebuild(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-025-001: torch CUDA failure blocks any rebuild."""
    ledger = _write_ledger(
        tmp_path,
        recommended_next_action="repair_selected_python_torch_cuda_before_full_receipt",
    )
    runner, calls = _runner(torch_cuda=False, llama_supports_before=False)

    artifact = mod.build_artifact(
        project_root=tmp_path,
        env_ledger_path=ledger,
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([1.0, 1.5]).__next__,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3207"
    assert artifact["milestone"] == "2026.05.297"
    assert artifact["env_ledger_artifact"] == str(ledger)
    assert artifact["rebuild_attempted"] is False
    assert artifact["rebuild_command_summary"] == []
    assert artifact["torch_cuda_available_after"] is False
    assert artifact["llama_cpp_cuda_build_detected_after"] is True
    assert artifact["clean_subprocess_gpu_offload_probe_passed"] is False
    assert artifact["cpu_fallback_only"] is True
    assert artifact["cuda_receipt_ready"] is False
    assert artifact["clean_rerun_allowed_candidate"] is False
    assert "selected_python_torch_cuda_unavailable" in artifact["blocker"]
    assert "CUDA unknown error" in artifact["blocker"]
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["honest_verdict"].startswith("blocked_selected_python_torch_cuda:")
    assert artifact["duration_s"] == pytest.approx(0.5)
    assert not any(call["command"][:3] == [SELECTED_PYTHON, "-m", "pip"] for call in calls)


def test_scenario_025_002_rebuild_then_clean_probe_opens_candidate(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-025-002/003: CUDA rebuild precedes clean pass."""
    ledger = _write_ledger(
        tmp_path, recommended_next_action="rebuild_llama_cpp_python_with_ggml_cuda"
    )
    runner, calls = _runner(
        torch_cuda=True,
        llama_supports_before=False,
        llama_supports_after=True,
    )

    artifact = mod.build_artifact(
        project_root=tmp_path,
        env_ledger_path=ledger,
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([10.0, 15.0]).__next__,
    )

    rebuild_calls = [
        call for call in calls if call["command"][:3] == [SELECTED_PYTHON, "-m", "pip"]
    ]
    assert len(rebuild_calls) == 1
    assert "--no-binary" in rebuild_calls[0]["command"]
    assert "llama-cpp-python==0.3.23" in rebuild_calls[0]["command"]
    assert rebuild_calls[0]["kwargs"]["env"]["CMAKE_ARGS"] == "-DGGML_CUDA=ON"
    assert rebuild_calls[0]["kwargs"]["env"]["FORCE_CMAKE"] == "1"

    assert artifact["rebuild_attempted"] is True
    assert any("GGML_CUDA=ON" in item for item in artifact["rebuild_command_summary"])
    assert artifact["rebuild_log_tail"] == [
        "stdout: building llama-cpp-python with CUDA",
        "stderr: Successfully built llama-cpp-python",
    ]
    assert artifact["torch_cuda_available_after"] is True
    assert artifact["llama_cpp_cuda_build_detected_after"] is True
    assert artifact["clean_subprocess_gpu_offload_probe_passed"] is True
    assert artifact["cpu_fallback_only"] is False
    assert artifact["cuda_receipt_ready"] is True
    assert artifact["clean_rerun_allowed_candidate"] is True
    assert artifact["blocker"] is None
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_025_rebuild_failure_and_cli_writer_are_honest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-025: rebuild failures stay blocked and CLI writes JSON."""
    ledger = _write_ledger(
        tmp_path, recommended_next_action="repair_llama_cpp_cuda_initialization_or_rebuild"
    )
    runner, _calls = _runner(
        torch_cuda=True,
        llama_supports_before=False,
        llama_supports_after=False,
        rebuild_returncode=1,
    )

    artifact = mod.build_artifact(
        project_root=tmp_path,
        env_ledger_path=ledger,
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([20.0, 23.0]).__next__,
        tests_run=["REQ-INFER-SOTA-025 focused"],
    )
    assert artifact["rebuild_attempted"] is True
    assert artifact["clean_subprocess_gpu_offload_probe_passed"] is False
    assert "llama_cpp_cuda_rebuild_failed" in artifact["blocker"]
    assert artifact["honest_verdict"].startswith("blocked_llama_cpp_cuda_rebuild:")
    assert artifact["tests_run"] == ["REQ-INFER-SOTA-025 focused"]

    output = tmp_path / "results" / "experiment_3207.json"
    written = mod.run_experiment(
        project_root=tmp_path,
        output_path=output,
        env_ledger_path=ledger,
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([30.0, 33.0]).__next__,
        tests_run=["writer"],
    )
    assert json.loads(output.read_text(encoding="utf-8")) == written

    calls: list[dict[str, Any]] = []

    def fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert (
        mod.main(
            [
                "--output",
                str(output),
                "--env-ledger",
                str(ledger),
                "--selected-python",
                SELECTED_PYTHON,
                "--test-run",
                "unit",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "output_path": output,
            "env_ledger_path": ledger,
            "selected_python": SELECTED_PYTHON,
            "tests_run": ["unit"],
        }
    ]


def test_req_025_helpers_preserve_missing_ledger_and_subprocess_failures(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-025: helper functions preserve blocked evidence."""
    assert mod._repo_root().name in {"carnot", "carnot-ebm"}
    assert mod._selected_python(tmp_path / "missing") != ""
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    assert mod._selected_python(tmp_path) == str(venv_python)
    assert mod._load_ledger(tmp_path / "missing.json")["load_error"].startswith("FileNotFoundError")
    assert mod._llama_version_from_ledger({}) is None
    assert mod._llama_version_from_ledger({"pip_show": {}}) is None
    assert mod._llama_version_from_ledger({"pip_show": {"llama-cpp-python": {}}}) is None
    assert mod._json_from_last_line({"stdout": "not-json\n", "stderr": "bad"})["error"] == "bad"
    assert mod._tail_lines("a\nb\n", label="x", limit=1) == ["x: b"]
    assert "stderr=none" in mod._llama_blocker(
        {"llama_cpp_import_ok": False, "llama_cpp_supports_gpu_offload": False},
        [],
        False,
    )
    assert mod._honest_verdict(ready=False, blocker=None).startswith(
        "blocked_llama_cpp_gpu_offload:"
    )

    result = mod._run_command([str(tmp_path / "missing-command")], timeout_s=1)
    assert result["returncode"] is None
    assert "FileNotFoundError" in result["stderr_summary"]

    ledger = _write_ledger(tmp_path, recommended_next_action="allow_full_local_sota_receipt_rerun")

    def missing_llama_runner(command: list[str], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        joined = "\n".join(command)
        if command[:3] == ["git", "status", "--porcelain"]:
            return _command(command)
        if command[0] == SELECTED_PYTHON and "exp3206_torch_clean_probe" in joined:
            return _command(command, stdout=_torch_payload(True))
        if command[0] == SELECTED_PYTHON and "exp3206_llama_cpp_clean_probe" in joined:
            return _command(
                command,
                stdout=json.dumps({"llama_cpp_import_ok": False}, sort_keys=True) + "\n",
                stderr="ModuleNotFoundError: No module named llama_cpp\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    artifact = mod.build_artifact(
        project_root=tmp_path,
        env_ledger_path=ledger,
        selected_python=SELECTED_PYTHON,
        env={"CUDA_VISIBLE_DEVICES": "0"},
        command_runner=missing_llama_runner,
        monotonic=iter([1.0, 2.0]).__next__,
    )
    assert artifact["rebuild_attempted"] is False
    assert artifact["blocker"].startswith("llama_cpp_gpu_offload_probe_failed:")
    assert artifact["cuda_receipt_ready"] is False
