"""Tests for Exp 3206 CUDA environment forensics ledger.

Spec: REQ-INFER-SOTA-024,
      SCENARIO-INFER-SOTA-024-001,
      SCENARIO-INFER-SOTA-024-002,
      SCENARIO-INFER-SOTA-024-003
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cuda_env_forensics_ledger_3206 as mod
from carnot.reporting.cuda_env_forensics_ledger_3206 import (
    REQUIRED_ARTIFACT_FIELDS,
    build_cuda_env_ledger,
    run_experiment,
)


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


def _runner(
    *,
    torch_cuda: bool = True,
    torch_count: int = 1,
    llama_gpu: bool = True,
    llama_stderr: str = "",
    nvidia_ok: bool = True,
) -> mod.CommandRunner:
    def fake(command: list[str], **_: Any) -> dict[str, Any]:
        joined = "\n".join(command)
        if command[:1] == ["nvidia-smi"] and "--query-gpu=" in joined:
            if not nvidia_ok:
                return _command(command, returncode=127, stderr="missing nvidia-smi\n")
            return _command(
                command,
                stdout="0, NVIDIA GeForce RTX 3090, 595.71.05, 24576, 512, 24064\n",
            )
        if command == ["nvidia-smi"]:
            return _command(command, stdout="NVIDIA-SMI 595.71.05\n")
        if command == ["nvcc", "--version"]:
            return _command(command, stdout="Cuda compilation tools, release 12.8, V12.8.93\n")
        if command[:3] == [SELECTED_PYTHON, "-m", "pip"] and command[-1] == "torch":
            return _command(
                command,
                stdout="Name: torch\nVersion: 2.11.0+cu128\nLocation: /repo/.venv/lib\n",
            )
        if (
            command[:3] == [SELECTED_PYTHON, "-m", "pip"]
            and command[-1] == "llama-cpp-python"
        ):
            return _command(
                command,
                stdout="Name: llama-cpp-python\nVersion: 0.3.23\nLocation: /repo/.venv/lib\n",
            )
        if command[0] == SELECTED_PYTHON and "exp3206_python_env_probe" in joined:
            return _command(
                command,
                stdout=json.dumps(
                    {
                        "executable": SELECTED_PYTHON,
                        "prefix": "/repo/.venv",
                        "base_prefix": "/usr",
                        "virtualenv": "/repo/.venv",
                        "sys_path": ["/usr/lib/python", "/repo/.venv/lib/python/site-packages"],
                    },
                    sort_keys=True,
                )
                + "\n",
            )
        if command[0] == SELECTED_PYTHON and "exp3206_torch_clean_probe" in joined:
            return _command(
                command,
                stdout=json.dumps(
                    {
                        "torch_import_ok": True,
                        "torch_version": "2.11.0+cu128",
                        "torch_cuda_version": "12.8" if torch_cuda else None,
                        "cuda_available": torch_cuda,
                        "device_count": torch_count if torch_cuda else 0,
                        "device_names": ["NVIDIA GeForce RTX 3090"] if torch_cuda else [],
                        "project_modules_preimport": [],
                    },
                    sort_keys=True,
                )
                + "\n",
            )
        if command[0] == SELECTED_PYTHON and "exp3206_llama_cpp_clean_probe" in joined:
            return _command(
                command,
                stdout=json.dumps(
                    {
                        "llama_cpp_import_ok": True,
                        "llama_cpp_version": "0.3.23",
                        "llama_cpp_origin": "/repo/.venv/lib/python/site-packages/llama_cpp/__init__.py",
                        "shared_library_path": "/repo/.venv/lib/libllama.so",
                        "llama_cpp_supports_gpu_offload": llama_gpu,
                        "llama_system_info": "CUDA = 1" if llama_gpu else "",
                    },
                    sort_keys=True,
                )
                + "\n",
                stderr=llama_stderr,
            )
        if command[:3] == ["git", "status", "--porcelain"]:
            return _command(command, stdout="")
        raise AssertionError(f"unexpected command: {command}")

    return fake


def test_req_infer_sota_024_spec_anchor_and_script_exist() -> None:
    """REQ-INFER-SOTA-024: OpenSpec declares the ledger before implementation."""
    spec = (mod.REPO_ROOT / "openspec/capabilities/llm-ebm-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-INFER-SOTA-024" in spec
    assert "SCENARIO-INFER-SOTA-024-001" in spec
    assert mod.DEFAULT_ARTIFACT_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_024_001_clean_cuda_stack_allows_receipt_rerun(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-024-001: clean torch and llama.cpp CUDA allows receipt."""
    artifact = build_cuda_env_ledger(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={
            "CUDA_VISIBLE_DEVICES": "0",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
            "PATH": "/usr/local/cuda/bin:/usr/bin",
            "CMAKE_ARGS": "-DGGML_CUDA=on",
            "FORCE_CMAKE": "1",
            "GGML_CUDA_FORCE_MMQ": "1",
        },
        command_runner=_runner(),
        monotonic=iter([10.0, 12.5]).__next__,
    )

    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3206"
    assert artifact["milestone"] == "2026.05.297"
    assert artifact["selected_python"] == SELECTED_PYTHON
    assert artifact["virtualenv"] == "/repo/.venv"
    assert artifact["nvidia_smi_available"] is True
    assert artifact["gpu_count_nvidia_smi"] == 1
    assert artifact["torch_version"] == "2.11.0+cu128"
    assert artifact["torch_cuda_version"] == "12.8"
    assert artifact["torch_cuda_available_clean_subprocess"] is True
    assert artifact["torch_cuda_device_count_clean_subprocess"] == 1
    assert artifact["llama_cpp_version"] == "0.3.23"
    assert artifact["llama_cpp_origin"].endswith("llama_cpp/__init__.py")
    assert artifact["llama_cpp_cuda_build_detected"] is True
    assert artifact["clean_subprocess_stderr_tail"] == []
    assert artifact["cuda_env_vars"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert artifact["cuda_env_vars"]["GGML_CUDA_FORCE_MMQ"] == "1"
    assert artifact["cuda_env_diagnosed"] is True
    assert artifact["cuda_init_clean"] is True
    assert artifact["recommended_next_action"] == "allow_full_local_sota_receipt_rerun"
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == pytest.approx(2.5)


def test_scenario_024_002_torch_cuda_failure_blocks_full_receipt(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-024-002: selected Python torch CUDA failure blocks receipt."""
    artifact = build_cuda_env_ledger(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={"PATH": "/usr/bin"},
        command_runner=_runner(torch_cuda=False),
        monotonic=iter([1.0, 1.2]).__next__,
    )

    assert artifact["nvidia_smi_available"] is True
    assert artifact["torch_cuda_available_clean_subprocess"] is False
    assert artifact["torch_cuda_device_count_clean_subprocess"] == 0
    assert artifact["cuda_env_diagnosed"] is True
    assert artifact["cuda_init_clean"] is False
    assert artifact["recommended_next_action"] == "repair_selected_python_torch_cuda_before_full_receipt"
    assert artifact["honest_verdict"].startswith("blocked_selected_python_torch_cuda:")


def test_scenario_024_003_llama_cpp_cuda_init_failure_preserved(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-024-003: llama.cpp CUDA init stderr stays in the ledger."""
    stderr = "ggml_cuda_init: failed to initialize CUDA: unknown error\n"
    artifact = build_cuda_env_ledger(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={"PATH": "/usr/bin"},
        command_runner=_runner(llama_gpu=False, llama_stderr=stderr),
        monotonic=iter([2.0, 2.3]).__next__,
    )

    assert artifact["llama_cpp_cuda_build_detected"] is True
    assert artifact["cuda_init_clean"] is False
    assert artifact["recommended_next_action"] == "repair_llama_cpp_cuda_initialization_or_rebuild"
    assert artifact["clean_subprocess_stderr_tail"] == [
        "llama_cpp: ggml_cuda_init: failed to initialize CUDA: unknown error"
    ]
    assert artifact["honest_verdict"].startswith("blocked_llama_cpp_cuda_init:")


def test_req_infer_sota_024_helpers_and_writer_cover_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-024: helpers preserve command failures and CLI wiring."""
    assert mod._repo_root().name in {"carnot", "carnot-ebm"}
    assert mod._selected_python(tmp_path / "missing") != ""
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    assert mod._selected_python(tmp_path) == str(venv_python)
    assert mod._virtualenv_from_python(SELECTED_PYTHON, {}, {}) == "/repo/.venv"
    assert mod._virtualenv_from_python("/usr/bin/python", {}, {"virtualenv": "/tmp/venv"}) == (
        "/tmp/venv"
    )
    assert mod._parse_pip_show("Name: torch\nVersion: 2.11\nBroken line\n") == {
        "Name": "torch",
        "Version": "2.11",
    }
    assert mod._json_from_last_line({"stdout": "not-json\n", "stderr": "bad"})["error"] == (
        "bad"
    )
    assert mod._stderr_tail("a\nb\nc\n", label="probe", limit=2) == ["probe: b", "probe: c"]
    assert mod._parse_nvidia_smi_csv("bad\n0, GPU, 595.71.05, 24576, 1, 24575\n")[0][
        "name"
    ] == "GPU"

    ok = mod._run_command(["printf", "ok"], timeout_s=5)
    assert ok["returncode"] == 0
    assert ok["stdout"] == "ok"
    missing = mod._run_command([str(tmp_path / "missing-command")], timeout_s=1)
    assert missing["returncode"] is None
    assert "FileNotFoundError" in missing["stderr_summary"]

    output = tmp_path / "results" / "experiment_3206.json"
    artifact = run_experiment(
        project_root=tmp_path,
        output_path=output,
        selected_python=SELECTED_PYTHON,
        env={"PATH": "/usr/bin"},
        command_runner=_runner(nvidia_ok=False),
        monotonic=iter([3.0, 3.1]).__next__,
        tests_run=["REQ-INFER-SOTA-024 focused"],
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["nvidia_smi_available"] is False
    assert artifact["recommended_next_action"] == "repair_nvidia_visibility_before_cuda_receipt"
    assert artifact["tests_run"] == ["REQ-INFER-SOTA-024 focused"]

    calls: list[dict[str, Any]] = []

    def fake_run_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert (
        mod.main(
            [
                "--output",
                str(tmp_path / "out.json"),
                "--selected-python",
                SELECTED_PYTHON,
                "--test-run",
                "unit",
                "--test-run",
                "coverage",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "output_path": tmp_path / "out.json",
            "selected_python": SELECTED_PYTHON,
            "tests_run": ["unit", "coverage"],
        }
    ]
