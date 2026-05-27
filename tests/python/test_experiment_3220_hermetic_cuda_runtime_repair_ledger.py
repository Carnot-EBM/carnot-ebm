"""Tests for Exp 3220 hermetic CUDA runtime repair ledger.

Spec: REQ-INFER-SOTA-026,
      SCENARIO-INFER-SOTA-026-001,
      SCENARIO-INFER-SOTA-026-002,
      SCENARIO-INFER-SOTA-026-003
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import hermetic_cuda_runtime_repair_ledger_3220 as mod


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


def _torch_payload(*, cuda_ok: bool, stage: str) -> str:
    return (
        json.dumps(
            {
                "probe": "exp3220_torch_cuda_probe",
                "stage": stage,
                "torch_import_ok": True,
                "torch_version": "2.11.0+cu128",
                "torch_cuda_version": "12.8" if cuda_ok else None,
                "cuda_available": cuda_ok,
                "device_count": 1 if cuda_ok else 0,
                "device_names": ["NVIDIA GeForce RTX 3090"] if cuda_ok else [],
                "import_order": [
                    "json",
                    "sys",
                    "torch",
                    "torch.cuda.is_available",
                    "torch.cuda.device_count",
                ],
                "project_modules_preimport": [],
            },
            sort_keys=True,
        )
        + "\n"
    )


def _cuda_bindings_payload(*, cuda_ok: bool, stage: str = "selected_after") -> str:
    return (
        json.dumps(
            {
                "probe": "exp3220_cuda_bindings_probe",
                "stage": stage,
                "cuda_bindings_import_ok": True,
                "cuda_runtime_ok": cuda_ok,
                "device_count": 1 if cuda_ok else 0,
                "device_names": ["NVIDIA GeForce RTX 3090"] if cuda_ok else [],
                "cuda_runtime_version": 12090,
                "cuda_driver_version": 13020,
                "import_order": ["json", "cuda.bindings.runtime", "cudaGetDeviceCount"],
            },
            sort_keys=True,
        )
        + "\n"
    )


def _llama_payload(*, supports_gpu: bool = True) -> str:
    return (
        json.dumps(
            {
                "probe": "exp3220_llama_cpp_linkage_probe",
                "llama_cpp_import_ok": True,
                "llama_cpp_version": "0.3.23",
                "llama_cpp_origin": "/repo/.venv/lib/python/site-packages/llama_cpp/__init__.py",
                "shared_library_path": "/repo/.venv/lib/python/site-packages/llama_cpp/lib/libllama.so",
                "llama_cpp_supports_gpu_offload": supports_gpu,
                "llama_system_info": "CUDA : ARCHS = 860" if supports_gpu else "",
            },
            sort_keys=True,
        )
        + "\n"
    )


def _nvidia_query_stdout() -> str:
    return "0, GPU-uuid, NVIDIA GeForce RTX 3090, 595.71.05, 24576, 2, 24126, 0, 48\n"


def _runner(
    *,
    selected_before: bool,
    selected_after: bool,
    isolated_cuda: bool = False,
    llama_supports_gpu: bool = True,
) -> tuple[mod.CommandRunner, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake(command: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append({"command": command, "kwargs": kwargs})
        joined = "\n".join(command)
        env = kwargs.get("env") or {}
        stage = env.get("CARNOT_EXP3220_PROBE_STAGE", "")
        if command[:1] == ["nvidia-smi"] and "--query-gpu=" in joined:
            return _command(command, stdout=_nvidia_query_stdout())
        if command == ["nvidia-smi"]:
            return _command(
                command,
                stdout="NVIDIA-SMI 595.71.05    Driver Version: 595.71.05    CUDA Version: 13.2\n",
            )
        if command[:3] == ["git", "status", "--porcelain"]:
            return _command(command)
        if command[:3] == [SELECTED_PYTHON, "-m", "pip"] and command[-1] == "torch":
            return _command(command, stdout="Name: torch\nVersion: 2.11.0+cu128\n")
        if command[:3] == [SELECTED_PYTHON, "-m", "pip"] and command[-1] == "cuda-bindings":
            return _command(command, stdout="Name: cuda-bindings\nVersion: 12.9.4\n")
        if command[0] == SELECTED_PYTHON and "exp3220_torch_cuda_probe" in joined:
            ok = selected_after if stage == "selected_after" else selected_before
            stderr = "" if ok else "CUDA initialization: CUDA unknown error\n"
            return _command(command, stdout=_torch_payload(cuda_ok=ok, stage=stage), stderr=stderr)
        if command[0] == SELECTED_PYTHON and "exp3220_cuda_bindings_probe" in joined:
            ok = selected_after
            stderr = "" if ok else "cudaErrorUnknown: 999\n"
            return _command(
                command,
                stdout=_cuda_bindings_payload(cuda_ok=ok, stage=stage),
                stderr=stderr,
            )
        if command[0] == SELECTED_PYTHON and "exp3220_llama_cpp_linkage_probe" in joined:
            stderr = "" if llama_supports_gpu else "ggml_cuda_init: failed to initialize CUDA\n"
            return _command(
                command, stdout=_llama_payload(supports_gpu=llama_supports_gpu), stderr=stderr
            )
        if command == ["/usr/bin/python3", "-m", "venv", "/tmp/exp3220-isolated"]:
            return _command(command, stdout="created venv\n")
        if command == [
            "/tmp/exp3220-isolated/bin/python",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "cuda-bindings==12.9.4",
        ]:
            return _command(command, stdout="Successfully installed cuda-bindings-12.9.4\n")
        if command[:3] == ["/tmp/exp3220-isolated/bin/python", "-m", "pip"]:
            return _command(command, stdout="Name: cuda-bindings\nVersion: 12.9.4\n")
        if command[0] == "/tmp/exp3220-isolated/bin/python" and (
            "exp3220_cuda_bindings_probe" in joined
        ):
            stderr = "" if isolated_cuda else "cudaErrorUnknown: 999\n"
            return _command(
                command,
                stdout=_cuda_bindings_payload(cuda_ok=isolated_cuda, stage="isolated_cuda_venv"),
                stderr=stderr,
            )
        raise AssertionError(f"unexpected command: {command}")

    return fake, calls


def test_req_infer_sota_026_spec_anchor_and_script_exist() -> None:
    """REQ-INFER-SOTA-026: OpenSpec declares the ledger before implementation."""
    spec = (mod.REPO_ROOT / "openspec/capabilities/llm-ebm-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-INFER-SOTA-026" in spec
    assert "SCENARIO-INFER-SOTA-026-001" in spec
    assert "SCENARIO-INFER-SOTA-026-002" in spec
    assert "SCENARIO-INFER-SOTA-026-003" in spec
    assert mod.DEFAULT_ARTIFACT_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_026_001_sanitized_selected_python_opens_candidate(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-026-001: sanitized selected Python can open the gate."""
    runner, _calls = _runner(selected_before=False, selected_after=True)

    artifact = mod.build_cuda_runtime_repair_ledger(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={
            "CUDA_VISIBLE_DEVICES": "0",
            "PATH": "/opt/rocm/bin:/opt/cuda/bin:/usr/bin",
            "LD_LIBRARY_PATH": "/opt/rocm/lib",
            "PYTHONPATH": "/repo/python",
        },
        command_runner=runner,
        create_isolated_venv=False,
        monotonic=iter([10.0, 12.0]).__next__,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3220"
    assert artifact["milestone"] == "2026.05.298"
    assert artifact["selected_python"] == SELECTED_PYTHON
    assert artifact["selected_python_cuda_ok_before"] is False
    assert artifact["selected_python_cuda_ok_after"] is True
    assert artifact["isolated_cuda_venv_created"] is False
    assert artifact["isolated_cuda_venv_cuda_ok"] is False
    assert artifact["cuda_visible_devices"] == "0"
    assert artifact["nvidia_smi_available"] is True
    assert artifact["gpu_count_nvidia_smi"] == 1
    assert artifact["driver_version"] == "595.71.05"
    assert artifact["torch_version_selected"] == "2.11.0+cu128"
    assert artifact["torch_cuda_version_selected"] == "12.8"
    assert artifact["cuda_receipt_ready_candidate"] is True
    assert (
        artifact["recommended_next_action"]
        == "allow_bounded_cuda_receipt_candidate_no_model_loaded"
    )
    assert artifact["inference_substrate"] == "cuda_runtime_forensics_no_model"
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["selected_python_probe_after"]["import_order"][2] == "torch"
    assert "model_specs" not in artifact
    assert "models_used" not in artifact
    assert any(
        item["kind"] == "path_contains_rocm" for item in artifact["environment_pollution_findings"]
    )
    assert any(
        item["action"] == "sanitized_selected_python_cuda_probe"
        for item in artifact["repair_actions_attempted"]
    )


def test_scenario_026_002_isolated_venv_separates_selected_venv_failure(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-026-002: isolated CUDA runtime can isolate venv failure."""
    runner, calls = _runner(selected_before=False, selected_after=False, isolated_cuda=True)

    artifact = mod.build_cuda_runtime_repair_ledger(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={"PATH": "/usr/bin:/opt/cuda/bin", "CUDA_VISIBLE_DEVICES": "0"},
        command_runner=runner,
        create_isolated_venv=True,
        isolated_venv_path=Path("/tmp/exp3220-isolated"),
        isolated_base_python="/usr/bin/python3",
        monotonic=iter([20.0, 27.0]).__next__,
    )

    assert artifact["selected_python_cuda_ok_before"] is False
    assert artifact["selected_python_cuda_ok_after"] is False
    assert artifact["isolated_cuda_venv_created"] is True
    assert artifact["isolated_cuda_venv_cuda_ok"] is True
    assert artifact["cuda_receipt_ready_candidate"] is False
    assert (
        artifact["recommended_next_action"] == "repair_selected_python_torch_cuda_or_recreate_venv"
    )
    assert artifact["honest_verdict"].startswith("blocked_selected_python_cuda:")
    assert artifact["isolated_cuda_venv"]["path"] == "/tmp/exp3220-isolated"
    assert artifact["isolated_cuda_venv"]["package_versions"]["cuda-bindings"] == "12.9.4"
    assert any(
        call["command"] == ["/usr/bin/python3", "-m", "venv", "/tmp/exp3220-isolated"]
        for call in calls
    )
    assert any(
        item["action"] == "create_isolated_cuda_venv" and item["status"] == "created"
        for item in artifact["repair_actions_attempted"]
    )


def test_scenario_026_003_pollution_and_writer_keep_nvidia_from_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-026-003: pollution is explicit and not receipt evidence."""
    runner, _calls = _runner(
        selected_before=False,
        selected_after=False,
        isolated_cuda=False,
        llama_supports_gpu=False,
    )
    env = {
        "PATH": "/opt/rocm/bin:/tools/Xilinx/2025.2/Vitis/bin:/opt/cuda/bin:/usr/bin",
        "LD_LIBRARY_PATH": "/opt/rocm/lib",
        "CUDA_HOME": "/opt/cuda",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": "/repo/python",
        "CMAKE_ARGS": "-DGGML_CUDA=ON",
        "FORCE_CMAKE": "1",
    }

    artifact = mod.build_cuda_runtime_repair_ledger(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=runner,
        create_isolated_venv=False,
        monotonic=iter([1.0, 2.0]).__next__,
        tests_run=["REQ-INFER-SOTA-026 focused"],
    )

    kinds = {item["kind"] for item in artifact["environment_pollution_findings"]}
    assert {
        "path_contains_rocm",
        "path_contains_xdna_tooling",
        "ld_library_path_contains_rocm",
        "pythonpath_set",
        "cmake_args_set",
        "force_cmake_set",
    } <= kinds
    assert artifact["nvidia_smi_available"] is True
    assert artifact["cuda_receipt_ready_candidate"] is False
    assert (
        artifact["recommended_next_action"]
        == "create_isolated_cuda_venv_to_disambiguate_selected_venv"
    )
    assert artifact["tests_run"] == ["REQ-INFER-SOTA-026 focused"]

    output = tmp_path / "results" / "experiment_3220.json"
    written = mod.run_experiment(
        project_root=tmp_path,
        output_path=output,
        selected_python=SELECTED_PYTHON,
        env=env,
        command_runner=runner,
        create_isolated_venv=False,
        monotonic=iter([3.0, 4.0]).__next__,
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
                "--selected-python",
                SELECTED_PYTHON,
                "--skip-isolated-venv",
                "--test-run",
                "unit",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "output_path": output,
            "selected_python": SELECTED_PYTHON,
            "create_isolated_venv": False,
            "isolated_venv_path": None,
            "isolated_base_python": None,
            "tests_run": ["unit"],
        }
    ]


def test_req_026_helpers_preserve_failure_evidence(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-026: helpers preserve parse and command failure evidence."""
    assert mod._repo_root().name in {"carnot", "carnot-ebm"}
    assert mod._selected_python(tmp_path / "missing") != ""
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    assert mod._selected_python(tmp_path) == str(venv_python)
    assert mod._parse_pip_show("Name: cuda-bindings\nVersion: 12.9.4\n") == {
        "Name": "cuda-bindings",
        "Version": "12.9.4",
    }
    assert mod._json_from_last_line({"stdout": "not-json\n", "stderr": "bad"})["error"] == "bad"
    assert (
        mod._parse_nvidia_smi_csv("bad\n0, uuid, GPU, 595.71.05, 1, 2, 3, 4, 5\n")[0][
            "memory_free_mib"
        ]
        == 3
    )
    assert mod._cuda_version_from_nvidia_smi("Driver Version: 595.71.05  CUDA Version: 13.2") == (
        "13.2"
    )
    assert mod._command_status({"returncode": None}) == "error"
    assert mod._command_status({"returncode": 1}) == "failed"
    assert mod._command_status({"returncode": 0}) == "ok"
    assert mod._default_isolated_venv_path(iter([42.0]).__next__) == (
        Path("/tmp") / "carnot-exp3220-cuda-only-42000000"
    )
    assert (
        mod._sanitized_cuda_env({"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})[
            "LD_LIBRARY_PATH"
        ]
        == "/usr/local/cuda/lib64"
    )
    assert (
        mod._recommended_next_action(
            nvidia_available=True,
            selected_after_ok=True,
            selected_equivalent_ok=False,
            isolated_created=False,
            isolated_ok=False,
            llama_probe={"llama_cpp_import_ok": True, "llama_cpp_supports_gpu_offload": False},
        )
        == "repair_llama_cpp_linkage_after_cuda_runtime_ok"
    )
    assert (
        mod._recommended_next_action(
            nvidia_available=True,
            selected_after_ok=False,
            selected_equivalent_ok=False,
            isolated_created=True,
            isolated_ok=False,
            llama_probe={},
        )
        == "repair_system_driver_cuda_runtime_boundary"
    )

    ok = mod._run_command(["printf", "ok"], timeout_s=5)
    assert ok["returncode"] == 0
    assert ok["stdout"] == "ok"
    result = mod._run_command([str(tmp_path / "missing-command")], timeout_s=1)
    assert result["returncode"] is None
    assert "FileNotFoundError" in result["stderr_summary"]
