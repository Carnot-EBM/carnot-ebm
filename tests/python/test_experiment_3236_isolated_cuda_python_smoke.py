"""Tests for Exp 3236 isolated CUDA Python smoke.

Spec refs: REQ-REPORT-3236, SCENARIO-REPORT-3236.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import isolated_cuda_python_smoke_3236 as mod


SELECTED_PYTHON = "/repo/.venv/bin/python"

REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "cuda_driver_visible",
    "selected_python_torch_import_ok",
    "selected_python_torch_cuda_available",
    "selected_python_device_count",
    "cuda_bindings_import_ok",
    "cuda_bindings_device_count",
    "cuda_python_smoke_passed",
    "recommended_next_task",
    "honest_verdict",
}


def _command(
    command: list[str],
    *,
    returncode: int | None = 0,
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


def _nvidia_query_stdout() -> str:
    return "0, GPU-uuid, NVIDIA GeForce RTX 3090, 595.71.05, 24576, 2, 24126, 0, 48\n"


def _torch_payload(*, import_ok: bool = True, cuda_ok: bool = True) -> str:
    payload: dict[str, Any] = {
        "probe": "exp3236_torch_cuda_probe",
        "project_modules_preimport": [],
        "import_order": ["json", "sys"],
        "selected_python_torch_import_ok": import_ok,
        "selected_python_torch_cuda_available": False,
        "selected_python_device_count": 0,
        "selected_python_device_names": [],
        "selected_python_torch_cuda_version": None,
        "torch_version": None,
    }
    if import_ok:
        payload.update(
            {
                "import_order": [
                    "json",
                    "sys",
                    "torch",
                    "torch.cuda.is_available",
                    "torch.cuda.device_count",
                    "torch.cuda.get_device_name",
                ],
                "selected_python_torch_cuda_available": cuda_ok,
                "selected_python_device_count": 1 if cuda_ok else 0,
                "selected_python_device_names": ["NVIDIA GeForce RTX 3090"] if cuda_ok else [],
                "selected_python_torch_cuda_version": "12.8",
                "torch_version": "2.11.0+cu128",
            }
        )
    else:
        payload["error"] = "ImportError: torch not installed"
    return json.dumps(payload, sort_keys=True) + "\n"


def _cuda_bindings_payload(*, import_ok: bool = True, cuda_ok: bool = True) -> str:
    payload: dict[str, Any] = {
        "probe": "exp3236_cuda_bindings_probe",
        "project_modules_preimport": [],
        "import_order": ["json", "sys"],
        "cuda_bindings_import_ok": import_ok,
        "cuda_bindings_runtime_ok": False,
        "cuda_bindings_device_count": 0,
        "cuda_bindings_device_names": [],
        "cuda_bindings_cuda_error": None,
        "cuda_bindings_runtime_version": None,
        "cuda_bindings_driver_version": None,
    }
    if import_ok:
        payload.update(
            {
                "import_order": [
                    "json",
                    "sys",
                    "cuda.bindings.runtime",
                    "cudaGetDeviceCount",
                    "cudaGetDeviceProperties",
                ],
                "cuda_bindings_runtime_ok": cuda_ok,
                "cuda_bindings_device_count": 1 if cuda_ok else 0,
                "cuda_bindings_device_names": ["NVIDIA GeForce RTX 3090"] if cuda_ok else [],
                "cuda_bindings_cuda_error": "cudaSuccess" if cuda_ok else "cudaErrorUnknown",
                "cuda_bindings_runtime_version": 12090 if cuda_ok else None,
                "cuda_bindings_driver_version": 13020 if cuda_ok else None,
            }
        )
    else:
        payload["error"] = "ModuleNotFoundError: No module named cuda"
    return json.dumps(payload, sort_keys=True) + "\n"


def _runner(
    *,
    nvidia_visible: bool = True,
    torch_import_ok: bool = True,
    torch_cuda_ok: bool = True,
    bindings_import_ok: bool = True,
    bindings_cuda_ok: bool = True,
) -> tuple[mod.CommandRunner, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake(command: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append({"command": command, "kwargs": kwargs})
        joined = "\n".join(command)
        if command[:1] == ["nvidia-smi"] and "--query-gpu=" in joined:
            return _command(command, stdout=_nvidia_query_stdout() if nvidia_visible else "")
        if command == ["nvidia-smi"]:
            stdout = (
                "NVIDIA-SMI 595.71.05 Driver Version: 595.71.05 CUDA Version: 13.2\n"
                if nvidia_visible
                else ""
            )
            return _command(command, returncode=0 if nvidia_visible else 1, stdout=stdout)
        if command[0] == SELECTED_PYTHON and "exp3236_torch_cuda_probe" in joined:
            stderr = "" if torch_cuda_ok else "CUDA initialization: CUDA unknown error\n"
            return _command(
                command,
                stdout=_torch_payload(import_ok=torch_import_ok, cuda_ok=torch_cuda_ok),
                stderr=stderr,
            )
        if command[0] == SELECTED_PYTHON and "exp3236_cuda_bindings_probe" in joined:
            stderr = "" if bindings_cuda_ok else "cudaErrorUnknown: 999\n"
            return _command(
                command,
                stdout=_cuda_bindings_payload(
                    import_ok=bindings_import_ok,
                    cuda_ok=bindings_cuda_ok,
                ),
                stderr=stderr,
            )
        raise AssertionError(f"unexpected command: {command}")

    return fake, calls


def test_req_report_3236_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3236: OpenSpec declares the smoke before implementation."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3236" in spec
    assert "SCENARIO-REPORT-3236" in spec
    assert mod.DEFAULT_ARTIFACT_PATH.as_posix() in spec
    assert "cuda_python_smoke_passed" in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3236_all_green_smoke_opens_next_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3236: all CUDA probes pass before the next gate opens."""

    runner, calls = _runner()
    artifact = mod.build_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={"PATH": "/usr/bin:/opt/cuda/bin", "CUDA_VISIBLE_DEVICES": "0"},
        command_runner=runner,
        monotonic=iter([10.0, 12.0]).__next__,
        tests_run=["REQ-REPORT-3236 focused"],
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3236"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.300"
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["cuda_driver_visible"] is True
    assert artifact["selected_python"] == SELECTED_PYTHON
    assert artifact["selected_python_torch_import_ok"] is True
    assert artifact["selected_python_torch_cuda_available"] is True
    assert artifact["selected_python_device_count"] == 1
    assert artifact["selected_python_torch_cuda_version"] == "12.8"
    assert artifact["selected_python_device_names"] == ["NVIDIA GeForce RTX 3090"]
    assert artifact["cuda_bindings_import_ok"] is True
    assert artifact["cuda_bindings_runtime_ok"] is True
    assert artifact["cuda_bindings_device_count"] == 1
    assert artifact["cuda_bindings_device_names"] == ["NVIDIA GeForce RTX 3090"]
    assert artifact["cuda_python_smoke_passed"] is True
    assert artifact["smoke_block_reasons"] == []
    assert artifact["recommended_next_task"] == "exp3237-llama-cpp-cuda-receipt-smoke-v2"
    assert artifact["no_llama_cpp_rebuild"] is True
    assert artifact["no_full_gguf_load"] is True
    assert artifact["no_mandated_gguf_model_inference"] is True
    assert artifact["tests_run"] == ["REQ-REPORT-3236 focused"]
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "repaired" not in artifact["honest_verdict"].lower()
    assert artifact["selected_python_torch_probe"]["project_modules_preimport"] == []
    assert artifact["cuda_bindings_probe"]["project_modules_preimport"] == []
    assert all("-I" in call["command"] for call in calls if call["command"][0] == SELECTED_PYTHON)
    assert {
        call["kwargs"]["env"]["CARNOT_EXP3236_PROBE_KIND"]
        for call in calls
        if call["command"][0] == SELECTED_PYTHON
    } == {"torch", "cuda_bindings"}


def test_scenario_report_3236_writer_preserves_torch_cuda_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3236: selected-Python CUDA failure is a complete blocked artifact."""

    runner, _calls = _runner(torch_cuda_ok=False)
    output = tmp_path / mod.DEFAULT_ARTIFACT_PATH

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=output,
        selected_python=SELECTED_PYTHON,
        env={"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": ""},
        command_runner=runner,
        monotonic=iter([1.0, 1.25]).__next__,
        tests_run=["writer"],
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == artifact
    assert artifact["cuda_driver_visible"] is True
    assert artifact["selected_python_torch_import_ok"] is True
    assert artifact["selected_python_torch_cuda_available"] is False
    assert artifact["selected_python_device_count"] == 0
    assert artifact["cuda_bindings_import_ok"] is True
    assert artifact["cuda_bindings_device_count"] == 1
    assert artifact["cuda_python_smoke_passed"] is False
    assert artifact["smoke_block_reasons"] == ["selected_python_torch_cuda_unavailable"]
    assert artifact["recommended_next_task"] == "repair_selected_python_torch_cuda_before_exp3237"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "cuda_python_smoke_passed=false" in artifact["honest_verdict"]


def test_scenario_report_3236_runtime_bindings_block_even_when_torch_passes(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3236: the isolated CUDA-only runtime probe must stay explicit."""

    runner, _calls = _runner(bindings_import_ok=False, bindings_cuda_ok=False)
    artifact = mod.build_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        env={"PATH": "/usr/bin"},
        command_runner=runner,
        monotonic=iter([2.0, 3.0]).__next__,
    )

    assert artifact["selected_python_torch_cuda_available"] is True
    assert artifact["selected_python_device_count"] == 1
    assert artifact["cuda_bindings_import_ok"] is False
    assert artifact["cuda_bindings_runtime_ok"] is False
    assert artifact["cuda_bindings_device_count"] == 0
    assert artifact["cuda_python_smoke_passed"] is False
    assert artifact["smoke_block_reasons"] == ["cuda_bindings_unavailable"]
    assert artifact["recommended_next_task"] == "repair_cuda_bindings_runtime_probe_before_exp3237"


def test_req_report_3236_helpers_and_cli_preserve_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-REPORT-3236: helper parsing and CLI wiring preserve bounded evidence."""

    assert mod._repo_root().name in {"carnot", "carnot-ebm"}
    assert mod._selected_python(tmp_path / "missing") != ""
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    assert mod._selected_python(tmp_path) == str(venv_python)
    assert mod._summarize("abcdef", limit=3) == "def"
    assert mod._parse_nvidia_smi_csv("bad\n0, uuid, GPU, 595.71.05, 1, 2, 3, 4, 5\n")[0][
        "memory_free_mib"
    ] == 3
    assert mod._parse_nvidia_smi_csv("0, too, short\n") == []
    assert mod._cuda_version_from_nvidia_smi("Driver Version: 595.71.05 CUDA Version: 13.2") == (
        "13.2"
    )
    assert mod._cuda_version_from_nvidia_smi("no cuda here") is None
    assert mod._json_from_last_line({"stdout": "not-json\n", "stderr": "bad"})["error"] == "bad"
    assert mod._stdout({"stdout": "ok"}) == "ok"
    assert mod._stderr({"stderr": "bad"}) == "bad"
    assert mod._probe_env({"CUDA_VISIBLE_DEVICES": ""}, probe_kind="torch")[
        "CUDA_VISIBLE_DEVICES"
    ] == "0"
    assert mod._as_int("not-an-int") == 0
    assert mod._smoke_block_reasons(
        driver_visible=False,
        torch_import_ok=False,
        torch_cuda_available=False,
        torch_device_count=0,
        cuda_bindings_import_ok=True,
        cuda_bindings_runtime_ok=False,
        cuda_bindings_device_count=0,
    ) == [
        "cuda_driver_not_visible",
        "selected_python_torch_import_failed",
        "cuda_bindings_runtime_no_devices",
    ]
    assert (
        mod._recommended_next_task(
            ["cuda_driver_not_visible", "selected_python_torch_import_failed"]
        )
        == "repair_nvidia_driver_visibility_before_exp3237"
    )
    assert (
        mod._recommended_next_task(["selected_python_torch_import_failed"])
        == "repair_selected_python_torch_import_before_exp3237"
    )
    assert (
        mod._recommended_next_task(["cuda_bindings_runtime_no_devices"])
        == "repair_cuda_bindings_runtime_device_count_before_exp3237"
    )
    assert (
        mod._recommended_next_task(["unknown_block"])
        == "inspect_cuda_python_smoke_block_before_exp3237"
    )
    ok = mod._run_command(["printf", "ok"], timeout_s=5)
    assert ok["returncode"] == 0
    assert ok["stdout"] == "ok"
    missing = mod._run_command([str(tmp_path / "missing-command")], timeout_s=1)
    assert missing["returncode"] is None
    assert "FileNotFoundError" in missing["stderr_summary"]

    runner, _calls = _runner()
    relative = mod.run_experiment(
        project_root=tmp_path,
        output_path=Path("relative_exp3236.json"),
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([4.0, 4.5]).__next__,
    )
    assert (tmp_path / "relative_exp3236.json").is_file()
    assert relative["cuda_python_smoke_passed"] is True

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
            ]
        )
        == 0
    )
    assert calls == [
        {
            "project_root": None,
            "output_path": tmp_path / "out.json",
            "selected_python": SELECTED_PYTHON,
            "tests_run": ["unit"],
        }
    ]
