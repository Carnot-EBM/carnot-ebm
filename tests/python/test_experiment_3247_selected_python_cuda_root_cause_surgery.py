"""Tests for Exp 3247 selected-Python CUDA root-cause surgery.

Spec refs: REQ-REPORT-3247, SCENARIO-REPORT-3247.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import selected_python_cuda_root_cause_surgery_3247 as mod


SELECTED_PYTHON = "/repo/.venv/bin/python"

REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "preconditions_checked",
    "cuda_root_cause_class",
    "selected_python_path",
    "selected_python_torch_cuda_available_before",
    "cuda_bindings_device_count_before",
    "repair_actions_attempted",
    "selected_python_torch_cuda_available_after",
    "cuda_bindings_device_count_after",
    "selected_python_cuda_repaired_candidate",
    "next_smoke_allowed",
    "random_seed",
    "reproducibility_checksum",
    "protected_files_untouched",
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


def _write_exp3236(
    root: Path,
    *,
    selected_python: str = SELECTED_PYTHON,
    torch_cuda_available: bool = False,
    bindings_device_count: int = 0,
) -> None:
    path = root / mod.EXP3236_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": "exp3236",
                "task_id": "exp3236-isolated-cuda-python-smoke-v1",
                "milestone": "2026.05.300",
                "cuda_driver_visible": True,
                "selected_python": selected_python,
                "torch_version": "2.11.0+cu128",
                "selected_python_torch_cuda_version": "12.8",
                "selected_python_torch_cuda_available": torch_cuda_available,
                "selected_python_device_count": 0,
                "cuda_bindings_import_ok": True,
                "cuda_bindings_runtime_ok": bindings_device_count > 0,
                "cuda_bindings_device_count": bindings_device_count,
                "cuda_python_smoke_passed": False,
                "smoke_block_reasons": [
                    "selected_python_torch_cuda_unavailable",
                    "cuda_bindings_runtime_no_devices",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _nvidia_query_stdout() -> str:
    return "0, GPU-uuid, NVIDIA GeForce RTX 3090, 595.71.05, 24576, 2, 24126, 0, 48\n"


def _torch_payload(*, cuda_ok: bool, count: int) -> str:
    payload = {
        "probe": "exp3247_torch_cuda_probe",
        "python_version": "3.14.4",
        "selected_python_torch_import_ok": True,
        "torch_version": "2.11.0+cu128",
        "selected_python_torch_cuda_version": "12.8",
        "selected_python_torch_cuda_available": cuda_ok,
        "selected_python_device_count": count,
        "selected_python_device_names": ["NVIDIA GeForce RTX 3090"] if count else [],
        "selected_python_cuda_error": None if cuda_ok else "CUDA unknown error",
    }
    return json.dumps(payload, sort_keys=True) + "\n"


def _bindings_payload(*, import_ok: bool, runtime_ok: bool, count: int) -> str:
    payload = {
        "probe": "exp3247_cuda_bindings_probe",
        "python_version": "3.14.4",
        "cuda_bindings_import_ok": import_ok,
        "cuda_bindings_runtime_ok": runtime_ok,
        "cuda_bindings_device_count": count,
        "cuda_bindings_device_names": ["NVIDIA GeForce RTX 3090"] if count else [],
        "cuda_bindings_cuda_error": "cudaSuccess" if runtime_ok else "999",
        "cuda_bindings_runtime_version": 12090 if runtime_ok else None,
        "cuda_bindings_driver_version": 13020 if runtime_ok else None,
    }
    return json.dumps(payload, sort_keys=True) + "\n"


def _runner(
    *,
    driver_visible: bool = True,
    before_torch_ok: bool = False,
    before_torch_count: int = 0,
    before_bindings_import_ok: bool = True,
    before_bindings_ok: bool = False,
    before_bindings_count: int = 0,
    after_torch_ok: bool = False,
    after_torch_count: int = 0,
    after_bindings_import_ok: bool = True,
    after_bindings_ok: bool = False,
    after_bindings_count: int = 0,
) -> tuple[mod.CommandRunner, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake(command: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append({"command": command, "kwargs": kwargs})
        joined = "\n".join(command)
        env = kwargs.get("env", {})
        stage = env.get("CARNOT_EXP3247_PROBE_STAGE", "before")
        if command[:1] == ["nvidia-smi"] and "--query-gpu=" in joined:
            return _command(
                command,
                returncode=0 if driver_visible else 1,
                stdout=_nvidia_query_stdout() if driver_visible else "",
                stderr="" if driver_visible else "nvidia-smi failed",
            )
        if command == ["nvidia-smi"]:
            stdout = (
                "NVIDIA-SMI 595.71.05 Driver Version: 595.71.05 CUDA Version: 13.2\n"
                if driver_visible
                else ""
            )
            return _command(command, returncode=0 if driver_visible else 1, stdout=stdout)
        if command[0] == SELECTED_PYTHON and "exp3247_torch_cuda_probe" in joined:
            cuda_ok = after_torch_ok if stage == "after" else before_torch_ok
            count = after_torch_count if stage == "after" else before_torch_count
            return _command(
                command,
                stdout=_torch_payload(cuda_ok=cuda_ok, count=count),
                stderr="" if cuda_ok else "CUDA initialization: CUDA unknown error\n",
            )
        if command[0] == SELECTED_PYTHON and "exp3247_cuda_bindings_probe" in joined:
            import_ok = (
                after_bindings_import_ok if stage == "after" else before_bindings_import_ok
            )
            runtime_ok = after_bindings_ok if stage == "after" else before_bindings_ok
            count = after_bindings_count if stage == "after" else before_bindings_count
            return _command(
                command,
                stdout=_bindings_payload(import_ok=import_ok, runtime_ok=runtime_ok, count=count),
                stderr="" if runtime_ok else "cudaGetDeviceCount returned 999\n",
            )
        raise AssertionError(f"unexpected command: {command}")

    return fake, calls


def test_req_report_3247_spec_anchor_exists() -> None:
    """REQ-REPORT-3247: OpenSpec declares the root-cause surgery contract."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3247" in spec
    assert "SCENARIO-REPORT-3247" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3247_blocks_exp3248_on_runtime_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3247: failed post-repair probes keep the next smoke blocked."""

    _write_exp3236(tmp_path)
    runner, calls = _runner()

    artifact = mod.build_artifact(
        tmp_path,
        selected_python=SELECTED_PYTHON,
        env={
            "PATH": "/usr/bin:/opt/cuda/bin",
            "CUDA_VISIBLE_DEVICES": "",
            "LD_LIBRARY_PATH": "",
            "VIRTUAL_ENV": "/repo/.venv",
        },
        command_runner=runner,
        monotonic=iter([10.0, 12.0]).__next__,
        tests_run=["REQ-REPORT-3247 focused"],
        device_nodes_world_accessible=True,
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3247"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.301"
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["preconditions_checked"] is True
    assert artifact["cuda_root_cause_class"] == "cuda_bindings_runtime_failure"
    assert artifact["selected_python_path"] == SELECTED_PYTHON
    assert artifact["selected_python_version"] == "3.14.4"
    assert artifact["torch_version_before"] == "2.11.0+cu128"
    assert artifact["torch_cuda_build_before"] == "12.8"
    assert artifact["cuda_bindings_import_ok_before"] is True
    assert artifact["environment_snapshot"]["CUDA_VISIBLE_DEVICES"] == ""
    assert artifact["environment_snapshot"]["LD_LIBRARY_PATH"] == ""
    assert artifact["environment_snapshot"]["active_virtual_environment"] == "/repo/.venv"
    assert artifact["selected_python_torch_cuda_available_before"] is False
    assert artifact["cuda_bindings_device_count_before"] == 0
    assert artifact["selected_python_torch_cuda_available_after"] is False
    assert artifact["cuda_bindings_device_count_after"] == 0
    assert artifact["selected_python_cuda_repaired_candidate"] is False
    assert artifact["next_smoke_allowed"] is False
    assert artifact["recommended_next_task"] == "keep_exp3248_blocked_repair_cuda_runtime"
    assert artifact["repair_actions_attempted"] == [
        {
            "action": "subprocess_only_normalize_cuda_visible_devices",
            "scope": "selected_project_environment",
            "safe": True,
            "persistent_changes": False,
            "destructive_package_operation": False,
            "cuda_visible_devices_after": "0",
            "result": "failed",
        }
    ]
    assert artifact["protected_files_untouched"] == {"scripts/research_conductor.py": True}
    assert artifact["exp3236_comparison"]["selected_python_same_resolved"] is True
    assert artifact["tests_run"] == ["REQ-REPORT-3247 focused"]
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "next_smoke_allowed=false" in artifact["honest_verdict"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert all(
        "returncode" in command and "stdout_excerpt" in command and "stderr_excerpt" in command
        for command in artifact["commands_run"]
    )
    assert {
        call["kwargs"]["env"]["CARNOT_EXP3247_PROBE_STAGE"]
        for call in calls
        if call["command"][0] == SELECTED_PYTHON
    } == {"before", "after"}


def test_scenario_report_3247_opens_exp3248_after_safe_subprocess_repair(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3247: post-repair torch and bindings probes gate Exp 3248."""

    _write_exp3236(tmp_path)
    runner, _calls = _runner(
        before_torch_ok=False,
        before_torch_count=0,
        before_bindings_ok=False,
        before_bindings_count=0,
        after_torch_ok=True,
        after_torch_count=1,
        after_bindings_ok=True,
        after_bindings_count=1,
    )
    artifact = mod.write_artifact(
        tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        selected_python=SELECTED_PYTHON,
        env={"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": ""},
        command_runner=runner,
        monotonic=iter([1.0, 1.5]).__next__,
        device_nodes_world_accessible=True,
    )
    saved = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert artifact["cuda_root_cause_class"] == "permission/device_visibility_failure"
    assert artifact["selected_python_torch_cuda_available_before"] is False
    assert artifact["cuda_bindings_device_count_before"] == 0
    assert artifact["selected_python_torch_cuda_available_after"] is True
    assert artifact["cuda_bindings_device_count_after"] == 1
    assert artifact["selected_python_cuda_repaired_candidate"] is True
    assert artifact["next_smoke_allowed"] is True
    assert artifact["recommended_next_task"] == "exp3248-selected-python-cuda-smoke-rerun"
    assert artifact["repair_actions_attempted"][0]["result"] == "candidate_repaired"
    assert "next_smoke_allowed=true" in artifact["honest_verdict"]


def test_req_report_3247_classification_helpers() -> None:
    """REQ-REPORT-3247: root-cause classes stay inside the allowed set."""

    prior = {
        "selected_python": "/repo/.venv/bin/python",
        "torch_version": "2.11.0+cu128",
        "selected_python_torch_cuda_version": "12.8",
    }
    baseline = {
        "driver_visible": True,
        "selected_python_path": "/repo/.venv/bin/python",
        "torch_import_ok": True,
        "torch_version": "2.11.0+cu128",
        "torch_cuda_build": "12.8",
        "torch_cuda_available": False,
        "cuda_bindings_import_ok": True,
        "cuda_bindings_runtime_ok": False,
        "cuda_bindings_device_count": 0,
        "cuda_bindings_error": "999",
        "device_nodes_world_accessible": True,
    }

    assert mod.classify_root_cause({**baseline, "driver_visible": False}, prior) == "driver_absent"
    assert (
        mod.classify_root_cause(
            {**baseline, "selected_python_path": "/tmp/throwaway/bin/python"},
            prior,
        )
        == "selected_python_env_mismatch"
    )
    assert mod.classify_root_cause({**baseline, "torch_import_ok": False}, prior) == (
        "selected_python_env_mismatch"
    )
    assert mod.classify_root_cause({**baseline, "torch_cuda_build": None}, prior) == (
        "torch_cuda_build_mismatch"
    )
    assert mod.classify_root_cause({**baseline, "device_nodes_world_accessible": False}, prior) == (
        "permission/device_visibility_failure"
    )
    assert mod.classify_root_cause(baseline, prior) == "cuda_bindings_runtime_failure"
    assert (
        mod.classify_root_cause(
            {
                **baseline,
                "torch_cuda_available": True,
                "cuda_bindings_runtime_ok": True,
                "cuda_bindings_device_count": 1,
            },
            prior,
        )
        == "unresolved"
    )


def test_req_report_3247_parsing_and_defaults(tmp_path: Path) -> None:
    """REQ-REPORT-3247: malformed evidence is preserved without crashing."""

    assert mod._repo_root().name in {"carnot", "carnot-ebm"}
    command = mod._run_command(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin"},
    )
    assert command["returncode"] == 0
    assert command["stdout_summary"].strip() == "ok"
    assert mod._summarize("abc", limit=2) == "bc"
    assert mod._json_from_last_line({"stdout": "not json\n{\"ok\": true}\n"}) == {"ok": True}
    assert mod._json_from_last_line({"stdout": "not json\n", "stderr": "boom"}) == {
        "error": "boom"
    }
    assert mod._json_from_last_line({"stdout": "", "stderr": "boom"}) == {"error": "boom"}
    assert mod._parse_nvidia_smi_csv("bad\n0, uuid, name, 595.71.05, 10, 1, 9, 0, 40\n") == [
        {
            "index": 0,
            "uuid": "uuid",
            "name": "name",
            "driver_version": "595.71.05",
            "memory_total_mib": 10,
            "memory_used_mib": 1,
            "memory_free_mib": 9,
            "utilization_gpu_pct": 0,
            "temperature_gpu_c": 40,
        }
    ]
    assert mod._cuda_version_from_nvidia_smi("CUDA Version: 13.2") == "13.2"
    assert mod._cuda_version_from_nvidia_smi("no version") is None
    assert mod._read_json_object(tmp_path / "missing.json") == {}
    checksum_path = tmp_path / "checksum.txt"
    checksum_path.write_text("checksum", encoding="utf-8")
    assert mod._sha256_file(checksum_path)
    readable = tmp_path / "nvidia0"
    readable.write_text("", encoding="utf-8")
    readable.chmod(0o666)
    assert mod._device_nodes_world_accessible([readable]) is True
    unreadable = tmp_path / "nvidiactl"
    unreadable.write_text("", encoding="utf-8")
    unreadable.chmod(0o000)
    try:
        assert mod._device_nodes_world_accessible([unreadable]) is False
    finally:
        unreadable.chmod(0o666)
    read_only = tmp_path / "nvidia-ro"
    read_only.write_text("", encoding="utf-8")
    read_only.chmod(0o444)
    try:
        assert mod._device_nodes_world_accessible([read_only]) is False
    finally:
        read_only.chmod(0o666)
    assert mod._device_nodes_world_accessible([tmp_path / "missing-nvidia"]) is False
    assert mod._selected_python(tmp_path / "missing") != ""
