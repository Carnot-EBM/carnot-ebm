"""Tests for Exp 3261 CUDA recovery confirmation smoke.

Spec refs: REQ-REPORT-3261, SCENARIO-REPORT-3261.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cuda_recovery_confirmation_smoke_3261 as mod


SELECTED_PYTHON = "/repo/.venv/bin/python"

REQUIRED_FIELDS = {
    "cuda_recovery_confirmation_smoke_v1_ready",
    "next_smoke_allowed",
    "cuda_python_smoke_passed",
    "gpu_count",
    "gpu_names",
    "driver_version",
    "matmul_verified",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
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


def _nvidia_stdout(*, names: list[str] | None = None) -> str:
    gpu_names = names or ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090"]
    return "".join(f"{index}, {name}, 595.71.05\n" for index, name in enumerate(gpu_names))


def _matmul_stdout(
    *,
    devices: list[int] | None = None,
    verified: bool = True,
    seed: int = 3261,
) -> str:
    requested = devices or [0, 1]
    payload = {
        "probe": "exp3261_cuda_matmul_probe",
        "random_seed": seed,
        "device_results": [
            {
                "device": f"cuda:{device}",
                "device_index": device,
                "matmul_verified": verified,
                "max_abs_error": 0.0 if verified else 1.0,
                "result_checksum": f"device-{device}-checksum",
            }
            for device in requested
        ],
    }
    return json.dumps(payload, sort_keys=True) + "\n"


def _runner(
    *,
    nvidia_returncode: int = 0,
    nvidia_stdout: str | None = None,
    torch_returncode: int = 0,
    matmul_stdout: str | None = None,
    matmul_returncode: int = 0,
) -> tuple[mod.CommandRunner, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake(command: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append({"command": command, "kwargs": kwargs})
        joined = "\n".join(command)
        if command[:1] == ["nvidia-smi"]:
            return _command(
                command,
                returncode=nvidia_returncode,
                stdout=nvidia_stdout if nvidia_stdout is not None else _nvidia_stdout(),
                stderr="" if nvidia_returncode == 0 else "nvidia-smi failed",
            )
        if command[0] == SELECTED_PYTHON and "assert torch.cuda.is_available()" in joined:
            return _command(
                command,
                returncode=torch_returncode,
                stdout="",
                stderr="" if torch_returncode == 0 else "AssertionError\n",
            )
        if command[0] == SELECTED_PYTHON and "exp3261_cuda_matmul_probe" in joined:
            return _command(
                command,
                returncode=matmul_returncode,
                stdout=matmul_stdout if matmul_stdout is not None else _matmul_stdout(),
                stderr="" if matmul_returncode == 0 else "matmul failed",
            )
        raise AssertionError(f"unexpected command: {command}")

    return fake, calls


def test_req_report_3261_spec_anchor_exists() -> None:
    """REQ-REPORT-3261: OpenSpec declares the CUDA recovery smoke first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3261" in spec
    assert "SCENARIO-REPORT-3261" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "cuda_python_smoke_passed" in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3261_two_gpu_recovery_smoke_opens_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3261: verified matmul on both RTX 3090s opens the gate."""

    runner, calls = _runner()

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        selected_python=SELECTED_PYTHON,
        env={"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0,1"},
        command_runner=runner,
        monotonic=iter([10.0, 12.25]).__next__,
    )

    output = tmp_path / mod.OUTPUT_REL_PATH
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3261"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.302"
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["cuda_recovery_confirmation_smoke_v1_ready"] is True
    assert artifact["next_smoke_allowed"] is True
    assert artifact["cuda_python_smoke_passed"] is True
    assert artifact["gpu_count"] == 2
    assert artifact["gpu_names"] == ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090"]
    assert artifact["driver_version"] == "595.71.05"
    assert artifact["matmul_verified"] is True
    assert artifact["matmul_devices_tested"] == ["cuda:0", "cuda:1"]
    assert artifact["blocked_reason"] == ""
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "next_smoke_allowed=true" in artifact["honest_verdict"]
    assert calls[0]["command"] == mod.NVIDIA_SMI_QUERY
    assert "assert torch.cuda.is_available()" in "\n".join(calls[1]["command"])
    assert "exp3261_cuda_matmul_probe" in "\n".join(calls[2]["command"])
    assert calls[2]["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_scenario_report_3261_blocks_without_rtx_3090(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3261: missing RTX 3090 leaves downstream smokes gated."""

    runner, calls = _runner(nvidia_stdout=_nvidia_stdout(names=["NVIDIA A10"]))

    artifact = mod.build_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([1.0, 1.5]).__next__,
    )

    assert artifact["cuda_recovery_confirmation_smoke_v1_ready"] is False
    assert artifact["next_smoke_allowed"] is False
    assert artifact["cuda_python_smoke_passed"] is False
    assert artifact["gpu_count"] == 1
    assert artifact["gpu_names"] == ["NVIDIA A10"]
    assert artifact["blocked_reason"] == "blocked_no_gpu"
    assert artifact["matmul_verified"] is False
    assert "blocked_no_gpu" in artifact["honest_verdict"]
    assert len(calls) == 1


def test_scenario_report_3261_blocks_when_selected_python_cuda_is_unavailable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3261: torch CUDA precondition failure blocks matmul."""

    runner, calls = _runner(torch_returncode=1)

    artifact = mod.build_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([2.0, 3.0]).__next__,
    )

    assert artifact["cuda_recovery_confirmation_smoke_v1_ready"] is False
    assert artifact["next_smoke_allowed"] is False
    assert artifact["cuda_python_smoke_passed"] is False
    assert artifact["blocked_reason"] == "blocked_cuda_unavailable"
    assert artifact["matmul_verified"] is False
    assert "blocked_cuda_unavailable" in artifact["honest_verdict"]
    assert len(calls) == 2


def test_scenario_report_3261_blocks_when_matmul_probe_fails(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3261: CUDA availability alone is not enough."""

    runner, _calls = _runner(matmul_stdout=_matmul_stdout(devices=[0, 1], verified=False))

    artifact = mod.build_artifact(
        project_root=tmp_path,
        selected_python=SELECTED_PYTHON,
        command_runner=runner,
        monotonic=iter([4.0, 5.0]).__next__,
    )

    assert artifact["cuda_recovery_confirmation_smoke_v1_ready"] is False
    assert artifact["next_smoke_allowed"] is False
    assert artifact["cuda_python_smoke_passed"] is False
    assert artifact["blocked_reason"] == "matmul_verification_failed"
    assert artifact["matmul_verified"] is False
    assert artifact["matmul_devices_tested"] == ["cuda:0", "cuda:1"]


def test_helpers_cover_selected_python_command_runner_and_parsers(tmp_path: Path) -> None:
    """REQ-REPORT-3261: helper behavior remains deterministic and bounded."""

    assert mod._selected_python(tmp_path) == sys.executable
    candidate = tmp_path / ".venv" / "bin" / "python"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# python placeholder\n", encoding="utf-8")

    assert mod._selected_python(tmp_path) == str(candidate)
    assert mod._summarize("abcdef", limit=3) == "def"
    assert mod._json_from_last_line({"stdout": "noise\n{\"ok\": true}\n", "stderr": ""}) == {
        "ok": True
    }
    assert mod._json_from_last_line({"stdout": "noise\n", "stderr": "bad"}) == {"error": "bad"}
    assert mod._all_matmuls_verified({"device_results": []}, [0]) is False
    assert mod._matmul_devices({"device_results": "not-a-list"}) == []

    result = mod._run_command([sys.executable, "-c", "print('ok')"], timeout_s=10)
    assert result["returncode"] == 0
    assert result["stdout_summary"].strip() == "ok"
