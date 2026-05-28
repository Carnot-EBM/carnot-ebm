"""Build the Exp 3261 CUDA recovery confirmation smoke artifact.

Spec refs: REQ-REPORT-3261, SCENARIO-REPORT-3261.

This smoke is intentionally small: it proves the selected project Python can
initialize CUDA and execute a real GPU matmul after the operator reboot, then
opens the downstream llama.cpp receipt gate only when that operation verifies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.cuda_recovery_confirmation_smoke.v1"
EXPERIMENT_ID = "exp3261"
TASK_ID = "exp3261-cuda-recovery-confirmation-smoke-v1"
ARTIFACT = "experiment_3261_cuda_recovery_confirmation_smoke_v1"
MILESTONE = "2026.05.302"
INFERENCE_SUBSTRATE = "hardware_smoke"
RANDOM_SEED = 3261

OUTPUT_REL_PATH = Path("results/experiment_3261_cuda_recovery_confirmation_smoke_v1.json")
NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,driver_version",
    "--format=csv,noheader,nounits",
]


def _selected_python(project_root: str | Path) -> str:
    """Return the conductor virtualenv Python when the repo has one."""

    candidate = Path(project_root) / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str | None, *, limit: int = 4000) -> str:
    """Bound command output while keeping the failure tail visible."""

    value = text or ""
    return value if len(value) <= limit else value[-limit:]


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 60,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> JsonDict:
    """Run a subprocess probe and return JSON-serializable command evidence."""

    cmd = [str(part) for part in command]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
            cwd=str(cwd) if cwd is not None else None,
        )
        return {
            "command": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "stdout_summary": _summarize(completed.stdout),
            "stderr_summary": _summarize(completed.stderr),
        }
    except Exception as exc:  # pragma: no cover - defensive command evidence path.
        error = f"{type(exc).__name__}: {exc}"
        return {
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": error,
            "stdout_summary": "",
            "stderr_summary": error,
        }


def _stdout(result: Mapping[str, Any]) -> str:
    return str(result.get("stdout") or result.get("stdout_summary") or "")


def _stderr(result: Mapping[str, Any]) -> str:
    return str(result.get("stderr") or result.get("stderr_summary") or "")


def _json_from_last_line(result: Mapping[str, Any]) -> JsonDict:
    """Parse the final JSON line emitted by a selected-Python probe."""

    for line in reversed(_stdout(result).splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}
    return {"error": _stderr(result) or _stdout(result) or "json_probe_unparseable"}


def _parse_nvidia_smi_devices(stdout: str) -> list[JsonDict]:
    """Parse the nvidia-smi CSV device inventory used by this smoke."""

    devices: list[JsonDict] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver_version": parts[2],
                }
            )
    return devices


def _nvidia_inventory(*, command_runner: CommandRunner) -> JsonDict:
    """Record GPU names and driver version before any Python CUDA work."""

    result = command_runner(NVIDIA_SMI_QUERY, timeout_s=10)
    devices = (
        _parse_nvidia_smi_devices(_stdout(result)) if result.get("returncode") == 0 else []
    )
    return {
        "command": result.get("command", NVIDIA_SMI_QUERY),
        "returncode": result.get("returncode"),
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
        "devices": devices,
        "gpu_count": len(devices),
        "gpu_names": [str(device["name"]) for device in devices],
        "driver_version": str(devices[0]["driver_version"]) if devices else "",
        "rtx_3090_present": any("RTX 3090" in str(device["name"]) for device in devices),
    }


def _torch_cuda_precondition(
    selected_python: str,
    *,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """Run the exact selected-Python Torch CUDA availability precondition."""

    command = [selected_python, "-c", "import torch; assert torch.cuda.is_available()"]
    result = command_runner(command, timeout_s=60, env=env)
    return {
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
        "passed": result.get("returncode") == 0,
    }


def _required_device_indices(gpu_count: int) -> list[int]:
    """Test cuda:0, and cuda:1 when the host reports at least two GPUs."""

    return list(range(min(gpu_count, 2)))


def _matmul_probe_script(device_indices: Sequence[int], random_seed: int) -> str:
    """Return the selected-Python script that performs deterministic GPU matmuls."""

    return f"""
import hashlib
import json
import torch

device_indices = {list(device_indices)!r}
random_seed = {random_seed}
device_results = []
offset = random_seed % 17
base = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
left_cpu = base + float(offset)
right_cpu = torch.flip(base, dims=(0,)) + float(offset + 1)
expected_cpu = left_cpu @ right_cpu
for device_index in device_indices:
    device = f"cuda:{{device_index}}"
    left_gpu = left_cpu.to(device)
    right_gpu = right_cpu.to(device)
    result_cpu = (left_gpu @ right_gpu).cpu()
    max_abs_error = float((result_cpu - expected_cpu).abs().max().item())
    verified = bool(torch.allclose(result_cpu, expected_cpu, rtol=0.0, atol=0.0))
    checksum = hashlib.sha256(
        json.dumps(result_cpu.tolist(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    device_results.append(
        {{
            "device": device,
            "device_index": device_index,
            "matmul_verified": verified,
            "max_abs_error": max_abs_error,
            "result_checksum": checksum,
        }}
    )
print(
    json.dumps(
        {{
            "probe": "exp3261_cuda_matmul_probe",
            "random_seed": random_seed,
            "device_results": device_results,
        }},
        sort_keys=True,
    )
)
"""


def _matmul_probe(
    selected_python: str,
    *,
    device_indices: Sequence[int],
    random_seed: int,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """Run the selected-Python CUDA matmul probe and attach command evidence."""

    script = _matmul_probe_script(device_indices, random_seed)
    command = [selected_python, "-c", script]
    result = command_runner(command, timeout_s=120, env=env)
    payload = _json_from_last_line(result)
    payload["command"] = result.get("command", command)
    payload["returncode"] = result.get("returncode")
    payload["stderr_summary"] = _summarize(_stderr(result))
    return payload


def _all_matmuls_verified(payload: Mapping[str, Any], expected_devices: Sequence[int]) -> bool:
    results = payload.get("device_results")
    if not isinstance(results, list) or len(results) != len(expected_devices):
        return False
    return all(bool(row.get("matmul_verified")) for row in results if isinstance(row, Mapping))


def _matmul_devices(payload: Mapping[str, Any]) -> list[str]:
    results = payload.get("device_results")
    if not isinstance(results, list):
        return []
    return [str(row.get("device")) for row in results if isinstance(row, Mapping)]


def _reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _principle_annotations() -> JsonDict:
    return {
        "next_smoke_allowed": "Gates the downstream llama.cpp receipt smoke.",
        "cuda_python_smoke_passed": "Requires a real selected-Python CUDA matmul, not only is_available().",
        "matmul_verified": "Requires copied-back GPU results to match the deterministic CPU expectation.",
    }


def _honest_verdict(*, passed: bool, blocked_reason: str) -> str:
    if passed:
        return (
            "complete: cuda_recovery_confirmation_smoke_v1_ready=true; "
            "cuda_python_smoke_passed=true; next_smoke_allowed=true"
        )
    return (
        "complete: cuda_recovery_confirmation_smoke_v1_ready=false; "
        f"blocked_reason={blocked_reason}; cuda_python_smoke_passed=false; "
        "next_smoke_allowed=false"
    )


def build_artifact(
    *,
    project_root: str | Path,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """REQ-REPORT-3261: build the CUDA recovery confirmation smoke artifact."""

    start = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)

    nvidia = _nvidia_inventory(command_runner=command_runner)
    gpu_count = int(nvidia["gpu_count"])
    gpu_names = list(nvidia["gpu_names"])
    driver_version = str(nvidia["driver_version"])
    blocked_reason = ""
    torch_precondition: JsonDict = {}
    matmul_probe: JsonDict = {}
    matmul_verified = False
    matmul_devices_tested: list[str] = []

    if not nvidia["rtx_3090_present"]:
        blocked_reason = "blocked_no_gpu"
    else:
        torch_precondition = _torch_cuda_precondition(
            selected,
            env=merged_env,
            command_runner=command_runner,
        )
        if not torch_precondition["passed"]:
            blocked_reason = "blocked_cuda_unavailable"
        else:
            device_indices = _required_device_indices(gpu_count)
            matmul_probe = _matmul_probe(
                selected,
                device_indices=device_indices,
                random_seed=random_seed,
                env=merged_env,
                command_runner=command_runner,
            )
            matmul_verified = _all_matmuls_verified(matmul_probe, device_indices)
            matmul_devices_tested = _matmul_devices(matmul_probe)
            if not matmul_verified:
                blocked_reason = "matmul_verification_failed"

    passed = blocked_reason == "" and matmul_verified
    checksum = _reproducibility_checksum(
        {
            "blocked_reason": blocked_reason,
            "driver_version": driver_version,
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
            "matmul_devices_tested": matmul_devices_tested,
            "matmul_probe_device_results": matmul_probe.get("device_results", []),
            "matmul_verified": matmul_verified,
            "random_seed": random_seed,
            "selected_python": selected,
        }
    )
    duration_s = round(monotonic() - start, 6)

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle_annotations": _principle_annotations(),
        "selected_python": selected,
        "cuda_recovery_confirmation_smoke_v1_ready": passed,
        "next_smoke_allowed": passed,
        "cuda_python_smoke_passed": passed,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "driver_version": driver_version,
        "nvidia_smi": nvidia,
        "torch_precondition": torch_precondition,
        "matmul_probe": matmul_probe,
        "matmul_devices_tested": matmul_devices_tested,
        "matmul_verified": matmul_verified,
        "blocked_reason": blocked_reason,
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(passed=passed, blocked_reason=blocked_reason),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    project_root: str | Path,
    output_path: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: ClockFn = time.perf_counter,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Build and write the Exp 3261 CUDA recovery confirmation smoke artifact."""

    root = Path(project_root)
    destination = Path(output_path) if output_path is not None else root / OUTPUT_REL_PATH
    if not destination.is_absolute():
        destination = root / destination
    artifact = build_artifact(
        project_root=root,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        monotonic=monotonic,
        random_seed=random_seed,
    )
    _write_json(destination, artifact)
    return artifact
