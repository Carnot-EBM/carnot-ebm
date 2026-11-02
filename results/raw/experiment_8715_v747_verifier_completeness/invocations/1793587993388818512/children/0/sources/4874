"""Run the Exp 3236 isolated CUDA Python smoke.

**Researcher summary:**
    This module checks only whether the local Python/CUDA boundary is alive. It
    deliberately stops before llama.cpp, GGUF loading, or any model inference so
    the result can be used as a small gate before more expensive receipt work.

**Detailed explanation for engineers:**
    A visible NVIDIA driver is not enough to prove that the selected project
    interpreter can initialize CUDA. PyTorch can fail inside a virtualenv while
    ``nvidia-smi`` still reports a healthy GPU. A direct ``cuda.bindings``
    runtime probe can also fail independently of PyTorch. Exp 3236 keeps those
    layers separate and records the exact boolean gate fields downstream tasks
    need before attempting llama.cpp offload receipts.

Spec refs: REQ-REPORT-3236, SCENARIO-REPORT-3236.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.isolated_cuda_python_smoke.v1"
EXPERIMENT_ID = "exp3236"
TASK_ID = "exp3236-isolated-cuda-python-smoke-v1"
ARTIFACT = "experiment_3236_isolated_cuda_python_smoke_v1"
MILESTONE = "2026.05.300"
INFERENCE_SUBSTRATE = "hardware_smoke"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_3236_isolated_cuda_python_smoke_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3236_isolated_cuda_python_smoke_v1.py"

NEXT_TASK = "exp3237-llama-cpp-cuda-receipt-smoke-v2"

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
]


def _repo_root() -> Path:
    """Return the repo root while honoring the experiment override used elsewhere."""

    return Path(os.environ.get("CARNOT_REPO_ROOT", REPO_ROOT)).resolve()


def _selected_python(project_root: str | Path) -> str:
    """Select the project virtualenv interpreter when the repository has one."""

    candidate = Path(project_root) / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str | None, *, limit: int = 4000) -> str:
    """Keep command evidence bounded while preserving the useful failure tail."""

    value = text or ""
    return value if len(value) <= limit else value[-limit:]


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> JsonDict:
    """Run a diagnostic command and return a JSON-serializable evidence packet."""

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
    except Exception as exc:
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
    """Parse the last JSON line from a subprocess probe result."""

    for line in reversed(_stdout(result).splitlines()):
        try:
            parsed = json.loads(line)
            return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            continue
    return {"error": _stderr(result) or _stdout(result) or "json_probe_unparseable"}


def _int_field(value: str) -> int | None:
    stripped = value.strip()
    return int(stripped) if stripped.lstrip("-").isdigit() else None


def _parse_nvidia_smi_csv(text: str) -> list[JsonDict]:
    """Parse the stable CSV inventory emitted by the nvidia-smi query."""

    rows: list[JsonDict] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 9 or _int_field(parts[0]) is None:
            continue
        rows.append(
            {
                "index": _int_field(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "driver_version": parts[3],
                "memory_total_mib": _int_field(parts[4]),
                "memory_used_mib": _int_field(parts[5]),
                "memory_free_mib": _int_field(parts[6]),
                "utilization_gpu_pct": _int_field(parts[7]),
                "temperature_gpu_c": _int_field(parts[8]),
            }
        )
    return rows


def _cuda_version_from_nvidia_smi(text: str) -> str | None:
    """Extract the CUDA version from raw nvidia-smi output when it is present."""

    match = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    return match.group(1) if match else None


def _nvidia_smi_inventory(*, command_runner: CommandRunner) -> JsonDict:
    """Capture driver visibility without treating it as sufficient CUDA proof."""

    query_result = command_runner(NVIDIA_SMI_QUERY, timeout_s=10)
    raw_result = command_runner(["nvidia-smi"], timeout_s=10)
    gpus = (
        _parse_nvidia_smi_csv(_stdout(query_result)) if query_result.get("returncode") == 0 else []
    )
    raw_stdout = _stdout(raw_result)
    return {
        "available": bool(gpus),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "driver_version": gpus[0]["driver_version"] if gpus else None,
        "cuda_version": _cuda_version_from_nvidia_smi(raw_stdout),
        "query_command": query_result.get("command", NVIDIA_SMI_QUERY),
        "query_returncode": query_result.get("returncode"),
        "query_stdout_summary": _summarize(_stdout(query_result)),
        "query_stderr_summary": _summarize(_stderr(query_result)),
        "raw_command": raw_result.get("command", ["nvidia-smi"]),
        "raw_returncode": raw_result.get("returncode"),
        "raw_stdout_summary": _summarize(raw_stdout),
        "raw_stderr_summary": _summarize(_stderr(raw_result)),
    }


def _probe_env(env: Mapping[str, str], *, probe_kind: str) -> dict[str, str]:
    """Build the child-process environment used by the clean CUDA probes."""

    probe_env = dict(env)
    if not probe_env.get("CUDA_VISIBLE_DEVICES"):
        probe_env["CUDA_VISIBLE_DEVICES"] = "0"
    probe_env["CARNOT_EXP3236_PROBE_KIND"] = probe_kind
    return probe_env


def _torch_cuda_probe(
    selected_python: str,
    *,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """Probe PyTorch CUDA in an isolated interpreter before project imports."""

    script = (
        "import json, sys\n"
        "order = ['json', 'sys']\n"
        "payload = {'probe': 'exp3236_torch_cuda_probe', "
        "'project_modules_preimport': [m for m in sys.modules if m == 'carnot' "
        "or m.startswith('carnot.') or m == 'scripts' or m.startswith('scripts.')]}\n"
        "try:\n"
        "    import torch\n"
        "    order.append('torch')\n"
        "except Exception as exc:\n"
        "    payload.update({'selected_python_torch_import_ok': False, "
        "'torch_version': None, 'selected_python_torch_cuda_version': None, "
        "'selected_python_torch_cuda_available': False, "
        "'selected_python_device_count': 0, 'selected_python_device_names': [], "
        "'import_order': order, 'error': f'{type(exc).__name__}: {exc}'})\n"
        "else:\n"
        "    cuda_error = None\n"
        "    try:\n"
        "        available = bool(torch.cuda.is_available())\n"
        "        order.append('torch.cuda.is_available')\n"
        "        count = int(torch.cuda.device_count()) if available else 0\n"
        "        order.append('torch.cuda.device_count')\n"
        "        names = [torch.cuda.get_device_name(i) for i in range(count)]\n"
        "        if count:\n"
        "            order.append('torch.cuda.get_device_name')\n"
        "    except Exception as exc:\n"
        "        available = False\n"
        "        count = 0\n"
        "        names = []\n"
        "        cuda_error = f'{type(exc).__name__}: {exc}'\n"
        "    payload.update({'selected_python_torch_import_ok': True, "
        "'torch_version': getattr(torch, '__version__', None), "
        "'selected_python_torch_cuda_version': getattr(torch.version, 'cuda', None), "
        "'selected_python_torch_cuda_available': available, "
        "'selected_python_device_count': count, "
        "'selected_python_device_names': names, 'selected_python_cuda_error': cuda_error, "
        "'import_order': order})\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    command = [selected_python, "-I", "-c", script]
    result = command_runner(
        command,
        timeout_s=60,
        env=_probe_env(env, probe_kind="torch"),
    )
    payload = _json_from_last_line(result)
    payload["command"] = result.get("command", command)
    payload["returncode"] = result.get("returncode")
    payload["stderr_summary"] = _summarize(_stderr(result))
    return payload


def _cuda_bindings_probe(
    selected_python: str,
    *,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """Probe the CUDA runtime directly through NVIDIA's Python bindings."""

    script = (
        "import json, sys\n"
        "order = ['json', 'sys']\n"
        "payload = {'probe': 'exp3236_cuda_bindings_probe', "
        "'project_modules_preimport': [m for m in sys.modules if m == 'carnot' "
        "or m.startswith('carnot.') or m == 'scripts' or m.startswith('scripts.')]}\n"
        "try:\n"
        "    import cuda.bindings.runtime as rt\n"
        "    order.append('cuda.bindings.runtime')\n"
        "    err, count = rt.cudaGetDeviceCount()\n"
        "    order.append('cudaGetDeviceCount')\n"
        "    runtime_err, runtime_version = rt.cudaRuntimeGetVersion()\n"
        "    driver_err, driver_version = rt.cudaDriverGetVersion()\n"
        "    ok = str(err).endswith('cudaSuccess') and bool(count and count > 0)\n"
        "    names = []\n"
        "    if ok:\n"
        "        for idx in range(int(count)):\n"
        "            prop_err, prop = rt.cudaGetDeviceProperties(idx)\n"
        "            if str(prop_err).endswith('cudaSuccess'):\n"
        "                raw_name = getattr(prop, 'name', b'')\n"
        "                name = raw_name.decode('utf-8', 'replace').rstrip('\\x00') if isinstance(raw_name, bytes) else str(raw_name)\n"
        "                names.append(name)\n"
        "        order.append('cudaGetDeviceProperties')\n"
        "    payload.update({'cuda_bindings_import_ok': True, "
        "'cuda_bindings_runtime_ok': ok, "
        "'cuda_bindings_device_count': int(count or 0) if ok else 0, "
        "'cuda_bindings_device_names': names, "
        "'cuda_bindings_cuda_error': str(err), "
        "'cuda_bindings_runtime_version': runtime_version if str(runtime_err).endswith('cudaSuccess') else None, "
        "'cuda_bindings_driver_version': driver_version if str(driver_err).endswith('cudaSuccess') else None, "
        "'import_order': order})\n"
        "except Exception as exc:\n"
        "    payload.update({'cuda_bindings_import_ok': False, "
        "'cuda_bindings_runtime_ok': False, 'cuda_bindings_device_count': 0, "
        "'cuda_bindings_device_names': [], 'cuda_bindings_cuda_error': None, "
        "'cuda_bindings_runtime_version': None, 'cuda_bindings_driver_version': None, "
        "'import_order': order, 'error': f'{type(exc).__name__}: {exc}'})\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    command = [selected_python, "-I", "-c", script]
    result = command_runner(
        command,
        timeout_s=60,
        env=_probe_env(env, probe_kind="cuda_bindings"),
    )
    payload = _json_from_last_line(result)
    payload["command"] = result.get("command", command)
    payload["returncode"] = result.get("returncode")
    payload["stderr_summary"] = _summarize(_stderr(result))
    return payload


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _smoke_block_reasons(
    *,
    driver_visible: bool,
    torch_import_ok: bool,
    torch_cuda_available: bool,
    torch_device_count: int,
    cuda_bindings_import_ok: bool,
    cuda_bindings_runtime_ok: bool,
    cuda_bindings_device_count: int,
) -> list[str]:
    reasons: list[str] = []
    if not driver_visible:
        reasons.append("cuda_driver_not_visible")
    if not torch_import_ok:
        reasons.append("selected_python_torch_import_failed")
    elif not (torch_cuda_available and torch_device_count > 0):
        reasons.append("selected_python_torch_cuda_unavailable")
    if not cuda_bindings_import_ok:
        reasons.append("cuda_bindings_unavailable")
    elif not (cuda_bindings_runtime_ok and cuda_bindings_device_count > 0):
        reasons.append("cuda_bindings_runtime_no_devices")
    return reasons


def _recommended_next_task(block_reasons: Sequence[str]) -> str:
    if not block_reasons:
        return NEXT_TASK
    if "cuda_driver_not_visible" in block_reasons:
        return "repair_nvidia_driver_visibility_before_exp3237"
    if "selected_python_torch_import_failed" in block_reasons:
        return "repair_selected_python_torch_import_before_exp3237"
    if "selected_python_torch_cuda_unavailable" in block_reasons:
        return "repair_selected_python_torch_cuda_before_exp3237"
    if "cuda_bindings_unavailable" in block_reasons:
        return "repair_cuda_bindings_runtime_probe_before_exp3237"
    if "cuda_bindings_runtime_no_devices" in block_reasons:
        return "repair_cuda_bindings_runtime_device_count_before_exp3237"
    return "inspect_cuda_python_smoke_block_before_exp3237"


def _principle_annotations() -> JsonDict:
    return {
        "probe_scope": "Driver visibility, selected Python torch CUDA, and cuda.bindings runtime are separate checks.",
        "no_model_inference": "This task does not load llama.cpp or any mandated GGUF model.",
        "gate_rule": "cuda_python_smoke_passed requires driver visibility, torch CUDA device count, and cuda.bindings device count.",
        "downstream_order": "Exp 3237 may run only after this artifact records cuda_python_smoke_passed=true.",
    }


def _honest_verdict(*, passed: bool, block_reasons: Sequence[str], next_task: str) -> str:
    if passed:
        return (
            "complete: cuda_python_smoke_passed=true; "
            f"recommended_next_task={next_task}; no_gguf_model_loaded=true"
        )
    joined = ",".join(block_reasons) if block_reasons else "unknown"
    return (
        "complete: cuda_python_smoke_passed=false; "
        f"blocked_by={joined}; recommended_next_task={next_task}; no_gguf_model_loaded=true"
    )


def build_artifact(
    *,
    project_root: str | Path,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: ClockFn = time.perf_counter,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-REPORT-3236: build the isolated CUDA Python smoke artifact."""

    start = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)

    nvidia = _nvidia_smi_inventory(command_runner=command_runner)
    torch_probe = _torch_cuda_probe(selected, env=merged_env, command_runner=command_runner)
    cuda_bindings_probe = _cuda_bindings_probe(
        selected,
        env=merged_env,
        command_runner=command_runner,
    )

    driver_visible = bool(nvidia["available"])
    torch_import_ok = bool(torch_probe.get("selected_python_torch_import_ok"))
    torch_cuda_available = bool(torch_probe.get("selected_python_torch_cuda_available"))
    torch_device_count = _as_int(torch_probe.get("selected_python_device_count"))
    cuda_bindings_import_ok = bool(cuda_bindings_probe.get("cuda_bindings_import_ok"))
    cuda_bindings_runtime_ok = bool(cuda_bindings_probe.get("cuda_bindings_runtime_ok"))
    cuda_bindings_device_count = _as_int(cuda_bindings_probe.get("cuda_bindings_device_count"))
    block_reasons = _smoke_block_reasons(
        driver_visible=driver_visible,
        torch_import_ok=torch_import_ok,
        torch_cuda_available=torch_cuda_available,
        torch_device_count=torch_device_count,
        cuda_bindings_import_ok=cuda_bindings_import_ok,
        cuda_bindings_runtime_ok=cuda_bindings_runtime_ok,
        cuda_bindings_device_count=cuda_bindings_device_count,
    )
    passed = not block_reasons
    next_task = _recommended_next_task(block_reasons)

    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle_annotations": _principle_annotations(),
        "selected_python": selected,
        "cuda_visible_devices": _probe_env(merged_env, probe_kind="artifact")[
            "CUDA_VISIBLE_DEVICES"
        ],
        "cuda_driver_visible": driver_visible,
        "nvidia_smi": nvidia,
        "selected_python_torch_import_ok": torch_import_ok,
        "selected_python_torch_cuda_available": torch_cuda_available,
        "selected_python_device_count": torch_device_count,
        "selected_python_device_names": list(
            torch_probe.get("selected_python_device_names") or []
        ),
        "selected_python_torch_cuda_version": torch_probe.get(
            "selected_python_torch_cuda_version"
        ),
        "torch_version": torch_probe.get("torch_version"),
        "selected_python_torch_probe": torch_probe,
        "cuda_bindings_import_ok": cuda_bindings_import_ok,
        "cuda_bindings_runtime_ok": cuda_bindings_runtime_ok,
        "cuda_bindings_device_count": cuda_bindings_device_count,
        "cuda_bindings_device_names": list(
            cuda_bindings_probe.get("cuda_bindings_device_names") or []
        ),
        "cuda_bindings_runtime_version": cuda_bindings_probe.get(
            "cuda_bindings_runtime_version"
        ),
        "cuda_bindings_driver_version": cuda_bindings_probe.get(
            "cuda_bindings_driver_version"
        ),
        "cuda_bindings_probe": cuda_bindings_probe,
        "cuda_python_smoke_passed": passed,
        "smoke_block_reasons": block_reasons,
        "recommended_next_task": next_task,
        "no_llama_cpp_rebuild": True,
        "no_full_gguf_load": True,
        "no_mandated_gguf_model_inference": True,
        "no_conductor_execution": True,
        "no_push": True,
        "tests_run": list(tests_run or []),
        "duration_s": round(monotonic() - start, 6),
        "honest_verdict": _honest_verdict(
            passed=passed,
            block_reasons=block_reasons,
            next_task=next_task,
        ),
    }
    return artifact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON for the conductor and downstream gates."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    selected_python: str | Path | None = None,
    tests_run: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: ClockFn = time.perf_counter,
) -> JsonDict:
    """Build and write the Exp 3236 isolated CUDA Python smoke artifact."""

    root = Path(project_root) if project_root is not None else _repo_root()
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    if not destination.is_absolute():
        destination = root / destination
    artifact = build_artifact(
        project_root=root,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        monotonic=monotonic,
        tests_run=tests_run,
    )
    _write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selected-python", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by the conductor-style experiment wrapper."""

    args = _parse_args(argv)
    run_experiment(
        project_root=None,
        output_path=args.output,
        selected_python=args.selected_python,
        tests_run=args.test_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
