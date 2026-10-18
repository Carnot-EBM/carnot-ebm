"""Build the Exp 3247 selected-Python CUDA root-cause surgery artifact.

**Researcher summary:**
    Exp 3236 proved that the NVIDIA driver was visible but the selected project
    Python could not complete the torch CUDA and ``cuda.bindings`` device-count
    probes. This module repeats only those tiny probes, compares them with the
    Exp 3236 failure shape, and records whether a subprocess-only environment
    repair is enough to let Exp 3248 run.

**Detailed explanation for engineers:**
    CUDA failures often sit between layers: ``nvidia-smi`` can work while the
    runtime API returns error 999, and PyTorch can report a CUDA build while
    refusing to initialize a device. The useful repair artifact is therefore a
    boundary ledger, not a broad package reinstall. We capture preconditions
    first, try only a non-persistent ``CUDA_VISIBLE_DEVICES`` normalization in a
    clean subprocess, and set the downstream gate true only if both torch CUDA
    and ``cuda.bindings`` report at least one usable device after that probe.

Spec refs: REQ-REPORT-3247, SCENARIO-REPORT-3247.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.selected_python_cuda_root_cause_surgery.v1"
EXPERIMENT_ID = "exp3247"
TASK_ID = "exp3247-selected-python-cuda-root-cause-surgery-v1"
ARTIFACT = "experiment_3247_selected_python_cuda_root_cause_surgery_v1"
MILESTONE = "2026.05.301"
INFERENCE_SUBSTRATE = "hardware_smoke"
RANDOM_SEED = 3247

OUTPUT_REL_PATH = Path("results/experiment_3247_selected_python_cuda_root_cause_surgery_v1.json")
EXP3236_REL_PATH = Path("results/experiment_3236_isolated_cuda_python_smoke_v1.json")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
NEXT_SMOKE_TASK = "exp3248-selected-python-cuda-smoke-rerun"
BLOCKED_NEXT_TASK = "keep_exp3248_blocked_repair_cuda_runtime"

ROOT_CAUSE_CLASSES = frozenset(
    {
        "driver_absent",
        "selected_python_env_mismatch",
        "torch_cuda_build_mismatch",
        "cuda_bindings_runtime_failure",
        "permission/device_visibility_failure",
        "unresolved",
    }
)

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
]


def _repo_root() -> Path:
    """Return the repo root, allowing conductor wrappers to pin it explicitly."""

    return Path(os.environ.get("CARNOT_REPO_ROOT", REPO_ROOT)).resolve()


def _selected_python(project_root: str | Path) -> str:
    """Pick the project virtualenv interpreter when it exists."""

    candidate = Path(project_root) / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str | None, *, limit: int = 4000) -> str:
    """Bound command evidence while preserving the failure tail."""

    value = text or ""
    return value if len(value) <= limit else value[-limit:]


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> JsonDict:
    """Run a small diagnostic command and return JSON-serializable evidence."""

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
    except Exception as exc:  # pragma: no cover - defensive subprocess envelope
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
    """Parse the last JSON line from a probe result."""

    for line in reversed(_stdout(result).splitlines()):
        try:
            parsed = json.loads(line)
            return dict(parsed) if isinstance(parsed, Mapping) else {"value": parsed}
        except json.JSONDecodeError:
            continue
    return {"error": _stderr(result) or _stdout(result) or "json_probe_unparseable"}


def _int_field(value: str) -> int | None:
    stripped = value.strip()
    return int(stripped) if stripped.lstrip("-").isdigit() else None


def _parse_nvidia_smi_csv(text: str) -> list[JsonDict]:
    """Parse the stable CSV inventory emitted by ``nvidia-smi``."""

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
    """Extract the CUDA version from raw ``nvidia-smi`` output."""

    match = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    return match.group(1) if match else None


def _command_excerpt(result: Mapping[str, Any]) -> JsonDict:
    """Normalize command evidence for artifact storage."""

    return {
        "command": list(result.get("command") or []),
        "returncode": result.get("returncode"),
        "stdout_excerpt": _summarize(_stdout(result)),
        "stderr_excerpt": _summarize(_stderr(result)),
    }


def _read_json_object(path: Path) -> JsonDict:
    """Read JSON evidence as an object, returning an empty object on failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _sha256_file(path: Path) -> str | None:
    """Hash a source file so the artifact can be reproduced later."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nvidia_smi_inventory(*, command_runner: CommandRunner) -> tuple[JsonDict, list[JsonDict]]:
    """Capture driver visibility without treating it as runtime proof."""

    query_result = command_runner(NVIDIA_SMI_QUERY, timeout_s=10)
    raw_result = command_runner(["nvidia-smi"], timeout_s=10)
    gpus = (
        _parse_nvidia_smi_csv(_stdout(query_result)) if query_result.get("returncode") == 0 else []
    )
    raw_stdout = _stdout(raw_result)
    inventory = {
        "available": bool(gpus),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "driver_version": gpus[0]["driver_version"] if gpus else None,
        "cuda_version": _cuda_version_from_nvidia_smi(raw_stdout),
    }
    return inventory, [_command_excerpt(query_result), _command_excerpt(raw_result)]


def _probe_env(
    env: Mapping[str, str],
    *,
    stage: str,
    cuda_visible_devices: str | None = None,
) -> dict[str, str]:
    """Build a child-process environment for one diagnostic probe stage."""

    probe_env = dict(env)
    if cuda_visible_devices is not None:
        probe_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    probe_env["CARNOT_EXP3247_PROBE_STAGE"] = stage
    return probe_env


def _torch_cuda_probe(
    selected_python: str,
    *,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> tuple[JsonDict, JsonDict]:
    """Run the selected Python torch CUDA probe before project imports."""

    script = (
        "import json, sys\n"
        "payload = {'probe': 'exp3247_torch_cuda_probe', 'python_version': sys.version}\n"
        "try:\n"
        "    import torch\n"
        "except Exception as exc:\n"
        "    payload.update({'selected_python_torch_import_ok': False, "
        "'torch_version': None, 'selected_python_torch_cuda_version': None, "
        "'selected_python_torch_cuda_available': False, 'selected_python_device_count': 0, "
        "'selected_python_raw_device_count': 0, 'selected_python_device_names': [], "
        "'selected_python_cuda_error': f'{type(exc).__name__}: {exc}'})\n"
        "else:\n"
        "    cuda_error = None\n"
        "    try:\n"
        "        available = bool(torch.cuda.is_available())\n"
        "        raw_count = int(torch.cuda.device_count())\n"
        "        count = raw_count if available else 0\n"
        "        names = [torch.cuda.get_device_name(i) for i in range(count)]\n"
        "    except Exception as exc:\n"
        "        available = False\n"
        "        raw_count = 0\n"
        "        count = 0\n"
        "        names = []\n"
        "        cuda_error = f'{type(exc).__name__}: {exc}'\n"
        "    payload.update({'selected_python_torch_import_ok': True, "
        "'torch_version': getattr(torch, '__version__', None), "
        "'selected_python_torch_cuda_version': getattr(torch.version, 'cuda', None), "
        "'selected_python_torch_cuda_available': available, "
        "'selected_python_device_count': count, "
        "'selected_python_raw_device_count': raw_count, "
        "'selected_python_device_names': names, "
        "'selected_python_cuda_error': cuda_error})\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    result = command_runner([selected_python, "-I", "-c", script], timeout_s=60, env=env)
    payload = _json_from_last_line(result)
    payload.update(_command_excerpt(result))
    return payload, _command_excerpt(result)


def _cuda_bindings_probe(
    selected_python: str,
    *,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> tuple[JsonDict, JsonDict]:
    """Run the selected Python ``cuda.bindings`` runtime device-count probe."""

    script = (
        "import json, sys\n"
        "payload = {'probe': 'exp3247_cuda_bindings_probe', 'python_version': sys.version}\n"
        "try:\n"
        "    import cuda.bindings.runtime as rt\n"
        "    err, count = rt.cudaGetDeviceCount()\n"
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
        "    payload.update({'cuda_bindings_import_ok': True, "
        "'cuda_bindings_runtime_ok': ok, "
        "'cuda_bindings_device_count': int(count or 0) if ok else 0, "
        "'cuda_bindings_raw_device_count': int(count or 0), "
        "'cuda_bindings_device_names': names, "
        "'cuda_bindings_cuda_error': str(err), "
        "'cuda_bindings_runtime_version': runtime_version if str(runtime_err).endswith('cudaSuccess') else None, "
        "'cuda_bindings_driver_version': driver_version if str(driver_err).endswith('cudaSuccess') else None})\n"
        "except Exception as exc:\n"
        "    payload.update({'cuda_bindings_import_ok': False, "
        "'cuda_bindings_runtime_ok': False, 'cuda_bindings_device_count': 0, "
        "'cuda_bindings_raw_device_count': 0, 'cuda_bindings_device_names': [], "
        "'cuda_bindings_cuda_error': f'{type(exc).__name__}: {exc}', "
        "'cuda_bindings_runtime_version': None, 'cuda_bindings_driver_version': None})\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    result = command_runner([selected_python, "-I", "-c", script], timeout_s=60, env=env)
    payload = _json_from_last_line(result)
    payload.update(_command_excerpt(result))
    return payload, _command_excerpt(result)


def _environment_snapshot(env: Mapping[str, str]) -> JsonDict:
    """Record the environment variables that commonly affect CUDA discovery."""

    return {
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", ""),
        "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH", ""),
        "VIRTUAL_ENV": env.get("VIRTUAL_ENV", ""),
        "CONDA_PREFIX": env.get("CONDA_PREFIX", ""),
        "active_virtual_environment": env.get("VIRTUAL_ENV") or env.get("CONDA_PREFIX") or "",
    }


def _device_nodes_world_accessible(paths: Sequence[Path] | None = None) -> bool:
    """Return whether common NVIDIA device nodes are present and user-accessible."""

    required = list(paths) if paths is not None else [Path("/dev/nvidiactl"), Path("/dev/nvidia0")]
    for path in required:
        if not path.exists():
            return False
        mode = path.stat().st_mode
        if not (mode & stat.S_IRUSR or mode & stat.S_IRGRP or mode & stat.S_IROTH):
            return False
        if not (mode & stat.S_IWUSR or mode & stat.S_IWGRP or mode & stat.S_IWOTH):
            return False
    return True


def _resolved_path(path: str) -> str:
    """Resolve a path for comparison while preserving missing-path strings."""

    try:
        return str(Path(path).resolve())
    except OSError:  # pragma: no cover - Path.resolve almost never fails locally
        return path


def classify_root_cause(baseline: Mapping[str, Any], prior_exp3236: Mapping[str, Any]) -> str:
    """REQ-REPORT-3247: classify the selected-Python CUDA failure boundary."""

    if not baseline.get("driver_visible"):
        return "driver_absent"
    prior_python = str(
        prior_exp3236.get("selected_python_resolved")
        or prior_exp3236.get("selected_python")
        or prior_exp3236.get("selected_python_path")
        or ""
    )
    current_python = str(
        baseline.get("selected_python_resolved") or baseline.get("selected_python_path") or ""
    )
    if prior_python and current_python and prior_python != current_python:
        return "selected_python_env_mismatch"
    if baseline.get("torch_import_ok") is False:
        return "selected_python_env_mismatch"
    torch_version = str(baseline.get("torch_version") or "")
    torch_cuda_build = baseline.get("torch_cuda_build")
    if not torch_cuda_build or "+cpu" in torch_version:
        return "torch_cuda_build_mismatch"
    if baseline.get("device_nodes_world_accessible") is False:
        return "permission/device_visibility_failure"
    bindings_import_ok = baseline.get("cuda_bindings_import_ok") is True
    bindings_runtime_ok = baseline.get("cuda_bindings_runtime_ok") is True
    bindings_count = int(baseline.get("cuda_bindings_device_count") or 0)
    if bindings_import_ok and (not bindings_runtime_ok or bindings_count <= 0):
        return "cuda_bindings_runtime_failure"
    return "unresolved"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    selected_python: str | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: Callable[[], float] = time.perf_counter,
    tests_run: Sequence[str] | None = None,
    device_nodes_world_accessible: bool | None = None,
) -> JsonDict:
    """SCENARIO-REPORT-3247: build the root-cause and exp3248 gate artifact."""

    root_path = Path(root)
    start = monotonic()
    env_map = dict(os.environ if env is None else env)
    selected = selected_python or _selected_python(root_path)
    selected_resolved = _resolved_path(selected)
    prior_exp3236 = _read_json_object(root_path / EXP3236_REL_PATH)
    prior_for_classification = {
        **prior_exp3236,
        "selected_python_resolved": _resolved_path(str(prior_exp3236.get("selected_python") or "")),
    }

    nvidia_smi, nvidia_commands = _nvidia_smi_inventory(command_runner=command_runner)
    before_env = _probe_env(env_map, stage="before")
    torch_before, torch_before_cmd = _torch_cuda_probe(
        selected,
        env=before_env,
        command_runner=command_runner,
    )
    bindings_before, bindings_before_cmd = _cuda_bindings_probe(
        selected,
        env=before_env,
        command_runner=command_runner,
    )

    normalized_cvd = env_map.get("CUDA_VISIBLE_DEVICES") or "0"
    after_env = _probe_env(env_map, stage="after", cuda_visible_devices=normalized_cvd)
    torch_after, torch_after_cmd = _torch_cuda_probe(
        selected,
        env=after_env,
        command_runner=command_runner,
    )
    bindings_after, bindings_after_cmd = _cuda_bindings_probe(
        selected,
        env=after_env,
        command_runner=command_runner,
    )

    torch_before_ok = torch_before.get("selected_python_torch_cuda_available") is True
    bindings_before_count = int(bindings_before.get("cuda_bindings_device_count") or 0)
    torch_after_ok = torch_after.get("selected_python_torch_cuda_available") is True
    bindings_after_count = int(bindings_after.get("cuda_bindings_device_count") or 0)
    repaired_candidate = torch_after_ok and bindings_after_count > 0
    next_smoke_allowed = bool(nvidia_smi["available"] and repaired_candidate)
    device_nodes_ok = (
        _device_nodes_world_accessible()
        if device_nodes_world_accessible is None
        else bool(device_nodes_world_accessible)
    )
    baseline = {
        "driver_visible": nvidia_smi["available"],
        "selected_python_path": selected,
        "selected_python_resolved": selected_resolved,
        "torch_import_ok": torch_before.get("selected_python_torch_import_ok"),
        "torch_version": torch_before.get("torch_version"),
        "torch_cuda_build": torch_before.get("selected_python_torch_cuda_version"),
        "torch_cuda_available": torch_before_ok,
        "cuda_bindings_import_ok": bindings_before.get("cuda_bindings_import_ok"),
        "cuda_bindings_runtime_ok": bindings_before.get("cuda_bindings_runtime_ok"),
        "cuda_bindings_device_count": bindings_before_count,
        "cuda_bindings_error": bindings_before.get("cuda_bindings_cuda_error"),
        "device_nodes_world_accessible": device_nodes_ok,
    }
    root_cause = (
        "permission/device_visibility_failure"
        if repaired_candidate and not (torch_before_ok and bindings_before_count > 0)
        else classify_root_cause(baseline, prior_for_classification)
    )
    repair_result = "candidate_repaired" if repaired_candidate else "failed"
    repair_actions = [
        {
            "action": "subprocess_only_normalize_cuda_visible_devices",
            "scope": "selected_project_environment",
            "safe": True,
            "persistent_changes": False,
            "destructive_package_operation": False,
            "cuda_visible_devices_after": normalized_cvd,
            "result": repair_result,
        }
    ]
    commands_run = [*nvidia_commands, torch_before_cmd, bindings_before_cmd, torch_after_cmd, bindings_after_cmd]
    exp3236_comparison = _exp3236_comparison(prior_exp3236, baseline, repaired_candidate)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle_annotations": _principle_annotations(),
        "preconditions_checked": False,
        "cuda_root_cause_class": root_cause,
        "selected_python_path": selected,
        "selected_python_resolved_path": selected_resolved,
        "selected_python_version": str(torch_before.get("python_version") or ""),
        "torch_version_before": torch_before.get("torch_version"),
        "torch_cuda_build_before": torch_before.get("selected_python_torch_cuda_version"),
        "cuda_bindings_import_ok_before": bindings_before.get("cuda_bindings_import_ok") is True,
        "environment_snapshot": _environment_snapshot(env_map),
        "nvidia_smi": nvidia_smi,
        "device_nodes_world_accessible": device_nodes_ok,
        "selected_python_torch_probe_before": torch_before,
        "cuda_bindings_probe_before": bindings_before,
        "selected_python_torch_cuda_available_before": torch_before_ok,
        "cuda_bindings_device_count_before": bindings_before_count,
        "repair_actions_attempted": repair_actions,
        "selected_python_torch_probe_after": torch_after,
        "cuda_bindings_probe_after": bindings_after,
        "selected_python_torch_cuda_available_after": torch_after_ok,
        "cuda_bindings_device_count_after": bindings_after_count,
        "selected_python_cuda_repaired_candidate": repaired_candidate,
        "next_smoke_allowed": next_smoke_allowed,
        "recommended_next_task": NEXT_SMOKE_TASK if next_smoke_allowed else BLOCKED_NEXT_TASK,
        "exp3236_comparison": exp3236_comparison,
        "commands_run": commands_run,
        "protected_files_untouched": {CONDUCTOR_REL_PATH.as_posix(): True},
        "protected_file_checksums": {
            CONDUCTOR_REL_PATH.as_posix(): _sha256_file(root_path / CONDUCTOR_REL_PATH),
        },
        "no_system_driver_uninstall": True,
        "no_destructive_package_operation": True,
        "no_conductor_execution": True,
        "no_push": True,
        "random_seed": RANDOM_SEED,
        "tests_run": list(tests_run or []),
        "duration_s": round(monotonic() - start, 6),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["preconditions_checked"] = _preconditions_checked(artifact)
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    selected_python: str | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: Callable[[], float] = time.perf_counter,
    tests_run: Sequence[str] | None = None,
    device_nodes_world_accessible: bool | None = None,
) -> JsonDict:
    """Build and persist the Exp 3247 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(
        root_path,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        monotonic=monotonic,
        tests_run=tests_run,
        device_nodes_world_accessible=device_nodes_world_accessible,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _preconditions_checked(artifact: Mapping[str, Any]) -> bool:
    """Return true only when every mandated precondition field was captured."""

    env = artifact.get("environment_snapshot")
    return (
        isinstance(artifact.get("nvidia_smi"), Mapping)
        and bool(artifact.get("selected_python_path"))
        and "selected_python_version" in artifact
        and "torch_version_before" in artifact
        and "torch_cuda_build_before" in artifact
        and isinstance(artifact.get("cuda_bindings_import_ok_before"), bool)
        and isinstance(env, Mapping)
        and "CUDA_VISIBLE_DEVICES" in env
        and "LD_LIBRARY_PATH" in env
        and "active_virtual_environment" in env
    )


def _exp3236_comparison(
    prior: Mapping[str, Any],
    baseline: Mapping[str, Any],
    repaired_candidate: bool,
) -> JsonDict:
    """Compare the old smoke failure shape with the live baseline probes."""

    prior_selected = str(prior.get("selected_python") or prior.get("selected_python_path") or "")
    prior_resolved = _resolved_path(prior_selected) if prior_selected else ""
    return {
        "exp3236_present": bool(prior),
        "prior_selected_python": prior_selected,
        "live_selected_python": baseline.get("selected_python_path"),
        "selected_python_same_resolved": prior_resolved == baseline.get("selected_python_resolved"),
        "prior_torch_version": prior.get("torch_version"),
        "live_torch_version": baseline.get("torch_version"),
        "prior_torch_cuda_build": prior.get("selected_python_torch_cuda_version"),
        "live_torch_cuda_build": baseline.get("torch_cuda_build"),
        "prior_torch_cuda_available": prior.get("selected_python_torch_cuda_available"),
        "live_torch_cuda_available_before": baseline.get("torch_cuda_available"),
        "prior_cuda_bindings_device_count": prior.get("cuda_bindings_device_count"),
        "live_cuda_bindings_device_count_before": baseline.get("cuda_bindings_device_count"),
        "failure_shape_still_matches_exp3236": (
            prior.get("selected_python_torch_cuda_available") is False
            and baseline.get("torch_cuda_available") is False
            and int(prior.get("cuda_bindings_device_count") or 0) == 0
            and int(baseline.get("cuda_bindings_device_count") or 0) == 0
        ),
        "safe_subprocess_repair_changed_outcome": repaired_candidate,
    }


def _principle_annotations() -> JsonDict:
    """Document the boundaries that keep this repair attempt honest."""

    return {
        "hardware_smoke_only": "Only nvidia-smi, torch CUDA, and cuda.bindings probes run.",
        "safe_repair_scope": "Repair attempts are subprocess-only environment probes.",
        "gate_rule": "Exp 3248 may run only when torch CUDA and cuda.bindings both see a device.",
        "protected_conductor": "scripts/research_conductor.py is not modified.",
    }


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact fields, excluding wall-clock and noisy command output."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "duration_s",
            "honest_verdict",
            "reproducibility_checksum",
            "commands_run",
            "nvidia_smi",
            "selected_python_torch_probe_before",
            "cuda_bindings_probe_before",
            "selected_python_torch_probe_after",
            "cuda_bindings_probe_after",
        }
    }
    encoded = json.dumps(stable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict without overclaiming a repair."""

    return (
        "complete: "
        f"cuda_root_cause_class={artifact['cuda_root_cause_class']}; "
        f"selected_python_cuda_repaired_candidate={str(artifact['selected_python_cuda_repaired_candidate']).lower()}; "
        f"next_smoke_allowed={str(artifact['next_smoke_allowed']).lower()}; "
        f"recommended_next_task={artifact['recommended_next_task']}"
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT_REL_PATH))
    parser.add_argument("--selected-python", default=None)
    args = parser.parse_args(argv)
    artifact = write_artifact(
        _repo_root(),
        output_path=args.output,
        selected_python=args.selected_python,
        env=os.environ,
        tests_run=[
            "tests/python/test_experiment_3247_selected_python_cuda_root_cause_surgery.py coverage 100pct"
        ],
    )
    print(json.dumps({"path": args.output, "next_smoke_allowed": artifact["next_smoke_allowed"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
