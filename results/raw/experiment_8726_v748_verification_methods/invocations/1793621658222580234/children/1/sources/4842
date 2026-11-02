"""Exp 3220 hermetic CUDA runtime repair ledger.

**Researcher summary:**
    This ledger determines whether the current CUDA failure belongs to the
    selected repository virtualenv, the system NVIDIA driver/runtime boundary,
    import order, llama.cpp linkage, or polluted ROCm/XDNA/CUDA environment
    variables.  It is a runtime forensics artifact only; it does not load a
    GGUF model and it does not claim model receipt.

**Detailed explanation for engineers:**
    Prior artifacts showed a split-brain CUDA state: ``nvidia-smi`` could see an
    RTX 3090, but the selected ``.venv`` Python could not initialize PyTorch CUDA
    and llama.cpp printed a CUDA initialization error.  That failure can be
    caused by the Python package set, driver/runtime mismatch, import-order side
    effects, dynamic-library search paths, or a llama.cpp linkage issue.  This
    module keeps those layers separate by using isolated subprocess probes,
    explicit ``CUDA_VISIBLE_DEVICES=0``, a sanitized environment rerun, and an
    optional temporary CUDA-only venv outside the repository source tree.

Spec: REQ-INFER-SOTA-026,
      SCENARIO-INFER-SOTA-026-001,
      SCENARIO-INFER-SOTA-026-002,
      SCENARIO-INFER-SOTA-026-003
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
SCHEMA_VERSION = "carnot.hermetic_cuda_runtime_repair_ledger.v1"
EXPERIMENT_ID = "exp3220"
MILESTONE = "2026.05.298"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_3220_hermetic_cuda_runtime_repair_ledger_v1.json")
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / "experiment_3220_hermetic_cuda_runtime_repair_ledger_v1.py"
)
INFERENCE_SUBSTRATE = "cuda_runtime_forensics_no_model"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "experiment_id",
    "milestone",
    "selected_python",
    "selected_python_cuda_ok_before",
    "selected_python_cuda_ok_after",
    "isolated_cuda_venv_created",
    "isolated_cuda_venv_cuda_ok",
    "cuda_visible_devices",
    "nvidia_smi_available",
    "gpu_count_nvidia_smi",
    "driver_version",
    "torch_version_selected",
    "torch_cuda_version_selected",
    "environment_pollution_findings",
    "repair_actions_attempted",
    "cuda_receipt_ready_candidate",
    "recommended_next_action",
    "inference_substrate",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
)

TRACKED_ENV_KEYS = (
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "CMAKE_ARGS",
    "FORCE_CMAKE",
)
POLLUTED_PATH_TOKENS = ("rocm", "xilinx", "vitis", "vivado", "xrt", "ryzenai", "xdna")
NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
]


def _repo_root() -> Path:
    """Return the repository root, honoring the same override used by experiments."""
    return Path(os.environ.get("CARNOT_REPO_ROOT", Path.cwd())).resolve()


def _selected_python(project_root: str | Path) -> str:
    """Select the repository virtualenv interpreter when it exists."""
    candidate = Path(project_root) / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str | None, *, limit: int = 4000) -> str:
    """Keep subprocess evidence bounded while preserving useful failure tails."""
    value = text or ""
    return value if len(value) <= limit else value[-limit:]


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> JsonDict:
    """Run a diagnostic command and return an auditable result dictionary."""
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
    """Parse the last JSON line from a subprocess result."""
    for line in reversed(_stdout(result).splitlines()):
        try:
            parsed = json.loads(line)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            continue
    return {"error": _stderr(result) or _stdout(result) or "json_probe_unparseable"}


def _parse_pip_show(text: str) -> JsonDict:
    """Parse the stable key/value subset emitted by ``pip show``."""
    parsed: JsonDict = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def _pip_show(
    python: str,
    package: str,
    *,
    command_runner: CommandRunner,
) -> JsonDict:
    """Run ``pip show`` through the given interpreter."""
    command = [python, "-m", "pip", "show", package]
    result = command_runner(command, timeout_s=30)
    return {
        "package": package,
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "metadata": _parse_pip_show(_stdout(result)),
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
    }


def _int_field(value: str) -> int | None:
    stripped = value.strip()
    return int(stripped) if stripped.lstrip("-").isdigit() else None


def _parse_nvidia_smi_csv(text: str) -> list[JsonDict]:
    """Parse NVIDIA inventory rows from the query used by this ledger."""
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
    """Extract the CUDA version displayed by raw ``nvidia-smi`` output."""
    match = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    return match.group(1) if match else None


def _nvidia_smi_inventory(*, command_runner: CommandRunner) -> JsonDict:
    """Capture parseable and raw NVIDIA driver/GPU evidence."""
    query_result = command_runner(NVIDIA_SMI_QUERY, timeout_s=10)
    raw_result = command_runner(["nvidia-smi"], timeout_s=10)
    gpus = (
        _parse_nvidia_smi_csv(_stdout(query_result)) if query_result.get("returncode") == 0 else []
    )
    raw_stdout = _stdout(raw_result)
    return {
        "available": bool(gpus),
        "gpu_count": len(gpus) if gpus else None,
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


def _tracked_env(env: Mapping[str, str]) -> JsonDict:
    """Return only environment variables that influence CUDA/runtime discovery."""
    return {key: env.get(key) for key in TRACKED_ENV_KEYS}


def _path_has_token(value: str | None, tokens: Sequence[str]) -> bool:
    lowered = value.lower() if value else ""
    return any(token in lowered for token in tokens)


def _environment_pollution_findings(env: Mapping[str, str]) -> list[JsonDict]:
    """Classify ROCm, XDNA, Python path, and CMake/CUDA environment pollution."""
    findings: list[JsonDict] = []
    path = env.get("PATH")
    ld_path = env.get("LD_LIBRARY_PATH")
    checks = [
        ("path_contains_rocm", "PATH", _path_has_token(path, ("rocm",))),
        ("path_contains_xdna_tooling", "PATH", _path_has_token(path, POLLUTED_PATH_TOKENS[1:])),
        ("ld_library_path_contains_rocm", "LD_LIBRARY_PATH", _path_has_token(ld_path, ("rocm",))),
        (
            "ld_library_path_contains_xdna_tooling",
            "LD_LIBRARY_PATH",
            _path_has_token(ld_path, POLLUTED_PATH_TOKENS[1:]),
        ),
        ("pythonpath_set", "PYTHONPATH", bool(env.get("PYTHONPATH"))),
        ("cmake_args_set", "CMAKE_ARGS", bool(env.get("CMAKE_ARGS"))),
        ("force_cmake_set", "FORCE_CMAKE", bool(env.get("FORCE_CMAKE"))),
    ]
    for kind, variable, present in checks:
        if present:
            findings.append(
                {
                    "kind": kind,
                    "severity": "warn",
                    "variable": variable,
                    "detail": f"{variable} may influence CUDA runtime probing",
                }
            )
    if not env.get("CUDA_VISIBLE_DEVICES"):
        findings.append(
            {
                "kind": "cuda_visible_devices_missing_or_empty",
                "severity": "info",
                "variable": "CUDA_VISIBLE_DEVICES",
                "detail": "probe will force CUDA_VISIBLE_DEVICES=0 in subprocesses",
            }
        )
    return findings


def _remove_polluted_path_components(value: str | None) -> str | None:
    """Drop ROCm/XDNA/Vitis/Xilinx path entries for the sanitized probe."""
    if value is None:
        return None
    kept = [
        part
        for part in value.split(os.pathsep)
        if part and not _path_has_token(part, POLLUTED_PATH_TOKENS)
    ]
    return os.pathsep.join(kept)


def _sanitized_cuda_env(env: Mapping[str, str]) -> dict[str, str]:
    """Build a subprocess-only environment that removes common CUDA polluters."""
    sanitized = dict(env)
    path = _remove_polluted_path_components(env.get("PATH"))
    ld_path = _remove_polluted_path_components(env.get("LD_LIBRARY_PATH"))
    if path:
        sanitized["PATH"] = path
    if ld_path:
        sanitized["LD_LIBRARY_PATH"] = ld_path
    else:
        sanitized.pop("LD_LIBRARY_PATH", None)
    sanitized["CUDA_VISIBLE_DEVICES"] = "0"
    sanitized.pop("PYTHONPATH", None)
    sanitized.pop("CMAKE_ARGS", None)
    sanitized.pop("FORCE_CMAKE", None)
    return sanitized


def _probe_env(env: Mapping[str, str], *, stage: str) -> dict[str, str]:
    """Attach explicit device and stage metadata to a clean subprocess probe."""
    probe_env = dict(env)
    probe_env["CUDA_VISIBLE_DEVICES"] = "0"
    probe_env["CARNOT_EXP3220_PROBE_STAGE"] = stage
    return probe_env


def _torch_cuda_probe(
    selected_python: str,
    *,
    env: Mapping[str, str],
    command_runner: CommandRunner,
    stage: str,
) -> JsonDict:
    """Probe PyTorch CUDA without importing any project modules first."""
    script = (
        "import json, sys\n"
        "order = ['json', 'sys']\n"
        "payload = {'probe': 'exp3220_torch_cuda_probe', 'stage': "
        + repr(stage)
        + ", 'project_modules_preimport': [m for m in sys.modules if m == 'carnot' or m.startswith('carnot.') or m == 'scripts' or m.startswith('scripts.')]}\n"
        "try:\n"
        "    import torch\n"
        "    order.append('torch')\n"
        "    available = bool(torch.cuda.is_available())\n"
        "    order.append('torch.cuda.is_available')\n"
        "    count = int(torch.cuda.device_count()) if available else 0\n"
        "    order.append('torch.cuda.device_count')\n"
        "    names = [torch.cuda.get_device_name(i) for i in range(count)]\n"
        "    if count:\n"
        "        order.append('torch.cuda.get_device_name')\n"
        "    payload.update({'torch_import_ok': True, 'torch_version': getattr(torch, '__version__', None), 'torch_cuda_version': getattr(torch.version, 'cuda', None), 'cuda_available': available, 'device_count': count, 'device_names': names, 'import_order': order})\n"
        "except Exception as exc:\n"
        "    payload.update({'torch_import_ok': False, 'torch_version': None, 'torch_cuda_version': None, 'cuda_available': False, 'device_count': 0, 'device_names': [], 'import_order': order, 'error': f'{type(exc).__name__}: {exc}'})\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    command = [selected_python, "-I", "-c", script]
    result = command_runner(command, timeout_s=60, env=_probe_env(env, stage=stage))
    payload = _json_from_last_line(result)
    payload["command"] = result.get("command", command)
    payload["returncode"] = result.get("returncode")
    payload["stderr_summary"] = _summarize(_stderr(result))
    return payload


def _cuda_bindings_probe(
    python: str,
    *,
    env: Mapping[str, str],
    command_runner: CommandRunner,
    stage: str,
) -> JsonDict:
    """Probe CUDA directly through NVIDIA's Python runtime bindings."""
    script = (
        "import json\n"
        "order = ['json']\n"
        "payload = {'probe': 'exp3220_cuda_bindings_probe', 'stage': " + repr(stage) + "}\n"
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
        "    payload.update({'cuda_bindings_import_ok': True, 'cuda_runtime_ok': ok, 'device_count': int(count or 0) if ok else 0, 'device_names': names, 'cuda_error': str(err), 'cuda_runtime_version': runtime_version if str(runtime_err).endswith('cudaSuccess') else None, 'cuda_driver_version': driver_version if str(driver_err).endswith('cudaSuccess') else None, 'import_order': order})\n"
        "except Exception as exc:\n"
        "    payload.update({'cuda_bindings_import_ok': False, 'cuda_runtime_ok': False, 'device_count': 0, 'device_names': [], 'import_order': order, 'error': f'{type(exc).__name__}: {exc}'})\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    command = [python, "-I", "-c", script]
    result = command_runner(command, timeout_s=60, env=_probe_env(env, stage=stage))
    payload = _json_from_last_line(result)
    payload["command"] = result.get("command", command)
    payload["returncode"] = result.get("returncode")
    payload["stderr_summary"] = _summarize(_stderr(result))
    return payload


def _llama_cpp_linkage_probe(
    selected_python: str,
    *,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """Inspect llama.cpp import/linkage metadata without loading a model."""
    script = (
        "import importlib.metadata, importlib.util, json\n"
        "payload = {'probe': 'exp3220_llama_cpp_linkage_probe', 'llama_cpp_import_ok': False}\n"
        "try:\n"
        "    import llama_cpp\n"
        "    from llama_cpp import llama_cpp as low\n"
        "    lib = getattr(low, '_lib', None)\n"
        "    printer = getattr(low, 'llama_print_system_info', None)\n"
        "    system_info = ''\n"
        "    if callable(printer):\n"
        "        raw = printer()\n"
        "        system_info = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else str(raw)\n"
        "    supports = getattr(low, 'llama_supports_gpu_offload', None)\n"
        "    payload.update({'llama_cpp_import_ok': True, 'llama_cpp_version': getattr(llama_cpp, '__version__', None) or importlib.metadata.version('llama-cpp-python'), 'llama_cpp_origin': importlib.util.find_spec('llama_cpp').origin, 'shared_library_path': getattr(lib, '_name', None), 'llama_cpp_supports_gpu_offload': bool(supports()) if callable(supports) else False, 'llama_system_info': system_info})\n"
        "except Exception as exc:\n"
        "    payload['error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    command = [selected_python, "-I", "-c", script]
    result = command_runner(command, timeout_s=60, env=_probe_env(env, stage="llama_cpp_linkage"))
    payload = _json_from_last_line(result)
    payload["command"] = result.get("command", command)
    payload["returncode"] = result.get("returncode")
    payload["stderr_summary"] = _summarize(_stderr(result))
    return payload


def _torch_probe_ok(probe: Mapping[str, Any]) -> bool:
    return bool(probe.get("cuda_available")) and int(probe.get("device_count") or 0) > 0


def _cuda_bindings_probe_ok(probe: Mapping[str, Any]) -> bool:
    return bool(probe.get("cuda_runtime_ok")) and int(probe.get("device_count") or 0) > 0


def _command_status(result: Mapping[str, Any]) -> str:
    if result.get("returncode") is None:
        return "error"
    if result.get("returncode") == 0:
        return "ok"
    return "failed"


def _default_isolated_venv_path(clock: ClockFn) -> Path:
    stamp = f"{int(clock() * 1_000_000)}"
    return Path("/tmp") / f"carnot-exp3220-cuda-only-{stamp}"


def _create_isolated_cuda_venv(
    *,
    base_python: str,
    venv_path: Path,
    package_spec: str,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> JsonDict:
    """Create and probe a temporary CUDA-only venv outside source paths."""
    create_command = [base_python, "-m", "venv", str(venv_path)]
    create_result = command_runner(create_command, timeout_s=120, env=dict(env))
    actions = [
        {
            "action": "create_isolated_cuda_venv",
            "status": "created" if create_result.get("returncode") == 0 else "failed",
            "command": create_result.get("command", create_command),
            "stderr_summary": _summarize(_stderr(create_result)),
        }
    ]
    isolated_python = str(venv_path / "bin" / "python")
    if create_result.get("returncode") != 0:  # pragma: no cover - defensive subprocess failure
        return {
            "created": False,
            "cuda_ok": False,
            "path": str(venv_path),
            "actions": actions,
            "probe": {},
            "package_versions": {},
        }

    install_command = [
        isolated_python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        package_spec,
    ]
    install_result = command_runner(install_command, timeout_s=600, env=dict(env))
    actions.append(
        {
            "action": "install_isolated_cuda_package",
            "status": _command_status(install_result),
            "command": install_result.get("command", install_command),
            "stdout_summary": _summarize(_stdout(install_result)),
            "stderr_summary": _summarize(_stderr(install_result)),
        }
    )
    if install_result.get("returncode") != 0:  # pragma: no cover - defensive subprocess failure
        return {
            "created": True,
            "cuda_ok": False,
            "path": str(venv_path),
            "actions": actions,
            "probe": {},
            "package_versions": {},
        }

    probe = _cuda_bindings_probe(
        isolated_python,
        env=env,
        command_runner=command_runner,
        stage="isolated_cuda_venv",
    )
    show = _pip_show(isolated_python, "cuda-bindings", command_runner=command_runner)
    actions.append(
        {
            "action": "probe_isolated_cuda_venv",
            "status": "cuda_ok" if _cuda_bindings_probe_ok(probe) else "cuda_failed",
            "command": probe.get("command", []),
            "stderr_summary": probe.get("stderr_summary", ""),
        }
    )
    return {
        "created": True,
        "cuda_ok": _cuda_bindings_probe_ok(probe),
        "path": str(venv_path),
        "actions": actions,
        "probe": probe,
        "package_versions": {"cuda-bindings": show["metadata"].get("Version")},
    }


def _path_modified(path: str, *, command_runner: CommandRunner) -> bool:
    """Return whether git currently reports a protected path as modified."""
    result = command_runner(["git", "status", "--porcelain", "--", path], timeout_s=10)
    return bool(_stdout(result).strip())


def _llama_linkage_has_cuda_error(probe: Mapping[str, Any]) -> bool:
    evidence = str(probe.get("stderr_summary") or probe.get("error") or "").lower()
    return "ggml_cuda_init" in evidence or "failed to initialize cuda" in evidence


def _recommended_next_action(
    *,
    nvidia_available: bool,
    selected_after_ok: bool,
    selected_equivalent_ok: bool,
    isolated_created: bool,
    isolated_ok: bool,
    llama_probe: Mapping[str, Any],
) -> str:
    if not nvidia_available:  # pragma: no cover - live defensive branch
        return "repair_nvidia_driver_visibility_before_cuda_receipt"
    if selected_after_ok or selected_equivalent_ok:
        if _llama_linkage_has_cuda_error(llama_probe) or (
            llama_probe.get("llama_cpp_import_ok")
            and not llama_probe.get("llama_cpp_supports_gpu_offload")
        ):
            return "repair_llama_cpp_linkage_after_cuda_runtime_ok"
        return "allow_bounded_cuda_receipt_candidate_no_model_loaded"
    if isolated_created and isolated_ok:
        return "repair_selected_python_torch_cuda_or_recreate_venv"
    if not isolated_created:
        return "create_isolated_cuda_venv_to_disambiguate_selected_venv"
    return "repair_system_driver_cuda_runtime_boundary"


def _honest_verdict(*, candidate: bool, action: str) -> str:
    if candidate:
        return f"complete: cuda_receipt_ready_candidate=true; recommended_next_action={action}"
    if action == "repair_selected_python_torch_cuda_or_recreate_venv":
        return f"blocked_selected_python_cuda: cuda_receipt_ready_candidate=false; recommended_next_action={action}"
    return f"blocked_cuda_runtime: cuda_receipt_ready_candidate=false; recommended_next_action={action}"


def build_cuda_runtime_repair_ledger(
    *,
    project_root: str | Path,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    create_isolated_venv: bool = True,
    isolated_venv_path: str | Path | None = None,
    isolated_base_python: str | None = None,
    monotonic: ClockFn = time.perf_counter,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3220 CUDA runtime repair ledger."""
    start = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)
    explicit_env = dict(merged_env)
    explicit_env["CUDA_VISIBLE_DEVICES"] = "0"
    sanitized_env = _sanitized_cuda_env(merged_env)

    nvidia = _nvidia_smi_inventory(command_runner=command_runner)
    torch_pip = _pip_show(selected, "torch", command_runner=command_runner)
    cuda_bindings_pip = _pip_show(selected, "cuda-bindings", command_runner=command_runner)
    before_probe = _torch_cuda_probe(
        selected,
        env=explicit_env,
        command_runner=command_runner,
        stage="selected_before",
    )
    after_probe = _torch_cuda_probe(
        selected,
        env=sanitized_env,
        command_runner=command_runner,
        stage="selected_after",
    )
    selected_cuda_probe = _cuda_bindings_probe(
        selected,
        env=sanitized_env,
        command_runner=command_runner,
        stage="selected_after",
    )
    llama_probe = _llama_cpp_linkage_probe(
        selected,
        env=sanitized_env,
        command_runner=command_runner,
    )

    selected_before_ok = _torch_probe_ok(before_probe)
    selected_after_ok = _torch_probe_ok(after_probe)
    selected_equivalent_ok = _cuda_bindings_probe_ok(selected_cuda_probe)
    repair_actions: list[JsonDict] = [
        {
            "action": "explicit_cuda_visible_devices_probe",
            "status": "cuda_ok" if selected_before_ok else "cuda_failed",
            "command": before_probe.get("command", []),
            "stderr_summary": before_probe.get("stderr_summary", ""),
        },
        {
            "action": "sanitized_selected_python_cuda_probe",
            "status": "cuda_ok" if selected_after_ok or selected_equivalent_ok else "cuda_failed",
            "command": after_probe.get("command", []),
            "stderr_summary": after_probe.get("stderr_summary", ""),
        },
    ]

    package_version = cuda_bindings_pip["metadata"].get("Version")
    package_spec = f"cuda-bindings=={package_version}" if package_version else "cuda-bindings"
    isolated_info: JsonDict = {
        "created": False,
        "cuda_ok": False,
        "path": None,
        "actions": [],
        "probe": {},
        "package_versions": {},
    }
    if create_isolated_venv:
        venv_path = (
            Path(isolated_venv_path)
            if isolated_venv_path is not None
            else _default_isolated_venv_path(monotonic)
        )
        base_python = isolated_base_python or sys.executable
        isolated_info = _create_isolated_cuda_venv(
            base_python=base_python,
            venv_path=venv_path,
            package_spec=package_spec,
            env=sanitized_env,
            command_runner=command_runner,
        )
        repair_actions.extend(isolated_info["actions"])
    else:
        repair_actions.append(
            {
                "action": "create_isolated_cuda_venv",
                "status": "skipped",
                "reason": "disabled_by_caller",
            }
        )

    isolated_created = bool(isolated_info.get("created"))
    isolated_ok = bool(isolated_info.get("cuda_ok"))
    candidate = bool(selected_after_ok or selected_equivalent_ok)
    action = _recommended_next_action(
        nvidia_available=bool(nvidia["available"]),
        selected_after_ok=selected_after_ok,
        selected_equivalent_ok=selected_equivalent_ok,
        isolated_created=isolated_created,
        isolated_ok=isolated_ok,
        llama_probe=llama_probe,
    )
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "selected_python": selected,
        "selected_python_cuda_ok_before": selected_before_ok,
        "selected_python_cuda_ok_after": bool(selected_after_ok or selected_equivalent_ok),
        "isolated_cuda_venv_created": isolated_created,
        "isolated_cuda_venv_cuda_ok": isolated_ok,
        "cuda_visible_devices": explicit_env.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_available": bool(nvidia["available"]),
        "gpu_count_nvidia_smi": nvidia["gpu_count"],
        "driver_version": nvidia["driver_version"],
        "torch_version_selected": after_probe.get("torch_version")
        or before_probe.get("torch_version")
        or torch_pip["metadata"].get("Version"),
        "torch_cuda_version_selected": after_probe.get("torch_cuda_version")
        or before_probe.get("torch_cuda_version"),
        "environment_pollution_findings": _environment_pollution_findings(merged_env),
        "repair_actions_attempted": repair_actions,
        "cuda_receipt_ready_candidate": candidate,
        "recommended_next_action": action,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "conductor_file_modified": _path_modified(
            "scripts/research_conductor.py", command_runner=command_runner
        ),
        "active_roadmap_modified": _path_modified(
            "research-roadmap.yaml", command_runner=command_runner
        ),
        "honest_verdict": _honest_verdict(candidate=candidate, action=action),
        "duration_s": round(monotonic() - start, 6),
        "nvidia_smi": nvidia,
        "environment_snapshot": _tracked_env(merged_env),
        "sanitized_environment_snapshot": _tracked_env(sanitized_env),
        "selected_python_probe_before": before_probe,
        "selected_python_probe_after": after_probe,
        "selected_python_cuda_runtime_probe_after": selected_cuda_probe,
        "llama_cpp_linkage_probe": llama_probe,
        "isolated_cuda_venv": {
            "path": isolated_info.get("path"),
            "probe": isolated_info.get("probe", {}),
            "package_versions": isolated_info.get("package_versions", {}),
        },
        "tests_run": list(tests_run or []),
    }
    return artifact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON for conductor and downstream gates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    create_isolated_venv: bool = True,
    isolated_venv_path: str | Path | None = None,
    isolated_base_python: str | None = None,
    monotonic: ClockFn = time.perf_counter,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 3220 hermetic CUDA runtime repair ledger."""
    root = Path(project_root) if project_root is not None else _repo_root()
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    artifact = build_cuda_runtime_repair_ledger(
        project_root=root,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        create_isolated_venv=create_isolated_venv,
        isolated_venv_path=isolated_venv_path,
        isolated_base_python=isolated_base_python,
        monotonic=monotonic,
        tests_run=tests_run,
    )
    _write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selected-python", default=None)
    parser.add_argument("--skip-isolated-venv", action="store_true")
    parser.add_argument("--isolated-venv-path", type=Path, default=None)
    parser.add_argument("--isolated-base-python", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by conductor-style experiment runs."""
    args = _parse_args(argv)
    run_experiment(
        output_path=args.output,
        selected_python=args.selected_python,
        create_isolated_venv=not args.skip_isolated_venv,
        isolated_venv_path=args.isolated_venv_path,
        isolated_base_python=args.isolated_base_python,
        tests_run=args.test_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
