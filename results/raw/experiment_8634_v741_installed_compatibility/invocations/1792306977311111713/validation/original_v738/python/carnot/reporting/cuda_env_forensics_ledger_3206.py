"""Exp 3206 CUDA environment forensics ledger.

**Researcher summary:**
    This ledger explains whether the selected Python environment can initialize
    CUDA through PyTorch and llama.cpp before the conductor attempts another
    rebuild or full local SOTA receipt.  It is an environment-forensics
    artifact, not a model-quality experiment, so it does not load a large GGUF.

**Detailed explanation for engineers:**
    Prior CUDA/offload work saw an RTX 3090 through ``nvidia-smi`` while the
    selected Python reported ``torch.cuda.is_available() == false`` and
    ``llama_cpp`` printed ``ggml_cuda_init: failed to initialize CUDA``.  Those
    facts can be caused by different layers: driver visibility, virtualenv
    package selection, dynamic-library search paths, import order, or a
    llama.cpp build/runtime mismatch.  This module records each layer in clean
    subprocesses so the next task can repair the right layer instead of
    treating a CPU fallback as runtime evidence.

Spec: REQ-INFER-SOTA-024,
      SCENARIO-INFER-SOTA-024-001,
      SCENARIO-INFER-SOTA-024-002,
      SCENARIO-INFER-SOTA-024-003
"""

from __future__ import annotations

import argparse
import json
import os
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
SCHEMA_VERSION = "carnot.cuda_env_forensics_ledger.v1"
EXPERIMENT_ID = "exp3206"
MILESTONE = "2026.05.297"
ARTIFACT = "experiment_3206_cuda_env_forensics_ledger_v1"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_3206_cuda_env_forensics_ledger_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3206_cuda_env_forensics_ledger_v1.py"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "experiment_id",
    "milestone",
    "selected_python",
    "virtualenv",
    "nvidia_smi_available",
    "gpu_count_nvidia_smi",
    "torch_version",
    "torch_cuda_version",
    "torch_cuda_available_clean_subprocess",
    "torch_cuda_device_count_clean_subprocess",
    "llama_cpp_version",
    "llama_cpp_origin",
    "llama_cpp_cuda_build_detected",
    "clean_subprocess_stderr_tail",
    "cuda_env_vars",
    "cuda_env_diagnosed",
    "cuda_init_clean",
    "recommended_next_action",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
)

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
]
CUDA_ENV_BASE_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "PATH",
    "CMAKE_ARGS",
    "FORCE_CMAKE",
)
CUDA_ENV_KEY_FRAGMENTS = ("CUDA", "CUDNN", "NVIDIA", "LLAMA", "LLAMA_CPP", "GGML")


def _repo_root() -> Path:
    """Return the repository root, honoring the same override used by experiments."""
    return Path(os.environ.get("CARNOT_REPO_ROOT", Path.cwd())).resolve()


def _selected_python(project_root: str | Path) -> str:
    """Select the project virtualenv interpreter when it exists."""
    candidate = Path(project_root) / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str | None, *, limit: int = 4000) -> str:
    """Keep command evidence bounded while preserving the tail where failures land."""
    value = text or ""
    return value if len(value) <= limit else value[-limit:]


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> JsonDict:
    """Run a diagnostic command and return enough evidence to audit failures."""
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
    """Parse the last JSON line from a subprocess, preserving parse failure text."""
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
    selected_python: str,
    package: str,
    *,
    command_runner: CommandRunner,
) -> JsonDict:
    """Run ``pip show`` through the selected interpreter."""
    command = [selected_python, "-m", "pip", "show", package]
    result = command_runner(command, timeout_s=20)
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
    """Parse the NVIDIA GPU inventory CSV used by the ledger."""
    rows: list[JsonDict] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6 or _int_field(parts[0]) is None:
            continue
        rows.append(
            {
                "index": _int_field(parts[0]),
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mib": _int_field(parts[3]),
                "memory_used_mib": _int_field(parts[4]),
                "memory_free_mib": _int_field(parts[5]),
            }
        )
    return rows


def _nvidia_smi_inventory(*, command_runner: CommandRunner) -> JsonDict:
    """Capture both parseable GPU inventory and raw ``nvidia-smi`` evidence."""
    query_result = command_runner(NVIDIA_SMI_QUERY, timeout_s=10)
    raw_result = command_runner(["nvidia-smi"], timeout_s=10)
    gpus = _parse_nvidia_smi_csv(_stdout(query_result)) if query_result.get("returncode") == 0 else []
    return {
        "available": bool(gpus),
        "gpu_count": len(gpus) if gpus else None,
        "gpus": gpus,
        "driver_version": gpus[0]["driver_version"] if gpus else None,
        "query_command": query_result.get("command", NVIDIA_SMI_QUERY),
        "query_returncode": query_result.get("returncode"),
        "query_stdout_summary": _summarize(_stdout(query_result)),
        "query_stderr_summary": _summarize(_stderr(query_result)),
        "raw_command": raw_result.get("command", ["nvidia-smi"]),
        "raw_returncode": raw_result.get("returncode"),
        "raw_stdout_summary": _summarize(_stdout(raw_result)),
        "raw_stderr_summary": _summarize(_stderr(raw_result)),
    }


def _nvcc_version(*, command_runner: CommandRunner) -> JsonDict:
    """Capture CUDA toolkit compiler provenance when ``nvcc`` is on PATH."""
    command = ["nvcc", "--version"]
    result = command_runner(command, timeout_s=10)
    return {
        "available": result.get("returncode") == 0,
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
    }


def _python_env_probe(selected_python: str, *, command_runner: CommandRunner) -> JsonDict:
    """Capture selected interpreter identity without importing project modules."""
    script = (
        "import json, os, sys\n"
        "payload = {\n"
        "  'probe': 'exp3206_python_env_probe',\n"
        "  'executable': sys.executable,\n"
        "  'prefix': sys.prefix,\n"
        "  'base_prefix': getattr(sys, 'base_prefix', None),\n"
        "  'virtualenv': os.environ.get('VIRTUAL_ENV'),\n"
        "  'sys_path': list(sys.path),\n"
        "}\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    command = [selected_python, "-c", script]
    result = command_runner(command, timeout_s=20)
    payload = _json_from_last_line(result)
    payload["command"] = result.get("command", command)
    payload["returncode"] = result.get("returncode")
    payload["stderr_summary"] = _summarize(_stderr(result))
    return payload


def _torch_clean_probe(selected_python: str, *, command_runner: CommandRunner) -> JsonDict:
    """Import torch first in an isolated subprocess and record CUDA visibility."""
    script = (
        "import json, sys\n"
        "payload = {'probe': 'exp3206_torch_clean_probe', 'project_modules_preimport': [m for m in sys.modules if m == 'carnot' or m.startswith('carnot.') or m == 'scripts' or m.startswith('scripts.')]}\n"
        "try:\n"
        "    import torch\n"
        "    available = bool(torch.cuda.is_available())\n"
        "    count = int(torch.cuda.device_count()) if available else 0\n"
        "    names = [torch.cuda.get_device_name(i) for i in range(count)]\n"
        "    payload.update({'torch_import_ok': True, 'torch_version': getattr(torch, '__version__', None), 'torch_cuda_version': getattr(torch.version, 'cuda', None), 'cuda_available': available, 'device_count': count, 'device_names': names})\n"
        "except Exception as exc:\n"
        "    payload.update({'torch_import_ok': False, 'torch_version': None, 'torch_cuda_version': None, 'cuda_available': False, 'device_count': 0, 'device_names': [], 'error': f'{type(exc).__name__}: {exc}'})\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    command = [selected_python, "-I", "-c", script]
    result = command_runner(command, timeout_s=60)
    payload = _json_from_last_line(result)
    payload["command"] = result.get("command", command)
    payload["returncode"] = result.get("returncode")
    payload["stderr_summary"] = _summarize(_stderr(result))
    return payload


def _llama_cpp_clean_probe(
    selected_python: str,
    *,
    command_runner: CommandRunner,
    env: Mapping[str, str],
) -> JsonDict:
    """Import llama_cpp in a clean subprocess and query offload metadata only."""
    script = (
        "import importlib.metadata, importlib.util, json\n"
        "payload = {'probe': 'exp3206_llama_cpp_clean_probe', 'llama_cpp_import_ok': False}\n"
        "try:\n"
        "    import llama_cpp\n"
        "    from llama_cpp import llama_cpp as low\n"
        "    lib = getattr(low, '_lib', None)\n"
        "    system_info = ''\n"
        "    printer = getattr(low, 'llama_print_system_info', None)\n"
        "    if callable(printer):\n"
        "        raw = printer()\n"
        "        system_info = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else str(raw)\n"
        "    supports = getattr(low, 'llama_supports_gpu_offload', None)\n"
        "    payload.update({\n"
        "      'llama_cpp_import_ok': True,\n"
        "      'llama_cpp_version': getattr(llama_cpp, '__version__', None) or importlib.metadata.version('llama-cpp-python'),\n"
        "      'llama_cpp_origin': importlib.util.find_spec('llama_cpp').origin,\n"
        "      'shared_library_path': getattr(lib, '_name', None),\n"
        "      'llama_cpp_supports_gpu_offload': bool(supports()) if callable(supports) else False,\n"
        "      'llama_system_info': system_info,\n"
        "    })\n"
        "except Exception as exc:\n"
        "    payload['error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    command = [selected_python, "-I", "-c", script]
    result = command_runner(command, timeout_s=60, env=dict(env))
    payload = _json_from_last_line(result)
    payload["command"] = result.get("command", command)
    payload["returncode"] = result.get("returncode")
    payload["stderr_summary"] = _summarize(_stderr(result))
    return payload


def _virtualenv_from_python(
    selected_python: str,
    env: Mapping[str, str],
    python_env: Mapping[str, Any],
) -> str | None:
    """Infer the virtualenv path from environment, subprocess metadata, or path shape."""
    path_parent = Path(selected_python).parent.parent
    inferred = str(path_parent) if path_parent.name == ".venv" else None
    return env.get("VIRTUAL_ENV") or python_env.get("virtualenv") or inferred


def _cuda_env_vars(env: Mapping[str, str]) -> JsonDict:
    """Collect CUDA, llama.cpp, and dynamic-loader environment knobs."""
    keys = set(CUDA_ENV_BASE_KEYS)
    keys.update(
        key
        for key in env
        if any(fragment in key.upper() for fragment in CUDA_ENV_KEY_FRAGMENTS)
    )
    return {key: env.get(key) for key in sorted(keys)}


def _stderr_tail(text: str, *, label: str, limit: int = 20) -> list[str]:
    """Return labeled stderr tail lines for compact top-level diagnostics."""
    return [f"{label}: {line}" for line in text.splitlines()[-limit:] if line.strip()]


def _clean_subprocess_stderr_tail(
    torch_probe: Mapping[str, Any],
    llama_probe: Mapping[str, Any],
) -> list[str]:
    """Merge clean torch and llama.cpp stderr tails."""
    return _stderr_tail(str(torch_probe.get("stderr_summary") or ""), label="torch") + _stderr_tail(
        str(llama_probe.get("stderr_summary") or ""), label="llama_cpp"
    )


def _llama_cpp_cuda_build_detected(
    llama_probe: Mapping[str, Any],
    stderr_tail: Sequence[str],
) -> bool:
    """Detect whether the llama.cpp backend exposes or attempted CUDA support."""
    evidence = " ".join(
        [
            str(llama_probe.get("llama_system_info") or ""),
            str(llama_probe.get("shared_library_path") or ""),
            " ".join(stderr_tail),
        ]
    ).lower()
    return bool(
        llama_probe.get("llama_cpp_supports_gpu_offload")
        or "cuda" in evidence
        or "cublas" in evidence
        or "ggml_cuda" in evidence
    )


def _path_modified(path: str, *, command_runner: CommandRunner) -> bool:
    """Return whether git currently reports a protected path as modified."""
    result = command_runner(["git", "status", "--porcelain", "--", path], timeout_s=10)
    return bool(_stdout(result).strip())


def _recommended_next_action(
    *,
    nvidia_available: bool,
    torch_probe: Mapping[str, Any],
    llama_probe: Mapping[str, Any],
    stderr_tail: Sequence[str],
) -> str:
    """Map forensics facts to the next allowed conductor action."""
    cuda_error = "failed to initialize cuda" in " ".join(stderr_tail).lower()
    checks = [
        (not nvidia_available, "repair_nvidia_visibility_before_cuda_receipt"),
        (not torch_probe.get("torch_import_ok", True), "install_torch_in_selected_python"),
        (
            not torch_probe.get("cuda_available") or int(torch_probe.get("device_count") or 0) == 0,
            "repair_selected_python_torch_cuda_before_full_receipt",
        ),
        (not llama_probe.get("llama_cpp_import_ok"), "install_llama_cpp_python_in_selected_python"),
        (cuda_error, "repair_llama_cpp_cuda_initialization_or_rebuild"),
        (
            not llama_probe.get("llama_cpp_supports_gpu_offload"),
            "rebuild_llama_cpp_python_with_ggml_cuda",
        ),
        (True, "allow_full_local_sota_receipt_rerun"),
    ]
    return next(action for failed, action in checks if failed)


def _honest_verdict(*, clean: bool, action: str) -> str:
    """Convert the diagnostic decision to a terminal verdict string."""
    prefix_by_action = {
        "allow_full_local_sota_receipt_rerun": "complete",
        "repair_nvidia_visibility_before_cuda_receipt": "blocked_nvidia_visibility",
        "install_torch_in_selected_python": "blocked_torch_import",
        "repair_selected_python_torch_cuda_before_full_receipt": "blocked_selected_python_torch_cuda",
        "install_llama_cpp_python_in_selected_python": "blocked_llama_cpp_import",
        "repair_llama_cpp_cuda_initialization_or_rebuild": "blocked_llama_cpp_cuda_init",
        "rebuild_llama_cpp_python_with_ggml_cuda": "blocked_llama_cpp_cuda_build",
    }
    prefix = "complete" if clean else prefix_by_action.get(action, "blocked_cuda_environment")
    return f"{prefix}: cuda_env_diagnosed=true; cuda_init_clean={str(clean).lower()}; recommended_next_action={action}"


def build_cuda_env_ledger(
    *,
    project_root: str | Path,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: ClockFn = time.perf_counter,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3206 ledger without mutating the CUDA environment."""
    start = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)

    nvidia = _nvidia_smi_inventory(command_runner=command_runner)
    nvcc = _nvcc_version(command_runner=command_runner)
    python_env = _python_env_probe(selected, command_runner=command_runner)
    torch_pip = _pip_show(selected, "torch", command_runner=command_runner)
    llama_pip = _pip_show(selected, "llama-cpp-python", command_runner=command_runner)
    torch_probe = _torch_clean_probe(selected, command_runner=command_runner)
    llama_probe = _llama_cpp_clean_probe(selected, command_runner=command_runner, env=merged_env)
    stderr_tail = _clean_subprocess_stderr_tail(torch_probe, llama_probe)
    cuda_build = _llama_cpp_cuda_build_detected(llama_probe, stderr_tail)
    action = _recommended_next_action(
        nvidia_available=bool(nvidia["available"]),
        torch_probe=torch_probe,
        llama_probe=llama_probe,
        stderr_tail=stderr_tail,
    )
    clean = bool(
        action == "allow_full_local_sota_receipt_rerun"
        and nvidia["available"]
        and torch_probe.get("cuda_available")
        and int(torch_probe.get("device_count") or 0) > 0
        and llama_probe.get("llama_cpp_import_ok")
        and llama_probe.get("llama_cpp_supports_gpu_offload")
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "selected_python": selected,
        "virtualenv": _virtualenv_from_python(selected, merged_env, python_env),
        "nvidia_smi_available": bool(nvidia["available"]),
        "gpu_count_nvidia_smi": nvidia["gpu_count"],
        "torch_version": torch_probe.get("torch_version") or torch_pip["metadata"].get("Version"),
        "torch_cuda_version": torch_probe.get("torch_cuda_version"),
        "torch_cuda_available_clean_subprocess": bool(torch_probe.get("cuda_available")),
        "torch_cuda_device_count_clean_subprocess": int(torch_probe.get("device_count") or 0),
        "llama_cpp_version": llama_probe.get("llama_cpp_version")
        or llama_pip["metadata"].get("Version"),
        "llama_cpp_origin": llama_probe.get("llama_cpp_origin"),
        "llama_cpp_cuda_build_detected": cuda_build,
        "clean_subprocess_stderr_tail": stderr_tail,
        "cuda_env_vars": _cuda_env_vars(merged_env),
        "cuda_env_diagnosed": True,
        "cuda_init_clean": clean,
        "recommended_next_action": action,
        "conductor_file_modified": _path_modified(
            "scripts/research_conductor.py", command_runner=command_runner
        ),
        "active_roadmap_modified": _path_modified("research-roadmap.yaml", command_runner=command_runner),
        "honest_verdict": _honest_verdict(clean=clean, action=action),
        "duration_s": round(monotonic() - start, 6),
        "nvidia_smi": nvidia,
        "nvcc": nvcc,
        "selected_python_probe": python_env,
        "selected_python_sys_path": python_env.get("sys_path", []),
        "pip_show": {"torch": torch_pip, "llama-cpp-python": llama_pip},
        "torch_clean_subprocess": torch_probe,
        "llama_cpp_clean_subprocess": llama_probe,
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
    monotonic: ClockFn = time.perf_counter,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 3206 CUDA environment ledger."""
    root = Path(project_root) if project_root is not None else _repo_root()
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    artifact = build_cuda_env_ledger(
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
    """CLI entrypoint used by conductor-style experiment runs."""
    args = _parse_args(argv)
    run_experiment(
        output_path=args.output,
        selected_python=args.selected_python,
        tests_run=args.test_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
