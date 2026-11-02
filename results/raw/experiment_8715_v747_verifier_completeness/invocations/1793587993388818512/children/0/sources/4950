"""Exp 3207 llama.cpp CUDA rebuild clean subprocess gate.

**Researcher summary:**
    This gate decides whether it is safe to rebuild llama-cpp-python for CUDA
    after Exp 3206's environment ledger.  If the selected Python still cannot
    initialize PyTorch CUDA, a llama.cpp rebuild would only churn the runtime
    while leaving the driver/Python blocker untouched, so the artifact records a
    terminal blocker instead of treating CPU fallback as success.

**Detailed explanation for engineers:**
    The prior ledger separated three layers: NVIDIA driver visibility, selected
    Python torch/CUDA initialization, and llama.cpp CUDA/offload support.  This
    module preserves that ordering.  It probes torch first in a clean subprocess,
    probes llama.cpp separately, and attempts the CUDA rebuild only when torch
    already sees a CUDA device and llama.cpp is the remaining CUDA/offload
    blocker.

Spec: REQ-INFER-SOTA-025,
      SCENARIO-INFER-SOTA-025-001,
      SCENARIO-INFER-SOTA-025-002,
      SCENARIO-INFER-SOTA-025-003
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.reporting import cuda_env_forensics_ledger_3206 as env3206


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.llama_cpp_cuda_rebuild_clean_subprocess.v1"
EXPERIMENT_ID = "exp3207"
MILESTONE = "2026.05.297"
DEFAULT_ENV_LEDGER_REL_PATH = Path("results/experiment_3206_cuda_env_forensics_ledger_v1.json")
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_3207_llama_cpp_cuda_rebuild_clean_subprocess_v1.json"
)
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / "experiment_3207_llama_cpp_cuda_rebuild_clean_subprocess_v1.py"
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "experiment_id",
    "milestone",
    "env_ledger_artifact",
    "rebuild_attempted",
    "rebuild_command_summary",
    "rebuild_log_tail",
    "torch_cuda_available_after",
    "llama_cpp_cuda_build_detected_after",
    "clean_subprocess_gpu_offload_probe_passed",
    "cpu_fallback_only",
    "cuda_receipt_ready",
    "clean_rerun_allowed_candidate",
    "blocker",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
)

REBUILD_ACTIONS = {
    "repair_llama_cpp_cuda_initialization_or_rebuild",
    "rebuild_llama_cpp_python_with_ggml_cuda",
}


def _repo_root() -> Path:
    """Return the repository root used for artifact paths."""
    return Path(os.environ.get("CARNOT_REPO_ROOT", Path.cwd())).resolve()


def _selected_python(project_root: str | Path) -> str:
    """Prefer the project virtualenv interpreter for runtime probes."""
    candidate = Path(project_root) / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> JsonDict:
    """Run a command with the same bounded evidence shape as the 3206 ledger."""
    return env3206._run_command(command, timeout_s=timeout_s, env=env, cwd=cwd)


def _json_from_last_line(result: Mapping[str, Any]) -> JsonDict:
    """Parse a JSON payload from a subprocess result."""
    return env3206._json_from_last_line(result)


def _tail_lines(text: str, *, label: str, limit: int = 12) -> list[str]:
    """Return a compact labeled tail for stdout/stderr evidence."""
    return env3206._stderr_tail(text, label=label, limit=limit)


def _load_ledger(path: str | Path) -> JsonDict:
    """Load Exp 3206's ledger, preserving file errors in the payload."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        return {"load_error": f"{type(exc).__name__}: {exc}"}
    return data if isinstance(data, dict) else {"load_error": "ledger_json_not_object"}


def _llama_version_from_ledger(ledger: Mapping[str, Any]) -> str | None:
    pip_show = ledger.get("pip_show")
    if not isinstance(pip_show, Mapping):
        return None
    llama_pkg = pip_show.get("llama-cpp-python")
    if not isinstance(llama_pkg, Mapping):
        return None
    metadata = llama_pkg.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    version = metadata.get("Version")
    return str(version) if version else None


def _rebuild_command(selected_python: str, ledger: Mapping[str, Any]) -> list[str]:
    """Build the minimum CUDA-enabled llama-cpp-python reinstall command."""
    package = "llama-cpp-python"
    version = _llama_version_from_ledger(ledger)
    if version:
        package = f"{package}=={version}"
    return [
        selected_python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-cache-dir",
        "--no-binary",
        "llama-cpp-python",
        package,
    ]


def _rebuild_env(env: Mapping[str, str]) -> dict[str, str]:
    """Apply the llama.cpp CUDA CMake knobs without mutating the caller env."""
    merged = dict(env)
    merged["CMAKE_ARGS"] = "-DGGML_CUDA=ON"
    merged["FORCE_CMAKE"] = "1"
    return merged


def _rebuild_command_summary(command: Sequence[str], env: Mapping[str, str]) -> list[str]:
    """Record auditable rebuild intent without dumping the whole environment."""
    return [
        " ".join(str(part) for part in command),
        f"CMAKE_ARGS={env.get('CMAKE_ARGS')}",
        f"FORCE_CMAKE={env.get('FORCE_CMAKE')}",
    ]


def _rebuild_log_tail(result: Mapping[str, Any]) -> list[str]:
    """Preserve the tail of rebuild stdout/stderr for blocked artifacts."""
    stdout = str(result.get("stdout") or result.get("stdout_summary") or "")
    stderr = str(result.get("stderr") or result.get("stderr_summary") or "")
    return _tail_lines(stdout, label="stdout", limit=6) + _tail_lines(
        stderr, label="stderr", limit=6
    )


def _cuda_init_failed(stderr_tail: Sequence[str]) -> bool:
    return "ggml_cuda_init" in " ".join(stderr_tail).lower() or (
        "failed to initialize cuda" in " ".join(stderr_tail).lower()
    )


def _llama_cuda_build_detected(
    llama_probe: Mapping[str, Any],
    stderr_tail: Sequence[str],
) -> bool:
    return env3206._llama_cpp_cuda_build_detected(llama_probe, stderr_tail)


def _torch_cuda_available(torch_probe: Mapping[str, Any]) -> bool:
    return bool(torch_probe.get("cuda_available")) and int(torch_probe.get("device_count") or 0) > 0


def _needs_rebuild(
    action: str,
    llama_probe: Mapping[str, Any],
    stderr_tail: Sequence[str],
) -> bool:
    return (
        action in REBUILD_ACTIONS
        or _cuda_init_failed(stderr_tail)
        or (
            bool(llama_probe.get("llama_cpp_import_ok"))
            and not bool(llama_probe.get("llama_cpp_supports_gpu_offload"))
        )
    )


def _torch_blocker(torch_probe: Mapping[str, Any], action: str) -> str:
    stderr = str(torch_probe.get("stderr_summary") or torch_probe.get("error") or "").strip()
    return (
        "selected_python_torch_cuda_unavailable: "
        f"cuda_available={bool(torch_probe.get('cuda_available'))}; "
        f"device_count={int(torch_probe.get('device_count') or 0)}; "
        f"stderr={stderr or 'none'}; "
        f"ledger_recommended_next_action={action}"
    )


def _llama_blocker(
    llama_probe: Mapping[str, Any],
    stderr_tail: Sequence[str],
    cuda_build_detected: bool,
) -> str:
    stderr = "; ".join(stderr_tail) if stderr_tail else "none"
    return (
        "llama_cpp_gpu_offload_probe_failed: "
        f"import_ok={bool(llama_probe.get('llama_cpp_import_ok'))}; "
        f"supports_gpu_offload={bool(llama_probe.get('llama_cpp_supports_gpu_offload'))}; "
        f"cuda_build_detected={cuda_build_detected}; stderr={stderr}"
    )


def _gpu_offload_probe_passed(
    *,
    torch_available: bool,
    llama_probe: Mapping[str, Any],
    stderr_tail: Sequence[str],
    cuda_build_detected: bool,
) -> bool:
    return bool(
        torch_available
        and llama_probe.get("llama_cpp_import_ok")
        and llama_probe.get("llama_cpp_supports_gpu_offload")
        and cuda_build_detected
        and not _cuda_init_failed(stderr_tail)
    )


def _honest_verdict(*, ready: bool, blocker: str | None) -> str:
    if ready:
        return "complete: cuda_receipt_ready=true; clean_rerun_allowed_candidate=true"
    if blocker and blocker.startswith("selected_python_torch_cuda_unavailable"):
        return f"blocked_selected_python_torch_cuda: {blocker}"
    if blocker and blocker.startswith("llama_cpp_cuda_rebuild_failed"):
        return f"blocked_llama_cpp_cuda_rebuild: {blocker}"
    return f"blocked_llama_cpp_gpu_offload: {blocker or 'unknown_blocker'}"


def build_artifact(
    *,
    project_root: str | Path,
    env_ledger_path: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: ClockFn = time.perf_counter,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3207 rebuild/blocked gate artifact."""
    start = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    ledger_path = (
        Path(env_ledger_path) if env_ledger_path is not None else root / DEFAULT_ENV_LEDGER_REL_PATH
    )
    ledger = _load_ledger(ledger_path)
    ledger_action = str(ledger.get("recommended_next_action") or "unknown")
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)

    torch_probe = env3206._torch_clean_probe(selected, command_runner=command_runner)
    torch_available = _torch_cuda_available(torch_probe)
    llama_probe = env3206._llama_cpp_clean_probe(
        selected,
        command_runner=command_runner,
        env=merged_env,
    )
    stderr_tail = env3206._clean_subprocess_stderr_tail(torch_probe, llama_probe)
    cuda_build_detected = _llama_cuda_build_detected(llama_probe, stderr_tail)

    rebuild_attempted = False
    rebuild_summary: list[str] = []
    rebuild_tail: list[str] = []
    blocker: str | None = None

    if not torch_available:
        blocker = _torch_blocker(torch_probe, ledger_action)
    elif _needs_rebuild(ledger_action, llama_probe, stderr_tail):
        rebuild_attempted = True
        rebuild_env = _rebuild_env(merged_env)
        command = _rebuild_command(selected, ledger)
        rebuild_summary = _rebuild_command_summary(command, rebuild_env)
        rebuild_result = command_runner(
            command,
            timeout_s=1800,
            env=rebuild_env,
            cwd=str(root),
        )
        rebuild_tail = _rebuild_log_tail(rebuild_result)
        if rebuild_result.get("returncode") != 0:
            stderr = str(rebuild_result.get("stderr") or rebuild_result.get("stderr_summary") or "")
            blocker = (
                "llama_cpp_cuda_rebuild_failed: "
                f"returncode={rebuild_result.get('returncode')}; stderr={stderr.strip() or 'none'}"
            )
        else:
            llama_probe = env3206._llama_cpp_clean_probe(
                selected,
                command_runner=command_runner,
                env=merged_env,
            )
            stderr_tail = env3206._clean_subprocess_stderr_tail(torch_probe, llama_probe)
            cuda_build_detected = _llama_cuda_build_detected(llama_probe, stderr_tail)

    offload_passed = _gpu_offload_probe_passed(
        torch_available=torch_available,
        llama_probe=llama_probe,
        stderr_tail=stderr_tail,
        cuda_build_detected=cuda_build_detected,
    )
    if blocker is None and not offload_passed:
        blocker = _llama_blocker(llama_probe, stderr_tail, cuda_build_detected)

    ready = bool(offload_passed and blocker is None)
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "env_ledger_artifact": str(ledger_path),
        "rebuild_attempted": rebuild_attempted,
        "rebuild_command_summary": rebuild_summary,
        "rebuild_log_tail": rebuild_tail,
        "torch_cuda_available_after": torch_available,
        "llama_cpp_cuda_build_detected_after": cuda_build_detected,
        "clean_subprocess_gpu_offload_probe_passed": offload_passed,
        "cpu_fallback_only": not ready,
        "cuda_receipt_ready": ready,
        "clean_rerun_allowed_candidate": ready,
        "blocker": blocker,
        "conductor_file_modified": env3206._path_modified(
            "scripts/research_conductor.py", command_runner=command_runner
        ),
        "active_roadmap_modified": env3206._path_modified(
            "research-roadmap.yaml", command_runner=command_runner
        ),
        "honest_verdict": _honest_verdict(ready=ready, blocker=blocker),
        "duration_s": round(monotonic() - start, 6),
        "selected_python": selected,
        "env_ledger_recommended_next_action": ledger_action,
        "env_ledger_honest_verdict": ledger.get("honest_verdict"),
        "torch_clean_subprocess_after": torch_probe,
        "llama_cpp_clean_subprocess_after": llama_probe,
        "clean_subprocess_stderr_tail_after": stderr_tail,
        "tests_run": list(tests_run or []),
    }
    return artifact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON for downstream conductor gates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    env_ledger_path: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    monotonic: ClockFn = time.perf_counter,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 3207 CUDA rebuild gate artifact."""
    root = Path(project_root) if project_root is not None else _repo_root()
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    artifact = build_artifact(
        project_root=root,
        env_ledger_path=env_ledger_path,
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
    parser.add_argument("--env-ledger", type=Path, default=None)
    parser.add_argument("--selected-python", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by conductor-style experiment runs."""
    args = _parse_args(argv)
    run_experiment(
        output_path=args.output,
        env_ledger_path=args.env_ledger,
        selected_python=args.selected_python,
        tests_run=args.test_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
