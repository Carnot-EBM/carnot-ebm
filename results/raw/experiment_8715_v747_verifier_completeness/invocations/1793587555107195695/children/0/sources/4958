"""Local SOTA GGUF runtime repair artifact for Exp 1463.

This module is intentionally a narrow operational wrapper around the Exp 1442
runtime preflight.  Exp 1442 already knows how to inspect the mandated SOTA GGUF
cache and run a tiny llama.cpp probe; Exp 1463 adds the missing runtime-repair
evidence: reproduce the old failure first, discover CUDA runtime libraries, add
the project-local CUDA runtime directories to the probe environment, and record
whether the repaired path produced a real non-empty local response.

Spec: REQ-INFER-SOTA-008,
      SCENARIO-INFER-SOTA-008-001,
      SCENARIO-INFER-SOTA-008-002
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot.reporting import live_sota_repair_runtime_preflight as preflight

DEFAULT_ARTIFACT_PATH = Path("results/experiment_1463_local_sota_gguf_runtime_repair.json")
DEFAULT_REPRODUCED_1442_PATH = Path("results/experiment_1463_reproduced_exp1442_current_probe.json")
MIDDLE_MOE_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MIDDLE_MOE_Q4_FILENAME = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "gpu_probe",
    "libcudart_resolution_attempted",
    "missing_cache_resolution_attempted",
    "models_found_in_cache",
    "models_missing_from_cache",
    "smoke_inference_results",
    "live_sota_model_inference_used",
    "local_sota_runtime_ready",
    "persistent_blockers",
    "honest_verdict",
)

JsonDict = dict[str, Any]
ProbeFn = Callable[..., JsonDict]
CudaDiscoveryFn = Callable[..., JsonDict]
MissingCacheResolutionFn = Callable[..., JsonDict]
WriteJsonFn = Callable[[Path, JsonDict], None]


def _utc_run_date() -> str:
    """Return the UTC run date used by conductor-managed experiment artifacts."""
    return time.strftime("%Y%m%d", time.gmtime())


def _repo_root() -> Path:
    """Return the repository root using the canonical experiment helper."""
    return preflight._repo_root()


def _write_json(path: Path, payload: JsonDict) -> None:
    """Write deterministic JSON so repeated conductor reads see stable bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize(text: str | None, *, limit: int = 1500) -> str:
    """Keep command evidence compact while preserving the failure prefix."""
    return preflight._summarize_stream(text, limit=limit)


def _run_command(command: list[str], *, timeout_s: int = 10, env: Mapping[str, str] | None = None) -> JsonDict:
    """Run a local diagnostic command and capture enough detail for the artifact.

    These diagnostics are evidence, not control flow.  Any command can be absent
    on a contributor machine, so exceptions are converted into structured rows
    instead of aborting the experiment.
    """
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
        )
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout_summary": "",
            "stderr_summary": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_summary": _summarize(completed.stdout),
        "stderr_summary": _summarize(completed.stderr),
    }


def _venv_python(project_root: Path) -> str:
    """Prefer the project venv Python because the CUDA wheels live there."""
    candidate = project_root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _candidate_cuda_library_dirs(project_root: Path) -> list[str]:
    """Return CUDA library directories likely needed by llama.cpp's CUDA build."""
    site_packages = sorted((project_root / ".venv" / "lib").glob("python*/site-packages"))
    candidates: list[Path] = []
    for site in site_packages:
        candidates.extend(
            [
                site / "nvidia" / "cuda_runtime" / "lib",
                site / "nvidia" / "cublas" / "lib",
            ]
        )
    return [str(path) for path in candidates]


def _prepend_existing_library_dirs(*, current: str, candidate_dirs: Sequence[str]) -> str:
    """Prepend existing candidate directories without duplicating LD entries."""
    existing: list[str] = []
    seen: set[str] = set()
    for raw in candidate_dirs:
        path = str(Path(raw))
        if path in seen or not Path(path).is_dir():
            continue
        seen.add(path)
        existing.append(path)

    current_parts = [part for part in current.split(":") if part]
    for part in current_parts:
        if part not in seen:
            existing.append(part)
            seen.add(part)
    return ":".join(existing)


def _extract_ldconfig_libs(ldconfig_stdout: str) -> dict[str, str]:
    """Pull CUDA-related library names from `ldconfig -p` output."""
    libs: dict[str, str] = {}
    for line in ldconfig_stdout.splitlines():
        if not any(token in line for token in ("libcudart", "libcuda", "libcublas")):
            continue
        if "=>" not in line:
            continue
        name = line.strip().split(" ", 1)[0]
        libs[name] = line.rsplit("=>", 1)[-1].strip()
    return libs


def discover_cuda_runtime_state(
    *,
    project_root: Path,
    env: Mapping[str, str] | None = None,
    prior_artifact: JsonDict | None = None,
) -> JsonDict:
    """Inspect CUDA runtime state before applying the project-local path repair."""
    del prior_artifact
    base_env = dict(os.environ if env is None else env)
    candidate_dirs = _candidate_cuda_library_dirs(project_root)
    existing_dirs = [path for path in candidate_dirs if Path(path).is_dir()]
    repaired_ld_path = ":".join(existing_dirs + [base_env.get("LD_LIBRARY_PATH", "")]).strip(":")
    repaired_env = {**base_env, "LD_LIBRARY_PATH": repaired_ld_path}

    ldconfig = _run_command(["ldconfig", "-p"], timeout_s=10)
    nvidia_smi = _run_command(["nvidia-smi"], timeout_s=10)
    package_cmd = [
        _venv_python(project_root),
        "-m",
        "pip",
        "show",
        "llama-cpp-python",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "torch",
    ]
    package_metadata = _run_command(package_cmd, timeout_s=20)

    libllama = project_root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    libllama = libllama / "site-packages" / "llama_cpp" / "lib" / "libllama.so"
    if not libllama.exists():
        matches = sorted((project_root / ".venv" / "lib").glob("python*/site-packages/llama_cpp/lib/libllama.so"))
        libllama = matches[0] if matches else libllama

    ldd_before = _run_command(["ldd", str(libllama)], timeout_s=10) if libllama.exists() else {}
    ldd_after = (
        _run_command(["ldd", str(libllama)], timeout_s=10, env=repaired_env)
        if libllama.exists()
        else {}
    )

    return {
        "ldconfig": ldconfig,
        "ldconfig_libs": _extract_ldconfig_libs(ldconfig.get("stdout_summary", "")),
        "nvidia_smi": nvidia_smi,
        "package_metadata": package_metadata,
        "environment": {
            key: base_env.get(key, "")
            for key in (
                "LD_LIBRARY_PATH",
                "CUDA_HOME",
                "CUDA_PATH",
                "CARNOT_REPO_ROOT",
                "HF_HOME",
                "HUGGINGFACE_HUB_CACHE",
            )
        },
        "candidate_library_dirs": candidate_dirs,
        "existing_library_dirs": existing_dirs,
        "libllama_path": str(libllama) if libllama.exists() else None,
        "libllama_ldd_before": ldd_before.get("stdout_summary", "") or ldd_before.get("stderr_summary", ""),
        "libllama_ldd_after": ldd_after.get("stdout_summary", "") or ldd_after.get("stderr_summary", ""),
    }


def _artifact_strings(payload: Any) -> list[str]:
    """Flatten artifact values into strings for precise blocker detection."""
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, Mapping):
        rows: list[str] = []
        for value in payload.values():
            rows.extend(_artifact_strings(value))
        return rows
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray)):
        rows = []
        for value in payload:
            rows.extend(_artifact_strings(value))
        return rows
    return []


def _has_libcudart_blocker(artifact: JsonDict) -> bool:
    """Return whether the reproduced probe hit the CUDA runtime loader failure."""
    joined = "\n".join(_artifact_strings(artifact)).lower()
    return "libcudart.so.12" in joined or "libcublas.so.12" in joined


def _repair_env_from_discovery(discovery: JsonDict, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a subprocess environment with discovered CUDA libraries first."""
    env = dict(os.environ if base_env is None else base_env)
    existing_dirs = [str(path) for path in discovery.get("existing_library_dirs", [])]
    current = env.get("LD_LIBRARY_PATH", "")
    parts = existing_dirs + [part for part in current.split(":") if part]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part or part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    env["LD_LIBRARY_PATH"] = ":".join(deduped)
    return env


def _libcudart_resolution_attempt(prior: JsonDict, discovery: JsonDict, repair_env: Mapping[str, str]) -> JsonDict:
    """Summarize whether the venv CUDA runtime path repair was available and used."""
    existing_dirs = [str(path) for path in discovery.get("existing_library_dirs", [])]
    prior_had_loader_error = _has_libcudart_blocker(prior)
    return {
        "attempted": prior_had_loader_error or bool(existing_dirs),
        "prior_probe_had_libcudart_blocker": prior_had_loader_error,
        "repair_applied": bool(existing_dirs),
        "repair_method": "prepend_project_venv_cuda_runtime_and_cublas_to_LD_LIBRARY_PATH"
        if existing_dirs
        else "no_project_venv_cuda_runtime_dirs_available",
        "ld_library_path_before": discovery.get("environment", {}).get("LD_LIBRARY_PATH", ""),
        "ld_library_path_after": repair_env.get("LD_LIBRARY_PATH", ""),
        "candidate_library_dirs": discovery.get("candidate_library_dirs", []),
        "existing_library_dirs": existing_dirs,
        "ldconfig_libs": discovery.get("ldconfig_libs", {}),
        "nvidia_smi": discovery.get("nvidia_smi", {}),
        "package_metadata": discovery.get("package_metadata", {}),
        "environment": discovery.get("environment", {}),
        "libllama_path": discovery.get("libllama_path"),
        "libllama_ldd_before": discovery.get("libllama_ldd_before", ""),
        "libllama_ldd_after": discovery.get("libllama_ldd_after", ""),
    }


def _persistent_blockers_from(artifact: JsonDict, missing_cache_resolution: JsonDict | None = None) -> list[str]:
    """Collect exact terminal blockers from the repaired runtime and cache probes."""
    blockers: list[str] = []
    for blocker in artifact.get("blockers", []) or []:
        if isinstance(blocker, str) and blocker:
            blockers.append(blocker)
    for row in artifact.get("smoke_inference_results", []) or []:
        if not isinstance(row, Mapping):
            continue
        blocker = row.get("blocker")
        if isinstance(blocker, str) and blocker:
            blockers.append(blocker)
    if missing_cache_resolution:
        blocker = missing_cache_resolution.get("blocker")
        if isinstance(blocker, str) and blocker:
            blockers.append(blocker)

    deduped: list[str] = []
    seen: set[str] = set()
    for blocker in blockers:
        if blocker in seen:
            continue
        seen.add(blocker)
        deduped.append(blocker)
    return deduped


def attempt_missing_cache_resolution(
    *,
    missing_models: Sequence[str],
    allow_download: bool = True,
    hf_id: str = MIDDLE_MOE_HF_ID,
    filename: str = MIDDLE_MOE_Q4_FILENAME,
    **_: Any,
) -> JsonDict:
    """Try to make the missing middle-MoE GGUF visible to the local cache resolver.

    The function first asks HuggingFace's cache API for an offline hit so an
    interrupted prior download can be relinked without network traffic.  If the
    model is still missing and downloads are allowed, it performs one direct
    bounded-by-caller download attempt.  Any exception is recorded exactly.
    """
    if hf_id not in set(missing_models):
        return {
            "attempted": False,
            "hf_id": hf_id,
            "filename": filename,
            "status": "already_resolved_before_cache_attempt",
            "path": None,
            "blocker": None,
        }

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except Exception as exc:
        return {
            "attempted": True,
            "hf_id": hf_id,
            "filename": filename,
            "status": "blocked_huggingface_hub_unavailable",
            "path": None,
            "blocker": f"{type(exc).__name__}: {exc}",
        }

    offline_error: str | None = None
    try:
        path = hf_hub_download(repo_id=hf_id, filename=filename, local_files_only=True)
        return {
            "attempted": True,
            "hf_id": hf_id,
            "filename": filename,
            "status": "resolved_from_existing_local_cache",
            "path": path,
            "blocker": None,
        }
    except Exception as exc:
        offline_error = f"{type(exc).__name__}: {exc}"

    if not allow_download:
        return {
            "attempted": True,
            "hf_id": hf_id,
            "filename": filename,
            "status": "blocked_online_download_not_allowed",
            "path": None,
            "blocker": offline_error,
        }

    try:
        path = hf_hub_download(repo_id=hf_id, filename=filename, local_files_only=False)
    except Exception as exc:
        return {
            "attempted": True,
            "hf_id": hf_id,
            "filename": filename,
            "status": "blocked_online_download_failed",
            "path": None,
            "blocker": f"{type(exc).__name__}: {exc}",
            "offline_probe_error": offline_error,
        }
    return {
        "attempted": True,
        "hf_id": hf_id,
        "filename": filename,
        "status": "downloaded_to_local_cache",
        "path": path,
        "blocker": None,
        "offline_probe_error": offline_error,
    }


def _reproduce_exp1442_probe(*, project_root: Path, run_date: str, **_: Any) -> JsonDict:
    """Run the unmodified Exp 1442 gate before applying runtime-path repair."""
    output_path = project_root / DEFAULT_REPRODUCED_1442_PATH
    return preflight.run_experiment(
        project_root=project_root,
        run_date=run_date,
        output_path=output_path,
    )


def _run_repaired_preflight(*, project_root: Path, run_date: str, env: Mapping[str, str], **_: Any) -> JsonDict:
    """Run the existing live probe with the repaired loader environment."""

    def smoke_probe(model: JsonDict) -> JsonDict:
        def command_runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.run(command, **kwargs, env=dict(env))

        return preflight.run_live_probe_subprocess(
            model,
            command_runner=command_runner,
            python_executable=sys.executable,
        )

    return preflight.build_live_runtime_preflight_artifact(
        project_root=project_root,
        run_date=run_date,
        smoke_probe_fn=smoke_probe,
    )


def _in_progress_artifact(*, project_root: Path, run_date: str) -> JsonDict:
    """Create the required durable bootstrap artifact before repair work starts."""
    return {
        "status": "in_progress",
        "artifact": "experiment_1463_local_sota_gguf_runtime_repair",
        "run_date": run_date,
        "project_root": str(project_root),
        "model_specs": [dict(spec) for spec in preflight.MANDATED_MODEL_SPECS],
        "gpu_probe": {},
        "libcudart_resolution_attempted": {},
        "missing_cache_resolution_attempted": {},
        "models_found_in_cache": [],
        "models_missing_from_cache": [spec["hf_id"] for spec in preflight.MANDATED_MODEL_SPECS],
        "smoke_inference_results": [],
        "live_sota_model_inference_used": False,
        "local_sota_runtime_ready": False,
        "persistent_blockers": ["experiment_1463_runtime_repair_in_progress"],
        "honest_verdict": "in_progress",
    }


def build_runtime_repair_artifact(
    *,
    project_root: Path,
    run_date: str,
    reproduce_probe_fn: ProbeFn = _reproduce_exp1442_probe,
    repaired_probe_fn: ProbeFn = _run_repaired_preflight,
    cuda_discovery_fn: CudaDiscoveryFn = discover_cuda_runtime_state,
    missing_cache_resolution_fn: MissingCacheResolutionFn = attempt_missing_cache_resolution,
    base_env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Build the terminal Exp 1463 artifact from reproduced and repaired probes."""
    prior = reproduce_probe_fn(project_root=project_root, run_date=run_date)
    discovery = cuda_discovery_fn(project_root=project_root, prior_artifact=prior, env=base_env)
    missing_cache = missing_cache_resolution_fn(
        project_root=project_root,
        run_date=run_date,
        missing_models=prior.get("models_missing_from_cache", []),
    )
    repair_env = _repair_env_from_discovery(discovery, base_env=base_env)
    repaired = repaired_probe_fn(project_root=project_root, run_date=run_date, env=repair_env)

    live_success = repaired.get("local_sota_runtime_ready") is True and repaired.get(
        "live_sota_model_inference_used"
    ) is True
    persistent_blockers = [] if live_success else _persistent_blockers_from(repaired, missing_cache)

    artifact: JsonDict = {
        "status": "complete",
        "artifact": "experiment_1463_local_sota_gguf_runtime_repair",
        "run_date": run_date,
        "project_root": str(project_root),
        "schema_version": 1,
        "model_specs": [dict(spec) for spec in preflight.MANDATED_MODEL_SPECS],
        "reproduced_exp1442_probe": prior,
        "gpu_probe": repaired.get("gpu_probe", prior.get("gpu_probe", {})),
        "libcudart_resolution_attempted": _libcudart_resolution_attempt(
            prior,
            discovery,
            repair_env,
        ),
        "missing_cache_resolution_attempted": missing_cache,
        "models_found_in_cache": repaired.get("models_found_in_cache", []),
        "models_missing_from_cache": repaired.get("models_missing_from_cache", []),
        "smoke_inference_results": repaired.get("smoke_inference_results", []),
        "live_sota_model_inference_used": bool(live_success),
        "local_sota_runtime_ready": bool(live_success),
        "persistent_blockers": persistent_blockers,
        "honest_verdict": (
            "local_sota_runtime_ready" if live_success else "blocked_persistent_local_sota_runtime"
        ),
    }
    return artifact


def run_experiment(
    *,
    project_root: Path | None = None,
    run_date: str | None = None,
    output_path: Path | None = None,
    reproduce_probe_fn: ProbeFn = _reproduce_exp1442_probe,
    repaired_probe_fn: ProbeFn = _run_repaired_preflight,
    cuda_discovery_fn: CudaDiscoveryFn = discover_cuda_runtime_state,
    missing_cache_resolution_fn: MissingCacheResolutionFn = attempt_missing_cache_resolution,
    write_json_fn: WriteJsonFn = _write_json,
    base_env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Write the bootstrap artifact, perform repair probes, then write final JSON."""
    root = project_root or _repo_root()
    date = run_date or _utc_run_date()
    path = output_path or root / DEFAULT_ARTIFACT_PATH
    write_json_fn(path, _in_progress_artifact(project_root=root, run_date=date))
    artifact = build_runtime_repair_artifact(
        project_root=root,
        run_date=date,
        reproduce_probe_fn=reproduce_probe_fn,
        repaired_probe_fn=repaired_probe_fn,
        cuda_discovery_fn=cuda_discovery_fn,
        missing_cache_resolution_fn=missing_cache_resolution_fn,
        base_env=base_env,
    )
    write_json_fn(path, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not perform the online middle-MoE cache-fill attempt.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by conductor-style experiment runs."""
    args = _parse_args(argv)
    if args.no_download:
        missing_fn = lambda **kwargs: attempt_missing_cache_resolution(
            **kwargs,
            allow_download=False,
        )
    else:
        missing_fn = attempt_missing_cache_resolution
    run_experiment(
        run_date=args.run_date,
        output_path=args.output,
        missing_cache_resolution_fn=missing_fn,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through python -m.
    raise SystemExit(main())
