"""Exp 3123 local SOTA cache/precondition manifest v2.

**Researcher summary:**
    This module writes a machine-readable authority that downstream live-LLM
    tasks must consult before making headline claims.  It answers three
    separate questions without loading a model: which mandated SOTA GGUF files
    are really present on disk, whether the shared ``cached_sota_pair()``
    helper can supply a two-model pair, and whether the local GPU preflight is
    visible enough for a live attempt.

**Detailed explanation for engineers:**
    A HuggingFace cache directory can exist even when the model file is absent:
    failed lookups leave `.no_exist` sentinels and zero-byte marker files.  The
    manifest therefore treats only nonzero GGUF files whose names match a
    mandated model family as present.  Legacy small models are recorded only as
    smoke-test options.  The code performs no downloads and no inference; it
    runs only filesystem inspection and lightweight diagnostic commands.

Spec: REQ-INFER-SOTA-023,
      SCENARIO-INFER-SOTA-023-001,
      SCENARIO-INFER-SOTA-023-002,
      SCENARIO-INFER-SOTA-023-003
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
ClockFn = Callable[[], float]

DEFAULT_ARTIFACT_PATH = Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json")
MANDATORY_HEADLINE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SMOKE_TEST_MODEL_IDS: tuple[str, ...] = (
    "google/gemma-4-E4B-it",
    "Qwen/Qwen3.5-0.8B",
)
DEFAULT_SOURCE_PATHS: tuple[Path, ...] = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("scripts/experiment_template.py"),
    Path("results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json"),
    Path("results/experiment_3120_cross_corpus_matrix_v24.json"),
    Path("results/experiment_3121_capstone_v290.json"),
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "sota_cache_manifest_v2_ready",
    "mandatory_headline_model_ids",
    "present_model_ids",
    "missing_model_ids",
    "selected_headline_model_ids",
    "cached_sota_pair_available",
    "any_single_sota_available",
    "headline_claim_allowed",
    "gpu_preflight",
    "downstream_usage",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
_MODEL_BY_ID = {model["hf_id"]: model for model in SOTA_GGUF_MODELS}
_QUANTIZATION_PREFERENCE: tuple[str, ...] = (
    "UD-Q4_K_M",
    "Q4_K_M",
    "UD-Q5_K_M",
    "Q5_K_M",
    "UD-Q4_K_S",
    "Q8_0",
    "MXFP4_MOE",
    "BF16",
)


def _repo_root() -> Path:  # pragma: no cover - exercised by direct CLI use.
    """Return the repository root for CLI invocations and conductor runs."""
    return Path(os.environ.get("CARNOT_REPO_ROOT", Path.cwd())).resolve()


def _run_date() -> str:  # pragma: no cover - wall-clock fallback for CLI use.
    """Return the current UTC date as YYYYMMDD."""
    return _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%d")


def _selected_python(project_root: Path) -> str:  # pragma: no cover - trivial CLI fallback.
    """Use the project virtualenv interpreter when it exists."""
    candidate = project_root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str | None, *, limit: int = 2000) -> str:
    """Keep command evidence compact while preserving the useful prefix."""
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
) -> JsonDict:  # pragma: no cover - integration fallback, injected in tests.
    """Run a local preflight command and return structured diagnostic evidence."""
    cmd = [str(part) for part in command]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
        )
    except Exception as exc:
        return {
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "stdout_summary": "",
            "stderr_summary": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_summary": _summarize(completed.stdout),
        "stderr_summary": _summarize(completed.stderr),
    }


def _stdout(result: Mapping[str, Any]) -> str:
    return str(result.get("stdout") or result.get("stdout_summary") or "")


def _stderr(result: Mapping[str, Any]) -> str:
    return str(result.get("stderr") or result.get("stderr_summary") or "")


def _torch_cuda_probe(selected_python: str, *, command_runner: CommandRunner) -> JsonDict:
    """Probe CUDA through the exact Python executable downstream tasks use."""
    command = [
        selected_python,
        "-c",
        "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())",
    ]
    result = command_runner(command, timeout_s=30)
    parts = _stdout(result).strip().split()
    return {
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "python_executable": selected_python,
        "torch_version": parts[0] if parts else None,
        "cuda_available": bool(
            result.get("returncode") == 0 and len(parts) >= 2 and parts[1] == "True"
        ),
        "cuda_device_count": int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0,
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
    }


def _nvidia_smi_inventory(*, command_runner: CommandRunner) -> JsonDict:
    """Record visible NVIDIA GPUs without allocating model memory."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = command_runner(command, timeout_s=10)
    gpus: list[JsonDict] = []
    if result.get("returncode") == 0:
        for line in _stdout(result).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 6 and parts[0].isdigit() and parts[2].isdigit():
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mib": int(parts[2]),
                        "memory_used_mib": int(parts[3]) if parts[3].isdigit() else None,
                        "memory_free_mib": int(parts[4]) if parts[4].isdigit() else None,
                        "driver_version": parts[5],
                    }
                )
    return {
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "available": bool(gpus),
        "gpus": gpus,
        "stdout_summary": _summarize(_stdout(result)),
        "stderr_summary": _summarize(_stderr(result)),
    }


def _gpu_preflight(
    *,
    selected_python: str,
    command_runner: CommandRunner,
) -> JsonDict:
    """Build the explicit compute preflight without importing model runtimes."""
    torch_probe = _torch_cuda_probe(selected_python, command_runner=command_runner)
    smi = _nvidia_smi_inventory(command_runner=command_runner)
    gpu_count = max(int(torch_probe["cuda_device_count"]), len(smi["gpus"]))
    return {
        "selected_python": selected_python,
        "cuda_available": bool(torch_probe["cuda_available"]),
        "gpu_count": gpu_count,
        "nvidia_smi_available": bool(smi["available"]),
        "torch_cuda_probe": torch_probe,
        "nvidia_smi_inventory": smi,
        "no_model_loaded": True,
        "no_inference_run": True,
    }


def _cache_roots(project_root: Path, env: Mapping[str, str]) -> JsonDict:
    """Return local roots that may contain mandated GGUF files."""
    hf_cache_raw = env.get("HUGGINGFACE_HUB_CACHE") or (
        str(Path(env["HF_HOME"]).expanduser() / "hub")
        if env.get("HF_HOME")
        else str(Path.home() / ".cache" / "huggingface" / "hub")
    )
    hf_cache = Path(hf_cache_raw).expanduser()
    return {
        "huggingface_hub_cache": str(hf_cache),
        "huggingface_hub_cache_exists": hf_cache.exists(),
        "project_models": str(project_root / "models"),
        "project_models_exists": (project_root / "models").exists(),
    }


def _model_filename_token(hf_id: str) -> str:
    """Return the filename stem that identifies a mandated model family."""
    return hf_id.split("/", 1)[-1].removesuffix("-GGUF").lower()


def _candidate_dirs(project_root: Path, hf_cache: Path, hf_id: str) -> list[tuple[Path, str]]:
    """List HuggingFace and project-local directories worth scanning."""
    basename = hf_id.split("/", 1)[-1]
    stripped = basename.removesuffix("-GGUF")
    models_root = project_root / "models"
    return [
        (hf_cache / f"models--{hf_id.replace('/', '--')}", "huggingface_hub_cache"),
        (models_root / stripped, "project_models"),
        (models_root / basename, "project_models"),
        (models_root / stripped.lower(), "project_models"),
        (models_root / basename.lower(), "project_models"),
    ]


def _candidate_record(path: Path, hf_id: str, source: str) -> JsonDict:
    """Convert a local GGUF path into cache-inventory evidence."""
    try:
        size = int(path.stat().st_size)
        exists = path.exists()
    except OSError:  # pragma: no cover - defensive for disappearing cache entries.
        size = 0
        exists = False
    filename = path.name.lower()
    model_token = _model_filename_token(hf_id)
    zero_or_noexist = size == 0 or ".no_exist" in str(path)
    usable = bool(
        exists
        and not zero_or_noexist
        and model_token in filename
        and "mmproj" not in filename
        and "imatrix" not in filename
    )
    return {
        "path": str(path),
        "source": source,
        "exists": exists,
        "size_bytes": size,
        "usable_candidate": usable,
        "is_zero_byte_marker": zero_or_noexist,
    }


def _candidate_records(project_root: Path, roots: Mapping[str, Any], hf_id: str) -> list[JsonDict]:
    """Search local HF snapshots and project model directories for one model."""
    hf_cache = Path(str(roots["huggingface_hub_cache"]))
    records: dict[str, JsonDict] = {}
    for directory, source in _candidate_dirs(project_root, hf_cache, hf_id):
        if directory.exists():
            for path in directory.rglob("*.gguf"):
                records.setdefault(str(path), _candidate_record(path, hf_id, source))
    return list(records.values())


def _quantization_suffix(path: str | None) -> str | None:
    """Extract a visible quantization token from a GGUF filename."""
    if path is None:
        return None
    filename = Path(path).name.lower()
    for token in _QUANTIZATION_PREFERENCE:
        if token.lower() in filename:
            return token
    return "unknown"


def _select_candidate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Select the preferred nonzero GGUF candidate deterministically."""
    usable = [record for record in records if record.get("usable_candidate")]
    if not usable:
        return None
    for token in _QUANTIZATION_PREFERENCE:
        matches = [
            record for record in usable if token.lower() in Path(str(record["path"])).name.lower()
        ]
        if matches:
            return max(matches, key=lambda record: int(record.get("size_bytes") or 0))
    return max(usable, key=lambda record: str(record["path"]))


def _inspect_mandated_cache(project_root: Path, env: Mapping[str, str]) -> list[JsonDict]:
    """Inspect every mandated model and record exact resolved path or missing status."""
    roots = _cache_roots(project_root, env)
    rows: list[JsonDict] = []
    for hf_id in MANDATORY_HEADLINE_MODEL_IDS:
        records = _candidate_records(project_root, roots, hf_id)
        selected = _select_candidate(records)
        selected_path = str(selected["path"]) if selected is not None else None
        spec = _MODEL_BY_ID.get(hf_id, {})
        rows.append(
            {
                "hf_id": hf_id,
                "name": spec.get("name"),
                "role": spec.get("role"),
                "expected_quantization": spec.get("quantization"),
                "cache_status": "resolved" if selected_path else "missing",
                "path": selected_path,
                "resolved_path": str(Path(selected_path).resolve()) if selected_path else None,
                "observed_quantization": _quantization_suffix(selected_path),
                "candidate_count": len(records),
                "usable_candidate_count": sum(1 for record in records if record["usable_candidate"]),
                "zero_byte_marker_count": sum(1 for record in records if record["is_zero_byte_marker"]),
                "candidate_paths": [record["path"] for record in records],
                "missing_status": None if selected_path else "missing_or_zero_byte_only",
            }
        )
    return rows


def _loadable_pair(model_specs: Any) -> bool:
    """Return whether cached_sota_pair yielded two concrete mandated GGUF specs."""
    return bool(
        isinstance(model_specs, list)
        and len(model_specs) >= 2
        and all(
            isinstance(spec, dict)
            and spec.get("hf_id") in MANDATORY_HEADLINE_MODEL_IDS
            and spec.get("model_path")
            for spec in model_specs[:2]
        )
    )


def _exercise_cached_sota_pair(cached_pair_fn: CachedPairFn) -> JsonDict:
    """Call the shared pair helper and preserve errors as data."""
    try:
        result = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception as exc:  # pragma: no cover - defensive around injected helpers.
        return {
            "called": True,
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
            "returned_two_loadable_specs": False,
        }
    return {
        "called": True,
        "result": result,
        "error": None,
        "returned_two_loadable_specs": _loadable_pair(result),
    }


def _sha256_file(path: Path) -> str | None:
    """Hash a source artifact without changing it."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_artifact_records(project_root: Path, source_paths: Sequence[Path]) -> list[JsonDict]:
    """Trace the manifest to the checked-in files it consumed."""
    rows: list[JsonDict] = []
    for relpath in source_paths:
        path = project_root / relpath
        sha = _sha256_file(path)
        rows.append(
            {
                "path": str(relpath),
                "present": path.exists(),
                "sha256": sha,
                "source_type": "json" if relpath.suffix == ".json" else "text",
                "readable_json_object": _readable_json_object(path) if relpath.suffix == ".json" else None,
            }
        )
    return rows


def _readable_json_object(path: Path) -> bool:
    """Return whether a JSON source parses as an object."""
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    except (OSError, json.JSONDecodeError):
        return False


def _prior_artifact_observations(project_root: Path, source_paths: Sequence[Path]) -> list[JsonDict]:
    """Extract prior model availability fields from checked-in JSON artifacts."""
    observations: list[JsonDict] = []
    watched_keys = {
        "present_model_ids",
        "missing_model_ids",
        "selected_headline_model_ids",
        "cached_sota_pair_available",
        "headline_claim_allowed",
    }
    for relpath in source_paths:
        if relpath.suffix != ".json":
            continue
        path = project_root / relpath
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and any(key in payload for key in watched_keys):
            observations.append(
                {
                    "path": str(relpath),
                    "present_model_ids": list(payload.get("present_model_ids") or []),
                    "missing_model_ids": list(payload.get("missing_model_ids") or []),
                    "selected_headline_model_ids": list(
                        payload.get("selected_headline_model_ids") or []
                    ),
                    "cached_sota_pair_available": payload.get("cached_sota_pair_available"),
                    "headline_claim_allowed": payload.get("headline_claim_allowed"),
                }
            )
    return observations


def _selected_headline_ids(
    *,
    present_model_ids: Sequence[str],
    pair_result: Mapping[str, Any],
) -> list[str]:
    """Choose auditable headline candidates from pair specs or single-model cache."""
    if pair_result.get("returned_two_loadable_specs"):
        selected = [
            str(spec["hf_id"])
            for spec in pair_result.get("result", [])[:2]
            if isinstance(spec, dict) and spec.get("hf_id") in present_model_ids
        ]
        if selected:
            return selected
    return list(present_model_ids)


def _preconditions_checked(
    *,
    gpu_preflight: Mapping[str, Any],
    pair_result: Mapping[str, Any],
    present_model_ids: Sequence[str],
) -> list[JsonDict]:
    """Build the gating checklist downstream tasks can replay."""
    return [
        {
            "resource": "mandated_sota_gguf_cache",
            "available": bool(present_model_ids),
            "detail": f"present_model_ids={list(present_model_ids)}",
        },
        {
            "resource": "cached_sota_pair",
            "available": bool(pair_result.get("returned_two_loadable_specs")),
            "detail": pair_result.get("result") if pair_result.get("error") is None else pair_result["error"],
        },
        {
            "resource": "venv_torch_cuda",
            "available": bool(gpu_preflight.get("cuda_available")),
            "detail": gpu_preflight.get("torch_cuda_probe", {}).get("stdout_summary")
            or gpu_preflight.get("torch_cuda_probe", {}).get("stderr_summary"),
            "command": gpu_preflight.get("torch_cuda_probe", {}).get("command"),
        },
        {
            "resource": "nvidia_smi_inventory",
            "available": bool(gpu_preflight.get("nvidia_smi_available")),
            "detail": gpu_preflight.get("nvidia_smi_inventory", {}).get("gpus", []),
            "command": gpu_preflight.get("nvidia_smi_inventory", {}).get("command"),
        },
    ]


def _downstream_usage(
    *,
    present_model_ids: Sequence[str],
    selected_headline_model_ids: Sequence[str],
    cached_sota_pair_available: bool,
    headline_claim_allowed: bool,
) -> JsonDict:
    """Define how live, pair, solver-only, and smoke-only tasks consume v2."""
    return {
        "live_llm_headline_tasks": {
            "requires_present_mandated_model": True,
            "minimum_attempted_present_mandated_models": 1,
            "allowed_model_ids": list(present_model_ids),
            "selected_headline_model_ids": list(selected_headline_model_ids),
            "must_check_gpu_preflight": True,
            "headline_claim_allowed": headline_claim_allowed,
            "required_action": (
                "attempt_at_least_one_present_mandated_model_or_write_blocked_diagnostic_artifact"
            ),
            "when_no_present_mandated_model": (
                "write_blocked_model_cache_or_diagnostic_artifact_before_verifier_repair_self_learning_or_energy_sidecar"
            ),
        },
        "pair_or_comparative_headline_tasks": {
            "requires_cached_sota_pair_available": True,
            "cached_sota_pair_available": cached_sota_pair_available,
            "minimum_present_mandated_models": 2,
            "headline_claim_allowed": bool(headline_claim_allowed and cached_sota_pair_available),
        },
        "solver_only_tasks": {
            "allowed_without_model_availability": True,
            "live_llm_headline_claim_allowed": False,
            "required_action": (
                "may_run_exact_solver_or_test_oracle_work_but_must_not_report_live_llm_headline_evidence"
            ),
        },
        "legacy_small_models": {
            "model_ids": list(SMOKE_TEST_MODEL_IDS),
            "allowed_only_for_cpu_smoke_tests": True,
            "headline_claim_allowed": False,
        },
    }


def _field_principles() -> JsonDict:
    """Record why each required manifest field exists."""
    return {
        "sota_cache_manifest_v2_ready": "downstream tasks need a machine-readable cache authority",
        "mandatory_headline_model_ids": "SOTA policy must be explicit",
        "present_model_ids": "local availability must be measured",
        "missing_model_ids": "unavailable models must not be silently ignored",
        "selected_headline_model_ids": "actual headline models must be auditable",
        "cached_sota_pair_available": "pair availability must be separated from solver readiness",
        "any_single_sota_available": "single-model bounded attempts must be distinguished from pair evidence",
        "headline_claim_allowed": "cache and compute gaps must block live model claims, not solver-only work",
        "gpu_preflight": "compute-bound tasks need explicit preconditions",
        "downstream_usage": "later tasks must know how to consume the manifest",
        "source_artifacts": "manifest must trace to concrete files",
        "inference_substrate": "preflight work must declare no live inference unless it happens",
        "honest_verdict": "terminal verdict must use complete/success/passed/shipped or blocked prefix",
    }


def _honest_verdict(
    *,
    any_single_sota_available: bool,
    cached_sota_pair_available: bool,
    headline_claim_allowed: bool,
    missing_model_count: int,
    gpu_preflight: Mapping[str, Any],
) -> str:
    """Map precondition state to a terminal verdict."""
    if not any_single_sota_available:
        return "blocked_model_cache: no mandated SOTA GGUF resolved locally"
    if not gpu_preflight.get("cuda_available"):
        return "blocked_cuda: selected Python did not report CUDA availability"
    return (
        "complete: sota_cache_manifest_v2_ready=true; "
        f"cached_sota_pair_available={cached_sota_pair_available}; "
        f"any_single_sota_available={any_single_sota_available}; "
        f"headline_claim_allowed={headline_claim_allowed}; "
        f"missing_model_ids={missing_model_count}"
    )


def _reproducibility_checksum(
    *,
    cache_inventory: Sequence[Mapping[str, Any]],
    source_artifacts: Sequence[Mapping[str, Any]],
    gpu_preflight: Mapping[str, Any],
    pair_result: Mapping[str, Any],
) -> str:
    """Hash deterministic manifest inputs without reading large model blobs."""
    payload = {
        "mandatory_headline_model_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
        "cache_inventory": [
            {
                "hf_id": row.get("hf_id"),
                "path": row.get("path"),
                "size_bytes": _size_for_checksum(row.get("path")),
                "cache_status": row.get("cache_status"),
            }
            for row in cache_inventory
        ],
        "source_artifacts": [
            {"path": row.get("path"), "sha256": row.get("sha256")} for row in source_artifacts
        ],
        "gpu_preflight": {
            "cuda_available": gpu_preflight.get("cuda_available"),
            "gpu_count": gpu_preflight.get("gpu_count"),
            "nvidia_smi_available": gpu_preflight.get("nvidia_smi_available"),
        },
        "cached_sota_pair_available": pair_result.get("returned_two_loadable_specs"),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _size_for_checksum(path: Any) -> int | None:
    """Return a file size for reproducibility without hashing huge GGUFs."""
    if not path:
        return None
    try:
        return int(Path(str(path)).stat().st_size)
    except OSError:  # pragma: no cover - defensive for disappearing cache entries.
        return None


def build_sota_cache_manifest_v2(
    *,
    project_root: str | Path,
    run_date: str,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    source_paths: Sequence[Path] = DEFAULT_SOURCE_PATHS,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3123 manifest without downloading or running models."""
    root = Path(project_root).resolve()
    runtime_env = dict(os.environ if env is None else env)
    python_executable = str(selected_python) if selected_python is not None else _selected_python(root)
    started = monotonic()
    cache_inventory = _inspect_mandated_cache(root, runtime_env)
    present_model_ids = [
        str(row["hf_id"]) for row in cache_inventory if row.get("cache_status") == "resolved"
    ]
    missing_model_ids = [
        model_id for model_id in MANDATORY_HEADLINE_MODEL_IDS if model_id not in present_model_ids
    ]
    pair_result = _exercise_cached_sota_pair(cached_pair_fn)
    cached_sota_pair_available = bool(pair_result["returned_two_loadable_specs"])
    any_single_sota_available = bool(present_model_ids)
    selected_headline_model_ids = _selected_headline_ids(
        present_model_ids=present_model_ids,
        pair_result=pair_result,
    )
    gpu_preflight = _gpu_preflight(
        selected_python=python_executable,
        command_runner=command_runner,
    )
    headline_claim_allowed = bool(
        any_single_sota_available
        and gpu_preflight["cuda_available"]
        and int(gpu_preflight["gpu_count"]) > 0
    )
    source_artifacts = _source_artifact_records(root, source_paths)
    checksum = _reproducibility_checksum(
        cache_inventory=cache_inventory,
        source_artifacts=source_artifacts,
        gpu_preflight=gpu_preflight,
        pair_result=pair_result,
    )
    finished = monotonic()
    return {
        "artifact": "experiment_3123_sota_cache_preconditions_manifest_v2",
        "schema": "carnot.sota_cache_preconditions_manifest.v2",
        "run_date": run_date,
        "sota_cache_manifest_v2_ready": True,
        "mandatory_headline_model_ids": list(MANDATORY_HEADLINE_MODEL_IDS),
        "present_model_ids": present_model_ids,
        "missing_model_ids": missing_model_ids,
        "selected_headline_model_ids": selected_headline_model_ids,
        "cached_sota_pair_available": cached_sota_pair_available,
        "any_single_sota_available": any_single_sota_available,
        "headline_claim_allowed": headline_claim_allowed,
        "gpu_preflight": gpu_preflight,
        "downstream_usage": _downstream_usage(
            present_model_ids=present_model_ids,
            selected_headline_model_ids=selected_headline_model_ids,
            cached_sota_pair_available=cached_sota_pair_available,
            headline_claim_allowed=headline_claim_allowed,
        ),
        "source_artifacts": source_artifacts,
        "inference_substrate": {
            "kind": "local_filesystem_cache_and_gpu_preflight",
            "cache_probe_performed": True,
            "executes_models": False,
            "executes_solvers": False,
            "executes_verifiers": False,
            "executes_repairs": False,
            "executes_self_learning": False,
            "executes_energy_sidecar": False,
            "downloads_models": False,
            "no_live_llm_inference": True,
            "live_model_calls": 0,
        },
        "cache_roots": _cache_roots(root, runtime_env),
        "cache_inventory": cache_inventory,
        "cached_sota_pair_result": pair_result,
        "smoke_test_model_ids": list(SMOKE_TEST_MODEL_IDS),
        "preconditions_checked": _preconditions_checked(
            gpu_preflight=gpu_preflight,
            pair_result=pair_result,
            present_model_ids=present_model_ids,
        ),
        "prior_artifact_model_id_observations": _prior_artifact_observations(
            root, source_paths
        ),
        "field_principles": _field_principles(),
        "reproducibility_checksum": checksum,
        "tests_run": list(tests_run or []),
        "duration_s": round(finished - started, 6),
        "honest_verdict": _honest_verdict(
            any_single_sota_available=any_single_sota_available,
            cached_sota_pair_available=cached_sota_pair_available,
            headline_claim_allowed=headline_claim_allowed,
            missing_model_count=len(missing_model_ids),
            gpu_preflight=gpu_preflight,
        ),
    }


def run_experiment(
    *,
    project_root: str | Path,
    run_date: str,
    output_path: str | Path = DEFAULT_ARTIFACT_PATH,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    source_paths: Sequence[Path] = DEFAULT_SOURCE_PATHS,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and persist the v2 manifest as stable JSON."""
    artifact = build_sota_cache_manifest_v2(
        project_root=project_root,
        run_date=run_date,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        cached_pair_fn=cached_pair_fn,
        source_paths=source_paths,
        monotonic=monotonic,
        tests_run=tests_run,
    )
    output = Path(output_path)
    if not output.is_absolute():
        output = Path(project_root) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    """CLI entrypoint used by conductor tasks and manual local refreshes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--run-date", default=_run_date())
    parser.add_argument("--project-root", type=Path, default=_repo_root())
    parser.add_argument("--selected-python", default=None)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        project_root=args.project_root,
        run_date=args.run_date,
        output_path=args.output,
        selected_python=args.selected_python,
    )
    print(json.dumps({"output": str(args.output), "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
