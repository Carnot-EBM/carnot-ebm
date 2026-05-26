"""Build the Exp 3151 live-inference authenticity preflight.

Spec refs: REQ-VERIFY-3151, SCENARIO-VERIFY-3151.

This module is a provenance gate, not a verifier benchmark. It exists because
live-call counters are not enough evidence that a large local model actually
loaded and generated text. The preflight records model paths, load command,
load timing, GPU/CPU substrate, transcript hashes, token counts, seed, and a
checksum before a later clean verifier rerun is allowed to rely on live SOTA
claims.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3151_live_inference_authenticity_preflight_v1"
SCHEMA = "carnot.live_inference_authenticity_preflight.v1"
OUTPUT_REL_PATH = Path("results/experiment_3151_live_inference_authenticity_preflight_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3151_live_inference_authenticity_preflight_v1.py"

EXP3123_REL_PATH = Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json")
EXP3139_REL_PATH = Path("results/experiment_3139_live_sota_verifier_rerun_v7.json")

DEFAULT_RANDOM_SEED = 20260526
DEFAULT_MINIMUM_DURATION_S = 60.0
DEFAULT_PROMPT = (
    "Exp 3151 authenticity smoke prompt. Reply with exactly one token: VALID."
)
DEFAULT_MAX_TOKENS = 8
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3151_live_inference_authenticity_preflight_v1.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3151_live_inference_authenticity_preflight_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/live_inference_authenticity_preflight_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_FIELDS = {
    "live_inference_authenticity_preflight_ready",
    "model_specs",
    "locally_usable_model_ids",
    "selected_model_ids",
    "preflight_passed",
    "live_call_count",
    "model_load_evidence",
    "transcript_hashes",
    "minimum_duration_requirement_s",
    "headline_claim_allowed",
    "blocked_reason",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
MANDATED_MODEL_POLICY: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "name": "Qwen3.6-35B-A3B",
        "role": "moe",
        "tier": "flagship_moe",
        "expected_quantization": "Q4_K_M",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "name": "Gemma4-31B-it",
        "role": "dense",
        "tier": "flagship_dense",
        "expected_quantization": "Q4_K_M",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "name": "Gemma4-26B-A4B-it",
        "role": "moe",
        "tier": "middle_moe",
        "expected_quantization": "Q4_K_M",
    },
)
SOURCE_REL_PATHS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("experiment_template_policy", Path("scripts/experiment_template.py"), True, "python"),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True, "text"),
    ("exp3123_sota_cache_manifest", EXP3123_REL_PATH, True, "json"),
    ("exp3139_prior_live_claim_artifact", EXP3139_REL_PATH, True, "json"),
    (
        "exp3151_module",
        Path("python/carnot/verify/live_inference_authenticity_preflight_v1.py"),
        False,
        "python",
    ),
    (
        "exp3151_script",
        Path("scripts/experiment_3151_live_inference_authenticity_preflight_v1.py"),
        False,
        "python",
    ),
    (
        "exp3151_tests",
        Path("tests/python/test_experiment_3151_live_inference_authenticity_preflight_v1.py"),
        False,
        "python",
    ),
)
SMOKE_WORKER_CODE = r'''
import argparse
import json
import time


def _extract_text(raw_response):
    if isinstance(raw_response, str):
        return raw_response
    if not isinstance(raw_response, dict):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    if "text" in first:
        return str(first.get("text") or "")
    message = first.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return ""


parser = argparse.ArgumentParser()
parser.add_argument("--exp3151-smoke-worker", action="store_true")
parser.add_argument("--model-path", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--max-tokens", type=int, default=8)
args = parser.parse_args()

started = time.monotonic()
load_wall_time_s = None
generation_wall_time_s = None
llm = None
try:
    from llama_cpp import Llama

    load_started = time.monotonic()
    llm = Llama(
        model_path=args.model_path,
        n_ctx=512,
        n_batch=64,
        n_ubatch=64,
        n_gpu_layers=-1,
        main_gpu=0,
        verbose=False,
    )
    load_wall_time_s = time.monotonic() - load_started
    generation_started = time.monotonic()
    raw = llm(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repeat_penalty=1.0,
        seed=args.seed,
    )
    generation_wall_time_s = time.monotonic() - generation_started
    output_text = _extract_text(raw)
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    print(
        json.dumps(
            {
                "ok": True,
                "runtime": "llama_cpp",
                "load_wall_time_s": round(load_wall_time_s, 6),
                "generation_wall_time_s": round(generation_wall_time_s, 6),
                "total_worker_wall_time_s": round(time.monotonic() - started, 6),
                "output_text": output_text,
                "usage": usage,
            },
            sort_keys=True,
        )
    )
except Exception as exc:
    print(
        json.dumps(
            {
                "ok": False,
                "runtime": "llama_cpp",
                "error": f"{type(exc).__name__}: {exc}",
                "load_wall_time_s": (
                    None if load_wall_time_s is None else round(load_wall_time_s, 6)
                ),
                "generation_wall_time_s": (
                    None if generation_wall_time_s is None else round(generation_wall_time_s, 6)
                ),
                "total_worker_wall_time_s": round(time.monotonic() - started, 6),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(1)
finally:
    close = getattr(llm, "close", None)
    if callable(close):
        close()
'''


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    selected_python: str | Path | None = None,
    command_runner: CommandRunner = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    minimum_duration_requirement_s: float = DEFAULT_MINIMUM_DURATION_S,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> JsonDict:
    """REQ-VERIFY-3151: build the authenticity preflight artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    runner = command_runner or run_command
    python_exe = str(selected_python) if selected_python is not None else selected_python_for(root_path)
    source_rows = source_artifacts(root_path)
    source_checksums = {
        row["path"]: row["sha256"] for row in source_rows if row.get("sha256")
    }
    exp3123 = read_json_object(root_path / EXP3123_REL_PATH)
    model_specs = inspect_model_specs(root_path, exp3123)
    usable_ids = [row["hf_id"] for row in model_specs if row["usable_locally"] is True]
    selected_model = first_usable_model(model_specs)
    selected_ids = [selected_model["hf_id"]] if selected_model else []
    substrate_probe = probe_substrate(python_exe, runner)
    smoke_result = maybe_run_smoke(
        selected_python=python_exe,
        selected_model=selected_model,
        substrate_probe=substrate_probe,
        command_runner=runner,
        random_seed=random_seed,
    )
    load_evidence = smoke_result["model_load_evidence"]
    transcript_hashes = smoke_result["transcript_hashes"]
    token_counts = smoke_result["token_counts"]
    live_call_count = len(transcript_hashes)
    finished = time.perf_counter() if now_s is None else float(now_s)
    duration_s = duration(start, finished)
    blocked_reason = determine_blocked_reason(
        usable_ids=usable_ids,
        substrate_probe=substrate_probe,
        smoke_blocker=smoke_result["runtime_blocker"],
        live_call_count=live_call_count,
        duration_s=duration_s,
        minimum_duration_requirement_s=minimum_duration_requirement_s,
    )
    preflight_passed = blocked_reason == "" and live_call_count > 0
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "live_inference_authenticity_preflight_ready": True,
        "model_specs": mark_selected_model(model_specs, selected_ids),
        "locally_usable_model_ids": usable_ids,
        "selected_model_ids": selected_ids if load_evidence["load_attempted"] else [],
        "preflight_passed": preflight_passed,
        "live_call_count": live_call_count,
        "model_load_evidence": load_evidence,
        "transcript_hashes": transcript_hashes,
        "token_counts": token_counts,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "minimum_duration_requirement_s": float(minimum_duration_requirement_s),
        "headline_claim_allowed": False,
        "blocked_reason": blocked_reason,
        "source_artifacts": source_rows,
        "source_checksums": source_checksums,
        "inference_substrate": inference_substrate(
            selected_python=python_exe,
            substrate_probe=substrate_probe,
            selected_model=selected_model,
            live_call_count=live_call_count,
            load_attempted=bool(load_evidence["load_attempted"]),
        ),
        "preflight_contract_for_exp3152": preflight_contract_for_exp3152(),
        "field_principles": field_principles(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration_s,
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    selected_python: str | Path | None = None,
    command_runner: CommandRunner = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    minimum_duration_requirement_s: float = DEFAULT_MINIMUM_DURATION_S,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    """Build and persist the Exp 3151 terminal JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        selected_python=selected_python,
        command_runner=command_runner,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
        minimum_duration_requirement_s=minimum_duration_requirement_s,
        random_seed=random_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def selected_python_for(root: Path) -> str:  # pragma: no cover - CLI environment fallback.
    """Return the project virtualenv Python when present."""

    candidate = root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
) -> JsonDict:  # pragma: no cover - subprocess glue covered by integration run.
    """Run one bounded local command and keep compact diagnostic evidence."""

    cmd = [str(part) for part in command]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
            check=False,
        )
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
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_summary": summarize(completed.stdout),
        "stderr_summary": summarize(completed.stderr),
    }


def summarize(text: str | None, *, limit: int = 2000) -> str:
    """Keep command evidence compact while preserving useful diagnostics."""

    value = text or ""
    return value if len(value) <= limit else value[:limit] + "...<truncated>"


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, treating malformed files as absent evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return every local file the preflight reads or cites."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_REL_PATHS:
        path = root / rel_path
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_json_object": (
                    bool(read_json_object(path)) if source_type == "json" else None
                ),
                "sha256": sha256_file(path),
            }
        )
    return rows


def sha256_file(path: Path) -> str | None:
    """Checksum a source file so the artifact traces exact local bytes."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_model_specs(root: Path, exp3123: Mapping[str, Any]) -> list[JsonDict]:
    """Combine mandated policy rows with current local path evidence."""

    inventory = [row for row in exp3123.get("cache_inventory", []) if isinstance(row, Mapping)]
    manifest_present_ids = {str(value) for value in exp3123.get("present_model_ids") or []}
    rows: list[JsonDict] = []
    for index, policy in enumerate(MANDATED_MODEL_POLICY):
        manifest_row = next((row for row in inventory if row.get("hf_id") == policy["hf_id"]), {})
        raw_path = manifest_row.get("path") or manifest_row.get("model_path") or manifest_row.get("resolved_path")
        evidence = path_evidence(root, raw_path)
        rows.append(
            {
                "hf_id": policy["hf_id"],
                "name": manifest_row.get("name") or policy["name"],
                "role": manifest_row.get("role") or policy["role"],
                "tier": policy["tier"],
                "policy_order": index,
                "expected_quantization": manifest_row.get("expected_quantization")
                or policy["expected_quantization"],
                "cache_status": manifest_row.get("cache_status") or "missing",
                "manifest_present": policy["hf_id"] in manifest_present_ids,
                "model_path": evidence["path"],
                "path_exists": evidence["exists"],
                "path_size_bytes": evidence["size_bytes"],
                "path_sha256_bounded": evidence["bounded_sha256"],
                "usable_locally": evidence["exists"] and int(evidence["size_bytes"] or 0) > 0,
                "selected_for_smoke": False,
                "legacy_small_model": False,
            }
        )
    return rows


def path_evidence(root: Path, raw_path: Any) -> JsonDict:
    """Return existence, size, and bounded hash evidence for a possible GGUF."""

    if not raw_path:
        return {"path": None, "exists": False, "size_bytes": None, "bounded_sha256": None}
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = root / path
    try:
        if not path.is_file():
            return {"path": str(path), "exists": False, "size_bytes": None, "bounded_sha256": None}
        stat = path.stat()
    except OSError:  # pragma: no cover - protects against disappearing files.
        return {"path": str(path), "exists": False, "size_bytes": None, "bounded_sha256": None}
    return {
        "path": str(path),
        "exists": stat.st_size > 0,
        "size_bytes": int(stat.st_size),
        "bounded_sha256": bounded_file_hash(path),
    }


def bounded_file_hash(path: Path) -> str:
    """Hash enough of a GGUF to identify it without reading all weights."""

    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def first_usable_model(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Select the first mandated local GGUF in policy order."""

    for row in model_specs:
        if row.get("usable_locally") is True:
            return dict(row)
    return None


def mark_selected_model(model_specs: Sequence[Mapping[str, Any]], selected_ids: Sequence[str]) -> list[JsonDict]:
    """Annotate which model was selected for the smoke call."""

    selected = set(selected_ids)
    return [dict(row) | {"selected_for_smoke": row.get("hf_id") in selected} for row in model_specs]


def probe_substrate(selected_python: str, command_runner: CommandRunner) -> JsonDict:
    """Record GPU and CPU substrate before any model load attempt."""

    torch_probe = torch_cuda_probe(selected_python, command_runner)
    smi = nvidia_smi_inventory(command_runner)
    gpu_count = max(int(torch_probe["cuda_device_count"]), len(smi["gpus"]))
    return {
        "selected_python": selected_python,
        "cuda_available": bool(torch_probe["cuda_available"]),
        "gpu_count": gpu_count,
        "torch_cuda_probe": torch_probe,
        "nvidia_smi_inventory": smi,
        "cpu_probe": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
        },
    }


def torch_cuda_probe(selected_python: str, command_runner: CommandRunner) -> JsonDict:
    """Probe CUDA through the same Python executable used for smoke loading."""

    command = [
        selected_python,
        "-c",
        "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())",
    ]
    result = command_runner(command, timeout_s=30)
    parts = stdout_of(result).strip().split()
    return {
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "torch_version": parts[0] if parts else None,
        "cuda_available": bool(
            result.get("returncode") == 0 and len(parts) >= 2 and parts[1] == "True"
        ),
        "cuda_device_count": int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0,
        "stdout_summary": summarize(stdout_of(result)),
        "stderr_summary": summarize(stderr_of(result)),
    }


def nvidia_smi_inventory(command_runner: CommandRunner) -> JsonDict:
    """Record visible NVIDIA GPUs without allocating model memory."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = command_runner(command, timeout_s=10)
    gpus: list[JsonDict] = []
    if result.get("returncode") == 0:
        for line in stdout_of(result).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 6 and parts[0].isdigit():
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mib": int_or_none(parts[2]),
                        "memory_used_mib": int_or_none(parts[3]),
                        "memory_free_mib": int_or_none(parts[4]),
                        "driver_version": parts[5],
                    }
                )
    return {
        "command": result.get("command", command),
        "returncode": result.get("returncode"),
        "available": bool(gpus),
        "gpus": gpus,
        "stdout_summary": summarize(stdout_of(result)),
        "stderr_summary": summarize(stderr_of(result)),
    }


def maybe_run_smoke(
    *,
    selected_python: str,
    selected_model: Mapping[str, Any] | None,
    substrate_probe: Mapping[str, Any],
    command_runner: CommandRunner,
    random_seed: int,
) -> JsonDict:
    """Run one bounded smoke call only when local model and GPU evidence exist."""

    default = default_smoke_result(selected_model)
    if selected_model is None:
        return default
    if substrate_probe.get("cuda_available") is not True or int(substrate_probe.get("gpu_count") or 0) <= 0:
        return default
    command = smoke_command(
        selected_python=selected_python,
        model_path=str(selected_model["model_path"]),
        random_seed=random_seed,
    )
    result = command_runner(command, timeout_s=600)
    payload, parse_error = parse_smoke_stdout(stdout_of(result))
    load_evidence = load_evidence_from_result(selected_model, command, result, payload, parse_error)
    if load_evidence["runtime_error"]:
        return default | {"model_load_evidence": load_evidence, "runtime_blocker": load_evidence["runtime_error"]}
    output_text = str(payload.get("output_text") or "")
    if not output_text.strip():  # pragma: no cover - defensive worker guard.
        load_evidence["runtime_error"] = "smoke worker returned empty output_text"
        return default | {"model_load_evidence": load_evidence, "runtime_blocker": load_evidence["runtime_error"]}
    transcript = transcript_hash_row(
        selected_model=selected_model,
        output_text=output_text,
        usage=payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {},
        random_seed=random_seed,
    )
    return {
        "model_load_evidence": load_evidence,
        "transcript_hashes": [transcript],
        "token_counts": token_counts_for(DEFAULT_PROMPT, output_text, payload.get("usage")),
        "runtime_blocker": "",
    }


def default_smoke_result(selected_model: Mapping[str, Any] | None) -> JsonDict:
    """Return the no-smoke evidence shape used by blocked preconditions."""

    return {
        "model_load_evidence": {
            "load_attempted": False,
            "runtime": "llama_cpp",
            "selected_model_id": selected_model.get("hf_id") if selected_model else None,
            "selected_model_path": selected_model.get("model_path") if selected_model else None,
            "path_exists": bool(selected_model and selected_model.get("path_exists")),
            "load_command": [],
            "load_command_sha256": "",
            "worker_code_sha256": sha256_text(SMOKE_WORKER_CODE),
            "returncode": None,
            "load_wall_time_s": None,
            "generation_wall_time_s": None,
            "total_worker_wall_time_s": None,
            "stdout_summary": "",
            "stderr_summary": "",
            "runtime_error": None,
        },
        "transcript_hashes": [],
        "token_counts": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "source": "none",
        },
        "runtime_blocker": "",
    }


def smoke_command(*, selected_python: str, model_path: str, random_seed: int) -> list[str]:
    """Build the exact local load command for the llama.cpp smoke worker."""

    return [
        selected_python,
        "-c",
        SMOKE_WORKER_CODE,
        "--exp3151-smoke-worker",
        "--model-path",
        model_path,
        "--seed",
        str(int(random_seed)),
        "--prompt",
        DEFAULT_PROMPT,
        "--max-tokens",
        str(DEFAULT_MAX_TOKENS),
    ]


def parse_smoke_stdout(stdout: str) -> tuple[JsonDict, str | None]:
    """Parse the worker's final JSON line."""

    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload), None
    return {}, "smoke worker did not emit a JSON object"  # pragma: no cover - malformed worker output.


def load_evidence_from_result(
    selected_model: Mapping[str, Any],
    command: Sequence[str],
    result: Mapping[str, Any],
    payload: Mapping[str, Any],
    parse_error: str | None,
) -> JsonDict:
    """Convert subprocess output into model-load evidence."""

    runtime_error = parse_error
    if result.get("returncode") != 0:
        runtime_error = str(payload.get("error") or stderr_of(result) or "smoke command failed")
    elif payload.get("ok") is not True:
        runtime_error = str(payload.get("error") or "smoke worker returned ok=false")
    return {
        "load_attempted": True,
        "runtime": str(payload.get("runtime") or "llama_cpp"),
        "selected_model_id": selected_model.get("hf_id"),
        "selected_model_path": selected_model.get("model_path"),
        "path_exists": bool(selected_model.get("path_exists")),
        "load_command": list(command),
        "load_command_sha256": stable_hash(list(command)),
        "worker_code_sha256": sha256_text(SMOKE_WORKER_CODE),
        "returncode": result.get("returncode"),
        "load_wall_time_s": float_or_none(payload.get("load_wall_time_s")),
        "generation_wall_time_s": float_or_none(payload.get("generation_wall_time_s")),
        "total_worker_wall_time_s": float_or_none(payload.get("total_worker_wall_time_s")),
        "stdout_summary": summarize(stdout_of(result)),
        "stderr_summary": summarize(stderr_of(result)),
        "runtime_error": runtime_error,
    }


def transcript_hash_row(
    *,
    selected_model: Mapping[str, Any],
    output_text: str,
    usage: Mapping[str, Any],
    random_seed: int,
) -> JsonDict:
    """Build replay-identifiable transcript evidence without long raw text."""

    token_counts = token_counts_for(DEFAULT_PROMPT, output_text, usage)
    prompt_hash = sha256_text(DEFAULT_PROMPT)
    response_hash = sha256_text(output_text)
    transcript_payload = {
        "model_id": selected_model.get("hf_id"),
        "model_path": selected_model.get("model_path"),
        "prompt_hash": prompt_hash,
        "response_hash": response_hash,
        "random_seed": int(random_seed),
        "token_counts": token_counts,
    }
    return {
        "model_id": selected_model.get("hf_id"),
        "prompt_hash": prompt_hash,
        "response_hash": response_hash,
        "transcript_sha256": stable_hash(transcript_payload),
        "prompt_token_count": token_counts["prompt_tokens"],
        "output_token_count": token_counts["completion_tokens"],
        "random_seed": int(random_seed),
    }


def token_counts_for(prompt: str, output_text: str, usage: Any) -> JsonDict:
    """Return token counts from llama.cpp usage or deterministic whitespace estimates."""

    if isinstance(usage, Mapping) and usage:
        prompt_tokens = int_or_none(usage.get("prompt_tokens")) or len(prompt.split())
        completion_tokens = int_or_none(usage.get("completion_tokens")) or len(output_text.split())
        total_tokens = int_or_none(usage.get("total_tokens")) or prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "source": "llama_cpp_usage",
        }
    prompt_tokens = len(prompt.split())
    completion_tokens = len(output_text.split())
    return {  # pragma: no cover - real-worker fallback when usage is absent.
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "source": "whitespace_estimate",
    }


def determine_blocked_reason(
    *,
    usable_ids: Sequence[str],
    substrate_probe: Mapping[str, Any],
    smoke_blocker: str,
    live_call_count: int,
    duration_s: float,
    minimum_duration_requirement_s: float,
) -> str:
    """Return the first conservative blocker for the preflight gate."""

    if not usable_ids:
        return "no mandated local SOTA GGUF path exists with nonzero size"
    if substrate_probe.get("cuda_available") is not True or int(substrate_probe.get("gpu_count") or 0) <= 0:
        return "CUDA/GPU substrate unavailable for mandated GGUF smoke call"
    if smoke_blocker:
        return smoke_blocker
    if live_call_count <= 0:
        return "smoke call did not produce transcript hash evidence"  # pragma: no cover
    if duration_s < float(minimum_duration_requirement_s):
        return (
            f"duration_s={duration_s} is shorter than minimum plausible "
            f"duration {float(minimum_duration_requirement_s)}"
        )
    return ""


def inference_substrate(
    *,
    selected_python: str,
    substrate_probe: Mapping[str, Any],
    selected_model: Mapping[str, Any] | None,
    live_call_count: int,
    load_attempted: bool,
) -> JsonDict:
    """Describe GPU/model/live status explicitly for downstream gates."""

    return {
        "kind": "live_inference_authenticity_preflight_v1",
        "selected_python": selected_python,
        "runtime": "llama_cpp",
        "gpu_probe": {
            "cuda_available": substrate_probe.get("cuda_available"),
            "gpu_count": substrate_probe.get("gpu_count"),
            "torch_cuda_probe": substrate_probe.get("torch_cuda_probe"),
            "nvidia_smi_inventory": substrate_probe.get("nvidia_smi_inventory"),
        },
        "cpu_probe": substrate_probe.get("cpu_probe"),
        "model_load_attempted": load_attempted,
        "executes_models": live_call_count > 0,
        "live_model_calls": int(live_call_count),
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "downloads_models": False,
        "legacy_small_model_used": False,
        "no_verifier_panel_scored": True,
        "selected_model_id": selected_model.get("hf_id") if selected_model else None,
        "selected_model_path": selected_model.get("model_path") if selected_model else None,
    }


def preflight_contract_for_exp3152() -> JsonDict:
    """Reusable downstream contract for the clean live SOTA verifier rerun."""

    return {
        "contract_id": "exp3152_live_sota_verifier_clean_rerun_preflight",
        "must_not_score_verifier_panel": True,
        "required_fields": sorted(REQUIRED_FIELDS),
        "required_live_evidence": [
            "selected_model_ids",
            "model_specs.path_exists",
            "model_load_evidence.load_command",
            "model_load_evidence.load_wall_time_s",
            "inference_substrate.gpu_probe",
            "transcript_hashes",
            "token_counts",
            "random_seed",
            "reproducibility_checksum",
            "minimum_duration_requirement_s",
        ],
        "gate": {
            "preflight_passed": True,
            "live_call_count_min": 1,
            "headline_claim_allowed_for_smoke_only": False,
            "legacy_small_models_allowed_for_headline": False,
            "minimum_duration_requirement_s": DEFAULT_MINIMUM_DURATION_S,
        },
    }


def field_principles() -> JsonDict:
    """Explain why the required fields exist."""

    return {
        "live_inference_authenticity_preflight_ready": (
            "live verifier reruns need a preflight artifact"
        ),
        "model_specs": "mandated local model policy must be visible",
        "locally_usable_model_ids": "actual model availability must be auditable",
        "selected_model_ids": "any smoke call must identify the selected model",
        "preflight_passed": "downstream gates need one conservative field",
        "live_call_count": "live evidence must not be inferred",
        "model_load_evidence": "claimed inference requires load evidence",
        "transcript_hashes": "outputs must be replay-identifiable without copying long text",
        "minimum_duration_requirement_s": "duration sanity must be explicit",
        "headline_claim_allowed": "smoke tests do not create headline evidence",
        "blocked_reason": "blocked preflights must be actionable",
        "source_artifacts": "preflight must trace to files",
        "inference_substrate": "GPU/model/live status must be explicit",
        "honest_verdict": "terminal verdict must expose complete or blocked state",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the preflight evidence that should reproduce across reruns."""

    payload = {
        "model_specs": artifact.get("model_specs"),
        "selected_model_ids": artifact.get("selected_model_ids"),
        "live_call_count": artifact.get("live_call_count"),
        "model_load_evidence": {
            key: artifact.get("model_load_evidence", {}).get(key)
            for key in (
                "selected_model_id",
                "selected_model_path",
                "path_exists",
                "load_command_sha256",
                "worker_code_sha256",
                "load_wall_time_s",
                "generation_wall_time_s",
                "runtime_error",
            )
        },
        "transcript_hashes": artifact.get("transcript_hashes"),
        "token_counts": artifact.get("token_counts"),
        "random_seed": artifact.get("random_seed"),
        "minimum_duration_requirement_s": artifact.get("minimum_duration_requirement_s"),
        "source_checksums": artifact.get("source_checksums"),
    }
    return stable_hash(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and fail-closed safety invariants."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3151 artifact missing required fields: {missing}")
    if artifact.get("headline_claim_allowed") is not False:
        raise ValueError("headline_claim_allowed must remain false for smoke preflights")
    if int(artifact.get("live_call_count", -1)) < 0:
        raise ValueError("live_call_count must be nonnegative")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("preflight_passed") is True:
        if int(artifact.get("live_call_count") or 0) <= 0:
            raise ValueError("passed preflight requires live call evidence")
        if not artifact.get("transcript_hashes"):
            raise ValueError("passed preflight requires transcript_hashes")
        if not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must use terminal success prefix")
    elif not verdict.startswith("blocked_") and verdict:
        raise ValueError("honest_verdict must be blocked_ when preflight does not pass")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return conductor-compatible terminal verdict wording."""

    if artifact.get("preflight_passed") is True:
        return (
            "complete: live_inference_authenticity_preflight_ready=true; "
            "preflight_passed=true; "
            f"live_call_count={artifact.get('live_call_count')}; "
            "headline_claim_allowed=false"
        )
    reason = str(artifact.get("blocked_reason") or "preflight did not pass")
    if "no mandated local SOTA GGUF" in reason:
        prefix = "blocked_no_mandated_sota_gguf"
    elif "CUDA/GPU substrate" in reason:
        prefix = "blocked_gpu_substrate"
    elif "shorter than minimum" in reason:
        prefix = "blocked_duration_too_short"
    else:
        prefix = "blocked_smoke_runtime"
    return f"{prefix}: preflight_passed=false; live_call_count={artifact.get('live_call_count')}; detail={reason}"


def stdout_of(result: Mapping[str, Any]) -> str:
    """Return command stdout, falling back to compact summaries."""

    return str(result.get("stdout") or result.get("stdout_summary") or "")


def stderr_of(result: Mapping[str, Any]) -> str:
    """Return command stderr, falling back to compact summaries."""

    return str(result.get("stderr") or result.get("stderr_summary") or "")


def int_or_none(value: Any) -> int | None:
    """Parse integers from command payloads without raising."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    """Parse floats from command payloads without raising."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sha256_text(text: str) -> str:
    """Return a SHA-256 digest for a string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    """Hash JSON-serializable evidence with canonical key ordering."""

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def duration(started_s: float, finished_s: float) -> float:
    """Return a nonnegative elapsed duration."""

    return round(max(0.0, float(finished_s) - float(started_s)), 6)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = build_artifact(REPO_ROOT)
    output = REPO_ROOT / OUTPUT_REL_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["live_inference_authenticity_preflight_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
