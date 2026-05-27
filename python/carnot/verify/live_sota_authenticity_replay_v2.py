"""Build the Exp 3165 live SOTA authenticity replay v2 artifact.

Spec refs: REQ-VERIFY-3165, SCENARIO-VERIFY-3165.

This module is a gate for later live verifier reruns. It does not score a
verifier panel or claim benchmark quality. Its job is narrower: prove that a
mandated local SOTA GGUF can be loaded and can produce fresh, replayable smoke
transcripts under the duration-corrected Exp 3164 contract. When any
precondition is absent, the module writes a complete blocked artifact rather
than inferring live evidence from stale or partial metadata.
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
RUN_DATE = "20260527"
ARTIFACT = "experiment_3165_live_sota_authenticity_replay_v2"
SCHEMA = "carnot.live_sota_authenticity_replay.v2"
OUTPUT_REL_PATH = Path("results/experiment_3165_live_sota_authenticity_replay_v2.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3165_live_sota_authenticity_replay_v2.py"

EXP3164_REL_PATH = Path("results/experiment_3164_duration_corrected_authenticity_contract_v2.json")
EXP3151_REL_PATH = Path("results/experiment_3151_live_inference_authenticity_preflight_v1.json")
EXP3123_REL_PATH = Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json")

DEFAULT_RANDOM_SEED = 20260527
DEFAULT_MAX_TOKENS = 8
DEFAULT_PROMPTS = (
    "Exp 3165 authenticity replay prompt A. Reply with exactly one token: READY.",
    "Exp 3165 authenticity replay prompt B. Reply with exactly one token: VERIFIED.",
)
MINIMUM_DISTINCT_SMOKE_CALLS = 2
IMPOSSIBLE_COMPLETION_TOKENS_PER_SECOND = 500.0

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
    "live_sota_authenticity_replay_v2_ready",
    "model_specs",
    "locally_usable_model_ids",
    "selected_model_ids",
    "unavailable_model_ids",
    "preflight_passed",
    "live_call_count",
    "model_load_evidence",
    "prompt_hashes",
    "transcript_hashes",
    "token_counts",
    "measured_work_policy_passed",
    "fake_evidence_rejection_passed",
    "headline_claim_allowed",
    "blocked_reason",
    "random_seed",
    "reproducibility_checksum",
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
    ("exp3164_v2_contract", EXP3164_REL_PATH, True, "json"),
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("experiment_template_policy", Path("scripts/experiment_template.py"), True, "python"),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True, "text"),
    ("exp3151_duration_failed_preflight", EXP3151_REL_PATH, True, "json"),
    ("exp3123_sota_cache_manifest", EXP3123_REL_PATH, True, "json"),
    (
        "exp3165_module",
        Path("python/carnot/verify/live_sota_authenticity_replay_v2.py"),
        False,
        "python",
    ),
    (
        "exp3165_script",
        Path("scripts/experiment_3165_live_sota_authenticity_replay_v2.py"),
        False,
        "python",
    ),
    (
        "exp3165_tests",
        Path("tests/python/test_experiment_3165_live_sota_authenticity_replay_v2.py"),
        False,
        "python",
    ),
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3165_live_sota_authenticity_replay_v2.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3165_live_sota_authenticity_replay_v2.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/live_sota_authenticity_replay_v2.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
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
parser.add_argument("--exp3165-smoke-worker", action="store_true")
parser.add_argument("--model-path", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--prompts-json", required=True)
parser.add_argument("--max-tokens", type=int, default=8)
args = parser.parse_args()
prompts = json.loads(args.prompts_json)

started = time.monotonic()
load_wall_time_s = None
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
    calls = []
    for index, prompt in enumerate(prompts):
        generation_started = time.monotonic()
        raw = llm(
            prompt,
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repeat_penalty=1.0,
            seed=args.seed + index,
        )
        generation_wall_time_s = time.monotonic() - generation_started
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        calls.append(
            {
                "prompt": prompt,
                "seed": args.seed + index,
                "output_text": _extract_text(raw),
                "generation_wall_time_s": round(generation_wall_time_s, 6),
                "usage": usage,
            }
        )
    print(
        json.dumps(
            {
                "ok": True,
                "runtime": "llama_cpp",
                "load_wall_time_s": round(load_wall_time_s, 6),
                "total_worker_wall_time_s": round(time.monotonic() - started, 6),
                "calls": calls,
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
    cache_root: Path | str | None = None,
    selected_python: str | Path | None = None,
    command_runner: CommandRunner | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> JsonDict:
    """REQ-VERIFY-3165: build the v2 replay artifact and fail closed."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    runner = command_runner or run_command
    python_exe = str(selected_python) if selected_python is not None else selected_python_for(root_path)
    hf_cache = Path(cache_root) if cache_root is not None else default_hf_cache_root()

    exp3164 = read_json_object(root_path / EXP3164_REL_PATH)
    exp3123 = read_json_object(root_path / EXP3123_REL_PATH)
    sources = source_artifacts(root_path)
    model_specs = inspect_model_specs(root_path, hf_cache, exp3123)
    selected_model = first_usable_model(model_specs)
    substrate_probe = probe_substrate(python_exe, runner)
    contract_ready = contract_allows_exp3165(exp3164)
    smoke = maybe_run_replay(
        selected_python=python_exe,
        selected_model=selected_model,
        substrate_probe=substrate_probe,
        contract_ready=contract_ready,
        command_runner=runner,
        random_seed=random_seed,
    )

    live_call_count = len(smoke["transcript_hashes"])
    selected_ids = [selected_model["hf_id"]] if smoke["model_load_evidence"]["load_attempted"] else []
    model_specs = mark_selected_model(model_specs, selected_ids)
    prompt_hashes = [row["prompt_hash"] for row in smoke["transcript_hashes"]]
    source_checksums = {row["path"]: row["sha256"] for row in sources if row.get("sha256")}
    controlled_return_codes = controlled_subprocess_return_codes(
        substrate_probe, smoke["model_load_evidence"]
    )
    finished = time.perf_counter() if now_s is None else float(now_s)

    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": duration(start, finished),
        "live_sota_authenticity_replay_v2_ready": True,
        "model_specs": model_specs,
        "locally_usable_model_ids": [row["hf_id"] for row in model_specs if row["usable_locally"]],
        "selected_model_ids": selected_ids,
        "unavailable_model_ids": [row["hf_id"] for row in model_specs if not row["usable_locally"]],
        "preflight_passed": False,
        "live_call_count": live_call_count,
        "model_load_evidence": smoke["model_load_evidence"],
        "prompt_hashes": prompt_hashes,
        "transcript_hashes": smoke["transcript_hashes"],
        "token_counts": smoke["token_counts"],
        "controlled_subprocess_return_codes": controlled_return_codes,
        "measured_work_policy_passed": False,
        "token_scaled_duration_policy": token_scaled_duration_policy(exp3164),
        "token_scaled_duration_policy_passed": False,
        "repeated_call_policy": repeated_call_policy(exp3164),
        "repeated_call_policy_passed": False,
        "fake_evidence_rejection_criteria": fake_evidence_rejection_criteria(exp3164),
        "fake_evidence_rejection_passed": False,
        "fake_evidence_rejection_violations": [],
        "headline_claim_allowed": False,
        "blocked_reason": "",
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "source_artifacts": sources,
        "source_checksums": source_checksums,
        "inference_substrate": inference_substrate(
            selected_python=python_exe,
            substrate_probe=substrate_probe,
            selected_model=selected_model,
            live_call_count=live_call_count,
            load_attempted=bool(smoke["model_load_evidence"]["load_attempted"]),
            contract_ready=contract_ready,
        ),
        "field_principles": field_principles(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["measured_work_policy_passed"] = measured_work_policy_passed(artifact)
    artifact["token_scaled_duration_policy_passed"] = token_scaled_duration_policy_passed(artifact)
    artifact["repeated_call_policy_passed"] = repeated_call_policy_passed(artifact)
    fake_violations = fake_evidence_violations(artifact)
    artifact["fake_evidence_rejection_violations"] = fake_violations
    artifact["fake_evidence_rejection_passed"] = not fake_violations
    artifact["blocked_reason"] = determine_blocked_reason(
        contract_ready=contract_ready,
        usable_ids=artifact["locally_usable_model_ids"],
        substrate_probe=substrate_probe,
        smoke_blocker=smoke["runtime_blocker"],
        artifact=artifact,
    )
    artifact["preflight_passed"] = artifact["blocked_reason"] == ""
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    cache_root: Path | str | None = None,
    selected_python: str | Path | None = None,
    command_runner: CommandRunner | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    """Build and persist the Exp 3165 terminal JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        cache_root=cache_root,
        selected_python=selected_python,
        command_runner=command_runner,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
        random_seed=random_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def selected_python_for(root: Path) -> str:  # pragma: no cover - environment glue.
    """Use the project virtualenv Python when present."""

    candidate = root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def default_hf_cache_root() -> Path:
    """Return the local HuggingFace hub cache path without network access."""

    return Path.home() / ".cache" / "huggingface" / "hub"


def run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
) -> JsonDict:  # pragma: no cover - subprocess glue is covered by injected runners.
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
    """Read one JSON object, treating absent or malformed files as failed evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return every local file the replay reads or cites."""

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
    """Return a source-file checksum when the file exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_model_specs(root: Path, cache_root: Path, exp3123: Mapping[str, Any]) -> list[JsonDict]:
    """Combine mandated policy rows with current local cache path evidence."""

    inventory = [row for row in exp3123.get("cache_inventory", []) if isinstance(row, Mapping)]
    manifest_present_ids = {str(value) for value in exp3123.get("present_model_ids") or []}
    rows: list[JsonDict] = []
    for index, policy in enumerate(MANDATED_MODEL_POLICY):
        manifest_row = next((row for row in inventory if row.get("hf_id") == policy["hf_id"]), {})
        raw_path = best_model_path(root, cache_root, policy, manifest_row)
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
                "cache_status": "resolved" if evidence["exists"] else "missing",
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


def best_model_path(
    root: Path,
    cache_root: Path,
    policy: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> str | None:
    """Prefer a manifest path, then scan the local cache for the mandated GGUF."""

    for key in ("path", "model_path", "resolved_path"):
        raw_path = manifest_row.get(key)
        if raw_path and path_evidence(root, raw_path)["exists"]:
            return str(raw_path)
    candidates = direct_cache_candidates(cache_root, str(policy["hf_id"]))
    return str(candidates[0]) if candidates else None


def direct_cache_candidates(cache_root: Path, hf_id: str) -> list[Path]:
    """Inspect the local HF cache layout for a model ID without downloading."""

    owner, name = hf_id.split("/", 1)
    model_dir = cache_root / f"models--{owner}--{name}" / "snapshots"
    if not model_dir.is_dir():
        return []
    paths = [
        path
        for path in model_dir.rglob("*.gguf")
        if path.is_file() and path.stat().st_size > 0 and "mmproj" not in path.name.lower()
    ]
    q4_paths = [path for path in paths if "Q4_K_M" in path.name]
    return sorted(q4_paths or paths)


def path_evidence(root: Path, raw_path: Any) -> JsonDict:
    """Return existence, size, and bounded hash evidence for a possible GGUF."""

    if not raw_path:
        return {"path": None, "exists": False, "size_bytes": None, "bounded_sha256": None}
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return {"path": str(path), "exists": False, "size_bytes": None, "bounded_sha256": None}
    stat = path.stat()
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


def mark_selected_model(
    model_specs: Sequence[Mapping[str, Any]], selected_ids: Sequence[str]
) -> list[JsonDict]:
    """Annotate which mandated model actually backed the smoke call."""

    selected = set(selected_ids)
    return [dict(row) | {"selected_for_smoke": row.get("hf_id") in selected} for row in model_specs]


def contract_allows_exp3165(exp3164: Mapping[str, Any]) -> bool:
    """Return whether the checked-in v2 contract is complete enough to consume."""

    contract = _mapping(_mapping(exp3164.get("reusable_contracts")).get("exp3165"))
    return bool(
        exp3164.get("duration_corrected_authenticity_contract_v2_ready") is True
        and contract.get("old_fixed_60s_rule_hard_gate") is False
        and contract.get("minimum_distinct_smoke_calls", 0) >= MINIMUM_DISTINCT_SMOKE_CALLS
    )


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


def maybe_run_replay(
    *,
    selected_python: str,
    selected_model: Mapping[str, Any] | None,
    substrate_probe: Mapping[str, Any],
    contract_ready: bool,
    command_runner: CommandRunner,
    random_seed: int,
) -> JsonDict:
    """Run the repeated smoke only when the v2 contract and substrate permit it."""

    default = default_smoke_result(selected_model)
    if selected_model is None or not contract_ready:
        return default
    if substrate_probe.get("cuda_available") is not True or int(substrate_probe.get("gpu_count") or 0) <= 0:
        return default
    command = smoke_command(
        selected_python=selected_python,
        model_path=str(selected_model["model_path"]),
        random_seed=random_seed,
    )
    result = command_runner(command, timeout_s=900)
    payload = first_json_line(stdout_of(result))
    load_evidence = load_evidence_from_result(selected_model, command, result, payload)
    if load_evidence["runtime_error"]:
        return default | {
            "model_load_evidence": load_evidence,
            "runtime_blocker": load_evidence["runtime_error"],
        }
    calls = _mapping_list(payload.get("calls"))
    transcripts = transcript_hash_rows(selected_model, calls, random_seed)
    if not transcripts:
        load_evidence["runtime_error"] = "smoke worker produced no replay transcripts"
        return default | {
            "model_load_evidence": load_evidence,
            "runtime_blocker": load_evidence["runtime_error"],
        }
    return {
        "model_load_evidence": load_evidence,
        "transcript_hashes": transcripts,
        "token_counts": aggregate_token_counts([row["token_counts"] for row in transcripts]),
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
            "per_call_generation_wall_time_s": [],
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
            "per_call": [],
        },
        "runtime_blocker": "",
    }


def smoke_command(*, selected_python: str, model_path: str, random_seed: int) -> list[str]:
    """Build the exact local load command for the llama.cpp replay worker."""

    return [
        selected_python,
        "-c",
        SMOKE_WORKER_CODE,
        "--exp3165-smoke-worker",
        "--model-path",
        model_path,
        "--seed",
        str(int(random_seed)),
        "--prompts-json",
        json.dumps(list(DEFAULT_PROMPTS)),
        "--max-tokens",
        str(DEFAULT_MAX_TOKENS),
    ]


def first_json_line(stdout: str) -> JsonDict:
    """Parse the worker's final JSON line."""

    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def load_evidence_from_result(
    selected_model: Mapping[str, Any],
    command: Sequence[str],
    result: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> JsonDict:
    """Convert subprocess output into model-load evidence."""

    calls = _mapping_list(payload.get("calls"))
    generation_times = [float_or_none(row.get("generation_wall_time_s")) or 0.0 for row in calls]
    runtime_error = None
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
        "generation_wall_time_s": round(sum(generation_times), 6),
        "per_call_generation_wall_time_s": generation_times,
        "total_worker_wall_time_s": float_or_none(payload.get("total_worker_wall_time_s")),
        "stdout_summary": summarize(stdout_of(result)),
        "stderr_summary": summarize(stderr_of(result)),
        "runtime_error": runtime_error,
    }


def transcript_hash_rows(
    selected_model: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
    random_seed: int,
) -> list[JsonDict]:
    """Build replay-identifiable transcript evidence without long raw text."""

    rows: list[JsonDict] = []
    for index, call in enumerate(calls):
        prompt = str(call.get("prompt") or DEFAULT_PROMPTS[min(index, len(DEFAULT_PROMPTS) - 1)])
        output_text = str(call.get("output_text") or "")
        if not output_text.strip():
            continue
        seed = int_or_none(call.get("seed"))
        if seed is None:
            seed = int(random_seed) + index
        tokens = token_counts_for(prompt, output_text, call.get("usage"))
        prompt_hash = sha256_text(prompt)
        response_hash = sha256_text(output_text)
        transcript_payload = {
            "model_id": selected_model.get("hf_id"),
            "model_path": selected_model.get("model_path"),
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "random_seed": seed,
            "token_counts": tokens,
        }
        rows.append(
            {
                "model_id": selected_model.get("hf_id"),
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "transcript_sha256": stable_hash(transcript_payload),
                "prompt_token_count": tokens["prompt_tokens"],
                "output_token_count": tokens["completion_tokens"],
                "random_seed": seed,
                "token_counts": tokens,
            }
        )
    return rows


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
    return {
        "prompt_tokens": len(prompt.split()),
        "completion_tokens": len(output_text.split()),
        "total_tokens": len(prompt.split()) + len(output_text.split()),
        "source": "whitespace_estimate",
    }


def aggregate_token_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate token counts while preserving per-call evidence."""

    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
    sources = {str(row.get("source") or "unknown") for row in rows}
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "source": sources.pop() if len(sources) == 1 else "mixed",
        "per_call": [dict(row) for row in rows],
    }


def controlled_subprocess_return_codes(
    substrate_probe: Mapping[str, Any], load_evidence: Mapping[str, Any]
) -> list[JsonDict]:
    """Expose every bounded subprocess return code that supports the replay."""

    return [
        {
            "name": "torch_cuda_probe",
            "returncode": _mapping(substrate_probe.get("torch_cuda_probe")).get("returncode"),
        },
        {
            "name": "nvidia_smi_inventory",
            "returncode": _mapping(substrate_probe.get("nvidia_smi_inventory")).get("returncode"),
        },
        {"name": "model_load_smoke_worker", "returncode": load_evidence.get("returncode")},
    ]


def measured_work_policy_passed(artifact: Mapping[str, Any]) -> bool:
    """Evaluate the v2 observed-work gate for claimed live smoke evidence."""

    if int(artifact.get("live_call_count") or 0) <= 0:
        return False
    load = _mapping(artifact.get("model_load_evidence"))
    tokens = _mapping(artifact.get("token_counts"))
    return bool(
        load.get("load_attempted") is True
        and load.get("path_exists") is True
        and load.get("returncode") == 0
        and float_or_none(load.get("load_wall_time_s")) is not None
        and float_or_none(load.get("generation_wall_time_s")) is not None
        and int_or_none(tokens.get("prompt_tokens")) is not None
        and int(tokens.get("prompt_tokens") or 0) > 0
        and int(tokens.get("completion_tokens") or 0) > 0
        and bool(artifact.get("prompt_hashes"))
        and bool(artifact.get("transcript_hashes"))
        and artifact.get("random_seed") is not None
        and bool(artifact.get("reproducibility_checksum"))
    )


def token_scaled_duration_policy(exp3164: Mapping[str, Any]) -> JsonDict:
    """Keep the v2 duration policy visible in the replay artifact."""

    policy = _mapping(exp3164.get("token_scaled_duration_policy"))
    ceiling = (
        _mapping(policy.get("one_prompt_smoke")).get(
            "reject_if_completion_tokens_per_second_gt"
        )
        or IMPOSSIBLE_COMPLETION_TOKENS_PER_SECOND
    )
    return {
        "policy_version": "v2",
        "fixed_60s_floor_hard_gate": False,
        "minimum_distinct_smoke_calls": MINIMUM_DISTINCT_SMOKE_CALLS,
        "reject_if_completion_tokens_per_second_gt": float(ceiling),
        "pass_fail_basis": "measured work evidence plus bounded token throughput",
    }


def token_scaled_duration_policy_passed(artifact: Mapping[str, Any]) -> bool:
    """Reject smoke evidence with impossible completion-token throughput."""

    load = _mapping(artifact.get("model_load_evidence"))
    tokens = _mapping(artifact.get("token_counts"))
    completion_tokens = int(tokens.get("completion_tokens") or 0)
    generation_s = float_or_none(load.get("generation_wall_time_s"))
    if completion_tokens <= 0 or generation_s is None or generation_s <= 0:
        return False
    ceiling = float(
        _mapping(artifact.get("token_scaled_duration_policy")).get(
            "reject_if_completion_tokens_per_second_gt",
            IMPOSSIBLE_COMPLETION_TOKENS_PER_SECOND,
        )
    )
    return completion_tokens / generation_s <= ceiling


def repeated_call_policy(exp3164: Mapping[str, Any]) -> JsonDict:
    """Return the repeated-call controls inherited from Exp 3164."""

    policy = _mapping(_mapping(exp3164.get("repeated_call_policy")).get("exp3165"))
    return {
        "minimum_distinct_smoke_calls": int(
            policy.get("minimum_distinct_smoke_calls") or MINIMUM_DISTINCT_SMOKE_CALLS
        ),
        "require_distinct_prompt_hashes": policy.get("require_distinct_prompt_hashes", True),
        "require_distinct_transcript_sha256": policy.get("require_distinct_transcript_sha256", True),
        "require_distinct_response_hashes_or_seeded_prompt_variants": policy.get(
            "require_distinct_response_hashes_or_seeded_prompt_variants", True
        ),
        "all_calls_must_reference_same_selected_model_or_declared_model_rotation": policy.get(
            "all_calls_must_reference_same_selected_model_or_declared_model_rotation", True
        ),
    }


def repeated_call_policy_passed(artifact: Mapping[str, Any]) -> bool:
    """Check that the replay produced fresh transcript evidence."""

    policy = _mapping(artifact.get("repeated_call_policy"))
    rows = _mapping_list(artifact.get("transcript_hashes"))
    minimum = int(policy.get("minimum_distinct_smoke_calls") or MINIMUM_DISTINCT_SMOKE_CALLS)
    if len(rows) < minimum:
        return False
    prompt_hashes = [str(row.get("prompt_hash") or "") for row in rows]
    transcript_hashes = [str(row.get("transcript_sha256") or "") for row in rows]
    response_hashes = [str(row.get("response_hash") or "") for row in rows]
    seeds = [row.get("random_seed") for row in rows]
    return bool(
        len(set(prompt_hashes)) == len(rows)
        and len(set(transcript_hashes)) == len(rows)
        and (
            len(set(response_hashes)) == len(rows)
            or len(set(seeds)) == len(rows)
        )
    )


def fake_evidence_rejection_criteria(exp3164: Mapping[str, Any]) -> list[str]:
    """Keep adversarial failure modes explicit for future contract consumers."""

    criteria = exp3164.get("fake_evidence_rejection_criteria")
    if isinstance(criteria, list) and criteria:
        return [str(item) for item in criteria]
    return [
        "reject no model loaded: load_attempted false, path proof absent, or load wall time missing",
        "reject missing transcript hashes: every live call needs transcript_sha256 and response_hash",
        "reject no seed/checksum: random_seed and reproducibility_checksum are mandatory",
        "reject impossible token throughput: completion tokens per generation second above the declared ceiling",
        "reject reused stale transcript hash: repeated calls must have fresh transcript hashes unless declared replay",
        "reject mismatch between selected model and local path: selected model ID must match the selected model-spec path",
        "reject wall-clock claims not supported by command output: duration fields must match worker stdout JSON",
        "reject uncontrolled subprocess outcomes: model load, CUDA probe, and GPU inventory return codes must be recorded",
    ]


def fake_evidence_violations(artifact: Mapping[str, Any]) -> list[str]:
    """Return fake-evidence rejection failures, if any."""

    if int(artifact.get("live_call_count") or 0) == 0:
        return []
    violations: list[str] = []
    load = _mapping(artifact.get("model_load_evidence"))
    rows = _mapping_list(artifact.get("transcript_hashes"))
    if not load.get("load_attempted") or not load.get("path_exists") or load.get("load_wall_time_s") is None:
        violations.append("no model loaded")
    if len(rows) != int(artifact.get("live_call_count") or 0) or any(
        not row.get("transcript_sha256") or not row.get("response_hash") for row in rows
    ):
        violations.append("missing transcript hashes")
    if artifact.get("random_seed") is None or not artifact.get("reproducibility_checksum"):
        violations.append("missing seed/checksum")
    if not token_scaled_duration_policy_passed(artifact):
        violations.append("impossible token throughput")
    transcript_hashes = [str(row.get("transcript_sha256") or "") for row in rows]
    if len(set(transcript_hashes)) != len(transcript_hashes):
        violations.append("reused stale transcript hash")
    if not selected_path_matches_spec(artifact):
        violations.append("selected model/local path mismatch")
    if not worker_output_supports_wall_clock(load):
        violations.append("wall-clock claims not supported by command output")
    returncodes = [row.get("returncode") for row in _mapping_list(artifact.get("controlled_subprocess_return_codes"))]
    if any(code != 0 for code in returncodes):
        violations.append("uncontrolled subprocess outcomes")
    return violations


def selected_path_matches_spec(artifact: Mapping[str, Any]) -> bool:
    """Verify selected model identity and path agree with the model-spec row."""

    load = _mapping(artifact.get("model_load_evidence"))
    selected_model_id = load.get("selected_model_id")
    selected_model_path = load.get("selected_model_path")
    for row in _mapping_list(artifact.get("model_specs")):
        if row.get("hf_id") == selected_model_id and row.get("selected_for_smoke") is True:
            return row.get("model_path") == selected_model_path
    return False


def worker_output_supports_wall_clock(load: Mapping[str, Any]) -> bool:
    """Check that duration fields agree with the worker's printed JSON."""

    payload = first_json_line(str(load.get("stdout_summary") or ""))
    return all(
        _float_matches(payload.get(key), load.get(key))
        for key in ("load_wall_time_s", "total_worker_wall_time_s")
    )


def determine_blocked_reason(
    *,
    contract_ready: bool,
    usable_ids: Sequence[str],
    substrate_probe: Mapping[str, Any],
    smoke_blocker: str,
    artifact: Mapping[str, Any],
) -> str:
    """Return the first conservative blocker for the preflight gate."""

    if not contract_ready:
        return "duration-corrected v2 contract unavailable or not ready for Exp 3165"
    if not usable_ids:
        return "no mandated local SOTA GGUF path exists with nonzero size"
    if substrate_probe.get("cuda_available") is not True or int(substrate_probe.get("gpu_count") or 0) <= 0:
        return "CUDA/GPU substrate unavailable for mandated GGUF replay smoke"
    if smoke_blocker:
        return smoke_blocker
    if not artifact.get("measured_work_policy_passed"):
        return "measured-work policy failed for live SOTA replay evidence"
    if not artifact.get("token_scaled_duration_policy_passed"):
        return "token-scaled duration policy failed for live SOTA replay evidence"
    if not artifact.get("repeated_call_policy_passed"):
        return (
            f"repeated-call smoke produced {artifact.get('live_call_count')} transcripts; "
            f"expected at least {MINIMUM_DISTINCT_SMOKE_CALLS}"
        )
    if not artifact.get("fake_evidence_rejection_passed"):
        return "fake-evidence rejection failed: " + ", ".join(
            str(item) for item in artifact.get("fake_evidence_rejection_violations", [])
        )
    return ""


def inference_substrate(
    *,
    selected_python: str,
    substrate_probe: Mapping[str, Any],
    selected_model: Mapping[str, Any] | None,
    live_call_count: int,
    load_attempted: bool,
    contract_ready: bool,
) -> JsonDict:
    """Describe GPU/model/live status explicitly for downstream gates."""

    return {
        "kind": "live_sota_authenticity_replay_v2",
        "selected_python": selected_python,
        "runtime": "llama_cpp",
        "gpu_probe": {
            "cuda_available": substrate_probe.get("cuda_available"),
            "gpu_count": substrate_probe.get("gpu_count"),
            "torch_cuda_probe": substrate_probe.get("torch_cuda_probe"),
            "nvidia_smi_inventory": substrate_probe.get("nvidia_smi_inventory"),
        },
        "cpu_probe": substrate_probe.get("cpu_probe"),
        "contract_ready": bool(contract_ready),
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


def field_principles() -> JsonDict:
    """Explain why the required fields exist."""

    return {
        "live_sota_authenticity_replay_v2_ready": "downstream rerun needs a completed preflight artifact",
        "model_specs": "mandated local model policy must be visible",
        "locally_usable_model_ids": "actual availability must be auditable",
        "selected_model_ids": "any smoke call must identify selected models",
        "unavailable_model_ids": "comparative gaps must stay visible",
        "preflight_passed": "downstream gates need one conservative field",
        "live_call_count": "live evidence must not be inferred",
        "model_load_evidence": "claimed inference requires load evidence",
        "prompt_hashes": "prompts must be replay-identifiable",
        "transcript_hashes": "outputs must be replay-identifiable",
        "token_counts": "duration plausibility requires measured work",
        "measured_work_policy_passed": "fast calls need a principled pass criterion",
        "fake_evidence_rejection_passed": "authenticity checks must be adversarial",
        "headline_claim_allowed": "smoke tests do not create headline evidence",
        "blocked_reason": "blocked preflights must be actionable",
        "random_seed": "methodology completeness must be explicit",
        "reproducibility_checksum": "rerun provenance must be checkable",
        "source_artifacts": "preflight must trace to the v2 contract",
        "inference_substrate": "GPU/model/live status must be explicit",
        "honest_verdict": "terminal verdict must expose complete or blocked state",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash replay evidence that should reproduce across reruns."""

    payload = {
        "model_specs": artifact.get("model_specs"),
        "selected_model_ids": artifact.get("selected_model_ids"),
        "live_call_count": artifact.get("live_call_count"),
        "model_load_evidence": {
            key: _mapping(artifact.get("model_load_evidence")).get(key)
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
        "prompt_hashes": artifact.get("prompt_hashes"),
        "transcript_hashes": artifact.get("transcript_hashes"),
        "token_counts": artifact.get("token_counts"),
        "random_seed": artifact.get("random_seed"),
        "source_checksums": artifact.get("source_checksums"),
    }
    return stable_hash(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and fail-closed safety invariants."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3165 artifact missing required fields: {missing}")
    if artifact.get("headline_claim_allowed") is not False:
        raise ValueError("headline_claim_allowed must remain false for smoke replays")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("preflight_passed") is True:
        if int(artifact.get("live_call_count") or 0) < MINIMUM_DISTINCT_SMOKE_CALLS:
            raise ValueError("passed preflight requires repeated live call count evidence")
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
            "complete: live_sota_authenticity_replay_v2_ready=true; "
            "preflight_passed=true; "
            f"live_call_count={artifact.get('live_call_count')}; "
            "headline_claim_allowed=false; live_verifier_rerun_may_proceed=true"
        )
    reason = str(artifact.get("blocked_reason") or "preflight did not pass")
    if "no mandated local SOTA GGUF" in reason:
        prefix = "blocked_no_mandated_sota_gguf"
    elif "contract" in reason:
        prefix = "blocked_contract_precondition"
    elif "CUDA/GPU substrate" in reason:
        prefix = "blocked_gpu_substrate"
    elif "repeated-call" in reason:
        prefix = "blocked_repeated_call_policy"
    elif "fake-evidence" in reason:
        prefix = "blocked_fake_evidence_rejection"
    elif "token-scaled" in reason:
        prefix = "blocked_token_scaled_duration_policy"
    else:
        prefix = "blocked_smoke_runtime"
    return (
        f"{prefix}: preflight_passed=false; "
        f"live_call_count={artifact.get('live_call_count')}; detail={reason}"
    )


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

    return sha256_bytes(text.encode("utf-8"))


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest for raw bytes."""

    return hashlib.sha256(value).hexdigest()


def stable_hash(value: Any) -> str:
    """Hash JSON-serializable evidence with canonical key ordering."""

    return sha256_text(json.dumps(value, sort_keys=True))


def duration(started_s: float, finished_s: float) -> float:
    """Return a nonnegative elapsed duration."""

    return round(max(0.0, float(finished_s) - float(started_s)), 6)


def _float_matches(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    left_value = float_or_none(left)
    right_value = float_or_none(right)
    if left_value is None or right_value is None:
        return False
    return abs(left_value - right_value) <= tolerance


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = build_artifact(REPO_ROOT)
    output_path = REPO_ROOT / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["live_sota_authenticity_replay_v2_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
