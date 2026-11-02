"""Build the Exp 3179 local SOTA GGUF receipt-smoke artifact.

Spec refs: REQ-VERIFY-3179, SCENARIO-VERIFY-3179.

This module is a preflight receipt writer, not a verifier benchmark. It tries
to prove that one mandated local SOTA GGUF can be invoked through llama.cpp and
that the resulting transcripts are fresh, hashed, and plausible under the Exp
3178 v3 contract. CPU execution is allowed only to prove wiring; it never
becomes headline evidence and never unlocks a clean verifier rerun.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
CommandRunner = Callable[..., JsonDict]
CachedPairProvider = Callable[[], list[dict[str, Any]] | None]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
ARTIFACT = "experiment_3179_local_sota_receipt_smoke_v3"
SCHEMA = "carnot.local_sota_receipt_smoke.v3"
OUTPUT_REL_PATH = Path("results/experiment_3179_local_sota_receipt_smoke_v3.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3179_local_sota_receipt_smoke_v3.py"
EXP3178_REL_PATH = Path("results/experiment_3178_receipt_backed_authenticity_contract_v3.json")

DEFAULT_RANDOM_SEED = 20260527
DEFAULT_MAX_TOKENS = 4
DEFAULT_WORKER_TIMEOUT_S = 300
DEFAULT_PROMPTS = (
    "Exp 3179 receipt smoke A. Reply with exactly one token: READY.",
    "Exp 3179 receipt smoke B. Reply with exactly one token: VERIFIED.",
)
MINIMUM_RECEIPTS = 2
MAX_COMPLETION_TOKENS_PER_SECOND = 500.0

SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)

SUBSTRATE_CLASSES = (
    "model_cache_missing",
    "loader_missing",
    "cuda_unavailable",
    "cuda_available_unhealthy",
    "cpu_fallback_receipt_only",
    "full_local_sota_receipt",
)

REQUIRED_RECEIPT_FIELDS = (
    "selected_model_id",
    "model_path",
    "model_file_hash",
    "loader_name",
    "substrate_used",
    "prompt_hashes",
    "transcript_hashes",
    "token_counts",
    "random_seed",
    "wall_clock_s",
    "command_hash",
    "subprocess_return_code",
    "stderr_tail",
    "throughput_plausibility",
    "replay_count",
)

PER_RECEIPT_REQUIRED_FIELDS = (
    "selected_model_id",
    "model_path",
    "model_file_hash",
    "loader_name",
    "substrate_used",
    "prompt_hash",
    "transcript_hash",
    "token_counts",
    "random_seed",
    "wall_clock_s",
    "command_hash",
    "subprocess_return_code",
    "stderr_tail",
    "throughput_plausibility",
    "replay_count",
)

REQUIRED_FIELDS = {
    "local_sota_receipt_smoke_v3_ready",
    "preflight_passed",
    "live_call_count",
    "mandated_model_inventory",
    "selected_model_ids",
    "substrate_classification",
    "cpu_fallback_used",
    "proof_receipts",
    "throughput_plausibility_passed",
    "headline_claim_allowed",
    "clean_rerun_allowed",
    "inference_substrate",
    "honest_verdict",
}

MANDATED_MODEL_POLICY: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "name": "Qwen3.6-35B-A3B",
        "role": "moe",
        "tier": "flagship_moe",
        "strength_rank": 1,
        "expected_quantization": "Q4_K_M",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "name": "Gemma4-31B-it",
        "role": "dense",
        "tier": "flagship_dense",
        "strength_rank": 2,
        "expected_quantization": "Q4_K_M",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "name": "Gemma4-26B-A4B-it",
        "role": "moe",
        "tier": "middle_moe",
        "strength_rank": 3,
        "expected_quantization": "Q4_K_M",
    },
)

SOURCE_REL_PATHS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("experiment_template_policy", Path("scripts/experiment_template.py"), True, "python"),
    ("verification_openspec", Path("openspec/capabilities/verification/spec.md"), True, "text"),
    ("exp3178_v3_contract", EXP3178_REL_PATH, True, "json"),
    (
        "exp3179_module",
        Path("python/carnot/verify/local_sota_receipt_smoke_v3.py"),
        False,
        "python",
    ),
    (
        "exp3179_script",
        Path("scripts/experiment_3179_local_sota_receipt_smoke_v3.py"),
        False,
        "python",
    ),
    (
        "exp3179_tests",
        Path("tests/python/test_experiment_3179_local_sota_receipt_smoke_v3.py"),
        False,
        "python",
    ),
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3179_local_sota_receipt_smoke_v3.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3179_local_sota_receipt_smoke_v3.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/local_sota_receipt_smoke_v3.py' --fail-under=100 --show-missing",
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
parser.add_argument("--exp3179-smoke-worker", action="store_true")
parser.add_argument("--model-id", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--prompts-json", required=True)
parser.add_argument("--max-tokens", type=int, default=4)
parser.add_argument("--n-gpu-layers", type=int, required=True)
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
        n_ctx=256,
        n_batch=32,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )
    load_wall_time_s = time.monotonic() - load_started
    calls = []
    for index, prompt in enumerate(prompts):
        generation_started = time.monotonic()
        generation_kwargs = {
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "repeat_penalty": 1.0,
            "seed": args.seed + index,
        }
        try:
            raw = llm(prompt, **generation_kwargs)
            seed_supported = True
        except TypeError:
            generation_kwargs.pop("seed")
            raw = llm(prompt, **generation_kwargs)
            seed_supported = False
        generation_wall_time_s = time.monotonic() - generation_started
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        calls.append(
            {
                "prompt": prompt,
                "seed": args.seed + index,
                "seed_supported": seed_supported,
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
                "model_id": args.model_id,
                "n_gpu_layers": args.n_gpu_layers,
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
                "model_id": args.model_id,
                "n_gpu_layers": args.n_gpu_layers,
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
    cached_pair_provider: CachedPairProvider | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    worker_timeout_s: int = DEFAULT_WORKER_TIMEOUT_S,
) -> JsonDict:
    """REQ-VERIFY-3179: build a live-or-blocked v3 local SOTA receipt artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    runner = command_runner or run_command
    python_exe = str(selected_python) if selected_python is not None else selected_python_for(root_path)
    hf_cache = Path(cache_root) if cache_root is not None else default_hf_cache_root()

    exp3178 = read_json_object(root_path / EXP3178_REL_PATH)
    v3_contract = v3_contract_summary(exp3178)
    cached_pair_probe = probe_cached_sota_pair(cached_pair_provider or cached_sota_pair)
    inventory = inspect_mandated_model_inventory(root_path, hf_cache, cached_pair_probe)
    selected_model = strongest_available_model(inventory)
    loader_probe = probe_loader(python_exe, runner)
    cuda_probe = probe_cuda(python_exe, runner)
    nvidia_probe = probe_nvidia_smi(runner)
    cuda_healthy = cuda_probe.get("cuda_available") is True and safe_int(
        cuda_probe.get("device_count")
    ) not in (None, 0)
    n_gpu_layers = -1 if cuda_healthy else 0
    prior_transcript_hashes = collect_prior_transcript_hashes(exp3178)

    smoke = maybe_run_smoke(
        selected_python=python_exe,
        selected_model=selected_model,
        loader_probe=loader_probe,
        v3_contract_ready=bool(v3_contract["ready"]),
        n_gpu_layers=n_gpu_layers,
        command_runner=runner,
        random_seed=random_seed,
        worker_timeout_s=int(worker_timeout_s),
    )
    receipts = smoke["proof_receipts"]
    live_call_count = len(receipts)
    receipts_fresh = receipts_are_fresh(receipts, prior_transcript_hashes)
    throughput_passed = throughput_plausibility_passed(receipts)
    if not receipts_fresh:
        throughput_passed = False
    all_required_receipt_fields_present = required_receipt_fields_present(receipts)
    substrate_classification = classify_substrate(
        selected_model=selected_model,
        loader_available=loader_probe["available"],
        cuda_healthy=cuda_healthy,
        live_call_count=live_call_count,
        throughput_passed=throughput_passed,
        worker_returncode=smoke["worker_returncode"],
    )
    preflight_passed = (
        bool(v3_contract["ready"])
        and live_call_count >= MINIMUM_RECEIPTS
        and throughput_passed
        and receipts_fresh
        and all_required_receipt_fields_present
        and smoke["worker_returncode"] == 0
    )
    cpu_fallback_used = preflight_passed and substrate_classification == "cpu_fallback_receipt_only"
    clean_rerun_allowed = preflight_passed and substrate_classification == "full_local_sota_receipt"
    blocked_reason = blocked_reason_for(
        v3_contract_ready=bool(v3_contract["ready"]),
        selected_model=selected_model,
        loader_probe=loader_probe,
        smoke=smoke,
        live_call_count=live_call_count,
        receipts_fresh=receipts_fresh,
        throughput_passed=throughput_passed,
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    sources = source_artifacts(root_path)

    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": duration(start, finished),
        "local_sota_receipt_smoke_v3_ready": True,
        "v3_contract": v3_contract,
        "cached_sota_pair_probe": cached_pair_probe,
        "mandated_model_inventory": mark_selected(inventory, selected_model),
        "selected_model_ids": [selected_model["hf_id"]] if selected_model else [],
        "loader_probe": loader_probe,
        "cuda_probe": cuda_probe,
        "nvidia_smi_probe": nvidia_probe,
        "substrate_classification": substrate_classification,
        "preflight_passed": preflight_passed,
        "live_call_count": live_call_count,
        "cpu_fallback_used": cpu_fallback_used,
        "proof_receipts": receipts,
        "prompt_hashes": [row["prompt_hash"] for row in receipts],
        "transcript_hashes": [row["transcript_hash"] for row in receipts],
        "token_counts": aggregate_token_counts(receipts),
        "all_required_receipt_fields_present": all_required_receipt_fields_present,
        "throughput_plausibility": throughput_plausibility_summary(receipts, throughput_passed),
        "throughput_plausibility_passed": throughput_passed,
        "stale_transcript_rejection_passed": receipts_fresh,
        "prior_transcript_hash_count": len(prior_transcript_hashes),
        "headline_claim_allowed": False,
        "clean_rerun_allowed": clean_rerun_allowed,
        "blocked_reason": "" if preflight_passed else blocked_reason,
        "controlled_subprocess_return_codes": controlled_return_codes(
            loader_probe, cuda_probe, nvidia_probe, smoke
        ),
        "inference_substrate": inference_substrate(
            selected_python=python_exe,
            selected_model=selected_model,
            substrate_classification=substrate_classification,
            loader_probe=loader_probe,
            cuda_probe=cuda_probe,
            n_gpu_layers=n_gpu_layers,
            live_call_count=live_call_count,
            worker_timeout_s=int(worker_timeout_s),
            preflight_passed=preflight_passed,
        ),
        "source_artifacts": sources,
        "source_checksums": {row["path"]: row.get("sha256") for row in sources},
        "field_principles": field_principles(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "honest_verdict": "",
    }
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
    cached_pair_provider: CachedPairProvider | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    worker_timeout_s: int = DEFAULT_WORKER_TIMEOUT_S,
) -> Path:
    """Build and persist the Exp 3179 JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        cache_root=cache_root,
        selected_python=selected_python,
        command_runner=command_runner,
        cached_pair_provider=cached_pair_provider,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
        random_seed=random_seed,
        worker_timeout_s=worker_timeout_s,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def selected_python_for(root: Path) -> str:  # pragma: no cover - environment glue.
    """Prefer the project virtualenv because llama.cpp is usually installed there."""

    candidate = root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def default_hf_cache_root() -> Path:  # pragma: no cover - environment glue.
    """Return the HuggingFace hub cache root without downloading anything."""

    return Path.home() / ".cache" / "huggingface" / "hub"


def run_command(
    command: Sequence[str],
    *,
    timeout_s: int = 10,
    env: Mapping[str, str] | None = None,
) -> JsonDict:  # pragma: no cover - subprocess glue is tested through injected runners.
    """Run one bounded local command and keep enough stderr/stdout for diagnosis."""

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
        return {"command": cmd, "returncode": None, "stdout": "", "stderr": error}
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object while treating malformed files as missing evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """List the local files that make the receipt decision auditable."""

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
                "readable_json_object": bool(read_json_object(path))
                if source_type == "json"
                else None,
                "sha256": sha256_file(path),
            }
        )
    return rows


def sha256_file(path: Path) -> str | None:
    """Return a checksum for present source files and None for absent ones."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_cached_sota_pair(provider: CachedPairProvider) -> JsonDict:
    """Call cached_sota_pair() and record the result without trusting it alone."""

    try:
        pair = provider()
    except Exception as exc:
        return {
            "called": True,
            "returned_pair": False,
            "model_ids": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    rows = [dict(row) for row in pair] if isinstance(pair, list) else []
    return {
        "called": True,
        "returned_pair": len(rows) >= 2,
        "model_ids": [str(row.get("hf_id")) for row in rows if row.get("hf_id")],
        "model_paths": [str(row.get("model_path")) for row in rows if row.get("model_path")],
        "error": "",
    }


def inspect_mandated_model_inventory(
    root: Path, cache_root: Path, cached_pair_probe: Mapping[str, Any]
) -> list[JsonDict]:
    """Inspect every mandated GGUF directly in the HF cache."""

    cached_pair_ids = {str(item) for item in cached_pair_probe.get("model_ids", [])}
    rows: list[JsonDict] = []
    for policy in MANDATED_MODEL_POLICY:
        candidates = direct_cache_candidates(cache_root, str(policy["hf_id"]))
        evidence = path_evidence(root, str(candidates[0]) if candidates else None)
        rows.append(
            {
                "hf_id": policy["hf_id"],
                "name": policy["name"],
                "role": policy["role"],
                "tier": policy["tier"],
                "strength_rank": policy["strength_rank"],
                "expected_quantization": policy["expected_quantization"],
                "cache_status": "resolved" if evidence["exists"] else "missing",
                "model_path": evidence["path"],
                "path_exists": evidence["exists"],
                "path_size_bytes": evidence["size_bytes"],
                "model_file_hash": evidence["bounded_sha256"],
                "cached_sota_pair_member": policy["hf_id"] in cached_pair_ids,
                "candidate_count": len(candidates),
                "selected_for_smoke": False,
                "legacy_small_model": False,
            }
        )
    return rows


def direct_cache_candidates(cache_root: Path, hf_id: str) -> list[Path]:
    """Return non-empty GGUF candidates from the local HF cache layout."""

    owner, name = hf_id.split("/", 1)
    snapshots = cache_root / f"models--{owner}--{name}" / "snapshots"
    if not snapshots.is_dir():
        return []
    paths = [
        path
        for path in snapshots.rglob("*.gguf")
        if path.is_file() and path.stat().st_size > 0 and "mmproj" not in path.name.lower()
    ]
    q4_paths = [path for path in paths if "Q4_K_M" in path.name]
    return sorted(q4_paths or paths)


def path_evidence(root: Path, raw_path: Any) -> JsonDict:
    """Normalize a possible model path into existence, size, and hash evidence."""

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
    """Hash a bounded prefix plus metadata so large GGUFs are identified cheaply."""

    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def strongest_available_model(inventory: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Pick the strongest mandated model by policy order, not by cache scan order."""

    for row in inventory:
        if row.get("path_exists") is True and safe_int(row.get("path_size_bytes")) not in (None, 0):
            return dict(row)
    return None


def mark_selected(
    inventory: Sequence[Mapping[str, Any]], selected_model: Mapping[str, Any] | None
) -> list[JsonDict]:
    """Annotate the inventory row chosen for the smoke attempt."""

    selected_id = selected_model.get("hf_id") if selected_model else None
    return [dict(row) | {"selected_for_smoke": row.get("hf_id") == selected_id} for row in inventory]


def probe_loader(selected_python: str, command_runner: CommandRunner) -> JsonDict:
    """Check llama_cpp import availability before any expensive model load."""

    code = (
        "import json\n"
        "print('exp3179_loader_probe')\n"
        "try:\n"
        "    import llama_cpp\n"
        "    from llama_cpp import Llama\n"
        "    print(json.dumps({'ok': True, 'loader_name': 'llama_cpp.Llama', "
        "'version': getattr(llama_cpp, '__version__', None)}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}))\n"
        "    raise SystemExit(1)\n"
    )
    result = command_runner([selected_python, "-c", code], timeout_s=20)
    payload = first_json_line(str(result.get("stdout") or ""))
    available = result.get("returncode") == 0 and payload.get("ok") is True
    return {
        "available": available,
        "loader_name": payload.get("loader_name") if available else "llama_cpp.Llama",
        "version": payload.get("version"),
        "returncode": result.get("returncode"),
        "error": "" if available else str(payload.get("error") or result.get("stderr") or ""),
        "stderr_tail": truncate_tail(str(result.get("stderr") or "")),
    }


def probe_cuda(selected_python: str, command_runner: CommandRunner) -> JsonDict:
    """Probe torch CUDA only when torch is importable in the selected Python."""

    code = (
        "import importlib.util, json\n"
        "print('exp3179_torch_cuda_probe')\n"
        "if importlib.util.find_spec('torch') is None:\n"
        "    print(json.dumps({'torch_present': False, 'torch_import_ok': False, "
        "'cuda_available': False, 'device_count': 0}))\n"
        "else:\n"
        "    try:\n"
        "        import torch\n"
        "        print(json.dumps({'torch_present': True, 'torch_import_ok': True, "
        "'torch_version': getattr(torch, '__version__', None), "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'device_count': int(torch.cuda.device_count()), "
        "'cuda_version': getattr(torch.version, 'cuda', None)}))\n"
        "    except Exception as exc:\n"
        "        print(json.dumps({'torch_present': True, 'torch_import_ok': False, "
        "'cuda_available': False, 'device_count': 0, "
        "'error': f'{type(exc).__name__}: {exc}'}))\n"
        "        raise SystemExit(1)\n"
    )
    result = command_runner([selected_python, "-c", code], timeout_s=20)
    payload = first_json_line(str(result.get("stdout") or ""))
    return {
        "torch_present": payload.get("torch_present") is True,
        "torch_import_ok": payload.get("torch_import_ok") is True,
        "torch_version": payload.get("torch_version"),
        "cuda_available": result.get("returncode") == 0 and payload.get("cuda_available") is True,
        "device_count": safe_int(payload.get("device_count")) or 0,
        "cuda_version": payload.get("cuda_version"),
        "returncode": result.get("returncode"),
        "error": str(payload.get("error") or ""),
        "stderr_tail": truncate_tail(str(result.get("stderr") or "")),
    }


def probe_nvidia_smi(command_runner: CommandRunner) -> JsonDict:
    """Record nvidia-smi visibility as supporting evidence, not as a CUDA oracle."""

    result = command_runner(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        timeout_s=10,
    )
    stdout = str(result.get("stdout") or "")
    rows = [line.strip() for line in stdout.splitlines() if line.strip()]
    return {
        "available": result.get("returncode") == 0,
        "returncode": result.get("returncode"),
        "gpu_rows": rows,
        "stderr_tail": truncate_tail(str(result.get("stderr") or "")),
    }


def maybe_run_smoke(
    *,
    selected_python: str,
    selected_model: Mapping[str, Any] | None,
    loader_probe: Mapping[str, Any],
    v3_contract_ready: bool,
    n_gpu_layers: int,
    command_runner: CommandRunner,
    random_seed: int,
    worker_timeout_s: int,
) -> JsonDict:
    """Run the bounded llama.cpp worker only after cheap preconditions pass."""

    if selected_model is None or loader_probe.get("available") is not True or not v3_contract_ready:
        return {
            "attempted": False,
            "worker_returncode": None,
            "runtime_blocker": "",
            "proof_receipts": [],
            "command": [],
            "command_hash": "",
            "worker_code_sha256": hash_text(SMOKE_WORKER_CODE),
            "stderr_tail": "",
            "worker_payload": {},
        }
    command = [
        selected_python,
        "-c",
        SMOKE_WORKER_CODE,
        "--exp3179-smoke-worker",
        "--model-id",
        str(selected_model["hf_id"]),
        "--model-path",
        str(selected_model["model_path"]),
        "--seed",
        str(int(random_seed)),
        "--prompts-json",
        json.dumps(list(DEFAULT_PROMPTS)),
        "--max-tokens",
        str(DEFAULT_MAX_TOKENS),
        "--n-gpu-layers",
        str(int(n_gpu_layers)),
    ]
    command_hash = stable_hash(command)
    result = command_runner(command, timeout_s=worker_timeout_s)
    payload = first_json_line(str(result.get("stdout") or ""))
    receipts = receipt_rows(
        selected_model=selected_model,
        calls=mapping_list(payload.get("calls")),
        random_seed=random_seed,
        command_hash=command_hash,
        worker_code_sha256=hash_text(SMOKE_WORKER_CODE),
        subprocess_return_code=safe_int(result.get("returncode")),
        stderr_tail=truncate_tail(str(result.get("stderr") or "")),
        substrate_used="full_local_sota_receipt" if n_gpu_layers == -1 else "cpu_fallback_receipt_only",
        loader_name=str(loader_probe.get("loader_name") or "llama_cpp.Llama"),
    )
    runtime_blocker = ""
    if result.get("returncode") != 0 or payload.get("ok") is not True:
        runtime_blocker = str(payload.get("error") or result.get("stderr") or "smoke worker failed")
    elif not receipts:
        runtime_blocker = "smoke worker produced no proof receipts"
    return {
        "attempted": True,
        "worker_returncode": result.get("returncode"),
        "runtime_blocker": runtime_blocker,
        "proof_receipts": receipts,
        "command": command,
        "command_hash": command_hash,
        "worker_code_sha256": hash_text(SMOKE_WORKER_CODE),
        "stderr_tail": truncate_tail(str(result.get("stderr") or "")),
        "worker_payload": payload,
    }


def receipt_rows(
    *,
    selected_model: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
    random_seed: int,
    command_hash: str,
    worker_code_sha256: str,
    subprocess_return_code: int | None,
    stderr_tail: str,
    substrate_used: str,
    loader_name: str,
) -> list[JsonDict]:
    """Turn worker calls into v3 proof receipts with prompt/transcript hashes."""

    receipts: list[JsonDict] = []
    for index, call in enumerate(calls):
        output_text = str(call.get("output_text") or "")
        if not output_text:
            continue
        prompt = str(call.get("prompt") or DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)])
        seed = safe_int(call.get("seed")) or int(random_seed) + index
        prompt_sha = hash_text(prompt)
        response_sha = hash_text(output_text)
        receipt = {
            "selected_model_id": str(selected_model["hf_id"]),
            "model_path": str(selected_model["model_path"]),
            "model_file_hash": selected_model.get("model_file_hash"),
            "loader_name": loader_name,
            "substrate_used": substrate_used,
            "prompt_hash": prompt_sha,
            "response_hash": response_sha,
            "transcript_hash": transcript_hash(str(selected_model["hf_id"]), prompt_sha, response_sha, seed),
            "token_counts": token_counts_for(prompt, output_text, mapping(call.get("usage"))),
            "random_seed": seed,
            "wall_clock_s": safe_float(call.get("generation_wall_time_s")),
            "command_hash": command_hash,
            "worker_code_sha256": worker_code_sha256,
            "subprocess_return_code": subprocess_return_code,
            "stderr_tail": stderr_tail,
            "throughput_plausibility": True,
            "replay_count": index + 1,
        }
        receipt["throughput_plausibility"] = receipt_throughput_plausible(receipt)
        receipts.append(receipt)
    return receipts


def token_counts_for(prompt: str, output_text: str, usage: Mapping[str, Any]) -> JsonDict:
    """Prefer llama.cpp usage counters and fall back to a transparent word estimate."""

    prompt_tokens = safe_int(usage.get("prompt_tokens"))
    completion_tokens = safe_int(usage.get("completion_tokens"))
    total_tokens = safe_int(usage.get("total_tokens"))
    if prompt_tokens is not None and completion_tokens is not None and total_tokens is not None:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "source": "llama_cpp_usage",
        }
    prompt_estimate = len(prompt.split())
    completion_estimate = len(output_text.split())
    return {
        "prompt_tokens": prompt_estimate,
        "completion_tokens": completion_estimate,
        "total_tokens": prompt_estimate + completion_estimate,
        "source": "whitespace_estimate",
    }


def receipt_throughput_plausible(receipt: Mapping[str, Any]) -> bool:
    """Reject impossible completion throughput instead of trusting wall time blindly."""

    counts = mapping(receipt.get("token_counts"))
    completion_tokens = safe_int(counts.get("completion_tokens")) or 0
    wall_clock_s = safe_float(receipt.get("wall_clock_s"))
    if completion_tokens <= 0:
        return True
    if wall_clock_s is None or wall_clock_s <= 0:
        return False
    return completion_tokens / wall_clock_s <= MAX_COMPLETION_TOKENS_PER_SECOND


def throughput_plausibility_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require every emitted receipt to pass the token-rate plausibility check."""

    return bool(receipts) and all(receipt.get("throughput_plausibility") is True for receipt in receipts)


def throughput_plausibility_summary(
    receipts: Sequence[Mapping[str, Any]], passed: bool
) -> JsonDict:
    """Summarize throughput evidence for downstream gates."""

    return {
        "passed": passed,
        "max_completion_tokens_per_second": MAX_COMPLETION_TOKENS_PER_SECOND,
        "per_receipt": [
            {
                "transcript_hash": row.get("transcript_hash"),
                "completion_tokens": mapping(row.get("token_counts")).get("completion_tokens"),
                "wall_clock_s": row.get("wall_clock_s"),
                "passed": row.get("throughput_plausibility") is True,
            }
            for row in receipts
        ],
    }


def receipts_are_fresh(receipts: Sequence[Mapping[str, Any]], prior_hashes: set[str]) -> bool:
    """Reject duplicate receipts and hashes already present in prior contracts."""

    hashes = [str(row.get("transcript_hash") or "") for row in receipts]
    hashes = [value for value in hashes if value]
    return len(hashes) == len(set(hashes)) and not (set(hashes) & prior_hashes)


def required_receipt_fields_present(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Check the Exp 3178 receipt field contract against per-call evidence rows."""

    return bool(receipts) and all(
        all(field in row and row.get(field) is not None for field in PER_RECEIPT_REQUIRED_FIELDS)
        for row in receipts
    )


def collect_prior_transcript_hashes(value: Any) -> set[str]:
    """Extract transcript-like hashes from a prior JSON contract recursively."""

    hashes: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if "transcript" in str(key).lower() and isinstance(item, str) and item:
                hashes.add(item)
            hashes.update(collect_prior_transcript_hashes(item))
    elif isinstance(value, list):
        for item in value:
            hashes.update(collect_prior_transcript_hashes(item))
    return hashes


def classify_substrate(
    *,
    selected_model: Mapping[str, Any] | None,
    loader_available: bool,
    cuda_healthy: bool,
    live_call_count: int,
    throughput_passed: bool,
    worker_returncode: Any,
) -> str:
    """Classify the terminal substrate using the Exp 3178 v3 class names."""

    if selected_model is None:
        return "model_cache_missing"
    if not loader_available:
        return "loader_missing"
    if live_call_count >= MINIMUM_RECEIPTS and throughput_passed and worker_returncode == 0:
        return "full_local_sota_receipt" if cuda_healthy else "cpu_fallback_receipt_only"
    return "cuda_available_unhealthy" if cuda_healthy else "cuda_unavailable"


def blocked_reason_for(
    *,
    v3_contract_ready: bool,
    selected_model: Mapping[str, Any] | None,
    loader_probe: Mapping[str, Any],
    smoke: Mapping[str, Any],
    live_call_count: int,
    receipts_fresh: bool,
    throughput_passed: bool,
) -> str:
    """Return the exact precondition or runtime blocker for no-pass artifacts."""

    if not v3_contract_ready:
        return "Exp 3178 v3 receipt contract missing or not ready"
    if selected_model is None:
        return "no mandated local SOTA GGUF path exists with nonzero size"
    if loader_probe.get("available") is not True:
        return f"llama_cpp loader/import unavailable: {loader_probe.get('error')}"
    if str(smoke.get("runtime_blocker") or ""):
        return str(smoke["runtime_blocker"])
    if live_call_count < MINIMUM_RECEIPTS:
        return f"receipt smoke produced {live_call_count} transcripts; expected at least {MINIMUM_RECEIPTS}"
    if not receipts_fresh:
        return "reused stale transcript hash"
    if not throughput_passed:
        return "throughput plausibility failed"
    return ""


def aggregate_token_counts(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate per-call token counts while preserving the per-receipt rows."""

    per_receipt = [mapping(row.get("token_counts")) for row in receipts]
    return {
        "prompt_tokens": sum(safe_int(row.get("prompt_tokens")) or 0 for row in per_receipt),
        "completion_tokens": sum(
            safe_int(row.get("completion_tokens")) or 0 for row in per_receipt
        ),
        "total_tokens": sum(safe_int(row.get("total_tokens")) or 0 for row in per_receipt),
        "source": "receipt_sum" if per_receipt else "none",
        "per_receipt": per_receipt,
    }


def controlled_return_codes(
    loader_probe: Mapping[str, Any],
    cuda_probe: Mapping[str, Any],
    nvidia_probe: Mapping[str, Any],
    smoke: Mapping[str, Any],
) -> list[JsonDict]:
    """Expose subprocess outcomes so failed preflights do not look uncontrolled."""

    return [
        {"name": "llama_cpp_loader_probe", "returncode": loader_probe.get("returncode")},
        {"name": "torch_cuda_probe", "returncode": cuda_probe.get("returncode")},
        {"name": "nvidia_smi_inventory", "returncode": nvidia_probe.get("returncode")},
        {"name": "model_smoke_worker", "returncode": smoke.get("worker_returncode")},
    ]


def inference_substrate(
    *,
    selected_python: str,
    selected_model: Mapping[str, Any] | None,
    substrate_classification: str,
    loader_probe: Mapping[str, Any],
    cuda_probe: Mapping[str, Any],
    n_gpu_layers: int,
    live_call_count: int,
    worker_timeout_s: int,
    preflight_passed: bool,
) -> JsonDict:
    """Declare exactly what local compute substrate was observed or used."""

    return {
        "kind": "local_sota_receipt_smoke_v3",
        "selected_python": selected_python,
        "downloads_models": False,
        "legacy_small_model_used": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_models": live_call_count > 0,
        "live_model_calls": int(live_call_count),
        "selected_model_id": selected_model.get("hf_id") if selected_model else None,
        "selected_model_path": selected_model.get("model_path") if selected_model else None,
        "loader_name": loader_probe.get("loader_name"),
        "loader_available": loader_probe.get("available") is True,
        "torch_cuda_available": cuda_probe.get("cuda_available") is True,
        "torch_cuda_device_count": safe_int(cuda_probe.get("device_count")) or 0,
        "n_gpu_layers": int(n_gpu_layers),
        "substrate_classification": substrate_classification,
        "cpu_fallback_used": preflight_passed and substrate_classification == "cpu_fallback_receipt_only",
        "worker_timeout_s": int(worker_timeout_s),
    }


def v3_contract_summary(exp3178: Mapping[str, Any]) -> JsonDict:
    """Summarize the contract this smoke is governed by."""

    classes = mapping(mapping(exp3178.get("substrate_classification_policy")).get("classes"))
    return {
        "path": EXP3178_REL_PATH.as_posix(),
        "ready": exp3178.get("receipt_backed_authenticity_contract_v3_ready") is True,
        "required_receipt_fields_present": set(exp3178.get("required_receipt_fields", []))
        >= set(REQUIRED_RECEIPT_FIELDS),
        "substrate_classes_present": set(classes) >= set(SUBSTRATE_CLASSES),
        "clean_rerun_unlock_requirements": [
            str(item) for item in exp3178.get("clean_rerun_unlock_requirements", [])
        ],
    }


def field_principles() -> JsonDict:
    """Explain the top-level fields downstream gates are expected to consume."""

    return {
        "local_sota_receipt_smoke_v3_ready": "downstream gates need a materialized artifact",
        "preflight_passed": "live reruns require explicit preflight status",
        "live_call_count": "no-call blocked artifacts must not look like live evidence",
        "mandated_model_inventory": "model availability must be auditable",
        "selected_model_ids": "local SOTA policy must be machine-readable",
        "substrate_classification": "blockers must distinguish cache, loader, CUDA, and CPU causes",
        "cpu_fallback_used": "CPU smoke must not be mistaken for headline performance",
        "proof_receipts": "authenticity must trace to prompt and transcript hashes",
        "throughput_plausibility_passed": "impossible speed claims must be rejected",
        "headline_claim_allowed": "smoke tests should not become headline evidence",
        "clean_rerun_allowed": "verifier tasks need explicit unlock signal",
        "inference_substrate": "live or blocked preflight substrate must be declared",
        "honest_verdict": "terminal status must be success-prefixed unless honestly blocked",
    }


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Produce the conductor-compatible terminal verdict string."""

    if artifact.get("preflight_passed") is True:
        clean = str(artifact.get("clean_rerun_allowed")).lower()
        return (
            "complete: local_sota_receipt_smoke_v3_ready=true; "
            f"substrate_classification={artifact.get('substrate_classification')}; "
            f"live_call_count={artifact.get('live_call_count')}; "
            f"clean_rerun_allowed={clean}"
        )
    reason = str(artifact.get("blocked_reason") or "unknown blocker")
    if "throughput" in reason.lower():
        prefix = "blocked_throughput_plausibility"
    elif artifact.get("substrate_classification") == "model_cache_missing":
        prefix = "blocked_model_cache_missing"
    elif artifact.get("substrate_classification") == "loader_missing":
        prefix = "blocked_loader_missing"
    elif artifact.get("substrate_classification") == "cuda_unavailable":
        prefix = "blocked_cuda_unavailable"
    elif artifact.get("substrate_classification") == "cuda_available_unhealthy":
        prefix = "blocked_cuda_available_unhealthy"
    else:
        prefix = "blocked_receipt_precondition"
    return (
        f"{prefix}: local_sota_receipt_smoke_v3_ready=true; "
        f"preflight_passed=false; live_call_count={artifact.get('live_call_count')}; "
        f"detail={reason}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that could be mistaken for headline or clean-rerun evidence."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("substrate_classification") not in SUBSTRATE_CLASSES:
        raise ValueError("substrate_classification is not a v3 substrate class")
    if artifact.get("headline_claim_allowed") is not False:
        raise ValueError("headline_claim_allowed must remain false for receipt smoke")
    if artifact.get("clean_rerun_allowed") is True and artifact.get(
        "substrate_classification"
    ) != "full_local_sota_receipt":
        raise ValueError("clean rerun requires full_local_sota_receipt")
    if artifact.get("preflight_passed") is True and int(artifact.get("live_call_count") or 0) < 2:
        raise ValueError("preflight_passed requires live_call_count >= 2")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (verdict.startswith(SUCCESS_PREFIXES) or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must use a terminal success or blocked prefix")
    if artifact.get("preflight_passed") is False and not verdict.startswith("blocked_"):
        raise ValueError("blocked preflight must use a blocked_ honest_verdict")


def transcript_hash(model_id: str, prompt_hash: str, response_hash: str, seed: int) -> str:
    """Hash transcript identity fields without storing full prompt/output text."""

    return stable_hash(
        {
            "model_id": model_id,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "seed": int(seed),
        }
    )


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 over a JSON-serializable value."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hash_text(payload)


def hash_text(value: str) -> str:
    """Return the SHA-256 of text using the repo's UTF-8 artifact convention."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first_json_line(text: str) -> JsonDict:
    """Parse the first JSON object emitted by a probe or worker command."""

    for line in text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def mapping(value: Any) -> JsonDict:
    """Normalize a JSON value into a mapping."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Normalize a JSON list into a list of mapping rows."""

    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def safe_int(value: Any) -> int | None:
    """Convert counters from JSON without raising on missing probe fields."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    """Convert wall-clock values from JSON without raising on missing fields."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truncate_tail(text: str, *, limit: int = 2000) -> str:
    """Keep stderr compact while preserving the most recent diagnostic lines."""

    compact = text.rstrip()
    return compact if len(compact) <= limit else compact[-limit:]


def duration(started_s: float, finished_s: float) -> float:
    """Return a non-negative rounded duration for stable artifacts."""

    return round(max(0.0, float(finished_s) - float(started_s)), 6)
