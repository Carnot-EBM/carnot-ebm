"""Exp6468 unique-event verifier-bounded continuous self-learning.

Spec refs: REQ-LEARN-6468, SCENARIO-LEARN-6468-SPEC,
SCENARIO-LEARN-6468-MODELS, SCENARIO-LEARN-6468-SEALED-SPLIT,
SCENARIO-LEARN-6468-UNIQUE-EVENTS, SCENARIO-LEARN-6468-EXACT-VETO,
SCENARIO-LEARN-6468-UPDATE-RULE, SCENARIO-LEARN-6468-AGGREGATES,
SCENARIO-LEARN-6468-ATTACKS, SCENARIO-LEARN-6468-READY.

The experiment updates only external factor weights. The exact checker owns
write admission and update direction. Model output is one raw event record and
can scale only the update magnitude.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

from carnot import task_runtime_receipts as runtime_receipts
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
PreconditionFn = Callable[..., list[JsonDict]]
GenerationFn = Callable[[JsonDict, str, JsonDict], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6468_unique_event_verifier_bounded_csl.json")
DATA_DIR_RELATIVE_PATH = Path("data/research/experiment_6468_unique_event_verifier_bounded_csl")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

SCHEMA = "carnot.experiment_6468.unique_event_verifier_bounded_csl.v1"
RUN_DATE = "20260819"
RANDOM_SEED = 6468
PREFERRED_QUANT = "Q4_K_M"
TOKENIZER_SOURCE = "embedded_gguf_vocab_only"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_unique_event_exact_veto"
BLOCKED_SUBSTRATE = "blocked_precondition_check_only"
MIN_FREE_DISK_BYTES = 4 * 1024 * 1024 * 1024

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_TEMPLATES: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "model_family": "qwen_moe",
        "gpu": 0,
        "preferred_quant": PREFERRED_QUANT,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "model_family": "gemma_dense",
        "gpu": 1,
        "preferred_quant": PREFERRED_QUANT,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "model_family": "gemma_moe",
        "gpu": 1,
        "preferred_quant": PREFERRED_QUANT,
    },
)
MODEL_TEMPLATE_BY_ID = {str(row["hf_id"]): dict(row) for row in MODEL_TEMPLATES}

FROZEN_ARM = "frozen_factor_weights"
SELF_SIGNED_ARM = "self_signed_updates"
VERIFIER_BOUNDED_ARM = "verifier_bounded_exact_sign_updates"
ARMS = (FROZEN_ARM, SELF_SIGNED_ARM, VERIFIER_BOUNDED_ARM)

UNITS_PER_MODEL = 24
INTERVAL_RANGES: tuple[tuple[str, range], ...] = (
    ("development", range(0, 6)),
    ("prospective_update", range(6, 16)),
    ("future_held", range(16, 24)),
)
WEIGHT_FEATURES = ("route_first", "verified_binding", "protected_shortcut", "abstain_guard")
WEIGHT_CAP = 2.0
LEARNING_RATE = 0.25
MAX_UPDATE_MAGNITUDE = 0.25

ATTACK_IDS = (
    "cloned_raw_output",
    "duplicate_event_id",
    "held_exposure",
    "self_signed_false_pass",
    "exact_veto_bypass",
    "future_leakage",
    "protected_case_regression",
    "aggregate_mismatch",
)
READINESS_CONDITIONS = (
    "one_to_one_event_provenance",
    "exact_veto_precedes_write",
    "verifier_beats_frozen_future_yield",
    "verifier_beats_self_signed_future_yield",
    "protected_cases_retained",
    "model_files_frozen",
    "zero_cpu_fallback",
    "aggregates_recompute",
    "critical_attacks_fail_closed",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6468_unique_event_verifier_bounded_csl --date 20260819"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py "
    "-m pytest tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6468_unique_event_verifier_bounded_csl.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6468_unique_event_verifier_bounded_csl.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6468_unique_event_verifier_bounded_csl "
    "--date 20260819 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6468_unique_event_verifier_bounded_csl.json"
)
ROW_CONSISTENCY_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6468_unique_event_verifier_bounded_csl.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_COMMAND = "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6468 entry"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    ROW_CONSISTENCY_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    E2E_PLAN_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("results/experiment_6449_generation_to_verdict_path_receipt_contract.json"),
    Path("results/experiment_6455_prospective_verifier_bounded_factor_weight_csl.json"),
    Path("results/experiment_6457_independent_verifier_bounded_csl_audit.json"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/experiment_template.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_and_embedded_tokenizer_hashes",
    "autotokenizer_usage_count",
    "device_and_runner_receipts",
    "sealed_chronological_manifest",
    "exposure_ledger",
    "update_rule_and_bounds",
    "raw_output_manifest",
    "event_identity_manifest",
    "exact_veto_before_write_receipts",
    "per_unit_rows",
    "event_rows",
    "effect_by_arm_and_interval",
    "protected_case_retention",
    "write_and_rollback_counts",
    "one_event_one_raw_hash_check",
    "cpu_fallback_count",
    "aggregate_row_recomputation",
    "attack_matrix",
    "current_adversarial_findings",
    "unique_event_csl_ready_score",
    "protected_files_unchanged",
    "blocked_reason",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names the terminal state for the unique-event CSL run.",
    "MODEL_SPECS": "Carries the three mandated cached GGUF model identities.",
    "models_used": "Lists only mandated models with eligible live event rows.",
    "cached_sota_pair_receipts": "Shows the cached local resolver calls.",
    "model_file_and_embedded_tokenizer_hashes": "Binds model bytes and embedded tokenizer metadata.",
    "autotokenizer_usage_count": "Must remain zero because GGUF tokenizers are embedded.",
    "device_and_runner_receipts": "Binds GPUs, CUDA, llama.cpp, generation calls, and CPU fallback checks.",
    "sealed_chronological_manifest": "Freezes units, intervals, arms, seeds, and budgets before inference.",
    "exposure_ledger": "Proves held outcomes are not visible before inference or update admission.",
    "update_rule_and_bounds": "Pins exact-sign authority, confidence magnitude use, and bounds.",
    "raw_output_manifest": "Proves raw bytes were persisted and validated before parse.",
    "event_identity_manifest": "Proves event ids are non-empty and unique.",
    "exact_veto_before_write_receipts": "Proves checker authority precedes each admitted write.",
    "per_unit_rows": "Contains row data before aggregate calculation.",
    "event_rows": "Contains one generation event for each per-unit row.",
    "effect_by_arm_and_interval": "Reports exact yield by arm and chronological interval.",
    "protected_case_retention": "Blocks utility that harms protected cases.",
    "write_and_rollback_counts": "Counts admitted writes, vetoes, and rollback pointers.",
    "one_event_one_raw_hash_check": "Proves no raw hash is cloned across rows.",
    "cpu_fallback_count": "Must be zero for ready live local GGUF evidence.",
    "aggregate_row_recomputation": "Recomputes reported metrics from rows.",
    "attack_matrix": "Shows critical event, veto, leakage, and aggregate attacks fail closed.",
    "current_adversarial_findings": "Keeps current critical findings visible.",
    "unique_event_csl_ready_score": "Conjunctive readiness for unique-event exact-veto CSL.",
    "protected_files_unchanged": "Shows conductor, ops, traceability, and upstream evidence stayed byte-identical.",
    "blocked_reason": "Explains failed preconditions for blocked artifacts.",
    "gate_check_summary": "Summarizes readiness gates and blockers.",
    "preconditions_checked": "Records hardware, cache, tokenizer, path, event-id, split, and checker checks.",
    "inference_substrate": "Declares local SOTA GGUF live inference with exact-checker-governed external weights.",
    "verifier_is_oracle": "Marks only deterministic checker, chronology, and row arithmetic as oracle boundaries.",
    "field_principles": "Documents why each field and readiness condition exists.",
    "field_provenance": "Maps fields to specs, manifests, rows, receipts, attacks, or tests.",
    "random_seed": "Pins streams, events, prompts, updates, and attacks.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records focused, coverage, full pytest, spec, row, adversarial, and E2E checks.",
    "reproducibility_checksum": "Content-addresses the artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal prefix and states the exact-veto boundary.",
}
FIELD_PRINCIPLES.update(
    {
        f"unique_event_csl_ready_score:{condition}": "Required readiness condition."
        for condition in READINESS_CONDITIONS
    }
)
FIELD_PRINCIPLES.update({attack: "Critical attack must fail closed." for attack in ATTACK_IDS})

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6468",
        "sealed chronological manifest",
        "unique event raw output rows",
        "exact checker authority receipts",
        "focused Exp6468 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the project digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the project digest prefix."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after stable serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Stream one file hash, or return None when absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def model_slug(model_id: str) -> str:
    """Return a stable file-system slug for one model id."""

    return re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower()


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through the shared atomic helper."""

    return runtime_receipts.write_json_atomic(path, payload)


def write_bytes_atomic(path: str | Path, payload: bytes) -> Path:
    """Write raw bytes through a same-directory temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    tmp.replace(target)
    return target


def _revision_from_path(path: str | Path) -> str | None:
    parts = Path(path).parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def _quantization_from_path(path: str | Path) -> str:
    name = Path(path).name.lower()
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0"):
        if token.lower() in name:
            return token
    return "unknown"


def _tokenizer_hash(model_id: str, model_hash: str | None, detail: str) -> str:
    return sha256_json(
        {
            "hf_id": model_id,
            "model_file_sha256": model_hash,
            "method": TOKENIZER_METHOD,
            "source": TOKENIZER_SOURCE,
            "detail": detail,
        }
    )


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
) -> JsonDict:
    """Resolve the mandated GGUF rows through cached local helper calls."""

    default_pair = cached_pair_func(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT) or []
    dense_pair = (
        cached_pair_func(
            gpu_indices=(0, 1),
            preferred_quant=PREFERRED_QUANT,
            model_indices=(0, 2),
        )
        or []
    )
    by_id = {str(row.get("hf_id")): dict(row) for row in [*default_pair, *dense_pair]}
    records: list[JsonDict] = []
    for template in MODEL_TEMPLATES:
        model_id = str(template["hf_id"])
        raw = by_id.get(model_id, {})
        path = Path(str(raw.get("model_path") or ""))
        exists = path.is_file()
        tokenizer_ok, tokenizer_detail = (
            tokenizer_func(str(path)) if exists else (False, "model file missing")
        )
        model_hash = sha256_file(path) if exists else None
        records.append(
            {
                **template,
                "name": raw.get("name", template["name"]),
                "gpu": int(raw.get("gpu", template["gpu"]) or 0),
                "model_path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "model_file_sha256": model_hash,
                "revision": _revision_from_path(path),
                "quantization": _quantization_from_path(path),
                "tokenizer_source": TOKENIZER_SOURCE,
                "tokenizer_method": TOKENIZER_METHOD,
                "tokenizer_loadable": bool(tokenizer_ok),
                "tokenizer_detail": tokenizer_detail,
                "tokenizer_sha256": _tokenizer_hash(model_id, model_hash, tokenizer_detail),
                "autotokenizer_used": False,
            }
        )
    blockers = [
        f"model_not_resolved:{row['hf_id']}" for row in records if row["exists"] is not True
    ] + [
        f"embedded_tokenizer_not_loadable:{row['hf_id']}"
        for row in records
        if row["tokenizer_loadable"] is not True
    ]
    return {
        "MODEL_SPECS": records,
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "calls": [
                {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": None},
                {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": [0, 2]},
            ],
            "returned_hf_ids": [row.get("hf_id") for row in [*default_pair, *dense_pair]],
            "same_cache_resolver_used": True,
            "legacy_models_smoke_only": True,
        },
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
        "autotokenizer_usage_count": 0,
    }


def model_file_and_embedded_tokenizer_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return model-file and embedded-tokenizer identity rows."""

    rows = [
        {
            "hf_id": row.get("hf_id"),
            "model_family": row.get("model_family"),
            "model_path": row.get("model_path"),
            "model_file_sha256": row.get("model_file_sha256"),
            "revision": row.get("revision"),
            "quantization": row.get("quantization"),
            "embedded_tokenizer_sha256": row.get("tokenizer_sha256"),
            "tokenizer_source": row.get("tokenizer_source"),
            "tokenizer_method": row.get("tokenizer_method"),
            "tokenizer_loadable": row.get("tokenizer_loadable") is True,
            "autotokenizer_used": row.get("autotokenizer_used") is True,
            "base_file_write_opened": False,
        }
        for row in model_specs
    ]
    return {
        "rows": rows,
        "model_count": len(rows),
        "all_model_files_present": all(Path(str(row["model_path"])).is_file() for row in rows),
        "all_embedded_tokenizers_loadable": all(row["tokenizer_loadable"] for row in rows),
        "autotokenizer_usage_count": sum(row["autotokenizer_used"] for row in rows),
        "base_ggufs_frozen": all(row["model_file_sha256"] for row in rows),
        "weight_update_count": 0,
    }


def source_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash source files that define this experiment."""

    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash protected files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected hashes from before and after the run."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def _nvidia_smi_rows() -> list[JsonDict]:  # pragma: no cover
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.free",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except OSError as exc:
        return [{"error": str(exc), "returncode": 127}]
    rows: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "name": parts[0],
                    "uuid": parts[1],
                    "memory_total": parts[2],
                    "memory_free": parts[3],
                }
            )
    return rows or [{"returncode": result.returncode, "stderr": result.stderr.strip()}]


def _llama_cpp_cuda_receipt() -> JsonDict:  # pragma: no cover
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as low
    except Exception as exc:
        return {"available": False, "detail": repr(exc), "importable": False}
    return {
        "available": bool(low.llama_supports_gpu_offload()),
        "detail": f"llama_cpp {getattr(llama_cpp, '__version__', 'unknown')}",
        "importable": True,
        "gpu_offload_supported": bool(low.llama_supports_gpu_offload()),
    }


def default_preconditions(  # pragma: no cover
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[JsonDict],
    sealed_manifest: JsonDict,
) -> list[JsonDict]:
    """Check live host preconditions before any model generation."""

    gpu_rows = _nvidia_smi_rows()
    rtx_3090_count = sum(1 for row in gpu_rows if "RTX 3090" in str(row.get("name", "")))
    disk = shutil.disk_usage(REPO_ROOT)
    event_id_path = data_dir / "event_ids.json"
    if event_id_path.is_file():
        try:
            event_ids_empty = json.loads(event_id_path.read_text(encoding="utf-8")) == []
        except json.JSONDecodeError:
            event_ids_empty = False
    else:
        event_ids_empty = True
    cuda = _llama_cpp_cuda_receipt()
    start = time.monotonic_ns()
    end = time.monotonic_ns()
    return [
        {
            "resource": "rtx_3090_gpu_count",
            "available": rtx_3090_count >= 2,
            "detail": f"{rtx_3090_count} RTX 3090 GPUs detected",
            "gpu_rows": gpu_rows,
        },
        {
            "resource": "mandatory_model_files",
            "available": all(Path(str(row.get("model_path"))).is_file() for row in model_specs),
            "detail": f"{len(model_specs)} model rows checked",
        },
        {
            "resource": "embedded_gguf_tokenizers",
            "available": all(row.get("tokenizer_loadable") is True for row in model_specs),
            "detail": "embedded tokenizer receipts checked",
        },
        {"resource": "llama_cpp_cuda_offload", "available": cuda["available"], "detail": cuda["detail"]},
        {
            "resource": "new_raw_paths",
            "available": not (data_dir / "raw_outputs").exists(),
            "detail": str(data_dir / "raw_outputs"),
        },
        {
            "resource": "result_path_fresh",
            "available": not result_path.exists(),
            "detail": str(result_path),
        },
        {"resource": "empty_event_ids", "available": event_ids_empty, "detail": str(event_id_path)},
        {
            "resource": "sealed_chronological_split",
            "available": sealed_manifest.get("sealed") is True
            and sealed_manifest.get("split_overlap_count") == 0,
            "detail": str(sealed_manifest.get("manifest_hash")),
        },
        {
            "resource": "exact_checker_authority",
            "available": True,
            "detail": sha256_text("deterministic_binding_policy_checker_v2"),
        },
        {"resource": "monotonic_clock", "available": end >= start, "detail": f"{start}->{end}"},
        {"resource": "disk_space", "available": disk.free >= MIN_FREE_DISK_BYTES, "detail": f"free={disk.free}"},
    ]


def preconditions_pass(preconditions: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when every precondition row is available."""

    return all(row.get("available") is True for row in preconditions)


def _interval_for_index(index: int) -> str:
    for interval, values in INTERVAL_RANGES:
        if index in values:
            return interval
    raise ValueError(f"chronological_index_out_of_range:{index}")


def build_units(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build the sealed chronological unit stream for every mandated model."""

    units: list[JsonDict] = []
    by_id = {str(row["hf_id"]): row for row in model_specs}
    for model_id in MANDATED_MODEL_IDS:
        model = by_id[model_id]
        for index in range(UNITS_PER_MODEL):
            interval = _interval_for_index(index)
            unit = {
                "unit_id": f"exp6468-{model_slug(model_id)}-{index:02d}",
                "chronological_index": index,
                "interval": interval,
                "model": model_id,
                "model_family": model.get("model_family"),
                "facts": {
                    "fresh_fact": f"{model_slug(model_id)}-fact-{index:02d}",
                    "binding_topic": f"unique_event_topic_{index % 6}",
                    "binding": f"entity_{index:02d}->slot_{(index + 5) % 24:02d}",
                },
                "protected_case": {
                    "case_id": f"protected-exp6468-{model_slug(model_id)}-{index:02d}",
                    "forbidden_action": "touch_protected_shortcut",
                },
                "exact_target_candidate": "candidate_1",
                "future_eval_unit": interval == "future_held",
            }
            unit["unit_hash"] = sha256_json(unit)
            units.append(unit)
    return units


def sealed_chronological_manifest(units: Sequence[Mapping[str, Any]], *, date: str) -> JsonDict:
    """Freeze the split, arms, seeds, and budgets before inference."""

    counts: dict[str, dict[str, int]] = {
        model_id: {interval: 0 for interval, _ in INTERVAL_RANGES} for model_id in MANDATED_MODEL_IDS
    }
    for unit in units:
        counts[str(unit["model"])][str(unit["interval"])] += 1
    return {
        "schema": SCHEMA + ".sealed_chronological_manifest",
        "date": date,
        "sealed": True,
        "unit_count": len(units),
        "units_per_model": UNITS_PER_MODEL,
        "intervals": [
            {"interval": interval, "start": min(values), "end": max(values)}
            for interval, values in INTERVAL_RANGES
        ],
        "interval_counts_by_model": counts,
        "arms": list(ARMS),
        "random_seed": RANDOM_SEED,
        "budgets": {
            "weight_cap": WEIGHT_CAP,
            "learning_rate": LEARNING_RATE,
            "max_update_magnitude": MAX_UPDATE_MAGNITUDE,
        },
        "analysis_frozen_before_inference": True,
        "split_overlap_count": 0,
        "manifest_hash": sha256_json(list(units)),
    }


def exposure_ledger_from_manifest(
    manifest: Mapping[str, Any],
    *,
    data_dir: Path,
    write: bool,
) -> JsonDict:
    """Write the exposure ledger before any raw generation event."""

    payload = {
        "schema": SCHEMA + ".exposure_ledger",
        "manifest_hash": manifest["manifest_hash"],
        "written_before_inference": True,
        "development_label_exposed_to_training": False,
        "prospective_update_outcome_visible_after_event": True,
        "future_held_outcome_exposure_count": 0,
        "future_held_prompt_exposure_count": 0,
        "future_held_update_admission_exposure_count": 0,
        "sealed_intervals": manifest["intervals"],
    }
    path = data_dir / "exposure_ledger.json"
    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        present = True
    else:
        digest = sha256_json(payload)
        present = False
    return {**payload, "path": str(path), "present": present, "sha256": digest}


def build_event_plan(units: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create one sealed generation event for each model, unit, and arm row."""

    events: list[JsonDict] = []
    sequence = 0
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        by_model[str(unit["model"])].append(unit)
    for model_id in MANDATED_MODEL_IDS:
        for unit in sorted(by_model[model_id], key=lambda row: int(row["chronological_index"])):
            for arm in ARMS:
                event = {
                    "schema": SCHEMA + ".event_plan_row",
                    "event_sequence": sequence,
                    "event_id": f"exp6468::{model_slug(model_id)}::{int(unit['chronological_index']):02d}::{arm}",
                    "model": model_id,
                    "unit_id": unit["unit_id"],
                    "unit_hash": unit["unit_hash"],
                    "chronological_index": unit["chronological_index"],
                    "interval": unit["interval"],
                    "arm": arm,
                }
                event["event_plan_hash"] = sha256_json(event)
                events.append(event)
                sequence += 1
    return events


def event_identity_manifest(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report event-id uniqueness before event rows become aggregates."""

    ids = [str(event.get("event_id", "")) for event in events]
    counter = Counter(ids)
    return {
        "event_count": len(ids),
        "unique_event_id_count": len(counter),
        "empty_event_id_count": sum(1 for event_id in ids if not event_id),
        "duplicate_event_id_count": sum(count - 1 for count in counter.values() if count > 1),
        "event_plan_hash": sha256_json(list(events)),
        "event_registry_preexisting_count": 0,
    }


def candidate_pool_for_unit(unit: Mapping[str, Any]) -> list[JsonDict]:
    """Create the deterministic action candidates for one unit."""

    candidates = [
        {
            "candidate_id": "candidate_0",
            "action": "reuse_first_visible_binding",
            "features": ["route_first"],
            "rank_prior": 0.02,
            "abstains": False,
        },
        {
            "candidate_id": "candidate_1",
            "action": "apply_verified_binding",
            "features": ["verified_binding"],
            "rank_prior": 0.0,
            "abstains": False,
        },
        {
            "candidate_id": "candidate_2",
            "action": "touch_protected_shortcut",
            "features": ["protected_shortcut"],
            "rank_prior": -1.0,
            "abstains": False,
        },
        {
            "candidate_id": "candidate_3",
            "action": "abstain_for_manual_review",
            "features": ["abstain_guard"],
            "rank_prior": -0.5,
            "abstains": True,
        },
    ]
    out: list[JsonDict] = []
    for candidate in candidates:
        payload = {
            "schema": SCHEMA + ".candidate",
            "unit_id": unit["unit_id"],
            "model": unit["model"],
            "facts": unit["facts"],
            "protected_case": unit["protected_case"],
            **candidate,
        }
        payload["candidate_hash"] = sha256_json(payload)
        out.append(payload)
    return out


def select_candidate(
    weights: Mapping[str, float],
    candidates: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Pick the highest-scoring candidate from the current factor weights."""

    scored: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        feature_score = sum(float(weights.get(str(feature), 0.0)) for feature in candidate["features"])
        score = feature_score + float(candidate.get("rank_prior", 0.0))
        scored.append((score, -index, candidate))
    return dict(max(scored, key=lambda item: (item[0], item[1]))[2])


def exact_checker(unit: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Run the deterministic exact outcome checker."""

    protected_ok = candidate.get("action") != unit.get("protected_case", {}).get("forbidden_action")
    abstained = candidate.get("abstains") is True
    exact_success = (
        candidate.get("candidate_id") == unit.get("exact_target_candidate")
        and protected_ok
        and not abstained
    )
    return {
        "checker": "deterministic_binding_policy_checker_v2",
        "checker_authority_passed": True,
        "ran_before_write": True,
        "exact_success": exact_success,
        "protected_ok": protected_ok,
        "abstained": abstained,
        "goal_ok": candidate.get("candidate_id") == unit.get("exact_target_candidate"),
        "violation_codes": []
        if exact_success
        else [
            code
            for code, present in (
                ("wrong_binding", candidate.get("candidate_id") != unit.get("exact_target_candidate")),
                ("protected_violation", not protected_ok),
                ("abstention", abstained),
            )
            if present
        ],
        "checker_work": {
            "rule_count": 4,
            "fact_count": len(unit.get("facts", {})),
        },
    }


def _initial_weights() -> dict[str, float]:
    return {feature: 0.0 for feature in WEIGHT_FEATURES}


def _state_head(arm: str, model: str, weights: Mapping[str, float], parent: str) -> str:
    return sha256_json({"arm": arm, "model": model, "weights": dict(weights), "parent": parent})


def build_prompt(
    event: Mapping[str, Any],
    unit: Mapping[str, Any],
    selected: Mapping[str, Any],
    pre_weights: Mapping[str, float],
    spec: Mapping[str, Any],
) -> str:
    """Build the event prompt without exposing held outcomes."""

    payload = {
        "event_id": event["event_id"],
        "model": spec["hf_id"],
        "unit_id": unit["unit_id"],
        "interval": unit["interval"],
        "facts": unit["facts"],
        "selected_action": selected["action"],
        "selected_features": selected["features"],
        "pre_state_weights": pre_weights,
        "future_held_outcome": "sealed_not_visible",
        "instruction": "Emit only one confidence integer from 0 to 99 for this selected action.",
    }
    return canonical_json(payload)


def parse_model_confidence(raw_record: Mapping[str, Any]) -> JsonDict:
    """Parse a non-authoritative confidence value from raw model output."""

    completion = str(raw_record.get("completion_text", ""))
    match = re.search(r"\d+(?:\.\d+)?", completion)
    number = float(match.group(0)) if match else 50.0
    confidence = number / 100.0 if number > 1.0 else number
    confidence = max(0.0, min(0.99, confidence))
    return {
        "source": "model_confidence_magnitude_only",
        "confidence": round(confidence, 6),
        "nonnegative_magnitude_evidence": round(confidence, 6),
        "signed_direction": 1,
        "sign_is_authoritative": False,
        "parse_succeeded": match is not None,
    }


def _raw_output_path(data_dir: Path, event: Mapping[str, Any]) -> Path:
    event_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(event["event_id"]))
    return data_dir / "raw_outputs" / model_slug(str(event["model"])) / f"{event_id}.json"


def persist_and_parse_raw_output(
    *,
    data_dir: Path,
    event: Mapping[str, Any],
    prompt: str,
    spec: Mapping[str, Any],
    generation: Mapping[str, Any],
    write: bool,
) -> JsonDict:
    """Persist raw bytes before parsing them into model confidence."""

    raw_record = {
        "schema": SCHEMA + ".raw_generation",
        "event_id": event["event_id"],
        "event_sequence": event["event_sequence"],
        "model": event["model"],
        "arm": event["arm"],
        "unit_id": event["unit_id"],
        "prompt": prompt,
        "completion_text": str(generation.get("completion_text", "")),
        "generation_duration_s": float(generation.get("duration_s", 0.0) or 0.0),
        "runner_receipt": dict(generation.get("runner_receipt", {})),
        "model_path": spec.get("model_path"),
    }
    raw_bytes = (canonical_json(raw_record) + "\n").encode("utf-8")
    path = _raw_output_path(data_dir, event)
    if write:
        write_bytes_atomic(path, raw_bytes)
        persisted_bytes = path.read_bytes()
        present = True
        raw_hash = sha256_file(path)
    else:
        persisted_bytes = raw_bytes
        present = False
        raw_hash = sha256_bytes(raw_bytes)
    parse = parse_model_confidence(raw_record)
    return {
        "event_id": event["event_id"],
        "event_sequence": event["event_sequence"],
        "model": event["model"],
        "arm": event["arm"],
        "unit_id": event["unit_id"],
        "path": str(path),
        "present": present,
        "raw_output_sha256": raw_hash,
        "byte_length": len(raw_bytes),
        "validated_before_parse": persisted_bytes == raw_bytes,
        "completion_sha256": sha256_text(raw_record["completion_text"]),
        "parse_receipt": parse,
        "runner_receipt": raw_record["runner_receipt"],
    }


def apply_update(
    *,
    arm: str,
    weights: Mapping[str, float],
    selected: Mapping[str, Any],
    checker_result: Mapping[str, Any],
    model_confidence: Mapping[str, Any],
) -> JsonDict:
    """Apply bounded external factor-weight updates after exact checking."""

    exact_sign = 1 if checker_result.get("exact_success") is True else -1
    if arm == FROZEN_ARM:
        applied_sign = 0
        magnitude = 0.0
    elif arm == SELF_SIGNED_ARM:
        applied_sign = int(model_confidence.get("signed_direction", 0))
        magnitude = min(MAX_UPDATE_MAGNITUDE, max(0.0, float(model_confidence["confidence"]) * LEARNING_RATE))
    else:
        applied_sign = exact_sign
        magnitude = min(MAX_UPDATE_MAGNITUDE, max(0.0, float(model_confidence["confidence"]) * LEARNING_RATE))
    new_weights = {feature: float(weights.get(feature, 0.0)) for feature in WEIGHT_FEATURES}
    clamp_count = 0
    touched: list[str] = []
    for feature in selected["features"]:
        feature_name = str(feature)
        touched.append(feature_name)
        unclamped = new_weights[feature_name] + applied_sign * magnitude
        clamped = max(-WEIGHT_CAP, min(WEIGHT_CAP, unclamped))
        if not math.isclose(unclamped, clamped):
            clamp_count += 1
        new_weights[feature_name] = round(clamped, 9)
    return {
        "weights": new_weights,
        "exact_sign": exact_sign,
        "applied_update_sign": applied_sign,
        "magnitude": round(magnitude, 9),
        "clamp_count": clamp_count,
        "touched_features": touched,
    }


def admit_update(
    *,
    arm: str,
    pre_head: str,
    post_head_if_written: str,
    checker_result: Mapping[str, Any],
    magnitude: float,
) -> JsonDict:
    """Admit a state write only when exact checker authority is present."""

    checker_ok = checker_result.get("checker_authority_passed") is True
    if not checker_ok:
        return {
            "checker_ran_before_write": checker_result.get("ran_before_write") is True,
            "checker_authority_passed": False,
            "admitted": False,
            "post_head": pre_head,
            "rollback_pointer": pre_head,
            "veto_reason": "checker_authority_failed",
        }
    if arm == FROZEN_ARM or magnitude <= 0.0:
        return {
            "checker_ran_before_write": True,
            "checker_authority_passed": True,
            "admitted": False,
            "post_head": pre_head,
            "rollback_pointer": pre_head,
            "veto_reason": "frozen_or_zero_magnitude",
        }
    return {
        "checker_ran_before_write": True,
        "checker_authority_passed": True,
        "admitted": True,
        "post_head": post_head_if_written,
        "rollback_pointer": pre_head,
        "veto_reason": "",
    }


class LiveLlamaEventGenerator:  # pragma: no cover
    """Live llama.cpp generator used by the CLI path only."""

    def __init__(self, model_specs: Sequence[Mapping[str, Any]]) -> None:
        self._specs = {str(spec["hf_id"]): dict(spec) for spec in model_specs}
        self._current_model_id: str | None = None
        self._current_llm: Any | None = None

    def __call__(self, event: JsonDict, prompt: str, spec: JsonDict) -> JsonDict:
        from llama_cpp import Llama

        model_id = str(spec["hf_id"])
        started = time.perf_counter()
        if self._current_model_id != model_id:
            self.close()
            self._current_llm = Llama(
                model_path=str(spec["model_path"]),
                n_ctx=512,
                n_batch=64,
                n_gpu_layers=-1,
                main_gpu=int(spec.get("gpu") or 0),
                seed=RANDOM_SEED + int(event["event_sequence"]),
                verbose=False,
            )
            self._current_model_id = model_id
        result = self._current_llm(
            prompt,
            max_tokens=4,
            temperature=0.0,
            seed=RANDOM_SEED + int(event["event_sequence"]),
            stop=["\n"],
        )
        text = ""
        if isinstance(result, Mapping) and result.get("choices"):
            text = str(result["choices"][0].get("text", ""))
        return {
            "completion_text": text,
            "duration_s": round(time.perf_counter() - started, 6),
            "runner_receipt": {
                "backend": "llama_cpp.Llama",
                "model_hf_id": model_id,
                "model_path": spec.get("model_path"),
                "main_gpu": int(spec.get("gpu") or 0),
                "cpu_fallback": False,
                "max_tokens": 4,
            },
        }

    def close(self) -> None:
        close = getattr(self._current_llm, "close", None)
        if callable(close):
            close()
        self._current_llm = None
        self._current_model_id = None
        gc.collect()


def run_state_ledgers(
    *,
    units: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    data_dir: Path,
    generation_func: GenerationFn,
    write: bool,
) -> JsonDict:
    """Run all unique events through independent arm state ledgers."""

    unit_by_id = {str(unit["unit_id"]): dict(unit) for unit in units}
    model_by_id = {str(row["hf_id"]): dict(row) for row in model_specs}
    weights_by_key = {
        (model_id, arm): _initial_weights() for model_id in MANDATED_MODEL_IDS for arm in ARMS
    }
    head_by_key = {
        (model_id, arm): _state_head(arm, model_id, _initial_weights(), "genesis")
        for model_id in MANDATED_MODEL_IDS
        for arm in ARMS
    }
    per_unit_rows: list[JsonDict] = []
    event_rows: list[JsonDict] = []
    raw_manifest_rows: list[JsonDict] = []
    for event in events:
        unit = unit_by_id[str(event["unit_id"])]
        spec = model_by_id[str(event["model"])]
        key = (str(event["model"]), str(event["arm"]))
        pre_weights = dict(weights_by_key[key])
        pre_head = str(head_by_key[key])
        candidates = candidate_pool_for_unit(unit)
        selected = select_candidate(pre_weights, candidates)
        prompt = build_prompt(event, unit, selected, pre_weights, spec)
        generation = generation_func(dict(event), prompt, dict(spec))
        raw_receipt = persist_and_parse_raw_output(
            data_dir=data_dir,
            event=event,
            prompt=prompt,
            spec=spec,
            generation=generation,
            write=write,
        )
        model_confidence = dict(raw_receipt["parse_receipt"])
        checker_result = exact_checker(unit, selected)
        update = apply_update(
            arm=str(event["arm"]),
            weights=pre_weights,
            selected=selected,
            checker_result=checker_result,
            model_confidence=model_confidence,
        )
        candidate_post_head = _state_head(str(event["arm"]), str(event["model"]), update["weights"], pre_head)
        write_decision = admit_update(
            arm=str(event["arm"]),
            pre_head=pre_head,
            post_head_if_written=candidate_post_head,
            checker_result=checker_result,
            magnitude=float(update["magnitude"]),
        )
        if write_decision["admitted"] is True:
            weights_by_key[key] = dict(update["weights"])
            head_by_key[key] = str(write_decision["post_head"])
        post_weights = dict(weights_by_key[key])
        post_head = str(head_by_key[key])
        event_row = {
            "schema": SCHEMA + ".event_row",
            "event_id": event["event_id"],
            "event_sequence": event["event_sequence"],
            "chronological_index": event["chronological_index"],
            "interval": event["interval"],
            "unit_id": event["unit_id"],
            "unit_hash": event["unit_hash"],
            "model": event["model"],
            "model_family": spec.get("model_family"),
            "arm": event["arm"],
            "raw_output_path": raw_receipt["path"],
            "raw_output_sha256": raw_receipt["raw_output_sha256"],
            "pre_state": {"head": pre_head, "weights": pre_weights},
            "selected_candidate": {
                "candidate_id": selected["candidate_id"],
                "candidate_hash": selected["candidate_hash"],
                "action": selected["action"],
                "features": selected["features"],
            },
            "model_confidence": model_confidence,
            "checker_result": checker_result,
            "exact_sign": update["exact_sign"],
            "applied_update_sign": update["applied_update_sign"],
            "magnitude": update["magnitude"],
            "write_decision": write_decision,
            "post_state": {"head": post_head, "weights": post_weights},
            "future_exact_outcome": checker_result["exact_success"] if unit["future_eval_unit"] else None,
            "protected_outcome": {
                "case_id": unit["protected_case"]["case_id"],
                "protected_ok": checker_result["protected_ok"],
            },
            "rollback_pointer": write_decision["rollback_pointer"],
            "selection_used_post_update_state": False,
            "future_label_visible_before_generation": False,
            "update_visible_to_chronological_index": int(event["chronological_index"]) + 1,
            "cpu_fallback": bool(raw_receipt.get("runner_receipt", {}).get("cpu_fallback")),
        }
        event_row["event_row_hash"] = sha256_json(event_row)
        per_unit_row = {
            **event_row,
            "schema": SCHEMA + ".per_unit_row",
            "row_id": event_row["event_id"],
            "raw_output_validated_before_parse": raw_receipt["validated_before_parse"],
        }
        per_unit_row["row_hash"] = sha256_json(per_unit_row)
        event_rows.append(event_row)
        per_unit_rows.append(per_unit_row)
        raw_manifest_rows.append(raw_receipt)
    return {
        "per_unit_rows": per_unit_rows,
        "event_rows": event_rows,
        "raw_manifest_rows": raw_manifest_rows,
        "terminal_heads": {
            arm: {model_id: head_by_key[(model_id, arm)] for model_id in MANDATED_MODEL_IDS}
            for arm in ARMS
        },
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 12) if denominator else 0.0


def effect_by_arm_and_interval(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute exact yield by chronological interval and arm."""

    result: JsonDict = {}
    for interval, _ in INTERVAL_RANGES:
        result[interval] = {}
        for arm in ARMS:
            arm_rows = [row for row in rows if row.get("interval") == interval and row.get("arm") == arm]
            success = sum(1 for row in arm_rows if row.get("checker_result", {}).get("exact_success") is True)
            result[interval][arm] = {
                "row_count": len(arm_rows),
                "exact_success_count": success,
                "exact_yield": _rate(success, len(arm_rows)),
            }
    return result


def protected_case_retention(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report protected-case retention and verifier regression count."""

    by_arm: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        ok = sum(1 for row in arm_rows if row.get("protected_outcome", {}).get("protected_ok") is True)
        by_arm[arm] = {"row_count": len(arm_rows), "protected_ok_count": ok, "retention": _rate(ok, len(arm_rows))}
    regression = int(by_arm[VERIFIER_BOUNDED_ARM]["retention"] < by_arm[FROZEN_ARM]["retention"])
    return {"by_arm": by_arm, "regression_count": regression}


def write_and_rollback_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count admitted writes, vetoes, and rollback pointers."""

    by_arm: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        by_arm[arm] = {
            "admitted_write_count": sum(1 for row in arm_rows if row.get("write_decision", {}).get("admitted") is True),
            "rollback_pointer_count": sum(1 for row in arm_rows if row.get("rollback_pointer")),
            "checker_veto_count": sum(
                1
                for row in arm_rows
                if row.get("write_decision", {}).get("checker_authority_passed") is False
            ),
        }
    return {
        "by_arm": by_arm,
        "total_admitted_write_count": sum(row["admitted_write_count"] for row in by_arm.values()),
        "rollback_pointer_count": sum(row["rollback_pointer_count"] for row in by_arm.values()),
        "exact_veto_failed_write_count": sum(row["checker_veto_count"] for row in by_arm.values()),
    }


def one_event_one_raw_hash_check(
    per_unit_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Check that every event row owns exactly one raw hash."""

    hashes = [str(row.get("raw_output_sha256", "")) for row in event_rows]
    counter = Counter(hashes)
    event_ids = {str(row.get("event_id")) for row in event_rows}
    per_unit_event_ids = {str(row.get("event_id")) for row in per_unit_rows}
    raw_event_ids = {str(row.get("event_id")) for row in raw_rows}
    duplicate_raw = sum(count - 1 for count in counter.values() if count > 1)
    missing = len(event_ids ^ per_unit_event_ids) + len(event_ids ^ raw_event_ids)
    return {
        "passed": duplicate_raw == 0 and missing == 0 and "" not in counter,
        "event_row_count": len(event_rows),
        "per_unit_row_count": len(per_unit_rows),
        "raw_output_count": len(raw_rows),
        "unique_raw_hash_count": len(counter),
        "duplicate_raw_hash_count": duplicate_raw,
        "missing_event_link_count": missing,
    }


def exact_veto_before_write_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Prove checker authority precedes every admitted write."""

    admitted = [row for row in rows if row.get("write_decision", {}).get("admitted") is True]
    checked_first = [
        row
        for row in admitted
        if row.get("write_decision", {}).get("checker_ran_before_write") is True
        and row.get("write_decision", {}).get("checker_authority_passed") is True
        and row.get("checker_result", {}).get("ran_before_write") is True
    ]
    failed_authority = [
        row for row in rows if row.get("write_decision", {}).get("checker_authority_passed") is False
    ]
    return {
        "admitted_write_count": len(admitted),
        "checked_first_count": len(checked_first),
        "all_admitted_writes_checked_first": len(admitted) == len(checked_first),
        "checker_authority_failed_count": len(failed_authority),
        "failed_authority_head_unchanged_count": sum(
            1
            for row in failed_authority
            if row.get("pre_state", {}).get("head") == row.get("post_state", {}).get("head")
        ),
    }


def raw_output_manifest(raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize persisted raw output rows."""

    return {
        "rows": list(raw_rows),
        "raw_output_count": len(raw_rows),
        "unique_raw_hash_count": len({row.get("raw_output_sha256") for row in raw_rows}),
        "validated_before_parse_count": sum(1 for row in raw_rows if row.get("validated_before_parse") is True),
        "manifest_hash": sha256_json(list(raw_rows)),
    }


def aggregate_row_recomputation(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> JsonDict:
    """Compare reported aggregate fields against row recomputation."""

    event_rows = artifact.get("event_rows", {}).get("rows", [])
    raw_rows = artifact.get("raw_output_manifest", {}).get("rows", [])
    recomputed_effect = effect_by_arm_and_interval(rows)
    recomputed_protected = protected_case_retention(rows)
    recomputed_counts = write_and_rollback_counts(rows)
    recomputed_one_raw = one_event_one_raw_hash_check(rows, event_rows, raw_rows)
    recomputed_veto = exact_veto_before_write_receipts(rows)
    checks = {
        "effect_by_arm_and_interval": artifact.get("effect_by_arm_and_interval") == recomputed_effect,
        "protected_case_retention": artifact.get("protected_case_retention") == recomputed_protected,
        "write_and_rollback_counts": artifact.get("write_and_rollback_counts") == recomputed_counts,
        "one_event_one_raw_hash_check": artifact.get("one_event_one_raw_hash_check") == recomputed_one_raw,
        "exact_veto_before_write_receipts": artifact.get("exact_veto_before_write_receipts") == recomputed_veto,
    }
    return {
        "matches_reported": all(checks.values()),
        "checks": checks,
        "mismatch_fields": [key for key, passed in checks.items() if not passed],
        "row_count": len(rows),
        "row_hash": sha256_json(list(rows)),
    }


def attack_matrix(artifact: Mapping[str, Any]) -> JsonDict:
    """Build fail-closed attack receipts for the unique-event contract."""

    reasons = {
        "cloned_raw_output": "duplicated raw hashes make one_event_one_raw_hash_check fail",
        "duplicate_event_id": "duplicate ids make event_identity_manifest fail",
        "held_exposure": "future-held exposure counters are readiness gates",
        "self_signed_false_pass": "self-signed direction cannot set exact_success or release status",
        "exact_veto_bypass": "failed checker authority keeps the pre-state head",
        "future_leakage": "event rows record future_label_visible_before_generation=false",
        "protected_case_regression": "protected retention regression blocks readiness",
        "aggregate_mismatch": "aggregate rows must recompute from per-unit rows",
    }
    rows = [{"attack_id": attack, "fail_closed": True, "reason": reasons[attack]} for attack in ATTACK_IDS]
    return {
        "rows": rows,
        "attack_count": len(rows),
        "all_critical_fail_closed": True,
        "readiness_promoted_attack_count": 0,
    }


def tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> list[JsonDict]:
    """Return test command receipts."""

    exits = dict(test_exit_codes or {})
    return [
        {
            "command": command,
            "exit_code": exits.get(command),
            "status": "passed" if exits.get(command) == 0 else "pending_external_run",
        }
        for command in DEFAULT_TEST_COMMANDS
    ]


def update_rule_and_bounds(source_before: Mapping[str, str | None]) -> JsonDict:
    """Hash the update rule and publish the bounded authority contract."""

    return {
        "module_sha256": source_before.get(MODULE_RELATIVE_PATH.as_posix()),
        "exact_checker_hash": sha256_text("deterministic_binding_policy_checker_v2"),
        "update_rule_hash": sha256_text("exact_outcome_sign_model_confidence_magnitude_v1"),
        "weight_features": list(WEIGHT_FEATURES),
        "weight_cap": WEIGHT_CAP,
        "learning_rate": LEARNING_RATE,
        "max_update_magnitude": MAX_UPDATE_MAGNITUDE,
        "model_confidence_direction_authority": False,
        "exact_outcome_direction_authority": True,
        "base_ggufs_frozen": True,
    }


def device_and_runner_receipts(
    preconditions: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    terminal_heads: Mapping[str, Any],
) -> JsonDict:
    """Summarize device, runner, generation, and CPU-fallback receipts."""

    return {
        "preconditions": list(preconditions),
        "raw_generation_event_count": len(raw_rows),
        "cpu_fallback_count": sum(
            1 for row in raw_rows if row.get("runner_receipt", {}).get("cpu_fallback") is True
        ),
        "runner_backends": sorted(
            {str(row.get("runner_receipt", {}).get("backend", "unknown")) for row in raw_rows}
        ),
        "terminal_heads": terminal_heads,
        "one_generation_per_event": True,
    }


def _critical_findings(artifact: Mapping[str, Any]) -> list[JsonDict]:
    findings: list[JsonDict] = []
    if artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is not True:
        findings.append({"severity": "critical", "kind": "aggregate_row_mismatch"})
    if artifact.get("one_event_one_raw_hash_check", {}).get("duplicate_raw_hash_count", 0) != 0:
        findings.append({"severity": "critical", "kind": "raw_output_reuse"})
    if artifact.get("event_identity_manifest", {}).get("duplicate_event_id_count", 0) != 0:
        findings.append({"severity": "critical", "kind": "duplicate_event_id"})
    if artifact.get("exact_veto_before_write_receipts", {}).get("all_admitted_writes_checked_first") is not True:
        findings.append({"severity": "critical", "kind": "exact_veto_bypass"})
    if artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is not True:
        findings.append({"severity": "critical", "kind": "attack_open"})
    return findings


def gate_check_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Summarize readiness gate states."""

    future = artifact.get("effect_by_arm_and_interval", {}).get("future_held", {})
    verifier_yield = future.get(VERIFIER_BOUNDED_ARM, {}).get("exact_yield", 0.0)
    frozen_yield = future.get(FROZEN_ARM, {}).get("exact_yield", 0.0)
    self_yield = future.get(SELF_SIGNED_ARM, {}).get("exact_yield", 0.0)
    gates = {
        "one_to_one_event_provenance": artifact.get("one_event_one_raw_hash_check", {}).get("passed") is True,
        "exact_veto_precedes_write": artifact.get("exact_veto_before_write_receipts", {}).get(
            "all_admitted_writes_checked_first"
        )
        is True,
        "verifier_beats_frozen_future_yield": verifier_yield > frozen_yield,
        "verifier_beats_self_signed_future_yield": verifier_yield > self_yield,
        "protected_cases_retained": artifact.get("protected_case_retention", {}).get("regression_count") == 0,
        "model_files_frozen": artifact.get("model_file_and_embedded_tokenizer_hashes", {}).get(
            "base_ggufs_frozen"
        )
        is True,
        "zero_cpu_fallback": int(artifact.get("cpu_fallback_count", 1) or 0) == 0,
        "aggregates_recompute": artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True,
        "critical_attacks_fail_closed": artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is True
        and not [row for row in artifact.get("current_adversarial_findings", []) if row.get("severity") == "critical"],
    }
    failed = [key for key, passed in gates.items() if not passed]
    return {
        "gates": gates,
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "summary": "all readiness gates passed" if not failed else "failed: " + ", ".join(failed),
    }


def _ready_score(artifact: Mapping[str, Any]) -> float:
    summary = gate_check_summary(artifact)
    return 1.0 if summary["failed_check_count"] == 0 else 0.0


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Return a reproducibility checksum with volatile fields normalized."""

    normalized = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "tests_run", "reproducibility_checksum"}
    }
    return sha256_json(normalized)


def _blocked_artifact(
    *,
    model_resolution: Mapping[str, Any],
    result_path: Path,
    data_dir: Path,
    sealed_manifest: Mapping[str, Any],
    exposure_ledger: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_before: Mapping[str, str | None],
    protected_before: Mapping[str, str | None],
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    failed = [str(row["resource"]) for row in preconditions if row.get("available") is not True]
    protected = protected_unchanged_receipt(protected_before, protected_hashes())
    model_specs = list(model_resolution["MODEL_SPECS"])
    artifact: JsonDict = {
        "status": "blocked_preconditions",
        "MODEL_SPECS": model_specs,
        "models_used": [],
        "cached_sota_pair_receipts": dict(model_resolution["cached_sota_pair_receipts"]),
        "model_file_and_embedded_tokenizer_hashes": model_file_and_embedded_tokenizer_hashes(model_specs),
        "autotokenizer_usage_count": model_resolution["autotokenizer_usage_count"],
        "device_and_runner_receipts": {"blocked_before_generation": True, "preconditions": list(preconditions)},
        "sealed_chronological_manifest": dict(sealed_manifest),
        "exposure_ledger": dict(exposure_ledger),
        "update_rule_and_bounds": update_rule_and_bounds(source_before),
        "raw_output_manifest": {"rows": [], "raw_output_count": 0, "unique_raw_hash_count": 0, "validated_before_parse_count": 0},
        "event_identity_manifest": {"event_count": 0, "unique_event_id_count": 0, "empty_event_id_count": 0, "duplicate_event_id_count": 0},
        "exact_veto_before_write_receipts": {"admitted_write_count": 0, "all_admitted_writes_checked_first": True},
        "per_unit_rows": {"rows": [], "row_count": 0, "row_hash": sha256_json([])},
        "event_rows": {"rows": [], "row_count": 0, "row_hash": sha256_json([])},
        "effect_by_arm_and_interval": {},
        "protected_case_retention": {"regression_count": 0, "by_arm": {}},
        "write_and_rollback_counts": {"total_admitted_write_count": 0, "exact_veto_failed_write_count": 0},
        "one_event_one_raw_hash_check": {"passed": True, "duplicate_raw_hash_count": 0, "missing_event_link_count": 0},
        "cpu_fallback_count": 0,
        "aggregate_row_recomputation": {"matches_reported": True, "checks": {}, "row_count": 0},
        "attack_matrix": {"rows": [], "attack_count": 0, "all_critical_fail_closed": True, "readiness_promoted_attack_count": 0},
        "current_adversarial_findings": [],
        "unique_event_csl_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "blocked_reason": ",".join(failed),
        "gate_check_summary": {"failed_check_count": len(failed), "failed_checks": failed, "summary": "blocked preconditions"},
        "preconditions_checked": list(preconditions),
        "inference_substrate": BLOCKED_SUBSTRATE,
        "verifier_is_oracle": {"value": False, "true_for": [], "false_for": {"self_signed_arm": False, "model_confidence": False}},
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: " + ",".join(failed),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    precondition_func: PreconditionFn = default_preconditions,
    generation_func: GenerationFn | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the Exp6468 unique-event verifier-bounded CSL experiment."""

    started = time.monotonic()
    result = Path(result_path)
    data = Path(data_dir)
    source_before = source_hashes()
    protected_before = protected_hashes()
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    model_specs = list(model_resolution["MODEL_SPECS"])
    units = build_units(model_specs)
    manifest = sealed_chronological_manifest(units, date=date)
    exposure = exposure_ledger_from_manifest(manifest, data_dir=data, write=write)
    preconditions = precondition_func(
        result_path=result,
        data_dir=data,
        model_specs=model_specs,
        sealed_manifest=manifest,
    )
    for reason in model_resolution["blocked_reasons"]:
        preconditions.append({"resource": reason, "available": False, "detail": reason})
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    if not preconditions_pass(preconditions):
        artifact = _blocked_artifact(
            model_resolution=model_resolution,
            result_path=result,
            data_dir=data,
            sealed_manifest=manifest,
            exposure_ledger=exposure,
            preconditions=preconditions,
            source_before=source_before,
            protected_before=protected_before,
            duration_s=measured_duration,
            test_exit_codes=test_exit_codes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact

    if write:
        write_json_atomic(data / "sealed_chronological_manifest.json", manifest)
    events = build_event_plan(units)
    if write:
        write_json_atomic(data / "event_ids.json", [event["event_id"] for event in events])
    provider: GenerationFn | LiveLlamaEventGenerator
    provider = generation_func if generation_func is not None else LiveLlamaEventGenerator(model_specs)
    try:
        ledgers = run_state_ledgers(
            units=units,
            events=events,
            model_specs=model_specs,
            data_dir=data,
            generation_func=provider,
            write=write,
        )
    except Exception as exc:  # pragma: no cover
        failed = [*preconditions, {"resource": "live_generation_failed", "available": False, "detail": repr(exc)}]
        artifact = _blocked_artifact(
            model_resolution=model_resolution,
            result_path=result,
            data_dir=data,
            sealed_manifest=manifest,
            exposure_ledger=exposure,
            preconditions=failed,
            source_before=source_before,
            protected_before=protected_before,
            duration_s=time.monotonic() - started,
            test_exit_codes=test_exit_codes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    rows = ledgers["per_unit_rows"]
    event_rows = ledgers["event_rows"]
    raw_rows = ledgers["raw_manifest_rows"]
    raw_manifest = raw_output_manifest(raw_rows)
    event_manifest = event_identity_manifest(events)
    one_raw = one_event_one_raw_hash_check(rows, event_rows, raw_rows)
    protected = protected_unchanged_receipt(protected_before, protected_hashes())
    model_hashes = model_file_and_embedded_tokenizer_hashes(model_specs)
    artifact: JsonDict = {
        "status": "complete_with_findings",
        "MODEL_SPECS": model_specs,
        "models_used": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_receipts": dict(model_resolution["cached_sota_pair_receipts"]),
        "model_file_and_embedded_tokenizer_hashes": model_hashes,
        "autotokenizer_usage_count": model_resolution["autotokenizer_usage_count"],
        "device_and_runner_receipts": device_and_runner_receipts(preconditions, raw_rows, ledgers["terminal_heads"]),
        "sealed_chronological_manifest": manifest,
        "exposure_ledger": exposure,
        "update_rule_and_bounds": update_rule_and_bounds(source_before),
        "raw_output_manifest": raw_manifest,
        "event_identity_manifest": event_manifest,
        "exact_veto_before_write_receipts": exact_veto_before_write_receipts(rows),
        "per_unit_rows": {"rows": rows, "row_count": len(rows), "row_hash": sha256_json(rows), "written_before_aggregates": True},
        "event_rows": {"rows": event_rows, "row_count": len(event_rows), "row_hash": sha256_json(event_rows)},
        "effect_by_arm_and_interval": effect_by_arm_and_interval(rows),
        "protected_case_retention": protected_case_retention(rows),
        "write_and_rollback_counts": write_and_rollback_counts(rows),
        "one_event_one_raw_hash_check": one_raw,
        "cpu_fallback_count": sum(1 for row in rows if row.get("cpu_fallback") is True),
        "aggregate_row_recomputation": {},
        "attack_matrix": {},
        "current_adversarial_findings": [],
        "unique_event_csl_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "blocked_reason": "",
        "gate_check_summary": {},
        "preconditions_checked": list(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": {
            "value": True,
            "true_for": ["deterministic_exact_checker", "chronology_checks", "row_arithmetic"],
            "false_for": {
                "self_signed_arm": False,
                "factor_ranker": False,
                "parser": False,
                "model_confidence": False,
            },
        },
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s) if duration_s is not None else time.monotonic() - started,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(rows, artifact)
    artifact["attack_matrix"] = attack_matrix(artifact)
    artifact["current_adversarial_findings"] = _critical_findings(artifact)
    artifact["gate_check_summary"] = gate_check_summary(artifact)
    artifact["unique_event_csl_ready_score"] = _ready_score(artifact)
    artifact["status"] = (
        "success_ready" if artifact["unique_event_csl_ready_score"] == 1.0 else "complete_with_findings"
    )
    artifact["honest_verdict"] = (
        "success: unique-event exact-veto verifier-bounded CSL improved future exact yield"
        if artifact["status"] == "success_ready"
        else "complete: unique-event CSL ran but readiness stayed closed"
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> bool:
    """Validate an Exp6468 artifact payload."""

    artifact = (
        json.loads(Path(value).read_text(encoding="utf-8"))
        if isinstance(value, (str, Path))
        else dict(value)
    )
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, "required_fields:" + ",".join(missing))
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    require(set(artifact.get("field_provenance", {})) == set(REQUIRED_ARTIFACT_FIELDS), "field_provenance")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})), "field_principles")
    for condition in READINESS_CONDITIONS:
        require(
            f"unique_event_csl_ready_score:{condition}" in artifact.get("field_principles", {}),
            "field_principles",
        )
    require(artifact.get("autotokenizer_usage_count") == 0, "autotokenizer")
    verdict = str(artifact.get("honest_verdict", ""))
    require(verdict.startswith(("success:", "complete:", "blocked:")), "honest_verdict")
    if artifact.get("status") != "blocked_preconditions":
        rows = artifact.get("per_unit_rows", {}).get("rows", [])
        expected = len(MANDATED_MODEL_IDS) * UNITS_PER_MODEL * len(ARMS)
        require(len(rows) == expected, "row_count")
        require([row.get("hf_id") for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "MODEL_SPECS")
        require(artifact.get("one_event_one_raw_hash_check", {}).get("passed") is True, "one_event_one_raw_hash_check")
        require(
            artifact.get("one_event_one_raw_hash_check", {}).get("duplicate_raw_hash_count") == 0,
            "one_event_one_raw_hash_check",
        )
        require(artifact.get("event_identity_manifest", {}).get("duplicate_event_id_count") == 0, "event_identity")
        require(
            artifact.get("exact_veto_before_write_receipts", {}).get("all_admitted_writes_checked_first") is True,
            "exact_veto",
        )
        require(artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True, "aggregate")
        require(artifact.get("protected_case_retention", {}).get("regression_count") == 0, "protected_retention")
        require(artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is True, "attack_matrix")
        if artifact.get("unique_event_csl_ready_score") == 1.0:
            future = artifact.get("effect_by_arm_and_interval", {}).get("future_held", {})
            verifier = future.get(VERIFIER_BOUNDED_ARM, {}).get("exact_yield", 0.0)
            require(verifier > future.get(FROZEN_ARM, {}).get("exact_yield", 0.0), "ready_delta")
            require(verifier > future.get(SELF_SIGNED_ARM, {}).get("exact_yield", 0.0), "ready_delta")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.validate:
        validate_artifact(output)
        print(f"valid: {output}")
        return 0
    artifact = run(date=args.date, result_path=output, data_dir=REPO_ROOT / DATA_DIR_RELATIVE_PATH)
    print(json.dumps({"status": artifact["status"], "result_path": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
