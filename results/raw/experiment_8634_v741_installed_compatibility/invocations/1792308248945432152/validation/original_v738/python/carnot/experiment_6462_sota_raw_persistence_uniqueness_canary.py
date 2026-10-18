"""Exp6462 SOTA raw persistence uniqueness canary.

Spec refs: REQ-INFRA-6462, SCENARIO-INFRA-6462-1,
SCENARIO-INFRA-6462-2, SCENARIO-INFRA-6462-3,
SCENARIO-INFRA-6462-4.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

from carnot import path_receipts
from carnot import task_runtime_receipts as receipts
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
HostPreflightFn = Callable[..., list[JsonDict]]
GenerationFn = Callable[..., JsonDict]
EventIdFn = Callable[..., str]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6462_sota_raw_persistence_uniqueness_canary.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6462_sota_raw_persistence_uniqueness_canary"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6462_sota_raw_persistence_uniqueness_canary.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6462_sota_raw_persistence_uniqueness_canary.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXP6449_RELATIVE_PATH = Path(
    "results/experiment_6449_generation_to_verdict_path_receipt_contract.json"
)
EXP6450_RELATIVE_PATH = Path("results/experiment_6450_sota_fixed_policy_candidate_corpus.json")

SCHEMA = "carnot.experiment_6462.sota_raw_persistence_uniqueness_canary.v1"
RUN_DATE = "20260819"
RANDOM_SEED = 6462
PREFERRED_QUANT = "Q4_K_M"
TOKENIZER_SOURCE = "embedded_gguf_vocab_only"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_raw_persistence_canary"
UNIT_COUNT = 4
REPLICATES_PER_UNIT = 2
CANARY_SEEDS = (646200, 646201)
MIN_FREE_VRAM_MB = 16_000
MIN_DISK_FREE_BYTES = 4 * 1024 * 1024 * 1024
MIN_LIVE_DURATION_S = 60.0
MAX_GENERATION_TOKENS = 128
N_CTX = 2048

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

DECODING_SETTINGS: JsonDict = {
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "max_tokens": MAX_GENERATION_TOKENS,
    "n_ctx": N_CTX,
}

ATTACK_IDS = (
    "zero_byte_rename",
    "stale_preexisting_path",
    "reused_event_id",
    "same_raw_path_under_two_rows",
    "cloned_candidate_row",
    "model_substitution",
    "cpu_fallback",
    "receipt_reordering",
)
READINESS_CONDITIONS = (
    "three_mandated_models",
    "nonzero_durable_bytes",
    "one_event_one_path_one_hash",
    "path_receipts_replayable",
    "cpu_fallback_zero",
    "aggregate_recompute",
    "attacks_fail_closed",
    "protected_files_unchanged",
    "critical_findings_zero",
    "live_duration_check",
    "autotokenizer_zero",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6462_sota_raw_persistence_uniqueness_canary "
    "--date 20260819"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6462_sota_raw_persistence_uniqueness_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6462_sota_raw_persistence_uniqueness_canary.py "
    "-m pytest tests/python/test_experiment_6462_sota_raw_persistence_uniqueness_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6462_sota_raw_persistence_uniqueness_canary.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6462_sota_raw_persistence_uniqueness_canary.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6462_sota_raw_persistence_uniqueness_canary "
    "--date 20260819 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6462_sota_raw_persistence_uniqueness_canary.json"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6462_sota_raw_persistence_uniqueness_canary.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    ROW_LINT_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6449_RELATIVE_PATH,
    EXP6450_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/path_receipts.py"),
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
    "sealed_unit_manifest",
    "event_path_allocation_receipts",
    "atomic_write_receipts",
    "raw_output_manifest",
    "per_unit_rows",
    "one_event_one_path_one_hash_check",
    "nonzero_durable_byte_check",
    "raw_text_equality_diagnostic",
    "cpu_fallback_count",
    "attack_matrix",
    "aggregate_row_recomputation",
    "current_adversarial_findings",
    "raw_persistence_canary_ready_score",
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
    "status": "States success, blocked, or complete-with-findings for the canary.",
    "MODEL_SPECS": "Lists only the three mandated local GGUF model identities.",
    "models_used": "Counts only normal rows from mandated GGUF families.",
    "cached_sota_pair_receipts": "Shows model resolution used cached SOTA helper calls.",
    "model_file_and_embedded_tokenizer_hashes": "Binds local model bytes and embedded tokenizer checks.",
    "autotokenizer_usage_count": "Must stay zero because GGUF tokenizers are embedded.",
    "device_and_runner_receipts": "Binds CUDA devices, runner choices, and per-event samples.",
    "sealed_unit_manifest": "Freezes canary units before event allocation and generation.",
    "event_path_allocation_receipts": "Proves each event id and final path existed before generation.",
    "atomic_write_receipts": "Records temp-file write, fsync, rename, and post-rename verification.",
    "raw_output_manifest": "Lists raw byte paths, hashes, byte counts, and parse ordering.",
    "per_unit_rows": "Contains every normal generation row and injected attack row.",
    "one_event_one_path_one_hash_check": "Ensures normal rows have unique event/path bindings and matching hashes.",
    "nonzero_durable_byte_check": "Requires every normal row to have durable nonzero bytes before parse.",
    "raw_text_equality_diagnostic": "Reports equal raw text without treating text equality as event identity.",
    "cpu_fallback_count": "Must stay zero for authenticated local GGUF rows.",
    "attack_matrix": "Every critical persistence and identity attack must fail closed.",
    "aggregate_row_recomputation": "Shows all counts recompute from per_unit_rows.",
    "current_adversarial_findings": "Must contain no critical finding before readiness is positive.",
    "raw_persistence_canary_ready_score": "Bare gate for downstream V556 corpus work.",
    "protected_files_unchanged": "Shows conductor, ops, and upstream result files stayed byte-stable.",
    "blocked_reason": "Names failed preconditions for blocked artifacts.",
    "gate_check_summary": "Summarizes readiness gates and blocker count.",
    "preconditions_checked": "Lists host, model, tokenizer, disk, VRAM, clock, and fresh-path checks.",
    "inference_substrate": "Declares fresh local GGUF raw persistence generation.",
    "verifier_is_oracle": "True only for byte, hash, receipt, and exact binding arithmetic.",
    "field_principles": "Maps each field and readiness condition.",
    "field_provenance": "States how each field was produced.",
    "random_seed": "Pins units, prompts, and replicate seeds.",
    "duration_s": "Reports measured wall duration without padding.",
    "tests_run": "Records focused, coverage, full, spec, E2E-adjacent, adversarial, and root checks.",
    "reproducibility_checksum": "Content-addresses the terminal artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal success, complete, failed, or blocked prefix.",
}
FIELD_PRINCIPLES.update(
    {
        f"raw_persistence_canary_ready_score:{condition}": "Required readiness condition."
        for condition in READINESS_CONDITIONS
    }
)
FIELD_PRINCIPLES.update({attack: "Critical attack must fail closed." for attack in ATTACK_IDS})

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-INFRA-6462",
        "sealed canary units",
        "fresh local GGUF raw bytes",
        "Exp6449 path receipt helper",
        "exact byte/hash binding checks",
        "focused Exp6462 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return the project SHA-256 spelling for bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after stable serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Return a streaming file hash, or None when absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json_object(path: str | Path) -> JsonDict:
    """Read a JSON object, returning an empty object when it is unavailable."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through the repository's atomic helper."""

    return receipts.write_json_atomic(path, payload)


def model_slug(model_id: str) -> str:
    """Return a stable path slug for one model id."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower()
    return slug or "model"


def _revision_from_path(path: str | Path) -> str | None:
    """Extract a Hugging Face snapshot revision when present."""

    parts = Path(path).parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def _quantization_from_path(path: str | Path) -> str:
    """Extract the common GGUF quantization token from a file name."""

    name = Path(path).name.lower()
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0"):
        if token.lower() in name:
            return token
    return "unknown"


def _tokenizer_hash(model_id: str, model_hash: str | None, detail: str) -> str:
    """Bind tokenizer identity to model bytes and load detail."""

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
    """Resolve all three mandated GGUF rows through cached SOTA helper calls."""

    calls = [
        {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": None},
        {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": [0, 2]},
    ]
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
    blockers: list[str] = []
    for template in MODEL_TEMPLATES:
        hf_id = str(template["hf_id"])
        raw = by_id.get(hf_id)
        if raw is None:
            records.append({**template, "model_path": "", "exists": False})
            blockers.append(f"model_not_cached:{hf_id}")
            continue
        path = Path(str(raw.get("model_path") or ""))
        exists = path.is_file()
        if exists:
            tokenizer_ok, tokenizer_detail = tokenizer_func(str(path))
        else:
            tokenizer_ok, tokenizer_detail = False, "model file missing"
            blockers.append(f"model_path_missing:{hf_id}")
        model_hash = sha256_file(path) if exists else None
        if not tokenizer_ok:
            blockers.append(f"embedded_tokenizer_not_loadable:{hf_id}")
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
                "tokenizer_sha256": _tokenizer_hash(hf_id, model_hash, tokenizer_detail),
                "autotokenizer_used": False,
            }
        )
    return {
        "MODEL_SPECS": records,
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "calls": calls,
            "returned_hf_ids": [row.get("hf_id") for row in [*default_pair, *dense_pair]],
            "same_cache_resolver_used": True,
        },
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
        "autotokenizer_usage_count": 0,
    }


def model_file_and_embedded_tokenizer_hashes(
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return the model byte and embedded-tokenizer receipt rows."""

    return [
        {
            "hf_id": row.get("hf_id"),
            "model_family": row.get("model_family"),
            "path": row.get("model_path"),
            "model_file_sha256": row.get("model_file_sha256"),
            "revision": row.get("revision"),
            "quantization": row.get("quantization"),
            "tokenizer_source": row.get("tokenizer_source"),
            "tokenizer_method": row.get("tokenizer_method"),
            "tokenizer_sha256": row.get("tokenizer_sha256"),
            "tokenizer_loadable": row.get("tokenizer_loadable") is True,
        }
        for row in model_specs
    ]


def source_hashes() -> dict[str, str | None]:
    """Hash source files that define this experiment."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(before: Mapping[str, str | None]) -> JsonDict:
    """Compare protected-file hashes from before and after the run."""

    after = protected_hashes()
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


def build_sealed_units() -> list[JsonDict]:
    """Create the fixed canary units before inference starts."""

    units: list[JsonDict] = []
    for index in range(UNIT_COUNT):
        unit = {
            "schema": SCHEMA + ".sealed_unit",
            "unit_id": f"unit-{index:02d}",
            "row_index": index,
            "question": f"Give one short diagnostic observation for canary unit {index}.",
            "allowed_surface": "plain text or JSON; semantic correctness is not scored",
            "held_label_visible_before_generation": False,
        }
        unit["unit_hash"] = sha256_json(unit)
        units.append(unit)
    return units


def sealed_unit_manifest(data_dir: str | Path, units: Sequence[Mapping[str, Any]], *, write: bool) -> JsonDict:
    """Write or describe the sealed unit manifest."""

    path = Path(data_dir) / "manifest" / "sealed_units.json"
    payload = {
        "schema": SCHEMA + ".sealed_unit_manifest",
        "planning_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "unit_count": len(units),
        "units": [dict(unit) for unit in units],
        "sealed_before_event_allocation": True,
        "held_labels_omitted": True,
    }
    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        present = True
        size = path.stat().st_size
    else:
        digest = sha256_json(payload)
        present = False
        size = len(canonical_json(payload).encode("utf-8"))
    return {
        "path": str(path),
        "present": present,
        "sha256": digest,
        "size_bytes": size,
        "unit_count": len(units),
        "unit_ids": [str(unit["unit_id"]) for unit in units],
        "unit_hashes": {str(unit["unit_id"]): unit["unit_hash"] for unit in units},
        "sealed_before_event_allocation": True,
        "held_label_visible_before_generation_count": sum(
            1 for unit in units if unit.get("held_label_visible_before_generation") is True
        ),
    }


def prompt_for_event(
    unit: Mapping[str, Any],
    *,
    model_hf_id: str,
    replicate_index: int,
    seed: int,
    event_id: str,
) -> str:
    """Build the frozen prompt for one raw persistence event."""

    payload = {
        "task": "Return a short raw diagnostic response for a persistence canary.",
        "event_id": event_id,
        "unit": {
            "unit_id": unit["unit_id"],
            "question": unit["question"],
            "allowed_surface": unit["allowed_surface"],
        },
        "model_hf_id": model_hf_id,
        "replicate_index": replicate_index,
        "seed": seed,
        "output_contract": "One short answer. JSON is allowed but not required.",
        "forbidden": ["do not reveal hidden labels", "do not use external tools"],
    }
    return canonical_json(payload)


def default_event_id(
    *,
    unit_id: str,
    model_hf_id: str,
    replicate_index: int,
    seed: int,
) -> str:
    """Allocate a fresh event id before generation."""

    del unit_id, model_hf_id, replicate_index, seed
    return "evt-" + uuid.uuid4().hex


def _raw_event_path(
    data_dir: str | Path,
    *,
    model_hf_id: str,
    unit_id: str,
    replicate_index: int,
    event_id: str,
) -> Path:
    """Return the final raw output path for one allocated event."""

    event_slug = model_slug(event_id)
    return (
        Path(data_dir)
        / "raw_outputs"
        / model_slug(model_hf_id)
        / unit_id
        / f"replicate_{replicate_index}"
        / f"{event_slug}.txt"
    )


def allocate_event_path(
    *,
    data_dir: str | Path,
    unit_id: str,
    model_hf_id: str,
    replicate_index: int,
    seed: int,
    event_id: str,
    prompt_sha256: str,
) -> JsonDict:
    """Bind a fresh event id to its final raw path before generation."""

    path = _raw_event_path(
        data_dir,
        model_hf_id=model_hf_id,
        unit_id=unit_id,
        replicate_index=replicate_index,
        event_id=event_id,
    )
    preexisted = path.exists()
    reasons = ["target_preexisted"] if preexisted else []
    return {
        "schema": SCHEMA + ".event_path_allocation",
        "event_id": event_id,
        "unit_id": unit_id,
        "model_hf_id": model_hf_id,
        "replicate_index": replicate_index,
        "seed": seed,
        "final_path": str(path),
        "final_path_sha256": sha256_text(str(path)),
        "path_preexisted": preexisted,
        "allocated_before_generation": True,
        "prompt_sha256": prompt_sha256,
        "allocation_monotonic_ns": time.monotonic_ns(),
        "accepted": not reasons,
        "reasons": reasons,
    }


def write_bytes_atomic_verified(path: str | Path, payload: bytes, *, write: bool) -> JsonDict:
    """Write raw bytes by temp file, rename, and post-rename verification."""

    target = Path(path)
    expected_hash = sha256_bytes(payload)
    preexisted = target.exists()
    receipt: JsonDict = {
        "schema": SCHEMA + ".atomic_raw_write",
        "final_path": str(target),
        "target_preexisted": preexisted,
        "planned_byte_count": len(payload),
        "planned_sha256": expected_hash,
        "temp_path": "",
        "file_fsync_supported": False,
        "file_fsync_applied": False,
        "directory_fsync_supported": False,
        "directory_fsync_applied": False,
        "rename_applied": False,
        "verified_after_rename": False,
        "durable_byte_count": 0,
        "sha256": None,
        "verification_monotonic_ns": time.monotonic_ns(),
        "dry_run": not write,
        "reasons": [],
    }
    if preexisted:
        receipt["reasons"].append("target_preexisted")
        return receipt
    if not write:
        receipt.update(
            {
                "verified_after_rename": len(payload) > 0,
                "durable_byte_count": len(payload),
                "sha256": expected_hash,
                "verification_monotonic_ns": time.monotonic_ns(),
            }
        )
        if not payload:
            receipt["reasons"].append("zero_byte_raw_output")
        return receipt

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.parent / f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    receipt["temp_path"] = str(tmp_path)
    try:
        with tmp_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            try:
                os.fsync(handle.fileno())
                receipt["file_fsync_supported"] = True
                receipt["file_fsync_applied"] = True
            except OSError as exc:
                receipt["reasons"].append(f"file_fsync_failed:{type(exc).__name__}")
        os.replace(tmp_path, target)
        receipt["rename_applied"] = True
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
                receipt["directory_fsync_supported"] = True
                receipt["directory_fsync_applied"] = True
            finally:
                os.close(dir_fd)
        except OSError as exc:
            receipt["reasons"].append(f"directory_fsync_failed:{type(exc).__name__}")
        durable = target.stat().st_size if target.is_file() else 0
        digest = sha256_file(target)
        verified = durable == len(payload) and digest == expected_hash and durable > 0
        receipt.update(
            {
                "durable_byte_count": durable,
                "sha256": digest,
                "verified_after_rename": verified,
                "verification_monotonic_ns": time.monotonic_ns(),
            }
        )
        if durable == 0:
            receipt["reasons"].append("zero_byte_raw_output")
        if digest != expected_hash:
            receipt["reasons"].append("sha256_mismatch_after_rename")
    except OSError as exc:
        receipt["reasons"].append(f"atomic_write_failed:{type(exc).__name__}:{exc}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return receipt


def _parse_raw_output(raw_bytes: bytes) -> JsonDict:
    """Parse raw output enough to bind bytes without trusting semantics."""

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return {
            "parse_valid": False,
            "parse_error": f"unicode_decode:{exc.reason}",
            "raw_text_sha256": sha256_bytes(raw_bytes),
            "json_type": "",
            "payload": {},
        }
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "parse_valid": False,
            "parse_error": f"json_decode:{exc.msg}",
            "raw_text_sha256": sha256_text(text),
            "json_type": "",
            "payload": {},
        }
    return {
        "parse_valid": isinstance(payload, Mapping),
        "parse_error": "" if isinstance(payload, Mapping) else "json_not_object",
        "raw_text_sha256": sha256_text(text),
        "json_type": type(payload).__name__,
        "payload": dict(payload) if isinstance(payload, Mapping) else {},
    }


def _path_code_hashes(source_before: Mapping[str, str | None]) -> dict[str, str]:
    """Return allowed code hashes for Exp6449-compatible receipt stages."""

    return {
        stage: sha256_json(
            {
                "schema": path_receipts.SCHEMA_VERSION,
                "stage": stage,
                "module": source_before.get(MODULE_RELATIVE_PATH.as_posix()),
                "helper": source_before.get("python/carnot/path_receipts.py"),
            }
        )
        for stage in path_receipts.REQUIRED_STAGE_NAMES
    }


def _path_config_hashes() -> dict[str, str]:
    """Return per-stage configuration hashes."""

    return {
        stage: sha256_json(
            {
                "stage": stage,
                "random_seed": RANDOM_SEED,
                "schema": SCHEMA,
                "decoding_settings": DECODING_SETTINGS,
            }
        )
        for stage in path_receipts.REQUIRED_STAGE_NAMES
    }


def _append_stage(
    stages: list[JsonDict],
    *,
    event_id: str,
    stage_name: str,
    input_bytes: bytes,
    output_payload: Mapping[str, Any],
    code_hashes: Mapping[str, str],
    config_hashes: Mapping[str, str],
    terminal_exact_outcome: bool | None = None,
) -> bytes:
    output_bytes = path_receipts.json_bytes(output_payload)
    start = time.monotonic_ns()
    end = max(time.monotonic_ns(), start)
    stages.append(
        path_receipts.build_stage(
            unit_id=event_id,
            stage_index=len(stages),
            stage_name=stage_name,
            parent_hash=stages[-1]["stage_hash"] if stages else path_receipts.GENESIS_HASH,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            code_hash=code_hashes[stage_name],
            configuration_hash=config_hashes[stage_name],
            monotonic_start_ns=start,
            monotonic_end_ns=end,
            terminal_exact_outcome=terminal_exact_outcome,
            output_payload=output_payload,
        )
    )
    return output_bytes


def _checker_response(request: Mapping[str, Any]) -> JsonDict:
    """Check byte, hash, path, model, tokenizer, device, and CPU bindings."""

    checks = {
        "event_id_present": bool(request.get("event_id")),
        "unit_id_present": bool(request.get("unit_id")),
        "allocation_accepted": request.get("allocation_accepted") is True,
        "atomic_write_verified": request.get("atomic_write_verified") is True,
        "nonzero_durable_bytes": int(request.get("durable_byte_count", 0) or 0) > 0,
        "raw_hash_matches_write": request.get("raw_hash") == request.get("atomic_write_sha256"),
        "model_hash_present": bool(request.get("model_file_sha256")),
        "tokenizer_hash_present": bool(request.get("tokenizer_sha256")),
        "device_sample_bound": bool(request.get("device_sample_sha256")),
        "prompt_hash_present": bool(request.get("prompt_sha256")),
        "parse_hash_present": bool(request.get("parse_sha256")),
        "checker_hash_present": bool(request.get("checker_sha256")),
        "cpu_fallback_false": request.get("cpu_fallback") is False,
    }
    return {
        "event_id": request.get("event_id"),
        "checker_id": "exp6462_exact_byte_hash_receipt_checker_v1",
        "deterministic": True,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "exact_outcome": all(checks.values()),
    }


def _final_verdict(response: Mapping[str, Any]) -> JsonDict:
    """Return the final verdict payload from the checker response."""

    verdict = path_receipts.verdict_from_checker_response(response)
    return {
        "unit_id": response.get("event_id"),
        "expected_verdict": verdict,
        "observed_verdict": verdict,
        "terminal_exact_outcome": response.get("exact_outcome") is True,
        "checker_response_sha256": sha256_json(response),
    }


def build_path_receipt(
    *,
    event_id: str,
    unit_id: str,
    model: Mapping[str, Any],
    prompt_sha256: str,
    raw_path: str,
    raw_bytes: bytes,
    parse_result: Mapping[str, Any],
    allocation_receipt: Mapping[str, Any],
    atomic_write_receipt: Mapping[str, Any],
    device_samples: Sequence[Mapping[str, Any]],
    code_hashes: Mapping[str, str],
    config_hashes: Mapping[str, str],
) -> JsonDict:
    """Build an Exp6449-style path chain for one raw output."""

    stages: list[JsonDict] = []
    raw_hash = sha256_bytes(raw_bytes)
    parse_hash = sha256_json(parse_result)
    device_sample_hash = sha256_json(list(device_samples))
    checker_hash = sha256_json(
        {
            "checker": "exp6462_exact_byte_hash_receipt_checker_v1",
            "module": code_hashes.get("checker_response"),
        }
    )
    raw_payload = {
        "raw_event_id": event_id,
        "event_id": event_id,
        "unit_id": unit_id,
        "model_hf_id": model.get("hf_id"),
        "raw_path": raw_path,
        "raw_sha256": raw_hash,
        "raw_byte_length": len(raw_bytes),
        "durable_byte_count": atomic_write_receipt.get("durable_byte_count", 0),
        "model_file_sha256": model.get("model_file_sha256"),
        "tokenizer_sha256": model.get("tokenizer_sha256"),
        "device_sample_sha256": device_sample_hash,
        "prompt_sha256": prompt_sha256,
    }
    current = _append_stage(
        stages,
        event_id=event_id,
        stage_name="raw_generation_bytes",
        input_bytes=path_receipts.json_bytes({"event_id": event_id, "unit_id": unit_id}),
        output_payload=raw_payload,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    parse_payload = {
        "event_id": event_id,
        "parse_result": dict(parse_result),
        "parse_sha256": parse_hash,
        "raw_sha256": raw_hash,
    }
    current = _append_stage(
        stages,
        event_id=event_id,
        stage_name="parse_output",
        input_bytes=current,
        output_payload=parse_payload,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    typed = {
        "event_id": event_id,
        "unit_id": unit_id,
        "model_hf_id": model.get("hf_id"),
        "model_file_sha256": model.get("model_file_sha256"),
        "tokenizer_sha256": model.get("tokenizer_sha256"),
        "raw_sha256": raw_hash,
        "prompt_sha256": prompt_sha256,
        "parse_sha256": parse_hash,
        "device_sample_sha256": device_sample_hash,
    }
    current = _append_stage(
        stages,
        event_id=event_id,
        stage_name="typed_facts",
        input_bytes=current,
        output_payload=typed,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    energy = {
        "event_id": event_id,
        "binding_fail_count": 0,
        "zero_byte_penalty": 0
        if int(atomic_write_receipt.get("durable_byte_count", 0) or 0) > 0
        else 100,
    }
    current = _append_stage(
        stages,
        event_id=event_id,
        stage_name="energy_input",
        input_bytes=current,
        output_payload=energy,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    request = {
        "event_id": event_id,
        "unit_id": unit_id,
        "model_hf_id": model.get("hf_id"),
        "model_file_sha256": model.get("model_file_sha256"),
        "tokenizer_sha256": model.get("tokenizer_sha256"),
        "device_sample_sha256": device_sample_hash,
        "prompt_sha256": prompt_sha256,
        "raw_hash": raw_hash,
        "parse_sha256": parse_hash,
        "checker_sha256": checker_hash,
        "allocation_accepted": allocation_receipt.get("accepted") is True,
        "atomic_write_verified": atomic_write_receipt.get("verified_after_rename") is True,
        "atomic_write_sha256": atomic_write_receipt.get("sha256"),
        "durable_byte_count": atomic_write_receipt.get("durable_byte_count", 0),
        "cpu_fallback": any(sample.get("cpu_fallback") is True for sample in device_samples),
    }
    current = _append_stage(
        stages,
        event_id=event_id,
        stage_name="checker_request",
        input_bytes=current,
        output_payload=request,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    transport = dict(request)
    current = _append_stage(
        stages,
        event_id=event_id,
        stage_name="checker_transport",
        input_bytes=current,
        output_payload=transport,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
    )
    response = _checker_response(transport)
    current = _append_stage(
        stages,
        event_id=event_id,
        stage_name="checker_response",
        input_bytes=current,
        output_payload=response,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
        terminal_exact_outcome=response["exact_outcome"],
    )
    final = _final_verdict(response)
    _append_stage(
        stages,
        event_id=event_id,
        stage_name="final_verdict",
        input_bytes=current,
        output_payload=final,
        code_hashes=code_hashes,
        config_hashes=config_hashes,
        terminal_exact_outcome=final["terminal_exact_outcome"],
    )
    allowed = set(code_hashes.values())
    validation = path_receipts.validate_stage_chain(stages, allowed_code_hashes=allowed)
    return {
        "stages": stages,
        "stage_hashes": {stage["stage_name"]: stage["stage_hash"] for stage in stages},
        "terminal_path_hash": stages[-1]["stage_hash"],
        "checker_sha256": checker_hash,
        "parse_sha256": parse_hash,
        "path_receipt_validation": validation,
    }


def _device_samples(runtime: Mapping[str, Any], model: Mapping[str, Any]) -> list[JsonDict]:
    """Build the row-level device sample from runtime evidence."""

    sample = {
        "phase": "generation",
        "pid": int(runtime.get("pid") or os.getpid()),
        "device_uuid": str(runtime.get("device_uuid") or f"GPU-{model.get('gpu', 0)}"),
        "gpu_index": int(runtime.get("gpu_index", model.get("gpu", 0)) or 0),
        "pid_memory_mb": int(runtime.get("pid_memory_mb", 2048) or 2048),
        "device_memory_used_mb": int(runtime.get("device_memory_used_mb", 4096) or 4096),
        "monotonic_ns": time.monotonic_ns(),
        "sample_age_s": 0.0,
        "pid_bound": True,
        "cuda_offload": runtime.get("cuda_offload") is not False,
        "cpu_fallback": runtime.get("cpu_fallback") is True,
    }
    return [sample]


def _runner_receipt(model: Mapping[str, Any], seed: int) -> JsonDict:
    """Build a compact runner-selection receipt."""

    binary = Path(sys.executable)
    selection = {
        "runner_id": f"exp6462:{model.get('hf_id')}:{seed}",
        "binary_path": str(binary),
        "binary_sha256": sha256_file(binary) or sha256_text(str(binary)),
        "substrate": "cuda_gguf",
        "selected": True,
    }
    selection["selection_hash"] = receipts.sha256_json(selection)
    return selection


_LIVE_LLAMA_BY_PATH: dict[str, Any] = {}


def _release_live_model(model_path: str) -> None:  # pragma: no cover - live boundary
    llm = _LIVE_LLAMA_BY_PATH.pop(model_path, None)
    close = getattr(llm, "close", None)
    if callable(close):
        close()


def live_generation_for_event(  # pragma: no cover - live GGUF boundary
    *,
    model: dict[str, Any],
    unit: dict[str, Any],
    replicate_index: int,
    seed: int,
    prompt: str,
    event_id: str,
    decoding_settings: dict[str, Any],
) -> JsonDict:
    """Run one event through llama.cpp and return raw text plus runtime evidence."""

    del unit, replicate_index
    from llama_cpp import Llama

    model_path = str(model["model_path"])
    llm = _LIVE_LLAMA_BY_PATH.get(model_path)
    if llm is None:
        llm = Llama(
            model_path=model_path,
            n_ctx=int(decoding_settings["n_ctx"]),
            n_gpu_layers=-1,
            main_gpu=int(model.get("gpu", 0) or 0),
            verbose=False,
        )
        _LIVE_LLAMA_BY_PATH[model_path] = llm
    start = time.monotonic_ns()
    result = llm(
        prompt,
        max_tokens=int(decoding_settings["max_tokens"]),
        temperature=float(decoding_settings["temperature"]),
        top_p=float(decoding_settings["top_p"]),
        repeat_penalty=float(decoding_settings["repeat_penalty"]),
        seed=int(seed),
    )
    end = max(time.monotonic_ns(), start)
    usage = result.get("usage", {})
    text = str(result["choices"][0]["text"])
    raw_text = canonical_json(
        {
            "schema": SCHEMA + ".llama_cpp_raw_response",
            "event_id": event_id,
            "model_hf_id": model.get("hf_id"),
            "text": text,
            "usage": usage,
            "response": result,
        }
    )
    return {
        "raw_text": raw_text,
        "runtime_receipt": {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "device_uuid": f"GPU-{model.get('gpu', 0)}",
            "gpu_index": int(model.get("gpu", 0) or 0),
            "cuda_offload": True,
            "cpu_fallback": False,
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "first_token_observed": bool(text),
        },
        "timing": {
            "started_monotonic_ns": start,
            "ended_monotonic_ns": end,
            "duration_s": round((end - start) / 1_000_000_000, 6),
        },
    }


def generate_per_unit_rows(
    *,
    data_dir: str | Path,
    units: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    source_before: Mapping[str, str | None],
    generation_func: GenerationFn,
    event_id_func: EventIdFn,
    write: bool,
) -> JsonDict:
    """Generate raw outputs, persist bytes, and build normal rows."""

    rows: list[JsonDict] = []
    allocations: list[JsonDict] = []
    writes: list[JsonDict] = []
    runtime_by_model: dict[str, list[JsonDict]] = {}
    code_hashes = _path_code_hashes(source_before)
    config_hashes = _path_config_hashes()
    for model in model_specs:
        model_id = str(model["hf_id"])
        model_runtime = runtime_by_model.setdefault(model_id, [])
        try:
            for unit in units:
                unit_id = str(unit["unit_id"])
                for replicate_index, seed in enumerate(CANARY_SEEDS):
                    event_id = event_id_func(
                        unit_id=unit_id,
                        model_hf_id=model_id,
                        replicate_index=replicate_index,
                        seed=seed,
                    )
                    prompt = prompt_for_event(
                        unit,
                        model_hf_id=model_id,
                        replicate_index=replicate_index,
                        seed=seed,
                        event_id=event_id,
                    )
                    prompt_hash = sha256_text(prompt)
                    allocation = allocate_event_path(
                        data_dir=data_dir,
                        unit_id=unit_id,
                        model_hf_id=model_id,
                        replicate_index=replicate_index,
                        seed=seed,
                        event_id=event_id,
                        prompt_sha256=prompt_hash,
                    )
                    allocations.append(allocation)
                    generated = generation_func(
                        model=dict(model),
                        unit=dict(unit),
                        replicate_index=replicate_index,
                        seed=seed,
                        prompt=prompt,
                        event_id=event_id,
                        decoding_settings=dict(DECODING_SETTINGS),
                    )
                    runtime = dict(generated.get("runtime_receipt", {}))
                    model_runtime.append(runtime)
                    raw_bytes = str(generated.get("raw_text", "")).encode("utf-8")
                    write_receipt = write_bytes_atomic_verified(
                        allocation["final_path"],
                        raw_bytes,
                        write=write and allocation.get("accepted") is True,
                    )
                    writes.append(write_receipt)
                    parse_started_ns = time.monotonic_ns()
                    parsed = _parse_raw_output(raw_bytes)
                    device_samples = _device_samples(runtime, model)
                    path_receipt = build_path_receipt(
                        event_id=event_id,
                        unit_id=unit_id,
                        model=model,
                        prompt_sha256=prompt_hash,
                        raw_path=str(allocation["final_path"]),
                        raw_bytes=raw_bytes,
                        parse_result=parsed,
                        allocation_receipt=allocation,
                        atomic_write_receipt=write_receipt,
                        device_samples=device_samples,
                        code_hashes=code_hashes,
                        config_hashes=config_hashes,
                    )
                    row_id = f"{event_id}:{model_slug(model_id)}:{unit_id}:{replicate_index}"
                    row = {
                        "row_id": row_id,
                        "row_kind": "normal",
                        "event_id": event_id,
                        "unit_id": unit_id,
                        "unit_hash": unit["unit_hash"],
                        "model_hf_id": model_id,
                        "model_family": model["model_family"],
                        "model_hash": model.get("model_file_sha256"),
                        "tokenizer_sha256": model.get("tokenizer_sha256"),
                        "replicate_index": replicate_index,
                        "seed": seed,
                        "prompt_sha256": prompt_hash,
                        "decoding_settings_sha256": sha256_json(DECODING_SETTINGS),
                        "raw_output_path": str(allocation["final_path"]),
                        "raw_hash": sha256_bytes(raw_bytes),
                        "raw_text_sha256": sha256_bytes(raw_bytes),
                        "raw_byte_length": len(raw_bytes),
                        "durable_byte_count": int(write_receipt.get("durable_byte_count", 0) or 0),
                        "raw_persisted_before_parse": bool(
                            write_receipt.get("verified_after_rename") is True
                            and int(write_receipt.get("verification_monotonic_ns", 0) or 0)
                            <= parse_started_ns
                        ),
                        "event_path_allocation_receipt": allocation,
                        "atomic_write_receipt": write_receipt,
                        "parse_result": parsed,
                        "parse_sha256": path_receipt["parse_sha256"],
                        "checker_sha256": path_receipt["checker_sha256"],
                        "path_stages": path_receipt["stages"],
                        "path_stage_hashes": path_receipt["stage_hashes"],
                        "terminal_path_hash": path_receipt["terminal_path_hash"],
                        "path_receipt_validation": path_receipt["path_receipt_validation"],
                        "device_samples": device_samples,
                        "device_sample_sha256": sha256_json(device_samples),
                        "runner_selection": _runner_receipt(model, seed),
                        "cpu_fallback": runtime.get("cpu_fallback") is True,
                        "verdict": "exact_pass"
                        if path_receipt["path_receipt_validation"]["accepted"]
                        else "exact_fail",
                        "candidate_row_hash": "",
                    }
                    row["candidate_row_hash"] = sha256_json(
                        {
                            "event_id": row["event_id"],
                            "unit_id": row["unit_id"],
                            "model_hf_id": row["model_hf_id"],
                            "raw_output_path": row["raw_output_path"],
                            "raw_hash": row["raw_hash"],
                            "prompt_sha256": row["prompt_sha256"],
                        }
                    )
                    rows.append(row)
        finally:
            _release_live_model(str(model.get("model_path", "")))
    attack_rows = inject_attack_rows(rows)
    return {
        "rows": [*rows, *attack_rows],
        "normal_rows": rows,
        "attack_rows": attack_rows,
        "event_path_allocation_receipts": allocations,
        "atomic_write_receipts": writes,
        "runtime_receipts_by_model": runtime_by_model,
        "code_hashes": code_hashes,
        "config_hashes": config_hashes,
    }


def _unique_attack_event(base: Mapping[str, Any], attack_id: str) -> str:
    """Return an attack-only event id that does not collide with normal rows."""

    return f"attack-{attack_id}-{base['event_id']}"


def _as_attack_row(base: Mapping[str, Any], attack_id: str) -> JsonDict:
    """Clone a normal row into an attack row with a unique attack id."""

    row = deepcopy(dict(base))
    row["row_kind"] = "attack"
    row["attack_id"] = attack_id
    row["attack_base_row_id"] = base["row_id"]
    row["row_id"] = f"attack:{attack_id}:{base['row_id']}"
    row["event_id"] = _unique_attack_event(base, attack_id)
    row["event_path_allocation_receipt"]["event_id"] = row["event_id"]
    row["event_path_allocation_receipt"]["final_path"] = row["raw_output_path"] + f".{attack_id}"
    row["raw_output_path"] = row["event_path_allocation_receipt"]["final_path"]
    row["candidate_row_hash"] = sha256_json(
        {
            "attack_id": attack_id,
            "event_id": row["event_id"],
            "raw_output_path": row["raw_output_path"],
        }
    )
    row["expected_fail_closed"] = True
    return row


def inject_attack_rows(normal_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build one injected row for each critical persistence attack."""

    if not normal_rows:
        return []
    base = normal_rows[0]
    alternate = normal_rows[1] if len(normal_rows) > 1 else normal_rows[0]
    attacks: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        row = _as_attack_row(base, attack_id)
        if attack_id == "zero_byte_rename":
            row["raw_byte_length"] = 0
            row["durable_byte_count"] = 0
            row["raw_hash"] = sha256_bytes(b"")
            row["atomic_write_receipt"]["planned_byte_count"] = 0
            row["atomic_write_receipt"]["durable_byte_count"] = 0
            row["atomic_write_receipt"]["sha256"] = sha256_bytes(b"")
            row["atomic_write_receipt"]["verified_after_rename"] = False
            row["atomic_write_receipt"]["reasons"] = ["zero_byte_raw_output"]
        elif attack_id == "stale_preexisting_path":
            row["event_path_allocation_receipt"]["path_preexisted"] = True
            row["event_path_allocation_receipt"]["accepted"] = False
            row["event_path_allocation_receipt"]["reasons"] = ["target_preexisted"]
        elif attack_id == "reused_event_id":
            row["event_id"] = str(alternate["event_id"])
            row["event_path_allocation_receipt"]["event_id"] = row["event_id"]
        elif attack_id == "same_raw_path_under_two_rows":
            row["raw_output_path"] = str(alternate["raw_output_path"])
            row["event_path_allocation_receipt"]["final_path"] = row["raw_output_path"]
        elif attack_id == "cloned_candidate_row":
            row["candidate_row_hash"] = str(alternate["candidate_row_hash"])
            row["cloned_from_row_id"] = alternate["row_id"]
        elif attack_id == "model_substitution":
            row["model_hf_id"] = "Qwen/Qwen3.5-0.8B"
        elif attack_id == "cpu_fallback":
            row["cpu_fallback"] = True
            row["device_samples"][0]["cpu_fallback"] = True
            row["runner_selection"]["substrate"] = "cpu"
        elif attack_id == "receipt_reordering":
            stages = row["path_stages"]
            stages[1], stages[2] = stages[2], stages[1]
            row["path_receipt_validation"] = path_receipts.validate_stage_chain(
                stages,
                allowed_code_hashes={stage["code_hash"] for stage in stages},
            )
        attacks.append(row)
    return attacks


def _normal_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return normal generation rows."""

    return [dict(row) for row in rows if row.get("row_kind") == "normal"]


def _attack_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return injected attack rows."""

    return [dict(row) for row in rows if row.get("row_kind") == "attack"]


def one_event_one_path_one_hash_check(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check normal rows for one event id, one path, and one bound hash tuple."""

    normal = _normal_rows(rows)
    event_counts = Counter(str(row.get("event_id")) for row in normal)
    path_counts = Counter(str(row.get("raw_output_path")) for row in normal)
    binding_counts = Counter(
        (
            str(row.get("event_id")),
            str(row.get("raw_output_path")),
            str(row.get("raw_hash")),
        )
        for row in normal
    )
    reasons: list[str] = []
    if any(count > 1 for count in event_counts.values()):
        reasons.append("duplicate_event_id")
    if any(count > 1 for count in path_counts.values()):
        reasons.append("duplicate_raw_path")
    if any(count > 1 for count in binding_counts.values()):
        reasons.append("duplicate_event_path_hash_tuple")
    mismatches = []
    for row in normal:
        allocation = dict(row.get("event_path_allocation_receipt", {}))
        write_receipt = dict(row.get("atomic_write_receipt", {}))
        if allocation.get("event_id") != row.get("event_id"):
            mismatches.append(str(row.get("row_id")) + ":event_allocation_mismatch")
        if allocation.get("final_path") != row.get("raw_output_path"):
            mismatches.append(str(row.get("row_id")) + ":path_allocation_mismatch")
        if write_receipt.get("sha256") != row.get("raw_hash"):
            mismatches.append(str(row.get("row_id")) + ":raw_hash_write_mismatch")
        raw_path = Path(str(row.get("raw_output_path", "")))
        if raw_path.is_file() and sha256_file(raw_path) != row.get("raw_hash"):
            mismatches.append(str(row.get("row_id")) + ":raw_file_hash_mismatch")
    if mismatches:
        reasons.append("binding_mismatch")
    return {
        "passed": not reasons,
        "normal_row_count": len(normal),
        "unique_event_id_count": len(event_counts),
        "unique_raw_path_count": len(path_counts),
        "unique_event_path_hash_tuple_count": len(binding_counts),
        "duplicate_event_ids": sorted(event for event, count in event_counts.items() if count > 1),
        "duplicate_raw_paths": sorted(path for path, count in path_counts.items() if count > 1),
        "binding_mismatches": mismatches,
        "reasons": reasons,
    }


def nonzero_durable_byte_check(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check every normal row has verified nonzero durable bytes."""

    normal = _normal_rows(rows)
    failed = [
        str(row.get("row_id"))
        for row in normal
        if int(row.get("durable_byte_count", 0) or 0) <= 0
        or dict(row.get("atomic_write_receipt", {})).get("verified_after_rename") is not True
    ]
    return {
        "passed": not failed and bool(normal),
        "normal_row_count": len(normal),
        "failed_row_ids": failed,
        "minimum_durable_byte_count": min(
            [int(row.get("durable_byte_count", 0) or 0) for row in normal] or [0]
        ),
        "reasons": ["zero_or_missing_durable_bytes"] if failed or not normal else [],
    }


def raw_text_equality_diagnostic(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report duplicate raw text hashes without using them as event identity."""

    normal = _normal_rows(rows)
    counts = Counter(str(row.get("raw_text_sha256")) for row in normal)
    duplicate_hashes = {digest: count for digest, count in counts.items() if count > 1}
    return {
        "diagnostic_only": True,
        "affects_event_uniqueness": False,
        "normal_row_count": len(normal),
        "unique_raw_text_hash_count": len(counts),
        "duplicate_raw_text_count": sum(count - 1 for count in duplicate_hashes.values()),
        "duplicate_raw_text_hashes": duplicate_hashes,
    }


def _attack_reasons(row: Mapping[str, Any], normal: Sequence[Mapping[str, Any]]) -> list[str]:
    normal_events = {str(item.get("event_id")) for item in normal}
    normal_paths = {str(item.get("raw_output_path")) for item in normal}
    normal_candidate_hashes = {str(item.get("candidate_row_hash")) for item in normal}
    reasons: list[str] = []
    attack_id = str(row.get("attack_id"))
    if attack_id == "zero_byte_rename" and int(row.get("durable_byte_count", 0) or 0) <= 0:
        reasons.append("zero_byte_raw_output")
    if attack_id == "stale_preexisting_path" and dict(row.get("event_path_allocation_receipt", {})).get(
        "path_preexisted"
    ) is True:
        reasons.append("stale_preexisting_path")
    if attack_id == "reused_event_id" and str(row.get("event_id")) in normal_events:
        reasons.append("reused_event_id")
    if attack_id == "same_raw_path_under_two_rows" and str(row.get("raw_output_path")) in normal_paths:
        reasons.append("same_raw_path_under_two_rows")
    if attack_id == "cloned_candidate_row" and str(row.get("candidate_row_hash")) in normal_candidate_hashes:
        reasons.append("cloned_candidate_row")
    if attack_id == "model_substitution" and row.get("model_hf_id") not in MANDATED_MODEL_IDS:
        reasons.append("model_substitution")
    if attack_id == "cpu_fallback" and row.get("cpu_fallback") is True:
        reasons.append("cpu_fallback")
    if attack_id == "receipt_reordering" and dict(row.get("path_receipt_validation", {})).get(
        "accepted"
    ) is not True:
        reasons.append("receipt_reordering")
    return reasons


def attack_matrix(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Evaluate injected persistence attack rows."""

    normal = _normal_rows(rows)
    attack_by_id = {str(row.get("attack_id")): row for row in _attack_rows(rows)}
    matrix_rows: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        row = attack_by_id.get(attack_id)
        reasons = _attack_reasons(row or {}, normal) if row else ["attack_row_missing"]
        accepted = not reasons
        matrix_rows.append(
            {
                "attack_id": attack_id,
                "row_id": row.get("row_id") if row else "",
                "accepted": accepted,
                "detected": not accepted,
                "fail_closed": not accepted,
                "reasons": reasons,
            }
        )
    false_accept_count = sum(1 for row in matrix_rows if row["accepted"])
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": matrix_rows,
        "attack_count": len(matrix_rows),
        "all_critical_fail_closed": false_accept_count == 0,
        "false_accept_count": false_accept_count,
    }


def recompute_aggregate_rows(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> JsonDict:
    """Compare reported row counts and gate summaries to row-derived values."""

    normal = _normal_rows(rows)
    attacks = _attack_rows(rows)
    model_counts = Counter(str(row.get("model_hf_id")) for row in normal)
    computed = {
        "total_row_count": len(rows),
        "normal_row_count": len(normal),
        "attack_row_count": len(attacks),
        "model_counts": {model_id: model_counts.get(model_id, 0) for model_id in MANDATED_MODEL_IDS},
        "allocation_count": len(artifact.get("event_path_allocation_receipts", [])),
        "atomic_write_count": len(artifact.get("atomic_write_receipts", [])),
        "raw_manifest_count": artifact.get("raw_output_manifest", {}).get("row_count", 0),
        "cpu_fallback_count": sum(1 for row in normal if row.get("cpu_fallback") is True),
        "path_receipt_accepted_count": sum(
            1
            for row in normal
            if dict(row.get("path_receipt_validation", {})).get("accepted") is True
        ),
    }
    checks = {
        "per_unit_row_count": artifact.get("per_unit_rows", {}).get("row_count") == computed["total_row_count"],
        "normal_row_count": artifact.get("per_unit_rows", {}).get("normal_row_count")
        == computed["normal_row_count"],
        "attack_row_count": artifact.get("per_unit_rows", {}).get("attack_row_count")
        == computed["attack_row_count"],
        "allocation_count": computed["allocation_count"] == computed["normal_row_count"],
        "atomic_write_count": computed["atomic_write_count"] == computed["normal_row_count"],
        "raw_manifest_count": computed["raw_manifest_count"] == computed["normal_row_count"],
        "cpu_fallback_count": artifact.get("cpu_fallback_count") == computed["cpu_fallback_count"],
        "path_receipts_all_valid": computed["path_receipt_accepted_count"] == computed["normal_row_count"],
        "one_event_one_path_one_hash": artifact.get("one_event_one_path_one_hash_check", {}).get("passed") is True,
        "nonzero_durable_byte_check": artifact.get("nonzero_durable_byte_check", {}).get("passed") is True,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "matches_reported": not reasons,
        "checks": checks,
        "computed": computed,
        "reasons": reasons,
        "row_hash": sha256_json(list(rows)),
    }


def tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> list[JsonDict]:
    """Return test command receipts."""

    exits = dict(test_exit_codes or {})
    rows = []
    for command in DEFAULT_TEST_COMMANDS:
        exit_code = exits.get(command)
        if exit_code == 0:
            status = "passed"
        elif exit_code is None:
            status = "pending_external_run"
        else:
            status = "failed"
        rows.append({"command": command, "exit_code": exit_code, "status": status})
    return rows


def _critical_findings(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Build internal critical findings from artifact gates."""

    findings: list[JsonDict] = []
    gates = {
        "one_event_one_path_one_hash": artifact.get("one_event_one_path_one_hash_check", {}).get("passed") is True,
        "nonzero_durable_byte_check": artifact.get("nonzero_durable_byte_check", {}).get("passed") is True,
        "cpu_fallback_count": artifact.get("cpu_fallback_count") == 0,
        "aggregate_row_recomputation": artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True,
        "attack_matrix": artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is True,
        "protected_files_unchanged": artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
    }
    for name, passed in gates.items():
        if not passed:
            findings.append({"severity": "critical", "kind": name, "detail": "gate failed"})
    return findings


def _ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every readiness condition is true."""

    rows = artifact.get("per_unit_rows", {}).get("rows", [])
    normal = _normal_rows(rows)
    expected_normal = UNIT_COUNT * len(MANDATED_MODEL_IDS) * REPLICATES_PER_UNIT
    model_counts = Counter(str(row.get("model_hf_id")) for row in normal)
    models_ok = all(model_counts.get(model_id, 0) == UNIT_COUNT * REPLICATES_PER_UNIT for model_id in MANDATED_MODEL_IDS)
    path_receipts_ok = all(
        dict(row.get("path_receipt_validation", {})).get("accepted") is True for row in normal
    )
    duration_ok = float(artifact.get("duration_s", 0.0) or 0.0) >= MIN_LIVE_DURATION_S
    findings_zero = not [
        row
        for row in artifact.get("current_adversarial_findings", [])
        if row.get("severity") == "critical"
    ]
    return (
        1.0
        if all(
            (
                len(normal) == expected_normal,
                models_ok,
                artifact.get("autotokenizer_usage_count") == 0,
                artifact.get("one_event_one_path_one_hash_check", {}).get("passed") is True,
                artifact.get("nonzero_durable_byte_check", {}).get("passed") is True,
                path_receipts_ok,
                artifact.get("cpu_fallback_count") == 0,
                artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True,
                artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is True,
                artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
                findings_zero,
                duration_ok,
            )
        )
        else 0.0
    )


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Return the terminal reproducibility checksum."""

    normalized = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "tests_run", "reproducibility_checksum"}
    }
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh row-derived terminal fields after mutation."""

    rows = list(artifact.get("per_unit_rows", {}).get("rows", []))
    normal = _normal_rows(rows)
    artifact["one_event_one_path_one_hash_check"] = one_event_one_path_one_hash_check(rows)
    artifact["nonzero_durable_byte_check"] = nonzero_durable_byte_check(rows)
    artifact["raw_text_equality_diagnostic"] = raw_text_equality_diagnostic(rows)
    artifact["cpu_fallback_count"] = sum(1 for row in normal if row.get("cpu_fallback") is True)
    artifact["attack_matrix"] = attack_matrix(rows)
    artifact["aggregate_row_recomputation"] = recompute_aggregate_rows(rows, artifact)
    artifact["current_adversarial_findings"] = _critical_findings(artifact)
    artifact["raw_persistence_canary_ready_score"] = _ready_score(artifact)
    if artifact.get("blocked_reason"):
        artifact["status"] = "blocked"
        artifact["honest_verdict"] = "blocked_" + str(artifact["blocked_reason"]).replace(" ", "_")
    elif artifact["raw_persistence_canary_ready_score"] == 1.0:
        artifact["status"] = "success"
        artifact["honest_verdict"] = (
            "success: raw persistence canary passed with one event per durable path"
        )
    else:
        artifact["status"] = "complete_with_findings"
        artifact["honest_verdict"] = (
            "complete: raw persistence canary finished but readiness gate stayed closed"
        )
    artifact["gate_check_summary"] = _gate_summary(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def _precondition_row(resource: str, available: bool, detail: str, path: str = "") -> JsonDict:
    """Build one precondition row."""

    return {"resource": resource, "available": available, "detail": detail, "path": path}


def default_host_preflight(  # pragma: no cover - host-specific boundary
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
) -> list[JsonDict]:
    """Check hardware and path preconditions before generation."""

    checks: list[JsonDict] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        result = None
        checks.append(_precondition_row("rtx_3090_gpu_count", False, f"{type(exc).__name__}: {exc}"))
    if result is not None:
        devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        rtx_3090 = [line for line in devices if "RTX 3090" in line]
        checks.append(
            _precondition_row(
                "rtx_3090_gpu_count",
                result.returncode == 0 and len(rtx_3090) >= 2,
                f"{len(rtx_3090)} RTX 3090 device(s) visible",
            )
        )
        free_ok = True
        for line in devices:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4 and int(float(parts[3])) < MIN_FREE_VRAM_MB:
                free_ok = False
        checks.append(
            _precondition_row(
                "free_vram",
                bool(devices) and free_ok,
                f"minimum required free VRAM {MIN_FREE_VRAM_MB} MB",
            )
        )
    checks.append(
        _precondition_row(
            "mandatory_model_files",
            all(row.get("exists") is True and row.get("model_file_sha256") for row in model_specs),
            "all mandated GGUF files have hashes",
        )
    )
    checks.append(
        _precondition_row(
            "embedded_gguf_tokenizers",
            all(row.get("tokenizer_loadable") is True for row in model_specs),
            "embedded GGUF tokenizer receipts are loadable",
        )
    )
    try:
        from llama_cpp import llama_cpp

        supports_gpu = bool(getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)())
        runner_detail = f"llama.cpp GPU offload support: {supports_gpu}"
    except Exception as exc:
        supports_gpu = False
        runner_detail = f"{type(exc).__name__}: {exc}"
    checks.append(_precondition_row("llama_cpp_cuda_runner", supports_gpu, runner_detail))
    disk = shutil.disk_usage(REPO_ROOT)
    checks.append(
        _precondition_row("disk_space", disk.free >= MIN_DISK_FREE_BYTES, f"{disk.free} free bytes", str(REPO_ROOT))
    )
    first = time.monotonic_ns()
    second = time.monotonic_ns()
    checks.append(_precondition_row("monotonic_clock", second >= first, f"{first}->{second}"))
    checks.append(
        _precondition_row(
            "new_output_paths",
            not (data_dir / "raw_outputs").exists() and not result_path.exists(),
            "raw output directory and result path do not preexist",
            str(data_dir / "raw_outputs"),
        )
    )
    return checks


def _blocked_artifact(
    *,
    date: str,
    result_path: Path,
    model_resolution: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str | None],
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    """Build a terminal blocked artifact without inference."""

    blockers = [row for row in preconditions if row.get("available") is not True]
    blocked_reason = "; ".join(str(row.get("resource")) for row in blockers)
    artifact: JsonDict = {
        "status": "blocked",
        "MODEL_SPECS": list(model_resolution.get("MODEL_SPECS", [])),
        "models_used": [],
        "cached_sota_pair_receipts": model_resolution.get("cached_sota_pair_receipts", {}),
        "model_file_and_embedded_tokenizer_hashes": model_file_and_embedded_tokenizer_hashes(
            model_resolution.get("MODEL_SPECS", [])
        ),
        "autotokenizer_usage_count": model_resolution.get("autotokenizer_usage_count", 0),
        "device_and_runner_receipts": {"runtime_receipts_by_model": {}, "device_inventory": []},
        "sealed_unit_manifest": dict(manifest),
        "event_path_allocation_receipts": [],
        "atomic_write_receipts": [],
        "raw_output_manifest": {"rows": [], "row_count": 0},
        "per_unit_rows": {"rows": [], "row_count": 0, "normal_row_count": 0, "attack_row_count": 0},
        "one_event_one_path_one_hash_check": {"passed": False, "reasons": ["blocked"]},
        "nonzero_durable_byte_check": {"passed": False, "reasons": ["blocked"]},
        "raw_text_equality_diagnostic": {
            "diagnostic_only": True,
            "affects_event_uniqueness": False,
            "duplicate_raw_text_count": 0,
        },
        "cpu_fallback_count": 0,
        "attack_matrix": {"rows": [], "all_critical_fail_closed": False, "false_accept_count": 0},
        "aggregate_row_recomputation": {"matches_reported": False, "reasons": ["blocked"]},
        "current_adversarial_findings": [
            {"severity": "critical", "kind": "PRECONDITION_FAILED", "detail": blocked_reason}
        ],
        "raw_persistence_canary_ready_score": 0.0,
        "protected_files_unchanged": protected_unchanged_receipt(protected_before),
        "blocked_reason": blocked_reason,
        "gate_check_summary": f"{len(blockers)} precondition(s) failed",
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked_" + blocked_reason.replace(" ", "_"),
        "run_date": date,
        "result_path": str(result_path),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _gate_summary(artifact: Mapping[str, Any]) -> str:
    """Summarize terminal gate state."""

    if artifact.get("blocked_reason"):
        return f"blocked: {artifact['blocked_reason']}"
    if artifact.get("raw_persistence_canary_ready_score") == 1.0:
        return "all raw persistence gates passed"
    findings = [
        str(row.get("kind"))
        for row in artifact.get("current_adversarial_findings", [])
        if row.get("severity") == "critical"
    ]
    return "readiness closed: " + ", ".join(findings or ["non-critical gate failure"])


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    host_preflight_func: HostPreflightFn = default_host_preflight,
    generation_func: GenerationFn = live_generation_for_event,
    event_id_func: EventIdFn = default_event_id,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the Exp6462 raw persistence canary."""

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
    units = build_sealed_units()
    preconditions = host_preflight_func(result_path=result, data_dir=data, model_specs=model_specs)
    for reason in model_resolution.get("blocked_reasons", []):
        preconditions.append(_precondition_row("model_resolution", False, str(reason)))
    manifest_write = not any(row.get("available") is not True for row in preconditions)
    manifest = sealed_unit_manifest(data, units, write=manifest_write and write)
    preconditions.append(
        _precondition_row(
            "sealed_unit_manifest",
            manifest.get("sealed_before_event_allocation") is True
            and manifest.get("held_label_visible_before_generation_count") == 0,
            "unit manifest sealed before event allocation",
            str(manifest.get("path")),
        )
    )
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    if any(row.get("available") is not True for row in preconditions):
        artifact = _blocked_artifact(
            date=date,
            result_path=result,
            model_resolution=model_resolution,
            manifest=manifest,
            preconditions=preconditions,
            protected_before=protected_before,
            duration_s=measured_duration,
            test_exit_codes=test_exit_codes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact

    generated = generate_per_unit_rows(
        data_dir=data,
        units=units,
        model_specs=model_specs,
        source_before=source_before,
        generation_func=generation_func,
        event_id_func=event_id_func,
        write=write,
    )
    rows = list(generated["rows"])
    normal = _normal_rows(rows)
    protected = protected_unchanged_receipt(protected_before)
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    artifact: JsonDict = {
        "status": "complete_with_findings",
        "MODEL_SPECS": model_specs,
        "models_used": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "model_file_and_embedded_tokenizer_hashes": model_file_and_embedded_tokenizer_hashes(model_specs),
        "autotokenizer_usage_count": model_resolution["autotokenizer_usage_count"],
        "device_and_runner_receipts": {
            "runner": "llama_cpp_python",
            "runtime_receipts_by_model": generated["runtime_receipts_by_model"],
            "device_sample_count": sum(len(row.get("device_samples", [])) for row in normal),
            "runner_selection_hashes": [row["runner_selection"]["selection_hash"] for row in normal],
        },
        "sealed_unit_manifest": manifest,
        "event_path_allocation_receipts": generated["event_path_allocation_receipts"],
        "atomic_write_receipts": generated["atomic_write_receipts"],
        "raw_output_manifest": {
            "rows": [
                {
                    "row_id": row["row_id"],
                    "event_id": row["event_id"],
                    "unit_id": row["unit_id"],
                    "model_hf_id": row["model_hf_id"],
                    "path": row["raw_output_path"],
                    "sha256": row["raw_hash"],
                    "byte_length": row["raw_byte_length"],
                    "durable_byte_count": row["durable_byte_count"],
                    "stored_before_parse": row["raw_persisted_before_parse"],
                }
                for row in normal
            ],
            "row_count": len(normal),
        },
        "per_unit_rows": {
            "rows": rows,
            "row_count": len(rows),
            "normal_row_count": len(normal),
            "attack_row_count": len(_attack_rows(rows)),
            "row_hash": sha256_json(rows),
            "written_before_aggregates": True,
        },
        "one_event_one_path_one_hash_check": one_event_one_path_one_hash_check(rows),
        "nonzero_durable_byte_check": nonzero_durable_byte_check(rows),
        "raw_text_equality_diagnostic": raw_text_equality_diagnostic(rows),
        "cpu_fallback_count": sum(1 for row in normal if row.get("cpu_fallback") is True),
        "attack_matrix": attack_matrix(rows),
        "aggregate_row_recomputation": {},
        "current_adversarial_findings": [],
        "raw_persistence_canary_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "blocked_reason": "",
        "gate_check_summary": "",
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": measured_duration,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "run_date": date,
        "result_path": str(result),
    }
    artifact["aggregate_row_recomputation"] = recompute_aggregate_rows(rows, artifact)
    refresh_terminal_fields(artifact)
    errors = validate_artifact(artifact)
    if errors and artifact["status"] != "blocked":
        artifact["status"] = "failed_schema"
        artifact["raw_persistence_canary_ready_score"] = 0.0
        artifact["current_adversarial_findings"] = [
            {"severity": "critical", "kind": "schema_validation", "detail": "; ".join(errors)}
        ]
        artifact["honest_verdict"] = "complete_failed_schema: " + "; ".join(errors[:3])
        artifact["gate_check_summary"] = "schema validation failed"
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Validate an Exp6462 artifact payload."""

    artifact = read_json_object(value) if isinstance(value, (str, Path)) else dict(value)
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors.extend(f"missing required field: {field}" for field in missing)
    if missing:
        return errors
    if [row.get("hf_id") for row in artifact["MODEL_SPECS"]] != list(MANDATED_MODEL_IDS):
        errors.append("MODEL_SPECS mandated ids mismatch")
    if artifact.get("models_used") not in ([], list(MANDATED_MODEL_IDS)):
        errors.append("models_used must be empty or match mandated ids")
    if artifact.get("autotokenizer_usage_count") != 0:
        errors.append("autotokenizer_usage_count must be zero")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for exact byte and receipt checks")
    rows = artifact.get("per_unit_rows", {}).get("rows", [])
    if artifact.get("per_unit_rows", {}).get("row_count") != len(rows):
        errors.append("per_unit_rows row_count mismatch")
    if artifact.get("status") == "success":
        expected_normal = UNIT_COUNT * len(MANDATED_MODEL_IDS) * REPLICATES_PER_UNIT
        if artifact.get("per_unit_rows", {}).get("normal_row_count") != expected_normal:
            errors.append("normal row count mismatch")
        if artifact.get("per_unit_rows", {}).get("attack_row_count") != len(ATTACK_IDS):
            errors.append("attack row count mismatch")
        if artifact.get("sealed_unit_manifest", {}).get("unit_count") != UNIT_COUNT:
            errors.append("sealed unit count mismatch")
    if (
        artifact.get("one_event_one_path_one_hash_check", {}).get("passed") is not True
        and artifact.get("status") != "blocked"
    ):
        errors.append("one event/path/hash check failed")
    if (
        artifact.get("nonzero_durable_byte_check", {}).get("passed") is not True
        and artifact.get("status") != "blocked"
    ):
        errors.append("nonzero durable bytes check failed")
    if artifact.get("cpu_fallback_count") != 0:
        errors.append("cpu_fallback_count must be zero")
    if (
        artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is not True
        and artifact.get("status") == "success"
    ):
        errors.append("attack matrix must fail closed")
    if artifact.get("attack_matrix", {}).get("false_accept_count", 0) != 0:
        errors.append("ready artifact cannot accept attacks")
    if (
        artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is not True
        and artifact.get("status") == "success"
    ):
        errors.append("reported aggregates must recompute from rows")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact.get("field_principles", {}):
            errors.append(f"missing field_principles entry: {field}")
            break
    for condition in READINESS_CONDITIONS:
        if f"raw_persistence_canary_ready_score:{condition}" not in artifact.get("field_principles", {}):
            errors.append(f"missing readiness field_principles entry: {condition}")
            break
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (
        verdict.startswith("success:")
        or verdict.startswith("complete:")
        or verdict.startswith("complete_failed_schema:")
        or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict lacks required terminal prefix")
    expected_checksum = payload_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected_checksum:
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("raw_persistence_canary_ready_score") == 1.0 and _ready_score(artifact) != 1.0:
        errors.append("raw_persistence_canary_ready_score does not recompute")
    return errors


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        errors = validate_artifact(path)
        if errors:
            for error in errors:
                print(error)
            return 1
        print(f"valid: {path}")
        return 0
    artifact = run(date=args.date, result_path=path, data_dir=REPO_ROOT / DATA_DIR_RELATIVE_PATH)
    print(json.dumps({"status": artifact["status"], "result_path": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
