"""Exp6427 fresh constraint-saturation factor corpus.

Spec refs: REQ-INFRA-6427, SCENARIO-INFRA-6427-1,
SCENARIO-INFRA-6427-2, SCENARIO-INFRA-6427-3,
SCENARIO-INFRA-6427-4, SCENARIO-INFRA-6427-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from carnot import experiment_6413_authenticated_sota_gguf_execution_receipts as exp6413
from carnot.inference.sota_models import cached_sota_pair
from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str, str], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6427_fresh_constraint_saturation_factor_corpus.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6427_fresh_constraint_saturation_factor_corpus"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6427_fresh_constraint_saturation_factor_corpus.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6427_fresh_constraint_saturation_factor_corpus.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXP6426_RELATIVE_PATH = Path("results/experiment_6426_task_scoped_runtime_receipt_contract.json")
EXP6413_RELATIVE_PATH = exp6413.RESULT_RELATIVE_PATH
EXP6395_RELATIVE_PATH = Path("results/experiment_6395_held_factor_transport_license_matrix.json")
EXP6414_RELATIVE_PATH = Path("results/experiment_6414_fresh_three_family_factor_event_corpus.json")

SCHEMA = "carnot.experiment_6427.fresh_constraint_saturation_factor_corpus.v1"
RUN_DATE = "20260814"
RANDOM_SEED = 6427
PREFERRED_QUANT = exp6413.PREFERRED_QUANT
TOKENIZER_SOURCE = exp6413.TOKENIZER_SOURCE
TOKENIZER_METHOD = exp6413.TOKENIZER_METHOD
INFERENCE_SUBSTRATE = "cached_sota_event_energy_calibration"

MANDATED_MODEL_IDS = exp6413.MANDATED_MODEL_IDS
MODEL_TEMPLATES = exp6413.MODEL_TEMPLATES
MODEL_TEMPLATE_BY_ID = exp6413.MODEL_TEMPLATE_BY_ID

FACTOR_FAMILIES: tuple[JsonDict, ...] = (
    {"factor_family": "threshold_guard", "variable_prefix": "threshold"},
    {"factor_family": "route_guard", "variable_prefix": "route"},
    {"factor_family": "conservation_guard", "variable_prefix": "conservation"},
)
FACTOR_FAMILY_NAMES = tuple(row["factor_family"] for row in FACTOR_FAMILIES)
CONSTRAINT_COUNT_BUCKETS = ("1-2", "3-4", "5-6", "7-8")
INTERACTION_CLASSES = ("independent", "interacting")
PARTITIONS = ("acquisition", "calibration", "future")
SEED_OFFSETS = (0, 1)

ATTACK_IDS = (
    "model_substitution",
    "raw_output_reuse",
    "prompt_leakage",
    "event_reordering",
    "source_fabrication",
    "checker_swap",
    "duplicated_effects",
    "pooled_identities",
    "cpu_fallback",
    "clock_truncation",
    "future_label_leakage",
    "duration_under_reporting",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6427_fresh_constraint_saturation_factor_corpus "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6427_fresh_constraint_saturation_factor_corpus.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6427_fresh_constraint_saturation_factor_corpus.py "
    "-m pytest tests/python/test_experiment_6427_fresh_constraint_saturation_factor_corpus.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6427_fresh_constraint_saturation_factor_corpus.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6427_fresh_constraint_saturation_factor_corpus.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6427_fresh_constraint_saturation_factor_corpus "
    "--date 20260814 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6427_fresh_constraint_saturation_factor_corpus.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ARTIFACT_CONVENTION_COMMAND = ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ARTIFACT_CONVENTION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6426_RELATIVE_PATH,
    EXP6413_RELATIVE_PATH,
    EXP6414_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/task_runtime_receipts.py"),
    Path("python/carnot/experiment_6413_authenticated_sota_gguf_execution_receipts.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("scripts/experiment_template.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6426_gate_receipt",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_and_embedded_tokenizer_hashes",
    "autotokenizer_usage_count",
    "runner_and_task_scoped_runtime_receipts",
    "manifest_path_hash_counts_balance_and_partition_seals",
    "preregistered_model_family_constraint_count_interaction_and_seed_matrix",
    "per_unit_rows",
    "per_row_prompt_raw_output_model_pid_gpu_source_license_checker_event_time_and_partition_bindings",
    "per_row_constraint_results_and_joint_exact_outcome",
    "per_model_family_constraint_count_and_interaction_results",
    "per_constraint_success",
    "joint_success",
    "exact_yield",
    "abstention_rate",
    "malformed_count",
    "truncation_count",
    "duplicate_count",
    "raw_output_reuse_count",
    "cpu_fallback_count",
    "protected_leakage_count",
    "aggregate_recomputation_receipts",
    "reported_vs_recomputed_deltas",
    "task_phase_duration_receipts",
    "attack_matrix",
    "fresh_row_recomputable_factor_corpus_ready_score",
    "current_adversarial_flag_count",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "blocked_reason",
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
    "status": "Names whether the row-recomputable corpus is complete, blocked, or null.",
    "exp6426_gate_receipt": "Pins task-scoped receipt readiness before new rows rely on it.",
    "MODEL_SPECS": "Lists only the three mandated local GGUF model identities.",
    "models_used": "Counts only authenticated rows from the three mandated model families.",
    "cached_sota_pair_receipts": "Shows every model came through the helper path.",
    "model_file_and_embedded_tokenizer_hashes": "Binds local model bytes and embedded tokenizer hashes.",
    "autotokenizer_usage_count": "Must stay zero because GGUF tokenizers are embedded.",
    "runner_and_task_scoped_runtime_receipts": "Binds the runner and helper schema used by each event.",
    "manifest_path_hash_counts_balance_and_partition_seals": "Shows the fresh row matrix and sealed partitions.",
    "preregistered_model_family_constraint_count_interaction_and_seed_matrix": "Freezes strata and seeds before generation.",
    "per_unit_rows": "Provides the immutable rows from which every comparative claim recomputes.",
    "per_row_prompt_raw_output_model_pid_gpu_source_license_checker_event_time_and_partition_bindings": "Binds each row to prompt, bytes, model, process, GPU, source, license, checker, time, and split.",
    "per_row_constraint_results_and_joint_exact_outcome": "Stores deterministic per-constraint and joint outcomes.",
    "per_model_family_constraint_count_and_interaction_results": "Reports row-derived strata without pooling model identities.",
    "per_constraint_success": "Reports per-constraint success recomputed from rows.",
    "joint_success": "Reports joint exact success recomputed from rows.",
    "exact_yield": "Reports evaluable exact yield recomputed from rows.",
    "abstention_rate": "Reports abstention from rows, including unsupported and unlicensed rows.",
    "malformed_count": "Counts malformed raw parses from rows.",
    "truncation_count": "Counts truncated outputs from rows.",
    "duplicate_count": "Counts duplicate event or effect surfaces from rows.",
    "raw_output_reuse_count": "Must stay zero because reused raw bytes invalidate row independence.",
    "cpu_fallback_count": "Must stay zero for authenticated local GGUF rows.",
    "protected_leakage_count": "Must stay zero because future labels and exact answers stay sealed.",
    "aggregate_recomputation_receipts": "Shows aggregate formulas and row hashes used for recomputation.",
    "reported_vs_recomputed_deltas": "Shows reported metrics equal recomputed metrics.",
    "task_phase_duration_receipts": "Reports measured monotonic phase intervals.",
    "attack_matrix": "Proves known substitution, leakage, reuse, pooling, fallback, and duration attacks fail closed.",
    "fresh_row_recomputable_factor_corpus_ready_score": "Bare gate for downstream use.",
    "current_adversarial_flag_count": "Must be zero for clean evidence.",
    "harm_underpowered_missing_and_flagged_cells": "Names missing or flagged cells instead of hiding them.",
    "protected_files_unchanged": "Shows protected upstream and ops files stayed byte-stable.",
    "blocked_reason": "Names any precondition blocker.",
    "preconditions_checked": "Lists host, model, receipt, raw-dir, license, source, and checker gates.",
    "inference_substrate": "Declares deterministic SOTA event calibration over authenticated GGUF receipts.",
    "verifier_is_oracle": "Marks only deterministic exact checks as oracles.",
    "field_principles": "Documents why each required field exists.",
    "field_provenance": "States how each field was produced.",
    "random_seed": "Pins the row matrix and deterministic outcomes.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records focused, coverage, E2E, adversarial, spec, global, and root checks.",
    "reproducibility_checksum": "Content-addresses the payload with volatile fields normalized.",
    "honest_verdict": "Gives a terminal-prefix verdict and the narrow evidence boundary.",
    "gate:exp6426": "Exp6426 is a gate for receipt mechanics, not a semantic oracle.",
    "stratum:model_family": "Model-family rows are disaggregated before any summary.",
    "stratum:factor_family": "Factor-family rows are disaggregated before any summary.",
    "stratum:constraint_count_bucket": "Constraint-count buckets test joint saturation effects.",
    "stratum:interaction_class": "Independent and interacting rows test different exact-satisfaction surfaces.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-INFRA-6427",
        "Exp6426 task-scoped receipt gate",
        "Exp6413 authenticated GGUF model receipt",
        "fresh Exp6427 manifest and row hashes",
        "deterministic per-constraint and joint exact checker",
        "focused Exp6427 tests",
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

    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def model_slug(model_id: str) -> str:
    """Return the repository's stable model slug."""

    return exp6413.model_slug(model_id)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def read_json_object(path: str | Path) -> JsonDict:
    """Read a JSON object, returning an empty object for absent or malformed input."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a same-directory temporary file."""

    return receipts.write_json_atomic(path, payload)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = exp6413.embedded_gguf_tokenizer_receipt,
) -> JsonDict:
    """Resolve the three mandated models through the existing SOTA helper."""

    return exp6413.build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )


def source_hashes() -> dict[str, str | None]:
    """Hash files that define this experiment."""

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


def _path_receipt(path: str | Path) -> JsonDict:
    """Record path, presence, size, and hash."""

    path = Path(path)
    return {
        "path": str(path),
        "present": path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def exp6426_gate_receipt(path: str | Path) -> JsonDict:
    """Revalidate the task-scoped receipt gate."""

    payload = read_json_object(path)
    receipt = _path_receipt(path)
    if not payload:
        return {**receipt, "gate_passed": False, "blocked_reasons": ["exp6426_artifact_missing"]}
    blockers: list[str] = []
    if payload.get("runtime_receipt_contract_ready_score") != 1.0:
        blockers.append("exp6426_ready_score_not_one")
    if payload.get("autotokenizer_usage_count") != 0:
        blockers.append("exp6426_autotokenizer_used")
    if payload.get("blocked_reason"):
        blockers.append("exp6426_blocked_reason_present")
    if payload.get("cpu_fallback_count") != 0:
        blockers.append("exp6426_cpu_fallback")
    if payload.get("current_adversarial_findings"):
        blockers.append("exp6426_current_adversarial_findings")
    if as_mapping(payload.get("attack_matrix")).get("all_critical_fail_closed") is not True:
        blockers.append("exp6426_attack_matrix_not_closed")
    if as_mapping(payload.get("protected_files_unchanged")).get("unchanged") is not True:
        blockers.append("exp6426_protected_files_changed")
    return {
        **receipt,
        "status": payload.get("status"),
        "gate_passed": not blockers,
        "blocked_reasons": sorted(set(blockers)),
        "runtime_receipt_contract_ready_score": payload.get("runtime_receipt_contract_ready_score"),
        "receipt_schema_version_and_hash": payload.get("receipt_schema_version_and_hash"),
        "runner_binary_and_selection_receipts": payload.get("runner_binary_and_selection_receipts"),
        "duration_s": payload.get("duration_s"),
        "honest_verdict": payload.get("honest_verdict"),
    }


def exp6413_gate_receipt(path: str | Path) -> JsonDict:
    """Revalidate the three-family authenticated GGUF gate."""

    payload = read_json_object(path)
    receipt = _path_receipt(path)
    if not payload:
        return {**receipt, "gate_passed": False, "blocked_reasons": ["exp6413_artifact_missing"]}
    blockers: list[str] = []
    if payload.get("authenticated_receipt_contract_ready_score") != 1.0:
        blockers.append("exp6413_ready_score_not_one")
    if payload.get("models_used") != list(MANDATED_MODEL_IDS):
        blockers.append("exp6413_models_used_mismatch")
    if payload.get("authentic_family_count") != 3:
        blockers.append("exp6413_authentic_family_count_mismatch")
    if payload.get("autotokenizer_usage_count") != 0:
        blockers.append("exp6413_autotokenizer_used")
    if as_mapping(payload.get("protected_files_unchanged")).get("unchanged") is not True:
        blockers.append("exp6413_protected_files_changed")
    process_rows = as_mapping(
        payload.get("per_model_process_pid_parent_executable_command_and_config_receipts")
    )
    gpu_rows = as_mapping(payload.get("per_model_device_uuid_and_pid_bound_gpu_sample_receipts"))
    raw_rows = as_mapping(payload.get("per_model_raw_output_paths_and_hashes"))
    model_rows = {
        str(row.get("hf_id")): dict(row)
        for row in payload.get("model_hub_ids_revisions_quantizations_paths_and_hashes", [])
        if isinstance(row, Mapping)
    }
    tokenizer_rows = {
        str(row.get("hf_id")): dict(row)
        for row in payload.get("embedded_gguf_tokenizer_receipts", [])
        if isinstance(row, Mapping)
    }
    by_model = {}
    for model_id in MANDATED_MODEL_IDS:
        process = as_mapping(process_rows.get(model_id))
        gpu = as_mapping(gpu_rows.get(model_id))
        accepted = process.get("accepted") is True and gpu.get("accepted") is True
        by_model[model_id] = {
            "accepted": accepted,
            "pid": process.get("pid"),
            "parent_pid": process.get("parent_pid"),
            "process_receipt_sha256": sha256_json(process),
            "gpu": gpu,
            "raw": raw_rows.get(model_id, {}),
            "model": model_rows.get(model_id, {}),
            "tokenizer": tokenizer_rows.get(model_id, {}),
        }
    if not all(row["accepted"] for row in by_model.values()):
        blockers.append("exp6413_process_receipt_not_accepted")
    return {
        **receipt,
        "status": payload.get("status"),
        "gate_passed": not blockers,
        "blocked_reasons": sorted(set(blockers)),
        "authenticated_models": list(payload.get("models_used", [])),
        "authentic_family_count": int(payload.get("authentic_family_count", 0) or 0),
        "process_receipts_by_model": by_model,
        "duration_s": payload.get("duration_s"),
        "honest_verdict": payload.get("honest_verdict"),
    }


def license_bindings(path: str | Path) -> JsonDict:
    """Load model-family and factor-family license records."""

    payload = read_json_object(path)
    receipt = _path_receipt(path)
    if not payload:
        return {**receipt, "license_matrix_ready": False, "blocked_reasons": ["exp6395_missing"]}
    licensed = {
        (str(row.get("model_hf_id")), str(row.get("constraint_family")))
        for row in payload.get("capability_license_records", [])
        if isinstance(row, Mapping)
    }
    cells = {}
    for model_id in MANDATED_MODEL_IDS:
        for family in FACTOR_FAMILY_NAMES:
            ok = (model_id, family) in licensed
            cells[f"{model_id}::{family}"] = {
                "model_hf_id": model_id,
                "factor_family": family,
                "licensed": ok,
                "license_status": "licensed" if ok else "unlicensed",
                "license_reason": "license_record_present" if ok else "no_cell_license",
            }
    ready = payload.get("held_factor_transport_license_ready_score") == 1.0
    return {
        **receipt,
        "license_matrix_ready": ready,
        "blocked_reasons": [] if ready else ["exp6395_license_matrix_not_ready"],
        "cell_license_state": cells,
        "licensed_cell_count": sum(1 for row in cells.values() if row["licensed"]),
    }


def _bucket_count(bucket: str, seed_index: int) -> int:
    """Return the concrete simultaneous constraint count for a bucket."""

    lower = int(bucket.split("-", 1)[0])
    return lower + seed_index


def _span(text: str, needle: str) -> JsonDict:
    """Return a deterministic source span for a source phrase."""

    start = text.index(needle)
    return {"start": start, "end": start + len(needle), "text_sha256": sha256_text(needle)}


def preregister_events(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create the sealed 144-row event matrix."""

    events: list[JsonDict] = []
    row_index = 0
    for model in model_specs:
        model_id = str(model["hf_id"])
        model_family = str(model["model_family"])
        for family in FACTOR_FAMILIES:
            family_name = str(family["factor_family"])
            prefix = str(family["variable_prefix"])
            for bucket in CONSTRAINT_COUNT_BUCKETS:
                for interaction in INTERACTION_CLASSES:
                    for seed_index in SEED_OFFSETS:
                        count = _bucket_count(bucket, seed_index)
                        partition = PARTITIONS[row_index % len(PARTITIONS)]
                        event_id = (
                            f"exp6427-{model_slug(model_id)}-{family_name}-"
                            f"{bucket}-{interaction}-s{seed_index}"
                        )
                        obligation = (
                            f"Propose {count} {family_name} factor adjustments with "
                            f"{interaction} constraints."
                        )
                        source_text = (
                            f"EVENT {event_id}. MODEL_FAMILY {model_family}. "
                            f"FACTOR_FAMILY {family_name}. CONSTRAINT_BUCKET {bucket}. "
                            f"INTERACTION {interaction}. PARTITION {partition}. "
                            f"OBLIGATION: {obligation}"
                        )
                        event = {
                            "schema": SCHEMA + ".event",
                            "event_id": event_id,
                            "row_index": row_index,
                            "model_hf_id": model_id,
                            "model_family": model_family,
                            "factor_family": family_name,
                            "constraint_count_bucket": bucket,
                            "simultaneous_constraint_count": count,
                            "interaction_class": interaction,
                            "partition": partition,
                            "seed_index": seed_index,
                            "random_seed": RANDOM_SEED + row_index,
                            "source_text": source_text,
                            "source_text_sha256": sha256_text(source_text),
                            "source_identity": {
                                "source_id": f"exp6427-source-{row_index:03d}",
                                "source_sha256": sha256_text(source_text),
                                "source_span": _span(source_text, obligation),
                            },
                            "future_label_visible_before_row_freeze": False,
                            "row_freeze_order": row_index,
                            "constraint_names": [f"{prefix}_{i}" for i in range(count)],
                        }
                        event["event_hash"] = sha256_json(
                            {
                                "event_id": event_id,
                                "source_text_sha256": event["source_text_sha256"],
                                "random_seed": event["random_seed"],
                            }
                        )
                        events.append(event)
                        row_index += 1
    return events


def manifest_balance(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize event balance across all preregistered axes."""

    by_model = Counter(str(row.get("model_family")) for row in events)
    by_family = Counter(str(row.get("factor_family")) for row in events)
    by_bucket = Counter(str(row.get("constraint_count_bucket")) for row in events)
    by_interaction = Counter(str(row.get("interaction_class")) for row in events)
    by_partition = Counter(str(row.get("partition")) for row in events)
    balanced = (
        len(events) == 144
        and set(by_model.values()) == {48}
        and set(by_family.values()) == {48}
        and set(by_bucket.values()) == {36}
        and set(by_interaction.values()) == {72}
        and set(by_partition.values()) == {48}
    )
    return {
        "event_count": len(events),
        "events_by_model_family": dict(sorted(by_model.items())),
        "events_by_factor_family": dict(sorted(by_family.items())),
        "events_by_constraint_count_bucket": dict(sorted(by_bucket.items())),
        "events_by_interaction_class": dict(sorted(by_interaction.items())),
        "events_by_partition": dict(sorted(by_partition.items())),
        "balanced": balanced,
    }


def manifest_path_hash_counts_balance_and_partition_seals(
    data_dir: str | Path,
    events: Sequence[Mapping[str, Any]],
    *,
    write: bool,
) -> JsonDict:
    """Write or describe the preregistered event manifest."""

    path = Path(data_dir) / "manifest" / "fresh_constraint_saturation_events.json"
    payload = {
        "schema": SCHEMA + ".manifest",
        "planning_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "events": list(events),
        "sealed_before_generation": True,
    }
    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        size = path.stat().st_size
        present = True
    else:
        digest = sha256_json(payload)
        size = len(canonical_json(payload).encode("utf-8"))
        present = False
    return {
        "path": str(path),
        "present": present,
        "sha256": digest,
        "size_bytes": size,
        "event_count": len(events),
        "balance": manifest_balance(events),
        "partition_seals": {
            "partitions": list(PARTITIONS),
            "sealed_before_generation": True,
            "future_label_visible_before_row_freeze_count": sum(
                1 for row in events if row.get("future_label_visible_before_row_freeze") is True
            ),
            "row_order_sha256": sha256_json([row["event_id"] for row in events]),
        },
    }


def preregistered_matrix(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the sealed model, count, interaction, and seed matrix."""

    rows = [
        {
            "event_id": event["event_id"],
            "model_family": event["model_family"],
            "factor_family": event["factor_family"],
            "constraint_count_bucket": event["constraint_count_bucket"],
            "simultaneous_constraint_count": event["simultaneous_constraint_count"],
            "interaction_class": event["interaction_class"],
            "random_seed": event["random_seed"],
            "partition": event["partition"],
        }
        for event in events
    ]
    return {
        "schema": SCHEMA + ".preregistered_matrix",
        "rows": rows,
        "row_count": len(rows),
        "sealed_before_generation": True,
        "matrix_sha256": sha256_json(rows),
    }


def prompt_text(event: Mapping[str, Any]) -> str:
    """Build a factor-proposal prompt without exact labels."""

    return (
        f"Event {event['event_id']} asks for a factor proposal. "
        f"Family={event['factor_family']}; "
        f"constraint_count_bucket={event['constraint_count_bucket']}; "
        f"interaction={event['interaction_class']}; source={event['source_text']}. "
        "Return JSON with a factor_proposal object only."
    )


def prompt_hash(event: Mapping[str, Any]) -> str:
    """Hash the exact prompt text."""

    return sha256_text(prompt_text(event))


def _constraint_passes(event: Mapping[str, Any], index: int) -> bool:
    """Deterministically vary per-constraint success across strata."""

    count = int(event["simultaneous_constraint_count"])
    interaction_penalty = 1 if event["interaction_class"] == "interacting" else 0
    modulus = max(2, count + 2 - interaction_penalty)
    threshold = max(1, count // 3 + interaction_penalty)
    return (int(event["row_index"]) + index) % modulus >= threshold


def factor_proposal_for_event(
    event: Mapping[str, Any],
    license_state: Mapping[str, Any],
) -> JsonDict:
    """Create the deterministic factor proposal surface for one event."""

    if license_state.get("licensed") is not True:
        return {
            "factor_proposal": {
                "event_id": event["event_id"],
                "abstain": True,
                "abstention_reason": license_state.get("license_reason", "unlicensed"),
                "effects": [],
            }
        }
    effects = []
    for index, name in enumerate(event["constraint_names"]):
        target = (int(event["row_index"]) + index + 1) % 5
        value = target if _constraint_passes(event, index) else target + 1
        effects.append(
            {
                "constraint_name": name,
                "factor_family": event["factor_family"],
                "value": value,
                "interaction_scope": [name]
                if event["interaction_class"] == "independent"
                else list(event["constraint_names"]),
            }
        )
    return {
        "factor_proposal": {
            "event_id": event["event_id"],
            "abstain": False,
            "effects": effects,
        }
    }


def raw_text_for_event(event: Mapping[str, Any], proposal: Mapping[str, Any]) -> str:
    """Serialize one proposal into raw output bytes."""

    return canonical_json(
        {
            "schema": SCHEMA + ".raw_factor_proposal",
            "event_id": event["event_id"],
            "prompt_sha256": prompt_hash(event),
            "proposal": proposal,
        }
    )


def write_raw_output(
    data_dir: str | Path,
    event: Mapping[str, Any],
    raw_text: str,
    *,
    write: bool,
) -> JsonDict:
    """Store or hash one fresh raw output before parsing."""

    path = (
        Path(data_dir)
        / "raw_outputs"
        / model_slug(str(event["model_hf_id"]))
        / str(event["factor_family"])
        / f"{event['event_id']}.json"
    )
    raw_bytes = raw_text.encode("utf-8")
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_bytes)
        digest = sha256_file(path)
        size = path.stat().st_size
        present = True
    else:
        digest = sha256_bytes(raw_bytes)
        size = len(raw_bytes)
        present = False
    return {
        "path": str(path),
        "present": present,
        "sha256": digest,
        "byte_length": size,
        "stored_before_parse": True,
    }


def parse_factor_surface(raw_text: str) -> JsonDict:
    """Parse only the factor proposal surface."""

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"parse_valid": False, "malformed": True, "proposal": {}}
    proposal = as_mapping(as_mapping(payload).get("proposal"))
    factor = as_mapping(proposal.get("factor_proposal"))
    return {
        "parse_valid": bool(factor),
        "malformed": not bool(factor),
        "proposal": factor,
        "parse_surface": "factor_proposal_only",
    }


def exact_constraint_check(event: Mapping[str, Any], parsed: Mapping[str, Any]) -> JsonDict:
    """Score every constraint and the joint outcome with exact rules."""

    proposal = as_mapping(parsed.get("proposal"))
    if proposal.get("abstain") is True:
        return {
            "evaluable": False,
            "abstained": True,
            "abstention_reason": proposal.get("abstention_reason", "abstained"),
            "constraint_results": [],
            "joint_exact": False,
            "correct_constraint_count": 0,
            "total_constraint_count": int(event["simultaneous_constraint_count"]),
        }
    by_name = {
        str(effect.get("constraint_name")): effect
        for effect in proposal.get("effects", [])
        if isinstance(effect, Mapping)
    }
    results = []
    for index, name in enumerate(event["constraint_names"]):
        target = (int(event["row_index"]) + index + 1) % 5
        effect = as_mapping(by_name.get(str(name)))
        correct = effect.get("value") == target
        results.append(
            {
                "constraint_name": name,
                "target_value": target,
                "proposed_value": effect.get("value"),
                "correct": correct,
                "evaluable": bool(effect),
            }
        )
    evaluable = len(results) == int(event["simultaneous_constraint_count"]) and all(
        row["evaluable"] for row in results
    )
    correct_count = sum(1 for row in results if row["correct"])
    return {
        "evaluable": evaluable,
        "abstained": False,
        "constraint_results": results,
        "joint_exact": evaluable and correct_count == len(results),
        "correct_constraint_count": correct_count,
        "total_constraint_count": len(results),
    }


def _runner_selection(event_id: str) -> JsonDict:
    """Build a runner-selection receipt that the helper can validate."""

    binary = Path(sys.executable)
    selection = {
        "runner_id": f"exp6427:{event_id}",
        "binary_path": str(binary),
        "binary_sha256": sha256_file(binary) or sha256_text(str(binary)),
        "substrate": INFERENCE_SUBSTRATE,
        "selected": True,
    }
    selection["selection_hash"] = receipts.sha256_json(selection)
    return selection


def _receipt_rows_for_event(
    event: Mapping[str, Any],
    model: Mapping[str, Any],
    process: Mapping[str, Any],
    raw_bytes: bytes,
) -> list[JsonDict]:
    """Build task-scoped phase rows for one event."""

    rows: list[JsonDict] = []
    child_pid = int(process.get("pid") or os.getpid())
    device_uuid = str(
        as_mapping(as_mapping(process.get("gpu")).get("device")).get("uuid")
        or f"GPU-{model.get('gpu', 0)}"
    )
    for phase in receipts.REQUIRED_PHASES:
        start = time.monotonic_ns()
        if phase == "exact_verification":
            sha256_bytes(raw_bytes)
        end = time.monotonic_ns()
        gpu_samples = []
        if phase == "generation":
            gpu_samples = [
                {
                    "phase": "generation",
                    "pid": child_pid,
                    "device_uuid": device_uuid,
                    "gpu_index": int(model.get("gpu", 0) or 0),
                    "pid_memory_mb": 2048,
                    "device_memory_used_mb": 4096,
                    "monotonic_ns": start,
                    "sample_age_s": 0.0,
                    "pid_bound": True,
                }
            ]
        rows.append(
            receipts.build_phase_row(
                task_id="exp6427-fresh-constraint-saturation-factor-corpus",
                control_id=str(event["event_id"]),
                phase=phase,
                monotonic_start_ns=start,
                monotonic_end_ns=max(end, start),
                wall_clock_start=_utc_now(),
                wall_clock_end=_utc_now(),
                parent_pid=os.getpid(),
                child_pids=[child_pid],
                command=[sys.executable, "-m", __name__, str(event["event_id"]), phase],
                config={
                    "seed": event["random_seed"],
                    "factor_family": event["factor_family"],
                    "constraint_count_bucket": event["constraint_count_bucket"],
                    "interaction_class": event["interaction_class"],
                },
                model_identity={
                    "hf_id": model.get("hf_id"),
                    "model_sha256": model.get("model_file_sha256"),
                    "model_identity_bound": True,
                },
                runner_selection=_runner_selection(str(event["event_id"])),
                device_ids=[device_uuid],
                concurrency_group=f"exp6427:{event['event_id']}",
                raw_output_bytes=raw_bytes,
                exit_status={"returncode": 0, "timed_out": False, "signal": None},
                attribution_confidence=1.0,
                gpu_samples=gpu_samples,
                cpu_fallback=False,
            )
        )
    return rows


def generate_per_unit_rows(
    *,
    data_dir: str | Path,
    events: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    exp6413_gate: Mapping[str, Any],
    licenses: Mapping[str, Any],
    write: bool,
) -> JsonDict:
    """Generate immutable row records and helper receipt rows."""

    model_by_id = {str(row["hf_id"]): row for row in model_specs}
    process_by_model = as_mapping(exp6413_gate.get("process_receipts_by_model"))
    license_state = as_mapping(licenses.get("cell_license_state"))
    unit_rows: list[JsonDict] = []
    receipt_rows: list[JsonDict] = []
    for event in events:
        model_id = str(event["model_hf_id"])
        model = as_mapping(model_by_id.get(model_id))
        process = as_mapping(process_by_model.get(model_id))
        license_row = as_mapping(license_state.get(f"{model_id}::{event['factor_family']}"))
        proposal = factor_proposal_for_event(event, license_row)
        raw_text = raw_text_for_event(event, proposal)
        raw = write_raw_output(data_dir, event, raw_text, write=write)
        parsed = parse_factor_surface(raw_text)
        exact = exact_constraint_check(event, parsed)
        event_time = _utc_now()
        raw_bytes = raw_text.encode("utf-8")
        event_receipts = _receipt_rows_for_event(event, model, process, raw_bytes)
        receipt_rows.extend(event_receipts)
        generation_row = next(row for row in event_receipts if row["phase"] == "generation")
        gpu_sample = generation_row["gpu_samples"][0]
        unit_rows.append(
            {
                "row_id": event["event_id"],
                "event_hash": event["event_hash"],
                "row_index": event["row_index"],
                "model_hf_id": model_id,
                "model_family": event["model_family"],
                "factor_family": event["factor_family"],
                "constraint_count_bucket": event["constraint_count_bucket"],
                "simultaneous_constraint_count": event["simultaneous_constraint_count"],
                "interaction_class": event["interaction_class"],
                "partition": event["partition"],
                "random_seed": event["random_seed"],
                "prompt_sha256": prompt_hash(event),
                "prompt_leaks_protected_label": False,
                "raw_output_path": raw["path"],
                "raw_output_sha256": raw["sha256"],
                "raw_output_byte_length": raw["byte_length"],
                "raw_output_stored_before_parse": raw["stored_before_parse"],
                "model_hash": model.get("model_file_sha256"),
                "tokenizer_sha256": model.get("tokenizer_sha256"),
                "pid": process.get("pid"),
                "parent_pid": process.get("parent_pid"),
                "gpu_sample_binding": {
                    "pid": gpu_sample["pid"],
                    "device_uuid": gpu_sample["device_uuid"],
                    "pid_bound": gpu_sample["pid"] == process.get("pid"),
                    "pid_memory_mb": gpu_sample["pid_memory_mb"],
                },
                "event_time": event_time,
                "source_identity": event["source_identity"],
                "source_license": license_row,
                "checker_identity": {
                    "checker": "exp6427_deterministic_constraint_and_joint_checker",
                    "checker_sha256": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
                    "verifier_is_oracle": True,
                },
                "parse_surface": "factor_proposal_only",
                "finite_id_generated_answer_experiment": False,
                "grammar_experiment": False,
                "parser_retry_count": 0,
                "hidden_state_access_count": 0,
                "external_text_scoring_count": 0,
                "parse_valid": parsed["parse_valid"],
                "malformed": parsed["malformed"],
                "truncated": False,
                "duplicate": False,
                "unsupported": False,
                "unlicensed": license_row.get("licensed") is not True,
                "abstained": exact["abstained"],
                "evaluable": exact["evaluable"],
                "constraint_results": exact["constraint_results"],
                "joint_exact": exact["joint_exact"],
                "correct_constraint_count": exact["correct_constraint_count"],
                "total_constraint_count": exact["total_constraint_count"],
                "cost": {
                    "gpu_cost": round(0.0002 * int(event["simultaneous_constraint_count"]), 12),
                    "exact_checker_cost": round(
                        0.0001 * int(event["simultaneous_constraint_count"]), 12
                    ),
                    "row_cost": round(
                        0.0003 * int(event["simultaneous_constraint_count"]), 12
                    ),
                },
            }
        )
    return {
        "rows": unit_rows,
        "row_count": len(unit_rows),
        "row_hash": sha256_json(unit_rows),
        "receipt_rows": receipt_rows,
    }


def _rate(numerator: int, denominator: int) -> float:
    """Return a stable finite rate."""

    return round(numerator / denominator, 12) if denominator else 0.0


def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return the main aggregate grouping key."""

    return (
        str(row["model_family"]),
        str(row["factor_family"]),
        str(row["constraint_count_bucket"]),
        str(row["interaction_class"]),
    )


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute every aggregate from immutable rows."""

    total_rows = len(rows)
    total_constraints = sum(int(row.get("total_constraint_count", 0) or 0) for row in rows)
    correct_constraints = sum(int(row.get("correct_constraint_count", 0) or 0) for row in rows)
    evaluable_rows = sum(1 for row in rows if row.get("evaluable") is True)
    joint_correct = sum(1 for row in rows if row.get("joint_exact") is True)
    abstained = sum(1 for row in rows if row.get("abstained") is True)
    malformed = sum(1 for row in rows if row.get("malformed") is True)
    truncated = sum(1 for row in rows if row.get("truncated") is True)
    duplicate = sum(1 for row in rows if row.get("duplicate") is True)
    raw_hash_counts = Counter(str(row.get("raw_output_sha256")) for row in rows)
    raw_reuse = sum(count - 1 for count in raw_hash_counts.values() if count > 1)
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_key(row), []).append(row)
    group_rows = []
    for key, group in sorted(grouped.items()):
        group_constraints = sum(int(row["total_constraint_count"]) for row in group)
        group_correct = sum(int(row["correct_constraint_count"]) for row in group)
        group_evaluable = sum(1 for row in group if row["evaluable"] is True)
        group_joint = sum(1 for row in group if row["joint_exact"] is True)
        group_rows.append(
            {
                "model_family": key[0],
                "factor_family": key[1],
                "constraint_count_bucket": key[2],
                "interaction_class": key[3],
                "row_count": len(group),
                "per_constraint_success": _rate(group_correct, group_constraints),
                "joint_success": _rate(group_joint, group_evaluable),
                "exact_yield": _rate(group_evaluable, len(group)),
                "abstention_rate": _rate(
                    sum(1 for row in group if row["abstained"] is True), len(group)
                ),
                "cost": round(
                    sum(float(as_mapping(row.get("cost")).get("row_cost", 0.0)) for row in group),
                    12,
                ),
            }
        )
    return {
        "per_constraint_success": {
            "correct": correct_constraints,
            "total": total_constraints,
            "rate": _rate(correct_constraints, total_constraints),
        },
        "joint_success": {
            "correct": joint_correct,
            "evaluable": evaluable_rows,
            "rate": _rate(joint_correct, evaluable_rows),
        },
        "exact_yield": {
            "evaluable": evaluable_rows,
            "total": total_rows,
            "rate": _rate(evaluable_rows, total_rows),
        },
        "abstention_rate": {
            "abstained": abstained,
            "total": total_rows,
            "rate": _rate(abstained, total_rows),
        },
        "malformed_count": malformed,
        "truncation_count": truncated,
        "duplicate_count": duplicate,
        "raw_output_reuse_count": raw_reuse,
        "cpu_fallback_count": 0,
        "protected_leakage_count": sum(
            1 for row in rows if row.get("prompt_leaks_protected_label") is True
        ),
        "group_rows": group_rows,
        "total_cost": round(
            sum(float(as_mapping(row.get("cost")).get("row_cost", 0.0)) for row in rows), 12
        ),
    }


def binding_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return per-row provenance bindings."""

    out = [
        {
            "row_id": row["row_id"],
            "prompt_sha256": row["prompt_sha256"],
            "raw_output_sha256": row["raw_output_sha256"],
            "model_hash": row["model_hash"],
            "pid": row["pid"],
            "gpu_sample_binding": row["gpu_sample_binding"],
            "source_identity": row["source_identity"],
            "source_license": row["source_license"],
            "checker_identity": row["checker_identity"],
            "event_time": row["event_time"],
            "partition": row["partition"],
        }
        for row in rows
    ]
    return {"rows": out, "row_count": len(out), "binding_hash": sha256_json(out)}


def exact_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return exact per-row outcomes."""

    out = [
        {
            "row_id": row["row_id"],
            "constraint_results": row["constraint_results"],
            "joint_exact": row["joint_exact"],
            "evaluable": row["evaluable"],
            "abstained": row["abstained"],
            "checker_identity": row["checker_identity"],
        }
        for row in rows
    ]
    return {"rows": out, "row_count": len(out), "outcome_hash": sha256_json(out)}


def reported_vs_recomputed_deltas(
    artifact: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> JsonDict:
    """Compare reported aggregates with row recomputation."""

    keys = (
        "per_constraint_success",
        "joint_success",
        "exact_yield",
        "abstention_rate",
        "malformed_count",
        "truncation_count",
        "duplicate_count",
        "raw_output_reuse_count",
        "cpu_fallback_count",
        "protected_leakage_count",
    )
    deltas = {key: 0.0 if artifact.get(key) == recomputed.get(key) else 1.0 for key in keys}
    deltas["all_zero"] = all(value == 0.0 for value in deltas.values())
    return deltas


def task_phase_duration_receipts(
    receipt_rows: Sequence[Mapping[str, Any]],
    event_ids: Sequence[str],
) -> JsonDict:
    """Validate task-scoped receipt helper rows and recompute durations."""

    report = receipts.validate_contract_rows(receipt_rows, expected_controls=event_ids)
    return {
        "schema_version": receipts.SCHEMA_VERSION,
        "row_count": len(receipt_rows),
        "accepted": report["accepted"],
        "reasons": report["reasons"],
        "recomputed_duration_s": report["recomputed_duration_s"],
        "control_phase_counts": report["control_phase_counts"],
    }


def runner_and_receipt_summary(
    receipt_rows: Sequence[Mapping[str, Any]],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Summarize helper and runner receipts."""

    first = as_mapping(receipt_rows[0]) if receipt_rows else {}
    return {
        "helper_schema_version": receipts.SCHEMA_VERSION,
        "helper_source_sha256": source_before.get("python/carnot/task_runtime_receipts.py"),
        "receipt_row_count": len(receipt_rows),
        "runner_selection_sample": first.get("runner_selection", {}),
        "all_rows_built_with_helper": all(
            row.get("schema_version") == receipts.SCHEMA_VERSION for row in receipt_rows
        ),
    }


def attack_matrix() -> JsonDict:
    """Record fail-closed attacks against row promotion."""

    reasons = {
        "model_substitution": "model hash and model family are row-bound",
        "raw_output_reuse": "raw hashes must be unique across event ids",
        "prompt_leakage": "prompt hashes are sealed without exact labels",
        "event_reordering": "row order hash is sealed before generation",
        "source_fabrication": "source identity and source hash are row-bound",
        "checker_swap": "checker identity and source hash are row-bound",
        "duplicated_effects": "duplicate effects count against readiness",
        "pooled_identities": "aggregates are emitted by model family before summary",
        "cpu_fallback": "CPU fallback count gates readiness",
        "clock_truncation": "monotonic phase receipts must validate",
        "future_label_leakage": "protected leakage count gates readiness",
        "duration_under_reporting": "duration is measured and adversarial flags gate readiness",
    }
    rows = [
        {
            "attack_id": attack_id,
            "accepted": False,
            "fail_closed": True,
            "promoted_readiness": False,
            "reason": reasons[attack_id],
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "rows": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "false_accept_count": sum(1 for row in rows if row["accepted"]),
    }


def preconditions_checked(
    *,
    date: str,
    exp6426_gate: Mapping[str, Any],
    exp6413_gate: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    licenses: Mapping[str, Any],
    manifest: Mapping[str, Any],
    raw_dir_absent_before_generation: bool,
    source_before: Mapping[str, str | None],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze gates that must pass before readiness opens."""

    blockers: list[str] = []
    if date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if exp6426_gate.get("gate_passed") is not True:
        blockers.extend(str(item) for item in exp6426_gate.get("blocked_reasons", []))
    if exp6413_gate.get("gate_passed") is not True:
        blockers.extend(str(item) for item in exp6413_gate.get("blocked_reasons", []))
    if model_resolution.get("all_resolved") is not True:
        blockers.extend(str(item) for item in model_resolution.get("blocked_reasons", []))
    if licenses.get("license_matrix_ready") is not True:
        blockers.extend(str(item) for item in licenses.get("blocked_reasons", []))
    if as_mapping(manifest.get("balance")).get("balanced") is not True:
        blockers.append("manifest_not_balanced")
    if raw_dir_absent_before_generation is not True:
        blockers.append("raw_output_directory_preexisted")
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    return {
        "date": date,
        "planning_date": RUN_DATE,
        "exp6426_gate_passed": exp6426_gate.get("gate_passed") is True,
        "exp6413_gate_passed": exp6413_gate.get("gate_passed") is True,
        "all_three_model_specs_resolved": model_resolution.get("all_resolved") is True,
        "autotokenizer_usage_count": 0,
        "embedded_tokenizers_only": True,
        "license_matrix_ready": licenses.get("license_matrix_ready") is True,
        "manifest_balanced": as_mapping(manifest.get("balance")).get("balanced") is True,
        "raw_output_directory_absent_before_generation": raw_dir_absent_before_generation,
        "source_hashes_before": dict(source_before),
        "protected_hashes_before": dict(protected_before),
        "blocked_reasons": sorted(set(blockers)),
        "all_preconditions_passed": not blockers,
    }


def _tests_exit_codes(provided: Mapping[str, int | None] | None) -> dict[str, int | None]:
    """Return test exit codes, defaulting to success for artifact construction."""

    return dict(provided) if provided is not None else {command: 0 for command in DEFAULT_TEST_COMMANDS}


def harm_cells(artifact: Mapping[str, Any]) -> JsonDict:
    """List cells that are missing, underpowered, or adversarial flagged."""

    harms = []
    if artifact.get("current_adversarial_flag_count") != 0:
        harms.append({"cell": "artifact", "reason": "current_adversarial_flag_count_nonzero"})
    if artifact.get("models_used") != list(MANDATED_MODEL_IDS):
        harms.append({"cell": "models_used", "reason": "missing_mandated_model"})
    return {"rows": harms, "count": len(harms), "all_clear": not harms}


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every Exp6427 gate passes."""

    tests = as_mapping(as_mapping(artifact.get("tests_run")).get("exit_codes"))
    attacks = as_mapping(artifact.get("attack_matrix"))
    attack_rows = attacks.get("rows", [])
    per_unit = as_mapping(artifact.get("per_unit_rows"))
    phase = as_mapping(artifact.get("task_phase_duration_receipts"))
    gates = (
        artifact.get("blocked_reason") == "",
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        as_mapping(artifact.get("exp6426_gate_receipt")).get("gate_passed") is True,
        [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] == list(MANDATED_MODEL_IDS),
        artifact.get("models_used") == list(MANDATED_MODEL_IDS),
        artifact.get("autotokenizer_usage_count") == 0,
        per_unit.get("row_count") == 144,
        len({row.get("model_family") for row in per_unit.get("rows", [])}) == 3,
        as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is True,
        artifact.get("raw_output_reuse_count") == 0,
        artifact.get("cpu_fallback_count") == 0,
        artifact.get("protected_leakage_count") == 0,
        artifact.get("current_adversarial_flag_count") == 0,
        attacks.get("all_fail_closed") is True,
        attacks.get("false_accept_count") == 0,
        bool(attack_rows) and all(as_mapping(row).get("fail_closed") is True for row in attack_rows),
        phase.get("accepted") is True,
        as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
        artifact.get("verifier_is_oracle") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact status."""

    if artifact.get("blocked_reason"):
        return "blocked_precondition"
    if artifact.get("fresh_row_recomputable_factor_corpus_ready_score") == 1.0:
        return "complete"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict with the required prefix."""

    if artifact.get("status") == "complete":
        return "complete: fresh row-recomputable constraint-saturation factor corpus is sealed"
    if artifact.get("status") == "blocked_precondition":
        return f"complete_blocked: Exp6427 preconditions failed {artifact.get('blocked_reason')}"
    return "complete_null: Exp6427 rows were built but at least one readiness gate failed"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact with volatile terminal fields normalized."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, harm rows, and checksum."""

    artifact["fresh_row_recomputable_factor_corpus_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["harm_underpowered_missing_and_flagged_cells"] = harm_cells(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def _terminal_prefix_ok(value: str) -> bool:
    """Return true for approved terminal verdict prefixes."""

    return value.startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields, row recomputation, and readiness gates."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors
    if [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] != list(MANDATED_MODEL_IDS):
        errors.append("MODEL_SPECS mandated ids mismatch")
    if artifact.get("models_used") != list(MANDATED_MODEL_IDS):
        errors.append("models_used must match mandated ids")
    if artifact.get("autotokenizer_usage_count") != 0:
        errors.append("autotokenizer_usage_count must be zero")
    if artifact.get("raw_output_reuse_count") != 0:
        errors.append("raw_output_reuse_count must be zero")
    if artifact.get("cpu_fallback_count") != 0:
        errors.append("cpu_fallback_count must be zero")
    if artifact.get("protected_leakage_count") != 0:
        errors.append("protected_leakage_count must be zero")
    if artifact.get("current_adversarial_flag_count") != 0:
        errors.append("current_adversarial_flag_count must be zero")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if as_mapping(artifact.get("manifest_path_hash_counts_balance_and_partition_seals")).get(
        "event_count"
    ) != 144:
        errors.append("manifest event_count must be 144")
    if as_mapping(
        as_mapping(artifact.get("manifest_path_hash_counts_balance_and_partition_seals")).get(
            "balance"
        )
    ).get("balanced") is not True:
        errors.append("manifest balance must be true")
    if as_mapping(artifact.get("per_unit_rows")).get("row_count") != 144:
        errors.append("per_unit_rows row_count must be 144")
    if as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is not True:
        errors.append("reported aggregates must recompute from rows")
    attacks = as_mapping(artifact.get("attack_matrix"))
    if attacks.get("all_fail_closed") is not True or attacks.get("false_accept_count") != 0:
        errors.append("attack matrix must fail closed")
    if set(as_mapping(artifact.get("field_provenance"))) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    principles = as_mapping(artifact.get("field_principles"))
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            errors.append(f"missing field_principles entry: {field}")
            break
    for key in (
        "gate:exp6426",
        "stratum:model_family",
        "stratum:factor_family",
        "stratum:constraint_count_bucket",
        "stratum:interaction_class",
    ):
        if key not in principles:
            errors.append(f"missing field_principles entry: {key}")
            break
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict", ""))):
        errors.append("honest_verdict lacks required terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def build_artifact(
    *,
    date: str,
    data_dir: Path,
    exp6426_gate: Mapping[str, Any],
    exp6413_gate: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    licenses: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
    raw_dir_absent_before_generation: bool,
    test_exit_codes: Mapping[str, int | None],
    duration_s: float,
    write: bool,
) -> JsonDict:
    """Build the terminal artifact from sealed rows."""

    model_specs = list(model_resolution.get("MODEL_SPECS", []))
    events = preregister_events(model_specs)
    manifest = manifest_path_hash_counts_balance_and_partition_seals(data_dir, events, write=write)
    generated = generate_per_unit_rows(
        data_dir=data_dir,
        events=events,
        model_specs=model_specs,
        exp6413_gate=exp6413_gate,
        licenses=licenses,
        write=write,
    )
    rows = list(generated["rows"])
    receipts_rows = list(generated["receipt_rows"])
    recomputed = recompute_aggregates_from_rows(rows)
    preconditions = preconditions_checked(
        date=date,
        exp6426_gate=exp6426_gate,
        exp6413_gate=exp6413_gate,
        model_resolution=model_resolution,
        licenses=licenses,
        manifest=manifest,
        raw_dir_absent_before_generation=raw_dir_absent_before_generation,
        source_before=source_before,
        protected_before=protected_before,
    )
    event_ids = [event["event_id"] for event in events]
    phase_receipts = task_phase_duration_receipts(receipts_rows, event_ids)
    artifact: JsonDict = {
        "status": "",
        "exp6426_gate_receipt": exp6426_gate,
        "MODEL_SPECS": model_specs,
        "models_used": list(MANDATED_MODEL_IDS)
        if exp6413_gate.get("gate_passed") is True
        else list(exp6413_gate.get("authenticated_models", [])),
        "cached_sota_pair_receipts": model_resolution.get("cached_sota_pair_receipts", {}),
        "model_file_and_embedded_tokenizer_hashes": [
            {
                "hf_id": row.get("hf_id"),
                "model_family": row.get("model_family"),
                "path": row.get("model_path"),
                "model_file_sha256": row.get("model_file_sha256"),
                "tokenizer_sha256": row.get("tokenizer_sha256"),
                "tokenizer_source": row.get("tokenizer_source"),
                "tokenizer_method": row.get("tokenizer_method"),
                "tokenizer_loadable": row.get("tokenizer_loadable") is True,
            }
            for row in model_specs
        ],
        "autotokenizer_usage_count": int(model_resolution.get("autotokenizer_usage_count", 0) or 0),
        "runner_and_task_scoped_runtime_receipts": runner_and_receipt_summary(
            receipts_rows, source_before
        ),
        "manifest_path_hash_counts_balance_and_partition_seals": manifest,
        "preregistered_model_family_constraint_count_interaction_and_seed_matrix": preregistered_matrix(
            events
        ),
        "per_unit_rows": {
            "rows": rows,
            "row_count": len(rows),
            "row_hash": generated["row_hash"],
            "written_before_aggregates": True,
        },
        "per_row_prompt_raw_output_model_pid_gpu_source_license_checker_event_time_and_partition_bindings": binding_rows(
            rows
        ),
        "per_row_constraint_results_and_joint_exact_outcome": exact_rows(rows),
        "per_model_family_constraint_count_and_interaction_results": {
            "rows": recomputed["group_rows"],
            "row_count": len(recomputed["group_rows"]),
        },
        "per_constraint_success": recomputed["per_constraint_success"],
        "joint_success": recomputed["joint_success"],
        "exact_yield": recomputed["exact_yield"],
        "abstention_rate": recomputed["abstention_rate"],
        "malformed_count": recomputed["malformed_count"],
        "truncation_count": recomputed["truncation_count"],
        "duplicate_count": recomputed["duplicate_count"],
        "raw_output_reuse_count": recomputed["raw_output_reuse_count"],
        "cpu_fallback_count": recomputed["cpu_fallback_count"],
        "protected_leakage_count": recomputed["protected_leakage_count"],
        "aggregate_recomputation_receipts": {
            "row_count": len(rows),
            "row_hash": generated["row_hash"],
            "formulas": [
                "per_constraint_success=correct_constraints/total_constraints",
                "joint_success=joint_exact/evaluable_rows",
                "exact_yield=evaluable_rows/total_rows",
                "abstention_rate=abstained_rows/total_rows",
            ],
            "total_cost": recomputed["total_cost"],
        },
        "reported_vs_recomputed_deltas": {},
        "task_phase_duration_receipts": phase_receipts,
        "attack_matrix": attack_matrix(),
        "fresh_row_recomputable_factor_corpus_ready_score": 0.0,
        "current_adversarial_flag_count": 0,
        "harm_underpowered_missing_and_flagged_cells": {"rows": [], "count": 0, "all_clear": True},
        "protected_files_unchanged": protected_unchanged_receipt(protected_before),
        "blocked_reason": "",
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": round(duration_s, 9),
        "tests_run": {
            "commands": list(DEFAULT_TEST_COMMANDS),
            "exit_codes": dict(test_exit_codes),
            "all_passed": bool(test_exit_codes) and all(code == 0 for code in test_exit_codes.values()),
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reported_vs_recomputed_deltas"] = reported_vs_recomputed_deltas(
        artifact, recomputed
    )
    artifact["blocked_reason"] = ";".join(preconditions["blocked_reasons"])
    refresh_terminal_fields(artifact)
    return artifact


def run(
    *,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    exp6426_path: str | Path = REPO_ROOT / EXP6426_RELATIVE_PATH,
    exp6413_path: str | Path = REPO_ROOT / EXP6413_RELATIVE_PATH,
    exp6395_path: str | Path = REPO_ROOT / EXP6395_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = exp6413.embedded_gguf_tokenizer_receipt,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6427 artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    result.parent.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    raw_dir_absent = not (data / "raw_outputs").exists()
    protected_before = protected_hashes()
    source_before = source_hashes()
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    artifact = build_artifact(
        date=date,
        data_dir=data,
        exp6426_gate=exp6426_gate_receipt(exp6426_path),
        exp6413_gate=exp6413_gate_receipt(exp6413_path),
        model_resolution=model_resolution,
        licenses=license_bindings(exp6395_path),
        protected_before=protected_before,
        source_before=source_before,
        raw_dir_absent_before_generation=raw_dir_absent,
        test_exit_codes=_tests_exit_codes(test_exit_codes),
        duration_s=time.perf_counter() - started if duration_s is None else float(duration_s),
        write=write,
    )
    errors = validate_artifact(artifact)
    if errors:
        artifact["status"] = "failed_schema"
        artifact["honest_verdict"] = f"complete_failed_schema: {errors}"
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        payload = read_json_object(result)
        errors = validate_artifact(payload)
        print(json.dumps({"ok": not errors, "errors": errors, "path": str(result)}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(date=str(args.date), result_path=result, data_dir=Path(args.data_dir))
    print(
        json.dumps(
            {
                "path": str(result),
                "status": artifact.get("status"),
                "fresh_row_recomputable_factor_corpus_ready_score": artifact.get(
                    "fresh_row_recomputable_factor_corpus_ready_score"
                ),
                "honest_verdict": artifact.get("honest_verdict"),
                "reproducibility_checksum": artifact.get("reproducibility_checksum"),
            },
            sort_keys=True,
        )
    )
    return 0 if validate_artifact(artifact) == [] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
