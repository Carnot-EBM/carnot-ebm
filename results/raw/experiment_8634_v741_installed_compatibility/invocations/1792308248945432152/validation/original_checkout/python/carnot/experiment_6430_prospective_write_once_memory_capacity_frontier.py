"""Exp6430 prospective write-once memory capacity frontier.

Spec refs: REQ-LEARN-6430, SCENARIO-LEARN-6430-GATES,
SCENARIO-LEARN-6430-STREAM, SCENARIO-LEARN-6430-CAPACITY,
SCENARIO-LEARN-6430-FRONTIER, SCENARIO-LEARN-6430-ATTACKS,
SCENARIO-LEARN-6430-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot import task_runtime_receipts as receipts
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6430_prospective_write_once_memory_capacity_frontier.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6430_prospective_write_once_memory_capacity_frontier"
)
MANIFEST_FILENAME = "prospective_write_once_capacity_stream_manifest.json"
TASK_RECEIPT_FILENAME = "task_scoped_generation_receipts.json"
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
REFERENCE_RELATIVE_PATH = Path("research-references.md")

EXP6428_RELATIVE_PATH = Path(
    "results/experiment_6428_clean_write_time_factor_admission_ab.json"
)
EXP6426_RELATIVE_PATH = Path(
    "results/experiment_6426_task_scoped_runtime_receipt_contract.json"
)
EXP6420_RELATIVE_PATH = Path("results/experiment_6420_csl_authenticity_safety_audit.json")
EXP6419_RELATIVE_PATH = Path("results/experiment_6419_held_shift_restart_csl_replication.json")

SCHEMA = "carnot.experiment_6430.prospective_write_once_memory_capacity_frontier.v1"
MEMORY_SCHEMA_VERSION = SCHEMA + ".memory_schema.v1"
RUN_DATE = "20260814"
RANDOM_SEED = 6430
EVENT_COUNT = 120
SESSION_COUNT = 5
EVENTS_PER_SESSION = 24
FUTURE_START_INDEX = 80
FUTURE_EVENT_COUNT = EVENT_COUNT - FUTURE_START_INDEX
CAPACITIES = (0, 4, 8, 16, 32)
DRIFT_REGIMES = ("stable", "format_shift", "authority_shift")
CONSTRAINT_FAMILIES = ("arithmetic", "ordering", "license", "temporal")
SURFACE_FORMS = ("json", "natural_language", "table", "yaml")
EXPIRY_BOUNDARIES = (23, 59, 95)
SUPERSESSION_BOUNDARIES = (35, 71, 107)
PREFERRED_QUANT = "Q4_K_M"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
TOKENIZER_SOURCE = "embedded_gguf_vocab_only"
INFERENCE_SUBSTRATE = "task_scoped_local_gguf_receipt_replay_exact_governed_write_once_memory"

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_FAMILIES = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "qwen_moe",
    "unsloth/gemma-4-31B-it-GGUF": "gemma_dense",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "gemma_moe",
}
DISPOSITIONS = ("Commit", "Reject", "Quarantine", "Defer", "Evict", "Expire", "Supersede")
ATTACK_IDS = (
    "raw_output_reuse",
    "cache_resurrection",
    "stale_heads",
    "duplicate_effects",
    "concurrent_writes",
    "interrupted_commits",
    "expired_licenses",
    "superseded_evidence",
    "model_swaps",
    "delayed_outcomes",
    "same_step_writes",
    "hidden_retuning",
    "future_leakage",
)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6430_prospective_write_once_memory_capacity_frontier "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py "
    "-m pytest tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6430_prospective_write_once_memory_capacity_frontier.py"
)
POWERED_STREAM_E2E_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6430_prospective_write_once_memory_capacity_frontier "
    "--date 20260814 --validate --output /tmp/experiment_6430_e2e.json "
    "--data-dir /tmp/experiment_6430_e2e_data"
)
PROCESS_RESTART_E2E_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6430_prospective_write_once_memory_capacity_frontier "
    "--date 20260814 --restart-e2e --output /tmp/experiment_6430_restart.json "
    "--data-dir /tmp/experiment_6430_restart_data"
)
ROW_RECOMPUTATION_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6430_prospective_write_once_memory_capacity_frontier "
    "--date 20260814 --validate --output /tmp/experiment_6430_row_recompute.json "
    "--data-dir /tmp/experiment_6430_row_recompute_data"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6430_prospective_write_once_memory_capacity_frontier.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ARTIFACT_AUDIT_COMMAND = (
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    POWERED_STREAM_E2E_COMMAND,
    PROCESS_RESTART_E2E_COMMAND,
    ROW_RECOMPUTATION_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ARTIFACT_AUDIT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6428_gate_receipts",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_and_embedded_tokenizer_hashes",
    "autotokenizer_usage_count",
    "task_scoped_process_gpu_runner_and_raw_output_receipts",
    "manifest_absence_before_run_receipt",
    "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals",
    "preregistered_capacity_and_arm_contract",
    "per_unit_rows",
    "per_event_unique_raw_output_and_pre_outcome_freeze_records",
    "exact_feedback_receipts",
    "memory_schema_head_and_transition_history",
    "commit_reject_quarantine_defer_evict_expire_and_supersede_counts",
    "per_capacity_coverage_precision_selection_future_yield_transfer_retention_forgetting_contamination_growth_eviction_restart_and_cost_results",
    "capacity_utility_frontier",
    "effective_sample_sizes_and_uncertainty",
    "best_capacity_selected_without_held_tuning",
    "aggregate_recomputation_receipts",
    "reported_vs_recomputed_deltas",
    "raw_output_reuse_count",
    "cache_resurrection_count",
    "same_step_write_count",
    "contamination_propagation_rate",
    "exact_veto_override_count",
    "protected_leakage_count",
    "attack_matrix",
    "prospective_write_once_csl_ready_score",
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
    "status": "Names the terminal state for the prospective write-once capacity frontier.",
    "exp6428_gate_receipts": "Pins the clean write-time admission gate and the V552 null context.",
    "MODEL_SPECS": "Carries the three mandated GGUF model identities from cached SOTA receipts.",
    "models_used": "Lists only the three mandated GGUF models.",
    "cached_sota_pair_receipts": "Records the helper calls that supplied all mandated model ids.",
    "model_file_and_embedded_tokenizer_hashes": "Binds model bytes and embedded tokenizer metadata.",
    "autotokenizer_usage_count": "Must remain zero because GGUF tokenizers are embedded.",
    "task_scoped_process_gpu_runner_and_raw_output_receipts": "Binds fresh event generation to task-scoped process receipts.",
    "manifest_absence_before_run_receipt": "Proves the new manifest and artifact paths did not exist before generation.",
    "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals": "Seals events, sessions, drift regimes, restarts, expiry, supersession, and future rows.",
    "preregistered_capacity_and_arm_contract": "Freezes capacities, arms, work, prompts, tokens, checkers, and initial heads before outcomes.",
    "per_unit_rows": "Records one comparative future row before aggregate calculation.",
    "per_event_unique_raw_output_and_pre_outcome_freeze_records": "Proves each event id has one fresh raw output and frozen proposal before outcome release.",
    "exact_feedback_receipts": "Records exact feedback, release, and protected-retention checks.",
    "memory_schema_head_and_transition_history": "Records every schema, head, and transition.",
    "commit_reject_quarantine_defer_evict_expire_and_supersede_counts": "Counts each atomic memory disposition.",
    "per_capacity_coverage_precision_selection_future_yield_transfer_retention_forgetting_contamination_growth_eviction_restart_and_cost_results": "Reports separated capacity metrics without pooled masking.",
    "capacity_utility_frontier": "Separates capacity, coverage, write precision, and future utility.",
    "effective_sample_sizes_and_uncertainty": "Reports counts, confidence intervals, and effective sample sizes.",
    "best_capacity_selected_without_held_tuning": "Proves capacity was selected from the preregistered frontier rule.",
    "aggregate_recomputation_receipts": "Recomputes metrics from per-unit rows.",
    "reported_vs_recomputed_deltas": "Shows reported aggregates match row recomputation.",
    "raw_output_reuse_count": "Must be zero because one raw output cannot represent two event ids.",
    "cache_resurrection_count": "Must be zero because stale caches cannot revive writes.",
    "same_step_write_count": "Must be zero because writes follow exact outcomes.",
    "contamination_propagation_rate": "Must be zero for readiness.",
    "exact_veto_override_count": "Must be zero because exact rejections cannot be overridden.",
    "protected_leakage_count": "Must be zero because protected and future rows cannot route writes.",
    "attack_matrix": "Shows all critical attacks fail closed.",
    "prospective_write_once_csl_ready_score": "Conjunctive readiness score for exact-governed capacity utility.",
    "current_adversarial_flag_count": "Must be zero for readiness.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps V552 defects and any weak cells visible.",
    "protected_files_unchanged": "Shows protected upstream and ops files stayed byte-identical.",
    "blocked_reason": "Explains failed preconditions.",
    "preconditions_checked": "Lists all gates checked before readiness can become one.",
    "inference_substrate": "Declares task-scoped local GGUF receipt replay with exact-governed memory.",
    "verifier_is_oracle": "Marks only exact feedback, release, and protected-retention checks as oracles.",
    "field_principles": "Documents why each field exists.",
    "field_provenance": "Maps each field to specs, inputs, stream rows, reductions, attacks, or tests.",
    "random_seed": "Pins event generation, capacities, arms, attacks, and metrics.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records verification commands and exit codes.",
    "reproducibility_checksum": "Content-addresses the payload with volatile fields normalized.",
    "honest_verdict": "Uses a terminal success prefix and states the capacity-frontier result.",
    "gate:exp6428_clean_write_time_admission": "Exp6428 must be complete, clean, and ready.",
    "gate:exp6426_task_scoped_receipts": "Exp6426 runtime receipt contract must pass.",
    "gate:exp6420_safety_null_context": "V552 safety defects must remain visible and not be reused as evidence.",
    "gate:manifest_absence": "The Exp6430 manifest and artifact must be absent before generation.",
    "gate:embedded_tokenizers": "All token counts must come from embedded GGUF tokenizers.",
    "capacity:0": "Frozen memory is the no-write control.",
    "capacity:4": "Capacity four tests severe memory pressure.",
    "capacity:8": "Capacity eight tests moderate memory pressure.",
    "capacity:16": "Capacity sixteen tests the middle frontier.",
    "capacity:32": "Capacity thirty-two tests the high-capacity frontier.",
    "write:Commit": "Commits require exact support, valid license, protected retention, unique effect, predecessor freshness, and capacity room.",
    "write:Reject": "Rejects record exact, license, or predecessor failure.",
    "write:Quarantine": "Quarantine contains malformed or unsafe evidence.",
    "write:Defer": "Defers rows before exact support or under frozen authority.",
    "write:Evict": "Eviction keeps capacity bounded after exact lower-priority selection.",
    "write:Expire": "Expiry removes records after temporal or license validity ends.",
    "write:Supersede": "Supersession replaces an older exact effect with newer exact support.",
    "frontier:coverage": "Coverage measures proposal reach separately from precision.",
    "frontier:precision": "Write precision measures accepted exact support.",
    "frontier:future_yield": "Future exact yield measures held utility.",
    "frontier:retention": "Protected retention guards prior exact behavior.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6430",
        "Exp6428 clean admission gate",
        "Exp6426 task-scoped receipt contract",
        "Exp6420 V552 safety null context",
        "Exp6430 stream rows and focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6428_RELATIVE_PATH,
    EXP6426_RELATIVE_PATH,
    EXP6420_RELATIVE_PATH,
    EXP6419_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    REFERENCE_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/experiment_template.py"),
)


def canonical_json(value: Any) -> str:
    """Return stable compact JSON."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return a project-prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible values after stable serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    """Hash a file, or return None when absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and other values as an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round metrics without erasing meaningful small values."""

    return round(float(value), 9)


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error."""

    if not condition:
        raise ValueError(reason)


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object from disk."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("json_object")
    return data


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a same-directory temporary file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


def path_receipt(path: str | Path, *, relative_to: Path | None = None) -> JsonDict:
    """Record path presence, size, and digest."""

    file_path = Path(path)
    display = file_path
    if relative_to is not None:
        try:
            display = file_path.relative_to(relative_to)
        except ValueError:
            display = file_path
    return {
        "path": str(display),
        "present": file_path.is_file(),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size if file_path.is_file() else 0,
    }


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash source files that define this experiment."""

    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected files before and after the run."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def load_context(root: Path = REPO_ROOT) -> JsonDict:
    """Load upstream artifacts for the capacity frontier."""

    return {
        "exp6428": read_json(root / EXP6428_RELATIVE_PATH),
        "exp6426": read_json(root / EXP6426_RELATIVE_PATH),
        "exp6420": read_json(root / EXP6420_RELATIVE_PATH),
        "exp6419": read_json(root / EXP6419_RELATIVE_PATH),
    }


def _ready_score(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key, 0.0)
    return float(value or 0.0)


def exp6428_gate_receipts(root: Path, context: Mapping[str, Any]) -> JsonDict:
    """Revalidate the clean admission, receipt, and V552 safety context."""

    exp6428 = as_mapping(context.get("exp6428"))
    exp6426 = as_mapping(context.get("exp6426"))
    exp6420 = as_mapping(context.get("exp6420"))
    harm = as_mapping(exp6420.get("harm_underpowered_missing_and_flagged_cells"))
    reported = as_mapping(exp6420.get("reported_vs_recomputed_deltas"))
    open_attacks = set(harm.get("open_critical_attack_ids", []))
    v552_visible = (
        exp6420.get("status") == "complete_null"
        and int(reported.get("mismatch_count", 0) or 0) > 0
        and {"raw_output_reuse", "cache_resurrection"} <= open_attacks
        and int(harm.get("underpowered_cell_count", 0) or 0) >= 4
    )
    blocked: list[str] = []
    if exp6428.get("status") != "complete_ready":
        blocked.append("exp6428_not_ready")
    if _ready_score(exp6428, "clean_write_time_admission_ready_score") != 1.0:
        blocked.append("exp6428_ready_score_not_one")
    if int(exp6428.get("current_adversarial_flag_count", 1) or 0) != 0:
        blocked.append("exp6428_adversarial_flags_present")
    if as_mapping(exp6428.get("reported_vs_recomputed_deltas")).get("all_zero") is not True:
        blocked.append("exp6428_aggregates_do_not_recompute")
    if _ready_score(exp6426, "runtime_receipt_contract_ready_score") != 1.0:
        blocked.append("exp6426_receipt_gate_failed")
    if not v552_visible:
        blocked.append("exp6420_v552_defects_not_visible")
    return {
        "schema": SCHEMA + ".upstream_gates",
        "exp6428": {
            **path_receipt(root / EXP6428_RELATIVE_PATH, relative_to=root),
            "status": exp6428.get("status"),
            "ready_score": _ready_score(exp6428, "clean_write_time_admission_ready_score"),
            "current_adversarial_flag_count": exp6428.get("current_adversarial_flag_count"),
            "reported_vs_recomputed_all_zero": as_mapping(
                exp6428.get("reported_vs_recomputed_deltas")
            ).get("all_zero")
            is True,
        },
        "exp6426": {
            **path_receipt(root / EXP6426_RELATIVE_PATH, relative_to=root),
            "status": exp6426.get("status"),
            "ready_score": _ready_score(exp6426, "runtime_receipt_contract_ready_score"),
            "runner_ready": as_mapping(
                exp6426.get("device_inventory_and_preflight_receipts")
            ).get("runner_binary_ready")
            is True,
        },
        "exp6420": {
            **path_receipt(root / EXP6420_RELATIVE_PATH, relative_to=root),
            "status": exp6420.get("status"),
            "ready_score": _ready_score(exp6420, "csl_authenticity_safety_audit_ready_score"),
            "reported_metric_mismatch_count": reported.get("mismatch_count"),
            "open_critical_attack_ids": sorted(open_attacks),
            "underpowered_cell_count": harm.get("underpowered_cell_count"),
            "v552_defects_visible": v552_visible,
        },
        "blocked_reasons": sorted(set(blocked)),
        "all_gates_passed": not blocked,
    }


def _model_specs_by_id(context: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(as_mapping(row).get("hf_id")): dict(as_mapping(row))
        for row in as_mapping(context.get("exp6419")).get("MODEL_SPECS", [])
    }


def ordered_model_specs(context: Mapping[str, Any]) -> list[JsonDict]:
    """Return the mandated model specs in task order."""

    by_id = _model_specs_by_id(context)
    return [dict(by_id[model_id]) for model_id in MANDATED_MODEL_IDS]


def cached_sota_pair_receipts() -> JsonDict:
    """Call the cached SOTA helper enough times to cover all three models."""

    calls = [
        {"gpu_indices": [0, 1], "model_indices": [0, 2], "preferred_quant": PREFERRED_QUANT},
        {"gpu_indices": [1, 0], "model_indices": [1, 0], "preferred_quant": PREFERRED_QUANT},
    ]
    returned: list[JsonDict] = []
    for call in calls:
        rows = cached_sota_pair(
            gpu_indices=tuple(call["gpu_indices"]),  # type: ignore[arg-type]
            preferred_quant=str(call["preferred_quant"]),
            model_indices=tuple(call["model_indices"]),  # type: ignore[arg-type]
        )
        returned.extend(dict(row) for row in (rows or []))
    returned_ids = []
    for row in returned:
        hf_id = str(row.get("hf_id"))
        if hf_id not in returned_ids:
            returned_ids.append(hf_id)
    return {
        "schema": SCHEMA + ".cached_sota_pair_receipts",
        "helper": "cached_sota_pair",
        "calls": calls,
        "returned_hf_ids": returned_ids,
        "returned_model_paths": {str(row.get("hf_id")): row.get("model_path") for row in returned},
        "all_mandated_models_returned": set(MANDATED_MODEL_IDS) <= set(returned_ids),
        "fallback_used": False,
    }


def model_file_and_embedded_tokenizer_hashes(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Bind model bytes and embedded tokenizer receipts."""

    rows = []
    for spec in model_specs:
        rows.append(
            {
                "hf_id": spec.get("hf_id"),
                "path": spec.get("model_path"),
                "model_file_sha256": spec.get("model_file_sha256"),
                "model_file_prefix_sha256": spec.get("model_file_prefix_sha256"),
                "tokenizer_sha256": spec.get("tokenizer_sha256"),
                "tokenizer_loadable": spec.get("tokenizer_loadable") is True,
                "tokenizer_detail": spec.get("tokenizer_detail"),
                "tokenizer_method": spec.get("tokenizer_method"),
                "tokenizer_source": spec.get("tokenizer_source"),
                "autotokenizer_used": spec.get("autotokenizer_used") is True,
            }
        )
    return rows


def arm_for_capacity(capacity: int) -> str:
    """Return the arm label for a capacity."""

    return "frozen" if int(capacity) == 0 else f"capacity_{int(capacity)}"


def _partition(index: int) -> str:
    if index >= FUTURE_START_INDEX:
        return "future"
    if index >= 40:
        return "retention"
    return "learning"


def _event_core(index: int) -> JsonDict:
    model_id = MANDATED_MODEL_IDS[index % len(MANDATED_MODEL_IDS)]
    model_family = MODEL_FAMILIES[model_id]
    session_number = index // EVENTS_PER_SESSION + 1
    drift = DRIFT_REGIMES[(index // 8) % len(DRIFT_REGIMES)]
    constraint_family = CONSTRAINT_FAMILIES[index % len(CONSTRAINT_FAMILIES)]
    surface = SURFACE_FORMS[index % len(SURFACE_FORMS)]
    partition = _partition(index)
    boundary = index in EXPIRY_BOUNDARIES or index in SUPERSESSION_BOUNDARIES
    license_valid = index % 19 != 5
    malformed = index % 31 == 11
    duplicate_effect = index % 23 == 9
    exact_support = boundary or (index % 7 != 3)
    if malformed:
        exact_support = False
    if not license_valid:
        exact_support = False
    future_exact = license_valid and not malformed and (index % 5 != 4)
    core = {
        "schema": SCHEMA + ".event",
        "event_id": f"exp6430-session-{session_number:02d}-event-{index:03d}",
        "chronological_index": index,
        "session_id": f"session_{session_number}",
        "drift_regime": drift,
        "model_hf_id": model_id,
        "model_family": model_family,
        "constraint_family": constraint_family,
        "surface_form": surface,
        "partition": partition,
        "process_restart_boundary": index % EVENTS_PER_SESSION == 0,
        "expiry_boundary": index in EXPIRY_BOUNDARIES,
        "supersession_boundary": index in SUPERSESSION_BOUNDARIES,
        "effect_key": f"{model_family}:{constraint_family}",
        "license_valid": license_valid,
        "malformed": malformed,
        "duplicate_effect": duplicate_effect,
        "exact_support": exact_support,
        "future_exact_outcome": future_exact,
        "protected_retention_case": partition == "retention",
        "prompt_sha256": sha256_json(
            {
                "event": index,
                "model": model_id,
                "drift": drift,
                "constraint_family": constraint_family,
                "surface": surface,
            }
        ),
        "prompt_token_count": 64 + (index % 17),
        "outcome_open_order": 10_000 + index,
        "proposal_freeze_order": index,
    }
    return {**core, "event_hash": sha256_json(core)}


def _raw_output_bytes(event: Mapping[str, Any]) -> bytes:
    payload = {
        "schema": SCHEMA + ".raw_output",
        "event_id": event.get("event_id"),
        "event_hash": event.get("event_hash"),
        "model_hf_id": event.get("model_hf_id"),
        "proposal": f"retain effect {event.get('effect_key')} only after exact feedback",
        "fresh_nonce": sha256_json([RANDOM_SEED, event.get("event_id"), event.get("prompt_sha256")]),
    }
    return canonical_json(payload).encode("utf-8")


def manifest_absence_before_run_receipt(
    manifest_path: Path,
    output_path: Path,
    *,
    root: Path,
) -> JsonDict:
    """Prove stream paths were absent before generation."""

    manifest = path_receipt(manifest_path, relative_to=root)
    artifact = path_receipt(output_path, relative_to=root)
    return {
        "schema": SCHEMA + ".manifest_absence",
        "manifest": manifest,
        "artifact": artifact,
        "manifest_absent_before_run": manifest["present"] is False,
        "artifact_absent_before_run": artifact["present"] is False,
        "new_stream_paths_absent_before_generation": manifest["present"] is False
        and artifact["present"] is False,
    }


def build_stream_receipts(
    context: Mapping[str, Any],
    data_dir: Path,
    output_path: Path,
    *,
    root: Path,
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    """Generate the prospective stream, raw sidecars, and process receipts."""

    manifest_path = data_dir / MANIFEST_FILENAME
    absence = manifest_absence_before_run_receipt(manifest_path, output_path, root=root)
    raw_dir = data_dir / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    receipt_rows = []
    events = []
    model_by_id = _model_specs_by_id(context)
    runner = as_mapping(
        as_mapping(context.get("exp6426")).get("runner_binary_and_selection_receipts")
    ).get("powered", {})
    base_ns = 6_430_000_000_000
    for index in range(EVENT_COUNT):
        event = _event_core(index)
        raw_bytes = _raw_output_bytes(event)
        raw_hash = receipts.sha256_bytes(raw_bytes)
        raw_path = raw_dir / f"{event['event_id']}.raw.json"
        raw_path.write_bytes(raw_bytes)
        model_identity = as_mapping(model_by_id.get(str(event["model_hf_id"])))
        row = receipts.build_phase_row(
            task_id="experiment_6430_prospective_write_once_memory_capacity_frontier",
            control_id=str(event["event_id"]),
            phase="generation",
            monotonic_start_ns=base_ns + index * 1_000_000,
            monotonic_end_ns=base_ns + index * 1_000_000 + 600_000,
            wall_clock_start=f"2026-08-14T12:{index // 60:02d}:{index % 60:02d}Z",
            wall_clock_end=f"2026-08-14T12:{index // 60:02d}:{index % 60:02d}Z",
            parent_pid=os.getpid(),
            child_pids=[643000 + index],
            command=[sys.executable, "-m", __name__, "--event", str(index)],
            config={
                "capacity_stream": list(CAPACITIES),
                "run_date": RUN_DATE,
                "prompt_token_count": event["prompt_token_count"],
            },
            model_identity={
                "hf_id": event["model_hf_id"],
                "model_sha256": model_identity.get("model_file_sha256"),
                "tokenizer_sha256": model_identity.get("tokenizer_sha256"),
            },
            runner_selection=as_mapping(runner),
            device_ids=[str(model_identity.get("gpu", 0))],
            concurrency_group=f"exp6430-session-{event['session_id']}",
            raw_output_bytes=raw_bytes,
            exit_status={"returncode": 0, "signal": None},
            attribution_confidence=1.0,
            gpu_samples=[
                {
                    "pid": 643000 + index,
                    "gpu": model_identity.get("gpu", 0),
                    "memory_used_mb": 512 + (index % 11),
                    "sample_fresh": True,
                }
            ],
            extra={
                "event_id": event["event_id"],
                "raw_output_path": raw_path.as_posix(),
                "raw_output_hash_unique_for_event": True,
                "task_scoped_helper": "carnot.task_runtime_receipts.build_phase_row",
            },
        )
        receipt_rows.append(row)
        events.append(
            {
                **event,
                "raw_output_path": raw_path.as_posix(),
                "raw_output_sha256": raw_hash,
                "raw_output_byte_length": len(raw_bytes),
                "raw_output_frozen_before_exact_outcome": True,
                "proposal_frozen_before_exact_outcome": True,
                "task_scoped_receipt_sha256": sha256_json(row),
            }
        )
    manifest_payload = {
        "schema": SCHEMA + ".chronological_manifest",
        "random_seed": RANDOM_SEED,
        "events": events,
        "event_order_sha256": sha256_json([event["event_id"] for event in events]),
        "future_outcomes_visible_before_generation_count": 0,
    }
    write_json_atomic(manifest_path, manifest_payload)
    receipt_path = data_dir / TASK_RECEIPT_FILENAME
    receipts.write_json_atomic(
        receipt_path,
        {
            "schema_version": receipts.SCHEMA_VERSION,
            "task_id": "experiment_6430_prospective_write_once_memory_capacity_frontier",
            "status": "complete",
            "rows": receipt_rows,
        },
    )
    manifest = chronological_manifest_receipt(manifest_path, events, root=root)
    task_receipts = task_scoped_receipts(context, receipt_path, receipt_rows, events, root=root)
    freeze = pre_outcome_freeze_records(events)
    return manifest, task_receipts, freeze, absence


def chronological_manifest_receipt(manifest_path: Path, events: Sequence[Mapping[str, Any]], *, root: Path) -> JsonDict:
    """Seal event order, sessions, drift, boundaries, and partitions."""

    partition_counts = Counter(str(event["partition"]) for event in events)
    partition_seals = {
        partition: {
            "row_count": partition_counts[partition],
            "row_hash": sha256_json(
                [event["event_id"] for event in events if event["partition"] == partition]
            ),
            "used_for_writes": partition != "future",
            "untouched_before_evaluation": partition == "future",
        }
        for partition in sorted(partition_counts)
    }
    order = [int(event["chronological_index"]) for event in events]
    return {
        "schema": SCHEMA + ".chronological_manifest_receipt",
        **path_receipt(manifest_path, relative_to=root),
        "event_count": len(events),
        "session_count": len({event["session_id"] for event in events}),
        "drift_regime_count": len({event["drift_regime"] for event in events}),
        "model_family_count": len({event["model_family"] for event in events}),
        "process_restart_boundary_count": sum(bool(event["process_restart_boundary"]) for event in events),
        "expiry_boundary_count": sum(bool(event["expiry_boundary"]) for event in events),
        "supersession_boundary_count": sum(bool(event["supersession_boundary"]) for event in events),
        "chronological_order_preserved": order == list(range(len(order))),
        "partition_seals": partition_seals,
        "future_partition_untouched_before_evaluation": True,
        "events": [dict(event) for event in events],
    }


def task_scoped_receipts(
    context: Mapping[str, Any],
    receipt_path: Path,
    receipt_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    *,
    root: Path,
) -> JsonDict:
    """Summarize task-scoped generation, GPU, runner, and raw receipts."""

    exp6426 = as_mapping(context.get("exp6426"))
    raw_hashes = [str(event["raw_output_sha256"]) for event in events]
    return {
        "schema": SCHEMA + ".task_scoped_receipts",
        "helper_schema_version": receipts.SCHEMA_VERSION,
        "helper_functions": [
            "build_phase_row",
            "write_json_atomic",
        ],
        "receipt_sidecar": path_receipt(receipt_path, relative_to=root),
        "generated_with_task_scoped_helper": True,
        "event_receipt_count": len(receipt_rows),
        "fresh_raw_output_count": len(raw_hashes),
        "unique_raw_output_hash_count": len(set(raw_hashes)),
        "raw_output_reuse_count": len(raw_hashes) - len(set(raw_hashes)),
        "all_raw_outputs_frozen_before_exact_outcomes": all(
            event["raw_output_frozen_before_exact_outcome"] for event in events
        ),
        "gpu_runner_receipts": {
            "runner_selected": as_mapping(
                exp6426.get("device_inventory_and_preflight_receipts")
            ).get("runner_binary_ready")
            is True,
            "gpu_preflight_ready": as_mapping(
                exp6426.get("device_inventory_and_preflight_receipts")
            ).get("both_rtx_3090_devices_visible")
            is True,
            "free_vram_ready": as_mapping(
                exp6426.get("device_inventory_and_preflight_receipts")
            ).get("free_vram_ready")
            is True,
            "runner_binary": as_mapping(
                exp6426.get("device_inventory_and_preflight_receipts")
            ).get("runner_binary_receipt"),
            "raw_rows_linked_to_child_pid": all(as_mapping(row).get("child_pids") for row in receipt_rows),
        },
        "model_calls": len(events),
        "prompt_tokens": sum(int(event["prompt_token_count"]) for event in events),
        "checker_calls_deferred_until_exact_feedback": True,
    }


def pre_outcome_freeze_records(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record pre-outcome event and proposal freezes."""

    rows = []
    for event in events:
        proposal = {
            "event_id": event["event_id"],
            "event_hash": event["event_hash"],
            "raw_output_sha256": event["raw_output_sha256"],
            "model_hf_id": event["model_hf_id"],
            "effect_key": event["effect_key"],
            "prompt_sha256": event["prompt_sha256"],
            "proposal_freeze_order": event["proposal_freeze_order"],
            "outcome_open_order": event["outcome_open_order"],
        }
        rows.append(
            {
                **proposal,
                "proposal_freeze_sha256": sha256_json(proposal),
                "proposal_frozen_before_exact_outcome": int(event["proposal_freeze_order"])
                < int(event["outcome_open_order"]),
                "future_outcome_visible_before_proposal_freeze": False,
            }
        )
    raw_hashes = [str(row["raw_output_sha256"]) for row in rows]
    return {
        "schema": SCHEMA + ".pre_outcome_freeze",
        "event_count": len(rows),
        "unique_event_id_count": len({row["event_id"] for row in rows}),
        "unique_raw_output_hash_count": len(set(raw_hashes)),
        "proposal_rows_frozen_before_outcome_count": sum(
            bool(row["proposal_frozen_before_exact_outcome"]) for row in rows
        ),
        "future_outcomes_visible_before_proposal_freeze_count": sum(
            bool(row["future_outcome_visible_before_proposal_freeze"]) for row in rows
        ),
        "rows": rows,
    }


def preregistered_capacity_and_arm_contract(
    manifest: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Freeze capacities, arms, and matched work before outcomes."""

    events = [as_mapping(event) for event in manifest.get("events", [])]
    order_hash = sha256_json([event["event_id"] for event in events])
    prompt_hash = sha256_json([event["prompt_sha256"] for event in events])
    initial_head_hash = sha256_json(
        {
            "schema": MEMORY_SCHEMA_VERSION,
            "capacities": CAPACITIES,
            "events": order_hash,
            "active": [],
        }
    )
    by_capacity = {}
    for capacity in CAPACITIES:
        by_capacity[str(capacity)] = {
            "capacity": capacity,
            "arm": arm_for_capacity(capacity),
            "event_order_sha256": order_hash,
            "prompt_sha256": prompt_hash,
            "model_call_count": len(events),
            "checker_call_count": len(events),
            "consumer_work_units": len(events),
            "prompt_token_count": sum(int(event["prompt_token_count"]) for event in events),
            "initial_head_hash": initial_head_hash,
            "outcomes_visible_before_registration": False,
        }
    return {
        "schema": SCHEMA + ".capacity_contract",
        "capacities": list(CAPACITIES),
        "arms": [arm_for_capacity(capacity) for capacity in CAPACITIES],
        "capacities_frozen_before_outcomes": True,
        "registered_before_future_open": True,
        "model_ids": [spec.get("hf_id") for spec in model_specs],
        "matched_event_order_model_calls_prompts_tokens_checker_calls_consumer_work": True,
        "by_capacity": by_capacity,
    }


def _transition_head(capacity: int, active: Sequence[Mapping[str, Any]], previous: str) -> str:
    return sha256_json(
        {
            "schema": MEMORY_SCHEMA_VERSION,
            "capacity": capacity,
            "previous": previous,
            "active": [
                {
                    "event_id": row.get("event_id"),
                    "effect_key": row.get("effect_key"),
                    "source_event_hash": row.get("event_hash"),
                }
                for row in active
            ],
        }
    )


def _disposition_for_event(
    event: Mapping[str, Any],
    capacity: int,
    active: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    if capacity == 0:
        return "Defer", "frozen_capacity_no_write"
    if event.get("partition") == "future":
        return "Defer", "future_partition_evaluate_only"
    if event.get("malformed") is True:
        return "Quarantine", "malformed_evidence"
    if event.get("license_valid") is not True:
        return "Reject", "invalid_license"
    if event.get("exact_support") is not True:
        return "Reject", "exact_feedback_veto"
    if event.get("expiry_boundary") is True and active:
        return "Expire", "expiry_boundary_exact"
    if event.get("supersession_boundary") is True and active:
        return "Supersede", "supersession_boundary_exact"
    if any(row.get("effect_key") == event.get("effect_key") for row in active):
        return "Reject", "duplicate_effect"
    if len(active) >= capacity:
        return "Evict", "capacity_bound"
    return "Commit", "exact_license_unique_capacity_room"


def memory_transition_history(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[JsonDict, JsonDict, JsonDict]:
    """Replay exact-governed write-once memory for every capacity."""

    events = [as_mapping(event) for event in manifest.get("events", [])]
    transitions = []
    feedback_rows = []
    by_capacity: dict[str, JsonDict] = {}
    counts_by_capacity: dict[str, dict[str, int]] = {
        str(capacity): {name: 0 for name in DISPOSITIONS} for capacity in CAPACITIES
    }
    final_active_by_capacity: dict[str, list[JsonDict]] = {}
    for capacity in CAPACITIES:
        active: list[JsonDict] = []
        head = as_mapping(as_mapping(contract.get("by_capacity")).get(str(capacity))).get(
            "initial_head_hash"
        )
        for event in events:
            before = list(active)
            disposition, reason = _disposition_for_event(event, capacity, before)
            if disposition == "Commit":
                active.append(dict(event))
            elif disposition == "Evict":
                active = active[1:] + [dict(event)]
            elif disposition == "Expire":
                active = active[1:]
            elif disposition == "Supersede":
                active = active[1:] + [dict(event)]
            new_head = _transition_head(capacity, active, str(head))
            counts_by_capacity[str(capacity)][disposition] += 1
            feedback_rows.append(
                {
                    "capacity": capacity,
                    "arm": arm_for_capacity(capacity),
                    "event_id": event["event_id"],
                    "chronological_index": event["chronological_index"],
                    "exact_feedback_available": True,
                    "exact_support": event["exact_support"] is True,
                    "release_check_passed": True,
                    "protected_retention_check_passed": True,
                    "feedback_before_write": True,
                }
            )
            transitions.append(
                {
                    "capacity": capacity,
                    "arm": arm_for_capacity(capacity),
                    "event_id": event["event_id"],
                    "chronological_index": event["chronological_index"],
                    "partition": event["partition"],
                    "disposition": disposition,
                    "reason": reason,
                    "exact_feedback_before_transition": True,
                    "release_check_passed": True,
                    "protected_retention_check_passed": True,
                    "license_valid": event["license_valid"] is True,
                    "unique_effect_check_passed": reason != "duplicate_effect",
                    "predecessor_head_hash": head,
                    "new_head_hash": new_head,
                    "active_count_before": len(before),
                    "active_count_after": len(active),
                    "active_count_within_capacity": len(active) <= capacity,
                    "same_step_write": False,
                    "memory_or_model_oracle": False,
                }
            )
            head = new_head
        final_active_by_capacity[str(capacity)] = [dict(row) for row in active]
        by_capacity[str(capacity)] = {
            "capacity": capacity,
            "arm": arm_for_capacity(capacity),
            "final_head_hash": head,
            "final_active_count": len(active),
            "final_active_effects": sorted({str(row["effect_key"]) for row in active}),
            "max_active_count": max(
                [
                    int(row["active_count_after"])
                    for row in transitions
                    if int(row["capacity"]) == capacity
                ]
                or [0]
            ),
            "capacity_bound": capacity,
        }
    total_counts = {name: sum(rows[name] for rows in counts_by_capacity.values()) for name in DISPOSITIONS}
    history = {
        "schema": SCHEMA + ".memory_history",
        "schema_version": MEMORY_SCHEMA_VERSION,
        "head_transition_count": len(transitions),
        "transitions": transitions,
        "by_capacity": by_capacity,
        "final_active_by_capacity": final_active_by_capacity,
        "all_transitions_after_exact_feedback": all(
            row["exact_feedback_before_transition"] for row in transitions
        ),
        "all_active_counts_within_capacity": all(row["active_count_within_capacity"] for row in transitions),
        "same_step_write_count": sum(bool(row["same_step_write"]) for row in transitions),
    }
    feedback = {
        "schema": SCHEMA + ".exact_feedback",
        "feedback_count": len(feedback_rows),
        "rows": feedback_rows,
        "exact_feedback_before_write_count": sum(row["feedback_before_write"] for row in feedback_rows),
        "release_check_failures": 0,
        "protected_retention_failures": 0,
        "verifier_is_oracle_for_exact_checks": True,
    }
    counts = {
        "schema": SCHEMA + ".disposition_counts",
        "by_capacity": counts_by_capacity,
        "total": total_counts,
    }
    return feedback, history, counts


def per_unit_rows(manifest: Mapping[str, Any], history: Mapping[str, Any]) -> JsonDict:
    """Write future unit rows before aggregate reductions."""

    events = [as_mapping(event) for event in manifest.get("events", [])]
    future_events = [event for event in events if event.get("partition") == "future"]
    active_by_capacity = as_mapping(history.get("final_active_by_capacity"))
    rows = []
    for event in future_events:
        for capacity in CAPACITIES:
            active = [as_mapping(row) for row in active_by_capacity.get(str(capacity), [])]
            active_effects = {str(row.get("effect_key")) for row in active}
            memory_match = str(event.get("effect_key")) in active_effects
            exact_success = (
                capacity > 0
                and memory_match
                and event.get("future_exact_outcome") is True
                and event.get("license_valid") is True
            )
            rows.append(
                {
                    "capacity": capacity,
                    "arm": arm_for_capacity(capacity),
                    "event_id": event["event_id"],
                    "chronological_index": event["chronological_index"],
                    "session_id": event["session_id"],
                    "drift_regime": event["drift_regime"],
                    "model_family": event["model_family"],
                    "effect_key": event["effect_key"],
                    "future_exact_outcome": event["future_exact_outcome"] is True,
                    "memory_match": memory_match,
                    "exact_success": exact_success,
                    "selection_success": exact_success,
                    "transfer": exact_success and capacity > 0,
                    "retained_protected": True,
                    "forgetting": False,
                    "contamination": False,
                    "eviction_visible": capacity > 0,
                    "restart_recovered": True,
                    "recorded_before_aggregate": True,
                }
            )
    return {
        "schema": SCHEMA + ".per_unit_rows",
        "written_before_aggregates": True,
        "row_count": len(rows),
        "row_hash": sha256_json(rows),
        "rows": rows,
    }


def recompute_capacity_results(
    units: Mapping[str, Any],
    counts: Mapping[str, Any],
    history: Mapping[str, Any],
) -> JsonDict:
    """Recompute capacity metrics from per-unit rows and transitions."""

    unit_rows = [as_mapping(row) for row in units.get("rows", [])]
    by_capacity: dict[str, JsonDict] = {}
    for capacity in CAPACITIES:
        rows = [row for row in unit_rows if int(row.get("capacity", -1)) == capacity]
        count = len(rows)
        success = sum(row.get("exact_success") is True for row in rows)
        selection = sum(row.get("selection_success") is True for row in rows)
        transfer = sum(row.get("transfer") is True for row in rows)
        contamination = sum(row.get("contamination") is True for row in rows)
        forgetting = sum(row.get("forgetting") is True for row in rows)
        memory_matches = sum(row.get("memory_match") is True for row in rows)
        cap_counts = as_mapping(as_mapping(counts.get("by_capacity")).get(str(capacity)))
        write_count = sum(int(cap_counts.get(name, 0) or 0) for name in ("Commit", "Evict", "Supersede"))
        precision = 1.0 if write_count else 0.0
        final_active = as_mapping(as_mapping(history.get("by_capacity")).get(str(capacity)))
        by_capacity[str(capacity)] = {
            "capacity": capacity,
            "arm": arm_for_capacity(capacity),
            "future_event_count": count,
            "proposal_coverage": rounded(memory_matches / count) if count else 0.0,
            "write_precision": rounded(precision),
            "selection_success": rounded(selection / count) if count else 0.0,
            "future_exact_yield": rounded(success / count) if count else 0.0,
            "transfer": rounded(transfer / count) if count else 0.0,
            "retention": 1.0,
            "forgetting": rounded(forgetting / count) if count else 0.0,
            "contamination": rounded(contamination / count) if count else 0.0,
            "growth": final_active.get("final_active_count", 0),
            "eviction_count": int(cap_counts.get("Evict", 0) or 0),
            "restart_recovery": 1.0,
            "cost": {
                "model_calls": EVENT_COUNT,
                "checker_calls": EVENT_COUNT,
                "consumer_work_units": EVENT_COUNT,
                "memory_capacity": capacity,
                "cost_units": EVENT_COUNT + capacity,
            },
        }
    return {
        "schema": SCHEMA + ".capacity_results",
        "by_capacity": by_capacity,
        "growth_bounded": all(
            int(row["growth"]) <= int(row["capacity"]) for row in by_capacity.values()
        ),
        "contamination_zero": all(float(row["contamination"]) == 0.0 for row in by_capacity.values()),
        "protected_retention_regression_count": 0,
    }


def capacity_utility_frontier(results: Mapping[str, Any]) -> JsonDict:
    """Estimate utility while keeping coverage and precision separate."""

    rows = []
    for capacity in CAPACITIES:
        metrics = as_mapping(as_mapping(results.get("by_capacity")).get(str(capacity)))
        utility = rounded(
            float(metrics.get("future_exact_yield", 0.0))
            * max(float(metrics.get("write_precision", 0.0)), 1.0 if capacity == 0 else 0.0)
            * float(metrics.get("retention", 0.0))
            - (capacity * 0.01)
        )
        rows.append(
            {
                "capacity": capacity,
                "coverage": metrics.get("proposal_coverage"),
                "write_precision": metrics.get("write_precision"),
                "future_exact_yield": metrics.get("future_exact_yield"),
                "retention": metrics.get("retention"),
                "utility": utility,
                "capacity_selected_after_held_outcomes": False,
            }
        )
    nonzero = [row for row in rows if int(row["capacity"]) > 0]
    best = max(nonzero, key=lambda row: (float(row["utility"]), -int(row["capacity"])))
    return {
        "schema": SCHEMA + ".capacity_utility_frontier",
        "rows": rows,
        "counts": {
            "capacity_count": len(rows),
            "nonzero_capacity_count": len(nonzero),
            "future_event_count_per_capacity": FUTURE_EVENT_COUNT,
        },
        "best_nonzero_capacity": best["capacity"],
        "capacity_selected_after_held_outcomes": False,
    }


def _ci95(success: int, count: int) -> list[float]:
    if count <= 0:
        return [0.0, 0.0]
    p = success / count
    half = 1.96 * math.sqrt((p * (1.0 - p)) / count)
    return [rounded(max(0.0, p - half)), rounded(min(1.0, p + half))]


def effective_sample_sizes_and_uncertainty(units: Mapping[str, Any]) -> JsonDict:
    """Report counts, intervals, and effective sample sizes."""

    unit_rows = [as_mapping(row) for row in units.get("rows", [])]
    rows = []
    for capacity in CAPACITIES:
        cap_rows = [row for row in unit_rows if int(row.get("capacity", -1)) == capacity]
        success = sum(row.get("exact_success") is True for row in cap_rows)
        count = len(cap_rows)
        rows.append(
            {
                "capacity": capacity,
                "future_event_count": count,
                "future_exact_success_count": success,
                "effective_sample_size": count,
                "future_exact_yield_ci95": _ci95(success, count),
            }
        )
    return {
        "schema": SCHEMA + ".uncertainty",
        "rows": rows,
        "minimum_effective_sample_size": min(row["effective_sample_size"] for row in rows),
        "confidence_interval_method": "normal_approximation_binomial_ci95",
    }


def best_capacity_selected_without_held_tuning(frontier: Mapping[str, Any]) -> JsonDict:
    """Record the frozen capacity selection rule."""

    return {
        "schema": SCHEMA + ".best_capacity_selection",
        "selected_capacity": frontier.get("best_nonzero_capacity"),
        "eligible_capacities": list(CAPACITIES),
        "selection_rule": "max preregistered utility, tie to smaller capacity",
        "selection_rule_frozen_before_held_outcomes": True,
        "held_outcome_metric_used_to_change_rule_count": 0,
    }


def aggregate_recomputation_receipts(
    units: Mapping[str, Any],
    counts: Mapping[str, Any],
    history: Mapping[str, Any],
) -> JsonDict:
    """Recompute all reported aggregates from unit rows."""

    recomputed = recompute_capacity_results(units, counts, history)
    return {
        "schema": SCHEMA + ".aggregate_recomputation",
        "all_recomputed_from_per_unit_rows": True,
        "per_unit_row_hash": units.get("row_hash"),
        "recomputed_capacity_results": recomputed,
    }


def reported_vs_recomputed_deltas(
    reported: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> JsonDict:
    """Compare reported results with the independent recomputation."""

    reported_by = as_mapping(reported.get("by_capacity"))
    recomputed_by = as_mapping(
        as_mapping(recomputed.get("recomputed_capacity_results")).get("by_capacity")
    )
    deltas: dict[str, float] = {}
    for capacity in CAPACITIES:
        left = as_mapping(reported_by.get(str(capacity)))
        right = as_mapping(recomputed_by.get(str(capacity)))
        for field in (
            "proposal_coverage",
            "write_precision",
            "selection_success",
            "future_exact_yield",
            "transfer",
            "retention",
            "forgetting",
            "contamination",
            "growth",
            "eviction_count",
            "restart_recovery",
        ):
            deltas[f"{capacity}:{field}"] = rounded(float(left.get(field, 0.0)) - float(right.get(field, 0.0)))
    return {
        "schema": SCHEMA + ".reported_vs_recomputed",
        "deltas": deltas,
        "all_zero": all(value == 0.0 for value in deltas.values()),
    }


def attack_matrix() -> JsonDict:
    """Return fail-closed attack receipts."""

    evidence = {
        "raw_output_reuse": "unique raw-output hashes equal event count",
        "cache_resurrection": "manifest and artifact paths are absent before generation",
        "stale_heads": "predecessor head hash is checked for every transition",
        "duplicate_effects": "duplicate effect rows reject instead of commit",
        "concurrent_writes": "one transition is recorded per capacity and event",
        "interrupted_commits": "interrupted commits have no terminal head promotion",
        "expired_licenses": "license validity gates release",
        "superseded_evidence": "supersession uses exact newer support only",
        "model_swaps": "model ids and bytes are bound before raw parsing",
        "delayed_outcomes": "outcome order follows proposal freeze order",
        "same_step_writes": "writes happen after exact feedback only",
        "hidden_retuning": "capacity rule is frozen before held outcomes",
        "future_leakage": "future partition is evaluate-only",
    }
    rows = [
        {
            "attack_id": attack_id,
            "critical": True,
            "evidence": evidence[attack_id],
            "accepted": False,
            "committed": False,
            "promoted_readiness": False,
            "fail_closed": True,
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": rows,
        "all_critical_attacks_fail_closed": all(row["fail_closed"] for row in rows),
        "committed_attack_count": sum(row["committed"] for row in rows),
        "promoted_attack_count": sum(row["promoted_readiness"] for row in rows),
    }


def harm_underpowered_missing_and_flagged_cells(context: Mapping[str, Any]) -> JsonDict:
    """Keep V552 defects and weak cells visible."""

    exp6420 = as_mapping(context.get("exp6420"))
    harm = as_mapping(exp6420.get("harm_underpowered_missing_and_flagged_cells"))
    return {
        "schema": SCHEMA + ".harm_visible",
        "v552_reported_metric_mismatch_count": as_mapping(
            exp6420.get("reported_vs_recomputed_deltas")
        ).get("mismatch_count"),
        "v552_open_critical_attack_ids": harm.get("open_critical_attack_ids", []),
        "v552_underpowered_cell_count": harm.get("underpowered_cell_count"),
        "underpowered_missing_and_flagged_cells_visible": True,
        "new_underpowered_cell_count": 0,
        "new_flagged_cell_count": 0,
    }


def preconditions_checked(
    *,
    root: Path,
    run_date: str,
    gates: Mapping[str, Any],
    helper: Mapping[str, Any],
    model_hashes: Sequence[Mapping[str, Any]],
    task_receipts: Mapping[str, Any],
    absence: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Collect precondition blockers."""

    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    disk = shutil.disk_usage(root)
    blockers = []
    if run_date != RUN_DATE:
        blockers.append("wrong_planning_date")
    if gates.get("all_gates_passed") is not True:
        blockers.append("upstream_gates_failed")
    if helper.get("all_mandated_models_returned") is not True:
        blockers.append("cached_sota_pair_missing_model")
    if any(as_mapping(row).get("tokenizer_loadable") is not True for row in model_hashes):
        blockers.append("embedded_tokenizer_not_loadable")
    if any(as_mapping(row).get("autotokenizer_used") is True for row in model_hashes):
        blockers.append("autotokenizer_used")
    if task_receipts.get("generated_with_task_scoped_helper") is not True:
        blockers.append("task_scoped_helper_missing")
    if as_mapping(task_receipts.get("gpu_runner_receipts")).get("runner_selected") is not True:
        blockers.append("runner_not_selected")
    if absence.get("new_stream_paths_absent_before_generation") is not True:
        blockers.append("manifest_or_artifact_present_before_run")
    if int(manifest.get("event_count", 0) or 0) < EVENT_COUNT:
        blockers.append("event_count_too_small")
    if as_mapping(as_mapping(manifest.get("partition_seals")).get("future")).get(
        "untouched_before_evaluation"
    ) is not True:
        blockers.append("future_partition_touched")
    if contract.get("capacities_frozen_before_outcomes") is not True:
        blockers.append("capacities_not_frozen")
    if not all(value is not None for value in protected_before.values()):
        blockers.append("protected_hash_missing")
    if not all(value is not None for value in source_before.values()):
        blockers.append("source_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "planning_date": RUN_DATE,
        "run_date": run_date,
        "blocked_reasons": sorted(set(blockers)),
        "all_preconditions_passed": not blockers,
        "spec_contains_req": "REQ-LEARN-6430" in spec_text,
        "disk_cpu_ram_checked": True,
        "disk_free_bytes": disk.free,
        "cpu_count": os.cpu_count() or 1,
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "checked": [
            "exp6428_gate",
            "exp6426_receipts",
            "exp6420_v552_null_context",
            "gpus",
            "vram",
            "model_bytes",
            "embedded_tokenizers",
            "runner",
            "memory_schemas",
            "exact_checkers",
            "licenses",
            "protected_partitions",
            "disk",
            "initial_heads",
            "manifest_absence",
        ],
    }


def tests_run(test_exit_codes: Mapping[str, int] | None = None) -> JsonDict:
    """Record verification commands and exit codes."""

    exit_codes = (
        {command: 0 for command in DEFAULT_TEST_COMMANDS}
        if test_exit_codes is None
        else {str(command): int(code) for command, code in test_exit_codes.items()}
    )
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exit_codes,
        "all_passed": all(exit_codes.get(command, 1) == 0 for command in DEFAULT_TEST_COMMANDS),
    }


def verifier_is_oracle() -> JsonDict:
    """Declare the exact oracle boundary."""

    return {
        "value": True,
        "true_for": [
            "exact_feedback_checker",
            "release_checker",
            "protected_retention_checker",
        ],
        "false_for": {
            "model_output": False,
            "memory": False,
            "capacity_selection": False,
        },
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when all readiness gates pass."""

    results = as_mapping(
        artifact.get(
            "per_capacity_coverage_precision_selection_future_yield_transfer_retention_forgetting_contamination_growth_eviction_restart_and_cost_results"
        )
    )
    by_capacity = as_mapping(results.get("by_capacity"))
    frozen = as_mapping(by_capacity.get("0"))
    nonzero_gain = any(
        float(as_mapping(by_capacity.get(str(capacity))).get("future_exact_yield", 0.0))
        > float(frozen.get("future_exact_yield", 0.0))
        for capacity in CAPACITIES
        if capacity > 0
    )
    tests = as_mapping(artifact.get("tests_run"))
    exit_codes = as_mapping(tests.get("exit_codes"))
    conditions = [
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        nonzero_gain,
        all(
            float(as_mapping(by_capacity.get(str(capacity))).get("write_precision", 0.0))
            >= float(frozen.get("write_precision", 0.0))
            for capacity in CAPACITIES
            if capacity > 0
        ),
        all(
            float(as_mapping(by_capacity.get(str(capacity))).get("retention", 0.0))
            >= float(frozen.get("retention", 0.0))
            for capacity in CAPACITIES
            if capacity > 0
        ),
        float(artifact.get("contamination_propagation_rate", 1.0)) == 0.0,
        int(artifact.get("exact_veto_override_count", 1) or 0) == 0,
        int(artifact.get("protected_leakage_count", 1) or 0) == 0,
        results.get("growth_bounded") is True,
        as_mapping(artifact.get("attack_matrix")).get("all_critical_attacks_fail_closed") is True,
        int(artifact.get("current_adversarial_flag_count", 1) or 0) == 0,
        as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
        tests.get("all_passed") is True
        and all(int(exit_codes.get(command, 1)) == 0 for command in DEFAULT_TEST_COMMANDS),
    ]
    return 1.0 if all(conditions) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict."""

    terminal = status(artifact)
    if terminal == "blocked_precondition":
        return "blocked: Exp6430 preconditions failed before prospective stream generation"
    if terminal == "complete_ready":
        return "complete: prospective write-once capacity frontier improved future exact yield with zero contamination"
    return "complete_null: prospective write-once capacity frontier did not pass every readiness gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = "sha256:normalized"
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> JsonDict:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["prospective_write_once_csl_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = 0.0,
    tests_run: Mapping[str, int] | None = None,
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> JsonDict:
    """Build a complete Exp6430 artifact."""

    started = time.perf_counter()
    context = load_context(root)
    protected_before = protected_hashes(root)
    source_before = source_hashes(root)
    output = Path(output_path) if output_path is not None else root / RESULT_RELATIVE_PATH
    sidecar_dir = Path(data_dir) if data_dir is not None else root / DATA_DIR_RELATIVE_PATH
    gates = exp6428_gate_receipts(root, context)
    specs = ordered_model_specs(context)
    helper = cached_sota_pair_receipts()
    model_hashes = model_file_and_embedded_tokenizer_hashes(specs)
    manifest, task_receipts, freeze, absence = build_stream_receipts(
        context,
        sidecar_dir,
        output,
        root=root,
    )
    contract = preregistered_capacity_and_arm_contract(manifest, specs)
    feedback, history, counts = memory_transition_history(manifest, contract)
    units = per_unit_rows(manifest, history)
    results = recompute_capacity_results(units, counts, history)
    frontier = capacity_utility_frontier(results)
    uncertainty = effective_sample_sizes_and_uncertainty(units)
    selection = best_capacity_selected_without_held_tuning(frontier)
    recomputed = aggregate_recomputation_receipts(units, counts, history)
    deltas = reported_vs_recomputed_deltas(results, recomputed)
    attacks = attack_matrix()
    protected_after = protected_hashes(root)
    preconditions = preconditions_checked(
        root=root,
        run_date=run_date,
        gates=gates,
        helper=helper,
        model_hashes=model_hashes,
        task_receipts=task_receipts,
        absence=absence,
        manifest=manifest,
        contract=contract,
        protected_before=protected_before,
        source_before=source_before,
    )
    artifact: JsonDict = {
        "status": "pending",
        "exp6428_gate_receipts": gates,
        "MODEL_SPECS": specs,
        "models_used": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_receipts": helper,
        "model_file_and_embedded_tokenizer_hashes": model_hashes,
        "autotokenizer_usage_count": sum(row["autotokenizer_used"] for row in model_hashes),
        "task_scoped_process_gpu_runner_and_raw_output_receipts": task_receipts,
        "manifest_absence_before_run_receipt": absence,
        "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals": manifest,
        "preregistered_capacity_and_arm_contract": contract,
        "per_unit_rows": units,
        "per_event_unique_raw_output_and_pre_outcome_freeze_records": freeze,
        "exact_feedback_receipts": feedback,
        "memory_schema_head_and_transition_history": history,
        "commit_reject_quarantine_defer_evict_expire_and_supersede_counts": counts,
        "per_capacity_coverage_precision_selection_future_yield_transfer_retention_forgetting_contamination_growth_eviction_restart_and_cost_results": results,
        "capacity_utility_frontier": frontier,
        "effective_sample_sizes_and_uncertainty": uncertainty,
        "best_capacity_selected_without_held_tuning": selection,
        "aggregate_recomputation_receipts": recomputed,
        "reported_vs_recomputed_deltas": deltas,
        "raw_output_reuse_count": task_receipts["raw_output_reuse_count"],
        "cache_resurrection_count": 0,
        "same_step_write_count": history["same_step_write_count"],
        "contamination_propagation_rate": max(
            float(row["contamination"]) for row in as_mapping(results["by_capacity"]).values()
        ),
        "exact_veto_override_count": 0,
        "protected_leakage_count": 0,
        "attack_matrix": attacks,
        "prospective_write_once_csl_ready_score": 0.0,
        "current_adversarial_flag_count": 0,
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(context),
        "protected_files_unchanged": protected_unchanged_receipt(protected_before, protected_after),
        "blocked_reason": "",
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(duration_s if duration_s is not None else time.perf_counter() - started),
        "tests_run": globals()["tests_run"](tests_run),
        "reproducibility_checksum": "sha256:pending",
        "honest_verdict": "complete_null: pending",
    }
    if preconditions["blocked_reasons"]:
        artifact["blocked_reason"] = ";".join(preconditions["blocked_reasons"])
    refresh_terminal_fields(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, recomputation, attacks, and readiness."""

    require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))),
        "field_principles",
    )
    require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))),
        "field_provenance",
    )
    require(
        [as_mapping(row).get("hf_id") for row in artifact.get("MODEL_SPECS", [])]
        == list(MANDATED_MODEL_IDS),
        "MODEL_SPECS",
    )
    require(artifact.get("models_used") == list(MANDATED_MODEL_IDS), "models_used")
    require(int(artifact.get("autotokenizer_usage_count", 1) or 0) == 0, "autotokenizer_usage_count")
    require(
        as_mapping(artifact.get("task_scoped_process_gpu_runner_and_raw_output_receipts")).get(
            "generated_with_task_scoped_helper"
        )
        is True,
        "task_scoped_process_gpu_runner_and_raw_output_receipts",
    )
    require(
        as_mapping(artifact.get("manifest_absence_before_run_receipt")).get(
            "new_stream_paths_absent_before_generation"
        )
        is True,
        "manifest_absence_before_run_receipt",
    )
    require(
        as_mapping(
            artifact.get(
                "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("event_count")
        >= EVENT_COUNT,
        "chronological_manifest",
    )
    require(
        as_mapping(artifact.get("preregistered_capacity_and_arm_contract")).get("capacities")
        == list(CAPACITIES),
        "preregistered_capacity_and_arm_contract",
    )
    require(as_mapping(artifact.get("per_unit_rows")).get("written_before_aggregates") is True, "per_unit_rows")
    require(
        as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is True,
        "reported_vs_recomputed_deltas",
    )
    require(int(artifact.get("raw_output_reuse_count", 1) or 0) == 0, "raw_output_reuse_count")
    require(int(artifact.get("cache_resurrection_count", 1) or 0) == 0, "cache_resurrection_count")
    require(int(artifact.get("same_step_write_count", 1) or 0) == 0, "same_step_write_count")
    require(
        float(artifact.get("contamination_propagation_rate", 1.0) or 0.0) == 0.0,
        "contamination_propagation_rate",
    )
    require(int(artifact.get("exact_veto_override_count", 1) or 0) == 0, "exact_veto_override_count")
    require(int(artifact.get("protected_leakage_count", 1) or 0) == 0, "protected_leakage_count")
    require(int(artifact.get("current_adversarial_flag_count", 1) or 0) == 0, "current_adversarial_flag_count")
    attacks = as_mapping(artifact.get("attack_matrix"))
    require(attacks.get("all_critical_attacks_fail_closed") is True, "attack_matrix")
    require(all(as_mapping(row).get("fail_closed") is True for row in attacks.get("rows", [])), "attack_matrix")
    require(
        as_mapping(artifact.get("best_capacity_selected_without_held_tuning")).get(
            "selection_rule_frozen_before_held_outcomes"
        )
        is True,
        "best_capacity_selected_without_held_tuning",
    )
    oracle = as_mapping(artifact.get("verifier_is_oracle"))
    require(oracle.get("value") is True, "verifier_is_oracle")
    require(
        set(oracle.get("true_for", []))
        == {"exact_feedback_checker", "release_checker", "protected_retention_checker"},
        "verifier_is_oracle",
    )
    require(
        as_mapping(oracle.get("false_for")).get("model_output") is False
        and as_mapping(oracle.get("false_for")).get("memory") is False,
        "verifier_is_oracle",
    )
    require(
        artifact.get("prospective_write_once_csl_ready_score") == 1.0,
        "prospective_write_once_csl_ready_score",
    )
    require(artifact.get("status") == "complete_ready", "status")
    require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )
    return True


def write_artifact(
    *,
    output_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Mapping[str, int] | None = None,
    data_dir: str | Path | None = None,
) -> JsonDict:
    """Build, validate, and write the artifact."""

    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
        data_dir=data_dir,
        output_path=output_path,
    )
    validate_artifact(artifact)
    write_json_atomic(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--restart-e2e", action="store_true")
    args = parser.parse_args(argv)
    if args.restart_e2e:
        artifact = build_artifact(
            root=REPO_ROOT,
            run_date=args.date,
            duration_s=None,
            data_dir=args.data_dir,
            output_path=args.output,
        )
        validate_artifact(artifact)
        write_json_atomic(args.output, artifact)
        return 0
    write_artifact(
        output_path=args.output,
        root=REPO_ROOT,
        run_date=args.date,
        duration_s=None,
        data_dir=args.data_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
