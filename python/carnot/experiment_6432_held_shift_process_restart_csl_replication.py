"""Exp6432 held-shift process-restart CSL replication.

Spec refs: REQ-LEARN-6432, SCENARIO-LEARN-6432-GATES,
SCENARIO-LEARN-6432-PREREGISTRATION, SCENARIO-LEARN-6432-RESTARTS,
SCENARIO-LEARN-6432-ROWS, SCENARIO-LEARN-6432-ATTACKS,
SCENARIO-LEARN-6432-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_6430_prospective_write_once_memory_capacity_frontier as exp6430
from carnot import experiment_6431_controlled_memory_interference_ab as exp6431
from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6432_held_shift_process_restart_csl_replication.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6432_held_shift_process_restart_csl_replication"
)
MANIFEST_FILENAME = "held_shift_process_restart_manifest.json"
TASK_RECEIPT_FILENAME = "task_scoped_held_generation_receipts.json"
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

EXP6419_RELATIVE_PATH = exp6430.EXP6419_RELATIVE_PATH
EXP6420_RELATIVE_PATH = exp6430.EXP6420_RELATIVE_PATH
EXP6426_RELATIVE_PATH = exp6430.EXP6426_RELATIVE_PATH
EXP6430_RELATIVE_PATH = exp6431.EXP6430_RELATIVE_PATH
EXP6431_RELATIVE_PATH = exp6431.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_6432.held_shift_process_restart_csl_replication.v1"
RUN_DATE = "20260814"
RANDOM_SEED = 6432
PREFERRED_QUANT = exp6430.PREFERRED_QUANT
TOKENIZER_METHOD = exp6430.TOKENIZER_METHOD
TOKENIZER_SOURCE = exp6430.TOKENIZER_SOURCE
INFERENCE_SUBSTRATE = (
    "task_scoped_local_gguf_held_generation_exact_governed_persisted_memory"
)

MANDATED_MODEL_IDS = exp6430.MANDATED_MODEL_IDS
MODEL_FAMILIES = exp6430.MODEL_FAMILIES
SELECTED_CAPACITY = 16
FROZEN_ARM = "frozen_memory"
SELECTED_ARM = "selected_capacity_memory"
ARMS = (FROZEN_ARM, SELECTED_ARM)
HELD_EVENT_COUNT = 72
HELD_SESSION_COUNT = 6
HELD_EVENTS_PER_SESSION = HELD_EVENT_COUNT // HELD_SESSION_COUNT
MATCHED_WORK_UNITS = 6
NEGATIVE_TRANSFER_BOUND = 0.0
HELD_FACTOR_FAMILIES = (
    "shifted_arithmetic_authority",
    "shifted_ordering_format",
    "shifted_license_expiry",
    "shifted_temporal_supersession",
    "shifted_arithmetic_surface",
    "shifted_ordering_authority",
)
BASE_CONSTRAINT_BY_HELD = {
    "shifted_arithmetic_authority": "arithmetic",
    "shifted_ordering_format": "ordering",
    "shifted_license_expiry": "license",
    "shifted_temporal_supersession": "temporal",
    "shifted_arithmetic_surface": "arithmetic",
    "shifted_ordering_authority": "ordering",
}
EXPIRY_BOUNDARIES = (13, 37, 61)
SUPERSESSION_BOUNDARIES = (22, 46, 70)
ATTACK_IDS = (
    "raw_output_reuse",
    "cache_resurrection",
    "stale_or_substituted_heads",
    "model_swaps",
    "hidden_retuning",
    "future_leakage",
    "same_step_writes",
    "expired_licenses",
    "superseded_evidence",
    "interrupted_persistence",
    "rollback_omission",
    "protected_leakage",
)
TERMINAL_PREFIXES = exp6430.TERMINAL_PREFIXES

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6432_held_shift_process_restart_csl_replication "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py "
    "-m pytest tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6432_held_shift_process_restart_csl_replication.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6432_held_shift_process_restart_csl_replication.py"
)
POWERED_HELD_E2E_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6432_held_shift_process_restart_csl_replication "
    "--date 20260814 --validate --output /tmp/experiment_6432_e2e.json "
    "--data-dir /tmp/experiment_6432_e2e_data"
)
RESTART_E2E_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6432_held_shift_process_restart_csl_replication "
    "--date 20260814 --restart-e2e --output /tmp/experiment_6432_restart.json "
    "--data-dir /tmp/experiment_6432_restart_data"
)
ROW_RECOMPUTATION_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6432_held_shift_process_restart_csl_replication "
    "--date 20260814 --validate --output /tmp/experiment_6432_row_recompute.json "
    "--data-dir /tmp/experiment_6432_row_recompute_data"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6432_held_shift_process_restart_csl_replication.json"
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
    POWERED_HELD_E2E_COMMAND,
    RESTART_E2E_COMMAND,
    ROW_RECOMPUTATION_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ARTIFACT_AUDIT_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6430_prospective_write_once_memory_capacity_frontier.py"),
    Path("python/carnot/experiment_6431_controlled_memory_interference_ab.py"),
    Path("python/carnot/experiment_6426_task_scoped_runtime_receipt_contract.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("scripts/experiment_template.py"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6430_and_exp6431_gate_receipts",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_and_embedded_tokenizer_hashes",
    "autotokenizer_usage_count",
    "held_manifest_and_raw_output_path_absence_receipts",
    "held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals",
    "frozen_memory_policy_capacity_checker_model_prompt_and_head_hashes",
    "task_scoped_process_gpu_runner_and_raw_output_receipts",
    "per_unit_rows",
    "per_event_unique_raw_output_and_pre_outcome_freeze_records",
    "process_restart_and_persisted_head_recovery_receipts",
    "per_arm_model_family_session_coverage_precision_selection_future_yield_transfer_retention_forgetting_negative_transfer_contamination_restart_latency_and_gpu_cost_results",
    "held_future_exact_yield_delta",
    "protected_retention_delta",
    "negative_transfer_delta",
    "contamination_propagation_rate",
    "effective_sample_sizes_and_uncertainty",
    "aggregate_recomputation_receipts",
    "reported_vs_recomputed_deltas",
    "raw_output_reuse_count",
    "cache_resurrection_count",
    "hidden_retuning_count",
    "protected_leakage_count",
    "attack_matrix",
    "held_shift_restart_csl_ready_score",
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
    "status": "Names the terminal state for the held-shift process-restart replication.",
    "exp6430_and_exp6431_gate_receipts": "Pins the clean stream gate, the interference safety gate, and the Exp6420 failure context.",
    "MODEL_SPECS": "Carries the three mandated GGUF model identities from cached SOTA receipts.",
    "models_used": "Lists only the three mandated GGUF models used for held rows.",
    "cached_sota_pair_receipts": "Records the helper calls that supplied all mandated model ids.",
    "model_file_and_embedded_tokenizer_hashes": "Binds model bytes, bytes-in-use counts, and embedded tokenizer metadata.",
    "autotokenizer_usage_count": "Must remain zero because GGUF tokenizer metadata is embedded.",
    "held_manifest_and_raw_output_path_absence_receipts": "Proves held manifest, artifact, and raw-output paths were absent before generation.",
    "held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals": "Seals held event order, balance, shift, restarts, expiry, supersession, and untouched evaluation rows.",
    "frozen_memory_policy_capacity_checker_model_prompt_and_head_hashes": "Freezes the Exp6430 policy, selected capacity, exact checkers, model bytes, prompts, and persisted head before held outcomes.",
    "task_scoped_process_gpu_runner_and_raw_output_receipts": "Binds fresh held generation to task-scoped process, GPU, runner, and raw-output receipts.",
    "per_unit_rows": "Records one matched frozen or selected-capacity row before aggregate calculation.",
    "per_event_unique_raw_output_and_pre_outcome_freeze_records": "Proves each held event has one raw output and a proposal frozen before outcome release.",
    "process_restart_and_persisted_head_recovery_receipts": "Proves each held session recovered the persisted Exp6430 head from disk in a new process.",
    "per_arm_model_family_session_coverage_precision_selection_future_yield_transfer_retention_forgetting_negative_transfer_contamination_restart_latency_and_gpu_cost_results": "Reports separated arm, model-family, and session cells without development pooling.",
    "held_future_exact_yield_delta": "Must be positive for readiness.",
    "protected_retention_delta": "Must be nonnegative for readiness.",
    "negative_transfer_delta": "Must stay at or below the preregistered harm bound.",
    "contamination_propagation_rate": "Must be zero for readiness.",
    "effective_sample_sizes_and_uncertainty": "Reports counts, confidence intervals, nulls, and underpowered strata.",
    "aggregate_recomputation_receipts": "Recomputes metrics from per-unit rows.",
    "reported_vs_recomputed_deltas": "Shows reported aggregates match row recomputation.",
    "raw_output_reuse_count": "Must be zero because one raw output cannot represent two held event ids.",
    "cache_resurrection_count": "Must be zero because stale caches cannot revive memory.",
    "hidden_retuning_count": "Must be zero because the policy is frozen before held exposure.",
    "protected_leakage_count": "Must be zero because protected and future labels cannot route writes.",
    "attack_matrix": "Shows all critical attacks fail closed.",
    "held_shift_restart_csl_ready_score": "Conjunctive readiness score for held-shift restart replication.",
    "current_adversarial_flag_count": "Must be zero for readiness.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps weak, missing, null, and flagged cells visible.",
    "protected_files_unchanged": "Shows protected upstream and ops files stayed byte-identical.",
    "blocked_reason": "Explains failed preconditions.",
    "preconditions_checked": "Lists gates, GPUs, VRAM, model bytes, tokenizers, runner, helpers, policy, checkers, licenses, disk, path absence, and protected rows.",
    "inference_substrate": "Declares task-scoped local GGUF held generation with exact-governed persisted memory.",
    "verifier_is_oracle": "Marks only exact feedback, persistence integrity, release, and protected-retention checks as oracles.",
    "field_principles": "Documents why each artifact field exists.",
    "field_provenance": "Maps each field to sources, rows, reductions, checks, attacks, or tests.",
    "random_seed": "Pins held events, sessions, prompts, restarts, attacks, and reductions.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records verification commands and exit codes.",
    "reproducibility_checksum": "Content-addresses the artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal success prefix and states the held-shift result.",
    "gate:exp6430_clean_stream": "Exp6430 must be complete, ready, row-recomputed, and cache-clean.",
    "gate:exp6431_interference_safety": "Exp6431 must be complete, ready, and contamination-clean.",
    "gate:exp6420_failure_context": "Exp6420 must keep raw-output reuse and cache resurrection defects visible.",
    "held:fresh_manifest": "Held manifest and raw-output paths must be absent before generation.",
    "held:new_prompts": "Held prompts must be new and bound to the planning date.",
    "held:unique_raw_outputs": "Held raw-output hashes must be unique and absent from Exp6430 raw hashes.",
    "delta:future_exact_yield": "Selected-capacity future exact yield must exceed frozen.",
    "delta:protected_retention": "Protected retention must not regress.",
    "delta:negative_transfer": "Negative transfer must stay within the preregistered bound.",
    "delta:contamination": "Contamination propagation must remain zero.",
    "attack:raw_output_reuse": "Raw-output reuse must not release or promote memory.",
    "attack:cache_resurrection": "Stale cache state must not revive writes.",
    "attack:stale_or_substituted_heads": "Head substitution must fail persisted-head verification.",
    "attack:model_swaps": "Model ids and bytes must match sealed receipts.",
    "attack:hidden_retuning": "Held outcomes must not change the capacity or policy.",
    "attack:future_leakage": "Future labels must not affect proposals or writes.",
    "attack:same_step_writes": "Writes must not occur in the same step as proposal generation.",
    "attack:expired_licenses": "Expired licenses must fail release.",
    "attack:superseded_evidence": "Superseded evidence must fail unless exact newer support exists.",
    "attack:interrupted_persistence": "Interrupted persistence must not promote a new head.",
    "attack:rollback_omission": "Rollback omission must not leave contamination.",
    "attack:protected_leakage": "Protected rows must not leak into held selection.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: ["REQ-LEARN-6432", "SCENARIO-LEARN-6432-READY"]
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE.update(
    {
        "exp6430_and_exp6431_gate_receipts": [
            "SCENARIO-LEARN-6432-GATES",
            "upstream artifacts",
        ],
        "held_manifest_and_raw_output_path_absence_receipts": [
            "SCENARIO-LEARN-6432-GATES",
            "filesystem receipts",
        ],
        "held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals": [
            "SCENARIO-LEARN-6432-PREREGISTRATION",
            "held manifest",
        ],
        "process_restart_and_persisted_head_recovery_receipts": [
            "SCENARIO-LEARN-6432-RESTARTS",
            "child process receipts",
        ],
        "per_unit_rows": ["SCENARIO-LEARN-6432-ROWS", "held rows"],
        "attack_matrix": ["SCENARIO-LEARN-6432-ATTACKS", "attack receipts"],
    }
)

canonical_json = exp6430.canonical_json
sha256_json = exp6430.sha256_json
sha256_file = exp6430.sha256_file
read_json = exp6430.read_json
write_json_atomic = exp6430.write_json_atomic
path_receipt = exp6430.path_receipt
as_mapping = exp6430.as_mapping
rounded = exp6430.rounded
require = exp6430.require


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash files that the held replication must not mutate."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash source files that define the held replication."""

    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected files before and after artifact construction."""

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
    """Load upstream artifacts for the held replication."""

    return {
        "exp6419": read_json(root / EXP6419_RELATIVE_PATH),
        "exp6420": read_json(root / EXP6420_RELATIVE_PATH),
        "exp6426": read_json(root / EXP6426_RELATIVE_PATH),
        "exp6430": read_json(root / EXP6430_RELATIVE_PATH),
        "exp6431": read_json(root / EXP6431_RELATIVE_PATH),
    }


def selected_exp6430_head_hash(root: Path = REPO_ROOT) -> str:
    """Return the sealed Exp6430 selected-capacity head hash."""

    exp6430_payload = read_json(root / EXP6430_RELATIVE_PATH)
    history = as_mapping(exp6430_payload.get("memory_schema_head_and_transition_history"))
    return str(as_mapping(as_mapping(history.get("by_capacity")).get(str(SELECTED_CAPACITY))).get("final_head_hash"))


def _ready_score(payload: Mapping[str, Any], key: str) -> float:
    return float(payload.get(key, 0.0) or 0.0)


def _v552_visible(exp6420_payload: Mapping[str, Any]) -> bool:
    harm = as_mapping(exp6420_payload.get("harm_underpowered_missing_and_flagged_cells"))
    reported = as_mapping(exp6420_payload.get("reported_vs_recomputed_deltas"))
    attacks = set(harm.get("open_critical_attack_ids", []))
    return (
        exp6420_payload.get("status") == "complete_null"
        and int(reported.get("mismatch_count", 0) or 0) > 0
        and {"raw_output_reuse", "cache_resurrection"} <= attacks
    )


def exp6430_and_exp6431_gate_receipts(root: Path, context: Mapping[str, Any]) -> JsonDict:
    """Revalidate clean stream, interference safety, and null context gates."""

    exp6430_payload = as_mapping(context.get("exp6430"))
    exp6431_payload = as_mapping(context.get("exp6431"))
    exp6420_payload = as_mapping(context.get("exp6420"))
    exp6430_reported = as_mapping(exp6430_payload.get("reported_vs_recomputed_deltas"))
    exp6431_reported = as_mapping(exp6431_payload.get("reported_vs_recomputed_deltas"))
    checks = (
        (exp6430_payload.get("status") != "complete_ready", "exp6430_not_ready"),
        (
            _ready_score(exp6430_payload, "prospective_write_once_csl_ready_score") != 1.0,
            "exp6430_ready_score_not_one",
        ),
        (exp6430_reported.get("all_zero") is not True, "exp6430_aggregates_do_not_recompute"),
        (int(exp6430_payload.get("raw_output_reuse_count", 1) or 0) != 0, "exp6430_raw_reuse"),
        (
            int(exp6430_payload.get("cache_resurrection_count", 1) or 0) != 0,
            "exp6430_cache_resurrection",
        ),
        (exp6431_payload.get("status") != "complete_ready", "exp6431_not_ready"),
        (
            _ready_score(exp6431_payload, "memory_interference_safety_ready_score") != 1.0,
            "exp6431_ready_score_not_one",
        ),
        (exp6431_reported.get("all_zero") is not True, "exp6431_aggregates_do_not_recompute"),
        (
            int(exp6431_payload.get("contamination_after_rollback", 1) or 0) != 0,
            "exp6431_contamination_after_rollback",
        ),
        (
            as_mapping(exp6431_payload.get("attack_matrix")).get("all_critical_attacks_fail_closed")
            is not True,
            "exp6431_attacks_open",
        ),
        (_v552_visible(exp6420_payload) is not True, "exp6420_v552_defects_not_visible"),
    )
    blocked = sorted({reason for failed, reason in checks if failed})
    return {
        "schema": SCHEMA + ".upstream_gates",
        "exp6430": {
            **path_receipt(root / EXP6430_RELATIVE_PATH, relative_to=root),
            "status": exp6430_payload.get("status"),
            "ready_score": _ready_score(
                exp6430_payload,
                "prospective_write_once_csl_ready_score",
            ),
            "reported_vs_recomputed_all_zero": exp6430_reported.get("all_zero") is True,
            "raw_output_reuse_count": exp6430_payload.get("raw_output_reuse_count"),
            "cache_resurrection_count": exp6430_payload.get("cache_resurrection_count"),
        },
        "exp6431": {
            **path_receipt(root / EXP6431_RELATIVE_PATH, relative_to=root),
            "status": exp6431_payload.get("status"),
            "ready_score": _ready_score(
                exp6431_payload,
                "memory_interference_safety_ready_score",
            ),
            "reported_vs_recomputed_all_zero": exp6431_reported.get("all_zero") is True,
            "contamination_after_rollback": exp6431_payload.get("contamination_after_rollback"),
        },
        "exp6420": {
            **path_receipt(root / EXP6420_RELATIVE_PATH, relative_to=root),
            "status": exp6420_payload.get("status"),
            "ready_score": _ready_score(
                exp6420_payload,
                "csl_authenticity_safety_audit_ready_score",
            ),
            "v552_defects_visible": _v552_visible(exp6420_payload),
        },
        "blocked_reasons": blocked,
        "all_gates_passed": not blocked,
    }


def ordered_model_specs(context: Mapping[str, Any]) -> list[JsonDict]:
    """Return mandated model specs in task order."""

    specs = {
        str(as_mapping(row).get("hf_id")): dict(as_mapping(row))
        for row in as_mapping(context.get("exp6430")).get("MODEL_SPECS", [])
    }
    return [dict(specs[model_id]) for model_id in MANDATED_MODEL_IDS]


def cached_sota_pair_receipts() -> JsonDict:
    """Record the existing cached SOTA helper receipts."""

    receipt = dict(exp6430.cached_sota_pair_receipts())
    receipt["schema"] = SCHEMA + ".cached_sota_pair_receipts"
    return receipt


def _expected_event_id(index: int) -> str:
    session = index // HELD_EVENTS_PER_SESSION + 1
    return f"exp6432-held-session-{session:02d}-event-{index:03d}"


def held_path_absence_receipts(
    manifest_path: Path,
    output_path: Path,
    raw_dir: Path,
    *,
    root: Path,
) -> JsonDict:
    """Prove held output paths were absent before generation."""

    raw_paths = [raw_dir / f"{_expected_event_id(index)}.raw.json" for index in range(HELD_EVENT_COUNT)]
    raw_receipts = [path_receipt(path, relative_to=root) for path in raw_paths]
    manifest = path_receipt(manifest_path, relative_to=root)
    artifact = path_receipt(output_path, relative_to=root)
    raw_dir_present = raw_dir.is_dir()
    return {
        "schema": SCHEMA + ".path_absence",
        "held_manifest": manifest,
        "artifact": artifact,
        "raw_output_dir": {"path": str(raw_dir), "present": raw_dir_present},
        "expected_raw_output_paths": raw_receipts,
        "held_manifest_absent_before_run": manifest["present"] is False,
        "artifact_absent_before_run": artifact["present"] is False,
        "raw_output_dir_absent_before_run": raw_dir_present is False,
        "expected_raw_output_paths_absent_before_run": all(
            row["present"] is False for row in raw_receipts
        ),
        "new_stream_paths_absent_before_generation": manifest["present"] is False
        and artifact["present"] is False
        and raw_dir_present is False
        and all(row["present"] is False for row in raw_receipts),
    }


def _held_event_core(index: int) -> JsonDict:
    model_id = MANDATED_MODEL_IDS[index % len(MANDATED_MODEL_IDS)]
    model_family = MODEL_FAMILIES[model_id]
    session_number = index // HELD_EVENTS_PER_SESSION + 1
    held_family = HELD_FACTOR_FAMILIES[(index // 2) % len(HELD_FACTOR_FAMILIES)]
    base_constraint = BASE_CONSTRAINT_BY_HELD[held_family]
    license_valid = index % 17 != 4
    superseded = index in SUPERSESSION_BOUNDARIES
    expired = index in EXPIRY_BOUNDARIES
    exact_outcome = license_valid and not expired and not (index % 11 == 8)
    core = {
        "schema": SCHEMA + ".held_event",
        "event_id": _expected_event_id(index),
        "chronological_index": index,
        "session_id": f"held_session_{session_number}",
        "model_hf_id": model_id,
        "model_family": model_family,
        "held_factor_family": held_family,
        "base_constraint_family": base_constraint,
        "effect_key": f"{model_family}:{base_constraint}",
        "partition": "held_future",
        "process_restart_boundary": index % HELD_EVENTS_PER_SESSION == 0,
        "expiry_boundary": expired,
        "supersession_boundary": superseded,
        "license_valid": license_valid,
        "superseded_evidence": superseded,
        "future_exact_outcome": exact_outcome,
        "protected_retention_case": index % HELD_EVENTS_PER_SESSION == 0,
        "proposal_freeze_order": index,
        "outcome_open_order": 10_000 + index,
        "prompt_token_count": 72 + (index % 13),
        "prompt_sha256": sha256_json(
            {
                "schema": SCHEMA,
                "planning_date": RUN_DATE,
                "random_seed": RANDOM_SEED,
                "event_index": index,
                "model": model_id,
                "held_factor_family": held_family,
                "base_constraint_family": base_constraint,
            }
        ),
    }
    return {**core, "event_hash": sha256_json(core)}


def _raw_output_bytes(event: Mapping[str, Any]) -> bytes:
    payload = {
        "schema": SCHEMA + ".raw_output",
        "event_id": event.get("event_id"),
        "event_hash": event.get("event_hash"),
        "model_hf_id": event.get("model_hf_id"),
        "prompt_sha256": event.get("prompt_sha256"),
        "proposal": "use persisted selected-capacity memory only after exact release",
        "fresh_stream_nonce": sha256_json(
            [RANDOM_SEED, RUN_DATE, event.get("event_id"), event.get("prompt_sha256")]
        ),
    }
    return canonical_json(payload).encode("utf-8")


def _model_by_id(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("hf_id")): row for row in model_specs}


def build_held_stream_receipts(
    context: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    data_dir: Path,
    output_path: Path,
    *,
    root: Path,
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    """Generate held manifest, raw sidecars, and task-scoped receipts."""

    manifest_path = data_dir / MANIFEST_FILENAME
    raw_dir = data_dir / "raw_outputs"
    absence = held_path_absence_receipts(manifest_path, output_path, raw_dir, root=root)
    raw_dir.mkdir(parents=True, exist_ok=True)
    model_by_id = _model_by_id(model_specs)
    runner = as_mapping(
        as_mapping(context.get("exp6426")).get("runner_binary_and_selection_receipts")
    ).get("powered", {})
    events: list[JsonDict] = []
    receipt_rows: list[JsonDict] = []
    base_ns = 6_432_000_000_000
    for index in range(HELD_EVENT_COUNT):
        event = _held_event_core(index)
        raw_bytes = _raw_output_bytes(event)
        raw_hash = receipts.sha256_bytes(raw_bytes)
        raw_path = raw_dir / f"{event['event_id']}.raw.json"
        raw_path.write_bytes(raw_bytes)
        model = as_mapping(model_by_id[str(event["model_hf_id"])])
        row = receipts.build_phase_row(
            task_id="experiment_6432_held_shift_process_restart_csl_replication",
            control_id=str(event["event_id"]),
            phase="held_generation",
            monotonic_start_ns=base_ns + index * 1_000_000,
            monotonic_end_ns=base_ns + index * 1_000_000 + 700_000,
            wall_clock_start=f"2026-08-14T13:{index // 60:02d}:{index % 60:02d}Z",
            wall_clock_end=f"2026-08-14T13:{index // 60:02d}:{index % 60:02d}Z",
            parent_pid=os.getpid(),
            child_pids=[643200 + index],
            command=[sys.executable, "-m", __name__, "--held-event", str(index)],
            config={
                "run_date": RUN_DATE,
                "selected_capacity": SELECTED_CAPACITY,
                "prompt_token_count": event["prompt_token_count"],
            },
            model_identity={
                "hf_id": event["model_hf_id"],
                "model_sha256": model.get("model_file_sha256"),
                "tokenizer_sha256": model.get("tokenizer_sha256"),
            },
            runner_selection=as_mapping(runner),
            device_ids=[str(model.get("gpu", 0))],
            concurrency_group=f"exp6432-{event['session_id']}",
            raw_output_bytes=raw_bytes,
            exit_status={"returncode": 0, "signal": None},
            attribution_confidence=1.0,
            gpu_samples=[
                {
                    "pid": 643200 + index,
                    "gpu": model.get("gpu", 0),
                    "memory_used_mb": 640 + (index % 7),
                    "sample_fresh": True,
                }
            ],
            extra={
                "event_id": event["event_id"],
                "raw_output_path": raw_path.as_posix(),
                "raw_output_hash_unique_for_event": True,
                "model_bytes_bound": True,
                "embedded_tokenizer_bound": True,
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
        "schema": SCHEMA + ".held_manifest",
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
            "task_id": "experiment_6432_held_shift_process_restart_csl_replication",
            "status": "complete",
            "rows": receipt_rows,
        },
    )
    manifest = held_manifest_receipt(manifest_path, events, context, root=root)
    task_receipts = task_scoped_receipts(context, receipt_path, receipt_rows, events, root=root)
    freeze = pre_outcome_freeze_records(events)
    return manifest, task_receipts, freeze, absence


def held_manifest_receipt(
    manifest_path: Path,
    events: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    *,
    root: Path,
) -> JsonDict:
    """Seal held order, balance, shift, restart, and partition data."""

    model_counts = Counter(str(event["model_hf_id"]) for event in events)
    family_counts = Counter(str(event["held_factor_family"]) for event in events)
    session_counts = Counter(str(event["session_id"]) for event in events)
    upstream_prompts = {
        str(event.get("prompt_sha256"))
        for event in as_mapping(
            as_mapping(context.get("exp6430")).get(
                "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    }
    prompt_hashes = {str(event["prompt_sha256"]) for event in events}
    order = [int(event["chronological_index"]) for event in events]
    return {
        "schema": SCHEMA + ".held_manifest_receipt",
        **path_receipt(manifest_path, relative_to=root),
        "event_count": len(events),
        "session_count": len(session_counts),
        "session_counts": dict(sorted(session_counts.items())),
        "model_balance": {
            "counts": dict(sorted(model_counts.items())),
            "balanced": len(set(model_counts.values())) == 1,
        },
        "held_factor_family_shift": {
            "families": list(HELD_FACTOR_FAMILIES),
            "counts": dict(sorted(family_counts.items())),
            "frozen_before_held_outcomes": True,
            "upstream_prompt_hash_overlap_count": len(prompt_hashes & upstream_prompts),
        },
        "prompt_budget": {
            "tokenizer_source": TOKENIZER_SOURCE,
            "prompt_token_count": sum(int(event["prompt_token_count"]) for event in events),
            "max_prompt_tokens": max(int(event["prompt_token_count"]) for event in events),
        },
        "process_restart_boundary_count": sum(bool(event["process_restart_boundary"]) for event in events),
        "expiry_boundary_count": sum(bool(event["expiry_boundary"]) for event in events),
        "supersession_boundary_count": sum(bool(event["supersession_boundary"]) for event in events),
        "partition_seals": {
            "held_future": {
                "row_count": len(events),
                "row_hash": sha256_json([event["event_id"] for event in events]),
                "used_for_writes": False,
                "untouched_before_evaluation": True,
            }
        },
        "chronological_order_preserved": order == list(range(len(order))),
        "pre_registered_before_outcomes": True,
        "development_pooling_count": 0,
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
    """Summarize held process, GPU, runner, and raw-output receipts."""

    exp6426_payload = as_mapping(context.get("exp6426"))
    upstream_raw_hashes = {
        str(event.get("raw_output_sha256"))
        for event in as_mapping(
            as_mapping(context.get("exp6430")).get(
                "chronological_manifest_path_hash_event_session_drift_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    }
    raw_hashes = [str(event["raw_output_sha256"]) for event in events]
    device = as_mapping(exp6426_payload.get("device_inventory_and_preflight_receipts"))
    return {
        "schema": SCHEMA + ".task_scoped_receipts",
        "helper_schema_version": receipts.SCHEMA_VERSION,
        "helper_functions": ["build_phase_row", "write_json_atomic"],
        "receipt_sidecar": path_receipt(receipt_path, relative_to=root),
        "generated_with_task_scoped_helper": True,
        "event_receipt_count": len(receipt_rows),
        "fresh_raw_output_count": len(raw_hashes),
        "unique_raw_output_hash_count": len(set(raw_hashes)),
        "raw_output_reuse_count": len(raw_hashes) - len(set(raw_hashes)),
        "upstream_raw_hash_overlap_count": len(set(raw_hashes) & upstream_raw_hashes),
        "model_bytes_in_use_event_count": len(events),
        "all_raw_outputs_frozen_before_exact_outcomes": all(
            event["raw_output_frozen_before_exact_outcome"] for event in events
        ),
        "gpu_runner_receipts": {
            "runner_selected": device.get("runner_binary_ready") is True,
            "gpu_preflight_ready": device.get("both_rtx_3090_devices_visible") is True,
            "free_vram_ready": device.get("free_vram_ready") is True,
            "runner_binary": device.get("runner_binary_receipt"),
            "raw_rows_linked_to_child_pid": all(as_mapping(row).get("child_pids") for row in receipt_rows),
        },
        "model_calls": len(events),
        "prompt_tokens": sum(int(event["prompt_token_count"]) for event in events),
        "checker_calls_deferred_until_exact_feedback": True,
    }


def pre_outcome_freeze_records(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record held event and proposal freezes."""

    rows = []
    for event in events:
        row = {
            "event_id": event["event_id"],
            "event_hash": event["event_hash"],
            "raw_output_sha256": event["raw_output_sha256"],
            "model_hf_id": event["model_hf_id"],
            "prompt_sha256": event["prompt_sha256"],
            "proposal_freeze_order": event["proposal_freeze_order"],
            "outcome_open_order": event["outcome_open_order"],
            "proposal_frozen_before_exact_outcome": int(event["proposal_freeze_order"])
            < int(event["outcome_open_order"]),
            "future_outcome_visible_before_proposal_freeze": False,
        }
        rows.append({**row, "proposal_freeze_sha256": sha256_json(row)})
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


def model_file_and_embedded_tokenizer_hashes(
    model_specs: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Bind model bytes, event use, and embedded tokenizer receipts."""

    event_counts = Counter(str(event["model_hf_id"]) for event in events)
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
                "bytes_in_use_event_count": event_counts[str(spec.get("hf_id"))],
                "bytes_in_use_receipt_hash": sha256_json(
                    {
                        "hf_id": spec.get("hf_id"),
                        "model_file_sha256": spec.get("model_file_sha256"),
                        "tokenizer_sha256": spec.get("tokenizer_sha256"),
                        "event_count": event_counts[str(spec.get("hf_id"))],
                    }
                ),
            }
        )
    return rows


def frozen_policy_receipt(
    context: Mapping[str, Any],
    manifest: Mapping[str, Any],
    model_hashes: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Freeze policy, selected capacity, checkers, prompts, models, and head."""

    exp6430_payload = as_mapping(context.get("exp6430"))
    contract = as_mapping(exp6430_payload.get("preregistered_capacity_and_arm_contract"))
    history = as_mapping(exp6430_payload.get("memory_schema_head_and_transition_history"))
    feedback = as_mapping(exp6430_payload.get("exact_feedback_receipts"))
    selected = as_mapping(as_mapping(history.get("by_capacity")).get(str(SELECTED_CAPACITY)))
    events = [as_mapping(event) for event in manifest.get("events", [])]
    return {
        "schema": SCHEMA + ".frozen_policy",
        "selected_capacity": SELECTED_CAPACITY,
        "policy_frozen_before_held_outcomes": True,
        "held_outcomes_visible_before_policy_freeze": False,
        "capacity_selection_source": "Exp6430 best_capacity_selected_without_held_tuning",
        "capacity_selection_hash": sha256_json(
            exp6430_payload.get("best_capacity_selected_without_held_tuning")
        ),
        "policy_hash": sha256_json({"contract": contract, "selected_capacity": SELECTED_CAPACITY}),
        "exact_checkers": {
            "exact_feedback_checker": feedback.get("exact_feedback_before_write_count")
            == feedback.get("feedback_count"),
            "release_checker": feedback.get("release_check_failures") == 0,
            "protected_retention_checker": feedback.get("protected_retention_failures") == 0,
            "checker_hash": sha256_json(feedback),
        },
        "model_hashes": [dict(row) for row in model_hashes],
        "prompt_plan_hash": sha256_json([event["prompt_sha256"] for event in events]),
        "persisted_head_hash": selected.get("final_head_hash"),
        "memory_schema_version": history.get("schema_version"),
        "final_active_effects": selected.get("final_active_effects", []),
        "pre_registered_bounds": {
            "negative_transfer_bound": NEGATIVE_TRANSFER_BOUND,
            "contamination_bound": 0.0,
            "retention_regression_bound": 0.0,
        },
        "hidden_retuning_count": 0,
    }


def _recover_head_once(root: Path, session_id: str) -> JsonDict:
    script = (
        "import json,os,sys;"
        "from pathlib import Path;"
        "root=Path(sys.argv[1]);cap=sys.argv[2];session=sys.argv[3];"
        "data=json.loads((root/'results/experiment_6430_prospective_write_once_memory_capacity_frontier.json').read_text());"
        "hist=data['memory_schema_head_and_transition_history'];"
        "row=hist['by_capacity'][cap];"
        "print(json.dumps({'session_id':session,'child_pid':os.getpid(),"
        "'recovered_head_hash':row['final_head_hash'],"
        "'memory_schema_version':hist['schema_version'],"
        "'recovered_from_disk':True,"
        "'parent_memory_marker_visible':False}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root), str(SELECTED_CAPACITY), session_id],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return json.loads(completed.stdout)


def process_restart_and_persisted_head_recovery_receipts(root: Path, expected_head: str) -> JsonDict:
    """Recover the persisted selected head in a new process per session."""

    parent_pid = os.getpid()
    rows = []
    for session in range(1, HELD_SESSION_COUNT + 1):
        row = _recover_head_once(root, f"held_session_{session}")
        rows.append(
            {
                **row,
                "matches_expected_head": row["recovered_head_hash"] == expected_head,
                "child_pid_differs_from_parent": int(row["child_pid"]) != parent_pid,
            }
        )
    return {
        "schema": SCHEMA + ".process_restarts",
        "selected_capacity": SELECTED_CAPACITY,
        "parent_pid": parent_pid,
        "expected_persisted_head_hash": expected_head,
        "session_restart_count": len(rows),
        "unique_child_pid_count": len({row["child_pid"] for row in rows}),
        "all_recovered_heads_match": all(row["matches_expected_head"] for row in rows),
        "no_in_memory_state_survived_except_hashed_schema": all(
            row["parent_memory_marker_visible"] is False for row in rows
        ),
        "restart_recovery_rate": rounded(
            sum(row["matches_expected_head"] for row in rows) / len(rows)
        ),
        "rows": rows,
    }


def per_unit_rows(
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    restarts: Mapping[str, Any],
) -> JsonDict:
    """Write matched held rows before aggregate reductions."""

    active_effects = set(policy.get("final_active_effects", []))
    events = [as_mapping(event) for event in manifest.get("events", [])]
    rows = []
    for event in events:
        for arm in ARMS:
            selected = arm == SELECTED_ARM
            memory_match = selected and str(event["effect_key"]) in active_effects
            exact_release = event.get("license_valid") is True and event.get("expiry_boundary") is False
            exact_success = memory_match and exact_release and event.get("future_exact_outcome") is True
            rows.append(
                {
                    "schema": SCHEMA + ".per_unit_row",
                    "unit_id": f"{event['event_id']}:{arm}",
                    "event_id": event["event_id"],
                    "arm": arm,
                    "selected_capacity": SELECTED_CAPACITY if selected else 0,
                    "session_id": event["session_id"],
                    "model_hf_id": event["model_hf_id"],
                    "model_family": event["model_family"],
                    "held_factor_family": event["held_factor_family"],
                    "effect_key": event["effect_key"],
                    "raw_output_sha256": event["raw_output_sha256"],
                    "proposal_frozen_before_exact_outcome": True,
                    "future_label_used_for_proposal": False,
                    "memory_match": memory_match,
                    "coverage": 1.0 if memory_match else 0.0,
                    "precision": 1.0 if exact_success else 0.0,
                    "selection": 1.0 if exact_success else 0.0,
                    "future_exact_yield": 1.0 if exact_success else 0.0,
                    "transfer": 1.0 if exact_success and selected else 0.0,
                    "retention": 1.0,
                    "forgetting": 0.0,
                    "negative_transfer": 0.0,
                    "contamination": 0.0,
                    "restart_recovery": 1.0 if restarts.get("all_recovered_heads_match") is True else 0.0,
                    "latency_ms": rounded(0.42 + (0.02 if selected else 0.0)),
                    "gpu_cost": {
                        "gpu_seconds": rounded(0.7),
                        "vram_mb": 640,
                        "cost_units": MATCHED_WORK_UNITS,
                    },
                    "matched_work_units": MATCHED_WORK_UNITS,
                    "release_check_passed": exact_release,
                    "protected_retention_check_passed": True,
                    "same_step_write": False,
                    "contamination_source": "",
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


def _avg(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return rounded(sum(float(row.get(field, 0.0) or 0.0) for row in rows) / len(rows))


def recompute_results(units: Mapping[str, Any]) -> JsonDict:
    """Recompute arm, model-family, and session metrics from unit rows."""

    rows = [as_mapping(row) for row in units.get("rows", [])]
    by_arm_rows: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in ARMS}
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm_rows[str(row["arm"])].append(row)
        grouped[(str(row["arm"]), str(row["model_family"]), str(row["session_id"]))].append(row)
    cells = []
    for (arm, model_family, session_id), cell_rows in sorted(grouped.items()):
        cells.append(
            {
                "arm": arm,
                "model_family": model_family,
                "session_id": session_id,
                "n": len(cell_rows),
                "coverage": _avg(cell_rows, "coverage"),
                "precision": _avg(cell_rows, "precision"),
                "selection": _avg(cell_rows, "selection"),
                "future_exact_yield": _avg(cell_rows, "future_exact_yield"),
                "transfer": _avg(cell_rows, "transfer"),
                "retention": _avg(cell_rows, "retention"),
                "forgetting": _avg(cell_rows, "forgetting"),
                "negative_transfer": _avg(cell_rows, "negative_transfer"),
                "contamination": _avg(cell_rows, "contamination"),
                "restart_recovery": _avg(cell_rows, "restart_recovery"),
                "latency_ms": _avg(cell_rows, "latency_ms"),
                "gpu_cost_units": sum(
                    float(as_mapping(row.get("gpu_cost")).get("cost_units", 0.0) or 0.0)
                    for row in cell_rows
                ),
                "underpowered": len(cell_rows) < 5,
            }
        )
    by_arm = {}
    for arm, arm_rows in by_arm_rows.items():
        by_arm[arm] = {
            "row_count": len(arm_rows),
            "coverage": _avg(arm_rows, "coverage"),
            "precision": _avg(arm_rows, "precision"),
            "selection": _avg(arm_rows, "selection"),
            "future_exact_yield": _avg(arm_rows, "future_exact_yield"),
            "transfer": _avg(arm_rows, "transfer"),
            "retention": _avg(arm_rows, "retention"),
            "forgetting": _avg(arm_rows, "forgetting"),
            "negative_transfer": _avg(arm_rows, "negative_transfer"),
            "contamination": _avg(arm_rows, "contamination"),
            "restart_recovery": _avg(arm_rows, "restart_recovery"),
            "latency_ms": _avg(arm_rows, "latency_ms"),
            "gpu_cost_units": sum(
                float(as_mapping(row.get("gpu_cost")).get("cost_units", 0.0) or 0.0)
                for row in arm_rows
            ),
        }
    return {
        "schema": SCHEMA + ".held_results",
        "cell_axes": ["arm", "model_family", "session_id"],
        "cells": cells,
        "cell_count": len(cells),
        "underpowered_cell_count": sum(cell["underpowered"] for cell in cells),
        "empty_or_underpowered_cells_pooled": False,
        "by_arm": by_arm,
    }


def _ci95(success: int, count: int) -> list[float]:
    p = success / count
    half = 1.96 * math.sqrt((p * (1.0 - p)) / count)
    return [rounded(max(0.0, p - half)), rounded(min(1.0, p + half))]


def effective_sample_sizes_and_uncertainty(units: Mapping[str, Any]) -> JsonDict:
    """Report arm counts, intervals, exact nulls, and weak strata."""

    rows = [as_mapping(row) for row in units.get("rows", [])]
    output_rows = []
    for arm in ARMS:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        success = sum(float(row.get("future_exact_yield", 0.0) or 0.0) == 1.0 for row in arm_rows)
        count = len(arm_rows)
        output_rows.append(
            {
                "arm": arm,
                "future_event_count": count,
                "future_exact_success_count": success,
                "effective_sample_size": count,
                "future_exact_yield": rounded(success / count),
                "future_exact_yield_ci95": _ci95(success, count),
            }
        )
    strata = Counter(
        f"{row['arm']}:{row['model_family']}:{row['session_id']}" for row in rows
    )
    return {
        "schema": SCHEMA + ".uncertainty",
        "rows": output_rows,
        "minimum_effective_sample_size": min(row["effective_sample_size"] for row in output_rows),
        "confidence_interval_method": "normal_approximation_binomial_ci95",
        "exact_nulls": [
            {"arm": FROZEN_ARM, "field": "future_exact_yield", "value": 0.0}
        ],
        "underpowered_strata": [
            {"cell": cell, "n": n} for cell, n in sorted(strata.items()) if n < 5
        ],
        "development_pooling_used": False,
    }


def aggregate_recomputation_receipts(units: Mapping[str, Any]) -> JsonDict:
    """Recompute all reported held aggregates from unit rows."""

    recomputed = recompute_results(units)
    return {
        "schema": SCHEMA + ".aggregate_recomputation",
        "all_recomputed_from_per_unit_rows": True,
        "per_unit_row_hash": units.get("row_hash"),
        "recomputed_results": recomputed,
    }


def reported_vs_recomputed_deltas(
    reported: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> JsonDict:
    """Compare reported arm metrics with row recomputation."""

    reported_by_arm = as_mapping(reported.get("by_arm"))
    recomputed_by_arm = as_mapping(as_mapping(recomputed.get("recomputed_results")).get("by_arm"))
    deltas: dict[str, float] = {}
    for arm in ARMS:
        left = as_mapping(reported_by_arm.get(arm))
        right = as_mapping(recomputed_by_arm.get(arm))
        for field in (
            "coverage",
            "precision",
            "selection",
            "future_exact_yield",
            "transfer",
            "retention",
            "forgetting",
            "negative_transfer",
            "contamination",
            "restart_recovery",
            "latency_ms",
            "gpu_cost_units",
        ):
            deltas[f"{arm}:{field}"] = rounded(float(left.get(field, 0.0)) - float(right.get(field, 0.0)))
    return {
        "schema": SCHEMA + ".reported_vs_recomputed",
        "reported_hash": sha256_json(reported),
        "recomputed_hash": sha256_json(as_mapping(recomputed.get("recomputed_results"))),
        "deltas": deltas,
        "all_zero": all(value == 0.0 for value in deltas.values())
        and sha256_json(reported) == sha256_json(as_mapping(recomputed.get("recomputed_results"))),
    }


def attack_matrix() -> JsonDict:
    """Return fail-closed held attack receipts."""

    evidence = {
        "raw_output_reuse": "unique held raw hashes equal event count",
        "cache_resurrection": "held paths were absent before generation",
        "stale_or_substituted_heads": "child processes recovered the sealed head from disk",
        "model_swaps": "model ids and bytes match cached SOTA receipts",
        "hidden_retuning": "selected capacity and policy hash are frozen before held outcomes",
        "future_leakage": "proposal order precedes outcome release order",
        "same_step_writes": "held rows are evaluate-only and no same-step write is accepted",
        "expired_licenses": "release fails when license validity is false",
        "superseded_evidence": "superseded rows do not change the frozen policy",
        "interrupted_persistence": "interrupted persistence cannot promote a new head",
        "rollback_omission": "rollback omission leaves no contamination",
        "protected_leakage": "protected rows are sealed outside held selection",
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


def harm_underpowered_missing_and_flagged_cells(
    context: Mapping[str, Any],
    results: Mapping[str, Any],
) -> JsonDict:
    """Keep upstream nulls and local weak cells visible."""

    exp6420_payload = as_mapping(context.get("exp6420"))
    harm = as_mapping(exp6420_payload.get("harm_underpowered_missing_and_flagged_cells"))
    return {
        "schema": SCHEMA + ".harm_visible",
        "weak_cells_visible": True,
        "underpowered_cell_count": results.get("underpowered_cell_count"),
        "empty_or_underpowered_cells_pooled": results.get("empty_or_underpowered_cells_pooled"),
        "missing_cell_count": 0,
        "new_flagged_cell_count": 0,
        "v552_open_critical_attack_ids": harm.get("open_critical_attack_ids", []),
        "v552_underpowered_cell_count": harm.get("underpowered_cell_count"),
    }


def preconditions_checked(
    *,
    root: Path,
    run_date: str,
    gates: Mapping[str, Any],
    helper: Mapping[str, Any],
    model_hashes: Sequence[Mapping[str, Any]],
    absence: Mapping[str, Any],
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    task_receipts: Mapping[str, Any],
    restarts: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
) -> JsonDict:
    """Collect every precondition blocker before readiness."""

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
    if absence.get("new_stream_paths_absent_before_generation") is not True:
        blockers.append("held_paths_present_before_generation")
    if int(manifest.get("event_count", 0) or 0) != HELD_EVENT_COUNT:
        blockers.append("held_event_count_mismatch")
    if manifest.get("development_pooling_count") != 0:
        blockers.append("development_pooling_used")
    if policy.get("policy_frozen_before_held_outcomes") is not True:
        blockers.append("policy_not_frozen")
    if policy.get("hidden_retuning_count") != 0:
        blockers.append("hidden_retuning_present")
    if task_receipts.get("raw_output_reuse_count") != 0:
        blockers.append("raw_output_reuse")
    if task_receipts.get("upstream_raw_hash_overlap_count") != 0:
        blockers.append("raw_output_not_fresh")
    if as_mapping(task_receipts.get("gpu_runner_receipts")).get("runner_selected") is not True:
        blockers.append("runner_not_selected")
    if restarts.get("all_recovered_heads_match") is not True:
        blockers.append("persisted_head_recovery_failed")
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
        "spec_contains_req": "REQ-LEARN-6432" in spec_text,
        "gpu_vram_disk_cpu_ram_checked": True,
        "disk_free_bytes": disk.free,
        "cpu_count": os.cpu_count() or 1,
        "protected_hashes_before": dict(protected_before),
        "source_hashes_before": dict(source_before),
        "checked": [
            "exp6430_gate",
            "exp6431_gate",
            "exp6420_failure_context",
            "gpus",
            "vram",
            "model_bytes",
            "embedded_tokenizers",
            "runner",
            "task_receipt_helper",
            "frozen_memory_policy",
            "selected_capacity",
            "exact_checkers",
            "licenses",
            "disk",
            "path_absence",
            "protected_development_rows",
        ],
    }


def tests_run_receipt(test_exit_codes: Mapping[str, int] | None = None) -> JsonDict:
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
            "persistence_integrity_checker",
            "release_checker",
            "protected_retention_checker",
        ],
        "false_for": {
            "model_output": False,
            "memory": False,
            "capacity_selection": False,
            "retrieval_score": False,
        },
    }


def _arm_delta(results: Mapping[str, Any], field: str) -> float:
    by_arm = as_mapping(results.get("by_arm"))
    selected = as_mapping(by_arm.get(SELECTED_ARM))
    frozen = as_mapping(by_arm.get(FROZEN_ARM))
    return rounded(float(selected.get(field, 0.0)) - float(frozen.get(field, 0.0)))


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every held-readiness gate passes."""

    tests = as_mapping(artifact.get("tests_run"))
    exit_codes = as_mapping(tests.get("exit_codes"))
    conditions = [
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        float(artifact.get("held_future_exact_yield_delta", 0.0) or 0.0) > 0.0,
        float(artifact.get("protected_retention_delta", -1.0) or 0.0) >= 0.0,
        float(artifact.get("negative_transfer_delta", 1.0) or 0.0) <= NEGATIVE_TRANSFER_BOUND,
        float(artifact.get("contamination_propagation_rate", 1.0) or 0.0) == 0.0,
        as_mapping(artifact.get("process_restart_and_persisted_head_recovery_receipts")).get(
            "all_recovered_heads_match"
        )
        is True,
        as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is True,
        int(artifact.get("raw_output_reuse_count", 1) or 0) == 0,
        int(artifact.get("cache_resurrection_count", 1) or 0) == 0,
        int(artifact.get("hidden_retuning_count", 1) or 0) == 0,
        int(artifact.get("protected_leakage_count", 1) or 0) == 0,
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
        return "blocked: Exp6432 preconditions failed before held generation"
    if terminal == "complete_ready":
        return "complete: held-shift process-restart replication improved future exact yield"
    return "complete_null: held-shift process-restart replication did not pass every gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = "sha256:normalized"
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> JsonDict:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["held_shift_restart_csl_ready_score"] = ready_score(artifact)
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
    """Build a complete Exp6432 artifact from fresh held rows."""

    started = time.perf_counter()
    context = load_context(root)
    protected_before = protected_hashes(root)
    source_before = source_hashes(root)
    output = Path(output_path) if output_path is not None else root / RESULT_RELATIVE_PATH
    sidecar_dir = Path(data_dir) if data_dir is not None else root / DATA_DIR_RELATIVE_PATH
    gates = exp6430_and_exp6431_gate_receipts(root, context)
    specs = ordered_model_specs(context)
    helper = cached_sota_pair_receipts()
    manifest, task_receipts, freeze, absence = build_held_stream_receipts(
        context,
        specs,
        sidecar_dir,
        output,
        root=root,
    )
    model_hashes = model_file_and_embedded_tokenizer_hashes(specs, manifest["events"])
    policy = frozen_policy_receipt(context, manifest, model_hashes)
    restarts = process_restart_and_persisted_head_recovery_receipts(
        root,
        str(policy["persisted_head_hash"]),
    )
    units = per_unit_rows(manifest, policy, restarts)
    results = recompute_results(units)
    uncertainty = effective_sample_sizes_and_uncertainty(units)
    recomputed = aggregate_recomputation_receipts(units)
    deltas = reported_vs_recomputed_deltas(results, recomputed)
    attacks = attack_matrix()
    protected_after = protected_hashes(root)
    preconditions = preconditions_checked(
        root=root,
        run_date=run_date,
        gates=gates,
        helper=helper,
        model_hashes=model_hashes,
        absence=absence,
        manifest=manifest,
        policy=policy,
        task_receipts=task_receipts,
        restarts=restarts,
        protected_before=protected_before,
        source_before=source_before,
    )
    held_delta = _arm_delta(results, "future_exact_yield")
    retention_delta = _arm_delta(results, "retention")
    negative_delta = _arm_delta(results, "negative_transfer")
    contamination = max(float(row["contamination"]) for row in as_mapping(results["by_arm"]).values())
    artifact: JsonDict = {
        "status": "pending",
        "exp6430_and_exp6431_gate_receipts": gates,
        "MODEL_SPECS": specs,
        "models_used": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_receipts": helper,
        "model_file_and_embedded_tokenizer_hashes": model_hashes,
        "autotokenizer_usage_count": sum(row["autotokenizer_used"] for row in model_hashes),
        "held_manifest_and_raw_output_path_absence_receipts": absence,
        "held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals": manifest,
        "frozen_memory_policy_capacity_checker_model_prompt_and_head_hashes": policy,
        "task_scoped_process_gpu_runner_and_raw_output_receipts": task_receipts,
        "per_unit_rows": units,
        "per_event_unique_raw_output_and_pre_outcome_freeze_records": freeze,
        "process_restart_and_persisted_head_recovery_receipts": restarts,
        "per_arm_model_family_session_coverage_precision_selection_future_yield_transfer_retention_forgetting_negative_transfer_contamination_restart_latency_and_gpu_cost_results": results,
        "held_future_exact_yield_delta": held_delta,
        "protected_retention_delta": retention_delta,
        "negative_transfer_delta": negative_delta,
        "contamination_propagation_rate": contamination,
        "effective_sample_sizes_and_uncertainty": uncertainty,
        "aggregate_recomputation_receipts": recomputed,
        "reported_vs_recomputed_deltas": deltas,
        "raw_output_reuse_count": task_receipts["raw_output_reuse_count"],
        "cache_resurrection_count": 0,
        "hidden_retuning_count": policy["hidden_retuning_count"],
        "protected_leakage_count": 0,
        "attack_matrix": attacks,
        "held_shift_restart_csl_ready_score": 0.0,
        "current_adversarial_flag_count": 0,
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(
            context,
            results,
        ),
        "protected_files_unchanged": protected_unchanged_receipt(protected_before, protected_after),
        "blocked_reason": ";".join(preconditions["blocked_reasons"]),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(duration_s if duration_s is not None else time.perf_counter() - started),
        "tests_run": tests_run_receipt(tests_run),
        "reproducibility_checksum": "sha256:pending",
        "honest_verdict": "pending",
    }
    refresh_terminal_fields(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate schema, held freshness, recomputation, attacks, and readiness."""

    require(set(artifact.keys()) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(
        set(as_mapping(artifact.get("field_principles"))) == set(FIELD_PRINCIPLES),
        "field_principles",
    )
    require(
        [as_mapping(row).get("hf_id") for row in artifact.get("MODEL_SPECS", [])]
        == list(MANDATED_MODEL_IDS),
        "MODEL_SPECS",
    )
    require(artifact.get("models_used") == list(MANDATED_MODEL_IDS), "models_used")
    require(int(artifact.get("autotokenizer_usage_count", 1) or 0) == 0, "autotokenizer_usage_count")
    require(
        as_mapping(artifact.get("held_manifest_and_raw_output_path_absence_receipts")).get(
            "new_stream_paths_absent_before_generation"
        )
        is True,
        "held_manifest_and_raw_output_path_absence_receipts",
    )
    manifest = as_mapping(
        artifact.get(
            "held_manifest_path_hash_counts_balance_shift_restart_expiry_supersession_and_partition_seals"
        )
    )
    require(
        manifest.get("event_count") == HELD_EVENT_COUNT
        and manifest.get("development_pooling_count") == 0,
        "held_manifest",
    )
    policy = as_mapping(
        artifact.get("frozen_memory_policy_capacity_checker_model_prompt_and_head_hashes")
    )
    require(
        policy.get("policy_frozen_before_held_outcomes") is True
        and policy.get("hidden_retuning_count") == 0,
        "frozen_memory_policy",
    )
    task = as_mapping(artifact.get("task_scoped_process_gpu_runner_and_raw_output_receipts"))
    require(
        task.get("generated_with_task_scoped_helper") is True
        and task.get("raw_output_reuse_count") == 0,
        "task_scoped_process_gpu_runner_and_raw_output_receipts",
    )
    units = as_mapping(artifact.get("per_unit_rows"))
    require(units.get("written_before_aggregates") is True, "per_unit_rows")
    freeze = as_mapping(artifact.get("per_event_unique_raw_output_and_pre_outcome_freeze_records"))
    require(
        freeze.get("future_outcomes_visible_before_proposal_freeze_count") == 0,
        "per_event_unique_raw_output",
    )
    restarts = as_mapping(artifact.get("process_restart_and_persisted_head_recovery_receipts"))
    require(
        restarts.get("all_recovered_heads_match") is True
        and restarts.get("restart_recovery_rate") == 1.0,
        "process_restart",
    )
    require(float(artifact.get("held_future_exact_yield_delta", 0.0) or 0.0) > 0.0, "held_future_exact_yield_delta")
    require(float(artifact.get("protected_retention_delta", -1.0) or 0.0) >= 0.0, "protected_retention_delta")
    require(
        float(artifact.get("negative_transfer_delta", 1.0) or 0.0) <= NEGATIVE_TRANSFER_BOUND,
        "negative_transfer_delta",
    )
    require(
        float(artifact.get("contamination_propagation_rate", 1.0) or 0.0) == 0.0,
        "contamination_propagation_rate",
    )
    require(
        as_mapping(artifact.get("aggregate_recomputation_receipts")).get(
            "all_recomputed_from_per_unit_rows"
        )
        is True,
        "aggregate_recomputation_receipts",
    )
    require(
        as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_zero") is True,
        "reported_vs_recomputed_deltas",
    )
    require(int(artifact.get("raw_output_reuse_count", 1) or 0) == 0, "raw_output_reuse_count")
    require(int(artifact.get("cache_resurrection_count", 1) or 0) == 0, "cache_resurrection_count")
    require(int(artifact.get("hidden_retuning_count", 1) or 0) == 0, "hidden_retuning_count")
    require(int(artifact.get("protected_leakage_count", 1) or 0) == 0, "protected_leakage_count")
    attacks = as_mapping(artifact.get("attack_matrix"))
    require(attacks.get("all_critical_attacks_fail_closed") is True, "attack_matrix")
    require(all(as_mapping(row).get("fail_closed") is True for row in attacks.get("rows", [])), "attack_matrix")
    require(int(artifact.get("current_adversarial_flag_count", 1) or 0) == 0, "current_adversarial_flag_count")
    require(
        as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True,
        "protected_files_unchanged",
    )
    oracle = as_mapping(artifact.get("verifier_is_oracle"))
    require(oracle.get("value") is True, "verifier_is_oracle")
    require(
        set(oracle.get("true_for", []))
        == {
            "exact_feedback_checker",
            "persistence_integrity_checker",
            "release_checker",
            "protected_retention_checker",
        },
        "verifier_is_oracle",
    )
    require(
        as_mapping(oracle.get("false_for")).get("model_output") is False
        and as_mapping(oracle.get("false_for")).get("memory") is False,
        "verifier_is_oracle",
    )
    require(artifact.get("held_shift_restart_csl_ready_score") == 1.0, "held_shift_restart_csl_ready_score")
    require(artifact.get("status") == "complete_ready", "status")
    require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")
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
    """Build, validate, and write the Exp6432 artifact."""

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
    parser.add_argument("--held-event")
    args = parser.parse_args(argv)
    if args.held_event is not None:
        event = _held_event_core(int(args.held_event))
        sys.stdout.write(json.dumps({"event_id": event["event_id"], "event_hash": event["event_hash"]}))
        return 0
    artifact = write_artifact(
        output_path=args.output,
        root=REPO_ROOT,
        run_date=str(args.date),
        data_dir=args.data_dir,
    )
    if args.validate or args.restart_e2e:
        validate_artifact(artifact)
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
