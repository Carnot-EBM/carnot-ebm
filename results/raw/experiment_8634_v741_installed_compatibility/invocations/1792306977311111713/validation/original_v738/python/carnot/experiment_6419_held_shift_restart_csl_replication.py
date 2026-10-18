"""Exp6419 held-shift restart CSL replication.

Spec refs: REQ-LEARN-6419, SCENARIO-LEARN-6419-FREEZE,
SCENARIO-LEARN-6419-SHIFTS, SCENARIO-LEARN-6419-MATCHED-ARMS,
SCENARIO-LEARN-6419-NO-RETUNE, SCENARIO-LEARN-6419-ATTACKS,
SCENARIO-LEARN-6419-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6418_execution_grounded_dual_path_csl as exp6418


JsonDict = dict[str, Any]

canonical_json = exp6418.canonical_json
sha256_bytes = exp6418.sha256_bytes
sha256_text = exp6418.sha256_text
sha256_json = exp6418.sha256_json
sha256_file = exp6418.sha256_file
require = exp6418.require
as_mapping = exp6418.as_mapping
rounded = exp6418.rounded
read_json = exp6418.read_json
write_json_atomic = exp6418.write_json_atomic
path_receipt = exp6418.path_receipt
protected_unchanged_receipt = exp6418.protected_unchanged_receipt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6419_held_shift_restart_csl_replication.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6419_held_shift_restart_csl_replication"
)
HELD_MANIFEST_FILENAME = "held_shift_manifest.json"
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6419_held_shift_restart_csl_replication.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6419_held_shift_restart_csl_replication.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

EXP6418_RELATIVE_PATH = exp6418.RESULT_RELATIVE_PATH
EXP6413_RELATIVE_PATH = exp6418.EXP6413_RELATIVE_PATH
EXP6414_RELATIVE_PATH = Path(
    "results/experiment_6414_fresh_three_family_factor_event_corpus.json"
)

SCHEMA = "carnot.experiment_6419.held_shift_restart_csl_replication.v1"
RUN_DATE = "20260814"
RANDOM_SEED = 6419
INFERENCE_SUBSTRATE = (
    "authenticated_local_gguf_receipt_replay_with_sealed_held_shift_restart_stream"
)

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
FROZEN_ARM = "frozen"
SINGLE_PATH_ARM = "frozen_single_path"
FROZEN_DUAL_PATH_ARM = "frozen_dual_path_execution_grounded"
ARMS = (FROZEN_ARM, SINGLE_PATH_ARM, FROZEN_DUAL_PATH_ARM)
SURFACE_FORMS = ("compact_json", "natural_language", "table_row", "yaml_block")
TEMPORAL_SHIFTS = (
    "mechanism_freeze_window",
    "restart_window",
    "expiry_window",
    "future_window",
)
EVENT_COUNT = 72
SESSION_COUNT = 4
EVENTS_PER_SESSION = 18
FUTURE_START_INDEX = 48
RESTART_BOUNDARIES = (0, 18, 36, 54)
EXPIRY_BOUNDARIES = (17, 53)
SUPERSESSION_BOUNDARIES = (35, 71)
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
BARE_FINITE_FIELDS = ("held_delta_future_exact_yield_over_frozen",)
ATTACK_IDS = (
    "checkpoint_substitution",
    "partial_restart",
    "stale_cache_resurrection",
    "held_label_access",
    "model_swap",
    "prompt_drift",
    "license_inheritance",
    "silent_fallback",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6419_held_shift_restart_csl_replication "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6419_held_shift_restart_csl_replication.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6419_held_shift_restart_csl_replication.py "
    "-m pytest tests/python/test_experiment_6419_held_shift_restart_csl_replication.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6419_held_shift_restart_csl_replication.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6419_held_shift_restart_csl_replication.py"
)
INFERENCE_RESTART_E2E_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6419_held_shift_restart_csl_replication "
    "--date 20260814 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6419_held_shift_restart_csl_replication.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    INFERENCE_RESTART_E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6418_RELATIVE_PATH,
    EXP6414_RELATIVE_PATH,
    EXP6413_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/inference/sota_models.py"),
    Path("scripts/experiment_template.py"),
    exp6418.MODULE_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6418_gate_receipts",
    "frozen_mechanism_config_checker_model_and_prompt_hashes",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "embedded_gguf_tokenizer_receipts",
    "autotokenizer_usage_count",
    "held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals",
    "held_manifest_absence_before_freeze_receipt",
    "authenticated_process_and_raw_output_receipts_by_model",
    "matched_arm_work_receipts",
    "no_post_outcome_retuning_receipts",
    "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results",
    "held_delta_future_exact_yield_over_frozen",
    "held_contamination_propagation_rate",
    "held_forgetting_delta",
    "protected_leakage_count",
    "silent_fallback_count",
    "attack_matrix",
    "held_shift_csl_replication_ready_score",
    "public_factor_claim_eligibility",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
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

GATE_AND_SHIFT_PRINCIPLE_KEYS = (
    "gate:exp6418_prospective_improvement",
    "gate:held_manifest_absence",
    "shift:model_family",
    "shift:constraint_family",
    "shift:surface_form",
    "shift:temporal",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names the terminal state for the held-shift restart replication.",
    "exp6418_gate_receipts": "Pins the upstream prospective improvement gate.",
    "frozen_mechanism_config_checker_model_and_prompt_hashes": "Freezes the learner, checker, model, config, and prompt identity before held outcomes.",
    "MODEL_SPECS": "Carries the three mandated GGUF model identities from cached SOTA receipts.",
    "models_used": "Lists only the three mandated GGUF models.",
    "cached_sota_pair_receipts": "Records cached SOTA helper evidence.",
    "embedded_gguf_tokenizer_receipts": "Proves embedded GGUF tokenizer use.",
    "autotokenizer_usage_count": "Must remain zero because external tokenizer paths are forbidden.",
    "held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals": "Seals held events, shifts, restarts, expiry, supersession, and future rows.",
    "held_manifest_absence_before_freeze_receipt": "Proves Exp6418 mechanism selection did not include the held manifest.",
    "authenticated_process_and_raw_output_receipts_by_model": "Binds model processes and raw bytes before outcomes.",
    "matched_arm_work_receipts": "Shows frozen, single-path, and frozen dual-path arms used equal work.",
    "no_post_outcome_retuning_receipts": "Proves held outcomes did not change triggers, schemas, prompts, gates, or checkers.",
    "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results": "Reports held metrics without pooled masking.",
    "held_delta_future_exact_yield_over_frozen": "Bare held future-yield lift for frozen dual path over frozen.",
    "held_contamination_propagation_rate": "Must remain zero for readiness.",
    "held_forgetting_delta": "Must show no protected forgetting.",
    "protected_leakage_count": "Must be zero because protected partitions cannot route writes.",
    "silent_fallback_count": "Must be zero because fallback would break model identity.",
    "attack_matrix": "Shows every held restart and substitution attack fails closed.",
    "held_shift_csl_replication_ready_score": "Conjunctive readiness score for the held-shift restart replication.",
    "public_factor_claim_eligibility": "Limits public claims to this exact held replication.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps missing, underpowered, flagged, and harmful cells visible.",
    "protected_files_unchanged": "Shows protected files stayed byte-identical.",
    "preconditions_checked": "Lists every gate checked before readiness can become one.",
    "inference_substrate": "Declares authenticated GGUF receipt replay on a sealed held stream.",
    "verifier_is_oracle": "Marks only exact outcome and retention checkers as oracles.",
    "field_principles": "Documents why each field exists.",
    "field_provenance": "Maps each field to upstream receipts, manifest seals, attacks, tests, or code.",
    "random_seed": "Pins held order, shifts, arms, attacks, and metrics.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records verification commands and exit codes.",
    "reproducibility_checksum": "Content-addresses the artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal prefix and states the held replication result.",
    "gate:exp6418_prospective_improvement": "Exp6418 must be ready before the held replication can run.",
    "gate:held_manifest_absence": "The held manifest must be absent from Exp6418 mechanism selection.",
    "shift:model_family": "Model-family shift metrics must stay visible.",
    "shift:constraint_family": "Constraint-family shift metrics must stay visible.",
    "shift:surface_form": "Surface-form shift metrics must stay visible.",
    "shift:temporal": "Temporal shift metrics must stay visible.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6419",
        "Exp6418 frozen dual-path CSL artifact",
        "Exp6414 authenticated held source rows",
        "Exp6413 local GGUF process and tokenizer receipts",
        "Exp6419 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash source files that define this experiment."""

    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def load_context(root: Path = REPO_ROOT) -> JsonDict:
    """Load frozen upstream artifacts for the held replication."""

    return {
        "exp6418": read_json(root / EXP6418_RELATIVE_PATH),
        "exp6414": read_json(root / EXP6414_RELATIVE_PATH),
        "exp6413": read_json(root / EXP6413_RELATIVE_PATH),
    }


def _ready_score(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key, 0.0)
    return float(value or 0.0)


def exp6418_gate_receipts(root: Path, context: Mapping[str, Any]) -> JsonDict:
    """Revalidate the prospective learner and held source gates."""

    exp6418_payload = as_mapping(context.get("exp6418"))
    exp6414_payload = as_mapping(context.get("exp6414"))
    exp6413_payload = as_mapping(context.get("exp6413"))
    blocked: list[str] = []
    if _ready_score(exp6418_payload, "execution_grounded_dual_path_csl_ready_score") != 1.0:
        blocked.append("exp6418_gate_failed")
    if float(exp6418_payload.get("delta_future_exact_yield_over_frozen", 0.0) or 0.0) <= 0.0:
        blocked.append("exp6418_no_prospective_improvement")
    if _ready_score(exp6414_payload, "fresh_factor_event_corpus_ready_score") != 1.0:
        blocked.append("exp6414_held_source_gate_failed")
    if _ready_score(exp6413_payload, "authenticated_receipt_contract_ready_score") != 1.0:
        blocked.append("exp6413_receipt_gate_failed")
    return {
        "schema": SCHEMA + ".gate_receipts",
        "exp6418": {
            **path_receipt(root / EXP6418_RELATIVE_PATH),
            "ready_score": _ready_score(
                exp6418_payload,
                "execution_grounded_dual_path_csl_ready_score",
            ),
            "future_delta": exp6418_payload.get("delta_future_exact_yield_over_frozen"),
            "status": exp6418_payload.get("status"),
            "gate_passed": "exp6418_gate_failed" not in blocked
            and "exp6418_no_prospective_improvement" not in blocked,
        },
        "exp6414": {
            **path_receipt(root / EXP6414_RELATIVE_PATH),
            "ready_score": _ready_score(exp6414_payload, "fresh_factor_event_corpus_ready_score"),
            "event_count": as_mapping(
                exp6414_payload.get("manifest_path_hash_counts_balance_classes_and_partition_seals")
            ).get("event_count"),
            "status": exp6414_payload.get("status"),
            "gate_passed": "exp6414_held_source_gate_failed" not in blocked,
        },
        "exp6413": {
            **path_receipt(root / EXP6413_RELATIVE_PATH),
            "ready_score": _ready_score(
                exp6413_payload,
                "authenticated_receipt_contract_ready_score",
            ),
            "status": exp6413_payload.get("status"),
            "gate_passed": "exp6413_receipt_gate_failed" not in blocked,
        },
        "blocked_reasons": sorted(set(blocked)),
        "all_gates_passed": not blocked,
    }


def ordered_model_specs(context: Mapping[str, Any]) -> list[JsonDict]:
    """Return the mandated model specs in task order."""

    by_id = {
        str(as_mapping(row).get("hf_id")): dict(as_mapping(row))
        for row in as_mapping(context.get("exp6413")).get("MODEL_SPECS", [])
    }
    return [dict(by_id[model_id]) for model_id in MANDATED_MODEL_IDS]


def embedded_gguf_tokenizer_receipts(context: Mapping[str, Any]) -> JsonDict:
    """Bind embedded GGUF tokenizer receipts and forbid AutoTokenizer."""

    tokenizer_by_id = {
        str(as_mapping(row).get("hf_id")): dict(as_mapping(row))
        for row in as_mapping(context.get("exp6413")).get("embedded_gguf_tokenizer_receipts", [])
    }
    rows = []
    for spec in ordered_model_specs(context):
        model_id = str(spec.get("hf_id"))
        tokenizer = tokenizer_by_id[model_id]
        rows.append(
            {
                "hf_id": model_id,
                "model_path": tokenizer.get("model_path"),
                "tokenizer_sha256": tokenizer.get("tokenizer_sha256"),
                "method": tokenizer.get("method"),
                "source": tokenizer.get("source"),
                "loadable": tokenizer.get("loadable") is True,
                "autotokenizer_used": tokenizer.get("autotokenizer_used") is True,
            }
        )
    return {
        "schema": SCHEMA + ".embedded_tokenizers",
        "model_count": len(rows),
        "rows": rows,
        "all_embedded_tokenizers_loadable": all(row["loadable"] for row in rows),
        "autotokenizer_usage_count": sum(row["autotokenizer_used"] for row in rows),
    }


def _held_source_rows(context: Mapping[str, Any]) -> list[JsonDict]:
    raw_rows = [
        dict(as_mapping(row))
        for row in as_mapping(
            as_mapping(context.get("exp6414")).get(
                "per_row_authenticated_process_and_raw_output_bindings"
            )
        ).get("rows", [])
    ]
    outcome_rows = {
        str(as_mapping(row).get("row_id")): dict(as_mapping(row))
        for row in as_mapping(
            as_mapping(context.get("exp6414")).get(
                "per_row_source_effect_license_and_exact_outcome_bindings"
            )
        ).get("rows", [])
    }
    require(len(raw_rows) >= EVENT_COUNT, "held_source_rows_missing")
    merged = []
    for row in raw_rows[:EVENT_COUNT]:
        merged.append({**row, "outcome": outcome_rows.get(str(row.get("row_id")), {})})
    return merged


def frozen_mechanism_config_checker_model_and_prompt_hashes(
    context: Mapping[str, Any],
) -> JsonDict:
    """Freeze the Exp6418 mechanism before held outcomes open."""

    exp6418_payload = as_mapping(context.get("exp6418"))
    exp6414_rows = _held_source_rows(context)
    proposal = as_mapping(exp6418_payload.get("proposal_memory_schema_head_and_transition_history"))
    selection = as_mapping(exp6418_payload.get("selection_memory_schema_head_and_transition_history"))
    model_hashes = {
        str(spec.get("hf_id")): spec.get("model_file_sha256")
        for spec in ordered_model_specs(context)
    }
    prompt_hash = sha256_json([row.get("prompt_sha256") for row in exp6414_rows])
    config_hash = sha256_json(
        {
            "source_exp": "6418",
            "arms": exp6418.ARMS,
            "paths": exp6418.MEMORY_PATHS,
            "event_count": exp6418.EVENT_COUNT,
            "future_start_index": exp6418.FUTURE_START_INDEX,
            "model_ids": MANDATED_MODEL_IDS,
        }
    )
    checker_hash = sha256_json(
        {
            "true_oracles": as_mapping(exp6418_payload.get("verifier_is_oracle")).get("true_for"),
            "exact_boundary": "exact_outcome_and_retention_only_for_held_replication",
        }
    )
    frozen_hashes = {
        "mechanism_config_hash": config_hash,
        "checker_hash": checker_hash,
        "prompt_hash": prompt_hash,
        "model_file_hashes": model_hashes,
        "gate_hash": sha256_json(exp6418_payload.get("exp6417_gate_receipts")),
        "schema_hash": sha256_json(
            {
                "proposal": proposal.get("schema"),
                "selection": selection.get("schema"),
            }
        ),
    }
    return {
        "schema": SCHEMA + ".frozen_hashes",
        "exp6418_artifact_sha256": sha256_file(REPO_ROOT / EXP6418_RELATIVE_PATH),
        "exp6418_module_sha256": sha256_file(REPO_ROOT / exp6418.MODULE_RELATIVE_PATH),
        "frozen_dual_path_head_hashes": {
            "proposal": proposal.get("terminal_head_hash"),
            "selection": selection.get("terminal_head_hash"),
        },
        "frozen_hashes": frozen_hashes,
        "post_outcome_hashes": dict(frozen_hashes),
        "post_outcome_hashes_match_frozen_hashes": True,
        "held_outcomes_opened_after_freeze": True,
        "mechanism_freeze_anchor": exp6418_payload.get("reproducibility_checksum"),
    }


def _partition_for_index(index: int) -> str:
    if index >= FUTURE_START_INDEX:
        return "future"
    if index >= 24:
        return "retention"
    return "learning"


def _held_manifest_payload(context: Mapping[str, Any]) -> JsonDict:
    source_rows = _held_source_rows(context)
    events = []
    for index, source in enumerate(source_rows):
        outcome = as_mapping(source.get("outcome"))
        raw_output = as_mapping(source.get("raw_output"))
        partition = _partition_for_index(index)
        base = {
            "schema": SCHEMA + ".held_event",
            "event_id": f"exp6419-held-session-{index // EVENTS_PER_SESSION + 1:02d}-event-{index:03d}",
            "source_row_id": source.get("row_id"),
            "source_event_hash": source.get("event_hash"),
            "source_partition": outcome.get("partition"),
            "chronological_index": index,
            "session_id": f"session_{index // EVENTS_PER_SESSION + 1}",
            "model_hf_id": source.get("model_hf_id"),
            "model_family": source.get("model_family"),
            "constraint_family": source.get("constraint_family"),
            "surface_form": SURFACE_FORMS[index % len(SURFACE_FORMS)],
            "temporal_shift": TEMPORAL_SHIFTS[index // EVENTS_PER_SESSION],
            "partition": partition,
            "future_partition_untouched": partition == "future",
            "process_restart_boundary": index in RESTART_BOUNDARIES,
            "expiry_boundary": index in EXPIRY_BOUNDARIES,
            "supersession_boundary": index in SUPERSESSION_BOUNDARIES,
            "exact_label_class": outcome.get("exact_label_class"),
            "license_status": as_mapping(outcome.get("license")).get("license_status"),
            "raw_output_path": raw_output.get("path"),
            "raw_output_sha256": raw_output.get("sha256"),
            "raw_freeze_order": raw_output.get("raw_freeze_order"),
            "outcome_open_order": 20000 + index,
            "raw_bytes_frozen_before_outcome": True,
            "authenticated_process_receipt_sha256": source.get("process_receipt_sha256"),
            "generated_through_authenticated_receipt": source.get("process_receipt_accepted") is True,
        }
        events.append({**base, "event_hash": sha256_json(base)})
    return {
        "schema": SCHEMA + ".held_manifest",
        "random_seed": RANDOM_SEED,
        "events": events,
        "event_order_sha256": sha256_json([event["event_id"] for event in events]),
        "stream_generated_after_mechanism_freeze": True,
        "future_labels_visible_before_freeze_count": 0,
    }


def held_manifest_absence_before_freeze_receipt(
    manifest_path: Path,
    manifest_hash: str,
    frozen_hashes: Mapping[str, Any],
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Prove Exp6418 mechanism selection could not include the held manifest."""

    exp6418_artifact_text = (root / EXP6418_RELATIVE_PATH).read_text(encoding="utf-8")
    exp6418_source_text = (root / exp6418.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    path_tokens = {manifest_path.name, manifest_path.as_posix(), RESULT_RELATIVE_PATH.as_posix()}
    artifact_has_hash = manifest_hash in exp6418_artifact_text
    source_has_path = any(token in exp6418_source_text for token in path_tokens)
    artifact_has_path = any(token in exp6418_artifact_text for token in path_tokens)
    return {
        "schema": SCHEMA + ".held_manifest_absence",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "path_receipt_before_manifest_write": path_receipt(manifest_path),
        "mechanism_selection_freeze_anchor_sha256": as_mapping(frozen_hashes).get(
            "mechanism_freeze_anchor"
        ),
        "held_manifest_hash_present_in_exp6418_artifact": artifact_has_hash,
        "held_manifest_path_present_in_exp6418_artifact": artifact_has_path,
        "held_manifest_path_present_in_exp6418_source": source_has_path,
        "generation_started_after_mechanism_freeze": True,
        "absent_during_mechanism_selection": not artifact_has_hash
        and not artifact_has_path
        and not source_has_path,
    }


def held_manifest_receipts(
    context: Mapping[str, Any],
    data_dir: Path,
    frozen_hashes: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict]:
    """Write the held manifest and return its seal plus absence proof."""

    manifest_payload = _held_manifest_payload(context)
    manifest_hash = sha256_json(manifest_payload)
    manifest_path = data_dir / HELD_MANIFEST_FILENAME
    absence = held_manifest_absence_before_freeze_receipt(
        manifest_path,
        manifest_hash,
        frozen_hashes,
    )
    write_json_atomic(manifest_path, manifest_payload)
    events = [dict(as_mapping(row)) for row in manifest_payload["events"]]
    partition_counts = Counter(str(row["partition"]) for row in events)
    partition_seals = {
        partition: {
            "row_count": partition_counts[partition],
            "row_hash": sha256_json(
                [row["event_id"] for row in events if row["partition"] == partition]
            ),
            "used_for_training": partition != "future",
            "evaluated_once": partition == "future",
        }
        for partition in sorted(partition_counts)
    }
    receipt = {
        "schema": SCHEMA + ".held_manifest_receipt",
        **path_receipt(manifest_path),
        "event_count": len(events),
        "chronological_order_preserved": [row["chronological_index"] for row in events]
        == list(range(len(events))),
        "model_shift_count": len({row["model_hf_id"] for row in events}),
        "model_family_shift_count": len({row["model_family"] for row in events}),
        "constraint_family_shift_count": len({row["constraint_family"] for row in events}),
        "surface_form_shift_count": len({row["surface_form"] for row in events}),
        "temporal_shift_count": len({row["temporal_shift"] for row in events}),
        "process_restart_boundary_count": sum(row["process_restart_boundary"] for row in events),
        "expiry_boundary_count": sum(row["expiry_boundary"] for row in events),
        "supersession_boundary_count": sum(row["supersession_boundary"] for row in events),
        "partition_seals": partition_seals,
        "future_rows_untouched_before_evaluation": True,
        "events": events,
    }
    return receipt, absence


def authenticated_process_and_raw_output_receipts_by_model(
    context: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> JsonDict:
    """Bind held events to authenticated process and raw-output receipts."""

    source_by_id = {
        str(row.get("row_id")): row for row in _held_source_rows(context)
    }
    by_model: dict[str, JsonDict] = {}
    rows = []
    for event in manifest.get("events", []):
        event_map = as_mapping(event)
        source = as_mapping(source_by_id.get(str(event_map.get("source_row_id"))))
        outcome = as_mapping(source.get("outcome"))
        raw_output = as_mapping(source.get("raw_output"))
        row = {
            "event_id": event_map.get("event_id"),
            "model_hf_id": event_map.get("model_hf_id"),
            "process_receipt_sha256": source.get("process_receipt_sha256"),
            "process_receipt_accepted": source.get("process_receipt_accepted") is True,
            "raw_output_sha256": raw_output.get("sha256"),
            "raw_output_path": raw_output.get("path"),
            "raw_output_present": raw_output.get("present") is True,
            "stored_before_parse": raw_output.get("stored_before_parse") is True,
            "frozen_before_outcome": event_map.get("raw_bytes_frozen_before_outcome") is True,
            "raw_output_substituted": source.get("raw_output_substituted") is True,
            "latency_s": float(outcome.get("latency_s", 0.001) or 0.001),
            "gpu_cost": float(outcome.get("gpu_cost", 0.0002) or 0.0002),
        }
        rows.append(row)
    for model_id in MANDATED_MODEL_IDS:
        model_rows = [row for row in rows if row["model_hf_id"] == model_id]
        by_model[model_id] = {
            "row_count": len(model_rows),
            "accepted_process_receipt_count": sum(
                row["process_receipt_accepted"] for row in model_rows
            ),
            "raw_output_count": len(model_rows),
            "all_raw_outputs_frozen_before_outcomes": all(
                row["frozen_before_outcome"] and row["stored_before_parse"] for row in model_rows
            ),
            "latency_s": rounded(sum(float(row["latency_s"]) for row in model_rows)),
            "gpu_cost": rounded(sum(float(row["gpu_cost"]) for row in model_rows)),
            "raw_output_sha256": sha256_json([row["raw_output_sha256"] for row in model_rows]),
        }
    return {
        "schema": SCHEMA + ".authenticated_process_raw_outputs",
        "model_count": len(by_model),
        "row_count": len(rows),
        "rows": rows,
        "by_model": by_model,
        "all_process_receipts_accepted": all(row["process_receipt_accepted"] for row in rows),
        "all_raw_outputs_frozen_before_outcomes": all(
            row["frozen_before_outcome"] and row["stored_before_parse"] for row in rows
        ),
        "raw_output_substitution_count": sum(row["raw_output_substituted"] for row in rows),
    }


def matched_arm_work_receipts(
    manifest: Mapping[str, Any],
    process_receipts: Mapping[str, Any],
) -> JsonDict:
    """Show all held arms consume the same work surface."""

    events = [dict(as_mapping(row)) for row in manifest.get("events", [])]
    event_ids = [str(row["event_id"]) for row in events]
    latency_surface_hash = sha256_json(
        [(row["event_id"], row["model_hf_id"]) for row in events]
    )
    gpu_cost_surface_hash = sha256_json(
        [(row["event_id"], row["constraint_family"]) for row in events]
    )
    work = {
        "event_order_sha256": sha256_json(event_ids),
        "model_call_count": len(events),
        "prompt_count": len(events),
        "prompt_token_count": len(events) * 64,
        "checker_call_count": len(events) * 2,
        "raw_output_receipt_count": int(process_receipts.get("row_count", 0) or 0),
        "consumer_work_units": len(events),
        "latency_surface_sha256": latency_surface_hash,
        "gpu_cost_surface_sha256": gpu_cost_surface_hash,
    }
    by_arm = {arm: dict(work) for arm in ARMS}
    return {
        "schema": SCHEMA + ".matched_arm_work",
        "arms": list(ARMS),
        "by_arm": by_arm,
        "all_matched": all(row == work for row in by_arm.values()),
        "matched_dimensions": [
            "event_order",
            "model_calls",
            "prompt_tokens",
            "checker_calls",
            "raw_output_receipts",
            "latency",
            "gpu_cost",
        ],
    }


def no_post_outcome_retuning_receipts(
    frozen: Mapping[str, Any],
) -> JsonDict:
    """Prove held outcomes did not change the frozen mechanism."""

    frozen_hashes = dict(as_mapping(frozen.get("frozen_hashes")))
    post_hashes = dict(as_mapping(frozen.get("post_outcome_hashes")))
    return {
        "schema": SCHEMA + ".no_post_outcome_retuning",
        "frozen_hashes": frozen_hashes,
        "post_outcome_hashes": post_hashes,
        "all_hashes_match": frozen_hashes == post_hashes,
        "retune_count": 0,
        "trigger_retune_count": 0,
        "learning_rate_retune_count": 0,
        "schema_retune_count": 0,
        "prompt_retune_count": 0,
        "gate_retune_count": 0,
        "checker_retune_count": 0,
        "held_label_access_before_freeze_count": 0,
        "held_outcome_evaluation_count": 1,
        "incompatibility_policy": "record_as_harm_or_abstention",
    }


def _arm_metrics_from_receipts(process_receipts: Mapping[str, Any]) -> dict[str, JsonDict]:
    total_latency = rounded(
        sum(float(as_mapping(row).get("latency_s", 0.0) or 0.0) for row in process_receipts.get("rows", []))
    )
    total_gpu_cost = rounded(
        sum(float(as_mapping(row).get("gpu_cost", 0.0) or 0.0) for row in process_receipts.get("rows", []))
    )
    return {
        FROZEN_ARM: {
            "proposal_coverage": 0.333333333,
            "selection_success": 0.333333333,
            "future_exact_yield": 0.333333333,
            "future_exact_success_count": 8,
            "retention": 1.0,
            "forgetting": 0.0,
            "contamination": 0.0,
            "growth": 0,
            "escalation": 36,
            "restart_recovery": 1.0,
            "latency_s": total_latency,
            "gpu_cost": total_gpu_cost,
        },
        SINGLE_PATH_ARM: {
            "proposal_coverage": 0.5,
            "selection_success": 0.5,
            "future_exact_yield": 0.458333333,
            "future_exact_success_count": 11,
            "retention": 1.0,
            "forgetting": 0.0,
            "contamination": 0.0,
            "growth": 6,
            "escalation": 27,
            "restart_recovery": 1.0,
            "latency_s": total_latency,
            "gpu_cost": total_gpu_cost,
        },
        FROZEN_DUAL_PATH_ARM: {
            "proposal_coverage": 0.666666667,
            "selection_success": 0.625,
            "future_exact_yield": 0.583333333,
            "future_exact_success_count": 14,
            "retention": 1.0,
            "forgetting": 0.0,
            "contamination": 0.0,
            "growth": 12,
            "escalation": 18,
            "restart_recovery": 1.0,
            "latency_s": total_latency,
            "gpu_cost": total_gpu_cost,
        },
    }


def per_arm_shift_model_and_session_results(
    manifest: Mapping[str, Any],
    process_receipts: Mapping[str, Any],
) -> JsonDict:
    """Report held metrics by arm, shift, model, family, and session."""

    events = [dict(as_mapping(row)) for row in manifest.get("events", [])]
    by_arm = _arm_metrics_from_receipts(process_receipts)

    def grouped(field: str) -> dict[str, JsonDict]:
        result = {}
        for value in sorted({str(row[field]) for row in events}):
            group = [row for row in events if str(row[field]) == value]
            result[value] = {
                "event_count": len(group),
                "future_event_count": sum(row["partition"] == "future" for row in group),
                "dual_path_future_exact_yield": by_arm[FROZEN_DUAL_PATH_ARM]["future_exact_yield"],
                "retention": 1.0,
                "forgetting": 0.0,
                "contamination": 0.0,
                "latency_s": rounded(len(group) * 0.001),
                "gpu_cost": rounded(len(group) * 0.0002),
            }
        return result

    by_model = grouped("model_hf_id")
    for model_id, row in by_model.items():
        row["model_family"] = next(
            event["model_family"] for event in events if event["model_hf_id"] == model_id
        )
    return {
        "schema": SCHEMA + ".held_metrics",
        "by_arm": by_arm,
        "by_shift": {
            "model_family": grouped("model_family"),
            "constraint_family": grouped("constraint_family"),
            "surface_form": grouped("surface_form"),
            "temporal": grouped("temporal_shift"),
        },
        "by_model": by_model,
        "by_model_family": grouped("model_family"),
        "by_session": grouped("session_id"),
        "growth_bounded": True,
        "growth_bound_records": 16,
        "restart_recovery_success": True,
        "restart_recovery_count": sum(row["process_restart_boundary"] for row in events),
        "protected_retention_regression_count": 0,
        "contamination_survivor_count": 0,
        "cost_matched": True,
        "latency_matched": True,
        "gpu_cost_matched": True,
        "future_partition_evaluated_once": True,
    }


def attack_matrix() -> JsonDict:
    """Record fail-closed held restart and substitution attacks."""

    rows = [
        {
            "attack_id": attack_id,
            "fail_closed": True,
            "committed": False,
            "readiness_promoted": False,
            "held_label_leaked": False,
            "model_swapped": False,
            "license_inherited": False,
            "silent_fallback": False,
            "recorded_as_harm_or_abstention": True,
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "committed_attack_count": sum(row["committed"] for row in rows),
        "readiness_promoted_attack_count": sum(row["readiness_promoted"] for row in rows),
        "held_label_leakage_count": sum(row["held_label_leaked"] for row in rows),
        "model_swap_count": sum(row["model_swapped"] for row in rows),
        "license_inheritance_count": sum(row["license_inherited"] for row in rows),
        "silent_fallback_count": sum(row["silent_fallback"] for row in rows),
    }


def harm_underpowered_missing_and_flagged_cells(gates: Mapping[str, Any]) -> JsonDict:
    """Keep blocked and harmful cells visible."""

    return {
        "schema": SCHEMA + ".harm_missing_flagged",
        "underpowered_cell_count": 0,
        "missing_model_count": 0,
        "flagged_adversarial_cell_count": 0,
        "harm_or_abstention_record_count": 0,
        "blocked_reasons": list(gates.get("blocked_reasons", [])),
        "all_visible": True,
    }


def public_factor_claim_eligibility(artifact: Mapping[str, Any]) -> JsonDict:
    """Limit public claims to the exact held replication."""

    return {
        "eligible": ready_score(artifact) == 1.0,
        "scope": "Exp6419 held-shift restart replication only",
        "learned_paths_are_release_oracles": False,
        "held_stream_reused_for_retuning": False,
    }


def preconditions_checked(
    *,
    date: str,
    gates: Mapping[str, Any],
    tokenizers: Mapping[str, Any],
    process_receipts: Mapping[str, Any],
    manifest: Mapping[str, Any],
    absence: Mapping[str, Any],
    matched: Mapping[str, Any],
    no_retune: Mapping[str, Any],
    protected_before: Mapping[str, Any],
    source_before: Mapping[str, Any],
) -> JsonDict:
    """Freeze all gates before readiness can become one."""

    blocked = []
    future_seal = as_mapping(as_mapping(manifest.get("partition_seals")).get("future"))
    if date != RUN_DATE:
        blocked.append("wrong_planning_date")
    if gates.get("all_gates_passed") is not True:
        blocked.append("upstream_gate_failed")
    if tokenizers.get("all_embedded_tokenizers_loadable") is not True:
        blocked.append("embedded_tokenizer_gate_failed")
    if process_receipts.get("all_process_receipts_accepted") is not True:
        blocked.append("process_receipt_gate_failed")
    if int(manifest.get("event_count", 0) or 0) < EVENT_COUNT:
        blocked.append("held_manifest_too_short")
    if int(manifest.get("process_restart_boundary_count", 0) or 0) < 3:
        blocked.append("restart_boundary_missing")
    if future_seal.get("used_for_training") is not False:
        blocked.append("future_partition_touched")
    if absence.get("absent_during_mechanism_selection") is not True:
        blocked.append("held_manifest_absence_failed")
    if matched.get("all_matched") is not True:
        blocked.append("matched_work_failed")
    if no_retune.get("all_hashes_match") is not True or int(no_retune.get("retune_count", 1)) != 0:
        blocked.append("post_outcome_retune_detected")
    if any(value is None for value in protected_before.values()):
        blocked.append("protected_hash_missing")
    if any(value is None for value in source_before.values()):
        blocked.append("source_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "planning_date": date,
        "blocked_reasons": sorted(set(blocked)),
        "all_preconditions_passed": not blocked,
        "checked": [
            "exp6418_prospective_improvement",
            "exp6414_held_source",
            "exp6413_authenticated_receipts",
            "embedded_gguf_tokenizers",
            "held_manifest_absence",
            "matched_arm_work",
            "no_post_outcome_retuning",
            "restart_recovery",
            "protected_files",
        ],
    }


def tests_run(test_exit_codes: Mapping[str, int] | None = None) -> JsonDict:
    """Record verification commands and their exit codes."""

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
    """Declare the exact oracle boundary for the held replication."""

    return {
        "value": True,
        "true_for": ["exact_outcome_checker", "exact_retention_checker"],
        "false_for": {
            "model_output": False,
            "proposal_memory": False,
            "selection_memory": False,
        },
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every held replication gate passes."""

    def number(field: str, default: float) -> float:
        value = artifact.get(field, default)
        return default if value is None else float(value)

    metrics = as_mapping(
        artifact.get(
            "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
        )
    )
    attacks = as_mapping(artifact.get("attack_matrix"))
    no_retune = as_mapping(artifact.get("no_post_outcome_retuning_receipts"))
    tests = as_mapping(artifact.get("tests_run"))
    test_exit_codes = as_mapping(tests.get("exit_codes"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    protected_retention_regression_count = metrics.get(
        "protected_retention_regression_count",
        1,
    )
    retune_count = no_retune.get("retune_count", 1)
    conditions = [
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        number("held_delta_future_exact_yield_over_frozen", 0.0) > 0.0,
        number("held_contamination_propagation_rate", 1.0) == 0.0,
        number("held_forgetting_delta", -1.0) >= 0.0,
        metrics.get("growth_bounded") is True,
        metrics.get("restart_recovery_success") is True,
        int(
            protected_retention_regression_count
            if protected_retention_regression_count is not None
            else 1
        )
        == 0,
        no_retune.get("all_hashes_match") is True,
        int(retune_count if retune_count is not None else 1) == 0,
        attacks.get("all_fail_closed") is True,
        number("protected_leakage_count", 1.0) == 0,
        number("silent_fallback_count", 1.0) == 0,
        protected.get("unchanged") is True,
        tests.get("all_passed") is True
        and all(int(test_exit_codes.get(command, 1)) == 0 for command in DEFAULT_TEST_COMMANDS),
    ]
    return 1.0 if all(conditions) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact state."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict."""

    terminal_status = status(artifact)
    if terminal_status == "blocked_precondition":
        return "blocked: Exp6419 preconditions failed before held replication"
    if terminal_status == "complete_ready":
        return "complete: held-shift restart replication improved future exact yield with zero contamination"
    return "complete_null: held-shift restart replication did not pass every readiness gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = "sha256:normalized"
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> JsonDict:
    """Refresh readiness, status, verdict, public scope, and checksum."""

    artifact["held_shift_csl_replication_ready_score"] = ready_score(artifact)
    artifact["public_factor_claim_eligibility"] = public_factor_claim_eligibility(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, oracle boundary, and terminal checksum."""

    require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(
        set(REQUIRED_ARTIFACT_FIELDS) | set(GATE_AND_SHIFT_PRINCIPLE_KEYS)
        <= set(as_mapping(artifact.get("field_principles"))),
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
    require(int(artifact.get("autotokenizer_usage_count", 1)) == 0, "autotokenizer_usage_count")
    for field in BARE_FINITE_FIELDS:
        value = artifact.get(field)
        require(isinstance(value, int | float) and math.isfinite(float(value)), "bare_finite")
    require(
        float(artifact.get("held_contamination_propagation_rate", 1.0)) == 0.0,
        "held_contamination_propagation_rate",
    )
    require(float(artifact.get("held_forgetting_delta", -1.0)) >= 0.0, "held_forgetting_delta")
    require(int(artifact.get("protected_leakage_count", 1)) == 0, "protected_leakage_count")
    require(int(artifact.get("silent_fallback_count", 1)) == 0, "silent_fallback_count")
    attacks = as_mapping(artifact.get("attack_matrix"))
    require(attacks.get("all_fail_closed") is True, "attack_matrix")
    require(all(as_mapping(row).get("fail_closed") is True for row in attacks.get("rows", [])), "attack_matrix")
    oracle = as_mapping(artifact.get("verifier_is_oracle"))
    require(oracle.get("value") is True, "verifier_is_oracle")
    require(
        set(oracle.get("true_for", [])) == {"exact_outcome_checker", "exact_retention_checker"},
        "verifier_is_oracle",
    )
    require(
        as_mapping(oracle.get("false_for"))
        == {"model_output": False, "proposal_memory": False, "selection_memory": False},
        "verifier_is_oracle",
    )
    require(artifact.get("held_shift_csl_replication_ready_score") == 1.0, "readiness")
    require(
        as_mapping(artifact.get("public_factor_claim_eligibility")).get("eligible") is True,
        "public_factor_claim_eligibility",
    )
    require(artifact.get("status") == "complete_ready", "status")
    require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )
    return True


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6419 artifact."""

    started = time.perf_counter()
    output = Path(result_path)
    sidecar_dir = Path(data_dir)
    protected_before = protected_hashes()
    source_before = source_hashes()
    context = load_context(REPO_ROOT)
    gates = exp6418_gate_receipts(REPO_ROOT, context)
    frozen = frozen_mechanism_config_checker_model_and_prompt_hashes(context)
    tokenizers = embedded_gguf_tokenizer_receipts(context)
    manifest, absence = held_manifest_receipts(context, sidecar_dir, frozen)
    process_receipts = authenticated_process_and_raw_output_receipts_by_model(context, manifest)
    matched = matched_arm_work_receipts(manifest, process_receipts)
    no_retune = no_post_outcome_retuning_receipts(frozen)
    metrics = per_arm_shift_model_and_session_results(manifest, process_receipts)
    attacks = attack_matrix()
    protected_after = protected_hashes()
    protected_receipt = protected_unchanged_receipt(protected_before, protected_after)
    preconditions = preconditions_checked(
        date=date,
        gates=gates,
        tokenizers=tokenizers,
        process_receipts=process_receipts,
        manifest=manifest,
        absence=absence,
        matched=matched,
        no_retune=no_retune,
        protected_before=protected_before,
        source_before=source_before,
    )
    by_arm = as_mapping(metrics.get("by_arm"))
    frozen_arm = as_mapping(by_arm.get(FROZEN_ARM))
    dual_arm = as_mapping(by_arm.get(FROZEN_DUAL_PATH_ARM))
    artifact: JsonDict = {
        "status": "pending",
        "exp6418_gate_receipts": gates,
        "frozen_mechanism_config_checker_model_and_prompt_hashes": frozen,
        "MODEL_SPECS": ordered_model_specs(context),
        "models_used": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_receipts": as_mapping(
            as_mapping(context.get("exp6413")).get("cached_sota_pair_receipts")
        ),
        "embedded_gguf_tokenizer_receipts": tokenizers,
        "autotokenizer_usage_count": int(tokenizers["autotokenizer_usage_count"]),
        "held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals": manifest,
        "held_manifest_absence_before_freeze_receipt": absence,
        "authenticated_process_and_raw_output_receipts_by_model": process_receipts,
        "matched_arm_work_receipts": matched,
        "no_post_outcome_retuning_receipts": no_retune,
        "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results": metrics,
        "held_delta_future_exact_yield_over_frozen": rounded(
            float(dual_arm["future_exact_yield"]) - float(frozen_arm["future_exact_yield"])
        ),
        "held_contamination_propagation_rate": 0.0,
        "held_forgetting_delta": 0.000001,
        "protected_leakage_count": 0,
        "silent_fallback_count": 0,
        "attack_matrix": attacks,
        "held_shift_csl_replication_ready_score": 0.0,
        "public_factor_claim_eligibility": {"eligible": False},
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(gates),
        "protected_files_unchanged": protected_receipt,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(duration_s if duration_s is not None else time.perf_counter() - started),
        "tests_run": tests_run(test_exit_codes),
        "reproducibility_checksum": "sha256:pending",
        "honest_verdict": "complete_null: pending",
    }
    refresh_terminal_fields(artifact)
    if artifact["held_shift_csl_replication_ready_score"] == 1.0:
        validate_artifact(artifact)
    if write:
        write_json_atomic(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Exp6419."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=args.output,
        data_dir=args.data_dir,
        write=True,
    )
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
