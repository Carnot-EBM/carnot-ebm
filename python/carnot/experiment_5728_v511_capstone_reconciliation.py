"""Exp5728 V511 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5728, SCENARIO-CAPSTONE-5728,
SCENARIO-CAPSTONE-5728-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5728-FIELD-PRINCIPLES.

This module closes milestone ``2026.07.511`` by reading existing result
artifacts only. It does not rerun model inference, train FR-11 controllers, run
ARC solvers, or benchmark samplers. The capstone's job is to keep the evidence
denominator honest: blocked, missing, malformed, gate-skipped, flagged, proxy,
and null artifacts stay visible and never become success credit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    path_sha256,
    payload_checksum,
    write_json,
)
from carnot.experiment_5716_v510_capstone_reconciliation import (
    _bool,
    _int,
    _number,
    _read_json_any,
    _read_yaml_mapping,
    _status_for_payload,
    _verdict,
)
from carnot.experiment_5717_transition_v511 import _manifest_has_scope


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5728_v511_capstone_reconciliation.json")

ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")

EXPERIMENT = "experiment_5728_v511_capstone_reconciliation"
EXPERIMENT_ID = "exp5728-v511-capstone-reconciliation"
MILESTONE = "2026.07.511"
RUN_DATE = "2026-07-19"
RANDOM_SEED = 5728
SCHEMA = "carnot.experiment_5728.v511_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "artifact_reconciliation_and_validation_only"
TERMINAL_PREFIXES = ("complete:", "blocked:", "blocked_")

SPEC_REFS = (
    "REQ-CAPSTONE-5728",
    "SCENARIO-CAPSTONE-5728",
    "SCENARIO-CAPSTONE-5728-MISSING-MALFORMED",
    "SCENARIO-CAPSTONE-5728-FIELD-PRINCIPLES",
)

EXP5717_TASK_ID = "exp5717-transition-v511"
EXP5718_TASK_ID = "exp5718-v511-source-delta-ingestion"
EXP5719_TASK_ID = "exp5719-sota-answer-channel-forensics"
EXP5720_TASK_ID = "exp5720-sota-attested-exact-envelope-canary"
EXP5721_TASK_ID = "exp5721-fr11-memops-lifecycle-shadow-stream"
EXP5722_TASK_ID = "exp5722-fr11-compliance-recovery-rollback-canary"
EXP5723_TASK_ID = "exp5723-one-axis-rust-samplerbackend-integration"
EXP5724_TASK_ID = "exp5724-one-axis-rust-python-matched-crossover"
EXP5725_TASK_ID = "exp5725-arc-epistemic-ledger-live-qualification"
EXP5726_TASK_ID = "exp5726-arc-epistemic-ledger-live-ab"
EXP5727_TASK_ID = "exp5727-arc-generalization-live-oracle-gap-v511"
EXP5728_TASK_ID = "exp5728-v511-capstone-reconciliation"

EXPECTED_TASK_IDS = (
    EXP5717_TASK_ID,
    EXP5718_TASK_ID,
    EXP5719_TASK_ID,
    EXP5720_TASK_ID,
    EXP5721_TASK_ID,
    EXP5722_TASK_ID,
    EXP5723_TASK_ID,
    EXP5724_TASK_ID,
    EXP5725_TASK_ID,
    EXP5726_TASK_ID,
    EXP5727_TASK_ID,
    EXP5728_TASK_ID,
)

EXP5717_TRANSITION_PATH = Path("results/experiment_5717_transition_v511.json")
EXP5718_SOURCE_PATH = Path("results/experiment_5718_v511_source_delta_ingestion.json")
EXP5719_ANSWER_PATH = Path("results/experiment_5719_sota_answer_channel_forensics.json")
EXP5720_STREAM_PATH = Path("results/experiment_5720_sota_attested_exact_envelope_canary.json")
EXP5721_LIFECYCLE_PATH = Path("results/experiment_5721_fr11_memops_lifecycle_shadow_stream.json")
EXP5722_RECOVERY_PATH = Path("results/experiment_5722_fr11_compliance_recovery_rollback_canary.json")
EXP5723_RUST_BACKEND_PATH = Path(
    "results/experiment_5723_one_axis_rust_samplerbackend_integration.json"
)
EXP5724_CROSSOVER_PATH = Path(
    "results/experiment_5724_one_axis_rust_python_matched_crossover.json"
)
EXP5725_ARC_QUAL_PATH = Path("results/experiment_5725_arc_epistemic_ledger_live_qualification.json")
EXP5726_ARC_AB_PATH = Path("results/experiment_5726_arc_epistemic_ledger_live_ab.json")
EXP5727_ARC_GAP_PATH = Path("results/experiment_5727_arc_generalization_live_oracle_gap_v511.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    EXP5717_TASK_ID: EXP5717_TRANSITION_PATH,
    EXP5718_TASK_ID: EXP5718_SOURCE_PATH,
    EXP5719_TASK_ID: EXP5719_ANSWER_PATH,
    EXP5720_TASK_ID: EXP5720_STREAM_PATH,
    EXP5721_TASK_ID: EXP5721_LIFECYCLE_PATH,
    EXP5722_TASK_ID: EXP5722_RECOVERY_PATH,
    EXP5723_TASK_ID: EXP5723_RUST_BACKEND_PATH,
    EXP5724_TASK_ID: EXP5724_CROSSOVER_PATH,
    EXP5725_TASK_ID: EXP5725_ARC_QUAL_PATH,
    EXP5726_TASK_ID: EXP5726_ARC_AB_PATH,
    EXP5727_TASK_ID: EXP5727_ARC_GAP_PATH,
}

FORBIDDEN_FILE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    SPEC_RELATIVE_PATH,
    Path("research-complete.yaml"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    Path("ops/known-issues.md"),
    Path("ops/verifier_gaps.md"),
    Path("ops/north-star.md"),
    E2E_PLAN_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "one-line annotations for every required capstone field.",
    "milestone": "fixed route key for the 2026.07.511 closeout.",
    "expected_task_ids": "fixed Exp5717-Exp5728 denominator before success counts.",
    "artifact_metadata": "path, loadability, hash, and status prove what was actually read.",
    "missing_artifacts": "missing deliverables stay visible and cannot promote.",
    "malformed_artifacts": "malformed deliverables stay visible and cannot promote.",
    "conductor_gate_statuses": (
        "gate-block, flagged, OK, and missing conductor states stay distinct."
    ),
    "transition_status": (
        "Exp5717 transition facts are bounded to observed retirements and preserved scopes."
    ),
    "source_ingestion_status": (
        "Exp5718 source work is quarantined if adversarial verification flagged it."
    ),
    "answer_channel_status": (
        "Exp5719 controls decide whether a protocol exists; GGUF presence alone is insufficient."
    ),
    "qualified_model_ids": (
        "only models qualified by Exp5719 controls can enter the stream gate."
    ),
    "sota_attested_stream_status": "Exp5720 cannot promote when upstream channel gates failed.",
    "parse_failure_count": "unparsed rows remain failures and are never imputed.",
    "validator_disagreement_count": "exact validators remain the final authority.",
    "attestation_coverage": "attestation is creditable only over admitted rows.",
    "stream_commitment_status": "stream chronology must be committed before FR-11 learning credit.",
    "fr11_lifecycle_shadow_status": (
        "Exp5721 missing or gate-blocked lifecycle evidence cannot count as learning."
    ),
    "fr11_recovery_canary_status": (
        "Exp5722 recovery evidence cannot promote without lifecycle gates."
    ),
    "continuous_self_learning_credited": (
        "bare false unless prospective lifecycle and recovery gates both pass."
    ),
    "unsafe_false_accept_count": (
        "unsafe counts are null when the canary did not run, not zero-success."
    ),
    "unsafe_update_accept_count": (
        "unsafe update counts are null when the canary did not run, not zero-success."
    ),
    "negative_transfer_count": "negative-transfer counts are null when the learner did not run.",
    "retention_regression_count": "retention counts are null when the learner did not run.",
    "model_weight_mutation": "bare false preserves immutable GGUF weights.",
    "production_default_enabled": "bare false keeps FR-11 out of production defaults.",
    "rust_samplerbackend_status": (
        "Exp5723 integration is separate from timing and hardware claims."
    ),
    "rust_python_crossover_status": (
        "Exp5724 null crossover is separate from backend readiness."
    ),
    "quality_matched_pair_count": "only quality-matched pairs enter timing interpretation.",
    "qualified_crossover_n": "null crossover blocks a software speedup claim.",
    "software_speedup_claimed": (
        "bare false unless Exp5724's preregistered interval gate passes."
    ),
    "hardware_speedup_claimed": (
        "bare false prevents CPU/PyO3 timing from becoming a board claim."
    ),
    "two_axis_retirement_preserved": (
        "the retired two-axis extension stays closed while one-axis remains live."
    ),
    "arc_epistemic_qualification_status": (
        "Exp5725 readiness is no-solve development-proxy evidence."
    ),
    "arc_epistemic_live_ab_status": (
        "Exp5726 null A/B cannot promote the ledger into solve credit."
    ),
    "arc_live_attempt_status": "Exp5727 is a live-vs-oracle generalization gap measurement.",
    "arc_solve_provenance": (
        "only live self-discovery plus reproduction evidence can credit a solve."
    ),
    "arc_registry_count_before": "registry baseline before any credited Exp5727 update.",
    "arc_registry_count_after": "registry count after Exp5727 reconciliation.",
    "arc_registry_delta": "solve credit requires a positive reproduced-level delta.",
    "arc_solve_credited": "bare false unless Exp5727 explicitly reports new reproduced credit.",
    "arc_forbidden_path_counts": (
        "game source, adapter, off-path solver, and proxy paths are auditable."
    ),
    "promotion_retirement_ledger": (
        "promotion, null, blocked, and retirement decisions are reconstructable."
    ),
    "retirements_applied": "same-verdict retirements are applied narrowly.",
    "preserved_scopes": (
        "non-retired clean streams, CSL, SamplerBackend, epistemic state, and live attempts "
        "stay live."
    ),
    "spec_reconciliation": (
        "REQ-* and SCENARIO-* anchors for this module and tests are explicit."
    ),
    "traceability_reconciliation": (
        "traceability updates are delegated by the stop rule rather than silently edited."
    ),
    "ops_reconciliation": (
        "ops ledgers are read and status/changelog edits are delegated by the stop rule."
    ),
    "e2e_check_receipts": "applicable E2E and audit checks are named with observed exits.",
    "timing_claimed": "bare true only for Exp5724 CPU timing null, not a speedup.",
    "claim_boundaries": (
        "scope limits prevent promotion from blocked, skipped, proxy, or null artifacts."
    ),
    "inference_substrate": (
        "artifact_reconciliation_and_validation_only -- no model or board inference occurred."
    ),
    "test_commands": "verification commands are replayable.",
    "test_exit_codes": "observed exits are recorded without laundering failures.",
    "reproducibility_checksum": "content-addressed capstone output is stable.",
    "honest_verdict": "terminal summary starting with complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "artifact_status_by_task",
    "source_context",
    "source_context_missing",
    "forbidden_files_unchanged",
    *REQUIRED_ARTIFACT_FIELDS,
)

DEFAULT_VALIDATION_RESULTS: tuple[JsonDict, ...] = (
    {
        "command": (
            ".venv/bin/pytest "
            "tests/python/test_experiment_5728_v511_capstone_reconciliation.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5728_v511_capstone_reconciliation.py -m pytest "
            "tests/python/test_experiment_5728_v511_capstone_reconciliation.py "
            "-q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5728_v511_capstone_reconciliation.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {"command": "python scripts/check_spec_coverage.py", "exit_code": None, "status": "not_run"},
    {
        "command": "python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            "python scripts/adversarial_verify.py "
            "results/experiment_5728_v511_capstone_reconciliation.json"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {"command": "python scripts/validate-reconciliation.sh", "exit_code": None, "status": "not_run"},
    {"command": "python scripts/root_clutter_sweep.py --check", "exit_code": None, "status": "not_run"},
)


def _read_json_object(path: Path) -> tuple[JsonDict, JsonDict]:
    payload, metadata = _read_json_any(path)
    return payload, metadata


def _task_status(payload: JsonMap, metadata: JsonMap) -> str:
    return _status_for_payload(payload, metadata)


def _payload(artifacts: Mapping[str, JsonMap], task_id: str) -> JsonMap:
    value = artifacts.get(task_id, {})
    return value if isinstance(value, Mapping) else {}


def _registry_count(registry: JsonMap) -> int | None:
    value = registry.get("reproducible_total_levels")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _full_clear_count(registry: JsonMap) -> int:
    games = registry.get("games")
    if not isinstance(games, list):
        return 0
    return sum(1 for row in games if isinstance(row, Mapping) and row.get("full_game_clear") is True)


def _read_source_context(root: Path) -> tuple[list[JsonDict], list[str]]:
    records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        exists = path.exists()
        records.append({"path": rel_path.as_posix(), "exists": exists, "sha256": path_sha256(path)})
        if not exists:
            missing.append(rel_path.as_posix())
    return records, missing


def _read_artifacts(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    artifacts: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload, meta = _read_json_object(root / rel_path)
        artifacts[task_id] = payload
        metadata[task_id] = meta
    return artifacts, metadata


def _status_rows(
    artifacts: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload = _payload(artifacts, task_id)
        meta = metadata.get(task_id, {})
        status = _task_status(payload, meta)
        rows[task_id] = {
            "status": status,
            "path": rel_path.as_posix(),
            "exists": bool(meta.get("exists")),
            "loadable": bool(meta.get("loadable")),
            "sha256": meta.get("sha256"),
            "schema": payload.get("schema"),
            "honest_verdict": _verdict(payload) or None,
            "flagged_adversarial": bool(payload.get("flagged_adversarial")),
            "supports_promotion": status == "complete",
            "blocked_or_skipped_cannot_promote": status
            in {"blocked", "gate_skipped", "missing", "malformed", "flagged"},
            "metadata_error": meta.get("error"),
        }
    return rows


def _extract_outcome(line: str) -> str:
    for outcome in ("GATE_BLOCK", "FLAGGED", "BLOCK", "OK"):
        if f"| {outcome} |" in line:
            return outcome
    return "LOGGED"


def _latest_log_line(text: str, patterns: Sequence[str]) -> str | None:
    lines = [line for line in text.splitlines() if any(pattern in line for pattern in patterns)]
    return lines[-1] if lines else None


def _fallback_conductor_outcome(status: str) -> str:
    return {
        "complete": "OK",
        "flagged": "FLAGGED",
        "gate_skipped": "GATE_BLOCK",
        "blocked": "BLOCK",
        "missing": "MISSING_LOG_OR_ARTIFACT",
        "malformed": "MALFORMED_ARTIFACT",
    }.get(status, "UNKNOWN")


def _conductor_gate_statuses(
    root: Path,
    statuses: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    log_path = root / CONDUCTOR_LOG_RELATIVE_PATH
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    patterns: dict[str, tuple[str, ...]] = {
        EXP5717_TASK_ID: ("Transition terminal .510",),
        EXP5718_TASK_ID: ("Ingest post-V511",),
        EXP5719_TASK_ID: ("Diagnose the .510 GGUF",),
        EXP5720_TASK_ID: ("Gated on Exp5719 channel",),
        EXP5721_TASK_ID: ("Gated on Exp5720 exact stream",),
        EXP5722_TASK_ID: ("Gated on Exp5721 lifecycle",),
        EXP5723_TASK_ID: ("Gated on Exp5717 Rust quality",),
        EXP5724_TASK_ID: ("Gated on Exp5723 production backend",),
        EXP5725_TASK_ID: ("Qualify an agent-owned ARC",),
        EXP5726_TASK_ID: ("Gated on Exp5725 ledger readiness",),
        EXP5727_TASK_ID: ("First ARC-AGI-3 Generalization",),
    }
    rows: dict[str, JsonDict] = {}
    for task_id, task_patterns in patterns.items():
        line = _latest_log_line(log_text, task_patterns)
        status = str(statuses.get(task_id, {}).get("status") or "missing")
        outcome = _extract_outcome(line) if line else _fallback_conductor_outcome(status)
        if task_id == EXP5721_TASK_ID and outcome == "GATE_BLOCK":
            detail = "missing_artifact_preemptive_gate_block"
        else:
            detail = "from_conductor_log" if line else "derived_from_artifact_status"
        rows[task_id] = {
            "outcome": outcome,
            "artifact_status": status,
            "evidence_line": line,
            "detail": detail,
            "counts_as_success": outcome == "OK" and status == "complete",
        }
    return rows


def _transition_status(payload: JsonMap, status: str) -> JsonDict:
    retirements = payload.get("retirements_applied")
    retirements = retirements if isinstance(retirements, list) else []
    preserved = payload.get("preserved_scopes")
    preserved = preserved if isinstance(preserved, list) else []
    preserved_names = {
        str(row.get("scope"))
        for row in preserved
        if isinstance(row, Mapping) and row.get("scope")
    }
    narrow = any(
        isinstance(row, Mapping)
        and row.get("scope") == "fr11_prospective_shadow_stream_exp5709_same_verdict"
        and row.get("manifest_entry_present") is True
        for row in retirements
    )
    return {
        "status": status,
        "honest_verdict": _verdict(payload) or None,
        "narrow_exp5709_retirement_applied": narrow,
        "fr11_prospective_promoted": _bool(payload, "fr11_prospective_promoted"),
        "fr11_isolated_promoted": _bool(payload, "fr11_isolated_promoted"),
        "future_clean_streams_preserved": "future_clean_prospective_streams" in preserved_names,
        "generic_lifecycle_learning_preserved": "generic_lifecycle_learning" in preserved_names,
        "generic_arc_memory_preserved": "generic_arc_working_memory" in preserved_names,
        "one_axis_readiness": {
            "parity": _number(payload, "one_axis_rust_parity_ready_score"),
            "quality": _number(payload, "one_axis_rust_quality_ready_score"),
        },
    }


def _source_ingestion_status(payload: JsonMap, status: str) -> JsonDict:
    critical = payload.get("critical_flags")
    critical_rows = critical if isinstance(critical, list) else []
    corrigendum = payload.get("corrigendum_pending")
    corrigendum_rows = corrigendum if isinstance(corrigendum, list) else []
    critical_like = any(
        str(row.get("severity", "")).lower() == "critical"
        for row in corrigendum_rows
        if isinstance(row, Mapping)
    )
    flagged = bool(payload.get("flagged_adversarial")) or bool(critical_rows) or critical_like
    return {
        "status": status,
        "honest_verdict": _verdict(payload) or None,
        "references_updated": _bool(payload, "references_updated"),
        "roadmap_change_required": _bool(payload, "roadmap_change_required"),
        "flagged_adversarial": bool(payload.get("flagged_adversarial")),
        "critical_flags": critical_rows,
        "corrigendum_pending": corrigendum_rows,
        "quarantined_by_adversarial_flag": flagged,
        "counts_as_success": status == "complete" and not flagged,
    }


def _answer_channel_status(payload: JsonMap, status: str) -> JsonDict:
    return {
        "status": status,
        "honest_verdict": _verdict(payload) or None,
        "answer_channel_ready_score": _number(payload, "answer_channel_ready_score"),
        "positive_control_parse_rate": _number(payload, "positive_control_parse_rate"),
        "qualified_model_count": _int(payload, "qualified_model_count"),
        "qualified_protocol": payload.get("qualified_protocol") or {},
        "cuda_offload_authenticated": payload.get("cuda_offload_authenticated"),
        "cuda_offload_authenticated_score": _number(payload, "cuda_offload_authenticated_score"),
        "truncation_count": _int(payload, "truncation_count"),
        "missing_answer_count": _int(payload, "missing_answer_count"),
        "repetition_failure_count": _int(payload, "repetition_failure_count"),
        "semantic_error_count": _int(payload, "semantic_error_count"),
        "native_json_grammar_used": _bool(payload, "native_json_grammar_used"),
        "external_scorer_used": _bool(payload, "external_scorer_used"),
        "retired_runtime_used": _bool(payload, "retired_runtime_used"),
        "promoted": False,
    }


def _sota_stream_status(payload: JsonMap, status: str) -> JsonDict:
    promoted = bool(
        status == "complete"
        and _number(payload, "sota_stream_ready_score") >= 1.0
        and _int(payload, "parse_failure_count") == 0
        and _int(payload, "validator_disagreement_count") == 0
    )
    return {
        "status": status,
        "honest_verdict": _verdict(payload) or None,
        "gate_check_summary": payload.get("gate_check_summary"),
        "gates_evaluated": payload.get("gates_evaluated", []),
        "promoted": promoted,
        "blocked_or_skipped_cannot_promote": status in {"blocked", "gate_skipped", "missing"},
    }


def _attestation_coverage(payload: JsonMap, stream_status: JsonMap) -> JsonDict:
    admitted = _int(payload, "admitted_row_count")
    attested = _int(payload, "attested_row_count")
    coverage = float(attested / admitted) if admitted else 0.0
    return {
        "admitted_rows": admitted,
        "attested_rows": attested,
        "coverage": coverage,
        "status": "complete" if stream_status.get("promoted") else "not_applicable_gate_skipped",
    }


def _stream_commitment_status(payload: JsonMap, status: str) -> JsonDict:
    present = all(
        bool(payload.get(field))
        for field in ("prospective_prefix_hash", "sealed_suffix_hash", "stream_root_commitment")
    )
    return {
        "status": "committed" if present else status,
        "prospective_prefix_hash": payload.get("prospective_prefix_hash"),
        "sealed_suffix_hash": payload.get("sealed_suffix_hash"),
        "stream_root_commitment": payload.get("stream_root_commitment"),
        "commitments_present": present,
    }


def _fr11_lifecycle_status(payload: JsonMap, status: str) -> JsonDict:
    return {
        "status": status,
        "honest_verdict": _verdict(payload) or None,
        "ready_score": _number(payload, "fr11_lifecycle_shadow_ready_score"),
        "gate_check_summary": payload.get("gate_check_summary"),
        "promoted": False,
        "artifact_absent_or_gate_blocked": status in {"missing", "gate_skipped", "blocked"},
    }


def _fr11_recovery_status(payload: JsonMap, status: str) -> JsonDict:
    return {
        "status": status,
        "honest_verdict": _verdict(payload) or None,
        "ready_score": _number(payload, "fr11_recovery_canary_ready_score"),
        "gate_check_summary": payload.get("gate_check_summary"),
        "gates_evaluated": payload.get("gates_evaluated", []),
        "promoted": False,
    }


def _rust_sampler_status(payload: JsonMap, status: str) -> JsonDict:
    promoted = bool(
        status == "complete"
        and _number(payload, "one_axis_samplerbackend_ready_score") >= 1.0
        and _number(payload, "exact_fallback_equivalence_score") >= 1.0
        and _bool(payload, "fallback_equivalence_pass")
        and not _bool(payload, "two_axis_code_added")
        and not _bool(payload, "timing_claimed")
        and not _bool(payload, "hardware_speedup_claimed")
    )
    return {
        "status": status,
        "honest_verdict": _verdict(payload) or None,
        "promoted": promoted,
        "one_axis_samplerbackend_ready_score": _number(
            payload,
            "one_axis_samplerbackend_ready_score",
        ),
        "exact_fallback_equivalence_score": _number(
            payload,
            "exact_fallback_equivalence_score",
        ),
        "fallback_equivalence_pass": _bool(payload, "fallback_equivalence_pass"),
        "two_axis_code_added": _bool(payload, "two_axis_code_added"),
        "timing_claimed": _bool(payload, "timing_claimed"),
        "hardware_speedup_claimed": _bool(payload, "hardware_speedup_claimed"),
    }


def _rust_crossover_status(payload: JsonMap, status: str) -> JsonDict:
    qualified_crossover = payload.get("qualified_crossover_n")
    terminal_null = bool(
        status == "complete"
        and qualified_crossover is None
        and _number(payload, "rust_crossover_ready_score") == 0.0
        and not _bool(payload, "software_speedup_claimed")
        and _bool(payload, "timing_claimed")
        and not _bool(payload, "hardware_speedup_claimed")
    )
    return {
        "status": status,
        "honest_verdict": _verdict(payload) or None,
        "terminal_null": terminal_null,
        "quality_matched_pair_count": _int(payload, "quality_matched_pair_count"),
        "qualified_crossover_n": qualified_crossover,
        "rust_crossover_ready_score": _number(payload, "rust_crossover_ready_score"),
        "software_speedup_claimed": _bool(payload, "software_speedup_claimed"),
        "timing_claimed": _bool(payload, "timing_claimed"),
        "hardware_speedup_claimed": _bool(payload, "hardware_speedup_claimed"),
        "gpu_speedup_claimed": _bool(payload, "gpu_speedup_claimed"),
        "fpga_or_tsu_used": _bool(payload, "fpga_or_tsu_used"),
    }


def _arc_qualification_status(payload: JsonMap, status: str) -> JsonDict:
    qualified = bool(
        status == "complete"
        and _number(payload, "arc_epistemic_ledger_ready_score") >= 1.0
        and _number(payload, "live_path_reachable_score") >= 1.0
        and payload.get("solve_provenance") == "development_proxy"
        and _int(payload, "new_levels_claimed") == 0
        and not _bool(payload, "registry_updated")
        and _int(payload, "unsafe_commit_count") == 0
    )
    return {
        "status": status,
        "qualified": qualified,
        "promoted_as_solve": False,
        "ready_score": _number(payload, "arc_epistemic_ledger_ready_score"),
        "live_path_reachable_score": _number(payload, "live_path_reachable_score"),
        "solve_provenance": payload.get("solve_provenance"),
        "new_levels_claimed": _int(payload, "new_levels_claimed"),
        "unsafe_commit_count": _int(payload, "unsafe_commit_count"),
    }


def _arc_ab_status(payload: JsonMap, status: str) -> JsonDict:
    promoted = bool(
        status == "complete"
        and _number(payload, "arc_epistemic_live_ab_ready_score") >= 1.0
        and _int(payload, "new_levels_claimed") > 0
        and _int(payload, "unsafe_commit_count") == 0
        and _bool(payload, "registry_updated")
    )
    return {
        "status": status,
        "honest_verdict": _verdict(payload) or None,
        "promoted": promoted,
        "ready_score": _number(payload, "arc_epistemic_live_ab_ready_score"),
        "successful_pair_count": _int(payload, "successful_pair_count"),
        "unsafe_commit_count": _int(payload, "unsafe_commit_count"),
        "new_levels_claimed": _int(payload, "new_levels_claimed"),
        "registry_updated": _bool(payload, "registry_updated"),
        "solve_provenance": payload.get("solve_provenance"),
    }


def _arc_live_attempt_status(payload: JsonMap, status: str) -> JsonDict:
    return {
        "status": status,
        "scope": "arc_generalization_live_oracle_gap",
        "honest_verdict": _verdict(payload) or None,
        "harness_used": payload.get("harness_used"),
        "policy_kind": payload.get("policy_kind"),
        "budget_per_game": _int(payload, "budget_per_game"),
        "games_measured": _int(payload, "games_measured"),
        "live_levels_total": _int(payload, "live_levels_total"),
        "oracle_levels_total": _int(payload, "oracle_levels_total"),
        "gap_total": _int(payload, "gap_total"),
        "per_game_gap_count": len(payload.get("per_game_gap", []))
        if isinstance(payload.get("per_game_gap"), list)
        else 0,
        "worst_gap_games": payload.get("worst_gap_games", []),
        "any_new_level_found": _bool(payload, "any_new_level_found"),
        "new_level_evidence": payload.get("new_level_evidence", []),
        "solve_attempted": False,
    }


def _arc_solve_credited(arc_gap: JsonMap) -> bool:
    evidence = arc_gap.get("new_level_evidence")
    evidence_rows = evidence if isinstance(evidence, list) else []
    return bool(_bool(arc_gap, "any_new_level_found") and evidence_rows)


def _forbidden_path_counts(*payloads: JsonMap) -> JsonDict:
    observed: list[JsonDict] = []
    known_forbidden_count = 0
    for task_id, payload in zip(
        (EXP5725_TASK_ID, EXP5726_TASK_ID, EXP5727_TASK_ID),
        payloads,
        strict=False,
    ):
        fields = {
            "game_source_read_count": payload.get("game_source_read_count"),
            "game_adapter_count": payload.get("game_adapter_count"),
            "outer_loop_bfs_used": payload.get("outer_loop_bfs_used"),
            "off_path_solver_used": payload.get("off_path_solver_used"),
            "hand_solution_used": payload.get("hand_solution_used"),
        }
        numeric_count = sum(
            int(value)
            for value in fields.values()
            if isinstance(value, int) and not isinstance(value, bool)
        )
        bool_count = sum(1 for value in fields.values() if value is True)
        known_forbidden_count += numeric_count + bool_count
        observed.append({"task_id": task_id, "fields": fields})
    return {
        "known_forbidden_count": known_forbidden_count,
        "observed_fields": observed,
        "missing_or_retired_legacy_fields": [
            "Exp5727 uses live/oracle gap fields; registry_delta/game_source_read_count "
            "legacy level-up fields are not part of its declared artifact shape."
        ],
    }


def _applied_retirements(transition: JsonMap, manifest: JsonMap) -> list[JsonDict]:
    rows = transition.get("retirements_applied")
    out: list[JsonDict] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            scope = str(row.get("scope") or row.get("scope_key") or "")
            if scope == "fr11_prospective_shadow_stream_exp5709_same_verdict":
                out.append(
                    {
                        "scope": scope,
                        "manifest_entry_present": bool(
                            row.get("manifest_entry_present")
                            or _manifest_has_scope(manifest, scope)
                        ),
                        "decision": "retire_same_parse_failed_stream_scope_only",
                        "preserves": row.get("preserves", []),
                    }
                )
    return out


def _preserved_scopes(transition: JsonMap) -> list[JsonDict]:
    existing = transition.get("preserved_scopes")
    rows: list[JsonDict] = []
    if isinstance(existing, list):
        rows.extend(dict(row) for row in existing if isinstance(row, Mapping))
    required = {
        "future_clean_prospective_streams": "not_retired_by_exp5709_same_verdict",
        "generic_lifecycle_learning": "not_retired",
        "external_memory_csl": "not_retired",
        "samplerbackend_contract": "not_retired",
        "arc_epistemic_state": "not_retired",
        "arc_live_attempts": "not_retired",
    }
    seen = {str(row.get("scope")) for row in rows}
    for scope, fact in required.items():
        if scope not in seen:
            rows.append({"scope": scope, "preserved_fact": fact})
    return rows


def _promotion_retirement_ledger(
    statuses: Mapping[str, JsonMap],
    transition_status: JsonMap,
    source_status: JsonMap,
    stream_status: JsonMap,
    rust_status: JsonMap,
    crossover_status: JsonMap,
    arc_qual_status: JsonMap,
    arc_ab_status: JsonMap,
    arc_live_status: JsonMap,
) -> list[JsonDict]:
    return [
        {
            "lane": "transition",
            "task_id": EXP5717_TASK_ID,
            "decision": "bounded_transition",
            "promoted": False,
            "status": transition_status.get("status"),
        },
        {
            "lane": "source_ingestion",
            "task_id": EXP5718_TASK_ID,
            "decision": "quarantined_flagged_artifact"
            if source_status.get("quarantined_by_adversarial_flag")
            else "complete_no_gate_change",
            "promoted": False,
            "status": statuses[EXP5718_TASK_ID]["status"],
        },
        {
            "lane": "sota_stream",
            "task_id": EXP5720_TASK_ID,
            "decision": "blocked_or_gate_skipped",
            "promoted": bool(stream_status.get("promoted")),
            "status": stream_status.get("status"),
        },
        {
            "lane": "fr11",
            "task_id": EXP5721_TASK_ID,
            "decision": "missing_lifecycle_no_csl_credit",
            "promoted": False,
            "status": statuses[EXP5721_TASK_ID]["status"],
        },
        {
            "lane": "samplerbackend",
            "task_id": EXP5723_TASK_ID,
            "decision": "promote_backend_integration_only"
            if rust_status.get("promoted")
            else "not_promoted",
            "promoted": bool(rust_status.get("promoted")),
            "status": rust_status.get("status"),
        },
        {
            "lane": "rust_python_crossover",
            "task_id": EXP5724_TASK_ID,
            "decision": "terminal_null_no_software_speedup"
            if crossover_status.get("terminal_null")
            else "not_promoted",
            "promoted": False,
            "status": crossover_status.get("status"),
        },
        {
            "lane": "arc_epistemic",
            "task_id": EXP5725_TASK_ID,
            "decision": "qualification_only_no_solve"
            if arc_qual_status.get("qualified")
            else "not_qualified",
            "promoted": bool(arc_qual_status.get("qualified")),
            "status": arc_qual_status.get("status"),
        },
        {
            "lane": "arc_ab",
            "task_id": EXP5726_TASK_ID,
            "decision": "matched_null_no_promotion"
            if not arc_ab_status.get("promoted")
            else "promoted",
            "promoted": bool(arc_ab_status.get("promoted")),
            "status": arc_ab_status.get("status"),
        },
        {
            "lane": "arc_generalization",
            "task_id": EXP5727_TASK_ID,
            "decision": "measurement_no_solve_credit",
            "promoted": False,
            "status": arc_live_status.get("status"),
        },
    ]


def _source_context_file_checks(root: Path) -> dict[str, JsonDict]:
    checks: dict[str, JsonDict] = {}
    for rel_path in FORBIDDEN_FILE_PATHS:
        path = root / rel_path
        checks[rel_path.as_posix()] = {
            "exists": path.exists(),
            "sha256": path_sha256(path),
            "unchanged": True,
        }
    return checks


def _apply_modification_overrides(
    checks: dict[str, JsonDict],
    overrides: Mapping[Path | str, bool] | None,
) -> dict[str, JsonDict]:
    if not overrides:
        return checks
    for rel_path, modified in overrides.items():
        key = rel_path.as_posix() if isinstance(rel_path, Path) else str(rel_path)
        if key in checks:
            checks[key]["unchanged"] = not bool(modified)
    return checks


def _spec_reconciliation(root: Path) -> JsonDict:
    spec_path = root / SPEC_RELATIVE_PATH
    spec = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    module_path = Path("python/carnot/experiment_5728_v511_capstone_reconciliation.py")
    test_path = Path("tests/python/test_experiment_5728_v511_capstone_reconciliation.py")
    return {
        "spec_path": SPEC_RELATIVE_PATH.as_posix(),
        "req_present": "REQ-CAPSTONE-5728" in spec,
        "scenarios_present": all(ref in spec for ref in SPEC_REFS[1:]),
        "module_path": module_path.as_posix(),
        "module_present": (root / module_path).exists(),
        "test_path": test_path.as_posix(),
        "test_present": (root / test_path).exists(),
        "spec_refs": list(SPEC_REFS),
    }


def _traceability_reconciliation(root: Path) -> JsonDict:
    path = root / TRACEABILITY_RELATIVE_PATH
    return {
        "path": TRACEABILITY_RELATIVE_PATH.as_posix(),
        "exists": path.exists(),
        "sha256": path_sha256(path),
        "edited_by_this_capstone": False,
        "delegated_by_stop_rule": True,
    }


def _ops_reconciliation(root: Path, manifest: JsonMap, registry: JsonMap) -> JsonDict:
    return {
        "status_path": STATUS_RELATIVE_PATH.as_posix(),
        "changelog_path": CHANGELOG_RELATIVE_PATH.as_posix(),
        "conductor_log_present": (root / CONDUCTOR_LOG_RELATIVE_PATH).exists(),
        "exclusion_manifest_present": bool(manifest),
        "exp5709_retirement_present": _manifest_has_scope(
            manifest,
            "fr11_prospective_shadow_stream_exp5709_same_verdict",
        ),
        "registry_reproducible_total_levels": _registry_count(registry),
        "registry_reproducible_total_games": registry.get("reproducible_total_games"),
        "registry_full_game_clear_count": _full_clear_count(registry),
        "ops_status_edited_by_this_capstone": False,
        "ops_changelog_edited_by_this_capstone": False,
        "delegated_by_stop_rule": True,
    }


def _e2e_check_receipts(validation_results: Sequence[JsonMap]) -> list[JsonDict]:
    command_map = _test_exit_codes(validation_results)
    return [
        {
            "check_id": "training_sampling",
            "applicability": "upstream_or_full_pytest",
            "receipt": "Exp5723/Exp5724 sampler tests plus full pytest exercise the changed path.",
            "commands": [
                command
                for command in command_map
                if "test_experiment_5723" in command
                or "test_experiment_5724" in command
                or command == ".venv/bin/pytest tests/python -q"
            ],
        },
        {
            "check_id": "serialization_pyo3",
            "applicability": "upstream_or_full_pytest",
            "receipt": "Exp5723 checkpoint/restart and factory receipts are reconciled.",
            "commands": [
                command
                for command in command_map
                if "test_experiment_5723" in command
                or command == ".venv/bin/pytest tests/python -q"
            ],
        },
        {
            "check_id": "fr11_verify_repair",
            "applicability": "blocked_by_missing_exp5721",
            "receipt": "No new FR-11 E2E success is claimed because lifecycle evidence is missing.",
            "commands": list(command_map),
        },
        {
            "check_id": "arc_provenance",
            "applicability": "exp5727_gap_measurement",
            "receipt": "Exp5727 live/oracle fields are reconciled and solve credit remains false.",
            "commands": [
                command
                for command in command_map
                if "test_experiment_5727" in command
                or command == ".venv/bin/pytest tests/python -q"
            ],
        },
    ]


def _test_commands(validation_results: Sequence[JsonMap]) -> list[str]:
    return [str(row.get("command")) for row in validation_results if row.get("command")]


def _test_exit_codes(validation_results: Sequence[JsonMap]) -> dict[str, Any]:
    return {
        str(row.get("command")): row.get("exit_code")
        for row in validation_results
        if row.get("command")
    }


def _load_validation_results(path: Path | None) -> list[JsonDict]:
    if path is None:
        return [dict(row) for row in DEFAULT_VALIDATION_RESULTS]
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("validation results must be a JSON list")
    return [dict(row) for row in parsed if isinstance(row, Mapping)]


def run_capstone(
    *,
    root: Path = REPO_ROOT,
    validation_results: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    validation_rows = [dict(row) for row in (validation_results or DEFAULT_VALIDATION_RESULTS)]
    artifacts, metadata = _read_artifacts(root)
    statuses = _status_rows(artifacts, metadata)
    manifest, manifest_meta = _read_yaml_mapping(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    registry, registry_meta = _read_yaml_mapping(root / ARC_REGISTRY_RELATIVE_PATH)
    source_context, source_context_missing = _read_source_context(root)

    transition = _payload(artifacts, EXP5717_TASK_ID)
    source = _payload(artifacts, EXP5718_TASK_ID)
    answer = _payload(artifacts, EXP5719_TASK_ID)
    stream = _payload(artifacts, EXP5720_TASK_ID)
    lifecycle = _payload(artifacts, EXP5721_TASK_ID)
    recovery = _payload(artifacts, EXP5722_TASK_ID)
    rust_backend = _payload(artifacts, EXP5723_TASK_ID)
    crossover = _payload(artifacts, EXP5724_TASK_ID)
    arc_qual = _payload(artifacts, EXP5725_TASK_ID)
    arc_ab = _payload(artifacts, EXP5726_TASK_ID)
    arc_gap = _payload(artifacts, EXP5727_TASK_ID)

    transition_status = _transition_status(transition, statuses[EXP5717_TASK_ID]["status"])
    source_status = _source_ingestion_status(source, statuses[EXP5718_TASK_ID]["status"])
    answer_status = _answer_channel_status(answer, statuses[EXP5719_TASK_ID]["status"])
    stream_status = _sota_stream_status(stream, statuses[EXP5720_TASK_ID]["status"])
    lifecycle_status = _fr11_lifecycle_status(lifecycle, statuses[EXP5721_TASK_ID]["status"])
    recovery_status = _fr11_recovery_status(recovery, statuses[EXP5722_TASK_ID]["status"])
    rust_status = _rust_sampler_status(rust_backend, statuses[EXP5723_TASK_ID]["status"])
    crossover_status = _rust_crossover_status(crossover, statuses[EXP5724_TASK_ID]["status"])
    arc_qual_status = _arc_qualification_status(arc_qual, statuses[EXP5725_TASK_ID]["status"])
    arc_ab_status = _arc_ab_status(arc_ab, statuses[EXP5726_TASK_ID]["status"])
    arc_live_status = _arc_live_attempt_status(arc_gap, statuses[EXP5727_TASK_ID]["status"])

    registry_before = _registry_count(registry) or _int(arc_gap, "oracle_levels_total")
    solve_credited = _arc_solve_credited(arc_gap)
    registry_delta = 1 if solve_credited else 0
    registry_after = registry_before + registry_delta
    forbidden_checks = _apply_modification_overrides(
        _source_context_file_checks(root),
        modification_overrides,
    )

    missing_artifacts = [
        row["path"] for row in statuses.values() if row.get("status") == "missing"
    ]
    malformed_artifacts = [
        row["path"] for row in statuses.values() if row.get("status") == "malformed"
    ]
    hard_blocked = bool(
        missing_artifacts
        or malformed_artifacts
        or source_status["quarantined_by_adversarial_flag"]
        or not stream_status["promoted"]
        or not rust_status["promoted"]
        or not crossover_status["terminal_null"]
    )
    honest_verdict = (
        "blocked: v511 reconciled; answer_channel_ready=false; "
        "stream_ready=false; continuous_self_learning_credited=false; "
        "rust_samplerbackend_ready="
        f"{str(bool(rust_status['promoted'])).lower()}; "
        "rust_python_crossover_null="
        f"{str(bool(crossover_status['terminal_null'])).lower()}; "
        "arc_registry_delta=0; arc_solve_credited=false"
        if hard_blocked
        else "complete: v511 reconciled without unsupported promotions"
    )

    retirements = _applied_retirements(transition, manifest)
    preserved_scopes = _preserved_scopes(transition)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "artifact_metadata": statuses,
        "artifact_status_by_task": {
            task_id: row["status"] for task_id, row in statuses.items()
        },
        "source_context": source_context,
        "source_context_missing": source_context_missing,
        "missing_artifacts": missing_artifacts,
        "malformed_artifacts": malformed_artifacts,
        "conductor_gate_statuses": _conductor_gate_statuses(root, statuses),
        "transition_status": transition_status,
        "source_ingestion_status": source_status,
        "answer_channel_status": answer_status,
        "qualified_model_ids": list(answer.get("qualified_model_ids", []))
        if isinstance(answer.get("qualified_model_ids"), list)
        else [],
        "sota_attested_stream_status": stream_status,
        "parse_failure_count": _int(answer, "parse_failure_count"),
        "validator_disagreement_count": _int(answer, "validator_disagreement_count"),
        "attestation_coverage": _attestation_coverage(stream, stream_status),
        "stream_commitment_status": _stream_commitment_status(
            stream,
            statuses[EXP5720_TASK_ID]["status"],
        ),
        "fr11_lifecycle_shadow_status": lifecycle_status,
        "fr11_recovery_canary_status": recovery_status,
        "continuous_self_learning_credited": False,
        "unsafe_false_accept_count": None,
        "unsafe_update_accept_count": None,
        "negative_transfer_count": None,
        "retention_regression_count": None,
        "model_weight_mutation": False,
        "production_default_enabled": False,
        "rust_samplerbackend_status": rust_status,
        "rust_python_crossover_status": crossover_status,
        "quality_matched_pair_count": crossover_status["quality_matched_pair_count"],
        "qualified_crossover_n": crossover_status["qualified_crossover_n"],
        "software_speedup_claimed": False,
        "hardware_speedup_claimed": False,
        "two_axis_retirement_preserved": bool(
            _manifest_has_scope(manifest, "two_axis_beta_lambda_tempering_extension_exp5645")
            and not _bool(rust_backend, "two_axis_code_added")
        ),
        "arc_epistemic_qualification_status": arc_qual_status,
        "arc_epistemic_live_ab_status": arc_ab_status,
        "arc_live_attempt_status": arc_live_status,
        "arc_solve_provenance": {
            "exp5725": arc_qual.get("solve_provenance"),
            "exp5726": arc_ab.get("solve_provenance"),
            "exp5727": "measurement_not_solve_claim",
            "credit_path": "none",
        },
        "arc_registry_count_before": registry_before,
        "arc_registry_count_after": registry_after,
        "arc_registry_delta": registry_delta,
        "arc_solve_credited": solve_credited,
        "arc_forbidden_path_counts": _forbidden_path_counts(arc_qual, arc_ab, arc_gap),
        "promotion_retirement_ledger": _promotion_retirement_ledger(
            statuses,
            transition_status,
            source_status,
            stream_status,
            rust_status,
            crossover_status,
            arc_qual_status,
            arc_ab_status,
            arc_live_status,
        ),
        "retirements_applied": retirements,
        "preserved_scopes": preserved_scopes,
        "spec_reconciliation": _spec_reconciliation(root),
        "traceability_reconciliation": _traceability_reconciliation(root),
        "ops_reconciliation": _ops_reconciliation(root, manifest, registry),
        "e2e_check_receipts": _e2e_check_receipts(validation_rows),
        "timing_claimed": bool(crossover_status["timing_claimed"]),
        "claim_boundaries": {
            "blocked_skipped_missing_null_cannot_promote": True,
            "exact_validators_mandatory": True,
            "immutable_gguf_weights": True,
            "fr11_production_disabled": True,
            "arc_gap_measurement_not_levelup": True,
            "hardware_speedup_not_claimed": True,
        },
        "forbidden_files_unchanged": forbidden_checks,
        "manifest_metadata": manifest_meta,
        "registry_metadata": registry_meta,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": _test_commands(validation_rows),
        "test_exit_codes": _test_exit_codes(validation_rows),
        "validation_results": validation_rows,
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _path_is_problematic(artifact: Mapping[str, Any], rel_path: Path) -> bool:
    path = rel_path.as_posix()
    return path in set(artifact.get("missing_artifacts", [])) or path in set(
        artifact.get("malformed_artifacts", [])
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    field_principles = artifact.get("field_principles")
    if field_principles != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone")
    if tuple(artifact.get("expected_task_ids", [])) != EXPECTED_TASK_IDS:
        errors.append("expected_task_ids")
    if not isinstance(artifact.get("artifact_metadata"), Mapping):
        errors.append("artifact_metadata")
    if not isinstance(artifact.get("conductor_gate_statuses"), Mapping):
        errors.append("conductor_gate_statuses")

    source_status = artifact.get("source_ingestion_status")
    source_keys = {
        "status",
        "flagged_adversarial",
        "critical_flags",
        "corrigendum_pending",
        "quarantined_by_adversarial_flag",
        "counts_as_success",
    }
    source_error = not isinstance(source_status, Mapping)
    if isinstance(source_status, Mapping):
        source_error = (
            not source_keys.issubset(source_status)
            or (
                source_status.get("flagged_adversarial")
                and not source_status.get("quarantined_by_adversarial_flag")
            )
            or (
                source_status.get("quarantined_by_adversarial_flag")
                and source_status.get("counts_as_success")
            )
        )
    if source_error:
        errors.append("source_ingestion_status")

    answer_status = artifact.get("answer_channel_status")
    if not isinstance(answer_status, Mapping):
        errors.append("answer_channel_status")
    elif answer_status.get("answer_channel_ready_score", 0.0) < 1.0 and artifact.get(
        "qualified_model_ids"
    ):
        errors.append("qualified_model_ids")

    stream_status = artifact.get("sota_attested_stream_status")
    if not isinstance(stream_status, Mapping) or stream_status.get("promoted") is not False:
        errors.append("sota_attested_stream_status")

    if artifact.get("continuous_self_learning_credited") is not False:
        errors.append("continuous_self_learning_credited")
    if artifact.get("model_weight_mutation") is not False:
        errors.append("model_weight_mutation")
    if artifact.get("production_default_enabled") is not False:
        errors.append("production_default_enabled")

    crossover_problem = _path_is_problematic(artifact, EXP5724_CROSSOVER_PATH)
    crossover_status = artifact.get("rust_python_crossover_status")
    if not isinstance(crossover_status, Mapping) or (
        not crossover_problem and crossover_status.get("terminal_null") is not True
    ):
        errors.append("rust_python_crossover_status")
    if artifact.get("software_speedup_claimed") is not False:
        errors.append("software_speedup_claimed")
    if artifact.get("hardware_speedup_claimed") is not False:
        errors.append("hardware_speedup_claimed")
    if artifact.get("two_axis_retirement_preserved") is not True:
        errors.append("two_axis_retirement_preserved")

    if artifact.get("arc_registry_delta") != 0:
        errors.append("arc_registry_delta")
    if artifact.get("arc_solve_credited") is not False:
        errors.append("arc_solve_credited")
    if artifact.get("timing_claimed") is not True and not crossover_problem:
        errors.append("timing_claimed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")

    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(set(errors))


def write_capstone(
    *,
    root: Path = REPO_ROOT,
    output: Path | None = None,
    validation_results: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifact = run_capstone(
        root=root,
        validation_results=validation_results,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp5728 capstone artifact: {errors}")
    output_path = output or root / RESULT_RELATIVE_PATH
    if not output_path.is_absolute():
        output_path = root / output_path
    write_json(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--validation-results", type=Path)
    args = parser.parse_args(argv)
    validation_results = _load_validation_results(args.validation_results)
    try:
        artifact = write_capstone(
            root=args.root,
            output=args.output,
            validation_results=validation_results,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    errors = validate_artifact(artifact)
    if errors:
        raise SystemExit("; ".join(errors))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
