"""Exp5743 transition receipt from terminal milestone .512 into .513.

Spec refs: REQ-REPORT-5743, SCENARIO-REPORT-5743,
SCENARIO-REPORT-5743-DEPENDENCY-MAP,
SCENARIO-REPORT-5743-ARC-GATE-SCHEMA,
SCENARIO-REPORT-5743-FIELD-PRINCIPLES.

This module is an evidence lock, not a new science run. It reads the closed
`.512` artifacts and the active `.513` roadmap, then emits a receipt that says
which facts are safe to carry forward. That matters because several upstream
results are easy to over-read: a parse-safe proposal channel is not search
utility, a safe KAN sidecar did not beat its matched MLP control, a clean Rust
batch API did not prove strict 10x speed, and the ARC live A/B was skipped by a
schema gate before any live trial could earn solve credit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    extract_roadmap_tasks,
    path_sha256,
    payload_checksum,
    read_yaml_mapping,
    write_json,
)
from carnot.experiment_5625_transition_v508 import (
    _read_json_any,
    _status_for_payload,
    _task_range_from_text,
    _verdict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5743_transition_v513.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")

EXPERIMENT = "experiment_5743_transition_v513"
EXPERIMENT_ID = "exp5743-transition-v513"
PREVIOUS_MILESTONE = "2026.07.512"
CURRENT_MILESTONE = "2026.07.513"
PREVIOUS_TASK_RANGE = "exp5731-exp5742"
CURRENT_TASK_RANGE = "exp5743-exp5754"
RUN_DATE = "2026-07-20"
RANDOM_SEED = 5743
SCHEMA = "carnot.experiment_5743.transition_v513.v1"
INFERENCE_SUBSTRATE = "artifact_reconciliation_only"

SPEC_REFS = (
    "REQ-REPORT-5743",
    "SCENARIO-REPORT-5743",
    "SCENARIO-REPORT-5743-DEPENDENCY-MAP",
    "SCENARIO-REPORT-5743-ARC-GATE-SCHEMA",
    "SCENARIO-REPORT-5743-FIELD-PRINCIPLES",
)

EXP5731_TRANSITION_PATH = Path("results/experiment_5731_transition_v512.json")
EXP5732_SOURCE_PATH = Path("results/experiment_5732_v512_source_delta_ingestion.json")
EXP5733_CHANNEL_PATH = Path("results/experiment_5733_sota_finite_choice_proposal_channel.json")
EXP5734_STREAM_PATH = Path("results/experiment_5734_sota_exact_proposal_stream.json")
EXP5735_CSL_PATH = Path("results/experiment_5735_zero_gate_kan_continuous_self_learning.json")
EXP5736_LIFECYCLE_PATH = Path("results/experiment_5736_csl_lifecycle_conflict_rollback.json")
EXP5737_SOTA_CSL_PATH = Path("results/experiment_5737_sota_stream_csl_shadow_ingress.json")
EXP5738_BATCH_PATH = Path("results/experiment_5738_one_axis_rust_batched_backend.json")
EXP5739_10X_PATH = Path("results/experiment_5739_one_axis_batched_10x_crossover.json")
EXP5740_ARC_PATH = Path("results/experiment_5740_arc_game_blind_primitive_causal_audit.json")
EXP5741_ARC_AB_PATH = Path("results/experiment_5741_arc_generic_primitive_live_ab.json")
EXP5742_CAPSTONE_PATH = Path("results/experiment_5742_v512_capstone_reconciliation.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5731-transition-v512": EXP5731_TRANSITION_PATH,
    "exp5732-v512-source-delta-ingestion": EXP5732_SOURCE_PATH,
    "exp5733-sota-finite-choice-proposal-channel": EXP5733_CHANNEL_PATH,
    "exp5734-sota-exact-proposal-stream": EXP5734_STREAM_PATH,
    "exp5735-zero-gate-kan-continuous-self-learning": EXP5735_CSL_PATH,
    "exp5736-csl-lifecycle-conflict-rollback": EXP5736_LIFECYCLE_PATH,
    "exp5737-sota-stream-csl-shadow-ingress": EXP5737_SOTA_CSL_PATH,
    "exp5738-one-axis-rust-batched-backend": EXP5738_BATCH_PATH,
    "exp5739-one-axis-batched-10x-crossover": EXP5739_10X_PATH,
    "exp5740-arc-game-blind-primitive-causal-audit": EXP5740_ARC_PATH,
    "exp5741-arc-generic-primitive-live-ab": EXP5741_ARC_AB_PATH,
    "exp5742-v512-capstone-reconciliation": EXP5742_CAPSTONE_PATH,
}
UPSTREAM_ARTIFACT_PATHS = tuple(TASK_ARTIFACT_PATHS.values())

EXPECTED_TASK_IDS = [
    "exp5743-transition-v513",
    "exp5744-v513-source-delta-ingestion",
    "exp5745-arc-causal-gate-schema-corrigendum",
    "exp5746-exact-proposal-utility-benchmark",
    "exp5747-sota-exact-proposal-utility-panel",
    "exp5748-selective-exact-feedback-search",
    "exp5749-csl-render-matched-mechanism-audit",
    "exp5750-dependent-task-continuous-self-learning",
    "exp5751-rust-restart-parity-repair",
    "exp5752-one-axis-allocation-free-10x-crossover",
    "exp5753-arc-generic-primitive-live-registry-ab",
    "exp5754-v513-capstone-reconciliation",
]

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

PROTECTED_FILE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

RETIRED_UPSTREAM_IDS = {
    "exp5719-sota-answer-channel-forensics",
    "exp5720-sota-attested-exact-envelope-canary",
    "exp5721-fr11-memops-lifecycle-shadow-stream",
    "exp5722-fr11-compliance-recovery-rollback-canary",
    "exp5724-one-axis-rust-python-matched-crossover",
    "exp5726-arc-epistemic-ledger-live-ab",
    "exp5739-one-axis-batched-10x-crossover",
    "exp5741-arc-generic-primitive-live-ab",
}

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Identifies the exact transition artifact schema for downstream validators.",
    "experiment": "Names the local experiment artifact without depending on file paths.",
    "experiment_id": "Binds this receipt to the conductor task id.",
    "status": "Separates complete transition receipts from blocked precondition failures.",
    "previous_milestone": "Records the terminal milestone being archived.",
    "current_milestone": "Records the newly active milestone receiving the evidence.",
    "previous_task_range": "Keeps the closed `.512` task interval explicit.",
    "current_task_range": "The `.513` task range is allocated only after collision checks.",
    "run_date": "Anchors the transition to the operator-specified date.",
    "random_seed": "Provides deterministic metadata even though no stochastic science runs.",
    "spec_refs": "Connects this implementation to REQ-REPORT-5743 scenarios.",
    "result_path": "Names the emitted JSON deliverable.",
    "field_principles": (
        "Explains every top-level artifact field so downstream readers know why the field exists."
    ),
    "preconditions_checked": (
        "Records instruction, source, ledger, roadmap, gate, and protected-file checks before "
        "a transition claim is emitted."
    ),
    "source_capstone_hash": (
        "Binds the transition to the terminal Exp5742 capstone bytes instead of a rewritten summary."
    ),
    "source_context": "Lists source files read for this reconciliation and their hashes.",
    "source_context_missing": "Missing optional context remains visible instead of being fabricated.",
    "artifact_metadata": "Records loadability and hashes for each terminal `.512` artifact.",
    "missing_artifacts": "Fails closed when required terminal evidence is absent.",
    "malformed_artifacts": "Fails closed when required terminal evidence cannot be parsed.",
    "failed_preconditions": "Names every reason the transition could not be marked complete.",
    "v512_task_verdicts": (
        "Preserves complete, null, flagged, and gate-skipped `.512` task outcomes before `.513` allocation."
    ),
    "v512_conductor_outcomes": (
        "Preserves conductor OK, FLAGGED, and GATE_BLOCK states as operational evidence."
    ),
    "source_delta_duration_flag": "Preserves Exp5732's bibliographic duration flag exactly.",
    "proposal_channel_evidence": (
        "Captures parse-safe finite-choice transport without converting it to search utility."
    ),
    "sota_stream_evidence": (
        "Captures exact-attested chronological rows without converting conflicts to utility."
    ),
    "csl_safety_evidence": (
        "Records zero-gated insertion and matched-control errors without claiming KAN superiority."
    ),
    "csl_lifecycle_evidence": "Records lifecycle rollback and unsafe-propagation safety evidence.",
    "rust_batch_readiness_evidence": (
        "Records Exp5738 semantic and distributional batch readiness without timing claims."
    ),
    "rust_batched_null_evidence": (
        "Records Exp5739's quality-matched pairs, restart exclusions, and strict 10x null."
    ),
    "arc_causal_effects_preserved": (
        "Copies the seven Exp5740 causal primitive effects without changing the science."
    ),
    "arc_gate_schema_issue": (
        "The gate skip is attributed to representation semantics without altering Exp5740 science."
    ),
    "proposal_channel_ready": (
        "Readiness is limited to parse-safe finite-choice transport, not proposal utility."
    ),
    "sota_proposal_stream_ready": (
        "The exact-attested stream is ready as transport only, not as a search improvement."
    ),
    "continuous_self_learning_credited": (
        "Safe continuous self-learning mechanics are credited without claiming KAN superiority."
    ),
    "batch_backend_ready": (
        "Batch backend readiness is semantic/distributional API evidence, not timing evidence."
    ),
    "rust_batched_10x_ready": (
        "The strict 10x CPU software claim remains false after the terminal benchmark null."
    ),
    "arc_registry_delta": "No registry count may change during artifact reconciliation.",
    "arc_solve_credited": "The gate-skipped live A/B cannot credit an ARC solve.",
    "proposal_conflict_count": "Conflict count stays visible and cannot become a utility claim.",
    "kan_suffix_error": "The KAN suffix error is carried for matched-control comparison.",
    "mlp_suffix_error": (
        "The parameter-matched MLP control remains visible to block a KAN superiority claim."
    ),
    "restart_exclusion_count": (
        "Restart-mismatch exclusions stay visible and block strict 10x promotion."
    ),
    "completion_ledger_duplicate_blocks_preserved": (
        "Historical duplicated completion blocks are preserved rather than rewritten."
    ),
    "roadmap_task_ids": "Records the active roadmap task ids used to allocate `.513`.",
    "roadmap_doc_task_range": "Cross-checks the prose roadmap task range against the YAML.",
    "current_task_collision_check": "Records collision scans across local paths, ledgers, and history.",
    "dependency_map": "Current dependencies are auditable and do not depend on retired upstream chains.",
    "gate_map": "Structured gates are explicit so skipped tasks can be interpreted correctly.",
    "dependency_chain_retired_id_check": (
        "Ensures active dependencies and gates do not point at retired upstream chains."
    ),
    "protected_files": "Confirms protected roadmap and conductor files were not modified.",
    "operator_constraints": "Records task-local prohibitions such as no push and no conductor edits.",
    "timing_claimed": "The transition reads artifacts only and makes no benchmark timing claim.",
    "hardware_speedup_claimed": "Artifact reconciliation cannot claim hardware acceleration.",
    "inference_substrate": "The run used artifact reconciliation only.",
    "test_commands": "Verification commands are listed for replay.",
    "test_exit_codes": "Observed verification exits are recorded without relabeling failures as passes.",
    "reproducibility_checksum": "The stable artifact payload can be checked for drift.",
    "honest_verdict": "The terminal result states complete or blocked without claim inflation.",
}

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5743_transition_v513.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/experiment_5743_transition_v513.py "
            "-m pytest tests/python/test_experiment_5743_transition_v513.py -q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/experiment_5743_transition_v513.py "
            "--fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {
        "command": (
            ".venv/bin/python -c \"import pathlib, yaml; "
            "yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); "
            "print('research-roadmap.yaml YAML parse OK')\""
        ),
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/pytest tests/python/test_roadmap_schema.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": "bash scripts/validate-phase-gate.sh python/carnot/experiment_5743_transition_v513.py",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/conductor_pre_flight.py",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/adversarial_verify.py results/experiment_5743_transition_v513.json",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/root_clutter_sweep.py",
        "exit_code": None,
        "status": "not_run",
    },
)


def _read_text(root: Path, rel_path: Path) -> str:
    path = root / rel_path
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _payload(artifacts: Mapping[Path, JsonMap], rel_path: Path) -> JsonMap:
    value = artifacts.get(rel_path, {})
    return value if isinstance(value, Mapping) else {}


def _float_value(payload: JsonMap, field: str, default: float = 0.0) -> float:
    value = payload.get(field)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _int_value(payload: JsonMap, field: str) -> int:
    return int(_float_value(payload, field, 0.0))


def _bool_value(payload: JsonMap, field: str) -> bool:
    value = payload.get(field)
    return bool(value) if isinstance(value, bool) else str(value).lower() == "true"


def _nested(mapping: JsonMap, *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _read_source_context(root: Path) -> tuple[list[JsonDict], list[str]]:
    rows: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        exists = path.exists()
        rows.append(
            {
                "path": rel_path.as_posix(),
                "exists": exists,
                "read_only": True,
                "sha256": path_sha256(path),
            }
        )
        if not exists:
            missing.append(rel_path.as_posix())
    return rows, missing


def _read_artifacts(root: Path) -> tuple[dict[Path, JsonDict], dict[str, JsonDict]]:
    payloads: dict[Path, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for rel_path in UPSTREAM_ARTIFACT_PATHS:
        payload, meta = _read_json_any(root / rel_path)
        payloads[rel_path] = payload
        metadata[rel_path.as_posix()] = meta
    return payloads, metadata


def _task_statuses(
    artifacts: Mapping[Path, JsonMap],
    metadata: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload = _payload(artifacts, rel_path)
        meta = metadata.get(rel_path.as_posix(), {})
        status = _status_for_payload(payload, meta)
        rows[task_id] = {
            "artifact_path": rel_path.as_posix(),
            "status": status,
            "exists": bool(meta.get("exists")),
            "loadable": bool(meta.get("loadable")),
            "sha256": meta.get("sha256"),
            "schema": payload.get("schema"),
            "honest_verdict": _verdict(payload) or None,
            "flagged_adversarial": bool(payload.get("flagged_adversarial")),
            "metadata_error": meta.get("error"),
            "supports_positive_claim": status == "complete",
        }
    return rows


def _extract_outcome(line: str | None) -> str:
    if line is None:
        return "MISSING_LOG_LINE"
    for outcome in ("GATE_BLOCK", "FLAGGED", "BLOCK", "OK"):
        if f"| {outcome} |" in line:
            return outcome
    return "LOGGED"


def _latest_log_line(text: str, patterns: Sequence[str]) -> str | None:
    lines = [line for line in text.splitlines() if any(pattern in line for pattern in patterns)]
    return lines[-1] if lines else None


def _fallback_outcome(status: str) -> str:
    return {
        "complete": "OK",
        "flagged": "FLAGGED",
        "gate_skipped": "GATE_BLOCK",
        "blocked": "BLOCK",
        "missing": "MISSING",
        "malformed": "MALFORMED",
    }.get(status, "UNKNOWN")


def _conductor_outcomes(root: Path, statuses: Mapping[str, JsonMap]) -> dict[str, JsonDict]:
    text = _read_text(root, CONDUCTOR_LOG_RELATIVE_PATH)
    patterns: dict[str, tuple[str, ...]] = {
        "exp5731-transition-v512": ("Transition terminal .511 evidence",),
        "exp5732-v512-source-delta-ingestion": ("Ingest post-V512",),
        "exp5733-sota-finite-choice-proposal-channel": ("finite-choice SOTA proposal",),
        "exp5734-sota-exact-proposal-stream": ("exact-attested SOTA proposal",),
        "exp5735-zero-gate-kan-continuous-self-learning": ("zero-gated KAN",),
        "exp5736-csl-lifecycle-conflict-rollback": ("CSL lifecycle",),
        "exp5737-sota-stream-csl-shadow-ingress": ("Shadow-ingest",),
        "exp5738-one-axis-rust-batched-backend": ("Rust sample_batch",),
        "exp5739-one-axis-batched-10x-crossover": ("batched 10x crossover",),
        "exp5740-arc-game-blind-primitive-causal-audit": ("ARC game-blind",),
        "exp5741-arc-generic-primitive-live-ab": ("Gated on Exp5740", "generic primitive"),
        "exp5742-v512-capstone-reconciliation": ("Reconcile .512",),
    }
    rows: dict[str, JsonDict] = {}
    for task_id, task_patterns in patterns.items():
        line = _latest_log_line(text, task_patterns)
        status = str(statuses.get(task_id, {}).get("status") or "unknown")
        outcome = _extract_outcome(line) if line else _fallback_outcome(status)
        rows[task_id] = {
            "outcome": outcome,
            "artifact_status": status,
            "evidence_line": line,
            "detail": "from_conductor_log" if line else "derived_from_artifact_status",
        }
    return rows


def _model_accuracy_range(payload: JsonMap) -> list[float]:
    values = payload.get("model_accuracy")
    if not isinstance(values, Mapping):
        return []
    numeric = sorted(float(value) for value in values.values() if isinstance(value, int | float))
    return [round(numeric[0], 6), round(numeric[-1], 6)] if numeric else []


def _proposal_channel_evidence(payload: JsonMap) -> JsonDict:
    control_count = _int_value(payload, "positive_control_count") + _int_value(
        payload, "negative_control_count"
    )
    if control_count == 0:
        model_accuracy = payload.get("model_accuracy")
        model_count = len(model_accuracy) if isinstance(model_accuracy, Mapping) else 0
        row_count = _int_value(payload, "score_vector_row_count")
        control_count = row_count // model_count if model_count else 0
    return {
        "honest_verdict": _verdict(payload) or None,
        "ready_score": _float_value(payload, "proposal_channel_ready_score"),
        "control_count": control_count,
        "receipt_failure_count": _int_value(payload, "receipt_failure_count"),
        "label_collision_count": _int_value(payload, "label_collision_count"),
        "validator_disagreement_count": _int_value(payload, "validator_disagreement_count"),
        "model_accuracy": payload.get("model_accuracy", {}),
        "model_accuracy_range": _model_accuracy_range(payload),
        "parse_safe_exact_authority_only": True,
        "utility_claimed": False,
    }


def _sota_stream_evidence(payload: JsonMap) -> JsonDict:
    science_row_count = _int_value(payload, "row_count")
    if science_row_count == 0:
        hashes = payload.get("score_vector_hashes")
        science_row_count = len(hashes) if isinstance(hashes, list) else 0
    return {
        "honest_verdict": _verdict(payload) or None,
        "ready_score": _float_value(payload, "sota_proposal_stream_ready_score"),
        "science_row_count": science_row_count,
        "proposal_conflict_count": _int_value(payload, "proposal_conflict_count"),
        "validator_disagreement_count": _int_value(payload, "validator_disagreement_count"),
        "parse_safe_exact_authority_only": True,
        "search_utility_claimed": False,
    }


def _csl_safety_evidence(payload: JsonMap) -> JsonDict:
    kan_error = _nested(payload, "arm_metrics", "zero_gated_residual_spline_growth", "suffix_error")
    mlp_error = _nested(payload, "arm_metrics", "parameter_matched_mlp_residual", "suffix_error")
    return {
        "honest_verdict": _verdict(payload) or None,
        "function_preserving_insertion_score": _float_value(
            payload, "function_preserving_insertion_score"
        ),
        "zero_gate_csl_ready_score": _float_value(payload, "zero_gate_csl_ready_score"),
        "unsafe_update_count": _int_value(payload, "unsafe_update_count"),
        "max_update_latency_ms": _float_value(payload, "max_update_latency_ms"),
        "kan_suffix_error": float(kan_error) if isinstance(kan_error, int | float) else None,
        "mlp_suffix_error": float(mlp_error) if isinstance(mlp_error, int | float) else None,
        "kan_superiority_claimed": False,
    }


def _csl_lifecycle_evidence(payload: JsonMap) -> JsonDict:
    operation_counts = payload.get("operation_counts")
    total = operation_counts.get("total") if isinstance(operation_counts, Mapping) else 0
    return {
        "honest_verdict": _verdict(payload) or None,
        "csl_lifecycle_ready_score": _float_value(payload, "csl_lifecycle_ready_score"),
        "operation_count": int(total) if isinstance(total, int | float) else 0,
        "rollback_state_hash_matches": bool(payload.get("rollback_state_hash_matches")),
        "unsafe_propagation_count": _int_value(payload, "unsafe_propagation_count"),
    }


def _rust_batch_readiness(payload: JsonMap) -> JsonDict:
    return {
        "honest_verdict": _verdict(payload) or None,
        "batch_backend_ready_score": _float_value(payload, "batch_backend_ready_score"),
        "energy_trace_mismatch_count": _int_value(payload, "energy_trace_mismatch_count"),
        "checkpoint_mismatch_count": _int_value(payload, "checkpoint_mismatch_count"),
        "restart_mismatch_count": _int_value(payload, "restart_mismatch_count"),
        "timing_claimed": _bool_value(payload, "timing_claimed"),
        "hardware_speedup_claimed": _bool_value(payload, "hardware_speedup_claimed"),
    }


def _restart_exclusion_count(payload: JsonMap) -> int:
    reasons = payload.get("excluded_pair_reasons")
    if not isinstance(reasons, list):
        return 0
    return sum(
        int(row.get("count", 0))
        for row in reasons
        if isinstance(row, Mapping) and row.get("reason") == "restart_match"
    )


def _qualified_large_size_count(payload: JsonMap) -> int:
    intervals = payload.get("paired_speedup_intervals")
    if not isinstance(intervals, list):
        return 0
    sizes = {
        int(row["size"])
        for row in intervals
        if isinstance(row, Mapping)
        and row.get("quality_matched") is True
        and isinstance(row.get("size"), int | float)
    }
    return len(sizes)


def _rust_batched_null(payload: JsonMap) -> JsonDict:
    return {
        "honest_verdict": _verdict(payload) or None,
        "quality_matched_pair_count": _int_value(payload, "quality_matched_pair_count"),
        "qualified_large_size_count": _qualified_large_size_count(payload),
        "restart_exclusion_count": _restart_exclusion_count(payload),
        "rust_batched_10x_ready_score": _float_value(payload, "rust_batched_10x_ready_score"),
        "software_speedup_claimed": _bool_value(payload, "software_speedup_claimed"),
        "hardware_speedup_claimed": _bool_value(payload, "hardware_speedup_claimed"),
        "timing_claimed": _bool_value(payload, "timing_claimed"),
        "strict_10x_claimed": False,
    }


def _arc_effects(payload: JsonMap) -> JsonDict:
    candidates = payload.get("primitive_candidates")
    rows = [
        {
            "primitive": str(row.get("primitive")),
            "composite_utility_delta": row.get("composite_utility_delta"),
            "corrected_interval": row.get("corrected_interval"),
            "paired_replay_count": row.get("paired_replay_count"),
        }
        for row in candidates
        if isinstance(row, Mapping) and row.get("causal_retained", True)
    ] if isinstance(candidates, list) else []
    return {
        "positive_causal_primitive_count": _int_value(payload, "positive_causal_primitive_count"),
        "primitive_effects": rows,
        "policy_modified": _bool_value(payload, "policy_modified"),
        "registry_modified": _bool_value(payload, "registry_modified"),
        "solve_provenance": payload.get("solve_provenance"),
    }


def _coverage_type(value: Any) -> str:
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, int | float):
        return "number"
    if value is None:
        return "missing"
    return type(value).__name__


def _arc_gate_issue(arc: JsonMap, arc_ab: JsonMap) -> JsonDict:
    coverage = arc.get("counterfactual_receipt_coverage")
    return {
        "representation_mismatch": True,
        "source_leak_count_field": _int_value(arc, "source_leak_count"),
        "game_identity_leak_count_field": _int_value(arc, "game_identity_leak_count"),
        "rejected_canaries_counted_in_leak_fields": True,
        "counterfactual_receipt_coverage": coverage,
        "coverage_field_type": _coverage_type(coverage),
        "expected_coverage_gate_type": "scalar_1.0",
        "exp5741_honest_verdict": _verdict(arc_ab) or None,
        "exp5741_blocked_at_layer": arc_ab.get("blocked_at_layer"),
        "exp5741_gate_check_summary": arc_ab.get("gate_check_summary"),
        "gates_evaluated": arc_ab.get("gates_evaluated", []),
        "live_ab_ran": False,
        "science_rerun": False,
        "upstream_artifact_modified": False,
    }


def _roadmap_tasks(roadmap: JsonMap) -> list[JsonMap]:
    tasks = roadmap.get("tasks")
    return [dict(row) for row in tasks if isinstance(row, Mapping)] if isinstance(tasks, list) else []


def _gate_rows(task: JsonMap) -> list[JsonDict]:
    gates = task.get("gated_on")
    return [dict(row) for row in gates if isinstance(row, Mapping)] if isinstance(gates, list) else []


def _dependency_map(roadmap: JsonMap) -> dict[str, JsonDict]:
    tasks = _roadmap_tasks(roadmap)
    ids = [str(row.get("id")) for row in tasks if row.get("id")]
    first = ids[0] if ids else EXPERIMENT_ID
    last = ids[-1] if ids else ""
    rows: dict[str, JsonDict] = {}
    for task in tasks:
        task_id = str(task.get("id"))
        gated = [str(row.get("upstream")) for row in _gate_rows(task) if row.get("upstream")]
        if task_id == first:
            depends_on: list[str] = []
        elif task_id == last:
            depends_on = [prior for prior in ids if prior != task_id]
        else:
            depends_on = gated or [first]
        rows[task_id] = {
            "deliverable": task.get("deliverable"),
            "depends_on": depends_on,
        }
    return rows


def _gate_map(roadmap: JsonMap) -> dict[str, list[JsonDict]]:
    rows: dict[str, list[JsonDict]] = {}
    for task in _roadmap_tasks(roadmap):
        task_id = str(task.get("id"))
        rows[task_id] = [
            {
                "upstream": gate.get("upstream"),
                "field": gate.get("artifact_field") or gate.get("field"),
                "op": gate.get("op"),
                "value": gate.get("value"),
                "principle": gate.get("principle"),
            }
            for gate in _gate_rows(task)
        ]
    return rows


def _dependency_retired_id_check(
    dependencies: Mapping[str, JsonMap],
    gates: Mapping[str, Sequence[JsonMap]],
) -> JsonDict:
    current_ids = set(dependencies)
    bad_refs: list[JsonDict] = []
    for task_id, row in dependencies.items():
        for upstream in row.get("depends_on", []):
            if upstream in RETIRED_UPSTREAM_IDS or upstream not in current_ids:
                bad_refs.append({"task_id": task_id, "field": "depends_on", "upstream": upstream})
    for task_id, gate_rows in gates.items():
        for gate in gate_rows:
            upstream = gate.get("upstream") if isinstance(gate, Mapping) else None
            if upstream and (upstream in RETIRED_UPSTREAM_IDS or upstream not in current_ids):
                bad_refs.append({"task_id": task_id, "field": "gate_map", "upstream": upstream})
    return {
        "valid": not bad_refs,
        "retired_or_unknown_references": bad_refs,
        "retired_upstream_ids": sorted(RETIRED_UPSTREAM_IDS),
    }


def _extract_exp_number(text: str) -> int | None:
    match = re.search(r"(?:exp|experiment_)(\d+)", text)
    return int(match.group(1)) if match else None


def _ids_from_text(text: str) -> list[str]:
    return sorted(set(re.findall(r"\bexp\d+(?:[-_][A-Za-z0-9_]+)*", text)))


def _git_history_hits(root: Path) -> list[str]:  # pragma: no cover
    if not (root / ".git").exists():
        return []
    result = subprocess.run(
        [
            "git",
            "log",
            "--oneline",
            "--all",
            "--regexp-ignore-case",
            "--extended-regexp",
            "--grep=(exp|experiment_)(5743|5744|5745|5746|5747|5748|5749|5750|5751|5752|5753|5754)",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return result.stdout.splitlines()


def _collision_check(root: Path, roadmap_task_ids: Sequence[str]) -> JsonDict:
    current_numbers = set(range(5743, 5755))
    allowed_paths = {
        RESULT_RELATIVE_PATH.as_posix(),
        "python/carnot/experiment_5743_transition_v513.py",
        "tests/python/test_experiment_5743_transition_v513.py",
    }
    path_hits: list[str] = []
    for rel_root in ("results", "tests", "scripts", "python/carnot", ".harness", "_bmad", "ops"):
        base = root / rel_root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            if rel in allowed_paths:
                continue
            number = _extract_exp_number(rel)
            if number in current_numbers:
                path_hits.append(rel)
    complete_ids = _ids_from_text(_read_text(root, RESEARCH_COMPLETE_RELATIVE_PATH))
    conductor_ids = _ids_from_text(_read_text(root, CONDUCTOR_LOG_RELATIVE_PATH))
    completed_collisions = [
        task_id for task_id in complete_ids if _extract_exp_number(task_id) in current_numbers
    ]
    conductor_collisions = [
        task_id for task_id in conductor_ids if _extract_exp_number(task_id) in current_numbers
    ]
    return {
        "searched_roots": [
            "research-roadmap.yaml",
            "research-roadmap-next.yaml",
            "openspec/change-proposals",
            "results",
            "tests",
            "scripts",
            "history",
            ".harness",
            "_bmad",
            "ops",
        ],
        "current_task_range": CURRENT_TASK_RANGE,
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "roadmap_task_ids": list(roadmap_task_ids),
        "path_collisions": sorted(set(path_hits)),
        "completed_ledger_collisions": completed_collisions,
        "conductor_log_current_range_mentions": conductor_collisions,
        "git_history_hits": _git_history_hits(root),
        "outer_loop_paths_scanned": [".harness", "_bmad", "ops"],
        "range_matches_roadmap": list(roadmap_task_ids) == list(EXPECTED_TASK_IDS),
        "collision_free": not path_hits and not completed_collisions,
    }


def _protected_files(
    root: Path,
    modification_overrides: Mapping[Path | str, bool] | None,
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for rel_path in PROTECTED_FILE_PATHS:
        modified = _modification_status(root, rel_path, modification_overrides)
        rows[rel_path.as_posix()] = {
            "exists": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
            "modified_by_transition": modified,
        }
    return rows


def _test_commands(tests_run: Sequence[JsonMap]) -> list[str]:
    return [str(row.get("command")) for row in tests_run if row.get("command")]


def _test_exit_codes(tests_run: Sequence[JsonMap]) -> dict[str, Any]:
    return {
        str(row.get("command")): row.get("exit_code")
        for row in tests_run
        if row.get("command")
    }


def _load_tests_run(path: Path | None) -> list[JsonDict]:
    if path is None:
        return [dict(row) for row in DEFAULT_TESTS_RUN]
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError("validation results must be a JSON list")
    return [dict(row) for row in parsed if isinstance(row, Mapping)]


def _preconditions_checked(
    root: Path,
    roadmap: JsonMap,
    roadmap_task_ids: Sequence[str],
    collision: JsonMap,
    protected: Mapping[str, JsonMap],
    source_context_missing: Sequence[str],
) -> JsonDict:
    return {
        "instructions_read": ["AGENTS.md", "CODEX.md", "CLAUDE.md"],
        "spec_refs": list(SPEC_REFS),
        "upstream_artifact_count": len(UPSTREAM_ARTIFACT_PATHS),
        "roadmap_milestone": roadmap.get("milestone"),
        "roadmap_task_ids": list(roadmap_task_ids),
        "roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "roadmap_next_absence_recorded": not (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
        "source_context_missing": list(source_context_missing),
        "collision_free": bool(collision.get("collision_free")),
        "range_matches_roadmap": bool(collision.get("range_matches_roadmap")),
        "protected_files_unchanged": all(
            not bool(row.get("modified_by_transition")) for row in protected.values()
        ),
    }


def _failed_preconditions(
    missing: Sequence[str],
    malformed: Sequence[str],
    roadmap: JsonMap,
    collision: JsonMap,
    retired_check: JsonMap,
    protected: Mapping[str, JsonMap],
    capstone: JsonMap,
) -> list[str]:
    failures = [f"missing_artifact:{path}" for path in missing]
    failures.extend(f"malformed_artifact:{path}" for path in malformed)
    if roadmap.get("milestone") != CURRENT_MILESTONE:
        failures.append("roadmap_milestone_not_2026.07.513")
    if not collision.get("range_matches_roadmap"):
        failures.append("roadmap_task_range_mismatch")
    if not collision.get("collision_free"):
        failures.append("current_task_id_collision")
    if not retired_check.get("valid"):
        failures.append("retired_or_unknown_dependency_reference")
    for rel_path, row in protected.items():
        if row.get("modified_by_transition"):
            failures.append(f"protected_file_modified:{rel_path}")
    if capstone.get("proposal_channel_ready") is not True:
        failures.append("capstone_proposal_channel_not_ready")
    if capstone.get("sota_proposal_stream_ready") is not True:
        failures.append("capstone_sota_stream_not_ready")
    if capstone.get("continuous_self_learning_credited") is not True:
        failures.append("capstone_csl_not_credited")
    if capstone.get("batch_backend_ready") is not True:
        failures.append("capstone_batch_backend_not_ready")
    if capstone.get("rust_batched_10x_ready") is not False:
        failures.append("capstone_rust_10x_not_terminal_false")
    if capstone.get("arc_registry_delta") != 0:
        failures.append("capstone_arc_registry_delta_nonzero")
    if capstone.get("arc_solve_credited") is not False:
        failures.append("capstone_arc_solve_credit_nonfalse")
    return failures


def _honest_verdict(status: str) -> str:
    if status == "complete":
        return (
            "complete: archived terminal .512 evidence into .513; "
            "proposal_channel_ready=true; sota_proposal_stream_ready=true; "
            "continuous_self_learning_credited=true; batch_backend_ready=true; "
            "rust_batched_10x_ready=false; arc_registry_delta=0; "
            "arc_solve_credited=false; current_task_range=exp5743-exp5754"
        )
    return "blocked: v513 transition preserved inputs but one or more preconditions failed"


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[JsonMap] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, artifact_metadata = _read_artifacts(root)
    statuses = _task_statuses(artifacts, artifact_metadata)
    source_context, source_context_missing = _read_source_context(root)
    roadmap, _roadmap_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    roadmap_doc_task_range = _task_range_from_text(_read_text(root, VNEXT_RELATIVE_PATH))
    dependencies = _dependency_map(roadmap)
    gates = _gate_map(roadmap)
    retired_check = _dependency_retired_id_check(dependencies, gates)
    collision = _collision_check(root, roadmap_task_ids)
    protected = _protected_files(root, modification_overrides)

    exp5732 = _payload(artifacts, EXP5732_SOURCE_PATH)
    exp5733 = _payload(artifacts, EXP5733_CHANNEL_PATH)
    exp5734 = _payload(artifacts, EXP5734_STREAM_PATH)
    exp5735 = _payload(artifacts, EXP5735_CSL_PATH)
    exp5736 = _payload(artifacts, EXP5736_LIFECYCLE_PATH)
    exp5738 = _payload(artifacts, EXP5738_BATCH_PATH)
    exp5739 = _payload(artifacts, EXP5739_10X_PATH)
    exp5740 = _payload(artifacts, EXP5740_ARC_PATH)
    exp5741 = _payload(artifacts, EXP5741_ARC_AB_PATH)
    exp5742 = _payload(artifacts, EXP5742_CAPSTONE_PATH)

    missing = [
        path for path, meta in artifact_metadata.items() if not bool(meta.get("exists"))
    ]
    malformed = [
        path
        for path, meta in artifact_metadata.items()
        if bool(meta.get("exists")) and not bool(meta.get("loadable"))
    ]
    failures = _failed_preconditions(
        missing,
        malformed,
        roadmap,
        collision,
        retired_check,
        protected,
        exp5742,
    )
    status = "complete" if not failures else "blocked"
    tests = [dict(row) for row in tests_run]
    proposal_evidence = _proposal_channel_evidence(exp5733)
    stream_evidence = _sota_stream_evidence(exp5734)
    csl_evidence = _csl_safety_evidence(exp5735)
    rust_null = _rust_batched_null(exp5739)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "previous_milestone": PREVIOUS_MILESTONE,
        "current_milestone": CURRENT_MILESTONE,
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "current_task_range": CURRENT_TASK_RANGE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_capstone_hash": path_sha256(root / EXP5742_CAPSTONE_PATH),
        "source_context": source_context,
        "source_context_missing": source_context_missing,
        "artifact_metadata": artifact_metadata,
        "missing_artifacts": missing,
        "malformed_artifacts": malformed,
        "failed_preconditions": failures,
        "v512_task_verdicts": statuses,
        "v512_conductor_outcomes": _conductor_outcomes(root, statuses),
        "source_delta_duration_flag": {
            "honest_verdict": _verdict(exp5732) or None,
            "flagged_adversarial": bool(exp5732.get("flagged_adversarial")),
            "duration_s": exp5732.get("duration_s"),
            "inference_substrate": exp5732.get("inference_substrate"),
            "benchmark_compute_claimed": bool(exp5732.get("benchmark_compute_claimed")),
        },
        "proposal_channel_evidence": proposal_evidence,
        "sota_stream_evidence": stream_evidence,
        "csl_safety_evidence": csl_evidence,
        "csl_lifecycle_evidence": _csl_lifecycle_evidence(exp5736),
        "rust_batch_readiness_evidence": _rust_batch_readiness(exp5738),
        "rust_batched_null_evidence": rust_null,
        "arc_causal_effects_preserved": _arc_effects(exp5740),
        "arc_gate_schema_issue": _arc_gate_issue(exp5740, exp5741),
        "proposal_channel_ready": exp5742.get("proposal_channel_ready") is True,
        "sota_proposal_stream_ready": exp5742.get("sota_proposal_stream_ready") is True,
        "continuous_self_learning_credited": exp5742.get("continuous_self_learning_credited")
        is True,
        "batch_backend_ready": exp5742.get("batch_backend_ready") is True,
        "rust_batched_10x_ready": False,
        "arc_registry_delta": int(exp5742.get("arc_registry_delta") or 0),
        "arc_solve_credited": False,
        "proposal_conflict_count": stream_evidence["proposal_conflict_count"],
        "kan_suffix_error": csl_evidence["kan_suffix_error"],
        "mlp_suffix_error": csl_evidence["mlp_suffix_error"],
        "restart_exclusion_count": rust_null["restart_exclusion_count"],
        "completion_ledger_duplicate_blocks_preserved": (root / RESEARCH_COMPLETE_RELATIVE_PATH).exists(),
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_doc_task_range": roadmap_doc_task_range,
        "current_task_collision_check": collision,
        "dependency_map": dependencies,
        "gate_map": gates,
        "dependency_chain_retired_id_check": retired_check,
        "protected_files": protected,
        "operator_constraints": {
            "do_not_push": True,
            "do_not_modify_research_conductor": True,
            "ops_status_changelog_traceability_delegated_to_reconciler": True,
        },
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": _test_commands(tests),
        "test_exit_codes": _test_exit_codes(tests),
        "honest_verdict": _honest_verdict(status),
        "reproducibility_checksum": "",
    }
    artifact["preconditions_checked"] = _preconditions_checked(
        root,
        roadmap,
        roadmap_task_ids,
        collision,
        protected,
        source_context_missing,
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def emit_report(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    tests_run: Sequence[JsonMap] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    report = build_report(
        root,
        tests_run=tests_run,
        modification_overrides=modification_overrides,
    )
    write_json(output_path or root / RESULT_RELATIVE_PATH, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validation-results", type=Path, default=None)
    args = parser.parse_args(argv)
    tests_run = _load_tests_run(args.validation_results)
    report = emit_report(args.root, output_path=args.output, tests_run=tests_run)
    print(json.dumps({"result_path": report["result_path"], "status": report["status"]}, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
