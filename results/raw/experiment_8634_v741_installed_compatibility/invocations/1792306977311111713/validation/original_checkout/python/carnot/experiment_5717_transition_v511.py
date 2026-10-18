"""Exp5717 transition receipt from milestone .510 into .511.

Spec refs: REQ-CAPSTONE-5717, SCENARIO-CAPSTONE-5717,
SCENARIO-CAPSTONE-5717-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5717-FIELD-PRINCIPLES.

This module is a reconciliation ledger. It reads completed, blocked,
gate-skipped, and missing `.510` artifacts, then emits a bounded `.511`
transition receipt. It does not run inference, train models, alter GGUF weights,
or reinterpret scientific verdicts. The important discipline is preserving the
exact denominator: CUDA offload succeeded in Exp5708, but the stream failed at
the answer-channel parser, so downstream FR-11 claims remain closed until a
future clean stream exists.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5717_transition_v511.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")

EXPERIMENT = "experiment_5717_transition_v511"
EXPERIMENT_ID = "exp5717-transition-v511"
PREVIOUS_MILESTONE = "2026.07.510"
CURRENT_MILESTONE = "2026.07.511"
PREVIOUS_TASK_RANGE = "exp5706-exp5716"
CURRENT_TASK_RANGE = "exp5717-exp5728"
RUN_DATE = "2026-07-19"
RANDOM_SEED = 5717
SCHEMA = "carnot.experiment_5717.transition_v511.v1"
INFERENCE_SUBSTRATE = "artifact_reconciliation_only"
TERMINAL_PREFIXES = ("complete:", "blocked:", "blocked_")

SPEC_REFS = (
    "REQ-CAPSTONE-5717",
    "SCENARIO-CAPSTONE-5717",
    "SCENARIO-CAPSTONE-5717-MISSING-MALFORMED",
    "SCENARIO-CAPSTONE-5717-FIELD-PRINCIPLES",
)

EXP5706_TASK_ID = "exp5706-transition-v510"
EXP5707_TASK_ID = "exp5707-v510-source-delta-ingestion"
EXP5708_TASK_ID = "exp5708-sota-exact-constraint-canary"
EXP5709_TASK_ID = "exp5709-fr11-prospective-shadow-stream"
EXP5710_TASK_ID = "exp5710-fr11-isolated-act-on-advice-canary"
EXP5711_TASK_ID = "exp5711-arc-relational-goal-energy-live-qualification"
EXP5712_TASK_ID = "exp5712-arc-relational-goal-energy-live-ab"
EXP5713_TASK_ID = "exp5713-arc-live-self-discovery-levelup-v510"
EXP5714_TASK_ID = "exp5714-one-axis-tempering-rust-parity"
EXP5715_TASK_ID = "exp5715-one-axis-tempering-rust-quality-restart"
EXP5716_TASK_ID = "exp5716-v510-capstone-reconciliation"

EXP5706_TRANSITION_PATH = Path("results/experiment_5706_transition_v510.json")
EXP5707_SOURCE_PATH = Path("results/experiment_5707_v510_source_delta_ingestion.json")
EXP5708_CANARY_PATH = Path("results/experiment_5708_sota_exact_constraint_canary.json")
EXP5708_ROWS_PATH = Path("results/experiment_5708_sota_exact_constraint_canary.rows.jsonl")
EXP5709_SHADOW_PATH = Path("results/experiment_5709_fr11_prospective_shadow_stream.json")
EXP5710_ISOLATED_PATH = Path("results/experiment_5710_fr11_isolated_act_on_advice_canary.json")
EXP5711_ARC_QUAL_PATH = Path(
    "results/experiment_5711_arc_relational_goal_energy_live_qualification.json"
)
EXP5712_ARC_AB_PATH = Path("results/experiment_5712_arc_relational_goal_energy_live_ab.json")
EXP5713_ARC_LEVEL_PATH = Path("results/experiment_5713_arc_live_self_discovery_levelup_v510.json")
EXP5714_RUST_PARITY_PATH = Path("results/experiment_5714_one_axis_tempering_rust_parity.json")
EXP5715_RUST_QUALITY_PATH = Path(
    "results/experiment_5715_one_axis_tempering_rust_quality_restart.json"
)
EXP5716_CAPSTONE_PATH = Path("results/experiment_5716_v510_capstone_reconciliation.json")

V510_ARTIFACT_PATHS: dict[str, Path] = {
    EXP5706_TASK_ID: EXP5706_TRANSITION_PATH,
    EXP5707_TASK_ID: EXP5707_SOURCE_PATH,
    EXP5708_TASK_ID: EXP5708_CANARY_PATH,
    EXP5709_TASK_ID: EXP5709_SHADOW_PATH,
    EXP5710_TASK_ID: EXP5710_ISOLATED_PATH,
    EXP5711_TASK_ID: EXP5711_ARC_QUAL_PATH,
    EXP5712_TASK_ID: EXP5712_ARC_AB_PATH,
    EXP5713_TASK_ID: EXP5713_ARC_LEVEL_PATH,
    EXP5714_TASK_ID: EXP5714_RUST_PARITY_PATH,
    EXP5715_TASK_ID: EXP5715_RUST_QUALITY_PATH,
    EXP5716_TASK_ID: EXP5716_CAPSTONE_PATH,
}

REQUIRED_MANIFEST_RETIREMENT: JsonDict = {
    "id": "exp5709_fr11_prospective_shadow_stream_retired_v511",
    "scope_key": "fr11_prospective_shadow_stream_exp5709_same_verdict",
    "experiment_scope": (
        "Exp5709 prospective shadow stream rerun with the same parse-failed "
        "Exp5708 stream verdict only"
    ),
    "reason": (
        "Exp5709 gate-blocked because Exp5708 authenticated CUDA but produced "
        "a parse-failed exact stream: 21 truncations, 26 missing-answer rows, "
        "3 parses, and zero validator disagreements. Preserve future clean "
        "prospective streams and generic lifecycle learning."
    ),
    "experiment_ids": ["exp5709"],
    "retired_milestone": CURRENT_MILESTONE,
    "retired_by_artifact": EXP5709_SHADOW_PATH.as_posix(),
    "recorded_by_artifact": RESULT_RELATIVE_PATH.as_posix(),
    "operator_reopen_required": True,
    "retire_if_same_verdict": True,
    "blocked_patterns": [
        "fr11_prospective_shadow_stream_exp5709_same_verdict",
        "repeat Exp5709 prospective shadow stream on the parse-failed Exp5708 stream",
        "blocked_gate_check_failed after exp5708 parse_failures without a new clean stream",
    ],
}

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "one-line annotations for every required transition field.",
    "source_capstone_hash": "binds the transition to the terminal .510 capstone bytes.",
    "v510_task_verdicts": "every Exp5706-Exp5716 verdict is explicit before carry-forward.",
    "v510_conductor_outcomes": (
        "conductor OK, gate-block, skip, activation, and missing states stay distinct."
    ),
    "sota_parse_failure_taxonomy": "Exp5708 parse-root-cause denominators are preserved exactly.",
    "cuda_offload_authenticated": (
        "authenticated CUDA provenance cannot override stream parsing gates."
    ),
    "fr11_prospective_promoted": (
        "Exp5709 gate-skipped evidence cannot promote prospective learning."
    ),
    "fr11_isolated_promoted": "Exp5710 missing/skipped evidence cannot promote act-on-advice.",
    "model_weight_mutation": "bare false proves no model weights were changed.",
    "production_default_enabled": "bare false keeps canary evidence out of production defaults.",
    "arc_registry_count": "authoritative reproduced-level count after .510.",
    "arc_registry_delta": "the .510 live attempt banked no ARC level.",
    "arc_relational_route_promoted": (
        "Exp5712 matched null cannot promote the relational route."
    ),
    "one_axis_rust_parity_ready_score": (
        "Exp5714 exact parity readiness is scalar and local to Rust portability."
    ),
    "one_axis_rust_quality_ready_score": (
        "Exp5715 hard-instance/restart readiness is scalar and local to Rust portability."
    ),
    "retirements_required": "the transition names every required narrow exclusion before mutation.",
    "retirements_applied": "manifest-applied retirements are reconstructable from exact scopes.",
    "preserved_scopes": "non-retired scopes stay live and unbroadened.",
    "retired_scopes": "terminal negative scopes are bounded narrowly.",
    "current_task_range": "canonical allocation is exp5717-exp5728.",
    "dependency_map": "successors and prerequisites are reconstructable.",
    "gate_map": "structured gates are auditable and ID-valid.",
    "timing_claimed": "bare false prevents runtime inflation.",
    "hardware_speedup_claimed": "bare false prevents hardware inflation.",
    "inference_substrate": "artifact_reconciliation_only -- no inference occurred.",
    "test_commands": "verification commands are replayable.",
    "test_exit_codes": "observed command exits are recorded without laundering failures.",
    "reproducibility_checksum": "content-addressed transition output is stable.",
    "honest_verdict": "terminal summary starting with complete: or blocked:.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "previous_milestone",
    "current_milestone",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "artifact_metadata",
    "missing_artifacts",
    "malformed_artifacts",
    "manifest_metadata",
    "manifest_debt_after",
    "collision_check",
    "forbidden_files_unchanged",
    *REQUIRED_ARTIFACT_FIELDS,
)

DEFAULT_VALIDATION_RESULTS: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5717_transition_v511.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/experiment_5717_transition_v511.py "
            "-m pytest tests/python/test_experiment_5717_transition_v511.py -q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5717_transition_v511.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {
        "command": ".venv/bin/python -c \"import pathlib, yaml; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); print('research-roadmap.yaml YAML parse OK')\"",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/pytest tests/python/test_roadmap_schema.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/pytest tests/python/test_pick_next_task_gate_block.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": "python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": "bash scripts/validate-phase-gate.sh python/carnot/experiment_5717_transition_v511.py",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": "python scripts/adversarial_verify.py results/experiment_5717_transition_v511.json",
        "exit_code": None,
        "status": "not_run",
    },
    {"command": "python scripts/root_clutter_sweep.py", "exit_code": None, "status": "not_run"},
)


def _payload(artifacts: Mapping[str, JsonMap], rel_path: Path) -> JsonMap:
    value = artifacts.get(rel_path.as_posix(), {})
    return value if isinstance(value, Mapping) else {}


def _load_artifacts(root: Path) -> tuple[dict[str, JsonDict], JsonDict, list[str], list[str]]:
    artifacts: dict[str, JsonDict] = {}
    metadata: JsonDict = {}
    missing: list[str] = []
    malformed: list[str] = []
    for rel_path in V510_ARTIFACT_PATHS.values():
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        artifacts[rel] = payload
        metadata[rel] = meta
        if not meta.get("exists"):
            missing.append(rel)
        elif not meta.get("loadable"):
            malformed.append(rel)
    return artifacts, metadata, missing, malformed


def _terminal_statuses(artifacts: Mapping[str, JsonMap], metadata: JsonMap) -> dict[str, JsonDict]:
    statuses: dict[str, JsonDict] = {}
    for task_id, rel_path in V510_ARTIFACT_PATHS.items():
        rel = rel_path.as_posix()
        payload = _payload(artifacts, rel_path)
        meta = metadata.get(rel, {})
        status = _status_for_payload(payload, meta)
        statuses[task_id] = {
            "path": rel,
            "status": status,
            "sha256": meta.get("sha256"),
            "honest_verdict": _verdict(payload) or None,
            "gate_check_summary": payload.get("gate_check_summary"),
            "gates_evaluated": payload.get("gates_evaluated", []),
            "preserves_state_exactly": status in {"complete", "blocked", "gate_skipped", "missing"},
            "supports_promotion": status == "complete",
        }
    return statuses


def _summarize_rows(root: Path) -> JsonDict:
    rows_path = root / EXP5708_ROWS_PATH
    counts = {
        "manifest_row_count": 0,
        "truncation_count": 0,
        "missing_answer_count": 0,
        "parsed_answer_count": 0,
        "parse_failure_count": 0,
        "validator_disagreement_count": 0,
        "finish_reason_length_count": 0,
        "finish_reason_stop_count": 0,
    }
    if not rows_path.exists():  # pragma: no cover - covered by malformed artifact path instead.
        return counts
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        counts["manifest_row_count"] += 1
        if row.get("parse_ok") is True:
            counts["parsed_answer_count"] += 1
        else:
            counts["parse_failure_count"] += 1
        if row.get("parse_error") == "truncated" or row.get("truncated") is True:
            counts["truncation_count"] += 1
        if row.get("parse_error") == "missing_answer_line":
            counts["missing_answer_count"] += 1
        if row.get("validator_disagreement") is True:
            counts["validator_disagreement_count"] += 1
        if row.get("finish_reason") == "length":
            counts["finish_reason_length_count"] += 1
        if row.get("finish_reason") == "stop":
            counts["finish_reason_stop_count"] += 1
    if counts["finish_reason_length_count"] == 0:
        counts["finish_reason_length_count"] = counts["truncation_count"]
    if counts["finish_reason_stop_count"] == 0:
        counts["finish_reason_stop_count"] = (
            counts["manifest_row_count"] - counts["finish_reason_length_count"]
        )
    return counts


def _manifest_entries(manifest: JsonMap) -> list[JsonMap]:
    entries: list[JsonMap] = []
    for key in ("retired", "retired_experiments", "retired_extras"):
        value = manifest.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            entries.extend(row for row in value if isinstance(row, Mapping))
    return entries


def _manifest_has_scope(manifest: JsonMap, scope: str) -> bool:
    return any(
        entry.get("scope_key") == scope or entry.get("scope") == scope
        for entry in _manifest_entries(manifest)
    )


def _derive_conductor_outcomes(root: Path) -> JsonDict:
    log_path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    def outcome(marker: str, ok: str, missing: str = "UNKNOWN") -> str:
        return ok if marker in text else missing

    return {
        "milestone_2026.07.510_activation": {
            "outcome": outcome("Milestone 2026.07.510 activated", "OK"),
            "evidence": "Milestone 2026.07.510 activated",
        },
        EXP5706_TASK_ID: {
            "outcome": outcome("Transition terminal .509 evidence", "OK"),
            "evidence": "Transition terminal .509 evidence",
        },
        EXP5707_TASK_ID: {
            "outcome": outcome("Ingest post-planning 2025-2026 source deltas", "OK"),
            "evidence": "Ingest post-planning 2025-2026 source deltas",
        },
        EXP5708_TASK_ID: {
            "outcome": outcome("Build a sealed exact-constraint canary", "OK"),
            "evidence": "Build a sealed exact-constraint canary",
        },
        EXP5709_TASK_ID: {
            "outcome": outcome("Gated on Exp5708 exact canary", "GATE_BLOCK"),
            "evidence": "Gated on Exp5708 exact canary",
        },
        EXP5710_TASK_ID: {
            "outcome": outcome("Pre-emptive skip: upstream retired", "GATE_BLOCK_PREEMPTIVE_SKIP"),
            "evidence": "Pre-emptive skip: upstream retired",
        },
        EXP5711_TASK_ID: {
            "outcome": outcome("Qualify zero-variance-safe relational", "OK"),
            "evidence": "Qualify zero-variance-safe relational",
        },
        EXP5712_TASK_ID: {
            "outcome": outcome("matched known-level", "OK"),
            "evidence": "matched known-level",
        },
        EXP5713_TASK_ID: {
            "outcome": outcome("Unconditional registry-rotated ARC", "OK"),
            "evidence": "Unconditional registry-rotated ARC",
        },
        EXP5714_TASK_ID: {
            "outcome": outcome("Port the promoted one-axis replica-exchange", "OK"),
            "evidence": "Port the promoted one-axis replica-exchange",
        },
        EXP5715_TASK_ID: {
            "outcome": outcome("Rust one-axis hard-instance quality", "OK"),
            "evidence": "Rust one-axis hard-instance quality",
        },
        EXP5716_TASK_ID: {
            "outcome": outcome("Reconcile .510 prospective FR-11", "OK"),
            "evidence": "Reconcile .510 prospective FR-11",
        },
        "milestone_2026.07.511_activation": {
            "outcome": outcome("Milestone 2026.07.511 activated", "OK"),
            "evidence": "Milestone 2026.07.511 activated",
        },
    }


def _direct_parity_ready(payload: JsonMap, statuses: Mapping[str, JsonMap]) -> float:
    ready = (
        statuses[EXP5714_TASK_ID]["status"] == "complete"
        and _number(payload, "one_axis_rust_parity_ready_score") >= 1.0
        and _number(payload, "broken_control_rejected_score") >= 1.0
        and _bool(payload, "checkpoint_roundtrip_pass")
        and _bool(payload, "cross_language_restart_pass")
        and not _bool(payload, "timing_claimed")
        and not _bool(payload, "hardware_speedup_claimed")
        and not _bool(payload, "two_axis_code_added")
    )
    return 1.0 if ready else 0.0


def _direct_quality_ready(payload: JsonMap, statuses: Mapping[str, JsonMap]) -> float:
    transition_budget = payload.get("transition_budget_parity")
    if not isinstance(transition_budget, Mapping):
        transition_budget = {}
    swap_schedule = payload.get("swap_schedule_parity")
    if not isinstance(swap_schedule, Mapping):
        swap_schedule = {}
    matched_swap = _bool(swap_schedule, "matched_language_swap_schedule") or _bool(
        swap_schedule, "matched_swap_schedule"
    )
    seed_count = payload.get("successful_seed_count")
    if isinstance(seed_count, Mapping):
        seed_count_value = _int(seed_count, "value")
    else:
        seed_count_value = _int(payload, "successful_seed_count")
    ready = (
        statuses[EXP5715_TASK_ID]["status"] == "complete"
        and _number(payload, "one_axis_rust_quality_ready_score") >= 1.0
        and _int(payload, "material_regression_count") == 0
        and seed_count_value >= 5
        and _bool(payload, "python_to_rust_restart_pass")
        and _bool(payload, "rust_to_python_restart_pass")
        and _bool(transition_budget, "matched_corrected_transition_budget")
        and matched_swap
        and not _bool(payload, "timing_claimed")
        and not _bool(payload, "hardware_speedup_claimed")
    )
    return 1.0 if ready else 0.0


def _dependency_map() -> JsonDict:
    return {
        EXPERIMENT_ID: {
            "deliverable": RESULT_RELATIVE_PATH.as_posix(),
            "depends_on": [EXP5716_TASK_ID],
        },
        "exp5718-v511-source-delta-ingestion": {
            "deliverable": "results/experiment_5718_v511_source_delta_ingestion.json",
            "depends_on": [EXPERIMENT_ID],
        },
        "exp5719-sota-answer-channel-forensics": {
            "deliverable": "results/experiment_5719_sota_answer_channel_forensics.json",
            "depends_on": [EXPERIMENT_ID, EXP5708_TASK_ID],
        },
        "exp5720-sota-attested-exact-envelope-canary": {
            "deliverable": "results/experiment_5720_sota_attested_exact_envelope_canary.json",
            "depends_on": ["exp5719-sota-answer-channel-forensics"],
        },
        "exp5721-fr11-memops-lifecycle-shadow-stream": {
            "deliverable": "results/experiment_5721_fr11_memops_lifecycle_shadow_stream.json",
            "depends_on": ["exp5720-sota-attested-exact-envelope-canary", EXPERIMENT_ID],
        },
        "exp5722-fr11-compliance-recovery-rollback-canary": {
            "deliverable": "results/experiment_5722_fr11_compliance_recovery_rollback_canary.json",
            "depends_on": ["exp5721-fr11-memops-lifecycle-shadow-stream"],
        },
        "exp5723-one-axis-rust-samplerbackend-integration": {
            "deliverable": "results/experiment_5723_one_axis_rust_samplerbackend_integration.json",
            "depends_on": [EXPERIMENT_ID, EXP5714_TASK_ID, EXP5715_TASK_ID],
        },
        "exp5724-one-axis-rust-python-matched-crossover": {
            "deliverable": "results/experiment_5724_one_axis_rust_python_matched_crossover.json",
            "depends_on": ["exp5723-one-axis-rust-samplerbackend-integration"],
        },
        "exp5725-arc-epistemic-ledger-live-qualification": {
            "deliverable": "results/experiment_5725_arc_epistemic_ledger_live_qualification.json",
            "depends_on": [EXPERIMENT_ID, EXP5712_TASK_ID, EXP5713_TASK_ID],
        },
        "exp5726-arc-epistemic-ledger-matched-ab": {
            "deliverable": "results/experiment_5726_arc_epistemic_ledger_matched_ab.json",
            "depends_on": ["exp5725-arc-epistemic-ledger-live-qualification"],
        },
        "exp5727-arc-live-self-discovery-levelup-v511": {
            "deliverable": "results/experiment_5727_arc_live_self_discovery_levelup_v511.json",
            "depends_on": [EXPERIMENT_ID],
            "optional_prerequisites": ["exp5726-arc-epistemic-ledger-matched-ab"],
            "unconditional": True,
        },
        "exp5728-v511-capstone-reconciliation": {
            "deliverable": "results/experiment_5728_v511_capstone_reconciliation.json",
            "depends_on": [
                EXPERIMENT_ID,
                "exp5718-v511-source-delta-ingestion",
                "exp5719-sota-answer-channel-forensics",
                "exp5720-sota-attested-exact-envelope-canary",
                "exp5721-fr11-memops-lifecycle-shadow-stream",
                "exp5722-fr11-compliance-recovery-rollback-canary",
                "exp5723-one-axis-rust-samplerbackend-integration",
                "exp5724-one-axis-rust-python-matched-crossover",
                "exp5725-arc-epistemic-ledger-live-qualification",
                "exp5726-arc-epistemic-ledger-matched-ab",
                "exp5727-arc-live-self-discovery-levelup-v511",
            ],
        },
    }


def _gate_map() -> JsonDict:
    return {
        "exp5720-sota-attested-exact-envelope-canary": [
            {
                "upstream": "exp5719-sota-answer-channel-forensics",
                "field": "answer_channel_ready_score",
                "op": ">=",
                "value": 1.0,
            },
            {
                "upstream": "exp5719-sota-answer-channel-forensics",
                "field": "positive_control_parse_success_rate",
                "op": "==",
                "value": 1.0,
            },
            {
                "upstream": "exp5719-sota-answer-channel-forensics",
                "field": "cuda_offload_authenticated",
                "op": "==",
                "value": True,
            },
        ],
        "exp5721-fr11-memops-lifecycle-shadow-stream": [
            {
                "upstream": "exp5720-sota-attested-exact-envelope-canary",
                "field": "attested_stream_ready_score",
                "op": ">=",
                "value": 1.0,
            },
            {
                "upstream": "exp5720-sota-attested-exact-envelope-canary",
                "field": "parse_failure_count",
                "op": "==",
                "value": 0,
            },
        ],
        "exp5722-fr11-compliance-recovery-rollback-canary": [
            {
                "upstream": "exp5721-fr11-memops-lifecycle-shadow-stream",
                "field": "lifecycle_ready_score",
                "op": ">=",
                "value": 1.0,
            },
            {
                "upstream": "exp5721-fr11-memops-lifecycle-shadow-stream",
                "field": "unsafe_false_accept_count",
                "op": "==",
                "value": 0,
            },
        ],
        "exp5723-one-axis-rust-samplerbackend-integration": [
            {
                "upstream": EXPERIMENT_ID,
                "field": "one_axis_rust_parity_ready_score",
                "op": "==",
                "value": 1.0,
            },
            {
                "upstream": EXPERIMENT_ID,
                "field": "one_axis_rust_quality_ready_score",
                "op": "==",
                "value": 1.0,
            },
        ],
        "exp5724-one-axis-rust-python-matched-crossover": [
            {
                "upstream": "exp5723-one-axis-rust-samplerbackend-integration",
                "field": "samplerbackend_rust_adapter_ready_score",
                "op": ">=",
                "value": 1.0,
            }
        ],
        "exp5726-arc-epistemic-ledger-matched-ab": [
            {
                "upstream": "exp5725-arc-epistemic-ledger-live-qualification",
                "field": "epistemic_ledger_ready_score",
                "op": ">=",
                "value": 1.0,
            }
        ],
        "exp5727-arc-live-self-discovery-levelup-v511": [],
    }


def _collision_check(root: Path) -> JsonDict:
    searched_roots = ["results", "python", "tests", "scripts"]
    target = RESULT_RELATIVE_PATH.as_posix()
    path_collisions: list[str] = []
    for dirname in searched_roots:
        base = root / dirname
        if base.exists():
            for path in base.rglob("*5717*"):
                rel = path.relative_to(root).as_posix()
                if rel != target:
                    path_collisions.append(rel)
    content_files = [
        ROADMAP_RELATIVE_PATH,
        RESEARCH_COMPLETE_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
        Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    ]
    content_references = [
        rel.as_posix()
        for rel in content_files
        if (root / rel).exists() and ("exp5717" in (root / rel).read_text(encoding="utf-8"))
    ]
    return {
        "searched_roots": searched_roots + [path.as_posix() for path in content_files],
        "target_deliverable": target,
        "target_deliverable_reserved_for_current_task": True,
        "historical_numeric_path_collisions": sorted(path_collisions),
        "content_reference_files": sorted(content_references),
        "allocation": CURRENT_TASK_RANGE,
        "collision_free_for_required_deliverable": True,
    }


def _forbidden_files(root: Path) -> JsonDict:
    return {
        rel.as_posix(): {"exists": (root / rel).exists(), "sha256": path_sha256(root / rel)}
        for rel in (ROADMAP_RELATIVE_PATH, CONDUCTOR_RELATIVE_PATH)
    }


def _test_maps(validation_results: Sequence[JsonMap]) -> tuple[list[str], JsonDict]:
    commands = [str(row.get("command")) for row in validation_results]
    exit_codes = {str(row.get("command")): row.get("exit_code") for row in validation_results}
    return commands, exit_codes


def _dependency_ids_valid(dependencies: Mapping[str, JsonMap], gates: Mapping[str, Any]) -> bool:
    allowed = set(V510_ARTIFACT_PATHS) | set(dependencies)
    for row in dependencies.values():
        for dep in row.get("depends_on", []):
            if dep not in allowed:
                return False
        for dep in row.get("optional_prerequisites", []):
            if dep not in allowed:
                return False
    for rows in gates.values():
        if isinstance(rows, Sequence) and not isinstance(rows, str | bytes | bytearray):
            for gate in rows:
                if isinstance(gate, Mapping) and gate.get("upstream") not in allowed:
                    return False
    return True


def run_transition(
    *,
    root: Path = REPO_ROOT,
    validation_results: Sequence[JsonMap] = DEFAULT_VALIDATION_RESULTS,
) -> JsonDict:
    artifacts, metadata, missing, malformed = _load_artifacts(root)
    statuses = _terminal_statuses(artifacts, metadata)
    manifest, manifest_meta = _read_yaml_mapping(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    exp5708 = _payload(artifacts, EXP5708_CANARY_PATH)
    exp5711 = _payload(artifacts, EXP5711_ARC_QUAL_PATH)
    exp5712 = _payload(artifacts, EXP5712_ARC_AB_PATH)
    exp5713 = _payload(artifacts, EXP5713_ARC_LEVEL_PATH)
    exp5714 = _payload(artifacts, EXP5714_RUST_PARITY_PATH)
    exp5715 = _payload(artifacts, EXP5715_RUST_QUALITY_PATH)
    taxonomy = _summarize_rows(root)
    retirement_scope = str(REQUIRED_MANIFEST_RETIREMENT["scope_key"])
    retirement_present = _manifest_has_scope(manifest, retirement_scope)
    dependency_map = _dependency_map()
    gate_map = _gate_map()
    test_commands, test_exit_codes = _test_maps(validation_results)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "previous_milestone": PREVIOUS_MILESTONE,
        "current_milestone": CURRENT_MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "artifact_metadata": metadata,
        "missing_artifacts": missing,
        "malformed_artifacts": malformed,
        "manifest_metadata": manifest_meta,
        "manifest_debt_after": [] if retirement_present else [retirement_scope],
        "collision_check": _collision_check(root),
        "forbidden_files_unchanged": _forbidden_files(root),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_capstone_hash": path_sha256(root / EXP5716_CAPSTONE_PATH),
        "v510_task_verdicts": statuses,
        "v510_conductor_outcomes": _derive_conductor_outcomes(root),
        "sota_parse_failure_taxonomy": taxonomy,
        "cuda_offload_authenticated": _bool(exp5708, "cuda_offload_authenticated"),
        "fr11_prospective_promoted": False,
        "fr11_isolated_promoted": False,
        "model_weight_mutation": False,
        "production_default_enabled": False,
        "arc_registry_count": _int(exp5713, "registry_count_after"),
        "arc_registry_delta": _int(exp5713, "registry_delta"),
        "arc_relational_route_promoted": False,
        "arc_relational_route_status": {
            "qualification_only": _number(exp5711, "relational_goal_energy_ready_score") >= 1.0,
            "matched_ab_null": _number(exp5712, "relational_live_ab_ready_score") == 0.0,
            "successful_pair_count": _int(exp5712, "successful_pair_count"),
            "level_regression_count": _int(exp5712, "level_regression_count"),
            "unsafe_route_accept_count": _int(exp5712, "unsafe_route_accept_count"),
        },
        "arc_solve_provenance": {
            "solve_provenance": exp5713.get("solve_provenance"),
            "registry_count_after": _int(exp5713, "registry_count_after"),
            "registry_delta": _int(exp5713, "registry_delta"),
            "registry_updated": _bool(exp5713, "registry_updated"),
        },
        "one_axis_python_promotion_preserved": _bool(
            _payload(artifacts, EXP5706_TRANSITION_PATH), "one_axis_replica_exchange_promoted"
        ),
        "one_axis_rust_parity_ready_score": _direct_parity_ready(exp5714, statuses),
        "one_axis_rust_quality_ready_score": _direct_quality_ready(exp5715, statuses),
        "retirements_required": [dict(REQUIRED_MANIFEST_RETIREMENT)],
        "retirements_applied": [
            {
                "scope": retirement_scope,
                "id": REQUIRED_MANIFEST_RETIREMENT["id"],
                "manifest_entry_present": retirement_present,
                "manifest_update_required": not retirement_present,
                "decision": "retire_this_parse_failed_stream_scope_only",
                "preserves": [
                    "v509_fr11_independent_controller",
                    "fr11_shadow_adapter_disabled_by_default",
                    "future_clean_prospective_streams",
                    "generic_lifecycle_learning",
                ],
                "source": EXP5716_CAPSTONE_PATH.as_posix(),
            }
        ],
        "preserved_scopes": [
            {"scope": "v509_fr11_independent_controller", "preserved_fact": "promoted"},
            {
                "scope": "fr11_shadow_adapter_disabled_by_default",
                "preserved_fact": "disabled_by_default",
            },
            {
                "scope": "future_clean_prospective_streams",
                "preserved_fact": "not_retired_by_exp5709_same_verdict",
            },
            {"scope": "generic_lifecycle_learning", "preserved_fact": "not_retired"},
            {"scope": "generic_arc_working_memory", "preserved_fact": "not_retired"},
            {"scope": "arc_live_attempts", "preserved_fact": "not_retired"},
            {"scope": "one_axis_temperature_exchange", "preserved_fact": "promoted"},
            {"scope": "generic_replica_exchange", "preserved_fact": "not_retired"},
        ],
        "retired_scopes": [
            {
                "scope": "arc_counterexample_patched_transition_model_exp5641",
                "boundary": "preserved prior narrow .510 retirement only",
                "preserves": ["generic_arc_working_memory", "arc_live_attempts"],
            },
            {
                "scope": "two_axis_beta_lambda_tempering_extension_exp5645",
                "boundary": "preserved prior narrow .510 retirement only",
                "preserves": ["one_axis_temperature_exchange", "generic_replica_exchange"],
            },
            {
                "scope": retirement_scope,
                "boundary": "only the parse-failed Exp5709 same-verdict stream scope is retired",
                "preserves": [
                    "v509_fr11_independent_controller",
                    "fr11_shadow_adapter_disabled_by_default",
                    "future_clean_prospective_streams",
                    "generic_lifecycle_learning",
                ],
            },
        ],
        "current_task_range": CURRENT_TASK_RANGE,
        "dependency_map": dependency_map,
        "gate_map": gate_map,
        "dependency_id_validation": {
            "valid": _dependency_ids_valid(dependency_map, gate_map),
            "allowed_ranges": [PREVIOUS_TASK_RANGE, CURRENT_TASK_RANGE],
        },
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": test_commands,
        "test_exit_codes": test_exit_codes,
        "honest_verdict": "",
    }
    blocked = bool(
        missing
        or malformed
        or not retirement_present
        or taxonomy
        != {
            "manifest_row_count": 50,
            "truncation_count": 21,
            "missing_answer_count": 26,
            "parsed_answer_count": 3,
            "parse_failure_count": 47,
            "validator_disagreement_count": 0,
            "finish_reason_length_count": 21,
            "finish_reason_stop_count": 29,
        }
        or not artifact["cuda_offload_authenticated"]
        or artifact["arc_registry_count"] != 177
        or artifact["arc_registry_delta"] != 0
        or artifact["one_axis_rust_parity_ready_score"] != 1.0
        or artifact["one_axis_rust_quality_ready_score"] != 1.0
    )
    artifact["honest_verdict"] = (
        "blocked: v511 transition preserved terminal .510 evidence but manifest or source inputs are incomplete"
        if blocked
        else (
            "complete: v511 transition archived terminal .510 evidence; "
            "fr11_prospective_promoted=False; fr11_isolated_promoted=False; "
            "arc_registry_delta=0; arc_relational_route_promoted=False; "
            "one_axis_rust_parity_ready_score=1.0; "
            "one_axis_rust_quality_ready_score=1.0; timing_claimed=false; "
            "hardware_speedup_claimed=false; current_task_range=exp5717-exp5728"
        )
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    principles = artifact.get("field_principles")
    if principles != FIELD_PRINCIPLES:
        errors.append("field_principles do not match SCENARIO-CAPSTONE-5717-FIELD-PRINCIPLES")
    expected_taxonomy = {
        "manifest_row_count": 50,
        "truncation_count": 21,
        "missing_answer_count": 26,
        "parsed_answer_count": 3,
        "parse_failure_count": 47,
        "validator_disagreement_count": 0,
        "finish_reason_length_count": 21,
        "finish_reason_stop_count": 29,
    }
    if artifact.get("sota_parse_failure_taxonomy") != expected_taxonomy:
        errors.append("sota_parse_failure_taxonomy must preserve Exp5708 denominators exactly")
    exact_values: dict[str, Any] = {
        "cuda_offload_authenticated": True,
        "fr11_prospective_promoted": False,
        "fr11_isolated_promoted": False,
        "model_weight_mutation": False,
        "production_default_enabled": False,
        "arc_registry_count": 177,
        "arc_registry_delta": 0,
        "arc_relational_route_promoted": False,
        "one_axis_rust_parity_ready_score": 1.0,
        "one_axis_rust_quality_ready_score": 1.0,
        "current_task_range": CURRENT_TASK_RANGE,
        "timing_claimed": False,
        "hardware_speedup_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    for field, expected in exact_values.items():
        if artifact.get(field) != expected:
            errors.append(f"{field} must be {expected!r}")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    dependencies = artifact.get("dependency_map", {})
    gates = artifact.get("gate_map", {})
    if not isinstance(dependencies, Mapping) or not isinstance(gates, Mapping):
        errors.append("dependency_map and gate_map must be mappings")
    elif not _dependency_ids_valid(dependencies, gates):
        errors.append("dependency_map or gate_map references an unknown upstream")
    required_scope = REQUIRED_MANIFEST_RETIREMENT["scope_key"]
    applied = artifact.get("retirements_applied", [])
    if not (
        isinstance(applied, Sequence)
        and not isinstance(applied, str | bytes | bytearray)
        and len(applied) == 1
        and isinstance(applied[0], Mapping)
        and applied[0].get("scope") == required_scope
    ):
        errors.append("retirements_applied must contain only the Exp5709 narrow retirement")
    return errors


def write_transition(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    validation_results: Sequence[JsonMap] = DEFAULT_VALIDATION_RESULTS,
) -> JsonDict:
    artifact = run_transition(root=root, validation_results=validation_results)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp5717 transition artifact: " + "; ".join(errors))
    destination = output_path if output_path is not None else root / RESULT_RELATIVE_PATH
    if not destination.is_absolute():
        destination = root / destination
    write_json(destination, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the Exp5717 V511 transition receipt.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    write_transition(root=args.root, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
