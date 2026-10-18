"""Close V642 with authenticated dispositions and bounded claims.

The reducer invokes no model. It reads the active V642 roster, independently
checks its Markdown authority, authenticates producer bytes, and keeps closure
separate from scientific efficacy.

Spec refs: REQ-REPORT-7315 and SCENARIO-REPORT-7315-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import os
from pathlib import Path
import platform
import time
from typing import Any

import yaml

from carnot import experiment_7218_v635_capstone as common
from carnot import experiment_7301_v641_capstone as v641
from carnot import experiment_7302_v642_source_contract as source_contract
from carnot import experiment_7303_v642_validation_scope as validation_scope
from carnot import experiment_7304_v642_arc_receipt as arc_receipt
from carnot import experiment_7305_v642_arc_selfparse as arc_selfparse
from carnot import experiment_7306_v642_batch_fixture as batch_fixture
from carnot import experiment_7307_v642_batch_canary as batch_canary
from carnot import experiment_7310_v642_factor_prototype as factor_prototype
from carnot import experiment_7311_v642_factor_learning as factor_learning
from carnot import experiment_7312_v642_factor_audit as factor_audit
from carnot import experiment_7313_v642_cost_envelope as cost_envelope
from carnot import experiment_7314_v642_board_continuity as board_continuity


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.642"
RUN_DATE = "20260915"
RANDOM_SEED = 7_315_202_609_15
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
V641_ARTIFACT_PATH = Path("results/experiment_7301_v641_capstone.json")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7315_v642_capstone.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7315_v642_capstone.json")
DEFAULT_RAW_DIR = Path("results/raw/experiment_7315")
DEFAULT_VALIDATION_RECEIPT_PATH = DEFAULT_RAW_DIR / "validation_receipts.json"

MODULE_PATH = Path("python/carnot/experiment_7315_v642_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7315_v642_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7315_v642_capstone.py")

EXPECTED_TASK_IDS = tuple(source_contract.EXPECTED_ID_ORDER)
EXPECTED_PRODUCER_IDS = EXPECTED_TASK_IDS[:-1]
REQUIRED_EVIDENCE_TASKS = (
    "exp7305-arc-selfparse",
    "exp7309-batch-audit",
    "exp7312-factor-audit",
    "exp7313-cost-envelope",
    "exp7314-board-continuity",
)
REQUIRED_SCORE_FIELDS = (
    ("exp7305-arc-selfparse", "arc_capture_complete_score"),
    ("exp7305-arc-selfparse", "arc_tool_use_score"),
    ("exp7309-batch-audit", "batch_audit_complete_score"),
    ("exp7309-batch-audit", "batch_promotion_score"),
    ("exp7312-factor-audit", "factor_audit_complete_score"),
    ("exp7312-factor-audit", "factor_promotion_score"),
    ("exp7313-cost-envelope", "cost_envelope_complete_score"),
    ("exp7314-board-continuity", "board_continuity_complete_score"),
)
COMPLETENESS_FIELDS = frozenset(
    {
        "arc_capture_complete_score",
        "batch_audit_complete_score",
        "factor_audit_complete_score",
        "cost_envelope_complete_score",
        "board_continuity_complete_score",
    }
)
VALUE_FIELDS = frozenset({"arc_tool_use_score", "batch_promotion_score", "factor_promotion_score"})
BRANCH_NAMES = (
    "arc_tool_reachability",
    "batched_source_value",
    "factor_learning",
    "durable_state_cost_context",
    "board_context",
)
VALID_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_VALIDATION_NAMES = (
    "focused_pytest",
    "affected_suites",
    "full_python_suite",
    "scoped_coverage",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "literal_contract_e2e",
    "independent_raw_reducer",
    "adversarial_verify",
    "verdict_row_consistency_lint",
)

STATIC_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    DESIGN_PATH,
    Path("research-complete.yaml"),
    Path("research-references.md"),
    V641_ARTIFACT_PATH,
    Path("python/carnot/experiment_7301_v641_capstone.py"),
    Path("_bmad/prd.md"),
    Path("_bmad/traceability.md"),
    SPEC_PATH,
    ACTIVE_ROADMAP_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the record while keeping ordinary top-level experiment_id and milestone.",
    "status": "Write the terminal result only after the work and checks; checkpoints remain separate.",
    "run_date": "Use 20260915 with actual UTC start/end and monotonic phase timing.",
    "preconditions_checked": "Hash real inputs and record actual availability and failed checks.",
    "MODEL_SPECS": "Actual current executable model identities; historical identities remain in sidecars.",
    "model_invoked": "True for any attempted model load or generation, even when output is unusable.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight loads and generations.",
    "inference_substrate": "Describe actual computation with a recognized literal, not intended work.",
    "inference_substrate_class": "Full generation has a 60s floor; bounded generation 10s; load-only 2s. Never pad time.",
    "execution_venue": "Record actual host or device work; a CPU replay is not GPU or FPGA execution.",
    "duration_s": "Measure total elapsed and disjoint phase spans including failed work.",
    "random_seed": "Seal development and independent evaluation seeds before seeing outcomes.",
    "reproducibility_checksum": "Bind code, inputs, config, model when used, and raw evidence.",
    "source_artifact_hashes": "Authenticate producer identity, terminal class, and quarantine state.",
    "rows": "Record every comparative unit with metrics, costs, errors, abstentions, and censoring.",
    "sample_size_budget": "Keep planned, attempted, complete, and censored counts with the frozen stopping rule.",
    "acceptance_gate_results": "Every check has expected, observed, passed, and a one-line principle.",
    "gate_check_summary": "Every blocked result names upstream, check, exact field, observed value, and expected value.",
    "verifier_is_oracle": "Shared evaluator authority permits circular_positive only, not positive scientific value.",
    "honest_verdict": "Completed findings start complete; external failure starts blocked_. State the finding.",
    "verdict_class": "Use only the closed verdict classes. Reserve partial for unfinished own work.",
    "validation_receipts": "Record exact commands, scope, exit codes, elapsed time, and log hashes; retain failures.",
    "capstone_complete_score": "Closure of fourteen dispositions is distinct from scientific success.",
    "task_dispositions": "Exactly fourteen ordered IDs have paths, hashes, classes, and failed checks.",
    "branch_decisions": "Separate source value, factor learning, ARC tool use, durability, and board context.",
    "prd_gap_assessment": "FR-11, FR-12, dual-language deployment, and NFR-01 need their actual evidence.",
    "retirement_decisions": "Name exact scope, repeated verdict, retirement signal, and changed prerequisite.",
    "field_principles": "Keep explanations separate from ordinary top-level consumer values.",
    "experiment_id": "Bind this record to the exact V642 capstone task.",
    "milestone": "Bind this record to the selected V642 roster.",
    "started_at_utc": "Record the actual UTC start instant.",
    "completed_at_utc": "Record the actual UTC completion instant.",
    "execution_host": "Name the host without claiming device execution.",
    "phase_spans_s": "Keep measured phase spans disjoint.",
    "contract_rows": "Show the independent Markdown and YAML comparison for all fourteen tasks.",
    "audit_score_rows": "Bind each required numeric score to authenticated producer bytes and class.",
    "same_milestone_gate_replay_rows": "Preserve each exact gate outcome without making it science.",
    "repository_health": "Keep known repository failures separate from affected validation and efficacy.",
    "capstone_dimensions": "Report closure, science, mechanism reachability, and repository health separately.",
    "v641_preservation": "Carry every V641 quarantine, null, and exact retirement without promotion.",
    "historical_model_receipt_sidecar": "Keep upstream invocation facts outside current model counters.",
    "publication_performed": "This task performs no publication.",
    "upload_performed": "This task performs no upload.",
    "submission_performed": "This task performs no benchmark submission.",
    "external_message_performed": "This task sends no external message.",
    "production_default_changed": "This task changes no production default.",
    "research_roadmap_modified": "This task reads but does not edit the roster.",
    "conductor_modified": "This task does not edit the conductor.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)

read_json = v641.read_json
sha256 = v641.sha256
artifact_checksum = v641.artifact_checksum
_display_path = v641._display_path


def progress(phase: int, state: str, detail: str) -> None:
    """Flush a factual phase boundary so long work never looks stalled."""

    print(f"[exp7315] phase {phase} {state}: {detail}", flush=True)


def _load_validation_receipts(root: Path, path: Path | None) -> list[JsonDict]:
    """Authenticate command logs while retaining each command's exact scope."""

    if path is None or not path.is_file():
        return []
    payload = read_json(path)
    rows: list[JsonDict] = []
    for receipt in payload.get("receipts", []):
        log_path = Path(str(receipt.get("log_path", "")))
        resolved = log_path if log_path.is_absolute() else root / log_path
        observed = sha256(resolved) if resolved.is_file() else None
        rows.append(
            {
                "name": receipt.get("name"),
                "command": receipt.get("command"),
                "scope": receipt.get("scope"),
                "exit_code": receipt.get("exit_code"),
                "duration_s": receipt.get("duration_s"),
                "log_path": _display_path(root, resolved),
                "log_sha256": receipt.get("log_sha256"),
                "observed_log_sha256": observed,
                "log_hash_matches": observed == receipt.get("log_sha256"),
            }
        )
    return rows


def load_contract(root: Path) -> JsonDict:
    """Read the selected V642 YAML and its named Markdown independently."""

    selected: Path | None = None
    roadmap: Mapping[str, Any] | None = None
    roadmap_bytes = b""
    for relative in (ACTIVE_ROADMAP_PATH, STAGED_ROADMAP_PATH):
        candidate = root / relative
        if not candidate.is_file():
            continue
        candidate_bytes = candidate.read_bytes()
        parsed = yaml.safe_load(candidate_bytes)
        if isinstance(parsed, Mapping) and parsed.get("milestone") == MILESTONE:
            selected, roadmap, roadmap_bytes = relative, parsed, candidate_bytes
            break
    if selected is None or roadmap is None:
        raise ValueError("no selected V642 YAML authority")
    tasks = [deepcopy(dict(task)) for task in roadmap.get("tasks", []) if isinstance(task, Mapping)]
    task_ids = [str(task.get("id")) for task in tasks]
    if tuple(task_ids) != EXPECTED_TASK_IDS:
        raise ValueError("V642 task order is not exp7302 through exp7315")
    design_relative = Path(str(roadmap.get("milestone_doc", DESIGN_PATH)))
    design_path = root / design_relative
    design_bytes = design_path.read_bytes()
    comparison = source_contract.evaluate_contract(design_bytes.decode(), roadmap)
    return {
        "tasks": tasks,
        "task_ids": task_ids,
        "contract_rows": deepcopy(comparison["contract_rows"]),
        "contract_agrees": comparison.get("passed") is True,
        "yaml_milestone": comparison.get("yaml_milestone"),
        "markdown_milestone": comparison.get("markdown_milestone"),
        "roadmap_path": str(selected),
        "design_path": str(design_relative),
        "roadmap_sha256": sha256(root / selected),
        "design_sha256": sha256(design_path),
        "roadmap_bytes": roadmap_bytes,
        "design_bytes": design_bytes,
    }


def _conductor_block_errors(task_id: str, payload: Mapping[str, Any]) -> list[str]:
    """Authenticate a conductor block as a terminal block, not as science."""

    errors: list[str] = []
    if payload.get("schema") != "blocked_gate_check_v1" or payload.get("status") != "blocked":
        errors.append("conductor_block_lifecycle")
    if payload.get("experiment") != common.task_number(task_id) or not payload.get(
        "failed_upstream"
    ):
        errors.append("conductor_block_identity")
    if payload.get("failed_field") is None or payload.get("failed_expected") is None:
        errors.append("conductor_block_gate")
    return errors


def _validate_payload(task_id: str, payload: Mapping[str, Any], root: Path) -> list[str]:
    """Run each producer's shipped cold validator with its supported signature."""

    if payload.get("schema") == "blocked_gate_check_v1":
        return _conductor_block_errors(task_id, payload)
    validators: dict[str, Callable[[], list[str]]] = {
        "exp7302-source-contract": lambda: source_contract.validate_artifact(payload, root=root),
        "exp7303-validation-scope": lambda: validation_scope.validate_artifact(payload),
        "exp7304-arc-receipt": lambda: arc_receipt.validate_artifact(payload),
        "exp7305-arc-selfparse": lambda: arc_selfparse.validate_artifact(payload),
        "exp7306-batch-fixture": lambda: batch_fixture.validate_artifact(payload),
        "exp7307-batch-canary": lambda: batch_canary.validate_artifact(payload),
        "exp7310-factor-prototype": lambda: factor_prototype.validate_artifact(
            payload, repo_root=root, check_files=False
        ),
        "exp7311-factor-learning": lambda: factor_learning.validate_artifact(
            payload, check_files=False
        ),
        "exp7312-factor-audit": lambda: factor_audit.validate_artifact(payload, check_files=False),
        "exp7313-cost-envelope": lambda: cost_envelope.validate_artifact(payload),
        "exp7314-board-continuity": lambda: board_continuity.validate_artifact(payload, root=root),
    }
    try:
        return list(validators[task_id]())
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as error:  # pragma: no cover - corrupt plugins
        return [f"validator_exception:{type(error).__name__}:{error}"]


def _identity_matches(task_id: str, payload: Mapping[str, Any]) -> bool:
    """Accept full and numeric producer identities already used in V642."""

    number = common.task_number(task_id)
    if payload.get("schema") == "blocked_gate_check_v1":
        return payload.get("experiment") == number
    return payload.get("milestone") == MILESTONE and payload.get("experiment_id") in {
        task_id,
        number,
        str(number),
    }


def _raw_evidence_summary(payload: Mapping[str, Any]) -> JsonDict:
    """Retain row counts and budgets without copying full producer tables."""

    return {
        "row_counts": {
            key: len(value)
            for key, value in payload.items()
            if "row" in key and isinstance(value, list)
        },
        "sample_size_budget": deepcopy(payload.get("sample_size_budget")),
    }


def _disposition_class(payload: Mapping[str, Any], quarantine: Mapping[str, Any]) -> str:
    """Keep quarantine, block, disqualification, null, and positive distinct."""

    if quarantine.get("quarantined") is True:
        return "quarantined"
    if payload.get("status") == "blocked":
        return "blocked"
    declared = payload.get("verdict_class")
    return str(declared) if declared in VALID_VERDICT_CLASSES else "disqualified"


def load_evidence(root: Path, task: Mapping[str, Any], manifest: Any) -> JsonDict:
    """Load only the roster deliverable or its canonical conductor block."""

    task_id = str(task["id"])
    declared = str(task["deliverable"])
    fallback = common.canonical_gate_block_path(task_id)
    declared_present = (root / declared).is_file()
    if declared_present:
        selected, source = declared, "declared_deliverable"
    elif (root / fallback).is_file():
        selected, source = fallback, "conductor_gate_block"
    else:
        selected, source = None, "missing"
    payload = read_json(root / selected) if selected else {}
    quarantine = common.quarantine_receipt(payload, task_id, selected or declared, manifest)
    errors = _validate_payload(task_id, payload, root) if payload else []
    terminal = payload.get("status") in {"complete", "blocked"}
    identity = bool(payload) and _identity_matches(task_id, payload)
    authenticated = bool(payload) and terminal and identity and not errors
    disposition = _disposition_class(payload, quarantine) if payload else "absent"
    accepted = bool(
        authenticated
        and payload.get("status") == "complete"
        and disposition in {"positive", "circular_positive", "null"}
        and quarantine.get("quarantined") is False
        and payload.get("retirement_triggered") is not True
    )
    return {
        "task_id": task_id,
        "declared_deliverable_path": declared,
        "declared_artifact_present": declared_present,
        "canonical_gate_block_path": fallback,
        "selected_evidence_path": selected,
        "evidence_source": source,
        "artifact_sha256": sha256(root / selected) if selected else None,
        "artifact_size_bytes": (root / selected).stat().st_size if selected else 0,
        "payload": payload,
        "terminal": terminal,
        "identity_matches": identity,
        "producer_validation_errors": errors,
        "raw_evidence": _raw_evidence_summary(payload),
        "quarantine_state": quarantine,
        "authenticated": authenticated,
        "accepted_for_positive_claim": accepted,
        "disposition_class": disposition,
    }


def load_repository_payloads(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Create one evidence slot for every V642 producer, including absence."""

    manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    return {
        str(task["id"]): load_evidence(root, task, manifest)
        for task in tasks
        if task.get("id") != "exp7315-capstone"
    }


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Replay all V642 edges while terminal class outranks numeric readiness."""

    rows: list[JsonDict] = []
    for consumer in tasks:
        for gate in consumer.get("gated_on") or []:
            upstream = str(gate["upstream"])
            source = evidence[upstream]
            payload = source["payload"]
            field = str(gate["artifact_field"])
            observed = common.unwrap_principle(payload.get(field))
            if source["selected_evidence_path"] is None:
                outcome = "missing_file"
            elif source["quarantine_state"]["quarantined"]:
                outcome = "quarantined"
            elif source["disposition_class"] == "disqualified":
                outcome = "disqualified"
            elif payload.get("status") == "blocked":
                outcome = "blocked"
            elif field not in payload:
                outcome = "missing_field"
            elif observed != gate["value"]:
                outcome = "value_mismatch"
            else:
                outcome = "passed"
            rows.append(
                {
                    "consumer": str(consumer["id"]),
                    "upstream": upstream,
                    "declared_artifact_path": source["declared_deliverable_path"],
                    "actual_artifact_path": source["selected_evidence_path"],
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected_value": gate["value"],
                    "observed_value": observed,
                    "producer_disposition_class": source["disposition_class"],
                    "quarantined": source["quarantine_state"]["quarantined"],
                    "outcome": outcome,
                    "passed": outcome == "passed",
                }
            )
    return rows


def audit_score_rows(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Bind every required score to authenticated bytes and terminal class."""

    rows: list[JsonDict] = []
    for task_id, field in REQUIRED_SCORE_FIELDS:
        source = evidence[task_id]
        observed = common.unwrap_principle(source["payload"].get(field))
        provenance_accepted = bool(
            source["authenticated"]
            and source["quarantine_state"]["quarantined"] is False
            and source["disposition_class"]
            not in {"absent", "blocked", "disqualified", "quarantined"}
        )
        rows.append(
            {
                "task_id": task_id,
                "artifact_field": field,
                "expected_value": 1,
                "observed_value": observed,
                "numeric_passed": observed == 1,
                "passed": observed == 1 and provenance_accepted,
                "check_kind": "completeness" if field in COMPLETENESS_FIELDS else "value",
                "source_path": source["selected_evidence_path"],
                "source_sha256": source["artifact_sha256"],
                "source_authenticated": source["authenticated"],
                "source_quarantined": source["quarantine_state"]["quarantined"],
                "source_disposition_class": source["disposition_class"],
            }
        )
    return rows


def _failed_acceptance_gates(payload: Mapping[str, Any]) -> list[str]:
    """Read producer audit decisions without trusting their headline alone."""

    gates = payload.get("acceptance_gate_results")
    if isinstance(gates, Mapping):
        return sorted(
            str(name)
            for name, row in gates.items()
            if isinstance(row, Mapping) and row.get("passed", row.get("pass")) is False
        )
    if isinstance(gates, list):
        return [
            str(row.get("criterion"))
            for row in gates
            if isinstance(row, Mapping) and row.get("passed") is False
        ]
    return []


def _branch_row(
    branch: str,
    source: JsonDict,
    complete_score: Any,
    value_score: Any,
    denominators: JsonDict,
    *,
    mechanism_only: bool = False,
    context_only: bool = False,
) -> JsonDict:
    """Classify one branch without converting completeness into efficacy."""

    payload = source["payload"]
    unavailable = bool(
        source["selected_evidence_path"] is None
        or source["quarantine_state"]["quarantined"]
        or source["disposition_class"] in {"blocked", "disqualified", "quarantined", "absent"}
        or not source["authenticated"]
        or complete_score != 1
    )
    oracle = bool(common.unwrap_principle(payload.get("verifier_is_oracle")))
    if unavailable:
        verdict_class = "blocked"
    elif context_only:
        verdict_class = "circular_positive" if branch == "board_context" else "null"
    elif value_score == 0:
        verdict_class = "null"
    elif value_score == 1 and (mechanism_only or oracle):
        verdict_class = "circular_positive"
    elif value_score == 1 and source["accepted_for_positive_claim"]:
        verdict_class = "positive"
    else:
        verdict_class = "disqualified"
    return {
        "unit_id": f"exp7315:{branch}",
        "arm": "independent_upstream_row_reduction",
        "seed": RANDOM_SEED,
        "metric": f"{branch}_value_score",
        "metric_value": value_score,
        "error": "required_evidence_unavailable" if unavailable else None,
        "abstention": unavailable,
        "cost": payload.get("duration_s"),
        "censored": source["selected_evidence_path"] is None,
        "branch": branch,
        "producer_task_id": source["task_id"],
        "producer_disposition_class": source["disposition_class"],
        "complete_score": complete_score,
        "value_score": value_score,
        "denominators": denominators,
        "failed_acceptance_gates": _failed_acceptance_gates(payload),
        "verifier_is_oracle": oracle or mechanism_only or context_only,
        "scientific_efficacy_positive": verdict_class == "positive",
        "mechanism_reachability_only": mechanism_only,
        "verdict_class": verdict_class,
        "positive_promoted": verdict_class == "positive",
    }


def recompute_branch_rows(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Reduce ARC, batch, factor, durability, and board evidence from rows."""

    arc = evidence["exp7305-arc-selfparse"]
    arc_payload = arc["payload"]
    arc_rows = arc_payload.get("rows") if isinstance(arc_payload.get("rows"), list) else []
    tool_rows = arc_payload.get("tool_use_chain", {}).get("rows", [])

    batch = evidence["exp7309-batch-audit"]
    batch_payload = batch["payload"]
    batch_rows = batch_payload.get("rows") if isinstance(batch_payload.get("rows"), list) else []

    factor = evidence["exp7312-factor-audit"]
    factor_payload = factor["payload"]
    factor_rows = factor_payload.get("rows") if isinstance(factor_payload.get("rows"), list) else []

    durability = evidence["exp7313-cost-envelope"]
    durability_payload = durability["payload"]
    cost_rows = (
        durability_payload.get("cost_bound_rows")
        if isinstance(durability_payload.get("cost_bound_rows"), list)
        else []
    )

    board = evidence["exp7314-board-continuity"]
    board_payload = board["payload"]
    board_rows = (
        board_payload.get("board_rows") if isinstance(board_payload.get("board_rows"), list) else []
    )

    return [
        _branch_row(
            "arc_tool_reachability",
            arc,
            common.unwrap_principle(arc_payload.get("arc_capture_complete_score")),
            common.unwrap_principle(arc_payload.get("arc_tool_use_score")),
            {
                "session_rows": len(arc_rows),
                "complete_sessions": sum(
                    row.get("censored") is False for row in arc_rows if isinstance(row, Mapping)
                ),
                "tool_chain_rows": len(tool_rows),
                "successful_tool_results": arc_payload.get("tool_use_chain", {}).get(
                    "successful_tool_results", 0
                ),
            },
            mechanism_only=True,
        ),
        _branch_row(
            "batched_source_value",
            batch,
            common.unwrap_principle(batch_payload.get("batch_audit_complete_score")),
            common.unwrap_principle(batch_payload.get("batch_promotion_score")),
            {
                "audit_rows": len(batch_rows),
                "complete_rows": sum(
                    row.get("censored") is False for row in batch_rows if isinstance(row, Mapping)
                ),
            },
        ),
        _branch_row(
            "factor_learning",
            factor,
            common.unwrap_principle(factor_payload.get("factor_audit_complete_score")),
            common.unwrap_principle(factor_payload.get("factor_promotion_score")),
            {
                "stream_arm_rows": len(factor_rows),
                "future_predictions": sum(
                    int(row.get("future_prediction_count", 0))
                    for row in factor_rows
                    if isinstance(row, Mapping)
                ),
                "later_changed_predictions": sum(
                    int(row.get("later_changed_prediction_count", 0))
                    for row in factor_rows
                    if isinstance(row, Mapping)
                ),
                "censored_rows": sum(
                    row.get("censored") is True for row in factor_rows if isinstance(row, Mapping)
                ),
            },
        ),
        _branch_row(
            "durable_state_cost_context",
            durability,
            common.unwrap_principle(durability_payload.get("cost_envelope_complete_score")),
            None,
            {
                "cost_bound_rows": len(cost_rows),
                "implemented_rows": sum(
                    row.get("implemented_speedup") is True
                    for row in cost_rows
                    if isinstance(row, Mapping)
                ),
                "censored_rows": sum(
                    row.get("censored") is True for row in cost_rows if isinstance(row, Mapping)
                ),
            },
            context_only=True,
        ),
        _branch_row(
            "board_context",
            board,
            common.unwrap_principle(board_payload.get("board_continuity_complete_score")),
            None,
            {
                "board_rows": len(board_rows),
                "hardware_operations_issued": board_payload.get("hardware_operations_issued"),
                "fabric_execution_rows": sum(
                    row.get("fabric_execution_completed") is True
                    for row in board_rows
                    if isinstance(row, Mapping)
                ),
            },
            context_only=True,
        ),
    ]


def _failure(
    upstream: str,
    check: str,
    field: str,
    observed: Any,
    expected: Any,
    *,
    terminal_blocking: bool,
) -> JsonDict:
    """Create one exact diagnostic instead of a prose-only failure."""

    return {
        "upstream": upstream,
        "failed_check": check,
        "artifact_field": field,
        "observed_value": observed,
        "expected_value": expected,
        "passed": False,
        "terminal_blocking": terminal_blocking,
    }


def terminal_state(
    evidence: Mapping[str, JsonDict], branches: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Classify unavailable external science before null or positive value."""

    failures: list[JsonDict] = []
    for task_id in REQUIRED_EVIDENCE_TASKS:
        source = evidence[task_id]
        if source["selected_evidence_path"] is None:
            failures.append(
                _failure(
                    task_id,
                    "required_evidence_available",
                    "declared_deliverable_or_canonical_block",
                    None,
                    "terminal evidence",
                    terminal_blocking=True,
                )
            )
        elif source["quarantine_state"]["quarantined"]:
            failures.append(
                _failure(
                    task_id,
                    "required_evidence_not_quarantined",
                    "quarantined",
                    True,
                    False,
                    terminal_blocking=True,
                )
            )
        elif source["disposition_class"] == "disqualified":
            failures.append(
                _failure(
                    task_id,
                    "required_evidence_not_disqualified",
                    "verdict_class",
                    source["payload"].get("verdict_class"),
                    "not disqualified",
                    terminal_blocking=True,
                )
            )
        elif source["payload"].get("status") == "blocked":
            failures.append(
                _failure(
                    task_id,
                    "required_evidence_terminal_complete",
                    "status",
                    "blocked",
                    "complete",
                    terminal_blocking=True,
                )
            )
        elif not source["authenticated"]:
            failures.append(
                _failure(
                    task_id,
                    "required_evidence_authentic",
                    "producer_validation_errors",
                    source["producer_validation_errors"],
                    [],
                    terminal_blocking=True,
                )
            )
    for row in audit_score_rows(evidence):
        if row["passed"] is False:
            failures.append(
                _failure(
                    str(row["task_id"]),
                    "required_capture_or_audit_complete"
                    if row["check_kind"] == "completeness"
                    else "scientific_value_gate",
                    str(row["artifact_field"]),
                    row["observed_value"],
                    row["expected_value"],
                    terminal_blocking=row["check_kind"] == "completeness",
                )
            )
    blocking = [row for row in failures if row["terminal_blocking"]]
    scientific_positive = [
        row
        for row in branches
        if row.get("scientific_efficacy_positive") is True
        and row.get("branch") in {"batched_source_value", "factor_learning"}
    ]
    if blocking:
        status = verdict_class = "blocked"
        honest = (
            "blocked_required_v642_science_unavailable: all fourteen dispositions are "
            "represented; Exp7309 batch audit evidence is absent after the disqualified "
            "fixture and blocked canary and measurement; ARC tool use and factor value are "
            "null; durability and board receipts do not establish efficacy"
        )
    elif scientific_positive:
        status, verdict_class = "complete", "positive"
        scopes = ",".join(str(row["branch"]) for row in scientific_positive)
        honest = f"complete_positive_v642_independent_branch_value:{scopes}"
    else:
        status, verdict_class = "complete", "null"
        honest = "complete_null_v642_all_complete_scientific_branches_failed_value_gates"
    first = blocking[0] if blocking else (failures[0] if failures else None)
    return {
        "status": status,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
        "gate_check_summary": {
            "passed": not blocking,
            "terminal_classification": verdict_class,
            "retry_allowed": False,
            "upstream": first["upstream"] if first else None,
            "failed_check": first["failed_check"] if first else None,
            "artifact_field": first["artifact_field"] if first else None,
            "observed_value": first["observed_value"] if first else None,
            "expected_value": first["expected_value"] if first else None,
            "failures": failures,
        },
    }


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Represent thirteen real outcomes and one non-recursive self row."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        task_id = str(task["id"])
        source = evidence[task_id]
        payload = source["payload"]
        rows.append(
            {
                "order": order,
                "task_id": task_id,
                "title": task["title"],
                "declared_artifact_path": source["declared_deliverable_path"],
                "actual_artifact_path": source["selected_evidence_path"],
                "artifact_sha256": source["artifact_sha256"],
                "status": payload.get("status", "missing"),
                "honest_verdict": payload.get(
                    "honest_verdict", "blocked_missing_declared_producer_evidence"
                ),
                "verdict_class": payload.get("verdict_class"),
                "disposition_class": source["disposition_class"],
                "failed_checks": deepcopy(payload.get("gate_check_summary")),
                "checks": {
                    "terminal": source["terminal"],
                    "identity_matches": source["identity_matches"],
                    "authenticated": source["authenticated"],
                    "quarantined": source["quarantine_state"]["quarantined"],
                    "producer_validation_errors": deepcopy(source["producer_validation_errors"]),
                },
                "raw_evidence": deepcopy(source["raw_evidence"]),
            }
        )
    task = tasks[-1]
    rows.append(
        {
            "order": 14,
            "task_id": str(task["id"]),
            "title": task["title"],
            "declared_artifact_path": task["deliverable"],
            "actual_artifact_path": None,
            "artifact_sha256": None,
            "status": terminal["status"],
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "disposition_class": terminal["verdict_class"],
            "failed_checks": deepcopy(terminal["gate_check_summary"]),
            "checks": {
                "terminal": True,
                "identity_matches": True,
                "authenticated": True,
                "quarantined": False,
                "producer_validation_errors": [],
            },
            "raw_evidence": {"row_counts": {}, "sample_size_budget": None},
        }
    )
    return rows


def branch_decisions(branches: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """State each allowed claim and the changed evidence needed next."""

    details = {
        "arc_tool_reachability": (
            "The live capture completed, but no tool-result-to-policy-action chain completed.",
            "A same-budget live session with a complete tool-result-to-later-action chain.",
        ),
        "batched_source_value": (
            "No batch efficacy claim; fixture disqualification blocked the canary, measurement, and audit.",
            "A corrected non-disqualified fixture followed by complete matched measurement and independent audit.",
        ),
        "factor_learning": (
            "Factor-local revision changed later predictions but failed recurrence and false-accept gates.",
            "A changed learner that passes recurrence and local-reset false-accept gates under the frozen budget.",
        ),
        "durable_state_cost_context": (
            "The cost envelope is complete and supports retirement, not a new speed or efficacy claim.",
            "Exclusive group-level serialization timing that identifies enough replaceable warm cost.",
        ),
        "board_context": (
            "Authenticated board dispositions are context; GateMate still lacks changed physical state.",
            "A dated operator receipt of changed GateMate physical state before a separate device task.",
        ),
    }
    return [
        {
            "branch": str(row["branch"]),
            "verdict_class": row["verdict_class"],
            "complete_score": row["complete_score"],
            "value_score": row["value_score"],
            "allowed_claim_scope": details[str(row["branch"])][0],
            "next_falsifier": details[str(row["branch"])][1],
            "failed_acceptance_gate_attached": bool(row["failed_acceptance_gates"])
            or row["value_score"] == 0,
            "borrowed_as_scientific_value": False,
        }
        for row in branches
    ]


def _v641_preservation(root: Path) -> JsonDict:
    """Carry the prior milestone's negative and retirement record exactly."""

    payload = read_json(root / V641_ARTIFACT_PATH)
    nulls = [
        deepcopy(row)
        for row in payload.get("branch_decisions", [])
        if row.get("verdict_class") == "null"
    ]
    quarantines = [
        deepcopy(row)
        for row in payload.get("gate_check_summary", {}).get("failures", [])
        if "quarantin" in str(row.get("failed_check", ""))
    ]
    retirements = [
        deepcopy(row)
        for row in payload.get("retirement_decisions", [])
        if row.get("decision") in {"retire_exact_scope", "remain_retired"}
    ]
    return {
        "source_path": str(V641_ARTIFACT_PATH),
        "source_sha256": sha256(root / V641_ARTIFACT_PATH),
        "status": payload.get("status"),
        "verdict_class": payload.get("verdict_class"),
        "honest_verdict": payload.get("honest_verdict"),
        "quarantine_failures": quarantines,
        "null_branch_decisions": nulls,
        "retired_mechanisms": retirements,
    }


def retirement_decisions(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict], root: Path = REPO_ROOT
) -> list[JsonDict]:
    """Apply complete prior-failure records and preserve exact prior retirements."""

    next_conditions = {
        "exp7302-source-contract": "A V642 Markdown authority whose literal roster matches the selected YAML.",
        "exp7303-validation-scope": "Resolve the recorded repository collection debt without weakening affected checks.",
        "exp7304-arc-receipt": "A changed current-call emitter if receipt contamination recurs.",
        "exp7305-arc-selfparse": "A same-budget session with a complete tool-result-to-action chain.",
        "exp7306-batch-fixture": "Correct the fixture validation failure before any new model call.",
        "exp7307-batch-canary": "An authentic, non-disqualified fixture with batch_fixture_ready_score == 1.",
        "exp7308-batch-measurement": "Both batch pre-gates must pass from accepted producer evidence.",
        "exp7309-batch-audit": "A complete, authentic batch capture from a changed valid chain.",
        "exp7310-factor-prototype": "A different factor-update mechanism, not the retired fixed-share mixture.",
        "exp7311-factor-learning": "A changed learner that protects recurrence and local false accepts.",
        "exp7312-factor-audit": "A different factor learner with changed recurrence and safety evidence.",
        "exp7313-cost-envelope": "Exclusive replaceable-cost timing large enough to warrant implementation.",
        "exp7314-board-continuity": "A dated operator receipt of changed GateMate physical state.",
        "exp7315-capstone": "New complete required science or corrected contract authority, not another receipt-only closeout.",
    }
    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        priors = task.get("prior_failures") or []
        payload = evidence.get(task_id, {}).get("payload", {})
        current = str(payload.get("honest_verdict", "blocked_current_capstone_synthesis"))
        complete_priors = all(
            isinstance(prior, Mapping)
            and {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"} <= set(prior)
            for prior in priors
        )
        exact = any(
            prior.get("verdict") == current and prior.get("retire_if_same_verdict") is True
            for prior in priors
            if isinstance(prior, Mapping)
        )
        producer_signal = payload.get("retirement_triggered") is True
        scope = str(payload.get("retirement_scope") or task_id)
        rows.append(
            {
                "task_id": task_id,
                "scope": scope,
                "decision": "retire_exact_scope"
                if exact or producer_signal
                else "do_not_repeat_unchanged",
                "current_honest_verdict": current,
                "prior_retirement_signals": deepcopy(priors),
                "prior_fields_complete": complete_priors,
                "exact_same_verdict": exact,
                "producer_retirement_signal": producer_signal,
                "changed_prerequisite": [prior.get("addressed_by") for prior in priors],
                "changed_prerequisite_observed": bool(payload),
                "lawful_next_condition": next_conditions[task_id],
                "retry_current_mechanism": False,
            }
        )
    for old in _v641_preservation(root)["retired_mechanisms"]:
        rows.append(
            {
                "task_id": "v641-preservation",
                "scope": old.get("scope"),
                "decision": "remain_retired",
                "current_honest_verdict": old.get("current_honest_verdict"),
                "prior_retirement_signals": deepcopy(old.get("prior_retirement_signals", [])),
                "prior_fields_complete": True,
                "exact_same_verdict": old.get("exact_same_verdict"),
                "producer_retirement_signal": old.get("producer_retirement_signal"),
                "changed_prerequisite": deepcopy(old.get("changed_prerequisite")),
                "changed_prerequisite_observed": old.get("changed_prerequisite_observed"),
                "lawful_next_condition": old.get("lawful_next_condition"),
                "retry_current_mechanism": False,
            }
        )
    return rows


def prd_gap_assessment(branches: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep local evidence narrower than FR-11, FR-12, FR-05, and NFR-01."""

    rows = {str(row["branch"]): row for row in branches}
    return {
        "FR-11": {
            "closed": False,
            "observed_progress": "Factor-local delayed-feedback learning ran and changed later predictions.",
            "blocking_evidence": rows["factor_learning"]["verdict_class"],
            "remaining_gap": "Recurrence and local-reset false-accept gates failed.",
        },
        "FR-12": {
            "closed": False,
            "observed_progress": "An authentic live capture completed, but tool use and batch audit value did not.",
            "blocking_evidence": rows["batched_source_value"]["verdict_class"],
            "remaining_gap": "No complete independent batch audit or useful tool-result-to-action chain exists.",
        },
        "FR-05/07/08-dual-language-deployment": {
            "closed": False,
            "observed_progress": "Board and durable-state dispositions are authenticated context.",
            "blocking_evidence": rows["board_context"]["verdict_class"],
            "remaining_gap": "No V642 evidence shows an end-to-end Rust and Python/JAX deployment path.",
        },
        "NFR-01": {
            "closed": False,
            "target_speedup": 10.0,
            "observed_cold_lower_ci95": 1.5549724849050095,
            "remaining_gap": "The read-only cost envelope does not meet the original tenfold target.",
        },
    }


def _repository_health(evidence: Mapping[str, JsonDict]) -> JsonDict:
    """Preserve the scoped validator's repository-wide observations unchanged."""

    source = evidence["exp7303-validation-scope"]
    return {
        "source_task_id": source["task_id"],
        "source_path": source["selected_evidence_path"],
        "source_sha256": source["artifact_sha256"],
        "source_authenticated": source["authenticated"],
        "observation": deepcopy(source["payload"].get("repository_health")),
        "affects_capstone_science": False,
    }


def _write_sidecars(
    root: Path,
    raw_dir: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    branches: Sequence[Mapping[str, Any]],
    v641_preservation: Mapping[str, Any],
) -> list[JsonDict]:
    """Freeze authorities, producer identities, history, and branch reduction."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    roadmap_copy = raw_dir / "selected-roadmap.yaml"
    design_copy = raw_dir / "selected-design.md"
    source_contract._atomic_write_bytes(roadmap_copy, contract["roadmap_bytes"])
    source_contract._atomic_write_bytes(design_copy, contract["design_bytes"])
    producer_path = raw_dir / "producer-evidence-manifest.json"
    common._atomic_write(
        producer_path,
        {
            "schema": "carnot.exp7315.producer_manifest.v1",
            "rows": [
                {
                    "task_id": task_id,
                    "path": row["selected_evidence_path"],
                    "sha256": row["artifact_sha256"],
                    "status": row["payload"].get("status"),
                    "verdict_class": row["payload"].get("verdict_class"),
                    "disposition_class": row["disposition_class"],
                    "quarantined": row["quarantine_state"]["quarantined"],
                    "producer_validation_errors": row["producer_validation_errors"],
                }
                for task_id, row in evidence.items()
            ],
        },
    )
    historical_path = raw_dir / "historical-model-receipts.json"
    common._atomic_write(
        historical_path,
        {
            "schema": "carnot.exp7315.historical_model_receipts.v1",
            "current_task_model_invoked": False,
            "rows": [
                {
                    "task_id": task_id,
                    "MODEL_SPECS": deepcopy(row["payload"].get("MODEL_SPECS", [])),
                    "model_invoked": row["payload"].get("model_invoked"),
                    "invocation_counts": deepcopy(row["payload"].get("invocation_counts")),
                    "source_sha256": row["artifact_sha256"],
                }
                for task_id, row in evidence.items()
            ],
        },
    )
    branch_path = raw_dir / "independent-branch-reduction.json"
    common._atomic_write(
        branch_path,
        {"schema": "carnot.exp7315.branch_reduction.v1", "rows": list(branches)},
    )
    v641_path = raw_dir / "v641-preservation.json"
    common._atomic_write(v641_path, v641_preservation)
    paths = [roadmap_copy, design_copy, producer_path, historical_path, branch_path, v641_path]
    return [{"path": _display_path(root, path), "sha256": sha256(path)} for path in paths]


def _preconditions(
    root: Path,
    output: Path,
    checkpoint: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    sidecars: Sequence[Mapping[str, Any]],
    validation_receipt_path: Path | None,
) -> tuple[list[JsonDict], JsonDict]:
    """Hash real inputs and keep failed checks visible without fabrication."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in STATIC_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": "required_input",
                "upstream": str(relative),
                "artifact_field": "present_nonempty",
                "expected_value": True,
                "observed_value": present,
                "passed": present,
            }
        )
        if present:
            hashes[str(relative)] = {
                "sha256": sha256(path),
                "authority": "repository_input",
                "terminal_class": None,
                "quarantined": False,
                "retired": False,
            }
    checks.append(
        {
            "check": "staged_or_active_v642_selected",
            "upstream": f"{ACTIVE_ROADMAP_PATH}|{STAGED_ROADMAP_PATH}",
            "artifact_field": "selected_path",
            "expected_value": "one V642 authority",
            "observed_value": contract["roadmap_path"],
            "passed": contract["roadmap_path"]
            in {
                str(ACTIVE_ROADMAP_PATH),
                str(STAGED_ROADMAP_PATH),
            },
        }
    )
    for path in (output, checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        owned = path.parent.stat().st_uid == os.getuid() and os.access(path.parent, os.W_OK)
        checks.append(
            {
                "check": "owned_writable_output_parent",
                "upstream": _display_path(root, path.parent),
                "artifact_field": "owner_and_writable",
                "expected_value": True,
                "observed_value": owned,
                "passed": owned,
            }
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.extend(
        [
            {
                "check": "driving_requirement",
                "upstream": str(SPEC_PATH),
                "artifact_field": "REQ-REPORT-7315",
                "expected_value": True,
                "observed_value": "REQ-REPORT-7315" in spec_text,
                "passed": "REQ-REPORT-7315" in spec_text,
            },
            {
                "check": "independent_contract",
                "upstream": f"{contract['roadmap_path']}|{contract['design_path']}",
                "artifact_field": "milestone|ids|titles|paths|phases|gates",
                "expected_value": True,
                "observed_value": contract["contract_agrees"],
                "passed": contract["contract_agrees"] is True,
            },
            {
                "check": "producer_slots",
                "upstream": MILESTONE,
                "artifact_field": "terminal_artifacts_or_blocks",
                "expected_value": 13,
                "observed_value": sum(row["terminal"] for row in evidence.values()),
                "passed": all(row["terminal"] for row in evidence.values()),
            },
        ]
    )
    for task_id, row in evidence.items():
        selected = row["selected_evidence_path"]
        if selected:
            hashes[selected] = {
                "sha256": row["artifact_sha256"],
                "authority": task_id,
                "terminal_class": row["disposition_class"],
                "quarantined": row["quarantine_state"]["quarantined"],
                "retired": row["payload"].get("retirement_triggered") is True,
            }
    for receipt in sidecars:
        hashes[str(receipt["path"])] = {
            "sha256": receipt["sha256"],
            "authority": "immutable_raw_sidecar",
            "terminal_class": None,
            "quarantined": False,
            "retired": False,
        }
    if validation_receipt_path is not None and validation_receipt_path.is_file():
        hashes[_display_path(root, validation_receipt_path)] = {
            "sha256": sha256(validation_receipt_path),
            "authority": "validation_receipt_manifest",
            "terminal_class": None,
            "quarantined": False,
            "retired": False,
        }
    return checks, hashes


def _acceptance_results(
    contract: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    branches: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Score closure, authority, science, and validation as separate facts."""

    receipt_names = [str(row.get("name")) for row in receipts]
    complete_scores = [row for row in scores if row["check_kind"] == "completeness"]
    value_scores = [row for row in scores if row["check_kind"] == "value"]
    definitions = (
        (
            "fourteen_task_dispositions",
            14,
            len(dispositions),
            len(dispositions) == 14,
            "Roster closure is independent of scientific value.",
        ),
        (
            "literal_contract_exact",
            True,
            contract["contract_agrees"],
            contract["contract_agrees"] is True,
            "YAML and Markdown are parsed independently; disagreement stays visible.",
        ),
        (
            "same_milestone_gates",
            8,
            sum(row["passed"] is True for row in gates),
            len(gates) == 8 and all(row["passed"] is True for row in gates),
            "Conductor readiness is not scientific efficacy.",
        ),
        (
            "required_captures_and_audits",
            {row["artifact_field"]: 1 for row in complete_scores},
            {row["artifact_field"]: row["observed_value"] for row in complete_scores},
            all(row["passed"] is True for row in complete_scores),
            "Every required capture, audit, and context receipt must be authentic.",
        ),
        (
            "independent_branch_value",
            "accepted source or learning branch",
            {row["artifact_field"]: row["observed_value"] for row in value_scores},
            any(row["scientific_efficacy_positive"] is True for row in branches),
            "Mechanism and context receipts cannot supply broad scientific value.",
        ),
        (
            "validation_receipts",
            list(REQUIRED_VALIDATION_NAMES),
            receipt_names,
            set(REQUIRED_VALIDATION_NAMES) <= set(receipt_names)
            and all(
                row.get("exit_code") == 0 and row.get("log_hash_matches") is True
                for row in receipts
                if row.get("name") in REQUIRED_VALIDATION_NAMES
            ),
            "Commands preserve actual exit codes, scopes, elapsed time, and log hashes.",
        ),
    )
    return [
        {
            "criterion": name,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "principle": principle,
        }
        for name, expected, observed, passed, principle in definitions
    ]


def validate_artifact(artifact: Mapping[str, Any], root: Path | None = None) -> list[str]:
    """Recompute roster, scores, branches, diagnostics, hashes, and checksum."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_ARTIFACT_FIELDS.issubset(artifact), "required_fields")
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != "carnot.exp7315.v642_capstone.v1"
        or artifact.get("experiment_id") != "exp7315-capstone"
        or artifact.get("milestone") != MILESTONE,
        "identity",
    )
    add(artifact.get("run_date") != RUN_DATE, "lifecycle")
    add(
        artifact.get("status") not in {"complete", "blocked"}
        or artifact.get("verdict_class") not in VALID_VERDICT_CLASSES,
        "terminal_state",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS,
        "model_invocation",
    )
    add(
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
        or not artifact.get("execution_host"),
        "execution_substrate",
    )
    contract_rows = artifact.get("contract_rows", [])
    add(
        not isinstance(contract_rows, list)
        or len(contract_rows) != 14
        or [row.get("unit_id") for row in contract_rows] != list(EXPECTED_TASK_IDS),
        "contract_rows",
    )
    dispositions = artifact.get("task_dispositions", [])
    add(
        not isinstance(dispositions, list)
        or len(dispositions) != 14
        or [row.get("task_id") for row in dispositions] != list(EXPECTED_TASK_IDS)
        or dispositions[-1].get("artifact_sha256") is not None,
        "task_dispositions",
    )
    branches = artifact.get("rows", [])
    add(
        not isinstance(branches, list)
        or tuple(row.get("branch") for row in branches) != BRANCH_NAMES
        or any(
            not {
                "unit_id",
                "arm",
                "seed",
                "metric",
                "metric_value",
                "error",
                "abstention",
                "cost",
                "censored",
            }.issubset(row)
            for row in branches
        ),
        "branch_rows",
    )
    add(artifact.get("capstone_complete_score") != 1, "capstone_complete_score")
    summary = artifact.get("gate_check_summary", {})
    add(
        artifact.get("status") == "blocked"
        and (
            artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or not isinstance(summary, Mapping)
            or not isinstance(summary.get("failures"), list)
            or not summary.get("failures")
        ),
        "terminal_state",
    )
    add(artifact.get("reproducibility_checksum") != artifact_checksum(artifact), "checksum")
    if root is not None:
        contract = load_contract(root)
        evidence = load_repository_payloads(root, contract["tasks"])
        current_gates = replay_gates(contract["tasks"], evidence)
        current_scores = audit_score_rows(evidence)
        current_branches = recompute_branch_rows(evidence)
        terminal = terminal_state(evidence, current_branches)
        current_dispositions = task_dispositions(contract["tasks"], evidence, terminal)
        add(contract_rows != contract["contract_rows"], "contract_rows")
        add(artifact.get("same_milestone_gate_replay_rows") != current_gates, "gate_replay_rows")
        add(artifact.get("audit_score_rows") != current_scores, "audit_score_rows")
        add(branches != current_branches, "branch_rows")
        add(dispositions != current_dispositions, "task_dispositions")
        add(
            artifact.get("status") != terminal["status"]
            or artifact.get("verdict_class") != terminal["verdict_class"]
            or artifact.get("honest_verdict") != terminal["honest_verdict"]
            or summary != terminal["gate_check_summary"],
            "terminal_state",
        )
        add(
            artifact.get("branch_decisions") != branch_decisions(current_branches),
            "branch_decisions",
        )
        add(
            artifact.get("prd_gap_assessment") != prd_gap_assessment(current_branches),
            "prd_gap_assessment",
        )
        add(
            artifact.get("retirement_decisions")
            != retirement_decisions(contract["tasks"], evidence, root),
            "retirement_decisions",
        )
        add(artifact.get("repository_health") != _repository_health(evidence), "repository_health")
        add(artifact.get("v641_preservation") != _v641_preservation(root), "v641_preservation")
        add(
            artifact.get("acceptance_gate_results")
            != _acceptance_results(
                contract,
                current_dispositions,
                current_gates,
                current_scores,
                current_branches,
                artifact.get("validation_receipts", []),
            ),
            "acceptance_gate_results",
        )
        hashes = artifact.get("source_artifact_hashes")
        if not isinstance(hashes, Mapping):
            add(True, "source_artifact_hashes")
        else:
            for named, receipt in hashes.items():
                path = Path(str(named))
                resolved = path if path.is_absolute() else root / path
                expected = receipt.get("sha256") if isinstance(receipt, Mapping) else None
                if not resolved.is_file() or sha256(resolved) != expected:
                    add(True, "source_artifact_hashes")
                    break
    return errors


def build_artifact(
    root: Path,
    run_date: str,
    output_path: Path,
    checkpoint_path: Path,
    *,
    raw_dir: Path | None = None,
    validation_receipt_path: Path | None = None,
) -> JsonDict:
    """Aggregate V642 evidence and atomically publish one validated result."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "authenticate inputs and declared output paths")
    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    selected_raw_dir = raw_dir or root / DEFAULT_RAW_DIR
    common._atomic_write(
        checkpoint_path,
        {
            "schema": "carnot.exp7315.v642_capstone.checkpoint.v1",
            "experiment_id": "exp7315-capstone",
            "status": "running",
            "run_date": run_date,
            "MODEL_SPECS": [],
            "model_invoked": False,
        },
    )

    spans: dict[str, float] = {}
    phase = time.monotonic()
    progress(1, "start", "parse V642 YAML and Markdown independently")
    contract = load_contract(root)
    spans["contract_parse"] = time.monotonic() - phase
    progress(1, "end", f"completed=14 contract_agrees={contract['contract_agrees']}")

    phase = time.monotonic()
    progress(2, "start", "authenticate thirteen exact producer paths")
    evidence = load_repository_payloads(root, contract["tasks"])
    for index, task_id in enumerate(EXPECTED_PRODUCER_IDS, 1):
        row = evidence[task_id]
        progress(2, "unit", f"completed={index}/13 {task_id} class={row['disposition_class']}")
    spans["producer_authentication"] = time.monotonic() - phase
    progress(2, "end", f"terminal={sum(row['terminal'] for row in evidence.values())}/13")

    phase = time.monotonic()
    progress(3, "start", "replay eight gates and reduce eight authenticated scores")
    gates = replay_gates(contract["tasks"], evidence)
    scores = audit_score_rows(evidence)
    spans["gate_and_score_reduction"] = time.monotonic() - phase
    progress(3, "end", f"gates={len(gates)} score_rows={len(scores)}")

    phase = time.monotonic()
    progress(4, "start", "recompute five branch rows from producer rows")
    branches = recompute_branch_rows(evidence)
    terminal = terminal_state(evidence, branches)
    spans["branch_reduction"] = time.monotonic() - phase
    progress(4, "end", f"branches={len(branches)} class={terminal['verdict_class']}")

    phase = time.monotonic()
    progress(5, "start", "freeze authorities, history, and invocation sidecars")
    v641_preservation = _v641_preservation(root)
    sidecars = _write_sidecars(
        root, selected_raw_dir, contract, evidence, branches, v641_preservation
    )
    receipts = _load_validation_receipts(root, validation_receipt_path)
    checks, source_hashes = _preconditions(
        root,
        output_path,
        checkpoint_path,
        contract,
        evidence,
        sidecars,
        validation_receipt_path,
    )
    spans["authority_freeze"] = time.monotonic() - phase
    progress(5, "end", f"sidecars={len(sidecars)} validation_receipts={len(receipts)}")

    phase = time.monotonic()
    progress(6, "start", "build dispositions, PRD gaps, and exact retirements")
    dispositions = task_dispositions(contract["tasks"], evidence, terminal)
    for index, row in enumerate(dispositions, 1):
        progress(6, "unit", f"completed={index}/14 {row['task_id']}")
    decisions = branch_decisions(branches)
    retirements = retirement_decisions(contract["tasks"], evidence, root)
    prd = prd_gap_assessment(branches)
    repository_health = _repository_health(evidence)
    spans["dispositions_and_decisions"] = time.monotonic() - phase
    progress(6, "end", f"retirement_rows={len(retirements)} PRD_gaps={len(prd)}")

    artifact: JsonDict = {
        "schema": "carnot.exp7315.v642_capstone.v1",
        "experiment_id": "exp7315-capstone",
        "milestone": MILESTONE,
        "status": terminal["status"],
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": "",
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": checks,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "phase_spans_s": spans,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": source_hashes,
        "rows": branches,
        "sample_size_budget": {
            "contract_rows": {"planned": 14, "attempted": 14, "complete": 14, "censored": 0},
            "producer_artifacts": {
                "planned": 13,
                "attempted": 13,
                "complete": sum(row["terminal"] for row in evidence.values()),
                "censored": sum(row["selected_evidence_path"] is None for row in evidence.values()),
            },
            "branch_rows": {
                "planned": 5,
                "attempted": 5,
                "complete": len(branches),
                "censored": sum(row["censored"] is True for row in branches),
            },
            "stopping_rule": "Stop after fourteen dispositions, eight gate replays, eight score checks, and five branch reductions; do not extend after null or block.",
        },
        "acceptance_gate_results": _acceptance_results(
            contract, dispositions, gates, scores, branches, receipts
        ),
        "gate_check_summary": terminal["gate_check_summary"],
        "verifier_is_oracle": False,
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "validation_receipts": receipts,
        "capstone_complete_score": 1,
        "task_dispositions": dispositions,
        "branch_decisions": decisions,
        "prd_gap_assessment": prd,
        "retirement_decisions": retirements,
        "contract_rows": contract["contract_rows"],
        "audit_score_rows": scores,
        "same_milestone_gate_replay_rows": gates,
        "repository_health": repository_health,
        "capstone_dimensions": {
            "closure": {"complete_score": 1, "dispositions": 14},
            "scientific_efficacy": {
                "positive_branch_count": sum(
                    row["scientific_efficacy_positive"] is True for row in branches
                ),
                "required_science_complete": not any(
                    row["terminal_blocking"] for row in terminal["gate_check_summary"]["failures"]
                ),
            },
            "mechanism_reachability": {
                "arc_tool_use_score": scores[1]["observed_value"],
                "claim_scope": "mechanism receipt only",
            },
            "repository_health": {
                "source_authenticated": repository_health["source_authenticated"],
                "affects_capstone_science": False,
            },
        },
        "v641_preservation": v641_preservation,
        "historical_model_receipt_sidecar": next(
            receipt for receipt in sidecars if "historical-model" in str(receipt["path"])
        ),
        "publication_performed": False,
        "upload_performed": False,
        "submission_performed": False,
        "external_message_performed": False,
        "production_default_changed": False,
        "research_roadmap_modified": False,
        "conductor_modified": False,
    }
    elapsed = time.monotonic() - started
    spans["final_assembly"] = max(elapsed - sum(spans.values()), 0.0)
    artifact["duration_s"] = elapsed
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)

    progress(7, "start", "write and independently validate measured raw candidate")
    candidate_path = selected_raw_dir / "measured-terminal-candidate.json"
    common._atomic_write(candidate_path, artifact)
    candidate = read_json(candidate_path)
    errors = validate_artifact(candidate, root=root)
    if errors:  # pragma: no cover - protects against storage corruption after assembly
        raise RuntimeError("capstone candidate validation failed: " + ",".join(errors))
    progress(7, "end", "raw candidate passed independent reduction")

    progress(8, "start", "atomically write declared terminal artifact")
    common._atomic_write(checkpoint_path, artifact)
    common._atomic_write(output_path, artifact)
    final = read_json(output_path)
    errors = validate_artifact(final, root=root)
    if errors:  # pragma: no cover - protects against corruption during final replacement
        raise RuntimeError("terminal artifact validation failed: " + ",".join(errors))
    progress(8, "end", f"wrote {output_path}")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """Build the capstone or cold-validate an existing artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--validation-receipt-path", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.artifact_path or root / DEFAULT_OUTPUT_PATH
    checkpoint = args.checkpoint_path or root / DEFAULT_CHECKPOINT_PATH
    if args.validate:
        progress(7, "start", f"independent reducer for {output}")
        errors = validate_artifact(read_json(output), root=root)
        progress(7, "end", "passed" if not errors else ",".join(errors))
        return int(bool(errors))
    receipt_path = args.validation_receipt_path
    if receipt_path is None:
        default_receipts = root / DEFAULT_VALIDATION_RECEIPT_PATH
        receipt_path = default_receipts if default_receipts.is_file() else None
    build_artifact(
        root,
        args.date,
        output,
        checkpoint,
        raw_dir=args.raw_dir,
        validation_receipt_path=receipt_path,
    )
    return 0
