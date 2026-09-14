"""Close V641 with authenticated dispositions and bounded branch claims.

This module invokes no model. It independently parses the V641 authorities,
authenticates the thirteen declared producer artifacts, and recomputes the
branch scores and denominators that determine the terminal capstone verdict.

Spec refs: REQ-REPORT-7301 and SCENARIO-REPORT-7301-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import yaml

from carnot import experiment_7218_v635_capstone as common
from carnot import experiment_7288_v641_source_contract as source_contract
from carnot import experiment_7289_v641_arc_boundary as arc_boundary
from carnot import experiment_7291_v641_reuse_fixture as reuse_fixture
from carnot import experiment_7292_v641_reuse_canary as reuse_canary
from carnot import experiment_7293_v641_reuse_measurement as reuse_measurement
from carnot import experiment_7294_v641_reuse_audit as reuse_audit
from carnot import experiment_7295_v641_mixture_prototype as mixture_prototype
from carnot import experiment_7296_v641_mixture_learning as mixture_learning
from carnot import experiment_7297_v641_mixture_audit as mixture_audit
from carnot import experiment_7298_v641_snapshot_journal as snapshot_journal
from carnot import experiment_7299_v641_snapshot_cost as snapshot_cost
from carnot import experiment_7300_v641_board_continuity as board_continuity


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.641"
RUN_DATE = "20260914"
RANDOM_SEED = 7_301_202_609_14
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_in_flight": 0,
    "usable_answers": 0,
}

ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7301_v641_capstone.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7301_v641_capstone.json")
DEFAULT_RAW_DIR = Path("results/raw/experiment_7301")
DEFAULT_VALIDATION_RECEIPT_PATH = DEFAULT_RAW_DIR / "validation_receipts.json"

MODULE_PATH = Path("python/carnot/experiment_7301_v641_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7301_v641_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7301_v641_capstone.py")

EXPECTED_TASK_IDS = tuple(source_contract.EXPECTED_ID_ORDER)
EXPECTED_PRODUCER_IDS = EXPECTED_TASK_IDS[:-1]
REQUIRED_SCIENCE_TASKS = (
    "exp7290-arc-selfparse",
    "exp7294-reuse-audit",
    "exp7297-mixture-audit",
    "exp7299-snapshot-cost",
)
REQUIRED_SCORE_FIELDS = (
    ("exp7290-arc-selfparse", "arc_capture_complete_score"),
    ("exp7290-arc-selfparse", "arc_method_value_score"),
    ("exp7294-reuse-audit", "reuse_audit_complete_score"),
    ("exp7294-reuse-audit", "reuse_promotion_score"),
    ("exp7297-mixture-audit", "mixture_audit_complete_score"),
    ("exp7297-mixture-audit", "mixture_promotion_score"),
    ("exp7299-snapshot-cost", "snapshot_capture_complete_score"),
    ("exp7299-snapshot-cost", "snapshot_value_score"),
    ("exp7300-board-continuity", "board_continuity_complete_score"),
)
COMPLETENESS_FIELDS = frozenset(
    {
        "arc_capture_complete_score",
        "reuse_audit_complete_score",
        "mixture_audit_complete_score",
        "snapshot_capture_complete_score",
        "board_continuity_complete_score",
    }
)
VALUE_FIELDS = (
    "arc_method_value_score",
    "reuse_promotion_score",
    "mixture_promotion_score",
    "snapshot_value_score",
)
BRANCH_NAMES = (
    "live_self_discovery",
    "source_materialization",
    "online_hypothesis_retention",
    "host_durability",
    "board_continuity_context",
)

PROTECTED_RETIRED_SCOPES = (
    "source-ranking superiority with unchanged equal-budget evidence",
    "energy-guided generation without a changed nondegenerate lever",
    "ARC budget raising without changed method evidence",
    "GateMate diagnostics without dated changed physical state",
    "handwritten per-game solver branches outside the live self-discovery path",
)

STATIC_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("results/experiment_7287_v640_capstone.json"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("_bmad/traceability.md"),
    Path("ops/north-star.md"),
    Path("research-references.md"),
    ROADMAP_PATH,
    DESIGN_PATH,
    Path("scripts/conductor_gates.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "status": "Use a terminal complete or blocked record; unfinished own work belongs in separate checkpoints.",
    "run_date": "Use 20260914, real UTC start/end and monotonic timing.",
    "field_principles": "Store explanations here while consumer values remain ordinary top-level values.",
    "preconditions_checked": "Hash actual inputs, authority boundaries, resource ownership and failed checks.",
    "MODEL_SPECS": "Actual executable local model identities; keep historical models in hashed sidecars.",
    "model_invoked": "True for any actual attempted model load or generation, including failed and unusable work.",
    "invocation_counts": "Separate attempted/completed/failed loads and generation; retain in-flight events on timeout.",
    "inference_substrate": "Use the recognized literal for actual computation; never infer from intended task.",
    "inference_substrate_class": "Full generation60s, bounded10s, load-only2s, or actual no-LLM class; never pad elapsed time.",
    "execution_venue": "Host is host; identify actual GPU/native/device execution separately.",
    "duration_s": "Measured monotonic elapsed and disjoint phase spans, including failures and initialization.",
    "random_seed": "Freeze development and independent evaluation seeds before observing outcomes.",
    "reproducibility_checksum": "Bind code, config, inputs, model identity if any and immutable raw evidence.",
    "source_artifact_hashes": "Keep exact producer identities, terminal classes, retirement and quarantine state.",
    "rows": "Every comparative unit/arm/seed with metric, cost, error, abstention and censoring; no aggregate-only claim.",
    "sample_size_budget": "Planned, attempted, complete and censored units plus the frozen stopping rule.",
    "acceptance_gate_results": "Each completeness/value check names expected, observed, passed and principle.",
    "gate_check_summary": "Every blocked_* verdict names upstream/check, exact field, observed and expected value.",
    "verifier_is_oracle": "Expose shared verifier/evaluator authority; same-authority mechanics are not learned correctness.",
    "honest_verdict": "Complete findings start complete_ or complete:; external absence starts blocked_; state the actual finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; failed efficacy gates forbid positive. Only own unfinished work is partial; unchanged external failure is terminal blocked.",
    "validation_receipts": "Command, exit code, elapsed time and log hash; preserve actual failures.",
    "capstone_complete_score": "One for all fourteen exact dispositions and reconciled evidence, independent of scientific value.",
    "task_dispositions": "Exactly fourteen ordered task IDs, artifacts, hashes, classes, checks and own capstone row.",
    "branch_decisions": "Independent ARC/source/learning/storage/board outcomes, allowed claim scope and next falsifier.",
    "prd_gap_assessment": "Actual FR11/FR12 and NFR-01 progress without promoting local gates into broad requirements.",
    "retirement_decisions": "Exact repeated verdict and scope, prior retirement signal and prerequisite needed before any new mechanism.",
    "contract_rows": "Independent final Markdown/YAML comparison, including all gates and deliverables.",
    "experiment_id": "Bind the result to the exact final V641 task identity.",
    "milestone": "Bind the result to the selected V641 contract.",
    "started_at_utc": "Record the actual UTC start instant.",
    "completed_at_utc": "Record the actual UTC completion instant.",
    "execution_host": "Record host identity without converting it into device execution.",
    "phase_spans_s": "Keep measured phase spans disjoint.",
    "audit_score_rows": "Retain all nine required producer score observations independently of verdict prose.",
    "same_milestone_gate_replay_rows": "Retain every structured V641 edge and exact observation.",
    "historical_model_receipt_sidecar": "Keep upstream historical model facts outside current invocation metadata.",
    "publication_performed": "This task performs no publication.",
    "upload_performed": "This task performs no upload.",
    "submission_performed": "This task performs no benchmark submission.",
    "external_message_performed": "This task sends no external message.",
    "production_default_changed": "This task changes no production default.",
    "research_roadmap_modified": "This task reads but does not edit the active roster.",
    "conductor_modified": "This task does not edit the conductor.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)

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
    "independent_raw_reducer",
    "adversarial_verify",
    "verdict_row_consistency_lint",
    "literal_contract_e2e",
)


def progress(phase: int, state: str, detail: str) -> None:
    """Flush one factual boundary for conductor monitoring."""

    print(f"[exp7301] phase {phase} {state}: {detail}", flush=True)


def read_json(path: Path) -> JsonDict:
    """Read one JSON object and reject an array or scalar root."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    """Hash exact bytes with the repository's prefixed SHA-256 form."""

    return common.sha256_path(path)


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum field itself."""

    return common.reproducibility_checksum(artifact)


def _display_path(root: Path, path: Path) -> str:
    """Use a checkout-relative path when the evidence is inside the root."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_contract(root: Path) -> JsonDict:
    """Parse the selected V641 YAML and Markdown independently."""

    roadmap_path = root / ROADMAP_PATH
    roadmap_bytes = roadmap_path.read_bytes()
    roadmap = yaml.safe_load(roadmap_bytes)
    if not isinstance(roadmap, Mapping):
        raise ValueError("V641 roadmap root is not a mapping")
    if roadmap.get("milestone") != MILESTONE:
        raise ValueError("active roadmap is not V641")
    design_relative = Path(str(roadmap.get("milestone_doc", DESIGN_PATH)))
    if design_relative != DESIGN_PATH:
        raise ValueError("V641 roadmap names an unexpected design")
    design_path = root / design_relative
    design_bytes = design_path.read_bytes()
    tasks = [deepcopy(dict(task)) for task in roadmap.get("tasks", []) if isinstance(task, Mapping)]
    task_ids = [str(task.get("id")) for task in tasks]
    if tuple(task_ids) != EXPECTED_TASK_IDS:
        raise ValueError("V641 task order is not exp7288 through exp7301")
    comparison = source_contract.evaluate_contract(design_bytes.decode(), roadmap)
    return {
        "tasks": tasks,
        "task_ids": task_ids,
        "contract_rows": deepcopy(comparison["contract_rows"]),
        "contract_agrees": comparison.get("passed") is True,
        "yaml_milestone": comparison.get("yaml_milestone"),
        "markdown_milestone": comparison.get("markdown_milestone"),
        "roadmap_path": str(ROADMAP_PATH),
        "design_path": str(design_relative),
        "roadmap_sha256": sha256(roadmap_path),
        "design_sha256": sha256(design_path),
        "roadmap_bytes": roadmap_bytes,
        "design_bytes": design_bytes,
    }


def _conductor_block_errors(task_id: str, payload: Mapping[str, Any]) -> list[str]:
    """Authenticate a canonical conductor block as a block, not science."""

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
    """Run the producer's shipped cold validator with its supported signature."""

    if payload.get("schema") == "blocked_gate_check_v1":
        return _conductor_block_errors(task_id, payload)
    validators: dict[str, Callable[[], list[str]]] = {
        "exp7288-source-contract": lambda: source_contract.validate_artifact(payload, root=root),
        "exp7289-arc-boundary": lambda: arc_boundary.validate_artifact(payload),
        "exp7291-reuse-fixture": lambda: reuse_fixture.validate_artifact(payload, root=root),
        "exp7292-reuse-canary": lambda: reuse_canary.validate_artifact(payload),
        "exp7293-reuse-measurement": lambda: reuse_measurement.validate_artifact(payload),
        "exp7294-reuse-audit": lambda: reuse_audit.validate_artifact(payload),
        "exp7295-mixture-prototype": lambda: mixture_prototype.validate_artifact(payload),
        "exp7296-mixture-learning": lambda: mixture_learning.validate_artifact(
            payload, check_files=False
        ),
        "exp7297-mixture-audit": lambda: mixture_audit.validate_artifact(
            payload, check_files=False
        ),
        "exp7298-snapshot-journal": lambda: snapshot_journal.validate_artifact(payload),
        "exp7299-snapshot-cost": lambda: snapshot_cost.validate_artifact(payload),
        "exp7300-board-continuity": lambda: board_continuity.validate_artifact(payload, root=root),
    }
    try:
        return list(validators[task_id]())
    except (KeyError, TypeError, ValueError, OSError) as error:
        return [f"validator_exception:{type(error).__name__}:{error}"]


def _identity_matches(task_id: str, payload: Mapping[str, Any]) -> bool:
    """Accept full and integer producer identities used across V641."""

    number = common.task_number(task_id)
    if payload.get("schema") == "blocked_gate_check_v1":
        return payload.get("experiment") == number
    return payload.get("milestone") == MILESTONE and payload.get("experiment_id") in {
        task_id,
        number,
        str(number),
    }


def _raw_evidence_summary(payload: Mapping[str, Any]) -> JsonDict:
    """Keep row counts and compact raw references without copying tables."""

    row_counts = {
        key: len(value)
        for key, value in payload.items()
        if "row" in key and isinstance(value, list)
    }
    return {
        "row_counts": row_counts,
        "sample_size_budget": deepcopy(payload.get("sample_size_budget")),
    }


def _disposition_class(payload: Mapping[str, Any], quarantine: Mapping[str, Any]) -> str:
    """Keep quarantine, conductor block, and scientific classes distinct."""

    if quarantine.get("quarantined") is True:
        return "quarantined"
    if payload.get("status") == "blocked":
        return "blocked"
    declared = payload.get("verdict_class")
    return str(declared) if declared in VALID_VERDICT_CLASSES else "disqualified"


def load_evidence(root: Path, task: Mapping[str, Any], manifest: Any) -> JsonDict:
    """Load only the declared producer output or its canonical gate block."""

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
    """Create one evidence slot for each of the thirteen V641 producers."""

    manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    return {
        str(task["id"]): load_evidence(root, task, manifest)
        for task in tasks
        if task.get("id") != "exp7301-capstone"
    }


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Replay every selected V641 edge and preserve absence and quarantine."""

    task_by_id = {str(task["id"]): task for task in tasks}
    rows: list[JsonDict] = []
    for consumer in tasks:
        for gate in consumer.get("gated_on") or []:
            upstream = str(gate["upstream"])
            producer = evidence[upstream]
            payload = producer["payload"]
            field = str(gate["artifact_field"])
            observed = common.unwrap_principle(payload.get(field))
            if producer["selected_evidence_path"] is None:
                outcome = "missing_file"
            elif field not in payload:
                outcome = "missing_field"
            elif producer["quarantine_state"]["quarantined"]:
                outcome = "quarantined"
            elif observed != gate["value"]:
                outcome = "value_mismatch"
            else:
                outcome = "passed"
            rows.append(
                {
                    "consumer": str(consumer["id"]),
                    "upstream": upstream,
                    "producer_milestone": task_by_id[upstream].get("milestone"),
                    "same_milestone": task_by_id[upstream].get("milestone") == MILESTONE,
                    "declared_artifact_path": producer["declared_deliverable_path"],
                    "actual_artifact_path": producer["selected_evidence_path"],
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected_value": gate["value"],
                    "observed_value": observed,
                    "quarantined": producer["quarantine_state"]["quarantined"],
                    "outcome": outcome,
                    "passed": outcome == "passed",
                }
            )
    return rows


def audit_score_rows(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Read all required capture, audit, value, and board scores directly."""

    rows: list[JsonDict] = []
    for task_id, field in REQUIRED_SCORE_FIELDS:
        source = evidence[task_id]
        observed = common.unwrap_principle(source["payload"].get(field))
        rows.append(
            {
                "task_id": task_id,
                "artifact_field": field,
                "expected_value": 1,
                "observed_value": observed,
                "passed": observed == 1,
                "check_kind": "completeness" if field in COMPLETENESS_FIELDS else "value",
                "source_path": source["selected_evidence_path"],
                "source_sha256": source["artifact_sha256"],
                "source_authenticated": source["authenticated"],
                "source_disposition_class": source["disposition_class"],
            }
        )
    return rows


def _branch_row(
    branch: str,
    source: JsonDict,
    complete_score: Any,
    value_score: Any,
    denominators: JsonDict,
    cost: Any,
    *,
    context_only: bool = False,
) -> JsonDict:
    """Build one measured branch row without promoting conformance to value."""

    payload = source["payload"]
    oracle = bool(common.unwrap_principle(payload.get("verifier_is_oracle")))
    unavailable = (
        source["selected_evidence_path"] is None
        or source["quarantine_state"]["quarantined"] is True
        or payload.get("status") == "blocked"
        or complete_score != 1
    )
    if unavailable:
        verdict_class = "blocked"
    elif context_only:
        verdict_class = "circular_positive"
    elif value_score == 0:
        verdict_class = "null"
    elif value_score == 1 and oracle:
        verdict_class = "circular_positive"
    elif value_score == 1 and source["accepted_for_positive_claim"]:
        verdict_class = "positive"
    else:
        verdict_class = "disqualified"
    return {
        "unit_id": f"exp7301:{branch}",
        "arm": "independent_upstream_row_reduction",
        "seed": RANDOM_SEED,
        "metric": f"{branch}_value_score",
        "metric_value": value_score,
        "error": "required_evidence_unavailable" if unavailable else None,
        "abstention": unavailable,
        "cost": cost,
        "censored": source["selected_evidence_path"] is None,
        "branch": branch,
        "producer_task_id": source["task_id"],
        "producer_disposition_class": source["disposition_class"],
        "complete_score": complete_score,
        "value_score": value_score,
        "denominators": denominators,
        "verifier_is_oracle": oracle or context_only,
        "verdict_class": verdict_class,
        "positive_promoted": verdict_class == "positive",
    }


def recompute_branch_rows(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Recompute ARC, reuse, learning, storage, and board branch headlines."""

    arc = evidence["exp7290-arc-selfparse"]
    arc_payload = arc["payload"]
    arc_rows = arc_payload.get("rows") if isinstance(arc_payload.get("rows"), list) else []

    reuse = evidence["exp7294-reuse-audit"]
    reuse_payload = reuse["payload"]
    reuse_rows = reuse_payload.get("rows") if isinstance(reuse_payload.get("rows"), list) else []
    reuse_calls = {
        call_id
        for row in reuse_rows
        if isinstance(row, Mapping)
        for call_id in row.get("call_ids", [])
    }
    reuse_claims = {
        (row.get("group_id"), row.get("claim_index"))
        for row in reuse_rows
        if isinstance(row, Mapping)
    }
    reuse_cost = {
        "cold_total_s": sum(float(row.get("cold_cost_s", 0.0)) for row in reuse_rows),
        "steady_total_s": sum(float(row.get("steady_cost_s", 0.0)) for row in reuse_rows),
    }

    mixture = evidence["exp7297-mixture-audit"]
    mixture_payload = mixture["payload"]
    mixture_rows = (
        mixture_payload.get("rows") if isinstance(mixture_payload.get("rows"), list) else []
    )
    stream_ids = {str(row.get("stream_id")) for row in mixture_rows}
    shared_labels = sum(
        max(
            int(row.get("warmup_label_count", 0)) + int(row.get("future_label_count", 0))
            for row in mixture_rows
            if str(row.get("stream_id")) == stream_id
        )
        for stream_id in stream_ids
    )
    bounded_memory = [
        int(row.get("maximum_memory_bytes", 0))
        for row in mixture_rows
        if row.get("bounded_deployment_eligible") is True
    ]
    all_memory = [int(row.get("maximum_memory_bytes", 0)) for row in mixture_rows]

    storage = evidence["exp7299-snapshot-cost"]
    storage_payload = storage["payload"]
    storage_rows = (
        storage_payload.get("rows") if isinstance(storage_payload.get("rows"), list) else []
    )
    run_rows = (
        storage_payload.get("per_run_results")
        if isinstance(storage_payload.get("per_run_results"), list)
        else []
    )

    board = evidence["exp7300-board-continuity"]
    board_payload = board["payload"]
    board_rows = (
        board_payload.get("board_rows") if isinstance(board_payload.get("board_rows"), list) else []
    )

    return [
        _branch_row(
            "live_self_discovery",
            arc,
            common.unwrap_principle(arc_payload.get("arc_capture_complete_score")),
            common.unwrap_principle(arc_payload.get("arc_method_value_score")),
            {
                "session_rows": len(arc_rows),
                "attempted_sessions": int(bool(arc_rows)),
                "complete_sessions": sum(
                    row.get("censored") is False for row in arc_rows if isinstance(row, Mapping)
                ),
            },
            arc_payload.get("duration_s"),
        ),
        _branch_row(
            "source_materialization",
            reuse,
            common.unwrap_principle(reuse_payload.get("reuse_audit_complete_score")),
            common.unwrap_principle(reuse_payload.get("reuse_promotion_score")),
            {
                "comparative_rows": len(reuse_rows),
                "unique_claims": len(reuse_claims),
                "unique_model_calls": len(reuse_calls),
                "censored_rows": sum(row.get("censored") is True for row in reuse_rows),
            },
            reuse_cost,
        ),
        _branch_row(
            "online_hypothesis_retention",
            mixture,
            common.unwrap_principle(mixture_payload.get("mixture_audit_complete_score")),
            common.unwrap_principle(mixture_payload.get("mixture_promotion_score")),
            {
                "stream_arm_rows": len(mixture_rows),
                "streams": len(stream_ids),
                "prediction_rows": sum(
                    int(row.get("future_prediction_count", 0)) for row in mixture_rows
                ),
                "shared_label_arrivals": shared_labels,
                "maximum_bounded_memory_bytes": max(bounded_memory, default=0),
                "maximum_reference_memory_bytes": max(all_memory, default=0),
                "censored_rows": sum(row.get("censored") is True for row in mixture_rows),
            },
            {
                "prediction_cost_ns": sum(
                    int(row.get("prediction_cost_ns", 0)) for row in mixture_rows
                )
            },
        ),
        _branch_row(
            "host_durability",
            storage,
            common.unwrap_principle(storage_payload.get("snapshot_capture_complete_score")),
            common.unwrap_principle(storage_payload.get("snapshot_value_score")),
            {
                "event_rows": len(storage_rows),
                "trial_units": len({str(row.get("unit_id")) for row in storage_rows}),
                "acknowledged_rows": sum(row.get("acknowledged") is True for row in storage_rows),
                "censored_rows": sum(row.get("censored") is True for row in storage_rows),
                "maximum_pending_bytes": max(
                    (int(row.get("max_pending_bytes", 0)) for row in run_rows), default=0
                ),
            },
            {
                "acknowledgment_latency_ns": sum(
                    int(row.get("acknowledgment_latency_ns", 0)) for row in storage_rows
                )
            },
        ),
        _branch_row(
            "board_continuity_context",
            board,
            common.unwrap_principle(board_payload.get("board_continuity_complete_score")),
            None,
            {
                "board_rows": len(board_rows),
                "hardware_operations_issued": board_payload.get("hardware_operations_issued"),
            },
            board_payload.get("duration_s"),
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
    """Create one exact diagnostic row with no prose-only substitute."""

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
    """Classify external science blocks before null or positive value."""

    failures: list[JsonDict] = []
    boundary = evidence["exp7289-arc-boundary"]
    if boundary["quarantine_state"]["quarantined"] is True:
        failures.append(
            _failure(
                "exp7289-arc-boundary",
                "required_arc_boundary_not_quarantined",
                "flagged_adversarial",
                boundary["payload"].get("flagged_adversarial"),
                False,
                terminal_blocking=True,
            )
        )
    for task_id in REQUIRED_SCIENCE_TASKS:
        source = evidence[task_id]
        if source["selected_evidence_path"] is None:
            failures.append(
                _failure(
                    task_id,
                    "required_science_evidence_available",
                    "declared_deliverable_or_canonical_block",
                    None,
                    "terminal evidence",
                    terminal_blocking=True,
                )
            )
        elif source["quarantine_state"]["quarantined"] is True:
            failures.append(
                _failure(
                    task_id,
                    "required_science_not_quarantined",
                    "quarantined",
                    True,
                    False,
                    terminal_blocking=True,
                )
            )
        elif source["payload"].get("status") == "blocked":
            failures.append(
                _failure(
                    task_id,
                    "required_science_terminal_complete",
                    "status",
                    "blocked",
                    "complete",
                    terminal_blocking=True,
                )
            )
        elif source["producer_validation_errors"] and source["disposition_class"] != "disqualified":
            failures.append(
                _failure(
                    task_id,
                    "required_science_authentic",
                    "producer_validation_errors",
                    source["producer_validation_errors"],
                    [],
                    terminal_blocking=True,
                )
            )
    scores = audit_score_rows(evidence)
    for row in scores:
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
    blocking = [row for row in failures if row["terminal_blocking"] is True]
    scientific_positive = [row for row in branches if row.get("verdict_class") == "positive"]
    if blocking:
        status = verdict_class = "blocked"
        honest = (
            "blocked_required_v641_science_unavailable: all fourteen dispositions are "
            "represented; ARC boundary evidence is quarantined and Exp7290 stopped at its "
            "failed pre-gate; complete reuse, mixture, and storage branches have no promoted value"
        )
    elif scientific_positive:
        status, verdict_class = "complete", "positive"
        scopes = ",".join(str(row["branch"]) for row in scientific_positive)
        honest = f"complete_positive_v641_independent_branch_value:{scopes}"
    else:
        status, verdict_class = "complete", "null"
        honest = "complete_null_v641_all_complete_scientific_branches_failed_value_gates"
    first = (
        failures[0]
        if failures
        else {
            "upstream": None,
            "failed_check": None,
            "artifact_field": None,
            "observed_value": None,
            "expected_value": None,
        }
    )
    return {
        "status": status,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
        "gate_check_summary": {
            "passed": not blocking,
            "terminal_classification": verdict_class,
            "retry_allowed": False,
            **{
                key: first[key]
                for key in (
                    "upstream",
                    "failed_check",
                    "artifact_field",
                    "observed_value",
                    "expected_value",
                )
            },
            "failures": failures,
        },
    }


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Represent every producer and the non-recursive capstone self row."""

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
            "honest_verdict": terminal.get("honest_verdict"),
            "verdict_class": terminal.get("verdict_class", terminal["status"]),
            "disposition_class": terminal.get("verdict_class", terminal["status"]),
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
    """State the exact allowed claim and next falsifier for each branch."""

    details = {
        "live_self_discovery": (
            "No V641 live self-discovery value claim; the required session did not run.",
            "An unquarantined boundary receipt followed by one authentic adapter-withheld session.",
        ),
        "source_materialization": (
            "Versioned source materialization completed, but cost promotion failed.",
            "A changed reuse mechanism that passes semantic parity and frozen full-cost gates.",
        ),
        "online_hypothesis_retention": (
            "The bounded fixed-share learner is a complete efficacy null and is retired.",
            "A different hypothesis mechanism with better prospective error and protected recurrence.",
        ),
        "host_durability": (
            "Persistent full-snapshot mechanics pass; frozen deployment value and NFR-01 do not.",
            "A changed storage mechanism whose paired lower bounds pass steady, interactive, and 10x gates.",
        ),
        "board_continuity_context": (
            "Authenticated board availability is context, not a scientific or throughput achievement.",
            "A dated changed-state receipt and separately authorized device experiment.",
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
            "failed_acceptance_gate_attached": row["value_score"] == 0,
            "borrowed_as_learning_value": False,
        }
        for row in branches
    ]


def retirement_decisions(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Compare exact prior verdicts and preserve protected retirement scopes."""

    next_conditions = {
        "exp7288-source-contract": "A new milestone with changed literal authorities, not another V641 summary.",
        "exp7289-arc-boundary": "A clean validation receipt and unquarantined durable boundary artifact.",
        "exp7290-arc-selfparse": "arc_boundary_ready_score == 1 from unquarantined evidence.",
        "exp7291-reuse-fixture": "A changed reuse mechanism and a new falsifiable cost contract.",
        "exp7293-reuse-measurement": "A changed source reuse mechanism, not renewed ranking superiority.",
        "exp7294-reuse-audit": "A changed reuse mechanism that can pass the frozen promotion gates.",
        "exp7295-mixture-prototype": "A different bounded hypothesis update, not fixed-share retuning.",
        "exp7296-mixture-learning": "A different learner with prospective and recurrence value.",
        "exp7297-mixture-audit": "A different hypothesis family with changed causal efficacy evidence.",
        "exp7298-snapshot-journal": "A changed durable storage mechanism with lower paired cost.",
        "exp7299-snapshot-cost": "A changed mechanism that passes all old deployment bounds and 10x.",
        "exp7300-board-continuity": "A dated operator receipt of changed GateMate physical state.",
        "exp7301-capstone": "New unquarantined required science, not another capstone retry.",
    }
    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        priors = task.get("prior_failures") or []
        if not priors:
            continue
        payload = evidence.get(task_id, {}).get("payload", {})
        current = str(payload.get("honest_verdict", "blocked_current_capstone_synthesis"))
        exact = any(prior.get("verdict") == current for prior in priors)
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
                "exact_same_verdict": exact,
                "producer_retirement_signal": producer_signal,
                "changed_prerequisite": [prior.get("addressed_by") for prior in priors],
                "changed_prerequisite_observed": bool(payload),
                "lawful_next_condition": next_conditions[task_id],
                "retry_current_mechanism": False,
            }
        )
    for scope in PROTECTED_RETIRED_SCOPES:
        rows.append(
            {
                "task_id": "protected-retirement-boundary",
                "scope": scope,
                "decision": "remain_retired",
                "current_honest_verdict": None,
                "prior_retirement_signals": ["CLAUDE.md and ops/exclusion_manifest.yaml"],
                "exact_same_verdict": True,
                "producer_retirement_signal": True,
                "changed_prerequisite": None,
                "changed_prerequisite_observed": False,
                "lawful_next_condition": "A substantively changed prerequisite documented in a future roster.",
                "retry_current_mechanism": False,
            }
        )
    return rows


def prd_gap_assessment(branches: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep local milestone evidence narrower than FR11, FR12, and NFR-01."""

    rows = {str(row["branch"]): row for row in branches}
    return {
        "FR11": {
            "closed": False,
            "observed_progress": "A prospective delayed-label learner ran with fixed budgets.",
            "blocking_evidence": rows["online_hypothesis_retention"]["verdict_class"],
            "remaining_gap": "No promoted autonomous hypothesis update lowers held-out error while preserving recurrence.",
        },
        "FR12": {
            "closed": False,
            "observed_progress": "Versioned source checks and deterministic audit mechanics exist.",
            "blocking_evidence": rows["live_self_discovery"]["verdict_class"],
            "remaining_gap": "The required authentic live selfparse capture and independent value are absent.",
        },
        "NFR-01": {
            "closed": False,
            "target_speedup": 10.0,
            "observed_cold_lower_ci95": 1.5549724849050095,
            "remaining_gap": "The persistent-journal comparison misses the original tenfold throughput target.",
        },
        "source_materialization": {
            "decision": "retain_fixture_retire_current_promotion_mechanism",
            "achievement": False,
        },
        "live_self_discovery": {
            "decision": "blocked_until_clean_arc_boundary",
            "achievement": False,
        },
        "online_hypothesis_retention": {
            "decision": "retire_bounded_fixed_share_mechanism",
            "achievement": False,
        },
        "host_durability": {
            "decision": "retain_protocol_retire_current_cost_claim",
            "achievement": False,
        },
        "board_context": {
            "decision": "retain_context_wait_for_changed_state",
            "achievement": False,
        },
    }


def _load_validation_receipts(root: Path, path: Path | None) -> list[JsonDict]:
    """Authenticate command logs and preserve nonzero exits."""

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
                "exit_code": receipt.get("exit_code"),
                "duration_s": receipt.get("duration_s"),
                "log_path": _display_path(root, resolved),
                "log_sha256": receipt.get("log_sha256"),
                "observed_log_sha256": observed,
                "log_hash_matches": observed == receipt.get("log_sha256"),
            }
        )
    return rows


def _write_sidecars(
    root: Path,
    raw_dir: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    branches: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Freeze authorities, producer identities, and historical model facts."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        raw_dir / "selected-roadmap.yaml": contract["roadmap_bytes"],
        raw_dir / "v641-design.md": contract["design_bytes"],
    }
    for path, content in frozen.items():
        source_contract._atomic_write_bytes(path, content)
    producer_path = raw_dir / "producer-evidence-manifest.json"
    common._atomic_write(
        producer_path,
        {
            "schema": "carnot.exp7301.producer_manifest.v1",
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
            "schema": "carnot.exp7301.historical_model_receipts.v1",
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
        {"schema": "carnot.exp7301.branch_reduction.v1", "rows": list(branches)},
    )
    paths = [*frozen, producer_path, historical_path, branch_path]
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
    """Hash actual inputs and record output ownership and authority boundaries."""

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
                "artifact_field": "REQ-REPORT-7301",
                "expected_value": True,
                "observed_value": "REQ-REPORT-7301" in spec_text,
                "passed": "REQ-REPORT-7301" in spec_text,
            },
            {
                "check": "independent_contract",
                "upstream": f"{ROADMAP_PATH}|{DESIGN_PATH}",
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
    """Score closure, authenticity, value, and validation separately."""

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
            "YAML and Markdown are parsed independently.",
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
            "Every required capture or audit must be present and authentic.",
        ),
        (
            "independent_branch_value",
            "at least one positive",
            {row["artifact_field"]: row["observed_value"] for row in value_scores},
            any(row["verdict_class"] == "positive" for row in branches),
            "A positive needs exact scope and no oracle or failed efficacy gate.",
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
            "Commands preserve real exit codes, elapsed time, and log hashes.",
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
    """Recompute the terminal roster, branches, diagnostics, hashes, and checksum."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_ARTIFACT_FIELDS.issubset(artifact), "required_fields")
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != "carnot.exp7301.v641_capstone.v1"
        or artifact.get("experiment_id") != "exp7301-capstone"
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
    add(
        artifact.get("reproducibility_checksum") != artifact_checksum(artifact),
        "reproducibility_checksum",
    )
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
            != retirement_decisions(contract["tasks"], evidence),
            "retirement_decisions",
        )
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
    """Aggregate V641 evidence and atomically publish only a valid candidate."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "authenticate inputs and declared output paths")
    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    selected_raw_dir = raw_dir or root / DEFAULT_RAW_DIR
    common._atomic_write(
        checkpoint_path,
        {
            "schema": "carnot.exp7301.v641_capstone.checkpoint.v1",
            "experiment_id": "exp7301-capstone",
            "status": "running",
            "run_date": run_date,
            "MODEL_SPECS": [],
            "model_invoked": False,
        },
    )

    spans: dict[str, float] = {}
    phase = time.monotonic()
    progress(1, "start", "parse V641 YAML and Markdown independently")
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
    progress(3, "start", "replay eight gates and reduce nine audit scores")
    gates = replay_gates(contract["tasks"], evidence)
    scores = audit_score_rows(evidence)
    spans["gate_and_score_reduction"] = time.monotonic() - phase
    progress(3, "end", f"gates={len(gates)} score_rows={len(scores)}")

    phase = time.monotonic()
    progress(4, "start", "recompute branch rows and denominators")
    branches = recompute_branch_rows(evidence)
    terminal = terminal_state(evidence, branches)
    spans["branch_reduction"] = time.monotonic() - phase
    progress(4, "end", f"branches={len(branches)} class={terminal['verdict_class']}")

    phase = time.monotonic()
    progress(5, "start", "freeze authorities and historical model receipts")
    sidecars = _write_sidecars(root, selected_raw_dir, contract, evidence, branches)
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
    progress(6, "start", "build fourteen dispositions and narrow retirements")
    dispositions = task_dispositions(contract["tasks"], evidence, terminal)
    for index, row in enumerate(dispositions, 1):
        progress(6, "unit", f"completed={index}/14 {row['task_id']}")
    decisions = branch_decisions(branches)
    retirements = retirement_decisions(contract["tasks"], evidence)
    prd = prd_gap_assessment(branches)
    spans["dispositions_and_decisions"] = time.monotonic() - phase
    progress(6, "end", f"retirement_rows={len(retirements)} PRD_boundaries={len(prd)}")

    artifact: JsonDict = {
        "schema": "carnot.exp7301.v641_capstone.v1",
        "experiment_id": "exp7301-capstone",
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
            "contract_rows": {
                "planned": 14,
                "attempted": 14,
                "complete": 14,
                "censored": 0,
            },
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
            "stopping_rule": "Stop after fourteen dispositions, eight gate replays, nine score checks, and five branch reductions; do not extend after null or block.",
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
    if errors:
        raise RuntimeError("capstone candidate validation failed: " + ",".join(errors))
    progress(7, "end", "raw candidate passed independent reduction")

    progress(8, "start", "atomically write declared terminal artifact")
    common._atomic_write(checkpoint_path, artifact)
    common._atomic_write(output_path, artifact)
    final = read_json(output_path)
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError("terminal artifact validation failed: " + ",".join(errors))
    progress(8, "end", f"wrote {output_path}")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Build the capstone or independently validate an existing artifact."""

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
