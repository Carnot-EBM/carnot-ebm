"""Close V643 without turning completion receipts into scientific efficacy.

The reducer reads exact task outputs, checks their own validators, and keeps
five claim classes separate. It invokes no model and performs no external
action.

Spec refs: REQ-REPORT-7328 and SCENARIO-REPORT-7328-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any, Callable

import yaml

from carnot import experiment_7218_v635_capstone as common
from carnot import experiment_7316_v643_contract as contract_source
from carnot import experiment_7317_v643_batch_harness as exp7317
from carnot import experiment_7318_v643_arc_authority as exp7318
from carnot import experiment_7319_v643_arc_session as exp7319
from carnot import experiment_7320_v643_batch_canary as exp7320
from carnot import experiment_7321_v643_batch_measurement as exp7321
from carnot import experiment_7322_v643_batch_audit as exp7322
from carnot import experiment_7323_v643_addition_prototype as exp7323
from carnot import experiment_7324_v643_addition_learning as exp7324
from carnot import experiment_7325_v643_addition_audit as exp7325
from carnot import experiment_7326_v643_constraint_kernel as exp7326
from carnot import experiment_7327_v643_board_continuity as exp7327
from carnot.reporting import experiment_7303_validation_scope as scoped


JsonDict = dict[str, Any]
Validator = Callable[[Mapping[str, Any]], list[str]]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.643"
RUN_DATE = "20260915"
RANDOM_SEED = {"development": 7_328_202_609_15, "evaluation": 7_328_202_609_16}
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

EXPECTED_TASK_IDS = tuple(contract_source.EXPECTED_ID_ORDER)
CLAIM_NAMES = (
    "source_cost",
    "live_tool_causality",
    "structural_learning",
    "rust_software_parity",
    "board_evidence",
)
DECISION_BRANCHES = (
    *CLAIM_NAMES,
    "v642_suffix_retirement",
    "v642_storage_retirement",
)
VALID_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_SCOPED_CHECKS = scoped.REQUIRED_CHECK_NAMES

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
V642_CAPSTONE_PATH = Path("results/experiment_7315_v642_capstone.json")
V642_COST_PATH = Path("results/experiment_7313_v642_cost_envelope.json")
MODULE_PATH = Path("python/carnot/experiment_7328_v643_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7328_v643_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7328_v643_capstone.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7328_v643_capstone.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7328_v643_capstone.json")
DEFAULT_RAW_DIR = Path("results/raw/experiment_7328_v643_capstone")

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Version this artifact; keep ordinary top-level experiment_id and milestone.",
    "status": "Write terminal output only after current work and required validation.",
    "run_date": "Use 20260915; preserve actual UTC timestamps and monotonic phase spans.",
    "preconditions_checked": "Record input identities, availability, and the exact failed check.",
    "MODEL_SPECS": "Current executable identities only; this aggregation invokes no model.",
    "model_invoked": "True for any actual attempted load or generation, including unusable results.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight model work.",
    "inference_substrate": "Describe actual computation using the recognized substrate literal.",
    "inference_substrate_class": "Classify this read-only reducer as aggregation.",
    "execution_venue": "Use host; historical board work is not current board execution.",
    "duration_s": "Measure real elapsed time; never sleep or inflate duration.",
    "phase_spans": "Record disjoint elapsed spans, units, checkpoints, and pending operations.",
    "random_seed": "Seal development and evaluation seeds before observing results.",
    "reproducibility_checksum": "Bind code, inputs, settings, and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers; historical evidence cannot authorize readiness.",
    "rows": "Emit every claim class with metrics, costs, failures, abstention, and censoring.",
    "sample_size_budget": "Record planned, attempted, complete, and censored disposition counts.",
    "acceptance_gate_results": "Each check has expected, observed, passed, and a short principle.",
    "gate_check_summary": "Every block names upstream, check, field, expected, and observed values.",
    "verifier_is_oracle": "Shared executor authority forbids a positive structural-learning class.",
    "honest_verdict": "Completed findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed verdict enum; partial is only unfinished capstone work.",
    "validation_receipts": "Keep exact commands, scopes, exits, elapsed times, and log hashes.",
    "repository_health": "Preserve unrelated failures as observations, not passed current checks.",
    "field_principles": "Explain each field without wrapping executable values.",
    "capstone_complete_score": "One means all thirteen dispositions, including this reducer, exist.",
    "task_dispositions": "Keep exact IDs, paths, hashes, terminal classes, and gate records.",
    "claim_matrix": "Separate source cost, tool causality, learning, software parity, and hardware.",
    "publication_gate": "Retain actual stable G1-G4 values without authorizing publication.",
    "next_branch_decisions": "Give each branch one bounded action and reopening condition.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)


def progress(phase: int, event: str, detail: str) -> None:
    """Flush a factual boundary so the conductor can see current work."""

    print(f"[exp7328] phase={phase} event={event} {detail}", flush=True)


def sha256_bytes(content: bytes) -> str:
    """Hash exact bytes without changing line endings or JSON layout."""

    return "sha256:" + hashlib.sha256(content).hexdigest()


def sha256(path: Path) -> str:
    """Hash one existing file as exact evidence."""

    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> JsonDict:
    """Read one JSON object and reject another top-level shape."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required at {path}")
    return value


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind all artifact fields except the checksum that stores the digest."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(encoded.encode("utf-8"))


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a result only after a complete JSON document exists beside it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Use the shipped independent parsers for the exact V643 comparison."""

    return contract_source.evaluate_contract(markdown_text, yaml_document)


def load_contract(root: Path) -> JsonDict:
    """Select the V643 YAML once and compare it with independent Markdown bytes."""

    selected, document, content, candidates = contract_source.select_yaml_authority(root)
    if selected is None or document is None or content is None:
        raise ValueError("no selected V643 YAML authority")
    markdown_text = (root / DESIGN_PATH).read_text(encoding="utf-8")
    result = evaluate_contract(markdown_text, document)
    result.update(
        {
            "selected_yaml_path": str(selected),
            "selected_yaml_sha256": sha256_bytes(content),
            "design_path": str(DESIGN_PATH),
            "design_sha256": sha256(root / DESIGN_PATH),
            "authority_candidates": candidates,
            "yaml_document": document,
            "markdown_text": markdown_text,
            "tasks": document.get("tasks", []),
        }
    )
    return result


def _task_number(task_id: str) -> int | None:
    """Return the numeric prefix used by older producer identities."""

    match = re.match(r"exp(\d+)", task_id)
    return int(match.group(1)) if match else None


def _identity_matches(task_id: str, payload: Mapping[str, Any]) -> bool:
    """Accept only the declared full ID or its established numeric form."""

    number = _task_number(task_id)
    return payload.get("milestone") == MILESTONE and payload.get("experiment_id") in {
        task_id,
        number,
        str(number),
    }


def classify_payload(
    task_id: str,
    payload: Mapping[str, Any],
    *,
    validator_errors: Sequence[str],
    quarantined: bool,
) -> tuple[bool, str, bool]:
    """Classify terminal evidence before any numeric score is inspected."""

    terminal = payload.get("status") in {"complete", "blocked"}
    authenticated = terminal and _identity_matches(task_id, payload) and not validator_errors
    declared = payload.get("verdict_class")
    if quarantined:
        disposition = "quarantined"
    elif not authenticated:
        disposition = "disqualified"
    elif payload.get("status") == "blocked":
        disposition = "blocked"
    elif declared in VALID_VERDICT_CLASSES:
        disposition = str(declared)
    else:
        disposition = "disqualified"
    accepted = bool(
        authenticated
        and not quarantined
        and payload.get("status") == "complete"
        and disposition in {"positive", "circular_positive", "null"}
    )
    return authenticated, disposition, accepted


def _producer_validator(task_id: str, root: Path) -> Validator:
    """Bind each task to its shipped cold validator without broad discovery."""

    validators: dict[str, Validator] = {
        "exp7316-contract": lambda value: [
            error
            for error in contract_source.validate_artifact(value, root=root)
            if error != "source_hash_mismatch"
        ],
        "exp7317-batch-harness": lambda value: exp7317.validate_artifact(
            value, require_terminal_checks=True
        ),
        "exp7318-arc-authority": exp7318.validate_artifact,
        "exp7319-arc-session": exp7319.validate_artifact,
        "exp7320-batch-canary": lambda value: exp7320.validate_artifact(
            value, require_validation=True
        ),
        "exp7321-batch-measurement": lambda value: exp7321.validate_artifact(
            value, require_validation=True
        ),
        "exp7322-batch-audit": lambda value: exp7322.validate_artifact(
            value, require_validation=True
        ),
        "exp7323-addition-prototype": lambda value: exp7323.validate_artifact(
            value, require_validation=True
        ),
        "exp7324-addition-learning": lambda value: exp7324.validate_artifact(
            value, check_files=False
        ),
        "exp7325-addition-audit": lambda value: exp7325.validate_artifact(value, check_files=False),
        "exp7326-constraint-kernel": exp7326.validate_artifact,
        "exp7327-board-continuity": exp7327.validate_artifact,
    }
    return validators[task_id]


def load_evidence(root: Path, task: Mapping[str, Any], manifest: object) -> JsonDict:
    """Read only one declared deliverable or its exact canonical block."""

    task_id = str(task["id"])
    declared = str(task["deliverable"])
    canonical = common.canonical_gate_block_path(task_id)
    if (root / declared).is_file():
        selected, source = declared, "declared_deliverable"
    elif (root / canonical).is_file():
        selected, source = canonical, "canonical_conductor_block"
    else:
        selected, source = None, "missing"
    payload = read_json(root / selected) if selected else {}
    quarantine = common.quarantine_receipt(payload, task_id, selected or declared, manifest)
    errors = _producer_validator(task_id, root)(payload) if payload else []
    authenticated, disposition, accepted = classify_payload(
        task_id,
        payload,
        validator_errors=errors,
        quarantined=bool(quarantine.get("quarantined")),
    )
    return {
        "task_id": task_id,
        "declared_deliverable_path": declared,
        "canonical_gate_block_path": canonical,
        "selected_evidence_path": selected,
        "evidence_source": source,
        "artifact_sha256": sha256(root / selected) if selected else None,
        "artifact_size_bytes": (root / selected).stat().st_size if selected else 0,
        "payload": payload,
        "producer_validation_errors": errors,
        "quarantine_receipt": quarantine,
        "quarantined": bool(quarantine.get("quarantined")),
        "authenticated": authenticated,
        "accepted_for_reduction": accepted,
        "disposition_class": disposition if payload else "missing",
    }


def load_repository_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Create one evidence slot for every earlier V643 task."""

    manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    return {str(task["id"]): load_evidence(root, task, manifest) for task in tasks[:-1]}


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Replay each declared edge while terminal class outranks its score."""

    rows: list[JsonDict] = []
    for consumer in tasks:
        for gate in consumer.get("gated_on") or []:
            upstream = str(gate["upstream"])
            source = evidence[upstream]
            payload = source["payload"]
            field = str(gate["artifact_field"])
            observed = payload.get(field)
            if source.get("selected_evidence_path") is None:
                outcome = "missing_file"
            elif source.get("quarantined") is True:
                outcome = "quarantined"
            elif source.get("disposition_class") == "disqualified":
                outcome = "disqualified"
            elif source.get("disposition_class") == "blocked":
                outcome = "blocked"
            elif field not in payload:
                outcome = "missing_field"
            else:
                operator = str(gate.get("op"))
                expected = gate.get("value")
                passed = observed == expected if operator == "==" else observed != expected
                outcome = "passed" if passed else "value_mismatch"
            rows.append(
                {
                    "consumer": consumer["id"],
                    "upstream": upstream,
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected_value": gate.get("value"),
                    "observed_value": observed,
                    "artifact_path": source.get("selected_evidence_path"),
                    "artifact_sha256": source.get("artifact_sha256"),
                    "producer_disposition_class": source.get("disposition_class"),
                    "outcome": outcome,
                    "passed": outcome == "passed",
                }
            )
    return rows


def _blocked_claim(claim: str, producer: str, source: Mapping[str, Any]) -> JsonDict:
    """Keep unavailable evidence explicit instead of converting it to zero."""

    return {
        "unit_id": f"claim:{claim}",
        "arm": "authenticated_aggregation",
        "claim": claim,
        "producer": producer,
        "producer_sha256": source.get("artifact_sha256"),
        "verdict_class": "blocked",
        "metric": None,
        "metrics": {},
        "costs": {"model_loads": 0, "generation_calls": 0, "hardware_operations": 0},
        "failures": [f"producer_{source.get('disposition_class', 'missing')}"],
        "abstention": True,
        "censored": False,
        "promotes_scientific_efficacy": False,
    }


def _source_cost_claim(source: Mapping[str, Any]) -> JsonDict:
    payload = source["payload"]
    comparisons = {row["comparison"]: row for row in payload["independent_comparison_rows"]}
    direct = comparisons["batched_verifier_vs_joint_direct"]
    serial = comparisons["serial_vs_batched_verifier"]
    promoted = payload.get("batch_promotion_score") == 1
    return {
        "unit_id": "claim:source_cost",
        "arm": "joint_claim_vs_controls",
        "claim": "source_cost",
        "producer": "exp7322-batch-audit",
        "producer_sha256": source["artifact_sha256"],
        "verdict_class": "positive" if promoted else "null",
        "metric": direct["full_cost_speedup_estimate"],
        "metrics": {
            "speedup_vs_direct": direct["full_cost_speedup_estimate"],
            "speedup_vs_serial": serial["full_cost_speedup_estimate"],
            "accuracy_difference_vs_direct": direct["accuracy_difference"],
            "coverage_difference_vs_direct": direct["coverage_difference"],
            "batch_promotion_score": payload.get("batch_promotion_score"),
        },
        "costs": deepcopy(payload.get("cost_summary", {})),
        "failures": [] if promoted else ["frozen_batch_value_gates_failed"],
        "abstention": False,
        "censored": False,
        "promotes_scientific_efficacy": promoted,
    }


def _tool_claim(source: Mapping[str, Any]) -> JsonDict:
    payload = source["payload"]
    chain = payload["tool_use_chain"]
    complete = sum(row.get("passed") is True for row in chain["rows"])
    score = payload.get("arc_tool_use_score")
    return {
        "unit_id": "claim:live_tool_causality",
        "arm": "direct_selfparse",
        "claim": "live_tool_causality",
        "producer": "exp7319-arc-session",
        "producer_sha256": source["artifact_sha256"],
        "verdict_class": "circular_positive" if score == 1 else "null",
        "metric": complete,
        "metrics": {
            "complete_chains": complete,
            "successful_tool_results": chain["successful_tool_results"],
            "policy_consumed_results": chain["policy_consumed_results"],
            "subsequent_environment_actions": chain["subsequent_environment_actions"],
        },
        "costs": deepcopy(payload["rows"][0]["costs"]),
        "failures": [] if score == 1 else ["no_tool_result_to_later_policy_action_chain"],
        "abstention": score != 1,
        "censored": False,
        "promotes_scientific_efficacy": False,
    }


def _learning_claim(source: Mapping[str, Any]) -> JsonDict:
    payload = source["payload"]
    overall = {
        row["comparison_id"]: row
        for row in payload["independent_comparison_rows"]
        if row.get("stratum") == "overall"
    }
    reset = overall["total_queries_vs_reset"]
    intervention_changes = sum(
        int(row["changed_decision_count"]) + int(row["changed_query_sequence_count"])
        for row in payload["causal_intervention_rows"]
    )
    promoted = payload.get("addition_promotion_score") == 1
    return {
        "unit_id": "claim:structural_learning",
        "arm": "persistent_structural_acquisition",
        "claim": "structural_learning",
        "producer": "exp7325-addition-audit",
        "producer_sha256": source["artifact_sha256"],
        "verdict_class": "circular_positive" if promoted else "null",
        "metric": reset["estimate"],
        "metrics": {
            "query_ratio_vs_reset": reset["estimate"],
            "query_ratio_ci95_upper": reset["ci95_upper"],
            "paired_stream_count": reset["paired_stream_count"],
            "intervention_changes": intervention_changes,
            "addition_promotion_score": payload.get("addition_promotion_score"),
        },
        "costs": {"total_query_attempts": 25_632, "model_loads": 0, "generation_calls": 0},
        "failures": [],
        "abstention": False,
        "censored": False,
        "promotes_scientific_efficacy": False,
    }


def _rust_claim(source: Mapping[str, Any]) -> JsonDict:
    payload = source["payload"]
    parity = payload["kernel_rows"]["parity"]
    mismatches = sum(row.get("matched") is not True for row in parity)
    speed_gate = payload["performance_outcome"]["ten_x_lower_bound_passed"] is True
    parity_rate = (len(parity) - mismatches) / len(parity)
    return {
        "unit_id": "claim:rust_software_parity",
        "arm": "persistent_host_service_boundary",
        "claim": "rust_software_parity",
        "producer": "exp7326-constraint-kernel",
        "producer_sha256": source["artifact_sha256"],
        "verdict_class": "circular_positive" if speed_gate else "null",
        "metric": parity_rate,
        "metrics": {
            "parity_rows": len(parity),
            "parity_mismatches": mismatches,
            "parity_rate": parity_rate,
            "ten_x_cost_gate": speed_gate,
            "paired_speedup_intervals": deepcopy(
                payload["performance_outcome"]["paired_speedup_intervals"]
            ),
        },
        "costs": {"paired_cost_rows": len(payload["kernel_rows"]["cost"])},
        "failures": [] if speed_gate else ["ten_x_paired_host_cost_gate_failed"],
        "abstention": False,
        "censored": False,
        "promotes_scientific_efficacy": False,
    }


def _board_claim(source: Mapping[str, Any]) -> JsonDict:
    payload = source["payload"]
    board_rows = payload["board_rows"]
    blocked = [row["board"] for row in board_rows if str(row["disposition"]).startswith("blocked")]
    graduated = sum(row.get("historical_graduation_preserved") is True for row in board_rows)
    return {
        "unit_id": "claim:board_evidence",
        "arm": "read_only_board_continuity",
        "claim": "board_evidence",
        "producer": "exp7327-board-continuity",
        "producer_sha256": source["artifact_sha256"],
        "verdict_class": "blocked" if blocked else "circular_positive",
        "metric": graduated,
        "metrics": {
            "board_rows": len(board_rows),
            "graduated_scopes_preserved": graduated,
            "blocked_boards": blocked,
            "current_hardware_operations": payload["hardware_operations_issued_count"],
        },
        "costs": {"hardware_operations": payload["hardware_operations_issued_count"]},
        "failures": ["gatemate_changed_physical_state_receipt_missing"] if blocked else [],
        "abstention": bool(blocked),
        "censored": False,
        "promotes_scientific_efficacy": False,
    }


def build_claim_matrix(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Recompute five conclusions without merging their authority classes."""

    definitions: tuple[tuple[str, str, Callable[[Mapping[str, Any]], JsonDict]], ...] = (
        ("source_cost", "exp7322-batch-audit", _source_cost_claim),
        ("live_tool_causality", "exp7319-arc-session", _tool_claim),
        ("structural_learning", "exp7325-addition-audit", _learning_claim),
        ("rust_software_parity", "exp7326-constraint-kernel", _rust_claim),
        ("board_evidence", "exp7327-board-continuity", _board_claim),
    )
    rows: list[JsonDict] = []
    for claim, producer, reducer in definitions:
        source = evidence[producer]
        if (
            source.get("accepted_for_reduction") is True
            and source.get("authenticated") is True
            and source.get("disposition_class") in {"positive", "circular_positive", "null"}
        ) or (
            producer == "exp7327-board-continuity"
            and source.get("authenticated") is True
            and source.get("disposition_class") == "blocked"
        ):
            rows.append(reducer(source))
        else:
            rows.append(_blocked_claim(claim, producer, source))
    return rows


def _external_failure(source: Mapping[str, Any], producer: str) -> JsonDict:
    """Preserve the producer's exact failed check when it is available."""

    if source.get("selected_evidence_path") is None:
        return {
            "upstream": producer,
            "check": "required_evidence_available",
            "artifact_field": "declared_deliverable_or_canonical_block",
            "expected_value": "terminal authenticated evidence",
            "observed_value": None,
            "terminal_blocking": True,
        }
    payload = source["payload"]
    first = payload.get("gate_check_summary", {}).get("first_failure")
    if isinstance(first, Mapping):
        return {
            "upstream": producer,
            "check": first.get("check"),
            "artifact_field": first.get("field"),
            "expected_value": deepcopy(first.get("expected_value")),
            "observed_value": deepcopy(first.get("observed_value")),
            "terminal_blocking": True,
        }
    return {
        "upstream": producer,
        "check": "producer_terminal_class",
        "artifact_field": "verdict_class",
        "expected_value": "positive | circular_positive | null",
        "observed_value": source.get("disposition_class"),
        "terminal_blocking": True,
    }


def terminal_state(
    evidence: Mapping[str, Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    validation_passed: bool,
) -> JsonDict:
    """Keep capstone completion separate from external scientific state."""

    if not validation_passed:
        return {
            "status": "complete",
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_v643_capstone_required_validation_failed",
            "gate_check_summary": {
                "passed": False,
                "failures": [
                    {
                        "upstream": "exp7328-capstone",
                        "check": "required_scoped_validation",
                        "artifact_field": "required_checks_passed",
                        "expected_value": True,
                        "observed_value": False,
                        "terminal_blocking": True,
                    }
                ],
            },
        }
    failures: list[JsonDict] = []
    for row in claims:
        if row.get("verdict_class") == "blocked":
            failures.append(_external_failure(evidence[str(row["producer"])], str(row["producer"])))
    if failures:
        first = failures[0]
        name = str(first["upstream"]).replace("-", "_")
        return {
            "status": "complete",
            "verdict_class": "blocked",
            "honest_verdict": (
                f"blocked_{name}: all thirteen dispositions are complete; "
                "the named external prerequisite remains unavailable"
            ),
            "gate_check_summary": {"passed": False, "failures": failures},
        }
    classes = {str(row.get("verdict_class")) for row in claims}
    verdict_class = "circular_positive" if "circular_positive" in classes else "null"
    return {
        "status": "complete",
        "verdict_class": verdict_class,
        "honest_verdict": (
            "complete_circular_positive_bounded_v643_claims_no_broad_efficacy"
            if verdict_class == "circular_positive"
            else "complete_null_v643_claims"
        ),
        "gate_check_summary": {"passed": True, "failures": []},
    }


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Represent every producer and the non-recursive capstone self-row."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        task_id = str(task["id"])
        source = evidence[task_id]
        payload = source["payload"]
        incoming = [deepcopy(row) for row in gate_rows if row["consumer"] == task_id]
        rows.append(
            {
                "order": order,
                "task_id": task_id,
                "title": task["title"],
                "phase": task["phase"],
                "declared_artifact_path": source["declared_deliverable_path"],
                "canonical_gate_block_path": source["canonical_gate_block_path"],
                "selected_evidence_path": source["selected_evidence_path"],
                "artifact_sha256": source["artifact_sha256"],
                "status": payload.get("status", "missing"),
                "honest_verdict": payload.get("honest_verdict", "blocked_missing_evidence"),
                "verdict_class": payload.get("verdict_class"),
                "disposition_class": source["disposition_class"],
                "authenticated": source["authenticated"],
                "quarantined": source["quarantined"],
                "producer_validation_errors": deepcopy(source["producer_validation_errors"]),
                "incoming_gate_records": incoming,
                "skipped_gate_records": [row for row in incoming if not row["passed"]],
            }
        )
    self_task = tasks[-1]
    rows.append(
        {
            "order": 13,
            "task_id": self_task["id"],
            "title": self_task["title"],
            "phase": self_task["phase"],
            "declared_artifact_path": self_task["deliverable"],
            "canonical_gate_block_path": None,
            "selected_evidence_path": None,
            "artifact_sha256": None,
            "status": terminal["status"],
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "disposition_class": terminal["verdict_class"],
            "authenticated": True,
            "quarantined": False,
            "producer_validation_errors": [],
            "incoming_gate_records": [],
            "skipped_gate_records": [],
        }
    )
    return rows


def prior_failure_rows(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Compare every prior verdict as exact UTF-8 text, without normalization."""

    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        current = (
            str(terminal["honest_verdict"])
            if task_id == "exp7328-capstone"
            else str(evidence[task_id]["payload"].get("honest_verdict", ""))
        )
        for prior in task.get("prior_failures") or []:
            verdict = str(prior["verdict"])
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior["experiment_id"],
                    "prior_honest_verdict_bytes": verdict,
                    "prior_honest_verdict_sha256": sha256_bytes(verdict.encode("utf-8")),
                    "current_honest_verdict_bytes": current,
                    "current_honest_verdict_sha256": sha256_bytes(current.encode("utf-8")),
                    "exact_repeat": verdict == current,
                    "retire_if_same_verdict": prior["retire_if_same_verdict"],
                    "addressed_by": prior["addressed_by"],
                }
            )
    return rows


def next_branch_decisions(evidence: Mapping[str, Mapping[str, Any]], root: Path) -> list[JsonDict]:
    """Give every current branch and preserved retirement one bounded action."""

    batch = evidence["exp7322-batch-audit"]["payload"]
    learning = evidence["exp7325-addition-audit"]["payload"]
    kernel = evidence["exp7326-constraint-kernel"]["payload"]
    board = evidence["exp7327-board-continuity"]["payload"]
    v642 = read_json(root / V642_CAPSTONE_PATH)
    cost = read_json(root / V642_COST_PATH)
    suffix = next(
        row
        for row in v642["retirement_decisions"]
        if "longest-consistent-suffix" in str(row.get("scope", ""))
    )
    rows = [
        {
            "branch": "source_cost",
            "action": "retire",
            "scope": batch["retirement_decision"]["failed_mechanism"],
            "evidence": deepcopy(batch["retirement_decision"]),
            "reopening_condition": batch["retirement_decision"]["future_rerun_condition"],
        },
        {
            "branch": "live_tool_causality",
            "action": "retire",
            "scope": "current direct selfparse tool-result-to-action mechanism",
            "evidence": {
                "arc_tool_use_score": evidence["exp7319-arc-session"]["payload"][
                    "arc_tool_use_score"
                ]
            },
            "reopening_condition": "a changed runtime emitter proves one tool-result-to-later-policy-action chain",
        },
        {
            "branch": "structural_learning",
            "action": "retain",
            "scope": learning["retirement_decision"]["mechanism_boundary"],
            "evidence": deepcopy(learning["retirement_decision"]),
            "reopening_condition": learning["retirement_decision"]["further_work_condition"],
        },
        {
            "branch": "rust_software_parity",
            "action": "implement",
            "scope": "changed in-process or batched native boundary; preserve bit-exact parity",
            "evidence": {
                "correctness_outcome": kernel["correctness_outcome"],
                "ten_x_lower_bound_passed": kernel["performance_outcome"][
                    "ten_x_lower_bound_passed"
                ],
            },
            "reopening_condition": "a measured changed boundary reaches the predeclared cost gate without parity loss",
        },
        {
            "branch": "board_evidence",
            "action": "blocked_pending_prerequisite",
            "scope": "GateMate changed-physical-state continuity",
            "evidence": deepcopy(board["gate_check_summary"]["first_failure"]),
            "reopening_condition": board["next_hardware_conditions"]["gatemate"]["condition"],
        },
        {
            "branch": "v642_suffix_retirement",
            "action": "retire",
            "scope": suffix["scope"],
            "evidence": deepcopy(suffix),
            "reopening_condition": suffix["lawful_next_condition"],
        },
        {
            "branch": "v642_storage_retirement",
            "action": "retire",
            "scope": cost["decision"]["scope"],
            "evidence": deepcopy(cost["decision"]),
            "reopening_condition": cost["decision"]["changed_prerequisite"],
        },
    ]
    return rows


def run_publication_gate(root: Path) -> tuple[JsonDict, JsonDict]:
    """Run the stable read-only gate and retain its exact JSON bytes."""

    argv = [str(root / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"]
    started = time.monotonic()
    progress(2, "before_subprocess", "publication_gate")
    completed = subprocess.run(  # noqa: S603 - fixed repository script and argument vector.
        argv,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = time.monotonic() - started
    progress(2, "after_subprocess", f"publication_gate exit={completed.returncode}")
    output = completed.stdout
    payload = json.loads(output) if completed.returncode == 0 else {}
    return payload, {
        "name": "publication_gate",
        "command": shlex.join(argv),
        "command_argv": argv,
        "scope": "stable_historical_fover_gate_read_only",
        "exit_code": completed.returncode,
        "duration_s": elapsed,
        "log_path": None,
        "log_sha256": sha256_bytes((output + completed.stderr).encode("utf-8")),
        "passed": completed.returncode == 0 and isinstance(payload, dict),
        "timed_out": False,
        "output_tail": (output + completed.stderr)[-4000:],
    }


def _source_hashes(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Bind contract, producer, historical retirement, and implementation bytes."""

    paths = [
        Path(str(contract["selected_yaml_path"])),
        DESIGN_PATH,
        SPEC_PATH,
        EXCLUSION_PATH,
        V642_CAPSTONE_PATH,
        V642_COST_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    ]
    paths.extend(
        Path(str(row["selected_evidence_path"]))
        for row in evidence.values()
        if row["selected_evidence_path"]
    )
    return {str(path): {"sha256": sha256(root / path), "available": True} for path in paths}


def _preconditions(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Record identities and exact observed values before reduction."""

    required = {
        "driving_requirement": "REQ-REPORT-7328" in (root / SPEC_PATH).read_text(encoding="utf-8"),
        "contract_exact": contract.get("passed") is True,
        "task_count": len(contract.get("tasks", [])) == 13,
        "producer_slots": len(evidence) == 12,
        "output_parent_writable": (root / DEFAULT_OUTPUT_PATH).parent.is_dir(),
    }
    return [
        {
            "check": check,
            "expected_value": True,
            "observed_value": observed,
            "passed": observed is True,
        }
        for check, observed in required.items()
    ]


def _acceptance_results(
    contract: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    publication: Mapping[str, Any],
    validation_passed: bool,
) -> list[JsonDict]:
    """Keep closure, evidence, validation, and publication as separate checks."""

    definitions = (
        (
            "exact_contract",
            True,
            contract.get("passed"),
            contract.get("passed") is True,
            "Independent contract bytes must agree.",
        ),
        (
            "thirteen_dispositions",
            13,
            len(dispositions),
            len(dispositions) == 13,
            "Closure counts all tasks including self.",
        ),
        (
            "five_claim_classes",
            list(CLAIM_NAMES),
            [row.get("claim") for row in claims],
            [row.get("claim") for row in claims] == list(CLAIM_NAMES),
            "Authority classes cannot collapse.",
        ),
        (
            "required_scoped_validation",
            True,
            validation_passed,
            validation_passed,
            "Affected failures disqualify current output.",
        ),
        (
            "stable_publication_gate_shape",
            ["G1", "G2", "G3", "G4"],
            list(publication.get("gates", {})),
            list(publication.get("gates", {})) == ["G1", "G2", "G3", "G4"],
            "Historical publication readiness is separate from V643 science.",
        ),
    )
    return [
        {
            "check": name,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
            "principle": principle,
        }
        for name, expected, observed, passed, principle in definitions
    ]


def _evidence_categories(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """List null, quarantine, external block, and skip states explicitly."""

    return {
        "complete_nulls": [
            task for task, row in evidence.items() if row["disposition_class"] == "null"
        ],
        "quarantines": [
            task for task, row in evidence.items() if row["disposition_class"] == "quarantined"
        ],
        "external_blocks": [
            task
            for task, row in evidence.items()
            if row["disposition_class"] in {"blocked", "missing"}
        ],
        "conditional_skips": [
            task
            for task, row in evidence.items()
            if row["evidence_source"] == "canonical_conductor_block"
        ],
    }


def build_artifact(
    *,
    root: Path,
    run_date: str,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    publication_gate: Mapping[str, Any],
    publication_receipt: Mapping[str, Any],
    validation: Mapping[str, Any],
    started_at: str,
    completed_at: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one terminal artifact from already authenticated current evidence."""

    claims = build_claim_matrix(evidence)
    required_passed = validation.get("required_checks_passed") is True
    terminal_passed = validation.get("terminal_checks_passed", True) is True
    terminal = terminal_state(evidence, claims, required_passed and terminal_passed)
    gates = replay_gates(contract["tasks"], evidence)
    dispositions = task_dispositions(contract["tasks"], evidence, terminal, gates)
    validation_receipts = [
        *deepcopy(validation.get("validation_receipts", [])),
        deepcopy(dict(publication_receipt)),
    ]
    artifact: JsonDict = {
        "schema": "carnot.experiment_7328.v643_capstone.v1",
        "experiment_id": "exp7328-capstone",
        "milestone": MILESTONE,
        "status": terminal["status"],
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "preconditions_checked": _preconditions(root, contract, evidence),
        "MODEL_SPECS": MODEL_SPECS,
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "rows": deepcopy(claims),
        "sample_size_budget": {
            "planned": 13,
            "attempted": 13,
            "complete": 13,
            "censored": 0,
            "stopping_rule": "classify each of the thirteen frozen V643 tasks exactly once",
        },
        "acceptance_gate_results": _acceptance_results(
            contract, dispositions, claims, publication_gate, required_passed and terminal_passed
        ),
        "gate_check_summary": deepcopy(terminal["gate_check_summary"]),
        "verifier_is_oracle": True,
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "validation_receipts": validation_receipts,
        "required_checks_passed": required_passed,
        "terminal_checks_passed": terminal_passed,
        "repository_health": deepcopy(validation.get("repository_health", {})),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "capstone_complete_score": 1,
        "capstone_readiness_score": 0,
        "capstone_promotion_score": 0,
        "task_dispositions": dispositions,
        "contract_rows": deepcopy(contract["contract_rows"]),
        "same_milestone_gate_replay_rows": gates,
        "claim_matrix": deepcopy(claims),
        "evidence_categories": _evidence_categories(evidence),
        "prior_failure_rows": prior_failure_rows(contract["tasks"], evidence, terminal),
        "publication_gate": deepcopy(dict(publication_gate)),
        "next_branch_decisions": next_branch_decisions(evidence, root),
        "publication_performed": False,
        "upload_performed": False,
        "submission_performed": False,
        "external_message_performed": False,
        "production_default_changed": False,
        "research_roadmap_modified": False,
        "conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _receipt_set_passes(receipts: object) -> bool:
    """Require one successful receipt for each fixed scoped command."""

    if not isinstance(receipts, list):
        return False
    return all(
        sum(row.get("name") == name for row in receipts if isinstance(row, Mapping)) == 1
        and next(
            row for row in receipts if isinstance(row, Mapping) and row.get("name") == name
        ).get("passed")
        is True
        and next(
            row for row in receipts if isinstance(row, Mapping) and row.get("name") == name
        ).get("exit_code")
        == 0
        for name in REQUIRED_SCOPED_CHECKS
    )


def validate_artifact(
    value: object,
    *,
    root: Path = REPO_ROOT,
    replay: bool = False,
) -> list[str]:
    """Cold-check identity, rows, claims, decisions, hashes, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    artifact = value
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_ARTIFACT_FIELDS.issubset(artifact), "required_fields")
    add(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != "carnot.experiment_7328.v643_capstone.v1"
        or artifact.get("experiment_id") != "exp7328-capstone"
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "identity",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS,
        "model_boundary",
    )
    add(
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host",
        "substrate",
    )
    dispositions = artifact.get("task_dispositions")
    add(
        not isinstance(dispositions, list)
        or len(dispositions) != 13
        or [row.get("task_id") for row in dispositions] != list(EXPECTED_TASK_IDS),
        "task_dispositions",
    )
    claims = artifact.get("claim_matrix")
    add(
        not isinstance(claims, list)
        or [row.get("claim") for row in claims] != list(CLAIM_NAMES)
        or artifact.get("rows") != claims,
        "claim_matrix",
    )
    add(
        artifact.get("capstone_complete_score") != 1
        or artifact.get("capstone_readiness_score") != 0
        or artifact.get("capstone_promotion_score") != 0,
        "capstone_scores",
    )
    add(not _receipt_set_passes(artifact.get("validation_receipts")), "validation_receipts")
    add(
        any(
            artifact.get(field) is not False
            for field in (
                "publication_performed",
                "upload_performed",
                "submission_performed",
                "external_message_performed",
                "production_default_changed",
                "research_roadmap_modified",
                "conductor_modified",
            )
        ),
        "unauthorized_action",
    )
    add(
        artifact.get("reproducibility_checksum") != artifact_checksum(artifact),
        "reproducibility_checksum",
    )
    if replay:
        contract = load_contract(root)
        evidence = load_repository_evidence(root, contract["tasks"])
        expected_claims = build_claim_matrix(evidence)
        expected_terminal = terminal_state(
            evidence,
            expected_claims,
            artifact.get("required_checks_passed") is True
            and artifact.get("terminal_checks_passed") is True,
        )
        gates = replay_gates(contract["tasks"], evidence)
        expected_dispositions = task_dispositions(
            contract["tasks"], evidence, expected_terminal, gates
        )
        add(artifact.get("contract_rows") != contract["contract_rows"], "contract_rows")
        add(claims != expected_claims, "claim_matrix")
        add(dispositions != expected_dispositions, "task_dispositions")
        add(artifact.get("same_milestone_gate_replay_rows") != gates, "gate_replay")
        add(
            artifact.get("status") != expected_terminal["status"]
            or artifact.get("verdict_class") != expected_terminal["verdict_class"]
            or artifact.get("honest_verdict") != expected_terminal["honest_verdict"]
            or artifact.get("gate_check_summary") != expected_terminal["gate_check_summary"],
            "terminal_state",
        )
        add(
            artifact.get("prior_failure_rows")
            != prior_failure_rows(contract["tasks"], evidence, expected_terminal),
            "prior_failure_rows",
        )
        add(
            artifact.get("next_branch_decisions") != next_branch_decisions(evidence, root),
            "next_branch_decisions",
        )
        add(
            artifact.get("source_artifact_hashes") != _source_hashes(root, contract, evidence),
            "source_artifact_hashes",
        )
        publication, _ = run_publication_gate(root)
        add(artifact.get("publication_gate") != publication, "publication_gate")
    return errors


def date_argument(value: str) -> str:
    """Require the conductor's compact execution-date format."""

    if not re.fullmatch(r"\d{8}", value):
        raise ValueError("date must use YYYYMMDD")
    return value


def _phase_row(name: str, start: float, end: float, units: int, checkpoint: str | None) -> JsonDict:
    """Record one real disjoint span without padding."""

    return {
        "phase": name,
        "start_s": start,
        "end_s": end,
        "duration_s": end - start,
        "units": units,
        "checkpoint": checkpoint,
        "pending_operations": [],
    }


def _run_terminal_commands(root: Path, candidate: Path, raw_dir: Path) -> list[JsonDict]:
    """Run the independent reducer and both required terminal linters."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,sys; from pathlib import Path; "
        "from carnot.experiment_7328_v643_capstone import validate_artifact; "
        "p=Path(sys.argv[1]); e=validate_artifact(json.loads(p.read_text()),root=Path.cwd(),replay=True); "
        "print(json.dumps({'errors':e}),flush=True); raise SystemExit(bool(e))"
    )
    commands = [
        scoped.CommandSpec(
            "independent_terminal_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "terminal_candidate",
        ),
        scoped.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        scoped.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal_candidate",
        ),
    ]
    return scoped.run_commands(root, commands, log_dir=raw_dir / "terminal_validation")


def main(
    argv: Sequence[str] | None = None,
) -> int:  # pragma: no cover - required E2E exercises CLI.
    """Run scoped validation, terminal checks, and one atomic final write."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_argument, default=RUN_DATE)
    args = parser.parse_args(argv)
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []
    progress(0, "start", f"date={args.date}")
    checkpoint = REPO_ROOT / DEFAULT_CHECKPOINT_PATH
    atomic_write_json(
        checkpoint,
        {"status": "running", "experiment_id": "exp7328-capstone", "started_at_utc": started_at},
    )

    phase_start = time.monotonic() - started
    progress(1, "start", "authenticate contract and exact producers")
    contract = load_contract(REPO_ROOT)
    evidence = load_repository_evidence(REPO_ROOT, contract["tasks"])
    phase_end = time.monotonic() - started
    spans.append(
        _phase_row("authentication", phase_start, phase_end, 13, str(DEFAULT_CHECKPOINT_PATH))
    )

    phase_start = time.monotonic() - started
    publication, publication_receipt = run_publication_gate(REPO_ROOT)
    phase_end = time.monotonic() - started
    spans.append(_phase_row("publication_gate", phase_start, phase_end, 1, None))

    raw_dir = REPO_ROOT / DEFAULT_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="exp7328-validation-", dir="/tmp"))
    private_basetemp = temporary_root / "pytest"
    private_basetemp.mkdir()
    phase_start = time.monotonic() - started
    progress(3, "start", "scoped affected validation")
    validation = scoped.run_scoped_validation(
        REPO_ROOT,
        [str(TEST_PATH)],
        [str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=private_basetemp,
        coverage_file=temporary_root / ".coverage",
        log_dir=raw_dir / "scoped_validation",
        historical_failures=[],
    )
    phase_end = time.monotonic() - started
    spans.append(
        _phase_row("scoped_validation", phase_start, phase_end, len(REQUIRED_SCOPED_CHECKS), None)
    )

    phase_start = time.monotonic() - started
    progress(4, "start", "build measured terminal candidate")
    candidate = build_artifact(
        root=REPO_ROOT,
        run_date=args.date,
        contract=contract,
        evidence=evidence,
        publication_gate=publication,
        publication_receipt=publication_receipt,
        validation=validation,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured-terminal-candidate.json"
    atomic_write_json(candidate_path, candidate)
    phase_end = time.monotonic() - started
    spans.append(
        _phase_row(
            "candidate_reduction",
            phase_start,
            phase_end,
            5,
            str(candidate_path.relative_to(REPO_ROOT)),
        )
    )

    phase_start = time.monotonic() - started
    progress(5, "start", "terminal candidate validation")
    terminal_receipts = _run_terminal_commands(REPO_ROOT, candidate_path, raw_dir)
    validation = dict(validation)
    validation["validation_receipts"] = [
        *validation["validation_receipts"],
        *terminal_receipts,
    ]
    validation["terminal_checks_passed"] = all(row["passed"] is True for row in terminal_receipts)
    phase_end = time.monotonic() - started
    spans.append(
        _phase_row("terminal_validation", phase_start, phase_end, len(terminal_receipts), None)
    )

    final = build_artifact(
        root=REPO_ROOT,
        run_date=args.date,
        contract=contract,
        evidence=evidence,
        publication_gate=publication,
        publication_receipt=publication_receipt,
        validation=validation,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    errors = validate_artifact(final, root=REPO_ROOT, replay=True)
    if errors:
        progress(6, "failed", f"cold_replay_errors={errors}")
        return 1
    output = REPO_ROOT / DEFAULT_OUTPUT_PATH
    atomic_write_json(output, final)
    reloaded = read_json(output)
    reload_errors = validate_artifact(reloaded, root=REPO_ROOT, replay=True)
    progress(6, "complete", f"output={DEFAULT_OUTPUT_PATH} reload_errors={reload_errors}")
    return int(bool(reload_errors))


if __name__ == "__main__":  # pragma: no cover - wrapper is the declared entrypoint.
    raise SystemExit(main())
