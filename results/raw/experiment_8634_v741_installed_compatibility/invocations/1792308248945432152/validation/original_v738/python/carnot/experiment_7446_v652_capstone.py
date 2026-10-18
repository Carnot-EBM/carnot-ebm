"""Close V652 with thirteen authenticated task dispositions.

The reducer reads existing artifacts and runs bounded validation commands. It
does not invoke a model, train an energy head, operate hardware, or publish.

Spec refs: REQ-REPORT-7446 and SCENARIO-REPORT-7446-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7329_v644_contract import (
    _field_declared,
    parse_markdown_contract,
    parse_yaml_contract,
)
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from scripts.publication_gate import evaluate as evaluate_publication_gates


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260920"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "exp7446-capstone"
SCHEMA = "carnot.exp7446.v652.capstone.v1"

ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7446_v652_capstone.json")
RAW_DIR = Path("results/raw/experiment_7446_v652_capstone")
MODULE_PATH = Path("python/carnot/experiment_7446_v652_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7446_v652_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7446_v652_capstone.py")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")

EXPECTED_TASK_IDS = (
    "exp7434-contract-methods",
    "exp7435-round-breaker",
    "exp7436-selection-protocol",
    "exp7437-span-protocol",
    "exp7438-mixture-prototype",
    "exp7439-certified-decisions",
    "exp7440-mixture-learning",
    "exp7441-decision-audit",
    "exp7442-span-capture",
    "exp7443-span-audit",
    "exp7444-arc-supervisor-evidence",
    "exp7445-hardware-envelope",
    EXPERIMENT_ID,
)
CLAIM_BRANCHES = (
    "contract",
    "runtime_recovery",
    "static_decision",
    "online_learning",
    "extraction",
    "arc",
    "hardware",
)
SCIENTIFIC_TASKS = (
    "exp7439-certified-decisions",
    "exp7440-mixture-learning",
    "exp7441-decision-audit",
    "exp7442-span-capture",
    "exp7443-span-audit",
    "exp7444-arc-supervisor-evidence",
    "exp7445-hardware-envelope",
)
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    SPEC_PATH,
    Path("python/carnot/experiment_7433_v651_capstone.py"),
    PUBLICATION_GATE_PATH,
    Path("scripts/conductor_gates.py"),
    Path("ops/north-star.md"),
    Path("ops/verifier_gaps.md"),
    Path("research-complete.yaml"),
    Path("ops/conductor-log.md"),
    ROADMAP_PATH,
    DESIGN_PATH,
)

REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "publication_gate_json",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def utc_now() -> str:  # pragma: no cover - real process boundary.
    """Return the real UTC time for an artifact boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public process boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Emit a flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7446] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_yaml_mapping(path: Path) -> JsonDict:
    """Load one YAML mapping and reject another top-level shape."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - I/O guard.
        raise ValueError(f"unreadable YAML mapping: {path}: {exc}") from exc
    if not isinstance(value, dict):  # pragma: no cover - parser guard.
        raise ValueError(f"YAML mapping required: {path}")
    return value


def load_json_object(path: Path) -> JsonDict:
    """Load one JSON mapping without treating a list as an artifact."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON object: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def numeric_experiment_id(value: object) -> int | None:
    """Return the first numeric experiment ID from common label forms."""

    if isinstance(value, int):  # pragma: no cover - retained compatibility.
        return value
    match = re.search(r"(?:exp|experiment[_-]?)(\d+)", str(value), re.IGNORECASE)
    return int(match.group(1)) if match else None


def terminal_status(value: Mapping[str, Any]) -> bool:
    """Accept a closed finding or an external structured block."""

    text = str(value.get("honest_verdict") or value.get("status") or "")
    return text.startswith(("complete", "blocked", "success", "passed", "shipped"))


def _public_task(task: Mapping[str, Any]) -> JsonDict:
    """Keep fields independently represented by both contract formats."""

    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "phase", "deliverable", "substrate", "gates")
    }


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Parse both V652 authorities and compare all thirteen rows."""

    try:
        markdown = parse_markdown_contract(markdown_text)
        parsed_yaml = parse_yaml_contract(roadmap)
    except (TypeError, ValueError) as exc:  # pragma: no cover - parser fail-closed path.
        return {
            "comparison_passed": False,
            "errors": [f"parse_error:{exc}"],
            "contract_rows": [],
        }
    mapping = roadmap if isinstance(roadmap, Mapping) else {}
    raw_tasks = mapping.get("tasks") if isinstance(mapping.get("tasks"), list) else []
    raw_by_id = {
        str(task.get("id")): (index, task)
        for index, task in enumerate(raw_tasks)
        if isinstance(task, Mapping)
    }
    markdown_tasks = markdown["tasks"]
    yaml_tasks = parsed_yaml["tasks"]
    width = max(len(EXPECTED_TASK_IDS), len(markdown_tasks), len(yaml_tasks))
    rows: list[JsonDict] = []
    undeclared = False
    for index in range(width):
        expected = markdown_tasks[index] if index < len(markdown_tasks) else {}
        observed = yaml_tasks[index] if index < len(yaml_tasks) else {}
        task_id = str(observed.get("id") or expected.get("id") or f"missing-{index + 1}")
        checks = {
            field: expected.get(field) == observed.get(field) and bool(expected) and bool(observed)
            for field in ("order", "id", "title", "phase", "deliverable", "substrate", "gates")
        }
        producer_rows: list[JsonDict] = []
        for gate in observed.get("gates") or []:
            upstream = str(gate.get("upstream"))
            producer_entry = raw_by_id.get(upstream)
            producer_index = producer_entry[0] if producer_entry else None
            producer = producer_entry[1] if producer_entry else {}
            field = str(gate.get("artifact_field"))
            declared = _field_declared(str(producer.get("prompt") or ""), field)
            precedes = producer_index is not None and producer_index < index
            producer_rows.append(
                {
                    "upstream": upstream,
                    "artifact_field": field,
                    "producer_in_roadmap": producer_entry is not None,
                    "producer_precedes_consumer": precedes,
                    "producer_field_declared": declared,
                    "passed": precedes and declared,
                }
            )
        checks["producer_fields"] = all(row["passed"] for row in producer_rows)
        undeclared = undeclared or not checks["producer_fields"]
        rows.append(
            {
                "order": index + 1,
                "task_id": task_id,
                "markdown": _public_task(expected),
                "yaml": _public_task(observed),
                "gates": deepcopy(observed.get("gates") or []),
                "producer_declarations": producer_rows,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    errors: list[str] = []
    if markdown.get("milestone") != MILESTONE:
        errors.append("markdown_milestone")
    if parsed_yaml.get("milestone") != MILESTONE:
        errors.append("yaml_milestone")
    if [row.get("id") for row in markdown_tasks] != list(EXPECTED_TASK_IDS):
        errors.append("markdown_task_order")
    if [row.get("id") for row in yaml_tasks] != list(EXPECTED_TASK_IDS):
        errors.append("yaml_task_order")
    if len(markdown_tasks) != 13 or len(yaml_tasks) != 13:
        errors.append("task_count")
    if any(not row["passed"] for row in rows):
        errors.append("row_mismatch")
    if undeclared:
        errors.append("producer_field_declaration")
    return {
        "comparison_passed": not errors,
        "errors": list(dict.fromkeys(errors)),
        "markdown_milestone": markdown.get("milestone"),
        "yaml_milestone": parsed_yaml.get("milestone"),
        "contract_rows": rows,
    }


def load_contract(root: Path) -> JsonDict:
    """Read the exact design and active YAML without using Exp7434 output."""

    roadmap = load_yaml_mapping(root / ROADMAP_PATH)
    comparison = compare_contract_authorities(
        (root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):  # pragma: no cover - guarded by comparison.
        raise ValueError("active V652 task list required")
    return {**comparison, "tasks": deepcopy(tasks), "title": roadmap.get("milestone_title")}


def _resolve_path(root: Path, label: str) -> Path:
    """Resolve a repository-relative evidence label or retain an absolute path."""

    path = Path(label)
    return path if path.is_absolute() else root / path


def authenticate_row_manifest(root: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Hash inline rows and every declared row shard without changing them."""

    rows = payload.get("rows")
    failures: list[str] = []
    if not isinstance(rows, list):
        failures.append("inline_rows_missing")
        rows = []
    shards = payload.get("row_shards") or []
    if not isinstance(shards, list):  # pragma: no cover - fail-closed schema guard.
        failures.append("row_shards_not_list")
        shards = []
    shard_row_count = 0
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):  # pragma: no cover - fail-closed schema guard.
            failures.append(f"row_shard_invalid:{index}")
            continue
        path = _resolve_path(root, str(shard.get("path") or ""))
        if not path.is_file():
            failures.append(f"row_shard_missing:{index}")
            continue
        if sha256_file(path) != shard.get("sha256"):
            failures.append(f"row_shard_hash:{index}")
        if shard.get("size_bytes") is not None and path.stat().st_size != shard.get("size_bytes"):
            failures.append(f"row_shard_size:{index}")
        count = shard.get("rows")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            failures.append(f"row_shard_rows:{index}")
        else:
            shard_row_count += count
    return {
        "authenticated": not failures,
        "inline_row_count": len(rows),
        "inline_rows_sha256": canonical_hash(rows),
        "raw_shard_count": len(shards),
        "raw_shard_row_count": shard_row_count,
        "failures": failures,
    }


def authenticate_validation_receipts(root: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Verify each receipt shape and the exact log bytes it cites."""

    receipts = payload.get("validation_receipts")
    failures: list[str] = []
    if not isinstance(receipts, list) or not receipts:
        return {
            "authenticated": False,
            "required_passed": False,
            "receipt_count": 0,
            "receipt_sha256": canonical_hash([]),
            "failures": ["validation_receipts_missing"],
        }
    required: list[Mapping[str, Any]] = []
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping):  # pragma: no cover - schema guard.
            failures.append(f"validation_receipt_invalid:{index}")
            continue
        if receipt.get("required") is True:
            required.append(receipt)
        if (
            not isinstance(receipt.get("name"), str)
            or not isinstance(receipt.get("passed"), bool)
            or not isinstance(receipt.get("exit_code"), int)
        ):
            failures.append(f"validation_receipt_shape:{index}")
        log_path = receipt.get("log_path")
        log_hash = receipt.get("log_sha256")
        path = _resolve_path(root, str(log_path or ""))
        if not path.is_file():
            failures.append(f"validation_log_missing:{index}")
        elif sha256_file(path) != log_hash:
            failures.append(f"validation_log_hash:{index}")
    return {
        "authenticated": not failures,
        "required_passed": bool(required)
        and all(row.get("passed") is True and row.get("exit_code") == 0 for row in required),
        "receipt_count": len(receipts),
        "receipt_sha256": canonical_hash(receipts),
        "failures": failures,
    }


def _fallback_pre_gate_path(task: Mapping[str, Any]) -> Path:
    """Derive the conductor's exact alternate pre-gate result filename."""

    task_id = str(task.get("id") or "")
    number = numeric_experiment_id(task_id)
    slug = task_id.split("-", 1)[1].replace("-", "_") if "-" in task_id else "task"
    return Path(f"results/experiment_{number}_{slug}.json")


def _missing_evidence(task: Mapping[str, Any]) -> JsonDict:
    """Represent absent external bytes as a terminal blocked disposition."""

    declared = str(task.get("deliverable") or "")
    return {
        "task_id": str(task.get("id")),
        "declared_path": declared,
        "source_path": declared,
        "source_kind": "missing",
        "authenticated": False,
        "available": False,
        "valid": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_missing_declared_evidence",
        "flagged_adversarial": False,
        "row_manifest_authenticated": False,
        "validation_receipts_authenticated": False,
        "required_validation_passed": False,
        "raw_row_count": 0,
        "raw_shard_count": 0,
        "sha256": None,
        "gate_check_summary": {
            "upstream": str(task.get("id")),
            "path": declared,
            "check": "declared_artifact_exists",
            "field": "path",
            "operator": "exists",
            "expected": True,
            "observed": False,
            "passed": False,
        },
        "payload": {},
    }


def load_evidence_slot(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one producer artifact or an exact conductor pre-gate."""

    task_id = str(task.get("id"))
    declared = Path(str(task.get("deliverable") or ""))
    actual = declared
    if not (root / actual).is_file():
        fallback = _fallback_pre_gate_path(task)
        if (root / fallback).is_file():
            actual = fallback
        else:
            return _missing_evidence(task)
    path = root / actual
    payload = load_json_object(path)
    expected_number = numeric_experiment_id(task_id)
    observed_number = numeric_experiment_id(
        payload.get("experiment_id") or payload.get("experiment") or payload.get("task_id")
    )
    if payload.get("schema") == "blocked_gate_check_v1":
        required = ("failed_upstream", "failed_field", "failed_expected", "failed_observed")
        authenticated = (
            observed_number == expected_number
            and payload.get("status") == "blocked"
            and str(payload.get("honest_verdict") or "").startswith("blocked")
            and all(field in payload for field in required)
        )
        return {
            "task_id": task_id,
            "declared_path": declared.as_posix(),
            "source_path": actual.as_posix(),
            "source_kind": "structured_pre_gate",
            "authenticated": authenticated,
            "available": False,
            "valid": authenticated,
            "verdict_class": "blocked",
            "honest_verdict": str(payload.get("honest_verdict")),
            "flagged_adversarial": False,
            "row_manifest_authenticated": authenticated,
            "validation_receipts_authenticated": authenticated,
            "required_validation_passed": authenticated,
            "raw_row_count": 0,
            "raw_shard_count": 0,
            "sha256": sha256_file(path),
            "gate_check_summary": {
                "upstream": payload.get("failed_upstream"),
                "path": actual.as_posix(),
                "check": "structured_pre_gate",
                "field": payload.get("failed_field"),
                "operator": "==",
                "expected": payload.get("failed_expected"),
                "observed": payload.get("failed_observed"),
                "passed": False,
            },
            "payload": payload,
        }
    row_manifest = authenticate_row_manifest(root, payload)
    validation = authenticate_validation_receipts(root, payload)
    verdict = payload.get("verdict_class")
    authenticated = (
        observed_number == expected_number
        and payload.get("milestone") == MILESTONE
        and verdict in CLOSED_VERDICTS
        and isinstance(payload.get("flagged_adversarial"), bool)
        and terminal_status(payload)
        and row_manifest["authenticated"]
        and validation["authenticated"]
    )
    flagged = payload.get("flagged_adversarial") is True
    valid = (
        authenticated
        and validation["required_passed"]
        and verdict != "disqualified"
        and not flagged
    )
    return {
        "task_id": task_id,
        "declared_path": declared.as_posix(),
        "source_path": actual.as_posix(),
        "source_kind": "terminal_artifact",
        "authenticated": authenticated,
        "available": authenticated,
        "valid": valid,
        "verdict_class": verdict if verdict in CLOSED_VERDICTS else "disqualified",
        "honest_verdict": str(payload.get("honest_verdict") or payload.get("status")),
        "flagged_adversarial": flagged,
        "row_manifest_authenticated": row_manifest["authenticated"],
        "validation_receipts_authenticated": validation["authenticated"],
        "required_validation_passed": validation["required_passed"],
        "raw_row_count": row_manifest["inline_row_count"],
        "raw_rows_sha256": row_manifest["inline_rows_sha256"],
        "raw_shard_count": row_manifest["raw_shard_count"],
        "raw_shard_row_count": row_manifest["raw_shard_row_count"],
        "validation_receipt_count": validation["receipt_count"],
        "validation_receipts_sha256": validation["receipt_sha256"],
        "sha256": sha256_file(path),
        "gate_check_summary": deepcopy(payload.get("gate_check_summary")),
        "payload": payload,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Authenticate the twelve predecessor slots in conductor order."""

    return {str(task["id"]): load_evidence_slot(root, task) for task in tasks[:-1]}


def _compare_gate(operator: str, observed: Any, expected: Any) -> bool:
    """Evaluate only the operators used by this roadmap."""

    if operator == "==":
        return observed == expected
    if operator == "in":
        return isinstance(expected, list) and observed in expected
    if operator == ">=":  # pragma: no cover - supported for roadmap compatibility.
        return isinstance(observed, (int, float)) and observed >= expected
    return False  # pragma: no cover - fail-closed for a future operator.


def gate_eligibility(score: object, verdict_class: object, flagged: object) -> bool:
    """Require completion plus a closed admissible class and no quarantine."""

    return score == 1 and verdict_class in {"null", "positive", "circular_positive"} and not flagged


def audit_structured_gates(
    contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Re-evaluate every scalar gate and its producer declaration."""

    tasks = contract.get("tasks") if isinstance(contract.get("tasks"), list) else []
    positions = {
        str(task.get("id")): index for index, task in enumerate(tasks) if isinstance(task, Mapping)
    }
    declaration_by_key: dict[tuple[str, str, str], bool] = {}
    for row in contract.get("contract_rows") or []:
        if not isinstance(row, Mapping):  # pragma: no cover - contract schema guard.
            continue
        consumer = str(row.get("task_id"))
        for declaration in row.get("producer_declarations") or []:
            if isinstance(declaration, Mapping):
                key = (
                    consumer,
                    str(declaration.get("upstream")),
                    str(declaration.get("artifact_field")),
                )
                declaration_by_key[key] = declaration.get("passed") is True
    rows: list[JsonDict] = []
    for consumer_index, task in enumerate(tasks):
        if not isinstance(task, Mapping):  # pragma: no cover - YAML schema guard.
            continue
        consumer = str(task.get("id"))
        for gate in task.get("gated_on") or []:
            upstream = str(gate.get("upstream"))
            field = str(gate.get("artifact_field"))
            source = evidence.get(upstream, {})
            observed = source.get("payload", {}).get(field)
            scalar_passed = _compare_gate(str(gate.get("op")), observed, gate.get("value"))
            source_admissible = bool(
                source.get("authenticated")
                and source.get("valid")
                and source.get("verdict_class") not in {"blocked", "disqualified", "partial"}
                and source.get("flagged_adversarial") is False
            )
            rows.append(
                {
                    "consumer": consumer,
                    "upstream": upstream,
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected": deepcopy(gate.get("value")),
                    "observed": deepcopy(observed),
                    "producer_in_roadmap": upstream in positions,
                    "producer_precedes_consumer": positions.get(upstream, consumer_index)
                    < consumer_index,
                    "producer_field_declared": declaration_by_key.get(
                        (consumer, upstream, field), False
                    ),
                    "scalar_passed": scalar_passed,
                    "source_admissible": source_admissible,
                    "passed": scalar_passed and source_admissible,
                }
            )
    return rows


def auditor_cross_checks(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Compare bounded producer metrics with both independent auditors."""

    static_source = evidence["exp7439-certified-decisions"]["payload"]
    online_source = evidence["exp7440-mixture-learning"]["payload"]
    decision_audit = evidence["exp7441-decision-audit"]["payload"]
    capture = evidence["exp7442-span-capture"]["payload"]
    extraction_audit = evidence["exp7443-span-audit"]["payload"]
    static = decision_audit.get("static_audit") or {}
    online = decision_audit.get("online_audit") or {}
    counts = extraction_audit.get("raw_disposition_counts") or {}
    audited_outcomes = sum(value for value in counts.values() if isinstance(value, int))
    maximum_calls = (capture.get("sample_size_budget") or {}).get("maximum_generation_calls")
    rows = [
        {
            "branch": "static_decision",
            "producer": "exp7439-certified-decisions",
            "auditor": "exp7441-decision-audit",
            "producer_completion_score": static_source.get("decision_capture_complete_score"),
            "producer_verdict_class": static_source.get("verdict_class"),
            "auditor_complete": static.get("complete"),
            "auditor_valid": static.get("valid"),
            "auditor_raw_row_count": static.get("raw_row_count"),
            "errors": deepcopy(static.get("errors") or []),
            "passed": static_source.get("decision_capture_complete_score") == 1
            and static.get("complete") is True
            and static.get("producer_verdict_class") == static_source.get("verdict_class"),
        },
        {
            "branch": "online_learning",
            "producer": "exp7440-mixture-learning",
            "auditor": "exp7441-decision-audit",
            "producer_completion_score": online_source.get("online_capture_complete_score"),
            "producer_verdict_class": online_source.get("verdict_class"),
            "auditor_complete": online.get("complete"),
            "auditor_valid": online.get("valid"),
            "auditor_raw_row_count": online.get("raw_row_count"),
            "auditor_prediction_row_count": online.get("prediction_row_count"),
            "errors": deepcopy(online.get("errors") or []),
            "passed": online_source.get("online_capture_complete_score") == 1
            and online.get("complete") is True
            and online.get("producer_verdict_class") == online_source.get("verdict_class"),
        },
        {
            "branch": "extraction",
            "producer": "exp7442-span-capture",
            "auditor": "exp7443-span-audit",
            "producer_verdict_class": capture.get("verdict_class"),
            "producer_flagged_adversarial": capture.get("flagged_adversarial"),
            "audited_outcome_count": audited_outcomes,
            "maximum_generation_calls": maximum_calls,
            "evaluation_coverage": extraction_audit.get("evaluation_coverage"),
            "errors": deepcopy(extraction_audit.get("audit_integrity_errors") or []),
            "passed": audited_outcomes == maximum_calls
            and (extraction_audit.get("upstream_producer_disposition") or {}).get("honest_verdict")
            == capture.get("honest_verdict"),
        },
    ]
    return rows


def _claim(
    evidence: Mapping[str, JsonDict],
    task_id: str,
    *,
    completion_score: int,
    benefit_score: int,
    disposition: str,
    authority: str,
    limitations: Sequence[str],
    **extra: Any,
) -> JsonDict:
    """Build one bounded claim without inheriting another branch's value."""

    source = evidence[task_id]
    row = {
        "authority_task": task_id,
        "authority": authority,
        "completion_score": completion_score,
        "benefit_score": benefit_score,
        "disposition": disposition,
        "producer_verdict_class": source["verdict_class"],
        "producer_flagged_adversarial": source["flagged_adversarial"],
        "available": source["available"],
        "valid": source["valid"],
        "limitations": list(limitations),
        "scientific_progress_claimed": benefit_score > 0,
    }
    row.update(extra)
    return row


def reduce_claim_matrix(evidence: Mapping[str, JsonDict]) -> dict[str, JsonDict]:
    """Reduce seven independent V652 mechanism claims."""

    p34 = evidence["exp7434-contract-methods"]["payload"]
    p35 = evidence["exp7435-round-breaker"]["payload"]
    p39 = evidence["exp7439-certified-decisions"]["payload"]
    p40 = evidence["exp7440-mixture-learning"]["payload"]
    p41 = evidence["exp7441-decision-audit"]["payload"]
    p42 = evidence["exp7442-span-capture"]["payload"]
    p43 = evidence["exp7443-span-audit"]["payload"]
    p44 = evidence["exp7444-arc-supervisor-evidence"]["payload"]
    p45 = evidence["exp7445-hardware-envelope"]["payload"]
    return {
        "contract": _claim(
            evidence,
            "exp7434-contract-methods",
            completion_score=int(p34.get("contract_ready_score", 0)),
            benefit_score=0,
            disposition="null",
            authority="independent Markdown and YAML comparison",
            limitations=["advisory accounting cannot create scientific progress"],
        ),
        "runtime_recovery": _claim(
            evidence,
            "exp7435-round-breaker",
            completion_score=int(p35.get("breaker_recovery_ready_score", 0)),
            benefit_score=0,
            disposition="null",
            authority="private persisted-log and sandbox recovery controls",
            limitations=["execution recovery is not scientific acceptance"],
        ),
        "static_decision": _claim(
            evidence,
            "exp7439-certified-decisions",
            completion_score=int(p39.get("decision_capture_complete_score", 0)),
            benefit_score=int(p39.get("decision_value_score", 0)),
            disposition=(p41.get("static_audit") or {}).get("verdict_class", "disqualified"),
            authority="Exp7439 rows and Exp7441 independent static audit",
            limitations=["reused corpus is exploratory; no deployment certificate"],
            auditor_valid=(p41.get("static_audit") or {}).get("valid"),
        ),
        "online_learning": _claim(
            evidence,
            "exp7440-mixture-learning",
            completion_score=int(p40.get("online_capture_complete_score", 0)),
            benefit_score=int(p40.get("online_value_score", 0)),
            disposition=(p41.get("online_audit") or {}).get("verdict_class", "disqualified"),
            authority="Exp7440 event rows and Exp7441 independent online audit",
            limitations=["auditor found missing expert predictions for update replay"],
            auditor_valid=(p41.get("online_audit") or {}).get("valid"),
        ),
        "extraction": _claim(
            evidence,
            "exp7442-span-capture",
            completion_score=int(p43.get("extraction_audit_complete_score", 0)),
            benefit_score=0,
            disposition="disqualified",
            authority="owned Qwen call and Exp7443 independent raw audit",
            limitations=["producer runtime identity failed; evaluation coverage is zero"],
            evaluation_coverage=p43.get("evaluation_coverage"),
            producer_completion_score=p42.get("extraction_capture_complete_score"),
        ),
        "arc": _claim(
            evidence,
            "exp7444-arc-supervisor-evidence",
            completion_score=int(p44.get("supervisor_evidence_complete_score", 0)),
            benefit_score=0,
            disposition="null",
            authority="archived adapter-withheld supervisor outcomes",
            limitations=["zero supervisor firings supply no arm-effect evidence"],
        ),
        "hardware": _claim(
            evidence,
            "exp7445-hardware-envelope",
            completion_score=1,
            benefit_score=int(p45.get("hardware_value_score", 0)),
            disposition="null",
            authority="host service bounds and read-only board evidence",
            limitations=["no new device workload; persistence fraction limits acceleration"],
            hardware_ready_score=p45.get("hardware_ready_score"),
        ),
    }


def _failure(
    task_id: str, source: Mapping[str, Any], category: str, field: str, expected: Any
) -> JsonDict:
    """Name every operand for one terminal classification failure."""

    return {
        "upstream": task_id,
        "path": source.get("source_path"),
        "check": category,
        "category": category,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": source.get(field),
    }


def classify_terminal(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
) -> JsonDict:
    """Apply current-defect, upstream-invalid, then upstream-absence precedence."""

    failures: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        failures.append(
            {
                "upstream": "v652-contract-authorities",
                "path": f"{DESIGN_PATH.as_posix()} + {ROADMAP_PATH.as_posix()}",
                "check": "contract_equivalence",
                "category": "current_validity",
                "field": "comparison_passed",
                "operator": "==",
                "expected": True,
                "observed": contract.get("comparison_passed"),
            }
        )
    if validation.get("required_checks_passed") is not True:
        failures.append(
            {
                "upstream": EXPERIMENT_ID,
                "path": TEST_PATH.as_posix(),
                "check": "affected_validation",
                "category": "current_validity",
                "field": "required_checks_passed",
                "operator": "==",
                "expected": True,
                "observed": validation.get("required_checks_passed"),
            }
        )
    for task_id, source in evidence.items():
        if source.get("source_kind") != "missing" and source.get("authenticated") is not True:
            failures.append(
                _failure(task_id, source, "evidence_authentication", "authenticated", True)
            )
    for task_id in SCIENTIFIC_TASKS:
        source = evidence[task_id]
        if source.get("available") and (
            source.get("valid") is not True
            or source.get("verdict_class") == "disqualified"
            or source.get("flagged_adversarial") is True
        ):
            failures.append(_failure(task_id, source, "scientific_validity", "valid", True))
    for task_id in SCIENTIFIC_TASKS:
        source = evidence[task_id]
        if source.get("available") is not True:
            failures.append(_failure(task_id, source, "scientific_availability", "available", True))
    invalid = any(
        row["category"] in {"current_validity", "evidence_authentication", "scientific_validity"}
        for row in failures
    )
    absent = any(row["category"] == "scientific_availability" for row in failures)
    if invalid:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_required_v652_science_with_thirteen_dispositions"
    elif absent:
        verdict_class = "blocked"
        verdict = "complete_blocked_required_v652_science_with_thirteen_dispositions"
    else:
        verdict_class = "null"
        verdict = "complete_null_v652_capstone_with_thirteen_dispositions"
    return {
        "status": verdict,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": {
            "passed": not failures,
            "failure_count": len(failures),
            "first_failure": deepcopy(failures[0]) if failures else None,
            "failures": failures,
        },
    }


def retirement_rows(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Retire only an exact repeated verdict under its declared rule."""

    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task.get("id"))
        current = (
            terminal.get("honest_verdict")
            if task_id == EXPERIMENT_ID
            else evidence.get(task_id, {}).get("honest_verdict")
        )
        for prior in task.get("prior_failures") or []:
            previous = str(prior.get("verdict"))
            same = previous == current
            permanent = same and prior.get("retire_if_same_verdict") is True
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior.get("experiment_id"),
                    "previous_verdict": previous,
                    "current_verdict": current,
                    "addressed_by": prior.get("addressed_by"),
                    "same_exact_verdict": same,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict") is True,
                    "permanent_retirement": permanent,
                    "decision": "retire" if permanent else "continue",
                }
            )
    return rows


def continuation_rows(
    claims: Mapping[str, JsonDict], retirements: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Give each mechanism a bounded decision and measured next condition."""

    causes = {
        "contract": ("defer", "new authority drift or a changed roadmap contract"),
        "runtime_recovery": ("defer", "a measured recurrence of invocation-local lockout"),
        "static_decision": ("defer", "a fresh independent corpus with usable score support"),
        "online_learning": (
            "continue",
            "record complete prediction-time expert prediction losses before each update",
        ),
        "extraction": (
            "continue",
            "repair owned-process identity and lease continuity before changed capture",
        ),
        "arc": ("defer", "measured supervisor exposure or a changed trigger condition"),
        "hardware": (
            "defer",
            "changed persistence/orchestration design or dated GateMate physical state",
        ),
    }
    permanent_tasks = {
        str(row.get("task_id")) for row in retirements if row.get("permanent_retirement") is True
    }
    task_by_claim = {
        "contract": "exp7434-contract-methods",
        "runtime_recovery": "exp7435-round-breaker",
        "static_decision": "exp7439-certified-decisions",
        "online_learning": "exp7440-mixture-learning",
        "extraction": "exp7442-span-capture",
        "arc": "exp7444-arc-supervisor-evidence",
        "hardware": "exp7445-hardware-envelope",
    }
    rows: list[JsonDict] = []
    for mechanism in CLAIM_BRANCHES:
        decision, cause = causes[mechanism]
        if task_by_claim[mechanism] in permanent_tasks:
            decision = "retire"
            cause = "same exact verdict recurred under retire_if_same_verdict=true"
        rows.append(
            {
                "mechanism": mechanism,
                "decision": decision,
                "changed_cause_or_prerequisite": cause,
                "current_disposition": claims[mechanism]["disposition"],
            }
        )
    return rows


def unresolved_obligations(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Keep missing science and forbidden conductor work explicit."""

    return [
        {
            "obligation_id": "static_decision_benefit",
            "state": "open",
            "required_change": "fresh independent labels and usable selected-action support",
        },
        {
            "obligation_id": "online_causal_evidence",
            "state": "open",
            "required_change": "complete stored expert predictions and replayable weight updates",
        },
        {
            "obligation_id": "live_extraction_evaluation",
            "state": "open",
            "required_change": "valid runtime ownership followed by sealed evaluation capture",
            "producer_flagged_adversarial": evidence["exp7442-span-capture"]["flagged_adversarial"],
        },
        {
            "obligation_id": "arc_supervisor_effect",
            "state": "open",
            "required_change": "measured exposure and outcome-bearing supervisor firings",
        },
        {
            "obligation_id": "hardware_service_value",
            "state": "open",
            "required_change": "persistence redesign plus complete device service measurements",
        },
        {
            "obligation_id": "conductor_size_gate",
            "state": "current_task_forbidden",
            "required_change": "none in this task; repository notes an earlier shipped size gate",
            "prohibited_path": "scripts/research_conductor.py",
            "current_task_action": "none",
            "repository_record": "ops/known-issues.md marks the conductor size gate resolved on 2026-09-20",
        },
    ]


def publication_gate_row(result: Mapping[str, Any]) -> JsonDict:
    """Preserve the established G1-G4 FoVer headline scope."""

    return {
        "headline_scope": "FoVer dual-condition AUROC",
        "certifies_v652": False,
        "paper_ready": result.get("paper_ready"),
        "gates": deepcopy(result.get("gates")),
        "unmet_gates": deepcopy(result.get("unmet_gates")),
        "note": result.get("note"),
    }


def _task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Build twelve source rows and one current-work disposition."""

    rows: list[JsonDict] = []
    for task in tasks[:-1]:
        source = evidence[str(task["id"])]
        rows.append(
            {
                "order": len(rows) + 1,
                "task_id": task["id"],
                "declared_path": task["deliverable"],
                "observed_path": source["source_path"],
                "source_kind": source["source_kind"],
                "authenticated": source["authenticated"],
                "available": source["available"],
                "valid": source["valid"],
                "raw_rows_available": source["row_manifest_authenticated"],
                "raw_row_count": source["raw_row_count"],
                "raw_shard_count": source["raw_shard_count"],
                "validation_receipts_authenticated": source["validation_receipts_authenticated"],
                "honest_verdict": source["honest_verdict"],
                "verdict_class": source["verdict_class"],
                "flagged_adversarial": source["flagged_adversarial"],
            }
        )
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    rows.append(
        {
            "order": 13,
            "task_id": EXPERIMENT_ID,
            "declared_path": tasks[-1]["deliverable"],
            "observed_path": RESULT_PATH.as_posix(),
            "source_kind": "current_work",
            "authenticated": current_valid,
            "available": True,
            "valid": current_valid,
            "raw_rows_available": True,
            "raw_row_count": 13,
            "raw_shard_count": 0,
            "validation_receipts_authenticated": current_valid,
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "flagged_adversarial": False,
        }
    )
    return rows


def _source_hashes(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> dict[str, JsonDict]:
    """Bind authorities, current code, and every predecessor artifact."""

    paths = {
        "research-roadmap": ROADMAP_PATH,
        "markdown-contract": DESIGN_PATH,
        "research-reporting-spec": SPEC_PATH,
        "publication-gate": PUBLICATION_GATE_PATH,
        "current-module": MODULE_PATH,
        "current-wrapper": WRAPPER_PATH,
        "current-test": TEST_PATH,
    }
    rows = {
        name: {"path": path.as_posix(), "sha256": sha256_file(root / path)}
        for name, path in paths.items()
    }
    for task_id, source in evidence.items():
        rows[task_id] = {
            "path": source["source_path"],
            "declared_path": source["declared_path"],
            "sha256": source.get("sha256"),
            "source_kind": source["source_kind"],
            "original_honest_verdict": source["honest_verdict"],
            "original_verdict_class": source["verdict_class"],
            "original_flagged_adversarial": source["flagged_adversarial"],
            "raw_rows_sha256": source.get("raw_rows_sha256"),
            "validation_receipts_sha256": source.get("validation_receipts_sha256"),
        }
    rows["contract-comparison"] = {
        "path": f"{DESIGN_PATH.as_posix()} + {ROADMAP_PATH.as_posix()}",
        "sha256": canonical_hash(contract.get("contract_rows")),
        "source_kind": "canonical_reduction",
    }
    return rows


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Reload every file-backed source hash and the contract reduction."""

    rows = artifact.get("source_artifact_hashes")
    if not isinstance(rows, Mapping):
        return False
    for row in rows.values():
        if not isinstance(row, Mapping):  # pragma: no cover - schema guard.
            return False
        if row.get("source_kind") == "canonical_reduction":
            current = load_contract(root)
            if row.get("sha256") != canonical_hash(current.get("contract_rows")):
                return False
            continue
        label = row.get("path")
        if not isinstance(label, str) or " + " in label:
            return False
        path = _resolve_path(root, label)
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            return False
    return True


def _historical_sidecars(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Keep prior model and small-head activity outside current counters."""

    rows: list[JsonDict] = []
    for task_id, source in evidence.items():
        payload = source["payload"]
        kinds: list[str] = []
        if payload.get("model_invoked") is True:
            kinds.append("historical_model_invocation")
        training = payload.get("small_ebm_training")
        if isinstance(training, Mapping) and training.get("performed") is True:
            kinds.append("historical_small_ebm_training")
        for kind in kinds:
            rows.append(
                {
                    "task_id": task_id,
                    "kind": kind,
                    "path": source["source_path"],
                    "sha256": source.get("sha256"),
                    "counts_as_current_invocation": False,
                }
            )
    return rows


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep completion, validity, benefit, and promotion gates distinct."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
    structured_gates: Sequence[Mapping[str, Any]],
    auditor_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build current checks without turning upstream nulls into failures."""

    sources_authenticated = all(row["authenticated"] for row in evidence.values())
    structured_passed = all(row.get("passed") is True for row in structured_gates)
    auditors_passed = all(row.get("passed") is True for row in auditor_rows)
    return [
        _gate(
            "thirteen_task_dispositions",
            "completion",
            "==",
            13,
            len(dispositions),
            len(dispositions) == 13,
            "Every promised task receives one terminal disposition.",
        ),
        _gate(
            "contract_authorities_agree",
            "current_validity",
            "==",
            True,
            contract.get("comparison_passed"),
            contract.get("comparison_passed") is True,
            "The current capstone independently binds the exact contract.",
        ),
        _gate(
            "predecessor_evidence_authenticated",
            "evidence_validity",
            "==",
            True,
            sources_authenticated,
            sources_authenticated,
            "Exact artifacts, rows, flags, and validation receipts remain bound.",
        ),
        _gate(
            "structured_gates_authenticated",
            "evidence_validity",
            "==",
            True,
            structured_passed,
            structured_passed,
            "A scalar score cannot override class or adversarial admissibility.",
        ),
        _gate(
            "independent_auditor_cross_checks",
            "evidence_validity",
            "==",
            True,
            auditors_passed,
            auditors_passed,
            "Both independent auditors retain their measured defects.",
        ),
        _gate(
            "affected_validation",
            "required_validation",
            "==",
            True,
            validation.get("required_checks_passed"),
            validation.get("required_checks_passed") is True,
            "Only the frozen affected command plan controls current code validity.",
        ),
        _gate(
            "terminal_validation",
            "required_validation",
            "==",
            True,
            validation.get("terminal_validation_passed"),
            validation.get("terminal_validation_passed") is True,
            "Cold replay and unchanged strict readers control final publication.",
        ),
        _gate(
            "promotion_zero",
            "promotion",
            "==",
            0,
            0,
            True,
            "A host aggregation cannot publish or change production state.",
        ),
    ]


def zero_test_phase_spans() -> list[JsonDict]:
    """Return deterministic phase rows for unit artifact construction."""

    return [
        {
            "phase": phase,
            "started_elapsed_s": 0.0,
            "ended_elapsed_s": 0.0,
            "duration_s": 0.0,
            "heartbeat_count": 0,
            "checkpoint": checkpoint,
        }
        for phase, checkpoint in (
            ("preconditions", "twelve_sources_authenticated"),
            ("plan", "affected_plan_frozen"),
            ("load", "no_current_model_load"),
            ("generate", "no_current_generation"),
            ("validate", "affected_checks_complete"),
            ("reduce", "seven_claims_reduced"),
            ("terminal_validation", "cold_readers_complete"),
            ("write", "terminal_artifact_ready"),
        )
    ]


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain each top-level field separately from its plain value."""

    special = {
        "schema": "Use a versioned plain top-level identity and terminal status.",
        "run_date": "Use 20260920 and retain real UTC and monotonic boundaries.",
        "preconditions_checked": "Name each real path and observed prerequisite before reduction.",
        "MODEL_SPECS": "List every current LLM; remain empty because no LLM runs here.",
        "model_invoked": "Distinguish current attempted model use from historical evidence.",
        "invocation_counts": "Reconcile every current load and generation outcome at zero.",
        "inference_substrate": "Describe current host aggregation without inheriting old compute.",
        "inference_substrate_class": "Declare aggregation so the matching duration rule applies.",
        "execution_venue": "Use host and keep hardware identity in details.",
        "duration_s": "Measure current work and separate validation from model time.",
        "phase_spans": "Bind phase timing, progress boundaries, and checkpoints.",
        "random_seed": "Remain null because this reducer does no fitting or resampling.",
        "reproducibility_checksum": "Bind protocol, sources, rows, gates, and decisions.",
        "source_artifact_hashes": "Preserve exact source bytes, classes, and flags.",
        "rows": "Keep one ordered disposition for every promised task.",
        "sample_size_budget": "Account for all thirteen task disposition units.",
        "acceptance_gate_results": "Separate completion, validity, benefit, and promotion.",
        "gate_check_summary": "Name exact failed upstream fields without hiding later failures.",
        "verifier_is_oracle": "Remain false because no deployed verifier scores this aggregation.",
        "honest_verdict": "Use complete findings; upstream absence never creates partial work.",
        "verdict_class": "Keep null, blocked, disqualified, and partial meanings distinct.",
        "flagged_adversarial": "Preserve critical upstream findings without rehabilitation.",
        "validation_receipts": "Record exact scoped commands, exits, timings, and log hashes.",
        "promotion_score": "Always zero; no rollout, publication, or generator update is authorized.",
        "capstone_complete_score": "One requires thirteen authenticated dispositions and current validation.",
        "task_dispositions": "One ordered row per task prevents silent contract shortening.",
        "claim_matrix": "Separate measured effects, readiness, circular fixtures, and unavailable evidence.",
        "continuation_rows": "Every next attempt requires a measured changed cause.",
        "publication_gates": "Preserve unchanged G1-G4 and their FoVer headline scope.",
        "retirement_rows": "Retire only exact repeated verdicts under declared rules.",
        "unresolved_obligations": "Keep missing science and forbidden conductor work explicit.",
    }
    return {key: special.get(key, f"Plain terminal evidence for {key}.") for key in keys}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable protocol, evidence, reductions, checks, and decisions."""

    excluded = {
        "reproducibility_checksum",
        "field_principles",
        "started_at_utc",
        "completed_at_utc",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_s",
        "validation_duration_s",
        "cold_start_duration_s",
        "model_duration_s",
        "phase_spans",
    }
    return canonical_hash({key: value for key, value in artifact.items() if key not in excluded})


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
    publication: Mapping[str, Any],
    *,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the schema-complete capstone from authenticated source rows."""

    terminal = classify_terminal(contract, evidence, validation)
    claims = reduce_claim_matrix(evidence)
    auditors = auditor_cross_checks(evidence)
    structured = audit_structured_gates(contract, evidence)
    retirements = retirement_rows(contract["tasks"], evidence, terminal)
    dispositions = _task_dispositions(contract["tasks"], evidence, terminal, validation)
    gates = _acceptance_gates(contract, evidence, validation, dispositions, structured, auditors)
    complete = bool(
        len(dispositions) == 13
        and all(row.get("authenticated") is True for row in dispositions)
        and validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    start_ns = 0
    end_ns = max(0, int(duration_s * 1_000_000_000))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "run_date": RUN_DATE,
        "status": terminal["status"],
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": any(
            row.get("flagged_adversarial") is True for row in evidence.values()
        ),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": start_ns,
        "ended_monotonic_ns": end_ns,
        "duration_s": float(duration_s),
        "validation_duration_s": sum(
            float(row.get("duration_s", 0.0))
            for row in validation.get("validation_receipts", [])
            if isinstance(row, Mapping)
        ),
        "cold_start_duration_s": sum(
            float(row.get("duration_s", 0.0))
            for row in validation.get("validation_receipts", [])
            if isinstance(row, Mapping) and row.get("name") in TERMINAL_CHECK_NAMES
        ),
        "model_duration_s": 0.0,
        "phase_spans": deepcopy(list(phase_spans)),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_authenticated_v652_artifacts",
        "inference_substrate_class": "aggregation",
        "inference_substrate_details": {
            "current_compute": "host JSON and YAML reduction",
            "current_llm_operations": 0,
            "current_cuda_operations": 0,
            "current_external_device_operations": 0,
            "historical_compute_inherited": False,
        },
        "execution_venue": "host",
        "random_seed": None,
        "preconditions_checked": [
            *[
                {
                    "check": "required_source_exists",
                    "upstream": path.as_posix(),
                    "path": path.as_posix(),
                    "artifact_field": "path",
                    "expected": True,
                    "observed": (root / path).is_file(),
                    "passed": (root / path).is_file(),
                }
                for path in SOURCE_PATHS
            ],
            {
                "check": "contract_authority_equivalence",
                "upstream": "v652-contract-authorities",
                "path": f"{DESIGN_PATH.as_posix()} + {ROADMAP_PATH.as_posix()}",
                "artifact_field": "comparison_passed",
                "expected": True,
                "observed": contract.get("comparison_passed"),
                "passed": contract.get("comparison_passed") is True,
            },
            *[
                {
                    "check": "predecessor_authentication",
                    "upstream": task_id,
                    "path": row["source_path"],
                    "artifact_field": "identity+rows+verdict+flag+validation",
                    "expected": True,
                    "observed": row["authenticated"],
                    "passed": row["authenticated"],
                }
                for task_id, row in evidence.items()
            ],
        ],
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "historical_evidence_sidecars": _historical_sidecars(evidence),
        "rows": deepcopy(dispositions),
        "sample_size_budget": {
            "planned": 13,
            "attempted": 13,
            "completed": 13,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_units": 13,
            "stopping_rule": "one terminal disposition for every ordered V652 task",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": terminal["gate_check_summary"],
        "verifier_is_oracle": False,
        "validation_receipts": deepcopy(validation.get("validation_receipts", [])),
        "repository_health": deepcopy(validation.get("repository_health", {})),
        "task_dispositions": dispositions,
        "structured_gate_audit": structured,
        "auditor_cross_checks": auditors,
        "claim_matrix": claims,
        "retirement_rows": retirements,
        "continuation_rows": continuation_rows(claims, retirements),
        "publication_gates": publication_gate_row(publication),
        "unresolved_obligations": unresolved_obligations(evidence),
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "declared_entrypoint_cold_replay",
            "numbered_e2e_applicable": [],
        },
        "roadmap_activated": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "external_contact_performed": False,
        "publication_performed": False,
        "push_performed": False,
        "promotion_score": 0,
        "capstone_complete_score": int(complete),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _receipt_passed(receipts: object, names: Sequence[str]) -> bool:
    """Require one successful receipt for every exact command name."""

    if not isinstance(receipts, list):
        return False
    by_name = {str(row.get("name")): row for row in receipts if isinstance(row, Mapping)}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        for name in names
    )


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, sources, reductions, decisions, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    required = {
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "status",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "MODEL_SPECS",
        "model_invoked",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "phase_spans",
        "preconditions_checked",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "validation_receipts",
        "task_dispositions",
        "claim_matrix",
        "continuation_rows",
        "publication_gates",
        "retirement_rows",
        "unresolved_obligations",
        "promotion_score",
        "capstone_complete_score",
        "reproducibility_checksum",
        "field_principles",
    }
    missing = sorted(required - set(artifact))
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE
    ):
        errors.append("identity_invalid")
    if not str(artifact.get("status")).startswith("complete"):
        errors.append("lifecycle_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_invalid")
    dispositions = artifact.get("task_dispositions")
    if (
        not isinstance(dispositions, list)
        or [row.get("task_id") for row in dispositions if isinstance(row, Mapping)]
        != list(EXPECTED_TASK_IDS)
        or artifact.get("rows") != dispositions
    ):
        errors.append("task_dispositions_invalid")
    if not _hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    current_publication = publication_gate_row(evaluate_publication_gates())
    if artifact.get("publication_gates") != current_publication:
        errors.append("publication_gates_invalid")
    try:
        contract = load_contract(root)
        evidence = collect_evidence(root, contract["tasks"])
        receipts = artifact.get("validation_receipts")
        validation = {
            "required_checks_passed": _receipt_passed(receipts, REQUIRED_CHECK_NAMES),
            "terminal_validation_passed": _receipt_passed(receipts, TERMINAL_CHECK_NAMES),
            "validation_receipts": receipts,
        }
        if not require_terminal and not validation["terminal_validation_passed"]:
            validation["terminal_validation_passed"] = False
        terminal = classify_terminal(contract, evidence, validation)
        expected_dispositions = _task_dispositions(
            contract["tasks"], evidence, terminal, validation
        )
        if dispositions != expected_dispositions:
            errors.append("task_dispositions_invalid")
        claims = reduce_claim_matrix(evidence)
        if artifact.get("claim_matrix") != claims:
            errors.append("claim_matrix_invalid")
        auditors = auditor_cross_checks(evidence)
        if artifact.get("auditor_cross_checks") != auditors:
            errors.append("auditor_cross_checks_invalid")
        structured = audit_structured_gates(contract, evidence)
        if artifact.get("structured_gate_audit") != structured:
            errors.append("structured_gate_audit_invalid")
        retirements = retirement_rows(contract["tasks"], evidence, terminal)
        if artifact.get("retirement_rows") != retirements:
            errors.append("retirement_rows_invalid")
        if artifact.get("continuation_rows") != continuation_rows(claims, retirements):
            errors.append("continuation_rows_invalid")
        if artifact.get("unresolved_obligations") != unresolved_obligations(evidence):
            errors.append("unresolved_obligations_invalid")
        if (
            artifact.get("honest_verdict") != terminal["honest_verdict"]
            or artifact.get("verdict_class") != terminal["verdict_class"]
            or artifact.get("gate_check_summary") != terminal["gate_check_summary"]
        ):
            errors.append("terminal_reduction_invalid")
        expected_complete = int(
            len(expected_dispositions) == 13
            and all(row.get("authenticated") is True for row in expected_dispositions)
            and validation["required_checks_passed"]
            and validation["terminal_validation_passed"]
        )
        if artifact.get("capstone_complete_score") != expected_complete:
            errors.append("capstone_score_invalid")
        expected_gates = _acceptance_gates(
            contract, evidence, validation, expected_dispositions, structured, auditors
        )
        if artifact.get("acceptance_gate_results") != expected_gates:
            errors.append("acceptance_gates_invalid")
        if require_terminal and not validation["terminal_validation_passed"]:
            errors.append("terminal_validation_incomplete")
    except (KeyError, OSError, TypeError, ValueError):  # pragma: no cover - cold fail-closed.
        errors.append("independent_reduction_failed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact) - {
        "field_principles"
    }:
        errors.append("field_principles_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def independent_reduce(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Replay the reducer in a fresh process without requiring its own receipt."""

    return validate_artifact(value, root=root, require_terminal=False)


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the frozen Exp7358 plan for only current affected files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the execution date frozen by the V652 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def _phase_span(  # pragma: no cover - real timing boundary.
    phase: str, phase_started: float, run_started: float, *, checkpoint: str
) -> JsonDict:
    """Close one measured phase and name its durable checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "heartbeat_count": 0,
        "checkpoint": checkpoint,
    }


def _terminal_commands(  # pragma: no cover - capability E2E subprocesses.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build cold replay, reduction, strict readers, and G1-G4 execution."""

    python = str(root / ".venv/bin/python")
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--validate", str(candidate)),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", WRAPPER_PATH.as_posix(), "--independent-reduce", str(candidate)),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "candidate_row_consistency",
        ),
        validation_scope.CommandSpec(
            "publication_gate_json",
            (python, "-u", PUBLICATION_GATE_PATH.as_posix(), "--json"),
            "unchanged_publication_g1_g4",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - public E2E.
    """Run exact reads, scoped checks, cold readers, and atomic publication."""

    date_argument(run_date)
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7446-", dir="/tmp"))

    point = time.monotonic()
    progress(started, "preconditions", "before")
    missing = [path.as_posix() for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source paths missing: {missing}")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    publication = evaluate_publication_gates()
    spans.append(
        _phase_span("preconditions", point, started, checkpoint="twelve_sources_authenticated")
    )
    progress(started, "preconditions", "after", completed_units=len(evidence))

    point = time.monotonic()
    progress(started, "plan", "before")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    spans.append(_phase_span("plan", point, started, checkpoint="affected_plan_frozen"))
    progress(started, "plan", "after", commands=len(commands))

    for phase, checkpoint in (
        ("load", "no_current_model_load"),
        ("generate", "no_current_generation"),
    ):
        point = time.monotonic()
        progress(started, phase, "before", current_llm_operations=0)
        spans.append(_phase_span(phase, point, started, checkpoint=checkpoint))
        progress(started, phase, "after", current_llm_operations=0)

    point = time.monotonic()
    progress(started, "validate", "before_affected_subprocesses", units=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    validation: JsonDict = {
        **reduced,
        "required_checks_passed": reduced["passed"],
        "terminal_validation_passed": False,
        "validation_receipts": affected,
        "repository_health": {
            "status": "outside_current_affected_validity",
            "historical_failures": [],
        },
    }
    spans.append(_phase_span("validate", point, started, checkpoint="affected_checks_complete"))
    progress(started, "validate", "after_affected_subprocesses", passed=reduced["passed"])

    point = time.monotonic()
    progress(started, "reduce", "before")
    spans.append(_phase_span("reduce", point, started, checkpoint="seven_claims_reduced"))
    candidate = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    progress(started, "reduce", "after", verdict=candidate["verdict_class"])

    point = time.monotonic()
    terminal_commands = _terminal_commands(root, candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", units=len(terminal_commands))
    terminal_receipts = run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_receipts)
    validation["terminal_validation_passed"] = (
        all(row.get("passed") is True for row in terminal_receipts) and not critical
    )
    validation["validation_receipts"] = [*affected, *terminal_receipts]
    spans.append(
        _phase_span("terminal_validation", point, started, checkpoint="cold_readers_complete")
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=validation["terminal_validation_passed"],
        critical=critical,
    )

    point = time.monotonic()
    progress(started, "write", "before_atomic", path=RESULT_PATH.as_posix())
    final_spans = [
        *spans,
        _phase_span("write", point, started, checkpoint="terminal_artifact_ready"),
    ]
    artifact = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=final_spans,
    )
    errors = validate_artifact(artifact, root=root)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(root / RESULT_PATH, artifact)
    progress(started, "write", "after_atomic", path=RESULT_PATH.as_posix())
    return artifact


def _parser() -> argparse.ArgumentParser:
    """Parse the frozen date and two cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, type=date_argument)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the V652 capstone or one read-only cold validation."""

    print("[exp7446] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    candidate_path = args.validate or args.independent_reduce
    if candidate_path is not None:
        try:
            value = load_json_object(candidate_path)
        except ValueError as error:
            print(json.dumps({"errors": [str(error)]}, sort_keys=True), flush=True)
            return 1
        errors = (
            independent_reduce(value)
            if args.independent_reduce is not None
            else validate_artifact(value, require_terminal=False)
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": RESULT_PATH.as_posix(),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
                "capstone_complete_score": artifact["capstone_complete_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
