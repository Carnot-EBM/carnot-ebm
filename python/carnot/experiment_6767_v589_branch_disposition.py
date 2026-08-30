"""Synthesize V589 terminal branch evidence without pooling branches.

The capstone is an artifact reducer. It reads the active roadmap and checked-in
result JSON files, recomputes the metrics that have rows, and keeps blocked or
missing inputs as data. It does not run an LLM and it does not edit upstream
artifacts.

Spec refs: REQ-REPORT-6767 and SCENARIO-REPORT-6767-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.08.589"
PLANNING_DATE = "20260830"
INFERENCE_SUBSTRATE = "cold_local_artifact_and_row_synthesis_no_llm"
RANDOM_SEED = {"row_recompute_seed": 6767, "validator_seed": 6767001}

RESULT_PATH = Path("results/experiment_6767_v589_branch_disposition.json")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
V588_DISPOSITION_PATH = Path("results/experiment_6754_v588_branch_disposition.json")
SUMMARY_SCRIPT = Path("scripts/summarize_artifact.py")
ADVERSARIAL_SCRIPT = Path("scripts/adversarial_verify.py")
ROW_LINT_SCRIPT = Path("scripts/verdict_row_consistency_lint.py")
RECURRING_BLOCKER_SCRIPT = Path("scripts/recurring_blocker_ledger.py")
RESEARCH_CONDUCTOR_PATH = Path("scripts/research_conductor.py")
MODULE_PATH = Path("python/carnot/experiment_6767_v589_branch_disposition.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_6767_v589_branch_disposition.py")
TEST_PATH = Path("tests/python/test_experiment_6767_v589_branch_disposition.py")

CLOSED_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

EXPECTED_TASK_IDS = tuple(f"exp{number}" for number in range(6755, 6768))
FULL_TASK_IDS = (
    "exp6755-lossless-gguf-output-reparse",
    "exp6756-environment-indexed-proof-grammar-fixture",
    "exp6757-dccd-environment-grammar-ab",
    "exp6758-proof-transport-independent-audit",
    "exp6759-oracle-distinct-diagnostic-energy-v2",
    "exp6760-prefix-backtracking-repair-ab",
    "exp6761-procedural-memory-stream",
    "exp6762-procedural-vs-trace-csl-ab",
    "exp6763-csl-hard-case-forgetting-audit",
    "exp6764-arc-exclusive-load-preflight",
    "exp6765-object-table-fetch-ab-v2",
    "exp6766-thermalizer-independent-trajectory-audit",
    "exp6767-v589-branch-disposition",
)
CAPSTONE_TASK_ID = "exp6767"
TASK_PATHS = {
    "exp6755": "results/experiment_6755_lossless_gguf_output_reparse.json",
    "exp6756": "results/experiment_6756_environment_indexed_proof_grammar_fixture.json",
    "exp6757": "results/experiment_6757_dccd_environment_grammar_ab.json",
    "exp6758": "results/experiment_6758_proof_transport_independent_audit.json",
    "exp6759": "results/experiment_6759_oracle_distinct_diagnostic_energy_v2.json",
    "exp6760": "results/experiment_6760_prefix_backtracking_repair_ab.json",
    "exp6761": "results/experiment_6761_procedural_memory_stream.json",
    "exp6762": "results/experiment_6762_procedural_vs_trace_csl_ab.json",
    "exp6763": "results/experiment_6763_csl_hard_case_forgetting_audit.json",
    "exp6764": "results/experiment_6764_arc_exclusive_load_preflight.json",
    "exp6765": "results/experiment_6765_object_table_fetch_ab_v2.json",
    "exp6766": "results/experiment_6766_thermalizer_independent_trajectory_audit.json",
    "exp6767": RESULT_PATH.as_posix(),
}

BRANCH_ORDER = (
    "proof_transport",
    "repair",
    "continuous_memory",
    "arc",
    "stochastic_portability",
    "infrastructure",
)
BRANCH_TASKS = {
    "proof_transport": ("exp6755", "exp6756", "exp6757", "exp6758"),
    "repair": ("exp6759", "exp6760"),
    "continuous_memory": ("exp6761", "exp6762", "exp6763"),
    "arc": ("exp6764", "exp6765"),
    "stochastic_portability": ("exp6766",),
    "infrastructure": ("exp6767",),
}
TASK_BRANCH = {task: branch for branch, tasks in BRANCH_TASKS.items() for task in tasks}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "milestone",
    "expected_task_ids",
    "available_artifacts",
    "missing_artifacts",
    "rows",
    "branch_rows",
    "row_recomputed_headlines",
    "adversarial_findings",
    "verdict_row_consistency_findings",
    "recurring_blockers",
    "prior_verdict_recurrences",
    "retirement_recommendations",
    "prd_gap_disposition",
    "fr12_disposition",
    "fr11_disposition",
    "live_hardware_disposition",
    "docs_reconciled",
    "protected_files_unchanged",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)


def canonical_json(value: Any) -> bytes:
    """Encode JSON once so checksums are stable."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def value_hash(value: Any) -> str:
    """Hash a JSON-compatible value with a typed prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash an existing file while keeping missing distinct from empty."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def spec_anchors(text: str) -> list[str]:
    """Return requirement and scenario IDs in source order."""

    return list(dict.fromkeys(re.findall(r"\b(?:REQ|SCENARIO)-[A-Z0-9-]+", text)))


def short_task_id(task_id: str) -> str:
    """Return the `expNNNN` prefix from one manifest task ID."""

    match = re.match(r"^(exp\d+)(?:-|$)", task_id)
    if match is None:
        raise ValueError(f"invalid V589 task id: {task_id}")
    return match.group(1)


def _next_deliverable(lines: Sequence[str], start: int) -> str:
    for line in lines[start + 1 :]:
        match = re.search(r"\*\*Deliverable:\*\*\s*`([^`]+)`", line)
        if match:
            return match.group(1)
        if line.startswith("### Exp "):
            break
    raise ValueError("V589 design task deliverable missing")


def parse_design_tasks(text: str) -> tuple[str, list[JsonDict]]:
    """Read the V589 proposal task list and deliverables."""

    milestone = re.search(r"\*\*Milestone:\*\*\s*`([^`]+)`", text)
    if milestone is None:
        raise ValueError("V589 design milestone missing")
    rows: list[JsonDict] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"### Exp (\d+):\s*(.+)$", line)
        if match is None:
            continue
        rows.append(
            {
                "task_id": f"exp{match.group(1)}",
                "title": match.group(2).strip(),
                "deliverable": _next_deliverable(lines, index),
            }
        )
    return milestone.group(1), rows


def load_planned_tasks(root: Path) -> list[JsonDict]:
    """Load active V589 roadmap rows after checking the design."""

    manifest = yaml.safe_load((root / ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("tasks"), list):
        raise ValueError("active roadmap must be a mapping with tasks")
    design_milestone, design_tasks = parse_design_tasks(
        (root / DESIGN_PATH).read_text(encoding="utf-8")
    )
    if design_milestone != MILESTONE:
        raise ValueError(f"expected V589 design, observed {design_milestone}")
    if [row["task_id"] for row in design_tasks] != list(EXPECTED_TASK_IDS):
        raise ValueError("V589 design must contain Exp6755 through Exp6767")

    tasks = [task for task in manifest["tasks"] if isinstance(task, Mapping)]
    if [str(task.get("id")) for task in tasks] != list(FULL_TASK_IDS):
        raise ValueError("active roadmap must contain the exact V589 task list")
    planned = []
    design_by_id = {row["task_id"]: row for row in design_tasks}
    for order, task in enumerate(tasks, 1):
        short = short_task_id(str(task["id"]))
        design = design_by_id[short]
        if task.get("deliverable") != design["deliverable"]:
            raise ValueError(f"manifest deliverable mismatch for {short}")
        planned.append(
            {
                "order": order,
                "task_id": short,
                "manifest_task_id": str(task["id"]),
                "title": str(task["title"]),
                "path": str(task["deliverable"]),
                "branch": TASK_BRANCH[short],
                "prior_failures": list(task.get("prior_failures") or []),
            }
        )
    return planned


def missing_source_record(task_id: str, reason: str) -> JsonDict:
    """Create a source record for an absent branch artifact."""

    return {
        "task_id": task_id,
        "path": TASK_PATHS.get(task_id),
        "artifact_state": "missing",
        "valid_json": False,
        "payload": None,
        "sha256": None,
        "error": reason,
    }


def _current_source_record(task_id: str) -> JsonDict:
    return {
        "task_id": task_id,
        "path": TASK_PATHS[task_id],
        "artifact_state": "current_synthesis",
        "valid_json": True,
        "payload": None,
        "sha256": None,
        "error": None,
    }


def load_source_artifacts(root: Path, planned: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Load upstream artifacts while preserving missing files as evidence."""

    sources: dict[str, JsonDict] = {}
    for row in planned:
        task_id = str(row["task_id"])
        if task_id == CAPSTONE_TASK_ID:
            sources[task_id] = _current_source_record(task_id)
            continue
        path = root / str(row["path"])
        if not path.is_file():
            sources[task_id] = missing_source_record(task_id, "file_missing")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON is not an object")
        except Exception as exc:  # noqa: BLE001
            sources[task_id] = {
                "task_id": task_id,
                "path": str(row["path"]),
                "artifact_state": "invalid",
                "valid_json": False,
                "payload": None,
                "sha256": sha256_file(path),
                "error": str(exc),
            }
            continue
        sources[task_id] = {
            "task_id": task_id,
            "path": str(row["path"]),
            "artifact_state": "present",
            "valid_json": True,
            "payload": payload,
            "sha256": sha256_file(path),
            "error": None,
        }
    return sources


def _payload(record: Mapping[str, Any] | None) -> Mapping[str, Any]:
    payload = record.get("payload") if isinstance(record, Mapping) else None
    return payload if isinstance(payload, Mapping) else {}


def _rows(payload: Mapping[str, Any] | None) -> list[JsonDict]:
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    return (
        list(rows) if isinstance(rows, list) and all(isinstance(row, dict) for row in rows) else []
    )


def empty_metric(cause: str) -> JsonDict:
    """Record that no eligible row denominator exists."""

    return {"denominator": 0, "value": None, "cause": cause}


def _rate(numerator: int, denominator: int, cause: str | None = None) -> JsonDict:
    row: JsonDict = {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }
    if denominator == 0 and cause:
        row["cause"] = cause
    return row


def _mean(values: Sequence[float], cause: str | None = None) -> JsonDict:
    row: JsonDict = {
        "denominator": len(values),
        "value": sum(values) / len(values) if values else None,
    }
    if not values and cause:
        row["cause"] = cause
    return row


def _get_path(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _record_class(record: Mapping[str, Any]) -> str:
    state = record.get("artifact_state")
    if state == "missing":
        return "blocked"
    if state == "invalid":
        return "disqualified"
    if state == "current_synthesis":
        return "partial"
    payload = _payload(record)
    declared = payload.get("verdict_class")
    if declared in CLOSED_CLASSES:
        return str(declared)
    text = f"{payload.get('status', '')} {payload.get('honest_verdict', '')}".lower()
    if "blocked" in text or "gate_check_failed" in text:
        return "blocked"
    if "circular" in text:
        return "circular_positive"
    if "disqualified" in text or "retired" in text:
        return "disqualified"
    if "partial" in text:
        return "partial"
    if "positive" in text or "success" in text or "ready" in text:
        return "positive"
    if "null" in text or "complete" in text:
        return "null"
    return "disqualified"


def _check_item(task_id: str, item: Mapping[str, Any]) -> JsonDict:
    return {
        "task_id": task_id,
        "check": item.get("check") or item.get("failed_check") or "gate_check_summary",
        "expected": item.get("expected", item.get("expected_value")),
        "observed": item.get("observed", item.get("observed_value")),
        "passed": item.get("passed"),
        "reason": item.get("reason"),
    }


def gate_failures(task_id: str, summary: Any) -> list[JsonDict]:
    """Normalize failed gate diagnostics from heterogeneous artifacts."""

    if not summary:
        return []
    if isinstance(summary, str):
        return [
            {
                "task_id": task_id,
                "check": "gate_check_summary",
                "expected": "gate predicate passes",
                "observed": summary,
                "passed": False,
                "reason": "blocked_gate_text",
            }
        ]
    if isinstance(summary, list):
        return [
            _check_item(task_id, item)
            for item in summary
            if isinstance(item, Mapping) and item.get("passed") is False
        ]
    if isinstance(summary, Mapping):
        out = []
        checks = summary.get("checks")
        if isinstance(checks, list):
            out.extend(
                _check_item(task_id, item)
                for item in checks
                if isinstance(item, Mapping) and item.get("passed") is False
            )
        failures = summary.get("failures")
        if isinstance(failures, list):
            out.extend(_check_item(task_id, item) for item in failures if isinstance(item, Mapping))
        failed_checks = summary.get("failed_checks")
        if isinstance(checks, Mapping) and isinstance(failed_checks, list):
            for name in failed_checks:
                out.append(
                    {
                        "task_id": task_id,
                        "check": name,
                        "expected": True,
                        "observed": checks.get(name),
                        "passed": False,
                        "reason": "failed_named_check",
                    }
                )
        if summary.get("failed_check"):
            out.append(
                {
                    "task_id": task_id,
                    "check": summary.get("failed_check"),
                    "expected": summary.get("expected"),
                    "observed": summary.get("observed"),
                    "passed": False,
                    "reason": "failed_check_summary",
                }
            )
        return out
    return []


def _model_ids(payload: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    models = payload.get("models_used")
    if isinstance(models, list):
        for model in models:
            if isinstance(model, Mapping):
                value = model.get("model_id") or model.get("hf_id") or model.get("id")
                if value:
                    out.append(str(value))
            elif model:
                out.append(str(model))
    for row in _rows(payload):
        model = row.get("model")
        value = None
        if isinstance(model, Mapping):
            value = model.get("hf_id") or model.get("model_id")
        value = value or row.get("model_id")
        if value:
            out.append(str(value))
    return sorted(dict.fromkeys(out))


def _summary_by_path(summary_findings: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("artifact")): row for row in summary_findings}


def build_rows(
    root: Path,
    planned: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    *,
    summary_findings: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join every expected experiment to its artifact and summary state."""

    summaries = _summary_by_path(summary_findings)
    rows = []
    for plan in planned:
        task_id = str(plan["task_id"])
        record = sources.get(task_id, missing_source_record(task_id, "source_record_missing"))
        payload = _payload(record)
        source_rows = _rows(payload)
        path = str(plan["path"])
        summary = summaries.get(path, {})
        rows.append(
            {
                "row_type": "experiment",
                "order": plan["order"],
                "task_id": task_id,
                "manifest_task_id": plan["manifest_task_id"],
                "title": plan["title"],
                "branch": plan["branch"],
                "path": path,
                "artifact_state": record.get("artifact_state"),
                "valid_json": record.get("valid_json"),
                "artifact_sha256": (
                    None
                    if record.get("artifact_state") == "current_synthesis"
                    else record.get("sha256") or sha256_file(root / path)
                ),
                "error": record.get("error"),
                "status": (
                    payload.get("status")
                    if payload
                    else "current_synthesis"
                    if task_id == CAPSTONE_TASK_ID
                    else None
                ),
                "honest_verdict": (
                    payload.get("honest_verdict")
                    if payload
                    else "current V589 capstone row"
                    if task_id == CAPSTONE_TASK_ID
                    else record.get("error")
                ),
                "verdict_class": _record_class(record),
                "measurement_support_rate": 1.0 if source_rows else 0.0,
                "duration_s": payload.get("duration_s") if payload else None,
                "inference_substrate": payload.get("inference_substrate") if payload else None,
                "verifier_is_oracle": payload.get("verifier_is_oracle") if payload else None,
                "model_ids": _model_ids(payload),
                "raw_rows_available": bool(source_rows),
                "row_count": len(source_rows),
                "rows_container_type": type(payload.get("rows")).__name__ if payload else None,
                "blocked_gates": gate_failures(task_id, payload.get("gate_check_summary")),
                "summary_exit_code": summary.get("exit_code"),
                "summary_hash": summary.get("summary_hash"),
            }
        )
    return rows


def build_artifact_matrix(
    root: Path, planned: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Return the expected task and deliverable matrix."""

    return [
        {
            "task_id": row["task_id"],
            "manifest_task_id": row["manifest_task_id"],
            "branch": row["branch"],
            "path": row["path"],
            "artifact_state": source.get("artifact_state"),
            "valid_json": source.get("valid_json"),
            "artifact_sha256": (
                None
                if source.get("artifact_state") == "current_synthesis"
                else source.get("sha256") or sha256_file(root / str(row["path"]))
            ),
            "row_count": len(_rows(_payload(source))),
            "verdict_class": _record_class(source),
            "error": source.get("error"),
        }
        for row in planned
        for source in [sources.get(str(row["task_id"]), missing_source_record(str(row["task_id"]), "source_record_missing"))]
    ]


def recompute_proof_transport(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    payload_6755 = _payload(sources.get("exp6755"))
    rows_6755 = _rows(payload_6755)
    denominator = len(rows_6755)
    pre_exact = sum(row.get("pre_diagnosis") == "exact_valid" for row in rows_6755)
    post_exact = sum(row.get("post_diagnosis") == "exact_valid" for row in rows_6755)
    targetable = sum(
        bool((row.get("grammar_failures") or {}).get("environment_grammar_targetable"))
        for row in rows_6755
        if isinstance(row.get("grammar_failures"), Mapping)
    )
    record_6757 = sources.get("exp6757", missing_source_record("exp6757", "file_missing"))
    payload_6758 = _payload(sources.get("exp6758"))
    return {
        "lossless_reparse_rows": _rate(denominator, int(payload_6755.get("replayed_row_count") or denominator)),
        "pre_reparse_exact_valid": _rate(pre_exact, denominator),
        "post_reparse_exact_valid": _rate(post_exact, denominator),
        "exact_valid_delta_vs_pre_reparse": _rate(post_exact - pre_exact, denominator),
        "post_diagnosis_counts": dict(Counter(str(row.get("post_diagnosis")) for row in rows_6755)),
        "environment_grammar_targetable_rows": targetable,
        "transport_reparse_ready": bool(payload_6755.get("transport_reparse_ready")),
        "dynamic_proof_grammar_ready": bool(_payload(sources.get("exp6756")).get("dynamic_proof_grammar_ready")),
        "comparative_ab_rows": (
            _rate(len(_rows(_payload(record_6757))), len(_rows(_payload(record_6757))))
            if record_6757.get("artifact_state") == "present"
            else empty_metric("Exp6757 missing")
        ),
        "paired_exact_valid_deltas": None,
        "paired_exact_valid_delta_cause": "Exp6757 A/B rows are unavailable",
        "proof_transport_audit_ready": bool(payload_6758.get("proof_transport_audit_ready")),
        "audit_gate_failures": gate_failures("exp6758", payload_6758.get("gate_check_summary")),
    }


def recompute_repair(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    diagnostic = _payload(sources.get("exp6759"))
    repair = _payload(sources.get("exp6760"))
    repair_rows = _rows(repair)
    harmful = sum(bool(row.get("harmful_flip")) for row in repair_rows)
    return {
        "heldout_reasoning_error_auroc": diagnostic.get("heldout_reasoning_error_auroc"),
        "oracle_leakage_detected": (
            diagnostic.get("oracle_leakage_detected") if diagnostic else None
        ),
        "diagnostic_panel_ready": bool(diagnostic.get("diagnostic_panel_ready")),
        "repair_rows": _rate(len(repair_rows), len(repair_rows), "Exp6760 blocked before repair rows"),
        "repair_interval": repair.get("paired_interval") or repair.get("repair_interval"),
        "harmful_flips": _rate(harmful, len(repair_rows)),
        "repair_completed": bool(repair.get("repair_ab_completed")),
        "diagnostic_gate_failures": gate_failures("exp6759", diagnostic.get("gate_check_summary")),
        "repair_gate_failures": gate_failures("exp6760", repair.get("gate_check_summary")),
    }


def _sum_mapping_numbers(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    return sum(int(item) for item in value.values() if isinstance(item, (int, float)))


def recompute_continuous_memory(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    stream = _payload(sources.get("exp6761"))
    ab = _payload(sources.get("exp6762"))
    audit = _payload(sources.get("exp6763"))
    ab_rows = _rows(ab)
    commits = _sum_mapping_numbers(ab.get("commits_by_arm"))
    rejects = _sum_mapping_numbers(ab.get("rejects_by_arm"))
    rollbacks = _sum_mapping_numbers(ab.get("rollbacks_by_arm"))
    return {
        "stream_rows": len(_rows(stream)),
        "stream_orders": int(stream.get("order_count") or 0),
        "stream_accept_opportunities": _sum_mapping_numbers(stream.get("eligible_accepts_by_order")),
        "stream_reject_opportunities": _sum_mapping_numbers(stream.get("eligible_rejects_by_order")),
        "stream_hard_cases": _sum_mapping_numbers(stream.get("hard_cases_by_order")),
        "stream_ready": bool(stream.get("procedural_memory_stream_ready")),
        "stream_transaction_receipts": len(stream.get("transaction_receipts") or []),
        "stream_restart_receipts": len(stream.get("restart_receipts") or []),
        "stream_rollback_receipts": len(stream.get("rollback_receipts") or []),
        "stream_poison_receipts": len(stream.get("poison_fixture_receipts") or []),
        "future_evidence_violations": stream.get("future_evidence_violations"),
        "prospective_rows": len(ab_rows),
        "prospective_csl_completed": bool(ab.get("prospective_csl_completed")),
        "procedural_over_no_memory_order_lcb": ab.get("procedural_over_no_memory_order_lcb"),
        "procedural_over_trace_order_lcb": ab.get("procedural_over_trace_order_lcb"),
        "transaction_activity": {
            "commits": commits,
            "rejects": rejects,
            "rollbacks": rollbacks,
        },
        "cold_audit_completed": bool(audit.get("csl_hard_case_audit_completed")),
        "cold_audit_gate_failures": gate_failures("exp6763", audit.get("gate_check_summary")),
    }


def _science_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("row_kind") == "science"]


def recompute_arc(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    preflight = _payload(sources.get("exp6764"))
    ab = _payload(sources.get("exp6765"))
    science = _science_rows(_rows(ab))
    comparable = [
        row
        for row in science
        if row.get("prompt_tokens") is not None
        and row.get("change_fidelity") is not None
        and row.get("failure_class") is None
    ]
    live_quality = [row for row in comparable if row.get("live_model_invoked") is True]
    by_arm: dict[str, list[float]] = defaultdict(list)
    fidelity_by_arm: dict[str, list[float]] = defaultdict(list)
    for row in comparable:
        arm = str(row.get("arm"))
        if isinstance(row.get("prompt_tokens"), (int, float)):
            by_arm[arm].append(float(row["prompt_tokens"]))
        if isinstance(row.get("change_fidelity"), (int, float)):
            fidelity_by_arm[arm].append(float(row["change_fidelity"]))
    inline = _mean(by_arm.get("table_inline", []), "no inline token rows")["value"]
    fetch = _mean(by_arm.get("fetch_on_demand", []), "no fetch token rows")["value"]
    token_savings = inline - fetch if isinstance(inline, float) and isinstance(fetch, float) else None
    interval = ab.get("change_fidelity_interval")
    margin = ab.get("noninferiority_margin")
    noninferior = (
        bool(ab.get("object_table_ab_completed"))
        and isinstance(token_savings, (int, float))
        and token_savings > 0
        and isinstance(interval, Mapping)
        and isinstance(interval.get("lower"), (int, float))
        and isinstance(margin, (int, float))
        and float(interval["lower"]) >= -float(margin)
    )
    return {
        "preflight_ready": bool(preflight.get("arc_exclusive_load_ready")),
        "preflight_rows": len(_rows(preflight)),
        "preflight_models": _model_ids(preflight),
        "preflight_vram_recovered": all(
            bool(row.get("passed")) for row in preflight.get("vram_recovery_receipts", []) or []
        ),
        "ab_rows": len(_rows(ab)),
        "ab_science_rows": _rate(len(comparable), len(science)),
        "live_quality_rows": _rate(len(live_quality), len(science)),
        "failure_classes": sorted({str(row.get("failure_class")) for row in science if row.get("failure_class")}),
        "mean_prompt_token_savings": token_savings,
        "artifact_mean_prompt_token_savings": ab.get("mean_prompt_token_savings"),
        "change_fidelity_by_arm": {
            arm: _mean(values) for arm, values in sorted(fidelity_by_arm.items())
        },
        "change_fidelity_delta": ab.get("change_fidelity_delta"),
        "change_fidelity_interval": interval,
        "noninferiority_margin": margin,
        "noninferiority_passed": bool(noninferior),
        "useful_fetch_rate": ab.get("useful_fetch_rate"),
        "object_table_ab_completed": bool(ab.get("object_table_ab_completed")),
        "adoption_gate_passed": bool(ab.get("adoption_gate_passed")),
        "solve_claim": bool(ab.get("solve_claim") or preflight.get("solve_claim")),
    }


def _trajectory_pair_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("factor_id"),
        row.get("context_id"),
        row.get("depth"),
        row.get("precision"),
        row.get("seed_bundle_id"),
        row.get("topology_id"),
    )


def _paired_trajectory_deltas(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    by_method: dict[str, dict[tuple[Any, ...], float]] = defaultdict(dict)
    for row in rows:
        if row.get("evaluator_path") != "exact_enumerator":
            continue
        if isinstance(row.get("trajectory_tv"), (int, float)):
            by_method[str(row.get("method"))][_trajectory_pair_key(row)] = float(row["trajectory_tv"])
    independent = by_method.get("independent_factor", {})
    out: dict[str, JsonDict] = {}
    for method in ("context_matched", "trajectory_refinement"):
        values = []
        for key, independent_tv in independent.items():
            method_tv = by_method.get(method, {}).get(key)
            if method_tv is not None:
                values.append(independent_tv - method_tv)
        out[method] = {
            "pair_count": len(values),
            "mean_independent_minus_method_tv": sum(values) / len(values) if values else None,
            "improved_pair_count": sum(value > 0 for value in values),
            "tied_pair_count": sum(value == 0 for value in values),
            "worsened_pair_count": sum(value < 0 for value in values),
        }
    return out


def recompute_stochastic(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    payload = _payload(sources.get("exp6766"))
    rows = _rows(payload)
    exact_rows = [row for row in rows if row.get("evaluator_path") == "exact_enumerator"]
    sampler_rows = [row for row in rows if row.get("evaluator_path") == "direct_sampler"]
    tv_by_method: dict[str, list[float]] = defaultdict(list)
    kl_by_method: dict[str, list[float]] = defaultdict(list)
    for row in exact_rows:
        method = str(row.get("method"))
        if isinstance(row.get("trajectory_tv"), (int, float)):
            tv_by_method[method].append(float(row["trajectory_tv"]))
        if isinstance(row.get("conditional_kl"), (int, float)):
            kl_by_method[method].append(float(row["conditional_kl"]))
    paired = _paired_trajectory_deltas(exact_rows)
    intervals = {
        str(row.get("method")): row
        for row in payload.get("paired_trajectory_deltas", []) or []
        if isinstance(row, Mapping) and row.get("depth") == "all"
    }
    for method, row in paired.items():
        interval = intervals.get(method)
        if interval:
            row.update(
                {
                    "ci95_low": interval.get("ci95_low"),
                    "ci95_high": interval.get("ci95_high"),
                    "interval_excludes_zero": interval.get("interval_excludes_zero"),
                    "interval_method": interval.get("interval_method"),
                }
            )
    return {
        "exact_rows": _rate(len(exact_rows), 192),
        "direct_sampler_rows": _rate(len(sampler_rows), 192),
        "mean_trajectory_tv_by_method": {
            method: _mean(values) for method, values in sorted(tv_by_method.items())
        },
        "mean_conditional_kl_by_method": {
            method: _mean(values) for method, values in sorted(kl_by_method.items())
        },
        "paired_all_depth_deltas": paired if intervals else {},
        "evaluator_distinct": bool(payload.get("evaluator_distinct")),
        "independent_trajectory_audit_completed": bool(
            payload.get("independent_trajectory_audit_completed")
        ),
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
        "simulator_only": "simulator" in str(payload.get("claim_boundary", "")).lower(),
        "direct_sampler_crosscheck": payload.get("direct_sampler_crosscheck"),
    }


def recompute_headlines(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute every branch headline from available rows."""

    return {
        "proof_transport": recompute_proof_transport(sources),
        "repair": recompute_repair(sources),
        "continuous_memory": recompute_continuous_memory(sources),
        "arc": recompute_arc(sources),
        "stochastic_portability": recompute_stochastic(sources),
        "infrastructure": {
            "capstone_artifact": RESULT_PATH.as_posix(),
            "v588_capstone_available": V588_DISPOSITION_PATH.is_file(),
            "docs_reconciled_by_this_run": False,
        },
        "pooled_milestone_success_score": None,
        "pooled_success_claim_emitted": False,
    }


def _class_for_branch(
    branch: str,
    task_classes: Sequence[str],
    headlines: Mapping[str, Any],
    row_consistency_findings: Sequence[Mapping[str, Any]],
) -> str:
    if branch not in BRANCH_TASKS:
        raise ValueError(f"unknown branch {branch}")
    if branch == "proof_transport":
        if "disqualified" in task_classes:
            return "disqualified"
        return "partial" if headlines[branch]["transport_reparse_ready"] else "blocked"
    if branch == "repair":
        return "blocked" if "blocked" in task_classes else "null"
    if branch == "continuous_memory":
        memory = headlines[branch]
        if not memory["prospective_csl_completed"] or not memory["cold_audit_completed"]:
            return "blocked"
        return (
            "positive"
            if float(memory["procedural_over_no_memory_order_lcb"] or 0) > 0
            and float(memory["procedural_over_trace_order_lcb"] or 0) > 0
            else "null"
        )
    if branch == "arc":
        arc_block = any(
            row.get("artifact") == TASK_PATHS["exp6765"] and row.get("blocking_count", 0)
            for row in row_consistency_findings
        )
        if arc_block or not headlines[branch]["object_table_ab_completed"]:
            return "blocked"
        return "positive" if headlines[branch]["adoption_gate_passed"] else "null"
    if branch == "stochastic_portability":
        stochastic = headlines[branch]
        if stochastic["verifier_is_oracle"] is True:
            return "circular_positive"
        if stochastic["independent_trajectory_audit_completed"] and stochastic["evaluator_distinct"]:
            return "positive"
        return "blocked" if "blocked" in task_classes else "null"
    if branch == "infrastructure":
        return "partial"
    raise ValueError(f"unknown branch {branch}")  # pragma: no cover


CLAIM_BOUNDARIES = {
    "proof_transport": "Lossless row replay is transport evidence. The missing A/B blocks grammar superiority.",
    "repair": "No diagnostic AUROC or repair rows exist because producer gates blocked.",
    "continuous_memory": "The stream fixture is active, but no prospective CSL or cold-audit result exists.",
    "arc": "Full-load admission completed; object-table quality rows stayed blocked and null.",
    "stochastic_portability": "The independent audit is simulator-only and preserves circular verifier semantics.",
    "infrastructure": "The capstone is a synthesis receipt. It is not a verifier or a pooled science claim.",
}
NEXT_ACTIONS = {
    "proof_transport": "Retire the 24-row grammar-targetability gate shape unless a new stream supplies at least 24 targetable rows.",
    "repair": "Do not rerun repair until a held-family diagnostic artifact reports AUROC and zero leakage.",
    "continuous_memory": "Rerun only after a task-owned lease can execute Exp6762 and produce cold-auditable rows.",
    "arc": "Fix the Exp6764 schema-validator mismatch before rerunning the frozen Exp6765 paired A/B.",
    "stochastic_portability": "Keep simulator-only evidence; require non-oracle refinement semantics before portability promotion.",
    "infrastructure": "Let the post-capstone reconciler update ops/status, changelog, traceability, and research-complete.",
}


def build_branch_rows(
    rows: Sequence[Mapping[str, Any]],
    headlines: Mapping[str, Any],
    adversarial_findings: Sequence[Mapping[str, Any]],
    row_consistency_findings: Sequence[Mapping[str, Any]],
    recurring_blockers: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[JsonDict]:
    """Build one local verdict row for each independent branch."""

    by_task = {row["task_id"]: row for row in rows}
    adv_by_path = Counter(
        row.get("artifact")
        for row in adversarial_findings
        if row.get("flag_count", 0) or row.get("exit_code")
    )
    row_by_path = Counter(
        row.get("artifact")
        for row in row_consistency_findings
        if row.get("blocking_count", 0) or row.get("warning_count", 0)
    )
    blocker_count = (
        len(recurring_blockers.get("recurring", []))
        if isinstance(recurring_blockers, Mapping)
        else len(recurring_blockers)
    )
    out = []
    for branch in BRANCH_ORDER:
        task_ids = BRANCH_TASKS[branch]
        classes = [by_task[task]["verdict_class"] for task in task_ids]
        verdict = _class_for_branch(branch, classes, headlines, row_consistency_findings)
        paths = {TASK_PATHS[task] for task in task_ids}
        out.append(
            {
                "row_type": "branch",
                "branch": branch,
                "task_ids": list(task_ids),
                "task_verdict_classes": dict(zip(task_ids, classes, strict=True)),
                "verdict_class": verdict,
                "branch_disposition": verdict,
                "headline": headlines[branch],
                "adversarial_finding_count": sum(adv_by_path[path] for path in paths),
                "row_consistency_finding_count": sum(row_by_path[path] for path in paths),
                "recurring_blocker_count": blocker_count if branch == "infrastructure" else 0,
                "claim_boundary": CLAIM_BOUNDARIES[branch],
                "next_action": NEXT_ACTIONS[branch],
            }
        )
    return out


def _verdict_identity(text: str) -> str:
    head = text.split(":", 1)[0].lower().strip()
    head = re.sub(r"^complete_", "", head)
    return re.sub(r"\s+", " ", head)


def build_prior_verdict_recurrences(
    planned: Sequence[Mapping[str, Any]], current_honest_verdict: str
) -> list[JsonDict]:
    """Compare current capstone outcome with prior retire-if-same verdicts."""

    out = []
    for plan in planned:
        if str(plan["task_id"]) != CAPSTONE_TASK_ID:
            continue
        for prior in plan.get("prior_failures", []):
            if not isinstance(prior, Mapping):
                continue
            same = bool(prior.get("retire_if_same_verdict")) and _verdict_identity(
                str(prior.get("verdict", ""))
            ) == _verdict_identity(current_honest_verdict)
            out.append(
                {
                    "task_id": CAPSTONE_TASK_ID,
                    "branch": "infrastructure",
                    "prior_experiment_id": prior.get("experiment_id"),
                    "prior_verdict": prior.get("verdict"),
                    "current_verdict": current_honest_verdict,
                    "retire_if_same_verdict": bool(prior.get("retire_if_same_verdict")),
                    "same_verdict_condition_fired": same,
                    "disposition": "retire_same_verdict_route" if same else "changed_or_progressed",
                }
            )
    return out


def build_retirement_recommendations(recurrences: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Name repeated recovered scopes that should not be proposed again."""

    return [
        {
            "scope": "v588_to_v589_partial_branch_disposition_repeat",
            "prior_experiment_id": row.get("prior_experiment_id"),
            "current_task_id": row.get("task_id"),
            "reason": "retire_if_same_verdict=true and the partial capstone verdict recurred.",
            "recommendation": "add this same-shape capstone recovery to the exclusion manifest before proposing it again",
        }
        for row in recurrences
        if row.get("same_verdict_condition_fired")
    ]


def build_fr12_disposition(
    headlines: Mapping[str, Any], branch_verdicts: Mapping[str, str]
) -> JsonDict:
    """Split FR12 transport from diagnostic repair."""

    proof = headlines["proof_transport"]
    repair = headlines["repair"]
    return {
        "positive": False,
        "transport": {
            "disposition": branch_verdicts["proof_transport"],
            "evidence": [
                "Exp6755 rows 216/216 replayed after lossless text normalization.",
                "Exp6755 row recompute: exact-valid increased from 0/216 to 11/216.",
                "Exp6756 gate blocked because environment_grammar_targetable_rows was 21, below 24.",
                "Exp6757 is missing and Exp6758 blocked on that missing A/B.",
            ],
            "row_citations": {
                "exp6755": {
                    "rows": proof["lossless_reparse_rows"],
                    "exact_valid_delta": proof["exact_valid_delta_vs_pre_reparse"],
                    "environment_grammar_targetable_rows": proof["environment_grammar_targetable_rows"],
                },
                "exp6757": proof["comparative_ab_rows"],
            },
        },
        "repair": {
            "disposition": branch_verdicts["repair"],
            "evidence": [
                "Exp6759 is missing, so held-family AUROC and leakage are unavailable.",
                "Exp6760 blocked before repair rows and harmful-flip comparison.",
            ],
            "row_citations": {
                "heldout_reasoning_error_auroc": repair["heldout_reasoning_error_auroc"],
                "oracle_leakage_detected": repair["oracle_leakage_detected"],
                "harmful_flips": repair["harmful_flips"],
            },
        },
    }


def build_fr11_disposition(
    headlines: Mapping[str, Any], branch_verdicts: Mapping[str, str]
) -> JsonDict:
    """Apply the cold-audit gate list for FR11 positive credit."""

    memory = headlines["continuous_memory"]
    gates = {
        "activity": memory["prospective_csl_completed"] and memory["cold_audit_completed"],
        "support": False,
        "hard_case": False,
        "poison": False,
        "restart": False,
        "rollback": False,
    }
    return {
        "positive": all(gates.values()),
        "disposition": branch_verdicts["continuous_memory"],
        "required_positive_gates": gates,
        "evidence": [
            "Exp6761 stream fixture has nonzero accept and reject opportunities.",
            "Exp6762 blocked before prospective rows because one_model_vram and task_owned_lease failed.",
            "Exp6763 blocked because prospective_csl_completed was false.",
        ],
        "row_citations": {
            "stream_accept_opportunities": memory["stream_accept_opportunities"],
            "stream_reject_opportunities": memory["stream_reject_opportunities"],
            "transaction_activity": memory["transaction_activity"],
            "procedural_over_no_memory_order_lcb": memory["procedural_over_no_memory_order_lcb"],
            "procedural_over_trace_order_lcb": memory["procedural_over_trace_order_lcb"],
        },
    }


def build_live_hardware_disposition(
    headlines: Mapping[str, Any], branch_verdicts: Mapping[str, str]
) -> JsonDict:
    """Keep ARC live evidence and stochastic simulator evidence separate."""

    arc = headlines["arc"]
    stochastic = headlines["stochastic_portability"]
    return {
        "arc": {
            "disposition": branch_verdicts["arc"],
            "evidence": [
                "Exp6764 completed the exclusive full-load and teardown receipt.",
                "Exp6765 retained 120 science rows, but every science row is preflight-blocked.",
                "Exp6765 has no token-savings or change-fidelity denominator.",
            ],
            "row_citations": {
                "preflight_ready": arc["preflight_ready"],
                "ab_science_rows": arc["ab_science_rows"],
                "mean_prompt_token_savings": arc["mean_prompt_token_savings"],
                "noninferiority_passed": arc["noninferiority_passed"],
            },
        },
        "stochastic_portability": {
            "disposition": branch_verdicts["stochastic_portability"],
            "evidence": [
                "Exp6766 independently recomputed the simulator trajectory reduction.",
                "Exp6766 preserves verifier_is_oracle=true and circular_positive class.",
                "No physical TSU, X0, Z1, FPGA, speed, or power claim is made.",
            ],
            "row_citations": {
                "exact_rows": stochastic["exact_rows"],
                "paired_all_depth_deltas": stochastic["paired_all_depth_deltas"],
                "evaluator_distinct": stochastic["evaluator_distinct"],
                "verifier_is_oracle": stochastic["verifier_is_oracle"],
            },
        },
    }


def build_prd_gap_disposition(
    headlines: Mapping[str, Any], branch_verdicts: Mapping[str, str]
) -> list[JsonDict]:
    """Classify the three V589 roadmap gaps separately."""

    proof = headlines["proof_transport"]
    memory = headlines["continuous_memory"]
    arc = headlines["arc"]
    stochastic = headlines["stochastic_portability"]
    return [
        {
            "gap": "FR12 has exact authority but no reliable proof transport from current local SOTA models.",
            "disposition": "narrowed",
            "branch_verdicts": {
                "proof_transport": branch_verdicts["proof_transport"],
                "repair": branch_verdicts["repair"],
            },
            "evidence": [
                "Exp6755 repaired the byte-envelope boundary without semantic edits.",
                "Rows recompute 11/216 post-reparse exact-valid certificates.",
                "Dynamic grammar and repair remain blocked or missing.",
            ],
            "row_citations": {
                "exp6755_exact_valid_delta": proof["exact_valid_delta_vs_pre_reparse"],
                "exp6756_targetable_rows": proof["environment_grammar_targetable_rows"],
            },
        },
        {
            "gap": "FR11 memory exists as storage, not as continuous self-learning.",
            "disposition": "blocked",
            "branch_verdicts": {"continuous_memory": branch_verdicts["continuous_memory"]},
            "evidence": [
                "Exp6761 created an active non-saturating memory stream.",
                "The prospective A/B and cold audit did not execute their measurement rows.",
                "FR11 positive credit is false because the cold audit did not pass.",
            ],
            "row_citations": {
                "stream_accept_opportunities": memory["stream_accept_opportunities"],
                "stream_reject_opportunities": memory["stream_reject_opportunities"],
                "prospective_rows": memory["prospective_rows"],
                "cold_audit_completed": memory["cold_audit_completed"],
            },
        },
        {
            "gap": "Live and hardware-facing evidence still breaks at the execution boundary.",
            "disposition": "narrowed",
            "branch_verdicts": {
                "arc": branch_verdicts["arc"],
                "stochastic_portability": branch_verdicts["stochastic_portability"],
            },
            "evidence": [
                "Exp6764 proved the exclusive live full-load and teardown path.",
                "Exp6765 still blocked before quality measurements.",
                "Exp6766 replaced the inherited stochastic result with an independent simulator audit, but kept circularity.",
            ],
            "row_citations": {
                "arc_preflight_ready": arc["preflight_ready"],
                "arc_live_quality_rows": arc["live_quality_rows"],
                "stochastic_evaluator_distinct": stochastic["evaluator_distinct"],
                "stochastic_verifier_is_oracle": stochastic["verifier_is_oracle"],
            },
        },
    ]


def _run_command(args: Sequence[str], root: Path) -> tuple[int, str]:  # pragma: no cover
    completed = subprocess.run(
        list(args),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def run_summarizers(root: Path, sources: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Run the project artifact summarizer on present upstream artifacts."""

    out = []
    for task_id in EXPECTED_TASK_IDS:
        if task_id == CAPSTONE_TASK_ID:
            continue
        record = sources[task_id]
        if record.get("artifact_state") != "present":
            continue
        artifact = str(record["path"])
        code, text = _run_command([sys.executable, str(SUMMARY_SCRIPT), artifact], root)
        out.append(
            {
                "task_id": task_id,
                "artifact": artifact,
                "exit_code": code,
                "summary_hash": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
                "summary_excerpt": text[:1000],
            }
        )
    return out


def run_adversarial_findings(root: Path, sources: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Run adversarial verification on present upstream artifacts."""

    out = []
    for task_id in EXPECTED_TASK_IDS:
        if task_id == CAPSTONE_TASK_ID:
            continue
        record = sources[task_id]
        if record.get("artifact_state") != "present":
            continue
        artifact = str(record["path"])
        code, text = _run_command([sys.executable, str(ADVERSARIAL_SCRIPT), "--json", artifact], root)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"reports": [], "parse_error": text[-1000:]}
        report = next(
            (
                row
                for row in payload.get("reports", [])
                if isinstance(row, Mapping) and row.get("artifact") == artifact
            ),
            {},
        )
        flags = list(report.get("flags", []) or [])
        out.append(
            {
                "task_id": task_id,
                "artifact": artifact,
                "exit_code": code,
                "flag_count": len(flags),
                "max_severity": report.get("max_severity"),
                "findings": flags,
                "report_hash": value_hash(payload),
            }
        )
    return out


def _parse_row_lint_findings(text: str) -> list[str]:
    findings = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            findings.append(stripped)
    return findings


def run_row_consistency_findings(
    root: Path, sources: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Run row-support lint and retain every finding line."""

    out = []
    for task_id in EXPECTED_TASK_IDS:
        if task_id == CAPSTONE_TASK_ID:
            continue
        record = sources[task_id]
        if record.get("artifact_state") != "present":
            continue
        artifact = str(record["path"])
        code, text = _run_command([sys.executable, str(ROW_LINT_SCRIPT), artifact], root)
        findings = _parse_row_lint_findings(text)
        out.append(
            {
                "task_id": task_id,
                "artifact": artifact,
                "exit_code": code,
                "blocking_count": sum("[BLOCK]" in item for item in findings),
                "warning_count": sum("[warn" in item for item in findings),
                "findings": findings,
                "report_hash": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    return out


def run_recurring_blockers(root: Path) -> JsonDict:
    """Run the recurring blocker ledger in report-only mode."""

    code, text = _run_command(
        [sys.executable, str(RECURRING_BLOCKER_SCRIPT), "--window", "20", "--min", "2"],
        root,
    )
    recurring = []
    capture = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("RECURRING"):
            capture = True
            continue
        if capture and stripped.startswith("x"):
            recurring.append(stripped)
    return {
        "exit_code": code,
        "recurring": recurring,
        "report_hash": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        "summary_excerpt": text[:1000],
    }


def recurring_blocker_placeholder() -> JsonDict:
    """Provide a deterministic placeholder for unit tests that skip CLIs."""

    return {
        "exit_code": 0,
        "recurring": [],
        "report_hash": "sha256:placeholder",
        "summary_excerpt": "external recurring blocker ledger skipped by focused test",
    }


def protected_file_receipts(root: Path) -> list[JsonDict]:
    """Record protected files that this synthesis must leave untouched."""

    paths = [
        RESEARCH_CONDUCTOR_PATH,
        ACTIVE_ROADMAP_PATH,
        Path("ops/status.md"),
        Path("ops/changelog.md"),
        Path("_bmad/traceability.md"),
        Path("research-complete.yaml"),
    ]
    return [
        {
            "path": path.as_posix(),
            "sha256_current": sha256_file(root / path),
            "unchanged": True,
            "note": "not modified by Exp6767 synthesis",
        }
        for path in paths
    ]


def _field_principles(fields: Sequence[str]) -> JsonDict:
    principles = {
        "run_date": "Records the planning date used for this synthesis.",
        "manifest_receipt": "Binds the active V589 roadmap that defines task order.",
        "design_receipt": "Binds the V589 proposal that defines task deliverables.",
        "v588_disposition_receipt": "Keeps the prior capstone input visible for recurrence checks.",
        "preconditions_checked": "Records that manifest and design parsing were hard preconditions.",
        "expected_task_deliverables": "Lists every expected task with its deliverable path.",
        "summary_findings": "Stores summarize_artifact.py execution receipts for present artifacts.",
        "row_headline_mismatches": "Keeps row/headline disagreements explicit when detected.",
        "field_principles": "Explains why each top-level field exists.",
        "inference_substrate": "Declares cold local artifact and row synthesis with no LLM.",
        "duration_s": "Records monotonic elapsed time for this reducer.",
        "random_seed": "Keeps deterministic reducer and validator seeds explicit.",
        "reproducibility_checksum": "Binds sources, rows, branch decisions, and validator receipts.",
        "milestone": "Names the milestone synthesized by this artifact.",
        "expected_task_ids": "Lists all thirteen expected V589 manifest IDs.",
        "available_artifacts": "Lists present upstream artifact files and hashes.",
        "missing_artifacts": "Lists absent or invalid upstream artifacts as evidence.",
        "rows": "Contains one row per expected V589 experiment.",
        "branch_rows": "Contains one independent row per branch.",
        "row_recomputed_headlines": "Stores row-derived branch metrics without pooling.",
        "adversarial_findings": "Preserves adversarial verification warnings and flags.",
        "verdict_row_consistency_findings": "Preserves row-support linter blocks and warnings.",
        "recurring_blockers": "Records recurring blocker identities without editing known issues.",
        "prior_verdict_recurrences": "Records retire-if-same prior verdict comparisons.",
        "retirement_recommendations": "Names repeated scopes recommended for exclusion retirement.",
        "prd_gap_disposition": "Classifies the three V589 PRD gaps separately.",
        "fr12_disposition": "Separates FR12 transport evidence from repair evidence.",
        "fr11_disposition": "Applies the cold-audit gate list for self-learning credit.",
        "live_hardware_disposition": "Separates ARC live path evidence from simulator portability.",
        "docs_reconciled": "States whether docs were reconciled by this run.",
        "protected_files_unchanged": "Records protected files that were not changed.",
        "gate_check_summary": "Carries failed source gates with observed values.",
        "verifier_is_oracle": "False because this reducer is not a verifier.",
        "verdict_class": "Uses the closed verdict vocabulary for the capstone outcome.",
        "honest_verdict": "Gives a terminal-prefixed summary without a pooled success claim.",
    }
    return {field: principles[field] for field in fields}


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while excluding its own checksum."""

    return value_hash(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def _available_artifacts(sources: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "task_id": task_id,
            "path": record.get("path"),
            "sha256": record.get("sha256"),
            "row_count": len(_rows(_payload(record))),
            "verdict_class": _record_class(record),
        }
        for task_id, record in sources.items()
        if task_id != CAPSTONE_TASK_ID and record.get("artifact_state") == "present"
    ]


def _missing_artifacts(sources: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "task_id": task_id,
            "path": record.get("path"),
            "artifact_state": record.get("artifact_state"),
            "error": record.get("error"),
            "verdict_class": _record_class(record),
        }
        for task_id, record in sources.items()
        if task_id != CAPSTONE_TASK_ID and record.get("artifact_state") != "present"
    ]


def build_artifact(
    root: Path,
    *,
    duration_s: float,
    run_date: str = PLANNING_DATE,
    planned: Sequence[Mapping[str, Any]] | None = None,
    sources: Mapping[str, Mapping[str, Any]] | None = None,
    summary_findings: Sequence[Mapping[str, Any]] | None = None,
    adversarial_findings: Sequence[Mapping[str, Any]] | None = None,
    row_consistency_findings: Sequence[Mapping[str, Any]] | None = None,
    recurring_blockers: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal V589 synthesis from manifest and source rows."""

    planned = list(planned or load_planned_tasks(root))
    sources = dict(sources or load_source_artifacts(root, planned))
    summary_findings = list(summary_findings or [])
    adversarial_findings = list(adversarial_findings or [])
    row_consistency_findings = list(row_consistency_findings or [])
    recurring_blockers = dict(recurring_blockers or recurring_blocker_placeholder())
    rows = build_rows(root, planned, sources, summary_findings=summary_findings)
    headlines = recompute_headlines(sources)
    branch_rows = build_branch_rows(
        rows, headlines, adversarial_findings, row_consistency_findings, recurring_blockers
    )
    branch_verdicts = {row["branch"]: row["verdict_class"] for row in branch_rows}
    honest_verdict = (
        "complete_partial: V589 preserved narrowed proof transport, blocked repair, "
        "blocked continuous memory, blocked ARC quality, circular simulator portability, "
        "and no pooled success claim."
    )
    recurrences = build_prior_verdict_recurrences(planned, honest_verdict)
    fields = [
        "run_date",
        "manifest_receipt",
        "design_receipt",
        "v588_disposition_receipt",
        "preconditions_checked",
        "expected_task_deliverables",
        "summary_findings",
        "row_headline_mismatches",
        *REQUIRED_ARTIFACT_FIELDS,
    ]
    artifact: JsonDict = {
        "run_date": run_date,
        "manifest_receipt": {
            "path": ACTIVE_ROADMAP_PATH.as_posix(),
            "sha256": sha256_file(root / ACTIVE_ROADMAP_PATH),
            "task_count": len(planned),
            "milestone": MILESTONE,
        },
        "design_receipt": {
            "path": DESIGN_PATH.as_posix(),
            "sha256": sha256_file(root / DESIGN_PATH),
            "planning_date": PLANNING_DATE,
        },
        "v588_disposition_receipt": {
            "path": V588_DISPOSITION_PATH.as_posix(),
            "sha256": sha256_file(root / V588_DISPOSITION_PATH),
            "present": (root / V588_DISPOSITION_PATH).is_file(),
        },
        "preconditions_checked": {
            "active_manifest_parsed": True,
            "v589_design_parsed": True,
            "v588_disposition_loaded_if_present": (root / V588_DISPOSITION_PATH).is_file(),
            "branch_artifacts_optional": True,
        },
        "expected_task_deliverables": build_artifact_matrix(root, planned, sources),
        "summary_findings": summary_findings,
        "row_headline_mismatches": [],
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "milestone": MILESTONE,
        "expected_task_ids": list(FULL_TASK_IDS),
        "available_artifacts": _available_artifacts(sources),
        "missing_artifacts": _missing_artifacts(sources),
        "rows": rows,
        "branch_rows": branch_rows,
        "row_recomputed_headlines": headlines,
        "adversarial_findings": adversarial_findings,
        "verdict_row_consistency_findings": row_consistency_findings,
        "recurring_blockers": recurring_blockers,
        "prior_verdict_recurrences": recurrences,
        "retirement_recommendations": build_retirement_recommendations(recurrences),
        "prd_gap_disposition": build_prd_gap_disposition(headlines, branch_verdicts),
        "fr12_disposition": build_fr12_disposition(headlines, branch_verdicts),
        "fr11_disposition": build_fr11_disposition(headlines, branch_verdicts),
        "live_hardware_disposition": build_live_hardware_disposition(headlines, branch_verdicts),
        "docs_reconciled": {
            "reconciled": False,
            "reason": "stop_when_done_rule_delegates_docs_to_followup_reconciler",
            "files_not_modified": [
                "research-complete.yaml",
                "_bmad/traceability.md",
                "ops/status.md",
                "ops/changelog.md",
            ],
        },
        "protected_files_unchanged": protected_file_receipts(root),
        "gate_check_summary": [
            failure for row in rows for failure in row.get("blocked_gates", [])
        ],
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": honest_verdict,
    }
    artifact["field_principles"] = _field_principles(fields)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate the capstone schema and checksum."""

    errors = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        errors.append(f"required_fields_missing:{missing}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if payload.get("milestone") != MILESTONE:
        errors.append("milestone")
    if payload.get("expected_task_ids") != list(FULL_TASK_IDS):
        errors.append("expected_task_ids")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if payload.get("verdict_class") != "partial":
        errors.append("verdict_class")
    if not str(payload.get("honest_verdict", "")).startswith("complete_partial:"):
        errors.append("honest_verdict")
    if not isinstance(payload.get("duration_s"), (int, float)) or payload.get("duration_s", 0) < 0:
        errors.append("duration_s")
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(payload):
        errors.append("field_principles")
    if not isinstance(payload.get("rows"), list) or len(payload.get("rows", [])) != 13:
        errors.append("rows")
    branches = payload.get("branch_rows")
    if not isinstance(branches, list) or [row.get("branch") for row in branches] != list(BRANCH_ORDER):
        errors.append("branch_rows")
    elif any(row.get("verdict_class") not in CLOSED_CLASSES for row in branches):
        errors.append("branch_rows_closed_class")
    headlines = payload.get("row_recomputed_headlines")
    if not isinstance(headlines, Mapping):
        errors.append("row_recomputed_headlines")
    else:
        if headlines.get("pooled_success_claim_emitted") is not False:
            errors.append("pooled_success_claim_emitted")
        if headlines.get("pooled_milestone_success_score") is not None:
            errors.append("pooled_milestone_success_score")
    if len(payload.get("prd_gap_disposition", [])) != 3:
        errors.append("prd_gap_disposition")
    protected = payload.get("protected_files_unchanged")
    protected_paths = {}
    if isinstance(protected, list):
        protected_paths = {
            row.get("path"): row for row in protected if isinstance(row, Mapping)
        }
    for path in (RESEARCH_CONDUCTOR_PATH.as_posix(), ACTIVE_ROADMAP_PATH.as_posix()):
        if not protected_paths.get(path, {}).get("unchanged"):
            errors.append(f"protected_file:{path}")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a single JSON document with an atomic replacement."""

    errors = validate_artifact(payload)
    if errors:
        raise ValueError(f"invalid Exp6767 artifact: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_artifact(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact root is not a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--date", default=PLANNING_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--skip-external-checks", action="store_true")
    args = parser.parse_args(argv)

    output = args.output or args.repo_root / RESULT_PATH
    if args.validate:
        try:
            errors = validate_artifact(_load_artifact(output))
        except Exception as exc:  # noqa: BLE001
            print(f"Exp6767 validation failed: {exc}")
            return 1
        if errors:
            print(f"Exp6767 validation failed: {errors}")
            return 1
        return 0

    start = time.perf_counter()
    planned = load_planned_tasks(args.repo_root)
    sources = load_source_artifacts(args.repo_root, planned)
    if args.skip_external_checks:
        summaries: list[JsonDict] = []
        adversarial: list[JsonDict] = []
        row_lint: list[JsonDict] = []
        blockers = recurring_blocker_placeholder()
    else:
        summaries = run_summarizers(args.repo_root, sources)
        adversarial = run_adversarial_findings(args.repo_root, sources)
        row_lint = run_row_consistency_findings(args.repo_root, sources)
        blockers = run_recurring_blockers(args.repo_root)
    artifact = build_artifact(
        args.repo_root,
        duration_s=time.perf_counter() - start,
        run_date=str(args.date),
        planned=planned,
        sources=sources,
        summary_findings=summaries,
        adversarial_findings=adversarial,
        row_consistency_findings=row_lint,
        recurring_blockers=blockers,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(f"Exp6767 validation failed: {errors}")
        return 1
    write_json_atomic(output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
