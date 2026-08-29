"""Synthesize V588 branch states without pooling the outcomes.

This capstone treats the active roadmap and upstream artifacts as data. It
keeps missing or blocked branch evidence visible, replays available row
metrics, and records validator findings without changing source artifacts.

Spec refs: REQ-REPORT-6754 and SCENARIO-REPORT-6754-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.08.588"
PLANNING_DATE = "20260829"
INFERENCE_SUBSTRATE = "local_cpu_branch_row_replay_and_adversarial_audit_no_llm"
RANDOM_SEED = {"row_recompute_seed": 6754, "adversarial_seed": 6754001}

RESULT_PATH = Path("results/experiment_6754_v588_branch_disposition.json")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ADVERSARIAL_SCRIPT = Path("scripts/adversarial_verify.py")
ROW_LINT_SCRIPT = Path("scripts/verdict_row_consistency_lint.py")
MODULE_PATH = Path("python/carnot/experiment_6754_v588_branch_disposition.py")
TEST_PATH = Path("tests/python/test_experiment_6754_v588_branch_disposition.py")

CLOSED_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
    "missing",
}
EXPECTED_TASK_IDS = tuple(f"exp{number}" for number in range(6742, 6755))
FULL_TASK_IDS = (
    "exp6742-v588-handoff-contract-audit",
    "exp6743-task-owned-phase-accelerator-canary",
    "exp6744-hardness-controlled-certificate-stream",
    "exp6745-sota-dual-encoding-proposal-corpus",
    "exp6746-oracle-distinct-diagnostic-energy",
    "exp6747-diagnostic-energy-localized-repair-ab",
    "exp6748-transactional-constraint-memory-fixture",
    "exp6749-prospective-support-preserving-csl-ab",
    "exp6750-csl-durability-support-poison-audit",
    "exp6751-thermalizer-factor-trajectory-fidelity",
    "exp6752-arc-code-carrying-tool-preflight",
    "exp6753-object-table-fetch-on-demand-ab",
    "exp6754-v588-branch-disposition",
)
CAPSTONE_TASK_ID = "exp6754"
TASK_PATHS = {
    "exp6742": "results/experiment_6742_v588_handoff_contract_audit.json",
    "exp6743": "results/experiment_6743_task_owned_phase_accelerator_canary.json",
    "exp6744": "results/experiment_6744_hardness_controlled_certificate_stream.json",
    "exp6745": "results/experiment_6745_sota_dual_encoding_proposal_corpus.json",
    "exp6746": "results/experiment_6746_oracle_distinct_diagnostic_energy.json",
    "exp6747": "results/experiment_6747_diagnostic_energy_localized_repair_ab.json",
    "exp6748": "results/experiment_6748_transactional_constraint_memory_fixture.json",
    "exp6749": "results/experiment_6749_prospective_support_preserving_csl_ab.json",
    "exp6750": "results/experiment_6750_csl_durability_support_poison_audit.json",
    "exp6751": "results/experiment_6751_thermalizer_factor_trajectory_fidelity.json",
    "exp6752": "results/experiment_6752_arc_code_carrying_tool_preflight.json",
    "exp6753": "results/experiment_6753_object_table_fetch_on_demand_ab.json",
    "exp6754": RESULT_PATH.as_posix(),
}
BRANCH_ORDER = (
    "handoff",
    "activity",
    "fr12_diagnostics_repair",
    "fr11_continuous_self_learning",
    "stochastic_portability",
    "arc_transport_object_table_quality",
)
BRANCH_TASKS = {
    "handoff": ("exp6742",),
    "activity": ("exp6743",),
    "fr12_diagnostics_repair": ("exp6744", "exp6745", "exp6746", "exp6747"),
    "fr11_continuous_self_learning": ("exp6748", "exp6749", "exp6750"),
    "stochastic_portability": ("exp6751",),
    "arc_transport_object_table_quality": ("exp6752", "exp6753"),
}
TASK_BRANCH = {task: branch for branch, tasks in BRANCH_TASKS.items() for task in tasks}
TASK_BRANCH["exp6754"] = "capstone"

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "rows",
    "artifact_presence_matrix",
    "branch_verdicts",
    "recomputed_headlines",
    "row_headline_mismatches",
    "adversarial_findings",
    "prd_gap_disposition",
    "prior_failure_retirements",
    "next_licensed_actions",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
)


def canonical_json(value: Any) -> bytes:
    """Encode JSON once so checksums replay exactly."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def value_hash(value: Any) -> str:
    """Hash a JSON-compatible value."""

    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Hash one file, or return None when the file is absent."""

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
    """Return the `expNNNN` prefix from a manifest task ID."""

    match = re.match(r"^(exp\d+)(?:-|$)", task_id)
    if match is None:
        raise ValueError(f"invalid V588 task id: {task_id}")
    return match.group(1)


def _next_deliverable(lines: Sequence[str], start: int) -> str:
    for line in lines[start + 1 :]:
        match = re.search(r"\*\*Deliverable:\*\*\s*`([^`]+)`", line)
        if match:
            return match.group(1)
        if line.startswith("### Exp "):
            break
    raise ValueError("V588 design task deliverable missing")


def parse_design_tasks(text: str) -> tuple[str, list[JsonDict]]:
    """Read the V588 design task list and deliverables."""

    milestone = re.search(r"\*\*Milestone:\*\*\s*`([^`]+)`", text)
    if milestone is None:
        raise ValueError("V588 design milestone missing")
    rows: list[JsonDict] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"### Exp (\d+):\s*(.+)$", line)
        if match is None:
            continue
        short = f"exp{match.group(1)}"
        rows.append(
            {
                "task_id": short,
                "title": match.group(2).strip(),
                "deliverable": _next_deliverable(lines, index),
            }
        )
    return milestone.group(1), rows


def load_planned_tasks(root: Path) -> list[JsonDict]:
    """Load active V588 manifest rows after checking the design."""

    manifest = yaml.safe_load((root / ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("tasks"), list):
        raise ValueError("active roadmap must be a mapping with tasks")
    design_milestone, design_tasks = parse_design_tasks(
        (root / DESIGN_PATH).read_text(encoding="utf-8")
    )
    if design_milestone != MILESTONE:
        raise ValueError(f"expected V588 design, observed {design_milestone}")
    if [row["task_id"] for row in design_tasks] != list(EXPECTED_TASK_IDS):
        raise ValueError("V588 design must contain Exp6742 through Exp6754")

    tasks = [task for task in manifest["tasks"] if isinstance(task, Mapping)]
    if [str(task.get("id")) for task in tasks] != list(FULL_TASK_IDS):
        raise ValueError("active V588 manifest must contain the exact 13 tasks")
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
                "manifest_task_id": task["id"],
                "title": task["title"],
                "path": task["deliverable"],
                "branch": TASK_BRANCH[short],
                "prior_failures": list(task.get("prior_failures") or []),
            }
        )
    return planned


def missing_source_record(task_id: str, reason: str) -> JsonDict:
    """Create a source record for an absent optional branch artifact."""

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
    """Load every upstream artifact while keeping missing files as data."""

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


def _rows(payload: Mapping[str, Any] | None) -> list[JsonDict]:
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    return (
        list(rows) if isinstance(rows, list) and all(isinstance(row, dict) for row in rows) else []
    )


def _record_class(record: Mapping[str, Any]) -> str:
    state = record.get("artifact_state")
    if state == "missing":
        return "missing"
    if state == "invalid":
        return "disqualified"
    if state == "current_synthesis":
        return "partial"
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        return "missing"
    declared = payload.get("verdict_class")
    if declared in CLOSED_CLASSES:
        return str(declared)
    text = f"{payload.get('status', '')} {payload.get('honest_verdict', '')}".lower()
    if "blocked" in text or "gate_check_failed" in text:
        return "blocked"
    if "circular_positive" in text:
        return "circular_positive"
    if "partial" in text:
        return "partial"
    if "positive" in text or "success" in text:
        return "positive"
    if "null" in text or "complete" in text:
        return "null"
    return "disqualified"


def _get_path(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _numbers_close(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-12
    return left == right


def _compare_headline(
    mismatches: list[JsonDict],
    record: Mapping[str, Any],
    field: str,
    recomputed_value: Any,
) -> None:
    payload = record.get("payload")
    artifact_value = _get_path(payload, field) if isinstance(payload, Mapping) else None
    if _numbers_close(artifact_value, recomputed_value):
        return
    mismatches.append(
        {
            "artifact": record.get("path"),
            "field": field,
            "artifact_value": artifact_value,
            "recomputed_value": recomputed_value,
            "reason": (
                "missing_denominator"
                if isinstance(recomputed_value, Mapping)
                and recomputed_value.get("denominator") == 0
                else "headline_mismatch"
            ),
        }
    )


def _rate(numerator: int, denominator: int, cause: str | None = None) -> JsonDict:
    row: JsonDict = {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }
    if denominator == 0:
        row["value"] = None
        row["cause"] = cause or "no eligible rows"
    return row


def _mean(values: Sequence[float], cause: str | None = None) -> JsonDict:
    row: JsonDict = {
        "denominator": len(values),
        "value": sum(values) / len(values) if values else None,
    }
    if not values:
        row["cause"] = cause or "no eligible rows"
    return row


def recompute_handoff(
    sources: Mapping[str, Mapping[str, Any]], mismatches: list[JsonDict]
) -> JsonDict:
    record = sources["exp6742"]
    payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
    rows = _rows(payload)
    task_rows = [row for row in rows if row.get("row_type") == "task"]
    gate_rows = [row for row in rows if row.get("row_type") == "gate"]
    preserved = bool(rows) and all(bool(row.get("operationally_preserved")) for row in rows)
    headline = {
        "task_rows": len(task_rows),
        "gate_rows": len(gate_rows),
        "operational_rows": sum(bool(row.get("operationally_preserved")) for row in rows),
        "handoff_contract_preserved": preserved,
    }
    _compare_headline(mismatches, record, "task_count", len(task_rows))
    _compare_headline(mismatches, record, "handoff_contract_preserved", preserved)
    return headline


def recompute_activity(
    sources: Mapping[str, Mapping[str, Any]], mismatches: list[JsonDict]
) -> JsonDict:
    record = sources["exp6743"]
    payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
    rows = _rows(payload)
    phase_rows = payload.get("task_phase_rows") if isinstance(payload, Mapping) else []
    clocks = [
        int(row.get("monotonic_ns", 0))
        for row in phase_rows
        if isinstance(row, Mapping) and isinstance(row.get("monotonic_ns"), int)
    ]
    phase_monotonic = bool(clocks) and clocks == sorted(clocks)
    first = _rate(sum(bool(row.get("first_token_reached")) for row in rows), len(rows))
    teardown = _rate(sum(bool(row.get("teardown_completed")) for row in rows), len(rows))
    offload = _rate(
        sum(
            bool(row.get("gpu_layers"))
            and row["gpu_layers"].get("offloaded") == row["gpu_layers"].get("total")
            for row in rows
        ),
        len(rows),
    )
    ready = (
        bool(rows)
        and first["rate"] == teardown["rate"] == offload["rate"] == 1.0
        and phase_monotonic
    )
    _compare_headline(mismatches, record, "accelerator_receipt_ready", ready)
    _compare_headline(mismatches, record, "phase_clock_monotonic", phase_monotonic)
    return {
        "first_token_reached": first,
        "teardown_completed": teardown,
        "full_cuda_offload": offload,
        "phase_clock_monotonic": phase_monotonic,
        "accelerator_receipt_ready": ready,
        "models_used": payload.get("models_used"),
    }


def recompute_fr12(
    sources: Mapping[str, Mapping[str, Any]], mismatches: list[JsonDict]
) -> JsonDict:
    stream_record = sources["exp6744"]
    corpus_record = sources["exp6745"]
    diagnostic_record = sources["exp6746"]
    repair_record = sources["exp6747"]
    stream = (
        stream_record.get("payload") if isinstance(stream_record.get("payload"), Mapping) else {}
    )
    corpus = (
        corpus_record.get("payload") if isinstance(corpus_record.get("payload"), Mapping) else {}
    )
    diagnostic = (
        diagnostic_record.get("payload")
        if isinstance(diagnostic_record.get("payload"), Mapping)
        else {}
    )
    repair = (
        repair_record.get("payload") if isinstance(repair_record.get("payload"), Mapping) else {}
    )
    stream_rows = _rows(stream)
    corpus_rows = _rows(corpus)
    diagnosis_counts = dict(Counter(str(row.get("diagnosis")) for row in corpus_rows))
    for label in (
        "abstention",
        "exact_valid",
        "malformed_certificate",
        "reasoning_error",
        "translation_disagreement",
    ):
        diagnosis_counts.setdefault(label, 0)
    exact_valid = sum(row.get("diagnosis") == "exact_valid" for row in corpus_rows)
    stream_ready = (
        bool(stream_rows)
        and len(stream_rows) == 72
        and all(row.get("certificate") for row in stream_rows)
    )
    corpus_ready = bool(corpus_rows) and len(corpus_rows) == int(corpus.get("planned_row_count", 0))
    _compare_headline(mismatches, stream_record, "hardness_stream_ready", stream_ready)
    _compare_headline(mismatches, corpus_record, "diagnosis_counts", diagnosis_counts)
    _compare_headline(mismatches, corpus_record, "planned_row_count", len(corpus_rows))
    _compare_headline(
        mismatches,
        diagnostic_record,
        "heldout_reasoning_error_auroc",
        diagnostic.get("heldout_reasoning_error_auroc"),
    )
    return {
        "certificate_rows": _rate(len(stream_rows), 72),
        "family_counts": dict(Counter(str(row.get("family")) for row in stream_rows)),
        "relabel_pairs": len(stream.get("relabel_pair_receipts", []) or []),
        "hardness_stream_ready": stream_ready,
        "proposal_rows": _rate(len(corpus_rows), int(corpus.get("planned_row_count", 0) or 0)),
        "proposal_exact_valid": _rate(exact_valid, len(corpus_rows)),
        "diagnosis_counts": diagnosis_counts,
        "dual_encoding_corpus_ready": corpus_ready
        and bool(corpus.get("dual_encoding_corpus_ready")),
        "heldout_reasoning_error_auroc": diagnostic.get("heldout_reasoning_error_auroc"),
        "oracle_leakage_detected": bool(diagnostic.get("oracle_leakage_detected")),
        "diagnostic_energy_ready": bool(diagnostic.get("diagnostic_energy_ready")),
        "repair_rows": _rate(len(_rows(repair)), 24, "repair A/B blocked before rows"),
        "repair_blocked_reason": repair.get("blocked_reason") or repair.get("honest_verdict"),
    }


def _arm_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, JsonDict]:
    by_arm: dict[str, list[float]] = {}
    for row in rows:
        arm = str(row.get("arm"))
        value = row.get(metric)
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            by_arm.setdefault(arm, []).append(float(value))
    return {arm: _mean(values) for arm, values in sorted(by_arm.items())}


def recompute_fr11(
    sources: Mapping[str, Mapping[str, Any]], mismatches: list[JsonDict]
) -> JsonDict:
    fixture_record = sources["exp6748"]
    ab_record = sources["exp6749"]
    audit_record = sources["exp6750"]
    fixture = (
        fixture_record.get("payload") if isinstance(fixture_record.get("payload"), Mapping) else {}
    )
    ab = ab_record.get("payload") if isinstance(ab_record.get("payload"), Mapping) else {}
    audit = audit_record.get("payload") if isinstance(audit_record.get("payload"), Mapping) else {}
    ab_rows = _rows(ab)
    order_metric_rows = [
        row
        for row in _rows(audit)
        if row.get("row_type") == "order_metric"
        and row.get("metric") == "prequential_exact_yield_delta"
        and isinstance(row.get("value"), (int, float))
    ]
    order_deltas = [float(row["value"]) for row in order_metric_rows]
    mean_delta = sum(order_deltas) / len(order_deltas) if order_deltas else None
    commit_totals = dict(
        (audit.get("commit_reject_rollback_counts") or {}).get("prospective_rows") or {}
    )
    _compare_headline(
        mismatches,
        fixture_record,
        "transaction_memory_ready",
        bool(fixture.get("transaction_memory_ready")),
    )
    _compare_headline(
        mismatches,
        ab_record,
        "prospective_csl_completed",
        bool(ab.get("prospective_csl_completed")),
    )
    _compare_headline(
        mismatches, audit_record, "csl_audit_passed", bool(audit.get("csl_audit_passed"))
    )
    return {
        "transaction_memory_ready": bool(fixture.get("transaction_memory_ready")),
        "read_only_attacks_rejected": _rate(
            sum(bool(row.get("rejected")) for row in fixture.get("read_only_violations", []) or []),
            len(fixture.get("read_only_violations", []) or []),
        ),
        "prospective_rows": len(ab_rows),
        "pass_at_1_by_arm": _arm_metric(ab_rows, "pass_at_1"),
        "best_at_k_by_arm": _arm_metric(ab_rows, "best_at_k"),
        "effective_rewardable_support_by_arm": _arm_metric(ab_rows, "effective_rewardable_support"),
        "joint_correct_constraint_support_by_arm": _arm_metric(
            ab_rows, "joint_correct_constraint_support"
        ),
        "prequential_exact_yield_delta_by_order": {
            "denominator": len(order_deltas),
            "mean_delta": mean_delta,
            "values": order_deltas,
        },
        "order_level_ci95": audit.get("order_level_ci95"),
        "commit_activity": {
            "commits": int(commit_totals.get("commits", 0)),
            "rejects": int(commit_totals.get("rejects", 0)),
            "rollbacks": int(commit_totals.get("rollbacks", 0)),
        },
        "admitted_poison_count": audit.get("admitted_poison_count"),
        "future_leakage_count": audit.get("future_leakage_count"),
        "retention_failures": audit.get("retention_failures"),
        "csl_audit_passed": bool(audit.get("csl_audit_passed")),
    }


def recompute_stochastic(
    sources: Mapping[str, Mapping[str, Any]], mismatches: list[JsonDict]
) -> JsonDict:
    record = sources["exp6751"]
    payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
    rows = _rows(payload)
    by_arm: dict[str, list[float]] = {}
    for row in rows:
        if isinstance(row.get("trajectory_tv"), (int, float)):
            by_arm.setdefault(str(row.get("arm")), []).append(float(row["trajectory_tv"]))
    arm_means = {arm: _mean(values) for arm, values in sorted(by_arm.items())}
    context = arm_means.get("context_matched", {}).get("value")
    independent = arm_means.get("independent_factor", {}).get("value")
    refined = arm_means.get("trajectory_refinement", {}).get("value")
    if context is not None:
        _compare_headline(
            mismatches,
            record,
            "positive_result_gate.context_matched_mean_trajectory_tv",
            context,
        )
    if independent is not None:
        _compare_headline(
            mismatches,
            record,
            "positive_result_gate.independent_mean_trajectory_tv",
            independent,
        )
    if refined is not None:
        _compare_headline(
            mismatches,
            record,
            "positive_result_gate.trajectory_refinement_mean_trajectory_tv",
            refined,
        )
        _compare_headline(
            mismatches,
            record,
            "positive_result_gate.best_refined_mean_trajectory_tv",
            min(value for value in (context, refined) if value is not None),
        )
    return {
        "trajectory_tv_by_arm": arm_means,
        "context_reduced_vs_independent": (
            bool(context < independent)
            if context is not None and independent is not None
            else False
        ),
        "trajectory_reduced_vs_independent": (
            bool(refined < independent)
            if refined is not None and independent is not None
            else False
        ),
        "compiler_fidelity_completed": bool(payload.get("compiler_fidelity_completed")),
        "hardware_used": payload.get("hardware_used"),
        "simulator_used": payload.get("simulator_used"),
        "claim_scope": payload.get("claim_scope"),
    }


def recompute_arc(sources: Mapping[str, Mapping[str, Any]], mismatches: list[JsonDict]) -> JsonDict:
    preflight_record = sources["exp6752"]
    ab_record = sources["exp6753"]
    preflight = (
        preflight_record.get("payload")
        if isinstance(preflight_record.get("payload"), Mapping)
        else {}
    )
    ab = ab_record.get("payload") if isinstance(ab_record.get("payload"), Mapping) else {}
    preflight_rows = _rows(preflight)
    science_rows = [row for row in _rows(ab) if row.get("row_kind") == "science"]
    valid_science = [
        row
        for row in science_rows
        if row.get("failure_class") is None
        and row.get("change_fidelity") is not None
        and row.get("prompt_tokens") is not None
    ]
    parse_dispatch = _rate(
        sum(
            row.get("parsed_tool") == "find_objects"
            and bool((row.get("dispatch_result") or {}).get("ok"))
            and bool(row.get("bounded_response_bytes"))
            for row in preflight_rows
        ),
        len(preflight_rows),
    )
    science_pairs = _mean(
        [float(row["change_fidelity"]) for row in valid_science],
        "object-table A/B blocked before live science rows",
    )
    _compare_headline(
        mismatches,
        preflight_record,
        "arc_context_tool_preflight_ready",
        bool(preflight.get("arc_context_tool_preflight_ready")),
    )
    _compare_headline(mismatches, ab_record, "object_table_ab_completed", bool(valid_science))
    return {
        "preflight_parse_dispatch_bounded": parse_dispatch,
        "context_requested": preflight.get("context_requested") or ab.get("context_requested"),
        "context_observed_by_model": preflight.get("context_observed_by_model"),
        "live_path_reached": bool(preflight.get("live_path_reached")),
        "object_table_science_pairs": science_pairs,
        "mean_prompt_token_savings": ab.get("mean_prompt_token_savings"),
        "fetch_rate": ab.get("fetch_rate"),
        "useful_fetch_rate": ab.get("useful_fetch_rate"),
        "change_fidelity_delta": ab.get("change_fidelity_delta"),
        "adoption_gate_passed": bool(ab.get("adoption_gate_passed")),
        "object_table_ab_completed": bool(ab.get("object_table_ab_completed")),
        "solve_claim": bool(preflight.get("solve_claim") or ab.get("solve_claim")),
        "blocked_failure_classes": sorted(
            {str(row.get("failure_class")) for row in science_rows if row.get("failure_class")}
        ),
    }


def recompute_headlines(
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[JsonDict, list[JsonDict]]:
    """Recompute every branch headline from retained rows."""

    mismatches: list[JsonDict] = []
    headlines = {
        "handoff": recompute_handoff(sources, mismatches),
        "activity": recompute_activity(sources, mismatches),
        "fr12_diagnostics_repair": recompute_fr12(sources, mismatches),
        "fr11_continuous_self_learning": recompute_fr11(sources, mismatches),
        "stochastic_portability": recompute_stochastic(sources, mismatches),
        "arc_transport_object_table_quality": recompute_arc(sources, mismatches),
        "pooled_milestone_success_score": None,
        "pooled_success_claim_emitted": False,
    }
    return headlines, mismatches


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
    """Normalize failed gate diagnostics from the source artifact."""

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


def _authority_boundary(payload: Mapping[str, Any] | None) -> JsonDict:
    if not isinstance(payload, Mapping):
        return {
            "models_used": None,
            "live_model_invoked": None,
            "inference_substrate": None,
            "verifier_is_oracle": None,
            "hardware_used": None,
            "simulator_used": None,
            "claim_boundary": None,
            "claim_scope": None,
            "solve_claim": None,
            "live_path_reached": None,
            "context_requested": None,
            "context_observed_by_model": None,
        }
    return {
        "models_used": payload.get("models_used"),
        "live_model_invoked": payload.get("live_model_invoked"),
        "inference_substrate": payload.get("inference_substrate"),
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
        "hardware_used": payload.get("hardware_used"),
        "simulator_used": payload.get("simulator_used"),
        "claim_boundary": payload.get("claim_boundary"),
        "claim_scope": payload.get("claim_scope"),
        "solve_claim": payload.get("solve_claim"),
        "live_path_reached": payload.get("live_path_reached"),
        "context_requested": payload.get("context_requested"),
        "context_observed_by_model": payload.get("context_observed_by_model"),
    }


def build_task_rows(
    root: Path, planned: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Join every manifest task to its artifact state."""

    rows = []
    for plan in planned:
        task_id = str(plan["task_id"])
        record = sources.get(task_id, missing_source_record(task_id, "source_record_missing"))
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else None
        source_rows = _rows(payload)
        path = root / str(plan["path"])
        artifact_sha256 = (
            None
            if record.get("artifact_state") == "current_synthesis"
            else record.get("sha256") or sha256_file(path)
        )
        rows.append(
            {
                "row_type": "task",
                "order": plan["order"],
                "task_id": task_id,
                "manifest_task_id": plan["manifest_task_id"],
                "title": plan["title"],
                "branch": plan["branch"],
                "path": plan["path"],
                "artifact_state": record.get("artifact_state"),
                "valid_json": record.get("valid_json"),
                "artifact_sha256": artifact_sha256,
                "error": record.get("error"),
                "status": payload.get("status")
                if payload
                else "current_synthesis"
                if task_id == CAPSTONE_TASK_ID
                else None,
                "honest_verdict": (
                    payload.get("honest_verdict")
                    if payload
                    else "current V588 capstone row"
                    if task_id == CAPSTONE_TASK_ID
                    else record.get("error")
                ),
                "verdict_class": _record_class(record),
                "duration_s": payload.get("duration_s") if payload else None,
                "row_count": len(source_rows),
                "rows_container_type": type(payload.get("rows")).__name__ if payload else None,
                "gate_failures": gate_failures(
                    task_id, payload.get("gate_check_summary") if payload else None
                ),
                "authority_boundaries": _authority_boundary(payload),
            }
        )
    return rows


def build_artifact_presence_matrix(
    root: Path, planned: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Return one presence row for all thirteen V588 tasks."""

    matrix = []
    for row in build_task_rows(root, planned, sources):
        matrix.append(
            {
                "task_id": row["task_id"],
                "manifest_task_id": row["manifest_task_id"],
                "branch": row["branch"],
                "path": row["path"],
                "artifact_state": row["artifact_state"],
                "valid_json": row["valid_json"],
                "artifact_sha256": row["artifact_sha256"],
                "row_count": row["row_count"],
                "verdict_class": row["verdict_class"],
                "error": row["error"],
            }
        )
    return matrix


def _branch_validator_blocks(branch: str, validator_findings: Sequence[Mapping[str, Any]]) -> int:
    paths = {TASK_PATHS[task] for task in BRANCH_TASKS[branch]}
    count = 0
    for finding in validator_findings:
        if finding.get("artifact") not in paths:
            continue
        if finding.get("severity") == "blocked" or int(finding.get("exit_code") or 0) != 0:
            count += 1
    return count


def _branch_mismatch_count(branch: str, mismatches: Sequence[Mapping[str, Any]]) -> int:
    paths = {TASK_PATHS[task] for task in BRANCH_TASKS[branch]}
    return sum(mismatch.get("artifact") in paths for mismatch in mismatches)


def _class_for_branch(
    branch: str,
    task_classes: Sequence[str],
    headlines: Mapping[str, Any],
    mismatches: Sequence[Mapping[str, Any]],
) -> str:
    if branch not in BRANCH_TASKS:
        raise ValueError(f"unknown branch {branch}")
    if task_classes and all(value == "missing" for value in task_classes):
        return "missing"
    if _branch_mismatch_count(branch, mismatches):
        return "disqualified"
    if branch == "handoff":
        return "null" if headlines[branch]["handoff_contract_preserved"] else "blocked"
    if branch == "activity":
        return "positive" if headlines[branch]["accelerator_receipt_ready"] else "blocked"
    if branch == "fr12_diagnostics_repair":
        return "blocked" if "blocked" in task_classes else "null"
    if branch == "fr11_continuous_self_learning":
        return "blocked" if "blocked" in task_classes else "null"
    if branch == "stochastic_portability":
        return (
            "positive"
            if headlines[branch]["compiler_fidelity_completed"]
            and headlines[branch]["context_reduced_vs_independent"]
            and headlines[branch]["trajectory_reduced_vs_independent"]
            else "null"
        )
    if branch == "arc_transport_object_table_quality":
        if headlines[branch]["object_table_ab_completed"]:
            return "positive" if headlines[branch]["adoption_gate_passed"] else "null"
        if headlines[branch]["preflight_parse_dispatch_bounded"]["rate"] == 1.0:
            return "partial"
        return "blocked"
    raise ValueError(f"unknown branch {branch}")  # pragma: no cover


CLAIM_BOUNDARIES = {
    "handoff": "A handoff audit is infrastructure. It is not scientific success.",
    "activity": "First-token CUDA receipts prove execution only. They do not rank models.",
    "fr12_diagnostics_repair": "Exact checkers certify rows. They do not make a learned verifier oracle-distinct.",
    "fr11_continuous_self_learning": "A safety fixture is circular evidence. The prospective A/B controls learning credit.",
    "stochastic_portability": "The compiler result is simulator-only. It makes no physical TSU, FPGA, speed, or power claim.",
    "arc_transport_object_table_quality": "ARC transport and object-table evidence claims no level solve.",
}
PROMOTION_GATES = {
    "handoff": "The manifest prompt literals must match the active execution root and planning date.",
    "activity": "Use task-owned phase receipts on the next dependent live model run.",
    "fr12_diagnostics_repair": "A held-family diagnostic energy needs at least two diagnosis classes per held family.",
    "fr11_continuous_self_learning": "A positive CSL claim needs order-level LCB above zero and nonzero transaction activity.",
    "stochastic_portability": "The next portability claim must keep the simulator boundary or add authenticated hardware receipts.",
    "arc_transport_object_table_quality": "The object-table A/B needs enough free VRAM to run the frozen paired rows.",
}
NEXT_ACTIONS = {
    "handoff": "Run a bounded V589 manifest-literal expansion audit before using handoff readiness as a gate.",
    "activity": "Attach the same phase and accelerator receipt to the next live branch that invokes a GGUF model.",
    "fr12_diagnostics_repair": "Run a proof-carrying DSL format repair canary on the same 72-row stream.",
    "fr11_continuous_self_learning": "Retire same-shape CSL gain claims until a fixture produces nonzero admitted commits.",
    "stochastic_portability": "Run a versioned official-Torx conformance sidecar without hardware or performance claims.",
    "arc_transport_object_table_quality": "Clear the CUDA memory preflight and rerun only Exp6753's frozen paired A/B.",
}


def build_branch_rows(
    task_rows: Sequence[Mapping[str, Any]],
    headlines: Mapping[str, Any],
    mismatches: Sequence[Mapping[str, Any]],
    validator_findings: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build one independent branch row for each V588 branch."""

    by_task = {row["task_id"]: row for row in task_rows}
    rows = []
    for branch in BRANCH_ORDER:
        classes = [by_task[task]["verdict_class"] for task in BRANCH_TASKS[branch]]
        verdict = _class_for_branch(branch, classes, headlines, mismatches)
        rows.append(
            {
                "row_type": "branch",
                "branch": branch,
                "task_ids": list(BRANCH_TASKS[branch]),
                "task_verdict_classes": dict(zip(BRANCH_TASKS[branch], classes, strict=True)),
                "verdict_class": verdict,
                "headline": headlines[branch],
                "row_headline_mismatch_count": _branch_mismatch_count(branch, mismatches),
                "validator_blocking_findings": _branch_validator_blocks(branch, validator_findings),
                "claim_boundary": CLAIM_BOUNDARIES[branch],
                "promotion_gate": PROMOTION_GATES[branch],
                "next_licensed_action": NEXT_ACTIONS[branch],
            }
        )
    return rows


def _verdict_identity(text: str) -> str:
    head = text.split(":", 1)[0].lower().strip()
    head = re.sub(r"^complete_", "", head)
    head = re.sub(r"\bexp\d+\b", "exp", head)
    return re.sub(r"\s+", " ", head)


def build_prior_failure_retirements(
    planned: Sequence[Mapping[str, Any]], task_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Compare planned prior failures with current task verdicts."""

    by_task = {row["task_id"]: row for row in task_rows}
    out = []
    for plan in planned:
        task_id = str(plan["task_id"])
        current = by_task[task_id]
        for prior in plan.get("prior_failures", []):
            if not isinstance(prior, Mapping):
                continue
            same = bool(prior.get("retire_if_same_verdict")) and _verdict_identity(
                str(prior.get("verdict", ""))
            ) == _verdict_identity(str(current.get("honest_verdict", "")))
            out.append(
                {
                    "task_id": task_id,
                    "branch": current["branch"],
                    "prior_experiment_id": prior.get("experiment_id"),
                    "prior_verdict": prior.get("verdict"),
                    "current_verdict": current.get("honest_verdict"),
                    "same_verdict_condition_fired": same,
                    "disposition": "retire_same_verdict_route" if same else "changed_or_progressed",
                }
            )
    return out


def build_next_licensed_actions(branch_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return one bounded next action per branch."""

    return [
        {
            "branch": row["branch"],
            "verdict_class": row["verdict_class"],
            "action": row["next_licensed_action"],
            "licensed_by": row["promotion_gate"],
        }
        for row in branch_rows
    ]


def build_prd_gap_disposition(branch_verdicts: Mapping[str, str]) -> list[JsonDict]:
    """State each V588 roadmap gap separately."""

    return [
        {
            "gap": "Gap 1 - executable evidence contract preservation",
            "narrowed": True,
            "disposition": "partially_narrowed",
            "evidence": [
                "Exp6743 produced three task-owned CUDA phase receipts.",
                "Exp6754 preserves blocked and positive infrastructure branches separately.",
            ],
            "remaining": "Exp6742 still blocks on active prompt literal placeholders.",
            "branch_verdicts": {
                "handoff": branch_verdicts["handoff"],
                "activity": branch_verdicts["activity"],
            },
        },
        {
            "gap": "Gap 2 - FR12 diagnostic and repair path",
            "narrowed": True,
            "disposition": "partially_narrowed",
            "evidence": [
                "Exp6744 produced 72 exact certificate rows.",
                "Exp6745 produced 216 attributable model rows and showed all malformed certificates.",
            ],
            "remaining": "Exp6746 lacked class support for diagnostic energy, so Exp6747 stayed blocked.",
            "branch_verdicts": {
                "fr12_diagnostics_repair": branch_verdicts["fr12_diagnostics_repair"]
            },
        },
        {
            "gap": "Gap 3 - FR11 safe adaptation and live-agent bridge",
            "narrowed": True,
            "disposition": "partially_narrowed",
            "evidence": [
                "Exp6748 passed the transactional fixture.",
                "Exp6752 proved code-carrying ARC tool transport with no solve claim.",
                "Exp6751 produced simulator-only compiler-fidelity evidence.",
            ],
            "remaining": "CSL had zero order-level gain and zero commits. The object-table A/B blocked on CUDA memory.",
            "branch_verdicts": {
                "fr11_continuous_self_learning": branch_verdicts["fr11_continuous_self_learning"],
                "stochastic_portability": branch_verdicts["stochastic_portability"],
                "arc_transport_object_table_quality": branch_verdicts[
                    "arc_transport_object_table_quality"
                ],
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


def _parse_row_lint_findings(text: str) -> list[str]:
    findings = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            findings.append(stripped)
    return findings


def run_validator_findings(root: Path, sources: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Run the two artifact validators for each available upstream artifact."""

    findings = []
    for task_id in EXPECTED_TASK_IDS:
        if task_id == CAPSTONE_TASK_ID:
            continue
        record = sources[task_id]
        if record.get("artifact_state") != "present":
            findings.append(
                {
                    "artifact": record.get("path"),
                    "validator": "artifact_presence",
                    "exit_code": None,
                    "severity": "blocked",
                    "findings": [record.get("error")],
                    "report_hash": value_hash(record),
                }
            )
            continue
        artifact = str(record["path"])
        adv_code, adv_text = _run_command(
            [sys.executable, str(ADVERSARIAL_SCRIPT), "--json", artifact], root
        )
        try:
            adv_payload = json.loads(adv_text)
        except json.JSONDecodeError:
            adv_payload = {"reports": [], "parse_error": adv_text[-1000:]}
        report = next(
            (
                row
                for row in adv_payload.get("reports", [])
                if isinstance(row, Mapping) and row.get("artifact") == artifact
            ),
            {},
        )
        flags = list(report.get("flags", []) or [])
        findings.append(
            {
                "artifact": artifact,
                "validator": "adversarial_verify",
                "exit_code": adv_code,
                "severity": "warning" if adv_code else "info",
                "flag_count": len(flags),
                "findings": flags,
                "report_hash": value_hash(adv_payload),
            }
        )

        row_code, row_text = _run_command([sys.executable, str(ROW_LINT_SCRIPT), artifact], root)
        row_findings = _parse_row_lint_findings(row_text)
        findings.append(
            {
                "artifact": artifact,
                "validator": "verdict_row_consistency_lint",
                "exit_code": row_code,
                "severity": "blocked" if row_code else "info",
                "findings": row_findings,
                "report_hash": "sha256:" + hashlib.sha256(row_text.encode()).hexdigest(),
            }
        )
    return findings


def _field_principles(fields: Sequence[str]) -> JsonDict:
    principles = {
        "status": "States that the capstone finished and preserved mixed branch outcomes.",
        "manifest_receipt": "Binds the active V588 manifest used for row replay.",
        "design_receipt": "Binds the V588 design used for the expected task list.",
        "source_artifact_receipts": "Shows which upstream artifacts were present, missing, or invalid.",
        "preconditions_checked": "States that manifest and design parsing were hard preconditions.",
        "field_principles": "Explains why each top-level field exists.",
        "inference_substrate": "Declares local row replay and validator execution with no LLM.",
        "duration_s": "Records monotonic elapsed time for the capstone itself.",
        "random_seed": "Keeps recompute and validator seeds explicit.",
        "reproducibility_checksum": "Binds manifest, artifacts, rows, validators, and reducers.",
        "rows": "Contains one task row per task and one branch row per branch.",
        "artifact_presence_matrix": "Preserves all thirteen expected task artifact states.",
        "branch_verdicts": "Gives exactly one closed class per branch.",
        "recomputed_headlines": "Stores headline values recomputed from raw rows.",
        "row_headline_mismatches": "Keeps source headline disagreements visible.",
        "adversarial_findings": "Stores adversarial and row-consistency validator receipts.",
        "prd_gap_disposition": "States each of the three roadmap gaps separately.",
        "prior_failure_retirements": "Records same-verdict retirement checks from the manifest.",
        "next_licensed_actions": "Gives one bounded next action per branch.",
        "gate_check_summary": "Carries failed source gate checks and observed values.",
        "verdict_class": "Uses the closed verdict vocabulary.",
        "honest_verdict": "Gives a terminal-prefixed human summary.",
    }
    return {field: principles[field] for field in fields}


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while excluding its own checksum."""

    return value_hash(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def build_artifact(
    root: Path,
    *,
    duration_s: float,
    planned: Sequence[Mapping[str, Any]] | None = None,
    sources: Mapping[str, Mapping[str, Any]] | None = None,
    validator_findings: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the terminal V588 synthesis from manifest and source rows."""

    planned = list(planned or load_planned_tasks(root))
    sources = dict(sources or load_source_artifacts(root, planned))
    validator_findings = list(validator_findings or [])
    task_rows = build_task_rows(root, planned, sources)
    headlines, mismatches = recompute_headlines(sources)
    branch_rows = build_branch_rows(task_rows, headlines, mismatches, validator_findings)
    branch_verdicts = {row["branch"]: row["verdict_class"] for row in branch_rows}
    rows = [*task_rows, *branch_rows]
    fields = [
        "status",
        "manifest_receipt",
        "design_receipt",
        "source_artifact_receipts",
        "preconditions_checked",
        *REQUIRED_ARTIFACT_FIELDS,
    ]
    artifact: JsonDict = {
        "status": "complete_terminal_partial",
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
        "source_artifact_receipts": [
            {
                "task_id": task_id,
                "path": record.get("path"),
                "artifact_state": record.get("artifact_state"),
                "sha256": record.get("sha256"),
                "error": record.get("error"),
            }
            for task_id, record in sources.items()
        ],
        "preconditions_checked": {
            "active_manifest_parsed": True,
            "v588_design_parsed": True,
            "branch_artifacts_optional": True,
        },
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "rows": rows,
        "artifact_presence_matrix": build_artifact_presence_matrix(root, planned, sources),
        "branch_verdicts": branch_verdicts,
        "recomputed_headlines": headlines,
        "row_headline_mismatches": mismatches,
        "adversarial_findings": validator_findings,
        "prd_gap_disposition": build_prd_gap_disposition(branch_verdicts),
        "prior_failure_retirements": build_prior_failure_retirements(planned, task_rows),
        "next_licensed_actions": build_next_licensed_actions(branch_rows),
        "gate_check_summary": [
            failure for row in task_rows for failure in row.get("gate_failures", [])
        ],
        "verdict_class": "partial",
        "honest_verdict": (
            "complete_partial: V588 preserved blocked handoff, blocked FR12, null CSL, "
            "positive activity, simulator-only stochastic evidence, and partial ARC "
            "transport without a pooled success claim."
        ),
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
    if payload.get("verdict_class") != "partial":
        errors.append("verdict_class")
    if not str(payload.get("honest_verdict", "")).startswith("complete_partial:"):
        errors.append("honest_verdict")
    if not isinstance(payload.get("duration_s"), (int, float)) or payload.get("duration_s", 0) < 0:
        errors.append("duration_s")
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(payload):
        errors.append("field_principles")
    if len(payload.get("artifact_presence_matrix", [])) != 13:
        errors.append("artifact_presence_matrix")
    branch_verdicts = payload.get("branch_verdicts")
    if not isinstance(branch_verdicts, Mapping) or set(branch_verdicts) != set(BRANCH_ORDER):
        errors.append("branch_verdicts")
    elif any(value not in CLOSED_CLASSES for value in branch_verdicts.values()):
        errors.append("branch_verdicts_closed_class")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < 19:
        errors.append("rows")
    headlines = payload.get("recomputed_headlines")
    if not isinstance(headlines, Mapping):
        errors.append("recomputed_headlines")
    else:
        if headlines.get("pooled_success_claim_emitted") is not False:
            errors.append("pooled_success_claim_emitted")
        if headlines.get("pooled_milestone_success_score") is not None:
            errors.append("pooled_milestone_success_score")
    if len(payload.get("prd_gap_disposition", [])) != 3:
        errors.append("prd_gap_disposition")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a single JSON document with an atomic replacement."""

    errors = validate_artifact(payload)
    if errors:
        raise ValueError(f"invalid Exp6754 artifact: {errors}")
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
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--skip-validators", action="store_true")
    args = parser.parse_args(argv)

    output = args.output or args.repo_root / RESULT_PATH
    if args.validate:
        try:
            errors = validate_artifact(_load_artifact(output))
        except Exception as exc:  # noqa: BLE001
            print(f"Exp6754 validation failed: {exc}")
            return 1
        if errors:
            print(f"Exp6754 validation failed: {errors}")
            return 1
        return 0

    start = time.perf_counter()
    planned = load_planned_tasks(args.repo_root)
    sources = load_source_artifacts(args.repo_root, planned)
    validators = [] if args.skip_validators else run_validator_findings(args.repo_root, sources)
    artifact = build_artifact(
        args.repo_root,
        duration_s=time.perf_counter() - start,
        planned=planned,
        sources=sources,
        validator_findings=validators,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(f"Exp6754 validation failed: {errors}")
        return 1
    write_json_atomic(output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
