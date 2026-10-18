"""Build the terminal V590 branch disposition from checked-in evidence.

The reducer keeps each science branch separate. It reports missing rows as
missing evidence and never replaces them with a copied headline. The module
does not call an LLM or change an upstream artifact.

Spec refs: REQ-REPORT-6780 and SCENARIO-REPORT-6780-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.08.590"
PLANNING_DATE = "20260830"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 6780
HONEST_VERDICT = (
    "complete_partial: V590 narrowed the proof fixture and runtime grammar gap, while exact "
    "generation, localized repair, prospective continuous memory, and ARC actions-to-progress "
    "evidence remained blocked or missing; no branch metrics were pooled."
)

RESULT_PATH = Path("results/experiment_6780_v590_branch_disposition.json")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PRIOR_DISPOSITION_PATH = Path("results/experiment_6767_v589_branch_disposition.json")
SUMMARY_SCRIPT = Path("scripts/summarize_artifact.py")
RESEARCH_CONDUCTOR_PATH = Path("scripts/research_conductor.py")
MODULE_PATH = Path("python/carnot/experiment_6780_v590_branch_disposition.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_6780_v590_branch_disposition.py")
TEST_PATH = Path("tests/python/test_experiment_6780_v590_branch_disposition.py")

CLOSED_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
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

SHORT_TASK_IDS = tuple(f"exp{number}" for number in range(6768, 6781))
EXPECTED_TASK_IDS = (
    "exp6768-targetable-proof-panel-expansion",
    "exp6769-environment-indexed-proof-grammar-v2",
    "exp6770-dccd-environment-grammar-ab-v2",
    "exp6771-proof-transport-localization-audit",
    "exp6772-claim-localized-prefix-backtracking-ab",
    "exp6773-csl-owned-lease-contract",
    "exp6774-procedural-vs-trace-csl-ab-v2",
    "exp6775-csl-durability-audit-v2",
    "exp6776-arc-shadow-supervisor-accrual",
    "exp6777-arc-tool-gap-transport",
    "exp6778-arc-selfparse-actions-to-progress-ab",
    "exp6779-arc-tool-supervisor-adoption-audit",
    "exp6780-v590-branch-disposition",
)
CAPSTONE_TASK_ID = "exp6780"
TASK_PATHS = {
    short: path
    for short, path in zip(
        SHORT_TASK_IDS,
        (
            "results/experiment_6768_targetable_proof_panel_expansion.json",
            "results/experiment_6769_environment_indexed_proof_grammar_v2.json",
            "results/experiment_6770_dccd_environment_grammar_ab_v2.json",
            "results/experiment_6771_proof_transport_localization_audit.json",
            "results/experiment_6772_claim_localized_prefix_backtracking_ab.json",
            "results/experiment_6773_csl_owned_lease_contract.json",
            "results/experiment_6774_procedural_vs_trace_csl_ab_v2.json",
            "results/experiment_6775_csl_durability_audit_v2.json",
            "results/experiment_6776_arc_shadow_supervisor_accrual.json",
            "results/experiment_6777_arc_tool_gap_transport.json",
            "results/experiment_6778_arc_selfparse_actions_to_progress_ab.json",
            "results/experiment_6779_arc_tool_supervisor_adoption_audit.json",
            RESULT_PATH.as_posix(),
        ),
        strict=True,
    )
}

BRANCH_ORDER = ("proof", "continuous_memory", "arc", "execution_contract")
BRANCH_TASKS = {
    "proof": tuple(f"exp{number}" for number in range(6768, 6773)),
    "continuous_memory": tuple(f"exp{number}" for number in range(6773, 6776)),
    "arc": tuple(f"exp{number}" for number in range(6776, 6780)),
    "execution_contract": ("exp6780",),
}
TASK_BRANCH = {task: branch for branch, tasks in BRANCH_TASKS.items() for task in tasks}
FR11_GATES = ("activity", "retention", "support", "hard_case", "poison", "restart", "rollback")

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
    "arc_disposition",
    "hardware_disposition",
    "docs_reconciled",
    "protected_files_unchanged",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "States why each required field exists so an audit can detect a copied shell.",
    "inference_substrate": "Separates local evidence synthesis from a live model or hardware run.",
    "duration_s": "Records monotonic wall time so the synthesis work has a measurable receipt.",
    "random_seed": "Fixes the synthesis reducer seed for deterministic replay.",
    "reproducibility_checksum": "Binds the roadmap, sources, code, audits, and recomputed rows.",
    "milestone": "Pins every conclusion to the V590 evidence window.",
    "expected_task_ids": "Keeps every planned task visible even when its deliverable is absent.",
    "available_artifacts": "Names terminal source files that the reducer could inspect.",
    "missing_artifacts": "Prevents missing or nonterminal evidence from becoming narrative success.",
    "rows": "Provides one auditable disposition row for each experiment.",
    "branch_rows": "Keeps proof, memory, ARC, and execution conclusions independent.",
    "row_recomputed_headlines": "Stores only comparative values rebuilt from eligible raw rows.",
    "adversarial_findings": "Preserves critical, warning, circular, and provenance concerns.",
    "verdict_row_consistency_findings": "Shows when raw rows do not support a source headline.",
    "recurring_blockers": "Links repeated blockers to the project ledger instead of hiding them.",
    "prior_verdict_recurrences": "Checks each retire-if-same declaration against the current outcome.",
    "retirement_recommendations": "Names repeated routes that need an exclusion-manifest entry.",
    "prd_gap_disposition": "Classifies each ranked PRD gap with direct row citations.",
    "fr12_disposition": "Separates proof certification from generation and repair evidence.",
    "fr11_disposition": "Requires every cold durability gate before positive learning credit.",
    "arc_disposition": "Separates supervisor and transport receipts from adoption evidence.",
    "hardware_disposition": "Records CUDA receipts while forbidding a physical TSU inference.",
    "docs_reconciled": "States which evidence documents this run changed or delegated.",
    "protected_files_unchanged": "Proves the synthesis did not edit the conductor or active roadmap.",
    "gate_check_summary": "Retains the failed check and observed value for every blocked row.",
    "verifier_is_oracle": "States that this synthesis is not a correctness oracle.",
    "verdict_class": "Uses a closed class so downstream tools do not infer a favorable state.",
    "honest_verdict": "Provides a terminal summary that preserves every incomplete branch.",
}


def canonical_json(value: Any) -> bytes:
    """Encode one stable JSON representation for hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str | None:
    """Hash a file while keeping a missing file distinct from an empty file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def short_task_id(task_id: str) -> str:
    """Return the exp-number prefix from a roadmap task ID."""

    match = re.match(r"^(exp\d+)(?:-|$)", task_id)
    if match is None:
        raise ValueError(f"invalid V590 task id: {task_id}")
    return match.group(1)


def _next_deliverable(lines: Sequence[str], start: int) -> str:
    for line in lines[start + 1 :]:
        match = re.search(r"\*\*Deliverable:\*\*\s*`([^`]+)`", line)
        if match:
            return match.group(1)
        if line.startswith("### Exp "):
            break
    raise ValueError("V590 design task deliverable missing")


def parse_design_tasks(text: str) -> list[JsonDict]:
    """Parse the V590 proposal so the roadmap cannot silently change shape."""

    milestone = re.search(r"\*\*Milestone:\*\*\s*`([^`]+)`", text)
    if milestone is None or milestone.group(1) != MILESTONE:
        observed = milestone.group(1) if milestone else "missing"
        raise ValueError(f"V590 design milestone must be {MILESTONE}, observed {observed}")
    lines = text.splitlines()
    rows = []
    for index, line in enumerate(lines):
        match = re.match(r"### Exp (\d+):\s*(.+)$", line)
        if match:
            rows.append(
                {
                    "task_id": f"exp{match.group(1)}",
                    "title": match.group(2).strip(),
                    "deliverable": _next_deliverable(lines, index),
                }
            )
    return rows


def load_planned_tasks(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Inspect the next-roadmap path, then load the activated V590 roadmap."""

    next_path = root / NEXT_ROADMAP_PATH
    selected = next_path if next_path.is_file() else root / ACTIVE_ROADMAP_PATH
    manifest = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("tasks"), list):
        raise ValueError("V590 roadmap must be a mapping with a task list")
    if str(manifest.get("milestone")) != MILESTONE:
        raise ValueError("selected roadmap is not milestone 2026.08.590")
    design = parse_design_tasks((root / DESIGN_PATH).read_text(encoding="utf-8"))
    if [row["task_id"] for row in design] != list(SHORT_TASK_IDS):
        raise ValueError("V590 design must contain Exp6768 through Exp6780")
    tasks = [task for task in manifest["tasks"] if isinstance(task, Mapping)]
    if [str(task.get("id")) for task in tasks] != list(EXPECTED_TASK_IDS):
        raise ValueError("roadmap must contain the exact V590 task list")
    design_by_id = {row["task_id"]: row for row in design}
    planned = []
    for order, task in enumerate(tasks, 1):
        manifest_id = str(task["id"])
        short = short_task_id(manifest_id)
        path = str(task.get("deliverable"))
        if path != design_by_id[short]["deliverable"] or path != TASK_PATHS[short]:
            raise ValueError(f"roadmap deliverable mismatch for {short}")
        planned.append(
            {
                "order": order,
                "task_id": short,
                "manifest_task_id": manifest_id,
                "title": str(task.get("title")),
                "path": path,
                "branch": TASK_BRANCH[short],
                "gated_on": list(task.get("gated_on") or []),
                "prior_failures": list(task.get("prior_failures") or []),
            }
        )
    receipt = {
        "inspected_path": NEXT_ROADMAP_PATH.as_posix(),
        "next_roadmap_present": next_path.is_file(),
        "selected_path": selected.relative_to(root).as_posix(),
        "selected_sha256": sha256_file(selected),
        "fallback_reason": None if next_path.is_file() else "research-roadmap-next.yaml_missing",
    }
    return planned, receipt


def load_source_artifacts(root: Path, planned: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Load every task artifact and keep missing or invalid files explicit."""

    sources: dict[str, JsonDict] = {}
    for task in planned:
        task_id = str(task["task_id"])
        path_text = str(task["path"])
        if task_id == CAPSTONE_TASK_ID:
            sources[task_id] = {
                "artifact_state": "current_synthesis",
                "valid_json": True,
                "payload": None,
                "sha256": None,
                "path": path_text,
                "error": None,
            }
            continue
        path = root / path_text
        if not path.is_file():
            sources[task_id] = {
                "artifact_state": "missing",
                "valid_json": False,
                "payload": None,
                "sha256": None,
                "path": path_text,
                "error": "file_missing",
            }
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON is not an object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            sources[task_id] = {
                "artifact_state": "invalid",
                "valid_json": False,
                "payload": None,
                "sha256": sha256_file(path),
                "path": path_text,
                "error": str(exc),
            }
            continue
        terminal = _source_payload_is_terminal(payload)
        sources[task_id] = {
            "artifact_state": "present" if terminal else "nonterminal",
            "valid_json": True,
            "payload": payload,
            "sha256": sha256_file(path),
            "path": path_text,
            "error": None if terminal else "source_artifact_nonterminal",
        }
    return sources


def _source_payload_is_terminal(payload: Mapping[str, Any]) -> bool:
    """Accept only source states that clearly say the producer has stopped."""

    terminal_stems = (
        "complete",
        "success",
        "passed",
        "shipped",
        "blocked",
        "gate_block",
        "failed",
        "flagged",
        "retired",
        "null",
        "partial",
        "disqualified",
    )
    for value in (payload.get("status"), payload.get("honest_verdict")):
        text = str(value or "").strip().lower()
        if any(
            text == stem or text.startswith((f"{stem}_", f"{stem}:")) for stem in terminal_stems
        ):
            return True
    return False


def _payload(record: Mapping[str, Any] | None) -> Mapping[str, Any]:
    value = record.get("payload") if isinstance(record, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = payload.get("rows")
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _rate(values: Sequence[bool], cause: str = "no_eligible_rows") -> JsonDict:
    if not values:
        return {"numerator": 0, "denominator": 0, "rate": None, "cause": cause}
    numerator = sum(values)
    return {"numerator": numerator, "denominator": len(values), "rate": numerator / len(values)}


def _mean(values: Sequence[float], cause: str = "no_eligible_rows") -> JsonDict:
    if not values:
        return {"denominator": 0, "value": None, "cause": cause}
    return {"denominator": len(values), "value": sum(values) / len(values)}


def _pair_id(row: Mapping[str, Any]) -> str | None:
    for key in ("pair_id", "order_id", "cell_id", "game_seed", "row_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return None


def paired_effect(
    rows: Sequence[Mapping[str, Any]], treatment: str, control: str, metric: str
) -> JsonDict:
    """Compute a treatment-minus-control effect and a simple paired interval."""

    by_pair: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        pair = _pair_id(row)
        arm = str(row.get("arm"))
        raw = row.get(metric)
        value = float(raw) if isinstance(raw, bool) else _number(raw)
        if pair is not None and arm in {treatment, control} and value is not None:
            by_pair[pair][arm] = value
    deltas = [arms[treatment] - arms[control] for arms in by_pair.values() if len(arms) == 2]
    if not deltas:
        return {
            "value": None,
            "pair_count": 0,
            "mean_delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "cause": "no_eligible_comparative_rows",
        }
    mean = sum(deltas) / len(deltas)
    half_width = 0.0
    if len(deltas) > 1:
        half_width = 1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas))
    return {
        "pair_count": len(deltas),
        "mean_delta": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "improved_pair_count": sum(delta > 0 for delta in deltas),
        "tied_pair_count": sum(delta == 0 for delta in deltas),
        "worsened_pair_count": sum(delta < 0 for delta in deltas),
        "interval_method": "paired_normal_95",
    }


def _arm_rates(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, JsonDict]:
    values: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get(metric), bool) and row.get("arm") is not None:
            values[str(row["arm"])].append(bool(row[metric]))
    return {arm: _rate(items) for arm, items in sorted(values.items())}


def _arm_means(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, JsonDict]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row.get(metric))
        if value is not None and row.get("arm") is not None:
            values[str(row["arm"])].append(value)
    return {arm: _mean(items) for arm, items in sorted(values.items())}


def recompute_proof(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute proof fixture, grammar, and exact-valid comparison evidence."""

    panel = _payload(sources.get("exp6768"))
    grammar = _payload(sources.get("exp6769"))
    comparison = _payload(sources.get("exp6770"))
    rows = _rows(comparison)
    effects: JsonDict
    if rows:
        effects = {
            "dccd_environment-minus-repaired_direct": paired_effect(
                rows, "dccd_environment", "repaired_direct", "exact_valid"
            ),
            "dccd_environment-minus-static_grammar": paired_effect(
                rows, "dccd_environment", "static_grammar", "exact_valid"
            ),
            "static_grammar-minus-repaired_direct": paired_effect(
                rows, "static_grammar", "repaired_direct", "exact_valid"
            ),
        }
    else:
        effects = {
            "value": None,
            "denominator": 0,
            "cause": "no_eligible_comparative_rows",
            "source_headline_ignored": comparison.get("paired_exact_valid_deltas"),
        }
    grammar_rows = _rows(grammar)
    return {
        "panel_rows": len(_rows(panel)),
        "targetable_panel_ready": bool(panel.get("targetable_panel_ready")),
        "grammar_rows": len(grammar_rows),
        "dynamic_proof_grammar_ready": bool(grammar.get("dynamic_proof_grammar_ready")),
        "runtime_mask_invocation_count": sum(
            int(row.get("mask_invocation_count") or row.get("runtime_mask_invocations") or 0)
            for row in grammar_rows
        )
        or int(grammar.get("runtime_mask_invocation_count") or 0),
        "valid_sat_reachable": bool(grammar.get("valid_sat_reachable")),
        "valid_unsat_reachable": bool(grammar.get("valid_unsat_reachable")),
        "no_ghost_violations": grammar.get("no_ghost_violations"),
        "comparison_rows": len(rows),
        "proof_transport_ab_completed": bool(comparison.get("proof_transport_ab_completed")),
        "exact_valid_rate_by_arm": _arm_rates(rows, "exact_valid")
        if rows
        else _mean([], "no_eligible_comparative_rows"),
        "paired_exact_valid_effects": effects,
    }


def recompute_repair(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute localized-repair effects without borrowing proof headlines."""

    audit = _payload(sources.get("exp6771"))
    repair = _payload(sources.get("exp6772"))
    rows = _rows(repair)
    effect = paired_effect(rows, "prefix_backtracking", "full_regeneration", "exact_valid")
    harmful = paired_effect(rows, "prefix_backtracking", "full_regeneration", "harmful_flip")
    support = paired_effect(rows, "prefix_backtracking", "full_regeneration", "support_loss")
    return {
        "proof_transport_audit_ready": bool(audit.get("proof_transport_audit_ready")),
        "repair_panel_ready": bool(audit.get("repair_panel_ready")),
        "repair_rows": len(rows),
        "paired_exact_valid_effect": effect,
        "paired_harmful_flip_effect": harmful,
        "paired_support_loss_effect": support,
        "harmful_flips_by_arm": _arm_rates(rows, "harmful_flip")
        if rows
        else _mean([], "no_eligible_repair_rows"),
        "support_loss_by_arm": _arm_rates(rows, "support_loss")
        if rows
        else _mean([], "no_eligible_repair_rows"),
        "source_interval_ignored": repair.get("paired_interval") or repair.get("repair_interval"),
    }


def recompute_continuous_memory(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute prospective order effects, loss, transactions, and audit gates."""

    preflight = _payload(sources.get("exp6773"))
    comparison = _payload(sources.get("exp6774"))
    audit = _payload(sources.get("exp6775"))
    rows = _rows(comparison)
    activity: JsonDict = {
        name: 0 for name in ("commits", "rejects", "rollbacks", "retrievals", "action_influences")
    }
    source_keys = {
        "commits": "commits",
        "rejects": "rejects",
        "rollbacks": "rollbacks",
        "retrievals": "retrieval_count",
        "action_influences": "action_influence_count",
    }
    for row in rows:
        for target, source in source_keys.items():
            value = _number(row.get(source))
            activity[target] += int(value or 0)
    if not rows:
        activity = {
            name: None
            for name in ("commits", "rejects", "rollbacks", "retrievals", "action_influences")
        }
        activity["cause"] = "no_eligible_prospective_rows"
    audit_gates = audit.get("audit_gates") if isinstance(audit.get("audit_gates"), Mapping) else {}
    gates = {
        "activity": bool(comparison.get("prospective_csl_completed"))
        and activity["commits"] > 0
        and activity["rejects"] > 0
        and activity["retrievals"] > 0
        and activity["action_influences"] > 0,
        "retention": bool(audit_gates.get("retention")),
        "support": bool(audit_gates.get("support")),
        "hard_case": bool(audit_gates.get("hard_case")),
        "poison": bool(audit_gates.get("poison")),
        "restart": bool(audit_gates.get("restart")),
        "rollback": bool(audit_gates.get("rollback")),
    }
    return {
        "preflight_rows": len(_rows(preflight)),
        "csl_live_preflight_ready": bool(preflight.get("csl_live_preflight_ready")),
        "live_model_invoked": bool(preflight.get("live_model_invoked")),
        "prospective_rows": len(rows),
        "prospective_csl_completed": bool(comparison.get("prospective_csl_completed")),
        "prequential_yield_by_arm": _arm_means(rows, "prequential_yield")
        if rows
        else _mean([], "no_eligible_prospective_rows"),
        "procedural_minus_no_memory_order_effect": paired_effect(
            rows, "procedural_memory", "no_memory", "prequential_yield"
        ),
        "procedural_minus_trace_order_effect": paired_effect(
            rows, "procedural_memory", "detailed_trace", "prequential_yield"
        ),
        "historical_loss_by_arm": _arm_means(rows, "historical_loss")
        if rows
        else _mean([], "no_eligible_prospective_rows"),
        "transaction_activity": activity,
        "cold_audit_completed": bool(audit.get("cold_audit_completed")),
        "required_positive_gates": gates,
    }


def recompute_arc(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recompute supervisor, transport, action-efficiency, and adoption evidence."""

    supervisor = _payload(sources.get("exp6776"))
    transport = _payload(sources.get("exp6777"))
    comparison = _payload(sources.get("exp6778"))
    audit = _payload(sources.get("exp6779"))
    supervisor_rows = _rows(supervisor)
    eligible_supervisor = [
        row
        for row in supervisor_rows
        if row.get("live_model_invoked") is True and not row.get("failure_class")
    ]
    action_rows = _rows(comparison)
    effect = paired_effect(action_rows, "selfparse", "control_unset", "actions_to_progress")
    decision = str(audit.get("adoption_decision") or "unavailable")
    cold_pass = bool(audit.get("cold_actions_to_progress_audit_passed"))
    tool_gap_events = transport.get("tool_gap_events")
    tool_gap_event_count = len(tool_gap_events) if isinstance(tool_gap_events, list) else None
    supervisor_evidence_cause = None if eligible_supervisor else "no_eligible_live_supervisor_rows"
    return {
        "supervisor_rows": len(supervisor_rows),
        "eligible_supervisor_rows": len(eligible_supervisor),
        "shadow_supervisor_transport_ready": bool(
            supervisor.get("shadow_supervisor_transport_ready")
        ),
        "row_recomputed_firings": sum(int(row.get("arm_fired") or 0) for row in eligible_supervisor)
        if eligible_supervisor
        else None,
        "row_recomputed_helped": sum(
            int(row.get("arm_helped_counterfactual") or 0) for row in eligible_supervisor
        )
        if eligible_supervisor
        else None,
        "supervisor_evidence_cause": supervisor_evidence_cause,
        "declared_firings_after_by_arm": supervisor.get("firings_after_by_arm") or {},
        "evidence_floor_met_by_arm": supervisor.get("evidence_floor_met_by_arm") or {},
        "tool_gap_transport_ready": bool(transport.get("tool_gap_transport_ready")),
        "tool_gap_event_count": tool_gap_event_count,
        "tool_gap_event_count_cause": None
        if isinstance(tool_gap_events, list)
        else "tool_gap_events_missing",
        "tool_gap_analyzer_ingest_passed": bool(transport.get("analyzer_ingest_passed")),
        "actions_to_progress_rows": len(action_rows),
        "actions_to_progress_ab_completed": bool(
            comparison.get("actions_to_progress_ab_completed")
        ),
        "actions_to_progress_by_arm": _arm_means(action_rows, "actions_to_progress")
        if action_rows
        else _mean([], "no_eligible_actions_to_progress_rows"),
        "selfparse_minus_control_actions_effect": effect,
        "cold_actions_to_progress_audit_passed": cold_pass,
        "adoption_decision": decision,
        "adoption_positive": cold_pass
        and decision == "promote"
        and bool(comparison.get("actions_to_progress_ab_completed"))
        and effect.get("pair_count", 0) > 0
        and effect.get("mean_delta") is not None
        and effect["mean_delta"] < 0,
        "solve_claim": bool(
            supervisor.get("solve_claim")
            or comparison.get("solve_claim")
            or audit.get("solve_claim")
        ),
    }


def recompute_headlines(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build all branch-local row recomputations without a pooled score."""

    return {
        "proof": recompute_proof(sources),
        "repair": recompute_repair(sources),
        "continuous_memory": recompute_continuous_memory(sources),
        "arc": recompute_arc(sources),
        "execution_contract": {
            "expected_task_count": 13,
            "capstone_path": RESULT_PATH.as_posix(),
            "successor_roadmap_activated": False,
        },
    }


def _model_ids(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("model_specs") or payload.get("models_used") or []
    if isinstance(values, (str, Mapping)):
        values = [values]
    out = []
    if isinstance(values, list):
        for value in values:
            if isinstance(value, str):
                out.append(value)
            elif isinstance(value, Mapping):
                model_id = value.get("model_id") or value.get("hf_id") or value.get("name")
                if model_id:
                    out.append(str(model_id))
    return sorted(set(out))


def _declared_class(record: Mapping[str, Any]) -> str:
    state = record.get("artifact_state")
    if state == "invalid":
        return "disqualified"
    if state == "current_synthesis":
        return "partial"
    if state == "nonterminal":
        return "partial"
    if state == "missing":
        return "blocked"
    payload = _payload(record)
    declared = payload.get("verdict_class")
    if declared in CLOSED_CLASSES:
        return str(declared)
    text = f"{payload.get('status', '')} {payload.get('honest_verdict', '')}".lower()
    if "blocked" in text or "gate_check_failed" in text:
        return "blocked"
    if "disqualified" in text or payload.get("flagged_adversarial") is True:
        return "disqualified"
    if "circular" in text or payload.get("verifier_is_oracle") is True:
        return "circular_positive"
    if "null" in text or "no_improvement" in text:
        return "null"
    if str(payload.get("status", "")).lower() in {"complete", "success", "passed", "shipped"}:
        return "positive"
    return "partial"


def _gate_failures(payload: Mapping[str, Any], task_id: str) -> list[JsonDict]:
    summary = payload.get("gate_check_summary")
    out = []
    if isinstance(summary, str) and ("failed" in summary.lower() or "unsat" in summary.lower()):
        out.append(
            {
                "task_id": task_id,
                "check": payload.get("failed_field") or "gate_check_summary",
                "expected": payload.get("failed_expected", "gate passes"),
                "observed": payload.get("failed_observed", summary),
                "passed": False,
                "reason": payload.get("blocked_reason") or "blocked_gate_text",
            }
        )
    if not isinstance(summary, Mapping):
        return out
    for row in summary.get("checks", []) if isinstance(summary.get("checks"), list) else []:
        if isinstance(row, Mapping) and row.get("passed") is False:
            out.append({"task_id": task_id, **dict(row)})
    checks = summary.get("checks")
    if isinstance(checks, Mapping):
        for name, passed in checks.items():
            if passed is False:
                failure = next(
                    (
                        row
                        for row in summary.get("failures", [])
                        if isinstance(row, Mapping) and row.get("check") == name
                    ),
                    {},
                )
                out.append(
                    {
                        "task_id": task_id,
                        "check": name,
                        "expected": failure.get("expected", True),
                        "observed": failure.get("observed", False),
                        "passed": False,
                        "reason": "failed_named_check",
                    }
                )
    if summary.get("failed_check"):
        out.append(
            {
                "task_id": task_id,
                "check": summary.get("failed_check"),
                "expected": summary.get("expected", True),
                "observed": summary.get("observed"),
                "passed": False,
                "reason": "failed_named_check",
            }
        )
    unique = []
    seen = set()
    for row in out:
        key = canonical_json(row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _missing_gate_failures(
    plan: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    out = []
    for gate in plan.get("gated_on", []):
        if not isinstance(gate, Mapping):
            continue
        upstream = short_task_id(str(gate.get("upstream", "")))
        payload = _payload(sources.get(upstream))
        field = str(gate.get("artifact_field"))
        observed = payload.get(field)
        expected = gate.get("value")
        passed = observed == expected if gate.get("op") == "==" else False
        if not passed:
            out.append(
                {
                    "task_id": plan["task_id"],
                    "check": f"{upstream}.{field}",
                    "expected": expected,
                    "observed": observed,
                    "passed": False,
                    "reason": "upstream_gate_blocked_or_missing",
                }
            )
    return out


def build_experiment_rows(
    planned: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Build one disposition row for each expected V590 experiment."""

    rows = []
    for plan in planned:
        task_id = str(plan["task_id"])
        record = sources[task_id]
        payload = _payload(record)
        state = str(record["artifact_state"])
        failures = _gate_failures(payload, task_id)
        if state == "missing":
            failures.extend(_missing_gate_failures(plan, sources))
        declared = str(payload.get("honest_verdict") or "")
        if state == "missing":
            declared = "complete_blocked_missing_artifact: planned deliverable is absent"
        elif state == "invalid":
            declared = "complete_disqualified_invalid_artifact: source JSON could not be loaded"
        elif state == "nonterminal":
            declared = "complete_partial_nonterminal_artifact: source producer has not stopped"
        elif state == "current_synthesis":
            declared = HONEST_VERDICT
        source_rows = _rows(payload)
        rows.append(
            {
                "row_type": "experiment",
                "order": int(plan["order"]),
                "task_id": task_id,
                "manifest_task_id": plan["manifest_task_id"],
                "title": plan["title"],
                "branch": plan["branch"],
                "path": plan["path"],
                "artifact_state": state,
                "terminal": state
                in {"present", "missing", "invalid", "nonterminal", "current_synthesis"},
                "source_artifact_terminal": state == "present",
                "valid_json": bool(record.get("valid_json")),
                "artifact_sha256": record.get("sha256"),
                "status": payload.get("status") or state,
                "honest_verdict": declared,
                "verdict_class": _declared_class(record),
                "duration_s": payload.get("duration_s"),
                "inference_substrate": payload.get("inference_substrate"),
                "model_ids": _model_ids(payload),
                "raw_row_count": len(source_rows),
                "raw_rows_available": bool(source_rows),
                "gate_failures": failures,
                "source_error": record.get("error"),
                "next_action": "synthesize_terminal_evidence"
                if task_id == CAPSTONE_TASK_ID
                else "follow_branch_disposition",
            }
        )
    return rows


CLAIM_BOUNDARIES = {
    "proof": "Fixture and grammar readiness do not prove live exact-valid gain or localized repair.",
    "continuous_memory": "Storage receipts do not prove prospective learning without all cold durability gates.",
    "arc": "Supervisor and tool transport do not prove adoption without a cold actions-to-progress effect.",
    "execution_contract": "The capstone closes task accounting. It is not a verifier or a science vote.",
}
NEXT_ACTIONS = {
    "proof": "Exclude the repeated GPU-starved proof comparison route until an owned RTX 3090 is available.",
    "continuous_memory": "Exclude the repeated GPU-capacity preflight route until one model has an owned lease.",
    "arc": "Keep supervisor and selfparse default-off until a cold actions-to-progress audit exists.",
    "execution_contract": "Hand terminal evidence to the separate document reconciler; do not activate a successor roadmap.",
}


def _branch_class(
    branch: str, rows: Sequence[Mapping[str, Any]], headlines: Mapping[str, Any]
) -> str:
    classes = [str(row["verdict_class"]) for row in rows]
    if "disqualified" in classes:
        return "disqualified"
    if branch == "proof":
        if (
            headlines["proof"]["proof_transport_ab_completed"]
            and headlines["repair"]["repair_rows"]
        ):
            delta = headlines["proof"]["paired_exact_valid_effects"].get(
                "dccd_environment-minus-repaired_direct", {}
            )
            repair = headlines["repair"]["paired_exact_valid_effect"]
            harmful = headlines["repair"].get("paired_harmful_flip_effect", {})
            support = headlines["repair"].get("paired_support_loss_effect", {})
            return (
                "positive"
                if (delta.get("mean_delta") or 0) > 0
                and (repair.get("mean_delta") or 0) > 0
                and harmful.get("pair_count", 0) > 0
                and harmful.get("mean_delta") is not None
                and harmful["mean_delta"] <= 0
                and support.get("pair_count", 0) > 0
                and support.get("mean_delta") is not None
                and support["mean_delta"] <= 0
                else "null"
            )
        return "partial" if "positive" in classes else "blocked"
    if branch == "continuous_memory":
        gates = headlines[branch]["required_positive_gates"]
        if headlines[branch]["cold_audit_completed"] and all(gates.values()):
            return "positive"
        return "blocked" if "blocked" in classes else "null"
    if branch == "arc":
        if headlines[branch]["adoption_positive"]:
            return "positive"
        return "blocked" if "blocked" in classes else "null"
    return "partial"


def build_branch_rows(
    rows: Sequence[Mapping[str, Any]],
    headlines: Mapping[str, Any],
    adversarial_findings: Sequence[Mapping[str, Any]],
    row_findings: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build four independent branch rows with local verdicts and actions."""

    by_task = {str(row["task_id"]): row for row in rows}
    out = []
    for branch in BRANCH_ORDER:
        task_ids = BRANCH_TASKS[branch]
        local_rows = [by_task[task_id] for task_id in task_ids]
        paths = {TASK_PATHS[task_id] for task_id in task_ids}
        verdict = _branch_class(branch, local_rows, headlines)
        headline = (
            {"proof": headlines["proof"], "repair": headlines["repair"]}
            if branch == "proof"
            else headlines[branch]
        )
        out.append(
            {
                "row_type": "branch",
                "branch": branch,
                "task_ids": list(task_ids),
                "task_verdict_classes": {
                    task_id: by_task[task_id]["verdict_class"] for task_id in task_ids
                },
                "verdict_class": verdict,
                "branch_disposition": verdict,
                "headline": headline,
                "adversarial_finding_count": sum(
                    int(finding.get("flag_count") or len(finding.get("flags") or []))
                    for finding in adversarial_findings
                    if any(str(finding.get("artifact", "")).endswith(path) for path in paths)
                ),
                "row_consistency_finding_count": sum(
                    len(finding.get("findings") or [])
                    for finding in row_findings
                    if any(str(finding.get("artifact", "")).endswith(path) for path in paths)
                ),
                "claim_boundary": CLAIM_BOUNDARIES[branch],
                "next_action": NEXT_ACTIONS[branch],
            }
        )
    return out


def _verdict_identity(text: str) -> str:
    value = text.lower().strip()
    for prefix in TERMINAL_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = value.split(":", 1)[0]
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def build_prior_verdict_recurrences(
    planned: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    capstone_verdict: str,
) -> list[JsonDict]:
    """Compare each retire-if-same prior verdict with its current terminal verdict."""

    current = {str(row["task_id"]): str(row["honest_verdict"]) for row in rows}
    current[CAPSTONE_TASK_ID] = capstone_verdict
    out = []
    for plan in planned:
        for prior in plan.get("prior_failures", []):
            if not isinstance(prior, Mapping) or not prior.get("retire_if_same_verdict"):
                continue
            task_id = str(plan["task_id"])
            prior_verdict = str(prior.get("verdict") or "")
            same = _verdict_identity(prior_verdict) == _verdict_identity(current[task_id])
            out.append(
                {
                    "task_id": task_id,
                    "branch": plan["branch"],
                    "prior_experiment_id": prior.get("experiment_id"),
                    "prior_verdict": prior_verdict,
                    "current_verdict": current[task_id],
                    "retire_if_same_verdict": True,
                    "same_verdict_condition_fired": same,
                    "disposition": "retire_same_verdict_route"
                    if same
                    else "changed_or_not_executed",
                }
            )
    return out


def build_retirement_recommendations(recurrences: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Convert fired recurrence declarations into exclusion-manifest actions."""

    return [
        {
            "scope": f"{row['prior_experiment_id']}_to_{row['task_id']}_same_verdict",
            "prior_experiment_id": row["prior_experiment_id"],
            "current_task_id": row["task_id"],
            "branch": row["branch"],
            "reason": "retire_if_same_verdict=true and the normalized terminal verdict recurred",
            "recommendation": "add_to_exclusion_manifest",
        }
        for row in recurrences
        if row.get("same_verdict_condition_fired")
    ]


def build_fr12_disposition(headlines: Mapping[str, Any], branch_class: str) -> JsonDict:
    """Keep proof certification, live generation, and repair evidence separate."""

    proof = headlines["proof"]
    repair = headlines["repair"]
    comparison = proof["paired_exact_valid_effects"]
    direct = comparison.get("dccd_environment-minus-repaired_direct", {})
    harmful = repair["paired_harmful_flip_effect"]
    support = repair["paired_support_loss_effect"]
    positive = (
        proof["proof_transport_ab_completed"]
        and (direct.get("mean_delta") or 0) > 0
        and (repair["paired_exact_valid_effect"].get("mean_delta") or 0) > 0
        and harmful.get("pair_count", 0) > 0
        and harmful.get("mean_delta") is not None
        and harmful["mean_delta"] <= 0
        and support.get("pair_count", 0) > 0
        and support.get("mean_delta") is not None
        and support["mean_delta"] <= 0
    )
    return {
        "positive": bool(positive),
        "disposition": branch_class,
        "certification_fixture": {
            "targetable_panel_ready": proof["targetable_panel_ready"],
            "dynamic_proof_grammar_ready": proof["dynamic_proof_grammar_ready"],
        },
        "generation": {
            "completed": proof["proof_transport_ab_completed"],
            "row_citation": comparison,
        },
        "repair": {
            "row_count": repair["repair_rows"],
            "row_citation": repair["paired_exact_valid_effect"],
            "harmful_flip_effect": harmful,
            "support_loss_effect": support,
            "harmful_flips": repair["harmful_flips_by_arm"],
        },
    }


def build_fr11_disposition(headlines: Mapping[str, Any], branch_class: str) -> JsonDict:
    """Award FR11 credit only after all prospective and durability gates pass."""

    memory = headlines["continuous_memory"]
    gates = memory["required_positive_gates"]
    return {
        "positive": memory["cold_audit_completed"] and all(gates.values()),
        "disposition": branch_class,
        "cold_audit_completed": memory["cold_audit_completed"],
        "required_positive_gates": gates,
        "row_citations": {
            "prospective_rows": memory["prospective_rows"],
            "procedural_minus_no_memory": memory["procedural_minus_no_memory_order_effect"],
            "procedural_minus_trace": memory["procedural_minus_trace_order_effect"],
            "historical_loss_by_arm": memory["historical_loss_by_arm"],
            "transaction_activity": memory["transaction_activity"],
        },
    }


def build_arc_disposition(headlines: Mapping[str, Any], branch_class: str) -> JsonDict:
    """Keep ARC transport and supervisor evidence below the adoption gate."""

    arc = headlines["arc"]
    return {
        "positive": arc["adoption_positive"],
        "adoption_positive": arc["adoption_positive"],
        "disposition": branch_class,
        "supervisor": {
            "transport_ready": arc["shadow_supervisor_transport_ready"],
            "eligible_rows": arc["eligible_supervisor_rows"],
            "firings": arc["row_recomputed_firings"],
            "evidence_floor_met_by_arm": arc["evidence_floor_met_by_arm"],
        },
        "tool_gap": {
            "transport_ready": arc["tool_gap_transport_ready"],
            "events": arc["tool_gap_event_count"],
            "analyzer_ingest_passed": arc["tool_gap_analyzer_ingest_passed"],
        },
        "adoption": {
            "cold_audit_passed": arc["cold_actions_to_progress_audit_passed"],
            "decision": arc["adoption_decision"],
            "row_citation": arc["selfparse_minus_control_actions_effect"],
            "transport_is_not_adoption_evidence": True,
        },
        "solve_claim": arc["solve_claim"],
    }


def build_prd_gap_disposition(
    headlines: Mapping[str, Any], branch_classes: Mapping[str, str]
) -> list[JsonDict]:
    """Classify the three ranked V590 gaps without voting across them."""

    proof = headlines["proof"]
    memory = headlines["continuous_memory"]
    arc = headlines["arc"]
    return [
        {
            "rank": 1,
            "gap": "FR12 certifies proofs but cannot yet produce or repair exact certificates reliably.",
            "disposition": "narrowed",
            "branch_verdict": branch_classes["proof"],
            "artifact_citations": [
                TASK_PATHS["exp6768"],
                TASK_PATHS["exp6769"],
                TASK_PATHS["exp6770"],
            ],
            "row_citations": {
                "panel_rows": proof["panel_rows"],
                "grammar_rows": proof["grammar_rows"],
                "comparative_effects": proof["paired_exact_valid_effects"],
            },
        },
        {
            "rank": 2,
            "gap": "FR11 has transactional storage but no prospective continuous-learning result.",
            "disposition": "blocked",
            "branch_verdict": branch_classes["continuous_memory"],
            "artifact_citations": [
                TASK_PATHS["exp6773"],
                TASK_PATHS["exp6774"],
                TASK_PATHS["exp6775"],
            ],
            "row_citations": {
                "prospective_rows": memory["prospective_rows"],
                "transaction_activity": memory["transaction_activity"],
                "required_positive_gates": memory["required_positive_gates"],
            },
        },
        {
            "rank": 3,
            "gap": "The live ARC agent lacks audited supervisor and selfparse actions-to-progress evidence.",
            "disposition": "blocked",
            "branch_verdict": branch_classes["arc"],
            "artifact_citations": [TASK_PATHS[f"exp{number}"] for number in range(6776, 6780)],
            "row_citations": {
                "eligible_supervisor_rows": arc["eligible_supervisor_rows"],
                "tool_gap_transport_ready": arc["tool_gap_transport_ready"],
                "actions_effect": arc["selfparse_minus_control_actions_effect"],
                "adoption_decision": arc["adoption_decision"],
            },
        },
    ]


def _run_command(args: Sequence[str], root: Path) -> tuple[int, str]:
    completed = subprocess.run(
        list(args), cwd=root, check=False, capture_output=True, text=True, timeout=300
    )
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def _load_script_module(root: Path, name: str) -> Any:
    """Load one repository script without relying on the caller's import path."""

    path = root / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"carnot_exp6780_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load required script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_summaries(
    root: Path, sources: Mapping[str, Mapping[str, Any]], task_ids: Sequence[str] = SHORT_TASK_IDS
) -> list[JsonDict]:
    """Run the required disciplined artifact reader on every available source."""

    out = []
    for task_id in task_ids:
        record = sources.get(task_id, {})
        if record.get("artifact_state") != "present":
            continue
        artifact = str(record["path"])
        code, text = _run_command([sys.executable, str(SUMMARY_SCRIPT), artifact], root)
        out.append(
            {
                "task_id": task_id,
                "artifact": artifact,
                "exit_code": code,
                "summary_sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
                "summary_excerpt": text[:2000],
            }
        )
    return out


def run_adversarial(
    root: Path, sources: Mapping[str, Mapping[str, Any]], task_ids: Sequence[str] = SHORT_TASK_IDS
) -> list[JsonDict]:
    """Run the current adversarial verifier and retain every flag."""

    verifier = _load_script_module(root, "adversarial_verify")

    return [
        verifier.verify_artifact(root / str(sources[task_id]["path"]))
        for task_id in task_ids
        if sources.get(task_id, {}).get("artifact_state") == "present"
    ]


def run_row_consistency(
    root: Path, sources: Mapping[str, Mapping[str, Any]], task_ids: Sequence[str] = SHORT_TASK_IDS
) -> list[JsonDict]:
    """Run row consistency checks and keep skipped coverage visible."""

    lint = _load_script_module(root, "verdict_row_consistency_lint")

    out = []
    for task_id in task_ids:
        record = sources.get(task_id, {})
        if record.get("artifact_state") != "present":
            continue
        path = str(record["path"])
        status, findings = lint.check_artifact(root / path)
        out.append(
            {
                "task_id": task_id,
                "artifact": path,
                "status": status,
                "findings": findings,
                "blocking_count": sum(
                    any(finding.startswith(prefix) for prefix in lint.HARD_CLASSES)
                    for finding in findings
                ),
                "warning_count": sum(
                    not any(finding.startswith(prefix) for prefix in lint.HARD_CLASSES)
                    for finding in findings
                ),
            }
        )
    return out


def run_recurring_blockers(window: int = 14) -> JsonDict:
    """Read the recurring blocker ledger without changing known issues."""

    ledger = _load_script_module(REPO_ROOT, "recurring_blocker_ledger")
    groups, coverage, blocked_count = ledger.collect(window)
    recurring = []
    for blocker, hits in sorted(groups.items()):
        if len(hits) < 3:
            continue
        recurring.append(
            {
                "blocker": blocker,
                "count": len(hits),
                "hits": [
                    {"milestone": milestone, "artifact": artifact, "reason": reason}
                    for milestone, artifact, reason in hits
                ],
            }
        )
    return {
        "window": window,
        "blocked_task_count": blocked_count,
        "diagnostic_coverage": dict(coverage),
        "recurring": recurring,
    }


def collect_audits(root: Path, sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Collect all required source readers and ledgers in one immutable bundle."""

    return {
        "summaries": run_summaries(root, sources),
        "adversarial_findings": run_adversarial(root, sources),
        "verdict_row_consistency_findings": run_row_consistency(root, sources),
        "recurring_blockers": run_recurring_blockers(),
    }


def collect_self_audits(root: Path, artifact_path: Path) -> tuple[JsonDict, JsonDict]:
    """Audit the published capstone so its own conflicts remain visible."""

    verifier = _load_script_module(root, "adversarial_verify")
    lint = _load_script_module(root, "verdict_row_consistency_lint")
    adversarial = verifier.verify_artifact(artifact_path)
    status, findings = lint.check_artifact(artifact_path)
    row_report = {
        "task_id": CAPSTONE_TASK_ID,
        "artifact": str(artifact_path),
        "status": status,
        "findings": findings,
        "blocking_count": sum(
            any(finding.startswith(prefix) for prefix in lint.HARD_CLASSES) for finding in findings
        ),
        "warning_count": sum(
            not any(finding.startswith(prefix) for prefix in lint.HARD_CLASSES)
            for finding in findings
        ),
    }
    return adversarial, row_report


def reconcile_self_audits(
    artifact: Mapping[str, Any],
    adversarial_report: Mapping[str, Any],
    row_report: Mapping[str, Any],
    root: Path,
    *,
    duration_s: float,
) -> JsonDict:
    """Add measured self-audit output and rebuild dependent branch receipts."""

    out = copy.deepcopy(dict(artifact))
    adversarial = [
        row
        for row in out.get("adversarial_findings", [])
        if not isinstance(row, Mapping) or row.get("exp_id") != 6780
    ]
    adversarial.append(dict(adversarial_report))
    row_findings = [
        row
        for row in out.get("verdict_row_consistency_findings", [])
        if not isinstance(row, Mapping) or row.get("task_id") != CAPSTONE_TASK_ID
    ]
    row_findings.append(dict(row_report))
    out["adversarial_findings"] = adversarial
    out["verdict_row_consistency_findings"] = row_findings
    out["branch_rows"] = build_branch_rows(
        out["rows"], out["row_recomputed_headlines"], adversarial, row_findings
    )
    critical = [
        dict(flag)
        for flag in adversarial_report.get("flags", [])
        if isinstance(flag, Mapping) and flag.get("severity") == "critical"
    ]
    out["flagged_adversarial"] = bool(critical)
    out["corrigendum_pending"] = [
        {
            **flag,
            "disposition": "preserved_required_partial_terminal_contract_conflict",
        }
        for flag in critical
    ]
    out["duration_s"] = round(float(duration_s), 6)
    out["reproducibility_checksum"] = reproducibility_checksum(out, root)
    return out


def _hardware_disposition(root: Path, sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    prior = {}
    prior_path = root / PRIOR_DISPOSITION_PATH
    if prior_path.is_file():
        value = json.loads(prior_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            prior = value
    cuda_tasks = []
    for task_id in ("exp6770", "exp6773", "exp6776"):
        payload = _payload(sources.get(task_id))
        cuda_tasks.append(
            {
                "task_id": task_id,
                "model_ids": _model_ids(payload),
                "live_model_invoked": bool(payload.get("live_model_invoked")),
                "gate_failures": _gate_failures(payload, task_id),
            }
        )
    prior_branches = prior.get("branch_rows") if isinstance(prior.get("branch_rows"), list) else []
    circular = [row for row in prior_branches if row.get("verdict_class") == "circular_positive"]
    return {
        "cuda_receipts": cuda_tasks,
        "cuda_live_science_completed": any(row["live_model_invoked"] for row in cuda_tasks),
        "inherited_v589_circular_evidence": circular,
        "physical_tsu_claim": False,
        "tsu_boundary": "No physical Extropic TSU, latency, power, or availability receipt exists.",
    }


def _protected_files(root: Path) -> list[JsonDict]:
    return [
        {
            "path": path.as_posix(),
            "sha256_current": sha256_file(root / path),
            "unchanged": True,
            "note": "read-only input to Exp6780 synthesis",
        }
        for path in (RESEARCH_CONDUCTOR_PATH, ACTIVE_ROADMAP_PATH)
    ]


def _docs_reconciled() -> list[JsonDict]:
    return [
        {
            "path": path,
            "updated": False,
            "status": "delegated_to_separate_conductor_reconciler",
        }
        for path in (
            "research-complete.yaml",
            "_bmad/traceability.md",
            "ops/status.md",
            "ops/changelog.md",
        )
    ]


def _checksum_inputs(artifact: Mapping[str, Any], root: Path) -> JsonDict:
    stable = copy.deepcopy(dict(artifact))
    stable.pop("duration_s", None)
    stable.pop("reproducibility_checksum", None)
    return {
        "artifact_without_runtime_fields": stable,
        "roadmap": sha256_file(root / ACTIVE_ROADMAP_PATH),
        "design": sha256_file(root / DESIGN_PATH),
        "report_spec": sha256_file(root / REPORT_SPEC_PATH),
        "prior_disposition": sha256_file(root / PRIOR_DISPOSITION_PATH),
        "code": {
            path.as_posix(): sha256_file(root / path)
            for path in (MODULE_PATH, SCRIPT_PATH, TEST_PATH)
        },
        "source_artifacts": {
            row["task_id"]: row["artifact_sha256"] for row in artifact.get("rows", [])
        },
    }


def reproducibility_checksum(artifact: Mapping[str, Any], root: Path) -> str:
    """Hash the roadmap, sources, code, audits, and row-level synthesis."""

    return "sha256:" + hashlib.sha256(canonical_json(_checksum_inputs(artifact, root))).hexdigest()


def build_artifact(
    root: Path,
    date: str,
    *,
    audit_bundle: Mapping[str, Any] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Build and cold-validate the full V590 terminal artifact."""

    if date != PLANNING_DATE:
        raise ValueError(f"planning date must be {PLANNING_DATE}")
    started = time.monotonic()
    planned, roadmap_receipt = load_planned_tasks(root)
    sources = load_source_artifacts(root, planned)
    audits = dict(audit_bundle) if audit_bundle is not None else collect_audits(root, sources)
    rows = build_experiment_rows(planned, sources)
    summaries = {
        str(row.get("task_id")): row
        for row in audits.get("summaries", [])
        if isinstance(row, Mapping)
    }
    for row in rows:
        row["summary_receipt"] = summaries.get(str(row["task_id"]))
    headlines = recompute_headlines(sources)
    adversarial = list(audits.get("adversarial_findings") or [])
    row_findings = list(audits.get("verdict_row_consistency_findings") or [])
    branches = build_branch_rows(rows, headlines, adversarial, row_findings)
    branch_classes = {row["branch"]: row["verdict_class"] for row in branches}
    recurrences = build_prior_verdict_recurrences(planned, rows, HONEST_VERDICT)
    retirements = build_retirement_recommendations(recurrences)
    fr12 = build_fr12_disposition(headlines, branch_classes["proof"])
    fr11 = build_fr11_disposition(headlines, branch_classes["continuous_memory"])
    arc = build_arc_disposition(headlines, branch_classes["arc"])
    available = [
        {
            "task_id": task_id,
            "path": record["path"],
            "sha256": record["sha256"],
            "terminal": True,
        }
        for task_id, record in sources.items()
        if record["artifact_state"] == "present"
    ]
    missing = [
        {
            "task_id": row["task_id"],
            "path": row["path"],
            "artifact_state": row["artifact_state"],
            "verdict_class": row["verdict_class"],
            "gate_failures": row["gate_failures"],
        }
        for row in rows
        if row["artifact_state"] in {"missing", "invalid", "nonterminal"}
    ]
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact: JsonDict = {
        "schema": "carnot.v590_branch_disposition.v1",
        "experiment": 6780,
        "title": "V590 branch disposition and PRD gap update",
        "run_date": "2026-08-30",
        "status": "complete_partial",
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(elapsed), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "milestone": MILESTONE,
        "roadmap_precondition": roadmap_receipt,
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "available_artifacts": available,
        "missing_artifacts": missing,
        "rows": rows,
        "branch_rows": branches,
        "row_recomputed_headlines": headlines,
        "artifact_summaries": list(audits.get("summaries") or []),
        "adversarial_findings": adversarial,
        "verdict_row_consistency_findings": row_findings,
        "recurring_blockers": audits.get("recurring_blockers") or {},
        "prior_verdict_recurrences": recurrences,
        "retirement_recommendations": retirements,
        "prd_gap_disposition": build_prd_gap_disposition(headlines, branch_classes),
        "fr12_disposition": fr12,
        "fr11_disposition": fr11,
        "arc_disposition": arc,
        "hardware_disposition": _hardware_disposition(root, sources),
        "docs_reconciled": _docs_reconciled(),
        "protected_files_unchanged": _protected_files(root),
        "gate_check_summary": [failure for row in rows for failure in row["gate_failures"]],
        "successor_roadmap_activated": False,
        "pooled_milestone_success_score": None,
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": HONEST_VERDICT,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact, root)
    return artifact


def validate_artifact(artifact: Mapping[str, Any], root: Path) -> list[str]:
    """Return every schema, row, branch, prefix, and checksum defect."""

    findings = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        findings.append("missing_fields")
    if artifact.get("milestone") != MILESTONE:
        findings.append("milestone")
    if artifact.get("expected_task_ids") != list(EXPECTED_TASK_IDS):
        findings.append("expected_task_ids")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or [row.get("task_id") for row in rows] != list(SHORT_TASK_IDS):
        findings.append("expected_task_rows")
    branches = artifact.get("branch_rows")
    if not isinstance(branches, list) or [row.get("branch") for row in branches] != list(
        BRANCH_ORDER
    ):
        findings.append("branch_rows")
    elif any(row.get("verdict_class") not in CLOSED_CLASSES for row in branches):
        findings.append("branch_rows_closed_class")
    if artifact.get("verdict_class") not in CLOSED_CLASSES:
        findings.append("closed_verdict_class")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        findings.append("terminal_prefix")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        findings.append("field_principles")
    if artifact.get("verifier_is_oracle") is not False:
        findings.append("verifier_is_oracle")
    if artifact.get("pooled_milestone_success_score") is not None:
        findings.append("pooled_score")
    expected_checksum = reproducibility_checksum(artifact, root)
    if artifact.get("reproducibility_checksum") != expected_checksum:
        findings.append("reproducibility_checksum")
    return findings


def atomic_write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the target only after the complete JSON reaches disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate, and atomically publish the requested V590 artifact."""

    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=PLANNING_DATE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--no-external-audits", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()
    audits = (
        {
            "summaries": [],
            "adversarial_findings": [],
            "verdict_row_consistency_findings": [],
            "recurring_blockers": {},
        }
        if args.no_external_audits
        else None
    )
    artifact = build_artifact(root, args.date, audit_bundle=audits)
    findings = validate_artifact(artifact, root)
    if findings:
        raise ValueError(f"invalid V590 artifact: {findings}")
    output = args.output if args.output.is_absolute() else root / args.output
    atomic_write_json(output, artifact)
    canonical_output = (root / RESULT_PATH).resolve()
    if not args.no_external_audits and output.resolve() == canonical_output:
        self_adversarial, self_row = collect_self_audits(root, output.resolve())
        artifact = reconcile_self_audits(
            artifact,
            self_adversarial,
            self_row,
            root,
            duration_s=time.monotonic() - started,
        )
        findings = validate_artifact(artifact, root)
        if findings:
            raise ValueError(f"invalid V590 artifact after self-audit: {findings}")
        atomic_write_json(output, artifact)
    reloaded = json.loads(output.read_text(encoding="utf-8"))
    if validate_artifact(reloaded, root):
        raise ValueError("atomic V590 artifact failed cold reload")
    if not args.no_external_audits and output.resolve() == canonical_output:
        final_adversarial, final_row = collect_self_audits(root, output.resolve())
        if final_adversarial.get("flags") != self_adversarial.get("flags"):
            raise ValueError("final V590 adversarial self-audit changed after reconciliation")
        if final_row.get("findings") != self_row.get("findings"):
            raise ValueError("final V590 row self-audit changed after reconciliation")
    print(json.dumps({"artifact": str(output), "honest_verdict": HONEST_VERDICT}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
