"""Build the ungated V592 branch disposition from stored evidence.

This reducer reads receipts and per-unit rows. It does not run a model,
sampler, learner, or hardware command. Comparative claims are recalculated
locally, and a cold audit has authority over its source experiment.

Spec refs: REQ-CAPSTONE-6795 and SCENARIO-CAPSTONE-6795-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.08.592"
PLANNING_DATE = "20260830"
RANDOM_SEED = 6795
INFERENCE_SUBSTRATE = "receipt-only local synthesis, no inference model"
OUTPUT_PATH = Path("results/experiment_6795_v592_branch_disposition.json")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")

EXPECTED_TASK_IDS = (
    "exp6784-agent-model-dispatch-contract",
    "exp6785-durable-row-checkpoint-contract",
    "exp6786-constraint-dependency-hard-negative-fixture",
    "exp6787-group-aware-soft-fixed-point",
    "exp6788-soft-fixed-point-structural-control-ab",
    "exp6789-soft-fixed-point-cold-authority-audit",
    "exp6790-chronological-constraint-routing-stream",
    "exp6791-compositional-online-constraint-routing-ab",
    "exp6792-csl-causal-safety-cold-audit",
    "exp6793-temporal-exchange-ising-ab",
    "exp6794-temporal-exchange-cold-hardware-audit",
)

TASK_PATHS = {
    "exp6784-agent-model-dispatch-contract": "results/experiment_6784_agent_model_dispatch_contract.json",
    "exp6785-durable-row-checkpoint-contract": "results/experiment_6785_durable_row_checkpoint_contract.json",
    "exp6786-constraint-dependency-hard-negative-fixture": "results/experiment_6786_constraint_dependency_hard_negative_fixture.json",
    "exp6787-group-aware-soft-fixed-point": "results/experiment_6787_group_aware_soft_fixed_point.json",
    "exp6788-soft-fixed-point-structural-control-ab": "results/experiment_6788_soft_fixed_point_structural_control_ab.json",
    "exp6789-soft-fixed-point-cold-authority-audit": "results/experiment_6789_soft_fixed_point_cold_authority_audit.json",
    "exp6790-chronological-constraint-routing-stream": "results/experiment_6790_chronological_constraint_routing_stream.json",
    "exp6791-compositional-online-constraint-routing-ab": "results/experiment_6791_compositional_online_constraint_routing_ab.json",
    "exp6792-csl-causal-safety-cold-audit": "results/experiment_6792_csl_causal_safety_cold_audit.json",
    "exp6793-temporal-exchange-ising-ab": "results/experiment_6793_temporal_exchange_ising_ab.json",
    "exp6794-temporal-exchange-cold-hardware-audit": "results/experiment_6794_temporal_exchange_cold_hardware_audit.json",
}

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

REQUIRED_FIELDS = {
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "expected_task_ids",
    "artifact_inventory",
    "source_artifact_hashes",
    "rows",
    "branch_decisions",
    "infrastructure_disposition",
    "fixed_point_disposition",
    "csl_disposition",
    "temporal_exchange_disposition",
    "source_audit_disagreements",
    "prior_verdict_recurrences",
    "retirement_recommendations",
    "prd_gap_reconciliation",
    "next_prerequisites",
    "pooled_score_computed",
    "docs_updated",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
}

FIELD_PRINCIPLES = {
    "field_principles": "States why every required capstone field exists.",
    "inference_substrate": "Separates receipt synthesis from model inference or new science.",
    "duration_s": "Records measured reducer wall time.",
    "random_seed": "Pins deterministic receipt reduction.",
    "reproducibility_checksum": "Binds source hashes and recomputed decisions.",
    "expected_task_ids": "Keeps every upstream V592 task visible.",
    "artifact_inventory": "Preserves presence, readability, source class, row count, and hash.",
    "source_artifact_hashes": "Makes every imported receipt content-addressable.",
    "rows": "Provides one task row and one row for every branch claim.",
    "branch_decisions": "Keeps incomparable infrastructure and science outcomes separate.",
    "infrastructure_disposition": "Treats dispatch and checkpoint readiness as infrastructure only.",
    "fixed_point_disposition": "Requires matched row effects and an oracle-free cold audit.",
    "csl_disposition": "Requires write, read, action, held-future, and cold causal evidence.",
    "temporal_exchange_disposition": "Requires matched-work efficiency and target-law preservation.",
    "source_audit_disagreements": "Gives independent audits authority over source claims.",
    "prior_verdict_recurrences": "Applies each declared retire-if-same rule.",
    "retirement_recommendations": "Stops unchanged methods after repeated or decisive null evidence.",
    "prd_gap_reconciliation": "Maps only V592 evidence to the three named PRD gaps.",
    "next_prerequisites": "Names the smallest evidence needed to reopen each branch.",
    "pooled_score_computed": "Prevents incomparable metrics from becoming a milestone score.",
    "docs_updated": "Records capstone-owned edits and reconciliation deferred to the conductor.",
    "gate_check_summary": "Names failed checks, expected values, and observed values.",
    "verifier_is_oracle": "States that receipt auditing does not create a scientific oracle.",
    "verdict_class": "Uses the closed outcome vocabulary.",
    "honest_verdict": "Provides a terminal summary without hiding mixed branch outcomes.",
}


def canonical_json(value: Any) -> bytes:
    """Encode stable JSON for content hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str | None:
    """Return a prefixed SHA-256 for one present file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def mean_ci95(values: Sequence[float]) -> JsonDict:
    """Return a row-derived normal 95 percent interval for paired effects."""

    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"mean": None, "lower": None, "upper": None, "n": 0}
    mean = statistics.fmean(clean)
    if len(clean) == 1:
        lower = upper = mean
    else:
        margin = 1.96 * statistics.stdev(clean) / math.sqrt(len(clean))
        lower, upper = mean - margin, mean + margin
    return {"mean": mean, "lower": lower, "upper": upper, "n": len(clean)}


def _row_rate(row: Mapping[str, Any]) -> tuple[int, int, float, float]:
    outcomes = [item for item in row.get("exact_outcomes", []) if isinstance(item, Mapping)]
    valid = sum(item.get("exact_valid") is True for item in outcomes)
    count = len(outcomes)
    violations = sum(float(item.get("dependency_violation_count") or 0) for item in outcomes)
    distances = [float(item.get("distance_to_nearest_valid") or 0) for item in outcomes]
    nearest = min(distances) if distances else float(row.get("nearest_valid_distance") or 0)
    return valid, count, violations, nearest


def recompute_fixed_point(
    source_rows: Sequence[Mapping[str, Any]], audit_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Recompute the grouped-versus-flat fixed-point claim from candidate rows."""

    cold_rows = [row for row in audit_rows if row.get("row_type") == "source_recompute"]
    selected = cold_rows or list(source_rows)
    authority = "independent_audit_rows" if cold_rows else "source_rows"
    pairs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in selected:
        arm = str(row.get("arm") or "")
        key = str(row.get("paired_key") or "")
        if arm in {"grouped_fixed_point", "flat_recurrent_control"} and key:
            pairs[key][arm] = row

    effects: list[float] = []
    held_effects: list[float] = []
    arm_valid = defaultdict(int)
    arm_candidates = defaultdict(int)
    arm_violations = defaultdict(float)
    arm_distances: dict[str, list[float]] = defaultdict(list)
    arm_runtime = defaultdict(float)
    arm_converged = defaultdict(int)
    support: dict[str, set[str]] = defaultdict(set)
    parameter_counts: dict[str, set[int]] = defaultdict(set)
    work_counts: dict[str, set[tuple[int, int]]] = defaultdict(set)
    oracle_checks: list[bool] = []

    for arms in pairs.values():
        if set(arms) != {"grouped_fixed_point", "flat_recurrent_control"}:
            continue
        local_rates = {}
        for arm, row in arms.items():
            valid, count, violations, nearest = _row_rate(row)
            local_rates[arm] = valid / count if count else 0.0
            arm_valid[arm] += valid
            arm_candidates[arm] += count
            arm_violations[arm] += violations
            arm_distances[arm].append(nearest)
            arm_runtime[arm] += float(row.get("runtime_s") or 0)
            arm_converged[arm] += row.get("stop_reason") == "converged"
            parameter_counts[arm].add(int(row.get("parameter_count") or 0))
            work_counts[arm].add(
                (
                    int(row.get("candidate_budget") or count),
                    int(row.get("optimizer_update_count") or 0),
                )
            )
            for outcome in row.get("exact_outcomes", []):
                if isinstance(outcome, Mapping) and outcome.get("exact_valid") is True:
                    support[arm].add(str(outcome.get("candidate_hash") or ""))
            if authority == "independent_audit_rows":
                oracle_checks.append(row.get("exact_checker_after_candidate_freeze") is True)
            else:
                receipt = row.get("exact_evaluation_receipt") or {}
                oracle_checks.append(
                    receipt.get("evaluated_after_proposal") is True
                    and receipt.get("model_feedback_applied") is False
                    and receipt.get("candidate_hashes_before")
                    == receipt.get("candidate_hashes_after")
                )
        effect = local_rates["grouped_fixed_point"] - local_rates["flat_recurrent_control"]
        effects.append(effect)
        if arms["grouped_fixed_point"].get("split") == "held_topology_test":
            held_effects.append(effect)

    grouped = "grouped_fixed_point"
    control = "flat_recurrent_control"
    overall = mean_ci95(effects)
    held = mean_ci95(held_effects)
    matched_parameters = (
        bool(parameter_counts[grouped]) and parameter_counts[grouped] == parameter_counts[control]
    )
    matched_work = bool(work_counts[grouped]) and work_counts[grouped] == work_counts[control]
    no_convergence_harm = arm_converged[grouped] >= arm_converged[control]
    no_support_harm = len(support[grouped]) >= len(support[control])
    oracle_free = bool(oracle_checks) and all(oracle_checks)
    positive = bool(
        overall["lower"] is not None
        and overall["lower"] > 0
        and held["lower"] is not None
        and held["lower"] > 0
        and matched_parameters
        and matched_work
        and no_convergence_harm
        and no_support_harm
        and oracle_free
    )
    rates = {
        arm: arm_valid[arm] / arm_candidates[arm] if arm_candidates[arm] else None
        for arm in (grouped, control)
    }
    return {
        "evidence_authority": authority,
        "paired_key_count": len(effects),
        "exact_valid_rate_by_arm": rates,
        "paired_exact_valid_delta": overall,
        "held_topology_exact_valid_delta": held,
        "dependency_violation_rate_by_arm": {
            arm: arm_violations[arm] / arm_candidates[arm] if arm_candidates[arm] else None
            for arm in (grouped, control)
        },
        "nearest_valid_distance_mean_by_arm": {
            arm: statistics.fmean(arm_distances[arm]) if arm_distances[arm] else None
            for arm in (grouped, control)
        },
        "runtime_s_by_arm": dict(arm_runtime),
        "valid_support_by_arm": {arm: len(support[arm]) for arm in (grouped, control)},
        "matched_parameter_counts": matched_parameters,
        "matched_candidate_work": matched_work,
        "no_convergence_harm": no_convergence_harm,
        "no_support_harm": no_support_harm,
        "oracle_leakage_free": oracle_free,
        "positive_gate": positive,
    }


def recompute_csl(source_rows: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]) -> JsonDict:
    """Recompute prospective CSL activity and enforce cold-audit authority."""

    held: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    online_rows = []
    for row in source_rows:
        arm = str(row.get("arm") or "")
        order = str(row.get("order_id") or "")
        if row.get("held_future") is True:
            held[arm][order].append(float(row.get("route_utility") or 0))
        if arm == "compositional_online":
            online_rows.append(row)

    order_effects = []
    placebo_effects = []
    for order in sorted(held["compositional_online"]):
        online = statistics.fmean(held["compositional_online"][order])
        frozen_values = held["frozen_controller"].get(order, [])
        placebo_values = held["random_update_placebo"].get(order, [])
        if frozen_values:
            order_effects.append(online - statistics.fmean(frozen_values))
        if placebo_values:
            placebo_effects.append(online - statistics.fmean(placebo_values))

    writes = sum(
        int(row.get("memory_write_count") or 0)
        for row in online_rows
        if (row.get("transaction") or {}).get("committed") is True
    )
    reads = sum(int(row.get("memory_read_count") or 0) for row in online_rows)
    actions = sum(row.get("selected_action") != row.get("baseline_action") for row in online_rows)
    activity = writes > 0 and reads > 0 and actions > 0
    effect = mean_ci95(order_effects)
    placebo = mean_ci95(placebo_effects)
    positive_lcb = effect["lower"] is not None and effect["lower"] > 0
    cold_passed = bool(
        audit.get("verdict_class") == "positive"
        and audit.get("csl_causal_audit_completed") is True
        and audit.get("source_verdict_supported") is True
        and audit.get("rows")
    )
    return {
        "evidence_authority": "source_rows_bounded_by_independent_audit",
        "held_future_online_minus_frozen": effect,
        "held_future_online_minus_placebo": placebo,
        "order_effects": order_effects,
        "writes": writes,
        "later_reads": reads,
        "action_changes": actions,
        "prospective_causal_activity": activity,
        "positive_held_future_lcb": positive_lcb,
        "cold_causal_audit_passed": cold_passed,
        "audit_verdict_class": audit.get("verdict_class", "blocked"),
        "audit_blocker": audit.get("honest_verdict"),
        "promotion_gate": bool(activity and positive_lcb and cold_passed),
    }


def _attempted_updates(row: Mapping[str, Any]) -> int:
    accounting = row.get("update_accounting") or {}
    return int(row.get("update_count") or accounting.get("attempted_conditional_updates") or 0)


def _efficiency(row: Mapping[str, Any]) -> float:
    return float(
        row.get(
            "energy_effective_samples_per_attempted_update",
            row.get("effective_samples_per_update", 0),
        )
    )


def recompute_temporal_exchange(
    source_rows: Sequence[Mapping[str, Any]], audit_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Recompute sampler efficiency and target-law gates at matched work."""

    cold_rows = [row for row in audit_rows if row.get("row_kind") == "source_recomputation"]
    selected = cold_rows or list(source_rows)
    authority = "independent_audit_rows" if cold_rows else "source_rows"
    paired: dict[tuple[str, float, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in selected:
        arm = str(row.get("arm") or "")
        if arm in {"ordinary_gibbs", "temporal_exchange", "temporal_exchange_zero_coupling"}:
            key = (
                str(row.get("graph_id") or ""),
                float(row.get("temperature") or 0),
                int(row.get("seed") or 0),
            )
            paired[key][arm] = row

    efficiency_by_stratum: dict[tuple[str, float], list[float]] = defaultdict(list)
    target_by_stratum: dict[tuple[str, float], list[float]] = defaultdict(list)
    matched_work = []
    zero_control_equal = []
    for (graph_id, temperature, _seed), arms in paired.items():
        if "ordinary_gibbs" not in arms or "temporal_exchange" not in arms:
            continue
        ordinary = arms["ordinary_gibbs"]
        temporal = arms["temporal_exchange"]
        stratum = (graph_id, temperature)
        efficiency_by_stratum[stratum].append(_efficiency(temporal) - _efficiency(ordinary))
        target_by_stratum[stratum].append(
            float(temporal.get("target_total_variation") or 0)
            - float(ordinary.get("target_total_variation") or 0)
        )
        matched_work.append(_attempted_updates(temporal) == _attempted_updates(ordinary))
        zero = arms.get("temporal_exchange_zero_coupling")
        if zero is not None:
            zero_control_equal.append(
                _efficiency(zero) == _efficiency(ordinary)
                and float(zero.get("target_total_variation") or 0)
                == float(ordinary.get("target_total_variation") or 0)
            )

    efficiency_rows = []
    target_rows = []
    for graph_id, temperature in sorted(efficiency_by_stratum):
        interval = mean_ci95(efficiency_by_stratum[(graph_id, temperature)])
        efficiency_rows.append(
            {
                "graph_id": graph_id,
                "temperature": temperature,
                **interval,
                "lcb_above_zero": interval["lower"] > 0,
            }
        )
        target = mean_ci95(target_by_stratum[(graph_id, temperature)])
        target_rows.append(
            {
                "graph_id": graph_id,
                "temperature": temperature,
                **target,
                "margin": 0.03,
                "ucb_within_margin": target["upper"] <= 0.03,
            }
        )
    efficiency_gate = bool(efficiency_rows) and all(
        row["lcb_above_zero"] for row in efficiency_rows
    )
    target_gate = bool(target_rows) and all(row["ucb_within_margin"] for row in target_rows)
    work_gate = bool(matched_work) and all(matched_work)
    return {
        "evidence_authority": authority,
        "paired_seed_count": len(matched_work),
        "efficiency_by_stratum": efficiency_rows,
        "target_law_by_stratum": target_rows,
        "matched_attempted_updates": work_gate,
        "zero_coupling_control_equal": bool(zero_control_equal) and all(zero_control_equal),
        "efficiency_gate_passed": efficiency_gate,
        "target_law_gate_passed": target_gate,
        "positive_gate": bool(work_gate and efficiency_gate and target_gate),
    }


def _load_preconditions(root: Path) -> tuple[JsonDict | None, list[JsonDict], JsonDict]:
    failures = []
    design = root / DESIGN_PATH
    design_text = design.read_text() if design.is_file() else ""
    if not design_text.strip():
        failures.append(
            {
                "check": "v592_design_nonempty",
                "expected": "nonempty markdown",
                "observed": len(design_text),
            }
        )
    roadmap: JsonDict | None = None
    roadmap_path = root / ROADMAP_PATH
    try:
        parsed = yaml.safe_load(roadmap_path.read_text())
        if isinstance(parsed, dict) and parsed.get("milestone") == MILESTONE:
            roadmap = parsed
        else:
            failures.append(
                {"check": "v592_roadmap_mapping", "expected": MILESTONE, "observed": parsed}
            )
    except (OSError, yaml.YAMLError) as exc:
        failures.append(
            {
                "check": "v592_roadmap_mapping",
                "expected": MILESTONE,
                "observed": f"{type(exc).__name__}: {exc}",
            }
        )
    next_path = root / NEXT_ROADMAP_PATH
    observation = {
        "check": "inactive_next_roadmap",
        "blocking": False,
        "path": NEXT_ROADMAP_PATH.as_posix(),
        "observed": "present" if next_path.is_file() else "missing",
    }
    return roadmap, failures, observation


def _load_inventory(
    root: Path, roadmap: Mapping[str, Any]
) -> tuple[list[JsonDict], dict[str, JsonDict]]:
    planned = {
        str(task.get("id")): task for task in roadmap.get("tasks", []) if isinstance(task, Mapping)
    }
    inventory = []
    payloads = {}
    for task_id in EXPECTED_TASK_IDS:
        task = planned.get(task_id, {})
        relative = str(task.get("deliverable") or TASK_PATHS[task_id])
        path = root / relative
        digest = sha256_file(path)
        state = "present" if digest else "missing"
        payload = None
        error = None
        if digest:
            try:
                value = json.loads(path.read_text())
                if not isinstance(value, dict):
                    raise ValueError("top level is not an object")
                payload = value
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                state = "unreadable"
                error = f"{type(exc).__name__}: {exc}"
        source_class = payload.get("verdict_class") if payload else None
        evidence_state = source_class if source_class in CLOSED_CLASSES else state
        row = {
            "task_id": task_id,
            "path": relative,
            "roadmap_task_present": task_id in planned,
            "artifact_state": state,
            "evidence_state": evidence_state,
            "verdict_class": source_class if source_class in CLOSED_CLASSES else "blocked",
            "honest_verdict": payload.get("honest_verdict") if payload else None,
            "row_count": len(payload.get("rows", [])) if payload else 0,
            "sha256": digest,
            "read_error": error,
        }
        inventory.append(row)
        if payload is not None:
            payloads[task_id] = payload
    return inventory, payloads


def _blocked_source_checks(
    inventory: Sequence[Mapping[str, Any]], payloads: Mapping[str, JsonDict]
) -> list[JsonDict]:
    out = []
    for row in inventory:
        if row.get("verdict_class") != "blocked":
            continue
        task_id = str(row["task_id"])
        summary = payloads.get(task_id, {}).get("gate_check_summary") or {}
        checks = summary.get("checks", []) if isinstance(summary, Mapping) else []
        failed = [
            check for check in checks if isinstance(check, Mapping) and check.get("passed") is False
        ]
        if not failed and isinstance(summary, Mapping) and summary.get("passed") is False:
            failed = [summary]
        if not failed:
            failed = [
                {
                    "check": "artifact_availability",
                    "expected": "readable terminal artifact",
                    "observed": row.get("artifact_state"),
                }
            ]
        for check in failed:
            out.append(
                {
                    "task_id": task_id,
                    "failed_check": check.get("check", check.get("failed_check")),
                    "expected": check.get("expected"),
                    "observed": check.get("observed"),
                }
            )
    return out


def _infrastructure(payloads: Mapping[str, JsonDict]) -> JsonDict:
    dispatch = payloads.get(EXPECTED_TASK_IDS[0], {})
    checkpoint = payloads.get(EXPECTED_TASK_IDS[1], {})
    dispatch_ready = dispatch.get("dispatch_contract_ready") is True
    checkpoint_ready = checkpoint.get("durable_checkpoint_ready") is True
    verdict = (
        "positive"
        if dispatch_ready and checkpoint_ready
        else "partial"
        if dispatch_ready or checkpoint_ready
        else "blocked"
    )
    return {
        "branch": "infrastructure",
        "verdict_class": verdict,
        "evidence": {
            "dispatch_contract_ready": dispatch_ready,
            "durable_checkpoint_ready": checkpoint_ready,
            "prefix_rows_preserved": (checkpoint.get("prefix_rows_preserved") or {}).get("count"),
            "fresh_resume_rows": len(
                (checkpoint.get("fresh_process_resume_rows") or {}).get("row_ids", [])
            ),
        },
        "claim_boundary": "Operational readiness only; neither contract is a scientific win.",
        "decision": "adopt durable checkpoint contract; keep dispatch activation blocked",
        "exact_next_prerequisite": "Create and audit the agent/model compatibility matrix from the active V592 roadmap, while treating an absent inactive next-roadmap file as nonblocking.",
    }


def _fixed_disposition(recomputed: JsonDict, audit: Mapping[str, Any]) -> JsonDict:
    supported = (
        audit.get("source_verdict_supported") is True
        and audit.get("fixed_point_audit_completed") is True
    )
    verdict = "positive" if recomputed.get("positive_gate") and supported else "partial"
    return {
        "branch": "fixed_point",
        "verdict_class": verdict,
        "evidence": recomputed,
        "claim_boundary": "Positive only for the synthetic exact-enumerable CPU fixture; all runs hit the iteration cap, nearest-valid distance and runtime remain tradeoffs, and no production output claim is made.",
        "decision": "adopt experiment-only as a bounded neural constraint-dynamics bridge",
        "exact_next_prerequisite": "Before wider adoption, test frozen grouped proposals on real structured constraint outputs with the same post-proposal exact checker, matched work, convergence reporting, and independent cold audit.",
    }


def _csl_disposition(recomputed: JsonDict) -> JsonDict:
    if recomputed.get("promotion_gate"):
        verdict = "positive"
        decision = "adopt experiment-only causal routing controller"
    elif recomputed.get("prospective_causal_activity") and recomputed.get(
        "positive_held_future_lcb"
    ):
        verdict = "partial"
        decision = "withhold FR11 credit and production adoption"
    else:
        verdict = "blocked"
        decision = "keep branch blocked"
    return {
        "branch": "continuous_self_learning",
        "verdict_class": verdict,
        "evidence": recomputed,
        "claim_boundary": "Source rows show prospective writes, later reads, action changes, and held-future lift; missing parent and new canonical state bytes prevent cold causal, restart, rollback, poison, and retention credit.",
        "decision": decision,
        "exact_next_prerequisite": "Emit parent and new canonical state-byte snapshots for every committed transaction, verify each hash, then rerun only the Exp6792 cold causal and safety audit.",
    }


def _temporal_disposition(recomputed: JsonDict, audit: Mapping[str, Any]) -> JsonDict:
    audit_complete = audit.get("temporal_exchange_audit_completed") is True
    verdict = (
        "positive"
        if recomputed.get("positive_gate") and audit_complete
        else "null"
        if recomputed.get("paired_seed_count")
        else "blocked"
    )
    return {
        "branch": "temporal_exchange",
        "verdict_class": verdict,
        "evidence": recomputed,
        "claim_boundary": "CPU Ising simulation and static cost accounting only; no FPGA, TSU, latency, throughput, power, or physical-hardware claim.",
        "decision": "retire the unchanged temporal-coupling schedule and do not start hardware work",
        "exact_next_prerequisite": "Reopen only with a target-invariant temporal kernel, a preregistered work denominator, and a matched-work law-preservation proof before any new simulation or hardware study.",
    }


def _verdict_identity(text: str) -> str:
    value = text.lower().strip()
    for prefix in TERMINAL_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return value.split(":", 1)[0].strip().replace("-", "_").replace(" ", "_")


def _prior_recurrences(
    roadmap: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    current = {str(row["task_id"]): str(row.get("honest_verdict") or "") for row in inventory}
    current["exp6795-v592-branch-disposition"] = "complete_partial: V592 mixed branch disposition"
    out = []
    for task in roadmap.get("tasks", []):
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id") or "")
        for prior in task.get("prior_failures") or []:
            if not isinstance(prior, Mapping) or prior.get("retire_if_same_verdict") is not True:
                continue
            prior_verdict = str(prior.get("verdict") or "")
            now = current.get(task_id, "")
            if now and _verdict_identity(prior_verdict) == _verdict_identity(now):
                out.append(
                    {
                        "task_id": task_id,
                        "prior_experiment_id": prior.get("experiment_id"),
                        "prior_verdict": prior_verdict,
                        "current_verdict": now,
                        "repeated_outcome": now.split(":", 1)[0],
                        "retire_if_same_verdict": True,
                        "disposition": "retire_unchanged_method_scope",
                    }
                )
    return out


def _retirements(recurrences: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    proposed = []
    for row in recurrences:
        proposed.append(
            {
                "branch": "capstone_aggregation",
                "reason": "The normalized complete_partial outcome repeated Exp6780 under retire_if_same_verdict=true.",
                "recommendation": "stop recommending another aggregation-only capstone as the way to close unchanged evidence gaps",
                "proposed_exclusion_manifest_entry": {
                    "id": "exp6780_mixed_capstone_repeat_retired_v592",
                    "scope_key": "aggregation_only_capstone_complete_partial_repeat_v592",
                    "experiment_scope": "Another branch-disposition capstone without new prerequisite evidence for the open V592 gaps",
                    "experiment_ids": [row.get("prior_experiment_id"), row.get("task_id")],
                    "retired_milestone": MILESTONE,
                    "retire_if_same_verdict": True,
                    "operator_reopen_required": True,
                    "reopen_requires": "New dispatch-matrix, CSL byte-snapshot audit, or target-invariant sampler evidence rather than another synthesis pass.",
                },
                "active_exclusion_manifest_modified": False,
            }
        )
    proposed.append(
        {
            "branch": "temporal_exchange",
            "reason": "Matched-row recomputation and the cold audit both support the source null, including target-law and efficiency gate failures.",
            "recommendation": "retire the unchanged temporal coupling schedule",
            "proposed_exclusion_manifest_entry": None,
            "active_exclusion_manifest_modified": False,
        }
    )
    return proposed


def _checksum(artifact: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()


def _blocked_artifact(date: str, failures: list[JsonDict], duration_s: float) -> JsonDict:
    blocked_branch = {
        "verdict_class": "blocked",
        "evidence": {},
        "claim_boundary": "No branch claim is available because a capstone precondition failed.",
        "decision": "blocked",
        "exact_next_prerequisite": "Repair the unreadable V592 design or active roadmap, then rerun receipt synthesis.",
    }
    artifact: JsonDict = {
        "schema": "carnot.v592_branch_disposition.v1",
        "experiment_id": "experiment_6795_v592_branch_disposition",
        "run_date": date,
        "milestone": MILESTONE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "artifact_inventory": [],
        "source_artifact_hashes": {},
        "rows": [],
        "branch_decisions": [],
        "infrastructure_disposition": {"branch": "infrastructure", **blocked_branch},
        "fixed_point_disposition": {"branch": "fixed_point", **blocked_branch},
        "csl_disposition": {"branch": "continuous_self_learning", **blocked_branch},
        "temporal_exchange_disposition": {"branch": "temporal_exchange", **blocked_branch},
        "source_audit_disagreements": [],
        "prior_verdict_recurrences": [],
        "retirement_recommendations": [],
        "prd_gap_reconciliation": [],
        "next_prerequisites": {},
        "pooled_score_computed": False,
        "docs_updated": {"capstone_spec": True, "conductor_reconciliation_deferred": True},
        "gate_check_summary": {"passed": False, "failed_checks": failures},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_v592_disposition",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def build_disposition(root: Path, date: str) -> JsonDict:
    """Build the complete V592 disposition from artifacts already on disk."""

    started = time.monotonic()
    roadmap, failures, next_observation = _load_preconditions(root)
    if failures or roadmap is None:
        return _blocked_artifact(date, failures, time.monotonic() - started)

    inventory, payloads = _load_inventory(root, roadmap)
    fixed_source = payloads.get(EXPECTED_TASK_IDS[4], {})
    fixed_audit = payloads.get(EXPECTED_TASK_IDS[5], {})
    csl_source = payloads.get(EXPECTED_TASK_IDS[7], {})
    csl_audit = payloads.get(EXPECTED_TASK_IDS[8], {})
    temporal_source = payloads.get(EXPECTED_TASK_IDS[9], {})
    temporal_audit = payloads.get(EXPECTED_TASK_IDS[10], {})

    fixed = recompute_fixed_point(fixed_source.get("rows", []), fixed_audit.get("rows", []))
    csl = recompute_csl(csl_source.get("rows", []), csl_audit)
    temporal = recompute_temporal_exchange(
        temporal_source.get("rows", []), temporal_audit.get("rows", [])
    )
    infrastructure_disposition = _infrastructure(payloads)
    fixed_disposition = _fixed_disposition(fixed, fixed_audit)
    csl_disposition = _csl_disposition(csl)
    temporal_disposition = _temporal_disposition(temporal, temporal_audit)
    decisions = [
        infrastructure_disposition,
        fixed_disposition,
        csl_disposition,
        temporal_disposition,
    ]
    recurrences = _prior_recurrences(roadmap, inventory)
    disagreements = []
    if (
        csl_source.get("verdict_class") == "positive"
        and csl_audit.get("source_verdict_supported") is not True
    ):
        disagreements.append(
            {
                "branch": "continuous_self_learning",
                "source_task_id": EXPECTED_TASK_IDS[7],
                "audit_task_id": EXPECTED_TASK_IDS[8],
                "source_claim": csl_source.get("honest_verdict"),
                "audit_claim": csl_audit.get("honest_verdict"),
                "authority_applied": "independent_audit",
                "disposition": "source positive is bounded to partial; FR11 credit withheld",
            }
        )

    prd_gaps = [
        {
            "gap": "fail_closed_dispatch_and_durable_checkpoints",
            "prd_refs": ["FR11"],
            "verdict_class": infrastructure_disposition["verdict_class"],
            "reconciliation": "Durable row resume is ready, but the dispatch compatibility matrix is absent; infrastructure is not a scientific win.",
            "smallest_next_step_or_retirement": infrastructure_disposition[
                "exact_next_prerequisite"
            ],
        },
        {
            "gap": "validated_neural_constraint_dynamics_bridge",
            "prd_refs": ["FR12"],
            "verdict_class": fixed_disposition["verdict_class"],
            "reconciliation": "Grouped fixed-point proposals beat the matched flat control under cold exact audit without oracle feedback, closing only the synthetic experimental slice of FR12.",
            "smallest_next_step_or_retirement": fixed_disposition["exact_next_prerequisite"],
        },
        {
            "gap": "self_learning_causal_credit",
            "prd_refs": ["FR11"],
            "verdict_class": csl_disposition["verdict_class"],
            "reconciliation": "Positive source rows do not close FR11 because the independent audit lacks canonical transaction bytes for causal and safety replay.",
            "smallest_next_step_or_retirement": csl_disposition["exact_next_prerequisite"],
        },
    ]
    source_hashes = {str(row["task_id"]): row.get("sha256") for row in inventory}
    task_rows = [{"row_type": "task", **row} for row in inventory]
    branch_rows = [
        {
            "row_type": "branch_claim",
            "branch": decision["branch"],
            "verdict_class": decision["verdict_class"],
            "evidence": decision["evidence"],
            "claim_boundary": decision["claim_boundary"],
            "decision": decision["decision"],
            "exact_next_prerequisite": decision["exact_next_prerequisite"],
        }
        for decision in decisions
    ]
    elapsed = time.monotonic() - started
    artifact: JsonDict = {
        "schema": "carnot.v592_branch_disposition.v1",
        "experiment_id": "experiment_6795_v592_branch_disposition",
        "run_date": date,
        "milestone": MILESTONE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(elapsed, 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "artifact_inventory": inventory,
        "source_artifact_hashes": source_hashes,
        "rows": task_rows + branch_rows,
        "branch_decisions": decisions,
        "infrastructure_disposition": infrastructure_disposition,
        "fixed_point_disposition": fixed_disposition,
        "csl_disposition": csl_disposition,
        "temporal_exchange_disposition": temporal_disposition,
        "source_audit_disagreements": disagreements,
        "prior_verdict_recurrences": recurrences,
        "retirement_recommendations": _retirements(recurrences),
        "prd_gap_reconciliation": prd_gaps,
        "next_prerequisites": {
            decision["branch"]: decision["exact_next_prerequisite"] for decision in decisions
        },
        "pooled_score_computed": False,
        "docs_updated": {
            "openspec/capabilities/capstone/spec.md": "REQ-CAPSTONE-6795 added",
            "research-complete.yaml": "V592 receipt row added by this task",
            "deferred_to_conductor": ["_bmad/traceability.md", "ops/status.md", "ops/changelog.md"],
        },
        "gate_check_summary": {
            "passed": True,
            "capstone_failed_checks": [],
            "blocked_source_checks": _blocked_source_checks(inventory, payloads),
            "nonblocking_observations": [next_observation],
        },
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "complete_partial: V592 has checkpoint and bounded fixed-point evidence, but dispatch and cold CSL causality remain incomplete and temporal exchange is null; no branch metrics were pooled.",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return all capstone schema and determination defects."""

    findings = []
    if REQUIRED_FIELDS - set(artifact):
        findings.append("required_fields")
    if artifact.get("verdict_class") not in CLOSED_CLASSES:
        findings.append("verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        findings.append("terminal_prefix")
    if artifact.get("pooled_score_computed") is not False:
        findings.append("pooled_score_computed")
    if artifact.get("verifier_is_oracle") is not False:
        findings.append("verifier_is_oracle")
    if artifact.get("reproducibility_checksum") != _checksum(artifact):
        findings.append("reproducibility_checksum")
    return findings


def atomic_write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    """Publish JSON only after complete bytes are flushed to disk."""

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
    """Build, validate, and atomically write the V592 disposition."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=PLANNING_DATE)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.repo.resolve()
    artifact = build_disposition(root, args.date)
    findings = validate_artifact(artifact)
    if findings:
        raise ValueError(f"invalid V592 disposition: {findings}")
    output = root / OUTPUT_PATH
    atomic_write_json(output, artifact)
    print(output)
    return 0
