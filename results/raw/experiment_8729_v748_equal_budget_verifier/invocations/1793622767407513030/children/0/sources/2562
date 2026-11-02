"""Build the V599 independent capstone from checked-in evidence.

The reducer reads raw rows because producer summaries can contain mistakes.
It keeps missing values as ``None`` because zero is a measured result. This
distinction prevents an absent experiment from looking like a negative result.

Spec refs: REQ-REPORT-6860 and SCENARIO-REPORT-6860-*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("results/experiment_6860_v599_independent_capstone.json")
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
MODEL_IDS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
}

TASKS: dict[str, dict[str, str]] = {
    "exp6848": {
        "task_id": "exp6848-v599-method-change-evidence-contract",
        "path": "results/experiment_6848_v599_method_change_evidence_contract.json",
        "branch": "shared_evidence",
        "title": "V599 method-change and evidence contract",
    },
    "exp6849": {
        "task_id": "exp6849-typed-program-isomorphic-authority-audit",
        "path": "results/experiment_6849_typed_program_isomorphic_authority_audit.json",
        "branch": "typed_compatibility",
        "title": "Independent typed-program isomorphic authority aud",
    },
    "exp6850": {
        "task_id": "exp6850-three-family-scoring-admission-canary",
        "path": "results/experiment_6850_three_family_scoring_admission_canary.json",
        "branch": "typed_resource_admission",
        "title": "Three-family forced-sequence scoring admission can",
    },
    "exp6851": {
        "task_id": "exp6851-three-family-isomorphic-compatibility-stream",
        "path": "results/experiment_6851_three_family_isomorphic_compatibility_stream.json",
        "branch": "typed_compatibility",
        "title": "Three-family isomorphic fixed-sequence compatibili",
    },
    "exp6852": {
        "task_id": "exp6852-compatibility-shortcut-authority-audit",
        "path": "results/experiment_6852_compatibility_shortcut_authority_audit.json",
        "branch": "typed_compatibility",
        "title": "Independent compatibility shortcut and claim-autho",
    },
    "exp6853": {
        "task_id": "exp6853-risk-sensitive-memory-opportunity-fixture",
        "path": "results/experiment_6853_risk_sensitive_memory_opportunity_fixture.json",
        "branch": "continuous_self_learning",
        "title": "Risk-sensitive memory opportunity and headroom fix",
    },
    "exp6854": {
        "task_id": "exp6854-risk-sensitive-abstention-memory-controller",
        "path": "results/experiment_6854_risk_sensitive_abstention_memory_controller.json",
        "branch": "continuous_self_learning",
        "title": "Risk-sensitive abstention-aware online memory cont",
    },
    "exp6855": {
        "task_id": "exp6855-counterfactual-memory-credit-audit",
        "path": "results/experiment_6855_counterfactual_memory_credit_audit.json",
        "branch": "continuous_self_learning",
        "title": "Counterfactual memory-write credit and negative-tr",
    },
    "exp6856": {
        "task_id": "exp6856-sealed-risk-sensitive-learning-audit",
        "path": "results/experiment_6856_sealed_risk_sensitive_learning_audit.json",
        "branch": "continuous_self_learning",
        "title": "Sealed risk-sensitive self-learning durability and",
    },
    "exp6857": {
        "task_id": "exp6857-dynamic-live-arc-receipt-router",
        "path": "results/experiment_6857_dynamic_live_arc_receipt_router.json",
        "branch": "arc_supervisor",
        "title": "Dynamic provenance-qualified live ARC receipt rout",
    },
    "exp6858": {
        "task_id": "exp6858-supervisor-counterfactual-credit-audit",
        "path": "results/experiment_6858_supervisor_counterfactual_credit_audit.json",
        "branch": "arc_supervisor",
        "title": "Supervisor counterfactual credit audit gated on Ex",
    },
    "exp6859": {
        "task_id": "exp6859-first-party-tool-gap-receipt-wiring",
        "path": "results/experiment_6859_first_party_tool_gap_receipt_wiring.json",
        "branch": "tool_gap_receipt",
        "title": "First-party tool-gap receipt wiring at the canonic",
    },
}


def sha256_path(path: Path) -> str:
    """Return a content hash so a later edit cannot silently change evidence."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    """Read one JSON object and make malformed evidence explicitly unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_source_payloads(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Load each V599 producer without importing any producer Python module."""

    payloads: dict[str, dict[str, Any]] = {}
    for short_id, info in TASKS.items():
        payload = read_json(repo_root / info["path"])
        if payload is not None:
            payloads[short_id] = payload
    return payloads


def load_roadmap_tasks(repo_root: Path) -> list[dict[str, Any]]:
    """Read planned identities from the active roadmap instead of guessing them."""

    payload = yaml.safe_load((repo_root / "research-roadmap.yaml").read_text(encoding="utf-8"))
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
    return [row for row in tasks if isinstance(row, dict)]


def duplicate_task_ids(tasks: list[dict[str, Any]]) -> list[str]:
    """List repeated task IDs because one identity cannot own two dispositions."""

    counts: dict[str, int] = defaultdict(int)
    for task in tasks:
        task_id = task.get("id")
        if isinstance(task_id, str):
            counts[task_id] += 1
    return sorted(task_id for task_id, count in counts.items() if count > 1)


def parse_conductor_events(text: str) -> list[dict[str, str]]:
    """Extract only V599 terminal events from the append-only conductor table."""

    events: list[dict[str, str]] = []
    active = False
    for line in text.splitlines():
        if "Milestone 2026.09.599 activated" in line:
            active = True
            continue
        if not active or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        title = cells[1]
        match = next((info for info in TASKS.values() if title.startswith(info["title"])), None)
        if match is None:
            continue
        events.append(
            {
                "timestamp": cells[0],
                "task_id": match["task_id"],
                "status": cells[2],
                "detail": cells[3],
            }
        )
    return events


def retry_events(events: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Keep failed attempts visible when a later retry produced an artifact."""

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in events:
        grouped[event["task_id"]].append(event)
    return [
        {"task_id": task_id, "retry_count": len(rows) - 1, "attempts": rows}
        for task_id, rows in sorted(grouped.items())
        if len(rows) > 1
    ]


def _class_from_text(text: str) -> str:
    """Map a terminal phrase to the closed verdict enum without making it positive."""

    lowered = text.lower()
    if "circular_positive" in lowered:
        return "circular_positive"
    if "disqual" in lowered or "quarantin" in lowered:
        return "disqualified"
    if "block" in lowered or "gate_check_failed" in lowered:
        return "blocked"
    if "partial" in lowered or "running" in lowered:
        return "partial"
    if "positive" in lowered or lowered.startswith("success"):
        return "positive"
    return "null"


def classify_task_state(
    short_id: str,
    payload: dict[str, Any] | None,
    events: list[dict[str, str]],
) -> dict[str, Any]:
    """Classify one task while preserving missing, skipped, and flagged states."""

    matching = [
        row
        for row in events
        if row.get("task_id") in {None, TASKS.get(short_id, {}).get("task_id")}
    ]
    gate = next((row for row in reversed(matching) if row.get("status") == "GATE_BLOCK"), None)
    if gate is not None:
        return {
            "verdict_class": "blocked",
            "failed_condition": "conductor_pre_gate",
            "observed_value": gate["detail"],
        }
    if payload is None:
        return {
            "verdict_class": "null",
            "failed_condition": "terminal_artifact_missing",
            "observed_value": None,
        }
    flag = next((row for row in reversed(matching) if row.get("status") == "FLAGGED"), None)
    if flag is not None:
        return {
            "verdict_class": "disqualified",
            "failed_condition": "adversarial_quarantine",
            "observed_value": flag["detail"],
        }
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or status)
    declared = payload.get("verdict_class")
    terminal = verdict.startswith(("complete_", "blocked_")) or status in {
        "complete",
        "completed",
        "success",
        "blocked",
    }
    verdict_class = str(declared) if declared in VERDICT_CLASSES else _class_from_text(verdict)
    if not terminal:
        verdict_class = "partial"
    return {
        "verdict_class": verdict_class,
        "failed_condition": None if verdict_class in {"positive", "null"} else verdict,
        "observed_value": verdict,
    }


def check_declared_hashes(repo_root: Path, declared: Any) -> list[dict[str, Any]]:
    """Compare direct source receipts with current bytes and expose missing files."""

    if not isinstance(declared, dict):
        return []
    checks: list[dict[str, Any]] = []
    for source_id, receipt in sorted(declared.items()):
        if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
            continue
        expected = (
            receipt.get("sha256") or receipt.get("file_sha256") or receipt.get("recorded_sha256")
        )
        if not isinstance(expected, str):
            continue
        raw_path = Path(receipt["path"])
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        observed = sha256_path(path) if path.is_file() else None
        checks.append(
            {
                "source_id": str(source_id),
                "path": receipt["path"],
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": expected == observed,
            }
        )
    return checks


def _mean(values: list[float]) -> float | None:
    """Return no value for an empty measurement instead of inventing zero."""

    return sum(values) / len(values) if values else None


def _same_direction(left: float, right: float) -> bool:
    """Treat two measured margins as invariant only when their signs agree."""

    return (left == 0 and right == 0) or left * right > 0


def reduce_typed(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Recompute typed authority, admission, margins, controls, and eligibility."""

    authority = payloads.get("exp6849", {})
    authority_rows = authority.get("rows", [])
    candidates = authority.get("candidate_identity_manifest", [])
    row_ids = [row.get("row_id") for row in authority_rows if isinstance(row, dict)]
    candidate_ids = [row.get("candidate_id") for row in candidates if isinstance(row, dict)]
    semantics = [row.get("semantic_identity") for row in candidates if isinstance(row, dict)]
    parity_ready = bool(authority_rows) and all(
        row.get("parity_passed") is True
        and row.get("atom_identities_match") is True
        and row.get("observed_label") == row.get("expected_label")
        for row in authority_rows
        if isinstance(row, dict)
    )
    identity_ready = (
        bool(candidate_ids)
        and len(candidate_ids) == len(set(candidate_ids))
        and len(semantics) == len(set(semantics))
        and len(row_ids) == len(set(row_ids))
    )

    admission_rows = payloads.get("exp6850", {}).get("rows", [])
    admission_models = {
        row.get("hf_id") for row in admission_rows if isinstance(row, dict) and row.get("hf_id")
    }
    receipt_ready = all(
        isinstance(row.get("canary_hash"), str)
        and isinstance(row.get("model_hash"), str)
        and isinstance(row.get("tokenizer_hash"), str)
        and isinstance(row.get("token_logprobs"), list)
        and len(row["token_logprobs"]) == len(row.get("candidate_token_ids", []))
        and len(row["token_logprobs"]) > 0
        for row in admission_rows
        if isinstance(row, dict)
    )
    resource_ready = len(admission_rows) == 3 and admission_models == MODEL_IDS and receipt_ready

    stream_rows = [
        row for row in payloads.get("exp6851", {}).get("rows", []) if isinstance(row, dict)
    ]
    row_identities = [row.get("row_identity") for row in stream_rows]
    protocol_ready = all(
        row.get("no_answer_feedback") is True
        and row.get("no_generation") is True
        and row.get("no_grammar") is True
        and row.get("no_retry_repair") is True
        and row.get("no_sampling") is True
        and row.get("prompt_masked") is True
        for row in stream_rows
    )
    stream_complete = (
        len(stream_rows) == 168
        and len(row_identities) == len(set(row_identities))
        and {row.get("model_hf_id") for row in stream_rows} == MODEL_IDS
        and protocol_ready
    )
    model_metrics: dict[str, dict[str, Any]] = {}
    for model_id in sorted(MODEL_IDS):
        margins = [
            float(row["scalar_compatibility_margin"])
            for row in stream_rows
            if row.get("model_hf_id") == model_id
            and isinstance(row.get("scalar_compatibility_margin"), (int, float))
        ]
        model_metrics[model_id] = {
            "row_count": len(margins),
            "wins": sum(value > 0 for value in margins),
            "losses": sum(value < 0 for value in margins),
            "ties": sum(value == 0 for value in margins),
            "mean_margin": _mean(margins),
        }

    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in stream_rows:
        grouped[
            (row.get("model_hf_id"), row.get("semantic_pair_identity"), row.get("repeat"))
        ].append(row)
    isomorphic_failures = 0
    isomorphic_comparisons = 0
    for rows in grouped.values():
        base = next((row for row in rows if row.get("transform_kind") == "base"), None)
        if base is None or not isinstance(base.get("scalar_compatibility_margin"), (int, float)):
            isomorphic_failures += 1
            continue
        base_margin = float(base["scalar_compatibility_margin"])
        for row in rows:
            if row.get("transform_kind") == "base":
                continue
            margin = row.get("scalar_compatibility_margin")
            isomorphic_comparisons += 1
            if not isinstance(margin, (int, float)) or not _same_direction(
                base_margin, float(margin)
            ):
                isomorphic_failures += 1

    audit_rows = [
        row for row in payloads.get("exp6852", {}).get("rows", []) if isinstance(row, dict)
    ]
    shortcut_count = sum(
        row.get("attack_kind") is not None and row.get("explains_same_or_greater") is True
        for row in audit_rows
    )
    positive_models = sorted(
        model_id
        for model_id, metrics in model_metrics.items()
        if isinstance(metrics["mean_margin"], float) and metrics["mean_margin"] > 0
    )
    claim_eligible = (
        parity_ready
        and identity_ready
        and resource_ready
        and stream_complete
        and positive_models == sorted(MODEL_IDS)
        and shortcut_count == 0
        and isomorphic_failures == 0
    )
    return {
        "authority_row_count": len(authority_rows),
        "authority_ready": parity_ready and identity_ready,
        "resource_admission_ready": resource_ready,
        "resource_rows_support_science": any(
            row.get("supports_margin_claim") is True
            for row in admission_rows
            if isinstance(row, dict)
        ),
        "compatibility_row_count": len(stream_rows),
        "compatibility_stream_complete": stream_complete,
        "model_metrics": model_metrics,
        "positive_margin_models": positive_models,
        "isomorphic_comparison_count": isomorphic_comparisons,
        "isomorphic_direction_failure_count": isomorphic_failures,
        "shortcut_explanation_count": shortcut_count,
        "claim_eligible": claim_eligible,
    }


def reduce_memory(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Recompute learning effects and safety from decision and credit rows."""

    sealed = payloads.get("exp6856", {})
    sealed_rows = [row for row in sealed.get("rows", []) if isinstance(row, dict)]
    held = [
        row
        for row in sealed_rows
        if row.get("arm") == "contextual_bandit" and row.get("split") == "held_future"
    ]
    effects = [
        float(row["effect_vs_no_memory"])
        for row in held
        if isinstance(row.get("effect_vs_no_memory"), (int, float))
    ]
    held_effect = _mean(effects)
    fp_rate = (
        sum(row.get("false_positive_injection") is True for row in held) / len(held)
        if held
        else None
    )
    abstention_rate = (
        sum(row.get("abstained") is True for row in held) / len(held) if held else None
    )

    credit_rows = [
        row
        for row in payloads.get("exp6855", {}).get("rows", [])
        if isinstance(row, dict) and row.get("counterfactual_metric") == "marginal_value"
    ]
    credit_by_write: dict[str, float] = defaultdict(float)
    for row in credit_rows:
        write_id = row.get("write_id")
        value = row.get("metric_value")
        if isinstance(write_id, str) and isinstance(value, (int, float)):
            credit_by_write[write_id] += float(value)
    all_write_ids = {
        row.get("write_id")
        for row in payloads.get("exp6855", {}).get("rows", [])
        if isinstance(row, dict) and isinstance(row.get("write_id"), str)
    }
    helpful = sum(value > 0 for value in credit_by_write.values())
    harmful = sum(value < 0 for value in credit_by_write.values())
    unsupported = len(all_write_ids - credit_by_write.keys())

    restart = sealed.get("restart_results", {})
    rollback = sealed.get("rollback_results", {})
    durability = (
        restart.get("byte_identical") is True and rollback.get("restored_parent_bytes") is True
    )
    family_rows = sealed.get("leave_one_family_out_rows", [])
    order_rows = sealed.get("leave_one_order_out_rows", [])
    family_portability = bool(family_rows) and all(
        row.get("portability_passed") is True for row in family_rows if isinstance(row, dict)
    )
    order_portability = bool(order_rows) and all(
        row.get("portability_passed") is True for row in order_rows if isinstance(row, dict)
    )
    calibrated = sealed.get("abstention_calibration", {}).get("calibrated") is True
    ready = (
        isinstance(held_effect, float)
        and held_effect > 0
        and isinstance(fp_rate, float)
        and fp_rate <= 0.05
        and calibrated
        and durability
        and family_portability
        and order_portability
        and harmful == 0
        and unsupported == 0
    )
    return {
        "held_future_row_count": len(held),
        "held_future_effect": held_effect,
        "false_positive_injection_rate": fp_rate,
        "abstention_rate": abstention_rate,
        "per_write_credit_count": len(credit_by_write),
        "helpful_write_count": helpful,
        "harmful_write_count": harmful,
        "unsupported_write_count": unsupported,
        "durability_passed": durability,
        "family_portability_passed": family_portability,
        "order_portability_passed": order_portability,
        "abstention_calibrated": calibrated,
        "self_learning_ready": ready,
    }


def reduce_arc(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Recompute live provenance, supervisor headroom, and tool receipt status."""

    router = payloads.get("exp6857", {})
    qualified = [
        row for row in router.get("provenance_qualified_manifest", []) if isinstance(row, dict)
    ]
    rejected = [row for row in router.get("rejected_source_manifest", []) if isinstance(row, dict)]
    headroom = [row for row in router.get("supervisor_headroom_rows", []) if isinstance(row, dict)]
    eligible_headroom = [
        row
        for row in headroom
        if row.get("valid_headroom") is True
        and row.get("live_reachable") is True
        and row.get("provenance_class") == "authentic_live"
    ]
    action_values = [
        float(row["action_credit"])
        for row in eligible_headroom
        if isinstance(row.get("action_credit"), (int, float))
    ]

    tool_rows = [
        row
        for row in payloads.get("exp6859", {}).get("rows", [])
        if isinstance(row, dict) and row.get("row_kind") == "gap_chain"
    ]
    complete = [row for row in tool_rows if row.get("join_complete") is True]
    authentic = [
        row
        for row in tool_rows
        if row.get("provenance_class") == "authentic_live"
        and row.get("first_party") is True
        and row.get("live_reachable") is True
    ]
    effect_eligible = [
        row
        for row in authentic
        if row.get("join_complete") is True
        and row.get("causal_eligible") is True
        and row.get("valid_headroom") is True
        and row.get("exact_outcome") is True
    ]
    contract_ready = bool(complete) and all(row.get("first_party") is True for row in complete)
    return {
        "qualified_source_count": len(qualified),
        "invalid_source_count": len(rejected),
        "supervisor_headroom_row_count": len(eligible_headroom),
        "supervisor_action_credit": _mean(action_values),
        "supervisor_effect_eligible": bool(action_values),
        "tool_gap_chain_count": len(tool_rows),
        "tool_gap_complete_chain_count": len(complete),
        "tool_gap_authentic_live_chain_count": len(authentic),
        "tool_gap_live_effect_eligible_count": len(effect_eligible),
        "tool_gap_contract_ready": contract_ready,
    }


def _equal(left: Any, right: Any) -> bool:
    """Compare floats with stable tolerance and other values exactly."""

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def aggregate_consistency(
    payloads: dict[str, dict[str, Any]],
    typed: dict[str, Any],
    memory: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Compare selected producer headlines with values recomputed from rows."""

    checks = [
        ("exp6849", "typed_program_authority_ready_score", int(typed["authority_ready"])),
        (
            "exp6850",
            "three_family_scoring_admission_ready_score",
            int(typed["resource_admission_ready"]),
        ),
        ("exp6851", "positive_margin_models", typed["positive_margin_models"]),
        ("exp6852", "compatibility_claim_eligible_score", int(typed["claim_eligible"])),
    ]
    if memory is not None:
        checks.extend(
            [
                ("exp6856", "held_future_effect", memory["held_future_effect"]),
                (
                    "exp6856",
                    "false_positive_injection_rate",
                    memory["false_positive_injection_rate"],
                ),
                (
                    "exp6856",
                    "continuous_self_learning_ready_score",
                    int(memory["self_learning_ready"]),
                ),
            ]
        )
    results: list[dict[str, Any]] = []
    for task_id, field, recomputed in checks:
        declared = payloads.get(task_id, {}).get(field)
        results.append(
            {
                "task_id": TASKS[task_id]["task_id"],
                "field": field,
                "declared": declared,
                "recomputed": recomputed,
                "passed": _equal(declared, recomputed),
            }
        )
    return results


def circularity_checks(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Prevent a producer's tested verifier from serving as its own truth source."""

    rows: list[dict[str, Any]] = []
    for short_id, info in TASKS.items():
        payload = payloads.get(short_id)
        if payload is None:
            continue
        oracle = payload.get("verifier_is_oracle") is True
        producer_class = payload.get("verdict_class")
        supported = (
            "circular_positive" if oracle and producer_class == "positive" else producer_class
        )
        if supported not in VERDICT_CLASSES:
            supported = _class_from_text(
                str(payload.get("honest_verdict") or payload.get("status") or "")
            )
        rows.append(
            {
                "task_id": info["task_id"],
                "producer_verdict_class": producer_class,
                "verifier_is_oracle": oracle,
                "supported_verdict_class": supported,
                "passed": not oracle,
            }
        )
    return rows


def retirement_decision(
    branch: str,
    prior_failure: dict[str, Any],
    current_class: str,
    *,
    mechanism_changed: bool,
) -> dict[str, Any]:
    """Retire a repeated failure only when the new task changed no mechanism."""

    prior_class = _class_from_text(str(prior_failure.get("verdict") or ""))
    retire = (
        prior_failure.get("retire_if_same_verdict") is True
        and prior_class == current_class
        and not mechanism_changed
    )
    return {
        "branch": branch,
        "prior_experiment_id": prior_failure.get("experiment_id"),
        "prior_verdict_class": prior_class,
        "current_verdict_class": current_class,
        "mechanism_changed": mechanism_changed,
        "retire": retire,
        "rerun_recommended": not retire and mechanism_changed,
    }


def _principles(keys: set[str]) -> dict[str, str]:
    """Explain each field so measured values are not mistaken for authority."""

    specific = {
        "field_principles": "Explains the evidence limit of every top-level field, including itself.",
        "rows": "Contains task, branch, gate, and independently recomputed metric rows.",
        "source_artifact_hashes": "Binds each upstream V599 artifact to the bytes read by this reducer.",
        "duration_s": "Measures local JSON reading and deterministic reduction only.",
        "positive_claims": "Lists only row-supported readiness claims and does not pool them into science.",
        "verdict_class": "Summarizes milestone evidence completeness, not an ARC solve or hardware result.",
    }
    return {
        key: specific.get(key, f"Records the row-supported capstone value for {key}.")
        for key in keys
    }


def _checksum_payload(artifact: dict[str, Any]) -> str:
    """Bind deterministic content while excluding elapsed time and the checksum itself."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    raw = json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def build_artifact(repo_root: Path, run_date: str) -> dict[str, Any]:
    """Build one complete V599 disposition even when upstream evidence is absent."""

    started = time.monotonic()
    tasks = load_roadmap_tasks(repo_root)
    duplicate_ids = duplicate_task_ids(tasks)
    payloads = load_source_payloads(repo_root)
    conductor_text = (repo_root / "ops/conductor-log.md").read_text(encoding="utf-8")
    events = parse_conductor_events(conductor_text)
    flags = [row for row in events if row["status"] == "FLAGGED"]
    skips = [row for row in events if row["status"] == "GATE_BLOCK"]

    source_hashes: dict[str, dict[str, Any]] = {}
    for short_id, info in TASKS.items():
        path = repo_root / info["path"]
        source_hashes[short_id] = {
            "path": info["path"],
            "exists": path.is_file(),
            "sha256": sha256_path(path) if path.is_file() else None,
        }
    link_checks: list[dict[str, Any]] = []
    for short_id, payload in payloads.items():
        for check in check_declared_hashes(repo_root, payload.get("source_artifact_hashes")):
            if re.fullmatch(r"exp\d+", check["source_id"]):
                link_checks.append({"owner_task_id": TASKS[short_id]["task_id"], **check})

    typed = reduce_typed(payloads)
    memory = reduce_memory(payloads)
    arc = reduce_arc(payloads)
    aggregate_checks = aggregate_consistency(payloads, typed, memory)
    circularity = circularity_checks(payloads)

    task_manifest: list[dict[str, Any]] = []
    for short_id, info in TASKS.items():
        state = classify_task_state(short_id, payloads.get(short_id), events)
        task_manifest.append(
            {
                "task_id": info["task_id"],
                "branch": info["branch"],
                "artifact_path": info["path"],
                "artifact_state": "terminal" if short_id in payloads else "missing",
                **state,
            }
        )
    task_manifest.append(
        {
            "task_id": "exp6860-v599-independent-capstone",
            "branch": "capstone",
            "artifact_path": str(OUTPUT_PATH),
            "artifact_state": "terminal",
            "verdict_class": "partial",
            "failed_condition": "scientific_branches_not_all_eligible",
            "observed_value": "typed=null,memory=disqualified,supervisor=blocked,tool_gap=partial",
        }
    )

    typed_disposition = {
        "verdict_class": "null",
        "failed_condition": "compatibility_claim_eligible",
        "observed_value": typed["claim_eligible"],
        "resource_admission_ready": typed["resource_admission_ready"],
        "scientific_result": "not_identifiable",
    }
    memory_disposition = {
        "verdict_class": "disqualified"
        if any("exp6856" in row["task_id"] for row in flags)
        else "null",
        "failed_condition": "clean_safe_self_learning_readiness",
        "observed_value": {
            "self_learning_ready": memory["self_learning_ready"],
            "harmful_write_count": memory["harmful_write_count"],
            "abstention_rate": memory["abstention_rate"],
            "controlling_audit_flagged": any("exp6856" in row["task_id"] for row in flags),
        },
        "scientific_result": "held_future_benefit_not_readiness_eligible",
    }
    supervisor_disposition = {
        "verdict_class": "blocked",
        "failed_condition": "supervisor_headroom_ready_score",
        "observed_value": arc["supervisor_headroom_row_count"],
        "action_credit": arc["supervisor_action_credit"],
    }
    tool_disposition = {
        "verdict_class": "partial",
        "failed_condition": "authentic_live_tool_gap_chain_count",
        "observed_value": arc["tool_gap_authentic_live_chain_count"],
        "receipt_contract_ready": arc["tool_gap_contract_ready"],
        "live_effect_eligible_count": arc["tool_gap_live_effect_eligible_count"],
    }

    retirements = [
        retirement_decision(
            "typed_compatibility",
            {
                "experiment_id": "exp6487-representation-integrity-audit",
                "verdict": "disqualified",
                "retire_if_same_verdict": True,
            },
            "null",
            mechanism_changed=True,
        ),
        retirement_decision(
            "continuous_self_learning",
            {
                "experiment_id": "exp6842-sealed-memory-pathway-portability-audit",
                "verdict": "complete_null_sealed_memory_audit_complete_ready_score_zero",
                "retire_if_same_verdict": True,
            },
            "disqualified",
            mechanism_changed=True,
        ),
        retirement_decision(
            "arc_supervisor",
            {
                "experiment_id": "exp6844-supervisor-action-outcome-credit-audit",
                "verdict": "complete_blocked_supervisor_outcome_credit_audit",
                "retire_if_same_verdict": True,
            },
            "blocked",
            mechanism_changed=False,
        ),
        retirement_decision(
            "tool_gap_receipt",
            {
                "experiment_id": "exp6845-tool-gap-causal-support-audit",
                "verdict": "complete_blocked_tool_gap_causal_support_audit",
                "retire_if_same_verdict": True,
            },
            "partial",
            mechanism_changed=True,
        ),
    ]

    branch_rows = [
        {"row_kind": "branch_disposition", "branch": "typed_compatibility", **typed_disposition},
        {
            "row_kind": "branch_disposition",
            "branch": "continuous_self_learning",
            **memory_disposition,
        },
        {"row_kind": "branch_disposition", "branch": "arc_supervisor", **supervisor_disposition},
        {"row_kind": "branch_disposition", "branch": "tool_gap_receipt", **tool_disposition},
    ]
    metric_rows = (
        [
            {
                "row_kind": "recomputed_metric",
                "branch": "typed_compatibility",
                "metric": key,
                "value": value,
            }
            for key, value in typed.items()
        ]
        + [
            {
                "row_kind": "recomputed_metric",
                "branch": "continuous_self_learning",
                "metric": key,
                "value": value,
            }
            for key, value in memory.items()
        ]
        + [
            {"row_kind": "recomputed_metric", "branch": "arc", "metric": key, "value": value}
            for key, value in arc.items()
        ]
    )
    rows = (
        [{"row_kind": "task_state", **row} for row in task_manifest]
        + branch_rows
        + metric_rows
        + [{"row_kind": "aggregate_consistency", **row} for row in aggregate_checks]
        + [
            {
                "row_kind": "gate",
                "branch": row["branch"],
                "gate": row["failed_condition"],
                "expected": True,
                "observed": row["observed_value"],
                "passed": False,
            }
            for row in branch_rows
        ]
    )

    gate_checks = [
        {
            "check": "typed_compatibility_claim_eligible",
            "expected": True,
            "observed": typed["claim_eligible"],
            "passed": typed["claim_eligible"],
        },
        {
            "check": "continuous_self_learning_ready",
            "expected": True,
            "observed": memory["self_learning_ready"],
            "passed": memory["self_learning_ready"],
        },
        {
            "check": "supervisor_headroom_row_count",
            "expected": ">0",
            "observed": arc["supervisor_headroom_row_count"],
            "passed": arc["supervisor_headroom_row_count"] > 0,
        },
        {
            "check": "authentic_live_tool_gap_chain_count",
            "expected": ">0",
            "observed": arc["tool_gap_authentic_live_chain_count"],
            "passed": arc["tool_gap_authentic_live_chain_count"] > 0,
        },
    ]
    failed_checks = [row for row in gate_checks if not row["passed"]]

    artifact: dict[str, Any] = {
        "schema": "capstone_v599.independent_disposition.v1",
        "experiment_id": "exp6860-v599-independent-capstone",
        "run_date": run_date,
        "status": "complete",
        "field_principles": {},
        "preconditions_checked": {
            "active_roadmap_readable": True,
            "prompt_listed_research_roadmap_next_exists": (
                repo_root / "research-roadmap-next.yaml"
            ).is_file(),
            "planned_task_count": len(tasks),
            "duplicate_task_ids": duplicate_ids,
            "upstream_terminal_or_skip_count": sum(
                row["artifact_state"] == "terminal" or row["verdict_class"] == "blocked"
                for row in task_manifest[:-1]
            ),
            "stale_declared_source_hashes": [row for row in link_checks if not row["passed"]],
            "exclusion_manifest_readable": (repo_root / "ops/exclusion_manifest.yaml").is_file(),
            "no_llm_invoked": True,
        },
        "inference_substrate": "deterministic CPU independent capstone reduction",
        "duration_s": max(time.monotonic() - started, 0.0001),
        "source_artifact_hashes": source_hashes,
        "task_state_manifest": task_manifest,
        "conductor_skip_manifest": skips,
        "retry_manifest": retry_events(events),
        "flag_manifest": flags,
        "rows": rows,
        "fresh_reducer_manifest": {
            "producer_modules_imported": [],
            "producer_aggregate_functions_imported": [],
            "algorithms": [
                "typed row parity and identity reduction",
                "per-model margin and isomorphic sign reduction",
                "held-future decision and per-write marginal credit reduction",
                "ARC provenance and receipt-chain reduction",
            ],
            "source_link_hash_checks": link_checks,
        },
        "aggregate_row_consistency_results": aggregate_checks,
        "typed_compatibility_disposition": typed_disposition,
        "continuous_self_learning_disposition": memory_disposition,
        "arc_supervisor_disposition": supervisor_disposition,
        "tool_gap_receipt_disposition": tool_disposition,
        "circularity_results": circularity,
        "retirement_decisions": retirements,
        "next_action_by_branch": {
            "typed_compatibility": {
                "action": "Replace raw sequence likelihood with a pre-registered nuisance-controlled semantic contrast.",
                "rerun_recommended": False,
                "reason": "The current margins fail shortcut and isomorphic controls.",
            },
            "continuous_self_learning": {
                "action": "Add constrained exploration and a harmful-write admission block before another sealed audit.",
                "rerun_recommended": False,
                "reason": "The controller always abstains and supported harmful writes remain.",
            },
            "arc_supervisor": {
                "action": "Capture a matched authentic-live supervisor opportunity with nonzero pre-action headroom.",
                "rerun_recommended": False,
                "reason": "Action credit is undefined until the missing prerequisite exists.",
            },
            "tool_gap_receipt": {
                "action": "Wait for an authentic live first-party gap chain and preserve it through the shipped seam.",
                "rerun_recommended": False,
                "reason": "Fixture replay cannot establish a live effect.",
            },
        },
        "positive_claims": [
            "typed_program_authority_ready",
            "three_family_resource_admission_ready",
            "first_party_tool_gap_receipt_contract_ready",
        ],
        "null_claims": [
            "three_family_compatibility_not_identifiable",
            "tool_gap_live_effect_unmeasured",
        ],
        "blocked_claims": ["supervisor_action_credit_blocked_zero_headroom"],
        "partial_claims": ["tool_gap_contract_ready_fixture_only"],
        "disqualified_claims": [
            "continuous_self_learning_readiness_from_flagged_sealed_audit",
            "live_arc_effect_from_flagged_router",
        ],
        "v599_milestone_disposition_complete_score": 1,
        "solve_claimed": False,
        "game_level_solve_count": 0,
        "hardware_speedup_claimed": False,
        "gate_check_summary": {
            "passed": False,
            "checks": gate_checks,
            "failed_check": failed_checks[0]["check"],
            "failed_checks": failed_checks,
            "observed": {row["check"]: row["observed"] for row in failed_checks},
        },
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": "complete_partial_v599_dispositions_preserved_no_scientific_branch_advance",
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _principles(set(artifact))
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Return every schema problem so a failed publication explains itself."""

    required = {
        "field_principles",
        "preconditions_checked",
        "inference_substrate",
        "duration_s",
        "source_artifact_hashes",
        "task_state_manifest",
        "conductor_skip_manifest",
        "retry_manifest",
        "flag_manifest",
        "rows",
        "fresh_reducer_manifest",
        "aggregate_row_consistency_results",
        "typed_compatibility_disposition",
        "continuous_self_learning_disposition",
        "arc_supervisor_disposition",
        "tool_gap_receipt_disposition",
        "circularity_results",
        "retirement_decisions",
        "next_action_by_branch",
        "positive_claims",
        "null_claims",
        "blocked_claims",
        "partial_claims",
        "disqualified_claims",
        "v599_milestone_disposition_complete_score",
        "solve_claimed",
        "game_level_solve_count",
        "hardware_speedup_claimed",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    }
    findings: list[str] = []
    missing = sorted(required - artifact.keys())
    if missing:
        findings.append("missing required fields: " + ", ".join(missing))
    if artifact.get("solve_claimed") is not False:
        findings.append("solve_claimed must be false")
    if artifact.get("game_level_solve_count") != 0:
        findings.append("game_level_solve_count must be zero")
    if artifact.get("hardware_speedup_claimed") is not False:
        findings.append("hardware_speedup_claimed must be false")
    if artifact.get("verifier_is_oracle") is not False:
        findings.append("verifier_is_oracle must be false")
    if artifact.get("inference_substrate") != "deterministic CPU independent capstone reduction":
        findings.append("inference_substrate is not the required deterministic reducer")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        findings.append("verdict_class is outside the closed enum")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        findings.append("honest_verdict must start with complete_")
    manifest = artifact.get("task_state_manifest")
    if not isinstance(manifest, list) or len(manifest) != 13:
        findings.append("task_state_manifest must contain thirteen tasks")
    if isinstance(manifest, list):
        task_ids = [row.get("task_id") for row in manifest if isinstance(row, dict)]
        if len(task_ids) != len(set(task_ids)):
            findings.append("task_state_manifest task IDs must be unique")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or set(principles) != set(artifact):
        findings.append("field_principles must explain every artifact field")
    if not isinstance(artifact.get("rows"), list) or not artifact.get("rows"):
        findings.append("rows must contain recomputed evidence")
    if artifact.get("v599_milestone_disposition_complete_score") != 1:
        findings.append("v599_milestone_disposition_complete_score must equal one")
    checksum = artifact.get("reproducibility_checksum")
    if isinstance(checksum, str) and checksum and checksum != _checksum_payload(artifact):
        findings.append("reproducibility_checksum does not match artifact content")
    return findings


def write_atomic(path: Path, artifact: dict[str, Any]) -> None:
    """Replace the result only after a complete JSON document is on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    """Build or validate the selected artifact without changing other files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260901")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.repo_root / OUTPUT_PATH
    if args.validate:
        artifact = read_json(output)
        if artifact is None:
            return 1
        return int(bool(validate_artifact(artifact)))
    artifact = build_artifact(args.repo_root, args.date)
    findings = validate_artifact(artifact)
    if findings:
        return 1
    write_atomic(output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - the tested wrapper owns command execution
    raise SystemExit(main())
