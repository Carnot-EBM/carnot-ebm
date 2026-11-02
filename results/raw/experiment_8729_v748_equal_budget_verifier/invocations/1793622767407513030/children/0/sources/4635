"""Cold unit reducers and byte custody for Exp7707.

REQ-REPORT-7707 and REQ-CL-7707. All empirical denominators use original
families. The caller keeps fixture controls outside those denominators.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file
from carnot.verify.record_addresses import analyze_answer, index_records, resolve_address


def failed_check(
    check: str, upstream: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Name one failed gate with exact operands and the actual source path."""
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
    }


def audit_inventory(
    root: Path, plan: list[tuple[int, str, str | None]]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Hash each planned producer; never use this audit's own output as an input."""
    rows: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_inputs": []}
    gates: list[dict[str, Any]] = []
    for number, label, ready_field in plan:
        path = root / label
        if number == 7707:
            rows.append({"upstream": f"Exp{number}", "path": label, "state": "planned_output"})
            continue
        if not path.is_file():
            hashes["missing_inputs"].append(label)
            rows.append({"upstream": f"Exp{number}", "path": label, "state": "missing"})
            if 7700 <= number <= 7706:
                gates.append(
                    failed_check("producer_exists", f"Exp{number}", path, "exists", True, False)
                )
            continue
        digest = sha256_file(path)
        try:
            value = json.loads(path.read_bytes())
        except (ValueError, UnicodeDecodeError):
            value = {}
        terminal = str(value.get("honest_verdict", "")).startswith("complete_")
        eligible = (
            terminal
            and value.get("verdict_class") in {"null", "positive", "circular_positive"}
            and value.get("flagged_adversarial") is False
        )
        state = (
            "producer"
            if eligible and (ready_field is None or value.get(ready_field) == 1)
            else "pre_gate"
        )
        hashes["producers" if state == "producer" else "pre_gate_receipts"][label] = digest
        rows.append(
            {
                "upstream": f"Exp{number}",
                "path": label,
                "state": state,
                "sha256": digest,
                "verdict_class": value.get("verdict_class"),
                "honest_verdict": value.get("honest_verdict"),
                "ready_field": ready_field,
                "ready_observed": value.get(ready_field) if ready_field else None,
            }
        )
        if 7700 <= number <= 7706 and state != "producer":
            field = ready_field if eligible and ready_field else "verdict_class"
            observed = value.get(field)
            gates.append(
                failed_check(
                    "producer_eligible",
                    f"Exp{number}",
                    path,
                    field,
                    1 if field == ready_field else "eligible_terminal",
                    observed,
                )
            )
    return rows, hashes, gates


def reduce_decisions(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recompute each Brier and decision cost from label, probability, and action."""
    seen: set[tuple[str, str]] = set()
    families: set[str] = set()
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reduced: list[dict[str, Any]] = []
    for row in rows:
        unit, arm = str(row["unit_id"]), str(row["arm"])
        if (unit, arm) in seen:
            raise ValueError("duplicate_unit_arm")
        seen.add((unit, arm))
        label, probability = row["label"], row["probability_error"]
        if (
            type(label) is not int
            or label not in (0, 1)
            or not isinstance(probability, (float, int))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("invalid_label_probability")
        action = row["typed_action"]
        if action not in {"accept", "reject", "escalate"}:
            raise ValueError("invalid_action")
        brier = (float(probability) - label) ** 2
        cost = 0.2 if action == "escalate" else float((action == "accept") == bool(label))
        if not math.isclose(float(row["brier"]), brier, abs_tol=1e-10):
            raise ValueError("brier_mismatch")
        if not math.isclose(float(row["decision_cost"]), cost, abs_tol=1e-10):
            raise ValueError("cost_mismatch")
        excluded, censored = bool(row.get("excluded", False)), bool(row.get("censored", False))
        item = {
            "unit_id": unit,
            "arm": arm,
            "raw_metrics": {
                "label": label,
                "probability_error": float(probability),
                "brier": brier,
                "decision_cost": cost,
                "action": action,
            },
            "counts": {"independent_family": 1},
            "excluded": excluded,
            "censored": censored,
            "provenance": row.get("provenance", {}),
        }
        reduced.append(item)
        families.add(unit)
        if not excluded:
            by_arm[arm].append(item)
    summary = {
        "independent_families": len(families),
        "row_count": len(reduced),
        "by_arm": {
            arm: {
                "n": len(items),
                "brier_mean": sum(r["raw_metrics"]["brier"] for r in items) / len(items),
                "cost_mean": sum(r["raw_metrics"]["decision_cost"] for r in items) / len(items),
                "non_escalation_coverage": sum(
                    r["raw_metrics"]["action"] != "escalate" for r in items
                )
                / len(items),
                "interval_inputs": [r["raw_metrics"]["brier"] for r in items],
            }
            for arm, items in by_arm.items()
        },
    }
    return reduced, summary


def reduce_feedback(events: list[dict[str, Any]], *, cap: int) -> dict[str, dict[str, int]]:
    """Count charged proposals and later use only after a unique released label."""
    seen_ids: set[str] = set()
    released: set[tuple[str, str]] = set()
    predictions: set[tuple[str, str, int]] = set()
    by_arm: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "spent": 0,
            "charged_rejections": 0,
            "later_predictions": 0,
            "feedback": 0,
            "predictions": 0,
        }
    )
    for event in sorted(events, key=lambda item: item["tick"]):
        event_id, arm, kind = str(event["event_id"]), str(event["arm"]), event["kind"]
        tick = event["tick"]
        if event_id in seen_ids:
            raise ValueError("duplicate_event_id")
        seen_ids.add(event_id)
        state = by_arm[arm]
        if kind == "prediction":
            unit = str(event["unit_id"])
            predictions.add((arm, unit, tick))
            state["predictions"] += 1
            if state["feedback"]:
                state["later_predictions"] += 1
        elif kind == "feedback":
            origin, unit = event["origin_tick"], str(event["unit_id"])
            if origin >= tick:
                raise ValueError("future_feedback")
            if (arm, unit, origin) not in predictions:
                raise ValueError("feedback_without_prediction")
            if (arm, unit) in released:
                raise ValueError("duplicate_feedback")
            released.add((arm, unit))
            state["feedback"] += 1
        elif kind == "proposal":
            state["spent"] += 1
            state["charged_rejections"] += int(not event["accepted"])
            if state["spent"] > cap:
                raise ValueError("credit_cap_exceeded")
        else:
            raise ValueError("event_kind_invalid")
    return dict(by_arm)


def challenge_boundaries() -> dict[str, bool]:
    """Run private attacks outside every empirical family denominator."""
    source = "```\n at café (src/a.js:4:2)\n at café (src/b.js:8:1)\n```"
    records = index_records(source)
    ambiguous = not resolve_address(source, records, quote="at café").valid
    cross = not resolve_address(
        source, records, span=(records[0].byte_end - 2, records[1].byte_start + 2)
    ).valid
    answer = analyze_answer(source, "`café` at `src/a.js:4` because it crashes.")
    unknown = answer["whole_answer_status"] == "unknown" and bool(answer["residual_unknown_text"])
    row = {
        "unit_id": "private",
        "arm": "energy",
        "label": 1,
        "probability_error": 0.8,
        "typed_action": "reject",
        "brier": 0.04,
        "decision_cost": 0.0,
    }
    try:
        reduce_decisions([{**row, "label": 0}])
    except ValueError:
        permuted = True
    else:
        permuted = False
    prediction = {
        "event_id": "p",
        "arm": "priority",
        "kind": "prediction",
        "tick": 1,
        "unit_id": "u",
    }
    feedback = {
        "event_id": "f",
        "arm": "priority",
        "kind": "feedback",
        "tick": 2,
        "origin_tick": 1,
        "unit_id": "u",
    }
    try:
        reduce_feedback([prediction, {**feedback, "origin_tick": 3}], cap=6)
    except ValueError:
        future = True
    else:
        future = False
    try:
        reduce_feedback([prediction, feedback, {**feedback, "event_id": "f2", "tick": 3}], cap=6)
    except ValueError:
        duplicate = True
    else:
        duplicate = False
    return {
        "ambiguous_quote_rejected": ambiguous,
        "cross_record_span_rejected": cross,
        "unknown_clause_uncertified": unknown,
        "permuted_label_rejected": permuted,
        "future_feedback_rejected": future,
        "duplicate_feedback_rejected": duplicate,
    }
