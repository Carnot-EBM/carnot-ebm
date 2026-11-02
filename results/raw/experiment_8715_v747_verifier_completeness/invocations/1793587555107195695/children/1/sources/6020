"""REQ-VERIFY-8264: focal syntax and acquisition cost do not prove benefit.

Public source identities choose the roster before any outputs exist. Every
intended view remains visible even when its control or response is unavailable.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import focal_capture_8263 as capture
from carnot.verify import focal_protocol_8263 as focal
from carnot.verify import sentence_transport_8179 as transport

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8264_v714_evidence_view_canary"
CLI = f"scripts/experiments/{NAME}.py"
VIEWS = ["original", "selected_evidence_deleted", "nonselected_deletion"]


def plan_slots(slots: list[Json], cache: list[Json], count: Callable[[str], int]) -> list[Json]:
    """Select complete public focal candidates without conditioning on control success."""
    result = []
    for row in sorted(slots, key=lambda r: r["source_cluster_id"]):
        cached = [r for r in cache if r["unit_id"] == row["unit_id"]]
        complete = {
            s["sentence_index"]
            for s in transport.partition(bytes.fromhex(row["answer_bytes"]))
            if s["complete"]
        }
        if not any(r["sentence_index"] in complete for r in cached):
            continue
        try:
            plan = focal.views(row, cached, count)
        except ValueError as error:
            plan = dict(status="unavailable", reason=str(error), views={})
        result.append(
            dict(
                unit_id=row["unit_id"],
                source_cluster_id=row["source_cluster_id"],
                original=row,
                cached=cached,
                plan=plan,
            )
        )
    return result


def schedule(plans: list[Json]) -> list[Json]:
    """Rotate the frozen three-view order and bind each request to a unique ID."""
    slots = []
    for index, row in enumerate(plans):
        order = VIEWS[index % 3 :] + VIEWS[: index % 3]
        for name in order:
            if name not in row["plan"]["views"]:
                continue
            view = row["plan"]["views"][name]
            slots.append(
                dict(
                    unit_id=row["unit_id"],
                    source_cluster_id=row["source_cluster_id"],
                    role="fit",
                    condition=name,
                    view=view,
                    request_id=canonical_hash([NAME, row["unit_id"], name, view]),
                )
            )
    return slots


def reduce(work: Json) -> Json:
    """Independently parse exact view/answer custody, retaining missing intended rows."""
    expected = schedule(work["plans"])
    calls = work["calls"]
    by_id = {r["request_id"]: r for r in calls}
    custody = len(by_id) == len(calls) and set(by_id) == {s["request_id"] for s in expected}
    rows, changes, controls, triplets = [], [], [], 0
    for plan in work["plans"]:
        probabilities = {}
        for name in VIEWS:
            slot = next(
                (s for s in expected if s["unit_id"] == plan["unit_id"] and s["condition"] == name),
                None,
            )
            call = by_id.get(slot["request_id"]) if slot else None
            parsed = None
            if call and slot and call["status"] == "completed":
                view = slot["view"]
                valid = (
                    call["resume_key"]
                    == canonical_hash(dict(role=slot["role"], unit_id=slot["unit_id"], view=view))
                    and call["answer_bytes"]
                    == view["answer_bytes"]
                    == plan["original"]["answer_bytes"]
                    and call["source_bytes"] == view["source_bytes"]
                    and call["view_sha256"] == canonical_hash(view)
                    and call["request_sha256"] == canonical_hash(view["request"])
                    and call["response_sha256"] == canonical_hash(call["transcript"])
                )
                custody = custody and valid
                try:
                    parsed = focal.parse(view, call["transcript"], canonical_hash(view["request"]))
                except (ValueError, TypeError):
                    parsed = dict(status="escalated", rows=[])
                if parsed["status"] == "completed" and valid:
                    probabilities[name] = parsed["rows"][0]["p_unsupported"]
            status = "completed" if name in probabilities else "failed" if call else "censored"
            rows.append(
                dict(
                    unit_id=plan["unit_id"],
                    source_cluster_id=plan["source_cluster_id"],
                    condition=name,
                    arm="focal_grammar",
                    status=status,
                    exclusion_reason=None
                    if status == "completed"
                    else (call or {}).get("error")
                    or plan["plan"].get("reason", "invalid_or_missing_response"),
                    numerator=int(status == "completed"),
                    denominator=1,
                    metric="syntax_yield",
                    p_unsupported=probabilities.get(name),
                )
            )
        if len(probabilities) == 3:
            triplets += 1
            changes.append(probabilities[VIEWS[1]] - probabilities[VIEWS[0]])
            controls.append(probabilities[VIEWS[2]] - probabilities[VIEWS[0]])
    missing = sum(
        p["plan"].get("reason") in {"missing_control", "no_second_complete_source_sentence"}
        for p in work["plans"]
    )
    return dict(
        rows=rows,
        complete_triplets=triplets,
        custody_passed=custody,
        syntax_yield=dict(
            numerator=sum(r["status"] == "completed" for r in rows), denominator=len(rows)
        ),
        missing_control_frequency=dict(numerator=missing, denominator=len(work["plans"])),
        selected_probability_changes=changes,
        control_probability_changes=controls,
    )


def forecast(rosters: Json, timings: list[Json], spans: Json) -> Json:
    """Use the slowest observed token rates with full rosters and retry allowance."""
    usable = [
        r
        for r in timings
        if r.get("prompt_n", 0) > 0
        and r.get("predicted_n", 0) > 0
        and r.get("prompt_ms", 0) > 0
        and r.get("predicted_ms", 0) > 0
    ]
    result = {}
    for role, count in [("fit", 128), ("tune", 64), ("reserved", 128)]:
        tokens = rosters.get(role, [])
        estimate = None
        known = [
            r
            for r in tokens
            if r.get("input_tokens") is not None and r.get("output_tokens") is not None
        ]
        if usable and known and all(v is not None for v in spans.values()):
            prefill = max(r["prompt_ms"] / r["prompt_n"] / 1000 for r in usable)
            generation = max(
                max(
                    r["predicted_ms"] / 1000,
                    r.get("response_wall_seconds", 0) - r["prompt_ms"] / 1000,
                )
                / r["predicted_n"]
                for r in usable
            )
            estimate = sum(spans.values()) + 1.25 * sum(
                r["input_tokens"] * prefill
                + r["output_tokens"] * generation
                + spans.get("serialization", 0)
                for r in known
            )
        result[role] = dict(
            intended_sources=count,
            intended_calls=count * 3,
            token_counts=tokens,
            projected_seconds=estimate,
            ready_score=int(estimate is not None and estimate <= 2400 and len(known) == count * 3),
            unavailable_call_count=count * 3 - len(known),
            estimate_scope="full_frozen_roster"
            if len(known) == count * 3
            else "measured_available_calls_lower_bound; missing calls block this branch",
            measurement_budget_seconds=2400,
            validation_budget_seconds=900,
            implementation_closeout_seconds=1200,
            planned_total_seconds=4500,
            conservative_tail="maximum observed per-token cost; 25 percent retry allowance",
        )
    return result


def allow_capture(elapsed: float, projected: float) -> bool:
    """Keep the original roster fixed when elapsed time consumes launch allowance."""
    return bool(projected <= min(2400, 4500 - 1200 - 900 - elapsed))


def decorate(slots: list[Json], rows: list[Json]) -> list[Json]:
    """Attach public custody operands to the qualified capture's primitive replies."""
    return [
        dict(
            row,
            request_id=slot["request_id"],
            source_cluster_id=slot["source_cluster_id"],
            condition=slot["condition"],
            source_bytes=slot["view"]["source_bytes"],
            answer_bytes=slot["view"]["answer_bytes"],
            view_sha256=canonical_hash(slot["view"]),
            request_sha256=canonical_hash(slot["view"]["request"]),
            response_sha256=canonical_hash(row["transcript"]),
        )
        for slot, row in zip(slots, rows, strict=True)
    ]


def private_work(raw: Path) -> Json:
    """Private fixed replies exercise custody only and never receive live provenance."""
    slots = [
        dict(
            unit_id=str(i),
            source_cluster_id=f"{i:02}",
            role="fit",
            source_bytes=b"One sentence. Another sentence.".hex(),
            answer_bytes=b"One sentence.".hex(),
        )
        for i in range(12)
    ]
    cache = [dict(unit_id=str(i), sentence_index=0, p_unsupported=0.8) for i in range(12)]
    plans = plan_slots(slots, cache, lambda _: 1)
    frozen = schedule(plans)
    dispatch = lambda wire: __import__("json").dumps(
        dict(__import__("json").loads(wire), text="0|B|0.50|[]")
    )
    captured = capture.capture(frozen, raw / "capture.jsonl", nullcontext(dispatch))
    return dict(
        plans=plans,
        calls=decorate(frozen, captured["rows"]),
        rosters={},
        timings=[],
        spans={},
        capture=captured,
    )
