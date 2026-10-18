"""REQ-VERIFY-8099-CAPTURE: measure source sensitivity without altered labels.

The original public roster fixes every denominator. Duplicate judgments estimate
transport variation; changed sources measure sensitivity, never human accuracy.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import qwen_development_capture_7995 as prior

Json = dict[str, Any]
risk = prior.risk
ROLES = dict(fit=128, tune=64)
ARMS = ("full_source", "duplicate", "source_removed", "source_mismatched")
ROSTER_KEYS = {
    "family_id",
    "unit_id",
    "source_id",
    "source_cluster_id",
    "role",
    "order",
    "slot",
    "model",
    "response_id",
    "source_ids",
    "task_type",
}


def config() -> Json:
    """Reserve the worst-case output before observing any parse or effect."""
    return dict(
        risk.config(),
        call_limit=288,
        output_tokens=27648,
        intended_families=192,
        latest_launch_s=3150,
        measured_work_cap_s=3270,
    )


def freeze(views: Json) -> list[Json]:
    """Accept public fields only and preserve the producer's sealed hash order."""
    if set(views) != set(ROLES):
        raise ValueError("role_roster")
    slots: list[Json] = []
    seen: set[str] = set()
    clusters: set[str] = set()
    originals: list[Json] = []
    for role, total in ROLES.items():
        view = views[role]
        if set(view) != {"request_rows", "roster"} or any(len(view[k]) != total for k in view):
            raise ValueError("role_count")
        public = risk.custody.index_unique(view["request_rows"], "family_id")
        for i, meta in enumerate(view["roster"]):
            if (
                set(meta) - ROSTER_KEYS
                or meta["role"] != role
                or meta["order"] != i + (128 if role == "tune" else 0)
            ):
                raise ValueError("public_roster_fields_or_order")
            fid, cluster = meta["family_id"], meta["source_cluster_id"]
            if fid in seen or cluster in clusters:
                raise ValueError("cross_role_overlap")
            seen.add(fid)
            clusters.add(cluster)
            original = public[fid]
            row = risk.freeze([original], lambda _: 0)[0]
            slots.append(
                dict(
                    family_id=fid + ":full_source",
                    unit_id=meta["unit_id"],
                    source_id=meta["source_id"],
                    role=role,
                    source_cluster_id=cluster,
                    public_hash=canonical_hash(original),
                    request=row["requests"]["full_source"],
                    visible_ids=row["visible_ids"],
                    public_eligible=row["eligible"],
                    exclusion_reason=None,
                    arm="full_source",
                    condition="complete_original_source",
                    source_bytes=original["source_bytes"],
                    answer_bytes=original["answer_bytes"],
                )
            )
            if role == "fit" and i < 32:
                originals.append(slots[-1])
    for i, full in enumerate(originals):
        donor = originals[(i + 1) % 32]
        for arm in ARMS[1:]:
            control = deepcopy(full)
            control.update(family_id=full["unit_id"] + ":" + arm, arm=arm, condition=arm)
            if arm != "duplicate":
                public = dict(
                    family_id=full["unit_id"],
                    answer_bytes=full["answer_bytes"],
                    source_bytes=donor["source_bytes"]
                    if arm == "source_mismatched"
                    else full["source_bytes"],
                )
                transformed = risk.freeze([public], lambda _: 0)[0]
                erased = arm == "source_removed"
                control.update(
                    request=transformed["requests"]["source_erased" if erased else "full_source"],
                    visible_ids=[] if erased else transformed["visible_ids"],
                    intervention_source_id=None if erased else donor["source_id"],
                )
            slots.append(control)
    return slots


def capture(
    frozen: list[Json],
    runtime: Any,
    raw: Path,
    identity: str,
    *,
    ledger: prior.Ledger,
    deadline_s: float = 3270,
    started: float | None = None,
    token_budget: int = 27648,
    blocked_reason: str | None = None,
) -> list[Json]:
    """Reuse qualified transport and recover an orphan start without another call."""
    slots = deepcopy(frozen)
    receipts = {r["call_id"]: r for r in ledger.rows if r["operation"] == "generation"}
    for i, slot in enumerate(slots):
        if blocked_reason:
            slot.update(public_eligible=False, exclusion_reason=blocked_reason)
        path = raw / f"slot-{i:03d}.json"
        if path.exists():
            checkpoint = json.loads(path.read_text())
            checkpoint["exclusion_reason"] = slot["exclusion_reason"]
            atomic_json(path, checkpoint)
        if slot["family_id"] in receipts and not path.exists():
            receipt = receipts[slot["family_id"]]
            if receipt["status"] != "running" or receipt["request_sha256"] != canonical_hash(
                slot["request"]
            ):
                raise ValueError("lost_raw_receipt")
            atomic_json(
                path,
                dict(
                    slot,
                    capture_identity=identity,
                    started=True,
                    status="running",
                    raw_response={},
                    input_tokens=None,
                    duration_s=0,
                    reserved_tokens=96,
                ),
            )
    rows: list[Json] = list(
        prior.capture(
            slots,
            runtime,
            raw,
            identity,
            ledger=ledger,
            deadline_s=deadline_s,
            started=started,
            token_budget=token_budget,
        )
    )
    for i, row in enumerate(rows):
        error, status = row.get("error", ""), row["parsed"]["status"]
        kind = None
        if not row["parsed"]["completed"]:
            kind = (
                "timeout"
                if error.startswith("TimeoutError")
                else "interrupted"
                if "interrupted" in error
                else "context"
                if error == "input_token_limit"
                else "transport"
                if error
                else "truncation"
                if status == "token_limit"
                else "invalid_probability"
                if status == "invalid_schema"
                else status
            )
        row.update(
            metric="bounded_probability_parse",
            issued_state=identity,
            failure_kind=kind,
            human_target=None,
            probability=row["parsed"]["probability"],
            exclusion_reason=row.get("exclusion_reason")
            or row.get("error")
            or (status if not row["parsed"]["completed"] else None),
        )
        atomic_json(raw / f"slot-{i:03d}.json", row)
    return rows


def reduce(rows: list[Json]) -> Json:
    """Cold-parse primitive calls and compare controls on original source units."""
    seen: set[str] = set()
    counts = dict(
        intended=288, eligible=0, independent=0, completed=0, excluded=0, censored=0, failed=0
    )
    role_valid: dict[str, set[str]] = {r: set() for r in ROLES}
    cells: Json = {}
    for row in rows:
        key = row["family_id"]
        if key in seen or row["role"] not in ROLES or row["arm"] not in ARMS:
            raise ValueError("slot_roster")
        seen.add(key)
        parsed = risk.transport.parse_response(row["raw_response"], row["visible_ids"])
        if (
            parsed != row["parsed"]
            or row["numerator"] != int(parsed["completed"])
            or row["probability"] != parsed["probability"]
            or row["denominator"] != 1
        ):
            raise ValueError("parse_drift")
        counts["eligible"] += int(row["public_eligible"])
        counts["completed"] += int(parsed["completed"])
        counts["excluded"] += int(row["status"] == "excluded")
        counts["censored"] += int(row["status"] == "censored")
        counts["failed"] += int(
            row["status"] == "failed" or row["started"] and not parsed["completed"]
        )
        if row["arm"] == "full_source" and parsed["completed"]:
            role_valid[row["role"]].add(row["source_cluster_id"])
        cells.setdefault(row["unit_id"], {})[row["arm"]] = row
    counts["independent"] = sum(map(len, role_valid.values()))
    diagnostics, paired = {}, []
    for arm in ARMS[1:]:
        differences = []
        control = [r for r in rows if r["arm"] == arm]
        for unit, group in cells.items():
            if arm not in group or "full_source" not in group:
                continue
            full, altered = group["full_source"], group[arm]
            if full["parsed"]["completed"] and altered["parsed"]["completed"]:
                delta = abs(full["probability"] - altered["probability"])
                differences.append(delta)
                paired.append(
                    dict(
                        source_id=full["source_id"],
                        unit_id=unit,
                        arm=arm,
                        full_probability=full["probability"],
                        control_probability=altered["probability"],
                        absolute_difference=delta,
                        human_target=None,
                        numerator=delta,
                        denominator=1,
                    )
                )
        diagnostics[arm] = dict(
            paired_count=len(differences),
            intended_count=32,
            mean_absolute_difference=sum(differences) / len(differences) if differences else None,
            parsed_count=sum(r["parsed"]["completed"] for r in control),
            parse_denominator=32,
        )
    for arm in ARMS[2:]:
        effect = diagnostics[arm]["mean_absolute_difference"]
        noise = diagnostics["duplicate"]["mean_absolute_difference"]
        diagnostics[arm]["difference_from_duplicate_variation"] = (
            effect - noise if effect is not None and noise is not None else None
        )
    roster = Counter(r["role"] for r in rows if r["arm"] == "full_source")
    ready = int(
        roster == ROLES
        and len(rows) == 288
        and len(role_valid["fit"]) >= 96
        and len(role_valid["tune"]) >= 48
    )
    return dict(
        **{k + "_count": v for k, v in counts.items()},
        sample_size_budget=dict(
            counts,
            intended_sources=192,
            intended_controls=96,
            independent_unit="sealed_original_source_cluster",
            call_limit=288,
            output_token_limit=27648,
        ),
        fit_capture_ready_score=ready,
        role_completion_counts={r: len(s) for r, s in role_valid.items()},
        source_intervention_rows=paired,
        duplicate_variation=diagnostics.pop("duplicate"),
        source_effect_diagnostics=diagnostics,
    )
