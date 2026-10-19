"""REQ-REPORT-8566: prove discrete identities using bounded, data-only operands.

This private study does not change the shared audit. Comparing event identities,
array values and assignment lists lets equal real counts remain visible while
corrupt evidence fails independently of the artifact's names or verdict.
"""

from __future__ import annotations

from copy import deepcopy
import math
import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash as digest
from scripts import adversarial_verify as av

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8566_v736_count_semantics_controls"
TASK = "exp8566-count-semantics-controls"
MODEL_SPECS: list[Json] = []
CONTRACT = dict(
    version="discrete-counter-proof-v1",
    max_rows=50000,
    max_coordinates=50000,
    max_total_coordinates=1000000,
    claim_type="exact numeric JSON integer; booleans and floats rejected",
    nonnegative=True,
    table_identity="SHA256 of canonical bounded data, plus unique namespace/event_id",
    operations={
        "rows": "events",
        "admitted": "events",
        "label_released": "events",
        "applied": "events",
        "scale_changes": "changes",
        "assignments": "assignments",
        "coordinates": "coordinates",
        "effective": "coordinates",
        "unique": "coordinates",
    },
    policy="No code evaluation, names, ID allowlists, verdicts or self-declared equality exemptions",
)
KEYS = dict(
    rows="completed_count",
    admitted="admission_events",
    label_released="label_release_events",
    applied="applied_update_events",
    scale_changes="decay_scale_changes",
    assignments="physical_coordinate_writes",
    coordinates="changed_coordinate_count",
    effective="effective_coefficient_changes",
    unique="unique_changed_coordinates",
)


def reduce(rows: list[Json]) -> Json:
    """Recompute every counter so saved totals cannot certify their own correctness."""
    if not rows or len(rows) > 50000:
        raise ValueError("table_bounds")
    totals = dict.fromkeys([*KEYS.values(), "intended_count"], 0)
    seen: set[str] = set()
    unique: set[tuple[str, int]] = set()
    work_budget = 0
    for row in rows:
        identity = row["namespace"] + ":" + row["event_id"]
        if identity in seen:
            raise ValueError("duplicate_event")
        seen.add(identity)
        before, after, writes = row["before"], row["after"], row["assignment_indices"]
        left, right = before["parameters"], after["parameters"]
        work_budget += len(left)
        if work_budget > 1000000:
            raise ValueError("work_budget")
        if not row["applied"] and (
            left != right or writes or before["decay_scale"] != after["decay_scale"]
        ):
            raise ValueError("inactive_update")
        if (
            len(left) != len(right)
            or len(left) > 50000
            or len(writes) > 50000
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in [*left, *right, before["decay_scale"], after["decay_scale"]]
            )
            or any(type(i) is not int or not 0 <= i < len(left) for i in writes)
            or any(type(row[k]) is not bool for k in ("admitted", "label_released", "applied"))
        ):
            raise ValueError("invalid_event")
        changed = [i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]
        values = dict(
            rows=1,
            admitted=int(row["admitted"]),
            label_released=int(row["label_released"]),
            applied=int(row["applied"]),
            scale_changes=int(before["decay_scale"] != after["decay_scale"]),
            assignments=len(writes),
            coordinates=len(changed),
            effective=sum(
                a * before["decay_scale"] != b * after["decay_scale"]
                for a, b in zip(left, right, strict=True)
            ),
            unique=0,
        )
        for operation, key in KEYS.items():
            totals[key] += values[operation]
        unique.update((row["namespace"], i) for i in changed)
    totals["intended_count"] = len(rows)
    totals["unique_changed_coordinates"] = len(unique)
    return totals


def prove(claim: Json, tables: Json) -> int:
    """An exact typed claim must match a hash-bound restricted table reduction."""
    operation = claim["operation"]
    if (
        type(claim["value"]) is not int
        or claim["value"] < 0
        or operation not in KEYS
        or claim["unit"] != CONTRACT["operations"][operation]
        or claim["table"] not in tables
    ):
        raise ValueError("invalid_claim")
    table = tables[claim["table"]]
    if digest(table) != claim["table_sha256"]:
        raise ValueError("table_hash")
    actual = reduce(table["rows"])[KEYS[operation]]
    if actual != claim["value"]:
        raise ValueError("mismatched_total")
    return int(actual)


def critical(value: Json) -> list[Json]:
    """Use the unchanged audit projection so no metric disappears from its view."""
    flags: list[av.Flag] = []
    av.check_tautology(av._normalize_principle_wrapped_fields(av._flatten_metrics(value)), flags)
    return [f.to_dict() for f in flags if f.kind == "TAUTOLOGY" and f.severity == "critical"]


def corpus() -> Json:
    """Freeze controls before the next experiment can change validator behavior."""

    def event(i: int) -> Json:
        return dict(
            namespace="synthetic",
            event_id=str(i),
            admitted=True,
            label_released=True,
            applied=True,
            before=dict(parameters=[0.0], decay_scale=1.0),
            after=dict(parameters=[1.0], decay_scale=0.9),
            assignment_indices=[0],
        )

    tables = {str(n): dict(rows=[event(i) for i in range(n)]) for n in (2400, 1800)}
    wide = event(0)
    wide["before"]["parameters"] = [0.0] * 47472
    wide["after"]["parameters"] = [1.0] * 47472
    wide["assignment_indices"] = list(range(47472))
    tables["47472"] = dict(rows=[wide])

    def claim(n: int, op: str) -> Json:
        return dict(
            value=n,
            unit=CONTRACT["operations"][op],
            operation=op,
            table=str(n),
            table_sha256=digest(tables[str(n)]),
        )

    cases = [
        dict(
            id=f"equal_{n}",
            value=n,
            left=claim(n, a),
            right=claim(n, b),
            expected="valid_discrete_identity",
        )
        for n, a, b in [
            (2400, "rows", "admitted"),
            (1800, "applied", "scale_changes"),
            (47472, "coordinates", "assignments"),
        ]
    ]
    unequal = deepcopy(cases[0])
    unequal.update(id="unequal", expected="valid_unequal")
    unequal["value"] = 2
    tables["unequal"] = dict(rows=[event(0), dict(event(1), admitted=False)])
    for side, op, n in [("left", "rows", 2), ("right", "admitted", 1)]:
        unequal[side] = dict(
            value=n,
            unit="events",
            operation=op,
            table="unequal",
            table_sha256=digest(tables["unequal"]),
        )
    cases.append(unequal)
    mutations = {
        "wrong_total": dict(value=2399),
        "absent": dict(table="absent"),
        "wrong_units": dict(unit="seconds"),
        "boolean": dict(value=True),
        "float": dict(value=2400.0),
        "negative": dict(value=-1),
        "nonfinite": dict(value="NaN"),
        "infinite": dict(value="Infinity"),
        "code": dict(operation="eval"),
        "hash": dict(table_sha256="wrong"),
    }
    for name, mutation in mutations.items():
        case = deepcopy(cases[0])
        case.update(id=name, expected="reject_count_evidence")
        case["left"].update(mutation)
        if name in {"nonfinite", "infinite"}:
            case["raw_numeric_json"] = mutation["value"]
        cases.append(case)
    tables["duplicate"] = dict(rows=[event(0), event(0)])
    case = deepcopy(cases[0])
    case.update(id="duplicate", expected="reject_count_evidence")
    case["left"].update(table="duplicate", table_sha256=digest(tables["duplicate"]))
    cases.append(case)
    for name, unit in [("loss", "loss"), ("latency", "seconds")]:
        cases.append(dict(id=name, value=0.913123456, expected="critical_continuous", unit=unit))
    return dict(version="count-controls-v1", tables=tables, cases=cases, contract=CONTRACT)


def classify(value: Json) -> list[Json]:
    """Record old critical results beside sealed semantic expectations, without exemptions."""
    rows = []
    for case in value["cases"]:
        actual = "critical_continuous"
        if case["expected"] != "critical_continuous":
            try:
                claim = dict(case["left"])
                if "raw_numeric_json" in case:
                    claim["value"] = json.loads(case["raw_numeric_json"])
                left = prove(claim, value["tables"])
                right = prove(case["right"], value["tables"])
                actual = "valid_discrete_identity" if left == right else "valid_unequal"
            except (ValueError, KeyError, TypeError):
                actual = "reject_count_evidence"
        measured = case.get("value", 2400)
        old = critical(
            dict(
                observed_alpha=case.get("left", {}).get("value", measured),
                observed_beta=case.get("right", {}).get("value", measured),
            )
        )
        rows.append(
            dict(
                unit_id=case["id"],
                value=measured,
                claimed_left_value=case.get("left", {}).get("value"),
                claimed_right_value=case.get("right", {}).get("value"),
                expected=case["expected"],
                actual=actual,
                expectation_met=actual == case["expected"],
                old_critical=bool(old),
                old_findings=old,
                status="completed",
                missing_reason=None,
            )
        )
    return rows


def authenticate(root: Path) -> Json:
    """Keep the original primary hash fixed even if an attacker rehashes its envelope."""
    from carnot.reporting.count_semantics_evidence_8566 import authenticate as checked

    return checked(root)


def measure(root: Path, raw: Path) -> Json:
    """Use the small evidence adapter rather than a historical validation controller."""
    from carnot.reporting.count_semantics_evidence_8566 import measure as measured

    return measured(root, raw)
