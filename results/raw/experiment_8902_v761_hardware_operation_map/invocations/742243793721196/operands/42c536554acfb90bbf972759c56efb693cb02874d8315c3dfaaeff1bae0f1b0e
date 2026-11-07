"""REQ-REPORT-8831: compare paid ordered checks on qualified encoded worlds."""

from functools import lru_cache
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.reporting import selective_methods_8829 as base
from carnot.reporting import evidence_operation_meter_8818 as meter
from carnot.reporting import evidence_set_methods_8768 as methods
from carnot.reporting import evidence_qualification_8777 as qualification
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
__all__ = ["base"]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8831_v756_ordered_evidence_cost", "exp8831-ordered-evidence-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/ordered_evidence_{s}_8831.py" for s in ["cost", "runner"]] + [
    CLI
]
TESTS = ["tests/python/test_ordered_evidence_cost_8831.py"]
UPSTREAM = ROOT / "results/experiment_8829_v756_selective_methods.json"
PIN = "sha256:80b78a54f399eed31035062bcde37659e46ee259ce339755067c9819939cad9c"
SEED, CAP, DRAWS = 7568831, 200000, 10000
SCHEDULES = base.SCHEDULES
stable = base.stable


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual boundaries so a fixed deadline never appears to be a stall."""
    print(f"[exp8831] phase={phase} completed={completed} pending={pending}", flush=True)


def conflicts(payload: Json) -> bool:
    """A global contradictory sign prevents acceptance even if another witness exists."""
    signs: dict[tuple[str, ...], set[bool]] = {}
    for s, r, o, sign in payload["facts"]:
        signs.setdefault((s, r, o), set()).add(sign)
    return any(len(v) > 1 for v in signs.values())


@lru_cache(maxsize=1)
def evaluator() -> Any:
    """Reuse frozen solvers, charging their previously unmetered parsing as well."""
    frozen = meter.freeze()
    parse = meter.instrument(methods.parse)
    prior = frozen.__globals__["qualification"].bounded.__globals__["original"].compare
    compare = meter.instrument(
        methods.compare,
        parse=parse,
        **{k: prior.__globals__[k] for k in ["decide", "historical", "historical_verifier"]},
    )
    bounded = meter.instrument(
        qualification.bounded,
        original=SimpleNamespace(parse=parse, compare=compare),
    )
    strong = meter.instrument(base.original.old.strong, original=SimpleNamespace(parse=parse))
    return meter.instrument(
        base.original.old.evaluate, strong=strong, qualification=SimpleNamespace(bounded=bounded)
    ), meter.instrument(conflicts)


def attempt(name: str, payload: Json, cap: int | None = CAP) -> Json:
    """Stop only on a complete typed decision; every unfinished attempt stays censored."""
    if name not in SCHEDULES:
        raise ValueError("schedule")
    evaluate, conflict_check = evaluator()
    counter = meter.Counter(cap)
    token = meter.CURRENT.set(counter)
    start, cpu = time.monotonic_ns(), time.process_time_ns()
    decision, exhausted, conflict = "insufficient", False, False
    calls = []
    try:
        # This dispatch lookup is real schedule work under the same ceiling.
        counter.charge("lookup_attempts")
        order = base.CHECKS[name]
        conflict = conflict_check(payload)
        for arm in order:
            counter.charge("other_iterations")
            decision, invocations = evaluate(arm, payload)
            calls.append(dict(arm=arm, decision=decision, evaluator_invocations=invocations))
            counter.charge("other_comparisons")
            if arm == "local_record":
                counter.charge("other_comparisons")
                if decision == "support":
                    decision = "insufficient"
            counter.charge("other_comparisons")
            if conflict:
                decision = "insufficient"
            counter.charge("other_comparisons")
            if decision != "insufficient":
                break
    except meter.Exhausted:
        decision, exhausted = "insufficient", True
    finally:
        meter.CURRENT.reset(token)
    end, cpu_end = time.monotonic_ns(), time.process_time_ns()
    reason = (
        "operation_budget_exhausted"
        if exhausted
        else "conflicting_signs"
        if conflict
        else "complete_witness"
        if decision == "support"
        else "explicit_refutation"
        if decision == "refute"
        else "no_complete_decisive_evidence"
    )
    return dict(
        decision=decision,
        primitive_limit=cap,
        primitive_operations=counter.total,
        operation_counts=counter.counts,
        budget_exhausted=exhausted,
        calls=calls,
        stop_reason=reason,
        successful_bindings=int(decision == "support"),
        wall_time_s=(end - start) / 1e9,
        cpu_time_s=(cpu_end - cpu) / 1e9,
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )


def measure(inputs: list[Json]) -> list[Json]:
    """Labels and construction metadata never enter a measured solver or its order."""
    evaluator()
    progress("before_benchmark", 0, len(inputs) * 3)
    rows = []
    last = time.monotonic()
    for unit in inputs:
        for arm in SCHEDULES:
            rows.append(
                dict(
                    attempt(arm, unit["input"]),
                    arm=arm,
                    seed=SEED,
                    **{
                        k: unit[k]
                        for k in ["unit_id", "cluster_id", "split", "condition", "input_sha256"]
                    },
                )
            )
        if time.monotonic() - last >= 30:
            progress("benchmark", len(rows), len(inputs) * 3 - len(rows))
            last = time.monotonic()
    progress("after_benchmark", len(rows), 0)
    return rows


def score(decisions: list[Json], inputs: list[Json]) -> list[Json]:
    """Independent labels open only after the caller durably seals every prediction."""
    units = {u["unit_id"]: u for u in inputs}
    expected = {(u, a) for u in units for a in SCHEDULES}
    if len(decisions) != len(expected) or {(r["unit_id"], r["arm"]) for r in decisions} != expected:
        raise ValueError("attempt_roster")
    labels = {}
    last = time.monotonic()
    progress("before_independent_labels", 0, len(units))
    for u, v in units.items():
        labels[u] = base.truth(v["input"])
        if time.monotonic() - last >= 30:
            progress("independent_labels", len(labels), len(units) - len(labels))
            last = time.monotonic()
    progress("after_independent_labels", len(labels), 0)
    return [
        dict(
            r,
            truth=labels[r["unit_id"]],
            metric=int(r["decision"] != "insufficient" and r["decision"] == labels[r["unit_id"]]),
            denominator=1,
            raw_numerator=int(
                r["decision"] != "insufficient" and r["decision"] == labels[r["unit_id"]]
            ),
            raw_denominator=1,
            valid_alternative=units[r["unit_id"]]["valid_alternative"],
            status="censored" if r["budget_exhausted"] else "completed",
            missing_reason="operation_budget_exhausted" if r["budget_exhausted"] else None,
        )
        for r in decisions
    ]


def select(rows: list[Json]) -> Json:
    """Choose cost on calibration at exact comparator quality, retaining evaluation losers."""
    if not rows or any(r["split"] != "calibration" for r in rows):
        raise ValueError("calibration_only")
    reference = {r["unit_id"]: r["decision"] for r in rows if r["arm"] == "always_joint"}
    menu = []
    for arm in SCHEDULES[:2]:
        sample = [r for r in rows if r["arm"] == arm]
        menu.append(
            dict(
                schedule=arm,
                paid_operations=sum(r["primitive_operations"] for r in sample),
                matched_decisions=all(r["decision"] == reference[r["unit_id"]] for r in sample),
            )
        )
    eligible = [r for r in menu if r["matched_decisions"]]
    chosen = (
        min(eligible, key=lambda r: (r["paid_operations"], r["schedule"])) if eligible else menu[0]
    )
    return dict(
        schedule=chosen["schedule"],
        menu=menu,
        calibration_rows_sha256=canonical_hash(rows),
        frozen_before_evaluation=True,
    )


def reduce(rows: list[Json], selected: Json) -> Json:
    """Resample paired complete worlds; variants never become independent samples."""
    worlds = []
    for wid in sorted({r["cluster_id"] for r in rows}):
        pair = {}
        for arm in [selected["schedule"], "always_joint"]:
            sample = [r for r in rows if r["cluster_id"] == wid and r["arm"] == arm]
            pair[arm] = dict(
                operations=sum(r["primitive_operations"] for r in sample) / len(sample),
                correct=sum(r["metric"] for r in sample) / len(sample),
                false_accept=sum(
                    r["decision"] == "support" and r["truth"] != "support" for r in sample
                )
                / len(sample),
                coverage=sum(r["decision"] != "insufficient" for r in sample) / len(sample),
                wall_time_s=sum(r["wall_time_s"] for r in sample) / len(sample),
            )
        worlds.append(dict(cluster_id=wid, arms=pair))
    chosen = [w["arms"][selected["schedule"]] for w in worlds]
    reference = [w["arms"]["always_joint"] for w in worlds]
    indices = np.random.default_rng(SEED).integers(0, len(worlds), (DRAWS, len(worlds)))
    estimates: dict[str, float] = {}
    intervals: Json = {}
    for name, key in [
        ("paid_operation_reduction", "operations"),
        ("correct_decisive_coverage_loss", "correct"),
        ("semantic_false_accept_increase", "false_accept"),
        ("decisive_coverage", "coverage"),
    ]:
        a = np.array([w[key] for w in chosen])
        b = np.array([w[key] for w in reference])
        if key == "operations":
            draws = 1 - a[indices].mean(axis=1) / b[indices].mean(axis=1)
            estimate = 1 - a.mean() / b.mean()
        else:
            delta = b - a if key == "correct" else a - b if key == "false_accept" else a
            draws, estimate = delta[indices].mean(axis=1), delta.mean()
        low, high = np.quantile(draws, [0.00625, 0.99375])
        intervals[name] = dict(
            estimate=float(estimate),
            lower=float(low),
            upper=float(high),
            confidence=0.9875,
            draws=DRAWS,
            cluster="whole_world",
            iid_certificate=False,
        )
        estimates[name] = float(estimate)
    rules: list[tuple[str, str, float, bool]] = [
        ("paid_operation_reduction", "lower", 0.20, True),
        ("correct_decisive_coverage_loss", "upper", 0.01, False),
        ("semantic_false_accept_increase", "upper", 0.01, False),
        ("decisive_coverage", "lower", 0.30, True),
    ]
    gates = {
        name: dict(
            passed=bool(
                intervals[name][bound] >= threshold
                if minimum
                else intervals[name][bound] <= threshold
            ),
            threshold=threshold,
            bound=bound,
            principle="Paired world uncertainty must retain exact decision quality at lower paid work",
        )
        for name, bound, threshold, minimum in rules
    }
    wall_delta = sum(w["wall_time_s"] for w in chosen) - sum(w["wall_time_s"] for w in reference)
    return dict(
        per_world_results=worlds,
        whole_world_intervals=intervals,
        acceptance_gates=gates,
        schedule_benefit_score=int(all(g["passed"] for g in gates.values())),
        secondary_wall_time=dict(
            selected_minus_reference_s=wall_delta,
            boundaries="dispatch, preprocessing, parsing, all scheduled checks; excludes unrestricted labels",
            agrees_with_paid_direction=bool(
                (wall_delta <= 0) == (estimates["paid_operation_reduction"] >= 0)
            ),
        ),
    )


def controls() -> list[Json]:
    """Hand specified adversaries expose absence, broken joins, conflicts and alternate witnesses."""
    edges = [[i, (i + 1) % 8] for i in range(8)]
    fixtures = base.original.variants(edges, 2, False)
    fixtures.append(
        dict(
            condition="valid_alternative", input=base.original.variants(edges, 2, True)[0]["input"]
        )
    )
    fixtures.append(
        dict(
            condition="explicit_refutation",
            input=dict(facts=[["a", "r", "b", False]], query=[["a", "r", "b"]]),
        )
    )
    fixtures.append(
        dict(
            condition="two_valid_witnesses",
            input=dict(
                facts=[["a", "r", "b", True], ["c", "r", "b", True]], query=[["?x", "r", "b"]]
            ),
        )
    )
    result = []
    for unit in fixtures:
        expected = (
            "support"
            if unit["condition"]
            in [
                "intact",
                "irrelevant_padding",
                "reordered_facts",
                "valid_alternative",
                "two_valid_witnesses",
            ]
            else "refute"
            if unit["condition"] == "explicit_refutation"
            else "insufficient"
        )
        measured = {arm: attempt(arm, unit["input"]) for arm in SCHEDULES}
        result.append(
            dict(
                condition=unit["condition"],
                input=unit["input"],
                expected=expected,
                measured=measured,
                passed=all(r["decision"] == expected for r in measured.values()),
                no_change=unit["condition"] in ["irrelevant_padding", "reordered_facts"],
                valid_alternatives=2
                if unit["condition"] == "two_valid_witnesses"
                else int(expected == "support"),
            )
        )
    return result
