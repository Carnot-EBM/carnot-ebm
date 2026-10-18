"""REQ-REPORT-8064: causal CPU learning with durable one-use admission.

The same event equations drive a cold reader. Labels enter those equations only
at their original release slots; repeated seeds do not add independent sources.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from carnot import experiment_8058_v698_sealed_evidence_methods as methods
from carnot.experiment_8021_v695_typed_decision_test import action, loss
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.verify import causal_online_8025 as old

Json = dict[str, Any]
Array = NDArray[np.float64]
CONFIG = methods.methods()["learning"]
ARMS = CONFIG["arms"]
START = time.monotonic()
FIELDS = dict(
    issue="issued_prediction_rows",
    release="feedback_release_rows",
    candidate="candidate_commit_rows",
    consume="admission_consumption_rows",
    alpha="alpha_check_rows",
    commit="durable_commit_rows",
    pending="pending_update_rows",
    budget="update_budget_rows",
    seal="final_head_seals",
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so a waiting process cannot resemble completed work."""
    print(
        f"[exp8064] {phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


class Journal:
    """Persist an immutable prefix so a crash cannot apply an update twice."""

    def __init__(self, raw: Path):
        raw.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(raw / "ledger.sqlite")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, kind TEXT, payload TEXT)"
        )
        for verb in ("UPDATE", "DELETE"):
            self.db.execute(
                f"CREATE TRIGGER IF NOT EXISTS forbid_{verb} BEFORE {verb} ON events BEGIN SELECT RAISE(ABORT,'immutable'); END"
            )
        self.prefix = self.db.execute("SELECT kind,payload FROM events ORDER BY seq").fetchall()
        self.index = 0

    def emit(self, kind: str, row: Json) -> None:
        """Compare resumed bytes before appending each synchronous transaction."""
        payload = json.dumps(row, sort_keys=True)
        if self.index < len(self.prefix):
            if (kind, payload) != self.prefix[self.index]:
                raise ValueError("prefix_drift")
        else:
            with self.db:
                self.db.execute(
                    "INSERT INTO events(seq,kind,payload) VALUES(?,?,?)",
                    (self.index, kind, payload),
                )
        self.index += 1

    def close(self) -> None:
        """Release the writer before a cold reader opens the completed journal."""
        self.db.close()


def adequate(y: Array, minimum: int) -> bool:
    """Missing rows or classes must defer without outcome-selected replacements."""
    return len(y) >= minimum and all(int(sum(y == c)) >= 2 for c in (0, 1))


def operands(head: Json, theta: Array, x: Array, y: Array) -> Json:
    """Typed losses retain per-row false accepts for paired baseline comparisons."""
    a, b = head["calibration"]
    ps = old.expit(a + b * (x @ theta))
    acts = [action(float(p)) for p in ps]
    return dict(
        probabilities=ps.tolist(),
        brier=float(np.mean((ps - y) ** 2)),
        typed_cost=float(np.mean([loss(v, int(t)) for v, t in zip(acts, y, strict=True)])),
        false_accepts=[
            i for i, (v, t) in enumerate(zip(acts, y, strict=True)) if v == "accept" and t == 1
        ],
    )


def guard(
    head: Json, incumbent: Array, candidate: Array, initial: Array, x: Array, y: Array
) -> tuple[list[Json], float | None]:
    """Both guarded arms use these exact comparisons; only label timing differs."""
    checks: list[Json] = []
    if not adequate(y, 4):
        return checks, None
    original = operands(head, initial, x, y)
    current = operands(head, incumbent, x, y)
    for alpha in methods.methods()["guard"]["alphas"]:
        proposed = operands(head, incumbent + alpha * (candidate - incumbent), x, y)
        reasons = []
        for baseline, name, margins in [
            (original, "initial", (0.01, 0.02)),
            (current, "incumbent", (0.0, 0.0)),
        ]:
            for metric, margin in zip(("brier", "typed_cost"), margins, strict=True):
                if proposed[metric] > baseline[metric] + margin:
                    reasons.append(name + "." + metric)
            if set(proposed["false_accepts"]) - set(baseline["false_accepts"]):
                reasons.append(name + ".false_accept")
        checks.append(
            dict(
                alpha=alpha,
                initial=original,
                incumbent=current,
                candidate=proposed,
                passed=not reasons,
                reasons=reasons,
                numerator=len(y),
                denominator=len(y),
            )
        )
    passing = [r["alpha"] for r in checks if r["passed"]]
    return checks, float(passing[0]) if passing else None


def trajectory(data: Json, seed: int, emit: Any, release: Any) -> None:
    """Advance original slots, committing proposals before that slot's feedback."""
    sources, head = data["sources"], data["head"]
    vectors = np.asarray(
        [
            old.design(head, r) if r["public_eligible"] else np.zeros(len(head["parameters"]))
            for r in sources
        ]
    )
    initial = np.asarray(head["parameters"], dtype=float)
    states = {arm: initial.copy() for arm in ARMS}
    released: Json = {}
    updates: list[int] = []
    admissions: list[int] = []
    consumed: set[int] = set()
    counts = {arm: 0 for arm in ARMS}
    pending: Json = {}
    a, b = head["calibration"]

    def write(kind: str, row: Json) -> None:
        emit(kind, deepcopy(dict(seed=seed, **row)))

    def eligible(i: int) -> bool:
        return bool(sources[i]["public_eligible"] and sources[i].get("eligible", True))

    def labels(ids: list[int]) -> Array:
        return np.asarray([released[str(i)] for i in ids], dtype=float)

    for slot, source in enumerate(sources):
        for arm in ARMS:
            p = (
                float(old.expit(a + b * float(vectors[slot] @ states[arm])))
                if source["public_eligible"]
                else None
            )
            write(
                "issue",
                dict(
                    slot=slot,
                    source=source["source_cluster_id"],
                    family_id=source["family_id"],
                    arm=arm,
                    probability=p,
                    action=action(p),
                    head_hash=canonical_hash(states[arm].tolist()),
                ),
            )
        if slot in CONFIG["attempt_slots"]:
            if pending:
                write(
                    "pending",
                    dict(
                        slot=slot,
                        arm="shared",
                        status="censored",
                        reason="next_attempt",
                        candidate_slot=pending["slot"],
                        consumed_slots=pending["observed"],
                    ),
                )
                pending = {}
            ids = updates[-32:]
            if not adequate(labels(ids), 16):
                write(
                    "pending",
                    dict(slot=slot, arm="shared", status="deferred", reason="update_support"),
                )
            else:
                candidates = {}
                for arm in ARMS[1:]:
                    theta = states[arm].copy()
                    for _ in range(4):
                        theta -= 0.01 * (
                            b
                            * (
                                vectors[ids].T
                                @ (old.expit(a + b * (vectors[ids] @ theta)) - labels(ids))
                            )
                            / len(ids)
                            + 0.002 * theta
                        )
                    candidates[arm] = theta.tolist()
                    counts[arm] += 4
                fresh = [
                    i
                    for i, r in enumerate(sources)
                    if i + 20 > slot
                    and eligible(i)
                    and methods.partition(r["source_cluster_id"]) == 0
                    and i not in consumed
                ][:12]
                pending = dict(
                    slot=slot,
                    candidates=candidates,
                    incumbents={arm: states[arm].tolist() for arm in ARMS},
                    candidate_hashes={
                        arm: canonical_hash(theta) for arm, theta in candidates.items()
                    },
                    incumbent_hashes={arm: canonical_hash(states[arm].tolist()) for arm in ARMS},
                    update_slots=ids,
                    reused_slots=list(admissions),
                    fresh_slots=fresh,
                    observed=[],
                )
                write("candidate", dict(pending, arm="shared"))
        origin = slot - 20
        if origin >= 0:
            y = release(origin)
            if y is not None and (type(y) is not int or y not in (0, 1)):
                raise ValueError("label_contract")
            released[str(origin)] = y
            valid = eligible(origin) and y is not None
            role = (
                "admission"
                if methods.partition(sources[origin]["source_cluster_id"]) == 0
                else "update"
            )
            write(
                "release",
                dict(
                    slot=origin,
                    release_slot=slot,
                    source=sources[origin]["source_cluster_id"],
                    family_id=sources[origin]["family_id"],
                    arm="shared",
                    y=y,
                    eligible=valid,
                    feedback_role=role,
                ),
            )
            if valid:
                (admissions if role == "admission" else updates).append(origin)
            if pending and origin in pending["fresh_slots"] and valid:
                consumed.add(origin)
                pending["observed"].append(origin)
                write(
                    "consume",
                    dict(
                        slot=origin,
                        release_slot=slot,
                        arm="fresh_admission",
                        candidate_slot=pending["slot"],
                        source=sources[origin]["source_cluster_id"],
                        y=y,
                    ),
                )
            if pending and len(pending["observed"]) == 12:
                fresh, reused = pending["fresh_slots"], pending["reused_slots"]
                ready = adequate(labels(fresh), 4) and adequate(labels(reused), 4)
                for arm in ARMS:
                    checks: list[Json] = []
                    alpha: float | None = 1.0 if arm == "unconditional" else None
                    if arm in ("reused_guard", "fresh_admission") and ready:
                        guard_ids = reused if arm == "reused_guard" else fresh
                        checks, alpha = guard(
                            head,
                            states[arm],
                            np.asarray(pending["candidates"][arm]),
                            initial,
                            vectors[guard_ids],
                            labels(guard_ids),
                        )
                        for check in checks:
                            write(
                                "alpha",
                                dict(
                                    check,
                                    slot=slot,
                                    candidate_slot=pending["slot"],
                                    arm=arm,
                                    guard_slots=guard_ids,
                                ),
                            )
                    if alpha is not None:
                        candidate = np.asarray(pending["candidates"][arm])
                        states[arm] += alpha * (candidate - states[arm])
                    write(
                        "commit",
                        dict(
                            slot=slot,
                            candidate_slot=pending["slot"],
                            arm=arm,
                            alpha=alpha,
                            status="frozen"
                            if arm == "frozen"
                            else "deferred"
                            if alpha is None
                            else "zero"
                            if alpha == 0
                            else "accepted",
                            parameters=states[arm].tolist(),
                            head_hash=canonical_hash(states[arm].tolist()),
                        ),
                    )
                pending = {}
    if pending:
        write(
            "pending",
            dict(
                slot=len(sources),
                arm="shared",
                status="censored",
                reason="stream_end",
                candidate_slot=pending["slot"],
                consumed_slots=pending["observed"],
            ),
        )
    for arm in ARMS:
        write(
            "budget",
            dict(arm=arm, gradients=counts[arm], cap=12, numerator=counts[arm], denominator=12),
        )
        write(
            "seal",
            dict(
                arm=arm,
                parameters=states[arm].tolist(),
                head_hash=canonical_hash(states[arm].tolist()),
            ),
        )


def measure(data: Json, raw: Path, *, budget_s: float = 1200) -> Json:
    """Resume verified seed prefixes, then seal all heads before retention access."""
    inputs = {k: data[k] for k in ("head", "sources", "seeds")}
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != inputs:
        raise ValueError("input_drift")
    atomic_json(path, inputs)
    vault = data.get("labels")
    deadline = time.monotonic() + budget_s
    costs = []
    total = len(data["seeds"])
    progress("benchmark_before", 0, total)
    for index, seed in enumerate(data["seeds"]):
        journal = Journal(raw / f"seed-{seed}")
        began = time.process_time_ns()

        def emit(kind: str, row: Json) -> None:
            if time.monotonic() > deadline:
                raise TimeoutError("numerical_budget")
            journal.emit(kind, row)

        def release(origin: int) -> Any:
            nonlocal vault
            if vault is None:
                vault = {
                    r["family_id"]: r["eligible_y"]
                    for r in json.loads(checked(data["target_reference"]).read_text())["rows"]
                }
            return vault[data["sources"][origin]["family_id"]]

        try:
            trajectory(inputs, seed, emit, release)
        finally:
            journal.close()
        costs.append(
            dict(
                seed=seed,
                cpu_ns=time.process_time_ns() - began,
                scope="gradient, scans, prefix replay and synchronous journal CPU",
            )
        )
        progress("seed_complete", index + 1, total - index - 1)
    if not (raw / "cpu_costs.json").exists():
        atomic_json(raw / "cpu_costs.json", dict(rows=costs))
    result = reduce(raw)
    atomic_json(raw / "final_head_seals.json", dict(rows=result["final_head_seals"]))
    progress("benchmark_after", total, 0)
    return result


def reduce(raw: Path) -> Json:
    """Rebuild every issue, gradient and guard without trusting producer totals."""
    data = json.loads((raw / "inputs.json").read_text())
    result: Json = {field: [] for field in FIELDS.values()}
    for seed in data["seeds"]:
        db = sqlite3.connect(f"file:{raw / f'seed-{seed}' / 'ledger.sqlite'}?mode=ro", uri=True)
        events = [
            (kind, json.loads(payload))
            for kind, payload in db.execute("SELECT kind,payload FROM events ORDER BY seq")
        ]
        db.close()
        labels = {row["slot"]: row["y"] for kind, row in events if kind == "release"}
        expected: list[tuple[str, Json]] = []
        trajectory(data, seed, lambda k, r: expected.append((k, r)), lambda i: labels[i])
        if expected != events:
            raise ValueError("event_order_or_operand_drift")
        for kind, row in events:
            result[FIELDS[kind]].append(row)
    result["cpu_update_costs"] = json.loads((raw / "cpu_costs.json").read_text())["rows"]
    first_seed = data["seeds"][0]
    feedback = {r["slot"]: r for r in result["feedback_release_rows"] if r["seed"] == first_seed}
    rows = []
    for r in result["issued_prediction_rows"]:
        released = feedback.get(r["slot"])
        valid = bool(released and released["eligible"])
        y = released["y"] if valid else None
        rows.append(
            dict(
                r,
                unit=f"stream/{r['slot']}",
                numerator=loss(r["action"], y) if valid else None,
                denominator=int(valid),
                brier=(r["probability"] - y) ** 2 if valid else None,
                y=y,
                status="completed" if valid else "censored" if released is None else "excluded",
                exclusion_reason=None
                if valid
                else "unreleased_tail"
                if released is None
                else "ineligible_complete_target",
            )
        )
    result["rows"] = rows
    n = len(data["sources"])
    completed = sum(r["eligible"] for r in feedback.values())
    result.update(
        intended_count=n,
        eligible_count=completed,
        independent_count=completed,
        completed_count=completed,
        censored_count=n - len(feedback),
        excluded_count=len(feedback) - completed,
        failed_count=0,
    )
    result["sample_size_budget"] = {
        k: result[k + "_count"]
        for k in (
            "intended",
            "eligible",
            "independent",
            "completed",
            "censored",
            "excluded",
            "failed",
        )
    }
    result["sample_size_budget"].update(seeds_are_independent=False, independent_datasets=1)
    decisions = [r["slot"] for r in result["durable_commit_rows"]]
    later = [
        r
        for r in rows
        if r["denominator"]
        and decisions
        and r["slot"] > min(decisions)
        and feedback[r["slot"]]["feedback_role"] == "update"
    ]
    result["per_seed_false_accept_rows"] = [
        dict(
            seed=seed,
            arm=arm,
            numerator=sum(
                r["action"] == "accept" and r["y"] == 1
                for r in later
                if r["seed"] == seed and r["arm"] == arm
            ),
            denominator=sum(r["seed"] == seed and r["arm"] == arm for r in later),
        )
        for seed in data["seeds"]
        for arm in ARMS
    ]
    result["later_prediction_rows"] = later
    return result
