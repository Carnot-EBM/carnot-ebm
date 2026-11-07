"""REQ-REPORT-8856: collect released-feedback evidence without selecting on gain.

These small CPU heads see constructed signed records. Exact symbolic truth can
check causality, but cannot establish correctness on natural language.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time
from typing import Any

from carnot.reporting import linear_ablation_methods_8854 as a
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work boundaries so durable execution remains observable."""
    print(f"[exp8856] phase={phase} completed={completed} pending={pending}", flush=True)


def semantic(value: Any) -> Any:
    """Exclude measured clocks from comparisons of independently repeated work."""
    if isinstance(value, dict):
        return {
            k: semantic(v) for k, v in value.items() if k not in {"wall_clock_ns", "duration_s"}
        }
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def measured(prediction: list[float], truth: str) -> Json:
    """Keep proper loss and decisive mistakes separate from abstention."""
    row = a.score(prediction, truth, 0.5)
    row.update(
        nll=-math.log(max(prediction[a.c.m.LABELS.index(truth)], 1e-30)),
        decisive_error=int(
            row["covered"]
            and row["action"]
            != {"support": "accept", "refute": "reject", "insufficient": "escalate"}[truth]
        ),
    )
    return row


def run_stream(raw: Path, plan: Json, heads: dict[int, Json]) -> Json:
    """Cache pure verifier calls only; every state transition still reaches disk."""
    with a.prior.upstream.primitives():
        return _run_stream(raw, plan, heads)


def _run_stream(raw: Path, plan: Json, heads: dict[int, Json]) -> Json:
    """Seal each prediction before release and restart the same durable store at24."""
    work: Json = dict(
        stream=plan,
        rows=[],
        feedback_rows=[],
        rejected_feedback_rows=[],
        template_reuse_rows=[],
        movement_rows=[],
        retention_rows=[],
        restart_rows=[],
        final_state_refs=[],
    )
    worlds = [a.payload(w) for w in plan["online"]]
    retained = [a.payload(w) for w in plan["retention"]]
    for arm, settings in a.ARMS.items():
        store = a.c.Store(raw / (arm + ".json"), deepcopy(heads[plan["seed"]]))
        identity = dict(cluster_id=plan["stream_id"], seed=plan["seed"], arm=arm)
        online: dict[str, Json] = {}
        moves = []

        def retention(checkpoint: int) -> None:
            before = canonical_hash(store.state)
            for ordinal, (recipe, value) in enumerate(zip(plan["retention"], retained), 1):
                counts = a.c.m.observe(value, store.state["templates"])["counts"]
                prediction = a.c.probabilities(store.state["parameters"], counts)
                truth = a.c.m.decide(value)["decision"]
                work["retention_rows"].append(
                    dict(
                        identity,
                        unit_id=recipe["unit_id"] + f":step{checkpoint}",
                        split="retention",
                        checkpoint=checkpoint,
                        ordinal=ordinal,
                        error_type=recipe["error_type"],
                        prediction=prediction,
                        truth=truth,
                        counts=counts,
                        state_hash=before,
                        learning_state_unchanged=before == canonical_hash(store.state),
                        wall_clock_ns=time.time_ns(),
                        denominator=1,
                        status="complete",
                        missing_reason=None,
                        **measured(prediction, truth),
                    )
                )
            a.require(before == canonical_hash(store.state), "retention_mutation")

        retention(0)
        delays = [0] * 48 if settings["diagnostic"] else plan["delays"]
        for ordinal in range(1, 61):
            if ordinal <= 48:
                event = f"q{ordinal}"
                before = canonical_hash(store.state)
                sealed = store.predict(event, worlds[ordinal - 1], 0.5)
                a.require(
                    before == sealed["state_hash"] == canonical_hash(store.active), "query_mutation"
                )
                store.close()
                online[event] = dict(
                    identity,
                    unit_id=plan["online"][ordinal - 1]["unit_id"],
                    split="online",
                    checkpoint=None,
                    ordinal=ordinal,
                    delay=delays[ordinal - 1],
                    error_type=plan["online"][ordinal - 1]["error_type"],
                    future_window=ordinal >= 25,
                    sealed=sealed,
                    prediction_seal=canonical_hash(sealed),
                    prediction=sealed["prediction"],
                    wall_clock_ns=time.time_ns(),
                    denominator=1,
                    status="sealed",
                    missing_reason="feedback_pending",
                )
                for template in sealed["template_uses"]:
                    work["template_reuse_rows"].append(
                        dict(
                            identity,
                            event_id=event,
                            ordinal=ordinal,
                            template=template,
                            provenance=deepcopy(store.state["template_provenance"][template]),
                            prediction_seal=canonical_hash(sealed),
                            future_window=ordinal >= 25,
                            wall_clock_ns=time.time_ns(),
                        )
                    )
            for at, delay in enumerate(delays, 1):
                if at + delay != ordinal:
                    continue
                event = f"q{at}"
                value = worlds[at - 1]
                truth = str(a.c.m.decide(value)["decision"])
                before = canonical_hash(store.state)
                if at == 1:
                    rejected = a.release(store, event, value, ordinal + 1, arm, delay, label=truth)
                    a.require(
                        rejected["status"] == "invalid" and before == canonical_hash(store.state),
                        "rejection_mutation",
                    )
                    work["rejected_feedback_rows"].append(
                        dict(
                            identity,
                            event_id=event,
                            reason="wrong_release_time",
                            wall_clock_ns=time.time_ns(),
                            **rejected,
                        )
                    )
                row = a.release(store, event, value, ordinal, arm, delay, label=truth)
                a.require(row["status"] == "complete", "release_integrity")
                template = a.c.m.normalize(value)
                row.update(
                    identity,
                    delay=delay,
                    state_before=before,
                    template=template,
                    addition_witness=deepcopy(
                        store.state.get("template_provenance", {}).get(template)
                    ),
                    wall_clock_ns=time.time_ns(),
                )
                work["feedback_rows"].append(row)
                moves.append(row)
                online[event].update(
                    measured(online[event]["prediction"], truth),
                    truth=truth,
                    status="complete",
                    missing_reason=None,
                )
                if at == 1:
                    committed = store.path.read_bytes()
                    rejected = a.release(store, event, value, ordinal, arm, delay, label=truth)
                    a.require(
                        rejected["status"] == "invalid" and committed == store.path.read_bytes(),
                        "duplicate_mutation",
                    )
                    work["rejected_feedback_rows"].append(
                        dict(
                            identity,
                            event_id=event,
                            reason="duplicate",
                            wall_clock_ns=time.time_ns(),
                            **rejected,
                        )
                    )
            a.require(
                len(store.state["pending"]) <= 32 and len(store.state["templates"]) <= 32,
                "capacity",
            )
            if ordinal == 24:
                before = canonical_hash(store.state)
                store = a.c.Store(store.path, {})
                unchanged = before == canonical_hash(store.state)
                a.require(unchanged, "restart_drift")
                work["restart_rows"].append(
                    dict(
                        identity,
                        checkpoint=24,
                        before=before,
                        after=canonical_hash(store.state),
                        unchanged=unchanged,
                        wall_clock_ns=time.time_ns(),
                    )
                )
                retention(24)
            if ordinal in (24, 48, 60):
                progress(plan["stream_id"] + "/" + arm, ordinal, 60 - ordinal)
        a.require(not store.state["pending"] and len(store.state["acks"]) == 48, "pending_drain")
        retention(48)
        work["rows"].extend(online.values())
        work["final_state_refs"].append(dict(identity, **a.prior.old.reference(store.path)))
        for low, high in ((1, 24), (25, 48)):
            selected = [r for r in moves if low <= r["prediction_ordinal"] <= high]
            work["movement_rows"].append(
                dict(
                    identity,
                    prediction_window=[low, high],
                    optimizer_steps=sum(r["optimizer_steps"] for r in selected),
                    gradient_norm_sum=sum(r["gradient_norm"] for r in selected),
                    parameter_path_length=sum(r["parameter_delta"] for r in selected),
                    template_admissions=sum(r["template_added"] for r in selected),
                )
            )
    work["rows"].extend(work["retention_rows"])
    integrity(work)
    return work


def integrity(work: Json) -> Json:
    """Recompute the paired census and causal identities without imposing a gain gate."""
    expected = a.census(dict(cohort_manifest=[work["stream"]]), "")
    key = lambda row: (row["unit_id"], row["arm"])
    rows = work["rows"]
    a.require(
        len(rows) == len(expected) and {key(r) for r in rows} == {key(r) for r in expected},
        "row_census",
    )
    online = {}
    for row in rows:
        a.require(
            all(row[k] == v for k, v in measured(row["prediction"], row["truth"]).items())
            and row["status"] == "complete"
            and row["missing_reason"] is None
            and row["denominator"] == 1,
            "row_integrity",
        )
        if row["split"] == "online":
            a.require(row["prediction_seal"] == canonical_hash(row["sealed"]), "prediction_seal")
            online[(row["arm"], row["ordinal"])] = row
        else:
            a.require(row["learning_state_unchanged"], "retention_mutation")
    prefixes: dict[str, list[str]] = {arm: [] for arm in a.ARMS}
    for row in work["feedback_rows"]:
        arm, at = row["arm"], row["prediction_ordinal"]
        sealed = online[(arm, at)]
        prefixes[arm].append(row["verified_label"])
        label = (
            prefixes[arm][(row["feedback_ordinal"] * 7) % len(prefixes[arm])]
            if "shuffled" in arm
            else row["verified_label"]
        )
        a.require(
            row["feedback_ordinal"] == at + sealed["delay"]
            and row["prediction_hash"] == sealed["prediction_seal"]
            and row["verified_label"] == sealed["truth"]
            and row["released_prefix"] == prefixes[arm]
            and row["feedback_label"] == label,
            "release_integrity",
        )
    a.require(
        len(work["feedback_rows"]) == 288
        and all(len(p) == 48 for p in prefixes.values())
        and len(work["restart_rows"]) == 6
        and all(r["unchanged"] for r in work["restart_rows"])
        and all(r["status"] == "invalid" for r in work["rejected_feedback_rows"]),
        "transition_census",
    )
    return dict(passed=True, completed_count=len(rows), feedback_count=288)


def result_rows(work: Json, reason: str) -> list[Json]:
    """Retain every frozen identity and denominator when owned work is unfinished."""
    key = lambda row: (row["unit_id"], row["arm"])
    completed = {key(row): row for stream in work["stream_results"] for row in stream["rows"]}
    return [completed.get(key(row), row) for row in a.census(work["protocol"], reason)]


def descriptives(rows: list[Json]) -> list[Json]:
    """Group observed losses by delay regime and interval without an acceptance claim."""
    groups: dict[tuple[str, str, str, int], list[Json]] = {}
    for row in rows:
        if row["split"] != "online" or row["status"] != "complete":
            continue
        regime = "delay8" if int(row["cluster_id"].rsplit("-", 1)[1]) < 12 else "delay4_8_12"
        interval = "1_24" if row["ordinal"] <= 24 else "25_48"
        groups.setdefault((row["arm"], regime, interval, row["delay"]), []).append(row)
    return [
        dict(
            arm=arm,
            delay_regime=regime,
            interval=interval,
            feedback_delay=delay,
            denominator=len(group),
            brier=sum(r["metric"] for r in group) / len(group),
            nll=sum(r["nll"] for r in group) / len(group),
            decisive_error=sum(r["decisive_error"] for r in group),
            covered=sum(r["covered"] for r in group),
            diagnostic_only=a.ARMS[arm]["diagnostic"],
        )
        for (arm, regime, interval, delay), group in sorted(groups.items())
    ]
