"""REQ-REPORT-8914: measure complete durable work without a benefit prerequisite.

The original candidate rule remains unchanged. Each released feedback receives
its own storage barriers, including feedback that never deploys a candidate.
"""

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import resource
import shutil
import time
from typing import Any

import numpy as np
from scipy.stats import t

from carnot.reporting import guarded_update_methods_8905 as methods
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.selective_runner_8816 import stream
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
__all__ = ["require", "methods"]
ROOT, SEED = methods.ROOT, 7628914
NAME, TASK = "experiment_8914_v762_guarded_durable_cost", "exp8914-guarded-durable-cost"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/guarded_durable_cost_8914.py",
    "python/carnot/reporting/guarded_durable_runner_8914.py",
    CLI,
]
TESTS = ["tests/python/test_guarded_durable_cost_8914.py"]
SOURCE = methods.NAME
PIN = "sha256:acbd0cd68c70e51811c36897180a2dc302341f9fa8fa2f381c3c6e642ab16779"
BENEFIT_PIN = "sha256:28f12afb613a933f778248172532fa84c71f474b631b3c9aaa9fd4d296048d74"
g = namespace(methods.g, "durable8914_store")
signed, old = methods.signed, methods.old


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Keep actual counts visible while the fixed deadline continues."""
    print(f"[exp8914] phase={phase} completed={completed} pending={pending}", flush=True)


def roster() -> list[Json]:
    """Freeze fresh paired worlds and randomized arm order before timing."""
    planner = namespace(old, "durable8914_roster")
    planner.SEED = SEED
    rng, rows = random.Random(SEED), []
    for index in range(300, 312):
        plan = planner.plan(index)
        plan["stream_id"] = f"cost8914-{index}"
        for role in ("online", "retention"):
            for i, unit in enumerate(plan[role]):
                unit["unit_id"] = f"{plan['stream_id']}/{role}/{i}"
        arms = list(g.ARMS)
        rng.shuffle(arms)
        for order, arm in enumerate(arms):
            rows.append(
                dict(
                    unit_id=plan["stream_id"] + "/" + arm,
                    cluster_id=plan["stream_id"],
                    seed=plan["seed"],
                    arm=arm,
                    order=order,
                    denominator=48,
                    plan=plan,
                )
            )
    return rows


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate exact producer bytes and all operands before generating worlds."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = sources / (SOURCE + ".json")
    d: Json = dict(
        check="missing_file",
        upstream=methods.TASK,
        artifact_field="guarded_methods_ready_score",
        expected=1,
        observed=None,
        path=str(path),
        hash=None,
    )
    refs: list[Json] = []
    diagnostics: list[Json] = []
    heads: Json = {}
    anchors: list[Json] = []
    progress("preconditions_before")
    try:
        d["hash"] = sha256_file(path)
        d["check"] = "rejected_producer"
        value = json.loads(path.read_bytes())
        require(isinstance(value, dict), "producer_object")
        d["check"] = "missing_field"
        observed = value["guarded_methods_ready_score"]
        d.update(check="observed_zero" if observed == 0 else "rejected_producer", observed=observed)
        require(type(observed) is int and observed == 1, "method_gate")
        d.update(
            check="rejected_producer",
            artifact_field="producer_bytes",
            expected=PIN,
            observed=d["hash"],
        )
        require(
            d["hash"] == PIN
            and value["experiment_id"] == 8905
            and value["task_id"] == methods.TASK
            and value["required_checks_passed"] is True
            and value["protocol"] == methods.protocol(),
            "original_producer",
        )
        refs.append(reference(path))
        operands = (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
        for i, ref in enumerate(operands):
            d.update(path=ref["path"], artifact_field="original_operand", expected=ref["sha256"])
            d["observed"] = d["hash"] = sha256_file(Path(ref["path"]))
            require(d["observed"] == d["expected"], "operand_drift")
            refs.append(ref)
            if i % 50 == 0:
                progress("authenticate_operands", i, len(operands) - i)
        head_path = Path(value["raw_directory"]) / "heads.json"
        require(reference(head_path) in refs, "head_custody")
        heads = json.loads(head_path.read_bytes())
        anchors = value["anchor_manifest"]
        require(anchors == methods.anchors(), "anchor_drift")
        prior = value["protocol"]
        used = {
            u["world_seed"]
            for role in ("pilot_manifest", "frozen_evaluation_manifest")
            for p in prior[role]
            for u in p["online"] + p["retention"]
        }
        require(
            not used & {u["world_seed"] for s in roster() for u in s["plan"]["online"]},
            "cost_world_overlap",
        )
        shutil.copyfile(path, raw / "original_producer.json")
    except (OSError, ValueError, KeyError, TypeError) as error:
        diagnostics.append(
            dict(
                d,
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else d["check"],
                error=str(error),
            )
        )
    progress("preconditions_after", 1, 0)
    return dict(
        external_ready=not diagnostics,
        diagnostics=diagnostics,
        references=refs,
        heads=heads,
        anchors=anchors,
        resources=dict(
            passed=shutil.disk_usage(raw).free > 512 * 1024**2, total_budget_s=4200, hard_cap_s=4800
        ),
    )


StoreBase: Any = g.Store


class Store(StoreBase):  # type: ignore[misc]
    """Keep the original state semantics while measuring real durable acknowledgments."""

    def __init__(self, path: Path, params: Json, arm: str, anchors: list[Json]) -> None:
        self.commits: list[Json] = []
        self.feedback: list[Json] = []
        self.fits: list[Json] = []
        super().__init__(path, params, arm, anchors)
        require(
            self.state["cursor"] == len(self.state["released"])
            and self.state["prediction_cursor"] == len(self.state["splits"])
            and set(self.state["splits"])
            == set(self.state["pending"]) | {r["group_id"] for r in self.state["released"]},
            "stale_cursor",
        )

    def commit(self, state: Json) -> None:
        """Acknowledge only after the file and containing directory reach storage."""
        began = time.perf_counter_ns()
        atomic_json(self.path, state)
        descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.state = deepcopy(state)
        self.commits.append(
            dict(
                cursor=state["cursor"],
                bytes=self.path.stat().st_size,
                duration_ns=time.perf_counter_ns() - began,
                file_fsync=1,
                directory_fsync=1,
            )
        )

    def fit_candidate(self, state: Json) -> None:
        """Charge gradients, guard scoring and rejected candidates using the original rule."""
        began, guard_ns = time.perf_counter_ns(), 0
        original = g.metrics

        def timed(params: Json, rows: list[Json]) -> tuple[float, float]:
            nonlocal guard_ns
            start = time.perf_counter_ns()
            result = original(params, rows)
            guard_ns += time.perf_counter_ns() - start
            return tuple(result)

        g.metrics = timed
        try:
            super().fit_candidate(state)
        finally:
            g.metrics = original
        self.fits.append(
            dict(
                state["promotions"][-1],
                inclusive_ns=time.perf_counter_ns() - began,
                guard_scoring_inclusive_ns=guard_ns,
            )
        )

    def release(self, batch: list[tuple[str, Json]], tick: int) -> None:
        """Retain whole-block validation and permutation, then commit every feedback."""
        began = time.perf_counter_ns()
        state = deepcopy(self.state)
        pending = [state["pending"].get(group) for group, _ in batch]
        require(bool(batch) and all(p is not None for p in pending), "release_unknown")
        for index, ((group, value), sealed) in enumerate(zip(batch, pending, strict=True)):
            require(
                sealed["ordinal"] == state["cursor"] + index + 1
                and tick >= sealed["due"]
                and canonical_hash(value) == sealed["evidence_hash"]
                and state["splits"][group] == sealed["split"],
                "release_boundary",
            )
        labels = [str(g.symbolic.decide(value)["decision"]) for _, value in batch]
        targets = (
            random.Random(g.SEED + state["cursor"]).sample(labels, len(labels))
            if state["arm"] == "guarded_permuted"
            else labels
        )
        state["blocks"].append(
            dict(groups=[group for group, _ in batch], labels=labels, targets=targets, tick=tick)
        )
        for (group, value), label, target in zip(batch, labels, targets, strict=True):
            sealed = state["pending"].pop(group)
            state["released"].append(dict(sealed, label=target, true_label=label))
            key = old.topology(value["query"])
            if key not in state["templates"]:
                state["templates"].append(key)
            state["cursor"] += 1
            if state["cursor"] in g.SCHEDULE:
                self.fit_candidate(state)
            self.commit(state)
            self.feedback.append(
                dict(
                    group_id=group,
                    cursor=state["cursor"],
                    target=target,
                    label=label,
                    duration_ns=time.perf_counter_ns() - began,
                    bytes=self.commits[-1]["bytes"],
                )
            )
            began = time.perf_counter_ns()


def segment(request: Path, stop: int) -> Json:
    """Run a child segment and persist all state needed by a genuinely cold process."""
    progress("segment_before", 0, stop)
    spec = json.loads(request.read_bytes())
    plan = spec["plan"]
    store = Store(request.parent / "state.json", spec["head"], spec["arm"], spec["anchors"])
    initial = deepcopy(store.state)
    values = [
        old.world(
            old.graphs()["development"][u["template"]],
            u["unit_id"] + "/",
            u["error_type"],
            u["world_seed"],
        )
        for u in plan["online"]
    ]
    due = [i + 1 + delay for i, delay in enumerate(plan["delays"])]
    released = store.state["cursor"]
    tick = max(due[:released]) if released else 0
    predictions: list[Json] = []
    while released < stop:
        tick += 1
        if tick <= 48:
            began = time.perf_counter_ns()
            sealed = store.predict(
                plan["online"][tick - 1]["unit_id"], values[tick - 1], tick, due[tick - 1]
            )
            predictions.append(dict(sealed, duration_ns=time.perf_counter_ns() - began))
        while released < stop and tick >= max(due[released : released + 8]):
            store.release(
                [(plan["online"][i]["unit_id"], values[i]) for i in range(released, released + 8)],
                tick,
            )
            released += 8
            progress("feedback_committed", released, 48 - released)
    result = dict(
        initial_state=initial,
        state=store.state,
        predictions=predictions,
        feedback=store.feedback,
        commits=store.commits,
        fits=store.fits,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        pid=os.getpid(),
    )
    atomic_json(request.parent / f"segment-{stop}.json", result)
    progress("segment_after", released, 48 - released)
    return result


def execute(raw: Path, spec: Json, head: Json, anchors: list[Json]) -> Json:
    """Charge cold startup, service and restart checks; keep uninterrupted control costs separate."""
    import sys

    began = time.perf_counter_ns()
    request = raw / "request.json"
    atomic_json(request, dict(spec, head=head, anchors=anchors))
    commands: list[Json] = []
    for stop in (24, 48):
        progress("subprocess_before_" + str(stop), 0, 1)
        commands.append(
            stream(
                "segment" + str(stop),
                [
                    sys.executable,
                    "-u",
                    str(ROOT / CLI),
                    "--segment",
                    str(request),
                    "--stop",
                    str(stop),
                ],
                raw / "logs",
                deadline=120,
            )
        )
        require(commands[-1]["passed"], "service_child_failure")
        progress("subprocess_after_" + str(stop), 1, 0)
    first = json.loads((raw / "segment-24.json").read_bytes())
    last = json.loads((raw / "segment-48.json").read_bytes())
    require(
        first["pid"] != last["pid"] and first["state"] == last["initial_state"],
        "cold_restart_state",
    )
    complete_ns = time.perf_counter_ns() - began
    control_request = raw / "control" / "request.json"
    atomic_json(control_request, dict(spec, head=head, anchors=anchors))
    control_began = time.perf_counter_ns()
    progress("control_subprocess_before", 0, 1)
    control_receipt = stream(
        "control",
        [sys.executable, "-u", str(ROOT / CLI), "--segment", str(control_request), "--stop", "48"],
        raw / "logs",
        deadline=120,
    )
    require(control_receipt["passed"], "control_child_failure")
    control = json.loads((control_request.parent / "segment-48.json").read_bytes())
    strip: Callable[[list[Json]], list[Json]] = lambda rows: [
        {k: v for k, v in row.items() if k != "duration_ns"} for row in rows
    ]
    require(
        strip(first["predictions"] + last["predictions"]) == strip(control["predictions"])
        and last["state"] == control["state"],
        "restart_prediction_drift",
    )
    progress("control_subprocess_after", 1, 0)
    result = dict(
        spec,
        status="complete",
        missing_reason=None,
        metric=complete_ns,
        complete_ns=complete_ns,
        segments=[first, last],
        commands=commands,
        control_receipt=control_receipt,
        control_numerical_training_calls=control["state"]["training_calls"],
        control_duration_ns=time.perf_counter_ns() - control_began,
        restart=dict(
            passed=True,
            feedback_cursor=24,
            before_pid=first["pid"],
            after_pid=last["pid"],
            exact_state=True,
            exact_predictions=True,
        ),
    )
    atomic_json(raw / "unit.json", result)
    return result


def intervals(rows: list[Json]) -> list[Json]:
    """Use only independent complete paired streams for simultaneous total-service intervals."""
    lookup = {(r["cluster_id"], r["arm"]): r for r in rows if r["status"] == "complete"}
    result = []
    for arm in g.ARMS[1:]:
        deltas = [
            lookup[c, arm]["complete_ns"] - lookup[c, g.ARMS[0]]["complete_ns"]
            for c in sorted({r["cluster_id"] for r in rows})
            if (c, arm) in lookup and (c, g.ARMS[0]) in lookup
        ]
        n = len(deltas)
        mean = float(np.mean(deltas)) if n else None
        margin = (
            float(t.ppf(1 - 0.01 / 6, n - 1) * np.std(deltas, ddof=1) / n**0.5) if n > 1 else None
        )
        result.append(
            dict(
                arm=arm,
                control=g.ARMS[0],
                independent_n=n,
                differences_ns=deltas,
                mean_ns=mean,
                lower_ns=mean - margin if mean is not None and margin is not None else None,
                upper_ns=mean + margin if mean is not None and margin is not None else None,
                familywise_confidence=0.99,
                family_size=3,
                method="Bonferroni paired Student-t; whole-service wall time including cold startup; Gaussian stream means assumed",
            )
        )
    return result


def recover(raw: Path, reason: str) -> Json:
    """Preserve the complete scheduled census, including failed cells and zero promotions."""
    rows: list[Json] = []
    service: list[Json] = []
    feedback: list[Json] = []
    durability: list[Json] = []
    restarts: list[Json] = []
    promotions: list[Json] = []
    timers: list[Json] = []
    for spec in roster():
        path = raw / spec["cluster_id"] / spec["arm"] / "unit.json"
        row = (
            json.loads(path.read_bytes())
            if path.is_file()
            else dict(spec, status="censored", metric=None, missing_reason=reason)
        )
        rows.append(row)
        tags = {
            k: row[k]
            for k in (
                "unit_id",
                "cluster_id",
                "arm",
                "seed",
                "denominator",
                "status",
                "missing_reason",
            )
        }
        if row["status"] != "complete":
            service.append(dict(tags, metric=None))
            continue
        segments = row["segments"]
        last = segments[-1]["state"]
        requests = [r for s in segments for r in s["predictions"]]
        releases = [r for s in segments for r in s["feedback"]]
        commits = [r for s in segments for r in s["commits"]]
        fits = [r for s in segments for r in s["fits"]]
        require(
            len(requests) == len(releases) == last["cursor"] == 48
            and len(commits) == 97
            and not last["pending"]
            and row["restart"]["passed"],
            "complete_service_parity",
        )
        service.append(
            dict(
                tags,
                metric=row["complete_ns"],
                complete_ns=row["complete_ns"],
                request_p50_ns=float(np.percentile([r["duration_ns"] for r in requests], 50)),
                request_p95_ns=float(np.percentile([r["duration_ns"] for r in requests], 95)),
                feedback_p50_ns=float(np.percentile([r["duration_ns"] for r in releases], 50)),
                feedback_p95_ns=float(np.percentile([r["duration_ns"] for r in releases], 95)),
                bytes_written=sum(c["bytes"] for c in commits),
                peak_rss_bytes=max(s["peak_rss_bytes"] for s in segments),
                training_examples=last["training_calls"],
                numerical_steps=last["training_calls"],
                guard_scores=last["guard_scores"],
                observations=len(requests),
                template_admissions=len(last["templates"]),
                template_hash=canonical_hash(last["templates"]),
                promotions=sum(p["promoted"] for p in fits),
                rejected_candidates=sum(not p["promoted"] for p in fits),
                rejected_work_inclusive_ns=sum(
                    p["inclusive_ns"] for p in fits if not p["promoted"]
                ),
            )
        )
        feedback.extend(dict(tags, **r) for r in releases)
        durability.extend(dict(tags, **r) for r in commits)
        restarts.append(dict(tags, **row["restart"]))
        promotions.extend(dict(tags, **p) for p in fits)
        timers.extend(
            dict(
                tags,
                timer=key,
                inclusive=True,
                duration_ns=sum(r[key] for r in fits),
                additive_partition=False,
            )
            for key in ("inclusive_ns", "guard_scoring_inclusive_ns")
        )
    counts = Counter(r["status"] for r in rows)
    for cluster in {r["cluster_id"] for r in service}:
        group = [r for r in service if r["cluster_id"] == cluster and r["status"] == "complete"]
        require(
            len(
                {
                    tuple(
                        r[k]
                        for k in (
                            "training_examples",
                            "numerical_steps",
                            "guard_scores",
                            "observations",
                            "template_admissions",
                            "template_hash",
                        )
                    )
                    for r in group
                }
            )
            <= 1,
            "unmatched_service_work",
        )
    return dict(
        rows=rows,
        service_cost_rows=service,
        feedback_cost_rows=feedback,
        durability_rows=durability,
        restart_rows=restarts,
        promotion_cost_rows=promotions,
        inclusive_timer_rows=timers,
        paired_total_service_intervals=intervals(rows),
        intended_count=48,
        completed_count=counts["complete"],
        failed_count=counts["failed"],
        censored_count=counts["censored"],
        excluded_count=0,
        independent_count=sum(
            all(r["status"] == "complete" for r in rows if r["cluster_id"] == cluster)
            for cluster in {r["cluster_id"] for r in rows}
        ),
        adaptation_cost_ready_score=int(counts["complete"] == 48),
        numerical_training_calls=sum(
            s["segments"][-1]["state"]["training_calls"] + s["control_numerical_training_calls"]
            for s in rows
            if s["status"] == "complete"
        ),
    )


def measure(raw: Path, pre: Json) -> Json:
    """Pilot a complete cell, then finish the frozen census within a fixed budget."""
    atomic_json(raw / "roster.json", dict(rows=roster()))
    began = time.monotonic()
    if pre["external_ready"] and pre["resources"]["passed"]:
        for i, spec in enumerate(roster()):
            progress("service_cell_before", i, 48 - i)
            cell = raw / spec["cluster_id"] / spec["arm"]
            try:
                require(time.monotonic() - began < 2400, "measurement_deadline")
                execute(cell, spec, pre["heads"][str(spec["seed"])], pre["anchors"])
                if i == 0:
                    projected = (time.monotonic() - began) * 48 + 1200
                    atomic_json(
                        raw / "pilot_runtime.json",
                        dict(
                            projected_total_s=projected, validation_reserve_s=1200, hard_cap_s=4800
                        ),
                    )
                    require(projected < 4200, "pilot_runtime_projection")
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                atomic_json(
                    cell / "unit.json",
                    dict(spec, status="failed", metric=None, missing_reason=str(error)),
                )
                break
            progress("service_cell_after", i + 1, 47 - i)
    return recover(
        raw, "external_blocked" if not pre["external_ready"] else "unfinished_owned_work"
    )
