"""REQ-REPORT-8053: compare the same guarded work through Python and PyO3.

The binding computes real gradients and calibrated probabilities. Policy decisions
stay in Python, so its conversion and guard overhead remain charged to native work.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np

from carnot import experiment_8040_v696_native_transaction_cost as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.verify import feedback_constrained_8051 as learning

Json = dict[str, Any]
CONFIG: Json = dict(
    seed=8053,
    warmups=5,
    repetitions=30,
    draws=10000,
    budget_s=600,
    tolerance=1e-10,
    statistic="ratio of summed paired elapsed times",
    interval="paired percentile bootstrap; repeats and seeds are not independent n",
    memory="same complete released update and guard pools",
    disk="SQLite synchronous FULL; fsynced atomic JSON checkpoint; directory fsync",
    scaling_sizes=[1, 4, 16],
    thread_limit=1,
)
BEGAN = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real elapsed time and counts so a quiet child remains observable."""
    print(
        f"[exp8053] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} "
        f"completed={completed} pending={pending}",
        flush=True,
    )


def fixtures(head: Json) -> list[Json]:
    """Artificial reset and boundary operands supply no natural workload credit."""
    h = copy.deepcopy(head)
    h.update(parameters=[0.0] * 110, decay_scale=1.0, calibration=[0.0, 1.0])
    x = np.zeros((8, 110))
    x[:, 0] = 1
    result = []
    for arm in ("unconstrained", "feedback_constrained"):
        for name, theta, delta in [
            ("accepted", 0.0, 0.0),
            ("rejected", 0.0, 10.0),
            ("reset", 10.0, 0.0),
        ]:
            if arm == "unconstrained" and name != "accepted":
                continue
            result.append(
                dict(
                    identity=f"synthetic/{arm}/{name}",
                    arm=arm,
                    seed=0,
                    slot=0,
                    natural=False,
                    **{"class": name},
                    before=[theta] * 110,
                    delta=[delta] * 110,
                    guard_design=x.tolist(),
                    guards=[[0, i % 2] for i in range(8)],
                    updates=[],
                    labels=[],
                    released=[],
                    synthetic_head=h,
                )
            )
    return result


def guard(
    head: Json, theta: Any, delta: Any, initial: Any, x: Any, y: Any, arm: str, native: Any
) -> tuple[Json, int, int, int]:
    """Use the installed probability formula for every baseline and alpha scan.

    The finite exposed guard checks empirical Brier, typed cost and new false
    accepts. It supplies no certificate for unseen decisions.
    """
    if native is None:
        return learning.guard(head, theta, delta, initial, x, y, arm), 0, 0, 0
    diagnostics = []
    kernel_ns = calls = ffi_ns = 0
    ready = len(y) >= 8 and min(int(sum(y == c)) for c in (0, 1)) >= 2
    if ready:

        def operands(parameters: Any) -> Json:
            nonlocal kernel_ns, calls, ffi_ns
            started = time.perf_counter_ns()
            h = dict(head, parameters=parameters.tolist(), decay_scale=1.0)
            ps, elapsed, _ = native.RustNumericalUpdate8027(json.dumps(h)).update_batch(
                x.tolist(), [None] * len(y)
            )
            ffi_ns += time.perf_counter_ns() - started - elapsed
            kernel_ns += elapsed
            calls += 1
            acts = [prior.old.action(p) for p in ps]
            return dict(
                probabilities=ps,
                actions=acts,
                brier=float(np.mean((np.array(ps) - y) ** 2)),
                typed_cost=float(
                    np.mean([learning.loss(a, int(v)) for a, v in zip(acts, y, strict=True)])
                ),
                false_accepts=[
                    i
                    for i, (a, v) in enumerate(zip(acts, y, strict=True))
                    if a == "accept" and v == 1
                ],
            )

        baseline = operands(initial)
        for alpha in learning.protocol.METHODS["guard"]["alphas"]:
            candidate = operands(theta + alpha * delta)
            reasons = []
            if candidate["brier"] > baseline["brier"] + 1e-12:
                reasons.append("brier")
            if candidate["typed_cost"] > baseline["typed_cost"] + 1e-12:
                reasons.append("typed_cost")
            added = sorted(set(candidate["false_accepts"]) - set(baseline["false_accepts"]))
            if added:
                reasons.append("new_false_accept")
            diagnostics.append(
                dict(
                    alpha=alpha,
                    baseline=baseline,
                    candidate=candidate,
                    new_false_accepts=added,
                    reasons=reasons,
                    admissible=not reasons,
                    numerator=len(y),
                    denominator=len(y),
                )
            )
    passing = [r["alpha"] for r in diagnostics if r["admissible"]]
    alpha = 1 if arm == "unconstrained" else passing[0] if passing else 0 if not ready else None
    return (
        dict(
            alpha=alpha,
            parameters=(initial if alpha is None else theta + alpha * delta).tolist(),
            diagnostics=diagnostics,
            reset=alpha is None,
            rejected=alpha != 1,
            status="waiting_guard" if not ready else "reset" if alpha is None else "commit",
        ),
        kernel_ns,
        calls,
        ffi_ns,
    )


def calculate(data: Json, w: Json, native: Any, path: str, *, reextract: bool = True) -> Json:
    """Replay each original gradient, all guard tests and a serialized restart."""
    components: Json = {}
    started = time.perf_counter_ns()
    head = copy.deepcopy(w.get("synthetic_head", data["head"]))
    initial = prior.old.coefficients(head)
    head.update(parameters=w["before"], decay_scale=1.0)
    rows = [data["sources"][i] for i in w["updates"] + [g[0] for g in w["guards"]]]
    raw = []
    for r in rows:
        public = (
            prior.old.features.extract(
                {k: r[k] for k in ("family_id", "source_bytes", "answer_bytes")}
            )["values"]
            if reextract
            else r["features"]
        )
        if public != r["features"]:
            raise ValueError("public_feature_drift")
        raw.append([r["q"], *public])
    components["feature_gather_ns"] = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    state = native.RustNumericalUpdate8027(json.dumps(head)) if path == "native" else None
    matrix = (
        (
            np.asarray(state.design(raw))
            if state is not None
            else prior.old.python_design(head, np.asarray(raw))
        )
        if raw
        else np.empty((0, 110))
    )
    components["design_and_initialization_ns"] = time.perf_counter_ns() - started
    count = len(w["updates"])
    if len(w["labels"]) != count or any(type(y) is not int or y not in (0, 1) for y in w["labels"]):
        raise ValueError("label_contract")
    started = time.perf_counter_ns()
    kernel = calls = hot = 0
    if count:
        if state is not None:
            _, kernel, hot = state.update_batch(matrix[:count].tolist(), w["labels"])
            calls += 1
            proposed = np.asarray(state.effective())
        else:
            _, kernel, hot = prior.old.python_batch(head, matrix[:count], w["labels"])
            proposed = prior.old.coefficients(head)
        delta = proposed - np.asarray(w["before"])
    else:
        delta = np.asarray(w["delta"])
    gradient_span = time.perf_counter_ns() - started
    components["gradient_arithmetic_ns"] = hot
    components["gradient_construction_ns"] = kernel - hot
    components["ffi_ns"] = gradient_span - kernel if state is not None else 0
    components["gradient_orchestration_ns"] = gradient_span - kernel if state is None else 0
    started = time.perf_counter_ns()
    x = np.asarray(w.get("guard_design", matrix[count:]), dtype=float)
    y = np.asarray([g[1] for g in w["guards"]])
    result, scan_ns, scan_calls, scan_ffi = guard(
        head,
        np.asarray(w["before"]),
        delta,
        initial,
        x,
        y,
        w["arm"],
        native if path == "native" else None,
    )
    calls += scan_calls
    components["guard_scans_ns"] = time.perf_counter_ns() - started - scan_ffi
    components["ffi_ns"] += scan_ffi
    return dict(
        result=result,
        delta=delta.tolist(),
        components=components,
        native_calls=calls,
        native_kernel_ns=(kernel + scan_ns) if state is not None else 0,
        head=dict(head, parameters=result["parameters"], decay_scale=1.0),
        guard_count=len(y),
        parameter_count=110,
        update_size=count,
    )


def comparison(a: Json, b: Json) -> Json:
    """Keep numeric tolerance separate from exact actions and policy decisions."""
    coefficient_error = float(np.max(np.abs(np.asarray(a["parameters"]) - b["parameters"])))
    probability_error = 0.0
    exact = all(a[k] == b[k] for k in ("alpha", "reset", "rejected", "status"))
    exact = exact and len(a["diagnostics"]) == len(b["diagnostics"])
    for x, y in zip(a["diagnostics"], b["diagnostics"], strict=True):
        exact = exact and all(
            x[k] == y[k] for k in ("alpha", "reasons", "admissible", "new_false_accepts")
        )
        for role in ("baseline", "candidate"):
            probability_error = max(
                probability_error,
                float(
                    np.max(np.abs(np.asarray(x[role]["probabilities"]) - y[role]["probabilities"]))
                ),
            )
            exact = exact and x[role]["actions"] == y[role]["actions"]
    return dict(
        coefficient_error=coefficient_error,
        probability_error=probability_error,
        exact_decisions=exact,
        passed=exact and max(coefficient_error, probability_error) <= CONFIG["tolerance"],
    )


def parity(data: Json, native: Any, raw: Path) -> Json:
    """Replay all current decisions, keeping the original producer as a third operand."""
    rows = []
    progress("parity_before", 0, len(data["cases"]))
    for source in data["sources"]:
        if source["public_eligible"]:
            actual = prior.old.features.extract(
                {k: source[k] for k in ("family_id", "source_bytes", "answer_bytes")}
            )["values"]
            if actual != source["features"]:
                raise ValueError("parity_public_feature_drift")
    for i, w in enumerate(data["cases"]):
        a = calculate(data, w, native, "python", reextract=False)
        b = calculate(data, w, native, "native", reextract=False)
        original = comparison(a["result"], w.get("expected", a["result"]))
        row = dict(
            identity=w["identity"],
            arm=w["arm"],
            seed=w["seed"],
            slot=w["slot"],
            natural=w["natural"],
            **{"class": w["class"]},
            **comparison(a["result"], b["result"]),
            original_passed=original["passed"],
            native_calls=b["native_calls"],
            alpha=b["result"]["alpha"],
            reset=b["result"]["reset"],
            rejected=b["result"]["rejected"],
        )
        restored = native.RustNumericalUpdate8027(json.dumps(b["head"]))
        row["restart_error"] = float(
            np.max(np.abs(np.asarray(restored.effective()) - a["result"]["parameters"]))
        )
        row["restart_equal"] = row["restart_error"] <= CONFIG["tolerance"]
        row["passed"] = row["passed"] and row["restart_equal"] and row["original_passed"]
        rows.append(row)
        if (i + 1) % 100 == 0:
            progress("parity_batch", i + 1, len(data["cases"]) - i - 1)
    atomic_json(raw / "parity_rows.json", dict(rows=rows))
    progress("parity_after", len(rows))
    return dict(parity_rows=rows, parity_passed=bool(rows) and all(r["passed"] for r in rows))


def transaction(data: Json, w: Json, native: Any, arm: str, file: Path) -> Json:
    """Charge rejected work and complete storage before a useful decision finishes."""
    began = time.perf_counter_ns()
    outcome = calculate(data, w, native, arm)
    components = outcome["components"]
    started = time.perf_counter_ns()
    payload = dict(
        head=outcome["head"],
        candidate_delta=outcome["delta"],
        acceptance=outcome["result"],
        replay_pool=w["released"],
        guards=w["guards"],
        identity=w["identity"],
    )
    encoded = json.dumps(payload, sort_keys=True).encode()
    components["serialization_ns"] = time.perf_counter_ns() - started
    file.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    db = sqlite3.connect(file.with_suffix(".sqlite"))
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE events(payload BLOB)")
    with db:
        db.execute("INSERT INTO events VALUES(?)", (encoded,))
    db.close()
    with file.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(file.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    components["storage_fsync_ns"] = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    restored = json.loads(file.read_bytes())
    db = sqlite3.connect(file.with_suffix(".sqlite"))
    journal = db.execute("SELECT payload FROM events").fetchone()[0]
    db.close()
    if journal != encoded:
        raise ValueError("journal_restart_drift")
    restored_state = (
        native.RustNumericalUpdate8027(json.dumps(restored["head"])).effective()
        if arm == "native"
        else prior.old.coefficients(restored["head"]).tolist()
    )
    restart_equal = bool(
        np.max(np.abs(np.asarray(restored_state) - outcome["result"]["parameters"]))
        <= CONFIG["tolerance"]
    )
    components["reload_ns"] = time.perf_counter_ns() - started
    elapsed = time.perf_counter_ns() - began
    components["orchestration_ns"] = elapsed - sum(components.values())
    return dict(
        identity=w["identity"],
        input_hash=canonical_hash(w),
        condition=w["arm"],
        transaction_class=w["class"],
        natural=w["natural"],
        arm=arm,
        transaction_ns=elapsed,
        components=components,
        native_kernel_ns=outcome["native_kernel_ns"],
        native_calls=outcome["native_calls"],
        effective=restored_state,
        alpha=outcome["result"]["alpha"],
        reset=outcome["result"]["reset"],
        rejected=outcome["result"]["rejected"],
        restart_equal=restart_equal,
        guard_count=outcome["guard_count"],
        update_size=outcome["update_size"],
        parameter_count=110,
        stored_bytes=len(encoded) + file.with_suffix(".sqlite").stat().st_size,
        replay_memory_bytes=len(json.dumps(w["released"]).encode()),
        checkpoint=reference(file),
        journal=reference(file.with_suffix(".sqlite")),
        numerator=1,
        denominator=1,
    )


def summarize(rows: list[Json], config: Json) -> Json:
    """Bootstrap paired complete costs; repeated timings do not create new sources."""
    speeds, components, rejected = [], [], []
    groups = sorted({(r["condition"], r["transaction_class"], r["natural"]) for r in rows})
    for condition, klass, natural in groups:
        parts = [
            sorted(
                [
                    r
                    for r in rows
                    if r["condition"] == condition
                    and r["transaction_class"] == klass
                    and r["natural"] == natural
                    and r["arm"] == arm
                    and not r["excluded"]
                ],
                key=lambda r: r["repetition"],
            )
            for arm in ("python", "native")
        ]
        a, b = (np.asarray([r["transaction_ns"] for r in p], dtype=float) for p in parts)
        draw = np.random.default_rng(config["seed"]).integers(0, len(a), (config["draws"], len(a)))
        ratios = a[draw].sum(axis=1) / b[draw].sum(axis=1)
        speeds.append(
            dict(
                condition=condition,
                transaction_class=klass,
                natural=natural,
                speedup=float(a.sum() / b.sum()),
                lower_95=float(np.quantile(ratios, 0.025)),
                upper_95=float(np.quantile(ratios, 0.975)),
                paired_repetitions=len(a),
                numerator=int(a.sum()),
                denominator=int(b.sum()),
                independent_streams=int(natural),
            )
        )
        for arm, part in zip(("python", "native"), parts, strict=True):
            duration = sum(r["transaction_ns"] for r in part) / 1e9
            useful = sum(not r["reset"] for r in part)
            row = dict(
                condition=condition,
                transaction_class=klass,
                natural=natural,
                arm=arm,
                numerator=useful,
                denominator=len(part),
                duration_s=duration,
                useful_decisions_per_second=useful / duration,
                completed_transactions_per_second=len(part) / duration,
                timing_ns={
                    k: prior.old.percentiles([r["components"][k] for r in part])
                    for k in part[0]["components"]
                },
                guard_sizes=sorted({r["guard_count"] for r in part}),
                parameter_count=110,
                stored_bytes=sum(r["stored_bytes"] for r in part),
                gradient_ns=sum(
                    r["components"]["gradient_arithmetic_ns"]
                    + r["components"]["gradient_construction_ns"]
                    for r in part
                ),
                guard_ns=sum(r["components"]["guard_scans_ns"] for r in part),
                ffi_ns=sum(r["components"]["ffi_ns"] for r in part),
            )
            components.append(row)
            if klass == "rejected":
                rejected.append(row)
    return dict(
        rows=rows,
        timing_components=components,
        rejected_work_cost=rejected,
        complete_numerical_transaction_speedup=speeds,
    )


def measure(data: Json, native: Any, raw: Path) -> Json:
    """Randomize paired paths and explicitly separate absent natural reset work."""
    cases = list(data["cases"])
    natural_classes = {w["class"] for w in cases}
    if "reset" not in natural_classes:
        cases += [w for w in fixtures(data["head"]) if w["class"] == "reset"]
    groups = sorted({(w["arm"], w["class"], w["natural"]) for w in cases})
    rows = []
    rng = np.random.default_rng(CONFIG["seed"])
    deadline = time.monotonic() + CONFIG["budget_s"]
    total = len(groups) * (CONFIG["repetitions"] + CONFIG["warmups"]) * 2
    progress("benchmark_before", 0, total)
    for condition, klass, natural in groups:
        pool = [
            w for w in cases if (w["arm"], w["class"], w["natural"]) == (condition, klass, natural)
        ]
        for repetition in range(-CONFIG["warmups"], CONFIG["repetitions"]):
            if time.monotonic() > deadline:
                raise TimeoutError("numerical_benchmark_budget")
            w = pool[int(rng.integers(0, len(pool)))]
            order = rng.permutation(["python", "native"]).tolist()
            pair = []
            for arm in order:
                row = transaction(
                    data,
                    w,
                    native,
                    arm,
                    raw / "transactions" / f"{condition}-{klass}-{natural}-{repetition}-{arm}.json",
                )
                row.update(
                    repetition=repetition,
                    pair_order=order,
                    excluded=repetition < 0,
                    exclusion_reason="warmup" if repetition < 0 else None,
                    seed=w["seed"],
                    slot=w["slot"],
                )
                pair.append(row)
                rows.append(row)
                atomic_json(raw / "timing_batches" / f"{len(rows):04d}.json", dict(row=row))
            if pair[0]["input_hash"] != pair[1]["input_hash"] or any(
                not r["restart_equal"] for r in pair
            ):
                raise ValueError("pair_restart_contract")
            if (
                np.max(np.abs(np.asarray(pair[0]["effective"]) - pair[1]["effective"]))
                > CONFIG["tolerance"]
            ):
                raise ValueError("pair_coefficient_contract")
            progress("benchmark_pair_after", len(rows), total - len(rows))
    atomic_json(raw / "transaction_rows.json", dict(rows=rows))
    result = summarize(rows, CONFIG)
    result["transaction_class_rows"] = [
        dict(
            condition=arm,
            transaction_class=klass,
            numerator=sum(
                w["natural"] and w["arm"] == arm and w["class"] == klass for w in data["cases"]
            ),
            denominator=sum(w["natural"] and w["arm"] == arm for w in data["cases"]),
            status="present"
            if any(w["natural"] and w["arm"] == arm and w["class"] == klass for w in data["cases"])
            else "absent",
            synthetic_natural_credit=0,
        )
        for arm in ("unconstrained", "feedback_constrained")
        for klass in ("accepted", "rejected", "reset")
    ]
    progress("benchmark_after", len(rows))
    return result


def scaling(head: Json, native: Any) -> Json:
    """A bounded synthetic sweep exposes update size and real binding failures."""
    rows = []
    for size in CONFIG["scaling_sizes"]:
        x = native.RustNumericalUpdate8027(json.dumps(head)).design([[0.5] * 9] * size)
        for path in ("python", "native"):
            started = time.perf_counter_ns()
            h = copy.deepcopy(head)
            if path == "native":
                state = native.RustNumericalUpdate8027(json.dumps(h))
                state.update_batch(x, [i % 2 for i in range(size)])
            else:
                prior.old.python_batch(h, np.asarray(x), [i % 2 for i in range(size)])
            rows.append(
                dict(
                    arm=path,
                    update_size=size,
                    parameters=110,
                    elapsed_ns=time.perf_counter_ns() - started,
                    numerator=size,
                    denominator=size,
                    natural_credit=0,
                )
            )
    state = native.RustNumericalUpdate8027(json.dumps(head))
    before = state.state_json()
    failure = None
    try:
        state.update_batch([[float("nan")] * 110], [1])
    except ValueError as error:
        failure = str(error)
    return dict(
        rows=rows,
        natural_credit=0,
        failed_transaction=dict(error=failure, state_unchanged=before == state.state_json()),
        passed=failure is not None and before == state.state_json(),
    )
