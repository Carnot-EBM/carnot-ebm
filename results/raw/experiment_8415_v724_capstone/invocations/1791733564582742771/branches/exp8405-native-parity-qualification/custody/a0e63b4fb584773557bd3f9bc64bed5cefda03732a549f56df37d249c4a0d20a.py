"""REQ-VERIFY-8352: isolate local table error from frozen global arithmetic."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import time
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.special import expit

from carnot.verify import local_update_isolation_8306 as kernel

Json = dict[str, Any]
Array: TypeAlias = NDArray[Any]
SEED = 7208352
CONFIGS = list(product([65, 257, 1025], ["float64", "int16"], ["nearest", "linear"]))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work counts so a supervisor can detect stalled phases."""
    print(f"[exp8352] phase={phase} completed={completed} pending={pending}", flush=True)


def basis(values: Array) -> Array:
    """Reuse the historical kernel; SciPy remains an independent reference."""
    return np.asarray([kernel.local_basis(float(x)) for x in values], dtype=np.float64)


def direct(head: Json, x: Array) -> Array:
    """Keep all global terms unchanged to measure only local approximation error."""
    c: Array = np.asarray(head["coefficients"], dtype=np.float64)
    z = c[0] * x[:, 0] + c[1]
    for j in range(4):
        z += basis(x[:, j + 1]) @ c[2 + 8 * j : 10 + 8 * j]
    return np.asarray(expit(z / head["temperature"]), dtype=np.float64)


def reference(head: Json, x: Array) -> Array:
    """Independent SciPy spline recursion detects basis ordering and endpoint faults."""
    c: Array = np.asarray(head["coefficients"], dtype=np.float64)
    z = c[0] * x[:, 0] + c[1]
    for j in range(4):
        z += BSpline(kernel.KNOTS, c[2 + 8 * j : 10 + 8 * j], 3)(x[:, j + 1])
    return np.asarray(expit(z / head["temperature"]), dtype=np.float64)


def panel(head: Json) -> tuple[Array, list[str], bool]:
    """Label-free boundaries expose flips that uniform random points can miss."""
    rng = np.random.default_rng(SEED)
    local = rng.random((4096, 4))
    rows = np.column_stack((rng.normal(size=4096), local)).tolist()
    kinds = ["random"] * 4096
    for feature, knot, offset in product(range(4), np.unique(kernel.KNOTS), [-1e-8, 0.0, 1e-8]):
        row = [0.0, 0.5, 0.5, 0.5, 0.5]
        row[feature + 1] = float(np.clip(knot + offset, 0, 1))
        rows.append(row)
        kinds.append("knot")
    c = head["coefficients"]
    available = c[0] != 0
    if available:
        local_logit = float(
            np.log(
                reference(head, np.array([[0.0, 0.5, 0.5, 0.5, 0.5]]))[0]
                / (1 - reference(head, np.array([[0.0, 0.5, 0.5, 0.5, 0.5]]))[0])
            )
        )
        for threshold, offset in product([0.25, 0.75], [-1e-6, 0.0, 1e-6]):
            holistic = (
                head["temperature"] * (np.log(threshold / (1 - threshold)) - local_logit)
            ) / c[0]
            rows.append([float(holistic + offset), 0.5, 0.5, 0.5, 0.5])
            kinds.append("action_boundary")
    return np.asarray(rows, dtype=np.float64), kinds, available


def encode(values: Array, storage: str) -> tuple[Array, int]:
    """Count overflow before clipping; ties to even match the declared fixed format."""
    if storage == "float64":
        return values.astype("<f8"), 0
    rounded = np.rint(values * 2048)
    saturation = int(np.sum((rounded < -32768) | (rounded > 32767)))
    return np.clip(rounded, -32768, 32767).astype("<i2"), saturation


def table(head: Json, size: int, storage: str) -> tuple[Array, int]:
    """Store edge sums, so table bytes include each feature's full local contribution."""
    b = basis(np.linspace(0, 1, size))
    c = np.asarray(head["coefficients"])[2:].reshape(4, 8)
    return encode(np.sum(c[:, None, :] * b[None, :, :], axis=2), storage)


def lookup(head: Json, x: Array, values: Array, storage: str, interpolation: str) -> Array:
    """Decode only local values; interpolation and sigmoid remain float64 operations."""
    position = x[:, 1:] * (values.shape[1] - 1)
    if interpolation == "nearest":
        local = values[np.arange(4), np.rint(position).astype(int)].astype(np.float64)
    else:
        lo = np.floor(position).astype(int)
        hi = np.minimum(lo + 1, values.shape[1] - 1)
        weight = position - lo
        local = values[np.arange(4), lo] * (1 - weight) + values[np.arange(4), hi] * weight
    local /= 2048 if storage == "int16" else 1
    c = head["coefficients"]
    return np.asarray(
        expit((c[0] * x[:, 0] + c[1] + local.sum(axis=1)) / head["temperature"]), dtype=np.float64
    )


def summarize(
    ref: Array,
    actual: Array,
    size: int,
    storage: str,
    interpolation: str,
    order: int,
    saturation: int,
) -> Json:
    """Far-margin flips disqualify a candidate; boundary flips remain visible."""
    error = float(np.max(np.abs(ref - actual)))
    flips = np.asarray([kernel.action(float(p)) for p in ref]) != np.asarray(
        [kernel.action(float(p)) for p in actual]
    )
    far = np.minimum(np.abs(ref - 0.25), np.abs(ref - 0.75)) >= 0.002
    return dict(
        configuration_id=f"{size}-{storage}-{interpolation}",
        grid_points=size,
        storage=storage,
        interpolation=interpolation,
        order=order,
        probability_error_max=error,
        action_flip_count=int(flips.sum()),
        boundary_flip_count=int((flips & ~far).sum()),
        far_flip_count=int((flips & far).sum()),
        saturation_count=saturation,
        table_bytes=4 * size * (8 if storage == "float64" else 2),
        candidate_passed=error <= 0.001 and not bool((flips & far).any()),
    )


def select(rows: list[Json]) -> Json | None:
    """Choose by frozen engineering criteria without making a utility claim."""
    passed = [r for r in rows if r["candidate_passed"]]
    return (
        min(passed, key=lambda r: (r["table_bytes"], r["probability_error_max"], r["order"]))
        if passed
        else None
    )


def events() -> Array:
    """Alternating constructed targets exercise refresh without opening human labels."""
    rng = np.random.default_rng(SEED)
    local = rng.random((64, 4))
    return np.column_stack((rng.normal(size=64), local))


def update(head: Json, x: Array, target: int) -> Json:
    """Apply the frozen temperature-scaled gradient and local-only norm cap."""
    result = deepcopy(head)
    phi = basis(x[1:]).ravel()
    gradient = (float(direct(head, x[None])[0]) - target) / head["temperature"] * phi
    gradient *= min(1.0, 1.0 / max(float(np.linalg.norm(gradient)), 1e-300))
    result["coefficients"][2:] = np.clip(
        np.asarray(head["coefficients"])[2:] - 0.01 * gradient, -4, 4
    ).tolist()
    return result


def refresh(
    old: Json, new: Json, values: Array, storage: str, *, miss: bool = False
) -> tuple[Array, list[list[int]]]:
    """Recompute entire affected edge sums so scoped refresh matches a full rebuild."""
    b = basis(np.linspace(0, 1, values.shape[1]))
    changed = (np.asarray(old["coefficients"])[2:] != np.asarray(new["coefficients"])[2:]).reshape(
        4, 8
    )
    result = values.copy()
    entries = []
    omitted = False
    for feature in range(4):
        ids = np.flatnonzero(np.any(b[:, changed[feature]] != 0, axis=1))
        entries.append(ids.tolist())
        computed, _ = encode(
            np.sum(
                b[ids] * np.asarray(new["coefficients"])[2 + 8 * feature : 10 + 8 * feature], axis=1
            ),
            storage,
        )
        if miss and not omitted and len(ids):
            different = np.flatnonzero(computed != result[feature, ids])
            if len(different):
                omit = int(different[0])
                ids, computed = np.delete(ids, omit), np.delete(computed, omit)
                omitted = True
        result[feature, ids] = computed
    return result, entries


def evaluation_timings(
    head: Json, x: Array, values: Array, storage: str, interpolation: str
) -> Array:
    """Pair single-vector calls; keep the warmup distinct from five measured repeats."""
    samples: Array = np.empty((len(x), 6, 2), dtype=np.int64)
    progress("before_benchmark_evaluation")
    for repetition in range(6):
        for index, row in enumerate(x):
            for method in [0, 1] if (repetition + index) % 2 == 0 else [1, 0]:
                began = time.perf_counter_ns()
                if method == 0:
                    direct(head, row[None])
                else:
                    lookup(head, row[None], values, storage, interpolation)
                samples[index, repetition, method] = time.perf_counter_ns() - began
            if (index + 1) % 512 == 0:
                progress(
                    "evaluation",
                    repetition * len(x) + index + 1,
                    (6 - repetition) * len(x) - index - 1,
                )
    progress("after_benchmark_evaluation", len(x) * 6, 0)
    return samples


def refresh_measurement(original: Json) -> tuple[list[Json], list[Json], bool]:
    """Each repetition restarts the coefficients; timings never add independent data."""
    rows: list[Json] = []
    finals: list[Json] = []
    passed = True
    inputs = events()
    for size, storage in product([65, 257, 1025], ["float64", "int16"]):
        progress("before_benchmark_refresh_" + str(size) + storage)
        trajectory = []
        for repetition in range(6):
            head = deepcopy(original)
            values, _ = table(head, size, storage)
            for index, x in enumerate(inputs):
                began = time.perf_counter_ns()
                new = update(head, x, index % 2)
                update_ns = time.perf_counter_ns() - began
                timed: Json = dict(
                    repetition=repetition - 1, warmup=repetition == 0, local_update_ns=update_ns
                )
                full = scoped = values
                entries: list[list[int]] = []
                for method in (
                    ["full", "scoped"] if (index + repetition) % 2 == 0 else ["scoped", "full"]
                ):
                    began = time.perf_counter_ns()
                    if method == "full":
                        full, _ = table(new, size, storage)
                    else:
                        scoped, entries = refresh(head, new, values, storage)
                    timed[method + "_refresh_ns"] = time.perf_counter_ns() - began
                equal = full.tobytes() == scoped.tobytes()
                passed = passed and equal
                for name, data in [("full", full), ("scoped", scoped)]:
                    began = time.perf_counter_ns()
                    data.tobytes()
                    timed[name + "_serialization_ns"] = time.perf_counter_ns() - began
                if repetition == 0:
                    trajectory.append(
                        dict(
                            grid_points=size,
                            storage=storage,
                            update=index,
                            x=x.tolist(),
                            target=index % 2,
                            coefficients=new["coefficients"],
                            affected_entries=entries,
                            identical_bytes=equal,
                            timings=[timed],
                        )
                    )
                else:
                    trajectory[index]["identical_bytes"] &= equal
                    trajectory[index]["timings"].append(timed)
                head, values = new, scoped
            progress("refresh", (repetition + 1) * 64, (5 - repetition) * 64)
        rows.extend(trajectory)
        finals.append(
            dict(grid_points=size, storage=storage, head=head, table_hex=values.tobytes().hex())
        )
        progress("after_benchmark_refresh_" + str(size) + storage, 384, 0)
    old_values, _ = table(original, 65, "float64")
    new = update(original, inputs[0], 0)
    broken, _ = refresh(original, new, old_values, "float64", miss=True)
    missed_rejected = broken.tobytes() != table(new, 65, "float64")[0].tobytes()
    return rows, finals, passed and missed_rejected
