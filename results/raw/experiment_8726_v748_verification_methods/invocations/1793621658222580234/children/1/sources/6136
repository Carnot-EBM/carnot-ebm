"""REQ-VERIFY-8333: independent dense arithmetic checks primitive local writes."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from carnot.verify import local_update_isolation_8306 as k

Json = dict[str, Any]


def dense(x: list[float]) -> list[float]:
    """Use a bottom-up recurrence so the reference does not call the tested basis."""
    result = [x[0], 1.0]
    for value in x[1:]:
        row = [float(a <= value < b) for a, b in zip(k.KNOTS[:-1], k.KNOTS[1:], strict=True)]
        for degree in range(1, 4):
            next_row = []
            for i in range(len(row) - 1):
                left = k.KNOTS[i + degree] - k.KNOTS[i]
                right = k.KNOTS[i + degree + 1] - k.KNOTS[i + 1]
                next_row.append(
                    ((value - k.KNOTS[i]) * row[i] / left if left else 0.0)
                    + ((k.KNOTS[i + degree + 1] - value) * row[i + 1] / right if right else 0.0)
                )
            row = next_row
        result.extend([float(i == 7) for i in range(8)] if value == 1 else row)
    return result


def probability(c: list[float], x: list[float], temperature: float = 1.0) -> float:
    """The frozen energy is a sigmoid; independent arithmetic needs no energy API."""
    eta = sum(a * b for a, b in zip(c, dense(x), strict=True)) / temperature
    return 1.0 / (1.0 + math.exp(-eta))


def step(c: list[float], event: Json, consumed: list[str]) -> list[float]:
    """Replay dense SGD independently, including clipping, duplicates and stale IDs."""
    if event["id"] in consumed or event.get("stale"):
        return list(c)
    consumed.append(event["id"])
    residual = probability(c, event["x"]) - event["y"]
    gradient = [0.0, residual if event.get("global_update") else 0.0]
    gradient.extend(residual * v for v in dense(event["x"])[2:])
    norm = math.sqrt(sum(v * v for v in gradient))
    scale = min(1.0, 1.0 / norm) if norm else 1.0
    return [
        min(4.0, max(-4.0, a - event.get("rate", 0.01) * b * scale))
        for a, b in zip(c, gradient, strict=True)
    ]


def durable(state: Json) -> Json:
    """Compare reverse dependencies and cache validity as well as issued predictions."""
    return k.semantic(state) | {
        key: state[key] for key in ["cache", "index", "version", "index_version"]
    }


def audit(work: Json, protocol: Json, *, claimed_zero: bool = True) -> Json:
    """Inspect every coefficient array and issued probability instead of an aggregate."""
    coefficient_error, probability_error, parity = 0.0, 0.0, 0.0
    release_count = 0
    valid = True
    for stored, trajectory in zip(work["states"], protocol["trajectories"], strict=True):
        for arm in ["full", "indexed"]:
            state = stored["arms"][arm]
            c, consumed = list(trajectory["coefficients"]), []
            timeline = [list(c)]
            for release, event in zip(state["releases"], trajectory["events"], strict=True):
                c = step(c, event, consumed)
                timeline.append(list(c))
                coefficient_error = max(
                    coefficient_error,
                    max(abs(a - b) for a, b in zip(c, release["coefficients"], strict=True)),
                )
                release_count += 1
            valid &= state["applied"] == consumed and state["stale_cache_count"] == 0
            for issued in state["issues"]:
                x = trajectory["cache_x"][int(issued["cache_id"])]
                causal = timeline[min(64, max(0, issued["slot"] - 8))]
                coefficient_error = max(
                    coefficient_error,
                    max(abs(a - b) for a, b in zip(causal, issued["coefficients"], strict=True)),
                )
                p = probability(causal, x)
                probability_error = max(probability_error, abs(p - issued["p"]))
                valid &= k.action(p) == issued["action"]
        left, right = stored["arms"]["full"], stored["arms"]["indexed"]
        parity = max(
            parity,
            max(abs(a["p"] - b["p"]) for a, b in zip(left["issues"], right["issues"], strict=True)),
        )
        valid &= durable(left) == durable(right)
    passed = bool(
        valid and release_count and max(coefficient_error, probability_error, parity) <= 1e-10
    )
    return dict(
        passed=passed,
        recomputed=passed and (not claimed_zero or parity == 0.0),
        coefficient_error_max=coefficient_error,
        probability_error_max=probability_error,
        dense_sparse_probability_error_max=parity,
        release_count=release_count,
    )


def controls(work: Json, protocol: Json) -> Json:
    """A false zero must be rejected even when both reported arms share corruption."""
    altered = deepcopy(work)
    altered["states"][0]["arms"]["indexed"]["releases"][0]["coefficients"][2] += 0.125
    corrupt = audit(altered, protocol)
    fabricated = deepcopy(altered)
    fabricated["states"][0]["arms"]["full"]["releases"][0]["coefficients"][2] += 0.125
    false_zero = audit(fabricated, protocol)
    return dict(
        recomputed=audit(work, protocol)["recomputed"],
        deliberate_error_rejected=not corrupt["passed"],
        false_zero_rejected=not false_zero["passed"],
        corrupted_coefficient=corrupt,
        fabricated_zero=false_zero,
    )


def derivatives() -> Json:
    """Check endpoint support and the temperature derivative without calling updates."""
    gradient_error, temperature_error, partition_error = 0.0, 0.0, 0.0
    active_error = 0
    c = k.static_fit()
    for value in [0.0, 1e-12, 0.2, 0.4, 0.6, 0.8, 1 - 1e-12, 1.0]:
        x = [0.3, *([value] * 4)]
        d = dense(x)
        partition_error = max(partition_error, abs(sum(d[2:10]) - 1))
        active_error += int(
            [i for i, v in enumerate(d) if v] != [i for i, v in enumerate(k.design(x)) if v]
        )
        for temperature in [0.5, 1.0, 2.0]:
            eta = sum(a * b for a, b in zip(c, d, strict=True))
            loss = lambda weights, t: math.log1p(
                math.exp(sum(a * b for a, b in zip(weights, d, strict=True)) / t)
            )
            for i in range(34):
                high, low = list(c), list(c)
                high[i] += 1e-5
                low[i] -= 1e-5
                finite = (loss(high, temperature) - loss(low, temperature)) / 2e-5
                gradient_error = max(
                    gradient_error,
                    abs(finite - probability(c, x, temperature) * d[i] / temperature),
                )
            finite_t = (loss(c, temperature + 1e-5) - loss(c, temperature - 1e-5)) / 2e-5
            temperature_error = max(
                temperature_error,
                abs(finite_t + probability(c, x, temperature) * eta / temperature**2),
            )
    return dict(
        passed=max(gradient_error, temperature_error, partition_error) < 1e-8 and active_error == 0,
        finite_difference_error_max=gradient_error,
        temperature_derivative_error_max=temperature_error,
        partition_error_max=partition_error,
        active_coordinate_mismatches=active_error,
    )


def feedback_control() -> Json:
    """Out-of-order feedback still consumes each ID once and never rewrites issues."""
    trajectory = k.manifest()["trajectories"][0]
    state = k.initial(trajectory)
    k.issue(state, 0)
    issued = deepcopy(state["issues"])
    c, consumed = list(state["coefficients"]), []
    events = [dict(trajectory["events"][i], id=str(i), stale=False) for i in [9, 2, 9, 1]]
    for event in events:
        c = step(c, event, consumed)
        k.update(state, event, "indexed")
    error = max(abs(a - b) for a, b in zip(c, state["coefficients"], strict=True))
    return dict(
        passed=error <= 1e-10 and state["applied"] == ["9", "2", "1"] and issued == state["issues"],
        events=events,
        consumed_ids=state["applied"],
        coefficient_error_max=error,
        issued_predictions_unchanged=issued == state["issues"],
    )
