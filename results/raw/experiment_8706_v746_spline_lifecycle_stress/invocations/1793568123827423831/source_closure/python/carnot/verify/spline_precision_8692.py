"""REQ-VERIFY-8692: CPU integers expose rounding loss without claiming hardware speed.

The host keeps scaling, the logit offset and sigmoid. Local cubic recursion,
residual accumulation and Brier coefficient writes use signed fixed point.
"""

from __future__ import annotations

from copy import deepcopy
import math
import random
import struct
import time
from typing import Any

import numpy as np
from carnot.verify import objective_checkpoint_8461 as original

Json = dict[str, Any]
SEED = 7458692
FORMATS = [(16, 12), (24, 18), (32, 24)]
__all__ = [
    "Fixed",
    "FORMATS",
    "SEED",
    "original",
    "encode",
    "local_basis",
    "effective",
    "decision",
    "probability",
    "step",
    "sequences",
    "vectors",
    "serialize",
]


class Fixed:
    """Check widened intermediates before rounding so Python cannot hide overflow."""

    def __init__(self, bits: int, fractional: int):
        if (bits, fractional) not in FORMATS:
            raise ValueError("format")
        self.bits, self.fractional, self.unit = bits, fractional, 1 << fractional
        self.low, self.high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        self.counts = dict(saturation=0, overflow=0, decay_underflow=0, renormalization=0)
        self.operations = dict(multiply=0, divide=0, saturation_checks=0, accumulator_checks=0)
        self.maximum = 0

    def checked(self, n: int) -> int:
        """A signed64 check is an observed bound, even though Python integers are wider."""
        self.operations["accumulator_checks"] += 1
        self.maximum = max(self.maximum, abs(n))
        if not -(1 << 63) <= n < (1 << 63):
            self.counts["overflow"] += 1
            raise OverflowError("signed64")
        return n

    def ratio(self, n: int, d: int) -> int:
        """Integer quotient/remainder makes negative ties independent of float rounding."""
        self.operations["divide"] += 1
        if d <= 0:
            raise ValueError("positive_denominator")
        n = self.checked(n)
        q, remainder = divmod(n, d)
        return q + int(2 * remainder > d or (2 * remainder == d and q % 2 != 0))

    def sat(self, n: int) -> int:
        """Clamp each stored value explicitly; no masked or wrapping integer conversion."""
        self.operations["saturation_checks"] += 1
        self.counts["saturation"] += int(n < self.low or n > self.high)
        return min(self.high, max(self.low, n))

    def q(self, x: float) -> int:
        """Host ingress uses the same ties-to-even rule for finite binary64 operands."""
        if not math.isfinite(x):
            raise ValueError("finite_operand")
        return self.sat(round(x * self.unit))

    def mul(self, a: int, b: int) -> int:
        """Round the checked double-width product only once before storing it."""
        return self.sat(self.ratio(self.product(a, b), self.unit))

    def product(self, a: int, b: int) -> int:
        """Count basis and residual products as well as stored fixed-point products."""
        self.operations["multiply"] += 1
        return self.checked(a * b)

    def div(self, a: int, b: int) -> int:
        """Widen before division to retain the requested fractional precision."""
        return self.sat(self.ratio(self.product(a, self.unit), b))


def encode(head: Json, f: Fixed) -> Json:
    """Copy the original state so controls cannot mutate a fitted checkpoint."""
    values = [*head["parameters"], head["decay_scale"], head["temperature"]]
    if (
        len(head["parameters"]) != 109
        or head["temperature"] <= 0
        or head["decay_scale"] <= 0
        or not all(math.isfinite(v) for v in values)
    ):
        raise ValueError("head_contract")
    return dict(
        parameters=[f.q(v) for v in head["parameters"]],
        decay_scale=f.q(head["decay_scale"]),
        temperature=f.q(head["temperature"]),
        scaler=deepcopy(head["scaler"]),
    )


def local_basis(f: Fixed, scaled: list[float]) -> tuple[list[int], list[int]]:
    """Cox recursion visits only the four local cubic coordinates, including clamped ends."""
    knots = [f.q(v) for v in original.sparse.KNOTS]
    ids, weights = [108], [f.unit]
    for j, value in enumerate(scaled):
        x = f.q(value)
        span = min(11, max(3, next((i - 1 for i in range(4, 16) if knots[i] > x), 11)))
        previous = {span: f.unit}
        for degree in range(1, 4):
            current = {}
            for i in range(span - degree, span + 1):
                n = 0
                for operand, width, prior in [
                    (x - knots[i], knots[i + degree] - knots[i], previous.get(i, 0)),
                    (
                        knots[i + degree + 1] - x,
                        knots[i + degree + 1] - knots[i + 1],
                        previous.get(i + 1, 0),
                    ),
                ]:
                    if width and prior:
                        n += f.ratio(f.product(operand, prior), width)
                current[i] = f.sat(n)
            previous = current
        for i, weight in previous.items():
            if weight:
                ids.append(j * 12 + i)
                weights.append(weight)
    return ids, weights


def effective(state: Json, f: Fixed) -> list[float]:
    """Compare effective coefficients separately from the stored decay multiplier."""
    return [v * state["decay_scale"] / f.unit**2 for v in state["parameters"]]


def decision(p: float) -> str:
    """Retain the shipped typed-decision thresholds exactly."""
    return "accept" if p <= 0.2 else "reject" if p >= 0.8 else "escalate"


def serialize(state: Json, f: Fixed) -> bytes:
    """Pack actual integers and host scaler; an external schema names the format."""
    numbers = [*state["parameters"], state["decay_scale"], state["temperature"]]
    return b"".join(v.to_bytes(f.bits // 8, "little", signed=True) for v in numbers) + struct.pack(
        "<18d", *state["scaler"]["minimum"], *state["scaler"]["maximum"]
    )


def probability(state: Json, raw: list[float], f: Fixed) -> tuple[float, Json]:
    """Separate host normalization from integer work so their costs cannot overlap."""
    if len(raw) != 9 or not all(math.isfinite(v) for v in raw):
        raise ValueError("input_contract")
    begin = time.perf_counter_ns()
    low, high = state["scaler"]["minimum"], state["scaler"]["maximum"]
    scaled = [
        min(1.0, max(0.0, (x - a) / (b - a if b > a else 1.0)))
        for x, a, b in zip(raw, low, high, strict=True)
    ]
    ids, weights = local_basis(f, scaled)
    basis_end = time.perf_counter_ns()
    total = 0
    for i, weight in zip(ids, weights, strict=True):
        total = f.checked(total + f.product(state["parameters"][i], weight))
    residual = f.mul(f.sat(f.ratio(total, f.unit)), state["decay_scale"]) / f.unit
    residual_end = time.perf_counter_ns()
    q = min(1 - 1e-4, max(1e-4, raw[0]))
    z = (math.log(q / (1 - q)) + residual) / (state["temperature"] / f.unit)
    p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    end = time.perf_counter_ns()
    return p, dict(
        ids=ids,
        weights=weights,
        residual=residual,
        spans=[
            dict(operation=k, start_ns=a, end_ns=b, duration_ns=b - a)
            for k, a, b in [
                ("host_scale_integer_basis", begin, basis_end),
                ("integer_residual", basis_end, residual_end),
                ("host_logit_sigmoid", residual_end, end),
            ]
        ],
    )


def step(state: Json, raw: list[float], label: int, f: Fixed) -> tuple[Json, Json]:
    """Apply the original Brier derivative with sparse data writes and global bias decay."""
    if type(label) is not int or label not in (0, 1):
        raise ValueError("binary_label")
    p, receipt = probability(state, raw, f)
    began = time.perf_counter_ns()
    result = deepcopy(state)
    copy_end = time.perf_counter_ns()
    scale = f.mul(state["decay_scale"], f.q(1 - 0.002 * 0.01))
    f.counts["decay_underflow"] += int(scale == state["decay_scale"])
    renormalization_writes = 0
    if scale < max(2, f.unit // 256):
        f.counts["renormalization"] += 1
        f.counts["decay_underflow"] += int(scale == 0)
        result["parameters"] = [f.mul(v, state["decay_scale"]) for v in state["parameters"]]
        scale = f.q(1 - 0.002 * 0.01)
        renormalization_writes = 109
    quantized_p = f.q(p)
    derivative = f.div(
        f.mul(f.mul(2 * (quantized_p - label * f.unit), quantized_p), f.unit - quantized_p),
        state["temperature"],
    )
    for i, weight in zip(receipt["ids"], receipt["weights"], strict=True):
        delta = f.div(f.mul(f.q(0.01), f.mul(derivative, weight)), scale)
        result["parameters"][i] = f.sat(result["parameters"][i] - delta)
    result["decay_scale"] = scale
    ended = time.perf_counter_ns()
    receipt.update(
        local_writes=len(receipt["ids"]),
        global_decay_writes=1,
        renormalization_writes=renormalization_writes,
        host_snapshot_coefficient_copies=109,
        probability=p,
    )
    receipt["spans"].append(
        dict(
            operation="host_state_copy",
            start_ns=began,
            end_ns=copy_end,
            duration_ns=copy_end - began,
        )
    )
    receipt["spans"].append(
        dict(
            operation="integer_brier_sparse_write",
            start_ns=copy_end,
            end_ns=ended,
            duration_ns=ended - copy_end,
        )
    )
    return result, receipt


def sequences() -> dict[str, list[int]]:
    """Fixed label controls measure arithmetic trajectories rather than learning benefit."""
    rng = random.Random(SEED)
    return dict(
        constant_zero=[0] * 256,
        constant_one=[1] * 256,
        alternating=[i % 2 for i in range(256)],
        seeded_bernoulli=[rng.randrange(2) for _ in range(256)],
    )


def vectors(heads: list[Json]) -> Json:
    """Select geometry before labels; deterministic bisection targets typed thresholds."""
    rng = random.Random(SEED)
    h = heads[0]
    low, high = h["scaler"]["minimum"], h["scaler"]["maximum"]

    def unscale(xs: list[float]) -> list[float]:
        """Keep raw clip probes even when a feature has a constant fit range."""
        return [a + x * (b - a if b > a else 1.0) for x, a, b in zip(xs, low, high, strict=True)]

    rows = [
        dict(kind="interior", x=unscale([rng.uniform(0.001, 0.999) for _ in range(9)]))
        for _ in range(128)
    ]
    boundaries = [-1e-6, 0.0, 1e-9, *[i / 9 for i in range(1, 9)], 1 - 1e-9, 1.0, 1 + 1e-6]
    for i in range(64):
        x = [0.5] * 9
        x[i % 9] = boundaries[(i // 9 + i) % len(boundaries)]
        rows.append(dict(kind="boundary", x=unscale(x)))
    searches = []
    for i in range(64):
        owner = heads[i % len(heads)]
        x = unscale([rng.random() for _ in range(9)])
        target = (0.2 if i % 2 == 0 else 0.8) + [-0.0005, 0.0, 0.0005][(i // 2) % 3]

        def at(q: float) -> float:
            """Only the original numerical head participates in threshold search."""
            return float(original.sparse.predict(owner, np.array([[q, *x[1:]]]))[0])

        a, b = 1e-4, 1 - 1e-4
        unreachable = not at(a) <= target <= at(b)
        for _ in range(48):
            mid = (a + b) / 2
            if at(mid) < target:
                a = mid
            else:
                b = mid
        x[0] = (a + b) / 2
        searches.append(
            dict(
                vector_id=192 + i,
                head_index=i % len(heads),
                target=target,
                achieved=at(x[0]),
                unreachable=unreachable,
            )
        )
        rows.append(dict(kind="threshold", x=x))
    return dict(
        seed=SEED,
        vectors=[dict(v, vector_id=i) for i, v in enumerate(rows)],
        threshold_searches=searches,
        label_accesses=0,
        sequences=sequences(),
    )
