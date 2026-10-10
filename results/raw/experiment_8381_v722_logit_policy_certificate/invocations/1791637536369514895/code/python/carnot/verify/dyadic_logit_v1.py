"""REQ-VERIFY-8381: exact rational actions avoid an undocumented sigmoid contract."""

from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import json
import math
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import threshold_guard_8362 as previous

Json = dict[str, Any]
L_HEX = "0x1.193ea7aad030bp+0"
L = F(float.fromhex(L_HEX))
VERSION = "dyadic_logit_v1"
ROUNDING = "exact-rational-global-and-interpolation;binary64-outward-endpoints"
MODEL_SPECS: list[Json] = []
KNOTS = tuple(F(float(v)) for v in previous.old.kernel.KNOTS)
progress = previous.progress


def region(lo: F, hi: F) -> str | None:
    """A whole closed interval must belong to one region; ties always escalate."""
    if hi < -L:
        return "accept"
    if lo > L:
        return "reject"
    if lo >= -L and hi <= L:
        return "escalate"
    return None


def outward(value: F, direction: int) -> float:
    """One adjacent float encloses rational conversion error even at exact ties."""
    return float(np.nextafter(float(value), math.inf * direction))


def de_boor(coefficients: list[F], x: F) -> F:
    """Exact local triangular evaluation supplies the fallback without polynomial fitting."""
    k = 7 if x == 1 else next(i for i in range(3, 8) if KNOTS[i] <= x < KNOTS[i + 1])
    d = coefficients[k - 3 : k + 1]
    for r in range(1, 4):
        for j in range(3, r - 1, -1):
            i = k - 3 + j
            a = (x - KNOTS[i]) / (KNOTS[i + 4 - r] - KNOTS[i])
            d[j] = (1 - a) * d[j - 1] + a * d[j]
    return d[3]


@lru_cache(maxsize=16)
def polynomials(signature: str) -> list[list[list[F]]]:
    """Exact Newton interpolation uses independent basis recursion on each cubic piece."""
    head = json.loads(signature)
    pieces = []
    unique = sorted(set(KNOTS))
    for feature in range(4):
        c = [F(v) for v in head["coefficients"][2 + 8 * feature : 10 + 8 * feature]]
        local = []
        for lo, hi in zip(unique[:-1], unique[1:], strict=True):
            xs = [lo + (hi - lo) * F(i, 4) for i in range(4)]
            ys = [
                sum((c[i] * previous.rational_basis(i, 3, x, list(KNOTS)) for i in range(8)), F(0))
                for x in xs
            ]
            for order in range(1, 4):
                for j in range(3, order - 1, -1):
                    ys[j] = (ys[j] - ys[j - 1]) / (xs[j] - xs[j - order])
            p = [ys[3]]
            for j in range(2, -1, -1):
                q = [F(0)] * (len(p) + 1)
                for i, v in enumerate(p):
                    q[i] -= xs[j] * v
                    q[i + 1] += v
                q[0] += ys[j]
                p = q
            local.append(p)
        pieces.append(local)
    return pieces


def signature(head: Json) -> str:
    """Only policy operands affect cached exact polynomials, not fit metadata."""
    return json.dumps({k: head[k] for k in ["coefficients", "temperature"]}, sort_keys=True)


def polynomial_value(p: list[F], x: F) -> F:
    """Horner evaluation has no rounding because every operation remains rational."""
    value = F(0)
    for coefficient in reversed(p):
        value = value * x + coefficient
    return value


def local_reference(head: Json, feature: int, x: F) -> F:
    """The independent polynomial path handles the right endpoint by its last piece."""
    unique = sorted(set(KNOTS))
    piece = (
        len(unique) - 2
        if x == 1
        else next(i for i in range(len(unique) - 1) if unique[i] <= x < unique[i + 1])
    )
    return polynomial_value(polynomials(signature(head))[feature][piece], x)


def exact(head: Json, x: list[F], *, polynomial: bool = False) -> F:
    """Temperature and global terms use exact interpretations of their binary64 bytes."""
    c = [F(v) for v in head["coefficients"]]
    local = [
        local_reference(head, j, x[j + 1])
        if polynomial
        else de_boor(c[2 + 8 * j : 10 + 8 * j], x[j + 1])
        for j in range(4)
    ]
    return (c[0] * x[0] + c[1] + sum(local, F(0))) / F(head["temperature"])


@lru_cache(maxsize=16)
def _certificate(sig: str, table_sig: str) -> Json:
    """Measured exact endpoint residuals replace assumptions about table construction rounding."""
    head, data = json.loads(sig), json.loads(table_sig)
    table = np.asarray(data, dtype="<i2")
    old = previous.certificate(head, table)
    cells, audit = [], True
    for j in range(4):
        endpoint_errors = [
            abs(local_reference(head, j, F(i, 64)) - F(int(table[j, i]), 2048)) for i in range(65)
        ]
        feature = []
        for i in range(64):
            lo, hi = F(i, 64), F(i + 1, 64)
            extrema: list[F] = []
            for k, (a, b) in enumerate(
                zip(sorted(set(KNOTS))[:-1], sorted(set(KNOTS))[1:], strict=True)
            ):
                if a <= hi and b >= lo:
                    p = polynomials(sig)[j][k]
                    extrema.extend(abs(2 * p[2] + 6 * p[3] * v) for v in [max(a, lo), min(b, hi)])
            maximum = max(extrema)
            audit &= maximum == F(old["cells"][j][i]["second_derivative_exact"])
            feature.append(str(maximum / (8 * 64**2) + max(endpoint_errors[i : i + 2])))
        cells.append(feature)
    return dict(
        version=VERSION,
        head_sha256=canonical_hash(head),
        table_sha256=canonical_hash(data),
        cells=cells,
        rounding=ROUNDING,
        independent_remainder_audit=bool(audit),
        exact_threshold=str(L),
    )


def certificate(head: Json, table: previous.Array) -> Json:
    """Recompute the expected table before allowing its bytes into the action contract."""
    previous.validate(head)
    return dict(json.loads(json.dumps(_certificate(signature(head), json.dumps(table.tolist())))))


def guard(head: Json, x: list[F], table: previous.Array, supplied: Json | None) -> Json:
    """Unsafe certificates fall back to the exact new policy, never the historical sigmoid."""
    previous.validate(head)
    x = [F(v) for v in x]
    if len(x) != 5 or any(not 0 <= v <= 1 for v in x[1:]):
        raise ValueError("features")
    reason = ""
    if supplied is None:
        reason = "missing_bounds"
    elif table.shape != (4, 65) or table.dtype != np.dtype("<i2"):
        reason = "table_format"
    elif np.any((table == -32768) | (table == 32767)):
        reason = "saturated_table"
    elif not np.array_equal(table, previous.old.table(head, 65, "int16")[0]):
        reason = "swapped_table_or_head"
    elif supplied != certificate(head, table) or not supplied["independent_remainder_audit"]:
        reason = "changed_or_underbound_certificate"
    interval = None
    candidate = None
    if not reason:
        c = [F(v) for v in head["coefficients"]]
        center, radius = c[0] * x[0] + c[1], F(0)
        for j in range(4):
            position = x[j + 1] * 64
            i = min(position.numerator // position.denominator, 63)
            weight = position - i
            center += ((1 - weight) * int(table[j, i]) + weight * int(table[j, i + 1])) / 2048
            assert supplied is not None
            radius += F(supplied["cells"][j][i])
        try:
            lower_exact = (center - radius) / F(head["temperature"])
            upper_exact = (center + radius) / F(head["temperature"])
            lo = outward(lower_exact, -1)
            hi = outward(upper_exact, 1)
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise OverflowError("interval")
            if F(lo) > lower_exact or F(hi) < upper_exact:
                raise ValueError("outward_rounding")
            interval = [str(F(lo)), str(F(hi))]
            candidate = region(F(lo), F(hi))
            reason = "certified" if candidate is not None else "threshold_intersection"
        except OverflowError:
            reason = "saturation_or_overflow"
        except ValueError:
            reason = "altered_rounding_contract"
    action = candidate
    if action is None:
        z = exact(head, x)
        action = region(z, z)
    return dict(
        x=[str(v) for v in x],
        action=action,
        fast_path=candidate is not None,
        interval=interval,
        fallback_reason=reason,
    )


def panel(head: Json) -> tuple[list[list[F]], list[str]]:
    """Exact threshold witnesses use rational holistic inputs; neighbors remain binary64."""
    frozen, _, _ = previous.old.panel(head)
    rows = [[F(float(v)) for v in x] for x in frozen[:4096]]
    kinds = ["random"] * len(rows)
    for j in range(4):
        for knot in sorted(set(KNOTS)):
            for value in [
                knot,
                F(float(np.nextafter(float(knot), -np.inf))),
                F(float(np.nextafter(float(knot), np.inf))),
            ]:
                if 0 <= value <= 1:
                    x = [F(0), *[F(1, 2)] * 4]
                    x[j + 1] = value
                    rows.append(x)
                    kinds.append("knot" if value == knot else "nextafter_knot")
    base = [F(0), *[F(1, 2)] * 4]
    for threshold in [-L, L]:
        holistic = (
            (threshold - exact(head, base)) * F(head["temperature"]) / F(head["coefficients"][0])
        )
        rows.append([holistic, *base[1:]])
        kinds.append("exact_threshold")
        for direction in [-np.inf, np.inf]:
            rows.append([F(float(np.nextafter(float(holistic), direction))), *base[1:]])
            kinds.append("nextafter_threshold")
    return rows, kinds


def evaluate(head: Json, table: previous.Array) -> list[Json]:
    """Keep every independent reference value and every migration difference for cold replay."""
    vectors, kinds = panel(head)
    cert = certificate(head, table)
    probabilities = previous.old.direct(head, np.asarray([[float(v) for v in x] for x in vectors]))
    rows = []
    for i, (x, kind) in enumerate(zip(vectors, kinds, strict=True)):
        row = guard(head, x, table, cert)
        z = exact(head, x, polynomial=True)
        lo, hi = [F(v) for v in row["interval"]] if row["interval"] else (z, z)
        row.update(
            vector_id=i,
            kind=kind,
            reference_logit=str(z),
            reference_action=region(z, z),
            interval_escape=not lo <= z <= hi,
            original_probability=float(probabilities[i]),
            original_action=previous.action(float(probabilities[i])),
            old_policy_input_scope="binary64_projection"
            if kind == "exact_threshold"
            else "binary64",
        )
        rows.append(row)
        if (i + 1) % 256 == 0:
            progress("dyadic_vectors", i + 1, len(vectors) - i - 1)
    progress("dyadic_vectors", len(vectors), 0)
    return rows
