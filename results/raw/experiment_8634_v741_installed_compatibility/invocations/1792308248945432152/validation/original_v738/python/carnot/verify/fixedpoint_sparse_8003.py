"""REQ-VERIFY-8003: integer CPU emulation preserves measurable numerical errors.

The input scaler remains a host operation. Lookup, products, sums and stored
updates use the declared integer format; no accelerator executes this code.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.verify import sparse_energy_7996 as s
from carnot.verify.typed_development_7997 import action

Json = dict[str, Any]
CONFIG = dict(
    bits=[16, 24],
    fractional_bits=12,
    rounding="nearest_ties_to_even",
    overflow="saturate",
    max_groups=128,
    group_budgets=dict(fit=96, tune=32),
    primary_seed=17,
    learning_rate=0.01,
    probability_error_max=0.01,
    action_agreement_min=0.99,
    threshold_margin=0.01,
    thresholds=[0.05, 0.75],
    basis_grid_intervals=4096,
    scaler_venue="host_float64_fit_scaler",
    sigmoid_venue="host_float64_final_readout",
)


class Fixed:
    """Count each overflow so a good probability cannot hide lost integer range."""

    def __init__(self, bits: int):
        self.minimum = -(1 << (bits - 1))
        self.maximum = (1 << (bits - 1)) - 1
        self.saturations = 0

    def clamp(self, value: int) -> int:
        """Signed storage saturates after each arithmetic operation."""
        self.saturations += int(value < self.minimum or value > self.maximum)
        return max(self.minimum, min(self.maximum, value))

    def encode(self, value: float) -> int:
        """Python rounding uses even integers at exact half-integer ties."""
        return self.clamp(round(value * 4096))

    def mul(self, left: int, right: int) -> int:
        """The wide product is rounded before returning to signed storage."""
        return self.clamp(round(left * right / 4096))

    def add(self, left: int, right: int) -> int:
        """Each stored accumulator obeys the same overflow rule as coefficients."""
        return self.clamp(left + right)


def grids(head: Json, fit: list[Json]) -> Json:
    """Only fit bounds and fit q values define the frozen lookup tables."""
    basis = sorted(set(np.linspace(0, 1, 4097).tolist() + s.KNOTS))
    q = sorted(
        set(
            np.linspace(1e-4, 1 - 1e-4, 4097).tolist()
            + [float(np.clip(r["q"], 1e-4, 1 - 1e-4)) for r in fit if r["q"] is not None]
        )
    )
    matrix = s.basis(np.repeat(np.asarray(basis)[:, None], 9, axis=1))[:, :12]
    return dict(
        basis=basis,
        basis_values=matrix.tolist(),
        q=q,
        logit_values=np.log(np.asarray(q) / (1 - np.asarray(q))).tolist(),
        fit_scaler=deepcopy(head["scaler"]),
        fit_group_count=len({r["source_cluster_id"] for r in fit}),
        frozen_before_tune=True,
    )


def nearest(grid: list[float], value: float) -> int:
    """Choose the closest frozen cell with the lower cell winning distance ties."""
    return int(np.argmin(np.abs(np.asarray(grid) - value)))


def emulate(head: Json, raw: list[float], y: int, table: Json, bits: int) -> Json:
    """One integer prediction and lazy-decay update expose every stored value."""
    q = Fixed(bits)
    scaled, clips = s.scale(np.asarray([raw], dtype=float), head["scaler"])
    ids, weights = [108], [4096]
    for feature, value in enumerate(scaled[0]):
        slot = nearest(table["basis"], q.encode(float(value)) / 4096)
        for index, weight in enumerate(table["basis_values"][slot]):
            if weight > 0:
                ids.append(feature * 12 + index)
                weights.append(q.encode(weight))
    theta = [q.encode(v) for v in head["parameters"]]
    decay = q.encode(head["decay_scale"])
    inv_t = q.encode(1 / head["temperature"])
    offset = q.encode(
        table["logit_values"][nearest(table["q"], float(np.clip(raw[0], 1e-4, 1 - 1e-4)))]
    )

    def probability(coefficients: list[int], multiplier: int) -> float:
        z = offset
        for i, weight in zip(ids, weights, strict=True):
            z = q.add(z, q.mul(weight, q.mul(coefficients[i], multiplier)))
        return float(expit(q.mul(z, inv_t) / 4096))

    p = probability(theta, decay)
    residual = q.mul(q.encode(p - y), inv_t)
    gradient = [q.mul(q.encode(0.002), q.mul(v, decay)) for v in theta]
    data_gradient = [q.mul(weight, residual) for weight in weights]
    for i, g in zip(ids, data_gradient, strict=True):
        gradient[i] = q.add(gradient[i], g)
    next_decay = q.mul(decay, q.encode(1 - 0.002 * CONFIG["learning_rate"]))
    inverse_decay = q.encode(4096 / next_decay)
    rate = q.encode(CONFIG["learning_rate"])
    changed = theta.copy()
    for i, g in zip(ids, data_gradient, strict=True):
        changed[i] = q.add(changed[i], -q.mul(q.mul(rate, g), inverse_decay))
    updated_probability = probability(changed, next_decay)
    effective = [q.mul(v, next_decay) / 4096 for v in changed]
    return dict(
        probability=p,
        updated_probability=updated_probability,
        gradient=[g / 4096 for g in gradient],
        updated_coefficients=effective,
        stored_coefficients=[v / 4096 for v in changed],
        decay_scale=next_decay / 4096,
        coefficient_touches=len(ids),
        active_indices=ids,
        saturation_count=q.saturations,
        clip_counts=clips,
        global_decay_writes=1,
        snapshot_coefficient_copies=109,
    )


def evaluate(head: Json, data: Json) -> Json:
    """Bound source groups and compare the actual sparse head without fitting it."""
    table = grids(head, data["fit"])
    selected, seen = [], set()
    for role in ("fit", "tune"):
        role_count = 0
        for row in s.usable(data[role]):
            key = row["source_cluster_id"]
            if (
                key not in seen
                and row.get("y") in (0, 1)
                and role_count < CONFIG["group_budgets"][role]
            ):
                seen.add(key)
                role_count += 1
                selected.append(dict(row, role=role, fixture=False))
    low, high = np.asarray(head["scaler"]["minimum"]), np.asarray(head["scaler"]["maximum"])
    for index, coordinate in enumerate(sorted(set(s.KNOTS))):
        raw = low + (high - low) * coordinate
        selected.append(
            dict(
                q=float(raw[0]),
                features=raw[1:].tolist(),
                y=index % 2,
                family_id=f"knot-{index}",
                source_cluster_id=f"fixture-{index}",
                role="knot_fixture",
                fixture=True,
            )
        )
    rows = []
    for bits in (16, 24):
        for index, source in enumerate(selected):
            x = np.asarray([source["q"], *source["features"]], dtype=float)
            p = float(s.predict(head, x[None, :])[0])
            changed, touches = s.update(head, x, source["y"], 0.01)
            fp = emulate(head, x.tolist(), source["y"], table, bits)
            near = min(abs(p - t) for t in CONFIG["thresholds"]) <= 0.01
            gradient = s.dense_gradient(head, x, source["y"]).tolist()
            coefficients = s.parameters(changed).tolist()
            updated_p = float(s.predict(changed, x[None, :])[0])
            rows.append(
                dict(
                    bits=bits,
                    family_id=source["family_id"],
                    source_cluster_id=source["source_cluster_id"],
                    role=source["role"],
                    fixture=source["fixture"],
                    seed=17,
                    arm=f"signed{bits}_q12",
                    numerator=max(
                        abs(p - fp["probability"]), abs(updated_p - fp["updated_probability"])
                    ),
                    initial_probability_error=abs(p - fp["probability"]),
                    updated_probability_error=abs(updated_p - fp["updated_probability"]),
                    denominator=1,
                    eligibility=True,
                    failure_status=False,
                    censor_status=False,
                    float_probability=p,
                    fixed_probability=fp["probability"],
                    float_updated_probability=updated_p,
                    fixed_updated_probability=fp["updated_probability"],
                    float_gradient=gradient,
                    fixed_gradient=fp["gradient"],
                    maximum_gradient_error=float(
                        np.max(np.abs(np.asarray(gradient) - fp["gradient"]))
                    ),
                    float_updated_coefficients=coefficients,
                    fixed_updated_coefficients=fp["updated_coefficients"],
                    maximum_update_error=float(
                        np.max(np.abs(np.asarray(coefficients) - fp["updated_coefficients"]))
                    ),
                    float_action=action(p),
                    fixed_action=action(fp["probability"]),
                    near_threshold=near,
                    action_eligible=not near,
                    action_agreed=action(p) == action(fp["probability"]),
                    float_coefficient_touches=touches["coefficient_touches"],
                    **{
                        k: fp[k]
                        for k in (
                            "coefficient_touches",
                            "saturation_count",
                            "clip_counts",
                            "global_decay_writes",
                            "snapshot_coefficient_copies",
                            "active_indices",
                            "decay_scale",
                        )
                    },
                )
            )
            if index % 32 == 0:
                print(
                    f"[exp8003] fixed_point bits={bits} completed={index + 1}/{len(selected)}",
                    flush=True,
                )
    gates = {}
    boundary_rows = []
    for bits in (16, 24):
        subset = [r for r in rows if r["bits"] == bits]
        away = [r for r in subset if r["action_eligible"]]
        error = max(r["numerator"] for r in subset)
        agreed = sum(r["action_agreed"] for r in away)
        saturation = sum(r["saturation_count"] for r in subset)
        gates[str(bits)] = dict(
            maximum_probability_error=error,
            action_agreement_numerator=agreed,
            action_agreement_denominator=len(away),
            typed_action_agreement=agreed / len(away) if away else None,
            unexpected_saturation_count=saturation,
            passed=bool(away) and error <= 0.01 and agreed / len(away) >= 0.99 and saturation == 0,
            scope="cpu_emulation_only",
        )
        for value in (
            0.5 / 4096,
            1.5 / 4096,
            -1.5 / 4096,
            -(1 << (bits - 1)) / 4096 - 1,
            (1 << (bits - 1)) / 4096 + 1,
        ):
            fixed = Fixed(bits)
            encoded = fixed.encode(value)
            boundary_rows.append(
                dict(
                    bits=bits,
                    input=value,
                    encoded=encoded,
                    saturation_count=fixed.saturations,
                    expected_saturation=abs(value) > (1 << (bits - 1)) / 4096,
                    scope="arithmetic_fixture_only",
                )
            )
    control = deepcopy(head)
    control["parameters"] = [0.0] * 108 + [0.5]
    before = float(s.predict(control, np.full((1, 9), 0.5))[0])
    updated, _ = s.update(control, np.full(9, 0.5), 0, 0.01)
    after = float(s.predict(updated, np.full((1, 9), 0.5))[0])
    return dict(
        fixed_point_rows=rows,
        quantization_gates=gates,
        quantization_ready_score=int(any(g["passed"] for g in gates.values())),
        qualifying_formats=[int(k) for k, g in gates.items() if g["passed"]],
        near_threshold_rows=[r for r in rows if r["near_threshold"]],
        fixed_point_boundary_rows=boundary_rows,
        lookup_grids=table,
        positive_control_results=dict(
            working=after < before,
            genuine_headroom=0 < before < 1,
            scope="protocol_fixture_only",
            before=before,
            after=after,
        ),
        independent_source_groups=len(seen),
    )
