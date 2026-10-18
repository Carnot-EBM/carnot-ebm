"""REQ-REPORT-8043: independent equations retain the producer's claim limits.

A missing source evaluation cannot supply hypotheses. Available learning rows
are rebuilt from causal events before any producer aggregate is compared.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.verify import learning_benefit_8039 as learning
from carnot import experiment_8040_v696_native_transaction_cost as native

Json = dict[str, Any]


def bootstrap(diff: list[float], margin: float, block: int = 1) -> Json:
    """Test the frozen minimum benefit so tiny gains cannot pass a zero test."""
    x = np.asarray(diff, dtype=float)
    result: Json = dict(
        gain=None,
        raw_p=1.0,
        margin=margin,
        draws=10000,
        completed_draws=0,
        censored_draws=10000,
        eligible_slots=int(np.isfinite(x).sum()),
        slot_count=len(x),
        block_length=block,
        centered_errors=[],
        interval=[None, None],
    )
    if not np.isfinite(x).any():
        return result
    length = min(block, len(x))
    rng = np.random.default_rng(6968043)
    starts = rng.integers(0, len(x) - length + 1, (10000, int(np.ceil(len(x) / length))))
    ids = (starts[:, :, None] + np.arange(length)).reshape(10000, -1)[:, : len(x)]
    samples = x[ids]
    n = np.isfinite(samples).sum(axis=1)
    draws = np.nansum(samples[n > 0], axis=1) / n[n > 0]
    mean = float(np.nanmean(x))
    errors = draws - mean
    result.update(
        gain=mean,
        raw_p=float((1 + np.sum(errors >= mean - margin - 1e-15)) / (1 + len(errors))),
        completed_draws=len(errors),
        censored_draws=10000 - len(errors),
        centered_errors=errors.tolist(),
        interval=(mean - np.quantile(errors, [0.975, 0.025])).tolist(),
    )
    return result


def family(primaries: list[Json], qualified: list[bool]) -> list[Json]:
    """Keep all three registered hypotheses in Holm even when evidence is absent."""
    ps = [r["raw_p"] if q else 1.0 for r, q in zip(primaries, qualified, strict=True)]
    result: list[Json] = [{} for _ in range(3)]
    previous = 0.0
    for rank, i in enumerate(sorted(range(3), key=lambda j: (ps[j], j))):
        r = {k: v for k, v in primaries[i].items() if k != "centered_errors"}
        adjusted = max(previous, min(1.0, ps[i] * (3 - rank)))
        previous = adjusted
        errors = primaries[i]["centered_errors"]
        lower = r["gain"] - float(np.quantile(errors, 1 - 0.05 / (3 - rank))) if errors else None
        result[i] = dict(
            r,
            hypothesis=f"H{i + 1}",
            family_p=ps[i],
            holm_adjusted_p=adjusted,
            holm_inverted_lower=lower,
            qualified=qualified[i],
            positive_claim=bool(
                qualified[i] and adjusted < 0.05 and lower is not None and lower > r["margin"]
            ),
            uncertainty_scope="conditional finite exposed trajectory; seeds add zero independent environments",
        )
    return result


def independent(data: Json, number: int) -> Json:
    """Use the shipped separate audit equations and compare every reconstructed field."""
    if number in (8040, 8042) and data.get("rows"):
        raw = Path(data["raw_directory"])
        if number == 8040:
            rows = json.loads((raw / "transaction_rows.json").read_text())["rows"]
            reduced = native.summarize(rows, data["config"])
            for key, observed in reduced.items():
                learning.equal("transaction_reduction:" + key, observed, data[key])
            return dict(
                measurement_available=True,
                rows=rows,
                costs=reduced,
                missing_service_components=data["missing_service_components"],
                complete_service_measured=False,
            )
        rows = json.loads((raw / "primitive_rows.json").read_text())["rows"]
        learning.equal("fallback_primitive_drift", rows, data["rows"])
        numeric = [r for r in rows if r["metric"] == "interval_containment"]
        denominator = len(numeric)
        fraction = sum(r["used_float64"] for r in numeric) / denominator
        learning.equal("fallback_fraction", fraction, data["fallback_fraction"])
        learning.equal("fallback_denominator", denominator, data["fallback_fraction_denominator"])
        return dict(
            measurement_available=True,
            rows=rows,
            fallback_fraction=fraction,
            fallback_denominator=denominator,
            complete_service_measured=False,
            missing_service_components=data["missing_cost_components"],
        )
    if number != 8039 or not data.get("rows"):
        return dict(
            measurement_available=False, missing_reason="no_authenticated_measurement_primitives"
        )
    bundle = json.loads(checked(data["audit_bundle"]).read_text())
    replay = learning.replay_trajectory(Path(bundle["trajectory"]))
    retained = learning.retention(
        bundle, replay, checked(data["retention_prediction_seal"]), cold=True
    )
    compared = learning.compare(replay["rows"], retained["retention_rows"])
    for fields in (replay, retained, compared):
        for key, observed in fields.items():
            learning.equal("producer_drift:" + key, observed, data[key])
    rows = [r for r in compared["later_loss_rows"] if r["comparator"] == "cumulative"]
    diff = []
    for slot in sorted({r["slot"] for r in rows}):
        group = [r for r in rows if r["slot"] == slot]
        diff.append(
            float(np.mean([r["paired_gain"] for r in group]))
            if all(r["paired_gain"] is not None for r in group)
            else float("nan")
        )
    primary = bootstrap(diff, 0.02, 32)
    gates = compared["primary_hypothesis_results"][0]["gates"]
    return dict(
        measurement_available=True,
        primary=primary,
        block_sensitivity=[bootstrap(diff, 0.02, n) for n in (16, 64)],
        scientific_qualified=all(gates.values()),
        producer_gates=gates,
        numerical_agreement=replay["numerical_agreement"],
        sample_size_budget=replay["sample_size_budget"],
        historical_exposure=bundle["historical_exposure"],
        rows=replay["rows"],
        retention_rows=retained["retention_rows"],
        secondary_results=compared["primary_hypothesis_results"][1:],
        retention_drift_rows=compared["retention_drift_rows"],
    )
