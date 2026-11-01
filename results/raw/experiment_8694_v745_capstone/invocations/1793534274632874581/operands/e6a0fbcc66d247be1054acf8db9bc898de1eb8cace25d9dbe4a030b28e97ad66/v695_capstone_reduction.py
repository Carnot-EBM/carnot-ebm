"""REQ-REPORT-8030: source and block operands retain their original claim limits.

We reuse qualified primitive readers. The capstone computes its own hypothesis
family so missing or disqualified producers cannot supply positive claims.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_8021_v695_typed_decision_test as static
from carnot import experiment_8019_v695_eligible_targets as targets_reader
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.verify import learning_retention_audit_8026 as learning
from carnot.verify import likelihood_calibration_8023 as source

Json = dict[str, Any]
HYPOTHESES = ("static_policy", "source_intervention", "online_update")


def bootstrap(diff: list[float], block: int = 1) -> Json:
    """Center real bootstrap draws to invert the one-sided test of zero benefit."""
    x = np.asarray(diff, dtype=float)
    valid = np.isfinite(x)
    result: Json = dict(
        gain=None,
        raw_p=1.0,
        independent=int(valid.sum()) if block == 1 else int(valid.any()),
        eligible_slots=int(valid.sum()),
        independent_unit="source group"
        if block == 1
        else "single chronological environment; block uncertainty",
        slot_count=len(x),
        block_length=block,
        draws=10000,
        descriptive_interval=[None, None],
        centered_errors=[],
    )
    if not valid.any():
        return result
    rng = np.random.default_rng(6958030)
    length = min(block, len(x))
    starts = rng.integers(0, len(x) - length + 1, (10000, int(np.ceil(len(x) / length))))
    ids = (starts[:, :, None] + np.arange(length)).reshape(10000, -1)[:, : len(x)]
    samples = x[ids]
    n = np.isfinite(samples).sum(axis=1)
    draws = np.nansum(samples[n > 0], axis=1) / n[n > 0]
    mean = float(np.nanmean(x))
    errors = draws - mean
    result.update(
        gain=mean,
        raw_p=float((1 + np.sum(errors >= mean)) / (1 + len(errors))),
        completed_draws=len(errors),
        censored_draws=10000 - len(errors),
        descriptive_interval=(mean - np.quantile(errors, [0.975, 0.025])).tolist(),
        centered_errors=errors.tolist(),
    )
    return result


def family(primaries: list[Json], qualified: list[bool]) -> list[Json]:
    """Holm includes all three hypotheses; failed science gates receive p=1."""
    ps = [r["raw_p"] if q else 1.0 for r, q in zip(primaries, qualified, strict=True)]
    order = sorted(range(3), key=lambda i: (ps[i], i))
    result: list[Json] = [{} for _ in order]
    previous = 0.0
    for rank, i in enumerate(order):
        r = {k: v for k, v in primaries[i].items() if k != "centered_errors"}
        adjusted = max(previous, min(1.0, ps[i] * (3 - rank)))
        previous = adjusted
        errors = primaries[i]["centered_errors"]
        lower = r["gain"] - float(np.quantile(errors, 1 - 0.05 / (3 - rank))) if errors else None
        result[i] = dict(
            r,
            hypothesis=HYPOTHESES[i],
            family_p=ps[i],
            holm_adjusted_p=adjusted,
            holm_inverted_lower=lower,
            qualified=qualified[i],
            positive_claim=bool(
                qualified[i]
                and adjusted < 0.05
                and lower is not None
                and lower > 0
                and r["gain"] >= (0.01 if i == 1 else 0.02)
            ),
        )
    return result


def independent(data: Json, number: int) -> Json:
    """Re-read original predictions, answer tokens and checkpoint states."""
    if number == 8019 and data.get("rows"):
        inputs = json.loads(checked(data["checkpoint_references"][0]).read_text())
        reduced = targets_reader.reduce(inputs)
        if reduced["support_by_role"] != data["support_by_role"] or reduced["rows"] != data["rows"]:
            raise ValueError("eligibility_or_complete_annotation_drift")
        return dict(
            measurement_available=True,
            support_by_role=reduced["support_by_role"],
            sample_size_budget=reduced["sample_size_budget"],
            target_orientation="one means any authenticated unsupported span in the complete answer",
        )
    if number not in (8021, 8024, 8026):
        return dict(measurement_available=bool(data.get("rows")))
    if not data.get("rows"):
        return dict(
            measurement_available=False,
            primary=bootstrap([]),
            missing_reason="no_terminal_measurement_primitives",
        )
    if number == 8021:
        inputs = json.loads(checked(data["checkpoint_references"][0]).read_text())
        seal = json.loads(checked(data["prediction_seal"]).read_text())
        access = json.loads(checked(data["label_access_receipt"]).read_text())
        original_targets = json.loads(checked(access["target_reference"]).read_text())["rows"]
        if original_targets != inputs["targets"]:
            raise ValueError("original_targets_changed")
        static.verify_seal(inputs["predictions"], seal["original_seal"])
        measurement = json.loads(checked(seal["measurement"]).read_text())
        original = [r for r in measurement["primitive_rows"] if r["role"] == "stream"]
        tune = [r for r in measurement["primitive_rows"] if r["role"] == "tune"]
        if (
            original != inputs["predictions"]
            or inputs["comparator"] != static.select_comparator(tune)
            or access["prediction_receipt"] != data["prediction_seal"]
            or access["retention_labels_opened"] is not False
            or access["stream_access_monotonic_ns"] <= seal["invocation_seal_checked_monotonic_ns"]
        ):
            raise ValueError("prediction_or_target_access_contract")
        diagnostic = static.reduce(inputs["predictions"], inputs["targets"], inputs["comparator"])
        rows = diagnostic["rows"]
        indexed = {(static.key(r), r["seed"], r["family_id"]): r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError("duplicate_prediction")
        pairs = []
        for r in rows:
            if static.key(r) != static.CONFIG["primary"]:
                continue
            b = indexed[(inputs["comparator"], r["seed"], r["family_id"])]
            if r["eligibility"] and b["eligibility"]:
                pairs.append(
                    dict(
                        source_cluster_id=r["source_cluster_id"],
                        gain=b["actual_cost"] - r["actual_cost"],
                        seed=r["seed"],
                    )
                )
        groups = sorted({r["source_cluster_id"] for r in pairs})
        diff = [
            float(np.mean([r["gain"] for r in pairs if r["source_cluster_id"] == g]))
            for g in groups
        ]
        return dict(
            measurement_available=True,
            primary=bootstrap(diff),
            diagnostic=diagnostic,
            primitive_sha256=canonical_hash(inputs),
            comparator=inputs["comparator"],
            seed_reduction="mean within source across all registered seeds",
        )
    if number == 8024:
        bundle = json.loads(checked(data["checkpoint_references"][0]).read_text())
        reduced = source.reduce(bundle["panel"], bundle["token_rows"])
        if not reduced["passed"]:
            raise ValueError("source_complete_answer_or_duplicate_control")
        features = reduced["source_feature_rows"]
        public = {r["family_id"]: r for r in bundle["panel"]["rows"]}
        features = [dict(r, q=public[r["family_id"]]["q"]) for r in features]
        labels = {r["family_id"]: r["eligible_y"] for r in bundle["targets"]}
        a = source.predict(bundle["treatment_head"], features)
        b = source.predict(bundle["comparator_head"], features)
        if (
            canonical_hash(dict(features=features, treatment=a.tolist(), comparator=b.tolist()))
            != bundle["prediction_seal"]
        ):
            raise ValueError("source_prediction_seal")
        paired = [
            (
                r["source_cluster_id"],
                float((q - labels[r["family_id"]]) ** 2 - (p - labels[r["family_id"]]) ** 2),
            )
            for r, p, q in zip(features, a, b, strict=True)
            if labels[r["family_id"]] is not None
        ]
        groups = sorted({g for g, d in paired})
        diff = [float(np.mean([d for g, d in paired if g == group])) for group in groups]
        reduced["support_passed"] = len(groups) >= 72 and all(
            len({r["source_cluster_id"] for r in features if labels[r["family_id"]] == y}) >= 12
            for y in (0, 1)
        )
        return dict(measurement_available=True, primary=bootstrap(diff), diagnostic=reduced)
    bundle = json.loads(checked(data["audit_bundle"]).read_text())
    audit = learning.reduce(bundle)
    rows = audit["independent_issue_rows"]
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["arm"], r["seed"], r["slot"]): r for r in rows}
    diff = []
    for slot in sorted({r["slot"] for r in rows if r["slot"] >= 36}):
        pairs = [(by[("decision_loss", s, slot)], by[("uniform", s, slot)]) for s in seeds]
        diff.append(
            float(np.mean([b["cost"] - a["cost"] for a, b in pairs]))
            if all(a["eligibility"] and b["eligibility"] for a, b in pairs)
            else float("nan")
        )
    return dict(
        measurement_available=True,
        primary=bootstrap(diff, 32),
        diagnostic=audit,
        block_sensitivity=[bootstrap(diff, n) for n in (16, 64)],
    )
