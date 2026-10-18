"""REQ-VERIFY-7984: matched information controls on cached original responses.

All heads have the same capacity. Interventions measure dependence on evidence;
human outcomes remain valid only for the complete original source and response.
"""

from __future__ import annotations

import math
from functools import lru_cache
import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import evidence_features_7980 as features
from carnot.verify import multivariate_energy_7982 as m
from carnot.verify import qwen_energy_calibration_7972 as scalar
from carnot.verify import source_alignment as alignment
from carnot.verify.source_projection import PUBLIC_KEYS

Json = dict[str, Any]
ARMS = ("full", "q_only", "source_only", "q_donor")
CONFIG = dict(
    m.CONFIG,
    primary="full",
    arms=list(ARMS),
    capacity=97,
    donor_seed=69284,
    length_bins="floor(log2(source_utf8_byte_length))",
    draws=10000,
    loss=dict(accept="5*y", reject="1-y", escalate=0.25),
)


def interventions(public: Json) -> Json:
    """Choose label-free bijections within roles and length bins before fitting.

    A minimum-cost assignment uses seeded hashes only to order valid pairings.
    Bins without a complete same-source-free permutation remain explicit nulls.
    """
    result = {}
    for role, rows in public.items():
        bins: Json = {}
        for row in rows:
            if set(row) != PUBLIC_KEYS:
                raise ValueError("public_fields")
            size = len(bytes.fromhex(row["source_bytes"]))
            bins.setdefault(int(math.log2(max(1, size))), []).append(row)
        donors = {}
        for group in bins.values():
            n = len(group)
            costs = np.full((n, n), 1e9)
            for i, left in enumerate(group):
                for j, right in enumerate(group):
                    if features.normalized(
                        bytes.fromhex(left["source_bytes"])
                    ) != features.normalized(bytes.fromhex(right["source_bytes"])):
                        costs[i, j] = (
                            int(
                                canonical_hash(
                                    [69284, role, left["family_id"], right["family_id"]]
                                )[-12:],
                                16,
                            )
                            / 16**12
                        )
            ii, jj = linear_sum_assignment(costs)
            if all(costs[i, j] < 1e9 for i, j in zip(ii, jj, strict=True)):
                donors.update(
                    {group[i]["family_id"]: group[j] for i, j in zip(ii, jj, strict=True)}
                )
        result[role] = []
        for row in rows:
            donor = donors.get(row["family_id"])
            swapped = dict(row, source_bytes=donor["source_bytes"]) if donor else None
            erased = dict(row, source_bytes="")
            result[role].append(
                dict(
                    family_id=row["family_id"],
                    original=row,
                    duplicate=dict(row),
                    erased=erased,
                    donor=swapped,
                    donor_family_id=donor["family_id"] if donor else None,
                    length_bin=int(math.log2(max(1, len(bytes.fromhex(row["source_bytes"]))))),
                    donor_exclusion=None if donor else "no_within_bin_derangement",
                    target_scope="original_source_only",
                    hashes={
                        k: canonical_hash(v)
                        for k, v in dict(
                            original=row, duplicate=row, erased=erased, donor=swapped
                        ).items()
                    },
                )
            )
        print(f"[exp7984] public_mapping role={role} rows={len(rows)}", flush=True)
    return result


@lru_cache(maxsize=4096)
def cached_features(source: str, answer: str) -> Any:
    """Cache exact public bytes only; labels and opaque join IDs cannot affect features."""
    if source:
        return features.extract(dict(family_id="public", source_bytes=source, answer_bytes=answer))[
            "values"
        ]
    units = alignment.sentence_spans(bytes.fromhex(answer))
    tail = alignment.pair_features(b"", units)[-4:]
    return [v for v in tail for _ in range(2)]


def source_features(row: Json) -> Any:
    """Empty evidence uses existing null-location features, without inventing truth."""
    values = cached_features(row["source_bytes"], row["answer_bytes"])
    return list(values) if values is not None else None


def arm_rows(
    rows: list[Json], views: list[Json], arm: str, intervention: str = "original"
) -> list[Json]:
    """Mask information while retaining nine inputs, IDs and original labels."""
    lookup = {v["family_id"]: v for v in views}
    result = []
    for index, row in enumerate(rows):
        view = lookup[row["family_id"]]
        public = view["donor" if arm == "q_donor" and intervention == "original" else intervention]
        vector = (
            row["features"]
            if intervention in {"original", "duplicate"} and arm != "q_donor"
            else source_features(public)
            if public is not None
            else None
        )
        result.append(
            dict(
                row,
                q=0.5 if arm == "source_only" and row["q"] is not None else row["q"],
                features=[0.0] * 8 if arm == "q_only" and vector is not None else vector,
            )
        )
        if index % 32 == 31:
            print(
                f"[exp7984] feature_inputs arm={arm} intervention={intervention} rows={index + 1}/{len(rows)}",
                flush=True,
            )
    return result


def matched_data(data: Json, views: Json) -> Json:
    """Use exactly the same eligible fit and tune targets in all four controls."""
    controls = {
        arm: {
            role: arm_rows(data[role], views[role], arm)
            for role in ("fit", "tune", "policy_design")
        }
        for arm in ARMS
    }
    for role in ("fit", "tune"):
        shared = set.intersection(
            *[{r["family_id"] for r in m.valid(c[role])} for c in controls.values()]
        )
        for current in controls.values():
            current[role] = [r for r in current[role] if r["family_id"] in shared]
    return controls


def fit(data: Json, views: Json) -> Json:
    """Use the exact Exp7982 Gibbs optimization budget for every information arm."""
    result: Json = dict(
        arms={},
        optimizer_work=dict(
            total_steps=2400,
            per_head_steps=200,
            coefficient_touches=97 * 2400,
            tune_temperature_scores=12 * 17,
            pretrained_model_calls=0,
        ),
    )
    controls = matched_data(data, views)
    for arm in ARMS:
        current = controls[arm]
        m.validate_data(current)
        fitting, tuning = m.valid(current["fit"]), m.valid(current["tune"])
        fs, ts = scalar.support(fitting, 128, 16), scalar.support(tuning, 32, 4)
        if not fs["passed"] or not ts["passed"]:
            raise ValueError("support_floor")
        x, tx = m.inputs(fitting), m.inputs(tuning)
        means, scales = x.mean(axis=0), x.std(axis=0)
        scales = np.where(scales < 1e-12, 1.0, scales)
        x, tx = (x - means) / scales, (tx - means) / scales
        y, ty = (np.asarray([r["y"] for r in rr]) for rr in (fitting, tuning))
        heads, checks = [], []
        for seed in m.SEEDS:
            started = time.monotonic()
            theta = np.random.default_rng(seed).normal(0, 0.2, 97)
            checks.append(dict(m.gradient_check("gibbs", theta, x[:3], []), seed=seed))
            moment, variance = np.zeros(97), np.zeros(97)
            initial = theta.copy()
            for step in range(1, 201):
                z, jac = m.logits_jacobian("gibbs", theta, x, [])
                gradient = jac.T @ (expit(z) - y) / len(x) + 0.002 * theta
                moment, variance = (
                    0.9 * moment + 0.1 * gradient,
                    0.999 * variance + 0.001 * gradient**2,
                )
                theta -= (
                    0.01
                    * (moment / (1 - 0.9**step))
                    / (np.sqrt(variance / (1 - 0.999**step)) + 1e-8)
                )
                if step % 50 == 0:
                    print(f"[exp7984] fit arm={arm} seed={seed} steps={step}/200", flush=True)
            zt = m.logits_jacobian("gibbs", theta, tx, [])[0]
            losses = [float(np.mean((expit(zt / t) - ty) ** 2)) for t in scalar.TEMPERATURES]
            heads.append(
                dict(
                    arm="gibbs",
                    seed=seed,
                    parameters=theta.tolist(),
                    initial_parameters=initial.tolist(),
                    temperature=scalar.TEMPERATURES[int(np.argmin(losses))],
                    knots=[],
                    tune_brier_choices=losses,
                    optimizer_steps=200,
                    parameter_count=97,
                    duration_s=time.monotonic() - started,
                )
            )
        result["arms"][arm] = dict(
            heads=heads,
            means=means.tolist(),
            scales=scales.tolist(),
            fit_support=fs,
            tune_support=ts,
            gradient_checks=checks,
            training_family_ids=[r["family_id"] for r in fitting],
            tuning_family_ids=[r["family_id"] for r in tuning],
        )
    return result


def probabilities(fitted: Json, arm: str, rows: list[Json]) -> list[Any]:
    """The sealed head reads only numeric inputs; failed inputs stay unavailable."""
    unit = fitted["arms"][arm]
    if not rows or rows[0]["q"] is None or rows[0]["features"] is None:
        return [None] * len(unit["heads"])
    x = (m.inputs(rows) - unit["means"]) / unit["scales"]
    return [float(m.predict(head, x)[0]) for head in unit["heads"]]


def action(p: float | None, cutoff: float) -> str:
    """Policies automate only low expected cost and always escalate missing scores."""
    if p is None or min(5 * p, 1 - p) >= cutoff:
        return "escalate"
    return "accept" if p < 1 / 6 else "reject"


def cost(decision: str, y: int | None) -> float | None:
    """Unknown truth cannot produce a factual cost, including escalation cost."""
    return (
        None
        if y is None
        else 0.25
        if decision == "escalate"
        else float(5 * y if decision == "accept" else 1 - y)
    )


def design(fitted: Json, rows: list[Json], views: list[Json]) -> Json:
    """Original policy-design labels choose automation thresholds for every arm.

    Every arm uses original evidence here, including the donor-trained stress
    control. Thresholds target half automation, then prefer lower design error.
    """
    policies = {}
    for arm in ARMS:
        # A donor-trained head is deployed on original evidence for policy design.
        deploy = "full" if arm == "q_donor" else arm
        rr = m.valid(arm_rows(rows, views, deploy))
        p = np.asarray([np.mean(probabilities(fitted, arm, [r])) for r in rr])
        y = np.asarray([r["y"] for r in rr])
        confidence = np.minimum(5 * p, 1 - p)
        candidates = []
        for cutoff in sorted(set(confidence.tolist())):
            active = confidence < cutoff
            error = (
                float(np.mean(np.where(p[active] < 1 / 6, 0, 1) != y[active]))
                if active.any()
                else 1.0
            )
            candidates.append((abs(int(active.sum()) - max(1, int(0.5 * len(p)))), error, cutoff))
        policies[arm] = dict(
            cutoff=min(candidates)[2] if candidates else 0.0,
            design_pairs=len(rr),
            target_automation=0.5,
            fit_role="original_policy_design",
        )
    return policies


def added_information(support: Json, comparisons: Json) -> bool:
    """Movement is not accuracy; only the preregistered Brier test permits benefit."""
    if not support["passed"]:
        return False
    c = comparisons["q_only_brier"]
    return bool(
        c["gain"] is not None
        and c["gain"] >= 0.01
        and c["interval"][0] > 0
        and c["adjusted_p"] < 0.05
    )


def evaluate(fitted: Json, policies: Json, rows: list[Json], views: list[Json]) -> Json:
    """Compare original truth and report separate label-free intervention movement.

    Seed probabilities are averaged before paired source-cluster resampling.
    Cost averages primitive seed decisions under the frozen design policy.
    Only original-source rows carry human targets or factual loss.
    """
    primitive, metrics, movements, errors = [], {}, [], []
    lookup = {v["family_id"]: v for v in views}
    for arm in ARMS:
        metrics[arm] = {}
        deploy = "full" if arm == "q_donor" else arm
        for row in rows:
            view = lookup[row["family_id"]]
            ps = {}
            for intervention in ("original", "erased", "donor", "duplicate"):
                current = arm_rows([row], [view], deploy, intervention)[0]
                scores = probabilities(fitted, arm, [current])
                ps[intervention] = float(np.mean(scores)) if scores[0] is not None else None
                for seed, p in zip(m.SEEDS, scores, strict=True):
                    original = intervention == "original"
                    y = row["y"] if original else None
                    decision = action(p, policies[arm]["cutoff"])
                    primitive.append(
                        dict(
                            family_id=row["family_id"],
                            source_cluster_id=row["source_cluster_id"],
                            role="evaluation",
                            arm=arm,
                            seed=seed,
                            intervention=intervention,
                            p=p,
                            y=y,
                            decision=decision if original else None,
                            actual_cost=cost(decision, y),
                            target_scope="original_source_only"
                            if original
                            else "mechanism_stress_no_truth",
                            intervention_hash=view["hashes"][intervention],
                            status="completed" if p is not None else "excluded",
                        )
                    )
            if ps["original"] is not None and row["y"] is not None:
                scores = probabilities(fitted, arm, [arm_rows([row], [view], deploy)[0]])
                unit = dict(
                    brier=(ps["original"] - row["y"]) ** 2,
                    cost=float(
                        np.mean(
                            [cost(action(p, policies[arm]["cutoff"]), row["y"]) for p in scores]
                        )
                    ),
                )
                metrics[arm].setdefault(row["source_cluster_id"], []).append(unit)
            if ps["original"] is not None:
                errors.append(abs(ps["original"] - ps["duplicate"]))
            else:
                errors.append(0.0 if ps["duplicate"] is None else 1.0)
            movements.append(
                dict(
                    family_id=row["family_id"],
                    arm=arm,
                    erased_delta=None
                    if ps["original"] is None or ps["erased"] is None
                    else ps["erased"] - ps["original"],
                    donor_delta=None
                    if ps["original"] is None or ps["donor"] is None
                    else ps["donor"] - ps["original"],
                    truth_claim=False,
                )
            )
        print(f"[exp7984] evaluation arm={arm} sources={len(rows)}", flush=True)
    comparisons = {}
    for control in ("q_only", "source_only"):
        for metric in ("brier", "cost"):
            ids = sorted(set(metrics[control]) & set(metrics["full"]))
            diff = np.asarray(
                [
                    np.mean([r[metric] for r in metrics[control][cid]])
                    - np.mean([r[metric] for r in metrics["full"][cid]])
                    for cid in ids
                ]
            )
            rng = np.random.default_rng(69284)
            if len(diff):
                boot = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
                signs = rng.choice([-1, 1], (10000, len(diff)))
                gain, interval = float(diff.mean()), np.quantile(boot, [0.025, 0.975]).tolist()
                p = float((1 + np.sum((diff * signs).mean(axis=1) >= gain)) / 10001)
            else:
                gain, interval, p = None, [None, None], 1.0
            comparisons[control + "_" + metric] = dict(
                gain=gain,
                interval=interval,
                raw_p=p,
                independent=len(ids),
                paired_source_ids=ids,
                paired_differences=diff.tolist(),
                target_scope="original_source_only",
            )
            print(
                f"[exp7984] paired_comparison={control}_{metric} independent={len(ids)}", flush=True
            )
    adjusted = scalar.holm({k: v["raw_p"] for k, v in comparisons.items()})
    for k, v in comparisons.items():
        v["adjusted_p"] = adjusted[k]
    eligible = m.valid(rows)
    support = scalar.support(eligible, 32, 8)
    return dict(
        rows=primitive,
        paired_comparisons=comparisons,
        policies=policies,
        evaluation_support=support,
        added_information_score=int(added_information(support, comparisons)),
        duplicate_parity=dict(
            passed=bool(errors) and max(errors) <= 1e-12,
            max_absolute_error=max(errors, default=0.0),
            tolerance=1e-12,
            pairs=len(errors),
        ),
        source_reliance=movements,
        summary={
            a: {
                k: float(np.mean([r[k] for rr in groups.values() for r in rr])) if groups else None
                for k in ("brier", "cost")
            }
            for a, groups in metrics.items()
        },
    )
