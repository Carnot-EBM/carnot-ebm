"""Scalar response calibration keeps every arm on the same information.

REQ-VERIFY-7972. Human targets are fallible development annotations. Energy
heads change only small coefficients; no generator or text ranker is updated.
"""

from __future__ import annotations

from collections import defaultdict
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

from carnot.verify.qwen_response_risk_7958 import decision

Json = dict[str, Any]
Array = NDArray[np.float64]
ARMS = ("raw_qwen", "platt", "isotonic", "gibbs", "spline")
CONTROLS = ARMS[:3]
SEEDS = (69101, 69102, 69103)
TEMPERATURES = tuple(float(t) for t in np.geomspace(0.25, 4, 17))
CONFIG = dict(
    steps=200,
    learning_rate=0.01,
    l2=1e-3,
    optimizer="Adam",
    seeds=list(SEEDS),
    temperatures=list(TEMPERATURES),
    bootstrap_draws=10000,
    loss=dict(accept="5y", reject="1-y", escalate=0.25),
    primary="gibbs",
    controls=list(CONTROLS),
    target_automation=0.5,
)


def validate_data(data: Json) -> None:
    """Limit predictor data to scalars and reject overlap between source roles."""
    if set(data) != {"fit", "tune", "policy_design", "evaluation"}:
        raise ValueError("role_roster")
    seen: set[str] = set()
    clusters: dict[str, str] = {}
    for role, rows in data.items():
        for r in rows:
            if set(r) != {"family_id", "source_cluster_id", "q", "y", "status"}:
                raise ValueError("scalar_fields")
            if r["family_id"] in seen or clusters.get(r["source_cluster_id"], role) != role:
                raise ValueError("cross_role")
            seen.add(r["family_id"])
            clusters[r["source_cluster_id"]] = role
            if r["q"] is not None and (
                type(r["q"]) not in (int, float) or not np.isfinite(r["q"]) or not 0 <= r["q"] <= 1
            ):
                raise ValueError("probability")
            if r["y"] is not None and (type(r["y"]) is not int or r["y"] not in (0, 1)):
                raise ValueError("label")


def valid(rows: list[Json]) -> list[Json]:
    """Probability measurements need both a valid judgment and a known target."""
    return [r for r in rows if r["q"] is not None and r["y"] is not None]


def support(rows: list[Json], minimum: int, per_class: int) -> Json:
    """Source clusters, rather than fits or repeats, determine support floors."""
    usable = valid(rows)
    n = len({r["source_cluster_id"] for r in usable})
    counts = {str(y): len({r["source_cluster_id"] for r in usable if r["y"] == y}) for y in (0, 1)}
    return dict(
        independent=n,
        class_counts=counts,
        minimum=minimum,
        per_class=per_class,
        passed=n >= minimum and min(counts.values()) >= per_class,
    )


def basis(q: Array) -> Array:
    """True cubic clamped bases expose at most four local coefficients per score."""
    knots = np.array([0.0] * 4 + [0.2, 0.4, 0.6, 0.8] + [1.0] * 4)
    return np.asarray(BSpline.design_matrix(q, knots, 3).toarray(), dtype=np.float64)


def logits_jacobian(arm: str, theta: Array, q: Array) -> tuple[Array, Array]:
    """Exact derivatives let cold CPU fitting avoid loading pretrained models."""
    if arm == "platt":
        clipped = np.clip(q, 1e-4, 1 - 1e-4)
        x = np.column_stack((np.log(clipped / (1 - clipped)), np.ones(len(q))))
        return x @ theta, x
    if arm == "spline":
        x = basis(q)
        return x @ theta, x
    if arm != "gibbs":
        raise ValueError("arm")
    w, b, v = theta[:16].reshape(2, 8), theta[16:24], theta[24:32]
    x0 = np.column_stack((q, np.zeros(len(q))))
    x1 = np.column_stack((q, np.ones(len(q))))
    h0, h1 = np.tanh(x0 @ w + b), np.tanh(x1 @ w + b)
    d0, d1 = (1 - h0 * h0) * v, (1 - h1 * h1) * v
    dw = x0[:, :, None] * d0[:, None, :] - x1[:, :, None] * d1[:, None, :]
    jac = np.column_stack((dw.reshape(len(q), 16), d0 - d1, h0 - h1, np.zeros(len(q))))
    return (h0 - h1) @ v, jac


def predict(head: Json, q: Array) -> Array:
    """Use only q and sealed coefficients for every deployed probability."""
    arm = head["arm"]
    if arm == "raw_qwen" or len(q) == 0:
        return q.copy()
    if arm == "isotonic":
        return np.interp(q, head["x"], head["p"])
    z, _ = logits_jacobian(arm, np.array(head["parameters"], dtype=float), q)
    return np.asarray(expit(z / head["temperature"]), dtype=np.float64)


def bce(p: Array, y: Array) -> float:
    """Clipping prevents logarithms of zero without changing reported scores."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))


def fit(fitting: list[Json], tuning: list[Json]) -> Json:
    """Fit on one role, tune only energy temperatures, and count actual work."""
    start = time.monotonic()
    f, t = valid(fitting), valid(tuning)
    q, y = (np.array([r[k] for r in f], dtype=float) for k in ("q", "y"))
    tq, ty = (np.array([r[k] for r in t], dtype=float) for k in ("q", "y"))
    iso_started = time.monotonic()
    iso = IsotonicRegression(out_of_bounds="clip").fit(q, y)
    iso_duration = time.monotonic() - iso_started
    heads: Json = dict(
        raw_qwen=[dict(arm="raw_qwen", parameter_count=0)],
        isotonic=[
            dict(
                arm="isotonic",
                x=iso.X_thresholds_.tolist(),
                p=iso.y_thresholds_.tolist(),
                parameter_count=len(iso.y_thresholds_) + len(iso.X_thresholds_),
                optimizer_steps=0,
                duration_s=iso_duration,
            )
        ],
    )
    for arm, count in (("platt", 2), ("gibbs", 33), ("spline", 8)):
        heads[arm] = []
        for seed in SEEDS:
            began = time.monotonic()
            theta = np.random.default_rng(seed).normal(0, 0.2, count)
            initial = theta.copy()
            m, v = np.zeros(count), np.zeros(count)
            for step in range(1, 201):
                z, jac = logits_jacobian(arm, theta, q)
                gradient = jac.T @ (expit(z) - y) / len(q) + 2e-3 * theta
                m, v = 0.9 * m + 0.1 * gradient, 0.999 * v + 0.001 * gradient * gradient
                theta -= 0.01 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                if step % 50 == 0:
                    print(
                        f"[exp7972] fit arm={arm} seed={seed} completed_steps={step} "
                        f"elapsed_s={time.monotonic() - start:.3f}",
                        flush=True,
                    )
            fit_duration = time.monotonic() - began
            tune_started = time.monotonic()
            zt, _ = logits_jacobian(arm, theta, tq)
            choices = (1.0,) if arm == "platt" else TEMPERATURES
            losses = [bce(np.asarray(expit(zt / temp)), ty) for temp in choices]
            temperature = choices[int(np.argmin(losses))]
            heads[arm].append(
                dict(
                    arm=arm,
                    seed=seed,
                    parameters=theta.tolist(),
                    initial_parameters=initial.tolist(),
                    temperature=temperature,
                    tune_losses=losses,
                    parameter_count=count,
                    optimizer_steps=200,
                    changed_coefficients=int(np.count_nonzero(theta != initial)),
                    coefficient_touches=count * 200,
                    fit_pairs=len(f),
                    tune_pairs=len(t),
                    initial_bce=bce(np.asarray(expit(logits_jacobian(arm, initial, q)[0])), y),
                    final_bce=bce(np.asarray(expit(logits_jacobian(arm, theta, q)[0])), y),
                    duration_s=time.monotonic() - began,
                    fit_duration_s=fit_duration,
                    tune_duration_s=time.monotonic() - tune_started,
                    local_basis_touches_per_step=int(np.count_nonzero(basis(q)))
                    if arm == "spline"
                    else None,
                )
            )
    return heads


def design(heads: Json, rows: list[Json]) -> Json:
    """Freeze an error-ranked automation policy using policy-design labels only."""
    policies = {}
    usable = valid(rows)
    q = np.array([r["q"] for r in usable], dtype=float)
    y = np.array([r["y"] for r in usable], dtype=float)
    for arm in ARMS:
        p = np.mean([predict(h, q) for h in heads[arm]], axis=0)
        chosen = np.where(p < 1 / 6, 0, 1)
        confidence = np.minimum(5 * p, 1 - p)
        grid = sorted(set(float(v) for v in confidence))
        target = max(1, int(np.floor(0.5 * len(p))))
        candidates = []
        for cutoff in grid:
            active = confidence < cutoff
            n = int(active.sum())
            errors = float(np.mean(chosen[active] != y[active])) if n else 1.0
            candidates.append((abs(n - target), errors, cutoff))
        cutoff = min(candidates)[2] if candidates else 0.0
        policies[arm] = dict(
            cutoff=cutoff,
            target_automation=0.5,
            design_pairs=len(p),
            design_rule="nearest automation then error then threshold; strict cutoff",
        )
    return policies


def holm(values: dict[str, float]) -> dict[str, float]:
    """Correct the six registered tests together, without choosing a winner later."""
    result, previous = {}, 0.0
    for i, (name, p) in enumerate(sorted(values.items(), key=lambda item: item[1])):
        previous = max(previous, min(1.0, (len(values) - i) * p))
        result[name] = previous
    return result


def benefit(summary: Json, comparisons: Json) -> bool:
    """Every standard scalar control must lose on both registered metrics."""
    return bool(
        summary["gibbs"]["automation"] >= 0.2
        and all(
            summary["gibbs"]["false_accepts"] <= summary[a]["false_accepts"]
            and all(
                comparisons[a + "_" + m]["interval"][0] > 0
                and comparisons[a + "_" + m]["adjusted_p"] < 0.05
                and (m != "cost" or comparisons[a + "_" + m]["gain"] >= 0.02)
                for m in ("cost", "brier")
            )
            for a in CONTROLS
        )
    )


def evaluate(heads: Json, policies: Json, rows: list[Json]) -> Json:
    """Reduce primitive decisions, averaging seeds before resampling clusters."""
    started = time.monotonic()
    probabilities, decisions, risk_rows, summary = [], [], [], {}
    costs: Json = {a: defaultdict(list) for a in ARMS}
    briers: Json = {a: defaultdict(list) for a in ARMS}
    for arm in ARMS:
        fa, automated, total = 0.0, 0.0, 0.0
        for r in rows:
            usable = r["q"] is not None and r["y"] is not None
            ps = (
                [float(predict(h, np.array([r["q"]]))[0]) for h in heads[arm]] if usable else [None]
            )
            unit_costs = []
            for seed_index, p in enumerate(ps):
                action = decision(p)
                cost = (
                    0.25
                    if action == "escalate"
                    else (5 * r["y"] if action == "accept" else 1 - r["y"])
                )
                unit_costs.append(cost)
                automated += (action != "escalate") / len(ps)
                fa += (action == "accept" and r["y"] == 1) / len(ps)
                decisions.append(
                    dict(
                        family_id=r["family_id"],
                        source_cluster_id=r["source_cluster_id"],
                        arm=arm,
                        seed_index=seed_index,
                        p=p,
                        y=r["y"],
                        decision=action,
                        actual_cost=cost,
                    )
                )
                if p is not None:
                    probabilities.append(
                        dict(
                            family_id=r["family_id"], arm=arm, seed_index=seed_index, p=p, y=r["y"]
                        )
                    )
            mean_cost = float(np.mean(unit_costs))
            total += mean_cost
            costs[arm][r["source_cluster_id"]].append(mean_cost)
            if usable:
                briers[arm][r["source_cluster_id"]].append(
                    float(np.mean([(p - r["y"]) ** 2 for p in ps]))
                )
        denom = len(rows)
        summary[arm] = dict(
            arm=arm,
            intended_cost=total / denom if denom else None,
            automation=automated / denom if denom else 0.0,
            false_accepts=fa,
            intended_denominator=denom,
            probability_denominator=len(valid(rows)),
            brier=float(np.mean([v for vs in briers[arm].values() for v in vs]))
            if briers[arm]
            else None,
        )
        qrows = valid(rows)
        q = np.array([r["q"] for r in qrows])
        p = np.mean([predict(h, q) for h in heads[arm]], axis=0)
        active = np.minimum(5 * p, 1 - p) < policies[arm]["cutoff"]
        error = np.where(
            p < 1 / 6, np.array([r["y"] for r in qrows]), 1 - np.array([r["y"] for r in qrows])
        )
        risk_rows.append(
            dict(
                arm=arm,
                target_automation=0.5,
                cutoff=policies[arm]["cutoff"],
                automation=float(active.sum() / denom) if denom else 0.0,
                error_rate=float(np.mean(error[active])) if active.any() else None,
                automated_count=int(active.sum()),
                intended_denominator=denom,
                descriptive=True,
            )
        )
    comparisons, raw = {}, {}
    for control in CONTROLS:
        for metric, groups in (("cost", costs), ("brier", briers)):
            ids = sorted(set(groups[control]) & set(groups["gibbs"]))
            diff = np.array(
                [np.mean(groups[control][cid]) - np.mean(groups["gibbs"][cid]) for cid in ids]
            )
            rng = np.random.default_rng(69100)
            if len(diff):
                boot = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
                signs = rng.choice([-1, 1], (10000, len(diff)))
                pv = float((1 + np.sum((diff * signs).mean(axis=1) >= diff.mean())) / 10001)
                ci = np.quantile(boot, [0.025, 0.975]).tolist()
                gain = float(diff.mean())
            else:
                ci, pv, gain = [None, None], 1.0, 0.0
            name = control + "_" + metric
            comparisons[name] = dict(gain=gain, interval=ci, independent=len(ids), raw_p=pv)
            raw[name] = pv
            print(
                f"[exp7972] uncertainty completed_units={len(comparisons)}/6 "
                f"elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    adjusted = holm(raw)
    for name in comparisons:
        comparisons[name]["adjusted_p"] = adjusted[name]
    floors = support(rows, 32, 8)
    win = floors["passed"] and benefit(summary, comparisons)
    return dict(
        rows=list(summary.values()),
        probability_rows=probabilities,
        decision_rows=decisions,
        confidence_intervals={k: v["interval"] for k, v in comparisons.items()},
        raw_p_values=raw,
        adjusted_p_values=adjusted,
        comparisons=comparisons,
        false_accept_counts={a: s["false_accepts"] for a, s in summary.items()},
        risk_coverage_rows=risk_rows,
        evaluation_support=floors,
        qwen_calibration_ready_score=int(floors["passed"]),
        qwen_calibration_benefit_score=int(win),
        honest_verdict="complete_positive_qwen_energy_calibration"
        if win
        else (
            "complete_null_qwen_energy_calibration"
            if floors["passed"]
            else "complete_null_insufficient_evaluation_support"
        ),
        verdict_class="positive" if win else "null",
        sample_size_budget=dict(
            unit="complete_response_slot",
            intended=len(rows),
            eligible=len([r for r in rows if r["y"] is not None]),
            started=len(rows),
            completed=len(valid(rows)),
            failed=sum(r["q"] is None and r["y"] is not None for r in rows),
            censored=sum(r["status"] == "censored" for r in rows),
            excluded=sum(r["y"] is None for r in rows),
            independent=floors["independent"],
            independent_unit="original_source_cluster",
        ),
    )


def positive_control() -> Json:
    """An independent synthetic holdout tests whether calibration gains are detected."""
    data = {
        role: [
            dict(
                family_id=f"synthetic-{role}-{i}",
                source_cluster_id=f"synthetic-{role}-{i}",
                q=0.1 if i % 2 == 0 else 0.9,
                y=1 - i % 2,
                status="completed",
            )
            for i in range(n)
        ]
        for role, n in dict(fit=128, tune=32, policy_design=32, evaluation=64).items()
    }
    heads = fit(data["fit"], data["tune"])
    policies = design(heads, data["policy_design"])
    reduced = evaluate(heads, policies, data["evaluation"])
    delta = reduced["comparisons"]["raw_qwen_brier"]["gain"]
    metrics = {r["arm"]: r for r in reduced["rows"]}
    return dict(
        verdict_class="circular_positive",
        synthetic=True,
        detected=delta > 0.1,
        raw_brier=metrics["raw_qwen"]["brier"],
        calibrated_brier=metrics["gibbs"]["brier"],
        brier_gain=delta,
        independent_of_human_science=True,
        methodology="Disjoint synthetic copies with inverted q; reducer detects Brier distortion.",
        comparisons=reduced["comparisons"],
        heads=heads,
        policies=policies,
        evaluation_rows=data["evaluation"],
    )
