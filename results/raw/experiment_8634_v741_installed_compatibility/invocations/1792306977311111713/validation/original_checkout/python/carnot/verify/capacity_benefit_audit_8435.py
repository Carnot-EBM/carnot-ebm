"""REQ-VERIFY-8435: reduce exposed histories without claiming independent replication.

Sibling answers and random admissions share questions. Averaging them before
resampling keeps the question, rather than a repeated prediction, as the unit.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from carnot.reporting import v726_contract_methods as methods
from carnot.verify import capacity_online_learning_8434 as stream

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8435_v726_capacity_benefit_audit", "exp8435-capacity-benefit-audit"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_capacity_benefit_audit_8435.py"
OWNED = [
    "python/carnot/verify/capacity_benefit_audit_8435.py",
    "python/carnot/reporting/capacity_benefit_report_8435.py",
    "python/carnot/reporting/capacity_benefit_runner_8435.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
ARMS, SEEDS = stream.ARMS, stream.SEEDS
SEED, DRAWS = 7258421, 10000
reference, read, checksum = methods.reference, methods.read, methods.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed counts so a quiet child cannot be confused with finished work."""
    print(f"[exp8435] phase={phase} completed={completed} pending={pending}", flush=True)


def question_contrasts(rows: list[Json], capacity: int) -> list[list[float]]:
    """Require all paired siblings and seeds before reducing a whole question."""
    groups: dict[str, dict[str, list[Json]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["capacity"] == capacity and row["block"] > 0:
            groups[row["question_hash"]][row["arm"]].append(row)
    result = []
    for arms in groups.values():
        if not all(
            len(arms[a]) == 15 and all(r["brier"] is not None for r in arms[a]) for a in ARMS
        ):
            continue
        brier = {a: float(np.mean([r["brier"] for r in arms[a]])) for a in ARMS}
        loss = {a: float(np.mean([r["decision_loss"] for r in arms[a]])) for a in ARMS}
        result.append(
            [
                brier[ARMS[1]] - brier[ARMS[2]],
                brier[ARMS[0]] - brier[ARMS[2]],
                loss[ARMS[2]] - loss[ARMS[0]],
            ]
        )
    return result


def conditional_bootstrap(values: list[list[float]]) -> Json:
    """The fixed draws describe these issued histories; adaptive streams are dependent."""
    if not values:
        return dict(estimate=None, ci95=None, gain=None, gain_ci95=None, decision=None)
    matrix: NDArray[np.float64] = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = []
    for start in range(0, DRAWS, 1000):
        progress("conditional_bootstrap", start, DRAWS - start)
        draws.extend(matrix[rng.integers(len(matrix), size=(1000, len(matrix)))].mean(axis=1))
    intervals = np.quantile(np.asarray(draws), [0.025, 0.975], axis=0)
    means = matrix.mean(axis=0)
    return dict(
        estimate=float(means[0]),
        ci95=intervals[:, 0].tolist(),
        gain=float(means[1]),
        gain_ci95=intervals[:, 1].tolist(),
        decision=float(means[2]),
    )


def retention_bound(rows: list[Json]) -> Json:
    """One resample indexes all windows; each draw's maximum prevents window selection."""
    groups: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["capacity"] == 8 and row["arm"] == ARMS[2]:
            groups[row["question_hash"]][row["boundary"]].append(
                row["brier"] - row["baseline_brier"]
            )
    matrix: NDArray[np.float64] = np.asarray(
        [
            [np.mean(g[b]) for b in range(5)]
            for g in groups.values()
            if all(len(g[b]) == 15 for b in range(5))
        ],
        dtype=float,
    )
    if len(matrix) != 50:
        return dict(
            upper95=None,
            windows=[dict(window=b, estimate=None, upper95=None) for b in range(5)],
            question_count=len(matrix),
        )
    rng = np.random.default_rng(SEED)
    draws = []
    for start in range(0, DRAWS, 1000):
        progress("simultaneous_retention", start, DRAWS - start)
        draws.extend(matrix[rng.integers(50, size=(1000, 50))].mean(axis=1))
    array = np.asarray(draws)
    return dict(
        upper95=float(np.quantile(array.max(axis=1), 0.95)),
        windows=[
            dict(
                window=b,
                estimate=float(matrix[:, b].mean()),
                upper95=float(np.quantile(array[:, b], 0.95)),
            )
            for b in range(5)
        ],
        question_count=50,
    )


def analyze(data: Json) -> Json:
    """Preserve every arm and intended bound before deciding whether benefit is measured."""
    rows = data["rows"]
    primary = conditional_bootstrap(question_contrasts(rows, 8))
    retention = retention_bound(data["retention_rows"])
    controls, bounds = [], []
    for seed in SEEDS:
        for capacity in (8, 16):
            for arm in ARMS:
                for block in range(5):
                    subset = [
                        r
                        for r in rows
                        if (r["seed"], r["capacity"], r["arm"], r["block"])
                        == (seed, capacity, arm, block)
                    ]
                    observed = [r for r in subset if r["brier"] is not None]
                    total = sum(r["brier"] for r in observed)
                    ident = dict(seed=seed, capacity=capacity, arm=arm, block=block)
                    controls.append(
                        dict(
                            ident,
                            intended=60,
                            completed=len(observed),
                            brier=float(np.mean([r["brier"] for r in observed]))
                            if observed
                            else None,
                            decision_loss=float(np.mean([r["decision_loss"] for r in observed]))
                            if observed
                            else None,
                        )
                    )
                    bounds.append(
                        dict(
                            ident,
                            intended=60,
                            missing=60 - len(observed),
                            brier_lower=total / 60,
                            brier_upper=(total + 60 - len(observed)) / 60,
                            decision_loss_lower=sum(r["decision_loss"] for r in observed) / 60,
                            decision_loss_upper=(
                                sum(r["decision_loss"] for r in observed) + 60 - len(observed)
                            )
                            / 60,
                        )
                    )
    inclusion: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    for r in data["admission_rows"]:
        inclusion[(r["capacity"], r["arm"], r["block"], r["question_hash"])].append(r)
    frequencies = [
        dict(
            capacity=k[0],
            arm=k[1],
            block=k[2],
            question_hash=k[3],
            empirical=sum(r["admitted"] for r in group if r["admitted"] is not None) / len(group),
            expected=group[0]["inclusion_probability"],
            seed_count=len(group),
            missing=sum(r["admitted"] is None for r in group),
        )
        for k, group in inclusion.items()
    ]
    vectors = {a: [r["probability"] for r in rows if r["arm"] == a] for a in ARMS}
    shuffled_equal = bool(vectors[ARMS[2]]) and vectors[ARMS[2]] == vectors[ARMS[3]]
    frozen_equal = bool(vectors[ARMS[0]]) and vectors[ARMS[0]] == vectors[ARMS[2]]
    paired = len(question_contrasts(rows, 8))
    paired_values = question_contrasts(rows, 8)
    primary_bounds = dict(
        intended=80,
        observed=paired,
        missing=80 - paired,
        lower=(sum(v[0] for v in paired_values) - (80 - paired)) / 80,
        upper=(sum(v[0] for v in paired_values) + (80 - paired)) / 80,
    )
    support = (
        paired == 80
        and retention["question_count"] == 50
        and len(rows) == 12000
        and all(r["brier"] is not None for r in rows)
    )
    effect = (
        primary["ci95"] is not None
        and primary["ci95"][0] > 0
        and primary["gain_ci95"][0] > 0
        and primary["decision"] <= 0
        and retention["upper95"] is not None
        and retention["upper95"] <= 0.01
    )
    qualified_claim = support and not shuffled_equal and not frozen_equal
    outcome = (
        "unmeasured"
        if primary["estimate"] is None
        else "exploratory_positive"
        if effect and qualified_claim
        else "null"
    )
    return dict(
        fifo_minus_uniform_brier=primary["estimate"],
        conditional_ci95=primary["ci95"],
        gain_vs_frozen=dict(estimate=primary["gain"], conditional_ci95=primary["gain_ci95"]),
        decision_loss_delta=primary["decision"],
        retention_simultaneous_upper95=retention["upper95"],
        per_window_retention=retention["windows"],
        block_seed_capacity_controls=controls,
        missing_data_bounds=bounds,
        primary_missing_bounds=primary_bounds,
        inclusion_frequencies=frequencies,
        control_degeneracy=dict(
            shuffled_matches_uniform=shuffled_equal, uniform_matches_frozen=frozen_equal
        ),
        support_adequate=support,
        affected_claim_qualified=qualified_claim,
        scientific_outcome=outcome,
        exploratory_learning_benefit_score=int(outcome == "exploratory_positive"),
        independent_count=len({r["question_hash"] for r in rows if r["brier"] is not None}),
        later_question_count=paired,
        bootstrap_draws=DRAWS,
        random_seed=SEED,
        secondary_capacity16=conditional_bootstrap(question_contrasts(rows, 16)),
        uncertainty_scope="Conditional descriptive uncertainty for exposed issued histories; adaptive streams are dependent; not a population generalization confidence interval.",
        retention_resampling="Identical question indices across all five windows; maximum increase per draw; one-sided95 percentile.",
        next_evidence_condition="A null retires this exact unchanged criterion/admission attempt. A qualified positive earns only a future independent stream, not deployment or a moat claim. Unqualified evidence requires resolving its named defect before a new scientific attempt.",
    )
