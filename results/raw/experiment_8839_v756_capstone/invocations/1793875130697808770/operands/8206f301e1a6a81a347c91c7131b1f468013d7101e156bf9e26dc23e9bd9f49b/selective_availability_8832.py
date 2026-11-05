"""REQ-REPORT-8832: distinguish finite support from learned discrimination.

The inherited topology corpus is not IID. Binomial bounds therefore diagnose
what support would permit under IID; they never issue deployment certificates.
"""

from collections import defaultdict
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.stats import beta

from carnot.reporting import consensus_selective_energy_8819 as legacy
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
__all__ = ["legacy"]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8832_v756_selective_availability", "exp8832-selective-availability"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/selective_availability{s}_8832.py" for s in ["", "_runner"]] + [
    CLI
]
TESTS = ["tests/python/test_selective_availability_8832.py"]
UPSTREAM = ROOT / "results/experiment_8829_v756_selective_methods.json"
SEEDS, STEPS, BUDGETS = list(legacy.SEEDS), 240, [60, 180, 360]
# Risk and coverage for 7 policies, 3 budgets and 6 cutoffs share one .01 family.
ALPHA = 0.01 / (7 * 3 * 6 * 2)
FLOOR = math.ceil(math.log(ALPHA) / math.log(0.95))


def upper(n: int, errors: int, alpha: float = ALPHA) -> float:
    """Even zero observed errors need independent accepted worlds to bound risk."""
    if not 0 <= errors <= n:
        raise ValueError("binomial_counts")
    if not n or errors == n:
        return 1.0
    return float(
        -math.expm1(math.log(alpha) / n) if not errors else beta.isf(alpha, errors + 1, n - errors)
    )


def lower(n: int, successes: int, alpha: float = ALPHA) -> float:
    """Coverage uncertainty uses the scheduled world count, including abstentions."""
    if not 0 <= successes <= n:
        raise ValueError("binomial_counts")
    return float(beta.ppf(alpha, successes, n - successes + 1)) if successes else 0.0


def prepare(cohort: Json) -> list[Json]:
    """Hash-fixed candidate lists avoid using truth to manufacture agreement."""
    result = []
    for unit in cohort["inputs"]:
        if unit["split"] == "canary":
            continue
        index = int(canonical_hash(unit["input"]).split(":")[1][:8], 16) % 4
        candidates = [legacy.old.LABELS[index]] * 3 if index < 3 else list(legacy.old.LABELS)
        result.append(
            dict(
                unit,
                world_id=unit["cluster_id"],
                candidates=candidates,
                agreement="unanimous" if index < 3 else "disagreement",
            )
        )
    return result


def fit(
    inputs: list[Json], labels: Json, seeds: list[int] | None = None, steps: int = STEPS
) -> Json:
    """Keep the existing NCE updates; retain actual post-fit gradients before calibration."""
    import jax
    import jax.numpy as jnp
    from carnot.training.nce import nce_loss

    train = [u for u in inputs if u["split"] == "fit"]
    x = np.asarray([legacy.old.signals(u["input"]) for u in train])
    y = np.asarray([legacy.old.LABELS.index(labels[u["unit_id"]]) for u in train])
    scaled = jnp.asarray(x, dtype=jnp.float32) / jnp.array([16.0, 2.0, 2.0, 1.0, 1.0])
    parameters: list[Json] = []
    gradients: list[Json] = []
    for seed in SEEDS if seeds is None else seeds:
        legacy.old.progress(
            "8832_before_model_load_and_training", len(parameters), 5 - len(parameters)
        )
        models = legacy.old.train(x, y, seed, steps)
        parameters.append(dict(seed=seed, models=models))
        for label, model in enumerate(models):
            params = (
                [(jnp.asarray(w), jnp.asarray(b)) for w, b in model["layers"]],
                jnp.asarray(model["output_weight"]),
                jnp.asarray(model["output_bias"]),
            )

            def loss(p: Any) -> Any:
                def energy(z: Any) -> Any:
                    for w, b in p[0]:
                        z = jax.nn.silu(w @ z + b)
                    return p[1] @ z + p[2]

                return nce_loss(energy, scaled[y == label], scaled[y != label])

            value, gradient = jax.value_and_grad(loss)(params)
            operand = jax.tree.map(lambda v: v.tolist(), gradient)
            gradients.append(
                dict(
                    seed=seed,
                    label=label,
                    loss=float(value),
                    gradient=operand,
                    gradient_hash=canonical_hash(operand),
                    scope="post-fit NCE gradient; not an additional optimizer update",
                )
            )
        legacy.old.progress(
            "8832_after_model_load_and_training", len(parameters), 5 - len(parameters)
        )
    return dict(
        parameters=parameters,
        gradients=gradients,
        training_rows=len(train),
        training_worlds=len({u["world_id"] for u in train}),
    )


def score(predictions: list[Json], cohort: Json) -> list[Json]:
    """Labels reach scoring only after prediction bytes and selection are sealed."""
    rows = []
    for row in predictions:
        chosen = cohort["certification_units"][row["cluster_id"]]
        candidate = int(canonical_hash(["8832/candidate", chosen]).split(":")[1][:8], 16) % 3
        rows.append(
            dict(
                row,
                truth=cohort["labels"][row["unit_id"]],
                certification_unit=row["unit_id"] == chosen and row["candidate_index"] == candidate,
            )
        )
    audit(rows, cohort)
    return rows


def audit(rows: list[Json], cohort: Json) -> None:
    """A candidate, variant or seed cannot be relabelled as another independent world."""
    units = {u["unit_id"]: u for u in cohort["inputs"]}
    identities = [(r["unit_id"], r["candidate_index"], r["arm"], r["seed"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicated_candidates")
    for row in rows:
        unit = units[row["unit_id"]]
        if (
            row["cluster_id"] != unit["cluster_id"]
            or row["split"] != unit["split"]
            or row["truth"] != cohort["labels"][row["unit_id"]]
        ):
            raise ValueError("cluster_or_label_drift")


def select(rows: list[Json], budget: int, arm: str, seed: int) -> Json:
    """A fixed candidate per world enters each cutoff; every failure keeps its counts."""
    if any(r["split"] != "calibration" for r in rows):
        raise ValueError("calibration_only")
    sample = sorted(
        [r for r in rows if r["arm"] == arm and r["seed"] == seed and r["certification_unit"]],
        key=lambda r: r["cluster_id"],
    )[:budget]
    if len(sample) != len({r["cluster_id"] for r in sample}):
        raise ValueError("replicated_world")
    menu = []
    for cutoff in legacy.MENU:
        stats = legacy.metrics(
            [dict(r, action=legacy.old.decide(r["probabilities"], cutoff)) for r in sample]
        )
        n, e = stats["decisive"], stats["wrong"]
        hi, lo = upper(n, e), lower(len(sample), n)
        reasons = []
        if n < FLOOR:
            reasons.append("insufficient_support")
        if e and hi > 0.05:
            reasons.append("discriminative_errors")
        if lo < 0.30:
            reasons.append("coverage_lower_below_.30")
        menu.append(
            dict(
                arm=arm,
                seed=seed,
                budget=budget,
                cutoff=cutoff,
                accepted=n,
                errors=e,
                denominator=len(sample),
                risk=stats["risk"],
                risk_upper=hi,
                zero_error_upper=upper(n, 0),
                coverage=stats["coverage"],
                coverage_lower=lo,
                eligible=n > 0 and hi <= 0.05 and lo >= 0.30,
                certificate_issued=False,
                missing_reason=reasons,
                correct_coverage=stats["correct_coverage"],
            )
        )
    safe = [r for r in menu if r["eligible"]]
    best = max(safe, key=lambda r: (r["correct_coverage"], -r["cutoff"])) if safe else None
    return dict(
        arm=arm,
        seed=seed,
        budget=budget,
        threshold=best["cutoff"] if best else None,
        cutoffs=menu,
        calibration_input_hash=canonical_hash(sample),
        frozen_before_evaluation=True,
    )


def reduce(rows: list[Json], draws: int = 2000) -> Json:
    """Freeze calibration rules and report held-out utility without multiplying worlds."""
    calibration = [r for r in rows if r["split"] == "calibration"]
    evaluation = [r for r in rows if r["split"] == "evaluation"]
    keys = sorted({(r["arm"], r["seed"]) for r in calibration})
    selected = [select(calibration, budget, arm, seed) for budget in BUDGETS for arm, seed in keys]
    bootstrap = []
    coverage = []
    served: list[Json] = []
    siblings = []
    decomposition = []
    last = time.monotonic()
    for choice in selected:
        arm, seed, budget = choice["arm"], choice["seed"], choice["budget"]
        worlds = sorted(
            {r["cluster_id"] for r in calibration if r["arm"] == arm and r["seed"] == seed}
        )[:budget]
        control = legacy.select(
            [r for r in calibration if r["cluster_id"] in worlds], arm, seed, draws
        )
        bootstrap.append(
            dict(
                control,
                budget=budget,
                interpretation="descriptive old whole-world bootstrap; no finite-sample certificate",
            )
        )
        local = [r for r in evaluation if r["arm"] == arm and r["seed"] == seed]
        scored = [
            dict(
                r,
                action="escalate"
                if choice["threshold"] is None
                else legacy.old.decide(r["probabilities"], choice["threshold"]),
            )
            for r in local
        ]
        variants = [("exact_binomial_rule", scored)]
        if arm == "learned_energy":
            all_cal = [
                r
                for r in calibration
                if r["arm"] == arm
                and r["seed"] == seed
                and r["cluster_id"] in worlds
                and r["certification_unit"]
            ]
            base = legacy.metrics(
                [dict(r, action=legacy.old.decide(r["probabilities"], 0.0)) for r in all_cal]
            )["coverage"]
            chosen = next(
                (c["coverage"] for c in choice["cutoffs"] if c["cutoff"] == choice["threshold"]),
                0.0,
            )
            probability = chosen / base if base else 0.0
            random_rows = [
                dict(
                    r,
                    arm="matched_random",
                    action=legacy.old.decide(r["probabilities"], 0.0)
                    if int(
                        canonical_hash([seed, r["unit_id"], r["candidate_index"]]).split(":")[1][
                            :13
                        ],
                        16,
                    )
                    / 16**13
                    < probability
                    else "escalate",
                )
                for r in local
            ]
            variants.append(("matched_random", random_rows))
        for method, scored_rows in variants:
            sample = [r for r in scored_rows if r["certification_unit"]]
            stats = legacy.metrics(sample)
            record = dict(
                arm=arm if method == "exact_binomial_rule" else method,
                seed=seed,
                budget=budget,
                method=method,
                denominator=stats["n"],
                accepted=stats["decisive"],
                errors=stats["wrong"],
                risk=stats["risk"],
                risk_upper=upper(stats["decisive"], stats["wrong"], 0.01 / 120),
                coverage=stats["coverage"],
                coverage_lower=lower(stats["n"], stats["decisive"], 0.01 / 120),
                useful_coverage=stats["correct_coverage"],
                useful_lower=lower(stats["n"], stats["decisive"] - stats["wrong"], 0.01 / 120),
                useful_upper=upper(
                    stats["n"], stats["n"] - stats["decisive"] + stats["wrong"], 0.01 / 120
                ),
                brier=stats["brier"],
            )
            # Upper useful coverage is the upper confidence bound on correct decisions.
            record["useful_upper"] = upper(
                stats["n"], stats["decisive"] - stats["wrong"], 0.01 / 120
            )
            coverage.append(record)
            served.extend(
                dict(
                    r,
                    budget=budget,
                    metric=int(
                        r["action"] != "escalate"
                        and r["action"]
                        == legacy.ACTIONS[legacy.old.target(r["candidate"], r["truth"])]
                    ),
                    denominator=1,
                    status="completed",
                    missing_reason=None,
                )
                for r in sample
            )
            siblings.append(
                dict(
                    arm=record["arm"],
                    seed=seed,
                    budget=budget,
                    metrics=legacy.metrics(scored_rows),
                    world_clustered=legacy.distribution(scored_rows, draws),
                    independent_count=len({r["cluster_id"] for r in scored_rows}),
                    interpretation="descriptive siblings; seeds and candidates are not independent",
                )
            )
        support = any(c["accepted"] < FLOOR for c in choice["cutoffs"])
        errors = any(c["errors"] and c["risk_upper"] > 0.05 for c in choice["cutoffs"])
        decomposition.append(
            dict(
                arm=arm,
                seed=seed,
                budget=budget,
                bottleneck="both"
                if support and errors
                else "insufficient_support"
                if support
                else "discriminative_errors"
                if errors
                else "neither",
                basis="registered cutoff rows; coverage failures retained separately",
            )
        )
        if time.monotonic() - last >= 30:
            legacy.old.progress("8832_reduce", len(bootstrap), len(selected) - len(bootstrap))
            last = time.monotonic()
    utility = []
    for learned in [r for r in coverage if r["arm"] == "learned_energy"]:
        bases = [
            r
            for r in coverage
            if r["arm"] in ["agreement_only", "fixed_evidence"] and r["budget"] == learned["budget"]
        ]
        strongest = max(bases, key=lambda r: r["useful_coverage"]) if bases else None
        safe = [
            r
            for r in bases
            if any(
                s["arm"] == r["arm"] and s["budget"] == r["budget"] and s["threshold"] is not None
                for s in selected
            )
        ]
        gain = learned["useful_lower"] - max(r["useful_upper"] for r in safe) if safe else None
        utility.append(
            dict(
                seed=learned["seed"],
                budget=learned["budget"],
                strongest_fixed_baseline=strongest["arm"] if strongest else None,
                useful_gain_lower=gain,
                availability_difference=learned["coverage"]
                - (strongest["coverage"] if strongest else 0.0),
                risk_difference=None,
                reason="baseline_no_safe_rule" if not safe else "IID_assumption_not_met",
                numerical_gates_passed=learned["risk_upper"] <= 0.05
                and learned["coverage_lower"] >= 0.30
                and gain is not None
                and gain >= 0.05,
                benefit=False,
            )
        )
    certificates = [c for s in selected for c in s["cutoffs"]]
    return dict(
        selected_threshold_rows=selected,
        certificate_rows=certificates,
        bootstrap_control_rows=bootstrap,
        coverage_rows=coverage,
        rows=served,
        sibling_summaries=siblings,
        failure_decomposition=decomposition,
        utility_comparisons=utility,
        support_floor_rows=[
            dict(
                budget=b,
                zero_error_accepted_floor=FLOOR,
                alpha_per_bound=ALPHA,
                maximum_accepted=b,
                feasible=b >= FLOOR,
            )
            for b in BUDGETS
        ],
        risk_numerators_denominators=[
            dict(
                arm=r["arm"],
                seed=r["seed"],
                budget=r["budget"],
                numerator=r["errors"],
                denominator=r["accepted"],
                scheduled=r["denominator"],
            )
            for r in coverage
        ],
        certificate_availability=[
            dict(
                arm=s["arm"],
                seed=s["seed"],
                budget=s["budget"],
                numerically_eligible=s["threshold"] is not None,
                certificate_issued=False,
            )
            for s in selected
        ],
        iid_assumptions=dict(
            iid_world_sampling=False,
            certificate_issued=False,
            reason="inherited rejection of duplicate and exposed topologies; nested budgets and shared-world seeds are dependent",
            scope="constructed fixed-world empirical diagnostic; no drift or best-of-n guarantee",
        ),
        selective_benefit_score=0,
    )
