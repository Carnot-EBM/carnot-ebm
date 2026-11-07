"""REQ-REPORT-8819: fixed Gibbs evidence with prospective selective-risk gates.

Only calibration chooses cutoffs. Seeds and evidence views share one world's
uncertainty; abstention never removes a candidate from unconditional Brier.
"""

from collections import defaultdict
from copy import deepcopy
import time
from typing import Any

import numpy as np

from carnot.reporting import consensus_energy_8806 as old
from carnot.reporting import selective_evidence_8816 as evidence
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
__all__ = ["old"]
SEEDS = [7558819 + i for i in range(5)]
MENU = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ACTIONS = ["accept", "reject", "escalate"]


def protocol() -> Json:
    """Register five bounds before fitting so a null cannot change the question."""
    return dict(
        seeds=SEEDS,
        steps=240,
        threshold_menu=MENU,
        confidence=0.99,
        endpoints=5,
        bootstrap_draws=2000,
        bootstrap_seed=7558819,
        split_counts=dict(fit=60, calibration=60, evaluation=240),
        gates=dict(
            risk=("upper", 0.05),
            coverage=("lower", 0.30),
            useful_coverage_gain=("lower", 0.05),
            brier_gain=("lower", 0.01),
            insufficient_to_decision=("upper", 0.05),
        ),
        selection="safe risk upper <=.05 and coverage lower >=.30; maximize correct coverage, then lowest cutoff; none => escalate",
        amendment="five 99-percent endpoints supersede Exp8816 four-endpoint family before any fit",
        intervals="paired whole-world bootstrap after averaging nested seeds; empirical constructed-population intervals",
        random="hash-uniform thinning of unthresholded learned decisive actions; probability frozen from calibration coverage",
        checks="five unchanged visible count features; shared per group, not multiplied by candidates or seeds",
    )


def construct(worlds: list[Json]) -> Json:
    """Keep constructed labels in a separate object so serving consumes only inputs."""
    rows = []
    for i, world in enumerate(sorted(worlds, key=lambda w: w["world_id"])):
        # Six vertices are new; the one-variable evidence representation stays fixed.
        base = evidence.variants(world["edges"], 1)[[0, 2, 4][i % 3]]["input"]
        hop = next(j for j, query in enumerate(base["query"]) if "?x" in query)
        for condition in old.CONDITIONS:
            payload = deepcopy(base)
            predicate = payload["query"][hop]
            if condition != "intact":
                index = next(
                    j for j, fact in enumerate(payload["facts"]) if fact[1] == predicate[1]
                )
                if condition == "missing_hop":
                    payload["facts"].pop(index)
                else:
                    payload["facts"][index][predicate.index("?x")] = "swapped_binding"
            truth = evidence.truth(payload)
            wrong = old.LABELS[(old.LABELS.index(truth) + 1 + i % 2) % 3]
            for agreement in old.AGREEMENTS:
                candidates = (
                    [truth] * 3
                    if agreement == "unanimous_correct"
                    else [wrong] * 3
                    if agreement == "unanimous_wrong"
                    else list(old.LABELS)
                )
                rows.append(
                    dict(
                        unit_id=f"{world['world_id']}/{condition}/{agreement}",
                        world_id=world["world_id"],
                        split=world["split"],
                        input=payload,
                        truth=truth,
                        condition=condition,
                        agreement=agreement,
                        candidates=candidates,
                    )
                )
    return dict(
        inputs=[{k: v for k, v in x.items() if k != "truth"} for x in rows],
        labels={x["unit_id"]: x["truth"] for x in rows},
        worlds=worlds,
    )


def metrics(rows: list[Json]) -> Json:
    """Wrong decisive actions use only decisions; Brier keeps every scheduled row."""
    n = len(rows)
    decisive = sum(x["action"] != "escalate" for x in rows)
    wrong = sum(
        x["action"] != "escalate" and x["action"] != ACTIONS[old.target(x["candidate"], x["truth"])]
        for x in rows
    )
    unknown = sum(x["truth"] == "insufficient" for x in rows)
    bad_unknown = sum(x["truth"] == "insufficient" and x["action"] != "escalate" for x in rows)
    brier = sum(
        sum(
            (p - int(i == old.target(x["candidate"], x["truth"]))) ** 2
            for i, p in enumerate(x["probabilities"])
        )
        for x in rows
    )
    return dict(
        n=n,
        decisive=decisive,
        wrong=wrong,
        unknown=unknown,
        bad_unknown=bad_unknown,
        risk=wrong / decisive if decisive else None,
        coverage=decisive / n if n else 0.0,
        correct_coverage=(decisive - wrong) / n if n else 0.0,
        insufficient_to_decision=bad_unknown / unknown if unknown else None,
        brier=brier / n if n else None,
    )


def distribution(rows: list[Json], draws: int) -> Json:
    """Resample world sufficient statistics, preserving all seeds inside each world."""
    groups: dict[str, list[Json]] = defaultdict(list)
    for row in rows:
        groups[row["cluster_id"]].append(row)
    stats = [metrics(groups[k]) for k in sorted(groups)]
    if not stats:
        return {
            k: [None, None, None]
            for k in ["risk", "coverage", "correct_coverage", "brier", "insufficient_to_decision"]
        }
    # Normalize counts within worlds before drawing; duplicating seeds adds no evidence.
    a = np.asarray(
        [
            [s[k] / s["n"] for k in ["wrong", "decisive", "bad_unknown", "unknown"]] + [s["brier"]]
            for s in stats
        ]
    )
    samples = a[np.random.default_rng(SEEDS[0]).integers(0, len(a), size=(draws, len(a)))].mean(
        axis=1
    )
    center = a.mean(axis=0)
    result: dict[str, list[Any]] = {}
    for key, num, den in [
        ("risk", 0, 1),
        ("coverage", 1, None),
        ("correct_coverage", None, None),
        ("brier", 4, None),
        ("insufficient_to_decision", 2, 3),
    ]:
        values = samples[:, 1] - samples[:, 0] if key == "correct_coverage" else samples[:, num]
        estimate = center[1] - center[0] if key == "correct_coverage" else center[num]
        if den is not None:
            if center[den] == 0 or np.any(samples[:, den] == 0):
                result[key] = [None, None, None]
                continue
            values = values / samples[:, den]
            estimate = estimate / center[den]
        result[key] = [float(estimate), *np.quantile(values, [0.005, 0.995]).tolist()]
    return result


def select(rows: list[Json], arm: str, seed: int, draws: int = 2000) -> Json:
    """Only independent calibration can admit a cutoff; retain every rejected menu row."""
    if any(x["split"] != "calibration" for x in rows):
        raise ValueError("calibration_only")
    local = [x for x in rows if x["arm"] == arm and x["seed"] == seed]
    menu = []
    for threshold in MENU:
        scored = [dict(x, action=old.decide(x["probabilities"], threshold)) for x in local]
        intervals = distribution(scored, draws)
        stats = metrics(scored)
        upper, lower = intervals["risk"][2], intervals["coverage"][1]
        menu.append(
            dict(
                threshold=threshold,
                metrics=stats,
                risk_upper=upper,
                coverage_lower=lower,
                eligible=upper is not None and upper <= 0.05 and lower >= 0.30,
            )
        )
    safe = [x for x in menu if x["eligible"]]
    best = (
        max(safe, key=lambda x: (x["metrics"]["correct_coverage"], -x["threshold"]))
        if safe
        else None
    )
    base = metrics([dict(x, action=old.decide(x["probabilities"], 0.0)) for x in local])["coverage"]
    coverage = best["metrics"]["coverage"] if best else 0.0
    return dict(
        arm=arm,
        seed=seed,
        threshold=best["threshold"] if best else None,
        status="selected" if best else "no_candidate",
        calibration_rows=menu,
        calibration_coverage=coverage,
        random_probability=coverage / base if base else 0.0,
        calibration_input_hash=canonical_hash(local),
    )


def predict(
    inputs: list[Json], parameters: list[Json], selected: list[Json]
) -> tuple[list[Json], list[Json]]:
    """Serving accepts payloads and weights only; random thinning never reads labels."""
    features = [old.signals(x["input"]) for x in inputs]
    predictions = {p["seed"]: old.probabilities(features, p["models"]) for p in parameters}
    choices = {(x["arm"], x["seed"]): x for x in selected}
    rows, checks = [], []
    for i, (unit, counts) in enumerate(zip(inputs, features, strict=True)):
        atom_outcomes = [
            old.upstream.original.compare(
                "local_record", dict(facts=unit["input"]["facts"], query=[query])
            )[0]
            for query in unit["input"]["query"][:5]
        ]
        fixed = [0.1, 0.8, 0.1] if counts[1] else [0.1, 0.1, 0.8] if counts[2] else [0.8, 0.1, 0.1]
        checks.append(
            dict(
                unit_id=unit["unit_id"],
                features=counts,
                paid_atom_checks=5,
                atom_outcomes=atom_outcomes,
                feature_record_visits=len(unit["input"]["query"]) * len(unit["input"]["facts"]),
                input_hash=canonical_hash(unit["input"]),
                scope="five paid single-atom checks shared by all arms; unchanged full-count extraction separately recorded",
            )
        )
        for j, answer in enumerate(unit["candidates"]):
            arms = [
                (
                    "agreement_only",
                    0,
                    [1.0, 0.0, 0.0] if len(set(unit["candidates"])) == 1 else [1 / 3] * 3,
                ),
                ("fixed_evidence", 0, old.action_probabilities(fixed, answer)),
            ]
            arms += [
                (
                    "learned_energy",
                    p["seed"],
                    old.action_probabilities(predictions[p["seed"]][i], answer),
                )
                for p in parameters
            ]
            for arm, seed, probs in arms:
                choice = choices.get((arm, seed), dict(threshold=0.5, random_probability=0.0))
                action = (
                    "escalate"
                    if choice["threshold"] is None
                    else old.decide(probs, choice["threshold"])
                )
                base = dict(
                    unit_id=unit["unit_id"],
                    cluster_id=unit["world_id"],
                    split=unit["split"],
                    candidate_index=j,
                    candidate=answer,
                    agreement=unit["agreement"],
                    condition=unit["condition"],
                    probabilities=probs,
                    arm=arm,
                    seed=seed,
                    action=action,
                )
                rows.append(base)
                if arm == "learned_energy" and selected:
                    u = (
                        int(canonical_hash([seed, unit["unit_id"], j]).split(":")[1][:13], 16)
                        / 16**13
                    )
                    rows.append(
                        dict(
                            base,
                            arm="matched_random",
                            action=old.decide(probs, 0.0)
                            if u < choice["random_probability"]
                            else "escalate",
                        )
                    )
        if (i + 1) % 180 == 0:
            old.progress("8819_predictions", i + 1, len(inputs) - i - 1)
    return rows, checks


def fit(cohort: Json, seeds: list[int] = SEEDS, steps: int = 240) -> Json:
    """Reuse the existing heads and features; fit and calibration have disjoint worlds."""
    train = [
        x for x in cohort["inputs"] if x["split"] == "fit" and x["agreement"] == "disagreement"
    ]
    calibration = [x for x in cohort["inputs"] if x["split"] == "calibration"]
    x = np.asarray([old.signals(row["input"]) for row in train])
    y = np.asarray([old.LABELS.index(cohort["labels"][row["unit_id"]]) for row in train])
    parameters: list[Json] = []
    controls: list[Json] = []
    costs: list[Json] = []
    for i, seed in enumerate(seeds):
        old.progress("8819_before_model_initialization_and_training", i, len(seeds) - i)
        start = time.monotonic()
        models = old.train(x, y, seed, steps)
        parameters.append(dict(seed=seed, models=models))
        costs.append(
            dict(
                arm="learned_energy",
                seed=seed,
                duration_s=time.monotonic() - start,
                gradient_steps=steps * 3,
            )
        )
        old.progress("8819_after_model_initialization_and_training", i + 1, len(seeds) - i - 1)
    cx = [old.signals(row["input"]) for row in calibration]
    cy = np.eye(3)[[old.LABELS.index(cohort["labels"][row["unit_id"]]) for row in calibration]]
    full = float(
        np.mean(np.sum((old.probabilities(cx, parameters[0]["models"]) - cy) ** 2, axis=1))
    )
    for name in ["permuted_released_labels", "all_features_ablated"]:
        old.progress("8819_before_control_training_" + name, 0, 1)
        xx = np.zeros_like(x) if name == "all_features_ablated" else x
        yy = (
            np.random.default_rng(seeds[0]).permutation(y)
            if name == "permuted_released_labels"
            else y
        )
        models = old.train(xx, yy, seeds[0], steps)
        cp = old.probabilities(
            np.zeros_like(cx).tolist() if name == "all_features_ablated" else cx, models
        )
        brier = float(np.mean(np.sum((cp - cy) ** 2, axis=1)))
        controls.append(
            dict(
                control=name,
                models=models,
                full_brier=full,
                brier=brier,
                degradation=brier - full,
                passed=brier - full >= 0.01,
                split="calibration",
            )
        )
        old.progress("8819_after_control_training_" + name, 1, 0)
    predictions, _ = predict(calibration, parameters, [])
    scored = [dict(row, truth=cohort["labels"][row["unit_id"]]) for row in predictions]
    keys = sorted({(r["arm"], r["seed"]) for r in scored})
    selected = [select(scored, arm, seed) for arm, seed in keys]
    return dict(
        parameters=parameters,
        selected_threshold_rows=selected,
        shortcut_controls=controls,
        train_worlds=sorted({r["world_id"] for r in train}),
        calibration_worlds=sorted({r["world_id"] for r in calibration}),
        calibration_predictions=scored,
        cost_rows=costs,
        numerical_training_calls=3 * len(seeds) + 6,
    )


def release(
    cohort: Json, predictions: list[Json], checks: list[Json], seal: str
) -> tuple[list[Json], Json]:
    """Open scoring labels only after complete probabilities, actions and checks seal."""
    inputs = [x for x in cohort["inputs"] if x["split"] == "evaluation"]
    expected = {x["unit_id"] for x in inputs}
    identities = {(x["unit_id"], x["candidate_index"], x["arm"], x["seed"]) for x in predictions}
    arms = {(x["arm"], x["seed"]) for x in predictions}
    if (
        canonical_hash(dict(rows=predictions, checks=checks)) != seal
        or len(identities) != len(predictions)
        or len(predictions) != len(expected) * 3 * len(arms)
        or {x["unit_id"] for x in predictions} != expected
        or {x["unit_id"] for x in checks} != expected
        or any("truth" in x or x["action"] not in ACTIONS for x in predictions)
    ):
        raise ValueError("prediction_seal")
    start = time.monotonic()
    oracles = {x["unit_id"]: old.upstream.old.strong(x["input"]) for x in inputs}
    oracle_s = time.monotonic() - start
    rows = [dict(x, truth=cohort["labels"][x["unit_id"]]) for x in predictions]
    for unit in inputs:
        for j, answer in enumerate(unit["candidates"]):
            target = old.target(answer, oracles[unit["unit_id"]])
            rows.append(
                dict(
                    unit_id=unit["unit_id"],
                    cluster_id=unit["world_id"],
                    split="evaluation",
                    candidate_index=j,
                    candidate=answer,
                    truth=cohort["labels"][unit["unit_id"]],
                    agreement=unit["agreement"],
                    condition=unit["condition"],
                    arm="exact_oracle",
                    seed=0,
                    action=ACTIONS[target],
                    probabilities=np.eye(3)[target].tolist(),
                )
            )
    for row in rows:
        target = old.target(row["candidate"], row["truth"])
        row.update(
            metric=sum((p - int(i == target)) ** 2 for i, p in enumerate(row["probabilities"])),
            denominator=1,
            status="completed",
            missing_reason=None,
        )
    return rows, dict(
        arm="exact_oracle",
        duration_s=oracle_s,
        exact_solver_calls=len(inputs),
        verdict_class="circular_positive",
        scope="separate exact reference; excluded from policy benefit",
    )


def summarize(rows: list[Json], selected: list[Json], draws: int = 2000) -> Json:
    """Five conjunctive gates describe learned policy benefit, never oracle correctness."""
    groups: dict[tuple[str, int], list[Json]] = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["seed"])].append(row)
    stats = [dict(arm=a, seed=s, **metrics(group)) for (a, s), group in sorted(groups.items())]
    learned = [x for x in rows if x["arm"] == "learned_energy"]
    worlds = sorted({x["cluster_id"] for x in learned})
    intervals = distribution(learned, draws)
    bounds = {
        k: dict(
            estimate=intervals[k][0],
            lower=intervals[k][1],
            upper=intervals[k][2],
            confidence=0.99,
            status="estimable" if intervals[k][0] is not None else "undefined_denominator",
        )
        for k in ["risk", "coverage", "insufficient_to_decision"]
    }
    fixed_ok = any(x["arm"] == "fixed_evidence" and x["threshold"] is not None for x in selected)
    by_world: dict[tuple[str, str], list[Json]] = defaultdict(list)
    for row in rows:
        by_world[(row["arm"], row["cluster_id"])].append(row)
    for key, metric, sign in [
        ("useful_coverage_gain", "correct_coverage", 1),
        ("brier_gain", "brier", -1),
    ]:
        estimate: Any
        values = []
        if fixed_ok:
            values = [
                sign
                * (
                    metrics(by_world[("learned_energy", w)])[metric]
                    - metrics(by_world[("fixed_evidence", w)])[metric]
                )
                for w in worlds
            ]
        if values:
            a = np.asarray(values)
            sample = a[
                np.random.default_rng(SEEDS[0]).integers(0, len(a), size=(draws, len(a)))
            ].mean(axis=1)
            estimate = float(a.mean())
            lower, upper = np.quantile(sample, [0.005, 0.995]).tolist()
        else:
            estimate = lower = upper = None
        bounds[key] = dict(
            estimate=estimate,
            lower=lower,
            upper=upper,
            confidence=0.99,
            status="estimable" if values else "not_estimable",
        )
    gates = {
        k: bounds[k][direction] is not None
        and (
            bounds[k][direction] <= limit if direction == "upper" else bounds[k][direction] >= limit
        )
        for k, (direction, limit) in protocol()["gates"].items()
    }
    wrong = [x for x in rows if x["agreement"] == "unanimous_wrong"]
    return dict(
        simultaneous_intervals=bounds,
        acceptance_gates=gates,
        benefit_passed=all(gates.values()),
        independent_count=len(worlds),
        risk_numerators_denominators=stats,
        coverage_rows=[
            {
                k: s[k]
                for k in [
                    "arm",
                    "seed",
                    "n",
                    "coverage",
                    "correct_coverage",
                    "insufficient_to_decision",
                ]
            }
            for s in stats
        ],
        brier_rows=[{k: s[k] for k in ["arm", "seed", "n", "brier"]} for s in stats],
        matched_random_rows=[s for s in stats if s["arm"] == "matched_random"],
        unanimous_wrong_false_acceptance=[
            dict(
                arm=a,
                seed=s,
                numerator=sum(
                    x["action"] == "accept" and x["candidate"] != x["truth"] for x in group
                ),
                denominator=len(group),
            )
            for (a, s), group in sorted(_groups(wrong).items())
        ],
    )


def _groups(rows: list[Json]) -> dict[tuple[str, int], list[Json]]:
    """Group wrong-agreement rows without treating their repeated seeds as worlds."""
    result: dict[tuple[str, int], list[Json]] = defaultdict(list)
    for row in rows:
        result[(row["arm"], row["seed"])].append(row)
    return result
