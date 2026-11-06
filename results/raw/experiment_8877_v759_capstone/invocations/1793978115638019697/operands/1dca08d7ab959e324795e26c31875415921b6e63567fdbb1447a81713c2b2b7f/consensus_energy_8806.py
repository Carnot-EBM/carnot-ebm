"""REQ-REPORT-8806: learn evidence compatibility on constructed relational worlds.

Counts are necessary local checks, not a general proof procedure. Single-variable
constructed worlds make these signals unusually informative; no live-model or
independent-domain claim follows from this experiment.
"""

from collections import Counter
from copy import deepcopy
import hashlib
import time
from typing import Any

import numpy as np

from carnot.reporting import repaired_evidence_8803 as upstream
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
__all__ = ["upstream"]
LABELS = ("support", "refute", "insufficient")
AGREEMENTS = ("unanimous_correct", "unanimous_wrong", "disagreement")
CONDITIONS = ("intact", "missing_hop", "binding_swap")
SEEDS = (7548806, 7548807, 7548808, 7548809, 7548810)
STEPS = 240


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work counts so a parent can distinguish progress from silence."""
    print(f"[exp8806] phase={phase} completed={completed} pending={pending}", flush=True)


def partition(worlds: list[Json]) -> tuple[list[str], list[str]]:
    """Choose training and calibration worlds before opening their released labels."""
    names = sorted(
        (w["world_id"] for w in worlds if w["split"] == "consensus_dev"),
        key=lambda x: hashlib.sha256(("8806/split/" + x).encode()).hexdigest(),
    )
    boundary = len(names) * 2 // 3
    return names[:boundary], names[boundary:]


def construct(worlds: list[Json]) -> list[Json]:
    """Cross evidence and agreement on the sealed structures, keeping truth private."""
    rows = []
    for i, world in enumerate(sorted(worlds, key=lambda w: w["world_id"])):
        base = upstream.variants(world["edges"])[[0, 8, 7][i % 3]]["input"]
        hop = next(j for j, q in enumerate(base["query"]) if "?x" in q)
        for condition in CONDITIONS:
            payload = deepcopy(base)
            predicate = payload["query"][hop]
            if condition != "intact":
                index = next(j for j, f in enumerate(payload["facts"]) if f[1] == predicate[1])
                if condition == "missing_hop":
                    payload["facts"].pop(index)
                else:
                    payload["facts"][index][predicate.index("?x")] = "swapped_binding"
            truth = upstream.old.strong(payload)
            wrong = LABELS[(LABELS.index(truth) + 1 + i % 2) % 3]
            for agreement in AGREEMENTS:
                candidates = (
                    [truth] * 3
                    if agreement == "unanimous_correct"
                    else [wrong] * 3
                    if agreement == "unanimous_wrong"
                    else list(LABELS)
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
        if (i + 1) % 20 == 0:
            progress("construct_worlds", i + 1, len(worlds) - i - 1)
    return rows


def signals(payload: Json) -> list[float]:
    """Count paid atom checks and necessary binding consistency without exact solving.

    A variable's positive domains must intersect. This necessary condition is
    weaker than a full relational join when a query has several variables.
    """
    upstream.original.parse(payload)
    supported = conflicted = missing = 0
    domains: dict[str, set[str]] = {}
    for query in payload["query"]:
        positive: list[Json] = []
        negative = False
        for fact in payload["facts"]:
            binding: Json = {}
            matches = True
            for term, value in zip(query, fact[:3], strict=True):
                matches &= (
                    binding.setdefault(term, value) == value
                    if term.startswith("?")
                    else term == value
                )
            if matches:
                if fact[3]:
                    positive.append(binding)
                else:
                    negative = True
        supported += bool(positive)
        conflicted += negative
        missing += not positive and not negative
        for variable in {t for t in query if t.startswith("?")}:
            values = {b[variable] for b in positive}
            if positive:
                domains[variable] = domains.get(variable, values) & values
    conflicted += sum(not values for values in domains.values())
    return [float(supported), float(conflicted), float(missing), 0.0, 1.0]


def target(answer: str, truth: str) -> int:
    """Insufficient evidence requires escalation, including an 'insufficient' candidate."""
    return 2 if truth == "insufficient" else int(answer != truth)


def decide(probabilities: list[float], threshold: float) -> str:
    """Commit only when accept or reject reaches a calibration-selected confidence."""
    index = int(np.argmax(probabilities))
    return (
        ("accept", "reject", "escalate")[index] if probabilities[index] >= threshold else "escalate"
    )


def probabilities(features: list[list[float]], models: list[Json]) -> Any:
    """Replay Gibbs weights in NumPy, with no training or oracle at serving time."""
    outputs = []
    for model in models:
        x = np.asarray(features, dtype=np.float64) / np.array([16.0, 2.0, 2.0, 1.0, 1.0])
        for weights, bias in model["layers"]:
            h = x @ np.asarray(weights).T + bias
            x = h / (1 + np.exp(-np.clip(h, -60, 60)))
        energy = x @ model["output_weight"] + model["output_bias"]
        outputs.append(1 / (1 + np.exp(np.clip(energy, -60, 60))))
    result = np.stack(outputs, axis=1)
    return result / result.sum(axis=1, keepdims=True)


def train(features: Any, labels: Any, seed: int, steps: int) -> list[Json]:
    """Use the established class-balanced NCE gradient pipeline for three Gibbs heads."""
    import jax
    import jax.numpy as jnp
    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    x = jnp.asarray(features, dtype=jnp.float32) / jnp.array([16.0, 2.0, 2.0, 1.0, 1.0])
    result = []
    for label in range(3):
        model = GibbsModel(
            GibbsConfig(input_dim=5, hidden_dims=[4]), key=jax.random.PRNGKey(seed + label)
        )
        data, noise = x[labels == label], x[labels != label]

        def loss(parameters: Any) -> Any:
            layers, out_w, out_b = parameters

            def energy(z: Any) -> Any:
                for w, b in layers:
                    z = jax.nn.silu(w @ z + b)
                return out_w @ z + out_b

            return nce_loss(energy, data, noise)

        gradient = jax.jit(jax.value_and_grad(loss))
        parameters = (model.layers, model.output_weight, jnp.asarray(model.output_bias))
        initial = float(loss(parameters))
        for epoch in range(steps):
            value, gradients = gradient(parameters)
            parameters = jax.tree.map(lambda p, g: p - 0.12 * g, parameters, gradients)
        final = float(loss(parameters))
        layers, out_w, out_b = parameters
        result.append(
            dict(
                layers=[(w.tolist(), b.tolist()) for w, b in layers],
                output_weight=out_w.tolist(),
                output_bias=float(out_b),
                initial_nce_loss=initial,
                final_nce_loss=final,
                gradient_steps=steps,
            )
        )
        progress(f"trained_seed_{seed}", label + 1, 2 - label)
    return result


def fit(cohort: list[Json], seeds: list[int], steps: int = STEPS) -> Json:
    """Train on 40 worlds and select confidence thresholds on the other 20 only."""
    unique = {r["world_id"]: r for r in cohort}
    training, calibration = partition(list(unique.values()))
    train_rows = [
        r for r in cohort if r["world_id"] in training and r["agreement"] == "disagreement"
    ]
    cal_rows = [r for r in cohort if r["world_id"] in calibration]
    x = np.asarray([signals(r["input"]) for r in train_rows])
    y = np.asarray([LABELS.index(r["truth"]) for r in train_rows])
    cx = [signals(r["input"]) for r in cal_rows]
    cy = np.eye(3)[[LABELS.index(r["truth"]) for r in cal_rows]]
    parameters, costs, controls = [], [], []
    for i, seed in enumerate(seeds):
        progress("before_numerical_training", i, len(seeds) - i)
        started = time.monotonic()
        models = train(x, y, seed, steps)
        training_s = time.monotonic() - started
        started = time.monotonic()
        predicted = probabilities(cx, models)
        best = (float("inf"), 0.5)
        for threshold in (0.34, 0.5, 0.6, 0.7, 0.8, 0.9):
            errors = []
            for row, probs in zip(cal_rows, predicted, strict=True):
                for answer in row["candidates"]:
                    action_probs = action_probabilities(probs, answer)
                    action = decide(action_probs, threshold)
                    errors.append(
                        int(
                            action != ("accept", "reject", "escalate")[target(answer, row["truth"])]
                        )
                        + 0.05 * (action == "escalate")
                    )
            best = min(best, (float(np.mean(errors)), threshold))
        parameters.append(
            dict(
                seed=seed,
                models=models,
                threshold=best[1],
                calibration_objective=best[0],
                calibration_worlds=calibration,
            )
        )
        costs.append(
            dict(
                arm="learned_energy",
                seed=seed,
                training_s=training_s,
                calibration_s=time.monotonic() - started,
                gradient_steps=steps * 3,
            )
        )
        if i == 0:
            full_brier = float(np.mean(np.sum((predicted - cy) ** 2, axis=1)))
            for name in ("permuted_released_labels", "all_features_ablated"):
                control_start = time.monotonic()
                xx = np.zeros_like(x) if name == "all_features_ablated" else x
                yy = (
                    np.random.default_rng(seed).permutation(y)
                    if name == "permuted_released_labels"
                    else y
                )
                control = train(xx, yy, seed, steps)
                cp = probabilities(
                    np.zeros_like(cx).tolist() if name == "all_features_ablated" else cx, control
                )
                brier = float(np.mean(np.sum((cp - cy) ** 2, axis=1)))
                controls.append(
                    dict(
                        control=name,
                        split="development_calibration",
                        models=control,
                        brier=brier,
                        full_brier=full_brier,
                        degradation=brier - full_brier,
                        passed=brier - full_brier >= 0.01,
                        duration_s=time.monotonic() - control_start,
                        training_worlds=training,
                        evaluation_worlds=calibration,
                    )
                )
        progress("after_numerical_training", i + 1, len(seeds) - i - 1)
    return dict(
        parameters=parameters,
        train_worlds=training,
        calibration_worlds=calibration,
        costs=costs,
        shortcut_controls=controls,
        numerical_training_calls=3 * len(seeds) + 6,
    )


def action_probabilities(truth_probs: Any, answer: str) -> list[float]:
    """Candidate identity maps predicted truth to action after the scorer has run."""
    accept = float(truth_probs[LABELS.index(answer)]) if answer != "insufficient" else 0.0
    return [accept, float(sum(truth_probs[:2])) - accept, float(truth_probs[2])]


def evaluate(
    cohort: list[Json], parameters: list[Json], cost_sink: list[Json] | None = None
) -> list[Json]:
    """Keep all candidates and nested seeds; exact truth is confined to scoring/oracle."""
    held = [r for r in cohort if r["split"] == "consensus_test"]
    started = time.monotonic()
    features = [signals(r["input"]) for r in held]
    checks_s = time.monotonic() - started
    predictions = {}
    forward_s = 0.0
    for p in parameters:
        started = time.monotonic()
        predictions[p["seed"]] = probabilities(features, p["models"])
        forward_s += time.monotonic() - started
    arm_times = dict(
        agreement_only=0.0, fixed_evidence=0.0, learned_energy=forward_s, exact_oracle=0.0
    )
    rows = []
    for i, (unit, counts) in enumerate(zip(held, features, strict=True)):
        started = time.monotonic()
        oracle_truth = upstream.old.strong(unit["input"])
        arm_times["exact_oracle"] += time.monotonic() - started
        fixed = [0.1, 0.8, 0.1] if counts[1] else [0.1, 0.1, 0.8] if counts[2] else [0.8, 0.1, 0.1]
        for j, answer in enumerate(unit["candidates"]):
            arms = [
                (
                    "agreement_only",
                    None,
                    [1.0, 0.0, 0.0] if len(set(unit["candidates"])) == 1 else [1 / 3] * 3,
                    0.5,
                ),
                ("fixed_evidence", None, action_probabilities(fixed, answer), 0.6),
                ("exact_oracle", None, np.eye(3)[target(answer, oracle_truth)].tolist(), 0.5),
            ]
            arms += [
                (
                    "learned_energy",
                    p["seed"],
                    action_probabilities(predictions[p["seed"]][i], answer),
                    p["threshold"],
                )
                for p in parameters
            ]
            expected = target(answer, unit["truth"])
            for arm, seed, probs, threshold in arms:
                started = time.monotonic()
                action = decide(probs, threshold)
                brier = float(np.sum((np.asarray(probs) - np.eye(3)[expected]) ** 2))
                rows.append(
                    dict(
                        unit_id=unit["unit_id"],
                        cluster_id=unit["world_id"],
                        candidate_index=j,
                        arm=arm,
                        seed=seed,
                        answer=answer,
                        agreement=unit["agreement"],
                        condition=unit["condition"],
                        truth=unit["truth"],
                        target=expected,
                        probabilities=probs,
                        action=action,
                        metric=brier,
                        denominator=1,
                        status="completed",
                        missing_reason=None,
                        false_accept=int(action == "accept" and answer != unit["truth"]),
                        covered=int(action != "escalate"),
                        features=counts,
                    )
                )
                arm_times[arm] += time.monotonic() - started
        if (i + 1) % 180 == 0:
            progress("evaluate_groups", i + 1, len(held) - i - 1)
    if cost_sink is not None:
        cost_sink.extend(
            dict(
                arm=arm,
                duration_s=elapsed,
                shared_constraint_check_s=checks_s
                if arm in {"learned_energy", "fixed_evidence"}
                else 0.0,
                exact_solver_calls=len(held) if arm == "exact_oracle" else 0,
                groups=len(held),
            )
            for arm, elapsed in arm_times.items()
        )
    return rows


def interval(deltas: list[float], draws: int = 10000) -> list[float]:
    """Resample independent worlds, never candidates, conditions or optimization seeds."""
    values = np.asarray(deltas)
    rng = np.random.default_rng(SEEDS[0])
    samples = rng.choice(values, size=(draws, len(values))).mean(axis=1)
    return [float(values.mean()), *np.quantile(samples, [0.0125, 0.9875]).tolist()]


def summarize(rows: list[Json], draws: int = 10000) -> Json:
    """Recompute the two prespecified comparisons and retain every comparator arm."""
    keys = sorted({(r["arm"], r["seed"]) for r in rows}, key=str)
    brier, false_accepts, coverage = [], [], []
    for arm, seed in keys:
        local = [r for r in rows if (r["arm"], r["seed"]) == (arm, seed)]
        wrong = [r for r in local if r["agreement"] == "unanimous_wrong"]
        identity = dict(arm=arm, seed=seed)
        brier.append(
            dict(
                identity, brier=float(np.mean([r["metric"] for r in local])), denominator=len(local)
            )
        )
        false_accepts.append(
            dict(
                identity,
                count=sum(r["false_accept"] for r in wrong),
                denominator=len(wrong),
                fraction=float(np.mean([r["false_accept"] for r in wrong])),
            )
        )
        coverage.append(
            dict(
                identity,
                fraction=float(np.mean([r["covered"] for r in local])),
                denominator=len(local),
            )
        )
    worlds = sorted({r["cluster_id"] for r in rows})
    deltas: dict[str, list[float]] = {
        k: [] for k in ("false_accept", "brier", "coverage_agreement", "coverage_fixed")
    }
    for world in worlds:
        selected = [r for r in rows if r["cluster_id"] == world]
        by_arm = {
            a: [r for r in selected if r["arm"] == a]
            for a in ("agreement_only", "fixed_evidence", "learned_energy")
        }

        def mean(a: str, key: str, wrong: bool = False) -> float:
            return float(
                np.mean(
                    [r[key] for r in by_arm[a] if not wrong or r["agreement"] == "unanimous_wrong"]
                )
            )

        deltas["false_accept"].append(
            mean("agreement_only", "false_accept", True)
            - mean("learned_energy", "false_accept", True)
        )
        deltas["brier"].append(mean("fixed_evidence", "metric") - mean("learned_energy", "metric"))
        deltas["coverage_agreement"].append(
            mean("learned_energy", "covered") - mean("agreement_only", "covered")
        )
        deltas["coverage_fixed"].append(
            mean("learned_energy", "covered") - mean("fixed_evidence", "covered")
        )
    bounds = {key: interval(values, draws) for key, values in deltas.items()}
    gates = dict(
        false_accept_lower_97_5=bounds["false_accept"][1] >= 0.05,
        brier_lower_97_5=bounds["brier"][1] >= 0.01,
        coverage_loss_at_most_02=min(bounds[k][0] for k in ("coverage_agreement", "coverage_fixed"))
        >= -0.02,
    )
    table = Counter(
        (r["truth"], r["answer"], r["agreement"], r["condition"])
        for r in rows
        if r["arm"] == "agreement_only"
    )
    return dict(
        brier_rows=brier,
        unanimous_wrong_false_accepts=false_accepts,
        coverage_rows=coverage,
        oracle_reference_rows=[
            dict(x, verdict_class="circular_positive") for x in brier if x["arm"] == "exact_oracle"
        ],
        intervals_97_5=bounds,
        bootstrap_draws=draws,
        independent_count=len(worlds),
        contingency_table=[
            dict(truth=k[0], answer=k[1], agreement=k[2], condition=k[3], count=n)
            for k, n in sorted(table.items())
        ],
        acceptance_gates=gates,
        benefit_passed=all(gates.values()),
    )
