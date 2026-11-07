"""REQ-VERIFY-8815: distinguish finite feature ambiguity from objective behavior.

Labels are exposed development supervision or retrospective diagnostic truth.
Neither source establishes future stream benefit or population generalization.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import partial
from itertools import permutations
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.memory import causal_memory_8802 as m
from carnot.reporting import evidence_intervention_corpus_8791 as corpus
from carnot.reporting.causal_learning_audit_8810 import load, reference, require
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_set_corpus_8768 import graph_roster

Json = dict[str, Any]
SEEDS = (11, 23, 37, 47, 59)
LRS = (0.01, 0.03, 0.1)
OBJECTIVES = ("mixed", "categorical", "linear")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Show actual phase boundaries so compilation cannot resemble a stalled run."""
    print(f"[exp8815] phase={phase} completed={completed} pending={pending}", flush=True)


def structure(payload: Json) -> str:
    """Ignore names and query ordering while preserving variable and constant roles."""
    terms = {t for q in payload["query"] for t in (q[0], q[2])}
    variables = sorted(t for t in terms if t.startswith("?"))
    constants = sorted(terms - set(variables))
    encodings = []
    for vs in permutations(range(len(variables))):
        for cs in permutations(range(len(constants))):
            names = dict(zip(variables, ["v" + str(i) for i in vs]))
            names.update(zip(constants, ["c" + str(i) for i in cs]))
            encodings.append(sorted((names[q[0]], names[q[2]]) for q in payload["query"]))
    return canonical_hash(min(encodings))


def collisions(rows: list[Json], mode: str) -> tuple[list[Json], float]:
    """Compute an empirical corpus optimum without claiming a population bound."""
    groups: dict[tuple[int, ...], list[Json]] = defaultdict(list)
    for row in rows:
        groups[tuple(row["counts"])].append(row)
    table, floor = [], 0.0
    for counts, group in sorted(groups.items()):
        labels = Counter(r["truth"] for r in group)
        conditional = {label: labels[label] / len(group) for label in m.LABELS}
        contribution = len(group) / len(rows) * (1 - sum(p * p for p in conditional.values()))
        floor += contribution
        table.append(
            dict(
                template_mode=mode,
                counts=list(counts),
                count=len(group),
                label_counts=dict(labels),
                conditional_labels=conditional,
                conflicting=len(labels) > 1,
                distinct_structures=len({r["structure"] for r in group}),
                floor_contribution=contribution,
            )
        )
    return table, floor


def development(excluded: set[str]) -> tuple[list[Json], Json]:
    """Use fresh canonical five-edge graphs, with all sibling interventions kept together."""
    historical = {
        structure(corpus.units(e, "historical")[0]["input"]) for e in graph_roster(5, 120)
    }
    rows, identities = [], []
    for edges in graph_roster(5, 240):
        units = corpus.units(edges, "development")
        key = structure(units[0]["input"])
        if key in historical | excluded:
            continue
        split = "fit" if len(identities) < 40 else "validation"
        identities.append(key)
        for index, row in enumerate(units):
            payload = row["input"]
            rows.append(
                dict(
                    unit_id=f"{key}/{index}",
                    structure=key,
                    split=split,
                    input=payload,
                    truth=row["expected"],
                    counts=m.observe(payload, [m.normalize(payload)])["counts"],
                )
            )
        if len(identities) == 60:
            break
    require(len(identities) == 60, "development_structure_shortfall")
    return rows, dict(
        fit=identities[:40],
        validation=identities[40:],
        excluded=sorted(historical | excluded),
        query_count=5,
        feature_count=6,
        hidden_width=4,
        primitive_checks=5,
        exposure="exposed development only; all variants cluster by graph",
    )


def initial(seed: int, objective: str) -> list[jax.Array]:
    """Give both Gibbs objectives exactly the same initial state for each nested seed."""
    if objective == "linear":
        return [jax.random.normal(jax.random.PRNGKey(seed), (3, 7)) * 0.01]
    return m.arrays(m.initial_parameters(seed))


def logits(parameters: list[jax.Array], x: jax.Array, objective: str) -> jax.Array:
    """Use shared Gibbs energies; the diagnostic linear head has class-specific slopes."""
    if objective == "linear":
        return parameters[0] @ jnp.concatenate((x[0, :6], jnp.ones(1)))
    return -jax.vmap(lambda candidate: m.energy(parameters, candidate))(x)


def loss(parameters: list[jax.Array], x: jax.Array, y: jax.Array, objective: str) -> jax.Array:
    """Categorical normalization compares the same three candidates exactly."""
    z = logits(parameters, x, objective)
    if objective == "mixed":
        e = -z
        noise = e[(y + jnp.asarray([1, 2])) % 3]
        nce = -jax.nn.log_sigmoid(-e[y]) - jnp.mean(jax.nn.log_sigmoid(noise))
        return jnp.sum((jax.nn.softmax(z) - jax.nn.one_hot(y, 3)) ** 2) + 0.1 * nce
    return -jax.nn.log_softmax(z)[y]


def dataset(rows: list[Json]) -> tuple[jax.Array, jax.Array]:
    """Candidate identity is the only appended bit; truth never enters the features."""
    return (
        jnp.asarray([[m.features(r["counts"], i) for i in range(3)] for r in rows]),
        jnp.asarray([m.LABELS.index(r["truth"]) for r in rows]),
    )


@partial(jax.jit, static_argnames=("objective", "steps"))
def train(
    parameters: list[jax.Array],
    x: jax.Array,
    y: jax.Array,
    vx: jax.Array,
    vy: jax.Array,
    order: jax.Array,
    lr: float,
    objective: str,
    steps: int,
) -> tuple[Any, Any]:
    """Record every SGD update at a fixed budget, including probability and loss curves."""

    def metrics(p: list[jax.Array]) -> jax.Array:
        probs = jax.nn.softmax(jax.vmap(lambda a: logits(p, a, objective))(x), axis=-1)
        valid = jax.nn.softmax(jax.vmap(lambda a: logits(p, a, objective))(vx), axis=-1)
        return jnp.asarray(
            [
                jnp.mean(jnp.sum((probs - jax.nn.one_hot(y, 3)) ** 2, axis=1)),
                jnp.mean(jnp.sum((valid - jax.nn.one_hot(vy, 3)) ** 2, axis=1)),
                jnp.mean(jnp.max(valid, axis=1) - jnp.min(valid, axis=1)),
            ]
        )

    def step(p: list[jax.Array], index: jax.Array) -> tuple[Any, Any]:
        at = order[index % len(order)]
        value, grads = jax.value_and_grad(loss)(p, x[at], y[at], objective)
        new = [a - lr * g for a, g in zip(p, grads)]
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in grads))
        return new, jnp.concatenate((jnp.asarray([value, norm]), metrics(new)))

    before = metrics(parameters)
    new, curve = jax.lax.scan(step, parameters, jnp.arange(steps))
    return new, jnp.concatenate(
        (
            jnp.asarray([[loss(parameters, x[order[0]], y[order[0]], objective), 0.0, *before]]),
            curve,
        )
    )


def fit(
    rows: list[Json], valid: list[Json], seed: int, objective: str, lr: float, steps: int = 400
) -> Json:
    """Freeze the same sample order across objectives and learning rates within a seed."""
    x, y = dataset(rows)
    vx, vy = dataset(valid)
    order = jnp.asarray(np.random.default_rng(seed).permutation(len(rows)))
    params, array = train(initial(seed, objective), x, y, vx, vy, order, lr, objective, steps)
    curve = [
        dict(
            update=i,
            loss=float(a),
            gradient_norm=float(b),
            train_brier=float(c),
            validation_brier=float(v),
            probability_spread=float(s),
        )
        for i, (a, b, c, v, s) in enumerate(np.asarray(array))
    ]
    return dict(
        seed=seed,
        objective=objective,
        lr=lr,
        updates=steps,
        curve=curve,
        validation_brier=curve[-1]["validation_brier"],
        parameters=[np.asarray(a).tolist() for a in params],
    )


def controls() -> list[Json]:
    """Use tiny constructed controls to check the optimizer without claiming natural benefit."""
    from jax.flatten_util import ravel_pytree

    rows = [
        dict(
            counts=m.observe(m.world("control", y), [m.normalize(m.world("control", y))])["counts"],
            truth=y,
        )
        for y in m.LABELS
    ]
    x, y = dataset(rows)
    checked = []
    for objective in OBJECTIVES:
        p = initial(11, objective)
        flat, restore = ravel_pytree(p)
        fn = lambda v: loss(restore(v), x[0], y[0], objective)
        analytic = np.asarray(jax.grad(fn)(flat))
        numerical = np.asarray(
            [
                (float(fn(flat.at[i].add(0.01))) - float(fn(flat.at[i].add(-0.01)))) / 0.02
                for i in range(len(flat))
            ]
        )
        error = float(np.max(np.abs(analytic - numerical)))
        checked.append(
            dict(
                check="finite_difference",
                objective=objective,
                max_absolute_error=error,
                tolerance=0.0001,
                passed=error < 0.0001,
            )
        )
        fitted = fit(rows, rows, 11, objective, 0.1)
        checked.append(
            dict(
                check="separable",
                objective=objective,
                before=fitted["curve"][0]["train_brier"],
                after=fitted["curve"][-1]["train_brier"],
                passed=fitted["curve"][-1]["train_brier"] < 0.60,
            )
        )
    p = initial(11, "mixed")
    actual, value, grads = m.train_step(p, x[0], y[0])
    own_value, own_grads = jax.value_and_grad(loss)(p, x[0], y[0], "mixed")
    checked.append(
        dict(
            check="mixed_update_parity",
            passed=bool(
                np.allclose(value, own_value, atol=1e-7)
                and all(
                    np.allclose(a, b - 0.03 * g, atol=1e-7) for a, b, g in zip(actual, p, own_grads)
                )
                and all(np.allclose(a, b, atol=1e-7) for a, b in zip(grads, own_grads))
            ),
        )
    )
    q = dict(zip(("w1", "b1", "w_out", "b_out"), [np.asarray(a).tolist() for a in p]))
    checked.append(
        dict(
            check="serving_parity",
            passed=all(
                np.allclose(
                    m.probabilities(q, r["counts"]),
                    np.asarray(jax.nn.softmax(logits(p, a, "categorical"))),
                    atol=1e-7,
                )
                for r, a in zip(rows, x)
            ),
        )
    )
    conflict = [dict(counts=rows[0]["counts"], truth=label, structure="same") for label in m.LABELS]
    fitted = fit(conflict, conflict, 11, "categorical", 0.1)
    _, floor = collisions(conflict, "identical")
    checked.append(
        dict(
            check="conflicting",
            empirical_floor=floor,
            observed=fitted["validation_brier"],
            passed=bool(
                fitted["validation_brier"] >= floor - 1e-6
                and np.allclose(dataset(conflict)[0][0], dataset(conflict)[0][1])
            ),
        )
    )
    shuffled = [dict(r, structure=str(i)) for i in range(30) for r in rows]
    labels = np.random.default_rng(8815).permutation([r["truth"] for r in shuffled])
    shuffled = [dict(r, truth=str(label)) for r, label in zip(shuffled, labels)]
    fitted = fit(shuffled, shuffled, 11, "categorical", 0.1)
    _, floor = collisions(shuffled, "shuffled")
    checked.append(
        dict(
            check="shuffled",
            empirical_floor=floor,
            observed=fitted["validation_brier"],
            changed_labels=sum(r["truth"] != rows[i % 3]["truth"] for i, r in enumerate(shuffled)),
            passed=fitted["validation_brier"] >= floor - 1e-6,
        )
    )
    return checked


def selection(fits: list[Json], disjoint: bool, controls_passed: bool) -> Json:
    """Choose learning rates on development only, then compare paired terminal updates."""
    rates = {
        o: min(
            LRS,
            key=lambda lr: np.mean(
                [r["validation_brier"] for r in fits if r["objective"] == o and r["lr"] == lr]
            ),
        )
        for o in OBJECTIVES
    }
    paired = []
    for seed in SEEDS:
        picked = {
            o: next(
                r for r in fits if r["objective"] == o and r["seed"] == seed and r["lr"] == rates[o]
            )
            for o in OBJECTIVES
        }
        delta = picked["mixed"]["validation_brier"] - picked["categorical"]["validation_brier"]
        paired.append(
            dict(
                seed=seed,
                improvement=delta,
                passed=delta >= 0.01,
                original_brier=picked["mixed"]["validation_brier"],
                categorical_brier=picked["categorical"]["validation_brier"],
                linear_brier=picked["linear"]["validation_brier"],
                updates=400,
            )
        )
    winners = sum(r["passed"] for r in paired)
    ready = int(winners >= 4 and disjoint and controls_passed)
    return dict(
        ready=ready,
        winning_seeds=winners,
        paired_rows=paired,
        learning_rates=rates,
        rule="mean development validation Brier chooses LR per objective;400 matched updates",
        continuation="freeze_categorical_candidate" if ready else "closed_no_bigger_sweep",
        configuration=dict(
            objective="categorical",
            lr=rates["categorical"],
            updates=400,
            seeds=list(SEEDS),
            hidden_width=4,
            features=list(m.COUNTS),
            candidate_bits=3,
            optimizer="SGD",
        ),
    )
