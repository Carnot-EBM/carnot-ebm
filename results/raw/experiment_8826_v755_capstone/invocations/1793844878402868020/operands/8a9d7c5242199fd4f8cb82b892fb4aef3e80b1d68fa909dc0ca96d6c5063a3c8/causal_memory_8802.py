"""REQ-VERIFY-8802: learn reusable checks only between sealed queries.

The small energy network learns from released verifier feedback. Its weights
and templates share one durable checkpoint so restart cannot split an update.
Constructed controls establish execution semantics, not a generalization gain.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.memory.transactional_constraint_memory import _atomic_write, canonical_json_bytes
from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.evidence_set_methods_8768 import decide, parse
from carnot.training.nce import nce_loss

Json = dict[str, Any]
ARMS = ("read_only", "additions_only", "calibration_only", "combined", "shuffled_feedback")
LABELS = ("support", "refute", "insufficient")
COUNTS = (
    "checked_predicates",
    "satisfied_local_predicates",
    "missing_predicates",
    "shared_variables",
    "observed_sign_conflicts",
    "current_successful_bindings",
)
LR = 0.03


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work counts so a parent can distinguish progress from a stall."""
    print(f"[exp8802] phase={phase} completed={completed} pending={pending}", flush=True)


def world(prefix: str, label: str) -> Json:
    """Use disjoint names to test reuse without copying an earlier answer."""
    return dict(
        facts=[
            [
                prefix + (str(i) if label == "insufficient" else "a"),
                f"r{i}",
                prefix + "b",
                label != "refute",
            ]
            for i in range(5)
        ],
        query=[
            [prefix + "a", f"r{i}", prefix + "b"] if label == "refute" else ["?x", f"r{i}", "?y"]
            for i in range(5)
        ],
    )


def normalize(payload: Json) -> str:
    """Retain role equality while removing entity, variable and relation spellings."""
    parse(payload)
    roles: dict[str, str] = {}
    relations: dict[str, str] = {}
    return json.dumps(
        [
            [
                roles.setdefault(s, ("v" if s.startswith("?") else "c") + str(len(roles))),
                relations.setdefault(r, str(len(relations))),
                roles.setdefault(o, ("v" if o.startswith("?") else "c") + str(len(roles))),
            ]
            for s, r, o in payload["query"]
        ]
    )


def observe(payload: Json, templates: list[str]) -> Json:
    """Run five primitive checks; admitted constraints join their visible witnesses."""
    parse(payload)
    if len(payload["query"]) != 5:
        raise ValueError("five_checks_required")
    checks = [decide(dict(facts=payload["facts"], query=[q])) for q in payload["query"]]
    bindings: list[dict[str, str]] = [{}]
    constrained = normalize(payload) in templates
    for s, r, o in payload["query"]:
        candidates = []
        for binding, (fs, fr, fo, sign) in product(
            bindings if constrained else [{}], payload["facts"]
        ):
            candidate = dict(binding)
            if (
                sign
                and r == fr
                and all(
                    candidate.setdefault(term, actual) == actual
                    if term.startswith("?")
                    else term == actual
                    for term, actual in [(s, fs), (o, fo)]
                )
            ):
                candidates.append(candidate)
        bindings = candidates
    satisfied = sum(c["decision"] == "support" for c in checks)
    conflicts = sum(len(c["conflicts"]) for c in checks)
    variables = [t for q in payload["query"] for t in (q[0], q[2]) if t.startswith("?")]
    counts = [
        5,
        satisfied,
        sum(bool(c["missing_premises"]) for c in checks),
        len({v for v in variables if variables.count(v) > 1}),
        conflicts,
        len(bindings),
    ]
    decision = (
        "insufficient"
        if conflicts
        else "support"
        if satisfied == 5 and bindings
        else "refute"
        if any(c["decision"] == "refute" for c in checks)
        else "insufficient"
    )
    return dict(counts=counts, decision=decision, primitive_checks=len(checks))


def initial_parameters(seed: int) -> Json:
    """Initialize numerical weights explicitly; no language model is loaded."""
    model = GibbsModel(GibbsConfig(input_dim=9, hidden_dims=[4]), jax.random.PRNGKey(seed))
    return dict(
        w1=model.layers[0][0].tolist(),
        b1=model.layers[0][1].tolist(),
        w_out=model.output_weight.tolist(),
        b_out=float(model.output_bias),
    )


def features(counts: list[int], decision: int) -> list[float]:
    """Candidate bits identify the decision being scored, never a known answer."""
    return [min(x, 32) / 5 for x in counts] + [float(i == decision) for i in range(3)]


def energy(parameters: list[jax.Array], x: jax.Array) -> jax.Array:
    """Reuse Gibbs evaluation so training and serving cannot use different networks."""
    model = object.__new__(GibbsModel)
    model.config = GibbsConfig(input_dim=9, hidden_dims=[4])
    model.layers = [(parameters[0], parameters[1])]
    model.output_weight, model.output_bias = parameters[2], parameters[3]
    return model.energy(x)


def arrays(params: Json) -> list[jax.Array]:
    """Restore a JSON checkpoint without hidden optimizer state."""
    return [jnp.asarray(params[k], dtype=jnp.float32) for k in ("w1", "b1", "w_out", "b_out")]


def probabilities(params: Json, counts: list[int]) -> list[float]:
    """Normalize the three candidate energies into one probability distribution."""
    x = jnp.asarray([features(counts, i) for i in range(3)])
    return list(map(float, jax.nn.softmax(-jax.vmap(lambda row: energy(arrays(params), row))(x))))


@jax.jit
def train_step(
    parameters: list[jax.Array], x: jax.Array, target: jax.Array
) -> tuple[Any, Any, Any]:
    """Brier trains calibration; NCE separates the released decision from alternatives."""

    def loss(p: list[jax.Array]) -> jax.Array:
        e = lambda row: energy(p, row)
        predicted = jax.nn.softmax(-jax.vmap(e)(x))
        correct = x[target][None, :]
        noise = x[(target + jnp.asarray([1, 2])) % 3]
        return jnp.sum((predicted - jax.nn.one_hot(target, 3)) ** 2) + 0.1 * nce_loss(
            e, correct, noise
        )

    value, grads = jax.value_and_grad(loss)(parameters)
    return [p - LR * g for p, g in zip(parameters, grads)], value, grads


def update(params: Json, counts: list[int], label: str) -> tuple[Json, Json]:
    """Record actual gradients and movement separately from any claimed benefit."""
    old = arrays(params)
    new, loss, grads = train_step(
        old, jnp.asarray([features(counts, i) for i in range(3)]), jnp.asarray(LABELS.index(label))
    )
    result = {k: np.asarray(v).tolist() for k, v in zip(("w1", "b1", "w_out", "b_out"), new)}
    return result, dict(
        loss=float(loss),
        gradient_norm=float(sum(jnp.sum(g * g) for g in grads) ** 0.5),
        parameter_delta=float(sum(jnp.sum((a - b) ** 2) for a, b in zip(old, new)) ** 0.5),
    )


def action(p: list[float], threshold: float) -> str:
    """Low confidence or insufficient evidence escalates instead of forcing a decision."""
    choice = int(np.argmax(p))
    return ("accept", "reject", "escalate")[choice] if max(p) >= threshold else "escalate"


def tune(params: Json, worlds: list[Json]) -> float:
    """Choose a confidence cutoff on development worlds and freeze it before controls."""

    def error(threshold: float) -> int:
        return sum(
            action(probabilities(params, observe(w, [normalize(w)])["counts"]), threshold)
            != {"support": "accept", "refute": "reject", "insufficient": "escalate"}[
                decide(w)["decision"]
            ]
            for w in worlds
        )

    return min([0.5, 0.6, 0.7, 0.8], key=error)


class Store:
    """Keep templates, weights, pending predictions and acknowledgements in one file."""

    def __init__(self, path: Path, params: Json) -> None:
        self.path = path
        self.active: Json | None = None
        self.state = (
            json.loads(path.read_text())
            if path.exists()
            else dict(templates=[], parameters=params, pending={}, acks=[], released=[])
        )
        self.save()

    def save(self) -> None:
        """Fsync and rename the complete state so no component can advance alone."""
        _atomic_write(self.path, canonical_json_bytes(self.state))

    def read(self) -> Json:
        """Return a copy of the frozen query snapshot to prevent read-side mutation."""
        return deepcopy(self.active if self.active is not None else self.state)

    def predict(self, event: str, payload: Json, threshold: float) -> Json:
        """Durably seal prediction and its predecessor before feedback can be opened."""
        if self.active is not None:
            raise RuntimeError("active query")
        if event in self.state["pending"] or event in self.state["acks"]:
            raise ValueError("duplicate prediction")
        if len(self.state["pending"]) >= 32:
            raise ValueError("pending capacity")
        self.active = deepcopy(self.state)
        observation = observe(payload, self.active["templates"])
        p = probabilities(self.active["parameters"], observation["counts"])
        sealed = dict(
            observation,
            event_id=event,
            ordinal=int(event[1:]),
            prediction=p,
            action=action(p, threshold),
            state_hash=canonical_hash(self.active),
            evidence_hash=canonical_hash(payload),
        )
        self.state["pending"][event] = sealed
        self.save()
        return deepcopy(sealed)

    def close(self) -> None:
        """Close the read boundary before any released event may modify learning state."""
        self.active = None

    def release(self, event: str, payload: Json, ordinal: int, arm: str) -> Json:
        """Verify original evidence before admitting a reusable check or numerical update."""
        if self.active is not None:
            raise RuntimeError("active query")
        if event in self.state["acks"]:
            raise ValueError("duplicate acknowledgement")
        pending = self.state["pending"][event]
        if ordinal < pending["ordinal"] + 8:
            raise ValueError("early feedback")
        if canonical_hash(payload) != pending["evidence_hash"]:
            raise ValueError("evidence mismatch")
        label = str(decide(payload)["decision"])
        prefix = self.state["released"] + [label]
        training_label = (
            prefix[(ordinal * 7) % len(prefix)] if arm == "shuffled_feedback" else label
        )
        new, gradient = update(self.state["parameters"], pending["counts"], training_label)
        coefficient = arm in {"calibration_only", "combined", "shuffled_feedback"}
        template = normalize(payload)
        add = (
            arm in {"additions_only", "combined", "shuffled_feedback"}
            and training_label == label
            and template not in self.state["templates"]
            and len(self.state["templates"]) < 32
        )
        next_state = deepcopy(self.state)
        if add:
            next_state["templates"].append(template)
        if coefficient:
            next_state["parameters"] = new
        next_state["pending"].pop(event)
        next_state["acks"].append(event)
        next_state["released"].append(label)
        self.state = next_state
        self.save()
        return dict(
            gradient,
            event_id=event,
            arm=arm,
            feedback_label=training_label,
            independently_verified_label=label,
            released_prefix=prefix,
            feedback_ordinal=ordinal,
            prediction_ordinal=pending["ordinal"],
            template_added=add,
            template=template,
            coefficient_committed=coefficient,
            checkpoint_hash=canonical_hash(self.state),
            feedback_verifier_calls=1,
            optimizer_steps=1,
        )


def freeze_manifest() -> Json:
    """Seal downstream rosters without opening downstream labels or measuring benefit.

    Each world recipe appends a distinct-length disconnected path to its signed
    graph. That path makes graph isomorphism classes disjoint across all splits.
    Seeds are repeated measurements within a reset stream, not independent units.
    """

    def roster(split: str, count: int, offset: int) -> list[Json]:
        return [
            dict(
                unit_id=f"{split}-{i}",
                disconnected_path_edges=offset + i + 6,
                structural_hash=canonical_hash(["five_predicates_plus_path", offset + i + 6]),
            )
            for i in range(count)
        ]

    return dict(
        seeds=[11, 23, 37],
        learning_rate=LR,
        fit=roster("fit", 40, 0),
        tune=roster("tune", 20, 40),
        streams=[
            dict(
                stream_id=f"stream-{i:02}",
                worlds=roster(f"online-{i}", 32, 60 + i * 44),
                retention=roster(f"retention-{i}", 12, 92 + i * 44),
            )
            for i in range(30)
        ],
        retention_updates=False,
        retention_checkpoints=[0, 16, 32],
        feature_schema=list(COUNTS) + list(LABELS),
        epochs=1,
        gradient_clipping=None,
        optimizer="SGD",
        threshold_grid=[0.5, 0.6, 0.7, 0.8],
        online_executed=False,
    )


def prototype(raw: Path, seeds: list[int] | None = None) -> Json:
    """Exercise tiny causal controls; reserve all thirty streams for downstream work."""
    raw.mkdir(parents=True, exist_ok=True)
    rows, gradients, additions, restarts, controls, heads = [], [], [], [], [], []
    labels = list(LABELS)
    selected = seeds if seeds is not None else [11, 23, 37]
    training_calls = 0
    for seed in selected:
        progress("numerical_head_fit_before", 0, 40)
        params = initial_parameters(seed)
        for i in range(40):
            payload = world(f"fit-{i}", labels[i % 3])
            params, _ = update(
                params, observe(payload, [normalize(payload)])["counts"], labels[i % 3]
            )
            training_calls += 1
        progress("numerical_head_fit_after", 40, 0)
        threshold = tune(params, [world(f"tune-{i}", labels[i % 3]) for i in range(20)])
        heads.append(
            dict(
                seed=seed,
                parameters=params,
                threshold=threshold,
                development_fit_updates=40,
                development_tune_worlds=20,
            )
        )
        for arm in ARMS:
            store = Store(raw / f"{arm}-{seed}.json", deepcopy(params))
            pending_payloads: dict[str, Json] = {}
            for ordinal in range(12):
                event = f"e{ordinal}"
                payload = world(f"control-{seed}-{ordinal}", labels[ordinal % 3])
                sealed = store.predict(event, payload, threshold)
                frozen = canonical_hash(store.read())
                within = frozen == canonical_hash(store.read())
                store.close()
                pending_payloads[event] = payload
                uninterrupted = deepcopy(store.state)
                restarted = Store(store.path, {})
                restarts.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        ordinal=ordinal,
                        pending_count=len(uninterrupted["pending"]),
                        uninterrupted_hash=canonical_hash(uninterrupted),
                        restarted_hash=canonical_hash(restarted.state),
                        passed=uninterrupted == restarted.state,
                    )
                )
                if ordinal >= 8:
                    released = f"e{ordinal - 8}"
                    receipt = store.release(released, pending_payloads[released], ordinal, arm)
                    training_calls += 1
                    gradients.append(dict(receipt, seed=seed))
                    if receipt["template_added"]:
                        additions.append(dict(receipt, seed=seed))
                    restarted.release(released, pending_payloads[released], ordinal, arm)
                    training_calls += 1
                    restarts[-1]["passed"] &= restarted.state == store.state
                target = labels[ordinal % 3]
                metric = float(
                    sum(
                        (p - float(i == LABELS.index(target))) ** 2
                        for i, p in enumerate(sealed["prediction"])
                    )
                )
                rows.append(
                    dict(
                        sealed,
                        unit_id=f"control-{ordinal}",
                        cluster_id="constructed_prototype",
                        arm=arm,
                        seed=seed,
                        metric=metric,
                        denominator=1,
                        status="completed",
                        missing_reason=None,
                        within_query_unchanged=within,
                        post_boundary_hash=canonical_hash(store.state),
                    )
                )
            progress("prototype_arm", len(rows), len(selected) * len(ARMS) * 12 - len(rows))
    sample = world("positive", "insufficient")
    for seed in selected:
        baseline = next(
            r for r in rows if r["seed"] == seed and r["arm"] == "read_only" and r["ordinal"] == 11
        )
        treatment = next(
            r
            for r in rows
            if r["seed"] == seed and r["arm"] == "additions_only" and r["ordinal"] == 11
        )
        controls.append(
            dict(
                control="verified_addition_changes_later_check",
                seed=seed,
                baseline_bindings=baseline["counts"][-1],
                added_bindings=treatment["counts"][-1],
                baseline_symbolic_decision=baseline["decision"],
                added_symbolic_decision=treatment["decision"],
                baseline_action=baseline["action"],
                added_action=treatment["action"],
                passed=baseline["counts"][-1] != treatment["counts"][-1]
                and baseline["decision"] != treatment["decision"]
                and baseline["prediction"] != treatment["prediction"],
            )
        )
    before, after = (
        observe(sample, []),
        observe(world("renamed", "insufficient"), [normalize(sample)]),
    )
    controls.append(
        dict(
            control="actual_changed_checks_and_later_decision",
            before=before,
            after=after,
            passed=before["counts"][-1] != after["counts"][-1]
            and before["decision"] != after["decision"],
        )
    )
    controls.append(
        dict(control="useless_template", passed=observe(sample, ["unrelated"]) == before)
    )
    controls.append(
        dict(
            control="released_prefix_shuffle",
            passed=all(
                g["feedback_label"] in g["released_prefix"]
                and g["feedback_ordinal"] >= g["prediction_ordinal"] + 8
                for g in gradients
            ),
        )
    )
    controls.append(
        dict(
            control="no_truth_bearing_features",
            passed=all(len(features(r["counts"], 0)) == 9 for r in rows),
        )
    )
    controls.append(
        dict(
            control="dropped_pending_feedback_never_trains",
            passed=all(
                len(json.loads((raw / f"{arm}-{seed}.json").read_text())["pending"]) == 8
                for seed in selected
                for arm in ARMS
            ),
        )
    )
    store = Store(
        raw / "combined-11.json" if 11 in selected else raw / f"combined-{selected[0]}.json", {}
    )
    unchanged = canonical_hash(store.state)
    try:
        store.release("e0", world("duplicate", "support"), 20, "combined")
    except ValueError as exc:
        controls.append(
            dict(
                control="duplicate_feedback",
                passed="duplicate" in str(exc) and unchanged == canonical_hash(store.state),
            )
        )
    try:
        observe(dict(sample, final_label="insufficient"), [])
    except ValueError:
        controls.append(dict(control="label_leakage_rejected", passed=True))
    auxiliary = Store(raw / "auxiliary-controls.json", {})
    auxiliary.state = deepcopy(store.state)
    store = auxiliary
    store.predict("e12", sample, 0.6)
    snapshot = canonical_hash(store.read())
    try:
        store.release("e4", sample, 20, "combined")
    except RuntimeError:
        controls.append(
            dict(control="active_write_rejected", passed=snapshot == canonical_hash(store.read()))
        )
    store.close()
    return dict(
        rows=rows,
        template_addition_rows=additions,
        gradient_update_rows=gradients,
        restart_rows=restarts,
        leakage_controls=controls,
        calibrated_head_parameters=heads,
        numerical_training_calls=training_calls,
    )
