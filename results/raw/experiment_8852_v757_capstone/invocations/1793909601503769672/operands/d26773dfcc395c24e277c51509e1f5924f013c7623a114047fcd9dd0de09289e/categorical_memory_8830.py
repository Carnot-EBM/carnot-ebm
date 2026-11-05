"""REQ-VERIFY-8830: opt-in normalized heads on the existing durable template store.

Only released verified labels train the small head. The original Gibbs mixed
loss remains available; no generator weights or future streams are opened.
"""

from copy import deepcopy
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.memory import causal_memory_8802 as m
from carnot.memory.transactional_constraint_memory import _atomic_write, canonical_json_bytes
from carnot.reporting import learning_signal_diagnosis_8815 as d
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def head(seed: int, objective: str = "mixed") -> Json:
    """Keep mixed loss as the default; callers must explicitly opt into alternatives."""
    if objective not in d.OBJECTIVES:
        raise ValueError("objective")
    return dict(
        objective=objective,
        lr=0.1,
        weights=[np.asarray(x).tolist() for x in d.initial(seed, objective)],
    )


@partial(jax.jit, static_argnames=("objective",))
def step(weights: list[jax.Array], x: jax.Array, y: jax.Array, lr: float, objective: str) -> Any:
    """Use the authenticated exact loss and one plain SGD step without hidden state."""
    value, grads = jax.value_and_grad(d.loss)(weights, x, y, objective)
    return [p - lr * g for p, g in zip(weights, grads)], value, grads


def probabilities(params: Json, counts: list[int]) -> list[float]:
    """Normalize all three candidate energies using the same serving-visible counts."""
    weights = [jnp.asarray(x, dtype=jnp.float32) for x in params["weights"]]
    x = jnp.asarray([m.features(counts, i) for i in range(3)])
    return list(map(float, jax.nn.softmax(d.logits(weights, x, params["objective"]))))


def update(params: Json, counts: list[int], label: str) -> tuple[Json, Json]:
    """Retain actual gradients and movement so an update counter is never the proof."""
    old = [jnp.asarray(x, dtype=jnp.float32) for x in params["weights"]]
    x = jnp.asarray([m.features(counts, i) for i in range(3)])
    new, value, grads = step(
        old, x, jnp.asarray(m.LABELS.index(label)), params["lr"], params["objective"]
    )
    result = dict(params, weights=[np.asarray(x).tolist() for x in new])
    return result, dict(
        loss=float(value),
        gradient_norm=float(sum(np.sum(np.asarray(g) ** 2) for g in grads) ** 0.5),
        parameter_delta=float(
            sum(np.sum((np.asarray(a) - np.asarray(b)) ** 2) for a, b in zip(old, new)) ** 0.5
        ),
        optimizer_steps=1,
    )


def binding_provenance(payload: Json) -> list[Json]:
    """Retain actual fact witnesses so a template addition is more than a counter."""
    witnesses: list[Json] = [dict(variables={}, fact_indices=[])]
    for predicate in payload["query"]:
        joined = []
        for prior in witnesses:
            for index, fact in enumerate(payload["facts"]):
                variables = dict(prior["variables"])
                if fact[3] and all(
                    variables.setdefault(term, actual) == actual
                    if term.startswith("?")
                    else term == actual
                    for term, actual in zip(predicate, fact[:3])
                ):
                    joined.append(
                        dict(variables=variables, fact_indices=prior["fact_indices"] + [index])
                    )
        witnesses = joined
    return witnesses


class Store(m.Store):
    """Reuse frozen reads and durable file format; commit admission before memory moves."""

    def commit(self, state: Json) -> None:
        """A write error leaves both the live object and durable predecessor intact."""
        _atomic_write(self.path, canonical_json_bytes(state))
        self.state = state

    def predict(self, event: str, payload: Json, threshold: float) -> Json:
        """Seal the decision and probabilities without asking the whole-query verifier."""
        if self.active is not None:
            raise RuntimeError("active query")
        if event in self.state["pending"] or event in self.state["acks"]:
            raise ValueError("duplicate prediction")
        if len(self.state["pending"]) >= 32:
            raise ValueError("pending capacity")
        snapshot = deepcopy(self.state)
        observation = m.observe(payload, snapshot["templates"])
        p = probabilities(snapshot["parameters"], observation["counts"])
        template = m.normalize(payload)
        sealed = dict(
            observation,
            event_id=event,
            ordinal=int(event[1:]),
            prediction=p,
            action=m.action(p, threshold),
            state_hash=canonical_hash(snapshot),
            evidence_hash=canonical_hash(payload),
            template_uses=[template] if template in snapshot["templates"] else [],
        )
        state = deepcopy(self.state)
        state["pending"][event] = sealed
        self.commit(state)
        self.active = snapshot
        return deepcopy(sealed)

    def release(
        self,
        event: str,
        payload: Json,
        ordinal: int,
        arm: str = "combined",
        *,
        label: str | None = None,
    ) -> Json:
        """Missing, early, changed or duplicate feedback is a byte-preserving no-op."""
        pending = self.state["pending"].get(event)
        if (
            self.active is not None
            or pending is None
            or label not in m.LABELS
            or ordinal < pending["ordinal"] + 8
            or canonical_hash(payload) != pending["evidence_hash"]
        ):
            return dict(status="invalid", optimizer_steps=0)
        verified = m.decide(payload)
        if label != verified["decision"]:
            return dict(status="invalid", optimizer_steps=0)
        new, gradient = update(self.state["parameters"], pending["counts"], label)
        state = deepcopy(self.state)
        template = m.normalize(payload)
        add = template not in state["templates"] and len(state["templates"]) < 32
        if add:
            state["templates"].append(template)
            state.setdefault("template_provenance", {})[template] = dict(
                event_id=event,
                evidence_hash=canonical_hash(payload),
                query=payload["query"],
                verifier=verified,
                binding_witnesses=binding_provenance(payload),
            )
        state["parameters"] = new
        state["pending"].pop(event)
        state["acks"].append(event)
        state["released"].append(label)
        self.commit(state)
        return dict(
            gradient,
            status="complete",
            arm=arm,
            event_id=event,
            feedback_label=label,
            prediction_hash=canonical_hash(pending),
            prediction_ordinal=pending["ordinal"],
            feedback_ordinal=ordinal,
            template_added=add,
            template=template,
            checkpoint_hash=canonical_hash(state),
        )


def controls() -> list[Json]:
    """A shuffled label control must lose real separable signal, not celebrate flatness."""
    rows = [
        dict(
            counts=m.observe(m.world("fixture", y), [m.normalize(m.world("fixture", y))])["counts"],
            truth=y,
        )
        for _ in range(30)
        for y in m.LABELS
    ]
    labels = np.random.default_rng(8830).permutation([r["truth"] for r in rows])
    shuffled = [dict(r, truth=str(y)) for r, y in zip(rows, labels)]
    checked = d.controls()
    for objective in ("categorical", "linear"):
        clean = d.fit(rows, rows, 11, objective, 0.1)
        broken = d.fit(shuffled, rows, 11, objective, 0.1)
        checked.append(
            dict(
                check="shuffled_damage",
                objective=objective,
                clean=clean["validation_brier"],
                shuffled=broken["validation_brier"],
                passed=clean["validation_brier"] + 0.05 < broken["validation_brier"],
            )
        )
    return checked
