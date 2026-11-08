"""REQ-REPORT-8892: preserve signed local evidence before a linear energy score.

This opt-in schema uses the existing durable store. It cannot read labels during
observation, and it leaves the six-coordinate legacy checkpoint format alone.
"""

from collections import Counter
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.reporting import topology_memory_methods_8881 as old
from carnot.memory import categorical_memory_8830 as categorical
from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
SCHEMA = "signed-local-v1"
EXTRA = ["explicit_refutation_fraction", "minimum_positive_binding_overlap"]


def observe(payload: Json, templates: list[str]) -> Json:
    """Scan every atom/fact pair once so neither oracle truth nor hidden work enters features.

    Positive candidates retain variable assignments. Pair overlap uses their shared
    variables only, so unrelated bindings cannot create false relation agreement.
    """
    symbolic.parse(payload)
    local = []
    signs: dict[tuple[str, ...], set[bool]] = {}
    for *triple, sign in payload["facts"]:
        signs.setdefault(tuple(triple), set()).add(sign)
    for atom in payload["query"]:
        positive, negative = [], 0
        for fact in payload["facts"]:
            binding: dict[str, str] = {}
            matched = True
            for term, actual in zip(atom, fact[:3], strict=True):
                matched &= (
                    binding.setdefault(term, actual) == actual
                    if term.startswith("?")
                    else term == actual
                )
            if matched:
                if fact[3]:
                    positive.append(binding)
                else:
                    negative += 1
        local.append((atom, positive, negative))
    # Equal keys have equal local counts; their order cannot change the last count.
    local.sort(key=lambda item: (len(item[1]), item[2], tuple(t.startswith("?") for t in item[0])))
    variables = [t for atom in payload["query"] for t in (atom[0], atom[2]) if t.startswith("?")]
    constrained = old.topology(payload["query"]) in templates
    bindings: list[dict[str, str]] = [{}]
    joins = 0
    for _, candidates, _ in local:
        joined = []
        for prior in bindings if constrained else [{}]:
            for candidate in candidates:
                joins += 1
                if all(prior.get(k, v) == v for k, v in candidate.items()):
                    joined.append(dict(prior, **candidate))
        bindings = joined
    overlaps, comparisons = [], 0
    for (a, pa, _), (b, pb, _) in combinations(local, 2):
        shared = sorted({v for v in a if v.startswith("?")} & {v for v in b if v.startswith("?")})
        if shared:
            left = {tuple(c[v] for v in shared) for c in pa}
            right = {tuple(c[v] for v in shared) for c in pb}
            comparisons += len(left) * len(right)
            overlaps.append(
                sum(x == y for x in left for y in right) / len(left | right)
                if left and right
                else 0.0
            )
    conflicts = sum(len(s) > 1 for s in signs.values()) * len(local)
    counts = [
        len(local),
        sum(bool(p) and not conflicts for _, p, _ in local),
        sum(not p for _, p, _ in local),
        sum(n > 1 for n in Counter(variables).values()),
        conflicts,
        len(bindings),
        sum(bool(n) for _, _, n in local) / len(local),
        min(overlaps, default=1.0),
    ]
    return dict(
        counts=counts,
        primitive_checks=len(local),
        work=dict(
            fact_comparisons=len(local) * len(payload["facts"]),
            sign_record_reads=len(payload["facts"]),
            binding_comparisons=joins,
            candidate_pair_comparisons=comparisons,
            atom_pairs=len(overlaps),
        ),
    )


def head(seed: int, arm: str, scale: list[float]) -> Json:
    """Extend the same initialized linear head with two zero coordinates for a fair start."""
    weights = np.asarray(categorical.head(seed, "linear")["weights"][0])
    params = dict(
        feature_schema=SCHEMA,
        feature_arm=arm,
        scale=scale,
        objective="linear",
        lr=0.1,
        weights=[
            np.concatenate((weights[:, :6], np.zeros((3, 2)), weights[:, 6:]), axis=1).tolist()
        ],
    )
    vector(params, [0.0] * 8)
    return params


def vector(params: Json, counts: list[float]) -> list[float]:
    """Refuse a schema change instead of reinterpreting an old checkpoint's numbers."""
    if (
        params.get("feature_schema") != SCHEMA
        or params.get("feature_arm") not in {"old", "signed"}
        or params.get("objective") != "linear"
        or params.get("lr") != 0.1
        or np.asarray(params.get("weights")).shape != (1, 3, 9)
        or len(params.get("scale", [])) != 2
        or len(counts) != 8
        or not all(np.isfinite(v) and v > 0 for v in params["scale"])
        or not np.all(np.isfinite(counts))
        or not np.all(np.isfinite(params["weights"]))
    ):
        raise ValueError("feature_schema_mismatch")
    extra = [v / scale for v, scale in zip(counts[6:], params["scale"], strict=True)]
    return (
        [min(v, 32) / 5 for v in counts[:6]]
        + (extra if params["feature_arm"] == "signed" else [0.0, 0.0])
        + [1.0]
    )


def probabilities(params: Json, counts: list[float]) -> list[float]:
    """Serving uses the same normalized linear energies as the unchanged SGD objective."""
    x = jnp.asarray(vector(params, counts), dtype=jnp.float32)
    return list(map(float, jax.nn.softmax(jnp.asarray(params["weights"][0]) @ x)))


@jax.jit
def step(weights: jax.Array, x: jax.Array, target: jax.Array) -> Any:
    """One plain SGD step keeps the existing linear categorical energy objective."""
    loss, grad = jax.value_and_grad(lambda w: -jax.nn.log_softmax(w @ x)[target])(weights)
    return weights - 0.1 * grad, loss, grad


def update(params: Json, counts: list[float], label: str) -> tuple[Json, Json]:
    """Retain gradients and real parameter movement instead of inferring training from counters."""
    x = jnp.asarray(vector(params, counts), dtype=jnp.float32)
    weights = jnp.asarray(params["weights"][0], dtype=jnp.float32)
    new, loss, grad = step(weights, x, jnp.asarray(primitive.LABELS.index(label)))
    return dict(params, weights=[np.asarray(new).tolist()]), dict(
        loss=float(loss),
        gradient_norm=float(jnp.linalg.norm(grad)),
        parameter_delta=float(jnp.linalg.norm(new - weights)),
        optimizer_steps=1,
    )


def engine() -> Any:
    """Configure isolated runner globals so original consumers and defaults cannot change."""
    runtime = namespace(old, "signed8892")
    runtime.observe = observe
    runtime.c = SimpleNamespace(probabilities=probabilities, update=update)

    class Store(old.Store):
        """Reject incompatible parameters before prediction or any durable write."""

        def __init__(self, path: Path, params: Json) -> None:
            super().__init__(path, params)
            vector(self.state["parameters"], [0.0] * 8)
            if params and any(
                params[k] != self.state["parameters"][k]
                for k in ("feature_schema", "feature_arm", "scale")
            ):
                raise ValueError("feature_schema_mismatch")

    runtime.Store = type("Store", (Store,), {"predict": runtime.Store.predict})
    return runtime
