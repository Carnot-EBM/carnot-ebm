"""REQ-REPORT-8959: anonymous graph context without traversal or truth features.

The same encoder architecture lets an ordinary classifier test whether any gain
comes from graph inputs rather than the energy head. Each arm trains separately.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.autoresearch import graph_calibration_benchmark as b
from carnot.models.gibbs import GibbsConfig, GibbsModel

Json = dict[str, Any]
KEYS = {"nodes", "edges", "query_node", "proposed_node", "features"}
WIDTH, ROUNDS, STEPS, RATE = 8, 3, 200, 0.01


def pack(rows: list[Json]) -> Json:
    """Reject extra fields before indexing; names only locate anonymous markers."""
    if not rows or any(set(row) != KEYS for row in rows):
        raise ValueError("detector_schema")
    confidence = b.features([row["features"] for row in rows])
    operands: dict[str, list[Any]] = {k: [] for k in ("x", "a", "mask", "s", "q", "c", "missing")}
    for row, conf in zip(rows, confidence, strict=True):
        nodes, edges = row["nodes"], row["edges"]
        if (
            not 1 <= len(nodes) <= 60
            or any(not isinstance(n, str) for n in nodes)
            or len(set(nodes)) != len(nodes)
            or row["query_node"] not in nodes
            or (row["proposed_node"] is not None and row["proposed_node"] not in nodes)
        ):
            raise ValueError("graph_nodes")
        index = {n: i for i, n in enumerate(nodes)}
        adjacency = np.zeros((60, 60))
        for edge in edges:
            if (
                len(edge) != 2
                or any(n not in index for n in edge)
                or edge[0] == edge[1]
                or adjacency[index[edge[0]], index[edge[1]]]
            ):
                raise ValueError("graph_edges")
            a, c = (index[n] for n in edge)
            adjacency[a, c] = adjacency[c, a] = 1
        source = np.zeros(60)
        source[index[row["query_node"]]] = 1
        answer = np.zeros(60)
        if row["proposed_node"] is not None:
            answer[index[row["proposed_node"]]] = 1
        mask = np.arange(60) < len(nodes)
        degree = adjacency.sum(axis=1)
        values = dict(
            x=np.stack([source, answer, mask, degree / max(1, len(nodes) - 1)], axis=1),
            a=adjacency / np.maximum(1, degree[:, None]),
            mask=mask,
            s=source,
            q=answer,
            c=conf,
            missing=float(row["proposed_node"] is None),
        )
        for key, value in values.items():
            operands[key].append(value)
    return {key: jnp.asarray(value) for key, value in operands.items()}


def initial(seed: int, relational: bool, energy: bool) -> Json:
    """Matched encoder seeds start equally, then labels train independent parameters."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)
    dim = 27 if relational else 2
    encoder = (
        dict(
            project=jax.random.normal(keys[0], (4, 8)) / 2,
            bias=jnp.zeros(8),
            self=jax.random.normal(keys[1], (8, 8)) / jnp.sqrt(8),
            neighbor=jax.random.normal(keys[2], (8, 8)) / jnp.sqrt(8),
            round_bias=jnp.zeros(8),
        )
        if relational
        else {}
    )
    model = GibbsModel(GibbsConfig(input_dim=dim, hidden_dims=[8]), keys[3])
    head = (
        (*model.layers[0], model.output_weight, jnp.asarray(model.output_bias))
        if energy
        else (jax.random.normal(keys[4], (dim,)) / jnp.sqrt(dim), jnp.asarray(0.0))
    )
    return dict(encoder=encoder, head=head)


def logits(params: Json, data: Json, relational: bool, energy: bool) -> Any:
    """Shared rounds and masked means commute with every node permutation."""
    x = data["c"]
    if relational:
        e = params["encoder"]
        h = jnp.tanh(data["x"] @ e["project"] + e["bias"])
        for _ in range(ROUNDS):
            h = jnp.tanh(h @ e["self"] + (data["a"] @ h) @ e["neighbor"] + e["round_bias"])
            h = h * data["mask"][:, :, None]
        pooled = h.sum(axis=1) / data["mask"].sum(axis=1)[:, None]
        x = jnp.concatenate(
            [
                (h * data["s"][:, :, None]).sum(axis=1),
                (h * data["q"][:, :, None]).sum(axis=1),
                pooled,
                x,
                data["missing"][:, None],
            ],
            axis=1,
        )
    head = params["head"]
    if energy:
        model = GibbsModel(
            GibbsConfig(input_dim=x.shape[1], hidden_dims=[8]), jax.random.PRNGKey(0)
        )
        model.layers = [(head[0], head[1])]
        model.output_weight, model.output_bias = head[2], head[3]
        return jax.vmap(model.energy)(x)
    return x @ head[0] + head[1]


def fit(rows: list[Json], truth: list[int], seed: int, relational: bool, energy: bool) -> Json:
    """Only fit operands enter full-batch population loss; no guard or future labels."""
    b.progress(f"8959_before_training_seed{seed}_relational{relational}_energy{energy}", 0, STEPS)
    data = pack(rows)
    y = b.labels(truth, len(rows))
    if len(set(truth)) != 2:
        raise ValueError("training_class_support")
    scaling = b._scaling(np.asarray(data["c"]))
    data["c"] = jnp.asarray(b._standardize(np.asarray(data["c"]), scaling))
    params = initial(seed, relational, energy)

    def loss(p: Json) -> Any:
        return b.population_loss(logits(p, data, relational, energy), y)

    gradient = jax.jit(jax.grad(loss))
    first = gradient(params)
    norm = lambda tree: float(sum(np.sum(np.asarray(x) ** 2) for x in jax.tree.leaves(tree)) ** 0.5)
    before = float(loss(params))

    def update(_: Any, p: Json) -> Json:
        return jax.tree.map(lambda value, grad: value - RATE * grad, p, gradient(p))

    chunk = jax.jit(lambda p: jax.lax.fori_loop(0, 50, update, p))
    for step in range(50, STEPS + 1, 50):
        params = chunk(params)
        jax.block_until_ready(params)
        b.progress("8959_training", step, STEPS - step)
    last = gradient(params)
    serialized = jax.tree.map(lambda a: np.asarray(a).tolist(), params)
    serialized["head"] = list(serialized["head"])
    result = dict(
        params=serialized,
        seed=seed,
        relational=relational,
        energy=energy,
        scaling=scaling,
        steps=STEPS,
        rate=RATE,
        gradient_norm=max(norm(first), norm(last)),
        encoder_gradient_norm=max(norm(first["encoder"]), norm(last["encoder"])),
        loss_before=before,
        loss_after=float(loss(params)),
    )
    b.progress("8959_after_training", STEPS, 0)
    return result


def predict(state: Json, rows: list[Json], temperature: float = 1.0) -> Any:
    """Training scales remain fixed when calibration or future graphs are scored."""
    if temperature not in b.GRID:
        raise ValueError("temperature_grid")
    data = pack(rows)
    data["c"] = jnp.asarray(b._standardize(np.asarray(data["c"]), state["scaling"]))
    params = dict(
        encoder={k: jnp.asarray(v) for k, v in state["params"]["encoder"].items()},
        head=tuple(jnp.asarray(v) for v in state["params"]["head"]),
    )
    return b._sigmoid(logits(params, data, state["relational"], state["energy"]) / temperature)


def ensemble(states: list[Json], rows: list[Json], temperature: float = 1.0) -> Any:
    """Seeds average probabilities without creating additional independent examples."""
    return np.mean([predict(state, rows, temperature) for state in states], axis=0)


def select_temperature(states: list[Json], calibration: list[Json], truth: list[int]) -> float:
    """One common arm temperature uses fresh calibration only, with ascending tie order."""
    y = b.labels(truth, len(calibration))
    return min(b.GRID, key=lambda t: float(np.mean((ensemble(states, calibration, t) - y) ** 2)))
