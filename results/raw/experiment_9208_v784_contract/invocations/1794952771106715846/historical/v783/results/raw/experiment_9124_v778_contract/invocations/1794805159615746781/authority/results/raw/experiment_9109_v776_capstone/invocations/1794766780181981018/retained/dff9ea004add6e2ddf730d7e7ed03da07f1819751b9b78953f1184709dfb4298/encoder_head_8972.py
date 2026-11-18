"""REQ-REPORT-8972: compare fresh heads without changing their frozen input basis."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.autoresearch import relational_graph_encoder as g

jax.config.update("jax_enable_x64", True)

Json = dict[str, Any]
b = g.b


def operands(state: Json, rows: list[Json]) -> Json:
    """Reuse fit-only confidence scaling so exposed labels cannot alter features."""
    data = g.pack(rows)
    data["c"] = jnp.asarray(b._standardize(np.asarray(data["c"]), state["scaling"]))
    return dict(data)


def features(state: Json, rows: list[Json]) -> Any:
    """An identity linear readout exposes the existing27 coordinates without new encoding."""
    params = dict(
        encoder={k: jnp.asarray(v) for k, v in state["params"]["encoder"].items()},
        head=(jnp.eye(27), jnp.zeros(27)),
    )
    return np.asarray(g.logits(params, operands(state, rows), True, False))


def head_logits(head: Any, x: Any, energy: bool) -> Any:
    """Use the same Gibbs forward path as the original joint-training experiment."""
    return g.logits(dict(encoder={}, head=head), dict(c=x), False, energy)


def head_loss(head: Any, x: Any, y: Any, energy: bool) -> Any:
    """Population log loss keeps the original unbalanced error prior."""
    return b.population_loss(head_logits(head, x, energy), y)


def full_loss(params: Json, data: Json, y: Any, energy: bool) -> Any:
    """Potential encoder derivatives measure sensitivity; frozen training never applies them."""
    return b.population_loss(g.logits(params, data, True, energy), y)


head_gradient = jax.jit(jax.grad(head_loss), static_argnames=("energy",))
full_gradient = jax.jit(jax.grad(full_loss), static_argnames=("energy",))


def norm(tree: Any) -> float:
    """Reduce all parameter leaves so a zero output layer does not hide later gradients."""
    return float(sum(np.sum(np.asarray(a, dtype=float) ** 2) for a in jax.tree.leaves(tree)) ** 0.5)


def gradient_row(params: Json, data: Json, y: Any, energy: bool, step: int) -> Json:
    """Record separate encoder and head norms at an actual optimization state."""
    grad = full_gradient(params, data, y, energy)
    return dict(
        step=step,
        encoder_gradient_norm=norm(grad["encoder"]),
        head_gradient_norm=norm(grad["head"]),
        fit_loss=float(full_loss(params, data, y, energy)),
        prediction_variance=float(np.var(b._sigmoid(g.logits(params, data, True, energy)))),
    )


def fit_head(
    state: Json, rows: list[Json], truth: list[int], energy: bool
) -> tuple[Json, list[Json]]:
    """Train only the new head; each seed starts from its original untrained head."""
    y = b.labels(truth, len(rows))
    if len(set(truth)) != 2:
        raise ValueError("training_class_support")
    data, x = operands(state, rows), jnp.asarray(features(state, rows))
    params = g.initial(state["seed"], True, energy)
    params["encoder"] = {k: jnp.asarray(v) for k, v in state["params"]["encoder"].items()}
    initial = params["head"]
    first = gradient_row(params, data, y, energy, 0)
    head = jax.tree.map(
        lambda p, grad: p - g.RATE * grad, initial, head_gradient(initial, x, y, energy)
    )
    second = gradient_row(dict(params, head=head), data, y, energy, 1)

    def update(_: Any, h: Any) -> Any:
        return jax.tree.map(lambda p, grad: p - g.RATE * grad, h, head_gradient(h, x, y, energy))

    b.progress(f"8972_before_head_seed{state['seed']}_energy{energy}", 1, 199)
    for start, stop in ((1, 50), (50, 100), (100, 150), (150, 200)):
        head = jax.jit(lambda h: jax.lax.fori_loop(start, stop, update, h))(head)
        jax.block_until_ready(head)
        b.progress("8972_head_steps", stop, 200 - stop)
    params["head"] = head
    final = gradient_row(params, data, y, energy, 200)
    serialized = jax.tree.map(lambda a: np.asarray(a).tolist(), params)
    serialized["head"] = list(serialized["head"])
    trained = dict(
        params=serialized,
        seed=state["seed"],
        relational=True,
        energy=energy,
        scaling=state["scaling"],
        steps=200,
        rate=0.01,
        loss_before=first["fit_loss"],
        loss_after=final["fit_loss"],
        initial_head=jax.tree.map(lambda a: np.asarray(a).tolist(), list(initial)),
    )
    b.progress("8972_after_head", 200, 0)
    return trained, [dict(r, actual_encoder_update_norm=0.0) for r in (first, second, final)]


def original_gradients(state: Json, rows: list[Json], truth: list[int]) -> list[Json]:
    """Measure the original joint path at0,1 and200; its first zero is not persistent proof."""
    y, data, energy = b.labels(truth, len(rows)), operands(state, rows), state["energy"]
    initial = g.initial(state["seed"], True, energy)
    first = gradient_row(initial, data, y, energy, 0)
    grad = full_gradient(initial, data, y, energy)
    second_params = jax.tree.map(lambda p, a: p - g.RATE * a, initial, grad)
    second = gradient_row(second_params, data, y, energy, 1)
    final_params = dict(
        encoder={k: jnp.asarray(v) for k, v in state["params"]["encoder"].items()},
        head=tuple(jnp.asarray(a) for a in state["params"]["head"]),
    )
    final = gradient_row(final_params, data, y, energy, 200)
    return [
        dict(
            r,
            prospective_encoder_step_norm=g.RATE * r["encoder_gradient_norm"],
            encoder_displacement_from_initial=norm(
                jax.tree.map(lambda p, q: p - q, final_params["encoder"], initial["encoder"])
            ),
        )
        for r in (first, second, final)
    ]
