"""REQ-REPORT-8919: change candidate learning without weakening its promotion guard.

The gradient still comes from the existing normalized three-class objective.
Both sequences are explicit JSON values so restart cannot invent optimizer state.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from carnot.memory import signed_memory_8892 as signed
from carnot.memory.causal_memory_8802 import LABELS
from carnot.reporting.causal_learning_audit_8810 import require

Json = dict[str, Any]
SCHEMA = "signed-interpolated-v1"
ALPHA, BETA, ETA = 0.25, 0.5, 0.1


def advance(fast: Any, slow: Any, gradient: Any, alpha: float = ALPHA) -> tuple[Any, Any]:
    """Keep the fast step anchored at fast, even though its gradient was measured elsewhere."""
    require(
        fast.shape == slow.shape == gradient.shape
        and all(np.all(np.isfinite(a)) for a in (fast, slow, gradient))
        and np.isfinite(alpha)
        and 0 <= alpha <= 1,
        "nonfinite_or_shape",
    )
    next_fast = fast - ETA * gradient
    next_slow = slow + alpha * (next_fast - slow)
    require(
        bool(np.all(np.isfinite(next_fast)) and np.all(np.isfinite(next_slow))), "nonfinite_update"
    )
    return next_fast, next_slow


def validate(state: Json) -> None:
    """Reject mixed feature schemas and invalid counters before a checkpoint is used."""
    for name in ("fast", "slow"):
        signed.vector(state[name], [0.0] * 8)
    require(
        type(state["update_count"]) is int
        and state["update_count"] >= 0
        and {k: v for k, v in state["fast"].items() if k != "weights"}
        == {k: v for k, v in state["slow"].items() if k != "weights"},
        "optimizer_schema",
    )


def update(
    state: Json, counts: list[float], label: str, *, alpha: float = ALPHA, beta: float = BETA
) -> tuple[Json, Json]:
    """Take exactly one existing-objective gradient, then submit the slow sequence."""
    validate(state)
    require(label in LABELS and np.isfinite(beta) and 0 <= beta <= 1, "optimizer_input")
    x = jnp.asarray(signed.vector(state["fast"], counts), dtype=jnp.float32)
    fast, slow = (np.asarray(state[n]["weights"][0]) for n in ("fast", "slow"))
    interpolated = (1 - beta) * fast + beta * slow
    _, loss, grad = signed.step(
        jnp.asarray(interpolated, dtype=jnp.float32), x, jnp.asarray(LABELS.index(label))
    )
    gradient = np.asarray(grad)
    next_fast, next_slow = advance(fast, slow, gradient, alpha)
    require(np.isfinite(float(loss)), "nonfinite_loss")
    result = dict(
        fast=dict(state["fast"], weights=[next_fast.tolist()]),
        slow=dict(state["slow"], weights=[next_slow.tolist()]),
        update_count=state["update_count"] + 1,
    )
    return result, dict(
        loss=float(loss),
        gradient=gradient.tolist(),
        gradient_norm=float(np.linalg.norm(gradient)),
        parameter_delta=float(np.linalg.norm(next_slow - slow)),
        optimizer_steps=1,
    )
