"""Small normalized sentence-risk trainer for REQ-VERIFY-7755.

Each answer sentence sees visible and null evidence. One response risk is
formed from their support product, then paired views are averaged if present.
This module trains fixture heads; it does not certify natural truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.verify import source_alignment as alignment

jax.config.update("jax_enable_x64", True)

SEEDS = (67501, 67502, 67503, 67504, 67505, 67801, 67802, 67803)
LEARNING_RATES = (0.01, 0.05)
EPOCHS_MAX = 40
PARAMETER_MAX = 4096
TEMPERATURES = tuple(float(x) for x in np.geomspace(0.25, 4.0, 17))
ARMS = ("energy_local", "response_set", "logistic_local", "mlp_local")
MODES = ("canonical", "ordinary", "constrained")


def temperature_grid() -> list[float]:
    """Return the fixed response-logit calibration grid."""
    return list(TEMPERATURES)


def _one_view(rows: list[dict[str, Any]], key: str) -> dict[str, jax.Array]:
    views = [row[key] for row in rows]
    if any(view["abstention"] or not view["answer_units"] for view in views):
        raise ValueError("abstained fixture")
    units = max(len(view["answer_units"]) for view in views)
    places = max(len(view["windows"]) + 1 for view in views)
    x = np.zeros((len(rows), units, places, alignment.FEATURE_DIM), dtype=np.float64)
    prior = np.zeros((len(rows), places), dtype=np.float64)
    unit_mask = np.zeros((len(rows), units), dtype=np.float64)
    place_mask = np.zeros((len(rows), places), dtype=np.float64)
    for i, view in enumerate(views):
        groups = view["group_ids"]
        count = len(set(groups))
        weights = [1 / ((count + 1) * groups.count(group)) for group in groups]
        weights.append(1 / (count + 1))
        prior[i, : len(weights)] = weights
        place_mask[i, : len(weights)] = 1
        unit_mask[i, : len(view["answer_units"])] = 1
        for u, sentence in enumerate(view["answer_units"]):
            for loc, window in enumerate([*view["windows"], b""]):
                x[i, u, loc] = alignment.pair_features(window, [sentence])
    return {
        "x": jnp.asarray(x),
        "prior": jnp.asarray(prior),
        "unit_mask": jnp.asarray(unit_mask),
        "place_mask": jnp.asarray(place_mask),
    }


def prepare_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad public features while keeping labels outside both view builders."""
    if not rows:
        raise ValueError("empty fixture")
    a = _one_view(rows, "view_a")
    b = _one_view(rows, "view_b")
    if a["unit_mask"].shape != b["unit_mask"].shape:
        raise ValueError("view sentence mismatch")
    known = np.full(a["unit_mask"].shape, -1, dtype=np.float64)
    for index, row in enumerate(rows):
        labels = row["known"]
        known[index, : len(labels)] = labels
    return {
        "a": a,
        "b": b,
        "known": jnp.asarray(known),
        "label": jnp.asarray([row["label"] for row in rows], dtype=jnp.float64),
    }


def init_params(arm: str, seed: int) -> dict[str, jax.Array]:
    """Initialize the registered shared head with seeded nonzero weights."""
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("unregistered arm or seed")
    rng = np.random.default_rng(seed)
    if arm == "mlp_local":
        return {
            "w1": jnp.asarray(rng.normal(0, 0.02, (alignment.FEATURE_DIM, 16))),
            "b1": jnp.zeros(16),
            "w2": jnp.asarray(rng.normal(0, 0.02, (16, 2))),
            "b2": jnp.zeros(2),
        }
    return {"w": jnp.asarray(rng.normal(0, 0.02, (alignment.FEATURE_DIM, 2))), "b": jnp.zeros(2)}


def parameter_count(params: dict[str, jax.Array], dual_variables: int = 0) -> int:
    """Count every fitted scalar, including registered dual variables."""
    return sum(int(value.size) for value in params.values()) + dual_variables


def energy_parameters(params: dict[str, jax.Array]) -> dict[str, Any]:
    """Convert the JAX head to the original finite-energy parameter form."""
    return {"weights": np.asarray(params["w"]).T.tolist(), "bias": np.asarray(params["b"]).tolist()}


def sentence_support(
    params: dict[str, jax.Array], view: dict[str, jax.Array], arm: str
) -> jax.Array:
    """Normalize binary energies or a shared control head per sentence."""
    x, prior = view["x"], view["prior"]
    if arm in ("energy_local", "response_set"):
        logits = (
            jnp.log(jnp.maximum(prior, 1e-300))[:, None, :, None]
            - jnp.einsum("buld,dc->bulc", x, params["w"])
            - params["b"]
        )
        logits = jnp.where(view["place_mask"][:, None, :, None] > 0, logits, -1e30)
        support = jnp.exp(
            jax.scipy.special.logsumexp(logits[..., 1], axis=2)
            - jax.scipy.special.logsumexp(logits, axis=(2, 3))
        )
    else:
        pooled = jnp.einsum("buld,bl->bud", x, prior)
        if arm == "logistic_local":
            logits = pooled @ params["w"] + params["b"]
        elif arm == "mlp_local":
            logits = jnp.tanh(pooled @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]
        else:
            raise ValueError("unknown arm")
        support = jax.nn.softmax(logits, axis=-1)[..., 1]
    return jnp.where(view["unit_mask"] > 0, support, 1.0)


def predict(params: dict[str, jax.Array], view: dict[str, jax.Array], arm: str) -> jax.Array:
    """Return raw unsupported response risk from the sentence product."""
    return 1 - jnp.prod(sentence_support(params, view, arm), axis=1)


def deployed_risk(
    params: dict[str, jax.Array], batch: dict[str, Any], arm: str, paired: bool
) -> jax.Array:
    """Average raw paired risks once; canonical predictions use view A."""
    a = predict(params, batch["a"], arm)
    return (a + predict(params, batch["b"], arm)) / 2 if paired else a


def _ce(probability: jax.Array, label: jax.Array) -> jax.Array:
    p = jnp.clip(probability, 1e-6, 1 - 1e-6)
    return -label * jnp.log(p) - (1 - label) * jnp.log1p(-p)


def _local_loss(
    params: dict[str, jax.Array], batch: dict[str, Any], arm: str, key: str
) -> jax.Array:
    known = batch["known"]
    mask = (known >= 0) * batch[key]["unit_mask"]
    support = sentence_support(params, batch[key], arm)
    per_sentence = _ce(support, jnp.maximum(known, 0)) * mask
    return jnp.mean(jnp.sum(per_sentence, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1))


def constraints(
    params: dict[str, jax.Array], batch: dict[str, Any], arm: str
) -> tuple[jax.Array, jax.Array]:
    """Return symmetric Bernoulli KL/2 and second-view label CE."""
    p = jnp.clip(predict(params, batch["a"], arm), 1e-6, 1 - 1e-6)
    q = jnp.clip(predict(params, batch["b"], arm), 1e-6, 1 - 1e-6)
    kl_pq = p * jnp.log(p / q) + (1 - p) * jnp.log((1 - p) / (1 - q))
    kl_qp = q * jnp.log(q / p) + (1 - q) * jnp.log((1 - q) / (1 - p))
    return jnp.mean((kl_pq + kl_qp) / 2), jnp.mean(_ce(q, batch["label"]))


def dual_step(duals: tuple[float, float], observed: tuple[float, float]) -> tuple[float, float]:
    """Project ascent on the two frozen constraint violations."""
    return (
        min(10.0, max(0.0, duals[0] + 0.01 * (observed[0] - 0.01))),
        min(10.0, max(0.0, duals[1] + 0.01 * (observed[1] - 0.70))),
    )


def loss(
    params: dict[str, jax.Array],
    batch: dict[str, Any],
    arm: str,
    mode: str,
    duals: tuple[float, float],
) -> jax.Array:
    """Train on the deployed response risk plus available local targets."""
    if mode not in MODES:
        raise ValueError("unknown training mode")
    paired = mode != "canonical"
    response = jnp.mean(_ce(deployed_risk(params, batch, arm, paired), batch["label"]))
    if arm != "response_set":
        local_a = _local_loss(params, batch, arm, "a")
        response += (local_a + _local_loss(params, batch, arm, "b")) / 2 if paired else local_a
    if mode == "constrained":
        divergence, second_ce = constraints(params, batch, arm)
        response += duals[0] * (divergence - 0.01) + duals[1] * (second_ce - 0.70)
    return response


_VALUE_GRAD = jax.jit(jax.value_and_grad(loss), static_argnames=("arm", "mode"))


def gradient_error(params: dict[str, jax.Array], batch: dict[str, Any], arm: str) -> float:
    """Compare one response/local bias derivative with a central difference."""
    epsilon = 1e-4
    gradient = jax.grad(loss)(params, batch, arm, "canonical", (0.0, 0.0))
    bias_key = "b2" if arm == "mlp_local" else "b"
    plus = {**params, bias_key: params[bias_key].at[0].add(epsilon)}
    minus = {**params, bias_key: params[bias_key].at[0].add(-epsilon)}
    finite = (
        loss(plus, batch, arm, "canonical", (0.0, 0.0))
        - loss(minus, batch, arm, "canonical", (0.0, 0.0))
    ) / (2 * epsilon)
    return abs(float(finite - gradient[bias_key][0]))


def response_nll(
    params: dict[str, jax.Array], batch: dict[str, Any], arm: str, paired: bool
) -> float:
    """Select on uncalibrated deployed response NLL only."""
    return float(jnp.mean(_ce(deployed_risk(params, batch, arm, paired), batch["label"])))


def _parameter_hash(params: dict[str, jax.Array]) -> str:
    payload = {key: np.asarray(value).tolist() for key, value in sorted(params.items())}
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fit(
    arm: str,
    train: dict[str, Any],
    tune: dict[str, Any],
    seed: int,
    learning_rate: float,
    epochs: int,
    mode: str,
    *,
    initial: dict[str, jax.Array] | None = None,
) -> dict[str, Any]:
    """Run bounded full-batch descent and retain every measured epoch."""
    if (
        seed not in SEEDS
        or learning_rate not in LEARNING_RATES
        or not 1 <= epochs <= EPOCHS_MAX
        or mode not in MODES
    ):
        raise ValueError("unregistered training budget")
    params = initial if initial is not None else init_params(arm, seed)
    if parameter_count(params, 2 if mode == "constrained" else 0) > PARAMETER_MAX:
        raise ValueError("parameter budget")
    initial_hash = _parameter_hash(params)
    duals = (0.0, 0.0)
    curve = []
    for epoch in range(epochs):
        value, gradient = _VALUE_GRAD(params, train, arm, mode, duals)
        norm = float(jnp.sqrt(sum(jnp.sum(g * g) for g in gradient.values())))
        if not np.isfinite(float(value)) or not np.isfinite(norm):
            raise ValueError("nonfinite training loss or gradient")
        params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, gradient)
        if mode == "constrained":
            observed = constraints(params, train, arm)
            duals = dual_step(duals, (float(observed[0]), float(observed[1])))
        curve.append(
            {"epoch": epoch + 1, "loss": float(value), "gradient_norm": norm, "dual": list(duals)}
        )
    return {
        "arm": arm,
        "mode": mode,
        "seed": seed,
        "learning_rate": learning_rate,
        "params": params,
        "curve": curve,
        "initial_hash": initial_hash,
        "final_hash": _parameter_hash(params),
        "duals": list(duals),
        "tune_nll": response_nll(params, tune, arm, mode != "canonical"),
    }


def temperature_risk(probability: float, temperature: float) -> float:
    """Scale the one aggregated response logit after raw view averaging."""
    if not 0 < probability < 1:
        return probability
    odds = np.log(probability) - np.log1p(-probability)
    return float(jax.nn.sigmoid(odds / temperature))


def calibrate(params: dict[str, jax.Array], tune: dict[str, Any], arm: str, paired: bool) -> float:
    """Fit temperature on the same aggregated tune risks used at deployment."""
    risks = np.asarray(deployed_risk(params, tune, arm, paired))
    labels = np.asarray(tune["label"])
    return min(
        TEMPERATURES,
        key=lambda t: float(
            np.mean(
                [
                    float(_ce(jnp.asarray(temperature_risk(float(p), t)), jnp.asarray(y)))
                    for p, y in zip(risks, labels, strict=True)
                ]
            )
        ),
    )


def action(probability: float) -> str:
    """Minimize the frozen accept, reject and escalate costs; ties escalate."""
    costs = {"accept": 5 * probability, "reject": 1 - probability, "escalate": 0.25}
    return min(("escalate", "accept", "reject"), key=lambda name: costs[name])


def decide(head: dict[str, Any], batch: dict[str, Any], index: int, paired: bool) -> dict[str, Any]:
    """Return one calibrated response risk and one typed decision."""
    raw = float(deployed_risk(head["params"], batch, head["arm"], paired)[index])
    calibrated = temperature_risk(raw, float(head["temperature"]))
    return {"raw_risk": raw, "probability_unsupported": calibrated, "action": action(calibrated)}


def save(path: Path, head: dict[str, Any]) -> None:
    """Save parameters atomically so a cold process sees a complete head."""
    data = {key: value for key, value in head.items() if key != "params"}
    data["params"] = {key: np.asarray(value).tolist() for key, value in head["params"].items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True))
    temporary.replace(path)


def load(path: Path) -> dict[str, Any]:
    """Rebuild the numeric head from saved JSON without model loading."""
    data = json.loads(path.read_text())
    data["params"] = {key: jnp.asarray(value) for key, value in data["params"].items()}
    return data
