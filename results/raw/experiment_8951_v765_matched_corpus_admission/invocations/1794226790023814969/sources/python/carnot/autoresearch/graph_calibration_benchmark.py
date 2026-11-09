"""Train a tiny selector from response confidence, with truth supplied separately.

The old calibration benchmark supplies the Gibbs/NCE classification pattern.
Its corpus and verifier labels are deliberately absent from this sibling.
REQ-REPORT-8937; SCENARIO-REPORT-8937-CONTRACT.
"""

from __future__ import annotations

import hashlib
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from carnot.models.gibbs import GibbsConfig, GibbsModel

Json = dict[str, Any]
Array: TypeAlias = NDArray[np.float64]
FEATURES = ("mean_token_nll", "first_answer_top2_margin")
SEED, STEPS, RATE = 764037, 200, 0.01
GRID = (0.5, 1.0, 2.0, 4.0)
ARMS = ("energy", "logistic", "prevalence", "nuisance")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush bounded work so the supervisor never mistakes compute for a stall."""
    print(f"[exp8937] phase={phase} completed={completed} pending={pending}", flush=True)


def features(rows: list[Json]) -> Array:
    """Exact keys prevent even rehashed oracle metadata from becoming an input."""
    if not rows or any(not isinstance(r, dict) or set(r) != set(FEATURES) for r in rows):
        raise ValueError("feature_schema")
    if any(type(r[k]) not in (int, float) for r in rows for k in FEATURES):
        raise ValueError("feature_numeric")
    x = np.asarray([[r[k] for k in FEATURES] for r in rows], dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("feature_range")
    return x


def nuisance(rows: list[Json]) -> Array:
    """Format validity is syntax only; distance-aware parser categories leak truth."""
    if not rows or any(
        set(r) != {"token_count", "format_valid"}
        or type(r["token_count"]) is not int
        or not 1 <= r["token_count"] <= 16
        or type(r["format_valid"]) is not bool
        for r in rows
    ):
        raise ValueError("nuisance_schema")
    return np.asarray([[r["token_count"], int(r["format_valid"])] for r in rows], dtype=float)


def labels(values: list[int], n: int) -> Array:
    """Reject malformed truth rather than silently coercing booleans or missing labels."""
    if len(values) != n or any(type(y) is not int or y not in (0, 1) for y in values):
        raise ValueError("binary_labels")
    return np.asarray(values, dtype=float)


def population_loss(logits: Any, y: Any) -> Any:
    """Empirical weighting restores the population prior absent from balanced NCE."""
    return jnp.mean(jax.nn.softplus(jnp.asarray(logits)) - jnp.asarray(y) * logits)


def _energy(params: Any, x: Any) -> Any:
    """Use the existing Gibbs forward pass, with explicit parameters for autodiff."""
    model = GibbsModel(GibbsConfig(input_dim=2, hidden_dims=[4]), jax.random.PRNGKey(SEED))
    model.layers = [(params[0], params[1])]
    model.output_weight, model.output_bias = params[2], params[3]
    return jax.vmap(model.energy)(x)


def _train(x: Array, y: Array, energy: bool) -> tuple[Any, Any]:
    """Full batches and fixed initialization remove label balancing and search choices."""
    model = GibbsModel(GibbsConfig(input_dim=2, hidden_dims=[4]), jax.random.PRNGKey(SEED))
    initial = (
        (*model.layers[0], model.output_weight, jnp.asarray(model.output_bias))
        if energy
        else jnp.zeros(x.shape[1] + 1)
    )

    def loss(params: Any) -> Any:
        z = _energy(params, jnp.asarray(x)) if energy else jnp.asarray(x) @ params[:-1] + params[-1]
        return population_loss(z, y)

    gradient = jax.jit(jax.grad(loss))
    params = initial
    progress("before_small_head_training", 0, STEPS)
    for step in range(STEPS):
        params = jax.tree.map(lambda p, g: p - RATE * g, params, gradient(params))
        if (step + 1) % 50 == 0:
            progress("small_head_training", step + 1, STEPS - step - 1)
    final, first = jax.tree.map(lambda a: np.asarray(a).tolist(), (params, initial))
    progress("after_small_head_training", STEPS, 0)
    return (list(final), list(first)) if energy else (final, first)


def _scaling(x: Array) -> Json:
    """A constant training feature keeps scale one and adds no artificial signal."""
    return dict(
        mean=x.mean(axis=0).tolist(), scale=np.where(x.std(axis=0) > 0, x.std(axis=0), 1).tolist()
    )


def _standardize(x: Array, scaling: Json) -> Array:
    """Reuse train operands on calibration and future rows without refitting."""
    return np.asarray((x - scaling["mean"]) / scaling["scale"], dtype=float)


def temperature(energy: Any, y: list[int]) -> float:
    """Ascending grid order makes exact Brier ties deterministic."""
    truth = labels(y, len(energy))
    return min(GRID, key=lambda t: float(np.mean((_sigmoid(np.asarray(energy) / t) - truth) ** 2)))


def _sigmoid(z: Any) -> Array:
    """Logaddexp avoids overflow at extreme finite logits."""
    return np.asarray(np.exp(-np.logaddexp(0, -np.asarray(z))), dtype=float)


def fit(
    train: list[Json],
    truth: list[int],
    train_nuisance: list[Json],
    calibration: list[Json],
    calibration_truth: list[int],
) -> Json:
    """The fitting API has no test operand, so test data cannot influence selection."""
    x, c, n = features(train), features(calibration), nuisance(train_nuisance)
    y, cy = labels(truth, len(x)), labels(calibration_truth, len(c))
    if len(n) != len(x) or len(set(y)) != 2 or len(set(cy)) != 2:
        raise ValueError("degenerate_class_or_roster")
    scaling, ns = _scaling(x), _scaling(n)
    x, c, n = _standardize(x, scaling), _standardize(c, scaling), _standardize(n, ns)
    e, ei = _train(x, y, True)
    logistic, li = _train(x, y, False)
    nuisance_state, ni = _train(n, y, False)
    return dict(
        energy=e,
        logistic=logistic,
        nuisance=nuisance_state,
        initial_states=dict(energy=ei, logistic=li, nuisance=ni),
        scaling=scaling,
        nuisance_scaling=ns,
        prevalence=float(y.mean()),
        temperature=temperature(_energy(tuple(jnp.asarray(v) for v in e), c), calibration_truth),
        numerical_training_calls=3 * STEPS,
    )


def score(state: Json, rows: list[Json], nuisance_rows: list[Json]) -> Json:
    """Only response confidence and a separate syntax view cross the scoring boundary."""
    x, n = features(rows), nuisance(nuisance_rows)
    if len(x) != len(n):
        raise ValueError("scoring_roster")
    shapes = [(4, 2), (4,), (4,), ()]
    if (
        len(state["energy"]) != 4
        or any(
            np.asarray(a).shape != shape or not np.isfinite(a).all()
            for a, shape in zip(state["energy"], shapes)
        )
        or any(
            np.asarray(state[k]).shape != (3,) or not np.isfinite(state[k]).all()
            for k in ("logistic", "nuisance")
        )
    ):
        raise ValueError("model_state")
    x, n = _standardize(x, state["scaling"]), _standardize(n, state["nuisance_scaling"])
    return dict(
        energy=_sigmoid(
            _energy(tuple(jnp.asarray(v) for v in state["energy"]), x) / state["temperature"]
        ).tolist(),
        logistic=_sigmoid(x @ state["logistic"][:2] + state["logistic"][2]).tolist(),
        nuisance=_sigmoid(n @ state["nuisance"][:2] + state["nuisance"][2]).tolist(),
        prevalence=[state["prevalence"]] * len(x),
    )


def decisions(probabilities: list[float]) -> list[str]:
    """Thresholds name actions; they provide no certified risk guarantee."""
    return ["accept" if p <= 0.1 else "reject" if p >= 0.9 else "escalate" for p in probabilities]


def half_selection(probabilities: Any, ids: list[str]) -> list[bool]:
    """Accept the most likely correct half, breaking confidence ties without labels."""
    p = np.asarray(probabilities)
    if (
        len(ids) != len(p)
        or len(p) == 0
        or len(p) % 2
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("half_coverage_requires_even_finite_roster")
    order = sorted(
        range(len(p)),
        key=lambda i: (p[i], hashlib.sha256(("8937/coverage/" + ids[i]).encode()).hexdigest(), i),
    )
    selected = set(order[: len(p) // 2])
    return [i in selected for i in range(len(p))]


def comparison_rows(
    probabilities: Json, truth: list[int], ids: list[str], clusters: list[str]
) -> list[Json]:
    """Keep every arm and unit so each headline can be reconstructed from rows."""
    y = labels(truth, len(ids))
    rows = []
    for arm in ARMS:
        p = probabilities[arm]
        selected = half_selection(p, ids)
        for i, decision in enumerate(decisions(p)):
            loss = float((p[i] - y[i]) ** 2)
            rows.append(
                dict(
                    unit_id=ids[i],
                    cluster_id=clusters[i],
                    seed=SEED,
                    arm=arm,
                    metric="brier",
                    value=loss,
                    numerator=loss,
                    denominator=1,
                    status="completed",
                    missing_reason=None,
                    p_error=p[i],
                    error=int(y[i]),
                    selected_50=selected[i],
                    decision=decision,
                    fixture_only=True,
                )
            )
    return rows


def _contrasts(prob: Array, y: Array, ids: list[str]) -> Array:
    """Use paired differences; each resample retains exactly half coverage per arm."""
    loss = np.mean((prob - y[:, None]) ** 2, axis=0)
    selected = [half_selection(prob[:, k], ids) for k in (0, 1)]
    risk = [float(y[mask].mean()) for mask in selected]
    return np.asarray([loss[2] - loss[0], loss[1] - loss[0], risk[0] - risk[1]])


def summarize(rows: list[Json]) -> Json:
    """Graph clusters, not repeated questions, are the independent bootstrap units."""
    grouped: dict[str, dict[str, Json]] = {}
    for row in rows:
        unit = grouped.setdefault(row["unit_id"], {})
        if row["arm"] in unit:
            raise ValueError("comparison_roster")
        unit[row["arm"]] = row
        loss = (row["p_error"] - row["error"]) ** 2
        if row["value"] != loss or row["numerator"] != loss or row["denominator"] != 1:
            raise ValueError("row_metric")
    if not grouped or any(set(r) != set(ARMS) for r in grouped.values()):
        raise ValueError("comparison_roster")
    ids = sorted(grouped)
    y = np.asarray([grouped[u]["energy"]["error"] for u in ids], dtype=float)
    prob = np.asarray([[grouped[u][a]["p_error"] for a in ARMS] for u in ids], dtype=float)
    clusters = [grouped[u]["energy"]["cluster_id"] for u in ids]
    names = sorted(set(clusters))
    groups = [np.flatnonzero(np.asarray(clusters) == name) for name in names]
    if any(len(g) != len(groups[0]) for g in groups) or len(ids) % 2:
        raise ValueError("equal_even_graph_roster")
    rng = np.random.default_rng(SEED)
    draws = np.empty((10000, 3))
    progress("before_graph_bootstrap", 0, 10000)
    for draw in range(10000):
        indexes = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws[draw] = _contrasts(prob[indexes], y[indexes], [ids[i] for i in indexes])
        if (draw + 1) % 2000 == 0:
            progress("graph_bootstrap", draw + 1, 9999 - draw)
    point = _contrasts(prob, y, ids)
    alpha = 0.05 / 3
    bounds = [
        float(np.quantile(draws[:, k], alpha if k < 2 else 1 - alpha, method="linear"))
        for k in range(3)
    ]
    contrasts = dict(
        zip(
            ("brier_vs_prevalence", "brier_vs_logistic", "accepted_error_vs_logistic"),
            [
                dict(
                    estimate=float(point[k]),
                    one_sided_bound=bounds[k],
                    direction="lower" if k < 2 else "upper",
                    passed=bounds[k] > 0 if k < 2 else bounds[k] <= 0,
                )
                for k in range(3)
            ],
        )
    )
    progress("after_graph_bootstrap", 10000, 0)
    return dict(
        draws=10000,
        alpha_per_contrast=alpha,
        independent_clusters=len(groups),
        contrasts=contrasts,
        benefit_passed=all(r["passed"] for r in contrasts.values()),
    )
