"""Adaptive topology updates for KAEM/KAN univariate energy splines.

Spec refs: REQ-KAN-2005, SCENARIO-KAN-2005.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.models.kaem_energy import UnivariateKAEMLayer

SPEC_TRACES = ["REQ-KAN-2005", "SCENARIO-KAN-2005"]
ARTIFACT_SCHEMA = "carnot.adaptive_energy_landscapes_kan.v1"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_2005_adaptive_energy_landscapes_kan.json")

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "status",
    "experiment_id",
    "spec_traces",
    "run_date",
    "structural_change_metrics",
    "energy_probe",
    "adaptive_mesh_refinement_ready",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class KAEMStructuralChangeMetrics:
    """Serializable summary of one KAEM spline topology update."""

    n_knots_before: int
    n_knots_after: int
    knots_added: int
    knots_removed: int
    added_positions: list[float]
    removed_positions: list[float]
    knot_positions_before: list[float]
    knot_positions_after: list[float]
    complexity_scores: list[float]
    high_complexity_threshold: float
    low_complexity_threshold: float
    min_knots: int
    max_knots: int
    changed: bool
    spec_traces: list[str] = field(default_factory=lambda: list(SPEC_TRACES))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe topology metrics for artifacts and logs."""

        return {
            "spec_traces": list(self.spec_traces),
            "n_knots_before": int(self.n_knots_before),
            "n_knots_after": int(self.n_knots_after),
            "knots_added": int(self.knots_added),
            "knots_removed": int(self.knots_removed),
            "added_positions": _float_list(self.added_positions),
            "removed_positions": _float_list(self.removed_positions),
            "knot_positions_before": _float_list(self.knot_positions_before),
            "knot_positions_after": _float_list(self.knot_positions_after),
            "complexity_scores": _float_list(self.complexity_scores),
            "high_complexity_threshold": float(self.high_complexity_threshold),
            "low_complexity_threshold": float(self.low_complexity_threshold),
            "min_knots": int(self.min_knots),
            "max_knots": int(self.max_knots),
            "changed": bool(self.changed),
        }


def adaptive_mesh_refine_layer(
    layer: UnivariateKAEMLayer,
    *,
    high_complexity_threshold: float | None = None,
    low_complexity_threshold: float | None = None,
    min_knots: int = 3,
    max_knots: int = 64,
    max_additions: int = 2,
    max_removals: int = 2,
) -> KAEMStructuralChangeMetrics:
    """Refine or coarsen a KAEM layer's shared one-dimensional spline mesh.

    Local complexity is measured as the absolute change in slope at each
    interior knot, aggregated across variables. High-complexity interior knots
    trigger a new knot in the steepest adjacent interval. Low-complexity
    interior knots are removed when the layer is above ``min_knots``.

    The layer is mutated in place. Existing control points are interpolated
    onto the new knot positions so the current landscape is preserved as a
    warm start for later fitting.
    """

    _validate_refinement_bounds(min_knots, max_knots, max_additions, max_removals)

    old_knots = np.asarray(layer._knots, dtype=np.float64)
    old_ctrl = np.asarray(layer.control_points, dtype=np.float64)
    if old_ctrl.ndim != 2 or old_ctrl.shape[1] != old_knots.size:
        raise ValueError(
            "layer control_points must have shape (n_vars, n_knots) matching layer._knots"
        )

    n_before = int(old_knots.size)
    complexity = _aggregate_complexity(old_ctrl, old_knots)
    high_threshold, low_threshold = _resolve_thresholds(
        complexity,
        high_complexity_threshold,
        low_complexity_threshold,
    )

    added_positions = _select_added_positions(
        old_ctrl=old_ctrl,
        old_knots=old_knots,
        complexity=complexity,
        threshold=high_threshold,
        max_additions=max_additions,
        max_knots=max_knots,
    )
    removed_positions = _select_removed_positions(
        old_knots=old_knots,
        complexity=complexity,
        threshold=low_threshold,
        max_removals=max_removals,
        min_knots=min_knots,
    )

    new_knots = _merge_knot_updates(old_knots, added_positions, removed_positions)
    new_ctrl = np.vstack([np.interp(new_knots, old_knots, row) for row in old_ctrl])

    if new_knots.size != old_knots.size or not np.allclose(new_knots, old_knots):
        layer.n_knots = int(new_knots.size)
        layer._knots = jnp.array(new_knots, dtype=jnp.float32)
        layer.control_points = jnp.array(new_ctrl, dtype=jnp.float32)

    return KAEMStructuralChangeMetrics(
        n_knots_before=n_before,
        n_knots_after=int(new_knots.size),
        knots_added=len(added_positions),
        knots_removed=len(removed_positions),
        added_positions=_float_list(added_positions),
        removed_positions=_float_list(removed_positions),
        knot_positions_before=_float_list(old_knots),
        knot_positions_after=_float_list(new_knots),
        complexity_scores=_float_list(complexity),
        high_complexity_threshold=float(high_threshold),
        low_complexity_threshold=float(low_threshold),
        min_knots=int(min_knots),
        max_knots=int(max_knots),
        changed=bool(len(added_positions) or len(removed_positions)),
    )


def build_adaptive_energy_landscape_kan_artifact(
    *,
    run_date: str = "20260513",
    tests_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the deterministic Exp 2005 AMR evidence artifact."""

    layer = UnivariateKAEMLayer(n_vars=1, n_knots=5, key=jrandom.PRNGKey(2005))
    layer.control_points = jnp.array([[0.0, 0.05, 0.95, 0.98, 1.0]], dtype=jnp.float32)
    layer._knots = jnp.linspace(-1.0, 1.0, 5)
    layer.n_knots = 5

    probe_points = np.array([-0.75, -0.25, 0.25, 0.75], dtype=np.float32)
    before = _energy_probe(layer, probe_points)
    metrics = layer.adaptive_mesh_refine(
        high_complexity_threshold=1.0,
        low_complexity_threshold=0.05,
        min_knots=3,
        max_knots=8,
        max_additions=1,
        max_removals=1,
    )
    after = _energy_probe(layer, probe_points)
    finite_after = bool(np.all(np.isfinite(after)))
    ready = bool(metrics.knots_added >= 1 and metrics.knots_removed >= 1 and finite_after)

    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "status": "complete",
        "experiment_id": 2005,
        "spec_traces": list(SPEC_TRACES),
        "run_date": str(run_date),
        "structural_change_metrics": metrics.to_dict(),
        "energy_probe": {
            "probe_points": _float_list(probe_points),
            "energy_before": _float_list(before),
            "energy_after": _float_list(after),
            "finite_after_refinement": finite_after,
        },
        "adaptive_mesh_refinement_ready": ready,
        "tests_run": list(tests_run or []),
        "honest_verdict": (
            "complete: adaptive KAEM spline topology updated "
            f"with +{metrics.knots_added}/-{metrics.knots_removed} knots"
        ),
    }
    validate_adaptive_energy_landscape_kan_artifact(artifact)
    return artifact


def write_adaptive_energy_landscape_kan_artifact(
    *,
    output_path: str | Path = DEFAULT_ARTIFACT_PATH,
    run_date: str = "20260513",
    tests_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write the Exp 2005 adaptive energy landscape KAN artifact."""

    artifact = build_adaptive_energy_landscape_kan_artifact(
        run_date=run_date,
        tests_run=tests_run,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_adaptive_energy_landscape_kan_artifact(artifact: dict[str, Any]) -> None:
    """Fail fast if the Exp 2005 artifact drifts from the required schema."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise AssertionError(f"missing required fields: {sorted(missing)}")
    if artifact["schema"] != ARTIFACT_SCHEMA:
        raise AssertionError("schema mismatch")
    if artifact["status"] != "complete":
        raise AssertionError("status must be complete")
    if artifact["experiment_id"] != 2005:
        raise AssertionError("experiment_id must be 2005")
    if artifact["spec_traces"] != SPEC_TRACES:
        raise AssertionError("spec_traces must match REQ-KAN-2005 and SCENARIO-KAN-2005")

    metrics = artifact["structural_change_metrics"]
    if metrics["knots_added"] < 1 or metrics["knots_removed"] < 1:
        raise AssertionError("artifact must record both added and removed knots")
    if metrics["n_knots_after"] != metrics["n_knots_before"]:
        raise AssertionError("deterministic Exp 2005 probe should add and remove one knot")
    if artifact["adaptive_mesh_refinement_ready"] is not True:
        raise AssertionError("adaptive_mesh_refinement_ready must be true")
    if artifact["energy_probe"]["finite_after_refinement"] is not True:
        raise AssertionError("refined energy probe must remain finite")


def _aggregate_complexity(control_points: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Compute max per-variable slope-change magnitude at each interior knot."""

    if knots.size < 3:
        return np.array([], dtype=np.float64)
    intervals = np.diff(knots)
    if np.any(intervals <= 0.0):
        raise ValueError("layer._knots must be strictly increasing")
    slopes = np.diff(control_points, axis=1) / intervals.reshape(1, -1)
    slope_changes = np.abs(np.diff(slopes, axis=1))
    return np.nan_to_num(np.max(slope_changes, axis=0), nan=0.0, posinf=0.0, neginf=0.0)


def _select_added_positions(
    *,
    old_ctrl: np.ndarray,
    old_knots: np.ndarray,
    complexity: np.ndarray,
    threshold: float,
    max_additions: int,
    max_knots: int,
) -> list[float]:
    """Pick midpoint positions for high-complexity intervals."""

    if max_additions <= 0 or old_knots.size >= max_knots or complexity.size == 0:
        return []

    interval_scores = _interval_slope_magnitude(old_ctrl, old_knots)
    candidate_order = sorted(
        range(complexity.size),
        key=lambda idx: (-float(complexity[idx]), float(old_knots[idx + 1])),
    )
    additions: list[float] = []
    for score_idx in candidate_order:
        if complexity[score_idx] <= threshold:
            continue
        if old_knots.size + len(additions) >= max_knots or len(additions) >= max_additions:
            break
        knot_idx = score_idx + 1
        left_interval = knot_idx - 1
        right_interval = knot_idx
        preferred = (
            right_interval
            if interval_scores[right_interval] >= interval_scores[left_interval]
            else left_interval
        )
        for interval_idx in (preferred, left_interval, right_interval):
            pos = float((old_knots[interval_idx] + old_knots[interval_idx + 1]) / 2.0)
            if _is_new_position(pos, old_knots, additions):
                additions.append(pos)
                break
    return additions


def _select_removed_positions(
    *,
    old_knots: np.ndarray,
    complexity: np.ndarray,
    threshold: float,
    max_removals: int,
    min_knots: int,
) -> list[float]:
    """Pick smooth interior knot positions to remove."""

    if max_removals <= 0 or old_knots.size <= min_knots or complexity.size == 0:
        return []

    candidate_order = sorted(
        range(complexity.size),
        key=lambda idx: (float(complexity[idx]), abs(float(old_knots[idx + 1]))),
    )
    removals: list[float] = []
    for score_idx in candidate_order:
        if complexity[score_idx] >= threshold:
            continue
        if old_knots.size - len(removals) <= min_knots or len(removals) >= max_removals:
            break
        removals.append(float(old_knots[score_idx + 1]))
    return removals


def _interval_slope_magnitude(control_points: np.ndarray, knots: np.ndarray) -> np.ndarray:
    intervals = np.diff(knots)
    slopes = np.abs(np.diff(control_points, axis=1) / intervals.reshape(1, -1))
    return np.max(slopes, axis=0)


def _merge_knot_updates(
    old_knots: np.ndarray,
    added_positions: Sequence[float],
    removed_positions: Sequence[float],
) -> np.ndarray:
    removed = {_position_key(pos) for pos in removed_positions}
    kept = [float(pos) for pos in old_knots if _position_key(float(pos)) not in removed]
    merged = sorted(kept + [float(pos) for pos in added_positions])
    return np.array(merged, dtype=np.float64)


def _resolve_thresholds(
    complexity: np.ndarray,
    high_complexity_threshold: float | None,
    low_complexity_threshold: float | None,
) -> tuple[float, float]:
    if complexity.size:
        mean = float(np.mean(complexity))
        std = float(np.std(complexity))
    else:
        mean = 0.0
        std = 0.0
    high = mean + std if high_complexity_threshold is None else float(high_complexity_threshold)
    low = max(0.0, mean - std) if low_complexity_threshold is None else float(low_complexity_threshold)
    return high, low


def _validate_refinement_bounds(
    min_knots: int,
    max_knots: int,
    max_additions: int,
    max_removals: int,
) -> None:
    if min_knots < 2:
        raise ValueError("min_knots must be >= 2")
    if max_knots < min_knots:
        raise ValueError("max_knots must be >= min_knots")
    if max_additions < 0 or max_removals < 0:
        raise ValueError("max_additions and max_removals must be non-negative")


def _energy_probe(layer: UnivariateKAEMLayer, probe_points: np.ndarray) -> np.ndarray:
    return np.array(
        [float(layer.energy(jnp.array([point], dtype=jnp.float32))) for point in probe_points],
        dtype=np.float64,
    )


def _is_new_position(pos: float, old_knots: np.ndarray, additions: Sequence[float]) -> bool:
    key = _position_key(pos)
    old_keys = {_position_key(float(old)) for old in old_knots}
    add_keys = {_position_key(float(add)) for add in additions}
    return key not in old_keys and key not in add_keys


def _position_key(pos: float) -> int:
    return int(round(float(pos) * 1_000_000_000))


def _float_list(values: Sequence[float] | np.ndarray) -> list[float]:
    return [float(value) for value in values]
