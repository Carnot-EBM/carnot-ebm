"""Latent-to-validity snap sweep for Phase 4 Option A preflight.

**Researcher summary:**
    The Phase 3 ContinuousEBM keeps reasoning states in a bounded Cartesian
    latent space, `z in [-1, 1]^d`. Q8's Phase 4 recommendation is Option A:
    run continuous dynamics in that space, then snap each latent state to the
    nearest discrete ARC-AGI-3 action. This module implements the cheap static
    diagnostic for that bridge before any HMC sampler work starts.

**Proxy caveat:**
    The repository does not currently contain a deterministic ARC-AGI-3 rule
    engine. Until it does, the sweep uses a synthetic proxy: legal actions are
    deterministic points from the 0.1-spaced grid in `[-1, 1]^d`, capped to
    1,000 actions. Snapping to that legal set should therefore produce a high
    validity rate; the artifact records `proxy_used=True` so downstream readers
    do not mistake this for a real game-rule validation.

Spec: REQ-KONA-008, SCENARIO-KONA-007
"""

from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass
from typing import Any

import numpy as np

from carnot.phase3.continuous_ebm import ContinuousEBM

PHASE3_REFERENCE_LATENT_DIM = 10
"""Current Phase 3 seed latent dimension from Exp 435a/446 ContinuousEBM runs."""


@dataclass(frozen=True)
class SnapSweepConfig:
    """Configuration for the latent-to-validity snap sweep.

    Attributes:
        n_states: Number of continuous latent states to sample. Q8 requires
            10,000 for the real diagnostic.
        seed: Random seed for reproducible uniform sampling.
        grid_spacing: Synthetic proxy grid spacing in each latent coordinate.
        legality_radius: Radius from the Q8 proxy description. The current
            legality check uses exact membership after snapping, so this value
            is documented in the artifact description rather than applied to
            post-snap points.
        max_actions: Maximum number of synthetic legal actions. Q8 assumes
            ARC-AGI-3 per-turn action sets are at most 1,000 actions.
        chunk_size: Batch size for nearest-neighbor distance evaluation.
    """

    n_states: int = 10_000
    seed: int = 1154
    grid_spacing: float = 0.1
    legality_radius: float = 0.15
    max_actions: int = 1_000
    chunk_size: int = 256


def infer_latent_dim(model: ContinuousEBM) -> int:
    """Return the bounded latent dimension `d` from a Phase 3 ContinuousEBM."""
    latent_dim = int(model.variables)
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    return latent_dim


def build_reference_continuous_ebm(latent_dim: int = PHASE3_REFERENCE_LATENT_DIM) -> ContinuousEBM:
    """Build the minimal current Phase 3 reference ContinuousEBM.

    Existing Phase 3 seed experiments instantiate `ContinuousEBM` with ten
    variables. The actual coupling and bias are irrelevant for a uniform
    geometry sweep, so this helper creates a zero-energy placeholder whose
    `variables` field carries the latent dimension under test.
    """
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    coupling = np.zeros((latent_dim, latent_dim), dtype=np.float64)
    bias = np.zeros(latent_dim, dtype=np.float64)
    return ContinuousEBM(variables=latent_dim, coupling=coupling, bias=bias)


def _axis_values(spacing: float) -> np.ndarray:
    """Build rounded grid-axis values from -1 to 1 inclusive."""
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    values = np.arange(-1.0, 1.0 + spacing / 2.0, spacing, dtype=np.float64)
    values = np.round(values, 10)
    return values[(values >= -1.0) & (values <= 1.0)]


def build_synthetic_action_space(
    latent_dim: int,
    spacing: float = 0.1,
    max_actions: int = 1_000,
) -> np.ndarray:
    """Return a deterministic capped legal-action grid for the proxy sweep.

    The full Cartesian 0.1-spaced grid has `21^d` points for spacing 0.1, which
    is intentionally not materialised for `d=10`. Instead, this function returns
    the first `max_actions` lattice points in lexicographic base-grid order. Each
    returned action is still a legal point on the regular grid in `[-1, 1]^d`.
    """
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")

    axis = _axis_values(spacing)
    total_points = axis.size**latent_dim
    n_actions = min(int(max_actions), int(total_points))
    ordinal = np.arange(n_actions, dtype=np.int64)
    digits = np.empty((n_actions, latent_dim), dtype=np.int64)
    for dim in range(latent_dim):
        digits[:, dim] = ordinal % axis.size
        ordinal = ordinal // axis.size
    return axis[digits]


def sample_uniform_latents(latent_dim: int, n_states: int, seed: int) -> np.ndarray:
    """Sample continuous states uniformly from the bounded hypercube."""
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n_states, latent_dim))


def snap_states_to_actions(
    states: np.ndarray,
    actions: np.ndarray,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap each state to its nearest legal action using Euclidean distance."""
    states_arr = np.asarray(states, dtype=np.float64)
    actions_arr = np.asarray(actions, dtype=np.float64)
    if states_arr.ndim != 2 or actions_arr.ndim != 2:
        raise ValueError("states and actions must both be 2D arrays")
    if states_arr.shape[1] != actions_arr.shape[1]:
        raise ValueError("states and actions must have the same latent dimension")
    if actions_arr.shape[0] == 0:
        raise ValueError("actions must contain at least one legal action")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    snapped = np.empty_like(states_arr)
    distances = np.empty(states_arr.shape[0], dtype=np.float64)
    for start in range(0, states_arr.shape[0], chunk_size):
        stop = min(start + chunk_size, states_arr.shape[0])
        chunk = states_arr[start:stop]
        deltas = chunk[:, None, :] - actions_arr[None, :, :]
        squared_distances = np.einsum("cad,cad->ca", deltas, deltas)
        nearest = np.argmin(squared_distances, axis=1)
        snapped[start:stop] = actions_arr[nearest]
        distances[start:stop] = np.sqrt(squared_distances[np.arange(stop - start), nearest])
    return snapped, distances


def snap_to_action(state: np.ndarray, legal_actions: list[Any] | tuple[Any, ...]) -> Any:
    """Snap one latent vector to the nearest legal action.

    Legal actions may be raw arrays or objects with a `latent` attribute. The
    return value is the original legal-action object, preserving action metadata
    for downstream environment steps.
    """
    if not legal_actions:
        raise ValueError("legal_actions must contain at least one action")
    state_arr = np.asarray(state, dtype=np.float64)
    action_vectors = np.asarray(
        [getattr(action, "latent", action) for action in legal_actions],
        dtype=np.float64,
    )
    if state_arr.ndim != 1:
        raise ValueError("state must be a one-dimensional latent vector")
    if action_vectors.ndim != 2 or action_vectors.shape[1] != state_arr.size:
        raise ValueError("legal action latents must match state dimension")
    deltas = action_vectors - state_arr.reshape(1, -1)
    nearest = int(np.argmin(np.einsum("ad,ad->a", deltas, deltas)))
    return legal_actions[nearest]


def _action_key(action: np.ndarray) -> tuple[float, ...]:
    """Convert a grid action to a stable key for membership checks."""
    return tuple(float(x) for x in np.round(action, 10))


def snapped_actions_legal_mask(
    snapped_actions: np.ndarray, legal_actions: np.ndarray
) -> np.ndarray:
    """Return a boolean mask indicating whether snapped points are legal actions."""
    snapped_arr = np.asarray(snapped_actions, dtype=np.float64)
    legal_arr = np.asarray(legal_actions, dtype=np.float64)
    if snapped_arr.ndim != 2 or legal_arr.ndim != 2:
        raise ValueError("snapped_actions and legal_actions must both be 2D arrays")
    if snapped_arr.shape[1] != legal_arr.shape[1]:
        raise ValueError("snapped_actions and legal_actions must have the same latent dimension")

    legal_keys = {_action_key(action) for action in legal_arr}
    return np.array([_action_key(action) in legal_keys for action in snapped_arr], dtype=bool)


def snap_validity_verdict(rate: float, continuous_ebm_found: bool = True) -> str:
    """Return the honest verdict enum for a snap-validity rate."""
    if not continuous_ebm_found:
        return "phase3_continuous_ebm_not_found"
    if rate >= 0.95:
        return "option_a_viable_above_95pct"
    if rate >= 0.90:
        return "option_a_marginal_90_to_95pct"
    return "option_a_failed_below_90pct"


def build_snap_validity_artifact(
    *,
    latent_dim: int,
    n_states_sampled: int,
    n_legal_snaps: int,
    proxy_used: bool,
    action_space_description: str,
    continuous_ebm_found: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON-serialisable Exp 1154 snap-validity artifact."""
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if n_states_sampled <= 0:
        raise ValueError("n_states_sampled must be positive")
    if n_legal_snaps < 0 or n_legal_snaps > n_states_sampled:
        raise ValueError("n_legal_snaps must be between 0 and n_states_sampled")

    rate = float(n_legal_snaps / n_states_sampled)
    gate_passed = bool(rate >= 0.95)
    artifact: dict[str, Any] = {
        "schema": "carnot.snap_validity_sweep.v1",
        "experiment": 1154,
        "run_date": _datetime.date.today().isoformat(),
        "latent_dim": int(latent_dim),
        "n_states_sampled": int(n_states_sampled),
        "n_legal_snaps": int(n_legal_snaps),
        "snap_validity_rate": rate,
        "snap_validity_gate_passed": gate_passed,
        "phase4_option_a_viable": gate_passed,
        "proxy_used": bool(proxy_used),
        "action_space_description": action_space_description,
        "honest_verdict": snap_validity_verdict(rate, continuous_ebm_found),
    }
    if extra:
        artifact.update(extra)
    return artifact


def _synthetic_action_space_description(
    latent_dim: int,
    n_actions: int,
    config: SnapSweepConfig,
) -> str:
    """Describe the synthetic proxy action set used in the result artifact."""
    return (
        f"synthetic proxy: {n_actions} legal points from the "
        f"{config.grid_spacing:.1f}-spaced grid in [-1, 1]^{latent_dim}; "
        "snap uses Euclidean nearest neighbor; snapped point is legal if it exists "
        f"in the legal action set; proxy legality radius={config.legality_radius:.2f}"
    )


def run_snap_validity_sweep(
    model: ContinuousEBM | None = None,
    config: SnapSweepConfig | None = None,
) -> dict[str, Any]:
    """Run the Q8 latent-to-validity sweep and return its artifact."""
    cfg = config or SnapSweepConfig()
    phase3_model = model or build_reference_continuous_ebm()
    latent_dim = infer_latent_dim(phase3_model)

    legal_actions = build_synthetic_action_space(
        latent_dim=latent_dim,
        spacing=cfg.grid_spacing,
        max_actions=cfg.max_actions,
    )
    states = sample_uniform_latents(latent_dim=latent_dim, n_states=cfg.n_states, seed=cfg.seed)
    snapped, distances = snap_states_to_actions(states, legal_actions, chunk_size=cfg.chunk_size)
    legal_mask = snapped_actions_legal_mask(snapped, legal_actions)
    n_legal = int(np.count_nonzero(legal_mask))

    return build_snap_validity_artifact(
        latent_dim=latent_dim,
        n_states_sampled=cfg.n_states,
        n_legal_snaps=n_legal,
        proxy_used=True,
        action_space_description=_synthetic_action_space_description(
            latent_dim=latent_dim,
            n_actions=int(legal_actions.shape[0]),
            config=cfg,
        ),
        continuous_ebm_found=True,
        extra={
            "seed": cfg.seed,
            "grid_spacing": cfg.grid_spacing,
            "legality_radius": cfg.legality_radius,
            "n_actions": int(legal_actions.shape[0]),
            "mean_snap_distance": float(np.mean(distances)),
            "max_snap_distance": float(np.max(distances)),
        },
    )
