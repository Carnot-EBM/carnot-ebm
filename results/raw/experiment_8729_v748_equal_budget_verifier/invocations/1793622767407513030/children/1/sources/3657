"""Nonlinear repair helpers for Phase 3 continuous latent states.

Spec: REQ-KONA-029, SCENARIO-KONA-029
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class NonlinearProjectionResult:
    """Result of a HardNet++-style damped local-linear projection.

    **Researcher summary:**
        The helper repairs nonlinear verifier inequalities ``g(z) <= 0`` by
        repeatedly linearising them around the current latent state and solving
        a damped least-squares projection for the active violations. The
        verified-span metric tracks whether copy/decode-like coordinates that do
        not need repair were preserved instead of being over-projected.

    Spec: REQ-KONA-029, SCENARIO-KONA-029
    """

    state: np.ndarray
    initial_violation_energy: float
    violation_energy: float
    violation_count: int
    convergence_steps: int
    distortion_l2: float
    converged: bool
    verified_span_reuse: float
    final_step_norm: float


def measure_nonlinear_violation(
    scores: np.ndarray,
    *,
    tolerance: float = 0.0,
) -> tuple[float, int]:
    """Measure squared-hinge violation energy and hard violation count."""
    values = np.asarray(scores, dtype=np.float64)
    hinge = np.maximum(values, 0.0)
    return float(hinge @ hinge), int(np.sum(values > tolerance))


def verified_span_reuse(
    initial_state: np.ndarray,
    repaired_state: np.ndarray,
    verified_span_indices: Sequence[int] | None,
    *,
    copy_tolerance: float = 0.15,
) -> float:
    """Return the fraction of verified-span coordinates preserved by repair."""
    if verified_span_indices is None:
        return 1.0
    indices = list(verified_span_indices)
    if not indices:
        return 1.0
    initial = np.asarray(initial_state, dtype=np.float64)
    repaired = np.asarray(repaired_state, dtype=np.float64)
    deltas = np.abs(repaired[indices] - initial[indices])
    return float(np.mean(deltas <= copy_tolerance))


def hardnetpp_damped_projection(
    state: np.ndarray,
    constraint_fn: Callable[[np.ndarray], np.ndarray],
    jacobian_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_steps: int = 32,
    damping: float = 1e-3,
    step_size: float = 0.8,
    anchor_weight: float = 0.01,
    tolerance: float = 1e-8,
    verified_span_indices: Sequence[int] | None = None,
) -> NonlinearProjectionResult:
    """Repair nonlinear inequality violations by damped local-linear projection.

    Args:
        state: One-dimensional bounded latent vector.
        constraint_fn: Callable returning nonlinear inequality scores ``g(z)``;
            positive entries are violations.
        jacobian_fn: Callable returning the Jacobian of ``g`` at ``z``.
        n_steps: Maximum construct/refine projection iterations.
        damping: Positive Tikhonov damping for the local least-squares solve.
        step_size: Projection step multiplier.
        anchor_weight: Small pull toward the original state to limit distortion.
        tolerance: Squared-hinge convergence tolerance.
        verified_span_indices: Coordinates expected to be reusable copy/decode
            state rather than repaired verifier state.

    Returns:
        NonlinearProjectionResult with the repaired state and diagnostics.

    Raises:
        ValueError: If shapes or hyperparameters are malformed.

    Spec: REQ-KONA-029, SCENARIO-KONA-029
    """
    z0 = np.asarray(state, dtype=np.float64)
    if z0.ndim != 1:
        raise ValueError("state must be one-dimensional")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if damping <= 0.0:
        raise ValueError("damping must be positive")
    if step_size < 0.0:
        raise ValueError("step_size must be non-negative")
    if anchor_weight < 0.0:
        raise ValueError("anchor_weight must be non-negative")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    scores = np.asarray(constraint_fn(z0), dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("constraint_fn must return a one-dimensional array")
    jacobian = np.asarray(jacobian_fn(z0), dtype=np.float64)
    if jacobian.shape != (scores.shape[0], z0.shape[0]):
        raise ValueError("jacobian_fn must return shape (n_constraints, state_dim)")

    initial_energy, initial_count = measure_nonlinear_violation(
        scores,
        tolerance=tolerance,
    )
    z = z0.copy()
    final_energy = initial_energy
    final_count = initial_count
    steps_taken = 0
    final_step_norm = 0.0

    for step in range(1, n_steps + 1):
        if final_energy <= tolerance and final_count == 0:
            break

        scores = np.asarray(constraint_fn(z), dtype=np.float64)
        jacobian = np.asarray(jacobian_fn(z), dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError("constraint_fn must return a one-dimensional array")
        if jacobian.shape != (scores.shape[0], z.shape[0]):
            raise ValueError("jacobian_fn must return shape (n_constraints, state_dim)")

        active = scores > tolerance
        if not np.any(active):
            final_energy, final_count = measure_nonlinear_violation(
                scores,
                tolerance=tolerance,
            )
            break

        active_jacobian = jacobian[active]
        residual = scores[active] + max(np.sqrt(tolerance), 1e-6)
        gram = active_jacobian @ active_jacobian.T
        gram = gram + damping * np.eye(active_jacobian.shape[0], dtype=np.float64)
        multipliers = np.linalg.solve(gram, residual)
        correction = active_jacobian.T @ multipliers
        anchor_scale = final_energy / (initial_energy + 1e-12)
        correction = correction + (anchor_weight * anchor_scale * (z - z0))

        candidate = z.copy()
        candidate_energy = final_energy
        candidate_count = final_count
        scaled_step = step_size
        for _ in range(8):
            trial = np.clip(
                z - scaled_step * correction,
                -1.0 + 1e-12,
                1.0 - 1e-12,
            )
            trial_energy, trial_count = measure_nonlinear_violation(
                constraint_fn(trial),
                tolerance=tolerance,
            )
            if trial_energy <= final_energy + tolerance:
                candidate = trial
                candidate_energy = trial_energy
                candidate_count = trial_count
                break
            scaled_step *= 0.5

        final_step_norm = float(np.linalg.norm(candidate - z))
        z = candidate
        final_energy = candidate_energy
        final_count = candidate_count
        steps_taken = step

    return NonlinearProjectionResult(
        state=z,
        initial_violation_energy=initial_energy,
        violation_energy=final_energy,
        violation_count=final_count,
        convergence_steps=steps_taken,
        distortion_l2=float(np.linalg.norm(z - z0)),
        converged=final_energy <= tolerance and final_count == 0,
        verified_span_reuse=verified_span_reuse(
            z0,
            z,
            verified_span_indices,
        ),
        final_step_norm=final_step_norm,
    )
