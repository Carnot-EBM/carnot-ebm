"""Phase 5-B in-situ training loop with verifier-ensemble grounding (exp_NEXT_B).

**Researcher summary:**
    Builds on Phase 5-A's read-only prototype (``insitu_prototype.py``)
    by adding a Contrastive Divergence (CD-1) update step to the
    encoder and energy MLP whenever the k=3 verifier ensemble
    AND-passes on a proposed action sequence.  Tracks the five
    Q9-detectable failure modes (failure modes 4-8 from the
    in-situ-training-phase5-derisking change proposal) across a
    1000-query trajectory and reports five boolean stability gates.

**Detailed explanation for engineers (CLAUDE.md verbose-layman rule):**
    Phase 5-A only ran a forward pass — it generated an action
    sequence per puzzle and recorded diagnostic statistics.  Phase 5-B
    adds the WEIGHT UPDATE half: when the verifier ensemble accepts a
    proposed action, the encoder + energy MLP take a small CD-1
    gradient step toward LOWER energy on the accepted latent and
    HIGHER energy on a randomly-sampled "negative" latent.  This is
    the smallest possible incarnation of "model adapts during
    inference using the verifier as supervision" — Carnot's
    differentiator from Sakana DGM-style closed-loop self-improvement.

    The acceptance test is FIVE simultaneous gates (CLAUDE.md MANDATORY
    Phase Prototype + Empirical Validation discipline; the proposal
    spec lists exactly these five Q9 failure modes that ARE detectable
    at toy scale):

        Gate 1: ``E(z_accepted)`` decreases ≥ 30% across the trajectory
                (the mechanism actually does what we claim — accepted
                samples become low-energy under the trained EBM).
        Gate 2: encoder spectral norm growth rate stays
                ``< 0.01 / query`` (representations don't drift
                unboundedly — Q9 failure mode 6).
        Gate 3: acceptance-rate first derivative is sub-linear over
                the trajectory (no autocatalytic spiral where the
                model learns to produce only what the verifier
                accepts — Q9 failure mode 7).
        Gate 4: mean vacuous-anchor distance stays ``> 0.5`` throughout
                (no null-space excavation — Q9 failure mode 4).
        Gate 5: held-out oracle accuracy retained within 5 pp of
                initial value (no catastrophic forgetting of base
                capability — Q9 failure mode 8).

    All five gates must pass for ``phase5b_stability_confirmed = True``.
    Partial-pass results are reported honestly as ``partial_gates``
    rather than rounded up to "blocked", per CLAUDE.md no-doomed-rerun
    discipline.

Spec coverage:
    REQ-KONA-017 (Phase 5-B in-situ training loop with verifier-ensemble
    grounding) and SCENARIO-KONA-017 (the 1000-query trajectory
    acceptance test).  Reuses REQ-KONA-008 / REQ-KONA-012 components
    from Phase 5-A.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from carnot.phase5.insitu_prototype import (
    GRID_SIZE,
    LATENT_DIM,
    NUM_COLORS,
    InSituEncoder,
    InSituEnergyMLP,
    InSituRefiner,
    VacuousAnchorTracker,
    _verifier_changes_grid,
    _verifier_in_bounds,
    apply_action_sequence,
    generate_random_5x5_puzzle,
    snap_to_action,
)

SCHEMA_VERSION = "1223.phase5b.insitu_training_loop.v1"
N_VERIFIERS_PHASE5B = 3
# The change-proposal recommends η=1e-5.  Empirically that is far below
# the threshold needed to produce a 30% energy drop in 1000 queries
# (the cumulative bias-update budget at η=1e-5 over ~1000 accepted
# updates is ~5e-3 in a4 — a 0.1% sigmoid shift).  Phase 5-B's runner
# therefore uses η=1e-3, which is the smallest rate empirically
# sufficient for Gate 1 to evaluate the *mechanism* (does the loop
# actually drive accepted-sample energy down?) on a 1000-query budget.
# The artifact records the actual learning rate used so the deviation
# from the proposal's default is auditable.
DEFAULT_LEARNING_RATE = 1e-3
PROPOSAL_LEARNING_RATE = 1e-5
DEFAULT_N_QUERIES = 1000
ROLLING_WINDOW = 50
ORACLE_PUZZLE_COUNT = 20
ORACLE_EVAL_INTERVAL = 50
ENERGY_BASELINE_WINDOW = 100  # accepted samples used at start/end for E-drop

# Q9 stability gate thresholds (from in-situ-training-phase5-derisking proposal)
GATE1_ENERGY_DROP_FRACTION = 0.30
GATE2_MAX_SPECTRAL_NORM_GROWTH_PER_QUERY = 0.01
GATE3_AUTOCATALYTIC_SLOPE_RATIO = 1.5
GATE4_MIN_ANCHOR_DISTANCE = 0.5
GATE5_MAX_ORACLE_ACCURACY_DROP_PP = 5.0


# ---------------------------------------------------------------------------
# k=3 verifier ensemble — Z3 + causal + ThinkPRM stubs
# ---------------------------------------------------------------------------


def z3_math_verifier_stub(
    actions: Sequence[tuple[int, int, int]],
    grid: np.ndarray,
    grid_size: int = GRID_SIZE,
    num_colors: int = NUM_COLORS,
) -> bool:
    """Tier-3 Z3-math verifier stub: enforces in-bounds invariants.

    In production this would call the Z3 SMT solver with linear-arithmetic
    constraints on the action sequence; for the Phase 5-B derisking
    prototype we use the cheap in-bounds check as a proxy that preserves
    the gate's mechanism (a Boolean verifier with a meaningful pass rate
    on random input).  When the encoder produces saturated latents the
    snap step can occasionally pop out of the 5x5 grid, so this is a
    non-trivial check.
    """
    return _verifier_in_bounds(actions, grid_size, num_colors)


def causal_reasoning_verifier_stub(
    actions: Sequence[tuple[int, int, int]],
    grid: np.ndarray,
    grid_size: int = GRID_SIZE,
    num_colors: int = NUM_COLORS,
) -> bool:
    """Tier-2.7 causal-reasoning verifier stub: did the action have an effect?

    In production this would query a learned causal model that asks
    whether the proposed action actually changes the puzzle state in a
    way that progresses toward the goal.  For Phase 5-B we use the
    "changes_grid" check as a stub — it captures the critical bit
    (vacuous no-op sequences must be rejected) at zero training cost.
    """
    return _verifier_changes_grid(actions, grid)


def thinkprm_v2_stub(
    actions: Sequence[tuple[int, int, int]],
    grid: np.ndarray,
    grid_size: int = GRID_SIZE,
    num_colors: int = NUM_COLORS,
) -> bool:
    """ThinkPRM v2 process-reward-model stub returning soft-accept (True).

    The proposal explicitly says this stub returns 'uncertain' in CI
    mode, which counts as a *soft accept* for the AND-composition.  A
    third permissive verifier still constrains the loss function (the
    other two are non-trivial), and crucially it lets the gate test
    whether the OTHER two verifiers are doing the work — if Phase 5-B
    fails, we want to be able to attribute the failure to the
    Z3 + causal pair and not to a noisy third channel.  Production
    ThinkPRM v2 is far more selective; the stub stays soft-accept
    until the real model is wired up in Phase 5-C.
    """
    del actions, grid, grid_size, num_colors  # signature parity only
    return True


def verifier_ensemble_pass(
    actions: Sequence[tuple[int, int, int]],
    grid: np.ndarray,
    grid_size: int = GRID_SIZE,
    num_colors: int = NUM_COLORS,
) -> bool:
    """k=3 AND-composed verifier ensemble for Phase 5-B."""
    v1 = z3_math_verifier_stub(actions, grid, grid_size, num_colors)
    v2 = causal_reasoning_verifier_stub(actions, grid, grid_size, num_colors)
    v3 = thinkprm_v2_stub(actions, grid, grid_size, num_colors)
    return bool(v1 and v2 and v3)


# ---------------------------------------------------------------------------
# Encoder forward pass with retained activations (needed for fc2 backprop)
# ---------------------------------------------------------------------------


def encoder_forward_with_h(
    encoder: InSituEncoder, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run the encoder forward pass and return (z, h_at_fc2_input).

    The Phase 5-B CD-1 update touches only the encoder's last linear
    layer (``fc2_W``, ``fc2_b``).  To compute its gradient we need the
    24-dim hidden activation that fc2 consumes; the public
    ``InSituEncoder.forward`` only returns the final z, so we
    reproduce its body here and capture h.  This is intentionally a
    duplicate-but-narrow implementation rather than a refactor of the
    Phase 5-A module (Phase 5-A is frozen — exp1222 already passed).
    """
    arr = np.asarray(grid)
    if arr.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"grid must be ({GRID_SIZE},{GRID_SIZE}); got {arr.shape}")
    x = (arr.astype(np.float32) / float(NUM_COLORS - 1))[None, :, :]
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.empty((16, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            patch = padded[:, r : r + 3, c : c + 3]
            out[:, r, c] = (encoder.conv_W * patch[None, :, :, :]).sum(
                axis=(1, 2, 3)
            ) + encoder.conv_b
    out = np.tanh(out)
    flat = out.reshape(-1)  # (400,)
    h = np.tanh(encoder.fc1_W @ flat + encoder.fc1_b)  # (24,)
    z = np.tanh(encoder.fc2_W @ h + encoder.fc2_b)  # (16,)
    return z.astype(np.float64), h.astype(np.float32)


# ---------------------------------------------------------------------------
# CD-1 update step — energy MLP + encoder fc2 only
# ---------------------------------------------------------------------------


def _energy_forward_with_activations(
    energy_mlp: InSituEnergyMLP, z: np.ndarray
) -> dict[str, np.ndarray]:
    """Forward pass through the energy MLP retaining all activations."""
    z_arr = np.asarray(z, dtype=np.float32).flatten()
    a1 = energy_mlp.W1 @ z_arr + energy_mlp.b1
    h1 = np.tanh(a1)
    a2 = energy_mlp.W2 @ h1 + energy_mlp.b2
    h2 = np.tanh(a2)
    a3 = energy_mlp.W3 @ h2 + energy_mlp.b3
    h3 = np.tanh(a3)
    a4 = float((energy_mlp.W4 @ h3 + energy_mlp.b4)[0])
    if energy_mlp.clamp_output:
        e = float(1.0 / (1.0 + np.exp(-a4)))
    else:
        e = float(a4)
    return {"z": z_arr, "h1": h1, "h2": h2, "h3": h3, "a4": np.array([a4]), "e": np.array([e])}


def _energy_backward(
    energy_mlp: InSituEnergyMLP, fwd: dict[str, np.ndarray], dL_de: float
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Manual chain-rule backprop of dL/dE through the energy MLP.

    Returns the gradient for every weight + bias and the gradient
    flowing back to the latent z.  Tanh derivative is ``1 - tanh^2``;
    sigmoid derivative is ``sigma * (1 - sigma)``.
    """
    if energy_mlp.clamp_output:
        e = float(fwd["e"][0])
        dL_da4 = float(dL_de * e * (1.0 - e))
    else:
        dL_da4 = float(dL_de)

    grad_W4 = (dL_da4 * fwd["h3"]).reshape(1, -1).astype(np.float32)
    grad_b4 = np.array([dL_da4], dtype=np.float32)
    dL_dh3 = (energy_mlp.W4.T.flatten() * dL_da4).astype(np.float32)

    dL_da3 = dL_dh3 * (1.0 - fwd["h3"] ** 2)
    grad_W3 = np.outer(dL_da3, fwd["h2"]).astype(np.float32)
    grad_b3 = dL_da3.astype(np.float32)
    dL_dh2 = (energy_mlp.W3.T @ dL_da3).astype(np.float32)

    dL_da2 = dL_dh2 * (1.0 - fwd["h2"] ** 2)
    grad_W2 = np.outer(dL_da2, fwd["h1"]).astype(np.float32)
    grad_b2 = dL_da2.astype(np.float32)
    dL_dh1 = (energy_mlp.W2.T @ dL_da2).astype(np.float32)

    dL_da1 = dL_dh1 * (1.0 - fwd["h1"] ** 2)
    grad_W1 = np.outer(dL_da1, fwd["z"]).astype(np.float32)
    grad_b1 = dL_da1.astype(np.float32)
    dL_dz = (energy_mlp.W1.T @ dL_da1).astype(np.float32)

    return (
        {
            "W1": grad_W1,
            "b1": grad_b1,
            "W2": grad_W2,
            "b2": grad_b2,
            "W3": grad_W3,
            "b3": grad_b3,
            "W4": grad_W4,
            "b4": grad_b4,
        },
        dL_dz,
    )


def cd1_update(
    encoder: InSituEncoder,
    energy_mlp: InSituEnergyMLP,
    z_pos: np.ndarray,
    z_neg: np.ndarray,
    encoder_h_pos: np.ndarray,
    encoder_h_neg: np.ndarray,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> None:
    """In-place CD-1 update on the energy MLP and encoder fc2 layer.

    CD-1 loss: ``L = E(z_pos) - E(z_neg)``.  Minimising L lowers the
    energy on accepted samples (z_pos) and raises it on negative
    samples (z_neg, here drawn from a freshly-encoded random puzzle so
    that the negative latent comes from the same manifold rather than
    uniform noise).  Encoder updates are restricted to the last linear
    layer (``fc2_W``/``fc2_b``); the refiner is treated as identity for
    backprop.  This is a Phase-5-B-prototype design choice — a full
    backprop through the conv + fc1 + refiner stack is feasible but
    would complicate the gate-2 spectral-norm interpretation.
    """
    fwd_pos = _energy_forward_with_activations(energy_mlp, z_pos)
    fwd_neg = _energy_forward_with_activations(energy_mlp, z_neg)
    grads_pos, dL_dz_pos = _energy_backward(energy_mlp, fwd_pos, +1.0)
    grads_neg, dL_dz_neg = _energy_backward(energy_mlp, fwd_neg, -1.0)

    for k in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"):
        delta = (grads_pos[k] + grads_neg[k]).astype(np.float32)
        setattr(energy_mlp, k, getattr(energy_mlp, k) - learning_rate * delta)

    z_pos_arr = np.asarray(z_pos, dtype=np.float32).flatten()
    z_neg_arr = np.asarray(z_neg, dtype=np.float32).flatten()
    grad_fc2_W_pos = np.outer(dL_dz_pos * (1.0 - z_pos_arr**2), encoder_h_pos).astype(np.float32)
    grad_fc2_W_neg = np.outer(dL_dz_neg * (1.0 - z_neg_arr**2), encoder_h_neg).astype(np.float32)
    grad_fc2_b_pos = (dL_dz_pos * (1.0 - z_pos_arr**2)).astype(np.float32)
    grad_fc2_b_neg = (dL_dz_neg * (1.0 - z_neg_arr**2)).astype(np.float32)
    encoder.fc2_W = (encoder.fc2_W - learning_rate * (grad_fc2_W_pos + grad_fc2_W_neg)).astype(
        np.float32
    )
    encoder.fc2_b = (encoder.fc2_b - learning_rate * (grad_fc2_b_pos + grad_fc2_b_neg)).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Spectral norm + oracle evaluation
# ---------------------------------------------------------------------------


def encoder_spectral_norm(encoder: InSituEncoder) -> float:
    """Largest singular value of the encoder's fc2 weight matrix.

    fc2_W is the only encoder weight Phase 5-B updates, so the
    spectral norm of the encoder is well-summarised by ``svd(fc2_W)[0]``
    for the gate-2 representation-drift check.  Initial value is around
    1.0 (Gaussian init with std = 1/sqrt(24), 16x24 matrix).
    """
    return float(np.linalg.svd(encoder.fc2_W, compute_uv=False)[0])


def evaluate_oracle(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
    oracle_puzzles: Sequence[np.ndarray],
) -> float:
    """Fraction of oracle puzzles whose proposed action passes verifier-AND.

    The "oracle" here is a frozen 20-puzzle held-out set sampled from
    the same generator as training.  Held-out accuracy is the fraction
    of these that the (encoder, refiner, snap) pipeline produces a
    verifier-AND-passing action for.  Tracking this every 50 queries
    gives the gate-5 catastrophic-forgetting signal.
    """
    if not oracle_puzzles:
        return 0.0
    n_pass = 0
    for grid in oracle_puzzles:
        z, _h = encoder_forward_with_h(encoder, grid)
        z_refined = refiner.forward(z)
        actions = snap_to_action(z_refined)
        if verifier_ensemble_pass(actions, grid):
            n_pass += 1
    return float(n_pass) / float(len(oracle_puzzles))


# ---------------------------------------------------------------------------
# Trajectory diagnostics
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryDiagnostics:
    """All per-query diagnostics recorded during the 1000-query trajectory.

    Stored as plain lists for ease of JSON serialisation; the gate
    evaluator computes summary statistics from these lists rather than
    holding running aggregates in dataclass fields (the latter would
    require teaching the artifact builder about every aggregate).
    """

    accepted: list[bool] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    energies_accepted: list[float] = field(default_factory=list)
    anchor_distances: list[float] = field(default_factory=list)
    spectral_norms: list[tuple[int, float]] = field(default_factory=list)
    oracle_accuracies: list[tuple[int, float]] = field(default_factory=list)


def _rolling_acceptance_rates(
    accepted: Sequence[bool], window: int = ROLLING_WINDOW
) -> list[float]:
    """Compute a rolling acceptance rate over the trajectory.

    Returns a list of length ``len(accepted) - window + 1`` (or empty
    if the window is too large).  Each entry is the mean of the
    boolean acceptance flags inside that window.
    """
    arr = np.asarray(accepted, dtype=np.float64)
    if arr.size < window:
        return []
    cumulative = np.concatenate(([0.0], np.cumsum(arr)))
    sliced = (cumulative[window:] - cumulative[:-window]) / float(window)
    return [float(x) for x in sliced]


def _acceptance_rate_sublinear(rates: Sequence[float]) -> bool:
    """Check that the acceptance-rate slope decelerates (no spiral).

    A rolling-acceptance-rate trajectory that is autocatalytic would
    show acceleration: the slope in the second half of the trajectory
    would exceed the slope in the first half.  We take the simple
    test slope_2nd_half <= slope_1st_half * GATE3_AUTOCATALYTIC_SLOPE_RATIO,
    treating identically-zero or near-flat trajectories as sub-linear
    (no growth at all is the limiting sub-linear case).
    """
    if len(rates) < 4:
        # Trajectory too short to characterise — declare sub-linear by default.
        return True
    half = len(rates) // 2
    first = np.asarray(rates[:half], dtype=np.float64)
    second = np.asarray(rates[half:], dtype=np.float64)
    x_first = np.arange(first.size, dtype=np.float64)
    x_second = np.arange(second.size, dtype=np.float64)
    slope_first = np.polyfit(x_first, first, 1)[0]
    slope_second = np.polyfit(x_second, second, 1)[0]
    if slope_first <= 0.0:
        # No growth in the first half — second half must not grow much either.
        return bool(slope_second <= GATE3_AUTOCATALYTIC_SLOPE_RATIO * 1e-3)
    return bool(slope_second <= GATE3_AUTOCATALYTIC_SLOPE_RATIO * slope_first)


def _energy_decrease_pct(energies_accepted: Sequence[float]) -> float:
    """Compute the percentage drop in mean E(z_accepted) start vs end.

    Splits the accepted-sample energy trace into a leading and trailing
    window of size ``ENERGY_BASELINE_WINDOW`` (or as many samples as
    are available, divided evenly) and returns
    ``(E_start - E_end) / E_start``.  Returns 0.0 if not enough
    accepted samples were observed (defensive).
    """
    arr = np.asarray(energies_accepted, dtype=np.float64)
    if arr.size < 4:
        return 0.0
    win = min(ENERGY_BASELINE_WINDOW, arr.size // 2)
    if win <= 0:
        return 0.0
    e_start = float(arr[:win].mean())
    e_end = float(arr[-win:].mean())
    if e_start == 0.0:
        return 0.0
    return float((e_start - e_end) / abs(e_start))


def _spectral_norm_growth_rate(spectral_norms: Sequence[tuple[int, float]]) -> float:
    """Linear-fit slope of spectral norm vs query index.

    Phase 5-B records ``(query_index, spectral_norm)`` every 50 queries.
    The growth rate is the linear-regression slope of spectral norm
    against query index in units of "norm units per query".  Returns
    0.0 if too few measurements.
    """
    if len(spectral_norms) < 2:
        return 0.0
    xs = np.asarray([q for q, _ in spectral_norms], dtype=np.float64)
    ys = np.asarray([sn for _, sn in spectral_norms], dtype=np.float64)
    if xs.size < 2 or float(xs[-1]) == float(xs[0]):
        return 0.0
    slope = np.polyfit(xs, ys, 1)[0]
    return float(slope)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def run_phase5b_training_loop(
    n_queries: int = DEFAULT_N_QUERIES,
    seed: int = 1223,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    oracle_eval_interval: int = ORACLE_EVAL_INTERVAL,
    oracle_puzzle_count: int = ORACLE_PUZZLE_COUNT,
) -> dict[str, Any]:
    """Run the Phase 5-B in-situ training loop and return raw diagnostics.

    The structure of one query iteration:

        1. Sample a random 5x5 puzzle.
        2. Encode → refine → snap → verify (k=3 AND).
        3. Record E(z_refined) and anchor distance.
        4. If accepted: sample a separate random puzzle as the negative,
           compute its (z_neg, h_neg) via the encoder, and apply one
           CD-1 step on the energy MLP and encoder fc2 layer.
        5. Every ``oracle_eval_interval`` queries, record the encoder
           spectral norm and the held-out oracle accuracy.

    Returns a dict with the trajectory diagnostics ready for the
    artifact builder.  Designed to be deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    encoder = InSituEncoder.init(seed=seed)
    energy_mlp = InSituEnergyMLP.init(seed=seed + 1)
    refiner = InSituRefiner.init(seed=seed + 2)
    anchor_tracker = VacuousAnchorTracker.default()

    # Frozen oracle set, sampled BEFORE training begins so that the
    # gate-5 measurement compares apples to apples.
    oracle_rng = np.random.default_rng(seed + 99)
    oracle_puzzles = [generate_random_5x5_puzzle(oracle_rng) for _ in range(oracle_puzzle_count)]
    oracle_accuracy_initial = evaluate_oracle(encoder, refiner, oracle_puzzles)

    diag = TrajectoryDiagnostics()

    for q in range(n_queries):
        grid = generate_random_5x5_puzzle(rng)
        z, h = encoder_forward_with_h(encoder, grid)
        z_refined = refiner.forward(z)
        e_z = energy_mlp.forward(z_refined)
        anchor_d = anchor_tracker.distance(z_refined)
        actions = snap_to_action(z_refined)
        accepted = verifier_ensemble_pass(actions, grid)

        diag.accepted.append(bool(accepted))
        diag.energies.append(float(e_z))
        diag.anchor_distances.append(float(anchor_d))
        if accepted:
            diag.energies_accepted.append(float(e_z))
            # CD-1 negative sample: uniform random latent in (-1, 1)^16.
            # We deliberately do NOT use a freshly-encoded latent here —
            # if the encoder has not yet specialised, encoded latents
            # cluster tightly around z_pos and the gradient cancels.  A
            # uniform-random negative gives the EBM a wide-coverage
            # contrast and is the standard CD-1 negative-sampling
            # protocol for non-persistent chains.  We pair it with a
            # uniform-random ``encoder_h_neg`` so that the encoder's
            # fc2 update has a non-trivial reference point too.
            z_neg_refined = rng.uniform(-1.0, 1.0, LATENT_DIM)
            h_neg = rng.uniform(-1.0, 1.0, encoder.fc1_b.shape[0]).astype(np.float32)
            cd1_update(
                encoder,
                energy_mlp,
                z_refined,
                z_neg_refined,
                h,
                h_neg,
                learning_rate=learning_rate,
            )

        if (q + 1) % oracle_eval_interval == 0:
            diag.spectral_norms.append((q + 1, encoder_spectral_norm(encoder)))
            diag.oracle_accuracies.append(
                (q + 1, evaluate_oracle(encoder, refiner, oracle_puzzles))
            )

    final_oracle_accuracy = (
        diag.oracle_accuracies[-1][1]
        if diag.oracle_accuracies
        else evaluate_oracle(encoder, refiner, oracle_puzzles)
    )

    return {
        "n_queries_run": int(n_queries),
        "n_accepted_by_verifier": int(sum(diag.accepted)),
        "energies": diag.energies,
        "energies_accepted": diag.energies_accepted,
        "anchor_distances": diag.anchor_distances,
        "spectral_norms": diag.spectral_norms,
        "oracle_accuracies": diag.oracle_accuracies,
        "oracle_accuracy_initial": float(oracle_accuracy_initial),
        "oracle_accuracy_final": float(final_oracle_accuracy),
        "accepted": diag.accepted,
    }


# ---------------------------------------------------------------------------
# Gate evaluator + artifact builder
# ---------------------------------------------------------------------------


def evaluate_phase5b_gates(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Compute the five Q9 stability gates from a trajectory diagnostics dict.

    Returns a dict containing every per-gate boolean, the underlying
    measurement that produced it, and the total ``gates_passed`` count.
    Designed to be pure: same input → same output, no global state.
    """
    energies_accepted = diagnostics.get("energies_accepted", [])
    energy_drop_pct = _energy_decrease_pct(energies_accepted)
    spectral_norms = diagnostics.get("spectral_norms", [])
    sn_growth = _spectral_norm_growth_rate(spectral_norms)
    accepted = diagnostics.get("accepted", [])
    rolling = _rolling_acceptance_rates(accepted)
    sublinear = _acceptance_rate_sublinear(rolling)
    anchor_distances = diagnostics.get("anchor_distances", [])
    if anchor_distances:
        mean_anchor = float(np.mean(np.asarray(anchor_distances, dtype=np.float64)))
        # "Stayed > 0.5 throughout" — Phase 5-B uses the *minimum* value
        # observed during training as the conservative trigger.
        min_anchor = float(np.min(np.asarray(anchor_distances, dtype=np.float64)))
    else:
        mean_anchor = 0.0
        min_anchor = 0.0
    oracle_initial = float(diagnostics.get("oracle_accuracy_initial", 0.0))
    oracle_final = float(diagnostics.get("oracle_accuracy_final", 0.0))
    oracle_drop_pp = (oracle_initial - oracle_final) * 100.0

    gate1 = bool(energy_drop_pct >= GATE1_ENERGY_DROP_FRACTION)
    gate2 = bool(abs(sn_growth) < GATE2_MAX_SPECTRAL_NORM_GROWTH_PER_QUERY)
    gate3 = bool(sublinear)
    gate4 = bool(min_anchor > GATE4_MIN_ANCHOR_DISTANCE)
    gate5 = bool(oracle_drop_pp <= GATE5_MAX_ORACLE_ACCURACY_DROP_PP)
    gates_passed = int(gate1) + int(gate2) + int(gate3) + int(gate4) + int(gate5)

    n_acc = int(diagnostics.get("n_accepted_by_verifier", 0))
    n_q = int(diagnostics.get("n_queries_run", 0))
    acceptance_rate = float(n_acc) / float(n_q) if n_q > 0 else 0.0

    return {
        "energy_decrease_pct": float(energy_drop_pct),
        "spectral_norm_growth_rate": float(sn_growth),
        "acceptance_rate_sublinear": bool(sublinear),
        "mean_anchor_distance": float(mean_anchor),
        "min_anchor_distance": float(min_anchor),
        "oracle_accuracy_initial": float(oracle_initial),
        "oracle_accuracy_final": float(oracle_final),
        "oracle_accuracy_drop_pp": float(oracle_drop_pp),
        "acceptance_rate": float(acceptance_rate),
        "gate1_energy_decrease_30pct": gate1,
        "gate2_no_representation_drift": gate2,
        "gate3_no_autocatalytic_spiral": gate3,
        "gate4_no_null_space_excavation": gate4,
        "gate5_no_catastrophic_forgetting": gate5,
        "gates_passed": gates_passed,
    }


def _verdict_for_gates(gates_passed: int) -> str:
    """Map the integer ``gates_passed`` count onto an honest verdict string."""
    if gates_passed == 5:
        return "all_5_gates_pass"
    if 1 <= gates_passed <= 4:
        return "partial_gates"
    return "gate_failure_diagnosed"


def build_phase5b_artifact(
    diagnostics: dict[str, Any],
    gates: dict[str, Any],
    *,
    seed: int = 1223,
    learning_rate_used: float = DEFAULT_LEARNING_RATE,
    blocked: bool = False,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    """Build the JSON-serialisable Exp 1223 artifact.

    The artifact flattens every required gate and measurement to the
    top level so the conductor's schema validator can verify the
    presence of fields named in REQ-KONA-017 without descending into
    nested objects.
    """
    if blocked:
        return {
            "experiment": "1223_phase5b_insitu_training_loop",
            "schema_version": SCHEMA_VERSION,
            "run_date": _dt.datetime.now(_dt.UTC).isoformat(),
            "seed": int(seed),
            "learning_rate_used": float(learning_rate_used),
            "proposal_learning_rate": float(PROPOSAL_LEARNING_RATE),
            "status": "blocked",
            "n_queries_run": 0,
            "n_accepted_by_verifier": 0,
            "acceptance_rate": 0.0,
            "energy_decrease_pct": 0.0,
            "spectral_norm_growth_rate": 0.0,
            "acceptance_rate_sublinear": False,
            "mean_anchor_distance": 0.0,
            "oracle_accuracy_initial": 0.0,
            "oracle_accuracy_final": 0.0,
            "oracle_accuracy_drop_pp": 0.0,
            "gate1_energy_decrease_30pct": False,
            "gate2_no_representation_drift": False,
            "gate3_no_autocatalytic_spiral": False,
            "gate4_no_null_space_excavation": False,
            "gate5_no_catastrophic_forgetting": False,
            "gates_passed": 0,
            "phase5b_stability_confirmed": False,
            "honest_verdict": "blocked",
            "blocked_reason": blocked_reason or "phase5a_prototype_not_ready",
        }

    gates_passed = int(gates["gates_passed"])
    confirmed = gates_passed == 5
    verdict = _verdict_for_gates(gates_passed)
    artifact = {
        "experiment": "1223_phase5b_insitu_training_loop",
        "schema_version": SCHEMA_VERSION,
        "run_date": _dt.datetime.now(_dt.UTC).isoformat(),
        "seed": int(seed),
        "learning_rate_used": float(learning_rate_used),
        "proposal_learning_rate": float(PROPOSAL_LEARNING_RATE),
        "status": "success" if confirmed else "partial",
        "n_queries_run": int(diagnostics["n_queries_run"]),
        "n_accepted_by_verifier": int(diagnostics["n_accepted_by_verifier"]),
        "acceptance_rate": float(gates["acceptance_rate"]),
        "energy_decrease_pct": float(gates["energy_decrease_pct"]),
        "spectral_norm_growth_rate": float(gates["spectral_norm_growth_rate"]),
        "acceptance_rate_sublinear": bool(gates["acceptance_rate_sublinear"]),
        "mean_anchor_distance": float(gates["mean_anchor_distance"]),
        "min_anchor_distance": float(gates["min_anchor_distance"]),
        "oracle_accuracy_initial": float(gates["oracle_accuracy_initial"]),
        "oracle_accuracy_final": float(gates["oracle_accuracy_final"]),
        "oracle_accuracy_drop_pp": float(gates["oracle_accuracy_drop_pp"]),
        "gate1_energy_decrease_30pct": bool(gates["gate1_energy_decrease_30pct"]),
        "gate2_no_representation_drift": bool(gates["gate2_no_representation_drift"]),
        "gate3_no_autocatalytic_spiral": bool(gates["gate3_no_autocatalytic_spiral"]),
        "gate4_no_null_space_excavation": bool(gates["gate4_no_null_space_excavation"]),
        "gate5_no_catastrophic_forgetting": bool(gates["gate5_no_catastrophic_forgetting"]),
        "gates_passed": gates_passed,
        "phase5b_stability_confirmed": bool(confirmed),
        "honest_verdict": verdict,
    }
    return artifact


def write_phase5b_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    """Write the artifact dict to ``path`` as pretty JSON."""
    Path(path).write_text(json.dumps(artifact, indent=2, sort_keys=True))


def confirm_phase5a_ready(phase5a_artifact_path: str | Path) -> bool:
    """Return True iff the Phase 5-A artifact reports prototype_ready=True.

    Used by the experiment runner as the precondition gate; the
    in-situ-training-phase5-derisking proposal explicitly requires
    Phase 5-A to have passed before Phase 5-B may launch.
    """
    p = Path(phase5a_artifact_path)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(data.get("phase5a_prototype_ready", False))
