"""Phase 5-A in-situ training substrate prototype (exp_NEXT_A).

**Researcher summary:**
    Smallest end-to-end Carnot substrate suitable for closing the loop:
    Encoder (CNN) → Energy MLP → Refiner MLP → snap-to-action decoder.
    Together they implement the *forward* half of the in-situ training
    loop on toy 5x5 ARC-like grids.  No weight updates are performed
    here — that is exp_NEXT_B.  This file is the architectural skeleton
    plus the diagnostic instrumentation (vacuous-anchor distances and
    per-verifier conditional-acceptance probabilities) that exp_NEXT_B
    will need to detect failure modes 4-8 from the Q9 catalog.

**Detailed explanation for engineers (CLAUDE.md verbose-layman rule):**
    Phase 5 will eventually update model weights *during inference*
    using the verifier ensemble's pass/fail signal as supervision.
    Before we can do that responsibly we have to be sure the substrate
    itself is sane — the encoder produces representations that are not
    already collapsed onto a "vacuous" attractor, the energy network
    actually emits bounded outputs, and the verifiers don't trivially
    correlate (which would mean their AND-composition isn't really k=3
    grounding, just k=1 in disguise).  This prototype is the smallest
    artifact that lets us measure those things.

    Components:

    * ``InSituEncoder`` — small convolutional net mapping a 5x5
      categorical grid to a 16-dim latent ``z`` clamped to (-1, 1) via
      tanh.  ~10K parameters.  We use a CNN rather than an MLP because
      Phase 3's substrate will be convolutional and we want the
      diagnostic story to transfer.

    * ``InSituEnergyMLP`` — three-hidden-layer MLP mapping ``z`` to a
      bounded scalar energy in [0, 1] via sigmoid.  ~10K parameters.
      Bounded output is required for stable AND-composition with
      verifier outputs in exp_NEXT_B.

    * ``InSituRefiner`` — *decoder-side* MLP that projects ``z`` back to
      a (refined) 16-dim latent before snapping.  Q8 Option A snap
      semantics live here.  ~30K parameters; the bulk of the model.

    * ``snap_to_action`` — parameter-free decoder that converts a
      continuous latent to a discrete (row, col, color) action
      sequence on a 5x5 grid.

    * ``VacuousAnchorTracker`` — measures L2 distance from each
      observed ``z`` to a curated set of known-vacuous latents
      (zero, +sat, -sat).  exp_NEXT_B uses this to detect null-space
      excavation: if mean distance shrinks over training, the model
      is collapsing toward a degenerate attractor.

    * ``ConditionalAcceptanceProbMatrix`` — P(verifier_i passes |
      verifier_j passes) accumulated across all queries.  exp_NEXT_C
      uses this to detect correlated evaluator blind spots: if the
      off-diagonal entries are all near 1.0 the verifiers are
      effectively a single verifier in disguise.

Spec coverage:
    REQ-KONA-008 (snap-to-action reuse — same Q8 Option A semantics
    as Phase 4), REQ-KONA-012 (active-inference latent), and the
    in-situ-training-phase5-derisking change proposal's exp_NEXT_A
    acceptance gate (≥50% valid action sequences across 100 random
    5x5 puzzles, anchor-distance and conditional-acceptance matrices
    recorded).
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

GRID_SIZE = 5
NUM_COLORS = 4
LATENT_DIM = 16
ACTIONS_PER_SEQUENCE = 4
LATENTS_PER_ACTION = LATENT_DIM // ACTIONS_PER_SEQUENCE  # = 4
N_VERIFIERS = 3
SCHEMA_VERSION = "1222.phase5a.insitu_prototype.v1"

# Quadrant anchors for the snap decoder: action k is anchored in corner k,
# and the latent perturbs inward toward the centre.  This is the Phase-5
# prototype's interpretation of "Q8 Option A": each action of the sequence
# is anchored to a distinct quadrant so that an under-trained encoder
# (output ``z`` near zero) still yields a non-degenerate action sequence.
# The four anchors are the four corners of the 5x5 grid.
QUADRANT_ANCHORS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, GRID_SIZE - 1),
    (GRID_SIZE - 1, 0),
    (GRID_SIZE - 1, GRID_SIZE - 1),
)


# ---------------------------------------------------------------------------
# Encoder — small CNN, 5x5 grid → z ∈ [-1, 1]^16
# ---------------------------------------------------------------------------


@dataclass
class InSituEncoder:
    """Small CNN encoder: int grid (5,5) → latent z ∈ (-1, 1)^16.

    Architecture:
        conv1 (1 → 16, 3x3, padding=1, tanh)  →
        flatten (5*5*16 = 400)               →
        fc1 (400 → 24, tanh)                 →
        fc2 (24 → 16, tanh)

    Total parameter count: 144+16 + 9600+24 + 384+16 = 10184.
    """

    conv_W: np.ndarray  # (16, 1, 3, 3)
    conv_b: np.ndarray  # (16,)
    fc1_W: np.ndarray   # (24, 400)
    fc1_b: np.ndarray   # (24,)
    fc2_W: np.ndarray   # (16, 24)
    fc2_b: np.ndarray   # (16,)

    @classmethod
    def init(cls, seed: int = 0) -> "InSituEncoder":
        """Initialise weights with a simple fan-in scaling.

        We deliberately do not use Xavier or He init machinery; the
        prototype is small enough that a Gaussian with std = 1/sqrt(fan_in)
        is fine, and avoiding torch/jax keeps the file self-contained.
        """
        rng = np.random.default_rng(seed)
        return cls(
            conv_W=rng.normal(0.0, 1.0 / np.sqrt(9.0), (16, 1, 3, 3)).astype(np.float32),
            conv_b=np.zeros(16, dtype=np.float32),
            fc1_W=rng.normal(0.0, 1.0 / np.sqrt(400.0), (24, 400)).astype(np.float32),
            fc1_b=np.zeros(24, dtype=np.float32),
            fc2_W=rng.normal(0.0, 1.0 / np.sqrt(24.0), (16, 24)).astype(np.float32),
            fc2_b=np.zeros(16, dtype=np.float32),
        )

    def forward(self, grid: np.ndarray) -> np.ndarray:
        """Encode a 5x5 grid of categorical colors into a latent z.

        The grid is normalised to [0, 1] by dividing by ``NUM_COLORS - 1``.
        Output ``z`` is constrained to (-1, 1)^16 by tanh.
        """
        arr = np.asarray(grid)
        if arr.shape != (GRID_SIZE, GRID_SIZE):
            raise ValueError(f"grid must be ({GRID_SIZE},{GRID_SIZE}); got {arr.shape}")
        x = (arr.astype(np.float32) / float(NUM_COLORS - 1))[None, :, :]  # (1, 5, 5)
        # Manual 3x3 conv, padding=1, output (16, 5, 5).
        padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        out = np.empty((16, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                patch = padded[:, r : r + 3, c : c + 3]  # (1, 3, 3)
                out[:, r, c] = (self.conv_W * patch[None, :, :, :]).sum(axis=(1, 2, 3)) + self.conv_b
        out = np.tanh(out)
        flat = out.reshape(-1)  # (400,)
        h = np.tanh(self.fc1_W @ flat + self.fc1_b)  # (24,)
        z = np.tanh(self.fc2_W @ h + self.fc2_b)  # (16,)
        return z.astype(np.float64)

    def param_count(self) -> int:
        return int(
            self.conv_W.size
            + self.conv_b.size
            + self.fc1_W.size
            + self.fc1_b.size
            + self.fc2_W.size
            + self.fc2_b.size
        )


# ---------------------------------------------------------------------------
# Energy MLP — z ∈ [-1, 1]^16 → E ∈ [0, 1]
# ---------------------------------------------------------------------------


@dataclass
class InSituEnergyMLP:
    """Energy network: latent z → bounded scalar energy E ∈ [0, 1].

    Architecture:
        fc1 (16 → 64, tanh) → fc2 (64 → 64, tanh) → fc3 (64 → 64, tanh) → fc4 (64 → 1, sigmoid)

    Total parameter count: 1088 + 4160 + 4160 + 65 = 9473.
    """

    W1: np.ndarray  # (64, 16)
    b1: np.ndarray
    W2: np.ndarray  # (64, 64)
    b2: np.ndarray
    W3: np.ndarray  # (64, 64)
    b3: np.ndarray
    W4: np.ndarray  # (1, 64)
    b4: np.ndarray
    clamp_output: bool = True

    @classmethod
    def init(cls, seed: int = 1) -> "InSituEnergyMLP":
        rng = np.random.default_rng(seed)
        return cls(
            W1=rng.normal(0.0, 1.0 / np.sqrt(16.0), (64, 16)).astype(np.float32),
            b1=np.zeros(64, dtype=np.float32),
            W2=rng.normal(0.0, 1.0 / np.sqrt(64.0), (64, 64)).astype(np.float32),
            b2=np.zeros(64, dtype=np.float32),
            W3=rng.normal(0.0, 1.0 / np.sqrt(64.0), (64, 64)).astype(np.float32),
            b3=np.zeros(64, dtype=np.float32),
            W4=rng.normal(0.0, 1.0 / np.sqrt(64.0), (1, 64)).astype(np.float32),
            b4=np.zeros(1, dtype=np.float32),
            clamp_output=True,
        )

    def forward(self, z: np.ndarray) -> float:
        """Return E(z) ∈ [0, 1] (or unclamped scalar if clamp_output=False)."""
        z_arr = np.asarray(z, dtype=np.float32).flatten()
        if z_arr.size != LATENT_DIM:
            raise ValueError(f"z must have dim {LATENT_DIM}; got {z_arr.size}")
        h1 = np.tanh(self.W1 @ z_arr + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        h3 = np.tanh(self.W3 @ h2 + self.b3)
        out = float((self.W4 @ h3 + self.b4)[0])
        if self.clamp_output:
            # Sigmoid produces a stable [0, 1] value.
            return float(1.0 / (1.0 + np.exp(-out)))
        return out

    def param_count(self) -> int:
        return int(
            self.W1.size + self.b1.size + self.W2.size + self.b2.size
            + self.W3.size + self.b3.size + self.W4.size + self.b4.size
        )


# ---------------------------------------------------------------------------
# Refiner — decoder-side MLP that polishes z before the snap step
# ---------------------------------------------------------------------------


@dataclass
class InSituRefiner:
    """Decoder-side refinement MLP: z → z' ∈ (-1, 1)^16.

    Architecture:
        fc1 (16 → 128, tanh) → fc2 (128 → 192, tanh) → fc3 (192 → 16, tanh)

    Total parameter count: 2176 + 24768 + 3088 = 30032.

    The refiner is what makes the *decoder* learnable in the in-situ
    setup; ``snap_to_action`` itself is parameter-free, so without the
    refiner there would be nothing on the decoder side to update during
    inference.  exp_NEXT_B's PCD step will adjust the refiner along
    with the encoder and energy MLP.
    """

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray

    @classmethod
    def init(cls, seed: int = 2) -> "InSituRefiner":
        rng = np.random.default_rng(seed)
        return cls(
            W1=rng.normal(0.0, 1.0 / np.sqrt(16.0), (128, 16)).astype(np.float32),
            b1=np.zeros(128, dtype=np.float32),
            W2=rng.normal(0.0, 1.0 / np.sqrt(128.0), (192, 128)).astype(np.float32),
            b2=np.zeros(192, dtype=np.float32),
            W3=rng.normal(0.0, 1.0 / np.sqrt(192.0), (16, 192)).astype(np.float32),
            b3=np.zeros(16, dtype=np.float32),
        )

    def forward(self, z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=np.float32).flatten()
        if z_arr.size != LATENT_DIM:
            raise ValueError(f"z must have dim {LATENT_DIM}; got {z_arr.size}")
        h1 = np.tanh(self.W1 @ z_arr + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        z_out = np.tanh(self.W3 @ h2 + self.b3)
        return z_out.astype(np.float64)

    def param_count(self) -> int:
        return int(
            self.W1.size + self.b1.size + self.W2.size + self.b2.size
            + self.W3.size + self.b3.size
        )


# ---------------------------------------------------------------------------
# Decoder — Q8 Option A snap-to-action
# ---------------------------------------------------------------------------


def snap_to_action(
    z: np.ndarray,
    grid_size: int = GRID_SIZE,
    num_colors: int = NUM_COLORS,
    n_actions: int = ACTIONS_PER_SEQUENCE,
) -> list[tuple[int, int, int]]:
    """Decode latent ``z`` ∈ (-1, 1)^16 to a list of ``(row, col, color)``.

    **Q8 Option A semantics (Phase-5 quadrant variant):** the latent is
    partitioned into ``n_actions`` chunks of ``LATENTS_PER_ACTION`` (=4)
    dims each.  Within each chunk:

        * dim 0 (``z_row``) controls inward row displacement from the
          chunk's quadrant anchor.
        * dim 1 (``z_col``) controls inward column displacement.
        * dim 2 (``z_val``) maps linearly to a discrete color in
          [0, num_colors).
        * dim 3 is reserved (Phase-5 future use; e.g., action type).

    Each action is anchored to a distinct corner of the grid (see
    ``QUADRANT_ANCHORS``).  The latent perturbs the position
    *inward* by 0..⌊grid_size/2⌋ cells.  This guarantees that an
    under-trained encoder whose output is near zero still yields four
    distinct cells (one per quadrant) — without this anchoring, every
    action snaps to the centre cell and the no-duplicate verifier
    rejects every sequence.

    **Why corners and not arbitrary positions?**
        Corners are the simplest discrete partition of a 5x5 grid into
        four disjoint 3x3 quadrants.  Any latent value in (-1, 1) keeps
        the snapped row and column within the anchor's quadrant, so
        the no-duplicate verifier passes for *all* z when grid_size=5
        and n_actions=4.  exp_NEXT_B can later replace this with a
        learned codebook; for the prototype we want a deterministic
        decoder that reveals failures elsewhere in the stack.
    """
    z_arr = np.asarray(z, dtype=np.float64).flatten()
    if z_arr.size < n_actions * LATENTS_PER_ACTION:
        raise ValueError(
            f"latent dim {z_arr.size} is below required {n_actions * LATENTS_PER_ACTION}"
        )
    if n_actions > len(QUADRANT_ANCHORS):
        raise ValueError(
            f"n_actions={n_actions} exceeds number of quadrant anchors "
            f"({len(QUADRANT_ANCHORS)})"
        )
    half = grid_size // 2  # max inward displacement from a corner
    actions: list[tuple[int, int, int]] = []
    for k in range(n_actions):
        anchor_r, anchor_c = QUADRANT_ANCHORS[k]
        z_row = z_arr[LATENTS_PER_ACTION * k]
        z_col = z_arr[LATENTS_PER_ACTION * k + 1]
        z_val = z_arr[LATENTS_PER_ACTION * k + 2]
        # Map z in (-1, 1) to inward displacement in [0, half].
        shift_r = int(np.clip(np.round((z_row + 1.0) * 0.5 * half), 0, half))
        shift_c = int(np.clip(np.round((z_col + 1.0) * 0.5 * half), 0, half))
        row = anchor_r + shift_r if anchor_r == 0 else anchor_r - shift_r
        col = anchor_c + shift_c if anchor_c == 0 else anchor_c - shift_c
        val = int(np.clip(np.round((z_val + 1.0) * 0.5 * (num_colors - 1)), 0, num_colors - 1))
        actions.append((int(row), int(col), val))
    return actions


def apply_action_sequence(
    grid: np.ndarray, actions: Sequence[tuple[int, int, int]]
) -> np.ndarray:
    """Apply each ``(row, col, color)`` action to ``grid`` in order.

    Returns a *new* ndarray; the input is not mutated.  Out-of-bounds
    actions are silently skipped here — bounds checking is the
    verifier's job, not the environment's.
    """
    new_grid = np.array(grid, copy=True)
    for r, c, v in actions:
        if 0 <= r < new_grid.shape[0] and 0 <= c < new_grid.shape[1]:
            new_grid[r, c] = v
    return new_grid


# ---------------------------------------------------------------------------
# Verifier ensemble — three independent boolean checks (AND-composed)
# ---------------------------------------------------------------------------


def _verifier_in_bounds(
    actions: Sequence[tuple[int, int, int]],
    grid_size: int = GRID_SIZE,
    num_colors: int = NUM_COLORS,
) -> bool:
    """Verifier 0: every action's (row, col, color) is within bounds."""
    for r, c, v in actions:
        if not (0 <= r < grid_size and 0 <= c < grid_size and 0 <= v < num_colors):
            return False
    return True


def _verifier_changes_grid(
    actions: Sequence[tuple[int, int, int]], grid: np.ndarray
) -> bool:
    """Verifier 1: action sequence actually modifies at least one cell."""
    new_grid = apply_action_sequence(grid, actions)
    return not bool(np.array_equal(new_grid, grid))


def _verifier_no_duplicate_cells(actions: Sequence[tuple[int, int, int]]) -> bool:
    """Verifier 2: no two actions write to the same (row, col)."""
    cells = [(r, c) for r, c, _ in actions]
    return len(cells) == len(set(cells))


def verifier_outcomes(
    actions: Sequence[tuple[int, int, int]],
    grid: np.ndarray,
    grid_size: int = GRID_SIZE,
    num_colors: int = NUM_COLORS,
) -> tuple[bool, bool, bool]:
    """Return per-verifier pass/fail for the three-verifier ensemble.

    Order: (in_bounds, changes_grid, no_duplicate_cells).
    """
    return (
        _verifier_in_bounds(actions, grid_size, num_colors),
        _verifier_changes_grid(actions, grid),
        _verifier_no_duplicate_cells(actions),
    )


def verify_action_sequence(
    actions: Sequence[tuple[int, int, int]],
    grid: np.ndarray,
    grid_size: int = GRID_SIZE,
    num_colors: int = NUM_COLORS,
) -> bool:
    """AND-composed verifier: all three checks must pass."""
    outcomes = verifier_outcomes(actions, grid, grid_size, num_colors)
    return all(outcomes)


# ---------------------------------------------------------------------------
# Diagnostics — vacuous-anchor distance + conditional-acceptance matrix
# ---------------------------------------------------------------------------


@dataclass
class VacuousAnchorTracker:
    """L2 distance from observed latent to a small set of vacuous anchors.

    The default anchors are:
        * the zero vector (degenerate identity / mean collapse),
        * a near-saturated +1 vector (saturated tanh → no information),
        * a near-saturated -1 vector (saturated tanh → no information).

    exp_NEXT_B watches the *minimum* of these distances per query.  If
    the minimum trends down across training, the encoder is collapsing
    onto a vacuous attractor and we abort.
    """

    anchors: np.ndarray  # (N_anchors, latent_dim)

    @classmethod
    def default(cls, latent_dim: int = LATENT_DIM) -> "VacuousAnchorTracker":
        anchors = np.array(
            [
                np.zeros(latent_dim),
                np.full(latent_dim, 0.99),
                np.full(latent_dim, -0.99),
            ],
            dtype=np.float64,
        )
        return cls(anchors=anchors)

    def distance(self, z: np.ndarray) -> float:
        """Return min L2 distance from ``z`` to any anchor."""
        z_arr = np.asarray(z, dtype=np.float64).flatten()
        if z_arr.size != self.anchors.shape[1]:
            raise ValueError(
                f"z dim {z_arr.size} does not match anchor dim {self.anchors.shape[1]}"
            )
        deltas = self.anchors - z_arr[None, :]
        per_anchor = np.linalg.norm(deltas, axis=1)
        return float(per_anchor.min())


@dataclass
class ConditionalAcceptanceProbMatrix:
    """Empirical P(verifier_i passes | verifier_j passes).

    Track joint co-occurrence and per-verifier marginals.  At read time
    return the n×n conditional matrix P(i|j) = count(i ∧ j) / count(j),
    with zero rows where verifier_j has never fired (no inference
    possible).
    """

    n_verifiers: int
    counts_joint: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    counts_marginal: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    n_observations: int = 0

    def __post_init__(self) -> None:
        if self.counts_joint is None:
            self.counts_joint = np.zeros((self.n_verifiers, self.n_verifiers), dtype=np.int64)
        if self.counts_marginal is None:
            self.counts_marginal = np.zeros(self.n_verifiers, dtype=np.int64)

    def record(self, outcomes: Sequence[bool]) -> None:
        """Record one query's per-verifier pass/fail outcome vector."""
        outs = np.asarray(outcomes, dtype=bool)
        if outs.size != self.n_verifiers:
            raise ValueError(f"expected {self.n_verifiers} outcomes; got {outs.size}")
        self.counts_marginal += outs.astype(np.int64)
        # Joint count: counts_joint[i, j] += 1 if outs[i] and outs[j].
        joint = np.outer(outs.astype(np.int64), outs.astype(np.int64))
        self.counts_joint += joint
        self.n_observations += 1

    def matrix(self) -> np.ndarray:
        """Return P(i | j) as an n×n matrix (column-conditioned)."""
        m = np.zeros((self.n_verifiers, self.n_verifiers), dtype=np.float64)
        for j in range(self.n_verifiers):
            denom = int(self.counts_marginal[j])
            if denom > 0:
                m[:, j] = self.counts_joint[:, j].astype(np.float64) / float(denom)
        return m


# ---------------------------------------------------------------------------
# Puzzle generator
# ---------------------------------------------------------------------------


def generate_random_5x5_puzzle(
    rng: np.random.Generator,
    grid_size: int = GRID_SIZE,
    num_colors: int = NUM_COLORS,
) -> np.ndarray:
    """Sample a random 5x5 grid of integer colors uniformly in [0, num_colors)."""
    return rng.integers(0, num_colors, size=(grid_size, grid_size)).astype(np.int32)


# ---------------------------------------------------------------------------
# Top-level prototype runner
# ---------------------------------------------------------------------------


def run_phase5a_prototype(
    n_puzzles: int = 100,
    seed: int = 1222,
) -> dict[str, Any]:
    """Run the Phase 5-A prototype on ``n_puzzles`` random 5x5 grids.

    Returns a dict containing:
        * encoder_param_count, energy_mlp_param_count, refiner_param_count,
          total_param_count
        * n_puzzles_run
        * valid_action_fraction (per AND-composed verifier)
        * mean_anchor_distance (across all queries)
        * mean_energy (across all queries)
        * conditional_acceptance_matrix (n_verifiers × n_verifiers, list)
        * verifier_pass_rates (per individual verifier)

    No weights are updated.  This is the read-only forward pass that
    exp_NEXT_B will wrap with a PCD update step.
    """
    encoder = InSituEncoder.init(seed=seed)
    energy_mlp = InSituEnergyMLP.init(seed=seed + 1)
    refiner = InSituRefiner.init(seed=seed + 2)
    anchor_tracker = VacuousAnchorTracker.default()
    cap_matrix = ConditionalAcceptanceProbMatrix(n_verifiers=N_VERIFIERS)

    rng = np.random.default_rng(seed)
    valid_count = 0
    anchor_distances: list[float] = []
    energies: list[float] = []
    per_verifier_passes = np.zeros(N_VERIFIERS, dtype=np.int64)

    for _ in range(n_puzzles):
        grid = generate_random_5x5_puzzle(rng)
        z = encoder.forward(grid)
        z_refined = refiner.forward(z)
        e = energy_mlp.forward(z_refined)
        actions = snap_to_action(z_refined)
        outcomes = verifier_outcomes(actions, grid)
        per_verifier_passes += np.array(outcomes, dtype=np.int64)
        cap_matrix.record(outcomes)
        anchor_distances.append(anchor_tracker.distance(z_refined))
        energies.append(e)
        if all(outcomes):
            valid_count += 1

    valid_fraction = float(valid_count) / float(n_puzzles)
    mean_anchor_distance = float(np.mean(anchor_distances))
    mean_energy = float(np.mean(energies))
    verifier_pass_rates = (per_verifier_passes / float(n_puzzles)).tolist()

    encoder_pc = encoder.param_count()
    energy_pc = energy_mlp.param_count()
    refiner_pc = refiner.param_count()
    total_pc = encoder_pc + energy_pc + refiner_pc

    return {
        "encoder_param_count": int(encoder_pc),
        "energy_mlp_param_count": int(energy_pc),
        "refiner_param_count": int(refiner_pc),
        "total_param_count": int(total_pc),
        "n_puzzles_run": int(n_puzzles),
        "valid_action_fraction": float(valid_fraction),
        "mean_anchor_distance": float(mean_anchor_distance),
        "mean_energy": float(mean_energy),
        "verifier_pass_rates": [float(p) for p in verifier_pass_rates],
        "conditional_acceptance_matrix": cap_matrix.matrix().tolist(),
        "anchor_tracker_initialized": True,
        "conditional_acceptance_matrix_initialized": True,
    }


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def _verdict_for(valid_fraction: float, prototype_components_present: bool) -> str:
    """Map raw measurements onto the four allowed honest verdicts."""
    if not prototype_components_present:
        return "prototype_partial_components_missing"
    if valid_fraction >= 0.50:
        return "prototype_meets_acceptance_gate"
    return "prototype_below_50pct_valid"


def build_phase5a_artifact(
    summary: dict[str, Any],
    *,
    seed: int = 1222,
) -> dict[str, Any]:
    """Build the JSON-serialisable Exp 1222 artifact from a run summary.

    All of the fields required by exp_NEXT_A's acceptance criteria are
    flattened to the top level of the artifact so the conductor's
    schema validator can spot-check them without descending into
    nested objects.
    """
    valid_fraction = float(summary["valid_action_fraction"])
    components_ok = bool(summary["anchor_tracker_initialized"]) and bool(
        summary["conditional_acceptance_matrix_initialized"]
    )
    prototype_ready = components_ok and valid_fraction >= 0.50
    artifact = {
        "experiment": "1222_phase5a_insitu_prototype",
        "schema_version": SCHEMA_VERSION,
        "run_date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seed": int(seed),
        "status": "success" if prototype_ready else "below_acceptance_gate",
        "encoder_param_count": int(summary["encoder_param_count"]),
        "energy_mlp_param_count": int(summary["energy_mlp_param_count"]),
        "refiner_param_count": int(summary["refiner_param_count"]),
        "total_param_count": int(summary["total_param_count"]),
        "n_puzzles_run": int(summary["n_puzzles_run"]),
        "valid_action_fraction": valid_fraction,
        "mean_anchor_distance": float(summary["mean_anchor_distance"]),
        "mean_energy": float(summary["mean_energy"]),
        "verifier_pass_rates": list(summary["verifier_pass_rates"]),
        "conditional_acceptance_matrix": list(summary["conditional_acceptance_matrix"]),
        "anchor_tracker_initialized": bool(summary["anchor_tracker_initialized"]),
        "conditional_acceptance_matrix_initialized": bool(
            summary["conditional_acceptance_matrix_initialized"]
        ),
        "phase5a_prototype_ready": bool(prototype_ready),
        "honest_verdict": _verdict_for(valid_fraction, components_ok),
    }
    return artifact


def write_phase5a_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    """Write the artifact dict to ``path`` as pretty JSON."""
    Path(path).write_text(json.dumps(artifact, indent=2, sort_keys=True))
