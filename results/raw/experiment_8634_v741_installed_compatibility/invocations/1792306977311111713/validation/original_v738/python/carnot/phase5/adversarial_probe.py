"""Phase 5-C adversarial probe — three attack classes vs Phase 5-A+B prototype.

exp_NEXT_C from openspec/change-proposals/in-situ-training-phase5-derisking.md.

Researcher summary:
    Hostile-reviewer round that tries to break the Phase 5-A+B prototype
    with three adversarial attack classes.  Runs BEFORE any Phase 5 scaling
    decision is committed (Phase Prototype + Empirical Validation +
    Adversarial Check discipline, CLAUDE.md).

Detailed explanation (verbose-layman rule):
    The Phase 5-A prototype showed that the in-situ training substrate runs,
    and Phase 5-B showed it is stable over 1000 queries.  But "runs and is
    stable" does not prove the architecture is grounded.  A hostile reviewer
    would ask: can the model *appear* to pass our gates while not actually
    working?

    Attack 1 — Single-verifier gaming: what if one verifier is much easier
    to pass than the others?  The model could learn to satisfy that one
    verifier alone, score it > 0.9, and our composite score would still look
    reasonable.  We check: how often does V0 (in-bounds) pass alone while the
    AND-composed composite fails?

    Attack 2 — Pairwise correlation exploitation: what if two verifiers are
    correlated — i.e., whenever one passes, the other almost always does too?
    Then our k=3 ensemble is really k=1 in disguise.  We measure the
    conditional acceptance probability matrix P(V_i passes | V_j passes) and
    look for off-diagonal entries > 0.7.

    Attack 3 — Joint null-space gradient attack: what if the energy MLP (the
    continuous surrogate that gradient-ascent runs on) has high values in
    regions where the actual verifiers would reject?  An adversary could
    gradient-ascend the MLP output to > 0.8 while the verifiers still say
    "invalid".  We simulate this by starting from invalid states and running
    50 steps of finite-difference gradient ascent.

Spec coverage:
    REQ-KONA-018 (Phase 5-C adversarial probe acceptance gate),
    exp_NEXT_C in openspec/change-proposals/in-situ-training-phase5-derisking.md,
    CLAUDE.md "Phase Prototype + Empirical Validation + Adversarial Check Discipline".
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.phase5.insitu_prototype import (
    LATENT_DIM,
    LATENTS_PER_ACTION,
    N_VERIFIERS,
    NUM_COLORS,
    ConditionalAcceptanceProbMatrix,
    InSituEnergyMLP,
    InSituEncoder,
    InSituRefiner,
    generate_random_5x5_puzzle,
    snap_to_action,
    verifier_outcomes,
    verify_action_sequence,
)

# V0 (in_bounds) is the "Z3MathVerifier" analog: the structural constraint
# that the snap decoder guarantees to almost always satisfy, making it the
# easiest single verifier to pass in isolation.
Z3_VERIFIER_IDX: int = 0

# Defense thresholds from the exp_NEXT_C acceptance gate specification.
ATTACK1_GAMING_THRESHOLD: float = 0.10   # < 10 % single-verifier gaming
ATTACK2_CORRELATION_THRESHOLD: float = 0.70  # < 0.70 max off-diagonal corr
ATTACK3_GAMING_THRESHOLD: float = 0.05   # < 5 % joint null-space gaming

SCHEMA_VERSION = "1224.phase5c.adversarial_probe.v1"


# ---------------------------------------------------------------------------
# Invalid-start construction helper
# ---------------------------------------------------------------------------


def _make_invalid_z_for_grid(
    grid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct a latent z that produces a no-change action sequence on grid.

    By writing the SAME color that already exists at each snap-decoded
    position, the action sequence passes V0 (in_bounds) but fails V1
    (changes_grid) because no cell is actually modified.

    Construction:
        1. Sample a small random z near zero so snap gives stable positions.
        2. Decode to 4 (row, col, _) positions.
        3. For each action k at (r, c), solve for the z_val component that
           makes snap produce grid[r, c] exactly.

    Inverse of snap's linear mapping:
        snap produces  val = round((z_val + 1) / 2 * (num_colors - 1))
        so the midpoint z_val for val = target_color is
            z_val_exact = target_color * 2/(num_colors-1) - 1
        which rounds exactly to target_color for any z_val within
        ±1/(num_colors-1) of the midpoint.
    """
    z = rng.uniform(-0.1, 0.1, LATENT_DIM).astype(np.float64)
    actions = snap_to_action(z)
    z_invalid = z.copy()
    for k, (r, c, _) in enumerate(actions):
        target_color = int(grid[r, c])
        z_val_exact = float(target_color) * (2.0 / float(NUM_COLORS - 1)) - 1.0
        z_invalid[LATENTS_PER_ACTION * k + 2] = np.clip(z_val_exact, -0.99, 0.99)
    return z_invalid


# ---------------------------------------------------------------------------
# Attack 1 — single-verifier gaming
# ---------------------------------------------------------------------------


def run_attack1_single_verifier_gaming(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
    n_samples: int = 200,
    seed: int = 1224,
) -> float:
    """Single-verifier gaming: fraction where V0 passes but k=3 composite fails.

    For each of n_samples random (grid, z) pairs we compute:
        Z3-verifier score  = float(V0 outcome) — binary 1.0 or 0.0.
        composite k=3 score = mean(V0, V1, V2) — in {0, 1/3, 2/3, 1}.

    Gaming is detected when Z3_score > 0.9 (V0 passes) AND composite < 0.5,
    meaning V1 AND V2 both fail and the AND-composed ensemble rejects despite
    V0 passing alone.

    A rate above ATTACK1_GAMING_THRESHOLD (10 %) signals that the architecture
    can be exploited by satisfying V0 alone.
    """
    rng = np.random.default_rng(seed)
    gaming_count = 0

    for _ in range(n_samples):
        grid = generate_random_5x5_puzzle(rng)
        z = encoder.forward(grid)
        z_refined = refiner.forward(z)
        actions = snap_to_action(z_refined)
        outcomes = verifier_outcomes(actions, grid)

        z3_score = float(outcomes[Z3_VERIFIER_IDX])
        composite_score = float(sum(outcomes)) / float(N_VERIFIERS)

        if z3_score > 0.9 and composite_score < 0.5:
            gaming_count += 1

    return float(gaming_count) / float(n_samples)


# ---------------------------------------------------------------------------
# Attack 2 — pairwise correlation exploitation
# ---------------------------------------------------------------------------


def run_attack2_pairwise_correlation(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
    n_samples: int = 200,
    seed: int = 1225,
) -> tuple[float, np.ndarray]:
    """Pairwise correlation: max off-diagonal P(V_i passes | V_j passes).

    Builds the empirical N×N conditional-acceptance matrix over n_samples
    queries and returns the maximum off-diagonal value.  A value above
    ATTACK2_CORRELATION_THRESHOLD (0.70) indicates correlated blind spots:
    two verifiers that co-occur so reliably that the AND-composition provides
    less independence than its k=3 label implies.

    Returns:
        (max_off_diagonal_correlation, full_conditional_matrix N×N ndarray)
    """
    rng = np.random.default_rng(seed)
    cap = ConditionalAcceptanceProbMatrix(n_verifiers=N_VERIFIERS)

    for _ in range(n_samples):
        grid = generate_random_5x5_puzzle(rng)
        z = encoder.forward(grid)
        z_refined = refiner.forward(z)
        actions = snap_to_action(z_refined)
        outcomes = verifier_outcomes(actions, grid)
        cap.record(outcomes)

    matrix = cap.matrix()
    off_diag_mask = ~np.eye(N_VERIFIERS, dtype=bool)
    max_corr = float(np.max(matrix[off_diag_mask]))
    return max_corr, matrix


# ---------------------------------------------------------------------------
# Attack 3 — joint null-space gradient attack
# ---------------------------------------------------------------------------


def _numerical_gradient(
    fn: Any,
    z: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Finite-difference central-difference gradient of scalar fn w.r.t. z.

    grad[i] = (fn(z + eps*e_i) - fn(z - eps*e_i)) / (2*eps).

    Used in Attack 3's gradient-ascent loop.  Pure numpy so no autodiff
    framework is needed; at 16 dims this costs 32 MLP forward passes per
    gradient step — well within budget for 50-step trajectories.
    """
    grad = np.zeros(len(z), dtype=np.float64)
    for i in range(len(z)):
        z_plus = z.copy()
        z_plus[i] += eps
        z_minus = z.copy()
        z_minus[i] -= eps
        grad[i] = (fn(z_plus) - fn(z_minus)) / (2.0 * eps)
    return grad


def run_attack3_joint_nullspace(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
    energy_mlp: InSituEnergyMLP,
    n_starts: int = 20,
    n_steps: int = 50,
    step_size: float = 0.05,
    seed: int = 1226,
    _max_rejection_attempts: int = 5000,
) -> float:
    """Joint null-space attack: gradient-ascend energy proxy from invalid starts.

    The energy MLP is the continuous surrogate the in-situ training loop uses
    for gradient steps.  This attack asks: can an adversary find latent z
    values where the energy MLP predicts "valid" (output > 0.8) while the
    actual boolean verifiers still reject?

    Procedure:
        1. Collect n_starts (grid, z) pairs where verify_action_sequence
           returns False.  Uses rejection sampling first (up to
           _max_rejection_attempts attempts); if budget exhausted, falls back
           to direct construction (_make_invalid_z_for_grid).
        2. For each pair, gradient-ascend the energy MLP output for n_steps,
           clamping z to (-1, 1)^16 after each step.
        3. Gaming detected: energy_mlp.forward(z_final) > 0.8 AND
           verify_action_sequence(snap_to_action(z_final), grid) is False.

    The _max_rejection_attempts parameter is exposed for testing only; the
    default of 5000 is correct for production use.

    Returns:
        Fraction of n_starts where joint null-space gaming is detected.
    """
    if n_starts == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    invalid_starts: list[tuple[np.ndarray, np.ndarray]] = []

    # Phase 1: rejection sampling — fast path for naturally invalid starts.
    for _ in range(_max_rejection_attempts):
        if len(invalid_starts) >= n_starts:
            break
        z_rand = rng.uniform(-1.0, 1.0, LATENT_DIM)
        grid = generate_random_5x5_puzzle(rng)
        if not verify_action_sequence(snap_to_action(z_rand), grid):
            invalid_starts.append((z_rand.copy(), grid.copy()))

    # Phase 2: direct construction for any remaining needed starts.
    # _make_invalid_z_for_grid writes same-color-as-existing at each position,
    # guaranteeing V1 (changes_grid) fails, so verify returns False reliably.
    while len(invalid_starts) < n_starts:
        grid = generate_random_5x5_puzzle(rng)
        z_inv = _make_invalid_z_for_grid(grid, rng)
        if not verify_action_sequence(snap_to_action(z_inv), grid):
            invalid_starts.append((z_inv, grid.copy()))

    gaming_count = 0
    actual = invalid_starts[:n_starts]

    for z_start, grid in actual:
        z = z_start.copy()

        def proxy(z_in: np.ndarray) -> float:
            # Wrapper keeps grid fixed; gradient is w.r.t. z_in only.
            return energy_mlp.forward(z_in)

        for _ in range(n_steps):
            grad = _numerical_gradient(proxy, z)
            z = np.clip(z + step_size * grad, -1.0, 1.0)

        final_energy = energy_mlp.forward(z)
        verifiers_pass = verify_action_sequence(snap_to_action(z), grid)

        if final_energy > 0.8 and not verifiers_pass:
            gaming_count += 1

    return float(gaming_count) / float(len(actual))


# ---------------------------------------------------------------------------
# Defense verdict
# ---------------------------------------------------------------------------


def evaluate_defense_verdict(
    gaming_rate_attack1: float,
    pairwise_max_correlation: float,
    joint_gaming_rate: float,
    conditional_matrix: np.ndarray,
) -> dict[str, Any]:
    """Map the three attack measurements onto defense verdict fields.

    Returns a dict with keys:
        attack1_blocked, attack2_blocked, attack3_blocked,
        all_attacks_blocked, failure_modes_discovered,
        architectural_revision_if_needed, honest_verdict.
    """
    attack1_blocked = gaming_rate_attack1 < ATTACK1_GAMING_THRESHOLD
    attack2_blocked = pairwise_max_correlation < ATTACK2_CORRELATION_THRESHOLD
    attack3_blocked = joint_gaming_rate < ATTACK3_GAMING_THRESHOLD
    all_attacks_blocked = attack1_blocked and attack2_blocked and attack3_blocked

    failure_modes: list[str] = []
    if not attack1_blocked:
        failure_modes.append(
            f"single_verifier_gaming: V0(in_bounds) gaming rate "
            f"{gaming_rate_attack1:.3f} >= 0.10 — V0 alone passes without "
            f"full composite acceptance"
        )
    if not attack2_blocked:
        failure_modes.append(
            f"pairwise_verifier_correlation: max P(V_i|V_j) = "
            f"{pairwise_max_correlation:.3f} >= 0.70 — correlated blind spots "
            f"detected; the quadrant-anchor decoder structurally guarantees V0 "
            f"(in_bounds) for all inputs and V2 (no_duplicate_cells) for most, "
            f"making their co-occurrence vacuous and reducing effective "
            f"independent coverage from k=3 to approximately k=1 (only V1 "
            f"changes_grid provides genuinely independent signal)"
        )
    if not attack3_blocked:
        failure_modes.append(
            f"joint_nullspace_gaming: gradient-ascent gaming rate "
            f"{joint_gaming_rate:.3f} >= 0.05 — energy MLP can be driven above "
            f"0.8 while actual verifiers reject, indicating an exploitable null "
            f"space in the untrained surrogate"
        )

    if all_attacks_blocked:
        architectural_revision = "none"
        honest_verdict = "all_attacks_blocked_architecture_validated"
    else:
        architectural_revision = _build_revision_text(
            attack1_blocked, attack2_blocked, attack3_blocked
        )
        honest_verdict = "partial_attack_success_revision_needed"

    return {
        "attack1_blocked": bool(attack1_blocked),
        "attack2_blocked": bool(attack2_blocked),
        "attack3_blocked": bool(attack3_blocked),
        "all_attacks_blocked": bool(all_attacks_blocked),
        "failure_modes_discovered": failure_modes,
        "architectural_revision_if_needed": architectural_revision,
        "honest_verdict": honest_verdict,
    }


def _build_revision_text(
    attack1_blocked: bool,
    attack2_blocked: bool,
    attack3_blocked: bool,
) -> str:
    """Build an architectural revision recommendation from which attacks succeeded."""
    parts: list[str] = []
    if not attack2_blocked:
        parts.append(
            "Replace V0(in_bounds) and V2(no_duplicate_cells) in the k=3 ensemble "
            "with structurally independent verifiers that have non-trivial fail "
            "modes on the quadrant-anchor decoder output. Candidates: (a) V3: "
            "require each written color differs from the original color at that "
            "cell — makes V1 and V3 jointly necessary for acceptance; (b) V4: "
            "require no two adjacent actions target neighbouring cells — "
            "orthogonal to the change check. Root cause: snap_to_action quadrant "
            "anchors structurally guarantee V0 and near-guarantee V2, so they "
            "provide no independent discriminative signal. Per Spera Theorem 9.2 "
            "(arXiv:2603.15973): verifier ensembles must be designed for "
            "joint-kernel orthogonality, not just individual coverage."
        )
    if not attack1_blocked:
        parts.append(
            "Add a stricter primary verifier with non-trivial fail probability "
            "on bounded-snap outputs (e.g., a semantic correctness check beyond "
            "structural bounds)."
        )
    if not attack3_blocked:
        parts.append(
            "Pre-train or fine-tune the energy MLP on actual verifier outcomes "
            "before gradient ascent is used in-situ to eliminate the exploitable "
            "null space in the random-initialisation surrogate."
        )
    return " | ".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_phase5c_artifact(
    *,
    start_time: _dt.datetime,
    seed: int,
    gaming_rate_attack1: float,
    pairwise_max_correlation: float,
    joint_gaming_rate: float,
    conditional_matrix: np.ndarray,
    verdict_dict: dict[str, Any],
    phase5b_stability_confirmed: bool,
) -> dict[str, Any]:
    """Build the JSON-serialisable Exp 1224 artifact from probe results."""
    end_time = _dt.datetime.now(_dt.timezone.utc)
    duration_s = (end_time - start_time).total_seconds()

    return {
        "experiment": "1224_phase5c_adversarial_probe",
        "schema_version": SCHEMA_VERSION,
        "run_date": start_time.isoformat(),
        "seed": int(seed),
        "status": "success",
        "duration_s": float(duration_s),
        # Required schema fields (REQ-KONA-018)
        "gaming_rate_attack1": float(gaming_rate_attack1),
        "pairwise_max_correlation": float(pairwise_max_correlation),
        "joint_gaming_rate": float(joint_gaming_rate),
        "attack1_blocked": verdict_dict["attack1_blocked"],
        "attack2_blocked": verdict_dict["attack2_blocked"],
        "attack3_blocked": verdict_dict["attack3_blocked"],
        "all_attacks_blocked": verdict_dict["all_attacks_blocked"],
        "failure_modes_discovered": verdict_dict["failure_modes_discovered"],
        "architectural_revision_if_needed": verdict_dict["architectural_revision_if_needed"],
        "adversarial_probe_complete": True,
        "honest_verdict": verdict_dict["honest_verdict"],
        # Diagnostic detail
        "conditional_acceptance_matrix": conditional_matrix.tolist(),
        "phase5b_stability_was_confirmed": bool(phase5b_stability_confirmed),
        "attack1_threshold": float(ATTACK1_GAMING_THRESHOLD),
        "attack2_threshold": float(ATTACK2_CORRELATION_THRESHOLD),
        "attack3_threshold": float(ATTACK3_GAMING_THRESHOLD),
    }


def write_phase5c_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    """Write the artifact dict to path as pretty-printed JSON."""
    Path(path).write_text(json.dumps(artifact, indent=2, sort_keys=True))
