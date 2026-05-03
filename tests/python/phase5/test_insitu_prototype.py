"""Tests for the Phase 5-A in-situ training prototype.

Spec coverage: REQ-KONA-008 (snap-to-action reuse), REQ-KONA-012
(active-inference latent), and the in-situ-training-phase5-derisking
change proposal's exp_NEXT_A acceptance gate.

Scenario coverage:
    SCENARIO-PHASE5A-001 — encoder produces latent in (-1, 1)^16 with
        ~10K parameters.
    SCENARIO-PHASE5A-002 — energy MLP produces a clamped scalar in
        [0, 1] with ~10K parameters.
    SCENARIO-PHASE5A-003 — refiner produces a 16-dim latent in (-1, 1)^16.
    SCENARIO-PHASE5A-004 — snap_to_action returns ``ACTIONS_PER_SEQUENCE``
        in-bounds (row, col, color) actions for any input latent.
    SCENARIO-PHASE5A-005 — quadrant-anchored snap yields four distinct
        cells when the latent is the zero vector (no-duplicate verifier
        trivially passes).
    SCENARIO-PHASE5A-006 — three-verifier ensemble runs and returns a
        boolean tuple of length 3.
    SCENARIO-PHASE5A-007 — vacuous-anchor tracker returns finite,
        non-negative L2 distances.
    SCENARIO-PHASE5A-008 — conditional-acceptance matrix accumulates
        joint counts and returns a column-conditioned probability matrix.
    SCENARIO-PHASE5A-009 — end-to-end prototype run on 100 random 5x5
        puzzles meets the ≥50% valid-action acceptance gate.
    SCENARIO-PHASE5A-010 — artifact builder emits all required fields
        with the correct honest_verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.phase5.insitu_prototype import (
    ACTIONS_PER_SEQUENCE,
    GRID_SIZE,
    LATENT_DIM,
    NUM_COLORS,
    N_VERIFIERS,
    QUADRANT_ANCHORS,
    SCHEMA_VERSION,
    ConditionalAcceptanceProbMatrix,
    InSituEncoder,
    InSituEnergyMLP,
    InSituRefiner,
    VacuousAnchorTracker,
    apply_action_sequence,
    build_phase5a_artifact,
    generate_random_5x5_puzzle,
    run_phase5a_prototype,
    snap_to_action,
    verifier_outcomes,
    verify_action_sequence,
    write_phase5a_artifact,
)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5A-001: encoder shape, range, parameter count
# ---------------------------------------------------------------------------


def test_encoder_produces_16d_latent_in_open_unit_cube_phase5a_001() -> None:
    """REQ-KONA-012, SCENARIO-PHASE5A-001."""
    encoder = InSituEncoder.init(seed=0)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    z = encoder.forward(grid)
    assert z.shape == (LATENT_DIM,)
    # tanh keeps z strictly in (-1, 1).
    assert np.all(z > -1.0) and np.all(z < 1.0)


def test_encoder_param_count_around_10k_phase5a_001() -> None:
    """SCENARIO-PHASE5A-001: encoder has ~10K parameters."""
    encoder = InSituEncoder.init(seed=0)
    pc = encoder.param_count()
    assert 8_000 <= pc <= 15_000, f"encoder param count {pc} outside ~10K target"


def test_encoder_rejects_wrong_grid_shape() -> None:
    encoder = InSituEncoder.init(seed=0)
    bad = np.zeros((4, 4), dtype=np.int32)
    with pytest.raises(ValueError):
        encoder.forward(bad)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5A-002: energy MLP shape, range, parameter count
# ---------------------------------------------------------------------------


def test_energy_mlp_produces_scalar_in_unit_interval_phase5a_002() -> None:
    """SCENARIO-PHASE5A-002: clamped energy in [0, 1]."""
    energy = InSituEnergyMLP.init(seed=1)
    z = np.zeros(LATENT_DIM, dtype=np.float64)
    e = energy.forward(z)
    assert isinstance(e, float)
    assert 0.0 <= e <= 1.0


def test_energy_mlp_param_count_around_10k_phase5a_002() -> None:
    """SCENARIO-PHASE5A-002: energy MLP has ~10K parameters."""
    energy = InSituEnergyMLP.init(seed=1)
    pc = energy.param_count()
    assert 8_000 <= pc <= 12_000, f"energy MLP param count {pc} outside ~10K target"


def test_energy_mlp_unclamped_returns_raw_scalar() -> None:
    energy = InSituEnergyMLP.init(seed=1)
    energy.clamp_output = False
    z = np.zeros(LATENT_DIM, dtype=np.float64)
    e = energy.forward(z)
    # Without sigmoid the output is unbounded — we just check finiteness.
    assert np.isfinite(e)


def test_energy_mlp_rejects_wrong_latent_size() -> None:
    energy = InSituEnergyMLP.init(seed=1)
    with pytest.raises(ValueError):
        energy.forward(np.zeros(5))


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5A-003: refiner shape and range
# ---------------------------------------------------------------------------


def test_refiner_returns_16d_latent_in_open_unit_cube_phase5a_003() -> None:
    refiner = InSituRefiner.init(seed=2)
    z = np.zeros(LATENT_DIM, dtype=np.float64)
    z_out = refiner.forward(z)
    assert z_out.shape == (LATENT_DIM,)
    assert np.all(z_out > -1.0) and np.all(z_out < 1.0)


def test_refiner_param_count_is_significant() -> None:
    refiner = InSituRefiner.init(seed=2)
    pc = refiner.param_count()
    assert pc >= 20_000, f"refiner expected to dominate decoder params; got {pc}"


def test_refiner_rejects_wrong_latent_size() -> None:
    refiner = InSituRefiner.init(seed=2)
    with pytest.raises(ValueError):
        refiner.forward(np.zeros(5))


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5A-004: snap_to_action shape and bounds
# ---------------------------------------------------------------------------


def test_snap_to_action_returns_n_actions_with_valid_components_phase5a_004() -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        z = rng.uniform(-1.0, 1.0, size=LATENT_DIM)
        actions = snap_to_action(z)
        assert len(actions) == ACTIONS_PER_SEQUENCE
        for r, c, v in actions:
            assert 0 <= r < GRID_SIZE
            assert 0 <= c < GRID_SIZE
            assert 0 <= v < NUM_COLORS


def test_snap_to_action_rejects_too_short_latent() -> None:
    with pytest.raises(ValueError):
        snap_to_action(np.zeros(3))


def test_snap_to_action_rejects_too_many_actions() -> None:
    n_too_many = len(QUADRANT_ANCHORS) + 1
    big_z = np.zeros(n_too_many * 4)  # enough dims to bypass size check
    with pytest.raises(ValueError, match="quadrant anchors"):
        snap_to_action(big_z, n_actions=n_too_many)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5A-005: quadrant-anchored snap yields distinct cells
# ---------------------------------------------------------------------------


def test_quadrant_snap_yields_distinct_cells_for_zero_latent_phase5a_005() -> None:
    """Zero latent → centre-of-quadrant cells; the four cells must differ."""
    actions = snap_to_action(np.zeros(LATENT_DIM))
    cells = [(r, c) for r, c, _ in actions]
    assert len(set(cells)) == ACTIONS_PER_SEQUENCE


def test_quadrant_snap_yields_distinct_cells_for_random_latent_phase5a_005() -> None:
    rng = np.random.default_rng(0)
    duplicate_count = 0
    for _ in range(50):
        z = rng.uniform(-1.0, 1.0, size=LATENT_DIM)
        actions = snap_to_action(z)
        cells = [(r, c) for r, c, _ in actions]
        if len(set(cells)) != len(cells):
            duplicate_count += 1
    # Quadrants overlap at row=2 and col=2 only at extremes; tolerate a few
    # but require the bulk of random samples to be duplicate-free.
    assert duplicate_count <= 5


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5A-006: three-verifier ensemble
# ---------------------------------------------------------------------------


def test_verifier_outcomes_returns_three_bools_phase5a_006() -> None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    actions = [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 1)]
    outs = verifier_outcomes(actions, grid)
    assert len(outs) == N_VERIFIERS
    for o in outs:
        assert isinstance(o, bool)
    assert all(outs) is True


def test_verifier_in_bounds_rejects_out_of_range_action() -> None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    actions = [(GRID_SIZE, 0, 0), (0, 0, 0), (1, 1, 0), (2, 2, 0)]
    outs = verifier_outcomes(actions, grid)
    assert outs[0] is False


def test_verifier_changes_grid_rejects_no_op_sequence() -> None:
    grid = np.full((GRID_SIZE, GRID_SIZE), 2, dtype=np.int32)
    # All actions write the same value the cells already hold.
    actions = [(0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)]
    outs = verifier_outcomes(actions, grid)
    assert outs[1] is False


def test_verifier_no_duplicate_cells_rejects_overlapping_actions() -> None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    actions = [(0, 0, 1), (0, 0, 2), (1, 1, 3), (2, 2, 0)]
    outs = verifier_outcomes(actions, grid)
    assert outs[2] is False


def test_verify_action_sequence_is_and_composed() -> None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    good = [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 1)]
    assert verify_action_sequence(good, grid) is True
    bad = [(0, 0, 1), (0, 0, 2), (1, 1, 3), (2, 2, 0)]
    assert verify_action_sequence(bad, grid) is False


def test_apply_action_sequence_skips_out_of_bounds() -> None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    actions = [(GRID_SIZE, 0, 1), (0, 0, 2)]
    new_grid = apply_action_sequence(grid, actions)
    # In-bounds action applied; out-of-bounds skipped silently.
    assert new_grid[0, 0] == 2
    # Original grid not mutated.
    assert grid[0, 0] == 0


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5A-007: vacuous-anchor tracker
# ---------------------------------------------------------------------------


def test_anchor_tracker_returns_nonnegative_finite_distance_phase5a_007() -> None:
    tracker = VacuousAnchorTracker.default()
    rng = np.random.default_rng(0)
    for _ in range(10):
        z = rng.uniform(-1.0, 1.0, size=LATENT_DIM)
        d = tracker.distance(z)
        assert np.isfinite(d)
        assert d >= 0.0


def test_anchor_tracker_zero_distance_at_zero_anchor() -> None:
    tracker = VacuousAnchorTracker.default()
    d = tracker.distance(np.zeros(LATENT_DIM))
    assert d == pytest.approx(0.0)


def test_anchor_tracker_rejects_wrong_dim() -> None:
    tracker = VacuousAnchorTracker.default()
    with pytest.raises(ValueError):
        tracker.distance(np.zeros(LATENT_DIM + 1))


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5A-008: conditional-acceptance matrix
# ---------------------------------------------------------------------------


def test_cap_matrix_records_joint_and_marginal_counts_phase5a_008() -> None:
    cap = ConditionalAcceptanceProbMatrix(n_verifiers=3)
    cap.record([True, True, False])
    cap.record([True, False, False])
    cap.record([True, True, True])
    assert cap.n_observations == 3
    # Verifier 0 fired in all 3, verifier 1 in 2, verifier 2 in 1.
    assert int(cap.counts_marginal[0]) == 3
    assert int(cap.counts_marginal[1]) == 2
    assert int(cap.counts_marginal[2]) == 1


def test_cap_matrix_returns_column_conditioned_probabilities_phase5a_008() -> None:
    cap = ConditionalAcceptanceProbMatrix(n_verifiers=3)
    cap.record([True, True, False])
    cap.record([True, False, True])
    m = cap.matrix()
    # P(0 | 1) = P(0 ∧ 1) / P(1) = 1 / 1 = 1.0
    assert m[0, 1] == pytest.approx(1.0)
    # P(2 | 1) = 0 / 1 = 0.0
    assert m[2, 1] == pytest.approx(0.0)
    # No verifier_j observed → column j of zeros (no division by zero).
    cap_empty = ConditionalAcceptanceProbMatrix(n_verifiers=3)
    m_empty = cap_empty.matrix()
    assert m_empty.shape == (3, 3)
    assert np.all(m_empty == 0.0)


def test_cap_matrix_rejects_wrong_outcome_length() -> None:
    cap = ConditionalAcceptanceProbMatrix(n_verifiers=3)
    with pytest.raises(ValueError):
        cap.record([True, False])


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5A-009: end-to-end run meets ≥50% acceptance gate
# ---------------------------------------------------------------------------


def test_generate_random_5x5_puzzle_returns_int_grid() -> None:
    rng = np.random.default_rng(0)
    g = generate_random_5x5_puzzle(rng)
    assert g.shape == (GRID_SIZE, GRID_SIZE)
    assert g.dtype == np.int32
    assert g.min() >= 0
    assert g.max() < NUM_COLORS


def test_run_phase5a_prototype_meets_acceptance_gate_phase5a_009() -> None:
    """exp_NEXT_A acceptance gate: ≥50% valid action sequences across 100 puzzles."""
    summary = run_phase5a_prototype(n_puzzles=100, seed=1222)
    assert summary["n_puzzles_run"] == 100
    assert summary["valid_action_fraction"] >= 0.50
    assert summary["anchor_tracker_initialized"] is True
    assert summary["conditional_acceptance_matrix_initialized"] is True
    assert summary["encoder_param_count"] >= 8_000
    assert summary["energy_mlp_param_count"] >= 8_000
    assert summary["total_param_count"] >= 30_000
    assert 0.0 <= summary["mean_energy"] <= 1.0
    cam = summary["conditional_acceptance_matrix"]
    assert len(cam) == N_VERIFIERS and len(cam[0]) == N_VERIFIERS


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5A-010: artifact builder emits all required fields
# ---------------------------------------------------------------------------


def test_build_phase5a_artifact_emits_required_fields_phase5a_010(tmp_path: Path) -> None:
    summary = run_phase5a_prototype(n_puzzles=20, seed=1222)
    artifact = build_phase5a_artifact(summary, seed=1222)
    required = {
        "experiment",
        "schema_version",
        "run_date",
        "seed",
        "status",
        "encoder_param_count",
        "energy_mlp_param_count",
        "refiner_param_count",
        "total_param_count",
        "n_puzzles_run",
        "valid_action_fraction",
        "mean_anchor_distance",
        "mean_energy",
        "verifier_pass_rates",
        "conditional_acceptance_matrix",
        "anchor_tracker_initialized",
        "conditional_acceptance_matrix_initialized",
        "phase5a_prototype_ready",
        "honest_verdict",
    }
    missing = required - artifact.keys()
    assert not missing, f"missing artifact fields: {missing}"
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["honest_verdict"] in {
        "prototype_meets_acceptance_gate",
        "prototype_below_50pct_valid",
        "prototype_partial_components_missing",
    }
    if artifact["valid_action_fraction"] >= 0.50:
        assert artifact["phase5a_prototype_ready"] is True
        assert artifact["honest_verdict"] == "prototype_meets_acceptance_gate"

    out = tmp_path / "art.json"
    write_phase5a_artifact(artifact, out)
    loaded = json.loads(out.read_text())
    assert loaded["experiment"] == "1222_phase5a_insitu_prototype"


def test_build_phase5a_artifact_below_gate_marks_prototype_not_ready() -> None:
    """Synthetic summary with valid_fraction < 0.5 must produce the honest fail verdict."""
    bogus_summary = {
        "encoder_param_count": 10_000,
        "energy_mlp_param_count": 10_000,
        "refiner_param_count": 30_000,
        "total_param_count": 50_000,
        "n_puzzles_run": 100,
        "valid_action_fraction": 0.10,
        "mean_anchor_distance": 1.0,
        "mean_energy": 0.5,
        "verifier_pass_rates": [1.0, 0.5, 0.2],
        "conditional_acceptance_matrix": [[1.0, 0.0, 0.0]] * 3,
        "anchor_tracker_initialized": True,
        "conditional_acceptance_matrix_initialized": True,
    }
    artifact = build_phase5a_artifact(bogus_summary, seed=0)
    assert artifact["phase5a_prototype_ready"] is False
    assert artifact["honest_verdict"] == "prototype_below_50pct_valid"


def test_build_phase5a_artifact_missing_components_marks_partial() -> None:
    bogus_summary = {
        "encoder_param_count": 10_000,
        "energy_mlp_param_count": 10_000,
        "refiner_param_count": 30_000,
        "total_param_count": 50_000,
        "n_puzzles_run": 100,
        "valid_action_fraction": 0.95,
        "mean_anchor_distance": 1.0,
        "mean_energy": 0.5,
        "verifier_pass_rates": [1.0, 1.0, 1.0],
        "conditional_acceptance_matrix": [[1.0, 1.0, 1.0]] * 3,
        "anchor_tracker_initialized": False,
        "conditional_acceptance_matrix_initialized": True,
    }
    artifact = build_phase5a_artifact(bogus_summary, seed=0)
    assert artifact["phase5a_prototype_ready"] is False
    assert artifact["honest_verdict"] == "prototype_partial_components_missing"
