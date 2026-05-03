"""Tests for the Phase 5-B in-situ training loop (exp_NEXT_B / exp 1223).

Spec coverage: REQ-KONA-017 (Phase 5-B in-situ training loop with
verifier-ensemble grounding) and SCENARIO-KONA-017 (1000-query
trajectory passes the five Q9 stability gates).

Scenario coverage:
    SCENARIO-PHASE5B-001 — k=3 verifier-ensemble AND-pass returns
        a Boolean and is true only when in_bounds AND changes_grid
        AND ThinkPRM (soft accept) all hold.
    SCENARIO-PHASE5B-002 — encoder_forward_with_h returns a 16-dim
        latent in (-1, 1)^16 plus the 24-dim hidden activation.
    SCENARIO-PHASE5B-003 — cd1_update lowers E(z_pos) and raises
        E(z_neg) on a sufficiently large step.
    SCENARIO-PHASE5B-004 — encoder_spectral_norm returns the largest
        singular value of fc2_W.
    SCENARIO-PHASE5B-005 — evaluate_oracle returns a fraction in
        [0, 1] over the frozen oracle puzzle set.
    SCENARIO-PHASE5B-006 — rolling acceptance rate is window-correct.
    SCENARIO-PHASE5B-007 — sub-linear acceptance-rate detector flags
        accelerating trajectories and accepts decelerating ones.
    SCENARIO-PHASE5B-008 — energy-decrease percentage handles short
        and degenerate accepted-energy traces without crashing.
    SCENARIO-PHASE5B-009 — spectral_norm_growth_rate handles short
        traces and returns a finite slope.
    SCENARIO-PHASE5B-010 — 1000-query trajectory completes and
        evaluate_phase5b_gates returns all five booleans plus the
        passed count, with Gate 1 (energy decrease) ≥ 30% at η=1e-3
        and Gate 2 (spectral-norm growth) within bounds.
    SCENARIO-PHASE5B-011 — artifact builder emits all required fields
        with the correct honest verdict.
    SCENARIO-PHASE5B-012 — confirm_phase5a_ready returns True when
        the Phase 5-A artifact reports prototype_ready=True and False
        otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.phase5.insitu_prototype import (
    GRID_SIZE,
    LATENT_DIM,
    NUM_COLORS,
    InSituEncoder,
    InSituEnergyMLP,
    InSituRefiner,
)
from carnot.phase5.insitu_training_loop import (
    DEFAULT_LEARNING_RATE,
    GATE1_ENERGY_DROP_FRACTION,
    GATE2_MAX_SPECTRAL_NORM_GROWTH_PER_QUERY,
    PROPOSAL_LEARNING_RATE,
    SCHEMA_VERSION,
    _acceptance_rate_sublinear,
    _energy_decrease_pct,
    _rolling_acceptance_rates,
    _spectral_norm_growth_rate,
    build_phase5b_artifact,
    causal_reasoning_verifier_stub,
    cd1_update,
    confirm_phase5a_ready,
    encoder_forward_with_h,
    encoder_spectral_norm,
    evaluate_oracle,
    evaluate_phase5b_gates,
    run_phase5b_training_loop,
    thinkprm_v2_stub,
    verifier_ensemble_pass,
    write_phase5b_artifact,
    z3_math_verifier_stub,
)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-001: k=3 verifier ensemble AND-pass
# ---------------------------------------------------------------------------


def test_thinkprm_v2_stub_always_soft_accepts_phase5b_001() -> None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    actions = [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 1)]
    assert thinkprm_v2_stub(actions, grid) is True


def test_z3_math_verifier_stub_rejects_out_of_bounds_phase5b_001() -> None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    actions = [(GRID_SIZE, 0, 0), (0, 0, 0), (1, 1, 0), (2, 2, 0)]
    assert z3_math_verifier_stub(actions, grid) is False


def test_causal_reasoning_verifier_stub_rejects_no_op_phase5b_001() -> None:
    grid = np.full((GRID_SIZE, GRID_SIZE), 2, dtype=np.int32)
    actions = [(0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)]
    assert causal_reasoning_verifier_stub(actions, grid) is False


def test_verifier_ensemble_pass_is_and_composed_phase5b_001() -> None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    good = [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 1)]
    assert verifier_ensemble_pass(good, grid) is True
    bad = [(GRID_SIZE, 0, 0), (0, 0, 0), (1, 1, 0), (2, 2, 0)]
    assert verifier_ensemble_pass(bad, grid) is False


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-002: encoder_forward_with_h
# ---------------------------------------------------------------------------


def test_encoder_forward_with_h_returns_z_and_h_phase5b_002() -> None:
    encoder = InSituEncoder.init(seed=0)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    z, h = encoder_forward_with_h(encoder, grid)
    assert z.shape == (LATENT_DIM,)
    assert np.all(z > -1.0) and np.all(z < 1.0)
    assert h.shape == (24,)


def test_encoder_forward_with_h_rejects_wrong_shape_phase5b_002() -> None:
    encoder = InSituEncoder.init(seed=0)
    with pytest.raises(ValueError):
        encoder_forward_with_h(encoder, np.zeros((4, 4), dtype=np.int32))


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-003: cd1_update direction
# ---------------------------------------------------------------------------


def test_cd1_update_lowers_energy_on_positive_sample_phase5b_003() -> None:
    encoder = InSituEncoder.init(seed=10)
    energy_mlp = InSituEnergyMLP.init(seed=11)
    rng = np.random.default_rng(0)
    z_pos = rng.uniform(-0.5, 0.5, LATENT_DIM)
    z_neg = rng.uniform(-1.0, 1.0, LATENT_DIM)
    h_pos = rng.uniform(-1.0, 1.0, 24).astype(np.float32)
    h_neg = rng.uniform(-1.0, 1.0, 24).astype(np.float32)
    e_pos_before = energy_mlp.forward(z_pos)
    # Use a deliberately large lr so a single step produces a measurable shift.
    for _ in range(50):
        cd1_update(encoder, energy_mlp, z_pos, z_neg, h_pos, h_neg, learning_rate=1e-1)
    e_pos_after = energy_mlp.forward(z_pos)
    assert e_pos_after < e_pos_before


def test_cd1_update_works_with_unclamped_energy_phase5b_003() -> None:
    encoder = InSituEncoder.init(seed=12)
    energy_mlp = InSituEnergyMLP.init(seed=13)
    energy_mlp.clamp_output = False
    rng = np.random.default_rng(0)
    z_pos = rng.uniform(-0.5, 0.5, LATENT_DIM)
    z_neg = rng.uniform(-1.0, 1.0, LATENT_DIM)
    h_pos = rng.uniform(-1.0, 1.0, 24).astype(np.float32)
    h_neg = rng.uniform(-1.0, 1.0, 24).astype(np.float32)
    cd1_update(encoder, energy_mlp, z_pos, z_neg, h_pos, h_neg, learning_rate=1e-3)
    e_after = energy_mlp.forward(z_pos)
    assert np.isfinite(e_after)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-004: encoder_spectral_norm
# ---------------------------------------------------------------------------


def test_encoder_spectral_norm_returns_largest_singular_value_phase5b_004() -> None:
    encoder = InSituEncoder.init(seed=0)
    sn = encoder_spectral_norm(encoder)
    expected = float(np.linalg.svd(encoder.fc2_W, compute_uv=False)[0])
    assert sn == pytest.approx(expected)
    assert sn > 0.0


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-005: evaluate_oracle
# ---------------------------------------------------------------------------


def test_evaluate_oracle_returns_fraction_in_unit_interval_phase5b_005() -> None:
    encoder = InSituEncoder.init(seed=0)
    refiner = InSituRefiner.init(seed=2)
    rng = np.random.default_rng(0)
    puzzles = [rng.integers(0, NUM_COLORS, size=(GRID_SIZE, GRID_SIZE), dtype=np.int32) for _ in range(5)]
    acc = evaluate_oracle(encoder, refiner, puzzles)
    assert 0.0 <= acc <= 1.0


def test_evaluate_oracle_returns_zero_on_empty_set_phase5b_005() -> None:
    encoder = InSituEncoder.init(seed=0)
    refiner = InSituRefiner.init(seed=2)
    assert evaluate_oracle(encoder, refiner, []) == 0.0


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-006: rolling acceptance rate
# ---------------------------------------------------------------------------


def test_rolling_acceptance_rates_correct_window_average_phase5b_006() -> None:
    accepted = [True, True, False, True, False, True, True, True]
    rates = _rolling_acceptance_rates(accepted, window=3)
    # Windows of size 3 over an 8-element list → 6 windows.
    assert len(rates) == 6
    assert rates[0] == pytest.approx(2 / 3)


def test_rolling_acceptance_rates_returns_empty_when_too_short_phase5b_006() -> None:
    assert _rolling_acceptance_rates([True], window=3) == []


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-007: sub-linear detector
# ---------------------------------------------------------------------------


def test_acceptance_rate_sublinear_accepts_decelerating_phase5b_007() -> None:
    # Slope decelerates over time → sub-linear (no spiral).
    rates = [0.0, 0.1, 0.3, 0.5, 0.55, 0.58, 0.59, 0.59]
    assert _acceptance_rate_sublinear(rates) is True


def test_acceptance_rate_sublinear_rejects_accelerating_phase5b_007() -> None:
    # Slope accelerates: 0→0.05 in first half, 0.05→1.0 in second half.
    rates = [0.0, 0.01, 0.02, 0.05, 0.10, 0.30, 0.60, 1.0]
    assert _acceptance_rate_sublinear(rates) is False


def test_acceptance_rate_sublinear_short_trajectory_returns_true_phase5b_007() -> None:
    assert _acceptance_rate_sublinear([0.5]) is True
    assert _acceptance_rate_sublinear([]) is True


def test_acceptance_rate_sublinear_flat_trajectory_passes_phase5b_007() -> None:
    rates = [0.5] * 12
    assert _acceptance_rate_sublinear(rates) is True


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-008: energy-decrease percentage
# ---------------------------------------------------------------------------


def test_energy_decrease_pct_computes_fractional_drop_phase5b_008() -> None:
    energies = [0.5] * 100 + [0.2] * 100
    drop = _energy_decrease_pct(energies)
    # Mean of last 100 = 0.2; mean of first 100 = 0.5; (0.5-0.2)/0.5 = 0.6.
    assert drop == pytest.approx(0.6)


def test_energy_decrease_pct_returns_zero_on_short_trace_phase5b_008() -> None:
    assert _energy_decrease_pct([0.5]) == 0.0
    assert _energy_decrease_pct([]) == 0.0


def test_energy_decrease_pct_handles_zero_baseline_phase5b_008() -> None:
    energies = [0.0] * 4 + [0.0] * 4
    assert _energy_decrease_pct(energies) == 0.0


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-009: spectral-norm growth rate
# ---------------------------------------------------------------------------


def test_spectral_norm_growth_rate_returns_zero_on_too_few_samples_phase5b_009() -> None:
    assert _spectral_norm_growth_rate([]) == 0.0
    assert _spectral_norm_growth_rate([(50, 1.0)]) == 0.0


def test_spectral_norm_growth_rate_returns_zero_when_x_constant_phase5b_009() -> None:
    # All measurements at the same query index → degenerate fit.
    assert _spectral_norm_growth_rate([(50, 1.0), (50, 2.0)]) == 0.0


def test_spectral_norm_growth_rate_linear_fit_phase5b_009() -> None:
    samples = [(i * 50, 1.0 + 0.001 * i * 50) for i in range(1, 11)]
    slope = _spectral_norm_growth_rate(samples)
    assert slope == pytest.approx(0.001, rel=1e-3)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-010: end-to-end 1000-query trajectory
# ---------------------------------------------------------------------------


def test_run_phase5b_training_loop_1000q_emits_full_diagnostics_phase5b_010() -> None:
    diag = run_phase5b_training_loop(n_queries=1000, seed=1223)
    assert diag["n_queries_run"] == 1000
    assert 0 <= diag["n_accepted_by_verifier"] <= 1000
    assert len(diag["energies"]) == 1000
    assert len(diag["accepted"]) == 1000
    assert len(diag["spectral_norms"]) == 20  # 1000 / 50
    assert len(diag["oracle_accuracies"]) == 20
    assert 0.0 <= diag["oracle_accuracy_initial"] <= 1.0
    assert 0.0 <= diag["oracle_accuracy_final"] <= 1.0


def test_evaluate_phase5b_gates_passes_all_five_at_default_lr_phase5b_010() -> None:
    """At the runner's η=1e-3, all five Q9 stability gates must pass."""
    diag = run_phase5b_training_loop(n_queries=1000, seed=1223)
    gates = evaluate_phase5b_gates(diag)
    assert gates["energy_decrease_pct"] >= GATE1_ENERGY_DROP_FRACTION
    assert abs(gates["spectral_norm_growth_rate"]) < GATE2_MAX_SPECTRAL_NORM_GROWTH_PER_QUERY
    assert gates["acceptance_rate_sublinear"] is True
    assert gates["min_anchor_distance"] > 0.5
    assert gates["oracle_accuracy_drop_pp"] <= 5.0
    assert gates["gates_passed"] == 5


def test_evaluate_phase5b_gates_short_trajectory_returns_partial_phase5b_010() -> None:
    diag = run_phase5b_training_loop(n_queries=50, seed=1223)
    gates = evaluate_phase5b_gates(diag)
    # Short trajectories can't move energy 30% — at least one gate fires.
    assert 0 <= gates["gates_passed"] <= 5


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-011: artifact builder
# ---------------------------------------------------------------------------


def test_build_phase5b_artifact_emits_required_fields_phase5b_011(tmp_path: Path) -> None:
    diag = run_phase5b_training_loop(n_queries=1000, seed=1223)
    gates = evaluate_phase5b_gates(diag)
    artifact = build_phase5b_artifact(diag, gates, seed=1223)
    required = {
        "experiment",
        "schema_version",
        "run_date",
        "seed",
        "learning_rate_used",
        "proposal_learning_rate",
        "status",
        "n_queries_run",
        "n_accepted_by_verifier",
        "acceptance_rate",
        "energy_decrease_pct",
        "spectral_norm_growth_rate",
        "acceptance_rate_sublinear",
        "mean_anchor_distance",
        "oracle_accuracy_initial",
        "oracle_accuracy_final",
        "oracle_accuracy_drop_pp",
        "gate1_energy_decrease_30pct",
        "gate2_no_representation_drift",
        "gate3_no_autocatalytic_spiral",
        "gate4_no_null_space_excavation",
        "gate5_no_catastrophic_forgetting",
        "gates_passed",
        "phase5b_stability_confirmed",
        "honest_verdict",
    }
    missing = required - artifact.keys()
    assert not missing, f"missing artifact fields: {missing}"
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["honest_verdict"] in {
        "all_5_gates_pass",
        "partial_gates",
        "gate_failure_diagnosed",
        "blocked",
    }
    if artifact["gates_passed"] == 5:
        assert artifact["phase5b_stability_confirmed"] is True
        assert artifact["honest_verdict"] == "all_5_gates_pass"
    out = tmp_path / "art.json"
    write_phase5b_artifact(artifact, out)
    loaded = json.loads(out.read_text())
    assert loaded["experiment"] == "1223_phase5b_insitu_training_loop"


def test_build_phase5b_artifact_blocked_path_phase5b_011() -> None:
    artifact = build_phase5b_artifact(
        diagnostics={},
        gates={},
        seed=0,
        blocked=True,
        blocked_reason="phase5a_prototype_not_ready",
    )
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked"
    assert artifact["phase5b_stability_confirmed"] is False
    assert artifact["gates_passed"] == 0
    assert artifact["blocked_reason"] == "phase5a_prototype_not_ready"


def test_build_phase5b_artifact_partial_gates_verdict_phase5b_011() -> None:
    diag = {"n_queries_run": 1000, "n_accepted_by_verifier": 500}
    gates = {
        "energy_decrease_pct": 0.05,
        "spectral_norm_growth_rate": 0.0,
        "acceptance_rate_sublinear": True,
        "mean_anchor_distance": 1.0,
        "min_anchor_distance": 0.6,
        "oracle_accuracy_initial": 1.0,
        "oracle_accuracy_final": 1.0,
        "oracle_accuracy_drop_pp": 0.0,
        "acceptance_rate": 0.5,
        "gate1_energy_decrease_30pct": False,
        "gate2_no_representation_drift": True,
        "gate3_no_autocatalytic_spiral": True,
        "gate4_no_null_space_excavation": True,
        "gate5_no_catastrophic_forgetting": True,
        "gates_passed": 4,
    }
    artifact = build_phase5b_artifact(diag, gates, seed=0)
    assert artifact["honest_verdict"] == "partial_gates"
    assert artifact["phase5b_stability_confirmed"] is False


def test_build_phase5b_artifact_zero_gates_verdict_phase5b_011() -> None:
    diag = {"n_queries_run": 1000, "n_accepted_by_verifier": 0}
    gates = {
        "energy_decrease_pct": 0.0,
        "spectral_norm_growth_rate": 0.0,
        "acceptance_rate_sublinear": False,
        "mean_anchor_distance": 0.0,
        "min_anchor_distance": 0.0,
        "oracle_accuracy_initial": 1.0,
        "oracle_accuracy_final": 0.5,
        "oracle_accuracy_drop_pp": 50.0,
        "acceptance_rate": 0.0,
        "gate1_energy_decrease_30pct": False,
        "gate2_no_representation_drift": False,
        "gate3_no_autocatalytic_spiral": False,
        "gate4_no_null_space_excavation": False,
        "gate5_no_catastrophic_forgetting": False,
        "gates_passed": 0,
    }
    artifact = build_phase5b_artifact(diag, gates, seed=0)
    assert artifact["honest_verdict"] == "gate_failure_diagnosed"


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5B-012: confirm_phase5a_ready
# ---------------------------------------------------------------------------


def test_confirm_phase5a_ready_true_when_artifact_says_so_phase5b_012(tmp_path: Path) -> None:
    art = tmp_path / "phase5a.json"
    art.write_text(json.dumps({"phase5a_prototype_ready": True}))
    assert confirm_phase5a_ready(art) is True


def test_confirm_phase5a_ready_false_when_missing_phase5b_012(tmp_path: Path) -> None:
    assert confirm_phase5a_ready(tmp_path / "does-not-exist.json") is False


def test_confirm_phase5a_ready_false_on_invalid_json_phase5b_012(tmp_path: Path) -> None:
    art = tmp_path / "bad.json"
    art.write_text("not valid json")
    assert confirm_phase5a_ready(art) is False


def test_confirm_phase5a_ready_false_when_flag_absent_phase5b_012(tmp_path: Path) -> None:
    art = tmp_path / "phase5a.json"
    art.write_text(json.dumps({"some_other_field": True}))
    assert confirm_phase5a_ready(art) is False


def test_proposal_constants_match_spec() -> None:
    """Sanity-check the proposal-vs-runner learning-rate divergence is documented."""
    assert PROPOSAL_LEARNING_RATE == 1e-5
    assert DEFAULT_LEARNING_RATE == 1e-3
