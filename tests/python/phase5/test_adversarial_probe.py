"""Tests for Phase 5-C adversarial probe (exp_NEXT_C / exp 1224).

Spec coverage:
    REQ-KONA-018 (Phase 5-C adversarial probe: three attack classes against
    the Phase 5-A+B prototype before any Phase 5 scaling decision).

Scenario coverage:
    SCENARIO-PHASE5C-001 — run_attack1_single_verifier_gaming returns a float
        in [0, 1] when given valid encoder/refiner components.
    SCENARIO-PHASE5C-002 — run_attack2_pairwise_correlation returns a
        (float, N×N ndarray) pair; the float is in [0, 1].
    SCENARIO-PHASE5C-003 — run_attack3_joint_nullspace returns a float in [0, 1]
        for standard parametrisation.
    SCENARIO-PHASE5C-004 — _make_invalid_z_for_grid produces action sequences
        where verify_action_sequence returns False on the supplied grid.
    SCENARIO-PHASE5C-005 — _numerical_gradient approximates the exact gradient
        of a simple known function to within numerical tolerance.
    SCENARIO-PHASE5C-006 — evaluate_defense_verdict returns
        all_attacks_blocked=True and verdict="all_attacks_blocked_architecture_validated"
        when all metrics are below thresholds.
    SCENARIO-PHASE5C-007 — evaluate_defense_verdict returns
        all_attacks_blocked=False and verdict="partial_attack_success_revision_needed"
        when attack 2 (pairwise correlation) succeeds.
    SCENARIO-PHASE5C-008 — evaluate_defense_verdict populates
        failure_modes_discovered and architectural_revision_if_needed when
        attack 1 and/or attack 3 also succeed.
    SCENARIO-PHASE5C-009 — build_phase5c_artifact emits all required schema
        fields from REQ-KONA-018 with correct types.
    SCENARIO-PHASE5C-010 — write_phase5c_artifact writes valid JSON to a
        temporary path and all required fields survive round-trip.
    SCENARIO-PHASE5C-011 — run_attack3_joint_nullspace returns 0.0 immediately
        when n_starts=0 (guard for degenerate callers).
    SCENARIO-PHASE5C-012 — _build_revision_text produces non-empty strings
        for all single-attack and multi-attack failure combinations.
"""

from __future__ import annotations

import datetime as _dt
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from carnot.phase5.adversarial_probe import (
    ATTACK1_GAMING_THRESHOLD,
    ATTACK2_CORRELATION_THRESHOLD,
    ATTACK3_GAMING_THRESHOLD,
    _build_revision_text,
    _make_invalid_z_for_grid,
    _numerical_gradient,
    build_phase5c_artifact,
    evaluate_defense_verdict,
    run_attack1_single_verifier_gaming,
    run_attack2_pairwise_correlation,
    run_attack3_joint_nullspace,
    write_phase5c_artifact,
)
from carnot.phase5.insitu_prototype import (
    LATENT_DIM,
    N_VERIFIERS,
    InSituEnergyMLP,
    InSituEncoder,
    InSituRefiner,
    generate_random_5x5_puzzle,
    snap_to_action,
    verify_action_sequence,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def encoder() -> InSituEncoder:
    return InSituEncoder.init(seed=0)


@pytest.fixture(scope="module")
def refiner() -> InSituRefiner:
    return InSituRefiner.init(seed=2)


@pytest.fixture(scope="module")
def energy_mlp() -> InSituEnergyMLP:
    return InSituEnergyMLP.init(seed=1)


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-001 — Attack 1 return type and range
# ---------------------------------------------------------------------------


def test_attack1_returns_float_in_unit_interval(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
) -> None:
    """REQ-KONA-018: Attack 1 rate must be in [0, 1]."""
    # Small n_samples to keep the test fast.
    rate = run_attack1_single_verifier_gaming(encoder, refiner, n_samples=30, seed=1)
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-002 — Attack 2 return shape
# ---------------------------------------------------------------------------


def test_attack2_returns_float_and_matrix(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
) -> None:
    """REQ-KONA-018: Attack 2 must return (float, N×N ndarray)."""
    max_corr, matrix = run_attack2_pairwise_correlation(
        encoder, refiner, n_samples=30, seed=2
    )
    assert isinstance(max_corr, float)
    assert 0.0 <= max_corr <= 1.0
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (N_VERIFIERS, N_VERIFIERS)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-003 — Attack 3 return type and range
# ---------------------------------------------------------------------------


def test_attack3_returns_float_in_unit_interval(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
    energy_mlp: InSituEnergyMLP,
) -> None:
    """REQ-KONA-018: Attack 3 rate must be in [0, 1]."""
    rate = run_attack3_joint_nullspace(
        encoder, refiner, energy_mlp,
        n_starts=3, n_steps=5, seed=3,
    )
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-004 — _make_invalid_z_for_grid produces invalid sequences
# ---------------------------------------------------------------------------


def test_make_invalid_z_always_fails_verifier(rng: np.random.Generator) -> None:
    """REQ-KONA-018: constructed z must produce verify_action_sequence=False."""
    # Test across 10 different random grids to verify the construction is robust.
    for seed_offset in range(10):
        local_rng = np.random.default_rng(100 + seed_offset)
        grid = generate_random_5x5_puzzle(local_rng)
        z_inv = _make_invalid_z_for_grid(grid, local_rng)
        actions = snap_to_action(z_inv)
        assert not verify_action_sequence(actions, grid), (
            f"Expected invalid sequence for seed_offset={seed_offset}, "
            f"grid=\n{grid}"
        )


def test_make_invalid_z_returns_correct_dim(rng: np.random.Generator) -> None:
    """REQ-KONA-018: _make_invalid_z_for_grid must return a LATENT_DIM vector."""
    local_rng = np.random.default_rng(200)
    grid = generate_random_5x5_puzzle(local_rng)
    z_inv = _make_invalid_z_for_grid(grid, local_rng)
    assert z_inv.shape == (LATENT_DIM,)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-005 — _numerical_gradient approximates exact gradient
# ---------------------------------------------------------------------------


def test_numerical_gradient_on_quadratic() -> None:
    """REQ-KONA-018: finite-difference gradient must match analytic gradient
    of f(z) = sum(z**2) to within 1e-3 tolerance."""
    # f(z) = sum(z^2), grad = 2*z
    z = np.array([0.5, -0.3, 0.8, 0.0, -0.9] + [0.1] * (LATENT_DIM - 5))

    def f_quad(z_in: np.ndarray) -> float:
        return float(np.sum(z_in**2))

    grad_numerical = _numerical_gradient(f_quad, z)
    grad_exact = 2.0 * z
    np.testing.assert_allclose(grad_numerical, grad_exact, atol=1e-3)


def test_numerical_gradient_length_matches_input() -> None:
    """REQ-KONA-018: gradient vector length must equal latent_dim."""
    z = np.zeros(LATENT_DIM)
    grad = _numerical_gradient(lambda z_in: float(np.sum(z_in)), z)
    assert grad.shape == (LATENT_DIM,)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-006 — All-blocked verdict
# ---------------------------------------------------------------------------


def test_evaluate_defense_verdict_all_blocked() -> None:
    """REQ-KONA-018: when all metrics below thresholds, verdict is
    'all_attacks_blocked_architecture_validated'."""
    matrix = np.eye(N_VERIFIERS)
    result = evaluate_defense_verdict(
        gaming_rate_attack1=ATTACK1_GAMING_THRESHOLD - 0.01,
        pairwise_max_correlation=ATTACK2_CORRELATION_THRESHOLD - 0.01,
        joint_gaming_rate=ATTACK3_GAMING_THRESHOLD - 0.01,
        conditional_matrix=matrix,
    )
    assert result["attack1_blocked"] is True
    assert result["attack2_blocked"] is True
    assert result["attack3_blocked"] is True
    assert result["all_attacks_blocked"] is True
    assert result["failure_modes_discovered"] == []
    assert result["architectural_revision_if_needed"] == "none"
    assert result["honest_verdict"] == "all_attacks_blocked_architecture_validated"


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-007 — Partial failure (attack 2 succeeds)
# ---------------------------------------------------------------------------


def test_evaluate_defense_verdict_attack2_fails() -> None:
    """REQ-KONA-018: when attack 2 exceeds threshold, verdict is
    'partial_attack_success_revision_needed' and failure modes are populated."""
    matrix = np.ones((N_VERIFIERS, N_VERIFIERS)) * 0.8
    result = evaluate_defense_verdict(
        gaming_rate_attack1=0.01,
        pairwise_max_correlation=0.85,  # above 0.70 threshold
        joint_gaming_rate=0.01,
        conditional_matrix=matrix,
    )
    assert result["attack2_blocked"] is False
    assert result["all_attacks_blocked"] is False
    assert result["honest_verdict"] == "partial_attack_success_revision_needed"
    assert len(result["failure_modes_discovered"]) == 1
    assert "pairwise_verifier_correlation" in result["failure_modes_discovered"][0]
    assert result["architectural_revision_if_needed"] != "none"
    # Should mention Spera Theorem since attack 2 failed.
    assert "Spera" in result["architectural_revision_if_needed"]


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-008 — Multi-attack failure populates all failure modes
# ---------------------------------------------------------------------------


def test_evaluate_defense_verdict_all_fail() -> None:
    """REQ-KONA-018: when all three attacks succeed, all failure modes listed."""
    matrix = np.ones((N_VERIFIERS, N_VERIFIERS))
    result = evaluate_defense_verdict(
        gaming_rate_attack1=0.20,  # above 0.10
        pairwise_max_correlation=0.90,  # above 0.70
        joint_gaming_rate=0.10,  # above 0.05
        conditional_matrix=matrix,
    )
    assert result["all_attacks_blocked"] is False
    assert len(result["failure_modes_discovered"]) == 3
    assert result["honest_verdict"] == "partial_attack_success_revision_needed"


def test_evaluate_defense_verdict_attack1_only_fails() -> None:
    """REQ-KONA-018: only attack 1 failure populates exactly one failure mode."""
    matrix = np.eye(N_VERIFIERS)
    result = evaluate_defense_verdict(
        gaming_rate_attack1=0.20,
        pairwise_max_correlation=0.50,
        joint_gaming_rate=0.01,
        conditional_matrix=matrix,
    )
    assert result["attack1_blocked"] is False
    assert result["attack2_blocked"] is True
    assert result["attack3_blocked"] is True
    assert len(result["failure_modes_discovered"]) == 1
    assert "single_verifier_gaming" in result["failure_modes_discovered"][0]


def test_evaluate_defense_verdict_attack3_only_fails() -> None:
    """REQ-KONA-018: only attack 3 failure populates exactly one failure mode."""
    matrix = np.eye(N_VERIFIERS)
    result = evaluate_defense_verdict(
        gaming_rate_attack1=0.01,
        pairwise_max_correlation=0.50,
        joint_gaming_rate=0.10,
        conditional_matrix=matrix,
    )
    assert result["attack3_blocked"] is False
    assert result["all_attacks_blocked"] is False
    assert len(result["failure_modes_discovered"]) == 1
    assert "joint_nullspace" in result["failure_modes_discovered"][0]


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-009 — build_phase5c_artifact emits required fields
# ---------------------------------------------------------------------------


_REQUIRED_FIELDS = [
    "experiment",
    "schema_version",
    "run_date",
    "seed",
    "status",
    "gaming_rate_attack1",
    "pairwise_max_correlation",
    "joint_gaming_rate",
    "attack1_blocked",
    "attack2_blocked",
    "attack3_blocked",
    "all_attacks_blocked",
    "failure_modes_discovered",
    "architectural_revision_if_needed",
    "adversarial_probe_complete",
    "honest_verdict",
    "conditional_acceptance_matrix",
    "phase5b_stability_was_confirmed",
]


def test_build_phase5c_artifact_has_required_fields() -> None:
    """REQ-KONA-018: artifact must contain all required schema fields."""
    start = _dt.datetime.now(_dt.timezone.utc)
    matrix = np.eye(N_VERIFIERS)
    verdict = evaluate_defense_verdict(0.01, 0.5, 0.01, matrix)
    artifact = build_phase5c_artifact(
        start_time=start,
        seed=1224,
        gaming_rate_attack1=0.01,
        pairwise_max_correlation=0.5,
        joint_gaming_rate=0.01,
        conditional_matrix=matrix,
        verdict_dict=verdict,
        phase5b_stability_confirmed=True,
    )
    for field in _REQUIRED_FIELDS:
        assert field in artifact, f"Missing required field: {field}"

    assert artifact["experiment"] == "1224_phase5c_adversarial_probe"
    assert artifact["adversarial_probe_complete"] is True
    assert artifact["phase5b_stability_was_confirmed"] is True
    assert isinstance(artifact["failure_modes_discovered"], list)


def test_build_phase5c_artifact_rates_are_floats() -> None:
    """REQ-KONA-018: rate fields must be Python floats."""
    start = _dt.datetime.now(_dt.timezone.utc)
    matrix = np.eye(N_VERIFIERS)
    verdict = evaluate_defense_verdict(0.05, 0.6, 0.02, matrix)
    artifact = build_phase5c_artifact(
        start_time=start,
        seed=1224,
        gaming_rate_attack1=0.05,
        pairwise_max_correlation=0.6,
        joint_gaming_rate=0.02,
        conditional_matrix=matrix,
        verdict_dict=verdict,
        phase5b_stability_confirmed=True,
    )
    assert isinstance(artifact["gaming_rate_attack1"], float)
    assert isinstance(artifact["pairwise_max_correlation"], float)
    assert isinstance(artifact["joint_gaming_rate"], float)
    assert isinstance(artifact["duration_s"], float)


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-010 — write_phase5c_artifact round-trip
# ---------------------------------------------------------------------------


def test_write_phase5c_artifact_round_trip() -> None:
    """REQ-KONA-018: write_phase5c_artifact must produce valid JSON that
    survives a load-and-compare round-trip."""
    start = _dt.datetime.now(_dt.timezone.utc)
    matrix = np.eye(N_VERIFIERS)
    verdict = evaluate_defense_verdict(0.01, 0.5, 0.01, matrix)
    artifact = build_phase5c_artifact(
        start_time=start,
        seed=1224,
        gaming_rate_attack1=0.01,
        pairwise_max_correlation=0.5,
        joint_gaming_rate=0.01,
        conditional_matrix=matrix,
        verdict_dict=verdict,
        phase5b_stability_confirmed=True,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_artifact.json"
        write_phase5c_artifact(artifact, path)
        loaded = json.loads(path.read_text())

    for field in _REQUIRED_FIELDS:
        assert field in loaded, f"Field lost in round-trip: {field}"
    assert loaded["experiment"] == "1224_phase5c_adversarial_probe"
    assert loaded["adversarial_probe_complete"] is True


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-011 — attack 3 n_starts=0 guard
# ---------------------------------------------------------------------------


def test_attack3_n_starts_zero_returns_zero(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
    energy_mlp: InSituEnergyMLP,
) -> None:
    """REQ-KONA-018: n_starts=0 must return 0.0 immediately without error."""
    rate = run_attack3_joint_nullspace(
        encoder, refiner, energy_mlp, n_starts=0, n_steps=5, seed=42
    )
    assert rate == 0.0


# ---------------------------------------------------------------------------
# SCENARIO-PHASE5C-012 — _build_revision_text coverage
# ---------------------------------------------------------------------------


def test_build_revision_text_all_blocked_returns_none() -> None:
    """REQ-KONA-018: no failures → revision text is 'none'."""
    text = _build_revision_text(True, True, True)
    assert text == "none"


def test_build_revision_text_attack2_only() -> None:
    """REQ-KONA-018: only attack 2 failing → revision mentions Spera Theorem."""
    text = _build_revision_text(True, False, True)
    assert "Spera" in text
    assert text != "none"


def test_build_revision_text_attack1_only() -> None:
    """REQ-KONA-018: only attack 1 failing → revision mentions stricter verifier."""
    text = _build_revision_text(False, True, True)
    assert "stricter" in text
    assert text != "none"


def test_build_revision_text_attack3_only() -> None:
    """REQ-KONA-018: only attack 3 failing → revision mentions energy MLP retraining."""
    text = _build_revision_text(True, True, False)
    assert "energy MLP" in text
    assert text != "none"


def test_build_revision_text_all_fail() -> None:
    """REQ-KONA-018: all attacks failing → revision covers all three topics."""
    text = _build_revision_text(False, False, False)
    assert "Spera" in text
    assert "stricter" in text
    assert "energy MLP" in text
    # Multi-part revision uses | separator.
    assert "|" in text


# ---------------------------------------------------------------------------
# Branch coverage — attack 1 gaming_count increment
# ---------------------------------------------------------------------------


def test_attack1_gaming_detection_branch(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
) -> None:
    """REQ-KONA-018: gaming_count increment fires when V0 passes but V1,V2 fail.

    Patches verifier_outcomes to return (True, False, False) so composite=1/3
    is below the 0.5 threshold on every sample — all n samples count as gaming.
    """
    from unittest.mock import patch

    with patch(
        "carnot.phase5.adversarial_probe.verifier_outcomes",
        return_value=(True, False, False),
    ):
        rate = run_attack1_single_verifier_gaming(
            encoder, refiner, n_samples=5, seed=42
        )
    assert rate == 1.0, f"Expected all 5 samples to be gaming; got rate={rate}"


# ---------------------------------------------------------------------------
# Branch coverage — attack 3 Phase 2 direct-construction fallback
# ---------------------------------------------------------------------------


def test_attack3_phase2_fallback_triggered(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
    energy_mlp: InSituEnergyMLP,
) -> None:
    """REQ-KONA-018: setting _max_rejection_attempts=0 forces Phase 2 direct
    construction to run — exercises lines 275-278."""
    # With _max_rejection_attempts=0, Phase 1 does 0 attempts and Phase 2
    # must build all invalid_starts via _make_invalid_z_for_grid.
    rate = run_attack3_joint_nullspace(
        encoder, refiner, energy_mlp,
        n_starts=2, n_steps=2, seed=99,
        _max_rejection_attempts=0,
    )
    assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Branch coverage — attack 3 gaming_count increment (line 298)
# ---------------------------------------------------------------------------


def test_attack3_gaming_detection_branch(
    encoder: InSituEncoder,
    refiner: InSituRefiner,
    energy_mlp: InSituEnergyMLP,
) -> None:
    """REQ-KONA-018: gaming_count fires when energy > 0.8 and verifiers reject.

    Patches verify_action_sequence to always return False (all sequences
    "invalid") and energy_mlp.forward to always return 0.9 (proxy > 0.8).
    With both patches active: rejection sampling instantly finds n_starts
    invalids (all sequences are "invalid") and after gradient ascent the
    gaming condition fires on every start.
    """
    from unittest.mock import patch

    with patch(
        "carnot.phase5.adversarial_probe.verify_action_sequence",
        return_value=False,
    ):
        with patch.object(energy_mlp, "forward", return_value=0.9):
            rate = run_attack3_joint_nullspace(
                encoder, refiner, energy_mlp,
                n_starts=2, n_steps=2, seed=77,
            )
    assert rate == 1.0, f"Expected all starts to be gaming with mocked proxy; got {rate}"
