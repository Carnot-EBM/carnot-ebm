"""Tests for Exp 1578 BRAIN REINFORCE training dynamics.

Spec refs: REQ-VERIFY-1578, SCENARIO-VERIFY-1578.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.training import brain_reinforce_training_dynamics as exp1578


def test_spec_mentions_exp1578_contract() -> None:
    """REQ-VERIFY-1578, SCENARIO-VERIFY-1578: Exp 1578 is spec-anchored."""

    spec = (exp1578.PROJECT_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-1578" in spec
    assert "SCENARIO-VERIFY-1578" in spec
    assert "experiment_1578_brain_reinforce_training_dynamics_at_k15.json" in spec
    assert "factorized gradient starvation real" in spec
    assert "starvation overstated" in spec
    assert "both parameterizations inadequate" in spec


def test_default_config_records_requested_k15_audit_regime() -> None:
    """REQ-VERIFY-1578: defaults match the requested BRAIN k=15 audit."""

    config = exp1578.TrainingDynamicsConfig()

    assert config.n == 16
    assert config.k == 15
    assert config.beta == pytest.approx(2.0)
    assert config.constraints_per_k == 10
    assert config.batch_size == 512
    assert config.max_iterations == 50_000
    assert config.factorized_parameter_count == 16
    assert config.linear_ar_parameter_count == 136


def test_problem_builder_matches_exp1562_uniform_k15_target() -> None:
    """REQ-VERIFY-1578: the finite-state target is the Exp 1562 k=15 regime."""

    config = exp1578.TrainingDynamicsConfig()
    problem = exp1578.build_problem(config)

    assert problem.states.shape == (65_536, 16)
    assert len(problem.constraints) == 10
    assert all(len(constraint.indices) == 15 for constraint in problem.constraints)
    assert problem.initial_factorized_kl() == pytest.approx(0.00133753526, abs=1e-10)
    assert problem.initial_linear_ar_kl() == pytest.approx(0.00133753526, abs=1e-10)


def test_training_trace_records_checkpoint_metrics() -> None:
    """SCENARIO-VERIFY-1578: REINFORCE traces expose KL, gradients, and escape."""

    config = exp1578.TrainingDynamicsConfig(
        n=6,
        k=4,
        constraints_per_k=3,
        batch_size=64,
        max_iterations=20,
        min_iterations=20,
        checkpoint_interval=10,
        convergence_kl_threshold=1.0,
    )
    problem = exp1578.build_problem(config)

    factorized = exp1578.train_factorized(problem, config)
    linear_ar = exp1578.train_linear_ar(problem, config)

    assert [point.iteration for point in factorized.checkpoints] == [0, 10, 20]
    assert [point.iteration for point in linear_ar.checkpoints] == [0, 10, 20]
    assert 0.0 <= factorized.gradient_active_fraction_first_1000 <= 1.0
    assert 0.0 <= linear_ar.gradient_active_fraction_first_1000 <= 1.0
    assert factorized.final_kl == factorized.checkpoints[-1].kl
    assert linear_ar.final_kl == linear_ar.checkpoints[-1].kl
    assert factorized.wall_time_s >= 0.0
    assert linear_ar.wall_time_s >= 0.0

    late_config = exp1578.TrainingDynamicsConfig(
        n=4,
        k=2,
        constraints_per_k=1,
        batch_size=128,
        max_iterations=10,
        min_iterations=10,
        checkpoint_interval=10,
        convergence_kl_threshold=0.4,
        factorized_learning_rate=0.1,
        linear_ar_learning_rate=0.1,
    )
    late_problem = exp1578.build_problem(late_config)

    assert exp1578.train_factorized(late_problem, late_config).convergence_iteration == 10
    assert exp1578.train_linear_ar(late_problem, late_config).convergence_iteration == 10


def test_verdict_classifier_distinguishes_all_allowed_outcomes() -> None:
    """REQ-VERIFY-1578: verdicts route to the three registered conclusions."""

    config = exp1578.TrainingDynamicsConfig(convergence_kl_threshold=0.01)
    factorized_starved = exp1578.TrainingTrace(
        parameterization="factorized",
        checkpoints=(),
        gradient_active_fraction_first_1000=0.05,
        convergence_iteration=None,
        wall_time_s=0.1,
        iterations_run=1000,
    )
    linear_good = exp1578.TrainingTrace(
        parameterization="linear_ar",
        checkpoints=(exp1578.CheckpointMetric(0, 0.004, 0.0, 0.0),),
        gradient_active_fraction_first_1000=0.9,
        convergence_iteration=1000,
        wall_time_s=0.1,
        iterations_run=1000,
    )
    both_bad = exp1578.TrainingTrace(
        parameterization="factorized",
        checkpoints=(exp1578.CheckpointMetric(0, 0.2, 0.0, 0.0),),
        gradient_active_fraction_first_1000=0.05,
        convergence_iteration=None,
        wall_time_s=0.1,
        iterations_run=1000,
    )
    factorized_ok = exp1578.TrainingTrace(
        parameterization="factorized",
        checkpoints=(exp1578.CheckpointMetric(0, 0.004, 0.0, 0.0),),
        gradient_active_fraction_first_1000=0.5,
        convergence_iteration=1000,
        wall_time_s=0.1,
        iterations_run=1000,
    )

    assert (
        exp1578.classify_training_dynamics(
            config=config,
            factorized=factorized_starved,
            linear_ar=linear_good,
        )
        == "factorized gradient starvation real"
    )
    assert (
        exp1578.classify_training_dynamics(
            config=config,
            factorized=both_bad,
            linear_ar=both_bad,
        )
        == "both parameterizations inadequate"
    )
    assert (
        exp1578.classify_training_dynamics(
            config=config,
            factorized=factorized_ok,
            linear_ar=linear_good,
        )
        == "starvation overstated"
    )


def test_run_experiment_writes_artifact_and_research_note(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1578: runner writes JSON and paper-v6 recommendation."""

    output = tmp_path / "experiment_1578.json"
    note = tmp_path / "brain-reinforce-training-dynamics-k15.md"
    artifact = exp1578.run_experiment(
        output_path=output,
        research_note_path=note,
        config=exp1578.TrainingDynamicsConfig(
            n=6,
            k=4,
            constraints_per_k=3,
            batch_size=64,
            max_iterations=20,
            min_iterations=20,
            checkpoint_interval=10,
            convergence_kl_threshold=1.0,
        ),
    )

    assert exp1578.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["status"] == "complete"
    assert artifact["brain_training_dynamics_verdict_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    note_text = note.read_text(encoding="utf-8")
    assert artifact["paper_v6_brain_recommendation"] in note_text
    assert artifact["honest_verdict"] in note_text


def test_validate_artifact_rejects_schema_and_verdict_errors() -> None:
    """REQ-VERIFY-1578: terminal artifact fields and verdicts are checked."""

    artifact = {
        "status": "complete",
        "factorized_gradient_active_fraction_first_1000": 0.5,
        "linear_ar_gradient_active_fraction_first_1000": 0.5,
        "factorized_final_kl": 0.001,
        "linear_ar_final_kl": 0.001,
        "factorized_converged": True,
        "linear_ar_converged": True,
        "brain_training_dynamics_verdict_ready": True,
        "paper_v6_brain_recommendation": "paper_v6: treat BRAIN starvation as overstated",
        "honest_verdict": "complete: starvation overstated",
    }

    assert exp1578.validate_artifact(artifact) is None

    missing = dict(artifact)
    missing.pop("linear_ar_final_kl")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1578.validate_artifact(missing)

    bad_status = dict(artifact, status="in_progress")
    with pytest.raises(ValueError, match="status must be complete"):
        exp1578.validate_artifact(bad_status)

    bad_verdict = dict(artifact, honest_verdict="starvation overstated")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1578.validate_artifact(bad_verdict)

    bad_ready = dict(artifact, brain_training_dynamics_verdict_ready=False)
    with pytest.raises(ValueError, match="verdict_ready"):
        exp1578.validate_artifact(bad_ready)

    bad_recommendation = dict(artifact, honest_verdict="complete: impossible verdict")
    with pytest.raises(ValueError, match="allowed verdict"):
        exp1578.validate_artifact(bad_recommendation)
