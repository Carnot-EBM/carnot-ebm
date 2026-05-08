"""Tests for Exp 1562 BRAIN Linear-AR k-sweep verification.

Spec refs: REQ-VERIFY-1562, SCENARIO-VERIFY-1562.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.scripts import dt_brain_correlations_verification as exp1562


def _kl(factorized: float, linear_ar: float, made_optional: float | None = None) -> dict[str, float | None]:
    return {
        "factorized": factorized,
        "linear_ar": linear_ar,
        "made_optional": made_optional,
    }


def test_spec_mentions_exp1562_contract() -> None:
    """REQ-VERIFY-1562, SCENARIO-VERIFY-1562: Exp 1562 is spec-anchored."""

    spec = (exp1562.PROJECT_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-1562" in spec
    assert "SCENARIO-VERIFY-1562" in spec
    assert "experiment_1562_brain_linear_ar_k_sweep_extended.json" in spec
    assert "brain_dropped" in spec


def test_config_records_requested_parameter_counts() -> None:
    """REQ-VERIFY-1562: parameter counts match factorized and Linear-AR contracts."""

    config = exp1562.SweepConfig()

    assert config.n == 16
    assert config.k_values == (4, 8, 12, 15)
    assert config.factorized_parameter_count == 16
    assert config.linear_ar_parameter_count == 136


def test_k4_optimizer_preserves_original_baseline() -> None:
    """SCENARIO-VERIFY-1562: deterministic seed preserves the partial k=4 run."""

    config = exp1562.SweepConfig(maxiter=250)
    problem = exp1562.build_problem(config, k=4)

    factorized = exp1562.optimize_factorized(problem, maxiter=config.maxiter)
    linear_ar = exp1562.optimize_linear_ar(problem, maxiter=config.maxiter)

    assert factorized.kl == pytest.approx(1.074764, abs=1e-4)
    assert linear_ar.kl == pytest.approx(0.334573, abs=1e-4)
    assert factorized.kl / linear_ar.kl == pytest.approx(3.21, abs=0.02)


def test_run_k_sweep_invokes_made_only_when_linear_ar_fails_gate() -> None:
    """REQ-VERIFY-1562: MADE is optional and triggered by the k=15 AR KL gate."""

    calls: list[tuple[int, bool]] = []
    first_pass = {
        4: _kl(1.0, 0.4),
        8: _kl(0.8, 0.3),
        12: _kl(0.6, 0.2),
        15: _kl(2.0, 0.2),
    }
    made_pass = {
        4: _kl(1.0, 0.4, 0.3),
        8: _kl(0.8, 0.3, 0.2),
        12: _kl(0.6, 0.2, 0.1),
        15: _kl(2.0, 0.2, 0.05),
    }

    def fake_optimizer(config: exp1562.SweepConfig, k: int, *, include_made: bool) -> dict[str, float | None]:
        calls.append((k, include_made))
        return dict(made_pass[k] if include_made else first_pass[k])

    sweep = exp1562.run_k_sweep(exp1562.SweepConfig(), optimizer=fake_optimizer)

    assert calls == [
        (4, False),
        (8, False),
        (12, False),
        (15, False),
        (4, True),
        (8, True),
        (12, True),
        (15, True),
    ]
    assert sweep[15]["made_optional"] == 0.05


def test_compute_kl_for_k_attaches_made_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-1562: per-k optimizer includes MADE only on request."""

    def fake_factorized(
        problem: exp1562.BrainCorrelationProblem,
        *,
        maxiter: int,
    ) -> exp1562.OptimizationResult:
        assert problem.k == 1
        assert maxiter == 7
        return exp1562.OptimizationResult("factorized", 0.4, True, 1, "ok")

    def fake_linear_ar(
        problem: exp1562.BrainCorrelationProblem,
        *,
        maxiter: int,
    ) -> exp1562.OptimizationResult:
        assert problem.config.n == 3
        assert maxiter == 7
        return exp1562.OptimizationResult("linear_ar", 0.2, True, 1, "ok")

    def fake_made(
        problem: exp1562.BrainCorrelationProblem,
        *,
        hidden_units: int,
        steps: int,
        learning_rate: float,
    ) -> exp1562.OptimizationResult:
        assert problem.states.shape == (8, 3)
        assert (hidden_units, steps, learning_rate) == (32, 11, 0.05)
        return exp1562.OptimizationResult("made_optional", 0.03, True, 1, "ok")

    monkeypatch.setattr(exp1562, "optimize_factorized", fake_factorized)
    monkeypatch.setattr(exp1562, "optimize_linear_ar", fake_linear_ar)
    monkeypatch.setattr(exp1562, "optimize_made", fake_made)

    row = exp1562.compute_kl_for_k(
        exp1562.SweepConfig(n=3, k_values=(1,), maxiter=7, made_steps=11, made_learning_rate=0.05),
        1,
        include_made=True,
    )

    assert row == {"factorized": 0.4, "linear_ar": 0.2, "made_optional": 0.03}


def test_build_artifact_maps_falsification_to_brain_dropped() -> None:
    """SCENARIO-VERIFY-1562: ratio below 5x honestly drops BRAIN."""

    config = exp1562.SweepConfig()
    kl_by_k = {
        4: _kl(1.074764, 0.334573),
        8: _kl(0.146665, 0.137829),
        12: _kl(0.010571, 0.010494),
        15: _kl(0.001337, 0.001336),
    }

    artifact = exp1562.build_artifact(config=config, kl_by_k=kl_by_k)

    assert artifact["status"] == "complete"
    assert artifact["brain_linear_ar_rescue_validated"] is False
    assert artifact["phase_3_recommendation"] == "brain_dropped"
    assert artifact["made_required_at_k15"] is False
    assert artifact["factorized_vs_ar_ratio_at_k15"] == pytest.approx(1.001, abs=0.01)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["kl_by_k_by_parameterization"]["15"]["made_optional"] is None


def test_build_artifact_maps_success_partial_and_made_required_cases() -> None:
    """REQ-VERIFY-1562: recommendations distinguish Linear-AR and MADE success."""

    config = exp1562.SweepConfig()
    linear_ar = exp1562.build_artifact(
        config=config,
        kl_by_k={
            4: _kl(1.0, 0.3),
            8: _kl(1.0, 0.2),
            12: _kl(1.0, 0.1),
            15: _kl(1.1, 0.05),
        },
    )
    made = exp1562.build_artifact(
        config=config,
        kl_by_k={
            4: _kl(1.0, 0.3, 0.2),
            8: _kl(1.0, 0.25, 0.2),
            12: _kl(1.0, 0.2, 0.15),
            15: _kl(2.0, 0.2, 0.05),
        },
    )
    partial = exp1562.build_artifact(
        config=config,
        kl_by_k={
            4: _kl(1.0, 0.3),
            8: _kl(1.0, 0.2),
            12: _kl(1.0, 0.1),
            15: _kl(0.6, 0.1),
        },
    )

    assert linear_ar["brain_linear_ar_rescue_validated"] is True
    assert linear_ar["phase_3_recommendation"] == "linear_ar_sufficient"
    assert partial["brain_linear_ar_rescue_validated"] is False
    assert partial["phase_3_recommendation"] == "brain_dropped"
    assert made["brain_linear_ar_rescue_validated"] is False
    assert made["made_required_at_k15"] is True
    assert made["phase_3_recommendation"] == "made_required"


def test_run_experiment_writes_complete_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1562: runner writes the terminal JSON schema."""

    output = tmp_path / "experiment_1562.json"
    kl_by_k = {
        4: _kl(1.074764, 0.334573),
        8: _kl(0.146665, 0.137829),
        12: _kl(0.010571, 0.010494),
        15: _kl(0.001337, 0.001336),
    }

    artifact = exp1562.run_experiment(
        output_path=output,
        config=exp1562.SweepConfig(),
        sweep_runner=lambda _config: kl_by_k,
    )

    assert exp1562.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_validate_artifact_rejects_missing_fields_and_bad_values() -> None:
    """REQ-VERIFY-1562: terminal artifacts are schema-checked."""

    config = exp1562.SweepConfig()
    valid = exp1562.build_artifact(
        config=config,
        kl_by_k={
            4: _kl(1.0, 0.3),
            8: _kl(1.0, 0.2),
            12: _kl(1.0, 0.1),
            15: _kl(1.1, 0.05),
        },
    )

    assert exp1562.validate_artifact(valid) is None

    missing = dict(valid)
    missing.pop("factorized_vs_ar_ratio_at_k15")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1562.validate_artifact(missing)

    bad_status = dict(valid)
    bad_status["status"] = "in_progress"
    with pytest.raises(ValueError, match="status must be complete"):
        exp1562.validate_artifact(bad_status)

    bad_verdict = dict(valid)
    bad_verdict["honest_verdict"] = "partial"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1562.validate_artifact(bad_verdict)

    bad_recommendation = dict(valid)
    bad_recommendation["phase_3_recommendation"] = "maybe"
    with pytest.raises(ValueError, match="invalid phase_3_recommendation"):
        exp1562.validate_artifact(bad_recommendation)

    with pytest.raises(ValueError, match="missing k values"):
        exp1562.build_artifact(
            config=config,
            kl_by_k={
                4: _kl(1.0, 0.3),
                8: _kl(1.0, 0.2),
                12: _kl(1.0, 0.1),
            },
        )

    with pytest.raises(ValueError, match="missing KL value"):
        exp1562.build_artifact(
            config=config,
            kl_by_k={
                4: {"linear_ar": 0.3, "made_optional": None},
                8: _kl(1.0, 0.2),
                12: _kl(1.0, 0.1),
                15: _kl(1.1, 0.05),
            },
        )
