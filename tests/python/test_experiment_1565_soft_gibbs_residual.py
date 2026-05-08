"""Tests for Exp 1565 Soft-Gibbs residual BRS.

Spec refs: REQ-SAMPLE-059, SCENARIO-SAMPLE-087.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from carnot.sampling import brs_residual as brs


class CyclingSampler:
    """Small deterministic prior sampler for tests."""

    def __init__(self, states: tuple[tuple[int, ...], ...]) -> None:
        self.states = states
        self.index = 0

    def __call__(self) -> tuple[int, ...]:
        state = self.states[self.index % len(self.states)]
        self.index += 1
        return state


class SequenceRng:
    """Minimal RNG stub exposing the random() method soft_brs needs."""

    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values
        self.index = 0

    def random(self) -> float:
        value = self.values[self.index]
        self.index += 1
        return value


def test_spec_mentions_exp1565_contract() -> None:
    """REQ-SAMPLE-059, SCENARIO-SAMPLE-087: Exp 1565 is spec-anchored."""

    spec = (brs.PROJECT_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-059" in spec
    assert "SCENARIO-SAMPLE-087" in spec
    assert "experiment_1565_soft_gibbs_residual_implementation.json" in spec
    assert "Soft-Gibbs residual" in spec


def test_req_sample_059_hard_brs_rejects_empty_intersection() -> None:
    """REQ-SAMPLE-059: Hard-BRS accepts nothing when verifier intersection is empty."""

    verifiers = brs.contradictory_verifiers()
    states = brs.enumerate_spin_states(n=8)
    sampler = CyclingSampler(states)

    result = brs.hard_brs(
        sampler,
        lambda y: all(verifier(y) for verifier in verifiers),
        n_steps=512,
    )

    assert result.total_steps == 512
    assert result.accepted_count == 0
    assert result.acceptance_rate == 0.0
    assert result.accepted_samples == ()


def test_req_sample_059_hard_brs_accepts_callable_and_container_sets() -> None:
    """REQ-SAMPLE-059: Hard-BRS supports concrete sets as well as predicates."""

    states = ((1, 1), (-1, 1), (1, -1))
    predicate_result = brs.hard_brs(CyclingSampler(states), lambda y: y[0] == 1, n_steps=3)
    container_result = brs.hard_brs(CyclingSampler(states), {(1, -1)}, n_steps=3)

    assert predicate_result.accepted_samples == ((1, 1), (1, -1))
    assert container_result.accepted_samples == ((1, -1),)
    with pytest.raises(ValueError, match="1D state"):
        brs.hard_brs(lambda: [[1, -1]], {(1, -1)}, n_steps=1)


def test_req_sample_059_soft_brs_uses_exp_beta_violation_acceptance() -> None:
    """REQ-SAMPLE-059: Soft-BRS accepts with A(y)=exp(-beta * V(y))."""

    verifiers = (
        lambda y: y[0] == 1,
        lambda y: y[1] == 1,
    )
    sampler = CyclingSampler(((1, -1), (-1, -1), (1, 1)))
    rng = SequenceRng((0.49, 0.30, 0.99))

    result = brs.soft_brs(
        sampler,
        verifiers,
        beta=math.log(2.0),
        n_steps=3,
        rng=rng,
    )

    assert result.proposal_violation_trace == (1, 2, 0)
    assert result.acceptance_probability_trace == pytest.approx((0.5, 0.25, 1.0))
    assert result.accepted_samples == ((1, -1), (1, 1))
    assert result.accepted_violation_trace == (1, 0)
    assert result.acceptance_rate == pytest.approx(2.0 / 3.0)


def test_scenario_sample_087_exact_contradictory_geometry() -> None:
    """SCENARIO-SAMPLE-087: contradictory verifiers have min residual V(y)=1."""

    verifiers = brs.contradictory_verifiers()
    states = brs.enumerate_spin_states(n=8)
    counts = brs.exact_violation_distribution(states, verifiers)

    assert counts == {1: 192, 2: 64}
    assert min(counts) == 1
    assert not any(all(verifier(state) for verifier in verifiers) for state in states)

    prior = brs.LatentSignPrior(n=8, seed=1565)
    samples = [prior() for _ in range(32)]
    assert all(len(sample) == 8 for sample in samples)
    assert all(set(sample).issubset({-1, 1}) for sample in samples)


def test_scenario_sample_087_run_experiment_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-087: runner writes the terminal JSON schema."""

    output_path = tmp_path / "experiment_1565.json"

    artifact = brs.run_experiment(output_path=output_path, decay_trials=768)

    assert brs.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["status"] == "complete"
    assert artifact["soft_gibbs_residual_implemented"] is True
    assert artifact["hard_brs_acceptance_rate"] == 0.0
    assert artifact["soft_brs_decay_confirmed"] is True
    assert artifact["min_violation_state_found"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["z_beta_curve"]) == 4
    assert [row["beta"] for row in artifact["z_beta_curve"]] == [0.5, 1.0, 2.0, 5.0]
    assert all(row["empirical_acceptance_rate"] > 0.0 for row in artifact["z_beta_curve"])
    assert artifact["min_violation_state_distribution"]
    assert all(row["violation_count"] == 1 for row in artifact["min_violation_state_distribution"])
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_req_sample_059_validate_artifact_rejects_bad_terminal_values() -> None:
    """REQ-SAMPLE-059: Exp 1565 artifacts require complete terminal semantics."""

    valid = {
        "status": "complete",
        "soft_gibbs_residual_implemented": True,
        "hard_brs_acceptance_rate": 0.0,
        "soft_brs_decay_confirmed": True,
        "min_violation_state_found": True,
        "z_beta_curve": [{"beta": 1.0, "empirical_acceptance_rate": 0.25}],
        "honest_verdict": "complete: soft_gibbs_residual_operational",
    }

    assert brs.validate_artifact(valid) is None

    missing = dict(valid)
    missing.pop("z_beta_curve")
    with pytest.raises(ValueError, match="missing required fields"):
        brs.validate_artifact(missing)

    bad_status = dict(valid, status="partial")
    with pytest.raises(ValueError, match="status must be complete"):
        brs.validate_artifact(bad_status)

    bad_hard_rate = dict(valid, hard_brs_acceptance_rate=0.1)
    with pytest.raises(ValueError, match="hard_brs_acceptance_rate"):
        brs.validate_artifact(bad_hard_rate)

    bad_verdict = dict(valid, honest_verdict="soft_gibbs_residual_operational")
    with pytest.raises(ValueError, match="honest_verdict"):
        brs.validate_artifact(bad_verdict)

    bad_implemented = dict(valid, soft_gibbs_residual_implemented=False)
    with pytest.raises(ValueError, match="soft_gibbs_residual_implemented"):
        brs.validate_artifact(bad_implemented)

    bad_decay = dict(valid, soft_brs_decay_confirmed=False)
    with pytest.raises(ValueError, match="soft_brs_decay_confirmed"):
        brs.validate_artifact(bad_decay)

    bad_min = dict(valid, min_violation_state_found=False)
    with pytest.raises(ValueError, match="min_violation_state_found"):
        brs.validate_artifact(bad_min)

    bad_curve = dict(valid, z_beta_curve=[])
    with pytest.raises(ValueError, match="z_beta_curve"):
        brs.validate_artifact(bad_curve)

    bad_curve_row = dict(valid, z_beta_curve=[{"beta": 1.0}])
    with pytest.raises(ValueError, match="z_beta_curve rows"):
        brs.validate_artifact(bad_curve_row)
