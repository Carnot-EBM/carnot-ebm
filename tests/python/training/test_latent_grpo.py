"""Tests for Latent-GRPO invalid-sample masking and reward noise.

Spec: REQ-LEARN-1187, SCENARIO-LEARN-1187, SCENARIO-LEARN-1188,
      SCENARIO-LEARN-1189.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from carnot.training.latent_grpo import (
    LatentGRPOTrainer,
    build_latent_grpo_artifact_fields,
    derive_latent_grpo_honest_verdict,
    mask_invalid_samples,
    one_sided_noise_injection,
)


@dataclass(frozen=True)
class Rollout:
    """Tiny immutable rollout used to prove reward updates preserve type."""

    text: str
    reward: float


def test_mask_invalid_samples_filters_nonfinite_flat_and_negative_ensembles() -> None:
    """SCENARIO-LEARN-1187: invalid k=5 energy ensembles are masked."""
    rollouts = [
        Rollout("valid", 1.0),
        Rollout("nan", 1.0),
        Rollout("inf", 1.0),
        Rollout("same", 1.0),
        Rollout("zeros", 1.0),
        Rollout("negative", 1.0),
    ]
    energies = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [math.nan, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, math.inf, 3.0, 4.0],
        [7.0, 7.0, 7.0, 7.0, 7.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, -2.0, -3.0, -4.0, -5.0],
    ]

    filtered, mask_rate = mask_invalid_samples(rollouts, energies)

    assert filtered == [rollouts[0]]
    assert math.isclose(mask_rate, 5 / 6)


def test_mask_invalid_samples_empty_input_has_zero_mask_rate() -> None:
    """REQ-LEARN-1187-1: empty rollout groups return no masked fraction."""
    filtered, mask_rate = mask_invalid_samples([], [])

    assert filtered == []
    assert mask_rate == 0.0


def test_mask_invalid_samples_accepts_scalar_energy_and_masks_empty_record() -> None:
    """REQ-LEARN-1187-2: scalar finite energy is valid; empty energy is invalid."""
    rollouts = [Rollout("scalar-valid", 0.0), Rollout("empty-invalid", 0.0)]

    filtered, mask_rate = mask_invalid_samples(rollouts, [0.5, []])

    assert filtered == [rollouts[0]]
    assert math.isclose(mask_rate, 0.5)


def test_mask_invalid_samples_requires_aligned_energy_records() -> None:
    """REQ-LEARN-1187-1: each rollout must have one energy record."""
    try:
        mask_invalid_samples([Rollout("a", 0.0)], [])
    except ValueError as exc:
        assert "length mismatch" in str(exc)
    else:
        raise AssertionError("mask_invalid_samples accepted misaligned inputs")


def test_one_sided_noise_only_changes_positive_rewards() -> None:
    """SCENARIO-LEARN-1188: negative rewards preserve the verifier boundary."""
    positive = Rollout("positive", 1.0)
    negative = Rollout("negative", -1.0)
    neutral = Rollout("neutral", 0.0)

    noisy_positive = one_sided_noise_injection(
        positive,
        noise_scale=0.01,
        rng=lambda mean, stdev: 0.25,
    )
    noisy_negative = one_sided_noise_injection(
        negative,
        noise_scale=0.01,
        rng=lambda mean, stdev: 0.25,
    )
    noisy_neutral = one_sided_noise_injection(
        neutral,
        noise_scale=0.01,
        rng=lambda mean, stdev: 0.25,
    )

    assert noisy_positive == Rollout("positive", 1.25)
    assert noisy_negative == negative
    assert noisy_neutral == neutral


def test_one_sided_noise_supports_mapping_rollouts_and_validates_scale() -> None:
    """REQ-LEARN-1187-3: mapping rewards are updated and scale must be valid."""
    noisy = one_sided_noise_injection(
        {"text": "positive", "reward": 1.0},
        noise_scale=0.01,
        rng=lambda mean, stdev: -0.1,
    )

    assert noisy == {"text": "positive", "reward": 0.9}

    try:
        one_sided_noise_injection(Rollout("bad-scale", 1.0), noise_scale=-0.01)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("one_sided_noise_injection accepted negative noise_scale")


def test_latent_grpo_trainer_masks_and_noises_before_gradient_update() -> None:
    """REQ-LEARN-1187-4: wrapper delegates only prepared valid rollouts."""

    class BaseTrainer:
        def __init__(self) -> None:
            self.received: list[Rollout] | None = None

        def gradient_update(self, rollouts: list[Rollout]) -> dict[str, int]:
            self.received = rollouts
            return {"n": len(rollouts)}

    base = BaseTrainer()
    trainer = LatentGRPOTrainer(base)
    rollouts = [
        Rollout("valid-positive", 1.0),
        Rollout("invalid-flat", 1.0),
        Rollout("valid-negative", -1.0),
    ]
    energies = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [4.0, 3.0, 2.0, 1.0, 0.0],
    ]

    result = trainer.gradient_update(
        rollouts,
        energies,
        noise_scale=0.01,
        rng=lambda mean, stdev: 0.5,
    )

    assert result == {"n": 2}
    assert base.received == [
        Rollout("valid-positive", 1.5),
        Rollout("valid-negative", -1.0),
    ]
    assert math.isclose(trainer.last_mask_rate, 1 / 3)
    assert trainer.last_n_masked == 1
    assert trainer.received == base.received


def test_latent_grpo_verdict_mapping_covers_no_delta_and_regression() -> None:
    """REQ-LEARN-1187-6: v4 comparison deltas map to the allowed verdicts."""
    assert derive_latent_grpo_honest_verdict(0.0) == "latent_grpo_no_delta"
    assert derive_latent_grpo_honest_verdict(-0.01) == "latent_grpo_regression"


def test_latent_grpo_artifact_fields_report_v4_delta_and_verdict() -> None:
    """SCENARIO-LEARN-1189: artifact fields include the v4 comparison."""
    fields = build_latent_grpo_artifact_fields(
        mask_rate=0.2,
        grpo_v4_baseline_pass_rate=0.26,
        latent_grpo_pass_rate=0.31,
        n_eval_questions=100,
    )

    assert fields["latent_grpo_implemented"] is True
    assert fields["mask_rate"] == 0.2
    assert fields["grpo_v4_baseline_pass_rate"] == 0.26
    assert fields["latent_grpo_pass_rate"] == 0.31
    assert math.isclose(fields["latent_grpo_delta_pp"], 0.05)
    assert fields["one_sided_noise_applied"] is True
    assert fields["n_eval_questions"] == 100
    assert fields["honest_verdict"] == "latent_grpo_above_v4"
