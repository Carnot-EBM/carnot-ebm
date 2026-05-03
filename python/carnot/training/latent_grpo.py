"""Latent-GRPO masking and one-sided reward-noise helpers.

Spec: REQ-LEARN-1187, SCENARIO-LEARN-1187, SCENARIO-LEARN-1188,
      SCENARIO-LEARN-1189.
"""

from __future__ import annotations

import math
import random
from copy import copy
from collections.abc import Callable, Iterable
from dataclasses import is_dataclass, replace
from typing import Any

REQUIRED_LATENT_GRPO_ARTIFACT_FIELDS = (
    "latent_grpo_implemented",
    "mask_rate",
    "grpo_v4_baseline_pass_rate",
    "latent_grpo_pass_rate",
    "latent_grpo_delta_pp",
    "one_sided_noise_applied",
    "n_eval_questions",
    "honest_verdict",
)

ALLOWED_LATENT_GRPO_VERDICTS = frozenset(
    {
        "latent_grpo_above_v4",
        "latent_grpo_no_delta",
        "latent_grpo_regression",
    }
)


def _energy_values(energy: Any) -> list[float]:
    if isinstance(energy, int | float):
        return [float(energy)]
    return [float(value) for value in energy]


def _is_invalid_energy(energy: Any) -> bool:
    values = _energy_values(energy)
    if not values:
        return True
    if any(not math.isfinite(value) for value in values):
        return True
    if len(values) < 5:
        return False
    if all(value == 0.0 for value in values):
        return True
    if len(set(values)) == 1:
        return True
    return all(value < 0.0 for value in values)


def mask_invalid_samples(
    rollouts: Iterable[Any],
    energies: Iterable[Any],
) -> tuple[list[Any], float]:
    """Remove rollouts with undefined or degenerate verifier energy records."""
    rollout_list = list(rollouts)
    energy_list = list(energies)
    if len(rollout_list) != len(energy_list):
        raise ValueError(
            f"rollout/energy length mismatch: {len(rollout_list)} vs {len(energy_list)}"
        )
    if not rollout_list:
        return [], 0.0

    filtered = [
        rollout
        for rollout, energy in zip(rollout_list, energy_list, strict=True)
        if not _is_invalid_energy(energy)
    ]
    n_masked = len(rollout_list) - len(filtered)
    return filtered, n_masked / len(rollout_list)


def _reward_value(rollout: Any) -> float:
    if isinstance(rollout, dict):
        return float(rollout["reward"])
    return float(getattr(rollout, "reward"))


def _replace_reward(rollout: Any, reward: float) -> Any:
    if isinstance(rollout, dict):
        updated = dict(rollout)
        updated["reward"] = float(reward)
        return updated
    if is_dataclass(rollout):
        return replace(rollout, reward=float(reward))
    updated = copy(rollout)  # pragma: no cover - defensive support for object rollouts.
    updated.reward = float(reward)  # pragma: no cover
    return updated  # pragma: no cover


def one_sided_noise_injection(
    rollout: Any,
    noise_scale: float = 0.01,
    *,
    rng: Callable[[float, float], float] | None = None,
) -> Any:
    """Add Gaussian reward noise only to positive-reward rollouts."""
    reward = _reward_value(rollout)
    if reward <= 0.0:
        return rollout
    if noise_scale < 0.0:
        raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
    noise = (rng or random.gauss)(0.0, float(noise_scale))
    return _replace_reward(rollout, reward + float(noise))


class LatentGRPOTrainer:
    """Wrapper that prepares rollouts before delegating GRPO updates."""

    def __init__(
        self,
        base_grpo_trainer: Any,
        *,
        noise_scale: float = 0.01,
        rng: Callable[[float, float], float] | None = None,
    ) -> None:
        self.base_grpo_trainer = base_grpo_trainer
        self.noise_scale = float(noise_scale)
        self.rng = rng
        self.last_mask_rate = 0.0
        self.last_n_masked = 0
        self.last_n_total = 0

    def prepare_rollouts(
        self,
        rollouts: Iterable[Any],
        energies: Iterable[Any],
        *,
        noise_scale: float | None = None,
        rng: Callable[[float, float], float] | None = None,
    ) -> list[Any]:
        rollout_list = list(rollouts)
        filtered, mask_rate = mask_invalid_samples(rollout_list, energies)
        self.last_mask_rate = float(mask_rate)
        self.last_n_total = len(rollout_list)
        self.last_n_masked = self.last_n_total - len(filtered)
        scale = self.noise_scale if noise_scale is None else float(noise_scale)
        noise_rng = self.rng if rng is None else rng
        return [one_sided_noise_injection(rollout, scale, rng=noise_rng) for rollout in filtered]

    def gradient_update(
        self,
        rollouts: Iterable[Any],
        energies: Iterable[Any],
        *args: Any,
        noise_scale: float | None = None,
        rng: Callable[[float, float], float] | None = None,
        **kwargs: Any,
    ) -> Any:
        prepared = self.prepare_rollouts(
            rollouts,
            energies,
            noise_scale=noise_scale,
            rng=rng,
        )
        return self.base_grpo_trainer.gradient_update(prepared, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_grpo_trainer, name)


def derive_latent_grpo_honest_verdict(
    latent_grpo_delta_pp: float,
    *,
    no_delta_tolerance_pp: float = 0.0,
) -> str:
    """Map the v4 comparison delta onto the Exp 1187 verdict set."""
    delta = float(latent_grpo_delta_pp)
    if abs(delta) <= float(no_delta_tolerance_pp):
        return "latent_grpo_no_delta"
    if delta > 0.0:
        return "latent_grpo_above_v4"
    return "latent_grpo_regression"


def build_latent_grpo_artifact_fields(
    *,
    mask_rate: float,
    grpo_v4_baseline_pass_rate: float,
    latent_grpo_pass_rate: float,
    n_eval_questions: int,
    one_sided_noise_applied: bool = True,
    no_delta_tolerance_pp: float = 0.0,
) -> dict[str, Any]:
    """Return the required Exp 1187 artifact fields."""
    delta = float(round(float(latent_grpo_pass_rate) - float(grpo_v4_baseline_pass_rate), 4))
    verdict = derive_latent_grpo_honest_verdict(
        delta,
        no_delta_tolerance_pp=no_delta_tolerance_pp,
    )
    return {
        "latent_grpo_implemented": True,
        "mask_rate": float(mask_rate),
        "grpo_v4_baseline_pass_rate": float(grpo_v4_baseline_pass_rate),
        "latent_grpo_pass_rate": float(latent_grpo_pass_rate),
        "latent_grpo_delta_pp": delta,
        "one_sided_noise_applied": bool(one_sided_noise_applied),
        "n_eval_questions": int(n_eval_questions),
        "honest_verdict": verdict,
    }


__all__ = [
    "ALLOWED_LATENT_GRPO_VERDICTS",
    "LatentGRPOTrainer",
    "REQUIRED_LATENT_GRPO_ARTIFACT_FIELDS",
    "build_latent_grpo_artifact_fields",
    "derive_latent_grpo_honest_verdict",
    "mask_invalid_samples",
    "one_sided_noise_injection",
]
