"""ActFocus token-energy weighting for verifier fast updates.

ActFocus separates answer-action tokens from explanatory reasoning tokens and
uses action-token energy variance as the credit-assignment signal.  In Carnot's
verify-repair loop this means verifier feedback tied to answer edits should be
retained more strongly than diffuse reasoning-token feedback.

Spec: REQ-LEARN-2242, SCENARIO-LEARN-2242.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from typing import Literal

TokenRole = Literal["reasoning", "action"]

_ACTION_MARKERS = {
    "answer",
    "correct",
    "corrected",
    "equals",
    "final",
    "fix",
    "fixed",
    "return",
    "therefore",
    "thus",
}


@dataclass(frozen=True)
class TokenEnergyRecord:
    """One token with a deterministic role and energy contribution."""

    index: int
    token: str
    role: TokenRole
    energy: float


@dataclass(frozen=True)
class ActFocusTokenWeight:
    """ActFocus weight assigned to one token-energy record."""

    index: int
    token: str
    role: TokenRole
    energy: float
    weight: float


def build_token_energy_trace(text: str, *, base_energy: float) -> list[TokenEnergyRecord]:
    """Build a deterministic ActFocus token-energy trace from one response.

    The helper is intentionally model-agnostic.  Live GGUF runs can replace the
    deterministic token energies with logprob or verifier energies later, but
    the weighting contract stays the same: tokens in the final answer/action
    span receive higher-variance energy, while explanatory reasoning tokens are
    damped.
    """

    tokens = text.split()
    records: list[TokenEnergyRecord] = []
    in_action_span = False
    for index, token in enumerate(tokens):
        normalized = _normalize_token(token)
        if normalized in _ACTION_MARKERS or normalized.startswith("final"):
            in_action_span = True
        role: TokenRole = "action" if in_action_span else "reasoning"
        records.append(
            TokenEnergyRecord(
                index=index,
                token=token,
                role=role,
                energy=_token_energy(token, role=role, base_energy=base_energy),
            )
        )
    return records


def energy_variance_by_role(records: list[TokenEnergyRecord]) -> dict[str, float]:
    """Return population energy variance for reasoning and action token roles."""

    return {
        "reasoning": _variance([record.energy for record in records if record.role == "reasoning"]),
        "action": _variance([record.energy for record in records if record.role == "action"]),
    }


def compute_actfocus_weights(records: list[TokenEnergyRecord]) -> list[ActFocusTokenWeight]:
    """Return per-token weights that downweight reasoning and upweight action."""

    variances = energy_variance_by_role(records)
    total = variances["reasoning"] + variances["action"]
    if total <= 0.0:
        action_fraction = 0.0
    else:
        action_fraction = variances["action"] / total

    action_weight = 1.0 + action_fraction
    reasoning_weight = max(0.1, 1.0 - 0.55 * action_fraction)
    return [
        ActFocusTokenWeight(
            index=record.index,
            token=record.token,
            role=record.role,
            energy=record.energy,
            weight=action_weight if record.role == "action" else reasoning_weight,
        )
        for record in records
    ]


def actfocus_fast_update_score(records: list[TokenEnergyRecord]) -> float:
    """Score how valuable a verifier-output fast update is to retain.

    The score is zero without action tokens.  Otherwise it combines action
    energy variance with the action share of weighted token energy, which makes
    final-answer edits dominate vague chain-of-thought commentary.
    """

    variances = energy_variance_by_role(records)
    action_variance = variances["action"]
    if action_variance <= 0.0:
        return 0.0

    weights = compute_actfocus_weights(records)
    action_energy = sum(row.energy * row.weight for row in weights if row.role == "action")
    total_energy = sum(row.energy * row.weight for row in weights)
    if total_energy <= 0.0:
        return 0.0
    return round(action_variance * (action_energy / total_energy), 6)


def _token_energy(token: str, *, role: TokenRole, base_energy: float) -> float:
    unit = _stable_unit_interval(token)
    if role == "action":
        return round(float(base_energy) * (0.6 + 0.85 * unit), 6)
    return round(float(base_energy) * (0.06 + 0.08 * unit), 6)


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def _normalize_token(token: str) -> str:
    return "".join(char for char in token.lower() if char.isalnum())


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(float(statistics.pvariance(values)), 6)


__all__ = [
    "ActFocusTokenWeight",
    "TokenEnergyRecord",
    "actfocus_fast_update_score",
    "build_token_energy_trace",
    "compute_actfocus_weights",
    "energy_variance_by_role",
]
