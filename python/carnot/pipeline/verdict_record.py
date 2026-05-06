"""Structured verdict records for verification APIs.

Spec: REQ-VERIFY-1408, REQ-VERIFY-1409, REQ-VERIFY-1410,
SCENARIO-VERIFY-1408
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

VerdictLabel = Literal["pass", "fail", "abstain"]

_VALID_VERDICTS = {"pass", "fail", "abstain"}


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def calibrated_confidence_from_energy(
    energy: float,
    *,
    threshold: float = 0.5,
    temperature: float = 1.0,
) -> float:
    """Map raw energy to a probability-like pass confidence.

    Lower energy maps to higher confidence.  The default sigmoid is a deterministic
    fallback calibration surface; callers can replace threshold/temperature with
    held-out Platt or isotonic parameters.

    Spec: REQ-VERIFY-1409
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if math.isnan(energy):
        return 0.0
    if energy == float("-inf"):
        return 1.0
    if energy == float("inf"):
        return 0.0

    scaled = (energy - threshold) / temperature
    if scaled >= 0.0:
        exp_neg = math.exp(-scaled)
        confidence = exp_neg / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(scaled)
        confidence = 1.0 / (1.0 + exp_pos)
    return _clamp01(confidence)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass
class VerdictRecord:
    """Structured verification verdict for downstream consumers.

    `calibrated_confidence` is the probability-like confidence that the response
    passes verification.  Lower energy therefore maps to higher confidence.

    Spec: REQ-VERIFY-1408
    """

    verdict: VerdictLabel
    energy: float
    calibrated_confidence: float
    producing_tier: int
    tier_reached: int
    rationale: str
    budget_ms_consumed: float
    repairs_applied: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(_VALID_VERDICTS)}")
        self.energy = float(self.energy)
        self.calibrated_confidence = _clamp01(float(self.calibrated_confidence))
        self.producing_tier = int(self.producing_tier)
        self.tier_reached = int(self.tier_reached)
        self.budget_ms_consumed = max(0.0, float(self.budget_ms_consumed))
        self.repairs_applied = [str(item) for item in self.repairs_applied]
        self.extras = dict(self.extras)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        payload = asdict(self)
        return _json_safe(payload)
