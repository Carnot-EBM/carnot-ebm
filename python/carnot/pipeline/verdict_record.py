"""Structured verdict records for verification APIs.

Spec: REQ-VERIFY-1408, REQ-VERIFY-1409, REQ-VERIFY-1410,
SCENARIO-VERIFY-1408
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

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


@dataclass(frozen=True)
class VerdictCalibration:
    """Held-out calibration parameters for verdict pass confidence.

    Spec: REQ-VERIFY-1409
    """

    threshold: float
    temperature: float
    n_heldout: int
    brier_score: float

    def confidence(self, energy: float) -> float:
        """Apply the fitted calibration to one energy value."""
        return calibrated_confidence_from_energy(
            energy,
            threshold=self.threshold,
            temperature=self.temperature,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible calibration summary."""
        return _json_safe(asdict(self))


def _brier_score(
    heldout_pairs: Sequence[tuple[float, bool]],
    *,
    threshold: float,
    temperature: float,
) -> float:
    total = 0.0
    for energy, passed in heldout_pairs:
        confidence = calibrated_confidence_from_energy(
            float(energy),
            threshold=threshold,
            temperature=temperature,
        )
        target = 1.0 if passed else 0.0
        total += (confidence - target) ** 2
    return total / len(heldout_pairs)


def fit_verdict_calibration(
    heldout_pairs: Sequence[tuple[float, bool]],
    *,
    temperatures: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
) -> VerdictCalibration:
    """Fit deterministic threshold/temperature parameters on held-out verdicts.

    Each held-out pair is `(energy, passed)`, where `passed=True` means the
    response should receive high pass confidence.  The fit performs a small
    deterministic grid search over observed energy thresholds and candidate
    temperatures, minimizing Brier score.  This is intentionally lightweight but
    gives downstream deployments an auditable post-hoc calibration step.

    Spec: REQ-VERIFY-1409
    """
    if not heldout_pairs:
        raise ValueError("heldout_pairs must not be empty")
    clean_pairs = [(float(energy), bool(passed)) for energy, passed in heldout_pairs]
    if any(math.isnan(energy) for energy, _passed in clean_pairs):
        raise ValueError("heldout_pairs must not contain NaN energy")
    if any(temperature <= 0.0 for temperature in temperatures):
        raise ValueError("temperatures must be positive")

    finite_energies = sorted({energy for energy, _passed in clean_pairs if math.isfinite(energy)})
    if not finite_energies:
        raise ValueError("heldout_pairs must contain at least one finite energy")
    thresholds = finite_energies
    if len(finite_energies) > 1:
        thresholds = sorted(
            {
                *finite_energies,
                *[
                    (left + right) / 2.0
                    for left, right in zip(finite_energies, finite_energies[1:], strict=False)
                ],
            }
        )

    best_threshold = thresholds[0]
    best_temperature = float(temperatures[0])
    best_score = float("inf")
    for threshold in thresholds:
        for temperature in temperatures:
            score = _brier_score(clean_pairs, threshold=threshold, temperature=float(temperature))
            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_temperature = float(temperature)

    return VerdictCalibration(
        threshold=best_threshold,
        temperature=best_temperature,
        n_heldout=len(clean_pairs),
        brier_score=best_score,
    )


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
