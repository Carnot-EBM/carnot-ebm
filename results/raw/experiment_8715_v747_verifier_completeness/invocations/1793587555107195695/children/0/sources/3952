"""ODAR-style free-energy routing for Carnot's verify-repair cascade.

**Researcher summary:**
    Fuses cheap Tier 0 probe outputs into a single expected free-energy score.
    Low EFE means the response looks low-risk enough to keep on the fast path;
    high EFE means the pipeline should spend compute on the deliberative
    verifier tiers.

**Detailed explanation for engineers:**
    ODAR frames routing as a risk-sensitive free-energy decision instead of an
    unconditional fall-through from cheap probes into expensive verification.
    Carnot's Tier 0 probes already produce small diagnostic records such as
    ``risk_score``, ``confidence``, ``is_unstable``, ``energy``, or ``verdict``.
    ``FreeEnergyRouter`` turns those heterogeneous records into normalized
    probe contributions:

    - risk: probability-like evidence that the response is unsafe or wrong;
    - ambiguity: uncertainty left by a low-confidence probe;
    - weight: optional probe importance, defaulting to one.

    The expected free energy (EFE) is the weighted mean of
    ``risk + ambiguity_weight * ambiguity``.  That simple form keeps the gate
    auditable: confident low-risk probes route fast, unstable or high-risk
    probes route to the deliberative verifier, and missing evidence never
    causes an optimistic skip.

Spec: REQ-ODAR-2243, SCENARIO-ODAR-2243
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum, StrEnum
import math
from typing import Any


class RoutingDecision(StrEnum):
    """Decision returned by the ODAR router."""

    FAST_PATH = "FAST_PATH"
    DELIBERATIVE = "DELIBERATIVE"


@dataclass(frozen=True)
class ProbeContribution:
    """One Tier 0 probe's normalized contribution to expected free energy."""

    name: str
    risk: float
    ambiguity: float
    weight: float
    free_energy: float
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for verification certificates."""
        return {
            "name": self.name,
            "risk": self.risk,
            "ambiguity": self.ambiguity,
            "weight": self.weight,
            "free_energy": self.free_energy,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class RoutingResult:
    """Full ODAR route evaluation with the scalar EFE and per-probe details."""

    decision: RoutingDecision
    expected_free_energy: float
    risk_threshold: float
    contributions: tuple[ProbeContribution, ...]

    def to_certificate(self) -> dict[str, Any]:
        """Return fields suitable for embedding in ``VerificationResult.certificate``."""
        return {
            "odar_decision": self.decision.value,
            "odar_expected_free_energy": self.expected_free_energy,
            "odar_risk_threshold": self.risk_threshold,
            "odar_contributions": [c.to_dict() for c in self.contributions],
        }


@dataclass
class FreeEnergyRouter:
    """Risk-sensitive Tier 0 fusion router.

    Args:
        risk_threshold: EFE below this value routes to ``FAST_PATH``.  EFE at
            or above this value routes to ``DELIBERATIVE``.
        ambiguity_weight: Multiplier on uncertainty from low-confidence probes.
            A modest default keeps risk evidence primary while still making
            ambiguous probe outputs pay a compute cost.
    """

    risk_threshold: float = 0.5
    ambiguity_weight: float = 0.25

    def expected_free_energy(self, probe_outputs: Any) -> float:
        """Compute the scalar EFE from Tier 0 probe outputs.

        Empty or unparseable probe evidence returns ``math.inf`` so the caller
        falls through to deliberative verification instead of skipping on
        missing data.
        """
        return self.evaluate(probe_outputs).expected_free_energy

    def route(self, probe_outputs: Any) -> RoutingDecision:
        """Return ``FAST_PATH`` when EFE is below threshold, else ``DELIBERATIVE``."""
        return self.evaluate(probe_outputs).decision

    def evaluate(self, probe_outputs: Any) -> RoutingResult:
        """Compute EFE and return the full routing result."""
        contributions = tuple(
            contribution
            for name, output in _iter_probe_outputs(probe_outputs)
            if (contribution := self._contribution_from_output(name, output)) is not None
        )
        if not contributions:
            efe = math.inf
        else:
            total_weight = sum(contribution.weight for contribution in contributions)
            efe = (
                sum(
                    contribution.weight * contribution.free_energy for contribution in contributions
                )
                / total_weight
            )

        decision = (
            RoutingDecision.FAST_PATH
            if math.isfinite(efe) and efe < self.risk_threshold
            else RoutingDecision.DELIBERATIVE
        )
        return RoutingResult(
            decision=decision,
            expected_free_energy=efe,
            risk_threshold=self.risk_threshold,
            contributions=contributions,
        )

    def _contribution_from_output(self, name: str, output: Any) -> ProbeContribution | None:
        data = _as_mapping(output)
        risk = _extract_risk(data)
        ambiguity = _extract_ambiguity(data)
        if risk is None and ambiguity is None:
            return None
        if risk is None:
            risk = 0.0
        if ambiguity is None:
            ambiguity = 0.0

        weight = _positive_float(data.get("weight"), default=1.0)
        free_energy = risk + self.ambiguity_weight * ambiguity
        return ProbeContribution(
            name=name,
            risk=risk,
            ambiguity=ambiguity,
            weight=weight,
            free_energy=free_energy,
            evidence=_json_safe_mapping(data),
        )


_DIRECT_SIGNAL_KEYS = {
    "ambiguity",
    "calibrated_confidence",
    "confidence",
    "energy",
    "entropy",
    "expected_free_energy",
    "is_spectrally_diffuse",
    "is_streaming_unstable",
    "is_unstable",
    "nup_score",
    "p_violation",
    "probability",
    "probability_correct",
    "risk",
    "risk_score",
    "score",
    "satisfied",
    "threshold",
    "uncertainty",
    "unstable",
    "value",
    "verdict",
    "violation_probability",
}

_RISK_KEYS = (
    "expected_free_energy",
    "risk_score",
    "risk",
    "p_violation",
    "violation_probability",
    "probability",
    "nup_score",
    "score",
    "energy",
    "value",
)

_AMBIGUITY_KEYS = (
    "ambiguity",
    "uncertainty",
    "entropy",
)

_CONFIDENCE_KEYS = (
    "confidence",
    "calibrated_confidence",
    "probability_correct",
)

_UNSTABLE_KEYS = (
    "is_unstable",
    "unstable",
    "is_streaming_unstable",
    "is_spectrally_diffuse",
)


def _iter_probe_outputs(probe_outputs: Any) -> tuple[tuple[str, Any], ...]:
    if probe_outputs is None:
        return ()
    if isinstance(probe_outputs, Mapping):
        keys = {str(key) for key in probe_outputs}
        if keys & _DIRECT_SIGNAL_KEYS:
            return (("probe", probe_outputs),)
        return tuple((str(name), output) for name, output in probe_outputs.items())
    if isinstance(probe_outputs, Sequence) and not isinstance(probe_outputs, str | bytes):
        return tuple((f"probe_{index}", output) for index, output in enumerate(probe_outputs))
    return (("probe", probe_outputs),)


def _as_mapping(output: Any) -> dict[str, Any]:
    if isinstance(output, Mapping):
        return dict(output)
    if is_dataclass(output) and not isinstance(output, type):
        return asdict(output)
    if hasattr(output, "_asdict"):
        return dict(output._asdict())

    data: dict[str, Any] = {}
    for key in _DIRECT_SIGNAL_KEYS:
        if hasattr(output, key):
            try:
                data[key] = getattr(output, key)
            except Exception:
                continue
    if data:
        return data
    return {"value": output}


def _extract_risk(data: Mapping[str, Any]) -> float | None:
    risk: float | None = None

    satisfied = data.get("satisfied")
    if isinstance(satisfied, bool):
        risk = 0.0 if satisfied else 1.0

    verdict_risk = _risk_from_verdict(data.get("verdict"))
    if verdict_risk is not None:
        risk = verdict_risk if risk is None else max(risk, verdict_risk)

    unstable_seen = False
    for key in _UNSTABLE_KEYS:
        if key in data:
            unstable_seen = True
            if bool(data[key]):
                risk = 1.0 if risk is None else max(risk, 1.0)
    if unstable_seen and risk is None:
        risk = 0.0

    for key in _RISK_KEYS:
        value = _finite_float(data.get(key))
        if value is None:
            continue
        threshold = _finite_float(data.get("threshold"))
        if key in {"score", "energy"} and threshold is not None:
            risk_value = _thresholded_risk(value, threshold)
        else:
            risk_value = _bounded_probability(value)
        risk = risk_value if risk is None else max(risk, risk_value)
        break

    return risk


def _extract_ambiguity(data: Mapping[str, Any]) -> float | None:
    for key in _AMBIGUITY_KEYS:
        value = _finite_float(data.get(key))
        if value is not None:
            return _bounded_probability(value)

    for key in _CONFIDENCE_KEYS:
        value = _finite_float(data.get(key))
        if value is not None:
            return 1.0 - _bounded_probability(value)

    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict in {"uncertain", "unknown", "abstain"}:
        return 0.5
    return None


def _risk_from_verdict(value: Any) -> float | None:
    if value is None:
        return None
    verdict = str(value).strip().lower()
    if verdict in {"incorrect", "violated", "violation", "fail", "failed", "unstable"}:
        return 1.0
    if verdict in {"correct", "pass", "passed", "verified", "stable", "satisfied"}:
        return 0.0
    if verdict in {"uncertain", "unknown", "abstain"}:
        return 0.5
    return None


def _thresholded_risk(value: float, threshold: float) -> float:
    scale = max(abs(threshold), 1.0)
    return _sigmoid(4.0 * ((value - threshold) / scale))


def _bounded_probability(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value
    return _sigmoid(value)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _positive_float(value: Any, *, default: float) -> float:
    number = _finite_float(value)
    if number is None or number <= 0.0:
        return default
    return number


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _json_safe_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe_value(value) for key, value in data.items()}


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_json_safe_value(item) for item in value]
    return repr(value)

class OdarRouter:
    """ODAR router that uses free-energy objective for iteration selection."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations

    def route(self, current_energy: float, prior_energy: float, iteration: int) -> tuple[bool, float, float]:
        """
        Evaluate if we should route to another repair iteration.
        Returns: (route_to_repair, variational_free_energy, expected_reduction)
        """
        delta_E = prior_energy - current_energy
        expected_reduction = delta_E * math.exp(-0.5 * iteration)
        complexity = abs(delta_E) / (iteration + 1)
        variational_free_energy = -expected_reduction + 0.3 * complexity

        # We route to repair if the expected gain is positive (VFE < 0)
        route_to_repair = variational_free_energy < 0 and iteration < self.max_iterations
        return route_to_repair, variational_free_energy, expected_reduction
