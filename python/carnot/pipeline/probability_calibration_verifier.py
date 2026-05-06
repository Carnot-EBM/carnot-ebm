"""Probability-calibration verifier for explicit P(event) claims.

Spec: REQ-VERIFY-1414, REQ-VERIFY-1415, SCENARIO-VERIFY-1414
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from carnot.pipeline.verdict_record import VerdictRecord, calibrated_confidence_from_energy


@dataclass(frozen=True)
class ProbabilityClaim:
    """Parsed probability claim.

    Spec: REQ-VERIFY-1414
    """

    event: str
    probability: float
    raw: str


@dataclass(frozen=True)
class ProbabilityEvidence:
    """Reference-class evidence atom used for probability calibration."""

    probability: float
    weight: float
    raw: str
    evidence_type: str


class ProbabilityCalibrationVerifier:
    """Deterministic side-car verifier for explicit probability claims.

    This verifier is intentionally lightweight: it does not train a calibration
    model. It extracts reference-class evidence from the chain and checks whether
    the explicit probability claim falls within the evidence-implied tolerance
    band.

    Spec: REQ-VERIFY-1414
    """

    _P_CLAIM = re.compile(
        r"P\((?P<event>[^)]+)\)\s*(?:=|is)\s*(?P<value>\d+(?:\.\d+)?%?)",
        re.IGNORECASE,
    )
    _PERCENT_CHANCE = re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*%\s*(?:chance|probability|risk)\s+"
        r"(?:of|for)\s+(?P<event>[A-Za-z][^.;,\n]*)",
        re.IGNORECASE,
    )
    _NAMED_PROBABILITY = re.compile(
        r"(?:probability|chance|risk)\s+(?:of|for)\s+(?P<event>[A-Za-z][^.;,\n]*)"
        r"\s+(?:is|=)\s*(?P<value>\d+(?:\.\d+)?%?)",
        re.IGNORECASE,
    )

    _OUT_OF = re.compile(
        r"(?P<count>\d+(?:\.\d+)?)\s+(?:out of|of)\s+(?P<total>\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    _FRACTION = re.compile(r"(?P<count>\d+(?:\.\d+)?)\s*/\s*(?P<total>\d+(?:\.\d+)?)")
    _BASE_RATE = re.compile(
        r"(?:base rate|reference rate|historical rate)\s*(?:is|=)\s*"
        r"(?P<value>\d+(?:\.\d+)?%?)",
        re.IGNORECASE,
    )
    _PERCENT_EVIDENCE = re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*%\s+"
        r"(?:of|among|in)\s+(?:comparable|similar|historical|reference)",
        re.IGNORECASE,
    )

    def __init__(self, *, tolerance: float = 0.10) -> None:
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = float(tolerance)

    def parse_claim(self, probability_claim: str) -> ProbabilityClaim | None:
        """Parse one explicit probability claim string."""
        for pattern in (self._P_CLAIM, self._PERCENT_CHANCE, self._NAMED_PROBABILITY):
            match = pattern.search(probability_claim)
            if match is None:
                continue
            probability = self._parse_probability(match.group("value"))
            if probability is None:
                return None
            return ProbabilityClaim(
                event=match.group("event").strip().rstrip("."),
                probability=probability,
                raw=match.group(0),
            )
        return None

    def extract_claims(self, chain: str) -> list[ProbabilityClaim]:
        """Extract explicit probability claims from a response."""
        claims: list[ProbabilityClaim] = []
        spans: list[tuple[int, int]] = []
        for pattern in (self._P_CLAIM, self._PERCENT_CHANCE, self._NAMED_PROBABILITY):
            for match in pattern.finditer(chain):
                if any(start <= match.start() < end for start, end in spans):
                    continue
                probability = self._parse_probability(match.group("value"))
                if probability is None:
                    continue
                claims.append(
                    ProbabilityClaim(
                        event=match.group("event").strip().rstrip("."),
                        probability=probability,
                        raw=match.group(0),
                    )
                )
                spans.append(match.span())
        return claims

    def extract_evidence(
        self,
        chain: str,
        *,
        exclude_text: str | None = None,
    ) -> list[ProbabilityEvidence]:
        """Extract simple reference-class probability evidence atoms."""
        text = chain.replace(exclude_text, " ") if exclude_text else chain
        evidence: list[ProbabilityEvidence] = []

        for pattern, evidence_type in (
            (self._OUT_OF, "count_out_of_total"),
            (self._FRACTION, "fraction"),
        ):
            for match in pattern.finditer(text):
                count = float(match.group("count"))
                total = float(match.group("total"))
                if total <= 0.0 or count < 0.0 or count > total:
                    continue
                evidence.append(
                    ProbabilityEvidence(
                        probability=count / total,
                        weight=max(total, 1.0),
                        raw=match.group(0),
                        evidence_type=evidence_type,
                    )
                )

        for match in self._BASE_RATE.finditer(text):
            probability = self._parse_probability(match.group("value"))
            if probability is None:
                continue
            evidence.append(
                ProbabilityEvidence(
                    probability=probability,
                    weight=1.0,
                    raw=match.group(0),
                    evidence_type="base_rate",
                )
            )

        for match in self._PERCENT_EVIDENCE.finditer(text):
            probability = self._parse_probability(match.group("value") + "%")
            if probability is None:
                continue
            evidence.append(
                ProbabilityEvidence(
                    probability=probability,
                    weight=1.0,
                    raw=match.group(0),
                    evidence_type="percent_reference",
                )
            )

        return evidence

    def score(self, chain: str, probability_claim: ProbabilityClaim | str) -> VerdictRecord:
        """Score one probability claim against evidence from the reasoning chain."""
        claim = (
            probability_claim
            if isinstance(probability_claim, ProbabilityClaim)
            else self.parse_claim(probability_claim)
        )
        if claim is None:
            return self._abstain("no_parseable_probability_claim", None, [])

        evidence = self.extract_evidence(chain, exclude_text=claim.raw)
        if not evidence:
            return self._abstain("no_reference_class_evidence", claim, evidence)

        implied_probability = self._weighted_mean(evidence)
        lo = max(0.0, implied_probability - self.tolerance)
        hi = min(1.0, implied_probability + self.tolerance)
        if lo <= claim.probability <= hi:
            energy = 0.0
            verdict = "pass"
            rationale = "claimed_probability_inside_reference_range"
        else:
            energy = min(abs(claim.probability - lo), abs(claim.probability - hi))
            verdict = "fail"
            rationale = "claimed_probability_outside_reference_range"

        return VerdictRecord(
            verdict=verdict,
            energy=energy,
            calibrated_confidence=calibrated_confidence_from_energy(
                energy,
                threshold=max(self.tolerance, 1e-6),
                temperature=max(self.tolerance, 1e-6),
            ),
            producing_tier=2,
            tier_reached=2,
            rationale=rationale,
            budget_ms_consumed=0.0,
            extras=self._extras(claim, evidence, implied_probability, (lo, hi)),
        )

    def score_text(self, chain: str) -> list[VerdictRecord]:
        """Score every explicit probability claim found in a response."""
        return [self.score(chain, claim) for claim in self.extract_claims(chain)]

    @staticmethod
    def _parse_probability(raw_value: str) -> float | None:
        value_text = raw_value.strip()
        is_percent = value_text.endswith("%")
        if is_percent:
            value_text = value_text[:-1].strip()
        try:
            value = float(value_text)
        except ValueError:
            return None
        if is_percent or value > 1.0:
            value /= 100.0
        if value < 0.0 or value > 1.0:
            return None
        return value

    @staticmethod
    def _weighted_mean(evidence: list[ProbabilityEvidence]) -> float:
        total_weight = sum(item.weight for item in evidence)
        if total_weight <= 0.0:
            return 0.0
        return sum(item.probability * item.weight for item in evidence) / total_weight

    def _abstain(
        self,
        rationale: str,
        claim: ProbabilityClaim | None,
        evidence: list[ProbabilityEvidence],
    ) -> VerdictRecord:
        return VerdictRecord(
            verdict="abstain",
            energy=0.0,
            calibrated_confidence=0.0,
            producing_tier=2,
            tier_reached=2,
            rationale=rationale,
            budget_ms_consumed=0.0,
            extras=self._extras(claim, evidence, None, None),
        )

    def _extras(
        self,
        claim: ProbabilityClaim | None,
        evidence: list[ProbabilityEvidence],
        implied_probability: float | None,
        implied_range: tuple[float, float] | None,
    ) -> dict[str, Any]:
        return {
            "event": claim.event if claim else None,
            "claimed_probability": claim.probability if claim else None,
            "implied_probability": implied_probability,
            "implied_range": list(implied_range) if implied_range else None,
            "tolerance": self.tolerance,
            "evidence_count": len(evidence),
            "evidence": [
                {
                    "probability": item.probability,
                    "weight": item.weight,
                    "raw": item.raw,
                    "type": item.evidence_type,
                }
                for item in evidence
            ],
        }
