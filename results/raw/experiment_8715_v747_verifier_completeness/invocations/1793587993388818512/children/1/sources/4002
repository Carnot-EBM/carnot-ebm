"""Claim-isolated semantic verifier with calibrated confidence and abstain.

Spec: REQ-VERIFY-046, REQ-VERIFY-047,
SCENARIO-VERIFY-047, SCENARIO-VERIFY-048, SCENARIO-VERIFY-049
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.semantic_grounding import (
    PromptClause,
    QuestionProfile,
    SemanticClaim,
    SemanticGroundingResult,
    SemanticGroundingVerifier,
    _claim_covers_clause,
    _clause_requires_grounding,
    _extract_keywords,
    _extract_quantities,
)
from carnot.pipeline.structured_reasoning import (
    StructuredReasoningController,
    get_repo_root,
)
from carnot.pipeline.typed_reasoning import extract_typed_reasoning as build_typed_reasoning_ir

if TYPE_CHECKING:
    from pathlib import Path

    from carnot.pipeline.typed_reasoning import TypedReasoningIR

RUN_DATE = "20260413"
_CALIBRATION_DEFAULT = "data/research/semantic_calibration_corpus_232.jsonl"
_JSON_MODES = {"structured_json", "minimal_json", "grammar_gated_json"}


@dataclass(frozen=True)
class SemanticVerifierV2Thresholds:
    """Thresholds derived from the checked-in Exp 232 calibration corpus."""

    support_max_error_probability: float
    violation_min_error_probability: float
    min_monitorability_confidence: float
    source_run_date: str
    calibration_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "support_max_error_probability": self.support_max_error_probability,
            "violation_min_error_probability": self.violation_min_error_probability,
            "min_monitorability_confidence": self.min_monitorability_confidence,
            "source_run_date": self.source_run_date,
            "calibration_rows": self.calibration_rows,
        }


@dataclass(frozen=True)
class SemanticClaimResult:
    """Claim-level semantic analysis for the v2 verifier."""

    claim_id: str
    text: str
    is_final: bool
    status: str
    answer_target_coverage: float
    premise_support: float
    monitorability_confidence: float
    semantic_error_probability: float
    matched_clause_ids: list[str] = field(default_factory=list)
    missing_clause_ids: list[str] = field(default_factory=list)
    missing_target_keywords: list[str] = field(default_factory=list)
    supporting_claim_ids: list[str] = field(default_factory=list)
    legacy_violation_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "is_final": self.is_final,
            "status": self.status,
            "answer_target_coverage": self.answer_target_coverage,
            "premise_support": self.premise_support,
            "monitorability_confidence": self.monitorability_confidence,
            "semantic_error_probability": self.semantic_error_probability,
            "matched_clause_ids": list(self.matched_clause_ids),
            "missing_clause_ids": list(self.missing_clause_ids),
            "missing_target_keywords": list(self.missing_target_keywords),
            "supporting_claim_ids": list(self.supporting_claim_ids),
            "legacy_violation_types": list(self.legacy_violation_types),
        }

    def to_constraint_result(self) -> ConstraintResult:
        missing_targets = ", ".join(self.missing_target_keywords) or "none"
        missing_clauses = ", ".join(self.missing_clause_ids) or "none"
        return ConstraintResult(
            constraint_type="semantic_verifier_v2",
            description=(
                f"Claim '{self.text}' failed calibrated semantic verification "
                f"(missing targets: {missing_targets}; missing clauses: {missing_clauses})"
            ),
            metadata={
                "satisfied": False,
                "claim_id": self.claim_id,
                "status": self.status,
                "answer_target_coverage": self.answer_target_coverage,
                "premise_support": self.premise_support,
                "monitorability_confidence": self.monitorability_confidence,
                "semantic_error_probability": self.semantic_error_probability,
                "matched_clause_ids": list(self.matched_clause_ids),
                "missing_clause_ids": list(self.missing_clause_ids),
                "missing_target_keywords": list(self.missing_target_keywords),
                "supporting_claim_ids": list(self.supporting_claim_ids),
                "legacy_violation_types": list(self.legacy_violation_types),
            },
        )


@dataclass(frozen=True)
class SemanticVerifierV2Result:
    """Structured result object for the v2 semantic verifier."""

    question_profile: QuestionProfile
    claims: list[SemanticClaim]
    claim_results: list[SemanticClaimResult]
    focus_claim_id: str | None
    verdict: str
    semantic_error_probability: float
    monitorability_confidence: float
    thresholds: SemanticVerifierV2Thresholds
    response_mode: str
    recommended_response_mode: str | None
    legacy_violation_types: list[str] = field(default_factory=list)
    run_date: str = RUN_DATE

    def to_constraint_results(self) -> list[ConstraintResult]:
        if self.verdict != "violated":
            return []
        violations = [claim for claim in self.claim_results if claim.status == "violated"]
        if not violations and self.focus_claim_id is not None:
            violations = [
                claim for claim in self.claim_results if claim.claim_id == self.focus_claim_id
            ]
        return [claim.to_constraint_result() for claim in violations]

    def to_dict(self) -> dict[str, object]:
        return {
            "question_profile": self.question_profile.to_dict(),
            "claims": [claim.to_dict() for claim in self.claims],
            "claim_results": [claim.to_dict() for claim in self.claim_results],
            "focus_claim_id": self.focus_claim_id,
            "verdict": self.verdict,
            "semantic_error_probability": self.semantic_error_probability,
            "monitorability_confidence": self.monitorability_confidence,
            "thresholds": self.thresholds.to_dict(),
            "response_mode": self.response_mode,
            "recommended_response_mode": self.recommended_response_mode,
            "legacy_violation_types": list(self.legacy_violation_types),
            "run_date": self.run_date,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


class SemanticVerifierV2:
    """Semantic verifier that only fails high-confidence claim-level mismatches."""

    def __init__(
        self,
        *,
        grounding_verifier: SemanticGroundingVerifier | None = None,
        calibration_path: Path | None = None,
        policy_path: Path | None = None,
        min_monitorability_confidence: float = 0.35,
    ) -> None:
        self._grounding = grounding_verifier or SemanticGroundingVerifier()
        self._routing_policy = StructuredReasoningController(policy_path=policy_path)
        self.thresholds = _load_thresholds(
            calibration_path=calibration_path,
            min_monitorability_confidence=min_monitorability_confidence,
        )

    def verify(
        self,
        question: str,
        response: str,
        *,
        typed_reasoning: TypedReasoningIR | None = None,
        semantic_grounding: SemanticGroundingResult | None = None,
        task_slice: str = "live_gsm8k_semantic_failure",
    ) -> SemanticVerifierV2Result:
        if typed_reasoning is None:
            try:
                typed_reasoning = build_typed_reasoning_ir(question=question, response=response)
            except Exception:
                typed_reasoning = None
        grounding = semantic_grounding or self._grounding.verify(
            question=question,
            response=response,
            typed_reasoning=typed_reasoning,
        )
        claims = _expand_claims(grounding.claims, typed_reasoning)
        response_mode = _response_mode(typed_reasoning, response)
        recommended_mode = self._routing_policy.recommended_mode(task_slice)
        legacy_types = sorted({violation.violation_type for violation in grounding.violations})

        if grounding.question_profile.is_code_like or not claims:
            return SemanticVerifierV2Result(
                question_profile=grounding.question_profile,
                claims=claims,
                claim_results=[],
                focus_claim_id=None,
                verdict="abstain",
                semantic_error_probability=_round_score(0.18),
                monitorability_confidence=0.0,
                thresholds=self.thresholds,
                response_mode=response_mode,
                recommended_response_mode=recommended_mode,
                legacy_violation_types=legacy_types,
            )

        required_clauses = [
            clause
            for clause in grounding.question_profile.prompt_clauses
            if _clause_requires_grounding(clause, grounding.question_profile)
        ]
        clause_matches = {
            claim.claim_id: _matched_clause_ids(required_clauses, claim) for claim in claims
        }
        overall_clause_ids = sorted(
            {clause_id for matched in clause_matches.values() for clause_id in matched}
        )
        overall_premise_support = _ratio(len(overall_clause_ids), len(required_clauses))
        focus_claim = _select_focus_claim(
            claims=claims,
            question_profile=grounding.question_profile,
            clause_matches=clause_matches,
        )

        claim_results: list[SemanticClaimResult] = []
        for claim in claims:
            target_coverage = _target_coverage(grounding.question_profile, claim)
            premise_support = _ratio(len(clause_matches[claim.claim_id]), len(required_clauses))
            monitorability = _monitorability_confidence(
                target_coverage=target_coverage,
                premise_support=(
                    overall_premise_support
                    if claim.claim_id == focus_claim.claim_id
                    else premise_support
                ),
                claim=claim,
                response_mode=response_mode,
                recommended_response_mode=recommended_mode,
            )
            raw_error = _raw_error_probability(
                claim=claim,
                focus_claim=focus_claim,
                target_coverage=target_coverage,
                premise_support=(
                    overall_premise_support
                    if claim.claim_id == focus_claim.claim_id
                    else premise_support
                ),
                legacy_violation_types=legacy_types,
                response_mode=response_mode,
                recommended_response_mode=recommended_mode,
                question_profile=grounding.question_profile,
            )
            semantic_error_probability = _calibrate_probability(raw_error, monitorability)
            status = _claim_status(
                semantic_error_probability=semantic_error_probability,
                monitorability_confidence=monitorability,
                thresholds=self.thresholds,
            )
            missing_clause_ids = [
                clause.clause_id
                for clause in required_clauses
                if clause.clause_id not in clause_matches[claim.claim_id]
            ]
            claim_results.append(
                SemanticClaimResult(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    is_final=claim.is_final,
                    status=status,
                    answer_target_coverage=target_coverage,
                    premise_support=(
                        overall_premise_support
                        if claim.claim_id == focus_claim.claim_id
                        else premise_support
                    ),
                    monitorability_confidence=monitorability,
                    semantic_error_probability=semantic_error_probability,
                    matched_clause_ids=list(clause_matches[claim.claim_id]),
                    missing_clause_ids=missing_clause_ids,
                    missing_target_keywords=_missing_target_keywords(
                        grounding.question_profile,
                        claim,
                    ),
                    supporting_claim_ids=_supporting_claim_ids(
                        claim=claim,
                        claims=claims,
                        clause_matches=clause_matches,
                        required_clauses=required_clauses,
                    ),
                    legacy_violation_types=(
                        legacy_types if claim.claim_id == focus_claim.claim_id else []
                    ),
                )
            )

        focus_result = next(
            claim_result
            for claim_result in claim_results
            if claim_result.claim_id == focus_claim.claim_id
        )
        overall_monitorability = focus_result.monitorability_confidence
        overall_error_probability = focus_result.semantic_error_probability
        verdict = _claim_status(
            semantic_error_probability=overall_error_probability,
            monitorability_confidence=overall_monitorability,
            thresholds=self.thresholds,
        )

        return SemanticVerifierV2Result(
            question_profile=grounding.question_profile,
            claims=claims,
            claim_results=claim_results,
            focus_claim_id=focus_claim.claim_id,
            verdict=verdict,
            semantic_error_probability=overall_error_probability,
            monitorability_confidence=overall_monitorability,
            thresholds=self.thresholds,
            response_mode=response_mode,
            recommended_response_mode=recommended_mode,
            legacy_violation_types=legacy_types,
        )


def verify_semantic_verifier_v2(
    question: str,
    response: str,
    *,
    typed_reasoning: TypedReasoningIR | None = None,
    semantic_grounding: SemanticGroundingResult | None = None,
    task_slice: str = "live_gsm8k_semantic_failure",
) -> SemanticVerifierV2Result:
    """Convenience helper for one-shot semantic verifier v2 calls."""
    return SemanticVerifierV2().verify(
        question=question,
        response=response,
        typed_reasoning=typed_reasoning,
        semantic_grounding=semantic_grounding,
        task_slice=task_slice,
    )


def _load_thresholds(
    *,
    calibration_path: Path | None,
    min_monitorability_confidence: float,
) -> SemanticVerifierV2Thresholds:
    path = calibration_path or (get_repo_root() / _CALIBRATION_DEFAULT)
    if not path.exists():
        return SemanticVerifierV2Thresholds(
            support_max_error_probability=0.125,
            violation_min_error_probability=0.45,
            min_monitorability_confidence=min_monitorability_confidence,
            source_run_date=RUN_DATE,
            calibration_rows=0,
        )

    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    bucket_scores: dict[str, list[float]] = {
        "true_positive": [],
        "false_positive": [],
        "true_negative": [],
    }
    run_date = RUN_DATE
    for row in rows:
        labels = row.get("labels", {})
        calibration = row.get("calibration", {})
        provenance = row.get("provenance", {})
        if isinstance(provenance, dict):
            run_date = str(provenance.get("calibration_artifact_run_date", run_date))
        bucket = labels.get("outcome_bucket")
        score = calibration.get("score")
        if isinstance(bucket, str) and isinstance(score, int | float) and bucket in bucket_scores:
            bucket_scores[bucket].append(float(score))

    true_negative = sorted(bucket_scores["true_negative"])
    false_positive = sorted(bucket_scores["false_positive"])
    true_positive = sorted(bucket_scores["true_positive"])
    support_max = (
        _round_score((_percentile(true_negative, 0.9) + _percentile(false_positive, 0.1)) / 2.0)
        if true_negative and false_positive
        else 0.125
    )
    violation_min = (
        _round_score((max(false_positive) + _percentile(true_positive, 0.5)) / 2.0)
        if false_positive and true_positive
        else 0.45
    )
    return SemanticVerifierV2Thresholds(
        support_max_error_probability=support_max,
        violation_min_error_probability=max(violation_min, support_max + 0.05),
        min_monitorability_confidence=min_monitorability_confidence,
        source_run_date=run_date,
        calibration_rows=len(rows),
    )


def _expand_claims(
    claims: list[SemanticClaim],
    typed_reasoning: TypedReasoningIR | None,
) -> list[SemanticClaim]:
    if typed_reasoning is None or not typed_reasoning.reasoning_steps:
        return claims

    seen = {claim.text.strip(): claim for claim in claims if claim.text.strip()}
    expanded = list(claims)
    for step in typed_reasoning.reasoning_steps:
        text = step.text.strip()
        if not text or text in seen:
            continue
        step_claim = SemanticClaim(
            claim_id=step.step_id,
            text=text,
            keywords=_extract_keywords(text),
            quantities=_extract_quantities(text),
        )
        seen[text] = step_claim
        expanded.append(
            SemanticClaim(
                claim_id=step.step_id,
                text=text,
                keywords=list(step_claim.keywords),
                quantities=list(step_claim.quantities),
                is_final=False,
                normalized_value=None,
            )
        )
    return expanded


def _response_mode(typed_reasoning: TypedReasoningIR | None, response: str) -> str:
    if typed_reasoning is not None:
        return typed_reasoning.provenance.extraction_method
    if response.strip():
        return "fallback_text"
    return "empty"


def _matched_clause_ids(required_clauses: list[PromptClause], claim: SemanticClaim) -> list[str]:
    return [clause.clause_id for clause in required_clauses if _claim_covers_clause(clause, claim)]


def _target_coverage(question_profile: QuestionProfile, claim: SemanticClaim) -> float:
    target_keywords = [
        keyword for keyword in question_profile.target_keywords if keyword not in {"many", "much"}
    ]
    if not target_keywords:
        return 1.0
    matched = sum(1 for keyword in target_keywords if keyword in claim.keywords)
    return _ratio(matched, len(target_keywords))


def _missing_target_keywords(
    question_profile: QuestionProfile,
    claim: SemanticClaim,
) -> list[str]:
    return [
        keyword
        for keyword in question_profile.target_keywords
        if keyword not in claim.keywords and keyword not in {"many", "much"}
    ]


def _select_focus_claim(
    *,
    claims: list[SemanticClaim],
    question_profile: QuestionProfile,
    clause_matches: dict[str, list[str]],
) -> SemanticClaim:
    return max(
        claims,
        key=lambda claim: (
            _target_coverage(question_profile, claim),
            _ratio(
                len(clause_matches[claim.claim_id]),
                max(len(question_profile.prompt_clauses), 1),
            ),
            1.0 if claim.is_final else 0.0,
            1.0 if claim.normalized_value is not None else 0.0,
            len(claim.quantities),
        ),
    )


def _monitorability_confidence(
    *,
    target_coverage: float,
    premise_support: float,
    claim: SemanticClaim,
    response_mode: str,
    recommended_response_mode: str | None,
) -> float:
    confidence = max(
        target_coverage,
        premise_support,
        0.25 if claim.normalized_value is not None else 0.0,
    )
    if response_mode == "direct_json":
        confidence += 0.1 if recommended_response_mode in _JSON_MODES else 0.05
    elif recommended_response_mode in _JSON_MODES:
        confidence -= 0.1
    return _clamp(confidence)


def _raw_error_probability(
    *,
    claim: SemanticClaim,
    focus_claim: SemanticClaim,
    target_coverage: float,
    premise_support: float,
    legacy_violation_types: list[str],
    response_mode: str,
    recommended_response_mode: str | None,
    question_profile: QuestionProfile,
) -> float:
    raw = 0.0
    missing_target_keywords = _missing_target_keywords(question_profile, claim)
    if (
        claim.claim_id == focus_claim.claim_id
        and "answer_target_mismatch" in legacy_violation_types
        and (target_coverage < 1.0 or missing_target_keywords)
    ):
        raw += 0.55
    if claim.claim_id == focus_claim.claim_id and "unsupported_reference" in legacy_violation_types:
        raw += 0.25
    if (
        "missing_quantity_coverage" in legacy_violation_types
        or "missing_entity_coverage" in legacy_violation_types
    ):
        raw += 0.2 * (1.0 - premise_support)
    raw += 0.2 * (1.0 - target_coverage)
    raw += 0.15 * (1.0 - premise_support)
    if claim.claim_id == focus_claim.claim_id and missing_target_keywords:
        raw += 0.1
    if recommended_response_mode in _JSON_MODES and response_mode != "direct_json":
        raw += 0.05
    return _clamp(raw)


def _calibrate_probability(raw_error: float, monitorability_confidence: float) -> float:
    return _round_score(
        _clamp(raw_error * monitorability_confidence + 0.18 * (1.0 - monitorability_confidence))
    )


def _claim_status(
    *,
    semantic_error_probability: float,
    monitorability_confidence: float,
    thresholds: SemanticVerifierV2Thresholds,
) -> str:
    if monitorability_confidence < thresholds.min_monitorability_confidence:
        return "abstain"
    if semantic_error_probability >= thresholds.violation_min_error_probability:
        return "violated"
    if semantic_error_probability <= thresholds.support_max_error_probability:
        return "supported"
    return "abstain"


def _supporting_claim_ids(
    *,
    claim: SemanticClaim,
    claims: list[SemanticClaim],
    clause_matches: dict[str, list[str]],
    required_clauses: list[PromptClause],
) -> list[str]:
    missing = {
        clause.clause_id
        for clause in required_clauses
        if clause.clause_id not in clause_matches[claim.claim_id]
    }
    if not missing:
        return []
    supporting: list[str] = []
    for other in claims:
        if other.claim_id == claim.claim_id:
            continue
        if missing & set(clause_matches[other.claim_id]):
            supporting.append(other.claim_id)
    return supporting


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = round((len(values) - 1) * fraction)
    return values[index]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return _round_score(numerator / denominator)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round_score(value: float) -> float:
    return round(value, 6)


__all__ = [
    "RUN_DATE",
    "SemanticClaimResult",
    "SemanticVerifierV2",
    "SemanticVerifierV2Result",
    "SemanticVerifierV2Thresholds",
    "verify_semantic_verifier_v2",
]
