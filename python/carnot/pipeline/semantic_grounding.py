"""Semantic grounding checks for question-target alignment.

Spec: REQ-VERIFY-020, REQ-VERIFY-021,
SCENARIO-VERIFY-020, SCENARIO-VERIFY-021
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from carnot.pipeline.extract import ConstraintResult

if TYPE_CHECKING:
    from carnot.pipeline.typed_reasoning import TypedReasoningIR

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:,|;|\bbut\b)\s*", re.IGNORECASE)
_CODE_LIKE_RE = re.compile(r"(```|`def\s+\w+|def\s+\w+\s*\(|->|class\s+\w+)")
_ASSUMPTION_MARKERS = (
    "assume",
    "assuming",
    "scenario",
    "interpret",
    "ambiguous",
    "implies",
)
_WORD_NUMBERS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "half": "0.5",
    "quarter": "0.25",
    "twice": "2",
    "double": "2",
    "triple": "3",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "give",
    "has",
    "have",
    "he",
    "her",
    "here",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "more",
    "of",
    "on",
    "or",
    "she",
    "so",
    "than",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "what",
    "when",
    "while",
    "with",
}
_TARGET_GENERIC_KEYWORDS = {
    "many",
    "much",
    "long",
    "total",
    "altogether",
    "number",
    "amount",
    "value",
}
_ADJUSTMENT_KEYWORDS = {
    "half",
    "quarter",
    "discount",
    "shipping",
    "fee",
    "free",
    "display",
    "promotion",
    "service",
    "late",
    "after",
    "before",
    "back",
    "home",
    "blueberry",
    "mint",
    "red",
    "green",
}
_UNIT_KEYWORDS = {
    "day",
    "days",
    "hour",
    "hours",
    "minute",
    "minutes",
    "month",
    "months",
    "mile",
    "miles",
    "mph",
    "liter",
    "liters",
    "milliliter",
    "milliliters",
    "gram",
    "grams",
    "kilogram",
    "kilograms",
}


@dataclass
class PromptClause:
    """One prompt-side clause that may need grounding in the response."""

    clause_id: str
    text: str
    keywords: list[str]
    quantities: list[str]
    role: str
    focus_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "clause_id": self.clause_id,
            "text": self.text,
            "keywords": list(self.keywords),
            "quantities": list(self.quantities),
            "role": self.role,
            "focus_keywords": list(self.focus_keywords),
        }


@dataclass
class SemanticClaim:
    """One atomic claim extracted from the response."""

    claim_id: str
    text: str
    keywords: list[str]
    quantities: list[str]
    is_final: bool = False
    normalized_value: object | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "keywords": list(self.keywords),
            "quantities": list(self.quantities),
            "is_final": self.is_final,
            "normalized_value": self.normalized_value,
        }


@dataclass
class SemanticGroundingViolation:
    """Structured semantic violation convertible into pipeline constraints."""

    violation_type: str
    description: str
    claim_id: str | None = None
    clause_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_constraint_result(self) -> ConstraintResult:
        return ConstraintResult(
            constraint_type="semantic_grounding",
            description=f"[{self.violation_type}] {self.description}",
            metadata={
                "satisfied": False,
                "violation_type": self.violation_type,
                "claim_id": self.claim_id,
                "clause_id": self.clause_id,
                **self.metadata,
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "violation_type": self.violation_type,
            "description": self.description,
            "claim_id": self.claim_id,
            "clause_id": self.clause_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class QuestionProfile:
    """Prompt-side grounding targets extracted deterministically."""

    question: str
    prompt_clauses: list[PromptClause]
    target_clause: PromptClause
    target_keywords: list[str]
    target_cues: list[str]
    is_code_like: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "prompt_clauses": [clause.to_dict() for clause in self.prompt_clauses],
            "target_clause": self.target_clause.to_dict(),
            "target_keywords": list(self.target_keywords),
            "target_cues": list(self.target_cues),
            "is_code_like": self.is_code_like,
        }


@dataclass
class SemanticGroundingResult:
    """Full semantic-grounding analysis for one question-response pair."""

    question_profile: QuestionProfile
    claims: list[SemanticClaim]
    violations: list[SemanticGroundingViolation]
    refinement_applied: bool = False

    @property
    def verified(self) -> bool:
        return len(self.violations) == 0

    def to_constraint_results(self) -> list[ConstraintResult]:
        return [violation.to_constraint_result() for violation in self.violations]

    def to_refinement_payload(self) -> dict[str, object]:
        return {
            "question": self.question_profile.to_dict(),
            "prompt_clauses": [clause.to_dict() for clause in self.question_profile.prompt_clauses],
            "claims": [claim.to_dict() for claim in self.claims],
            "violations": [violation.to_dict() for violation in self.violations],
        }


class SemanticGroundingRefiner(Protocol):
    """Optional hook for ambiguous semantic-grounding cases."""

    def __call__(
        self,
        payload: dict[str, object],
        violations: list[SemanticGroundingViolation],
    ) -> list[SemanticGroundingViolation]: ...


class SemanticGroundingVerifier:
    """Deterministic semantic verifier with an optional refinement hook."""

    def __init__(self, refiner: SemanticGroundingRefiner | None = None) -> None:
        self._refiner = refiner

    def verify(
        self,
        question: str,
        response: str,
        typed_reasoning: TypedReasoningIR | None = None,
    ) -> SemanticGroundingResult:
        profile = _build_question_profile(question)
        if profile.is_code_like:
            return SemanticGroundingResult(
                question_profile=profile,
                claims=[],
                violations=[],
            )

        claims = _extract_claims(response, typed_reasoning)
        violations = _deterministic_violations(profile, claims)
        refinement_applied = False

        if self._refiner is not None and any(
            violation.metadata.get("needs_refinement") for violation in violations
        ):
            violations = list(
                self._refiner(
                    {
                        "question": profile.to_dict(),
                        "prompt_clauses": [clause.to_dict() for clause in profile.prompt_clauses],
                        "claims": [claim.to_dict() for claim in claims],
                    },
                    violations,
                )
            )
            refinement_applied = True

        return SemanticGroundingResult(
            question_profile=profile,
            claims=claims,
            violations=_dedupe_violations(violations),
            refinement_applied=refinement_applied,
        )


def verify_semantic_grounding(
    question: str,
    response: str,
    typed_reasoning: TypedReasoningIR | None = None,
    refiner: SemanticGroundingRefiner | None = None,
) -> SemanticGroundingResult:
    """Convenience helper for one-shot semantic-grounding verification."""
    return SemanticGroundingVerifier(refiner=refiner).verify(
        question=question,
        response=response,
        typed_reasoning=typed_reasoning,
    )


def _build_question_profile(question: str) -> QuestionProfile:
    cleaned = " ".join(question.strip().split())
    if _looks_code_like(cleaned):
        return QuestionProfile(
            question=question,
            prompt_clauses=[],
            target_clause=PromptClause("target", cleaned, [], [], "target"),
            target_keywords=[],
            target_cues=[],
            is_code_like=True,
        )

    sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(cleaned) if sentence.strip()]
    target_text = _target_text(sentences, cleaned)
    target_prefix = ""
    target_match = re.search(
        r"\b(?:by how many|how many|how much|how long|what is|what's)\b",
        target_text,
        flags=re.IGNORECASE,
    )
    if target_match is not None and target_match.start() > 0:
        target_prefix = target_text[: target_match.start()].strip(" ,")
        target_text = target_text[target_match.start() :].strip()
    target_clause = PromptClause(
        clause_id="target",
        text=target_text,
        keywords=_extract_keywords(target_text),
        quantities=_extract_quantities(target_text),
        role="target",
    )

    prompt_clauses: list[PromptClause] = []
    clause_index = 1
    if target_prefix:
        for part in _CLAUSE_SPLIT_RE.split(target_prefix):
            clause_text = part.strip()
            if not clause_text:
                continue
            prompt_clauses.append(
                PromptClause(
                    clause_id=f"p{clause_index}",
                    text=clause_text,
                    keywords=_extract_keywords(clause_text),
                    quantities=_extract_quantities(clause_text),
                    role="premise",
                )
            )
            clause_index += 1
    for sentence in sentences:
        if sentence == _target_text(sentences, cleaned):
            continue
        for part in _CLAUSE_SPLIT_RE.split(sentence):
            clause_text = part.strip()
            if not clause_text:
                continue
            prompt_clauses.append(
                PromptClause(
                    clause_id=f"p{clause_index}",
                    text=clause_text,
                    keywords=_extract_keywords(clause_text),
                    quantities=_extract_quantities(clause_text),
                    role="premise",
                )
            )
            clause_index += 1

    keyword_counts = Counter(
        keyword
        for clause in [*prompt_clauses, target_clause]
        for keyword in clause.keywords
        if keyword not in _TARGET_GENERIC_KEYWORDS
    )
    for clause in [*prompt_clauses, target_clause]:
        unique_keywords = [
            keyword
            for keyword in clause.keywords
            if keyword_counts[keyword] == 1 and keyword not in _TARGET_GENERIC_KEYWORDS
        ]
        clause.focus_keywords = unique_keywords or [
            keyword for keyword in clause.keywords if keyword not in _TARGET_GENERIC_KEYWORDS
        ]

    return QuestionProfile(
        question=question,
        prompt_clauses=prompt_clauses,
        target_clause=target_clause,
        target_keywords=target_clause.focus_keywords or target_clause.keywords,
        target_cues=_target_cues(target_text),
    )


def _extract_claims(
    response: str,
    typed_reasoning: TypedReasoningIR | None,
) -> list[SemanticClaim]:
    claims: list[SemanticClaim] = []
    if typed_reasoning is not None and typed_reasoning.atomic_claims:
        for claim in typed_reasoning.atomic_claims:
            text = claim.text.strip()
            if not text:
                continue
            claims.append(
                SemanticClaim(
                    claim_id=claim.claim_id,
                    text=text,
                    keywords=_extract_keywords(text),
                    quantities=_extract_quantities(f"{text} {claim.value or ''}"),
                    normalized_value=claim.value,
                )
            )
        if typed_reasoning.final_answer is not None:
            answer_text = typed_reasoning.final_answer.text.strip() or str(
                typed_reasoning.final_answer.normalized
            )
            claims.append(
                SemanticClaim(
                    claim_id="final_answer",
                    text=answer_text,
                    keywords=_extract_keywords(answer_text),
                    quantities=_extract_quantities(
                        f"{answer_text} {typed_reasoning.final_answer.normalized or ''}"
                    ),
                    is_final=True,
                    normalized_value=typed_reasoning.final_answer.normalized,
                )
            )
    else:
        fragments = [
            _clean_fragment(fragment)
            for fragment in _SENTENCE_RE.split(response.strip())
            if _clean_fragment(fragment)
        ]
        if not fragments and response.strip():
            fragments = [_clean_fragment(response)]
        for index, fragment in enumerate(fragments, start=1):
            claims.append(
                SemanticClaim(
                    claim_id=f"cl{index}",
                    text=fragment,
                    keywords=_extract_keywords(fragment),
                    quantities=_extract_quantities(fragment),
                    normalized_value=_final_numeric_value(fragment),
                )
            )

    if claims:
        final_index = next(
            (
                index
                for index, claim in enumerate(claims)
                if re.search(r"^\s*(?:answer|final)\b", claim.text, flags=re.IGNORECASE)
            ),
            len(claims) - 1,
        )
        claims[final_index].is_final = True

    return claims


def _deterministic_violations(
    profile: QuestionProfile,
    claims: list[SemanticClaim],
) -> list[SemanticGroundingViolation]:
    if not claims or not _question_is_groundable(profile):
        return []

    uncovered_clauses = [
        clause
        for clause in profile.prompt_clauses
        if _clause_requires_grounding(clause, profile)
        and not any(_claim_covers_clause(clause, claim) for claim in claims)
    ]

    violations = [_missing_clause_violation(profile, clause) for clause in uncovered_clauses]

    target_violation = _answer_target_violation(profile, claims, uncovered_clauses)
    if target_violation is not None:
        violations.append(target_violation)

    unsupported_violation = _unsupported_reference_violation(profile, claims, uncovered_clauses)
    if unsupported_violation is not None:
        violations.append(unsupported_violation)

    return violations


def _claim_covers_clause(clause: PromptClause, claim: SemanticClaim) -> bool:
    if set(clause.quantities) & set(claim.quantities):
        return True

    focus_keywords = set(clause.focus_keywords or clause.keywords)
    return bool(focus_keywords & set(claim.keywords))


def _missing_clause_violation(
    profile: QuestionProfile,
    clause: PromptClause,
) -> SemanticGroundingViolation:
    violation_type = "missing_quantity_coverage" if clause.quantities else "missing_entity_coverage"
    return SemanticGroundingViolation(
        violation_type=violation_type,
        description=f"Prompt clause is not grounded in the response: '{clause.text}'",
        clause_id=clause.clause_id,
        metadata={
            "clause_text": clause.text,
            "target_keywords": list(profile.target_keywords),
            "taxonomy_hint": _taxonomy_hint(profile, clause, violation_type),
            "focus_keywords": list(clause.focus_keywords),
            "quantities": list(clause.quantities),
        },
    )


def _answer_target_violation(
    profile: QuestionProfile,
    claims: list[SemanticClaim],
    uncovered_clauses: list[PromptClause],
) -> SemanticGroundingViolation | None:
    if not profile.target_keywords:
        return None

    support_claims = _support_claims(claims)
    support_keywords = {keyword for claim in support_claims for keyword in claim.keywords}
    support_text = " ".join(claim.text.lower() for claim in support_claims)

    missing_target_keywords = [
        keyword
        for keyword in profile.target_keywords
        if keyword not in support_keywords and keyword not in _TARGET_GENERIC_KEYWORDS
    ]
    requires_difference = "difference" in profile.target_cues
    requires_event_specific = "event_specific" in profile.target_cues

    if requires_difference and not _has_difference_signal(support_text):
        return SemanticGroundingViolation(
            violation_type="answer_target_mismatch",
            description=(
                "The response does not compute the requested comparison or net quantity "
                f"for target '{profile.target_clause.text}'."
            ),
            claim_id=support_claims[-1].claim_id if support_claims else None,
            metadata={
                "target_keywords": list(profile.target_keywords),
                "taxonomy_hint": "question_grounding_failures",
            },
        )

    if missing_target_keywords and (requires_event_specific or uncovered_clauses):
        return SemanticGroundingViolation(
            violation_type="answer_target_mismatch",
            description=(
                "The response answers a related quantity but misses the requested target "
                f"keywords: {', '.join(missing_target_keywords)}."
            ),
            claim_id=support_claims[-1].claim_id if support_claims else None,
            metadata={
                "missing_target_keywords": missing_target_keywords,
                "target_keywords": list(profile.target_keywords),
                "taxonomy_hint": "question_grounding_failures",
            },
        )

    return None


def _unsupported_reference_violation(
    profile: QuestionProfile,
    claims: list[SemanticClaim],
    uncovered_clauses: list[PromptClause],
) -> SemanticGroundingViolation | None:
    prompt_quantities = {
        quantity for clause in profile.prompt_clauses for quantity in clause.quantities
    }

    for claim in claims:
        lowered = claim.text.lower()
        markers = [marker for marker in _ASSUMPTION_MARKERS if marker in lowered]
        if not markers:
            continue
        introduced_quantities = [
            quantity for quantity in claim.quantities if quantity not in prompt_quantities
        ]
        if not uncovered_clauses and not introduced_quantities:
            continue
        return SemanticGroundingViolation(
            violation_type="unsupported_reference",
            description=f"Response introduces an unsupported assumption: '{claim.text}'",
            claim_id=claim.claim_id,
            metadata={
                "unsupported_markers": markers,
                "introduced_quantities": introduced_quantities,
                "needs_refinement": True,
                "taxonomy_hint": _taxonomy_hint(
                    profile,
                    uncovered_clauses[0] if uncovered_clauses else profile.target_clause,
                    "unsupported_reference",
                ),
            },
        )

    return None


def _support_claims(claims: list[SemanticClaim]) -> list[SemanticClaim]:
    final_index = next(
        (index for index, claim in enumerate(claims) if claim.is_final),
        len(claims) - 1,
    )
    support = [claims[final_index]]
    if final_index > 0 and _looks_like_bare_answer(claims[final_index].text):
        support.insert(0, claims[final_index - 1])
    return support


def _taxonomy_hint(
    profile: QuestionProfile,
    clause: PromptClause,
    violation_type: str,
) -> str:
    clause_keywords = set(clause.focus_keywords or clause.keywords)
    lower_question = profile.question.lower()

    if violation_type == "answer_target_mismatch":
        return "question_grounding_failures"

    if violation_type == "unsupported_reference":
        if _is_nested_binding_case(lower_question):
            return "entity_quantity_binding_errors"
        return "question_grounding_failures"

    if clause_keywords & _ADJUSTMENT_KEYWORDS:
        return "omitted_premises"

    if clause_keywords & _UNIT_KEYWORDS or set(profile.target_cues) & {"aggregation"}:
        return "unit_aggregation_errors"

    if _is_nested_binding_case(lower_question):
        return "entity_quantity_binding_errors"

    if clause_keywords & set(profile.target_keywords):
        return "question_grounding_failures"

    return "omitted_premises"


def _target_cues(target_text: str) -> list[str]:
    lowered = target_text.lower()
    cues: list[str] = []
    if any(
        _contains_phrase(lowered, phrase)
        for phrase in ("profit", "left", "remain", "remaining", "out")
    ):
        cues.append("difference")
    if any(
        _contains_phrase(lowered, phrase)
        for phrase in ("fewer", "more", "longer", "shorter", "by how many")
    ):
        cues.append("difference")
    if any(_contains_phrase(lowered, phrase) for phrase in ("now", "currently")):
        cues.append("final_state")
    if any(
        _contains_phrase(lowered, phrase)
        for phrase in ("total", "altogether", "in total", "across")
    ):
        cues.append("aggregation")
    if _contains_phrase(lowered, "checked in during") or _contains_phrase(lowered, "still have"):
        cues.append("event_specific")
    if any(_contains_phrase(lowered, phrase) for phrase in ("each", "per")):
        cues.append("per_entity")
    return cues


def _target_text(sentences: list[str], fallback: str) -> str:
    for sentence in reversed(sentences):
        if "?" in sentence:
            return sentence
    return sentences[-1] if sentences else fallback


def _extract_keywords(text: str) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(text.lower()):
        raw_token = match.group(0)
        if raw_token in _STOPWORDS:
            continue
        token = _singularize(raw_token)
        if token in _STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            keywords.append(token)
    return keywords


def _extract_quantities(text: str) -> list[str]:
    quantities: list[str] = []
    seen: set[str] = set()

    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        normalized = token.rstrip("%")
        if normalized not in seen:
            seen.add(normalized)
            quantities.append(normalized)
        if token.endswith("%"):
            percent = str(float(normalized) / 100.0).rstrip("0").rstrip(".")
            if percent and percent not in seen:
                seen.add(percent)
                quantities.append(percent)

    for token in _TOKEN_RE.findall(text.lower()):
        word_normalized = _WORD_NUMBERS.get(token)
        if word_normalized is not None and word_normalized not in seen:
            seen.add(word_normalized)
            quantities.append(word_normalized)

    return quantities


def _has_difference_signal(text: str) -> bool:
    return any(
        token in text
        for token in (
            "-",
            "minus",
            "difference",
            "profit",
            "left",
            "remain",
            "remaining",
            "fewer",
            "more than",
            "less than",
            "longer",
            "shorter",
        )
    )


def _looks_like_bare_answer(text: str) -> bool:
    cleaned = text.strip().lower()
    return bool(re.fullmatch(r"(?:answer|final)\s*:\s*[-$]?\d+(?:\.\d+)?", cleaned))


def _looks_code_like(question: str) -> bool:
    return bool(_CODE_LIKE_RE.search(question))


def _is_nested_binding_case(text: str) -> bool:
    lowered = text.lower()
    return lowered.count(" each ") >= 1 and any(
        phrase in lowered for phrase in ("how many", "how much", "cost", "profit", "revenue")
    )


def _question_is_groundable(profile: QuestionProfile) -> bool:
    if profile.is_code_like:
        return False
    if any(clause.quantities for clause in profile.prompt_clauses):
        return True
    return bool(
        set(profile.target_cues) & {"difference", "aggregation", "event_specific", "per_entity"}
    )


def _clause_requires_grounding(profile_clause: PromptClause, profile: QuestionProfile) -> bool:
    clause_keywords = set(profile_clause.focus_keywords or profile_clause.keywords)
    if profile_clause.quantities:
        return True
    if clause_keywords & _ADJUSTMENT_KEYWORDS:
        return True
    if clause_keywords & _UNIT_KEYWORDS:
        return True
    return bool(clause_keywords & set(profile.target_keywords))


def _dedupe_violations(
    violations: list[SemanticGroundingViolation],
) -> list[SemanticGroundingViolation]:
    deduped: list[SemanticGroundingViolation] = []
    seen: set[tuple[str, str | None, str | None, str]] = set()
    for violation in violations:
        key = (
            violation.violation_type,
            violation.claim_id,
            violation.clause_id,
            violation.description,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(violation)
    return deduped


def _clean_fragment(text: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text.strip())
    cleaned = re.sub(r"[*_`$]", "", cleaned)
    return " ".join(cleaned.split())


def _final_numeric_value(text: str) -> str | None:
    values = _extract_quantities(text)
    return values[-1] if values else None


def _singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("shes", "ches", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
    return re.search(pattern, text) is not None


__all__ = [
    "PromptClause",
    "QuestionProfile",
    "SemanticClaim",
    "SemanticGroundingResult",
    "SemanticGroundingVerifier",
    "SemanticGroundingViolation",
    "verify_semantic_grounding",
]
