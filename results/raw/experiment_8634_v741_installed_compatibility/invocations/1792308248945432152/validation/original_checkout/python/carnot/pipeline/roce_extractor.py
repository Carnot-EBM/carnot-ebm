"""Reasoning-time open constraint elicitation for prompt-derived checks.

**Researcher summary:**
    ROCE extracts open-world acceptance criteria from a prompt before answer
    generation. This module converts those criteria into `ConstraintResult`
    objects with explicit formal predicates that downstream verifiers can
    evaluate against a candidate response.

**Detailed explanation for engineers:**
    The older `constraint_extractor.py` module focuses on dynamic response
    checkers. This module focuses on the shared extraction pipeline contract in
    `extract.py`: `ROCEExtractor.extract()` returns `ConstraintResult`
    instances whose metadata is already formalized as predicate + arguments.

Spec: REQ-EXTRACT-1763, SCENARIO-EXTRACT-1763, SCENARIO-EXTRACT-1764
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from carnot.pipeline.extract import ConstraintResult

_SPEC_REFS = ["REQ-EXTRACT-1763", "SCENARIO-EXTRACT-1763", "SCENARIO-EXTRACT-1764"]

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass(frozen=True)
class ROCEConstraint:
    """Formal prompt constraint before conversion to `ConstraintResult`.

    Attributes:
        kind: Broad verifier category, used in `ConstraintResult.constraint_type`.
        predicate: Formal predicate name, such as `required_text`.
        arguments: Predicate arguments with JSON-serializable values.
        description: Human-readable constraint summary.
        raw_phrase: Exact prompt phrase or span that triggered the constraint.
        confidence: Deterministic extractor confidence. Rule-based matches use 1.0.
        spec_refs: OpenSpec references that justify this output schema.

    Spec: REQ-EXTRACT-1763-2
    """

    kind: str
    predicate: str
    arguments: dict[str, Any]
    description: str
    raw_phrase: str
    confidence: float = 1.0
    spec_refs: list[str] = field(default_factory=lambda: list(_SPEC_REFS))

    def to_constraint_result(self) -> ConstraintResult:
        """Convert this formal ROCE constraint to the pipeline result type."""
        return ConstraintResult(
            constraint_type=f"roce_{self.kind}",
            description=self.description,
            metadata={
                "source": "roce",
                "predicate": self.predicate,
                "arguments": dict(self.arguments),
                "raw_phrase": self.raw_phrase,
                "confidence": self.confidence,
                "spec_refs": list(self.spec_refs),
            },
        )


class ROCEExtractor:
    """Extract open-world prompt constraints as formal verification metadata.

    Spec: REQ-EXTRACT-1763-1, REQ-EXTRACT-1763-3, REQ-EXTRACT-1763-5
    """

    @property
    def supported_domains(self) -> list[str]:
        return ["roce", "prompt", "open_world"]

    def extract(self, text: str, domain: str | None = None) -> list[ConstraintResult]:
        """Extract ROCE constraints from unstructured prompt text."""
        if domain is not None and domain not in self.supported_domains:
            return []

        constraints = self.extract_formal(text)
        return [constraint.to_constraint_result() for constraint in constraints]

    def extract_formal(self, text: str) -> list[ROCEConstraint]:
        """Return deduplicated formal constraints before result conversion."""
        if not text.strip():
            return []

        constraints: list[ROCEConstraint] = []
        constraints.extend(self._extract_required_text(text))
        constraints.extend(self._extract_forbidden_text(text))
        constraints.extend(self._extract_json_constraints(text))
        constraints.extend(self._extract_list_constraints(text))
        constraints.extend(self._extract_word_bounds(text))
        constraints.extend(self._extract_numeric_ranges(text))
        constraints.extend(self._extract_boundary_text(text))
        constraints.extend(self._extract_output_shape(text))
        constraints.extend(self._extract_arithmetic_equalities(text))
        constraints.extend(self._extract_conditional_guards(text))
        return _dedupe_constraints(constraints)

    def _extract_required_text(self, text: str) -> list[ROCEConstraint]:
        patterns = [
            r"(?:must|should|required to)\s+(?:include|contain|mention|use)\s+"
            r"(?:the\s+)?(?:word|phrase|term)?\s*['\"]([^'\"]+)['\"]",
            r"(?:must|should|required to)\s+(?:include|contain|mention|use)\s+"
            r"the\s+(?:word|phrase|term)\s+([A-Za-z0-9_:-]+)",
            r"(?:^|[.!?]\s+)(?:please\s+)?(?:include|contain|mention|use)\s+"
            r"(?:the\s+)?(?:word|phrase|term)?\s*['\"]([^'\"]+)['\"]",
        ]
        results: list[ROCEConstraint] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                term = match.group(1).strip()
                results.append(
                    ROCEConstraint(
                        kind="content",
                        predicate="required_text",
                        arguments={"term": term},
                        description=f"Response must contain '{term}'",
                        raw_phrase=match.group(0),
                    )
                )
        return results

    def _extract_forbidden_text(self, text: str) -> list[ROCEConstraint]:
        patterns = [
            r"(?:do\s+not|don't|never)\s+(?:include|contain|mention|use|say)\s+"
            r"(?:the\s+)?(?:word|phrase|term)?\s*['\"]([^'\"]+)['\"]",
            r"(?:do\s+not|don't|never)\s+(?:include|contain|mention|use|say)\s+"
            r"the\s+(?:word|phrase|term)\s+([A-Za-z0-9_:-]+)",
            r"(?:avoid|exclude|without)\s+(?:using|mentioning|including)?\s*"
            r"['\"]([^'\"]+)['\"]",
        ]
        results: list[ROCEConstraint] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                term = match.group(1).strip()
                results.append(
                    ROCEConstraint(
                        kind="content",
                        predicate="forbidden_text",
                        arguments={"term": term},
                        description=f"Response must not contain '{term}'",
                        raw_phrase=match.group(0),
                    )
                )
        return results

    def _extract_json_constraints(self, text: str) -> list[ROCEConstraint]:
        results: list[ROCEConstraint] = []
        json_match = re.search(r"\bJSON\b(?:\s+(?:object|array|response|answer))?", text)
        if json_match:
            results.append(
                ROCEConstraint(
                    kind="format",
                    predicate="format_json",
                    arguments={},
                    description="Response must be valid JSON",
                    raw_phrase=json_match.group(0),
                )
            )

        for match in re.finditer(
            r"(strict\s+key\s+order\s+\{.+?\})(?:\s+and\s+no\s+other\s+top-level\s+keys)?",
            text,
            re.IGNORECASE,
        ):
            keys = re.findall(r'"([^"]+)"\s*:', match.group(1))
            if keys:
                raw = match.group(0)
                results.append(
                    ROCEConstraint(
                        kind="schema",
                        predicate="json_required_keys",
                        arguments={
                            "keys": keys,
                            "ordered": True,
                            "no_extra_keys": "no other top-level keys" in text.lower(),
                        },
                        description=f"JSON response must contain keys {keys}",
                        raw_phrase=raw,
                    )
                )

        for match in re.finditer(r"(?:with|using)\s+keys?\s+(.+?)(?:[.;]|$)", text, re.IGNORECASE):
            keys = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
            if keys:
                results.append(
                    ROCEConstraint(
                        kind="schema",
                        predicate="json_required_keys",
                        arguments={"keys": keys, "ordered": False, "no_extra_keys": False},
                        description=f"JSON response must contain keys {keys}",
                        raw_phrase=match.group(0),
                    )
                )
        return results

    def _extract_list_constraints(self, text: str) -> list[ROCEConstraint]:
        results: list[ROCEConstraint] = []
        list_patterns = [
            ("bullet", r"bullet(?:ed)?\s+(?:points?|list)"),
            ("numbered", r"numbered\s+list"),
        ]
        for style, pattern in list_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results.append(
                    ROCEConstraint(
                        kind="format",
                        predicate="format_list",
                        arguments={"style": style},
                        description=f"Response must use a {style} list",
                        raw_phrase=match.group(0),
                    )
                )

        count_word = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        for match in re.finditer(
            rf"(?:give|list|provide|use)\s+(?:exactly\s+)?{count_word}\s+"
            r"(?:items?|entries|examples|reasons|bullet\s+points?|points?)",
            text,
            re.IGNORECASE,
        ):
            count = _number_from_token(match.group(1))
            results.append(
                ROCEConstraint(
                    kind="cardinality",
                    predicate="exact_item_count",
                    arguments={"count": count},
                    description=f"Response must contain exactly {count} items",
                    raw_phrase=match.group(0),
                )
            )
        for match in re.finditer(
            rf"exactly\s+{count_word}\s+"
            r"(?:items?|entries|examples|reasons|bullet\s+points?|points?)",
            text,
            re.IGNORECASE,
        ):
            count = _number_from_token(match.group(1))
            results.append(
                ROCEConstraint(
                    kind="cardinality",
                    predicate="exact_item_count",
                    arguments={"count": count},
                    description=f"Response must contain exactly {count} items",
                    raw_phrase=match.group(0),
                )
            )
        return results

    def _extract_word_bounds(self, text: str) -> list[ROCEConstraint]:
        results: list[ROCEConstraint] = []
        max_patterns = [
            r"(?:under|at\s+most|no\s+more\s+than|fewer\s+than|less\s+than)\s+(\d+)\s+words?",
            r"(\d+)\s+words?\s+or\s+fewer",
        ]
        min_patterns = [
            r"(?:at\s+least|minimum|no\s+fewer\s+than)\s+(\d+)\s+words?",
            r"(\d+)\s+words?\s+or\s+more",
        ]
        for pattern in max_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                limit = int(match.group(1))
                results.append(
                    ROCEConstraint(
                        kind="length",
                        predicate="word_count_at_most",
                        arguments={"limit": limit},
                        description=f"Response must be at most {limit} words",
                        raw_phrase=match.group(0),
                    )
                )
        for pattern in min_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                limit = int(match.group(1))
                results.append(
                    ROCEConstraint(
                        kind="length",
                        predicate="word_count_at_least",
                        arguments={"limit": limit},
                        description=f"Response must be at least {limit} words",
                        raw_phrase=match.group(0),
                    )
                )
        return results

    def _extract_numeric_ranges(self, text: str) -> list[ROCEConstraint]:
        results: list[ROCEConstraint] = []
        for match in re.finditer(
            r"between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        ):
            low = float(match.group(1))
            high = float(match.group(2))
            results.append(
                ROCEConstraint(
                    kind="numeric",
                    predicate="numeric_range",
                    arguments={"low": low, "high": high},
                    description=f"Numeric response value must be between {low} and {high}",
                    raw_phrase=match.group(0),
                )
            )
        return results

    def _extract_boundary_text(self, text: str) -> list[ROCEConstraint]:
        results: list[ROCEConstraint] = []
        for predicate, verb_pattern, argument_key, description in [
            ("starts_with", r"(?:start|begin)\s+(?:your\s+)?(?:response\s+)?with", "text", "begin"),
            (
                "ends_with",
                r"(?:end|finish|close|conclude)\s+(?:your\s+)?(?:response\s+)?with",
                "text",
                "end",
            ),
        ]:
            pattern = rf"{verb_pattern}\s+['\"]([^'\"]+)['\"]"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1).strip()
                results.append(
                    ROCEConstraint(
                        kind="boundary",
                        predicate=predicate,
                        arguments={argument_key: value},
                        description=f"Response must {description} with '{value}'",
                        raw_phrase=match.group(0),
                    )
                )
        return results

    def _extract_output_shape(self, text: str) -> list[ROCEConstraint]:
        results: list[ROCEConstraint] = []
        single_line = re.search(r"\bsingle[-\s]line\b", text, re.IGNORECASE)
        if single_line:
            results.append(
                ROCEConstraint(
                    kind="format",
                    predicate="single_line",
                    arguments={},
                    description="Response must be a single line",
                    raw_phrase=single_line.group(0),
                )
            )

        final_only = re.search(
            r"(?:return|provide|give)\s+only\s+the\s+final\s+answer|final\s+answer\s+only",
            text,
            re.IGNORECASE,
        )
        if final_only:
            results.append(
                ROCEConstraint(
                    kind="scope",
                    predicate="final_answer_only",
                    arguments={},
                    description="Response must include only the final answer",
                    raw_phrase=final_only.group(0),
                )
            )

        answer_type = re.search(
            r"answer\s+with\s+a\s+single\s+(number|integer|word|sentence)",
            text,
            re.IGNORECASE,
        )
        if answer_type:
            results.append(
                ROCEConstraint(
                    kind="type",
                    predicate="answer_type",
                    arguments={"type": answer_type.group(1).lower()},
                    description=f"Response answer type must be {answer_type.group(1).lower()}",
                    raw_phrase=answer_type.group(0),
                )
            )
        return results

    def _extract_arithmetic_equalities(self, text: str) -> list[ROCEConstraint]:
        results: list[ROCEConstraint] = []
        pattern = r"(-?\d+(?:\.\d+)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+)\s*=\s*(-?\d+(?:\.\d+)?)"
        for match in re.finditer(pattern, text):
            left = " ".join(match.group(1).split())
            right = match.group(2)
            results.append(
                ROCEConstraint(
                    kind="arithmetic",
                    predicate="arithmetic_equality",
                    arguments={"left": left, "right": right},
                    description=f"Response must state arithmetic equality {left} = {right}",
                    raw_phrase=match.group(0),
                )
            )
        return results

    def _extract_conditional_guards(self, text: str) -> list[ROCEConstraint]:
        results: list[ROCEConstraint] = []
        pattern = (
            r"\bif\s+(?:the\s+)?response\s+(?:contains|includes|mentions)\s+"
            r"['\"]([^'\"]+)['\"]\s*,?\s*(?:it|the\s+response)\s+must\s+"
            r"(?:also\s+)?(?:contain|include|mention)\s+['\"]([^'\"]+)['\"]"
        )
        for match in re.finditer(pattern, text, re.IGNORECASE):
            guard_term = match.group(1).strip()
            required_term = match.group(2).strip()
            results.append(
                ROCEConstraint(
                    kind="conditional",
                    predicate="conditional_required_text",
                    arguments={
                        "guard": {"predicate": "contains_text", "term": guard_term},
                        "then": {"predicate": "required_text", "term": required_term},
                    },
                    description=(
                        f"If response contains '{guard_term}', it must contain "
                        f"'{required_term}'"
                    ),
                    raw_phrase=match.group(0),
                )
            )
        return results


def extract_roce_constraints(text: str, domain: str | None = None) -> list[ConstraintResult]:
    """Convenience wrapper for one-shot ROCE extraction."""
    return ROCEExtractor().extract(text, domain=domain)


def _number_from_token(token: str) -> int:
    lowered = token.lower()
    if lowered.isdigit():
        return int(lowered)
    return _NUMBER_WORDS[lowered]


def _dedupe_constraints(constraints: list[ROCEConstraint]) -> list[ROCEConstraint]:
    seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
    deduped: list[ROCEConstraint] = []
    for constraint in constraints:
        key = (constraint.predicate, _normalized_items(constraint.arguments))
        if key not in seen:
            seen.add(key)
            deduped.append(constraint)
    return deduped


def _normalized_items(arguments: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(
        sorted((_normalize_value(key), _normalize_value(value)) for key, value in arguments.items())
    )


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, dict):
        return _normalized_items(value)
    return value


__all__ = ["ROCEConstraint", "ROCEExtractor", "extract_roce_constraints"]
