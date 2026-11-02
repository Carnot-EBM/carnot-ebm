"""Cross-sentence semantic consistency verifier.

Spec: REQ-VERIFY-1107, SCENARIO-VERIFY-1107
"""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation


class SemanticConsistencyVerifier:
    """Checks logical consistency across sentences in a reasoning chain.

    The verifier detects direct ``A``/``not A`` contradictions and repeated
    numeric claims for the same subject with conflicting values. It is a cheap
    cross-sentence relation checker, not a language-model or token-statistical
    verifier.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def score(self, text: str) -> float:
        """Return cross-sentence inconsistency energy in [0, 1]."""
        if not text or not text.strip():
            return 0.5

        sentences = _split_sentences(text)
        if not sentences:
            return 0.5

        numeric_claims = _extract_numeric_claims(sentences)
        prop_claims = _extract_polarity_claims(sentences)

        violations = 0
        checks = 0

        for values in numeric_claims.values():
            if len(values) < 2:
                continue
            checks += 1
            if len(set(values)) > 1:
                violations += 1

        for polarities in prop_claims.values():
            if len(polarities) < 2:
                continue
            checks += 1
            if True in polarities and False in polarities:
                violations += 1

        if checks == 0:
            return 0.0
        return float(max(0.0, min(1.0, violations / checks)))


def _split_sentences(text: str) -> list[str]:
    cleaned = text.replace("\\n", "\n")
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [part.strip(" -*\t") for part in parts if part.strip(" -*\t")]


def _extract_numeric_claims(sentences: list[str]) -> dict[str, list[Decimal]]:
    claims: dict[str, list[Decimal]] = defaultdict(list)
    pattern = re.compile(
        r"\b(?P<key>[A-Za-z][A-Za-z0-9 _-]{1,48}?)\s*"
        r"(?P<verb>=|:|is|are|was|were|equals?|remains?)\s*"
        r"(?:\$)?(?P<num>[-+]?\d[\d,]*(?:\.\d+)?)%?"
        r"(?!\s*[+\-*/×÷])",
        flags=re.IGNORECASE,
    )
    for sentence in sentences:
        for match in pattern.finditer(sentence):
            key = _canonical_key(match.group("key"))
            if not key:
                continue
            value = _parse_decimal(match.group("num"))
            if value is not None:
                claims[key].append(value)
    return claims


def _extract_polarity_claims(sentences: list[str]) -> dict[str, set[bool]]:
    claims: dict[str, set[bool]] = defaultdict(set)
    pattern = re.compile(
        r"\b(?P<subject>[A-Za-z][A-Za-z0-9 _-]{1,48}?)\s+"
        r"(?P<verb>is|are|was|were|has|have|can|will|equals?)\s+"
        r"(?P<neg>not\s+)?(?P<object>[A-Za-z][A-Za-z0-9 _-]{1,48})",
        flags=re.IGNORECASE,
    )
    for sentence in sentences:
        for match in pattern.finditer(sentence):
            subject = _canonical_key(match.group("subject"))
            obj = _canonical_object(match.group("object"))
            if not subject or not obj:
                continue
            key = f"{subject} {match.group('verb').lower()} {obj}"
            claims[key].add(match.group("neg") is None)
    return claims


def _canonical_key(raw: str) -> str:
    cleaned = raw.lower()
    cleaned = re.sub(r"[^a-z0-9 _-]+", " ", cleaned)
    words = [w for w in cleaned.split() if w not in _STOPWORDS]
    if not words:
        return ""
    words = words[-3:]
    key = " ".join(words)
    if len(key) < 3:
        return ""
    return key


def _canonical_object(raw: str) -> str:
    cleaned = raw.lower()
    cleaned = re.sub(r"[^a-z0-9 _-]+", " ", cleaned)
    words = [w for w in cleaned.split() if w not in _STOPWORDS]
    if not words:
        return ""
    return " ".join(words[:4])


def _parse_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "now",
    "of",
    "so",
    "that",
    "the",
    "then",
    "therefore",
    "this",
    "to",
    "we",
}
