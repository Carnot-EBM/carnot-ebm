"""Public-byte advisory predicates for REQ-VERIFY-7853.

These signals describe visible wording differences. They do not certify truth.
"""

from __future__ import annotations

import unicodedata

NEGATORS = frozenset({"no", "not", "never", "neither", "without", "cannot"})
STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "in",
        "on",
        "and",
        "or",
        "for",
        "with",
    }
)
SIGNAL_NAMES = (
    "unmatched_decimal",
    "missing_negator",
    "low_content_jaccard",
    "unmatched_titlecase",
)
NAMES = (
    "constant",
    *(
        "and_" + "_".join(SIGNAL_NAMES[i] for i in range(4) if mask & (1 << i))
        for mask in range(1, 16)
    ),
)


def tokens(data: bytes) -> list[str]:
    """Decode strictly so malformed public bytes cannot gain silent features."""
    text = data.decode("utf-8", "strict")
    found: list[str] = []
    run = ""
    for char in text:
        if char.isalnum():
            run += char
        elif run:
            found.append(run)
            run = ""
    if run:
        found.append(run)
    return found


def _decimal(token: str) -> str:
    if not token or not all(char.isdecimal() for char in token):
        return ""
    return "".join(str(unicodedata.decimal(char)) for char in token).lstrip("0") or "0"


def signals(source: bytes, answer: bytes) -> tuple[int, int, int, int]:
    """Use only the two public byte fields, never metadata or evaluator labels."""
    source_tokens = tokens(source)
    answer_tokens = tokens(answer)
    source_folded = {token.casefold() for token in source_tokens}
    answer_folded = {token.casefold() for token in answer_tokens}
    source_numbers = {_decimal(token) for token in source_tokens} - {""}
    answer_numbers = {_decimal(token) for token in answer_tokens} - {""}
    source_content = {token for token in source_folded if token.isalpha() and token not in STOP}
    answer_content = {token for token in answer_folded if token.isalpha() and token not in STOP}
    union = source_content | answer_content
    similarity = len(source_content & answer_content) / len(union) if union else 1.0
    return (
        int(bool(answer_numbers - source_numbers)),
        int(bool((answer_folded & NEGATORS) - source_folded)),
        int(similarity < 0.20),
        int(
            any(
                token.istitle() and token.casefold() not in source_folded
                for token in answer_tokens[1:]
            )
        ),
    )


def features(source: bytes, answer: bytes) -> dict[str, float]:
    """Expand four immutable advisory bits into one bias and fifteen ANDs."""
    bits = signals(source, answer)
    return {
        name: float(mask == 0 or all(bits[i] for i in range(4) if mask & (1 << i)))
        for mask, name in enumerate(NAMES)
    }
