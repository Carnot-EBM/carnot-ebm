"""Character-scanner symbolic encoder for the Exp6745 certificate DSL.

This module does not import or call the token-regex encoder. It uses explicit
character checks and builds the same normalized semantic structure.

Spec: REQ-VERIFY-6745 and SCENARIO-VERIFY-6745-DUAL.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ENCODER_ID = "carnot.dual_certificate.character_scanner.v1"


def _positive_integer(text: str, prefix: str) -> int:
    """Read a positive decimal integer after one required prefix character."""

    if not text.startswith(prefix):
        raise ValueError(f"term does not start with {prefix}: {text}")
    digits = text[len(prefix) :]
    if not digits or digits[0] == "0" or any(character not in "0123456789" for character in digits):
        raise ValueError(f"term has no positive integer: {text}")
    return int(digits)


def encode_certificate(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Scan one syntax-checked certificate and emit semantic constraints."""

    if parsed.get("parser_status") != "parseable":
        raise ValueError("certificate is not parseable")
    claim = str(parsed.get("claim"))
    terms = [str(term) for term in parsed.get("terms", [])]
    bindings: list[dict[str, Any]] = []
    core: list[int] = []
    if claim == "SAT":
        grouped: dict[int, list[bool]] = {}
        for term in terms:
            if term.count("=") != 1:
                raise ValueError(f"invalid SAT term: {term}")
            variable_text, value_text = term.split("=", 1)
            variable = _positive_integer(variable_text, "x")
            if value_text not in {"0", "1"}:
                raise ValueError(f"invalid SAT value: {term}")
            value = value_text == "1"
            values = grouped.setdefault(variable, [])
            if value not in values:
                values.append(value)
        bindings = [
            {"variable": variable, "values": sorted(grouped[variable])}
            for variable in sorted(grouped)
        ]
    elif claim == "UNSAT":
        seen: dict[int, None] = {}
        for term in terms:
            seen[_positive_integer(term, "c")] = None
        core = sorted(seen)
    else:
        raise ValueError(f"unsupported claim: {claim}")
    return {
        "encoder_id": ENCODER_ID,
        "normalized_constraints": {
            "claim": claim,
            "bindings": bindings,
            "core_clause_indices": core,
        },
    }
