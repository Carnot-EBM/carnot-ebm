"""Token-based symbolic encoder for the Exp6745 certificate DSL.

This encoder groups already parsed DSL tokens with regular expressions. A
second module uses a separate character scanner. Their outputs use one shared
semantic shape so the experiment can compare constraints instead of text.

Spec: REQ-VERIFY-6745 and SCENARIO-VERIFY-6745-DUAL.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import re
from typing import Any


ENCODER_ID = "carnot.dual_certificate.token_regex.v1"
_SAT_TERM = re.compile(r"x([1-9][0-9]*)=([01])")
_UNSAT_TERM = re.compile(r"c([1-9][0-9]*)")


def encode_certificate(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one syntax-checked certificate into normalized constraints.

    Duplicate SAT bindings become a set of required values. This preserves a
    contradiction such as ``x1=0 x1=1`` instead of silently choosing one term.
    """

    if parsed.get("parser_status") != "parseable":
        raise ValueError("certificate is not parseable")
    claim = str(parsed.get("claim"))
    terms = [str(term) for term in parsed.get("terms", [])]
    if claim == "SAT":
        grouped: dict[int, set[bool]] = defaultdict(set)
        for term in terms:
            match = _SAT_TERM.fullmatch(term)
            if match is None:
                raise ValueError(f"invalid SAT term: {term}")
            grouped[int(match.group(1))].add(match.group(2) == "1")
        bindings = [
            {"variable": variable, "values": sorted(values)}
            for variable, values in sorted(grouped.items())
        ]
        core: list[int] = []
    elif claim == "UNSAT":
        core_values = []
        for term in terms:
            match = _UNSAT_TERM.fullmatch(term)
            if match is None:
                raise ValueError(f"invalid UNSAT term: {term}")
            core_values.append(int(match.group(1)))
        bindings = []
        core = sorted(set(core_values))
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
