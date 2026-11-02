"""Check only exact, typed propositions in native tool source.

The model proposes pointers. This module checks their bytes and the original
answer's narrow proposition independently, so plausible prose stays unknown.
Spec: REQ-REPORT-7665 and SCENARIO-REPORT-7665-POINTER.
"""

from __future__ import annotations

import json
from typing import Any

from carnot.verify.tool_source_atoms import parse_atoms


def parse_pointers(response: str) -> list[dict[str, Any]]:
    """Accept one small schema; reject partial or changed model structures."""
    value = json.loads(response)
    if not isinstance(value, dict) or set(value) != {"pointers"}:
        raise ValueError("pointer_object_required")
    pointers = value["pointers"]
    if not isinstance(pointers, list) or len(pointers) > 32:
        raise ValueError("bounded_pointer_list_required")
    for pointer in pointers:
        if not isinstance(pointer, dict) or set(pointer) != {
            "claim_index",
            "source_byte_start",
            "source_byte_end",
            "relation",
        }:
            raise ValueError("pointer_fields_invalid")
        if not all(
            type(pointer[key]) is int
            for key in ("claim_index", "source_byte_start", "source_byte_end")
        ) or pointer["relation"] not in {"supports", "contradicts", "unknown"}:
            raise ValueError("pointer_values_invalid")
    return pointers


def check_pointers(row: dict[str, Any], pointers: list[dict[str, Any]]) -> dict[str, Any]:
    """Count support only when the cited bytes match independent narrow truth."""
    source = row["source"]
    source_bytes = source.encode("utf-8")
    atoms = parse_atoms(source)
    witnesses = row["truth"]["witnesses"]
    proposition_rows = []
    invalid = unsupported = exact = 0
    for pointer in pointers:
        start, end = pointer["source_byte_start"], pointer["source_byte_end"]
        claim_index = pointer["claim_index"]
        claim = witnesses[claim_index] if 0 <= claim_index < len(witnesses) else None
        atom = next(
            (item for item in atoms if item["byte_start"] == start and item["byte_end"] == end),
            None,
        )
        valid = bool(
            atom
            and 0 <= start < end <= len(source_bytes)
            and source_bytes[start:end] == atom["text"].encode("utf-8")
        )
        if not valid or claim is None:
            invalid += 1
        grounded = bool(
            valid
            and claim
            and claim["status"] == "observed"
            and claim["evidence_span"] == [start, end]
        )
        if pointer["relation"] == "supports":
            exact += int(grounded)
            unsupported += int(not grounded)
        proposition_rows.append(
            {
                "original_answer_span": [claim["byte_start"], claim["byte_end"]] if claim else None,
                "generated_pointer": pointer,
                "independent_predicate_truth": claim["status"] if claim else "unknown",
                "source_atom_text": atom["text"] if valid and atom else None,
                "exact_supported": grounded and pointer["relation"] == "supports",
                "residual_semantics": "Whole-answer and causal meaning remain unchecked.",
            }
        )
    return {
        "pointer_count": len(pointers),
        "exact_supported": exact,
        "invalid_pointers": invalid,
        "unsupported_certifications": unsupported,
        "proposition_rows": proposition_rows,
    }
