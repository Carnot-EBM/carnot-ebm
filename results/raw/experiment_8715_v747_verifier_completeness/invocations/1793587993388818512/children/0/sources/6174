"""Bind generated evidence to original bytes before checking narrow relations.

The model can suggest a source line, but only the independent source reader can
decide what that line means for the untouched answer. Spec: REQ-VERIFY-7676.
"""

from __future__ import annotations

import json
import re
from typing import Any

from carnot.verify.tool_source_relations import QUALIFIER, index_relations, verify_relations


ARMS = ("numeric_offset", "exact_quote")
KINDS = {"stack_frame", "grep_quote", "definition_in_scope"}


def bind_quote(source: str, quote: str) -> list[int] | None:
    """Return a byte span only when the proposed bytes have one occurrence."""
    if not quote:
        return None
    source_bytes, quote_bytes = source.encode("utf-8"), quote.encode("utf-8")
    first = source_bytes.find(quote_bytes)
    if first < 0 or source_bytes.find(quote_bytes, first + 1) >= 0:
        return None
    return [first, first + len(quote_bytes)]


def _parse(text: str, arm: str) -> list[dict[str, Any]]:
    """Reject mixed addressing and changed semantic fields before any scoring."""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"proposals"}:
        raise ValueError("proposal_object_required")
    proposals = value["proposals"]
    if not isinstance(proposals, list) or len(proposals) > 32:
        raise ValueError("bounded_proposals_required")
    address = {"source_quote"} if arm == "exact_quote" else {"source_byte_start", "source_byte_end"}
    required = {"claim_index", "kind", "arguments", "polarity", "modifiers", "relation"} | address
    for proposal in proposals:
        if not isinstance(proposal, dict) or set(proposal) != required:
            raise ValueError("proposal_fields_invalid")
        if type(proposal["claim_index"]) is not int or proposal["kind"] not in KINDS:
            raise ValueError("proposal_type_invalid")
        if not isinstance(proposal["arguments"], dict) or not all(
            isinstance(key, str) and type(item) in {str, int}
            for key, item in proposal["arguments"].items()
        ):
            raise ValueError("arguments_invalid")
        if proposal["polarity"] not in {"positive", "negative"} or proposal["relation"] not in {
            "supports",
            "contradicts",
            "unknown",
        }:
            raise ValueError("relation_invalid")
        if not isinstance(proposal["modifiers"], list) or not all(
            isinstance(item, str) for item in proposal["modifiers"]
        ):
            raise ValueError("modifiers_invalid")
        if arm == "exact_quote" and not isinstance(proposal["source_quote"], str):
            raise ValueError("quote_invalid")
        if arm == "numeric_offset" and not all(type(proposal[key]) is int for key in address):
            raise ValueError("offset_invalid")
    return proposals


def reduce_proposal(row: dict[str, Any], arm: str, text: str, finish_reason: str) -> dict[str, Any]:
    """Keep pointer validity and relation support as separate measurements."""
    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    truncated = finish_reason == "length"
    try:
        proposals = _parse(text, arm)
    except (ValueError, json.JSONDecodeError, TypeError):
        proposals = []
        schema_valid = False
    else:
        schema_valid = True
    source = row["source"]
    source_bytes = source.encode("utf-8")
    decision = verify_relations(source, row["answer"])
    records = index_relations(source)
    expected_modifiers = [match.group().lower() for match in QUALIFIER.finditer(row["answer"])]
    proposition_rows = []
    for proposal in proposals:
        if arm == "exact_quote":
            span = bind_quote(source, proposal["source_quote"])
        else:
            start, end = proposal["source_byte_start"], proposal["source_byte_end"]
            span = [start, end] if 0 <= start < end <= len(source_bytes) else None
        finding = (
            decision["findings"][proposal["claim_index"]]
            if 0 <= proposal["claim_index"] < len(decision["findings"])
            else None
        )
        record = next((item for item in records if item["evidence_span"] == span), None)
        unique = bool(
            record
            and span
            and (
                arm == "numeric_offset"
                or bind_quote(source, source_bytes[span[0] : span[1]].decode("utf-8")) == span
            )
        )
        modifiers_retained = sorted(proposal["modifiers"]) == sorted(expected_modifiers)
        semantics_match = bool(
            finding
            and record
            and proposal["kind"] == finding["kind"] == record["kind"]
            and proposal["arguments"] == finding["arguments"] == record["arguments"]
            and proposal["polarity"] == finding["polarity"]
            and modifiers_retained
        )
        independent = finding["status"] if finding else "unknown"
        supported = bool(
            unique
            and semantics_match
            and independent == "supported"
            and proposal["relation"] == "supports"
            and not truncated
        )
        proposition_rows.append(
            {
                "original_answer_span": finding["answer_span"] if finding else None,
                "original_answer_text": finding["text"] if finding else None,
                "source_span": span,
                "unique_evidence": unique,
                "typed_arguments_match": semantics_match,
                "qualifier_retained": modifiers_retained,
                "independent_status": independent,
                "supported": supported,
                "contradicted": bool(unique and semantics_match and independent == "contradicted"),
                "unknown": not supported and independent != "contradicted",
                "generated_proposal": proposal,
            }
        )
    return {
        "schema_valid": schema_valid,
        "truncated": truncated,
        "proposal_count": len(proposals),
        "unique_evidence": sum(item["unique_evidence"] for item in proposition_rows),
        "full_proposition_supported": sum(item["supported"] for item in proposition_rows),
        "contradicted_content": sum(item["contradicted"] for item in proposition_rows),
        "unknown_content": sum(item["unknown"] for item in proposition_rows),
        "qualifier_retained": sum(item["qualifier_retained"] for item in proposition_rows),
        "unknown_remainder": int(
            bool(decision["residual_unverified_text"]) or decision["status"] == "unknown"
        ),
        "proposition_rows": proposition_rows,
    }
