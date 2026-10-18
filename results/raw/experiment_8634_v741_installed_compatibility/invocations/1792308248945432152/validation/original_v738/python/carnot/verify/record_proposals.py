"""Check one Qwen proposal against original source records and answer spans.

The generated support word is only a claim. REQ-VERIFY-7702.
"""

from __future__ import annotations

import json
from typing import Any

from carnot.verify.record_addresses import analyze_answer, resolve_address


ARMS = ("opaque_index", "explicit_record")
_COMMON = {"kind", "arguments", "polarity", "modifiers", "relation", "source_quote"}


def _proposal(text: str, arm: str) -> dict[str, Any]:
    """Accept exactly one proposal and only the chosen arm's reference fields."""

    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"proposal"}:
        raise ValueError("one_proposal_object_required")
    proposal = value["proposal"]
    references = {"claim_index"} if arm == "opaque_index" else {"sentence_id", "proposition_id"}
    if not isinstance(proposal, dict) or set(proposal) != _COMMON | references:
        raise ValueError("proposal_fields_invalid")
    if arm == "opaque_index":
        if type(proposal["claim_index"]) is not int:
            raise ValueError("claim_index_invalid")
    elif not all(isinstance(proposal[key], str) for key in references):
        raise ValueError("sentence_or_proposition_id_invalid")
    if not isinstance(proposal["kind"], str) or not isinstance(proposal["source_quote"], str):
        raise ValueError("kind_or_quote_invalid")
    if not isinstance(proposal["arguments"], dict) or not all(
        isinstance(key, str) and type(value) in {str, int}
        for key, value in proposal["arguments"].items()
    ):
        raise ValueError("arguments_invalid")
    if proposal["polarity"] not in {"positive", "negative"}:
        raise ValueError("polarity_invalid")
    if proposal["relation"] not in {"supports", "contradicts", "unknown"}:
        raise ValueError("relation_invalid")
    if not isinstance(proposal["modifiers"], list) or not all(
        isinstance(item, str) for item in proposal["modifiers"]
    ):
        raise ValueError("modifiers_invalid")
    return proposal


def reduce_response(row: dict[str, Any], arm: str, text: str, finish_reason: str) -> dict[str, Any]:
    """Separate schema, location, claim binding and independent narrow truth."""

    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    analysis = analyze_answer(row["source"], row["answer"])
    truncated = finish_reason == "length"
    try:
        proposal = _proposal(text, arm)
    except (ValueError, TypeError):
        proposal = None
    propositions = analysis["propositions"]
    claimed = None
    if proposal is not None:
        if arm == "opaque_index":
            index = proposal["claim_index"]
            claimed = propositions[index] if 0 <= index < len(propositions) else None
        else:
            claimed = next(
                (
                    item
                    for item in propositions
                    if item["proposition_id"] == proposal["proposition_id"]
                    and item["sentence_id"] == proposal["sentence_id"]
                    and any(
                        sentence["sentence_id"] == item["sentence_id"]
                        and sentence["answer_span"][0] <= item["answer_span"][0]
                        and item["answer_span"][1] <= sentence["answer_span"][1]
                        for sentence in analysis["sentences"]
                    )
                ),
                None,
            )
    address = (
        resolve_address(
            row["source"], analysis_records(row["source"]), quote=proposal["source_quote"]
        )
        if proposal is not None
        else None
    )
    record = next(
        (
            item
            for item in analysis["records"]
            if address and item["record_id"] == address.record_id
        ),
        None,
    )
    typed = bool(
        claimed
        and proposal
        and proposal["kind"] == claimed["kind"]
        and proposal["arguments"] == claimed["arguments"]
        and proposal["polarity"] == claimed["polarity"]
        and not proposal["modifiers"]
    )
    bound = bool(
        typed
        and record
        and any(
            tuple_row["kind"] == claimed["kind"]
            and tuple_row["arguments"] == claimed["arguments"]
            and tuple_row["polarity"] == claimed["polarity"]
            for tuple_row in record["tuples"]
        )
    )
    truth = claimed["narrow_status"] if bound and claimed else "unknown"
    supported = bool(
        not truncated and bound and truth == "supported" and proposal["relation"] == "supports"
    )
    return {
        "schema_valid": proposal is not None,
        "truncated": truncated,
        "proposal_count": int(proposal is not None),
        "exact_span": bool(address and address.exact_span),
        "unique_containing_record": bool(address and address.unique_containment),
        "record_id": address.record_id if address else None,
        "address_reason": address.reason if address else "schema_invalid",
        "claim_alignment": claimed is not None,
        "correct_binding": bound,
        "tuple_truth": truth,
        "supported": supported,
        "false_support": bool(
            proposal and proposal["relation"] == "supports" and truth != "supported"
        ),
        "unknown_remainder": bool(analysis["residual_unknown_text"] or not supported),
        "residual_answer_text": analysis["residual_unknown_text"],
        "original_sentence": next(
            (
                item["full_text"]
                for item in analysis["sentences"]
                if claimed and item["sentence_id"] == claimed["sentence_id"]
            ),
            None,
        ),
        "original_proposition": claimed["text"] if claimed else None,
        "generated_proposal": proposal,
    }


def analysis_records(source: str):
    """Build the independent record objects used by the address resolver."""

    from carnot.verify.record_addresses import index_records

    return index_records(source)
