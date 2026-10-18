"""Build four CPU fixture views from authenticated source bytes.

REQ-REPORT-7868-V683. A scripted reply checks request construction. It cannot
tell us whether any edited source changes the truth of the answer.
"""

from __future__ import annotations

import json
import re
from typing import Any

from carnot.verify import context_sufficiency_7854 as context
from carnot.verify import source_interventions as source

Json = dict[str, Any]


def _count(text: str) -> int:
    """Give fixtures a reproducible word count without loading a tokenizer."""
    return len(text.split())


def audit_views(row: Json, requests: list[Json]) -> bool:
    """Check every visible span against original UTF-8 bytes, independently of syntax."""
    raw = row["complete_source"].encode()
    answer = row["complete_response"].encode()
    offsets = source.sentence_offsets(raw)
    target = source.target_span(answer)
    for request in requests:
        body = json.loads(request["messages"][1]["content"])
        ids = body["visible_source_sentence_ids"]
        expected = b"".join(raw[offsets[i]["start_byte"] : offsets[i]["end_byte"]] for i in ids)
        if (
            body["complete_source"].encode() != expected
            or body["source_sentence_offsets"] != [offsets[i] for i in ids]
            or body["original_answer"].encode() != answer[: target["end_byte"]].strip()
        ):
            return False
    return True


def fixture_case(row: Json, reply_text: str, finish: str, seed: int) -> Json:
    """Keep each failure and the requests actually possible after nomination."""
    raw = row["complete_source"].encode()
    answer = row["complete_response"].encode()
    first = source.target_span(answer)
    base: Json = {
        "family_id": row["family_id"],
        "seed": seed,
        "status": "excluded_no_complete_sentence",
        "syntax_valid": False,
        "source_byte_fidelity": True,
        "semantic_sensitivity": None,
        "matching_error": None,
        "requests": [],
        "finish_reason": finish,
        "source_sha256": row["source_sha256"],
        "answer_sha256": row["response_sha256"],
    }
    if not raw:
        base["status"] = "excluded_empty_source"
        return base
    if not re.search(rb"[.!?]+$", answer[: first["end_byte"]].strip()):
        return base
    frozen = context.freeze_protocol(seed=seed)
    full = context.build_request(row, "full_source", None, None, frozen, _count)
    requests = [full]
    offsets = source.sentence_offsets(raw)
    parsed = source.parse_reply(reply_text, finish, list(range(len(offsets))))
    status = parsed["disposition"]
    base.update(
        status=status,
        syntax_valid=status != "invalid_parse",
        requests=requests,
        source_byte_fidelity=audit_views(row, requests),
        nominated_risk=parsed["unsupported_probability"],
        witness_id=parsed["source_sentence_id"],
    )
    if status != "completed":
        return base
    witness = parsed["source_sentence_id"]
    control = context.choose_control(row, witness, _count, seed=seed)
    requests.extend(
        context.build_request(row, arm, witness, None, frozen, _count)
        for arm in ("witness_only", "witness_neighbors")
    )
    if control is None:
        base.update(status="excluded_no_matched_context", requests=requests)
        return base
    requests.append(context.build_request(row, "matched_control", witness, control, frozen, _count))
    witness_span = offsets[witness]
    control_span = offsets[control]
    witness_size = _count(raw[witness_span["start_byte"] : witness_span["end_byte"]].decode())
    control_size = _count(raw[control_span["start_byte"] : control_span["end_byte"]].decode())
    base.update(
        status="completed",
        requests=requests,
        control_id=control,
        matching_error=abs(witness_size - control_size),
        source_byte_fidelity=audit_views(row, requests),
    )
    return base
