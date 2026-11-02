"""Parse frozen proposal bytes without using the proposal producer.

The parser accepts only the recorded server-sent event stream and the frozen
JSON schema. It does not search for JSON, repair fields, or coerce values. This
keeps parse agreement separate from the code that produced the source rows.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any, Mapping


PARSER_VERSION = "carnot.exp6824.strict_sse_json_parser.v1"
MODULE_PATH = "python/carnot/experiment_6824_cold_parser.py"
CANDIDATE_FIELDS = {"action", "authority_chain", "candidate_id", "soft_progress"}
ACTION_FIELDS = {"data", "kind"}


class ColdParseError(ValueError):
    """Report corrupt transport evidence that prevents a cold replay."""


def sha256_bytes(value: bytes) -> str:
    """Return the source artifact's prefixed SHA-256 notation."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def incomplete_candidate(index: int, reason: str) -> dict[str, Any]:
    """Retain one deterministic candidate slot when strict parsing fails."""

    return {
        "action": None,
        "authority_chain": [],
        "candidate_id": f"candidate_{index}",
        "candidate_index": index,
        "parse_failure": reason,
        "parse_state": "incomplete",
        "soft_progress": None,
    }


def _incomplete(reason: str) -> dict[str, Any]:
    """Build the fixed two-slot result required by the frozen manifest."""

    return {
        "candidates": [incomplete_candidate(index, reason) for index in range(2)],
        "parse_failure": reason,
        "parse_state": "incomplete",
    }


def decode_sse(raw_api: bytes) -> bytes:
    """Concatenate content deltas from an exact server-sent event stream."""

    try:
        text = raw_api.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ColdParseError("raw API stream is not UTF-8") from exc
    content: list[str] = []
    saw_data = False
    for line in text.splitlines():
        if not line:
            continue
        if not line.startswith("data: "):
            raise ColdParseError("SSE data line required")
        saw_data = True
        payload = line[6:]
        if payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ColdParseError("invalid SSE event JSON") from exc
        if not isinstance(event, dict) or not isinstance(event.get("choices"), list):
            raise ColdParseError("SSE event JSON lacks choices")
        for choice in event["choices"]:
            if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                raise ColdParseError("SSE choice lacks delta")
            fragment = choice["delta"].get("content")
            if fragment is not None:
                if not isinstance(fragment, str):
                    raise ColdParseError("SSE content delta is not text")
                content.append(fragment)
    if not saw_data:
        raise ColdParseError("SSE data line required")
    return "".join(content).encode("utf-8")


def _candidate_failure(candidate: Any, index: int) -> str | None:
    """Return the first schema failure for one candidate, if present."""

    if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_FIELDS:
        return "invalid_candidate_fields"
    action = candidate["action"]
    if not isinstance(action, dict) or set(action) != ACTION_FIELDS:
        return "invalid_action"
    kind = action["kind"]
    if isinstance(kind, bool) or not isinstance(kind, (str, int)) or kind == "":
        return "invalid_action"
    chain = candidate["authority_chain"]
    if (
        not isinstance(chain, list)
        or not chain
        or any(not isinstance(item, str) or not item for item in chain)
        or chain != sorted(set(chain))
    ):
        return "invalid_authority_chain"
    if candidate["candidate_id"] != f"candidate_{index}":
        return "invalid_candidate_id"
    progress = candidate["soft_progress"]
    if isinstance(progress, bool) or not isinstance(progress, int):
        return "invalid_soft_progress"
    return None


def parse_proposal(raw_output: bytes) -> dict[str, Any]:
    """Parse exactly one two-candidate JSON object without recovery logic."""

    if not raw_output:
        return _incomplete("empty_output")
    try:
        text = raw_output.decode("utf-8")
    except UnicodeDecodeError:
        return _incomplete("utf8_decode_error")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return _incomplete("json_decode_error")
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        return _incomplete("invalid_top_level_fields")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 2:
        return _incomplete("invalid_candidate_count")
    for index, candidate in enumerate(candidates):
        reason = _candidate_failure(candidate, index)
        if reason is not None:
            return _incomplete(reason)
    parsed = [
        {
            **candidate,
            "candidate_index": index,
            "parse_failure": None,
            "parse_state": "complete",
        }
        for index, candidate in enumerate(candidates)
    ]
    return {"candidates": parsed, "parse_failure": None, "parse_state": "complete"}


def _decode_field(receipt: Mapping[str, Any], field: str) -> bytes:
    """Decode one mandatory base64 field and name corrupt evidence."""

    value = receipt.get(field)
    if not isinstance(value, str):
        raise ColdParseError(f"{field} is missing")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ColdParseError(f"{field} is not valid base64") from exc


def parse_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Verify transport receipts before parsing their proposal bytes."""

    raw_api = _decode_field(receipt, "raw_api_response_b64")
    raw_output = _decode_field(receipt, "raw_output_b64")
    if len(raw_api) != receipt.get("raw_api_response_len"):
        raise ColdParseError("raw API length mismatch")
    if sha256_bytes(raw_api) != receipt.get("raw_api_response_sha256"):
        raise ColdParseError("raw API hash mismatch")
    if len(raw_output) != receipt.get("raw_output_len"):
        raise ColdParseError("raw output length mismatch")
    output_hash = sha256_bytes(raw_output)
    if output_hash != receipt.get("raw_output_sha256"):
        raise ColdParseError("raw output hash mismatch")
    if decode_sse(raw_api) != raw_output:
        raise ColdParseError("SSE content does not equal raw output bytes")
    return {
        **parse_proposal(raw_output),
        "raw_output_bytes": raw_output,
        "raw_output_sha256": output_hash,
    }
