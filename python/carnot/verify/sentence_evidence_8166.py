"""REQ-VERIFY-8166: preserve intact text while checking local transport.

The conservative punctuation rule can refuse ambiguous text. It never shortens
the answer. Quote checks establish byte custody, not semantic truth.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import math
import re
from typing import Any

Json = dict[str, Any]
ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "u.s",
    "u.k",
}


def partition(answer: bytes) -> list[Json]:
    """Keep every byte in indexed sentences so qualifiers cannot be dropped.

    Headers stay attached to the following sentence. A final unfinished fragment
    remains visible and forces escalation instead of receiving a fabricated label.
    """
    text = answer.decode("utf-8")
    if not text.strip():
        return []
    ends = []
    for match in re.finditer(r"[.!?]+[\"\'”’)]*(?:\s+|$)", text):
        stem = text[: match.start()]
        word = re.search(r"([\w.]+)$", stem)
        token = word[1] if word else ""
        if match[0].startswith(".") and (
            token.casefold() in ABBREVIATIONS
            or token.isdigit()
            or (len(token) == 1 and token.isupper())
        ):
            continue
        ends.append(match.end())
    complete = bool(ends and ends[-1] == len(text))
    if not complete:
        ends.append(len(text))
    rows = []
    start = 0
    for index, end in enumerate(ends):
        rows.append(
            dict(
                sentence_index=index,
                byte_start=len(text[:start].encode()),
                byte_end=len(text[:end].encode()),
                sentence_bytes=text[start:end].encode().hex(),
                complete=complete or end != len(text),
            )
        )
        start = end
    return rows


def requests(row: Json, count: Callable[[str], int] | None = None) -> Json:
    """Bundle complete sentences with intact source and answer before capture.

    UTF-8 byte count is a conservative upper bound for the historical byte-fallback
    tokenizer. Future dispatch must check its exact frozen tokenizer independently.
    """
    source, answer = (bytes.fromhex(row[k]) for k in ("source_bytes", "answer_bytes"))
    sentences = partition(answer)
    reason = (
        "missing_source"
        if not source.strip()
        else "missing_sentences"
        if not sentences
        else "incomplete_sentence"
        if not all(r["complete"] for r in sentences)
        else "sentence_capacity"
        if len(sentences) > 8
        else None
    )
    calls = []
    if reason is None:
        for i in range(0, len(sentences), 4):
            payload = json.dumps(
                dict(
                    instruction="Return one JSON item per sentence_index: p_unsupported, relation (entailed/contradicted/baseless), verbatim source quote or null, UTF-8 byte_start and byte_end or null.",
                    source=source.decode(),
                    answer=answer.decode(),
                    sentences=[
                        dict(r, text=bytes.fromhex(r["sentence_bytes"]).decode())
                        for r in sentences[i : i + 4]
                    ],
                ),
                ensure_ascii=False,
            )
            n = count(payload) if count else len(payload.encode())
            if type(n) is not int or n < 0 or n > 6000:
                reason = "input_token_limit"
            calls.append(
                dict(
                    request_index=i // 4,
                    payload=payload,
                    input_upper_bound=n,
                    token_count_measured=count is not None,
                    maximum_output_tokens=256,
                    sentence_indices=[r["sentence_index"] for r in sentences[i : i + 4]],
                    instruction="Return one JSON item per sentence_index: p_unsupported, relation (entailed/contradicted/baseless), verbatim source quote or null, UTF-8 byte_start and byte_end or null.",
                )
            )
    return dict(
        sentences=sentences,
        requests=[] if reason else calls,
        status="escalated" if reason else "completed",
        exclusion_reason=reason,
        human_target=None,
    )


def parse(transcript: str, source: bytes, indices: list[int]) -> Json:
    """Validate each transport field independently; semantic claims stay predictions."""
    rows = []
    valid = True
    try:
        values = json.loads(transcript)
        if (
            not isinstance(values, list)
            or len(values) != len(indices)
            or not indices
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("sentence_coverage")
        for index, value in zip(indices, values, strict=True):
            p = value["p_unsupported"]
            fields = (
                type(value["sentence_index"]) is int
                and value["sentence_index"] == index
                and type(p) in (int, float)
                and math.isfinite(p)
                and 0 <= p <= 1
                and value["relation"] in ("entailed", "contradicted", "baseless")
            )
            quote, start, end = (value[k] for k in ("quote", "byte_start", "byte_end"))
            null_valid = quote is None and start is None and end is None
            quote_valid = (
                isinstance(quote, str)
                and bool(quote.strip())
                and type(start) is int
                and type(end) is int
                and 0 <= start < end <= len(source)
                and source[start:end] == quote.encode()
            )
            valid = valid and fields and (null_valid or quote_valid)
            rows.append(
                dict(
                    value,
                    quote_valid=bool(quote_valid),
                    fields_valid=fields,
                    quote_offsets_valid=null_valid or bool(quote_valid),
                    human_target=None,
                )
            )
    except (ValueError, KeyError, TypeError):
        valid = False
    return dict(
        rows=rows, status="completed" if valid else "escalated", quote_validity_is_entailment=False
    )


def local_features(rows: list[Json], count: int) -> list[float] | None:
    """Missing local evidence escalates a whole source, preserving paired units."""
    if (
        count == 0
        or len(rows) != count
        or [r["sentence_index"] for r in rows] != list(range(count))
    ):
        return None
    return [
        sum(r["p_unsupported"] for r in rows) / count,
        max(r["p_unsupported"] for r in rows),
        sum(r["relation"] == "contradicted" for r in rows) / count,
        sum(r["relation"] == "baseless" for r in rows) / count,
    ]
