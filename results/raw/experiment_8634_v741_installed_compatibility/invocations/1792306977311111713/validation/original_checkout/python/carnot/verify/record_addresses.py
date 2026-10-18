"""Resolve exact source bytes before assigning narrow answer certificates.

An address proves where text appears. It cannot prove a sentence's meaning.
The caller receives full sentences and smaller checkable propositions separately.
Spec: REQ-VERIFY-7700 and SCENARIO-VERIFY-7700-ADDRESS/CLAIM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Sequence

from carnot.verify.tool_source_atoms import digest, parse_atoms, verify_answer
from carnot.verify.tool_source_relations import index_relations, verify_relations


FEATURE_VOCABULARY = {
    "version": "record_span_protocol.v1",
    "statuses": ["supported", "contradicted", "unknown"],
    "certificate_types": ["bound_tuple", "atom_membership", "atom_definition"],
    "coverage": ["narrow_supported", "narrow_contradicted", "narrow_unknown", "residual_unknown"],
    "address_contract": "Unique UTF-8 span inside one visible source record; no semantic authority.",
}


@dataclass(frozen=True)
class SourceRecord:
    """Keep the exact visible line bytes and its independently parsed tuples."""

    record_id: str
    byte_start: int
    byte_end: int
    source_bytes: str
    dialect: str
    source_id: str
    line: int
    tuples: tuple[dict[str, Any], ...]
    polarity: str
    complete: bool


@dataclass(frozen=True)
class AddressResolution:
    """Report byte validity and containment without a truth label."""

    valid: bool
    reason: str
    span: tuple[int, int] | None
    record_id: str | None
    exact_span: bool
    unique_containment: bool


def index_records(source: str) -> list[SourceRecord]:
    """Reuse native atom boundaries so subquotes remain tied to visible lines."""

    atoms = parse_atoms(source)
    relations = index_relations(source)
    return [
        SourceRecord(
            record_id=f"R{index + 1:03d}",
            byte_start=atom["byte_start"],
            byte_end=atom["byte_end"],
            source_bytes=atom["text"],
            dialect=atom["dialect"],
            source_id=atom["source_id"],
            line=atom["line"],
            tuples=tuple(
                relation
                for relation in relations
                if relation["evidence_span"] == [atom["byte_start"], atom["byte_end"]]
            ),
            polarity="positive",
            complete=atom["complete"],
        )
        for index, atom in enumerate(atoms)
    ]


def resolve_address(
    source: str,
    records: Sequence[SourceRecord],
    *,
    quote: str | None = None,
    span: Sequence[int] | None = None,
) -> AddressResolution:
    """Accept one exact byte address only when one visible record contains it."""

    def reject(reason: str, location: tuple[int, int] | None = None) -> AddressResolution:
        return AddressResolution(False, reason, location, None, False, False)

    encoded = source.encode("utf-8")
    if (quote is None) == (span is None):
        return reject("one_address_required")
    if quote is not None:
        if not quote:
            return reject("empty_quote")
        quoted = quote.encode("utf-8")
        start = encoded.find(quoted)
        if start < 0:
            return reject("quote_absent")
        if encoded.find(quoted, start + 1) >= 0:
            return reject("duplicate_quote")
        location = (start, start + len(quoted))
    else:
        if (
            span is None
            or len(span) != 2
            or any(type(item) is not int for item in span)
            or not 0 <= span[0] < span[1] <= len(encoded)
        ):
            return reject("invalid_span")
        location = (span[0], span[1])
        try:
            encoded[: location[0]].decode("utf-8")
            encoded[: location[1]].decode("utf-8")
        except UnicodeDecodeError:
            return reject("utf8_split", location)
    for record in records:
        if (
            record.byte_start < 0
            or record.byte_end > len(encoded)
            or record.byte_start >= record.byte_end
            or encoded[record.byte_start : record.byte_end] != record.source_bytes.encode("utf-8")
        ):
            return reject("invalid_record", location)
    containing = [
        record
        for record in records
        if record.byte_start <= location[0] and location[1] <= record.byte_end
    ]
    if len(containing) > 1:
        return reject("ambiguous_overlap", location)
    if len(containing) == 1:
        return AddressResolution(True, "resolved", location, containing[0].record_id, True, True)
    touches = sum(
        record.byte_start <= location[0] < record.byte_end
        or record.byte_start < location[1] <= record.byte_end
        for record in records
    )
    return reject("cross_record" if touches > 1 else "outside_record", location)


def _sentences(answer: str) -> list[dict[str, Any]]:
    """Keep every original sentence byte, including its trailing whitespace."""

    rows: list[dict[str, Any]] = []
    start = 0
    boundaries = re.finditer(r"(?<=[.!?])\s+|$", answer)
    for boundary in boundaries:
        end = boundary.end()
        text = answer[start:end]
        if text.strip():
            rows.append(
                {
                    "sentence_id": f"S{len(rows) + 1:03d}",
                    "answer_span": [len(answer[:start].encode()), len(answer[:end].encode())],
                    "full_text": text,
                }
            )
        start = end
    return rows


def analyze_answer(source: str, answer: str) -> dict[str, Any]:
    """Check explicit small claims and leave unparsed sentence bytes visible."""

    records = index_records(source)
    sentences = _sentences(answer)
    relation = verify_relations(source, answer)
    atom = verify_answer(source, answer)
    relation_claims = [row for row in relation["findings"] if row["kind"] != "unhandled_clause"]
    candidates: list[dict[str, Any]] = []
    for claim in relation_claims:
        isolated = verify_relations(source, claim["text"] + ".")
        candidates.append(
            {
                "kind": claim["kind"],
                "arguments": claim["arguments"],
                "polarity": claim["polarity"],
                "answer_span": claim["answer_span"],
                "text": claim["text"],
                "narrow_status": isolated["status"],
                "certificate_type": "bound_tuple",
            }
        )
    for witness in atom["witnesses"]:
        location = [witness["byte_start"], witness["byte_end"]]
        if any(
            claim["answer_span"][0] < location[1] and location[0] < claim["answer_span"][1]
            for claim in candidates
        ):
            continue
        status = witness["status"]
        candidates.append(
            {
                "kind": witness["kind"],
                "arguments": {
                    key: witness[key] for key in ("source_id", "line", "name") if key in witness
                },
                "polarity": "positive",
                "answer_span": location,
                "text": witness["text"],
                "narrow_status": {
                    "observed": "supported",
                    "scoped_contradiction": "contradicted",
                    "unknown": "unknown",
                }[status],
                "certificate_type": (
                    "atom_membership" if witness["kind"] == "path_line" else "atom_definition"
                ),
            }
        )
    candidates.sort(key=lambda row: (row["answer_span"][0], row["answer_span"][1]))
    unclaimed = bytearray(answer.encode("utf-8"))
    propositions = []
    for row in candidates:
        start, end = row["answer_span"]
        sentence = next(
            item for item in sentences if item["answer_span"][0] <= start < item["answer_span"][1]
        )
        unclaimed[start:end] = b" " * (end - start)
        propositions.append(
            {
                **row,
                "proposition_id": f"P{len(propositions) + 1:03d}",
                "sentence_id": sentence["sentence_id"],
            }
        )
    residual = bytes(unclaimed).decode("utf-8").strip(" \t\r\n.,!?;:")
    statuses = {row["narrow_status"] for row in propositions}
    whole = (
        "contradicted"
        if "contradicted" in statuses
        else "supported"
        if statuses == {"supported"} and not residual
        else "unknown"
    )
    return {
        "source_sha256": digest(source),
        "answer_sha256": digest(answer),
        "source_bytes_utf8_hex": source.encode("utf-8").hex(),
        "answer_bytes_utf8_hex": answer.encode("utf-8").hex(),
        "records": [{**asdict(row), "tuples": list(row.tuples)} for row in records],
        "sentences": sentences,
        "propositions": propositions,
        "residual_unknown_text": residual,
        "whole_answer_status": whole,
        "feature_vocabulary": FEATURE_VOCABULARY,
    }


def bind_proposition(
    analysis: dict[str, Any], resolution: AddressResolution, proposition_id: str
) -> dict[str, Any]:
    """Check the original claim against one addressed record, not copied prose."""

    proposition = next(
        (row for row in analysis["propositions"] if row["proposition_id"] == proposition_id), None
    )
    record = next(
        (row for row in analysis["records"] if row["record_id"] == resolution.record_id), None
    )
    bound = False
    if proposition is not None and record is not None and resolution.valid:
        if proposition["certificate_type"] == "bound_tuple":
            bound = any(
                row["kind"] == proposition["kind"]
                and row["arguments"] == proposition["arguments"]
                and row["polarity"] == proposition["polarity"]
                for row in record["tuples"]
            )
        else:
            args = proposition["arguments"]
            bound = record["line"] == args["line"] and (
                "source_id" not in args or record["source_id"] == args["source_id"]
            )
    return {
        "proposition_id": proposition_id,
        "record_id": resolution.record_id,
        "exact_span": resolution.exact_span,
        "unique_containment": resolution.unique_containment,
        "claim_binding": bound,
        "tuple_truth": proposition["narrow_status"]
        if bound and proposition is not None
        else "unknown",
        "certificate_type": proposition["certificate_type"] if proposition is not None else None,
    }


def replay_analysis(source: str, answer: str, saved: dict[str, Any]) -> None:
    """Reject any changed source, answer, span, sentence, proposition or status."""

    if saved != analyze_answer(source, answer):
        raise ValueError("record_analysis_replay_mismatch")
