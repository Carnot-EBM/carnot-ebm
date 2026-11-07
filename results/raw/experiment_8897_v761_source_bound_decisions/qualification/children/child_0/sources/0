"""REQ-REPORT-8897: keep exact record truth separate from caller-owned source coverage.

The source is canonical UTF8 JSON with a ``records`` list of stable IDs and signed
facts. A caller supplies its complete ledger and exact bytes. This contract says
nothing about whether those records capture the meaning of natural-language text.
"""

import hashlib
import json
from typing import Any

from ._indexed_witness_meter import parse
from .size_aware_witness import verify_size_aware_witness

__all__ = ["verify_source_bound_witness"]


def _ids(value: Any, count: int) -> list[str]:
    """Stable unique IDs prevent a repeated fact from hiding a missing source record."""
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(not isinstance(v, str) or not v or v.strip() != v for v in value)
    ):
        raise ValueError("missing_record_ids")
    if len(set(value)) != count:
        raise ValueError("duplicate_record_ids")
    return value


def _records(records: Any, query: Any) -> dict[str, Any]:
    """Compare stable signed bindings so ledger order has no effect on coverage."""
    if not isinstance(records, list) or any(
        not isinstance(row, dict) or set(row) != {"record_id", "fact"} for row in records
    ):
        raise ValueError("ledger_records")
    ids = _ids([row["record_id"] for row in records], len(records))
    parse(dict(facts=[row["fact"] for row in records], query=query), paid=False)
    return {key: row["fact"] for key, row in zip(ids, records, strict=True)}


def verify_source_bound_witness(
    payload: Any,
    *,
    record_ids: Any = None,
    source_ledger: Any = None,
    source_bytes: Any = None,
    cap: int | None = 200000,
) -> dict[str, Any]:
    """Verify visible records unchanged, then qualify only the supplied typed source.

    Absent provenance leaves both source claims unknown. A valid ledger must match
    canonical source bytes and every visible ID-to-fact binding. Missing source
    records yield incomplete coverage even when the logical answer is supported.
    Invalid provenance is reported separately and cannot change the exact answer.
    """
    exact = verify_size_aware_witness(payload, cap=cap)
    coverage, errors = "unknown", []
    if source_ledger is not None:
        try:
            if not isinstance(source_ledger, dict) or set(source_ledger) != {
                "records",
                "source_sha256",
            }:
                raise ValueError("ledger_fields")
            full = _records(source_ledger["records"], payload["query"])
            if not isinstance(source_bytes, bytes):
                raise ValueError("source_bytes_required")
            if (
                source_ledger["source_sha256"]
                != "sha256:" + hashlib.sha256(source_bytes).hexdigest()
            ):
                raise ValueError("source_hash_mismatch")
            source = json.loads(source_bytes)
            if not isinstance(source, dict) or set(source) != {"records"}:
                raise ValueError("typed_source_fields")
            if source_bytes != json.dumps(source, sort_keys=True, separators=(",", ":")).encode():
                raise ValueError("typed_source_encoding")
            if _records(source["records"], payload["query"]) != full:
                raise ValueError("source_ledger_disagreement")
            visible_ids = _ids(record_ids, len(payload["facts"]))
            visible = dict(zip(visible_ids, payload["facts"], strict=True))
            if any(key not in full or full[key] != fact for key, fact in visible.items()):
                raise ValueError("visible_record_mismatch")
            coverage = "complete" if visible == full else "incomplete"
        except ValueError as error:
            coverage, errors = "invalid", [str(error)]
    return dict(
        exact,
        record_decision=exact["decision"],
        source_coverage_status=coverage,
        source_semantics_status="unknown",
        ledger_errors=errors,
    )
