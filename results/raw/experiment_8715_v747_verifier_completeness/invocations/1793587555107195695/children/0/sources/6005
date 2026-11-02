"""Authenticate source families and build label masks for REQ-VERIFY-7894-V685."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file
from carnot.verify import source_alignment


class InputBlocked(ValueError):
    """Keep external evidence failure separate from an owned training failure."""

    def __init__(self, operands: list[dict[str, Any]]):
        self.operands = operands
        super().__init__(str(operands))


def operand(path: Path, field: str, op: str, expected: Any, observed: Any) -> dict[str, Any]:
    """Keep every failed upstream comparison reviewable by exact file bytes."""
    return {
        "upstream_id": "exp7892-source-boundary",
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }


def _shard_rows(
    upstream: Path, artifact: dict[str, Any], key: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for index, reference in enumerate(artifact.get(key, [])):
        path = Path(reference["path"])
        expected = reference["sha256"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            raise InputBlocked(
                [operand(upstream, f"{key}[{index}].sha256", "==", expected, actual)]
            )
        sources.append({"role": key, "path": str(path), "sha256": actual})
        try:
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        except (ValueError, UnicodeError) as exc:
            raise InputBlocked(
                [operand(upstream, f"{key}[{index}].jsonl", "valid", True, str(exc))]
            ) from exc
    if not sources:
        raise InputBlocked([operand(upstream, key, "nonempty", True, None)])
    return rows, sources


def _known(answer: bytes, label: int, offsets: list[list[int]]) -> list[int]:
    """Unmarked units in a positive response stay unknown, never negative."""
    units = source_alignment.sentence_spans(answer)
    if label == 0:
        return [1] * len(units)
    result = []
    start = 0
    for unit in units:
        stop = start + len(unit)
        result.append(0 if any(left < stop and right > start for left, right in offsets) else -1)
        start = stop
    return result


def load_records(upstream: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join exact public bytes to evaluator labels after the producer gate passes."""
    if not upstream.is_file():
        raise InputBlocked([operand(upstream, "artifact", "exists", True, None)])
    try:
        artifact = json.loads(upstream.read_text())
    except (ValueError, UnicodeError) as exc:
        raise InputBlocked([operand(upstream, "artifact", "valid_json", True, str(exc))]) from exc
    checks = (
        ("experiment_id", "==", 7892),
        ("source_boundary_ready_score", "==", 1),
        ("flagged_adversarial", "==", False),
        ("verdict_class", "in", ["positive", "circular_positive", "null"]),
    )
    failures = [
        operand(upstream, field, op, expected, artifact.get(field))
        for field, op, expected in checks
        if (artifact.get(field) not in expected if op == "in" else artifact.get(field) != expected)
    ]
    if failures:
        raise InputBlocked(failures)
    public, sources = _shard_rows(upstream, artifact, "public_shards")
    evaluator, more = _shard_rows(upstream, artifact, "evaluator_shards")
    sources = [
        {"role": "upstream", "path": str(upstream), "sha256": sha256_file(upstream)},
        *sources,
        *more,
    ]
    by_id = {row["family_id"]: row for row in evaluator}
    if len(by_id) != len(evaluator) or len(public) != len(evaluator):
        raise InputBlocked([operand(upstream, "family_id", "unique_join", len(public), len(by_id))])
    records = []
    for row in public:
        if set(row) != {"family_id", "source_bytes", "answer_bytes"}:
            raise InputBlocked(
                [operand(upstream, "public_columns", "==", "byte_only", sorted(row))]
            )
        family = row["family_id"]
        if family not in by_id:
            raise InputBlocked([operand(upstream, "family_id", "joined", True, family)])
        ev = by_id[family]
        source = bytes.fromhex(row["source_bytes"])
        answer = bytes.fromhex(row["answer_bytes"])
        offsets = ev["annotation_byte_offsets"]
        if (
            ev["observed"] != 1
            or ev["human_label"] not in (0, 1)
            or any(
                not (isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(answer))
                for a, b in offsets
            )
        ):
            raise InputBlocked([operand(upstream, "evaluator_label", "valid", True, family)])
        records.append(
            {
                "id": family,
                "group": family,
                "role": ev["role"],
                "source": source,
                "answer": answer,
                "label": ev["human_label"],
                "known": _known(answer, ev["human_label"], offsets),
                "source_cluster_id": "sha256:" + hashlib.sha256(source).hexdigest(),
            }
        )
    if len({row["id"] for row in records}) != len(records):
        raise InputBlocked([operand(upstream, "family_id", "unique", True, False)])
    return records, sources
