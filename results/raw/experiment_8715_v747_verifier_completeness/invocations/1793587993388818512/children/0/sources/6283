"""REQ-VERIFY-8629: bounded lexical evidence describes overlap, never entailment.

The opt-in path keeps only a working batch and a three-sentence rolling context.
Exact byte keys live on disk so hash collisions cannot merge distinct windows.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify.evidence_features_7980 import normalized
from carnot.verify.source_alignment import MAX_ANSWER_UNITS, NEGATIONS, TOKEN_PATTERN
from carnot.verify.source_projection import PUBLIC_KEYS

Json = dict[str, Any]


@dataclass(frozen=True)
class Limits:
    """Reserve conservative working space before parsing, avoiding partial scores."""

    source_bytes: int = 512 * 1024
    answer_bytes: int = 64 * 1024
    working_bytes: int = 256 * 1024 * 1024
    disk_bytes: int = 32 * 1024 * 1024
    seconds: float = 30.0


DEFAULT_LIMITS = Limits()


def record_bytes(value: Json) -> bytes:
    """One canonical line binds both byte custody and ordered digest to the record."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def window_hash(value: bytes) -> str:
    """A content hash helps audit windows but never substitutes for byte equality."""
    return hashlib.sha256(value).hexdigest()


def spans(data: bytes) -> Iterator[tuple[int, int, bytes]]:
    """Match legacy Unicode punctuation while counting complete UTF-8 byte offsets."""
    text = data.decode("utf-8", "strict")
    start, byte_start = 0, 0
    for match in re.finditer(r"[.!?]+(?:\s+|$)", text):
        end = match.end()
        part = text[start:end].encode()
        yield byte_start, byte_start + len(part), part
        start, byte_start = end, byte_start + len(part)
    if start < len(text):
        yield byte_start, len(data), text[start:].encode()


def windows(data: bytes) -> Iterator[tuple[int, int, bytes]]:
    """Two passes preserve sentence-first order without retaining the sentence list."""
    yield from spans(data)
    context: deque[tuple[int, int, bytes]] = deque(maxlen=3)
    for span in spans(data):
        context.append(span)
        if len(context) == 3:
            yield context[0][0], context[-1][1], b"".join(s[2] for s in context)


def lexical(window: bytes, units: list[bytes]) -> list[float]:
    """Compute only the four old lexical values, preserving answer reduction order."""
    tokens = set(re.findall(TOKEN_PATTERN, window.decode().casefold()))
    numbers = {t for t in tokens if t.isdecimal()}
    totals: list[list[float]] = [[] for _ in range(4)]
    for unit in units:
        answer = set(re.findall(TOKEN_PATTERN, unit.decode().casefold()))
        answer_numbers = {t for t in answer if t.isdecimal()}
        overlap = len(tokens & answer) / max(len(answer), 1)
        values = [
            overlap,
            float(bool(tokens & NEGATIONS) != bool(answer & NEGATIONS)),
            float(bool(answer_numbers) and numbers != answer_numbers),
            1.0 - overlap,
        ]
        for i, value in enumerate(values):
            totals[i].append(value)
    return [sum(v) / len(units) for v in totals]


def extract(
    row: Json, primitive: Path, *, chunk_size: int = 32, limits: Limits = DEFAULT_LIMITS
) -> Json:
    """Stream every original window or return an explicit resource abstention.

    The conservative reservation covers input decoding, token sets and batches.
    Disk and elapsed-time checks run inside the loop; incomplete logs never qualify.
    """
    started = time.monotonic()
    if type(chunk_size) is not int or not 1 <= chunk_size <= 128:
        raise ValueError("chunk_size")
    if set(row) != PUBLIC_KEYS or not isinstance(row["family_id"], str) or not row["family_id"]:
        raise ValueError("public_fields")
    try:
        source, answer = (bytes.fromhex(row[k]) for k in ("source_bytes", "answer_bytes"))
        source.decode("utf-8", "strict")
        answer.decode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("public_bytes") from error
    reservation = 4 * 1024 * 1024 + 256 * (len(source) + len(answer)) + chunk_size * 256
    resource = (
        "source_bytes_over_budget"
        if len(source) > limits.source_bytes
        else "answer_bytes_over_budget"
        if len(answer) > limits.answer_bytes
        else "working_memory_over_budget"
        if reservation > limits.working_bytes
        else None
    )
    if resource:
        return _result(row, source, None, resource, 0, 0, "", False, started, reservation)
    units = [w for _, _, w in spans(answer)]
    reason = (
        "answer_units_over_budget"
        if len(units) > MAX_ANSWER_UNITS
        else ("empty_answer" if not units else "empty_source" if not source.strip() else None)
    )
    primitive.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="exp8629-dedup-", dir="/var/tmp") as directory:
        database = Path(directory) / "dedup.sqlite"
        with sqlite3.connect(database) as conn, primitive.open("wb") as log:
            conn.execute("PRAGMA cache_size=-1024")
            conn.execute("PRAGMA temp_store=FILE")
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute(
                "CREATE TABLE seen (value BLOB PRIMARY KEY, id INTEGER, "
                "f0 REAL, f1 REAL, f2 REAL, f3 REAL) WITHOUT ROWID"
            )
            iterator = windows(source)
            maxima, count, total = [0.0] * 4, 0, 0
            digest = hashlib.sha256()
            while batch := list(itertools.islice(iterator, chunk_size)):
                for start, end, window in batch:
                    disk = conn.execute("PRAGMA page_count").fetchone()[0] * 4096 + log.tell()
                    resource = (
                        "response_time_over_budget"
                        if time.monotonic() - started >= limits.seconds
                        else "dedup_disk_over_budget"
                        if disk > limits.disk_bytes
                        else None
                    )
                    if resource:
                        return _result(
                            row,
                            source,
                            None,
                            resource,
                            count,
                            total,
                            digest.hexdigest(),
                            False,
                            started,
                            reservation,
                        )
                    key = window.strip()
                    found = conn.execute("SELECT id FROM seen WHERE value=?", (key,)).fetchone()
                    fresh = found is None
                    group = count if fresh else found[0]
                    if fresh:
                        features = lexical(key, units) if not reason else [None] * 4
                        conn.execute(
                            "INSERT INTO seen VALUES (?, ?, ?, ?, ?, ?)", (key, count, *features)
                        )
                        count += 1
                        if not reason:
                            for i, value in enumerate(features):
                                maxima[i] = max(maxima[i], value)
                    record = dict(
                        index=total,
                        start=start,
                        end=end,
                        window_hash=window_hash(key),
                        group_id=group,
                        first_occurrence=fresh,
                        answer_unit_ids=list(range(len(units))),
                    )
                    encoded = record_bytes(record)
                    log.write(encoded)
                    digest.update(encoded)
                    total += 1
            disk = conn.execute("PRAGMA page_count").fetchone()[0] * 4096 + log.tell()
            resource = (
                "response_time_over_budget"
                if time.monotonic() - started >= limits.seconds
                else "dedup_disk_over_budget"
                if disk > limits.disk_bytes
                else None
            )
            sums = (
                [
                    sum(r[0] for r in conn.execute(f"SELECT f{i} FROM seen ORDER BY id"))
                    for i in range(4)
                ]
                if count and not reason and not resource
                else [0.0] * 4
            )
            values = (
                [v for i in range(4) for v in (sums[i] / count, maxima[i])]
                if (count and not reason and not resource)
                else None
            )
            return _result(
                row,
                source,
                values,
                resource or reason,
                count,
                total,
                digest.hexdigest(),
                resource is None,
                started,
                reservation,
            )


def _result(
    row: Json,
    source: bytes,
    values: list[float] | None,
    reason: str | None,
    count: int,
    total: int,
    digest: str,
    complete: bool,
    started: float,
    reservation: int,
) -> Json:
    """Keep old-domain feature hashes independent of new custody and cost fields."""
    feature = dict(
        family_id=row["family_id"],
        values=values,
        abstention=reason,
        window_count=count,
        source_normalized_hash=normalized(source),
    )
    return dict(
        features=dict(feature, feature_hash=canonical_hash(feature)),
        total_window_count=total,
        complete_window_digest=digest,
        complete=complete,
        latency_s=time.monotonic() - started,
        reserved_working_bytes=reservation,
    )
