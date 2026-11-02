"""REQ-REPORT-8066: durable public vectors avoid repeated lexical extraction.

Only public bytes enter this service. State-dependent probabilities and evaluator
labels stay with the caller so a cache hit cannot reuse an old policy decision.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import resource
import sqlite3
import time
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.verify import evidence_features_7980 as features
from carnot.verify import source_alignment as alignment
from carnot.verify.source_projection import PUBLIC_KEYS

Json = dict[str, Any]


def extractor_identity() -> Json:
    """Bind executable dependencies and constants rather than a mutable nickname."""
    return dict(
        extractor=[sha256_file(Path(m.__file__)) for m in (features, alignment)],
        config=dict(
            max_windows=alignment.MAX_WINDOWS,
            max_answer_units=alignment.MAX_ANSWER_UNITS,
            token_pattern=alignment.TOKEN_PATTERN,
            negations=sorted(alignment.NEGATIONS),
        ),
        version="8066-v1",
        schema=list(features.FEATURES),
    )


class FeatureCache:
    """Bound SQLite entries and page memory; integrity failures recompute publicly."""

    def __init__(self, path: Path, *, capacity: int = 256, identity: Json | None = None):
        if not 1 <= capacity <= 4096:
            raise ValueError("cache_capacity")
        self.path, self.capacity = path, capacity
        self.identity = identity if identity is not None else extractor_identity()
        self.identity_bytes = json.dumps(self.identity, sort_keys=True).encode()
        self.events: deque[Json] = deque(maxlen=512)
        started = time.perf_counter_ns()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA cache_size=-64")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS features(seq INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT UNIQUE,vector TEXT,digest TEXT)"
        )
        self.create_ns = time.perf_counter_ns() - started

    def key(self, row: Json) -> str:
        """Length-prefixed exact bytes prevent delimiter and normalization collisions."""
        if set(row) != PUBLIC_KEYS or not isinstance(row["family_id"], str) or not row["family_id"]:
            raise ValueError("public_fields")
        digest = hashlib.sha256()
        for value in (
            self.identity_bytes,
            row["family_id"].encode(),
            bytes.fromhex(row["source_bytes"]),
            bytes.fromhex(row["answer_bytes"]),
        ):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        return digest.hexdigest()

    def get(self, row: Json) -> Json:
        """Charge hashing, disk lookup, extraction, invalidation and durable writes."""
        started = time.perf_counter_ns()
        key = self.key(row)
        hashing = time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        saved = self.db.execute("SELECT vector,digest FROM features WHERE key=?", (key,)).fetchone()
        values = None
        status = "miss"
        if saved:
            try:
                values = json.loads(saved[0])
                if (
                    not isinstance(values, list)
                    or len(values) != 8
                    or any(
                        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                        for v in values
                    )
                    or saved[1] != canonical_hash(dict(key=key, values=values))
                ):
                    raise ValueError("cache_integrity")
                status = "hit"
            except (ValueError, TypeError):
                values, status = None, "corrupt_miss"
        loading = time.perf_counter_ns() - started
        extracting = storage = invalidating = evicted = 0
        if values is None:
            started = time.perf_counter_ns()
            values = features.extract(row)["values"]
            if values is None:
                raise ValueError("feature_abstention")
            extracting = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            with self.db:
                self.db.execute("DELETE FROM features WHERE key=?", (key,))
                count = self.db.execute("SELECT count(*) FROM features").fetchone()[0]
                evicted = max(0, count - self.capacity + 1)
                self.db.execute(
                    "DELETE FROM features WHERE seq IN (SELECT seq FROM features ORDER BY seq LIMIT ?)",
                    (evicted,),
                )
                invalidating = time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                self.db.execute(
                    "INSERT INTO features(key,vector,digest) VALUES(?,?,?)",
                    (key, json.dumps(values), canonical_hash(dict(key=key, values=values))),
                )
            storage = time.perf_counter_ns() - started
        self.events.append(
            dict(
                key=key,
                status=status,
                hashing_ns=hashing,
                loading_ns=loading,
                extraction_ns=extracting,
                storage_ns=storage,
                invalidation_ns=invalidating,
                evicted=evicted,
                numerator=int(status == "hit"),
                denominator=1,
            )
        )
        return dict(values=values)

    def memory(self) -> Json:
        """Report configured resident bounds separately from allocated disk bytes."""
        return dict(
            resident_bound_bytes=65536 + 512 * 2048 + len(self.identity_bytes),
            disk_bytes=self.path.stat().st_size,
            entries=self.db.execute("SELECT count(*) FROM features").fetchone()[0],
            capacity=self.capacity,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )

    def close(self) -> None:
        """Close committed storage so a new process verifies its own disk view."""
        self.db.close()
