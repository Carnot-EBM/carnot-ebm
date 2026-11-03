"""Public lexical measurements keep human outcomes outside predictor inputs.

REQ-VERIFY-7980. These signals describe word agreement, not factual truth.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import qwen_energy_calibration_7972 as calibration
from carnot.verify import source_alignment as alignment
from carnot.verify.source_projection import PUBLIC_KEYS

Json = dict[str, Any]
FEATURES = tuple(
    f"{name}_{stat}"
    for name in ("overlap", "negation_mismatch", "numeric_mismatch", "uncovered_fraction")
    for stat in ("mean", "maximum")
)
SEED = b"seed69280"


def normalized(source: bytes) -> str:
    """Merge formatting variants so one source cannot occupy two roles."""
    text = unicodedata.normalize("NFKC", source.decode()).casefold()
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def extract(row: Json) -> Json:
    """Measure complete original text and abstain instead of truncating it."""
    if set(row) != PUBLIC_KEYS or not isinstance(row["family_id"], str) or not row["family_id"]:
        raise ValueError("public_fields")
    try:
        source, answer = (bytes.fromhex(row[k]) for k in ("source_bytes", "answer_bytes"))
        view = alignment.prepare(source, answer)
    except (ValueError, TypeError, UnicodeError) as error:
        raise ValueError("public_bytes") from error
    reason = view["abstention"] or ("empty_source" if not source.strip() else None)
    windows = dict.fromkeys(w.strip() for w in view["windows"])
    pairs = (
        [alignment.pair_features(w, view["answer_units"])[-4:] for w in windows]
        if not reason
        else []
    )
    values = (
        [
            v
            for i in range(4)
            for v in (sum(p[i] for p in pairs) / len(pairs), max(p[i] for p in pairs))
        ]
        if pairs
        else None
    )
    value = dict(
        family_id=row["family_id"],
        values=values,
        abstention=reason,
        window_count=len(windows),
        source_normalized_hash=normalized(source),
    )
    return dict(value, feature_hash=canonical_hash(value))


def extract_file(public: Path, output: Path) -> None:
    """The child receives one public descriptor and never an evaluator path."""
    rows = json.loads(public.read_text())["request_rows"]
    if len({r["family_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate_family")
    features = []
    for index, row in enumerate(rows):
        features.append(extract(row))
        if index % 32 == 0:
            print(f"[exp7980-public] completed={index + 1}/{len(rows)}", flush=True)
    atomic_json(output, dict(rows=features))


def replay_features(public: Path, output: Path) -> None:
    """Cold computation must match every published feature byte and slot."""
    rows = json.loads(public.read_text())["request_rows"]
    features = json.loads(output.read_text())["rows"]
    expected = []
    for index, row in enumerate(rows):
        expected.append(extract(row))
        if index % 32 == 0:
            print(f"[exp7980-cold] completed={index + 1}/{len(rows)}", flush=True)
    if expected != features:
        raise ValueError("feature_drift")


def reserve(
    sources: list[Json],
    responses: list[Json],
    excluded_hashes: set[str],
    excluded_ids: set[str],
    size: int = 96,
) -> tuple[list[Json], list[Json]]:
    """Select TRAIN identities before consulting quality or human outcomes."""
    indexed = {s["source_id"]: s for s in sources}
    groups: dict[str, list[Json]] = {}
    for response in responses:
        if response["split"] != "train" or response["source_id"] not in indexed:
            continue
        source = indexed[response["source_id"]]["source_info"]
        blob = (
            source
            if isinstance(source, str)
            else json.dumps(source, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        ).encode()
        key = normalized(blob)
        groups.setdefault(key, []).append(
            dict(
                response_id=response["id"],
                source_id=response["source_id"],
                source_bytes=blob.hex(),
                answer_bytes=response["response"].encode().hex(),
            )
        )
    roster, public = [], []
    order = sorted(groups, key=lambda key: hashlib.sha256(SEED + key.encode()).hexdigest())
    for key in order:
        candidates = groups[key]
        if key in excluded_hashes or any(
            r["source_id"] in excluded_ids
            or hashlib.sha256(bytes.fromhex(r["source_bytes"])).hexdigest() in excluded_hashes
            for r in candidates
        ):
            continue
        row = min(
            candidates,
            key=lambda r: (
                hashlib.sha256(SEED + r["response_id"].encode()).hexdigest(),
                r["response_id"],
            ),
        )
        fid = canonical_hash(dict(source_hash=key, response_id=row["response_id"]))
        roster.append(
            dict(
                family_id=fid,
                source_id=row["source_id"],
                response_id=row["response_id"],
                normalized_source_hash=key,
                selection_hash=hashlib.sha256(SEED + key.encode()).hexdigest(),
            )
        )
        public.append(
            dict(family_id=fid, source_bytes=row["source_bytes"], answer_bytes=row["answer_bytes"])
        )
        if len(roster) == size:
            break
    return roster, public


def predicates(features: list[Json], fit: list[Json], heads: list[Json]) -> Json:
    """Fit truth vectors alone choose thresholds and residual associations."""
    indexed = {r["family_id"]: r for r in fit}
    usable = [r for r in features if r["values"] is not None]
    matrix = np.array([r["values"] for r in usable]).reshape((-1, 8))
    labels = [indexed[r["family_id"]] for r in usable]
    eligible = [
        i
        for i, r in enumerate(labels)
        if r["y"] in (0, 1) and r["q"] is not None and r["status"] == "completed"
    ]
    q = np.array([labels[i]["q"] for i in eligible])
    p = np.mean([calibration.predict(h, q) for h in heads], axis=0)
    residual = np.array([labels[i]["y"] for i in eligible]) - p
    unary, vectors, seen = [], [], set()
    for index, name in enumerate(FEATURES):
        for percentile in (25, 75):
            threshold = float(np.percentile(matrix[:, index], percentile)) if len(matrix) else None
            truth = (
                tuple(bool(v > threshold) for v in matrix[:, index])
                if threshold is not None
                else ()
            )
            reason = "constant" if len(set(truth)) < 2 else "duplicate" if truth in seen else None
            seen.add(truth)
            unary.append(
                dict(
                    id=f"{name}_q{percentile}",
                    feature=name,
                    index=index,
                    operator=">",
                    threshold=threshold,
                    inert_reason=reason,
                )
            )
            vectors.append(truth)
    candidates, truth_seen = [], set()
    active = sorted(
        (i for i, r in enumerate(unary) if not r["inert_reason"]), key=lambda i: unary[i]["id"]
    )
    for left, right in itertools.combinations(active, 2):
        truth = tuple(a and b for a, b in zip(vectors[left], vectors[right], strict=True))
        if len(set(truth)) < 2 or truth in truth_seen:
            continue
        truth_seen.add(truth)
        x = np.array([truth[i] for i in eligible], dtype=float)
        score = float(abs(np.mean((x - x.mean()) * residual))) if len(x) else 0.0
        candidates.append(
            dict(
                predicates=[unary[left]["id"], unary[right]["id"]],
                association=score,
                fit_truth=list(truth),
            )
        )
    candidates.sort(key=lambda r: (-r["association"], r["predicates"]))
    selected = [
        dict(predicates=r["predicates"], association=r["association"]) for r in candidates[:8]
    ]
    return dict(
        unary=unary,
        conjunctions=selected + [None] * (8 - len(selected)),
        selection_trace=candidates,
        fit_family_ids=[r["family_id"] for r in usable],
        fit_residual_count=len(eligible),
        selection_scope="fit_only_signed_residual_covariance",
    )
