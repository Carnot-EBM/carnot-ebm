"""Public byte projection for exposed source families (REQ-REPORT-7838).

Only original source and answer bytes and an opaque join ID enter this module.
The evaluator keeps labels, roles, annotations and confidence elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import evidence_views, source_alignment

PUBLIC_KEYS = frozenset({"family_id", "source_bytes", "answer_bytes"})
FEATURE_KEYS = frozenset(
    {
        "family_id",
        "feature_dim",
        "view_a_windows",
        "view_b_windows",
        "abstention",
        "feature_hash",
        "views",
    }
)


def canonical_bytes(value: Any) -> bytes:
    """Serialize a public value with one stable byte spelling for cold replay."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    """Bind a public value to its exact normalized JSON representation."""
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def public_row(row: dict[str, Any]) -> dict[str, str]:
    """Copy only original bytes and the opaque family join from a source row."""
    return {key: row[key] for key in ("family_id", "source_bytes", "answer_bytes")}


def extract_row(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild both complete sentence layouts and derive label-free features."""
    if set(row) != PUBLIC_KEYS or not isinstance(row["family_id"], str) or not row["family_id"]:
        raise ValueError("invalid_public_fields")
    try:
        source = bytes.fromhex(row["source_bytes"])
        answer = bytes.fromhex(row["answer_bytes"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_public_bytes") from exc
    pair = evidence_views.prepare_views(source, answer)
    views = evidence_views.serialize_pair(pair)
    for arm in ("a", "b"):
        view = pair[arm]
        for blob, offsets, parts in (
            (source, view["source_offsets"], view["source_sentences"]),
            (answer, view["answer_offsets"], view["answer_units"]),
        ):
            if b"".join(parts) != blob or offsets != evidence_views._offsets(parts):
                raise ValueError("invalid_sentence_offsets")
        if any(
            source[start:end] != window
            for (start, end), window in zip(view["window_offsets"], view["windows"], strict=True)
        ):
            raise ValueError("invalid_window_offsets")
        if any(len(vector) != source_alignment.FEATURE_DIM for vector in view["pair_features"]):
            raise ValueError("feature_dim_drift")
    return {
        "family_id": row["family_id"],
        "feature_dim": source_alignment.FEATURE_DIM,
        "view_a_windows": len(pair["a"]["windows"]),
        "view_b_windows": len(pair["b"]["windows"]),
        "abstention": pair["a"]["abstention"],
        "feature_hash": digest({arm: views[arm]["pair_features"] for arm in ("a", "b")}),
        "views": views,
    }


def validate_features(
    public: list[dict[str, Any]], features: list[dict[str, Any]], expected_ids: list[str]
) -> None:
    """Cold-recompute every public feature and reject missing or changed joins."""
    ids = [row.get("family_id") for row in public]
    if ids != expected_ids or len(ids) != len(set(ids)) or len(features) != len(ids):
        raise ValueError("family_roster_drift")
    for row, feature in zip(public, features, strict=True):
        if set(feature) != FEATURE_KEYS or canonical_bytes(feature) != canonical_bytes(
            extract_row(row)
        ):
            raise ValueError("feature_drift")


def masked_loss(
    logits: list[float], labels: list[int], observed: list[int]
) -> tuple[float, list[float]]:
    """Give unknown positions exactly zero loss and derivative for any sentinel."""
    if len(logits) != len(labels) or len(labels) != len(observed):
        raise ValueError("mask_shape")
    if any(mask not in (0, 1) for mask in observed):
        raise ValueError("mask_value")
    if any(label not in (0, 1) for label, mask in zip(labels, observed, strict=True) if mask):
        raise ValueError("known_label_value")
    count = sum(observed)
    if count == 0:
        return 0.0, [0.0 for _ in logits]
    total = 0.0
    gradients = []
    for logit, label, mask in zip(logits, labels, observed, strict=True):
        if not mask:
            gradients.append(0.0)
            continue
        total += (max(logit, 0.0) + math.log1p(math.exp(-abs(logit))) - label * logit) / count
        gradients.append(((1.0 / (1.0 + math.exp(-logit))) - label) / count)
    return total, gradients


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one explicitly named shard without changing row order."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Publish one complete shard atomically with stable row spelling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def extract_file(public_path: Path, output_path: Path) -> None:
    """Run extraction from a public-only descriptor in a separate process."""
    started = time.monotonic()
    public = read_jsonl(public_path)
    ids = [row.get("family_id") for row in public]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_family")
    output = []
    for index, row in enumerate(public, 1):
        output.append(extract_row(row))
        if index % 32 == 0:
            print(
                f"[source-projection] elapsed_s={time.monotonic() - started:.3f} completed={index}",
                flush=True,
            )
    write_jsonl(output_path, output)
    atomic_json(
        output_path.with_suffix(".manifest.json"),
        {"public_sha256": digest(public), "features_sha256": digest(output), "family_ids": ids},
    )


def replay_file(public_path: Path, output_path: Path) -> None:
    """Verify published public and feature shards against exact current bytes."""
    public = read_jsonl(public_path)
    features = read_jsonl(output_path)
    manifest = json.loads(output_path.with_suffix(".manifest.json").read_text())
    if (
        digest(public) != manifest["public_sha256"]
        or digest(features) != manifest["features_sha256"]
    ):
        raise ValueError("manifest_hash_drift")
    validate_features(public, features, manifest["family_ids"])


def main(argv: list[str] | None = None) -> int:
    """Expose only public extraction and independent cold replay commands."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3 or args[0] not in {"extract", "replay"}:
        raise ValueError("usage: extract|replay PUBLIC OUTPUT")
    public, output = Path(args[1]), Path(args[2])
    if args[0] == "extract":
        extract_file(public, output)
    else:
        replay_file(public, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
