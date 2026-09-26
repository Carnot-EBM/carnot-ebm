"""Reduce bounded semantic decisions without treating a quote as proof.

REQ-VERIFY-7716, SCENARIO-VERIFY-7716-ADDRESS.
"""

from __future__ import annotations

import json
import re
from typing import Any


DECISIONS = frozenset({"support", "contradiction", "unknown"})


def sentence_windows(source: str) -> list[dict[str, Any]]:
    """Index every source character once, including separators and whitespace."""
    if not source:
        return []
    windows = []
    start = 0
    for match in re.finditer(r"[.!?](?:\s+|$)", source):
        end = match.end()
        windows.append(
            {"index": len(windows), "start": start, "end": end, "text": source[start:end]}
        )
        start = end
    if start < len(source):
        windows.append(
            {"index": len(windows), "start": start, "end": len(source), "text": source[start:]}
        )
    return windows


def human_relation(annotation_types: list[str] | None) -> str | None:
    """Map human spans conservatively; baseless claims have no contradiction label."""
    if annotation_types is None:
        return None
    if "Evident Conflict" in annotation_types:
        return "contradiction"
    if annotation_types:
        return "unknown"
    return "support"


def reduce_response(
    source: str, text: str, finish_reason: str, annotation_types: list[str] | str | None
) -> dict[str, Any]:
    """Score syntax, unique quote address, and human agreement separately."""
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        pass
    valid = bool(
        isinstance(parsed, dict)
        and set(parsed) == {"decision", "quote"}
        and parsed.get("decision") in DECISIONS
        and isinstance(parsed.get("quote"), str)
    )
    decision = parsed["decision"] if valid else None
    quote = parsed["quote"] if valid else None
    address = bool(quote and source.count(quote) == 1)
    labels = [annotation_types] if isinstance(annotation_types, str) else annotation_types
    human = human_relation(labels)
    return {
        "schema_valid": valid,
        "decision": decision,
        "quote": quote,
        "address_valid": address,
        "human_annotation": human,
        "human_label_agreement": decision == human if decision and human else None,
        "semantic_verified": False,
        "unknown": decision == "unknown",
        "truncated": finish_reason == "length",
        "censored": finish_reason in {"length", "timeout", "error"},
    }


def reduce_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count families once and retain a paired latency comparison."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["family_id"], {})[row["arm"]] = row
    pairs = []
    for family_id, arms in sorted(grouped.items()):
        if {"whole_source", "indexed_windows"} <= set(arms):
            whole, indexed = arms["whole_source"], arms["indexed_windows"]
            pairs.append(
                {
                    "family_id": family_id,
                    "latency_delta_s": indexed["latency_s"] - whole["latency_s"],
                    "decision_changed": indexed["metrics"]["decision"]
                    != whole["metrics"]["decision"],
                }
            )
    return {
        "paired_families": len(pairs),
        "schema_valid_calls": sum(bool(row["metrics"]["schema_valid"]) for row in rows),
        "unknown_calls": sum(row["metrics"]["decision"] == "unknown" for row in rows),
        "pairs": pairs,
    }
