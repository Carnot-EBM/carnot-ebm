"""Reduce post-cutoff ARC supervisor receipts without running a game.

REQ-REPORT-7860 keeps source authentication in the existing delta reader. This
module adds the later write cutoff and attempt-level provenance boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.agentic.arc_supervisor_refinement import extract_rows
from carnot.reporting.arc_supervisor_delta import inspect_sources
from carnot.reporting.current_work_receipt import sha256_file

OUTCOME_KEYS = (
    "fired",
    "helped",
    "resolved_by_levelup",
    "actions_to_levelup",
    "stagnations_unredirected",
)


def _receipt_lookup(raw: Path) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    """Find the original fields because the older source screen keeps fewer keys."""

    try:
        document = json.loads(raw.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    lookup: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for row in extract_rows(document):
        receipt = row.get("trajectory_supervisor")
        if not isinstance(receipt, dict):
            continue
        redirects = receipt.get("redirects")
        if not isinstance(redirects, list):
            continue
        for redirect in redirects:
            if not isinstance(redirect, dict):
                continue
            identity = redirect.get("redirect_id", redirect.get("id", redirect.get("action_index")))
            key = (row.get("game"), row.get("seed"), identity)
            lookup.setdefault(key, {"row": row, "redirect": redirect})
    return lookup


def reduce_ledger(root: Path, cutoff_ns: int, prior_hashes: set[str]) -> dict[str, Any]:
    """Keep every screened disposition and count only new live attempt receipts."""

    screened = inspect_sources(root, "20260928", prior_hashes)
    lookups: dict[str, dict[tuple[Any, Any, Any], dict[str, Any]]] = {}
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for source in screened:
        if source.get("status") == "source":
            continue
        row = dict(source)
        label = row.get("source_path")
        raw = root / label if isinstance(label, str) else None
        producer_label = row.get("producer_path")
        producer = root / producer_label if isinstance(producer_label, str) else None
        if raw is not None and raw.is_file():
            row["source_sha256"] = sha256_file(raw)
        if row.get("status") in {"completed", "censored"} or row.get("reason") == "duplicate_retry":
            if raw is None or producer is None or not raw.is_file() or not producer.is_file():
                row.update(status="excluded", reason="missing_authenticated_source")
            elif min(raw.stat().st_mtime_ns, producer.stat().st_mtime_ns) <= cutoff_ns:
                row.update(status="excluded", reason="before_cutoff")
            else:
                lookup = lookups.setdefault(label, _receipt_lookup(raw))
                original = lookup.get((row.get("game"), row.get("seed"), row.get("redirect_id")))
                if original is None:
                    row.update(status="excluded", reason="missing_original_attempt")
                else:
                    episode = original["row"]
                    redirect = original["redirect"]
                    attempt = episode.get("attempt", row.get("redirect_id"))
                    row["attempt"] = attempt
                    provenance = episode.get("solve_provenance", episode.get("provenance"))
                    row["solve_provenance"] = provenance
                    row["source_family"] = episode.get("source_family", "live_supervisor_receipt")
                    row["fired"] = redirect.get("fired")
                    row["helped"] = redirect.get("helped")
                    row["resolved_by_levelup"] = redirect.get("resolved_by_levelup")
                    row["actions_to_levelup"] = redirect.get("actions_to_levelup")
                    row["stagnations_unredirected"] = episode["trajectory_supervisor"].get(
                        "stagnations_unredirected"
                    )
                    identity = (row.get("game"), row.get("seed"), attempt)
                    if provenance != "live_agent_self_discovery":
                        row.update(status="excluded", reason="non_live_provenance")
                    elif identity in seen:
                        row.update(status="excluded", reason="duplicate_attempt")
                    else:
                        seen.add(identity)
        output.append(row)
    output.sort(
        key=lambda row: (
            str(row.get("game", "")),
            str(row.get("seed", "")),
            str(row.get("attempt", "")),
            0 if row.get("status") == "completed" else 1,
            str(row.get("reason", "")),
        )
    )
    live = [row for row in output if row.get("status") in {"completed", "censored"}]
    fired = sum(row.get("fired") is True for row in live)
    counts = {
        name: sum(row.get("status") == name for row in output)
        for name in ("completed", "censored", "excluded")
    }
    return {
        "outcome_rows": output,
        "no_new_outcomes": not live,
        "firings": fired,
        "new_level_solves": 0,
        "recommendation_rows": [],
        "sample_size_budget": {
            "intended": len(output),
            "eligible": len(live),
            "started": len(live),
            "completed": counts["completed"],
            "censored": counts["censored"],
            "excluded": counts["excluded"],
            "independent_n": len({(r.get("game"), r.get("seed"), r.get("attempt")) for r in live}),
        },
    }
