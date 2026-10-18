"""Cross-run tool-gap refinement over live tool-loop evidence (REQ-ARC-WMTE-6770).

WHY. The induction tool loop serves a closed, hand-authored tool set. When the
model calls a tool outside that set, the dispatch layer refuses it — and until
2026-08-29 the refusal's IDENTITY (which tool was wanted) was discarded; only a
counter that also mixes in malformed JSON survived. This module is the
tool-side sibling of arc_supervisor_refinement.py: it ingests the loop's
`tool_gap_events` from run rows into a durable ledger, applies a frozen
evidence contract, and emits a written TOOL-GAP SPECIFICATION for a human.

It never authors a tool. A demanded NAME is demand evidence, not a design:
the model asking for `get_full_grid` does not bound the response, and an
unbounded retrieval tool rebuilds the prompt the tool set exists to shrink.
A human authors the tool via `register_candidate_tool` (default off, enabled
per-run by CARNOT_ARC_INDUCE_CANDIDATE_TOOLS) — the same selection-over-a-
curated-set boundary the supervisor's arm table keeps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Deliberate reuse: the rows-document shape, the row hash, and the
# .git-pruned directory scan are the supervisor refinement tool's, so the two
# refinement ledgers can never disagree about what a row or a scan is.
from carnot.agentic.arc_supervisor_refinement import (
    extract_rows,
    receipt_id_for_row,
    scan_inputs,
)

LEDGER_SCHEMA = "carnot.arc.tool_gap_ledger.v1"
SPECIFICATION_SCHEMA = "carnot.arc.tool_gap_specification.v1"
DEFAULT_LEDGER_PARTS = ("ops", "arc_tool_gap_ledger.json")
GAPS_DOC = "ops/arc_tool_gaps.md"

# Evidence floors, frozen. One hallucinated name in one turn is model noise;
# the same nonexistent name demanded repeatedly across independent runs is a
# missing capability. Three events across two distinct rows is the smallest
# shape that cannot be a single turn's fixation inside one run.
MIN_EVENTS_PER_GAP = 3
MIN_DISTINCT_ROWS = 2
MAX_EXAMPLES_PER_GAP = 5

CAUSAL_CAVEAT = (
    "A demanded tool name is demand evidence, not a design. bad_arguments "
    "events are noisier than unknown_tool events: a TypeError raised inside "
    "a tool body is indistinguishable at the dispatch seam from a signature "
    "mismatch. A human authors any new tool, or nobody does."
)

STATUS_NO_ROWS = "no_rows_ingested"
STATUS_NO_CAPTURE = "no_capture_capable_rows"
STATUS_NO_EVENTS = "no_gap_events_nothing_to_specify"
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_SPECIFICATION = "specification_available"


def stats_dicts_in_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Every tool-loop stats dict a row carries, wherever the harness put it.

    Rows in the wild hold stats under `tool_loop_stats`, `stats`, and the live
    agent's `tool_loop` subset — a fixed key list here would be the
    pattern-narrower-than-concept bug. So this walks the row and matches on
    the capture key itself. It does not descend INTO a matched dict, so a
    stats dict is counted once."""

    found: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("tool_gap_events"), list):
                found.append(node)
                return
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(row)
    return found


def _looks_like_loop_stats(node: Any) -> bool:
    """Pre-capture-era stats: a dict with loop counters but no gap key."""

    return isinstance(node, dict) and (
        "tool_calls_by_name" in node or "tool_call_parse_failures" in node
    )


def row_has_pre_capture_stats(row: dict[str, Any]) -> bool:
    def _walk(node: Any) -> bool:
        if isinstance(node, dict):
            if _looks_like_loop_stats(node) and "tool_gap_events" not in node:
                return True
            return any(_walk(value) for value in node.values())
        if isinstance(node, list):
            return any(_walk(value) for value in node)
        return False

    return _walk(row)


def _events_from_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """(events, dropped) for one row. Dropped rides along because a session
    that truncated at the capture bound must not read as a complete count
    downstream (adversarial review 2026-08-29, F8)."""
    events: list[dict[str, Any]] = []
    dropped = 0
    for stats in stats_dicts_in_row(row):
        for event in stats.get("tool_gap_events") or []:
            if isinstance(event, dict) and event.get("kind") in ("unknown_tool", "bad_arguments"):
                events.append(dict(event))
        value = stats.get("tool_gap_events_dropped")
        if isinstance(value, int) and value > 0:
            dropped += value
    return events, dropped


def empty_ledger() -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "created_at": None,
        "updated_at": None,
        "entries": {},
        "specification": None,
    }


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty_ledger()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != LEDGER_SCHEMA:
        raise ValueError(f"unsupported ledger schema in {path}")
    if not isinstance(data.get("entries"), dict):
        raise ValueError(f"malformed ledger entries in {path}")
    return data


def save_ledger(ledger: dict[str, Any], path: Path) -> None:
    """Atomic replace: a crash mid-write must not half-destroy the evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ingest_files(ledger: dict[str, Any], files: Sequence[Path], now_iso: str) -> dict[str, int]:
    counts = {
        "files_read": 0,
        "rows_seen": 0,
        "capture_rows_new": 0,
        "capture_rows_duplicate": 0,
        "pre_capture_rows": 0,
        "rows_without_stats": 0,
        "events_ingested": 0,
    }
    entries = ledger["entries"]
    for file_path in files:
        doc = json.loads(file_path.read_text(encoding="utf-8"))
        counts["files_read"] += 1
        for row in extract_rows(doc):
            counts["rows_seen"] += 1
            stats_dicts = stats_dicts_in_row(row)
            if not stats_dicts:
                # Distinguish "ran before capture existed" from "no loop ran":
                # the first is a population limit worth naming, not a zero.
                if row_has_pre_capture_stats(row):
                    counts["pre_capture_rows"] += 1
                else:
                    counts["rows_without_stats"] += 1
                continue
            row_id = receipt_id_for_row(row)
            if row_id in entries:
                counts["capture_rows_duplicate"] += 1
                continue
            events, dropped = _events_from_row(row)
            entries[row_id] = {
                "row_id": row_id,
                "source": str(file_path),
                "game": row.get("game"),
                "seed": row.get("seed"),
                "ingested_at": now_iso,
                "events": events,
                "events_dropped": dropped,
            }
            counts["capture_rows_new"] += 1
            counts["events_ingested"] += len(events)
    return counts


def _gap_key(event: dict[str, Any]) -> tuple[str, str]:
    if event.get("kind") == "unknown_tool":
        return ("unknown_tool", str(event.get("requested_tool")))
    return ("bad_arguments", str(event.get("tool")))


def render_markdown_entry(gap: dict[str, Any], ledger_path: str) -> str:
    """A ready-to-append entry in the ops/verifier_gaps.md schema convention.

    Rendered, never written: appending to the gaps doc stays a human act, the
    same boundary REQ-ARC-WMTE-6720 keeps for arm-table changes."""

    kind, name = gap["kind"], gap["name"]
    if kind == "unknown_tool":
        title = f"model demanded tool `{name}` which does not exist"
        failure = (
            "the generator wrote tool calls for a name outside the active set; "
            "each refusal costs a loop turn and the capability stays unserved"
        )
        missing = (
            f"a callable tool named `{name}`; observed argument keys: "
            f"{sorted(gap['argument_keys']) or 'none captured'}"
        )
    else:
        title = f"model repeatedly mis-calls `{name}` (signature mismatch)"
        failure = (
            f"calls to the existing tool `{name}` raise bad-arguments errors; "
            "the model imagines a signature the schema does not declare"
        )
        missing = (
            f"either a schema/description fix for `{name}`, or a variant tool "
            f"matching the demanded signature; sample errors: {gap['examples'][:2]}"
        )
    return "\n".join(
        [
            f"### TOOLGAP-{kind.upper()}-{name}: {title}",
            "- status: open",
            (
                f"- evidence: {gap['events']} events across {gap['distinct_rows']} "
                f"distinct run rows ({ledger_path})"
            ),
            f"- failure mode: {failure}",
            f"- missing capability: {missing}",
            (
                "- candidate design: HUMAN-AUTHORED, bounded, registered via "
                "`arc_induction_tools.register_candidate_tool` and enabled per-run by "
                "`CARNOT_ARC_INDUCE_CANDIDATE_TOOLS` (default off); this analyzer "
                "never generates one"
            ),
            "- priority: medium (rank by refused-call frequency)",
        ]
    )


def evaluate(ledger: dict[str, Any], now_iso: str, ledger_path: str) -> dict[str, Any]:
    """Apply the frozen contract to the whole ledger. Pure function of ledger
    content, so re-evaluation after source deletion reproduces the answer."""

    entries = list(ledger["entries"].values())
    gaps: dict[tuple[str, str], dict[str, Any]] = {}
    total_events = 0
    for entry in entries:
        for event in entry.get("events", []):
            total_events += 1
            key = _gap_key(event)
            gap = gaps.setdefault(
                key,
                {
                    "kind": key[0],
                    "name": key[1],
                    "events": 0,
                    "rows": set(),
                    "argument_keys": set(),
                    "examples": [],
                },
            )
            gap["events"] += 1
            gap["rows"].add(entry["row_id"])
            for arg in event.get("argument_keys") or []:
                gap["argument_keys"].add(str(arg))
            if len(gap["examples"]) < MAX_EXAMPLES_PER_GAP:
                gap["examples"].append(str(event.get("error") or event.get("requested_tool")))

    per_gap: list[dict[str, Any]] = []
    specifications: list[dict[str, Any]] = []
    for gap in sorted(gaps.values(), key=lambda g: (-g["events"], g["kind"], g["name"])):
        row = {
            "kind": gap["kind"],
            "name": gap["name"],
            "events": gap["events"],
            "distinct_rows": len(gap["rows"]),
            "argument_keys": sorted(gap["argument_keys"]),
            "examples": gap["examples"],
            "meets_floor": (
                gap["events"] >= MIN_EVENTS_PER_GAP and len(gap["rows"]) >= MIN_DISTINCT_ROWS
            ),
            "events_shortfall": max(0, MIN_EVENTS_PER_GAP - gap["events"]),
            "rows_shortfall": max(0, MIN_DISTINCT_ROWS - len(gap["rows"])),
        }
        per_gap.append(row)
        if row["meets_floor"]:
            specifications.append(
                {
                    "audience": "human",
                    "instruction": (
                        "Author ONE bounded tool for this demand and register it "
                        "via register_candidate_tool in "
                        "python/carnot/agentic/arc_induction_tools.py, default "
                        "off behind CARNOT_ARC_INDUCE_CANDIDATE_TOOLS. This tool "
                        "never generates a schema or an implementation."
                    ),
                    "gap": {k: row[k] for k in ("kind", "name", "events", "distinct_rows")},
                    "markdown_entry": render_markdown_entry(row, ledger_path),
                    "append_target": GAPS_DOC,
                }
            )

    capture_rows = len(entries)
    if capture_rows == 0:
        counts = ledger.get("last_ingest_counts") or {}
        pre_capture = int(counts.get("pre_capture_rows") or 0)
        status = STATUS_NO_CAPTURE if pre_capture else STATUS_NO_ROWS
    elif total_events == 0:
        status = STATUS_NO_EVENTS
    elif specifications:
        status = STATUS_SPECIFICATION
    else:
        status = STATUS_INSUFFICIENT

    return {
        "schema": SPECIFICATION_SCHEMA,
        "generated_at": now_iso,
        "status": status,
        "contract": {
            "min_events_per_gap": MIN_EVENTS_PER_GAP,
            "min_distinct_rows": MIN_DISTINCT_ROWS,
            "rules": [
                "specification: a gap with events >= floor across >= 2 distinct rows",
                "below either floor: insufficient_evidence, said loudly",
                "the analyzer renders a markdown entry; a human appends and authors",
            ],
        },
        "recommendation_only": True,
        "causal_caveat": CAUSAL_CAVEAT,
        "evidence": {
            "capture_rows": capture_rows,
            "gap_events": total_events,
            # Events lost to the per-session capture bound. Non-zero means
            # every per-gap count below is a FLOOR, not a total (F8).
            "gap_events_dropped_total": sum(
                int(entry.get("events_dropped") or 0) for entry in entries
            ),
            "games": sorted({str(entry.get("game")) for entry in entries}),
        },
        "per_gap": per_gap,
        "specifications": specifications,
    }


def render_report(spec: dict[str, Any]) -> str:
    """Human report. Empty states are loud and honest, never padded."""

    lines = ["ARC tool-gap refinement (REQ-ARC-WMTE-6770)"]
    status = spec["status"]
    if status == STATUS_NO_ROWS:
        lines.append("NO ROWS INGESTED — nothing to analyze.")
    elif status == STATUS_NO_CAPTURE:
        lines.append(
            "NO CAPTURE-CAPABLE ROWS — every ingested row predates tool-gap "
            "capture. This is absence of evidence, not evidence of absence."
        )
    elif status == STATUS_NO_EVENTS:
        lines.append(
            "NO GAP EVENTS — capture ran and the model demanded nothing "
            "outside the active tool set. This honest empty is the result."
        )
    elif status == STATUS_INSUFFICIENT:
        lines.append(
            "INSUFFICIENT EVIDENCE — gap events exist but no gap crosses the "
            f"floor ({MIN_EVENTS_PER_GAP} events across {MIN_DISTINCT_ROWS} "
            "distinct rows). No specification is emitted at these counts."
        )
    else:
        lines.append("SPECIFICATION AVAILABLE — for a human to author; nothing is generated.")
    evidence = spec["evidence"]
    lines.append(
        f"evidence: {evidence['capture_rows']} capture rows, "
        f"{evidence['gap_events']} gap events "
        f"({evidence['gap_events_dropped_total']} dropped at the capture bound — "
        "counts are floors when non-zero), "
        f"games={','.join(evidence['games']) or 'none'}"
    )
    for row in spec["per_gap"]:
        lines.append(
            f"  {row['kind']}:{row['name']}: events={row['events']} "
            f"rows={row['distinct_rows']} "
            f"shortfall=({row['events_shortfall']} events, {row['rows_shortfall']} rows)"
        )
    for item in spec["specifications"]:
        lines.append(f"  SPECIFY {item['gap']['kind']}:{item['gap']['name']} — entry ready to")
        lines.append(f"  append to {item['append_target']}:")
        lines.append(item["markdown_entry"])
    lines.append(f"caveat: {spec['causal_caveat']}")
    return "\n".join(lines)


def _default_ledger_path() -> Path:
    from carnot.paths import repo_path

    return repo_path(*DEFAULT_LEDGER_PARTS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unattended cross-run tool-gap refinement (REQ-ARC-WMTE-6770). "
            "Ingests tool_gap_events from rows.json files or scan directories "
            "into a durable ledger, then prints a specification for a human. "
            "Never edits TOOL_SCHEMAS and never generates a tool."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="rows.json files, or directories scanned for rows.json (repo clones pruned)",
    )
    parser.add_argument("--ledger", type=Path, default=None, help="ledger path override")
    parser.add_argument("--json", action="store_true", help="print the specification as JSON")
    args = parser.parse_args(argv)

    ledger_path = args.ledger if args.ledger is not None else _default_ledger_path()
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        ledger = load_ledger(ledger_path)
        files = scan_inputs(args.inputs)
        counts = ingest_files(ledger, files, now_iso)
        ledger["last_ingest_counts"] = counts
        specification = evaluate(ledger, now_iso, str(ledger_path))
        specification["ingest_counts"] = counts
        ledger["specification"] = specification
        if ledger["created_at"] is None:
            ledger["created_at"] = now_iso
        ledger["updated_at"] = now_iso
        save_ledger(ledger, ledger_path)
    except (OSError, ValueError) as exc:
        # Fail loud: an unattended step that swallows an unreadable input
        # would report clean without having looked.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(specification, indent=2, sort_keys=True))
    else:
        print(render_report(specification))
    print(f"ledger: {ledger_path}")
    return 0
