"""Cross-run refinement over live trajectory-supervisor outcomes (REQ-ARC-WMTE-6720).

WHY. REQ-ARC-WMTE-6640 made redirects measurable, but each run's receipt
dies with its scratch directory. This module is the unattended between-runs
step: it ingests applied receipts into a durable ledger under ops/, then
applies a frozen evidence contract that RECOMMENDS arm-table changes and
never applies them. A level-up FOLLOWING a redirect is not proof the
redirect caused it, so a human applies changes, or nobody does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER

LEDGER_SCHEMA = "carnot.arc.supervisor_refinement_ledger.v1"
RECOMMENDATION_SCHEMA = "carnot.arc.supervisor_refinement_recommendation.v1"
DEFAULT_LEDGER_PARTS = ("ops", "arc_supervisor_refinement_ledger.json")

# Evidence floor. At fired=10 and helped=0, a true follow rate of 0.25 gives
# zero credits with probability 0.75**10 = 0.056 — the smallest count where
# an all-zero record rejects even a modest rate near the 5 percent level.
# Below this, the tool says "insufficient evidence" instead of ranking noise.
MIN_FIRED_PER_ARM = 10

# One-sided 95 percent normal quantile for Wilson score bounds. Stdlib-only
# on purpose: the unattended step must not depend on scipy being installed.
WILSON_Z = 1.6448536269514722

CAUSAL_CAVEAT = (
    "resolved_by_levelup records that a level-up FOLLOWED a redirect inside "
    "the same progress-free span. One level-up credits every pending "
    "redirect (observed live 2026-08-27: two arms credited by one level-up). "
    "It is not evidence of cause. A human applies these, or nobody does."
)

STATUS_NO_RECEIPTS = "no_receipts_ingested"
STATUS_NO_FIRINGS = "no_firings_nothing_to_refine"
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_RECOMMENDATION = "recommendation_available"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def receipt_id_for_row(row: dict[str, Any]) -> str:
    """Hash the FULL source row, not just the receipt. A byte-identical copy
    dedupes; a genuine re-run differs in measured fields (wall time, frame
    counts) and correctly counts as new evidence."""

    return "sha256:" + hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def classify_receipt(row: dict[str, Any]) -> str:
    """One of: applied | shadow | error | other | absent (REQ-6720 rule 1).

    Only an explicit applied receipt is redirect evidence. Shadow receipts
    hold counterfactuals that never ran; ingesting one would count a
    redirect that was never applied — the field-names-lie class."""

    receipt = row.get("trajectory_supervisor")
    if not isinstance(receipt, dict):
        return "absent"
    if "error" in receipt:
        return "error"
    if receipt.get("mode") == "shadow" or "would_have_redirects" in receipt:
        return "shadow"
    if (
        receipt.get("mode") == "applied"
        and receipt.get("enabled") is True
        and isinstance(receipt.get("redirects"), list)
    ):
        return "applied"
    return "other"


def _evidence_from_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    receipt = row["trajectory_supervisor"]
    redirects: list[dict[str, Any]] = []
    for item in receipt.get("redirects") or []:
        if not isinstance(item, dict) or item.get("arm") is None:
            continue
        redirects.append(
            {
                "arm": str(item.get("arm")),
                "action_index": item.get("action_index"),
                "level": item.get("level"),
                "resolved_by_levelup": item.get("resolved_by_levelup") is True,
                "actions_to_levelup": item.get("actions_to_levelup"),
            }
        )
    return {
        "source": source,
        "game": row.get("game"),
        "seed": row.get("seed"),
        "harness_arm": row.get("arm"),
        "window": receipt.get("window"),
        "mode": "applied",
        "actions_observed": receipt.get("actions_observed"),
        "stagnations_unredirected": int(receipt.get("stagnations_unredirected") or 0),
        "levels": row.get("levels"),
        "redirects": redirects,
    }


def extract_rows(doc: Any) -> list[dict[str, Any]]:
    """Accept both shapes the harness has written: a bare list of rows, or
    an object with a `rows` list. Anything else holds no rows."""

    if isinstance(doc, list):
        return [row for row in doc if isinstance(row, dict)]
    if isinstance(doc, dict) and isinstance(doc.get("rows"), list):
        return [row for row in doc["rows"] if isinstance(row, dict)]
    return []


def _walk_rows_files(root: Path) -> Iterator[Path]:
    """Yield rows.json files under root, skipping any directory that holds a
    `.git` entry (REQ-6720 rule 3). A nested repo clone swept by a recursive
    glob once inflated a corpus from 86 rows to 2,212; a worktree marks
    itself with a `.git` FILE, so both forms prune."""

    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames or ".git" in filenames:
            dirnames[:] = []
            continue
        if "rows.json" in filenames:
            yield Path(dirpath) / "rows.json"


def scan_inputs(inputs: Iterable[Path | str]) -> list[Path]:
    """Resolve explicit files plus directory scans into a stable file list.
    A missing input raises: the unattended step must fail loud, never
    silently report clean without reading what it was pointed at."""

    found: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            found.add(path)
        elif path.is_dir():
            found.update(_walk_rows_files(path))
        else:
            raise FileNotFoundError(f"input does not exist: {path}")
    return sorted(found, key=str)


def empty_ledger() -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "created_at": None,
        "updated_at": None,
        "entries": {},
        "recommendation": None,
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
    """Atomic replace so a crash mid-write cannot half-destroy the only
    durable copy of the evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ingest_files(ledger: dict[str, Any], files: Sequence[Path], now_iso: str) -> dict[str, int]:
    counts = {
        "files_read": 0,
        "rows_seen": 0,
        "applied_new": 0,
        "applied_duplicate": 0,
        "shadow_observed": 0,
        "error_rows": 0,
        "other_receipts": 0,
        "rows_without_receipt": 0,
    }
    entries = ledger["entries"]
    for file_path in files:
        doc = json.loads(file_path.read_text(encoding="utf-8"))
        counts["files_read"] += 1
        for row in extract_rows(doc):
            counts["rows_seen"] += 1
            kind = classify_receipt(row)
            if kind == "applied":
                receipt_id = receipt_id_for_row(row)
                if receipt_id in entries:
                    counts["applied_duplicate"] += 1
                else:
                    entry = _evidence_from_row(row, str(file_path))
                    entry["receipt_id"] = receipt_id
                    entry["ingested_at"] = now_iso
                    entries[receipt_id] = entry
                    counts["applied_new"] += 1
            elif kind == "shadow":
                counts["shadow_observed"] += 1
            elif kind == "error":
                counts["error_rows"] += 1
            elif kind == "other":
                counts["other_receipts"] += 1
            else:
                counts["rows_without_receipt"] += 1
    return counts


def wilson_bounds(helped: int, fired: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval. Chosen over a normal approximation because it
    behaves at 0/n and n/n — exactly the counts the retire and raise rules
    read — and needs only math.sqrt."""

    if fired <= 0:
        return (0.0, 1.0)
    phat = helped / fired
    z2 = z * z
    denominator = 1.0 + z2 / fired
    center = phat + z2 / (2.0 * fired)
    margin = z * math.sqrt(phat * (1.0 - phat) / fired + z2 / (4.0 * fired * fired))
    return ((center - margin) / denominator, (center + margin) / denominator)


def _new_arm_cells(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for entry in entries:
        arms_fired = {redirect["arm"] for redirect in entry.get("redirects", [])}
        if entry.get("stagnations_unredirected", 0) > 0 and set(ARM_ORDER) <= arms_fired:
            cells.append(
                {
                    "game": entry.get("game"),
                    "seed": entry.get("seed"),
                    "window": entry.get("window"),
                    "source": entry.get("source"),
                    "levels": entry.get("levels"),
                    "arms_fired": sorted(arms_fired),
                    "stagnations_unredirected": entry.get("stagnations_unredirected"),
                }
            )
    return cells


def evaluate(ledger: dict[str, Any], now_iso: str) -> dict[str, Any]:
    """Apply the frozen contract to the whole ledger. Pure function of the
    ledger content so a re-evaluation after source deletion reproduces the
    same answer (SCENARIO-6720-2)."""

    entries = list(ledger["entries"].values())
    redirects: list[dict[str, Any]] = []
    for entry in entries:
        redirects.extend(entry.get("redirects", []))

    arms_seen = sorted({redirect["arm"] for redirect in redirects} - set(ARM_ORDER))
    arm_names = [*ARM_ORDER, *arms_seen]
    per_arm: list[dict[str, Any]] = []
    for arm in arm_names:
        arm_redirects = [redirect for redirect in redirects if redirect["arm"] == arm]
        fired = len(arm_redirects)
        helped = sum(1 for redirect in arm_redirects if redirect["resolved_by_levelup"])
        lower, upper = wilson_bounds(helped, fired)
        actions = sorted(
            redirect["actions_to_levelup"]
            for redirect in arm_redirects
            if redirect["resolved_by_levelup"] and redirect["actions_to_levelup"] is not None
        )
        per_arm.append(
            {
                "arm": arm,
                "fired": fired,
                "helped": helped,
                "help_follow_rate": round(helped / fired, 6) if fired else None,
                "wilson_lower": round(lower, 6),
                "wilson_upper": round(upper, 6),
                "actions_to_levelup": actions,
                "meets_floor": fired >= MIN_FIRED_PER_ARM,
                "floor_shortfall": max(0, MIN_FIRED_PER_ARM - fired),
            }
        )

    recommendations: list[dict[str, Any]] = []
    for arm_row in per_arm:
        fired = arm_row["fired"]
        helped = arm_row["helped"]
        if fired >= MIN_FIRED_PER_ARM and helped == 0:
            recommendations.append(
                {
                    "kind": "retire_candidate",
                    "arm": arm_row["arm"],
                    "evidence": {
                        "fired": fired,
                        "helped": helped,
                        "wilson_upper": arm_row["wilson_upper"],
                    },
                    "why": (
                        f"{fired} firings and zero follow-ups under a metric that "
                        "shares credit generously; the help-follow rate is below "
                        f"{arm_row['wilson_upper']} at one-sided 95 percent."
                    ),
                }
            )
            continue
        others = [row for row in per_arm if row["arm"] != arm_row["arm"]]
        others_fired = sum(row["fired"] for row in others)
        others_helped = sum(row["helped"] for row in others)
        _, others_upper = wilson_bounds(others_helped, others_fired)
        if (
            fired >= MIN_FIRED_PER_ARM
            and others_fired >= MIN_FIRED_PER_ARM
            and arm_row["wilson_lower"] > others_upper
        ):
            recommendations.append(
                {
                    "kind": "raise_priority_candidate",
                    "arm": arm_row["arm"],
                    "evidence": {
                        "fired": fired,
                        "helped": helped,
                        "wilson_lower": arm_row["wilson_lower"],
                        "others_fired": others_fired,
                        "others_helped": others_helped,
                        "others_wilson_upper": round(others_upper, 6),
                    },
                    "why": (
                        "this arm's lower bound exceeds the pooled other arms' "
                        "upper bound under the same post-hoc crediting bias."
                    ),
                }
            )

    cells = _new_arm_cells(entries)
    new_arm_specification: dict[str, Any] | None = None
    if cells:
        new_arm_specification = {
            "audience": "human",
            "instruction": (
                "Propose ONE new curated arm for ARM_ORDER in "
                "python/carnot/agentic/arc_trajectory_supervisor.py. Arm growth "
                "stays human on a 27B generator; this tool never generates an "
                "arm implementation."
            ),
            "trigger": (
                "every existing arm fired and stagnation continued "
                "(stagnations_unredirected > 0) in the cells below"
            ),
            "cells": cells,
        }

    if not entries:
        status = STATUS_NO_RECEIPTS
    elif not redirects:
        status = STATUS_NO_FIRINGS
    elif recommendations or new_arm_specification:
        status = STATUS_RECOMMENDATION
    else:
        status = STATUS_INSUFFICIENT

    return {
        "schema": RECOMMENDATION_SCHEMA,
        "generated_at": now_iso,
        "status": status,
        "contract": {
            "min_fired_per_arm": MIN_FIRED_PER_ARM,
            "wilson_z": WILSON_Z,
            "rules": [
                "retire_candidate: fired >= floor and helped == 0",
                "raise_priority_candidate: arm and pooled others both at floor, "
                "arm lower bound > others upper bound",
                "new_arm_specification: a receipt fired every arm and still "
                "recorded stagnations_unredirected > 0",
            ],
        },
        "recommendation_only": True,
        "causal_caveat": CAUSAL_CAVEAT,
        "evidence": {
            "receipts": len(entries),
            "redirects": len(redirects),
            "games": sorted({str(entry.get("game")) for entry in entries}),
            "stagnations_unredirected_total": sum(
                int(entry.get("stagnations_unredirected") or 0) for entry in entries
            ),
        },
        "per_arm": per_arm,
        "recommendations": recommendations,
        "new_arm_specification": new_arm_specification,
    }


def render_report(recommendation: dict[str, Any]) -> str:
    """Human report. The insufficient case is deliberately LOUD: a quiet
    weak ranking is the churn mode this tool exists to refuse."""

    lines: list[str] = ["ARC supervisor refinement (REQ-ARC-WMTE-6720)"]
    status = recommendation["status"]
    if status == STATUS_NO_RECEIPTS:
        lines.append(
            "NO RECEIPTS INGESTED — nothing to refine. This honest empty "
            "report satisfies the generalization-floor slot."
        )
    elif status == STATUS_NO_FIRINGS:
        lines.append(
            "NO FIRINGS — the supervisor never redirected. Nothing to "
            "refine; this report satisfies the generalization-floor slot."
        )
    elif status == STATUS_INSUFFICIENT:
        lines.append(
            "INSUFFICIENT EVIDENCE — firings exist but no rule crosses its "
            f"floor (min {MIN_FIRED_PER_ARM} firings per arm). No ranking is "
            "emitted; a ranking at these counts would be noise."
        )
    else:
        lines.append("RECOMMENDATION AVAILABLE — recommendation only; a human applies.")
    evidence = recommendation["evidence"]
    lines.append(
        f"evidence: {evidence['receipts']} receipts, {evidence['redirects']} "
        f"redirects, games={','.join(evidence['games']) or 'none'}"
    )
    for row in recommendation["per_arm"]:
        lines.append(
            f"  {row['arm']}: fired={row['fired']} helped={row['helped']} "
            f"wilson=[{row['wilson_lower']}, {row['wilson_upper']}] "
            f"floor_shortfall={row['floor_shortfall']}"
        )
    for item in recommendation["recommendations"]:
        lines.append(f"  RECOMMEND {item['kind']} arm={item['arm']}: {item['why']}")
    spec = recommendation.get("new_arm_specification")
    if spec:
        lines.append(
            "  NEW ARM SPECIFICATION (for a human): every arm fired and "
            f"stagnation continued in {len(spec['cells'])} cell(s):"
        )
        for cell in spec["cells"]:
            lines.append(
                f"    game={cell['game']} seed={cell['seed']} "
                f"window={cell['window']} "
                f"stagnations_unredirected={cell['stagnations_unredirected']}"
            )
    lines.append(f"caveat: {recommendation['causal_caveat']}")
    return "\n".join(lines)


def _default_ledger_path() -> Path:
    from carnot.paths import repo_path

    return repo_path(*DEFAULT_LEDGER_PARTS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unattended cross-run supervisor refinement (REQ-ARC-WMTE-6720). "
            "Ingests applied trajectory-supervisor receipts from rows.json "
            "files or scan directories into a durable ledger, then prints a "
            "recommendation. Never mutates the arm table."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="rows.json files, or directories scanned for rows.json (nested repo clones pruned)",
    )
    parser.add_argument("--ledger", type=Path, default=None, help="ledger path override")
    parser.add_argument("--json", action="store_true", help="print the recommendation as JSON")
    args = parser.parse_args(argv)

    ledger_path = args.ledger if args.ledger is not None else _default_ledger_path()
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        ledger = load_ledger(ledger_path)
        files = scan_inputs(args.inputs)
        counts = ingest_files(ledger, files, now_iso)
        recommendation = evaluate(ledger, now_iso)
        recommendation["ingest_counts"] = counts
        ledger["recommendation"] = recommendation
        if ledger["created_at"] is None:
            ledger["created_at"] = now_iso
        ledger["updated_at"] = now_iso
        save_ledger(ledger, ledger_path)
    except (OSError, ValueError) as exc:
        # Fail loud (REQ-6720 rule 6): an unattended step that swallows an
        # unreadable input would report clean without having looked.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(recommendation, indent=2, sort_keys=True))
    else:
        print(render_report(recommendation))
    print(f"ledger: {ledger_path}")
    return 0
