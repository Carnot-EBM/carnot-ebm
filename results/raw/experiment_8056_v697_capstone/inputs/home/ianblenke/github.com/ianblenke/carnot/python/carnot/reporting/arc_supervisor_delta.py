"""Read authenticated live supervisor receipts without starting new game episodes.

The source producer controls eligibility. A copied or changed raw file cannot add
independent evidence. Following a redirect is an association, not a causal effect.
REQ-REPORT-7845 and SCENARIO-REPORT-7845-RECEIPT define this boundary.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

from carnot.agentic.arc_supervisor_refinement import classify_receipt, extract_rows, wilson_bounds
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.reporting.current_work_receipt import sha256_file


def verify_prior(path: Path, expected_hash: str) -> list[dict[str, Any]]:
    """Report the exact missing or changed baseline operand before reading a delta."""

    observed = sha256_file(path) if path.is_file() else "missing"
    if observed == expected_hash:
        return []
    return [
        {
            "upstream_id": "exp7831-arc-supervisor-refinement",
            "path": str(path),
            "sha256": observed if observed != "missing" else None,
            "artifact_field": "sha256",
            "op": "==",
            "expected": expected_hash,
            "observed": observed,
        }
    ]


def _source_rows(
    root: Path, producer: Path, baseline_date: str, prior_hashes: set[str]
) -> list[dict[str, Any]]:
    """Use producer hashes and date to decide whether raw bytes can enter the delta."""

    try:
        document = json.loads(producer.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [
            {
                "producer_path": producer.relative_to(root).as_posix(),
                "producer_sha256": sha256_file(producer),
                "status": "excluded",
                "reason": "malformed_producer",
            }
        ]
    if not isinstance(document, dict) or str(document.get("run_date", "")) <= baseline_date:
        return []
    hashes = document.get("source_artifact_hashes", {})
    if not isinstance(hashes, dict):
        return []
    output: list[dict[str, Any]] = []
    for label, expected in hashes.items():
        if (
            not isinstance(label, str)
            or not label.startswith("results/raw/")
            or not label.endswith(".json")
        ):
            continue
        raw = (root / label).resolve()
        if not raw.is_relative_to(root.resolve()):
            continue
        record: dict[str, Any] = {
            "source_path": label,
            "producer_path": producer.relative_to(root).as_posix(),
            "producer_sha256": sha256_file(producer),
            "run_date": document["run_date"],
            "role": "science_producer",
        }
        output.append(record)
        if not raw.is_file():
            record.update(status="excluded", reason="missing_raw")
            continue
        actual = sha256_file(raw)
        record["source_sha256"] = actual
        expected_hash = expected.get("sha256") if isinstance(expected, dict) else expected
        if isinstance(expected_hash, str) and not expected_hash.startswith("sha256:"):
            expected_hash = "sha256:" + expected_hash
        if actual != expected_hash:
            record.update(
                status="excluded", reason="raw_hash_mismatch", expected_sha256=expected_hash
            )
            continue
        if actual in prior_hashes:
            record.update(status="excluded", reason="prior_hash_inventory")
            continue
        if document.get("flagged_adversarial") is True or document.get("verdict_class") in {
            "blocked",
            "disqualified",
            "partial",
        }:
            record.update(status="excluded", reason="disqualified_producer")
            continue
        raw_doc = json.loads(raw.read_text(encoding="utf-8"))
        receipt_seen = False
        eligible_firing = False
        for row in extract_rows(raw_doc):
            receipt = row.get("trajectory_supervisor")
            if not isinstance(receipt, dict):
                continue
            receipt_seen = True
            item = dict(record)
            item.update(game=row.get("game"), seed=row.get("seed"))
            kind = classify_receipt(row)
            if kind == "shadow":
                item.update(status="shadow", reason="unfired_control")
                output.append(item)
                continue
            if kind != "applied":
                item.update(status="excluded", reason="not_applied")
                output.append(item)
                continue
            redirects = receipt.get("redirects")
            if not isinstance(redirects, list) or not redirects:
                item.update(status="excluded", reason="no_firings")
                output.append(item)
                continue
            for redirect in redirects:
                firing = dict(item)
                if (
                    not isinstance(redirect, dict)
                    or not all(
                        key in redirect
                        for key in ("arm", "resolved_by_levelup", "actions_to_levelup")
                    )
                    or redirect.get("arm") not in ARM_ORDER
                    or not isinstance(redirect.get("resolved_by_levelup"), bool)
                ):
                    firing.update(status="excluded", reason="malformed_redirect")
                else:
                    reason = (row.get("termination") or {}).get("reason")
                    censored = redirect["resolved_by_levelup"] is False and reason in {
                        "action_limit",
                        "time_limit",
                        "timeout",
                        "collection_cap",
                    }
                    firing.update(
                        status="censored" if censored else "completed",
                        reason=reason if censored else None,
                        redirect_id=redirect.get(
                            "redirect_id", redirect.get("id", redirect.get("action_index"))
                        ),
                        arm=redirect["arm"],
                        resolved_by_levelup=redirect["resolved_by_levelup"],
                        actions_to_levelup=redirect["actions_to_levelup"],
                        stagnations_unredirected=receipt.get("stagnations_unredirected"),
                        arms_enabled=receipt.get("arms_enabled"),
                        arms_used=receipt.get("arms_used"),
                    )
                    eligible_firing = True
                output.append(firing)
        record.update(
            status="source" if eligible_firing else "excluded",
            reason=None
            if eligible_firing
            else ("no_eligible_firing" if receipt_seen else "no_supervisor_receipt"),
        )
    return output


def inspect_sources(root: Path, baseline_date: str, prior_hashes: set[str]) -> list[dict[str, Any]]:
    """Enumerate current producers and keep every screened receipt disposition."""

    rows: list[dict[str, Any]] = []
    for producer in sorted((root / "results").glob("experiment_*_*.json")):
        match = re.match(r"experiment_(\d+)_", producer.name)
        if match is None or int(match.group(1)) <= 7831:
            continue
        rows.extend(_source_rows(root, producer, baseline_date, prior_hashes))
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("status") not in {"completed", "censored"}:
            continue
        key = (
            row.get("source_sha256"),
            row.get("game"),
            row.get("seed"),
            row.get("redirect_id"),
            row.get("arm"),
        )
        if key in seen:
            row.update(status="excluded", reason="duplicate_retry")
        seen.add(key)
    return rows


def reduce_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count observed closed outcomes and apply the frozen conservative arm rule."""

    source_count = len({r["source_sha256"] for r in rows if r.get("status") == "source"})
    fires = [r for r in rows if r.get("status") in {"completed", "censored"}]
    per_game: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"closed": 0, "progress": 0, "censored": 0})
    )
    arms: dict[str, dict[str, Any]] = {}
    recommendations: list[dict[str, Any]] = []
    for arm in sorted({r["arm"] for r in fires}):
        selected = [r for r in fires if r["arm"] == arm]
        closed = [r for r in selected if r["status"] == "completed"]
        progress = sum(r["resolved_by_levelup"] is True for r in closed)
        games = {str(r["game"]) for r in closed}
        lower, upper = wilson_bounds(progress, len(closed))
        arms[arm] = {
            "fired": len(selected),
            "closed": len(closed),
            "censored": len(selected) - len(closed),
            "progress": progress,
            "games": len(games),
            "wilson_lower": lower,
            "wilson_upper": upper,
        }
        if len(closed) >= 20 and len(games) >= 3 and progress == 0:
            recommendations.append({"arm": arm, "kind": "shadow_retirement", "causal_claim": False})
        for row in selected:
            cell = per_game[str(row["game"])][arm]
            cell["censored" if row["status"] == "censored" else "closed"] += 1
            cell["progress"] += int(row["resolved_by_levelup"] is True)
    exhausted = any(
        isinstance(r.get("arms_enabled"), list)
        and set(r["arms_enabled"]) <= set(r.get("arms_used") or [])
        and isinstance(r.get("stagnations_unredirected"), int)
        and r["stagnations_unredirected"] > 0
        for r in fires
    )
    return {
        "honest_verdict": "complete_null_no_new_supervisor_outcomes"
        if source_count == 0
        else "complete_null_observational_supervisor_delta",
        "verdict_class": "null",
        "new_source_count": source_count,
        "per_game_results": {game: dict(arm_rows) for game, arm_rows in sorted(per_game.items())},
        "arm_statistics": arms,
        "recommendation_rows": recommendations,
        "general_mechanism_requirement": "Reusable game-blind search constraint after all arms are spent"
        if exhausted
        else None,
    }
