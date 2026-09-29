"""Reduce authenticated, unseen live supervisor events. REQ-REPORT-7899-V685."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.agentic.arc_supervisor_refinement import extract_rows, wilson_bounds
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file


def _event_key(row: dict[str, Any], redirect: dict[str, Any]) -> str:
    """Bind a redirect to the run and arm that emitted it."""

    identity = redirect.get("redirect_id", redirect.get("id", redirect.get("action_index")))
    return "|".join(
        str(value)
        for value in (
            row.get("game"),
            row.get("seed"),
            row.get("attempt"),
            identity,
            redirect.get("arm"),
        )
    )


def _excluded(path: Path, reason: str, **extra: Any) -> dict[str, Any]:
    """Keep a failed custody check visible in the primitive ledger."""

    return {"source_path": str(path), "status": "excluded", "reason": reason, **extra}


def reduce_receipts(
    root: Path, producers: list[Path], baseline: dict[str, str], registry: dict[str, int]
) -> dict[str, Any]:
    """Authenticate raw producer bytes before counting any new event."""

    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for producer in producers:
        if not producer.is_file():
            raise FileNotFoundError(producer)
        try:
            document = json.loads(producer.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            rows.append(_excluded(producer, "malformed_producer_json"))
            continue
        if not isinstance(document, dict) or not isinstance(
            document.get("source_artifact_hashes"), dict
        ):
            rows.append(_excluded(producer, "missing_source_hashes"))
            continue
        if document.get("flagged_adversarial") is True or document.get("verdict_class") in {
            "blocked",
            "disqualified",
            "partial",
        }:
            rows.append(_excluded(producer, "unqualified_producer"))
            continue
        for label, expected in document["source_artifact_hashes"].items():
            if (
                not isinstance(label, str)
                or not label.startswith("results/raw/")
                or not label.endswith(".json")
            ):
                continue
            raw = (root / label).resolve()
            if not raw.is_relative_to(root.resolve()):
                rows.append(_excluded(raw, "path_escape"))
                continue
            if not raw.is_file():
                rows.append(_excluded(raw, "missing_raw"))
                continue
            digest = sha256_file(raw)
            declared = expected.get("sha256") if isinstance(expected, dict) else expected
            if digest != declared:
                rows.append(_excluded(raw, "raw_hash_mismatch", source_sha256=digest))
                continue
            if baseline.get("raw:" + digest) == digest:
                rows.append(_excluded(raw, "prior_raw_hash", source_sha256=digest))
                continue
            try:
                raw_document = json.loads(raw.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                rows.append(_excluded(raw, "malformed_raw_json", source_sha256=digest))
                continue
            for episode in extract_rows(raw_document):
                receipt = episode.get("trajectory_supervisor")
                if not isinstance(receipt, dict):
                    continue
                redirects = receipt.get("redirects")
                if receipt.get("mode") != "applied" or not isinstance(redirects, list):
                    rows.append(_excluded(raw, "not_applied", source_sha256=digest))
                    continue
                for redirect in redirects:
                    if (
                        not isinstance(redirect, dict)
                        or any(
                            key not in redirect
                            for key in ("arm", "fired", "resolved_by_levelup", "actions_to_levelup")
                        )
                        or redirect.get("arm") not in ARM_ORDER
                        or redirect.get("fired") is not True
                        or not isinstance(redirect.get("resolved_by_levelup"), bool)
                    ):
                        rows.append(_excluded(raw, "malformed_redirect", source_sha256=digest))
                        continue
                    key = _event_key(episode, redirect)
                    content_hash = canonical_hash({"episode": episode, "redirect": redirect})
                    row = {
                        "family": "trajectory_supervisor",
                        "unit": key,
                        "arm": redirect["arm"],
                        "seed": episode.get("seed"),
                        "game": episode.get("game"),
                        "event_id": key,
                        "content_sha256": content_hash,
                        "source_path": str(raw),
                        "source_sha256": digest,
                        "producer_path": str(producer),
                        "producer_sha256": sha256_file(producer),
                        "fired": True,
                        "resolved_by_levelup": redirect["resolved_by_levelup"],
                        "actions_to_levelup": redirect["actions_to_levelup"],
                        "stagnations_unredirected": receipt.get("stagnations_unredirected"),
                        "solve_provenance": episode.get(
                            "solve_provenance", episode.get("provenance")
                        ),
                        "new_level_solve_credited": False,
                    }
                    if key in baseline:
                        row.update(
                            status="excluded",
                            reason="prior_event"
                            if baseline[key] == content_hash
                            else "revised_prior_event",
                        )
                    elif key in seen:
                        row.update(
                            status="excluded",
                            reason="duplicate_event"
                            if seen[key] == content_hash
                            else "revised_duplicate_event",
                        )
                    elif row["solve_provenance"] != "live_agent_self_discovery":
                        row.update(status="excluded", reason="non_live_provenance")
                    elif (
                        episode.get("game") is None
                        or episode.get("seed") is None
                        or episode.get("attempt") is None
                    ):
                        row.update(status="excluded", reason="missing_identity")
                    else:
                        reason = (episode.get("termination") or {}).get("reason")
                        censored = not redirect["resolved_by_levelup"] and reason in {
                            "action_limit",
                            "time_limit",
                            "timeout",
                            "collection_cap",
                        }
                        row.update(
                            status="censored" if censored else "completed",
                            reason=reason if censored else None,
                        )
                        seen[key] = content_hash
                    rows.append(row)
    live = [row for row in rows if row["status"] in {"completed", "censored"}]
    per_game: dict[str, dict[str, Any]] = {}
    for row in live:
        cell = per_game.setdefault(str(row["game"]), {"eligible": 0, "arms": {}})
        cell["eligible"] += 1
        arm = cell["arms"].setdefault(
            row["arm"],
            {
                "firings": 0,
                "resolved_by_levelup": 0,
                "actions_to_levelup": [],
                "stagnations_unredirected": [],
                "source_run_hashes": [],
            },
        )
        arm["firings"] += 1
        arm["resolved_by_levelup"] += int(row["resolved_by_levelup"])
        if isinstance(row["actions_to_levelup"], int):
            arm["actions_to_levelup"].append(row["actions_to_levelup"])
        if isinstance(row["stagnations_unredirected"], int):
            arm["stagnations_unredirected"].append(row["stagnations_unredirected"])
        if row["source_sha256"] not in arm["source_run_hashes"]:
            arm["source_run_hashes"].append(row["source_sha256"])
    recommendations = []
    for arm in sorted({row["arm"] for row in live}):
        selected = [row for row in live if row["arm"] == arm]
        games = {row["game"] for row in selected}
        if len(selected) >= 20 and len(games) >= 3:
            closed = [row for row in selected if row["status"] == "completed"]
            low, high = wilson_bounds(
                sum(row["resolved_by_levelup"] for row in closed), len(closed)
            )
            recommendations.append(
                {
                    "arm": arm,
                    "fired": len(selected),
                    "games": len(games),
                    "resolved": len(closed),
                    "censored": len(selected) - len(closed),
                    "outcome_interval": [low, high],
                    "recommendation": "review_selection_or_retirement",
                    "causal_claim": False,
                    "production_default_changed": False,
                }
            )
    counts = {
        state: sum(row["status"] == state for row in rows)
        for state in ("completed", "failed", "censored", "excluded")
    }
    return {
        "rows": rows,
        "new_outcome_count": len(live),
        "firings": len(live),
        "per_game_results": per_game,
        "recommendation_rows": recommendations,
        "sample_size_budget": {
            "intended": len(rows),
            "eligible": len(live),
            "started": len(live),
            **counts,
            "independent_n": len({row["event_id"] for row in live}),
        },
        "new_level_solves": 0,
        "registry_precheck": registry,
    }


def replay_delta(delta: dict[str, Any]) -> list[str]:
    """Recount claims from primitive rows without rerunning the source scan."""

    live = [row for row in delta["rows"] if row["status"] in {"completed", "censored"}]
    errors = []
    if len(live) != delta["new_outcome_count"]:
        errors.append("new_outcome_count")
    if sum(row.get("fired") is True for row in live) != delta["firings"]:
        errors.append("firings")
    return errors
