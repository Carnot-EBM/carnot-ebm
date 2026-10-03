"""Count only authenticated supervisor outcomes after a sealed snapshot.

The older receipt parser checks producer hashes and attempt identity. This
reader adds the current cutoff and the live solve registry boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting.arc_supervisor_receipt_delta import _receipt_lookup, reduce_ledger
from carnot.reporting.current_work_receipt import sha256_file


def check_inputs(
    prior: Path, source: Path, registry: Path, expected: tuple[str, str, str]
) -> dict[str, Any]:
    """Pin exact bytes before scanning because a moving source invalidates N."""

    checks: list[dict[str, Any]] = []
    for path, digest, role in zip(
        (prior, source, registry),
        expected,
        ("historical_cutoff", "historical_code", "live_registry_precheck"),
        strict=True,
    ):
        actual = sha256_file(path) if path.is_file() else "missing"
        checks.append(
            {
                "upstream_id": path.stem,
                "path": str(path),
                "sha256": None if actual == "missing" else actual,
                "artifact_field": "sha256",
                "op": "==",
                "expected": digest,
                "observed": actual,
                "role": role,
                "exposure_status": "read_only",
            }
        )
    failures = [row for row in checks if row["expected"] != row["observed"]]
    levels: dict[str, int] = {}
    prior_hashes: set[str] = set()
    cutoff = 0
    if not failures:
        snapshot = json.loads(prior.read_text(encoding="utf-8"))
        parsed = yaml.safe_load(registry.read_text(encoding="utf-8"))
        for path, field, observed, wanted in (
            (prior, "verdict_class", snapshot.get("verdict_class"), "null"),
            (registry, "schema_version", parsed.get("schema_version"), 1),
        ):
            check = {
                "upstream_id": path.stem,
                "path": str(path),
                "sha256": sha256_file(path),
                "artifact_field": field,
                "op": "==",
                "expected": wanted,
                "observed": observed,
                "role": "historical_cutoff" if path == prior else "live_registry_precheck",
                "exposure_status": "read_only",
            }
            checks.append(check)
            if observed != wanted:
                failures.append(check)
        games = parsed.get("games", [])
        iterable = games.values() if isinstance(games, dict) else games
        levels = {
            str(item["game"]): int(item["levels_reproduced"])
            for item in iterable
            if isinstance(item, dict) and "game" in item and "levels_reproduced" in item
        }
        if isinstance(games, dict):
            levels = {
                str(name): int(item["levels_reproduced"])
                for name, item in games.items()
                if isinstance(item, dict) and "levels_reproduced" in item
            }
        prior_hashes = {
            value
            for value in snapshot.get("source_artifact_hashes", {}).values()
            if isinstance(value, str)
        }
        cutoff = max(prior.stat().st_mtime_ns, source.stat().st_mtime_ns)
    return {
        "checks": checks,
        "failures": failures,
        "registry_levels": levels,
        "prior_hashes": prior_hashes,
        "cutoff_ns": cutoff,
    }


def summarize(
    root: Path, cutoff_ns: int, prior_hashes: set[str], registry_levels: dict[str, int]
) -> dict[str, Any]:
    """Keep screened rows and measure gains only from unique live attempts."""

    delta = reduce_ledger(root, cutoff_ns, prior_hashes)
    games: dict[str, dict[str, Any]] = {}
    solves = 0
    lookups: dict[str, dict[tuple[Any, Any, Any], dict[str, Any]]] = {}
    for row in delta["outcome_rows"]:
        if row.get("status") not in {"completed", "censored"}:
            continue
        game = str(row.get("game"))
        cell = games.setdefault(
            game,
            {
                "eligible": 0,
                "level_progress": 0,
                "gains": 0,
                "regressions": 0,
                "lost_wins": 0,
                "registry_levels_reproduced": registry_levels.get(game, 0),
                "mechanism_hashes": [],
            },
        )
        cell["eligible"] += 1
        if row.get("resolved_by_levelup") is True:
            cell["level_progress"] += 1
        if row.get("helped") is True:
            cell["gains"] += 1
        if row.get("helped") is False:
            cell["regressions"] += 1
        if row.get("resolved_by_levelup") is False:
            cell["lost_wins"] += 1
        label = row.get("source_path")
        if isinstance(label, str):
            raw = (root / label).resolve()
            if raw.is_file() and raw.is_relative_to(root.resolve()):
                original = lookups.setdefault(label, _receipt_lookup(raw)).get(
                    (row.get("game"), row.get("seed"), row.get("redirect_id"))
                )
                if original is not None:
                    episode = original["row"]
                    level = episode.get("levels_completed")
                    mechanism = episode.get("mechanism_hash")
                    row["levels_completed"] = level
                    row["mechanism_hash"] = mechanism
                    row["score_numerator"] = episode.get("score_numerator")
                    row["score_denominator"] = episode.get("score_denominator")
                    if isinstance(mechanism, str) and mechanism not in cell["mechanism_hashes"]:
                        cell["mechanism_hashes"].append(mechanism)
                    if (
                        isinstance(level, int)
                        and level > registry_levels.get(game, 0)
                        and isinstance(mechanism, str)
                        and row.get("solve_provenance") == "live_agent_self_discovery"
                        and row.get("resolved_by_levelup") is True
                        and episode.get("live_route_reachable") is True
                    ):
                        row["new_level_solve_credited"] = True
                        solves += 1
        row["registry_levels_reproduced"] = registry_levels.get(game, 0)
        row.setdefault("new_level_solve_credited", False)
    delta["new_live_outcome_count"] = delta["sample_size_budget"]["eligible"]
    delta["new_level_solves"] = solves
    delta["per_game_results"] = games
    return delta
