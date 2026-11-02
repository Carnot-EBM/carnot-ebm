"""Frozen V675 ARC qualification schedule and cold evidence reduction.

The scored policy and archive implementation come from the qualified V674
modules. This module only fixes custody and the new run's time bound.
"""

from __future__ import annotations

from typing import Any

from carnot.agentic.arc_generalization_runtime import run_episode
from carnot.experiment_7748_v674_arc_runner_qualification import make_agent_factory

GAMES = ("cd82", "dc22", "lf52", "m0r0", "sk48", "tn36", "sb26", "sc25")
SEEDS = (67501, 67502)
ARMS = ("off", "total", "organic")


def schedule_rows() -> list[dict[str, Any]]:
    """Return all 48 intended units before observing any result."""
    return [
        {
            "episode_id": f"{game}:{seed}:{arm}",
            "game": game,
            "seed": seed,
            "arm": arm,
            "max_actions": 2000,
            "max_seconds": 75,
            "status": "unstarted",
            "exclusions": [],
            "censoring": None,
            "raw_path": None,
            "metrics": None,
        }
        for game in GAMES
        for seed in SEEDS
        for arm in ARMS
    ]


def cold_reduce(schedule: list[dict[str, Any]], probes: list[dict[str, Any]]) -> dict[str, int]:
    """Recount complete evidence and reject altered schedules or source counters."""
    if schedule != schedule_rows():
        raise ValueError("schedule_changed")
    ids = [str(row["episode_id"]) for row in probes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_probe")
    actions = 0
    for probe in probes:
        if probe["actions_charged"] != len(probe["actions"]):
            raise ValueError("action_count")
        actions += probe["actions_charged"]
        cells: dict[str, dict[str, int]] = {}
        for event in probe["counter_event_rows"]:
            if event["event"] != "observation":
                continue
            cell = str(event.get("cell", "fixture"))
            counts = cells.setdefault(
                cell, {"seen": 0, "organic_seen": 0, "replay_seen": 0, "reset_seen": 0}
            )
            source = str(event["provenance"])
            if source not in ("organic", "replay", "reset"):
                raise ValueError("counter_provenance")
            counts["seen"] += 1
            counts[f"{source}_seen"] += 1
            if any(event[key] != value for key, value in counts.items()):
                raise ValueError("counter_provenance")
    return {
        "intended": len(schedule),
        "started": len(probes),
        "completed": sum(probe["error"] is None for probe in probes),
        "actions": actions,
    }


def run_probe(game: str, seed: int, arm: str, arcade: Any, max_actions: int) -> dict[str, Any]:
    """Cross the actual scored E3 wrapper with adapter inputs withheld."""
    events: list[dict[str, Any]] = []
    unit = {
        "episode_id": f"{game}:{seed}:{arm}",
        "game": game,
        "seed": seed,
        "arm": arm,
        "max_actions": max_actions,
        "max_seconds": 75,
    }
    episode = run_episode(unit, arcade, agent_factory=make_agent_factory(arm, events))
    return {
        **unit,
        "actions": episode["telemetry"],
        "actions_charged": len(episode["telemetry"]),
        "counter_event_rows": events,
        "raw_metrics": episode["raw_metrics"],
        "counts": episode["counts"],
        "censoring": episode["censoring"],
        "exclusions": episode["exclusions"],
        "error": episode["error"],
        "policy_entry": episode["policy_entry"],
        "solve_provenance": "development_proxy"
        if game == "fixture"
        else "live_agent_self_discovery",
        "new_solve_credit": False,
        "claim_scope": "fixture_only" if game == "fixture" else "adapter_withheld_public",
    }
