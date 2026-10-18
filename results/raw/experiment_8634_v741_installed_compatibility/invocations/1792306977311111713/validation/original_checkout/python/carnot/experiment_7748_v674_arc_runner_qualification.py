"""Task-owned scored ARC controls and custody for REQ-ARC-WMTE-7748."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import random
from typing import Any

import numpy as np

from carnot.agentic.arc_competition_agent import make_carnot_agent
from carnot.agentic.arc_go_explore import (
    GoExploreReplayArchive,
    _coarse_cell,
    _frame_grid,
    _frame_level,
)
from carnot.agentic.arc_generalization_runtime import run_episode


class QualificationArchive(GoExploreReplayArchive):
    """Keep the production replay path while varying only its cell ranking."""

    def __init__(self, arm: str, events: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.arm = arm
        self.events = events

    def observe(
        self,
        frame: Any,
        path: Sequence[Mapping[str, Any]] | None,
        *,
        provenance: str = "organic",
    ) -> None:
        """Record the source at the observation call, before the next action changes state."""
        before = self._observations
        super().observe(frame, path, provenance=provenance)
        if self._observations > before:
            key = _coarse_cell(_frame_grid(frame), _frame_level(frame), bins=self.bins)
            entry = self._cells[key]
            self.events.append(
                {
                    "event": "observation",
                    "provenance": provenance,
                    "cell": repr(key),
                    "seen": entry["seen"],
                    "organic_seen": entry["organic_seen"],
                    "replay_seen": entry["replay_seen"],
                    "reset_seen": entry["reset_seen"],
                    "observation_index": self._observations,
                }
            )

    def _select_via_selector(
        self, eligible_items: list[tuple[tuple, dict[str, Any]]]
    ) -> dict[str, Any] | None:
        """Use the same production tie order with total or organic sightings."""
        if self.arm == "organic":
            return super()._select_via_selector(eligible_items)
        if len(eligible_items) < 2:
            return None
        self._selector_calls += 1
        chosen = min(
            (entry for _, entry in eligible_items),
            key=lambda entry: (
                int(entry.get("visits", 0)) - int(entry.get("seen", 0)),
                -int(entry.get("depth", 0)),
                tuple((step["action"], repr(step.get("data"))) for step in entry["prefix"]),
            ),
        )
        self._selector_used += 1
        return chosen

    def select_prefix(
        self, *, current_path: Sequence[Mapping[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        prefix = super().select_prefix(current_path=current_path)
        if prefix:
            self.events.append(
                {
                    "event": "selection",
                    "arm": self.arm,
                    "cell": repr(self.last_selected_cell),
                    "prefix_length": len(prefix),
                    "selected_prefixes": self._selected_prefixes,
                }
            )
        return prefix

    @classmethod
    def restore(
        cls, snapshot: Mapping[str, Any], arm: str, events: list[dict[str, Any]]
    ) -> QualificationArchive:
        """Reload the production counter schema and reattach the task-only selector."""
        restored = GoExploreReplayArchive.from_snapshot(snapshot)
        archive = cls(arm, events, bins=restored.bins, selector=restored.selector)
        archive.__dict__.update(restored.__dict__)
        return archive


def make_agent_factory(arm: str, events: list[dict[str, Any]]) -> Any:
    """Build every arm through the same scored E3 policy factory."""
    if arm not in {"off", "total", "organic"}:
        raise ValueError("unknown_arm")

    def factory(base: type, **kwargs: Any) -> type:
        parent = make_carnot_agent(base, organic_visits=arm != "off", **kwargs)
        if arm == "off":
            return parent

        class ObservedAgent(parent):
            def __init__(self, *args: Any, **agent_kwargs: Any) -> None:
                super().__init__(*args, **agent_kwargs)
                self._policy.explorer.go_explore_archive = QualificationArchive(
                    arm, events, bins=6, max_cells=256, selector="organic_visits"
                )

        return ObservedAgent

    return factory


def run_qualification_episode(
    game: str, seed: int, arm: str, arcade: Any, max_actions: int
) -> dict[str, Any]:
    """Drive the scored wrapper using only SDK frames and retain each event."""
    random.seed(seed)
    np.random.seed(seed)
    events: list[dict[str, Any]] = []
    unit = {
        "game": game,
        "seed": seed,
        "arm": arm,
        "episode_id": f"{game}:{seed}:{arm}",
        "max_actions": max_actions,
        "max_seconds": 45,
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


def cold_reduce(rows: list[dict[str, Any]], intended: int) -> dict[str, int]:
    """Reject dropped schedule rows and counts that disagree with raw actions."""
    if len(rows) != intended or len({row["episode_id"] for row in rows}) != intended:
        raise ValueError("schedule_row_count")
    for row in rows:
        if row["actions_charged"] != len(row["actions"]):
            raise ValueError("action_count")
        if row["new_solve_credit"]:
            raise ValueError("duplicate_solve_credit")
    return {
        "intended": intended,
        "started": sum(row.get("status") != "unstarted" for row in rows),
        "completed": sum(row.get("status") != "unstarted" and row["error"] is None for row in rows),
        "actions": sum(row["actions_charged"] for row in rows),
    }
