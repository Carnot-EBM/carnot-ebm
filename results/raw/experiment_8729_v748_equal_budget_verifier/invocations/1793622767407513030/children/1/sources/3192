"""Direct scored ARC qualification controls for REQ-ARC-WMTE-7817."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.agentic.arc_competition_agent import make_carnot_agent
from carnot.agentic.arc_generalization_runtime import run_episode
from carnot.agentic.arc_go_explore import (
    GoExploreReplayArchive,
    _coarse_cell,
    _frame_grid,
    _frame_level,
)
from carnot.experiment_7803_v678_arc_runner_qualification import (
    freeze_panel as historical_panel_check,
)


FROZEN_MANIFEST_SHA256 = "f95c233d82006e797248345e666155ba1ce9a29a21a52d7e14b75c24f1f5a364"
ALLOWED_NAMES = (
    *(f"probe_{game}_{arm}" for game in ("r11l", "cd82") for arm in ("off", "total", "organic")),
    "worktree_imports",
    "focused_pytest",
    "changed_module_coverage",
    "cli_coverage",
    "coverage_combine",
    "changed_module_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "e2e_009",
    "e2e_011",
    "e2e_013",
    "e2e_009_smoke",
    "cold_reduce",
    "repository_health_full_python_suite",
    "adversarial_verify",
    "strict_row_lint",
    "cold_replay",
)


def sha256(path: Path) -> str:
    """Identify completed bytes, rather than a mutable live log path."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_plan(path: Path) -> list[dict[str, Any]]:
    """Reject every child outside the prospective, byte-frozen command list."""
    if sha256(path) != FROZEN_MANIFEST_SHA256:
        raise ValueError("undeclared_child_or_manifest_mutation")
    manifest = json.loads(path.read_text())
    commands = manifest["commands"]
    if tuple(row["name"] for row in commands) != ALLOWED_NAMES:
        raise ValueError("undeclared_child")
    if [row["name"] for row in commands if row["classification"] == "required"] != manifest[
        "required_checks"
    ]:
        raise ValueError("required_classification_changed")
    return [
        {"name": row["name"], "argv": row["argv"], "classification": row["classification"]}
        for row in commands
    ]


def seal_log(parent: Path, name: str, attempt: int, data: bytes) -> Path:
    """Copy completed child output once to an attempt-specific durable file."""
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    digest = hashlib.sha256(data).hexdigest()
    path = parent / f"{attempt:04d}_{name}_{digest}.log"
    with path.open("xb") as stream:
        stream.write(data)
    return path


def verify_log(path: Path, expected_sha256: str) -> bool:
    """Cold readers detect a changed byte after the receipt was sealed."""
    return path.is_file() and sha256(path) == expected_sha256


def validate_receipts(
    path: Path, receipts: Sequence[Mapping[str, Any]], *, sdk_ok: bool
) -> list[str]:
    """A missing, duplicate, failed, or timed-out required child closes readiness."""
    required = [row["name"] for row in command_plan(path) if row["classification"] == "required"]
    failed = [
        name
        for name in required
        if len(matches := [row for row in receipts if row.get("name") == name]) != 1
        or matches[0].get("exit_code") != 0
        or matches[0].get("passed") is not True
        or matches[0].get("timed_out") is True
    ]
    return [*failed, *([] if sdk_ok else ["sdk_transport"])]


def freeze_panel(path: Path) -> dict[str, Any]:
    """Read the immutable V677 schedule without depending on the mutable roadmap."""
    return historical_panel_check(path)


class ObservedArchive(GoExploreReplayArchive):
    """Record the source of a frame while using the current archive selector."""

    def __init__(self, arm: str, events: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.arm = arm
        self.events = events

    def observe(
        self, frame: Any, path: Sequence[Mapping[str, Any]] | None, *, provenance: str = "organic"
    ) -> None:
        """Count replay and reset separately at the observation call."""
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
                }
            )

    def _select_via_selector(
        self, eligible_items: list[tuple[tuple, dict[str, Any]]]
    ) -> dict[str, Any] | None:
        """Only the explicit local total arm changes the ranking count."""
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
        """Retain which cell was actually selected, after eligibility checks."""
        prefix = super().select_prefix(current_path=current_path)
        if prefix:
            self.events.append(
                {
                    "event": "selection",
                    "arm": self.arm,
                    "cell": repr(self.last_selected_cell),
                    "prefix_length": len(prefix),
                }
            )
        return prefix


def make_agent_factory(arm: str, events: list[dict[str, Any]]) -> Any:
    """Use the scored E3 factory; keep archive options local to this probe."""
    if arm not in {"off", "total", "organic"}:
        raise ValueError("unknown_arm")

    def factory(base: type, **kwargs: Any) -> type:
        parent = make_carnot_agent(base, organic_visits=arm != "off", **kwargs)
        if arm == "off":
            return parent

        class ObservedAgent(parent):
            def __init__(self, *args: Any, **agent_kwargs: Any) -> None:
                super().__init__(*args, **agent_kwargs)
                self._policy.explorer.go_explore_archive = ObservedArchive(
                    arm, events, bins=6, max_cells=256, selector="organic_visits"
                )

        return ObservedAgent

    return factory


def positive_selector_fixture() -> dict[str, Any]:
    """Use visible frames to prove replay changes one ranking without solve credit."""

    def frame(color: int) -> SimpleNamespace:
        return SimpleNamespace(
            frame=np.asarray([[color, 0], [0, 0]], dtype=np.int16), levels_completed=0
        )

    evidence: dict[str, Any] = {"off_archive": None, "new_solve_credit": False}
    for arm in ("total", "organic"):
        events: list[dict[str, Any]] = []
        archive = ObservedArchive(arm, events, bins=2, selector="organic_visits")
        for color in (2, 3):
            archive.observe(frame(color), [{"action": color, "data": None}])
        cell = _coarse_cell(frame(2).frame, 0, bins=2)
        before = archive._cells[cell]["organic_seen"]
        for _ in range(4):
            archive.observe(frame(2), [{"action": 2, "data": None}], provenance="replay")
        after = archive._cells[cell]["organic_seen"]
        replay = archive._cells[cell]["replay_seen"]
        archive.observe(frame(3), [{"action": 3, "data": None}])
        for color in (2, 3):
            archive._cells[_coarse_cell(frame(color).frame, 0, bins=2)]["visits"] = 2
        evidence[f"{arm}_prefix"] = archive.select_prefix()
        evidence[f"{arm}_events"] = events
        evidence["organic_seen_before_replay"] = before
        evidence["organic_seen_after_replay"] = after
        evidence["replay_seen_after_replay"] = replay
    return evidence


def run_probe(game: str, arm: str, arcade: Any) -> dict[str, Any]:
    """Drive real SDK transitions through E3 without calling an old experiment main."""
    import random

    random.seed(67501)
    np.random.seed(67501)
    events: list[dict[str, Any]] = []
    unit = {
        "episode_id": f"{game}:67501:{arm}",
        "game": game,
        "seed": 67501,
        "arm": arm,
        "max_actions": 12,
        "max_seconds": 75,
    }
    episode = run_episode(unit, arcade, agent_factory=make_agent_factory(arm, events))
    actions = episode["telemetry"]
    return {
        **unit,
        "actions": actions,
        "actions_charged": len(actions),
        "counter_event_rows": events,
        "raw_metrics": episode["raw_metrics"],
        "counts": episode["counts"],
        "censoring": episode["censoring"],
        "exclusions": episode["exclusions"],
        "error": episode["error"],
        "policy_entry": episode["policy_entry"],
        "solve_provenance": "live_agent_self_discovery",
        "new_solve_credit": False,
        "claim_scope": "adapter_withheld_public_transport_only",
    }


def sdk_probe_ok(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require actual observed transitions for each game and local arm."""
    expected = {(game, arm) for game in ("r11l", "cd82") for arm in ("off", "total", "organic")}
    if (
        len(rows) != len(expected)
        or {(row.get("game"), row.get("arm")) for row in rows} != expected
    ):
        return False
    return all(
        row.get("error") is None
        and row.get("new_solve_credit") is False
        and row.get("counts", {}).get("sdk_transitions", 0) > 0
        and row.get("policy_entry", {}).get("policy_class") == "E3AgentPolicy"
        and all(
            action.get("induction_attempt_count") == 0
            and action.get("actual_observation", {}).get("frame_sha256")
            for action in row.get("actions", [])
        )
        for row in rows
    )
