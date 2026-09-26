"""Select bounded information probes from this episode's visible ARC observations.

REQ-ARC-PROBE-7680. A changed pixel suggests a testable goal, not a win rule.
The SDK level counter and the existing two-sided contract retain goal authority.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence
import time

import numpy as np

from carnot.agentic.arc_active_reward_machine_frontier import (
    FRAME_CHANGED_NO_LEVEL,
    SAME_FRAME_NO_LEVEL,
    RewardMachineFrontier,
    RewardMachineHypothesis,
    RewardMachineTransition,
    TransitionEvidence,
)
from carnot.agentic.arc_two_sided_goal_contract import TwoSidedGoalEvidenceContract


def _visible_grid(frame: Any) -> list[list[int]] | None:
    """Copy the public final layer so later SDK mutation cannot rewrite evidence."""
    try:
        layers = getattr(frame, "frame", None)
        grid = np.asarray(layers[-1] if isinstance(layers, (list, tuple)) else layers)
        if grid.ndim != 2 or grid.size == 0 or grid.size > 4096:
            return None
        return grid.astype(int).tolist()
    except (TypeError, ValueError, IndexError, AttributeError):
        return None


def _level(frame: Any) -> int | None:
    try:
        return int(getattr(frame, "levels_completed"))
    except (TypeError, ValueError, AttributeError):
        return None


class ArcProbeProtocol:
    """Keep a tiny episode bank and ask the existing frontier for one legal test."""

    def __init__(self, mode: str = "guided") -> None:
        if mode not in {"guided", "novelty"}:
            raise ValueError("mode must be guided or novelty")
        self.mode = mode
        self.pending: dict[str, Any] | None = None
        self.effects: list[dict[str, Any]] = []
        self.goal_hypotheses: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.observed_actions = 0
        self.sdk_progress = 0
        self.goal_confirmations = 0
        self.virtual_engine_calls = 0
        self.tick = 0
        self.frontier = RewardMachineFrontier(capacity=2, timeout_ticks=4)
        self.contract = TwoSidedGoalEvidenceContract()

    def record_action(self, frame: Any, move: tuple[Any, Any]) -> None:
        """Freeze the actual dispatched action before its result is visible."""
        grid = _visible_grid(frame)
        level = _level(frame)
        action, data = move
        if grid is None or level is None or not isinstance(action, int) or data is not None:
            self.pending = None
            return
        self.pending = {"grid": grid, "level": level, "action": action, "frame_id": id(frame)}

    def observe(self, frame: Any) -> None:
        """Consume one real next observation; a level-up starts a new bank."""
        pending = self.pending
        if pending is None:
            return
        if id(frame) == pending["frame_id"]:
            self.pending = None
            return
        after = _visible_grid(frame)
        level = _level(frame)
        if after is None or level is None or np.shape(after) != np.shape(pending["grid"]):
            self.pending = None
            return
        self.tick += 1
        self.observed_actions += 1
        if level > pending["level"]:
            self.sdk_progress += 1
            self.effects.clear()
            self.goal_hypotheses.clear()
            self.frontier = RewardMachineFrontier(capacity=2, timeout_ticks=4)
            self.pending = None
            return
        if level < pending["level"]:
            self.pending = None
            return
        before_array = np.asarray(pending["grid"])
        after_array = np.asarray(after)
        changed = int(np.count_nonzero(before_array != after_array))
        action = int(pending["action"])
        self.effects.append({"action": action, "changed": changed, "tick": self.tick})
        self.effects = self.effects[-16:]
        counts = Counter(int(value) for value in after_array.flat)
        for hypothesis in self.goal_hypotheses:
            count = counts[int(hypothesis["value"])]
            reached = (
                count >= hypothesis["threshold"]
                if hypothesis["direction"] > 0
                else count <= hypothesis["threshold"]
            )
            if reached:
                hypothesis["state"] = "refuted"
        if changed:
            self._add_hypotheses(before_array, after_array, action)
        if self.frontier.diagnostics()["pending_probe"]:
            self.frontier.observe_action_result(
                action=action,
                tick=self.tick,
                level_before=level,
                level_after=level,
                frame_before_hash=str(hash(before_array.tobytes())),
                frame_after_hash=str(hash(after_array.tobytes())),
                source_transition_id=f"live:{self.tick}",
            )
        self.pending = None

    def _add_hypotheses(self, before: np.ndarray, after: np.ndarray, action: int) -> None:
        """Propose feature thresholds above observed values; no negative proves them."""
        old = Counter(int(value) for value in before.flat)
        new = Counter(int(value) for value in after.flat)
        for color in sorted(set(old) | set(new)):
            delta = new[color] - old[color]
            if delta == 0:
                continue
            row = {
                "id": f"color:{color}:{'up' if delta > 0 else 'down'}",
                "feature": "visible_color_count",
                "value": color,
                "direction": 1 if delta > 0 else -1,
                "threshold": new[color] + (1 if delta > 0 else -1),
                "source_action": action,
                "state": "unknown",
            }
            if all(item["id"] != row["id"] for item in self.goal_hypotheses):
                self.goal_hypotheses.append(row)
        self.goal_hypotheses = self.goal_hypotheses[-16:]

    def select(
        self,
        frame: Any,
        legal_actions: Sequence[Any],
        base_move: tuple[Any, Any],
    ) -> tuple[Any, Any]:
        """Admit one cheap legal probe or retain the scored policy's legal move."""
        started = time.monotonic()
        legal = tuple(
            sorted({int(a) for a in legal_actions if isinstance(a, int) and 1 <= a <= 5})
        )[:5]
        base_action, base_data = base_move
        fallback = base_move if base_action in legal else ((legal[0], None) if legal else base_move)
        chosen: int | None = None
        reason = "insufficient_support"
        if not legal:
            reason = "no_legal_candidates"
        elif len(self.decisions) >= 8:
            reason = "probe_budget_exhausted"
        elif _visible_grid(frame) is None or _level(frame) is None:
            reason = "unreadable_observation"
        elif base_data is not None:
            reason = "structured_action_fallback"
        elif self.mode == "novelty":
            counts = Counter(row["action"] for row in self.effects)
            if len(self.effects) >= 2:
                chosen = min(legal, key=lambda action: (counts[action], action))
                reason = "novel_legal_action"
        elif len(self.effects) >= 2 and any(
            row["state"] == "unknown" for row in self.goal_hypotheses
        ):
            changed = [row for row in self.effects if row["changed"] > 0 and row["action"] in legal]
            if changed:
                different = [row for row in changed if row["action"] != base_action]
                tested = int((different or changed)[-1]["action"])
                if tested in legal:
                    source = TransitionEvidence(
                        source_transition_id=f"live:{changed[-1]['tick']}",
                        source_tick=int(changed[-1]["tick"]),
                        source_action=tested,
                        observed_symbol=FRAME_CHANGED_NO_LEVEL,
                        visible_frame_hash_before="visible_before",
                        visible_frame_hash_after="visible_after",
                    )
                    self.frontier = RewardMachineFrontier(
                        (
                            RewardMachineHypothesis(
                                "effect_repeats",
                                ("start",),
                                "start",
                                "start",
                                (
                                    RewardMachineTransition(
                                        "start",
                                        tested,
                                        "start",
                                        FRAME_CHANGED_NO_LEVEL,
                                        evidence=(source,),
                                    ),
                                ),
                            ),
                            RewardMachineHypothesis(
                                "effect_is_contextual",
                                ("start",),
                                "start",
                                "start",
                                (
                                    RewardMachineTransition(
                                        "start",
                                        tested,
                                        "start",
                                        SAME_FRAME_NO_LEVEL,
                                    ),
                                ),
                            ),
                        ),
                        capacity=2,
                        timeout_ticks=4,
                    )
                    split = self.frontier.choose_legal_disagreement(
                        legal_actions=legal,
                        candidate_actions=(tested,),
                        tick=self.tick,
                        base_policy_action=fallback[0],
                    )
                    chosen = split.action
                    reason = split.reason
        if time.monotonic() - started > 0.01:
            chosen = None
            reason = "decision_time_exceeded"
        # No probe alters the SDK action set. An unsupported or unsafe candidate falls back.
        if chosen not in legal:
            chosen = None
        self.decisions.append(
            {
                "tick": self.tick,
                "mode": self.mode,
                "legal_actions": list(legal),
                "base_action": base_action,
                "selected_action": chosen if chosen is not None else fallback[0],
                "admitted": chosen is not None,
                "reason": reason,
                "goal_guard": self.contract.evaluate(
                    "unconfirmed_visible_goal",
                    (),
                    firing_witness_ids=(),
                    nonfiring_contrast_ids=(),
                ).state,
            }
        )
        self.decisions = self.decisions[-8:]
        return (chosen, None) if chosen is not None else fallback

    def diagnostics(self) -> dict[str, Any]:
        """Expose bounded evidence without upgrading a conjecture to a solve."""
        return {
            "enabled": True,
            "mode": self.mode,
            "goal_hypotheses": [dict(row) for row in self.goal_hypotheses],
            "decisions": [dict(row) for row in self.decisions],
            "observed_actions": self.observed_actions,
            "sdk_progress": self.sdk_progress,
            "goal_confirmations": self.goal_confirmations,
            "virtual_engine_calls": self.virtual_engine_calls,
            "frontier": self.frontier.diagnostics(),
        }

    def checkpoint(self) -> dict[str, Any]:
        """Save visible evidence and the pending actual action for a process restart."""
        return {
            "mode": self.mode,
            "pending": self.pending,
            "effects": self.effects,
            "goal_hypotheses": self.goal_hypotheses,
            "decisions": self.decisions,
            "observed_actions": self.observed_actions,
            "sdk_progress": self.sdk_progress,
            "tick": self.tick,
        }

    @classmethod
    def from_checkpoint(cls, value: dict[str, Any]) -> ArcProbeProtocol:
        """Reload an owned checkpoint without importing future outcome labels."""
        probe = cls(str(value["mode"]))
        probe.pending = value.get("pending")
        probe.effects = list(value.get("effects", ()))[:16]
        probe.goal_hypotheses = list(value.get("goal_hypotheses", ()))[:16]
        probe.decisions = list(value.get("decisions", ()))[:8]
        probe.observed_actions = int(value.get("observed_actions", 0))
        probe.sdk_progress = int(value.get("sdk_progress", 0))
        probe.tick = int(value.get("tick", 0))
        return probe
