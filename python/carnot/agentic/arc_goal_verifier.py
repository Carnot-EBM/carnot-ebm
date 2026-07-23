"""Goal-hypothesis verifier for the greedy-direct ARC agent (2026-07-23, REQ-ARC-WMTE-5830).

WHY. The 2026-07-23 winner-recipe reproduction
(docs/research-notes/arc-winner-recipe-reproduction-2026-07-23.md) faithfully rebuilt the leaderboard
winners' full recipe (gemma-4-31B + greedy-direct + object segmentation + reflection memory) and found
it STILL discovers 0 levels -- because the model induces the WRONG GOAL: it learns real dynamics and
forms+pursues a systematic goal hypothesis (e.g. bp35: "GOAL: fill column 63 with 15s"), completes it,
and never wins because that was not the actual win condition. The winner recipe supplies a fluent goal
HYPOTHESIZER (reflection memory) but has NO goal VERIFIER to reject a wrong hypothesis. This is exactly
Carnot's verification-first thesis (ops/verifier_gaps.md GAP-ARCH-GOAL-NOT-VERIFIED). This module is that
verifier.

WHAT IT VERIFIES, AND WHY IT IS ORACLE-DISTINCT. A goal hypothesis is a CLAIM ("this is how you win").
The verifier tests that claim against the ONLY ground-truth win signal a live hidden-game agent has: the
OBSERVABLE LEVEL COUNTER (frame.levels_completed / frame_level). A hypothesis is FALSIFIED when the agent
actively pursues it (real frame-changing activity) for long enough but the level counter does NOT advance
-- i.e. "I keep achieving this goal and nothing wins, so this is not the win condition." Crucially this
is NOT circular / NOT reading the win predicate (`is_level_complete`, which a hidden-game agent must never
read): it uses only the level-UP EVENTS the agent observes anyway (the reward signal every RL/discovery
agent legitimately sees). `verifier_is_oracle=False`: it never reads what winning IS, only whether a
level-up happened. The falsified goals are fed back so the model MUST hypothesize a DIFFERENT win
condition instead of burning its whole budget on one wrong one -- the exact failure mode observed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def extract_goal(notes: str) -> str:
    """Pull the model's current GOAL hypothesis out of its reflection notes (the 'GOAL:' line)."""
    if not notes:
        return ""
    m = re.search(r"GOAL:\s*(.+)", notes, re.IGNORECASE)
    if not m:
        return ""
    # take the first line of the GOAL section, trimmed
    return m.group(1).splitlines()[0].strip()[:200]


# The DEFINITIVE falsification signal from the winner-recipe reproduction: the model's own PROGRESS
# reports it COMPLETED/ACHIEVED its goal (bp35: "column 63 filled") yet no level-up fired -> the goal is
# demonstrably NOT the win condition. Independent of the activity-count heuristic, and it matches the
# exact observed failure mode.
_COMPLETION_RE = re.compile(
    r"PROGRESS:\s*(.+)",
    re.IGNORECASE,
)
_COMPLETION_WORDS = (
    "filled",
    "all cells",
    "all grid",
    "complete",
    "completed",
    "achieved",
    "done",
    "finished",
    "entire",
    "every cell",
    "fully",
)


def progress_indicates_completion(notes: str) -> bool:
    """True if the model's PROGRESS line reports having achieved/completed its goal."""
    m = _COMPLETION_RE.search(notes or "")
    if not m:
        return False
    line = m.group(1).splitlines()[0].lower()
    return any(w in line for w in _COMPLETION_WORDS)


@dataclass
class GoalVerifier:
    """Falsifies goal hypotheses against the observable level counter. See module docstring.

    goal_patience: min actions ON a single goal before it can be falsified (give it a fair chance).
    min_activity:  min frame-changing actions on the goal before falsification (don't falsify a goal
                   the agent never actually pursued -- e.g. a no-op stall is not evidence against it)."""

    goal_patience: int = 12
    min_activity: int = 4
    current_goal: str = ""
    actions_on_goal: int = 0
    changes_on_goal: int = 0
    levelups_on_goal: int = 0
    falsified: list[str] = field(default_factory=list)
    supported: list[str] = field(default_factory=list)
    goal_switches: int = 0

    def set_goal(self, goal: str) -> None:
        """Register the (possibly new) current goal. A genuinely NEW goal resets the pursuit counters."""
        g = (goal or "").strip()
        if not g:
            return
        if g != self.current_goal:
            self.current_goal = g
            self.actions_on_goal = 0
            self.changes_on_goal = 0
            self.levelups_on_goal = 0
            self.goal_switches += 1

    def observe(self, *, frame_changed: bool, leveled_up: bool) -> None:
        """Record one executed action's outcome under the current goal."""
        if not self.current_goal:
            return
        self.actions_on_goal += 1
        if frame_changed:
            self.changes_on_goal += 1
        if leveled_up:
            self.levelups_on_goal += 1
            if self.current_goal not in self.supported:
                self.supported.append(self.current_goal)

    def verdict(self) -> str:
        """'supported' (a level-up fired while pursuing it), 'falsified' (pursued with real activity but
        no level-up), or 'pending'."""
        if self.levelups_on_goal > 0:
            return "supported"
        if self.actions_on_goal >= self.goal_patience and self.changes_on_goal >= self.min_activity:
            return "falsified"
        return "pending"

    def maybe_falsify(self) -> bool:
        """If the current goal's verdict is 'falsified', record it (once) and return True."""
        if self.verdict() == "falsified" and self.current_goal:
            if self.current_goal not in self.falsified:
                self.falsified.append(self.current_goal)
            return True
        return False

    def falsify_on_reported_completion(self, notes: str) -> bool:
        """DEFINITIVE falsification: the model's PROGRESS reports it completed the current goal, yet no
        level-up has fired under it -> the goal is not the win condition. Records it and returns True."""
        if (
            self.current_goal
            and self.levelups_on_goal == 0
            and self.changes_on_goal >= 1
            and progress_indicates_completion(notes)
        ):
            if self.current_goal not in self.falsified:
                self.falsified.append(self.current_goal)
            return True
        return False

    def feedback(self) -> str:
        """Text injected into the reflection + decision prompts so the model abandons wrong goals."""
        if not self.falsified:
            return ""
        joined = "; ".join(self.falsified[-6:])
        return (
            "FALSIFIED GOALS (you pursued these and the level counter did NOT advance -- so they are NOT "
            "the win condition; do NOT pursue them again, hypothesize a DIFFERENT win condition): "
            + joined
        )

    def stats(self) -> dict:
        return {
            "falsified_goals": list(self.falsified),
            "supported_goals": list(self.supported),
            "goal_switches": self.goal_switches,
            "current_goal": self.current_goal,
        }
