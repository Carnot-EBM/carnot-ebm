"""Trajectory-level strategy supervisor for the live ARC agent (REQ-ARC-WMTE-6600).

WHY. After the one-shot induction latch fires and a plan exhausts without a
level-up, the live cascade returns to explore and never changes strategy for
the rest of the level. This module adapts the supervisor from NVIDIA's AVO
harness (arXiv 2603.24517): detect stagnation from trajectory statistics,
then redirect strategy. AVO redirects by open-ended re-planning with a
frontier model; our generator is weak, so the redirect is a closed decision
table over levers the agent already has. Deterministic; no LLM call.
See docs/research-notes/avo-adaptation-for-local-generator-2026-08-21.md.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# The closed arm vocabulary, in firing order. Order is a diagnosis ladder:
# an installed goal bias that survived a whole stagnant window is steering
# and not working, so remove it first; then buy a fresh world model with the
# evidence that accumulated; then change the search draw itself.
ARM_DROP_GOAL_BIAS = "drop_goal_bias"
ARM_ALLOW_REINDUCTION = "allow_reinduction"
ARM_FORCE_DIVERSITY = "force_exploration_diversity"
# A FOURTH RUNG, added 2026-08-29 and DEFAULT OFF (REQ-ARC-WMTE-6760).
#
# The ladder's second rung buys a fresh world model with accumulated evidence, but it re-draws
# through the SAME single-shot induction that just failed to explain the level. The callable-tool
# loop is a different draw: the model queries transitions and executes candidate engines instead
# of writing one blind. Its transport gate passed at ceiling on 2026-08-28 (20/20 attempt, 20/20
# parse-to-dispatch) so it RUNS; whether it induces BETTER is what the resumed holdout-equalized
# A/B measures, and that has not reported.
#
# So this arm exists, is fired last, and is gated OFF until that evidence lands. Wiring an
# unmeasured lever into the live scored path is the thing this project's disciplines exist to
# stop; leaving the supervisor unable to reach a capability it should be finetuning against is
# the opposite failure. A default-off arm with an outcome ledger is how both are avoided: the
# arm can be measured before it is trusted.
ARM_TOOL_LOOP_REINDUCTION = "tool_loop_reinduction"

ARM_ORDER = (
    ARM_DROP_GOAL_BIAS,
    ARM_ALLOW_REINDUCTION,
    ARM_FORCE_DIVERSITY,
    ARM_TOOL_LOOP_REINDUCTION,
)


def tool_loop_arm_enabled() -> bool:
    """Is the fourth rung armed? Default OFF until the A/B reports.

    Exact-match "1", so a stray truthy value cannot switch a live-path strategy change on by
    accident -- the same discipline the worktree guard's override uses.
    """

    return os.environ.get("CARNOT_ARC_SUPERVISOR_TOOL_ARM") == "1"


@dataclass(frozen=True)
class TrajectorySnapshot:
    """One per-action view of the run, built by the policy from state it
    already tracks. The supervisor reads only this — it never touches the
    policy or the explorer directly, so it stays trivially unit-testable."""

    level: int
    goal_bias_installed: bool
    induced: bool
    induction_attempts: int
    new_transitions_since_induction: int
    diversity_active: bool


@dataclass(frozen=True)
class Redirect:
    """A strategy change the supervisor asks the policy to apply. The policy
    owns the mutation; this object is also the receipt row's content."""

    arm: str
    action_index: int
    level: int
    diagnosis: str


class TrajectorySupervisor:
    """Mechanical stagnation detector plus bounded strategy redirection.

    Counts actions since the last level-up or redirect. At `window` stagnant
    actions it fires the first eligible unused arm (see ARM_ORDER), then
    restarts the window. Each arm fires at most once per level; a level-up
    resets everything. Bounded by construction: at most len(ARM_ORDER)
    redirects per level, no matter how long the run is.
    """

    def __init__(
        self,
        *,
        window: int = 400,
        reinduction_evidence_floor: int = 200,
        reinduction_attempt_cap: int = 3,
    ) -> None:
        self.window = max(1, int(window))
        self.reinduction_evidence_floor = max(0, int(reinduction_evidence_floor))
        self.reinduction_attempt_cap = max(0, int(reinduction_attempt_cap))
        self._actions_total = 0
        self._actions_since_progress = 0
        self._last_level: int | None = None
        self._arms_used: set[str] = set()
        self._redirects: list[dict] = []
        # Stagnation windows that fired NO arm (REQ-ARC-WMTE-6640 rule 4).
        # When this grows while no arm fires, the closed table has run out
        # of ideas — the written trigger for a human to propose a new arm.
        self._stagnations_unredirected = 0

    def observe(self, snapshot: TrajectorySnapshot) -> Redirect | None:
        """Feed one action's snapshot; get back a redirect or None."""

        self._actions_total += 1
        if self._last_level is None:
            self._last_level = int(snapshot.level)
        if int(snapshot.level) > self._last_level:
            # Progress. Credit every redirect still waiting on an outcome
            # (REQ-ARC-WMTE-6640 rule 2): the unresolved set is exactly the
            # set fired since the last progress event. Runs only here, so
            # the per-action cost of the routed path does not change.
            for row in self._redirects:
                if not row["resolved_by_levelup"]:
                    row["resolved_by_levelup"] = True
                    row["actions_to_levelup"] = self._actions_total - row["action_index"]
            # Start the level fresh: arms become available again and
            # the stagnation count restarts.
            self._last_level = int(snapshot.level)
            self._actions_since_progress = 0
            self._arms_used.clear()
            return None
        self._actions_since_progress += 1
        if self._actions_since_progress < self.window:
            return None
        arm, diagnosis = self._first_eligible_arm(snapshot)
        # Restart the window either way. When nothing is eligible now, a lever
        # may become eligible later (e.g. a bias installed by a fresh
        # induction), and re-checking every action would spam the same answer.
        self._actions_since_progress = 0
        if arm is None:
            self._stagnations_unredirected += 1
            return None
        self._arms_used.add(arm)
        redirect = Redirect(
            arm=arm,
            action_index=self._actions_total,
            level=int(snapshot.level),
            diagnosis=diagnosis,
        )
        self._redirects.append(
            {
                "arm": redirect.arm,
                "action_index": redirect.action_index,
                "level": redirect.level,
                "diagnosis": redirect.diagnosis,
                # Outcome fields start present, not absent (REQ-ARC-WMTE-6640
                # rule 1). A run that ends here reads an honest "false", and
                # no end-of-run finalize step is needed.
                "resolved_by_levelup": False,
                "actions_to_levelup": None,
            }
        )
        return redirect

    def _first_eligible_arm(self, s: TrajectorySnapshot) -> tuple[str | None, str]:
        """The decision table. Fixed order, one winner, plain-words diagnosis."""

        if ARM_DROP_GOAL_BIAS not in self._arms_used and s.goal_bias_installed:
            return (
                ARM_DROP_GOAL_BIAS,
                f"goal bias installed through {self.window} stagnant actions; "
                "it is steering and not working",
            )
        if (
            ARM_ALLOW_REINDUCTION not in self._arms_used
            and s.induced
            and s.new_transitions_since_induction >= self.reinduction_evidence_floor
            and s.induction_attempts < self.reinduction_attempt_cap
        ):
            return (
                ARM_ALLOW_REINDUCTION,
                f"induction latch set with {s.new_transitions_since_induction} new "
                "transitions the model has never seen",
            )
        # Fires only AFTER a plain re-induction has already been spent on this level and the
        # stagnation continued -- that is the written evidence that the single-shot draw is not
        # the thing that will explain this level, which is the only honest reason to pay for a
        # multi-turn loop. Reaching this rung with every earlier arm used is also exactly the
        # "all arms exhausted and stagnation continued" state the refinement spec calls the
        # specification for a NEW arm.
        if (
            tool_loop_arm_enabled()
            and ARM_TOOL_LOOP_REINDUCTION not in self._arms_used
            and ARM_ALLOW_REINDUCTION in self._arms_used
            and s.induction_attempts < self.reinduction_attempt_cap
        ):
            return (
                ARM_TOOL_LOOP_REINDUCTION,
                "single-shot re-induction was already spent on this level and stagnation "
                "continued; re-induce through the callable-tool loop instead",
            )
        if ARM_FORCE_DIVERSITY not in self._arms_used and not s.diversity_active:
            return (
                ARM_FORCE_DIVERSITY,
                "deterministic frontier draw exhausted its ideas; switch to the "
                "randomized top-k draw",
            )
        return None, ""

    def receipt(self) -> dict:
        """The evidence a run artifact carries (REQ-ARC-WMTE-6600 rule 6;
        outcome attribution per REQ-ARC-WMTE-6640)."""

        # Per-arm fired/helped counts (REQ-ARC-WMTE-6640 rule 3). Every arm
        # appears, zeros included: an unfired arm must be visibly zero, not
        # absent — absence is what made the 2026-08-21 A/B unreadable.
        arm_outcomes = {arm: {"fired": 0, "helped": 0} for arm in ARM_ORDER}
        for row in self._redirects:
            outcome = arm_outcomes[row["arm"]]
            outcome["fired"] += 1
            if row["resolved_by_levelup"]:
                outcome["helped"] += 1
        return {
            "enabled": True,
            "window": self.window,
            "actions_observed": self._actions_total,
            "arms_used": sorted(self._arms_used),
            "redirects": list(self._redirects),
            "arm_outcomes": arm_outcomes,
            "stagnations_unredirected": self._stagnations_unredirected,
        }


class TraceAutomatonSupervisor:
    """Apply one frozen, game-blind action redirect on the live policy seam.

    The object sees only outcomes that the policy already observed before the
    next action. It never sees a game ID or a future frame. A reset redirect is
    intentionally conservative because reset needs no game-specific payload.
    The caller must opt in by installing an instance on one E3 policy.
    """

    def __init__(self, frozen_fsm: Mapping[str, Any]) -> None:
        if frozen_fsm.get("schema") != "carnot.arc.trace_fsm.v1":
            raise ValueError("unsupported trace automaton schema")
        thresholds = frozen_fsm.get("thresholds") or {}
        self.same_action_threshold = max(1, int(thresholds["same_action_run"]))
        self.stagnation_threshold = max(1, int(thresholds["actions_since_observed_change"]))
        self.overhead_threshold = max(1, int(thresholds.get("consecutive_navigation_or_replay", 2)))
        self.frozen_fsm = dict(frozen_fsm)
        self._last_action_key: str | None = None
        self._same_action_run = 0
        self._actions_since_change = 0
        self._overhead_run = 0
        self._rows: list[dict[str, Any]] = []

    @staticmethod
    def _action_key(move: Any) -> str:
        kind, data = move if isinstance(move, tuple) and len(move) == 2 else (None, None)
        if isinstance(data, Mapping):
            data_key = tuple(sorted((str(key), repr(value)) for key, value in data.items()))
        else:
            data_key = repr(data)
        return repr((kind, data_key))

    def select_action(
        self,
        proposed_action: Any,
        *,
        previous_frame_changed: bool | None,
        level_progress_since_previous_action: bool,
        action_role_is_overhead: bool = False,
    ) -> Any:
        """Return the selected action and retain a next-outcome-linked receipt."""

        if self._rows and self._rows[-1]["next_outcome"] is None:
            self._rows[-1]["next_outcome"] = {
                "observed": previous_frame_changed is not None,
                "frame_changed": previous_frame_changed,
                "level_progress": bool(level_progress_since_previous_action),
            }
        if level_progress_since_previous_action or previous_frame_changed is True:
            self._actions_since_change = 0
        elif previous_frame_changed is False:
            self._actions_since_change += 1

        action_key = self._action_key(proposed_action)
        if action_key == self._last_action_key:
            self._same_action_run += 1
        else:
            self._last_action_key = action_key
            self._same_action_run = 1
        self._overhead_run = self._overhead_run + 1 if action_role_is_overhead else 0

        if not self._rows:
            state = "bootstrap"
        elif (
            self._same_action_run >= self.same_action_threshold
            and self._actions_since_change >= self.stagnation_threshold
        ) or self._overhead_run >= self.overhead_threshold:
            state = "stagnant_repeat"
        elif level_progress_since_previous_action or previous_frame_changed is True:
            state = "productive"
        else:
            state = "observing"

        fired = state == "stagnant_repeat"
        selected_action = ("RESET", None) if fired else proposed_action
        influenced = selected_action != proposed_action
        self._rows.append(
            {
                "action_index": len(self._rows),
                "state": state,
                "fired": fired,
                "arm": "reset_after_stagnant_repeat" if fired else None,
                "pre_action_features": {
                    "previous_frame_changed": previous_frame_changed,
                    "same_action_run": self._same_action_run,
                    "actions_since_observed_change": self._actions_since_change,
                    "level_progress_since_previous_action": bool(
                        level_progress_since_previous_action
                    ),
                    "action_role_is_overhead": bool(action_role_is_overhead),
                    "consecutive_navigation_or_replay": self._overhead_run,
                },
                "proposed_action": proposed_action,
                "selected_action": selected_action,
                "action_influenced": influenced,
                "blocked_valid_action": influenced,
                "next_outcome": None,
            }
        )
        return selected_action

    def finalize(self) -> None:
        """Close the final row without inventing an outcome after the run ends."""

        if self._rows and self._rows[-1]["next_outcome"] is None:
            self._rows[-1]["next_outcome"] = {
                "observed": False,
                "frame_changed": None,
                "level_progress": False,
            }

    def receipt(self) -> dict[str, Any]:
        """Return action, firing, influence, and exact-outcome accounting."""

        firings = sum(int(row["fired"]) for row in self._rows)
        influences = sum(int(row["action_influenced"]) for row in self._rows)
        return {
            "enabled": True,
            "schema": "carnot.arc.trace_fsm.receipt.v1",
            "fsm_schema": self.frozen_fsm["schema"],
            "actions_observed": len(self._rows),
            "firings": firings,
            "action_influences": influences,
            "blocked_valid_actions": sum(int(row["blocked_valid_action"]) for row in self._rows),
            "rows": list(self._rows),
        }
