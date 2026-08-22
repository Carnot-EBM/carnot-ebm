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

from dataclasses import dataclass

# The closed arm vocabulary, in firing order. Order is a diagnosis ladder:
# an installed goal bias that survived a whole stagnant window is steering
# and not working, so remove it first; then buy a fresh world model with the
# evidence that accumulated; then change the search draw itself.
ARM_DROP_GOAL_BIAS = "drop_goal_bias"
ARM_ALLOW_REINDUCTION = "allow_reinduction"
ARM_FORCE_DIVERSITY = "force_exploration_diversity"
ARM_ORDER = (ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY)


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
