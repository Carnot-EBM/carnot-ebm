"""PER-ACTION PROVENANCE for the SCORED ARC agent -- inert by default.

**Researcher summary (why this file exists at all).**
    The project can say how many levels the live agent banks and how good its induced
    world models look, but it cannot say *where its actions come from*. Three independent
    2026-07/2026-08 measurements converged on the same uncomfortable place:

      * deleting the LLM induction tier entirely left the agent's action sequence
        BYTE-IDENTICAL on 5 of 6 games;
      * 0 of 22 `stall` inductions ever cleared the goal gate, while 4 of 6
        `level_up_reinduction` events did -- so `plan_found` correlates with banking
        because banking TRIGGERS a trivially-passing re-induction, not because planning
        causes banking;
      * tn36 holds an induced engine with held-out accuracy 1.0 and 25/25 changing
        transitions correct, and banked 0 levels in 346 actions.

    Each of those is an INFERENCE that the induce->verify->plan pipeline is not on the
    causal path to banking a level. None of them measured the thing that would settle it:
    the ACTION. If almost no action the agent spends is plan-derived, the mediation null
    stops being an inference and becomes an accounting fact, and it names the stage to
    attack. If many ARE plan-derived and the plans are simply wrong, that is a different
    and equally actionable answer. This module is the instrument that produces that
    accounting, and nothing else -- it decides nothing and changes nothing.

**Detailed explanation for engineers.**
    `E3AgentPolicy.next_move` is the single choke point through which every action the
    SCORED agent emits passes (`make_carnot_agent` -> `CarnotAgent.choose_action` ->
    `self._policy.next_move`). This module is a passive recorder hung off that choke
    point plus a set of constant-string branch labels written at each `return` site in
    the two functions that actually choose an action (`E3AgentPolicy.next_move` and
    `StepwiseExplorer.next_move`/`_serve`).

    THE DESIGN CONSTRAINT THAT DOMINATES EVERYTHING HERE: this must not change what the
    agent does. An instrument that perturbs the run answers a question about the
    instrument. So:

      * The recorder is constructed ONLY when `CARNOT_ARC_ACTION_PROVENANCE=1`. When the
        flag is unset the policy holds `self._provenance = None` and every call site is a
        single `is None` test -- no allocation, no I/O, no RNG draw, no branch taken.
      * The branch labels in the hot path are BARE CONSTANT-STRING ASSIGNMENTS to an
        instance attribute (`self._prov_branch = "..."`). A constant assignment cannot
        change control flow, cannot consume randomness, and cannot raise for any object
        that has a `__dict__` -- which is why they are written unconditionally rather
        than behind the flag: an unconditional assignment is trivially provable inert,
        whereas a flag-guarded one adds a branch whose two sides must then be argued
        equivalent.
      * Every recorder entry point is wrapped so that an exception inside the instrument
        can never propagate into the agent. A crashed measurement is a lost row; a
        crashed agent is a zeroed game.

    WHAT IS NOT DONE HERE, DELIBERATELY. The recorder never calls the induced engine to
    ask what it predicted. Executing induced (LLM-authored) code inside the measuring
    process is forbidden in this repo, and it would also mean the instrument re-runs the
    very computation it is measuring. Consequently the "did execution diverge from the
    plan" question is answered with OBSERVATIONAL fields only -- did the frame change,
    was the plan abandoned before it was consumed -- and those fields are named so that
    no reader can mistake them for a model-prediction check. See
    `ActionProvenanceRecorder.record`'s field notes.

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-6070
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional

# The env flag. DEFAULT OFF: unset (or any value other than "1") means the shipped agent
# behaves exactly as it did before this module existed. Named with the CARNOT_ARC_ prefix
# every other runtime lever in this package uses.
PROVENANCE_ENV_FLAG = "CARNOT_ARC_ACTION_PROVENANCE"

# Where a run drops its rows when the caller does not name a path. Kept out of `results/`
# on purpose: `results/**` is the research record, and an instrument that writes there on
# every run is the exact mechanism CLAUDE.md's Test-Run Record Integrity Discipline
# exists to stop.
#
# KNOWN AND DELIBERATE: when this is unset, `resolve_path` falls back to
# `<cwd>/arc_provenance/`. Run from a checkout that is ALSO the cwd, that creates an
# UNTRACKED directory inside the repo, which shows up as `??` in `git status`. That is a
# nuisance, not a record-integrity violation -- nothing tracked is rewritten -- and it
# matches the sibling `_emit_generator_liveness_witness`, which writes `<cwd>/arc_liveness/`
# by exactly the same rule. Stated here rather than left for someone to discover, because
# "it appeared in my git status" should cost a reader ten seconds, not an investigation.
# Neither directory is in `.gitignore`; deliberately not adding one, since `.gitignore` is
# shared and the sibling convention predates this module. Measurement drivers should set
# this variable (or pass `path=`) rather than rely on the fallback -- the probe in
# `scripts/arc_action_provenance_probe.py` never calls `flush()` at all, it reads
# `to_dict()` directly.
PROVENANCE_DIR_ENV = "CARNOT_ARC_ACTION_PROVENANCE_DIR"

SCHEMA = "carnot.arc.action_provenance.v1"


def provenance_enabled() -> bool:
    """True when the operator has explicitly switched the instrument on.

    Deliberately strict equality against ``"1"`` rather than a truthiness test, matching
    the `_playbook_exemplars_gate_on` / `_sge_candidate_router_requested` convention in
    `arc_competition_agent.py`. A stray ``CARNOT_ARC_ACTION_PROVENANCE=0`` in an
    environment must not silently arm an instrument.
    """
    return os.environ.get(PROVENANCE_ENV_FLAG) == "1"


# --------------------------------------------------------------------------------------
# Branch vocabularies.
#
# These are the CLOSED sets of places an action can be chosen from. They are written down
# as constants (rather than left implicit in the string literals at the return sites) for
# one reason: the analysis has to be able to assert that it saw a label it understands. A
# row carrying an unknown label means the agent grew a new decision path that the
# accounting silently mis-attributed, and that must fail loudly rather than be bucketed
# into "other".
# --------------------------------------------------------------------------------------

#: `E3AgentPolicy.next_move`'s return sites. Every action the SCORED agent emits leaves
#: through exactly one of these.
TOP_BRANCHES: tuple[str, ...] = (
    # phase == "explore", no induction triggered: the tier-1 StepwiseExplorer chose it.
    "explore.explorer",
    # phase == "induce" produced a plan AND the plan is executable from the current state.
    "induce.plan_from_current",
    # phase == "induce" produced a plan that must be replayed from a reset first.
    "induce.plan_needs_reset",
    # phase == "induce" produced NO plan: fall straight back to the explorer.
    "induce.no_plan.explorer",
    # phase == "induce" DEFERRED by the cross-level carry (REQ-ARC-XLEVEL-CARRY-1,
    # flag-gated, default off): the boundary reinduction lacked enough new-level
    # evidence to verify a carried engine, so this action gathers one more tracked
    # transition via the explorer and the reinduction stays pending.
    "induce.carry_deferred.explorer",
    # phase == "execute": a step off an already-installed plan.
    "execute.plan_step",
    # plan exhausted or never existed: back to the explorer.
    "exhausted.explorer",
)

#: `StepwiseExplorer.next_move`'s return sites. Only meaningful on rows whose
#: `top_branch` ends in `.explorer`.
EXPLORER_BRANCHES: tuple[str, ...] = (
    "bootstrap_reset",  # no root yet: RESET to obtain the first frame
    "pending_drain",  # serving an already-queued navigation/probe step
    "go_explore_replay",  # go-explore archive replay sequence
    "depth_ride.qd_sequence",  # quality-diversity sequence at the current node
    "depth_ride.pop_untested",  # depth-first ride: expand the current node in place
    "frontier.qd_sequence",  # best frontier IS the current node, QD sequence
    "frontier.pop_untested",  # best frontier IS the current node, expand in place
    "frontier.navigate",  # navigate to a different frontier node, then probe
    "explored_out",  # nothing left to expand: (None, None)
)

#: `StepwiseExplorer._serve`'s three item kinds. Only meaningful when the explorer branch
#: is `pending_drain` or `frontier.navigate`.
SERVE_KINDS: tuple[str, ...] = (
    "reset",  # a RESET step in a replay-from-root sequence
    "probe",  # an actual frontier probe (the action being tested)
    "navigation",  # a forward-walk / replay step taken purely to reach a node
)


class ActionProvenanceRecorder:
    """Accumulates one row per action the policy emits. Decides nothing.

    Holds rows in memory and writes them once at `flush()`. Rows are small (a few hundred
    bytes) and an episode is bounded by `MAX_ACTIONS = 400` on the scored path, so the
    memory cost is trivial and the alternative -- an open file handle written per action --
    would put I/O latency inside the agent's decision loop.
    """

    def __init__(self, game: str, *, path: Optional[str] = None, run_label: str = "") -> None:
        self.game = str(game)
        self.run_label = str(run_label)
        self.rows: list[dict[str, Any]] = []
        self.path = path
        # Monotonic counter of DISTINCT plan objects the policy has installed. Incremented
        # by `note_plan_object` when it observes `self.plan` become a different object.
        # This is exact rather than inferred: the comparison happens while BOTH the old and
        # the new list are alive in the caller's frame, so there is no id()-reuse hazard.
        self.plan_epoch = 0
        self._last_plan_obj: Any = None
        # Index into `E3AgentPolicy.induction_attempts` of the attempt whose call installed
        # the currently-live plan. None when no plan has ever been installed.
        self.plan_installed_by_attempt: Optional[int] = None
        # Bookkeeping for the "was the plan abandoned" observation.
        self.plans_installed = 0
        self.plans_consumed_fully = 0
        self.plans_abandoned = 0
        self.errors: list[str] = []

    # -- plan-epoch accounting ---------------------------------------------------------

    def note_plan_object(self, plan: Any, attempt_index: Optional[int]) -> None:
        """Record that `plan` is the policy's current plan object.

        Called from the `next_move` wrapper AFTER the routed call, with the old object
        still referenced by the wrapper's local, so `is not` is a sound identity test.
        """
        try:
            if plan is self._last_plan_obj:
                return
            # A genuinely new list object. Only count it as an INSTALL when it is
            # non-empty: `self.plan = []` at a level boundary is a clear, not an install.
            self._last_plan_obj = plan
            if plan:
                self.plan_epoch += 1
                self.plans_installed += 1
                self.plan_installed_by_attempt = attempt_index
        except Exception as exc:  # pragma: no cover - the instrument must never raise
            self.errors.append(f"note_plan_object: {exc!r}"[:200])

    # -- the row -----------------------------------------------------------------------

    def record(self, row: Mapping[str, Any]) -> None:
        """Append one action row. Never raises."""
        try:
            self.rows.append(dict(row))
        except Exception as exc:  # pragma: no cover
            self.errors.append(f"record: {exc!r}"[:200])

    # -- output ------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """The accounting the whole exercise exists to produce.

        `plan_derived_actions` is the numerator of the question "of the N actions the
        agent spends, how many came from the induce->verify->plan pipeline". It counts
        rows whose top branch is a plan step -- `execute.plan_step` and
        `induce.plan_from_current` -- and nothing else. A RESET emitted so that a plan can
        be replayed from root (`induce.plan_needs_reset`) is counted SEPARATELY rather
        than folded in, because it is an action spent BECAUSE of a plan without being an
        action the plan chose; conflating the two would flatter the pipeline.
        """
        rows = self.rows
        n = len(rows)

        def _count(pred) -> int:
            return sum(1 for r in rows if pred(r))

        plan_steps = _count(lambda r: r.get("top_branch") in ("execute.plan_step",))
        plan_from_current = _count(lambda r: r.get("top_branch") == "induce.plan_from_current")
        plan_reset = _count(lambda r: r.get("top_branch") == "induce.plan_needs_reset")
        explorer = _count(lambda r: str(r.get("top_branch", "")).endswith("explorer"))
        by_top: dict[str, int] = {}
        by_explorer: dict[str, int] = {}
        by_serve: dict[str, int] = {}
        for r in rows:
            by_top[str(r.get("top_branch"))] = by_top.get(str(r.get("top_branch")), 0) + 1
            eb = r.get("explorer_branch")
            if eb:
                by_explorer[str(eb)] = by_explorer.get(str(eb), 0) + 1
            sk = r.get("explorer_serve_kind")
            if sk:
                by_serve[str(sk)] = by_serve.get(str(sk), 0) + 1
        # EXPANSION vs OVERHEAD. Orthogonal to the plan/explorer split above and, on the
        # measurements taken so far, the larger effect. An "expansion" is an action that
        # TESTS something the agent has not tried: a frontier probe, or an in-place pop of
        # an untested action. Everything the explorer serves with a `navigation` or `reset`
        # kind is the agent walking or replaying back to a state it has already seen, in
        # order to reach the node it wants to expand. Computed here rather than in an
        # analysis script so the number is part of the record and cannot be re-derived a
        # different way by the next reader.
        expansion_branches = (
            "depth_ride.pop_untested",
            "frontier.pop_untested",
            "depth_ride.qd_sequence",
            "frontier.qd_sequence",
        )
        expansions = _count(
            lambda r: (
                r.get("explorer_serve_kind") == "probe"
                or r.get("explorer_branch") in expansion_branches
            )
        )
        overhead = _count(lambda r: r.get("explorer_serve_kind") in ("navigation", "reset"))
        return {
            "game": self.game,
            "run_label": self.run_label,
            "actions_recorded": n,
            "new_information_expansions": expansions,
            "new_information_expansion_fraction": (round(expansions / n, 6) if n else None),
            "navigation_or_replay_actions": overhead,
            "navigation_or_replay_fraction": round(overhead / n, 6) if n else None,
            "plan_derived_actions": plan_steps + plan_from_current,
            "plan_derived_fraction": (
                round((plan_steps + plan_from_current) / n, 6) if n else None
            ),
            "plan_step_actions": plan_steps,
            "plan_from_current_actions": plan_from_current,
            "reset_for_plan_replay_actions": plan_reset,
            "explorer_actions": explorer,
            "explorer_fraction": round(explorer / n, 6) if n else None,
            "by_top_branch": by_top,
            "by_explorer_branch": by_explorer,
            "by_serve_kind": by_serve,
            "plan_epochs": self.plan_epoch,
            "plans_installed": self.plans_installed,
            "plans_consumed_fully": self.plans_consumed_fully,
            "plans_abandoned": self.plans_abandoned,
            "unknown_top_branches": sorted(set(by_top) - set(TOP_BRANCHES)),
            "unknown_explorer_branches": sorted(set(by_explorer) - set(EXPLORER_BRANCHES)),
            "recorder_errors": self.errors[:20],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "summary": self.summary(), "rows": self.rows}

    def resolve_path(self) -> str:
        if self.path:
            return self.path
        base = os.environ.get(PROVENANCE_DIR_ENV) or os.path.join(os.getcwd(), "arc_provenance")
        label = self.run_label or "run"
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in f"{self.game}_{label}")
        return os.path.join(base, f"action_provenance_{safe}.json")

    def flush(self) -> Optional[str]:
        """Write the rows. Returns the path written, or None on failure. Never raises."""
        try:
            path = self.resolve_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=1, default=str)
            return path
        except Exception as exc:  # pragma: no cover
            self.errors.append(f"flush: {exc!r}"[:200])
            return None


# The most recently ARMED recorder, so a driver that does not own the policy object can
# still retrieve the rows. `run_bounded_progress` (the standard offline-arcade driver)
# constructs the policy internally and returns only a metrics dataclass, and reaching the
# rows any other way would mean editing that shared file -- which this instrument
# deliberately does not do. Populated ONLY when the flag is set, so an unarmed run leaves
# this None and holds no reference to anything.
_LAST_RECORDER: Optional["ActionProvenanceRecorder"] = None


def last_recorder() -> Optional["ActionProvenanceRecorder"]:
    """The most recently armed recorder in this process, or None.

    NOT SAFE UNDER THE SWARM, and that is a real limit rather than a theoretical one. The
    competition framework's `swarm.py` runs one `Agent` PER GAME PER THREAD in a SINGLE
    process, so with several games armed at once this returns whichever thread constructed
    its policy last -- not "this game's rows". Anything running more than one game must go
    through `E3AgentPolicy.action_provenance()` on the policy it owns, which is what
    `CarnotAgent.cleanup` does. This accessor exists solely for a single-game measurement
    driver that does not own the policy object (`run_bounded_progress` constructs it
    internally and returns only a metrics dataclass), and reaching the rows any other way
    would mean editing that shared file.
    """
    return _LAST_RECORDER


def maybe_make_recorder(game: str, *, run_label: str = "") -> Optional[ActionProvenanceRecorder]:
    """The single construction point. Returns None -- the inert state -- unless armed."""
    global _LAST_RECORDER
    if not provenance_enabled():
        return None
    try:
        rec = ActionProvenanceRecorder(game, run_label=run_label)
    except Exception:  # pragma: no cover - arming must never break a run
        return None
    _LAST_RECORDER = rec
    return rec
