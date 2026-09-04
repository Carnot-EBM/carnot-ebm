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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any


OBLIGATION_SCHEMA = "carnot.arc.operational_obligation.v3"
AUTOMATON_SCHEMA = "carnot.arc.operational_obligation_automaton.v3"
EVENT_SCHEMA = "carnot.arc.operational_obligation_events.v3"
PRIORITY_ORDER = ("hard", "binding", "soft")

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
            pending = [row for row in self._redirects if not row["resolved_by_levelup"]]
            for row in pending:
                row["resolved_by_levelup"] = True
                row["actions_to_levelup"] = self._actions_total - row["action_index"]
                # REQ-ARC-WMTE-7013: how many redirects this ONE level-up credited.
                # Three arms fired at 120/240/360 were all credited by the level-up at
                # 885 (r11l, 2026-09-03), in the control arm too. A reader needs this
                # count to tell a sole credit from a shared one; `helped` cannot say.
                row["co_credited_count"] = len(pending)
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
                # REQ-ARC-WMTE-7013: None until a level-up credits this row.
                "co_credited_count": None,
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
        # REQ-ARC-WMTE-7013: credit that does not smear. `helped_sole` counts
        # credits where this redirect was the ONLY one pending; `helped_share`
        # splits each level-up evenly over the redirects it credited. Kept in
        # a separate key so `arm_outcomes` keeps its exact 6640 shape.
        arm_credit = {arm: {"helped_sole": 0, "helped_share": 0.0} for arm in ARM_ORDER}
        for row in self._redirects:
            outcome = arm_outcomes[row["arm"]]
            outcome["fired"] += 1
            if row["resolved_by_levelup"]:
                outcome["helped"] += 1
                k = row.get("co_credited_count")
                if isinstance(k, int) and k > 0:
                    credit = arm_credit[row["arm"]]
                    if k == 1:
                        credit["helped_sole"] += 1
                    credit["helped_share"] += 1.0 / k
        # Round once at emit, so the receipt and the refinement report agree on the same rows.
        for credit in arm_credit.values():
            credit["helped_share"] = round(credit["helped_share"], 4)
        return {
            "enabled": True,
            "window": self.window,
            "actions_observed": self._actions_total,
            "arms_used": sorted(self._arms_used),
            "redirects": list(self._redirects),
            "arm_outcomes": arm_outcomes,
            "arm_credit": arm_credit,
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


class OperationalObligationError(ValueError):
    """Reject an ambiguous contract with one stable machine-readable code.

    The code is part of the receipt. Callers do not have to parse explanatory
    prose to decide whether a contract failed closed.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        message = code if not detail else f"{code}: {detail}"
        super().__init__(message)


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON once so hashes and fresh processes compare exact bytes."""

    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def canonical_obligation_bytes(obligations: Sequence[Mapping[str, Any]]) -> bytes:
    """Wrap obligation records in the one accepted v3 source envelope."""

    return canonical_json_bytes(
        {"obligations": [dict(record) for record in obligations], "schema": OBLIGATION_SCHEMA}
    )


def canonical_event_bytes(events: Sequence[Mapping[str, Any]]) -> bytes:
    """Wrap replay events in the one accepted v3 event envelope."""

    return canonical_json_bytes(
        {"events": [dict(event) for event in events], "schema": EVENT_SCHEMA}
    )


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    code: str,
) -> None:
    if set(value) != expected:
        raise OperationalObligationError(
            code,
            f"expected={sorted(expected)!r} observed={sorted(value)!r}",
        )


def _require_sorted_fact_list(value: Any, *, code: str, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise OperationalObligationError(code, f"{field} must be a string list")
    if value != sorted(value) or len(value) != len(set(value)):
        raise OperationalObligationError(code, f"{field} must be sorted and unique")
    return value


def _validate_action(action: Any, *, code: str) -> None:
    if not isinstance(action, Mapping):
        raise OperationalObligationError(code, "action must be an object")
    _require_exact_keys(action, {"data", "kind"}, code=code)
    kind = action.get("kind")
    if isinstance(kind, bool) or not isinstance(kind, (int, str)) or kind == "":
        raise OperationalObligationError(code, "action kind must be a nonempty string or integer")
    try:
        canonical_json_bytes(action)
    except (TypeError, ValueError) as exc:
        raise OperationalObligationError(code, "action data is not canonical JSON") from exc


def _validate_obligation(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise OperationalObligationError("invalid_obligation", "record must be an object")
    _require_exact_keys(
        record,
        {"action", "contract", "obligation_id"},
        code="invalid_obligation_fields",
    )
    obligation_id = record.get("obligation_id")
    if not isinstance(obligation_id, str) or not obligation_id:
        raise OperationalObligationError("invalid_obligation_id")
    _validate_action(record.get("action"), code="invalid_action")

    contract = record.get("contract")
    if not isinstance(contract, Mapping):
        raise OperationalObligationError("invalid_contract")
    required_contract_fields = {
        "authority",
        "execution_consequence",
        "fallback",
        "prerequisite",
        "priority",
    }
    missing = required_contract_fields - set(contract)
    missing_codes = {
        "authority": "absent_authority",
        "execution_consequence": "consequence_deletion",
        "fallback": "missing_fallback",
        "prerequisite": "ambiguous_prerequisite",
        "priority": "unknown_priority",
    }
    if missing:
        first = sorted(missing)[0]
        raise OperationalObligationError(missing_codes[first])
    _require_exact_keys(contract, required_contract_fields, code="invalid_contract_fields")

    prerequisite = contract["prerequisite"]
    if not isinstance(prerequisite, Mapping):
        raise OperationalObligationError("ambiguous_prerequisite")
    _require_exact_keys(
        prerequisite,
        {"all_of", "none_of"},
        code="ambiguous_prerequisite",
    )
    all_of = _require_sorted_fact_list(
        prerequisite["all_of"],
        code="ambiguous_prerequisite",
        field="all_of",
    )
    none_of = _require_sorted_fact_list(
        prerequisite["none_of"],
        code="ambiguous_prerequisite",
        field="none_of",
    )
    if set(all_of) & set(none_of):
        raise OperationalObligationError("ambiguous_prerequisite", "fact appears in both sets")

    authority = contract["authority"]
    if not isinstance(authority, Mapping):
        raise OperationalObligationError("absent_authority")
    _require_exact_keys(authority, {"issuer", "order"}, code="absent_authority")
    issuer = authority.get("issuer")
    order = authority.get("order")
    if not isinstance(issuer, str) or not issuer:
        raise OperationalObligationError("absent_authority")
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise OperationalObligationError("invalid_authority_order")

    fallback = contract["fallback"]
    if not isinstance(fallback, Mapping):
        raise OperationalObligationError("missing_fallback")
    _require_exact_keys(fallback, {"action", "reason"}, code="missing_fallback")
    _validate_action(fallback.get("action"), code="missing_fallback")
    if not isinstance(fallback.get("reason"), str) or not fallback["reason"]:
        raise OperationalObligationError("missing_fallback")

    consequence = contract["execution_consequence"]
    if not isinstance(consequence, Mapping):
        raise OperationalObligationError("consequence_deletion")
    _require_exact_keys(consequence, {"add", "remove"}, code="consequence_deletion")
    add = _require_sorted_fact_list(
        consequence["add"],
        code="consequence_deletion",
        field="add",
    )
    remove = _require_sorted_fact_list(
        consequence["remove"],
        code="consequence_deletion",
        field="remove",
    )
    if not add and not remove:
        raise OperationalObligationError("consequence_deletion")
    if set(add) & set(remove):
        raise OperationalObligationError("ambiguous_consequence")

    priority = contract["priority"]
    if not isinstance(priority, Mapping):
        raise OperationalObligationError("unknown_priority")
    _require_exact_keys(priority, {"class", "weight"}, code="unknown_priority")
    priority_class = priority.get("class")
    weight = priority.get("weight")
    if priority_class not in PRIORITY_ORDER:
        raise OperationalObligationError("unknown_priority")
    if isinstance(weight, bool) or not isinstance(weight, int):
        raise OperationalObligationError("invalid_priority_weight")
    if priority_class == "soft" and weight <= 0:
        raise OperationalObligationError("invalid_priority_weight")
    if priority_class != "soft" and weight != 0:
        raise OperationalObligationError("invalid_priority_weight")


def _reject_unbounded_cycles(obligations: Sequence[Mapping[str, Any]]) -> None:
    producers: dict[str, set[str]] = {}
    for record in obligations:
        consequence = record["contract"]["execution_consequence"]
        for fact in consequence["add"]:
            producers.setdefault(fact, set()).add(record["obligation_id"])
    edges: dict[str, set[str]] = {record["obligation_id"]: set() for record in obligations}
    for record in obligations:
        target = record["obligation_id"]
        for fact in record["contract"]["prerequisite"]["all_of"]:
            for source in producers.get(fact, set()):
                edges[source].add(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise OperationalObligationError("unbounded_cycle", node)
        if node in visited:
            return
        visiting.add(node)
        for child in sorted(edges[node]):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for obligation_id in sorted(edges):
        visit(obligation_id)


def compile_operational_obligations(source: bytes) -> dict[str, Any]:
    """Compile canonical five-field contracts to a deterministic automaton.

    Strict bytes make an audit meaningful. If a producer changes whitespace or
    list order, the compiler refuses the source instead of silently normalizing
    a different contract into the expected hash.
    """

    if not isinstance(source, bytes):
        raise OperationalObligationError("non_canonical_serialization", "source must be bytes")
    try:
        payload = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperationalObligationError("non_canonical_serialization") from exc
    if canonical_json_bytes(payload) != source:
        raise OperationalObligationError("non_canonical_serialization")
    if not isinstance(payload, Mapping):
        raise OperationalObligationError("invalid_obligation_envelope")
    _require_exact_keys(
        payload,
        {"obligations", "schema"},
        code="invalid_obligation_envelope",
    )
    if payload.get("schema") != OBLIGATION_SCHEMA:
        raise OperationalObligationError("unsupported_obligation_schema")
    obligations = payload.get("obligations")
    if not isinstance(obligations, list) or not obligations:
        raise OperationalObligationError("empty_obligation_set")
    for record in obligations:
        _validate_obligation(record)
    identities = [record["obligation_id"] for record in obligations]
    if len(identities) != len(set(identities)):
        raise OperationalObligationError("duplicate_obligation_id")
    if identities != sorted(identities):
        raise OperationalObligationError("non_canonical_obligation_order")
    _reject_unbounded_cycles(obligations)

    compiled: dict[str, Any] = {
        "initial_state": "awaiting_event",
        "obligations": obligations,
        "priority_order": list(PRIORITY_ORDER),
        "schema": AUTOMATON_SCHEMA,
        "source_hash": _sha256_bytes(source),
        "states": ["awaiting_event", "selected", "fallback", "no_op", "conflict"],
        "transitions": [
            {"from": "awaiting_event", "to": "conflict", "when": "invalid_event"},
            {"from": "awaiting_event", "to": "fallback", "when": "no_candidate"},
            {"from": "awaiting_event", "to": "no_op", "when": "selected_no_state_change"},
            {"from": "awaiting_event", "to": "selected", "when": "selected_state_change"},
        ],
    }
    compiled["automaton_hash"] = _sha256_bytes(canonical_json_bytes(compiled))
    return compiled


def _priority_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    contract = record["contract"]
    return (
        PRIORITY_ORDER.index(contract["priority"]["class"]),
        contract["authority"]["order"],
        record["obligation_id"],
    )


class OperationalObligationSupervisor:
    """Evaluate v3 events through exact authority and priority rules."""

    def __init__(self, compiled_automaton: Mapping[str, Any]) -> None:
        compiled = json.loads(canonical_json_bytes(compiled_automaton))
        if compiled.get("schema") != AUTOMATON_SCHEMA:
            raise OperationalObligationError("unsupported_automaton_schema")
        expected = compile_operational_obligations(
            canonical_obligation_bytes(compiled.get("obligations") or [])
        )
        if compiled != expected:
            raise OperationalObligationError("compiled_automaton_mismatch")
        self.compiled_automaton = compiled
        self._obligations = {record["obligation_id"]: record for record in compiled["obligations"]}

    @staticmethod
    def _prerequisite_active(record: Mapping[str, Any], facts: set[str]) -> bool:
        prerequisite = record["contract"]["prerequisite"]
        return set(prerequisite["all_of"]).issubset(facts) and not (
            set(prerequisite["none_of"]) & facts
        )

    @staticmethod
    def _candidate_matches(candidate: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
        return (
            candidate["action"] == record["action"]
            and record["contract"]["authority"]["issuer"] in candidate["authority_chain"]
        )

    def _validate_event(self, event: Any, expected_sequence: int) -> None:
        if not isinstance(event, Mapping):
            raise OperationalObligationError("invalid_event")
        _require_exact_keys(
            event,
            {"candidates", "event_id", "obligation_ids", "observed_facts", "sequence"},
            code="invalid_event_fields",
        )
        if event.get("sequence") != expected_sequence:
            raise OperationalObligationError("replay_reorder")
        if not isinstance(event.get("event_id"), str) or not event["event_id"]:
            raise OperationalObligationError("invalid_event_id")
        obligations = _require_sorted_fact_list(
            event.get("obligation_ids"),
            code="non_canonical_event",
            field="obligation_ids",
        )
        if any(obligation_id not in self._obligations for obligation_id in obligations):
            raise OperationalObligationError("unknown_obligation_id")
        _require_sorted_fact_list(
            event.get("observed_facts"),
            code="non_canonical_event",
            field="observed_facts",
        )
        candidates = event.get("candidates")
        if not isinstance(candidates, list):
            raise OperationalObligationError("invalid_candidates")
        candidate_ids: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise OperationalObligationError("invalid_candidate")
            _require_exact_keys(
                candidate,
                {"action", "authority_chain", "candidate_id", "soft_progress"},
                code="invalid_candidate_fields",
            )
            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise OperationalObligationError("invalid_candidate_id")
            candidate_ids.append(candidate_id)
            _validate_action(candidate.get("action"), code="invalid_candidate_action")
            authorities = _require_sorted_fact_list(
                candidate.get("authority_chain"),
                code="invalid_candidate_authority",
                field="authority_chain",
            )
            if not authorities:
                raise OperationalObligationError("invalid_candidate_authority")
            progress = candidate.get("soft_progress")
            if isinstance(progress, bool) or not isinstance(progress, int):
                raise OperationalObligationError("invalid_soft_progress")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise OperationalObligationError("duplicate_candidate_id")

    def _evaluate_event(self, event: Mapping[str, Any], facts: set[str]) -> dict[str, Any]:
        facts.update(event["observed_facts"])
        declared = [self._obligations[item] for item in event["obligation_ids"]]
        active = [record for record in declared if self._prerequisite_active(record, facts)]
        hard = [record for record in active if record["contract"]["priority"]["class"] == "hard"]
        binding = sorted(
            (record for record in active if record["contract"]["priority"]["class"] == "binding"),
            key=lambda record: (
                record["contract"]["authority"]["order"],
                record["obligation_id"],
            ),
        )
        energies: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        legal_candidates: list[
            tuple[tuple[Any, ...], Mapping[str, Any], list[Mapping[str, Any]]]
        ] = []

        for candidate in sorted(event["candidates"], key=lambda item: item["candidate_id"]):
            matched = [record for record in active if self._candidate_matches(candidate, record)]
            same_action = [record for record in active if candidate["action"] == record["action"]]
            spoofed = [
                record
                for record in same_action
                if record["contract"]["authority"]["issuer"] not in candidate["authority_chain"]
            ]
            hard_count = sum(int(record not in matched) for record in hard)
            binding_vector = [int(record not in matched) for record in binding]
            soft_score = candidate["soft_progress"] + sum(
                record["contract"]["priority"]["weight"]
                for record in matched
                if record["contract"]["priority"]["class"] == "soft"
            )
            energy = [hard_count, binding_vector, -soft_score]
            accepted = hard_count == 0 and bool(matched) and not spoofed
            energies.append(
                {
                    "accepted": accepted,
                    "candidate_id": candidate["candidate_id"],
                    "energy": energy,
                }
            )
            if accepted:
                key = (hard_count, tuple(binding_vector), -soft_score, candidate["candidate_id"])
                legal_candidates.append((key, candidate, matched))
            else:
                reason = (
                    "authority_spoof"
                    if spoofed
                    else "hard_violation"
                    if hard_count
                    else "unbound_candidate"
                )
                conflicts.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "first_conflict": (
                            spoofed[0]["obligation_id"]
                            if spoofed
                            else hard[0]["obligation_id"]
                            if hard_count
                            else None
                        ),
                        "reason": reason,
                    }
                )

        legal_candidates.sort(key=lambda item: item[0])
        legal_actions = sorted(
            {canonical_json_bytes(item[1]["action"]).decode("utf-8") for item in legal_candidates}
        )
        state_before_execution = canonical_json_bytes(sorted(facts))
        selected_candidate_id: str | None = None
        selected_energy: list[Any] | None = None
        hard_violation_count = 0

        if legal_candidates:
            key, candidate, matched = legal_candidates[0]
            selected_candidate_id = candidate["candidate_id"]
            selected_action = candidate["action"]
            selected_energy = [key[0], list(key[1]), key[2]]
            hard_violation_count = int(key[0])
            for record in matched:
                consequence = record["contract"]["execution_consequence"]
                facts.difference_update(consequence["remove"])
                facts.update(consequence["add"])
            state_bytes = canonical_json_bytes(sorted(facts))
            changed = state_bytes != state_before_execution
            certificate = {
                "kind": "selected" if changed else "no_op",
                "reason": "exact_lexicographic_minimum"
                if changed
                else "execution_consequence_already_satisfied",
            }
        else:
            selected_record = min(declared, key=_priority_key) if declared else None
            selected_action = (
                selected_record["contract"]["fallback"]["action"]
                if selected_record is not None
                else {"data": None, "kind": "NOOP"}
            )
            state_bytes = state_before_execution
            changed = False
            if not active:
                reason = "stale_prerequisite"
            elif any(item["reason"] == "authority_spoof" for item in conflicts):
                reason = "authority_spoof"
            else:
                reason = "no_legal_candidate"
            certificate = {"kind": "no_candidate", "reason": reason}

        conflict_bytes = canonical_json_bytes(conflicts)
        legal_action_bytes = canonical_json_bytes(legal_actions)
        selected_action_bytes = canonical_json_bytes(selected_action)
        row: dict[str, Any] = {
            "candidate_energies": energies,
            "certificate": certificate,
            "conflict_certificate_bytes": conflict_bytes.decode("utf-8"),
            "conflict_certificates": conflicts,
            "event_id": event["event_id"],
            "hard_violation_count": hard_violation_count,
            "legal_action_bytes": legal_action_bytes.decode("utf-8"),
            "legal_action_set": [json.loads(item) for item in legal_actions],
            "selected_action": selected_action,
            "selected_action_bytes": selected_action_bytes.decode("utf-8"),
            "selected_candidate_id": selected_candidate_id,
            "selected_energy": selected_energy,
            "sequence": event["sequence"],
            "state_bytes": state_bytes.decode("utf-8"),
            "state_changed": changed,
            "state_hash": _sha256_bytes(state_bytes),
        }
        row["row_hash"] = _sha256_bytes(canonical_json_bytes(row))
        return row

    def replay(
        self,
        event_source: bytes,
        *,
        initial_facts: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Replay canonical events and retain exact state and decision bytes."""

        if not isinstance(event_source, bytes):
            raise OperationalObligationError("non_canonical_event_serialization")
        try:
            payload = json.loads(event_source.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OperationalObligationError("non_canonical_event_serialization") from exc
        if canonical_json_bytes(payload) != event_source:
            raise OperationalObligationError("non_canonical_event_serialization")
        if not isinstance(payload, Mapping):
            raise OperationalObligationError("invalid_event_envelope")
        _require_exact_keys(payload, {"events", "schema"}, code="invalid_event_envelope")
        if payload.get("schema") != EVENT_SCHEMA:
            raise OperationalObligationError("unsupported_event_schema")
        events = payload.get("events")
        if not isinstance(events, list):
            raise OperationalObligationError("invalid_event_envelope")
        event_ids: set[str] = set()
        facts = set(
            _require_sorted_fact_list(
                sorted(initial_facts), code="invalid_initial_facts", field="initial_facts"
            )
        )
        rows: list[dict[str, Any]] = []
        for sequence, event in enumerate(events):
            self._validate_event(event, sequence)
            if event["event_id"] in event_ids:
                raise OperationalObligationError("duplicate_event_id")
            event_ids.add(event["event_id"])
            rows.append(self._evaluate_event(event, facts))
        return rows


def read_supervisor_contract(contract: Mapping[str, Any]) -> Any:
    """Read legacy v1 or compiled v3 without changing either contract."""

    schema = contract.get("schema")
    if schema == "carnot.arc.trace_fsm.v1":
        return TraceAutomatonSupervisor(contract)
    if schema == AUTOMATON_SCHEMA:
        return OperationalObligationSupervisor(contract)
    raise OperationalObligationError("unsupported_supervisor_contract", str(schema))
