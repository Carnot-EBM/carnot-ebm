"""Two-sided live ARC goal evidence contract for REQ-ARC-WMTE-6386.

The contract is deliberately small. It decides whether a goal predicate has
enough runtime evidence to terminate search. Search can still use an unverified
goal as a weak probe-ordering hint, but that state never earns solve credit.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


ACCEPTED = "accepted"
REJECTED = "rejected"
UNVERIFIABLE = "unverifiable"
CONTRACT_VERSION = "6386.1"
DEFAULT_EXP6258_PATH = "results/experiment_6258_goal_veto_confusion_matrix.json"


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class GoalEvidenceEvent:
    """One visible runtime event in the bounded goal-evidence window.

    The event records only what the live agent can see: a frame hash, its action,
    the legal action set, the observed level counter, and the local tick. The
    predicate result is the candidate goal's own result on that visible state.
    """

    event_id: str
    tick: int
    predicate_fired: bool
    level_before: int
    level_after: int
    action: Any = None
    legal_actions: tuple[Any, ...] = ()
    visible_frame_hash: str = ""
    deadline_tick: int | None = None
    reversal_of: str | None = None

    @property
    def level_up(self) -> bool:
        return int(self.level_after) > int(self.level_before)

    def payload(self) -> tuple[Any, ...]:
        return (
            int(self.tick),
            bool(self.predicate_fired),
            int(self.level_before),
            int(self.level_after),
            self.action,
            tuple(self.legal_actions),
            str(self.visible_frame_hash),
            self.deadline_tick,
            self.reversal_of,
        )


@dataclass(frozen=True)
class GoalEvidenceDecision:
    hypothesis_id: str
    state: str
    reasons: tuple[str, ...]
    firing_witness_ids: tuple[str, ...] = ()
    nonfiring_contrast_ids: tuple[str, ...] = ()
    window_event_ids: tuple[str, ...] = ()
    accepted_firing_witness_ids: tuple[str, ...] = ()
    accepted_nonfiring_contrast_ids: tuple[str, ...] = ()
    duplicate_event_count: int = 0
    reversed_event_ids: tuple[str, ...] = ()
    contradiction_event_ids: tuple[str, ...] = ()

    @property
    def termination_allowed(self) -> bool:
        return self.state == ACCEPTED

    @property
    def solve_credit_allowed(self) -> bool:
        return self.state == ACCEPTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "state": self.state,
            "reasons": list(self.reasons),
            "termination_allowed": self.termination_allowed,
            "solve_credit_allowed": self.solve_credit_allowed,
            "firing_witness_ids": list(self.firing_witness_ids),
            "nonfiring_contrast_ids": list(self.nonfiring_contrast_ids),
            "window_event_ids": list(self.window_event_ids),
            "accepted_firing_witness_ids": list(self.accepted_firing_witness_ids),
            "accepted_nonfiring_contrast_ids": list(self.accepted_nonfiring_contrast_ids),
            "duplicate_event_count": int(self.duplicate_event_count),
            "reversed_event_ids": list(self.reversed_event_ids),
            "contradiction_event_ids": list(self.contradiction_event_ids),
        }


@dataclass(frozen=True)
class _WindowEvents:
    events: tuple[GoalEvidenceEvent, ...]
    duplicate_count: int
    reversed_ids: tuple[str, ...]
    contradiction_ids: tuple[str, ...]


@dataclass
class TwoSidedGoalEvidenceContract:
    """Verify a goal predicate with one positive and one negative witness.

    A win-free window is not proof. It is unverifiable unless the caller supplied
    a pre-registered firing witness and the predicate fires on that level-up.
    """

    max_window_ticks: int = 64
    max_window_events: int = 128
    probe_rank_count: int = field(default=0, init=False)

    def _window_events(
        self,
        events: Sequence[GoalEvidenceEvent],
        *,
        window_start_tick: int,
        current_tick: int | None,
        deadline_tick: int | None,
    ) -> _WindowEvents:
        end_candidates = [tick for tick in (current_tick, deadline_tick) if tick is not None]
        window_end = min(end_candidates) if end_candidates else None
        by_id: dict[str, GoalEvidenceEvent] = {}
        duplicate_count = 0
        reversed_ids: set[str] = set()
        contradiction_ids: set[str] = set()

        for event in sorted(events, key=lambda item: (int(item.tick), str(item.event_id))):
            if int(event.tick) < int(window_start_tick):
                continue
            if int(event.tick) - int(window_start_tick) > int(self.max_window_ticks):
                continue
            if window_end is not None and int(event.tick) > int(window_end):
                continue
            if event.reversal_of:
                reversed_ids.add(str(event.reversal_of))
                continue
            existing = by_id.get(str(event.event_id))
            if existing is not None:
                if existing.payload() == event.payload():
                    duplicate_count += 1
                    continue
                contradiction_ids.add(str(event.event_id))
                continue
            if len(by_id) >= int(self.max_window_events):
                continue
            by_id[str(event.event_id)] = event

        kept = tuple(
            event for event_id, event in by_id.items() if str(event_id) not in reversed_ids
        )
        return _WindowEvents(
            events=kept,
            duplicate_count=duplicate_count,
            reversed_ids=tuple(sorted(reversed_ids)),
            contradiction_ids=tuple(sorted(contradiction_ids)),
        )

    def evaluate(
        self,
        hypothesis_id: str,
        events: Sequence[GoalEvidenceEvent],
        *,
        firing_witness_ids: Sequence[str],
        nonfiring_contrast_ids: Sequence[str],
        window_start_tick: int = 0,
        current_tick: int | None = None,
        deadline_tick: int | None = None,
    ) -> GoalEvidenceDecision:
        window = self._window_events(
            events,
            window_start_tick=window_start_tick,
            current_tick=current_tick,
            deadline_tick=deadline_tick,
        )
        firing_ids = tuple(str(item) for item in firing_witness_ids)
        contrast_ids = tuple(str(item) for item in nonfiring_contrast_ids)
        event_by_id = {str(event.event_id): event for event in window.events}
        reasons: list[str] = []
        hard_reject = False

        if window.contradiction_ids:
            reasons.append("contradictory_duplicate")
            hard_reject = True

        if not firing_ids:
            reasons.append("no_preregistered_firing_witness")
        if not contrast_ids:
            reasons.append("no_preregistered_nonfiring_contrast")

        accepted_firing: list[str] = []
        firing_present_but_failed = False
        for event_id in firing_ids:
            event = event_by_id.get(event_id)
            if event is None:
                continue
            if event.level_up and event.predicate_fired:
                accepted_firing.append(event_id)
            elif event.level_up or event.predicate_fired:
                firing_present_but_failed = True

        if not accepted_firing:
            reasons.append("missing_firing_witness")
            if firing_present_but_failed:
                hard_reject = True

        accepted_contrast: list[str] = []
        for event_id in contrast_ids:
            event = event_by_id.get(event_id)
            if event is None:
                continue
            if event.level_up:
                reasons.append("contrast_leveled_up")
                hard_reject = True
            elif event.predicate_fired:
                reasons.append("contrast_fired")
                hard_reject = True
            else:
                accepted_contrast.append(event_id)

        if not accepted_contrast:
            reasons.append("missing_nonfiring_contrast")

        if hard_reject:
            state = REJECTED
        elif accepted_firing and accepted_contrast:
            state = ACCEPTED
            reasons = ["bounded_two_sided_evidence"]
        else:
            state = UNVERIFIABLE

        return GoalEvidenceDecision(
            hypothesis_id=str(hypothesis_id),
            state=state,
            reasons=tuple(dict.fromkeys(reasons)),
            firing_witness_ids=firing_ids,
            nonfiring_contrast_ids=contrast_ids,
            window_event_ids=tuple(str(event.event_id) for event in window.events),
            accepted_firing_witness_ids=tuple(accepted_firing),
            accepted_nonfiring_contrast_ids=tuple(accepted_contrast),
            duplicate_event_count=int(window.duplicate_count),
            reversed_event_ids=window.reversed_ids,
            contradiction_event_ids=window.contradiction_ids,
        )

    def rank_one_legal_probe(
        self,
        decision: GoalEvidenceDecision,
        candidates: Sequence[Mapping[str, Any]],
        *,
        legal_actions: Sequence[Any],
        preferred_action: Any,
    ) -> list[dict[str, Any]]:
        rows = [dict(row) for row in candidates]
        if decision.state != UNVERIFIABLE:
            return rows
        legal = set(legal_actions)
        if preferred_action not in legal:
            return rows
        for index, row in enumerate(rows):
            action = row.get("action", row.get("action_id"))
            if action == preferred_action:
                self.probe_rank_count += 1
                return [rows[index], *rows[:index], *rows[index + 1 :]]
        return rows


class LiveTwoSidedGoalContract:
    """Default-off live adapter for E3AgentPolicy and StepwiseExplorer.

    The adapter starts in `unverifiable`. Tests and future measured arms can set
    an accepted decision, but the shipped default has no path to termination.
    """

    def __init__(
        self,
        decision: GoalEvidenceDecision | None = None,
        contract: TwoSidedGoalEvidenceContract | None = None,
    ) -> None:
        self.contract = contract or TwoSidedGoalEvidenceContract()
        self._decision = decision
        self._last_diagnostics: dict[str, Any] = {
            "enabled": True,
            "state": UNVERIFIABLE,
            "termination_allowed": False,
            "solve_credit_allowed": False,
        }

    def set_decision(self, decision: GoalEvidenceDecision) -> None:
        self._decision = decision

    def termination_decision(self) -> GoalEvidenceDecision:
        if self._decision is not None:
            return self._decision
        return GoalEvidenceDecision(
            hypothesis_id="live_goal",
            state=UNVERIFIABLE,
            reasons=("missing_live_two_sided_evidence",),
        )

    def guard_predicate(self, predicate):
        def _guarded(value: Any) -> bool:
            decision = self.termination_decision()
            self._last_diagnostics = {
                "enabled": True,
                "state": decision.state,
                "termination_allowed": decision.termination_allowed,
                "solve_credit_allowed": decision.solve_credit_allowed,
                "reasons": list(decision.reasons),
            }
            if not decision.termination_allowed:
                return False
            try:
                return bool(predicate(value))
            except Exception:
                return False

        return _guarded

    def rank_candidates(self, frame: Any, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        decision = self.termination_decision()
        legal_actions = tuple(getattr(frame, "available_actions", ()) or ())
        preferred_action = getattr(frame, "preferred_goal_probe_action", None)
        ranked = self.contract.rank_one_legal_probe(
            decision,
            candidates,
            legal_actions=legal_actions,
            preferred_action=preferred_action,
        )
        self._last_diagnostics = {
            "enabled": True,
            "state": decision.state,
            "termination_allowed": decision.termination_allowed,
            "solve_credit_allowed": decision.solve_credit_allowed,
            "probe_rank_count": int(self.contract.probe_rank_count),
        }
        return ranked

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)


def coerce_two_sided_goal_contract(value: Any | bool | None = None) -> LiveTwoSidedGoalContract | None:
    if value is None:
        value = os.environ.get("CARNOT_ARC_TWO_SIDED_GOAL_CONTRACT", "0") == "1"
    if value is False:
        return None
    if isinstance(value, LiveTwoSidedGoalContract):
        return value
    if hasattr(value, "termination_decision") and hasattr(value, "rank_candidates"):
        return value
    if value is True:
        return LiveTwoSidedGoalContract()
    return None


def exp6258_fixture_boundary(root: Path | str) -> dict[str, Any]:
    path = Path(root) / DEFAULT_EXP6258_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))
    counts = {
        "true_accept": int(artifact.get("n_true_accept", 0)),
        "false_accept": int(artifact.get("n_false_accept_degenerate_admitted", 0)),
        "false_reject": int(artifact.get("n_false_reject_discriminating_discarded", 0)),
        "true_reject": int(artifact.get("n_true_reject", 0)),
    }
    return {
        "path": str(path),
        "path_sha256": sha256_file(path),
        "n_predicates": int(artifact.get("n_predicates", 0)),
        "acceptance_precision": float(artifact.get("acceptance_precision", 0.0)),
        "counts": counts,
        "confusion_boundary": (
            "old live veto accepted specificity==1.0 even when fires_on_real_win was false"
        ),
    }


def _old_label(outcome: str) -> str:
    lowered = str(outcome).lower()
    if lowered.startswith("false_accept"):
        return "false_accept"
    if lowered.startswith("false_reject"):
        return "false_reject"
    if lowered == "true_accept":
        return "true_accept"
    if lowered == "true_reject":
        return "true_reject"
    return lowered


def _decision_for_exp6258_row(
    row: Mapping[str, Any],
    contract: TwoSidedGoalEvidenceContract,
) -> GoalEvidenceDecision:
    specificity = float(row.get("specificity", 0.0))
    fires_on_real_win = bool(row.get("fires_on_real_win"))
    event_prefix = f"{row.get('source')}:{row.get('game')}:{row.get('arm')}"
    win = GoalEvidenceEvent(
        event_id=f"{event_prefix}:win",
        tick=1,
        predicate_fired=fires_on_real_win,
        level_before=0,
        level_after=1,
        action=0,
        legal_actions=(0, 1, 2, 3),
        visible_frame_hash=f"{event_prefix}:visible-win",
    )
    contrast = GoalEvidenceEvent(
        event_id=f"{event_prefix}:contrast",
        tick=2,
        predicate_fired=specificity < 1.0,
        level_before=1,
        level_after=1,
        action=1,
        legal_actions=(0, 1, 2, 3),
        visible_frame_hash=f"{event_prefix}:visible-contrast",
    )
    return contract.evaluate(
        event_prefix,
        [win, contrast],
        firing_witness_ids=(win.event_id,),
        nonfiring_contrast_ids=(contrast.event_id,),
        window_start_tick=0,
        current_tick=2,
        deadline_tick=3,
    )


def replay_exp6258_contract(root: Path | str) -> dict[str, Any]:
    path = Path(root) / DEFAULT_EXP6258_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))
    contract = TwoSidedGoalEvidenceContract()
    old_counts = {
        "true_accept": int(artifact.get("n_true_accept", 0)),
        "false_accept": int(artifact.get("n_false_accept_degenerate_admitted", 0)),
        "false_reject": int(artifact.get("n_false_reject_discriminating_discarded", 0)),
        "true_reject": int(artifact.get("n_true_reject", 0)),
    }
    new_counts = {ACCEPTED: 0, REJECTED: 0, UNVERIFIABLE: 0}
    rows: list[dict[str, Any]] = []
    new_false_accept = 0
    new_false_reject = 0
    prior_false_accept_fixed = 0

    for row in artifact.get("per_predicate", []):
        decision = _decision_for_exp6258_row(row, contract)
        old = _old_label(str(row.get("outcome", "")))
        new_counts[decision.state] += 1
        if old == "false_accept" and decision.state == ACCEPTED:
            new_false_accept += 1
        if old == "true_accept" and decision.state != ACCEPTED:
            new_false_reject += 1
        if old == "false_accept" and decision.state in {REJECTED, UNVERIFIABLE}:
            prior_false_accept_fixed += 1
        rows.append(
            {
                "source": row.get("source"),
                "game": row.get("game"),
                "arm": row.get("arm"),
                "specificity": row.get("specificity"),
                "fires_on_real_win": row.get("fires_on_real_win"),
                "old_label": old,
                "new_state": decision.state,
                "new_reasons": list(decision.reasons),
            }
        )

    accepted_total = int(new_counts[ACCEPTED])
    accepted_true = sum(
        1 for row in rows if row["old_label"] == "true_accept" and row["new_state"] == ACCEPTED
    )
    total = len(rows)
    return {
        "old_confusion_matrix": old_counts,
        "new_confusion_matrix": dict(new_counts),
        "per_predicate": rows,
        "new_false_accept_count": int(new_false_accept),
        "new_false_reject_count": int(new_false_reject),
        "prior_false_accepts_rejected_or_unverifiable": int(prior_false_accept_fixed),
        "admission_precision": accepted_true / accepted_total if accepted_total else 0.0,
        "admission_coverage": accepted_total / total if total else 0.0,
    }


def adversarial_fixture_manifest() -> dict[str, Any]:
    contract = TwoSidedGoalEvidenceContract(max_window_ticks=5)
    fixtures = {
        "constant_false": contract.evaluate(
            "constant_false",
            [
                GoalEvidenceEvent("win", 1, False, 0, 1),
                GoalEvidenceEvent("contrast", 2, False, 1, 1),
            ],
            firing_witness_ids=("win",),
            nonfiring_contrast_ids=("contrast",),
            deadline_tick=3,
            current_tick=2,
        ),
        "constant_true": contract.evaluate(
            "constant_true",
            [
                GoalEvidenceEvent("win", 1, True, 0, 1),
                GoalEvidenceEvent("contrast", 2, True, 1, 1),
            ],
            firing_witness_ids=("win",),
            nonfiring_contrast_ids=("contrast",),
            deadline_tick=3,
            current_tick=2,
        ),
        "delayed_trigger_before_deadline": contract.evaluate(
            "delayed_trigger_before_deadline",
            [
                GoalEvidenceEvent("contrast", 1, False, 0, 0),
                GoalEvidenceEvent("win", 4, True, 0, 1),
            ],
            firing_witness_ids=("win",),
            nonfiring_contrast_ids=("contrast",),
            deadline_tick=5,
            current_tick=4,
        ),
        "duplicate_identical": contract.evaluate(
            "duplicate_identical",
            [
                GoalEvidenceEvent("win", 1, True, 0, 1),
                GoalEvidenceEvent("win", 1, True, 0, 1),
                GoalEvidenceEvent("contrast", 2, False, 1, 1),
            ],
            firing_witness_ids=("win",),
            nonfiring_contrast_ids=("contrast",),
            deadline_tick=3,
            current_tick=2,
        ),
        "contradictory_duplicate": contract.evaluate(
            "contradictory_duplicate",
            [
                GoalEvidenceEvent("win", 1, True, 0, 1),
                GoalEvidenceEvent("win", 1, False, 0, 1),
                GoalEvidenceEvent("contrast", 2, False, 1, 1),
            ],
            firing_witness_ids=("win",),
            nonfiring_contrast_ids=("contrast",),
            deadline_tick=3,
            current_tick=2,
        ),
        "no_win_window": contract.evaluate(
            "no_win_window",
            [GoalEvidenceEvent("contrast", 1, False, 0, 0)],
            firing_witness_ids=("win",),
            nonfiring_contrast_ids=("contrast",),
            deadline_tick=3,
            current_tick=1,
        ),
    }
    return {name: decision.as_dict() for name, decision in fixtures.items()}
