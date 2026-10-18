"""Bounded active reward-machine frontier for visible ARC events.

Spec refs: REQ-ARC-ARM-6387,
SCENARIO-ARC-ARM-6387-LEGAL-DISAGREEMENT,
SCENARIO-ARC-ARM-6387-ABSTAIN-AND-BOUNDS,
SCENARIO-ARC-ARM-6387-TWO-SIDED-EVIDENCE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_two_sided_goal_contract import (
    GoalEvidenceEvent,
    TwoSidedGoalEvidenceContract,
)


REWARD_MACHINE_FRONTIER_VERSION = "6387.1"
SAME_FRAME_NO_LEVEL = "same_frame_no_level"
FRAME_CHANGED_NO_LEVEL = "frame_changed_no_level"
LEVEL_UP = "level_up"
UNKNOWN = "unknown"
VISIBLE_EVENT_SYMBOLS = (SAME_FRAME_NO_LEVEL, FRAME_CHANGED_NO_LEVEL, LEVEL_UP)


@dataclass(frozen=True)
class TransitionEvidence:
    """Visible source record that supports one automaton transition."""

    source_transition_id: str
    source_tick: int
    source_action: int
    observed_symbol: str
    visible_frame_hash_before: str
    visible_frame_hash_after: str
    source: str = "live_agent_visible_transition"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_transition_id": str(self.source_transition_id),
            "source_tick": int(self.source_tick),
            "source_action": int(self.source_action),
            "observed_symbol": str(self.observed_symbol),
            "visible_frame_hash_before": str(self.visible_frame_hash_before),
            "visible_frame_hash_after": str(self.visible_frame_hash_after),
            "source": str(self.source),
        }


@dataclass(frozen=True)
class RewardMachineTransition:
    source_state: str
    action: int
    target_state: str
    predicted_symbol: str
    data: Any = None
    evidence: tuple[TransitionEvidence, ...] = ()

    def matches(self, action: int, data: Any = None) -> bool:
        if int(self.action) != int(action):
            return False
        return self.data is None or _stable_payload(self.data) == _stable_payload(data)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_state": str(self.source_state),
            "action": int(self.action),
            "data": self.data,
            "target_state": str(self.target_state),
            "predicted_symbol": str(self.predicted_symbol),
            "evidence": [row.as_dict() for row in self.evidence],
        }


@dataclass(frozen=True)
class RewardMachineHypothesis:
    hypothesis_id: str
    states: tuple[str, ...]
    start_state: str
    current_state: str
    transitions: tuple[RewardMachineTransition, ...]
    hidden_source_path: str = ""

    @property
    def evidence_count(self) -> int:
        return sum(len(transition.evidence) for transition in self.transitions)

    def predict(self, action: int, data: Any = None) -> str:
        for transition in self.transitions:
            if transition.source_state == self.current_state and transition.matches(action, data):
                return str(transition.predicted_symbol)
        return UNKNOWN

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": str(self.hypothesis_id),
            "states": list(self.states),
            "start_state": str(self.start_state),
            "current_state": str(self.current_state),
            "transitions": [transition.as_dict() for transition in self.transitions],
            "evidence_count": int(self.evidence_count),
            "hidden_source_path": str(self.hidden_source_path),
        }


@dataclass(frozen=True)
class ProbeSelection:
    action: int | None
    data: Any = None
    reason: str = ""
    expected_elimination: int = 0
    prediction_buckets: tuple[dict[str, Any], ...] = ()
    frozen_hypothesis_ids: tuple[str, ...] = ()
    frozen_predictions: dict[str, str] = field(default_factory=dict)
    legal_actions: tuple[int, ...] = ()
    fallback_action: Any = None
    frozen_before_outcome: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "data": self.data,
            "reason": str(self.reason),
            "expected_elimination": int(self.expected_elimination),
            "prediction_buckets": [dict(row) for row in self.prediction_buckets],
            "frozen_hypothesis_ids": list(self.frozen_hypothesis_ids),
            "frozen_predictions": dict(self.frozen_predictions),
            "legal_actions": list(self.legal_actions),
            "fallback_action": self.fallback_action,
            "frozen_before_outcome": bool(self.frozen_before_outcome),
        }


@dataclass(frozen=True)
class FrontierUpdate:
    state: str
    observed_symbol: str = ""
    evaluated_hypothesis_ids: tuple[str, ...] = ()
    kept_hypothesis_ids: tuple[str, ...] = ()
    eliminated_hypothesis_ids: tuple[str, ...] = ()
    wrong_elimination_count: int = 0
    action_frozen_before_outcome: bool = False
    two_sided_admission: dict[str, dict[str, Any]] = field(default_factory=dict)
    arc_solve_claim: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": str(self.state),
            "observed_symbol": str(self.observed_symbol),
            "evaluated_hypothesis_ids": list(self.evaluated_hypothesis_ids),
            "kept_hypothesis_ids": list(self.kept_hypothesis_ids),
            "eliminated_hypothesis_ids": list(self.eliminated_hypothesis_ids),
            "wrong_elimination_count": int(self.wrong_elimination_count),
            "action_frozen_before_outcome": bool(self.action_frozen_before_outcome),
            "two_sided_admission": {
                key: dict(value) for key, value in self.two_sided_admission.items()
            },
            "arc_solve_claim": False,
        }


@dataclass(frozen=True)
class _PendingProbe:
    action: int
    data: Any
    tick: int
    legal_actions: tuple[int, ...]
    hypothesis_ids: tuple[str, ...]
    predictions: dict[str, str]


class RewardMachineFrontier:
    """Small deterministic frontier that only acts on legal prediction splits."""

    def __init__(
        self,
        hypotheses: Sequence[RewardMachineHypothesis] = (),
        *,
        capacity: int = 5,
        timeout_ticks: int = 8,
        contract: TwoSidedGoalEvidenceContract | None = None,
        enabled: bool = True,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.timeout_ticks = max(1, int(timeout_ticks))
        self.contract = contract or TwoSidedGoalEvidenceContract()
        self.enabled = bool(enabled)
        self.hypotheses: dict[str, RewardMachineHypothesis] = {}
        self._order: dict[str, int] = {}
        self._next_order = 0
        self._pending: _PendingProbe | None = None
        self._seen_evidence_ids: set[str] = set()
        self._events: dict[str, list[GoalEvidenceEvent]] = {}
        self._firing_ids: dict[str, set[str]] = {}
        self._contrast_ids: dict[str, set[str]] = {}
        self._stats: dict[str, int] = {
            "eviction_count": 0,
            "abstention_count": 0,
            "contradiction_count": 0,
            "timeout_count": 0,
            "duplicate_evidence_count": 0,
            "hypothesis_elimination_count": 0,
            "wrong_elimination_count": 0,
            "legal_action_mutation_count": 0,
            "base_policy_fallback_count": 0,
            "probe_selection_count": 0,
        }
        self._latency_s_total = 0.0
        for hypothesis in hypotheses:
            self.add_hypothesis(hypothesis)

    def add_hypothesis(self, hypothesis: RewardMachineHypothesis) -> None:
        hypothesis_id = str(hypothesis.hypothesis_id)
        if hypothesis_id not in self._order:
            self._order[hypothesis_id] = self._next_order
            self._next_order += 1
        self.hypotheses[hypothesis_id] = hypothesis
        while len(self.hypotheses) > self.capacity:
            evicted = min(self.hypotheses, key=lambda item: self._order.get(item, 0))
            self.hypotheses.pop(evicted, None)
            self._stats["eviction_count"] += 1

    def choose_legal_disagreement(
        self,
        *,
        legal_actions: Sequence[Any],
        candidate_actions: Sequence[Any] | None = None,
        tick: int,
        base_policy_action: Any = None,
    ) -> ProbeSelection:
        started = time.perf_counter()
        try:
            legal = _normalise_legal_actions(legal_actions)
            candidates = (
                _normalise_legal_actions(candidate_actions)
                if candidate_actions is not None
                else legal
            )
            legal_candidates = tuple(action for action in candidates if action in set(legal))
            best = self._best_legal_split(legal_candidates)
            if best is None:
                self._stats["abstention_count"] += 1
                if base_policy_action is not None:
                    self._stats["base_policy_fallback_count"] += 1
                return ProbeSelection(
                    action=None,
                    reason="no_safe_legal_disagreement",
                    legal_actions=legal,
                    fallback_action=base_policy_action,
                )

            action, expected, buckets, predictions = best
            hypothesis_ids = tuple(sorted(predictions))
            self._pending = _PendingProbe(
                action=action,
                data=None,
                tick=int(tick),
                legal_actions=legal,
                hypothesis_ids=hypothesis_ids,
                predictions=dict(predictions),
            )
            self._stats["probe_selection_count"] += 1
            return ProbeSelection(
                action=action,
                data=None,
                reason="legal_prediction_disagreement",
                expected_elimination=expected,
                prediction_buckets=tuple(buckets),
                frozen_hypothesis_ids=hypothesis_ids,
                frozen_predictions=dict(predictions),
                legal_actions=legal,
                fallback_action=base_policy_action,
                frozen_before_outcome=True,
            )
        finally:
            self._latency_s_total += time.perf_counter() - started

    def force_freeze_for_testing(self, *, action: int, tick: int, data: Any = None) -> None:
        predictions = {
            hypothesis_id: hypothesis.predict(int(action), data)
            for hypothesis_id, hypothesis in self.hypotheses.items()
        }
        self._pending = _PendingProbe(
            action=int(action),
            data=data,
            tick=int(tick),
            legal_actions=(int(action),),
            hypothesis_ids=tuple(sorted(predictions)),
            predictions=dict(predictions),
        )

    def observe_action_result(
        self,
        *,
        action: int,
        tick: int,
        level_before: int,
        level_after: int,
        frame_before_hash: str,
        frame_after_hash: str,
        source_transition_id: str,
        data: Any = None,
    ) -> FrontierUpdate:
        started = time.perf_counter()
        try:
            source_id = str(source_transition_id)
            if source_id in self._seen_evidence_ids:
                self._stats["duplicate_evidence_count"] += 1
                return FrontierUpdate(state="duplicate")
            pending = self._pending
            if pending is None:
                return FrontierUpdate(state="no_pending_probe")
            if int(tick) - int(pending.tick) > self.timeout_ticks:
                self._pending = None
                self._stats["timeout_count"] += 1
                return FrontierUpdate(
                    state="timeout",
                    evaluated_hypothesis_ids=pending.hypothesis_ids,
                    action_frozen_before_outcome=True,
                )

            self._seen_evidence_ids.add(source_id)
            observed = visible_event_symbol(
                level_before=level_before,
                level_after=level_after,
                frame_before_hash=frame_before_hash,
                frame_after_hash=frame_after_hash,
            )
            evaluated = pending.hypothesis_ids
            matched = tuple(
                hypothesis_id
                for hypothesis_id in evaluated
                if pending.predictions.get(hypothesis_id) == observed
            )
            two_sided = self._feed_two_sided_contract(
                pending,
                observed_tick=int(tick),
                level_before=int(level_before),
                level_after=int(level_after),
                frame_after_hash=str(frame_after_hash),
                source_transition_id=source_id,
            )
            if not matched:
                self._pending = None
                self._stats["contradiction_count"] += 1
                return FrontierUpdate(
                    state="contradiction",
                    observed_symbol=observed,
                    evaluated_hypothesis_ids=evaluated,
                    action_frozen_before_outcome=True,
                    two_sided_admission=two_sided,
                )

            eliminated = tuple(hypothesis_id for hypothesis_id in evaluated if hypothesis_id not in matched)
            for hypothesis_id in eliminated:
                self.hypotheses.pop(hypothesis_id, None)
            self._stats["hypothesis_elimination_count"] += len(eliminated)
            self._pending = None
            return FrontierUpdate(
                state="evaluated",
                observed_symbol=observed,
                evaluated_hypothesis_ids=evaluated,
                kept_hypothesis_ids=matched,
                eliminated_hypothesis_ids=eliminated,
                wrong_elimination_count=0,
                action_frozen_before_outcome=True,
                two_sided_admission=two_sided,
            )
        finally:
            self._latency_s_total += time.perf_counter() - started

    def diagnostics(self) -> dict[str, Any]:
        evidence_events = sum(len(events) for events in self._events.values())
        return {
            "enabled": bool(self.enabled),
            "version": REWARD_MACHINE_FRONTIER_VERSION,
            "n_hypotheses": int(len(self.hypotheses)),
            "capacity": int(self.capacity),
            "timeout_ticks": int(self.timeout_ticks),
            "hypotheses": [hypothesis.as_dict() for hypothesis in self.hypotheses.values()],
            "pending_probe": self._pending is not None,
            "evidence_event_count": int(evidence_events),
            "latency_s_total": round(float(self._latency_s_total), 8),
            **{key: int(value) for key, value in self._stats.items()},
        }

    def _best_legal_split(
        self,
        legal_candidates: Sequence[int],
    ) -> tuple[int, int, list[dict[str, Any]], dict[str, str]] | None:
        if not self.enabled or len(self.hypotheses) < 2:
            return None
        scored: list[tuple[int, int, list[dict[str, Any]], dict[str, str]]] = []
        for action in legal_candidates:
            predictions = {
                hypothesis_id: hypothesis.predict(action)
                for hypothesis_id, hypothesis in self.hypotheses.items()
            }
            if any(symbol == UNKNOWN for symbol in predictions.values()):
                continue
            buckets = _prediction_buckets(predictions)
            if len(buckets) < 2:
                continue
            largest_bucket = max(int(bucket["n_hypotheses"]) for bucket in buckets)
            expected = len(predictions) - largest_bucket
            if expected <= 0:  # pragma: no cover
                continue
            scored.append((int(action), int(expected), buckets, predictions))
        if not scored:
            return None
        return sorted(scored, key=lambda row: (-row[1], row[0]))[0]

    def _feed_two_sided_contract(
        self,
        pending: _PendingProbe,
        *,
        observed_tick: int,
        level_before: int,
        level_after: int,
        frame_after_hash: str,
        source_transition_id: str,
    ) -> dict[str, dict[str, Any]]:
        decisions: dict[str, dict[str, Any]] = {}
        for hypothesis_id in pending.hypothesis_ids:
            predicted = pending.predictions.get(hypothesis_id, UNKNOWN)
            event_id = f"{source_transition_id}:{hypothesis_id}"
            event = GoalEvidenceEvent(
                event_id=event_id,
                tick=int(observed_tick),
                predicate_fired=(predicted == LEVEL_UP),
                level_before=int(level_before),
                level_after=int(level_after),
                action=int(pending.action),
                legal_actions=tuple(int(action) for action in pending.legal_actions),
                visible_frame_hash=str(frame_after_hash),
                deadline_tick=int(pending.tick) + int(self.timeout_ticks),
            )
            self._events.setdefault(hypothesis_id, []).append(event)
            if predicted == LEVEL_UP:
                self._firing_ids.setdefault(hypothesis_id, set()).add(event_id)
            else:
                self._contrast_ids.setdefault(hypothesis_id, set()).add(event_id)
            decision = self.contract.evaluate(
                hypothesis_id,
                self._events[hypothesis_id],
                firing_witness_ids=sorted(self._firing_ids.get(hypothesis_id, set())),
                nonfiring_contrast_ids=sorted(self._contrast_ids.get(hypothesis_id, set())),
                window_start_tick=0,
                current_tick=int(observed_tick),
                deadline_tick=int(pending.tick) + int(self.timeout_ticks),
            )
            decisions[hypothesis_id] = decision.as_dict()
        return decisions


def visible_event_symbol(
    *,
    level_before: int,
    level_after: int,
    frame_before_hash: str,
    frame_after_hash: str,
) -> str:
    if int(level_after) > int(level_before):
        return LEVEL_UP
    if str(frame_before_hash) == str(frame_after_hash):
        return SAME_FRAME_NO_LEVEL
    return FRAME_CHANGED_NO_LEVEL


def reward_machine_frontier_from_transitions(
    transitions: Sequence[Any],
    *,
    capacity: int = 5,
    timeout_ticks: int = 8,
) -> RewardMachineFrontier:
    hypotheses: list[RewardMachineHypothesis] = []
    for index, transition in enumerate(list(transitions)[: max(1, int(capacity))]):
        action = int(getattr(transition, "action", 0) or 0)
        before_hash = _array_hash(getattr(transition, "grid", []))
        after_hash = _array_hash(getattr(transition, "next_grid", []))
        symbol = visible_event_symbol(
            level_before=int(getattr(transition, "level_before", 0) or 0),
            level_after=int(getattr(transition, "level_after", 0) or 0),
            frame_before_hash=before_hash,
            frame_after_hash=after_hash,
        )
        evidence = TransitionEvidence(
            source_transition_id=f"live-transition:{index}",
            source_tick=index,
            source_action=action,
            observed_symbol=symbol,
            visible_frame_hash_before=before_hash,
            visible_frame_hash_after=after_hash,
        )
        hypotheses.append(
            RewardMachineHypothesis(
                hypothesis_id=f"observed_visible_event_{index}",
                states=("q0", f"q_{symbol}"),
                start_state="q0",
                current_state="q0",
                transitions=(
                    RewardMachineTransition(
                        source_state="q0",
                        action=action,
                        target_state=f"q_{symbol}",
                        predicted_symbol=symbol,
                        data=getattr(transition, "data", None),
                        evidence=(evidence,),
                    ),
                ),
            )
        )
    return RewardMachineFrontier(
        hypotheses,
        capacity=capacity,
        timeout_ticks=timeout_ticks,
    )


def default_fixture_manifest() -> dict[str, Any]:
    return {
        "version": REWARD_MACHINE_FRONTIER_VERSION,
        "game_blind": True,
        "fixtures": {
            "unique_disagreement": {
                "hypotheses": 2,
                "legal_actions": [1, 2],
                "expected_probe_action": 2,
            },
            "no_disagreement": {
                "hypotheses": 2,
                "expected": "base_policy_fallback",
            },
            "delayed_evidence": {
                "deadline_ticks": 1,
                "expected": "timeout_no_elimination",
            },
            "repeated_frames": {
                "visible_symbol": SAME_FRAME_NO_LEVEL,
                "expected": "duplicate_collapsed",
            },
            "contradictory_evidence": {
                "expected": "contradiction_no_wrong_elimination",
            },
            "deadline_timeout": {
                "expected": "stale_probe_abstention",
            },
        },
        "visible_event_symbols": list(VISIBLE_EVENT_SYMBOLS),
        "candidate_count_range": [2, 5],
        "forbidden_access_counts": {
            "hidden_source_reads": 0,
            "offline_search_calls": 0,
            "adapter_lookup_calls": 0,
            "oracle_result_before_action_reads": 0,
        },
    }


def hypothesis_capacity_eviction_abstention_and_timeout_rules() -> dict[str, Any]:
    return {
        "capacity": "bounded integer; Exp6387 default is five active hypotheses",
        "eviction": "deterministic oldest-active eviction when capacity is exceeded",
        "abstention": "no safe legal disagreement, unknown prediction, or fewer than two hypotheses",
        "contradiction": "all active predictions mismatch the outcome; no mass elimination",
        "duplicate": "source_transition_id is processed once",
        "timeout": "pending probe expires after timeout_ticks before evidence can update beliefs",
        "action_freeze": "action, legal set, active ids, and predictions are copied before outcome",
    }


def _prediction_buckets(predictions: Mapping[str, str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for hypothesis_id, symbol in predictions.items():
        grouped.setdefault(str(symbol), []).append(str(hypothesis_id))
    return [
        {
            "predicted_symbol": symbol,
            "hypotheses": sorted(ids),
            "n_hypotheses": len(ids),
        }
        for symbol, ids in sorted(grouped.items())
    ]


def _normalise_legal_actions(raw_actions: Sequence[Any] | None) -> tuple[int, ...]:
    out: list[int] = []
    for raw in raw_actions or ():
        if isinstance(raw, Mapping):
            raw = raw.get("action", raw.get("action_id"))
        elif isinstance(raw, tuple) and raw:
            raw = raw[0]
        if hasattr(raw, "value"):
            raw = raw.value
        if isinstance(raw, str) and raw.upper().startswith("ACTION"):
            raw = raw.upper().replace("ACTION", "", 1)
        try:
            action = int(raw)
        except (TypeError, ValueError):
            continue
        if action != 0 and action not in out:
            out.append(action)
    return tuple(out)


def _stable_payload(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _array_hash(value: Any) -> str:
    arr = np.asarray(value)
    payload = {
        "shape": list(arr.shape),
        "values": arr.astype(int, copy=False).tolist() if arr.size else [],
    }
    return _stable_payload(payload)
