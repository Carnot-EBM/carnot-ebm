"""Deterministic memory transition verifier.

This helper is intentionally small and no-LLM. It checks a proposed post-state
against explicit fixture evidence before the caller is allowed to replace a
persistent memory state. The three scores mirror the TrustMem intuition in a
repo-local, auditable form: cover required facts, preserve unrelated memories,
and make only evidence-backed changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


JsonDict = dict[str, Any]
MemoryState = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class MemoryTransitionProposal:
    """One proposed memory write with enough evidence to verify it locally."""

    transition_id: str
    label: str
    source_stress_event_id: str
    prior_state: MemoryState
    proposed_state: MemoryState
    expected_state: MemoryState
    protected_keys: tuple[str, ...]
    safe_expected: bool


@dataclass(frozen=True)
class TransitionVerification:
    """Verifier result returned before any caller commits a memory write."""

    transition_id: str
    label: str
    accepted: bool
    coverage_score: float
    preservation_score: float
    faithfulness_score: float
    rejection_reasons: tuple[str, ...]
    model_weights_mutated: bool = False

    def to_json(self) -> JsonDict:
        """Return a stable JSON-shaped representation for result artifacts."""

        return {
            "transition_id": self.transition_id,
            "label": self.label,
            "accepted": self.accepted,
            "coverage_score": self.coverage_score,
            "preservation_score": self.preservation_score,
            "faithfulness_score": self.faithfulness_score,
            "rejection_reasons": list(self.rejection_reasons),
            "model_weights_mutated": self.model_weights_mutated,
        }


class MemoryTransitionVerifier:
    """Score and gate proposed memory writes without touching model weights."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = float(threshold)

    def verify(self, proposal: MemoryTransitionProposal) -> TransitionVerification:
        """Score a proposed transition without mutating any persistent state."""

        prior = _copy_state(proposal.prior_state)
        proposed = _copy_state(proposal.proposed_state)
        expected = _copy_state(proposal.expected_state)
        coverage = self._coverage_score(proposed, expected)
        preservation = self._preservation_score(prior, proposed, proposal.protected_keys)
        faithfulness = self._faithfulness_score(prior, proposed, expected)
        reasons = self._rejection_reasons(coverage, preservation, faithfulness)
        return TransitionVerification(
            transition_id=proposal.transition_id,
            label=proposal.label,
            accepted=not reasons,
            coverage_score=coverage,
            preservation_score=preservation,
            faithfulness_score=faithfulness,
            rejection_reasons=tuple(reasons),
        )

    def commit_if_safe(
        self,
        persistent_state: MemoryState,
        proposal: MemoryTransitionProposal,
    ) -> tuple[TransitionVerification, JsonDict]:
        """Return the committed state only when verification accepts the write."""

        before = _copy_state(persistent_state)
        decision = self.verify(proposal)
        if decision.accepted:
            return decision, _copy_state(proposal.proposed_state)
        return decision, before

    def _coverage_score(self, proposed: JsonDict, expected: JsonDict) -> float:
        if not expected:
            return 1.0
        covered = sum(1 for key, value in expected.items() if proposed.get(key) == value)
        return covered / len(expected)

    def _preservation_score(
        self,
        prior: JsonDict,
        proposed: JsonDict,
        protected_keys: tuple[str, ...],
    ) -> float:
        if not protected_keys:
            return 1.0
        preserved = sum(1 for key in protected_keys if proposed.get(key) == prior.get(key))
        return preserved / len(protected_keys)

    def _faithfulness_score(
        self,
        prior: JsonDict,
        proposed: JsonDict,
        expected: JsonDict,
    ) -> float:
        changed_keys = {
            key for key in set(prior) | set(proposed) if prior.get(key) != proposed.get(key)
        }
        expected_changed_keys = {
            key for key in set(prior) | set(expected) if prior.get(key) != expected.get(key)
        }
        if not changed_keys:
            return 1.0 if not expected_changed_keys else 0.0
        faithful = sum(
            1
            for key in changed_keys
            if key in expected_changed_keys and proposed.get(key) == expected.get(key)
        )
        return faithful / len(changed_keys)

    def _rejection_reasons(
        self,
        coverage: float,
        preservation: float,
        faithfulness: float,
    ) -> list[str]:
        reasons: list[str] = []
        if coverage < self.threshold:
            reasons.append("coverage_below_threshold")
        if preservation < self.threshold:
            reasons.append("preservation_below_threshold")
        if faithfulness < self.threshold:
            reasons.append("faithfulness_below_threshold")
        return reasons


def _copy_state(state: MemoryState) -> JsonDict:
    return deepcopy({str(key): dict(value) for key, value in state.items()})
