"""Opt-in query-time verifier signal from verified FR-11 memory.

The policy in this module is deliberately tiny: a caller supplies a bounded
replay case set, a verified-memory index, and an explicit opt-in flag. When the
flag is absent or false, the policy suppresses memory hits and returns the
baseline verifier-only decision. This keeps SessionMemory useful as a measured
query-time signal without making learned memory global default behavior.

Spec: REQ-LEARN-1484, SCENARIO-LEARN-1484, SCENARIO-LEARN-1485.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


BASELINE_SIGNAL = "baseline_verifier_only"
MEMORY_SIGNAL = "verified_memory_repair_hint"


@dataclass(frozen=True)
class QueryReplayCase:
    """A single replay row with the expected memory-signal behavior."""

    case_id: str
    expects_memory_signal: bool
    source: str = "bounded_replay"


@dataclass(frozen=True)
class VerifiedMemoryIndex:
    """Set of memory IDs allowed to influence query-time verification."""

    verified_ids: frozenset[str]

    @classmethod
    def from_ids(cls, ids: Iterable[str]) -> VerifiedMemoryIndex:
        return cls(frozenset(str(item) for item in ids))

    def contains(self, case_id: str) -> bool:
        return str(case_id) in self.verified_ids


@dataclass(frozen=True)
class QueryMemoryDecision:
    """Policy decision for one replay case."""

    case_id: str
    memory_enabled: bool
    memory_hit: bool
    verifier_signal: str
    task_success: bool
    soundness_mistake: bool
    completeness_mistake: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "memory_enabled": self.memory_enabled,
            "memory_hit": self.memory_hit,
            "verifier_signal": self.verifier_signal,
            "task_success": self.task_success,
            "soundness_mistake": self.soundness_mistake,
            "completeness_mistake": self.completeness_mistake,
        }


@dataclass(frozen=True)
class QueryMemoryEvaluation:
    """Aggregate metrics for a memory-disabled or memory-enabled replay pass."""

    memory_enabled: bool
    decisions: tuple[QueryMemoryDecision, ...]

    @property
    def task_success_rate(self) -> float:
        if not self.decisions:
            return 0.0
        successes = sum(1 for decision in self.decisions if decision.task_success)
        return successes / len(self.decisions)

    @property
    def soundness_mistakes(self) -> int:
        return sum(1 for decision in self.decisions if decision.soundness_mistake)

    @property
    def completeness_mistakes(self) -> int:
        return sum(1 for decision in self.decisions if decision.completeness_mistake)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_enabled": self.memory_enabled,
            "task_success_rate": self.task_success_rate,
            "soundness_mistakes": self.soundness_mistakes,
            "completeness_mistakes": self.completeness_mistakes,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def evaluate_query_time_memory_policy(
    cases: Iterable[QueryReplayCase],
    memory_index: VerifiedMemoryIndex,
    *,
    memory_enabled: bool = False,
) -> QueryMemoryEvaluation:
    """Evaluate the opt-in memory signal on replay cases.

    A soundness mistake is a memory hit on a negative-control case. A
    completeness mistake is a missing memory hit on a case that should benefit
    from verified memory. When memory is disabled, even known IDs do not hit.

    Spec: REQ-LEARN-1484-3, REQ-LEARN-1484-4, REQ-LEARN-1484-5.
    """

    enabled = bool(memory_enabled)
    decisions: list[QueryMemoryDecision] = []
    for case in cases:
        memory_hit = enabled and memory_index.contains(case.case_id)
        soundness_mistake = memory_hit and not case.expects_memory_signal
        completeness_mistake = case.expects_memory_signal and not memory_hit
        decisions.append(
            QueryMemoryDecision(
                case_id=case.case_id,
                memory_enabled=enabled,
                memory_hit=memory_hit,
                verifier_signal=MEMORY_SIGNAL if memory_hit else BASELINE_SIGNAL,
                task_success=not soundness_mistake and not completeness_mistake,
                soundness_mistake=soundness_mistake,
                completeness_mistake=completeness_mistake,
            )
        )
    return QueryMemoryEvaluation(memory_enabled=enabled, decisions=tuple(decisions))
