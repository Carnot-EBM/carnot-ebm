"""Live program-synthesis action-effect proposal filter for ARC exploration.

Spec refs: REQ-ARC-WMTE-4689,
SCENARIO-ARC-WMTE-4689-HELDOUT-REJECTION,
SCENARIO-ARC-WMTE-4689-PROPOSAL-PRUNING.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    ProgrammaticExpert,
    Transition,
    detect_cell,
    induce_programmatic_object_experts,
    to_logical,
)


JsonDict = dict[str, Any]


def _candidate_action(candidate: Any) -> int | None:
    if isinstance(candidate, Mapping):
        value = candidate.get("action", candidate.get("action_id", candidate.get("kind")))
    else:
        value = getattr(candidate, "action", getattr(candidate, "action_id", None))
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _candidate_data(candidate: Any) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get("data")
    return getattr(candidate, "data", None)


def _candidate_row(candidate: Any) -> JsonDict:
    if isinstance(candidate, Mapping):
        action = _candidate_action(candidate)
        return {"action": int(action or 0), "data": candidate.get("data")}
    action = _candidate_action(candidate)
    return {"action": int(action or 0), "data": getattr(candidate, "data", None)}


def _frame_grid(frame_or_grid: Any, *, cell: int) -> np.ndarray | None:
    try:
        if isinstance(frame_or_grid, np.ndarray):
            grid = np.asarray(frame_or_grid)
        else:
            from carnot.agentic.arc_agi3_world_model import grid_of

            grid = np.asarray(grid_of(frame_or_grid))
        logical_cell = int(cell) if int(cell) > 0 else detect_cell(grid)
        return to_logical(grid, logical_cell)
    except Exception:
        return None


@dataclass
class ProgramSynthesisFilterResult:
    """REQ-ARC-WMTE-4689: held-out-validated programs plus rejection counts."""

    proposal_filter: "ActionEffectProposalFilter"
    program_trust_weights: list[JsonDict]
    heldout_programs_kept: int
    heldout_programs_rejected: int
    proposer_used: bool = False
    llm_proposal_ok: bool = False
    residual: str = ""


@dataclass
class ActionEffectProposalFilter:
    """SCENARIO-ARC-WMTE-4689-PROPOSAL-PRUNING: prune to trusted visible effects."""

    game: str
    experts: Sequence[ProgrammaticExpert]
    program_trust_weights: Sequence[Mapping[str, Any]]
    heldout_programs_kept: int
    heldout_programs_rejected: int
    cell: int = 1
    fallback_all_on_empty: bool = True
    _candidate_sets_scored: int = 0
    _candidates_seen: int = 0
    _candidates_kept: int = 0
    _candidates_pruned: int = 0
    _fallback_no_match: int = 0
    _prediction_errors: int = 0
    _last_kept_actions: list[int] = field(default_factory=list)

    def _predicts_effect(self, grid: np.ndarray, candidate: Mapping[str, Any]) -> bool:
        action = _candidate_action(candidate)
        if action is None:
            return False
        data = _candidate_data(candidate)
        for expert in self.experts:
            if not expert.applies(grid, action, data):
                continue
            try:
                predicted = expert.predict(grid, action, data)
            except Exception:
                self._prediction_errors += 1
                continue
            if predicted.shape == grid.shape and not np.array_equal(predicted, grid):
                return True
        return False

    def filter_candidates(
        self,
        frame_or_grid: Any,
        candidates: Sequence[Mapping[str, Any] | Any],
    ) -> list[JsonDict]:
        rows = [_candidate_row(candidate) for candidate in candidates]
        if not rows or not self.experts:
            return rows
        grid = _frame_grid(frame_or_grid, cell=int(self.cell))
        if grid is None:
            return rows

        self._candidate_sets_scored += 1
        self._candidates_seen += len(rows)
        kept = [row for row in rows if self._predicts_effect(grid, row)]
        self._last_kept_actions = [int(row["action"]) for row in kept]
        if kept:
            self._candidates_kept += len(kept)
            self._candidates_pruned += len(rows) - len(kept)
            return kept
        self._fallback_no_match += 1
        if self.fallback_all_on_empty:
            return rows
        self._candidates_pruned += len(rows)
        return []

    def rank_candidates(
        self,
        frame_or_grid: Any,
        candidates: Sequence[Mapping[str, Any] | Any],
    ) -> list[JsonDict]:
        return self.filter_candidates(frame_or_grid, candidates)

    def diagnostics(self) -> JsonDict:
        return {
            "enabled": bool(self.experts),
            "game": str(self.game),
            "heldout_programs_kept": int(self.heldout_programs_kept),
            "heldout_programs_rejected": int(self.heldout_programs_rejected),
            "candidate_sets_scored": int(self._candidate_sets_scored),
            "candidates_seen": int(self._candidates_seen),
            "candidates_kept": int(self._candidates_kept),
            "candidates_pruned": int(self._candidates_pruned),
            "fallback_no_match": int(self._fallback_no_match),
            "prediction_errors": int(self._prediction_errors),
            "last_kept_actions": list(self._last_kept_actions),
            "verifier_is_oracle": False,
        }


def induce_action_effect_proposal_filter(
    *,
    game: str,
    transitions: Sequence[Transition],
    proposer: Any = None,
    cell: int = 1,
    trust_threshold: float = 0.75,
    heldout_fraction: float = 0.34,
    max_programs: int = 8,
) -> ProgramSynthesisFilterResult:
    """SCENARIO-ARC-WMTE-4689-HELDOUT-REJECTION: reject held-out failures."""

    induction = induce_programmatic_object_experts(
        game=str(game),
        transitions=list(transitions),
        proposer=proposer,
        cell=int(cell),
        trust_threshold=float(trust_threshold),
        heldout_fraction=float(heldout_fraction),
        max_experts=int(max_programs),
    )
    weights = [dict(row) for row in induction.expert_trust_weights]
    kept = sum(1 for row in weights if bool(row.get("kept")))
    rejected = sum(1 for row in weights if not bool(row.get("kept")))
    proposal_filter = ActionEffectProposalFilter(
        game=str(game),
        experts=list(induction.experts),
        program_trust_weights=weights,
        heldout_programs_kept=kept,
        heldout_programs_rejected=rejected,
        cell=int(cell),
    )
    residual = induction.residual
    if not residual:
        residual = "none" if kept else "heldout_transitions_too_sparse"
    elif residual == "experts_overfit_prefix":
        residual = "heldout_transitions_too_sparse"
    return ProgramSynthesisFilterResult(
        proposal_filter=proposal_filter,
        program_trust_weights=weights,
        heldout_programs_kept=kept,
        heldout_programs_rejected=rejected,
        proposer_used=bool(induction.proposer_used),
        llm_proposal_ok=bool(induction.llm_proposal_ok),
        residual=residual,
    )


def coerce_program_synthesis_filter(value: Any) -> ActionEffectProposalFilter | None:
    if isinstance(value, ActionEffectProposalFilter):
        return value
    return None
