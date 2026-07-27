"""E3 — Carnot Executable-World-Model solver for ARC-AGI-3 (induce -> VERIFY ->
refactor -> plan), after arXiv:2605.05138 "Executable World Models for ARC-AGI-3 in
the Era of Coding Agents" (GPT-5.5 fully solves 15/25 games).

The paper's own thesis IS Carnot's: "LLMs are most reliable when used not as final
authorities, but as PROPOSAL mechanisms inside systems that can check their outputs."
So the LLM (codex / gpt-5.5) PROPOSES an executable Python world model; CARNOT'S
VERIFIER grounds it by checking the model reproduces the game's real offline
transitions, and HALTS planning the instant the model's prediction diverges from the
environment. The verifier is the moat; the LLM is the (swappable) proposer.

Loop:
  1. collect_transitions(game)      -- gather (grid, action, data, next_grid) offline (zero quota)
  2. proposer.induce(...)           -- codex writes results/arc_e3/<game>/world_model.py
                                       with engine(grid, action, data)->grid + is_level_complete(grid)
  3. WorldModelVerifier.score(...)  -- % of transitions the engine reproduces exactly;
                                       returns the failing transitions as mismatch artifacts
  4. proposer.refactor(...)         -- feed the mismatches back; codex fixes/simplifies (MDL proxy)
  5. plan_and_execute(...)          -- plan to is_level_complete INSIDE the verified model,
                                       execute in the real env, halt on any predicted!=observed divergence

This module is representation-careful: ARC-AGI-3 frames are 64x64 pixel renders of a
coarser LOGICAL grid. We detect the logical cell size and run the whole pipeline at
LOGICAL resolution (the paper's "settled ASCII frame"), so the induced model reasons
about game cells, not pixels.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
# THE INDUCED-ENGINE STORE. `load_engine` READS from here and `LocalGGUFProposer._gen_to_file`
# / the refactor path WRITE here, unconditionally, on every successful induction.
#
# REQ-ARC-WMTE-6016 (2026-07-27): `CARNOT_ARC_E3_DIR` makes it redirectable WITHOUT a
# monkeypatch. An A/B that runs several arms against the same store is not running an A/B:
# arm A's inductions overwrite the engines arm B then starts from, so a cross-arm delta
# confounds the flag with starting-engine drift. That is not hypothetical -- a four-arm run
# on 2026-07-27 had rewritten 11 engines from its FIRST arm before it was stopped, including
# both engines named as the origin incident for GAP-WM-TRUST-GATE, and 15 of 75 rows of an
# already-published artifact stopped reproducing as a result (ft09's legacy_accuracy went
# 1.0000 -> 0.0000 because its 12-bare-`return grid` engine had been replaced in place).
#
# Module-global by design so the existing `e3.E3_DIR = ...` monkeypatch (see
# scripts/experiments/arc3_local_scaffold_induction_ab.py and
# tests/python/test_induce_split_fallback.py) keeps working: both readers and writers resolve
# it at CALL time, not at import time. The env var is the same override for a subprocess that
# cannot patch the module.
E3_DIR = (
    Path(os.environ["CARNOT_ARC_E3_DIR"])
    if os.environ.get("CARNOT_ARC_E3_DIR")
    else (REPO / "results" / "arc_e3")
)
# Pristine, READ-ONLY copies of the engines as they stood at the commit that named them the
# GAP-WM-TRUST-GATE origin incident. The mutable store above is rewritten by any induction
# run, so a test that asserts "the new gate rejects the real degenerate engines" must read
# from HERE or it silently becomes a test of whatever ran most recently.
E3_ORIGIN_FIXTURES_DIR = REPO / "results" / "arc_e3_origin_fixtures"

# ---------------------------------------------------------------------------
# logical-resolution helpers (the "settled ASCII frame")
# ---------------------------------------------------------------------------


def detect_cell(grid: np.ndarray) -> int:
    """Largest c in {8,4,2,1} (divisor of 64) for which the grid is EXACTLY constant
    within every c x c block -> the logical cell size. Lossless downsample factor."""
    if grid.ndim != 2:
        # A degenerate/malformed frame (e.g. a post-terminal empty sentinel,
        # shape (0,) -- the g50t apply_g50t_label failure class) has no
        # meaningful logical cell size. 1 (no downsampling) is the same safe
        # fallback already used below when no clean divisor is found; this
        # avoids `h, w = grid.shape` raising ValueError on every one of
        # next_move's several call sites (multiple were found unguarded
        # during the 2026-07-12 exp5587 cascade check).
        return 1
    h, w = grid.shape
    for c in (8, 4, 2):
        if h % c or w % c:
            continue
        blocks = grid.reshape(h // c, c, w // c, c)
        # constant within each block iff min == max over the (c,c) axes
        if np.array_equal(blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))):
            return c
    return 1


def to_logical(grid: np.ndarray, cell: int) -> np.ndarray:
    if grid.ndim != 2:
        # Can't logically downsample a malformed grid -- return it unchanged
        # (same degenerate-input contract detect_cell now honors above)
        # rather than raising and killing the caller's remaining action
        # budget on the live scored path.
        return grid
    h, w = grid.shape
    return grid[::cell, ::cell] if cell > 1 else grid


def to_ascii(logical: np.ndarray) -> str:
    """Compact one-char-per-cell ASCII (single trailing digit of the color)."""
    return "\n".join("".join(str(int(v))[-1] for v in row) for row in logical)


def _rle_grid(g: np.ndarray) -> str:
    """Lossless run-length encoding of a FULL grid for the induce prompt: one line per row,
    'r<row>:<v0>x<n0>,<v1>x<n1>,...' -- each row's runs cover ALL columns left-to-right with NO
    gaps, so the column position is implicit (the running sum of prior run counts in that row),
    never spelled out. On large boards (e.g. lp85's 64x64 logical grid), `to_ascii`'s
    one-char-per-cell render was the dominant fixed cost of `induce_prompt` -- a SINGLE full grid
    measured ~6-7K tokens, so an 8-transition window (up to two full-grid renders + per-transition
    deltas) measured 18,355 tokens against a 13,824-token available budget and overflowed with
    `exceed_context_size_error` (ops/known-issues.md task 11, exp5593). An earlier attempt at this
    fix spelled out an explicit `r<row>c<col>:<value>x<count>` per run (matching `_rle_delta`'s
    style) -- measured on lp85's REAL grids, that per-run column overhead made the encoding barely
    smaller than `to_ascii` for medium-length runs and up to 24% LARGER for a grid with many short
    runs (`_rle_delta` pays that overhead once per DIFF, a rare event; a FULL grid pays it once per
    RUN, hundreds of times). Dropping the explicit column (implicit from the row's own running
    count) removed that dominant per-run overhead."""
    g = np.asarray(g)
    h, w = g.shape
    lines = []
    for r in range(h):
        c = 0
        runs = []
        while c < w:
            v = g[r, c]
            c0 = c
            while c < w and g[r, c] == v:
                c += 1
            runs.append(f"{int(v)}x{c - c0}")
        lines.append(f"r{r}:" + ",".join(runs))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# transition collection (zero quota — offline sim)
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    grid: np.ndarray  # logical grid BEFORE
    action: int
    data: Optional[dict]
    next_grid: np.ndarray  # logical grid AFTER
    level_before: int
    level_after: int


@dataclass
class ProgrammaticExpert:
    """REQ-ARC-WMTE-4677: one small object-level precondition/effect factor."""

    name: str
    object_class: str
    precondition: Callable[[np.ndarray, int, Any], bool]
    effect: Callable[[np.ndarray, int, Any], np.ndarray]
    action: int | None = None
    trust: float = 0.0
    heldout_correct: int = 0
    heldout_total: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def applies(self, grid: np.ndarray, action: int, data: Any = None) -> bool:
        if self.action is not None and int(action) != int(self.action):
            return False
        try:
            return bool(self.precondition(np.asarray(grid), int(action), data))
        except Exception:
            return False

    def predict(self, grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        return np.asarray(self.effect(np.asarray(grid).copy(), int(action), data))

    def summary(self, *, kept: bool) -> dict[str, Any]:
        return {
            "name": self.name,
            "object_class": self.object_class,
            "trust": round(float(self.trust), 6),
            "heldout_correct": int(self.heldout_correct),
            "heldout_total": int(self.heldout_total),
            "kept": bool(kept),
        }


@dataclass
class ProgrammaticExpertInductionResult:
    """REQ-ARC-WMTE-4677: trusted factors plus the rejected-factor ledger."""

    experts: list[ProgrammaticExpert]
    expert_trust_weights: list[dict[str, Any]]
    proposer_used: bool = False
    llm_proposal_ok: bool = False
    residual: str = ""


@dataclass
class FactoredSubgoalPlanResult:
    """SCENARIO-ARC-WMTE-4677-PRODUCT-PLANNING: product-model plan diagnostics."""

    planned: bool
    plan: list[dict[str, Any]] = field(default_factory=list)
    subgoal_decomposition: list[dict[str, Any]] = field(default_factory=list)
    per_subgoal_reachable: list[dict[str, Any]] = field(default_factory=list)
    expert_trust_weights: list[dict[str, Any]] = field(default_factory=list)
    final_grid: np.ndarray | None = None
    residual: str = ""


def _color_rewrite_expert(
    *,
    name: str,
    object_class: str,
    action: int | None,
    from_color: int,
    to_color: int,
    metadata: Mapping[str, Any] | None = None,
) -> ProgrammaticExpert:
    src = int(from_color)
    dst = int(to_color)
    action_id = None if action is None else int(action)

    def _precondition(grid: np.ndarray, candidate_action: int, _data: Any) -> bool:
        return (action_id is None or int(candidate_action) == action_id) and bool(
            np.any(np.asarray(grid) == src)
        )

    def _effect(grid: np.ndarray, _candidate_action: int, _data: Any) -> np.ndarray:
        out = np.asarray(grid).copy()
        out[out == src] = dst
        return out

    return ProgrammaticExpert(
        name=name,
        object_class=object_class,
        precondition=_precondition,
        effect=_effect,
        action=action_id,
        metadata={
            "kind": "color_rewrite",
            "from_color": src,
            "to_color": dst,
            **dict(metadata or {}),
        },
    )


def _exact_delta_expert(transition: Transition, index: int) -> ProgrammaticExpert:
    base = np.asarray(transition.grid).copy()
    target = np.asarray(transition.next_grid).copy()
    action_id = int(transition.action)
    changed = np.argwhere(base != target)
    signature = [(int(r), int(c), int(base[r, c]), int(target[r, c])) for r, c in changed]

    def _precondition(grid: np.ndarray, candidate_action: int, _data: Any) -> bool:
        candidate = np.asarray(grid)
        return (
            int(candidate_action) == action_id
            and candidate.shape == base.shape
            and all(int(candidate[r, c]) == before for r, c, before, _after in signature)
        )

    def _effect(grid: np.ndarray, _candidate_action: int, _data: Any) -> np.ndarray:
        out = np.asarray(grid).copy()
        for r, c, _before, after in signature:
            out[r, c] = after
        return out

    colors = sorted({int(after) for _r, _c, _before, after in signature})
    return ProgrammaticExpert(
        name=f"exact_delta_action_{action_id}_{index}",
        object_class="cells_" + "_".join(str(color) for color in colors[:4]),
        precondition=_precondition,
        effect=_effect,
        action=action_id,
        metadata={"kind": "exact_delta", "changed_cells": len(signature)},
    )


def _normalise_programmatic_experts(rows: Sequence[Any]) -> list[ProgrammaticExpert]:
    experts: list[ProgrammaticExpert] = []
    for index, row in enumerate(rows):
        if isinstance(row, ProgrammaticExpert):
            experts.append(row)
            continue
        if not isinstance(row, Mapping):
            continue
        precondition = row.get("precondition")
        effect = row.get("effect")
        if callable(precondition) and callable(effect):
            experts.append(
                ProgrammaticExpert(
                    name=str(row.get("name") or f"expert_{index}"),
                    object_class=str(row.get("object_class") or row.get("object") or "object"),
                    precondition=precondition,
                    effect=effect,
                    action=(None if row.get("action") is None else int(row["action"])),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
            continue
        if row.get("kind") == "color_rewrite" or {
            "from_color",
            "to_color",
        }.issubset(row.keys()):
            experts.append(
                _color_rewrite_expert(
                    name=str(row.get("name") or f"color_rewrite_{index}"),
                    object_class=str(row.get("object_class") or f"color_{row.get('from_color')}"),
                    action=(None if row.get("action") is None else int(row["action"])),
                    from_color=int(row["from_color"]),
                    to_color=int(row["to_color"]),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
    return experts


def _stratified_prefix_heldout(
    transitions: Sequence[Transition],
    heldout_fraction: float,
) -> tuple[list[Transition], list[Transition]]:
    rows = list(transitions)
    if len(rows) < 2:
        return rows, rows
    n_suffix = max(1, int(round(len(rows) * max(0.0, min(1.0, heldout_fraction)))))
    heldout_indices = set(range(max(0, len(rows) - n_suffix), len(rows)))
    by_action: dict[int, list[int]] = {}
    for i, transition in enumerate(rows):
        by_action.setdefault(int(transition.action), []).append(i)
    for indices in by_action.values():
        if len(indices) > 1:
            heldout_indices.add(indices[-1])
    prefix = [row for i, row in enumerate(rows) if i not in heldout_indices]
    heldout = [row for i, row in enumerate(rows) if i in heldout_indices]
    return (prefix or rows[:1], heldout or rows[-1:])


def _fallback_experts_from_transitions(
    transitions: Sequence[Transition],
) -> list[ProgrammaticExpert]:
    experts: list[ProgrammaticExpert] = []
    seen: set[tuple[Any, ...]] = set()
    for index, transition in enumerate(transitions):
        before = np.asarray(transition.grid)
        after = np.asarray(transition.next_grid)
        if before.shape != after.shape or np.array_equal(before, after):
            continue
        changed = before != after
        from_values = sorted({int(v) for v in before[changed].flatten().tolist()})
        to_values = sorted({int(v) for v in after[changed].flatten().tolist()})
        if len(from_values) == 1 and len(to_values) == 1:
            key = ("color", int(transition.action), from_values[0], to_values[0])
            if key in seen:
                continue
            seen.add(key)
            experts.append(
                _color_rewrite_expert(
                    name=f"color_{from_values[0]}_to_{to_values[0]}_action_{int(transition.action)}",
                    object_class=f"color_{from_values[0]}",
                    action=int(transition.action),
                    from_color=from_values[0],
                    to_color=to_values[0],
                    metadata={"source": "transition_color_delta"},
                )
            )
        else:
            key = ("exact", int(transition.action), to_ascii(before))
            if key in seen:
                continue
            seen.add(key)
            experts.append(_exact_delta_expert(transition, index))
    return experts


def _score_expert_on_transitions(
    expert: ProgrammaticExpert,
    transitions: Sequence[Transition],
) -> ProgrammaticExpert:
    total = 0
    correct = 0
    for transition in transitions:
        if not expert.applies(transition.grid, int(transition.action), transition.data):
            continue
        total += 1
        try:
            pred = expert.predict(transition.grid, int(transition.action), transition.data)
        except Exception:
            continue
        if pred.shape == np.asarray(transition.next_grid).shape and np.array_equal(
            pred,
            np.asarray(transition.next_grid),
        ):
            correct += 1
    expert.heldout_total = int(total)
    expert.heldout_correct = int(correct)
    expert.trust = float(correct) / float(total) if total else 0.0
    return expert


def induce_programmatic_object_experts(
    *,
    game: str,
    transitions: Sequence[Transition],
    proposer: Any = None,
    cell: int = 1,
    trust_threshold: float = 0.75,
    heldout_fraction: float = 0.34,
    max_experts: int = 8,
) -> ProgrammaticExpertInductionResult:
    """REQ-ARC-WMTE-4677: induce factors, weight by held-out trust, keep stable ones."""

    rows = list(transitions)
    if not rows:
        return ProgrammaticExpertInductionResult(
            experts=[],
            expert_trust_weights=[],
            residual="experts_overfit_prefix",
        )
    prefix, heldout = _stratified_prefix_heldout(rows, heldout_fraction)
    proposed_rows: list[Any] = []
    proposer_used = False
    llm_ok = False
    provider = getattr(proposer, "induce_programmatic_experts", None)
    if callable(provider):
        proposer_used = True
        try:
            proposed_rows = list(
                provider(
                    game=game,
                    transitions=list(prefix),
                    heldout_transitions=list(heldout),
                    cell=int(cell),
                    max_experts=int(max_experts),
                )
                or []
            )
            llm_ok = bool(proposed_rows)
        except TypeError:
            try:
                proposed_rows = list(provider(game, list(prefix)) or [])
                llm_ok = bool(proposed_rows)
            except Exception:
                proposed_rows = []
        except Exception:
            proposed_rows = []
    experts = _normalise_programmatic_experts(proposed_rows)
    if not experts:
        experts.extend(_fallback_experts_from_transitions(prefix))

    deduped: list[ProgrammaticExpert] = []
    seen: set[str] = set()
    for expert in experts:
        key = json.dumps(
            {
                "name": expert.name,
                "action": expert.action,
                "object_class": expert.object_class,
                "metadata": expert.metadata,
            },
            sort_keys=True,
            default=str,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(expert)
        if len(deduped) >= int(max_experts):
            break

    threshold = max(0.0, min(1.0, float(trust_threshold)))
    kept: list[ProgrammaticExpert] = []
    weights: list[dict[str, Any]] = []
    for expert in deduped:
        scored = _score_expert_on_transitions(expert, heldout)
        is_kept = scored.heldout_total > 0 and scored.trust >= threshold
        if is_kept:
            kept.append(scored)
        weights.append(scored.summary(kept=is_kept))

    residual = (
        "" if kept else ("experts_overfit_prefix" if deduped else "expert_factors_not_independent")
    )
    return ProgrammaticExpertInductionResult(
        experts=kept,
        expert_trust_weights=weights,
        proposer_used=proposer_used,
        llm_proposal_ok=llm_ok,
        residual=residual,
    )


class ProductWorldModel:
    """REQ-ARC-WMTE-4677: executable product composition of trusted factors."""

    def __init__(self, experts: Sequence[ProgrammaticExpert]) -> None:
        self.experts = list(experts)

    def engine(self, grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        start = np.asarray(grid)
        out = start.copy()
        trust = np.full(start.shape, -1.0, dtype=float)
        for expert in self.experts:
            if not expert.applies(start, int(action), data):
                continue
            pred = expert.predict(start, int(action), data)
            if pred.shape != start.shape:
                continue
            changed = pred != start
            stronger = changed & (float(expert.trust) >= trust)
            out[stronger] = pred[stronger]
            trust[stronger] = float(expert.trust)
        return out


def _normalise_factored_subgoals(rows: Sequence[Any]) -> list[dict[str, Any]]:
    subgoals: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            predicate = row.get("predicate") or row.get("is_level_complete")
            if callable(predicate):
                subgoals.append(
                    {
                        "name": str(row.get("name") or f"subgoal_{index}"),
                        "predicate": predicate,
                        "source": str(row.get("source") or "a1_goal_induction"),
                        "score": float(row.get("score") or 0.0),
                    }
                )
            continue
        predicate = getattr(row, "predicate", None)
        if callable(predicate):
            subgoals.append(
                {
                    "name": str(getattr(row, "name", f"subgoal_{index}")),
                    "predicate": predicate,
                    "source": str(getattr(row, "source", "a1_goal_induction")),
                    "score": float(getattr(row, "score", 0.0) or 0.0),
                }
            )
    return subgoals


def _apply_factored_plan(
    engine: Callable[[np.ndarray, int, Any], np.ndarray],
    start_grid: np.ndarray,
    plan: Sequence[Mapping[str, Any]] | None,
) -> np.ndarray:
    grid = np.asarray(start_grid)
    for step in list(plan or []):
        grid = np.asarray(engine(grid.copy(), int(step["action"]), step.get("data")))
    return grid


def plan_factored_subgoal_sequence(
    *,
    start_grid: np.ndarray,
    final_goal: Callable[[np.ndarray], bool],
    experts: Sequence[ProgrammaticExpert],
    subgoals: Sequence[Any] = (),
    value_head: Callable[[np.ndarray], float] | None = None,
    max_subgoals: int = 3,
    max_nodes: int = 20000,
    max_depth: int = 40,
) -> FactoredSubgoalPlanResult:
    """SCENARIO-ARC-WMTE-4677-PRODUCT-PLANNING: plan through the product model."""

    product = ProductWorldModel(experts)
    current = np.asarray(start_grid)
    full_plan: list[dict[str, Any]] = []
    decomposition: list[dict[str, Any]] = []
    reachable_rows: list[dict[str, Any]] = []
    weights = [expert.summary(kept=True) for expert in experts]

    def _leg(goal: Callable[[np.ndarray], bool], grid: np.ndarray) -> list[dict[str, Any]] | None:
        try:
            if bool(goal(np.asarray(grid))):
                return []
        except Exception:
            return None
        return plan_in_model(
            product.engine,
            goal,
            np.asarray(grid),
            max_nodes=max_nodes,
            max_depth=max_depth,
            goal_energy=value_head,
        )

    ordered = sorted(
        _normalise_factored_subgoals(subgoals),
        key=lambda row: (float(row.get("score") or 0.0), str(row.get("name") or "")),
        reverse=True,
    )[: max(0, int(max_subgoals))]
    for subgoal in ordered:
        leg = _leg(subgoal["predicate"], current)
        reached = leg is not None
        row = {
            "name": subgoal["name"],
            "source": subgoal["source"],
            "reachable": bool(reached),
            "plan_length": len(leg or []),
            "score": round(float(subgoal.get("score") or 0.0), 6),
        }
        decomposition.append(dict(row))
        reachable_rows.append(dict(row))
        if not reached:
            return FactoredSubgoalPlanResult(
                planned=False,
                plan=full_plan,
                subgoal_decomposition=decomposition,
                per_subgoal_reachable=reachable_rows,
                expert_trust_weights=weights,
                final_grid=current,
                residual="product_model_plans_live_invalid",
            )
        full_plan.extend(dict(step) for step in leg)
        current = _apply_factored_plan(product.engine, current, leg)

    final_leg = _leg(final_goal, current)
    final_reached = final_leg is not None
    final_row = {
        "name": "final_goal",
        "source": "terminal_goal_predicate",
        "reachable": bool(final_reached),
        "plan_length": len(final_leg or []),
        "score": 1.0,
    }
    decomposition.append(dict(final_row))
    reachable_rows.append(dict(final_row))
    if not final_reached:
        return FactoredSubgoalPlanResult(
            planned=False,
            plan=full_plan,
            subgoal_decomposition=decomposition,
            per_subgoal_reachable=reachable_rows,
            expert_trust_weights=weights,
            final_grid=current,
            residual="product_model_plans_live_invalid",
        )
    full_plan.extend(dict(step) for step in final_leg)
    final_grid = _apply_factored_plan(product.engine, current, final_leg)
    return FactoredSubgoalPlanResult(
        planned=True,
        plan=full_plan,
        subgoal_decomposition=decomposition,
        per_subgoal_reachable=reachable_rows,
        expert_trust_weights=weights,
        final_grid=final_grid,
        residual="none",
    )


def collect_transitions(
    game: str, n: int = 120, warmup: bool = False, seed: int = 0
) -> tuple[list[Transition], int]:
    """Explore the offline sim and record logical-resolution transitions. Uses the
    salience-ordered candidate generator so the dataset covers meaningful actions, not
    just raster order. Returns (transitions, cell_size)."""
    import random
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_over, _game_action
    from carnot.agentic.arc_graph_explore import rich_action_candidates, _warm
    from carnot.agentic import arc_solver_kit as kit

    rng = random.Random(seed)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, warmup)
    cell = detect_cell(grid_of(f))
    trans: list[Transition] = []
    restarts = 0
    while len(trans) < n and restarts < n:
        cands = rich_action_candidates(f)
        if not cands:
            f = _warm(env, warmup)
            restarts += 1
            continue
        c = cands[rng.randrange(min(len(cands), 8))]  # bias to salient, keep some variety
        g0 = to_logical(grid_of(f), cell)
        l0 = _levels_completed(f)
        nf = env.step(_game_action(GameAction, c.action_id), data=c.data)
        if nf is None:
            f = _warm(env, warmup)
            restarts += 1
            continue
        g1 = to_logical(grid_of(nf), cell)
        l1 = _levels_completed(nf)
        trans.append(Transition(g0, int(c.action_id), c.data, g1, l0, l1))
        if _game_over(nf) or l1 > l0:
            f = _warm(env, warmup)
            restarts += 1
        else:
            f = nf
    return trans, cell


# ---------------------------------------------------------------------------
# THE CARNOT VERIFIER — grounds the LLM-induced model against reality
# ---------------------------------------------------------------------------

# ===========================================================================
# REQ-ARC-WMTE-6010 / REQ-ARC-WMTE-6011 -- the two INDEPENDENT, DEFAULT-OFF
# repairs to world-model verification, measured today (2026-07-27) as the two
# reasons `induction_attempts_planned == 0` on 174/174 rows of the first-win
# measurement while the generator itself was healthy (103 calls / 94 responses
# / 0 errors).
#
# THEY PUSH IN OPPOSITE DIRECTIONS, WHICH IS WHY THEY ARE TWO FLAGS AND NOT ONE.
#   * Masking the HUD REMOVES cells that were unattainable by construction, and
#     should therefore RAISE every measured fidelity number.
#   * Closing the trust gate REJECTS degenerate engines that today pass, and
#     should therefore LOWER the pass rate.
# Shipped together behind a single flag and measured together, a null result is
# uninterpretable -- "both worked and cancelled" and "neither did" produce the
# same number. Two independent flags make the four-arm matrix (control /
# mask-only / gate-only / both) possible, which is the only design that can
# attribute the effect. Neither flag is flipped here; the flip is the operator's
# call on the strength of that matrix.
# ===========================================================================

# ---- REQ-ARC-WMTE-6010: the HUD is inside the exact-match comparison --------
# Transitions are recorded as FULL logical grids (arc_competition_agent.py's
# E3AgentPolicy.next_move, via `to_logical(grid_of(latest), self.cell)`). A HUD
# mask EXISTS and is live in the explorer (`_compute_hud_mask_from_frame`, the
# SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED / _COLLAPSE_GUARD / _STAGE2_CONFIRM trio),
# but as of 2026-07-27 `grep hud_mask` returned ZERO hits in this module, in
# arc_llm_reinduction.py, and in arc_world_model_trust_energy.py -- verified
# directly. So on any game with a monotone step counter EVERY frame differs in
# the HUD, full-grid exact match is UNATTAINABLE BY CONSTRUCTION, and
#
# QUALIFICATION (measured 2026-07-27, second review): "a monotone step counter"
# describes a MINORITY of the games this mask is applied to. Measuring
# P(masked region changes | play area unchanged) over real transitions
# (n=60, seed 0) across the 17 games with a resolvable mask:
#     free-running counter (p >= 0.95):  6  -- bp35 lf52 s5i5 sp80 tu93 vc33
#     mixed (0.05 < p < 0.95):           7  -- ka59 cd82 dc22 su15 wa30 m0r0 g50t
#     game-COUPLED chrome (p <= 0.05):   2  -- ar25 ft09
#     unmeasurable (no still frames):    2  -- re86 tr87
# A free-running tick would sit at p = 1.0 everywhere. On the coupled games the
# masked row moves only when the game state also moves -- it is a score or
# progress readout, not an action-independent clock -- so the
# "unattainable by construction" argument does NOT cover them, and masking
# there is discarding real signal rather than removing noise. This does not
# make the mask dishonest; it means the justification is narrower than the
# application, which is why REQ-ARC-WMTE-6015's swallow guard exists.
#
# `cell_recall`'s change mask is dominated by counter cells rather than by game
# state. Part of the measured median-0.0 trust score is therefore a MEASUREMENT
# ARTIFACT, not purely a capability wall.
#
# The repair masks at COMPARE time, never at RECORD time: `Transition` keeps the
# full-fidelity grids it always kept (never-prune; a historical transitions dump
# stays byte-comparable), and only the comparison collapses HUD cells. The mask
# is supplied in LOGICAL coordinates -- see `logical_hud_mask`, which downsamples
# the explorer's FRAME-coordinate mask by the same `grid[::cell, ::cell]` stride
# `to_logical` uses, so the two are aligned by construction rather than by luck.
SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED = False
SUBMITTED_WORLD_MODEL_HUD_MASK_MODE = "compare_time_logical_mask_from_explorer_frame_mask"

# ---- REQ-ARC-WMTE-6011: GAP-WM-TRUST-GATE (ops/verifier_gaps.md) -----------
# An IDENTITY engine (`return grid`) scores 0.725 on `WorldModelVerifier.score`
# and PASSES the `accuracy >= 0.5` gate -- 0.5 being the threshold recorded in
# ops/verifier_gaps.md's GAP-WM-TRUST-GATE entry and used by
# `binary_exact_gate_pass`'s default. THRESHOLD CORRIGENDUM (2026-07-27): the
# threshold the AGENT ACTUALLY SHIPS is `min_heldout_accuracy=1.0`
# (arc_competition_agent.py:5593 and :5719). The two are not interchangeable --
# admission-flip counts differ by ~10x between them (see
# `change_gate_decision`'s legacy_accuracy_* keys) -- so every claim below is
# tagged with the threshold it was measured at, and any statement about LIVE
# behaviour uses 1.0. The 0.725-passes-0.5 framing is the DOCUMENTED gap, which
# is what this requirement was written against; it remains true at 0.5, and at
# 1.0 the identity engine is rejected by the incumbent gate for a different
# reason (it is not exact) while STILL being ranked and trusted through the
# hidden-state `trust_pass` branch, which is what REQ-6013 addresses.
# `accuracy` is full-grid exact
# match DENOMINATED OVER ALL TRANSITIONS INCLUDING NO-OPS and lp85's corpus is
# ~87 no-ops to 33 changing. Confirmed on disk 2026-07-27:
# results/arc_e3/ft09/world_model.py is 12 bare `return grid` branches and
# `is_level_complete -> False`; results/arc_e3/lp85/world_model.py mutates only
# on `action == 6 and grid[py, px] == 9`. Both were reported to the operator as
# "the good model" and are in fact degenerate.
#
# WHY NOT `cell_recall`, AND WHY NOT THE EXISTING `score_change_weighted_
# consistency`: both mask to TRUE changes only
# (`pred[changed] == next_grid[changed]`), so neither can see a cell the engine
# wrote that reality did NOT change. They punish misses and ignore spurious
# writes -- they are recall, not fidelity. A "write garbage everywhere" engine
# is invisible to them in exactly the way the identity engine is invisible to
# `accuracy`. So the gate quantity here is SYMMETRIC by construction: it scores
# over the UNION of the truly-changed cells and the engine-written cells, which
# makes a spurious write cost exactly what a missed change costs.
SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED = False
SUBMITTED_WORLD_MODEL_CHANGE_GATE_MODE = "symmetric_union_change_fidelity_plus_nondegeneracy_floor"
# Calibrated from the measured separation between the real on-disk degenerates
# and a hand-written correct engine -- see the REQ-ARC-WMTE-6011 witness in
# tests/python/test_arc_world_model_change_gate.py, which asserts the separation
# rather than the constants, so a re-tune cannot silently empty the pass region.
WORLD_MODEL_CHANGE_FIDELITY_THRESHOLD = 0.5
WORLD_MODEL_MIN_CORRECT_CHANGED_CELLS = 1
# Ceiling on the fraction of TRULY-UNCHANGED transitions the engine changes anyway. 0.25 is
# calibrated from the measured separation, not picked round: the honest hand-written dc22
# navigation engine hallucinates on 0.0000 of its no-ops, while the adversarial
# "correct-but-invents-a-change-on-every-no-op" engine that defeated the fidelity-only gate
# sits at 1.0000. Anything in between is a real judgement call, so the threshold is set well
# clear of the honest engine rather than tight against the attack.
WORLD_MODEL_MAX_NOOP_HALLUCINATION_RATE = 0.25

# ---- REQ-ARC-WMTE-6013: the change gate's HIDDEN-STATE branch coverage hole ----
# REQ-6011 above shipped `change_gate_decision` wired into exactly ONE of the agent's two
# admission branches -- the `else` (non-hidden-state) one. The OTHER branch, taken for the
# 11 HIDDEN_STATE_GAME_IDS, admits on `trust_pass` from `select_trusted_world_model` and
# never calls the change gate. That branch covers EVERY one of the 0.08-wall games
# (cn04/ar25/sc25/sk48/wa30), so the gate had zero coverage on precisely the games the
# whole programme exists to move.
#
# THE HOLE, MEASURED (results/experiment_6012_hidden_state_trust_gate_hole.json, 33 matched
# rows = 11 games x 3 seeds): an engine that is correct on every real change AND ALSO writes
# cells reality never wrote is ADMITTED by the live hidden-state gate on 31/33 rows -- the
# SAME 31 rows on which it admits the honest engine. Both attack arms score EXACTLY (not
# approximately) the honest engine's `consistency`. The cause is structural and is the same
# one REQ-6011 names for `cell_recall`: `score_change_weighted_consistency` masks to TRUE
# changes only (`pred[changed] == next[changed]`), so a cell the engine invented outside that
# mask is arithmetically invisible to it. It is recall, not fidelity.
#
# The repair routes the hidden-state branch's ADMIT/REJECT decision through the SAME
# symmetric union-fidelity `change_gate_decision` the plain branch uses -- REPLACING
# `trust_pass`, exactly as the plain branch replaces its `accuracy >= 0.5`, not AND-ing with
# it. AND-ing was considered and rejected on measurement: exp6012 found the live gate is not
# merely blind, it is ALSO too strict in the other direction -- it rejects the hand-written
# honest dc22 engine on 2/3 seeds where REQ-6011 admits it 3/3. Keeping `trust_pass` as a
# conjunct would import that false-reject wholesale, so the arm would confound "the symmetric
# metric helps" with "the old metric still vetoes".
#
# The CANDIDATE RANKING is deliberately NOT touched: `select_trusted_world_model` still picks
# by trust energy. Only the final admit/reject changes. Ranking and admission are separable
# concerns, and changing both at once would make a per-arm delta unattributable.
#
# WHY A THIRD FLAG RATHER THAN FOLDING INTO REQ-6011'S. The two branches gate DISJOINT game
# sets, so per-game attribution is automatic and a shared flag would not confound. But the
# branches replace DIFFERENT incumbent metrics (`trust_pass` from held-out recall here vs
# `accuracy` there) with different measured risk profiles, so an experimenter must be able
# to isolate them. The default is therefore "follow REQ-6011's flag" -- which keeps the
# four-arm matrix at four arms and stops the gate arm from being a silent no-op on the 11
# wall games -- with an explicit env override that separates them when a measurement needs it.
# None means follow; True/False pin it.
SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED: Optional[bool] = None


def _flag_env(name: str, default: bool) -> bool:
    """Read a per-arm override, falling back to the shipped SUBMITTED_* default.

    The four-arm matrix (control / mask-only / gate-only / both) needs to select an arm
    WITHOUT editing the shipped constants, because an arm selected by editing a constant
    cannot be run concurrently with its own control and cannot be reproduced from a command
    line in an artifact. Unset env -> the shipped default, so the submitted path is
    unaffected by the existence of this knob.
    """

    import os

    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def world_model_hud_mask_enabled() -> bool:
    """REQ-ARC-WMTE-6010 arm selector. CARNOT_ARC_WM_HUD_MASK=1 turns compare-time masking on."""

    return _flag_env("CARNOT_ARC_WM_HUD_MASK", SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED)


def world_model_change_gate_enabled() -> bool:
    """REQ-ARC-WMTE-6011 arm selector. CARNOT_ARC_WM_CHANGE_GATE=1 turns the change gate on."""

    return _flag_env("CARNOT_ARC_WM_CHANGE_GATE", SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED)


def world_model_change_gate_hidden_state_enabled() -> bool:
    """REQ-ARC-WMTE-6013 arm selector for the hidden-state branch.

    Resolution order, most specific first:
      1. CARNOT_ARC_WM_CHANGE_GATE_HIDDEN_STATE -- the explicit isolation override.
      2. SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED, when pinned to True/False.
      3. Otherwise FOLLOW REQ-6011's flag, so turning the change gate on covers BOTH
         admission branches instead of silently skipping the 11 hidden-state games.

    Step 3 is what keeps the four-arm matrix at four arms. Step 1 is what lets a follow-up
    measurement separate the two branches without a code edit.
    """

    import os

    raw = os.environ.get("CARNOT_ARC_WM_CHANGE_GATE_HIDDEN_STATE")
    if raw is not None and raw != "":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED is not None:
        return bool(SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED)
    return world_model_change_gate_enabled()


def logical_hud_mask(frame_mask: Any, cell: int) -> Optional[np.ndarray]:
    """Downsample a FRAME-coordinate HUD mask to LOGICAL-grid coordinates.

    `to_logical` is a plain stride `grid[::cell, ::cell]`, so the logical grid's
    (r, c) is the frame's (r*cell, c*cell) and the mask must be taken with the
    IDENTICAL stride. Doing anything cleverer (e.g. "any HUD pixel in the block")
    would silently mask cells `to_logical` never sampled, which is the
    over-masking direction -- and over-masking destroys CORRECTNESS while
    under-masking only costs efficiency, the same asymmetry that keeps the
    explorer's own edge-bar detector behind a Stage-2 confirmation gate.

    Returns None when there is nothing to align (no mask, malformed mask, bad
    cell size). None is NEVER a silent no-op at the call sites: every caller
    records an explicit `hud_mask_status` saying WHY no mask was applied.
    """

    if frame_mask is None:
        return None
    try:
        mask = np.asarray(frame_mask, dtype=bool)
    except Exception:
        return None
    if mask.ndim != 2:
        return None
    c = int(cell) if cell else 1
    if c < 1:
        return None
    out = mask[::c, ::c] if c > 1 else mask
    return out if bool(out.any()) else None


# ---- REQ-ARC-WMTE-6015: THE SWALLOW GUARD ---------------------------------
# Found 2026-07-27 by measuring mask coverage per game rather than trusting the "it is only
# a monotone step counter" story. On two of the 17 games with a resolvable mask, the
# explorer's HUD classifier selects a row where the GAME STATE lives, not chrome:
#
#     game   changed-cells-inside-mask   changing transitions, raw -> masked
#     lf52          1.0000                       60 -> 0     (the entire game is deleted)
#     su15          0.7568                       28 -> 1
#     ...  every other game               0.0000 .. 0.2219   (s5i5 0.2219 is the highest)
#
# On lf52 the mask makes the corpus DYNAMICS-FREE: nothing changes, so the IDENTITY engine
# is a perfect model of a game with no mechanics, and it is admitted. That is exactly the
# laundering an adversarial review flagged in the aggregate ("the mask helps a zero-knowledge
# engine ~1.7x more than a real one"), here located in a specific mechanism on specific games
# rather than left as a statistical worry.
#
# `apply_hud_mask`'s own docstring already names the asymmetry that decides the fix:
# "over-masking destroys CORRECTNESS while under-masking only costs efficiency". So the guard
# is REFUSE-ON-DOUBT: a mask that swallows the dynamics is not applied at all, and the reason
# is recorded. Refusing degrades that game to the pre-REQ-6010 behaviour (measurably worse,
# per the very artifact this repair is built on) -- which is the correct direction, because
# the alternative is a metric that scores an engine on a game it has deleted.
#
# THRESHOLD PROVENANCE. 0.5 is not picked round; it is the midpoint of the only wide gap in
# the measured distribution above (0.2219 -> 0.7568), so it is well clear of the worst honest
# game AND of the least-bad swallowing one. The zero-dynamics case is checked SEPARATELY and
# unconditionally, because "the corpus has changes and the mask leaves none" is a swallow at
# any threshold and must not depend on a tunable.
HUD_MASK_MAX_CHANGED_CELL_OVERLAP = 0.5


def hud_mask_swallow_check(transitions: Sequence["Transition"], mask: Optional[np.ndarray]) -> dict:
    """Does this mask delete the game rather than the chrome? Returns an auditable record.

    A dict, not a bool, for the same reason `change_gate_decision` returns one: a caller must
    be able to show WHY a mask was refused, and a reviewer must be able to see the measured
    quantity next to the threshold that judged it. `swallows=False` with
    `n_changed_cells_total == 0` is reported as `no_dynamics_to_swallow` -- an unmeasurable
    verdict, NOT a clean one, so a consumer cannot read "we checked and it is fine" off a
    corpus where the check could not fire.
    """

    rec = {
        "checked": False,
        "swallows": False,
        "reason": "no_mask",
        "changed_cell_overlap": 0.0,
        "overlap_threshold": float(HUD_MASK_MAX_CHANGED_CELL_OVERLAP),
        "raw_changing_transitions": 0,
        "masked_changing_transitions": 0,
        "n_changed_cells_total": 0,
        "n_changed_cells_inside_mask": 0,
    }
    if mask is None:
        return rec
    rows = list(transitions)
    if not rows:
        rec["reason"] = "no_transitions"
        return rec
    m = np.asarray(mask, dtype=bool)
    total = inside = raw_changing = masked_changing = 0
    for t in rows:
        g0 = np.asarray(t.grid)
        g1 = np.asarray(t.next_grid)
        if g0.shape != g1.shape or g0.shape != m.shape:
            continue
        ch = g0 != g1
        if not ch.any():
            continue
        raw_changing += 1
        total += int(ch.sum())
        inside += int((ch & m).sum())
        if not np.array_equal(apply_hud_mask(g0, m), apply_hud_mask(g1, m)):
            masked_changing += 1
    rec.update(
        {
            "checked": True,
            "raw_changing_transitions": raw_changing,
            "masked_changing_transitions": masked_changing,
            "n_changed_cells_total": total,
            "n_changed_cells_inside_mask": inside,
            "changed_cell_overlap": round(float(inside / total), 6) if total else 0.0,
        }
    )
    if total == 0:
        rec["reason"] = "no_dynamics_to_swallow"
        return rec
    if raw_changing > 0 and masked_changing == 0:
        # The corpus has changes and the mask leaves none. TWO different situations produce
        # this, and they are NOT distinguishable from inside this corpus:
        #   (a) the mask really does cover the game (lf52: overlap 1.0, 60 -> 0);
        #   (b) the mask is honest chrome and this corpus genuinely contains no state
        #       change, so the only cells that moved were the counter's.
        # Both are refused -- `apply_hud_mask`'s stated asymmetry (over-masking destroys
        # correctness, under-masking only costs efficiency) makes refusing the safe
        # direction -- but they are given DIFFERENT reasons, because (b) is a statement
        # about the corpus and (a) is a defect in the mask, and an operator reading a
        # refusal needs to know which one they are looking at. Collapsing them would be the
        # same clean-vs-unmeasurable conflation `noop_ok_is_vacuous` exists to prevent.
        rec["swallows"] = True
        rec["reason"] = (
            "mask_removes_all_dynamics"
            if inside < total
            else "no_changed_cells_outside_mask_cannot_distinguish"
        )
        return rec
    if rec["changed_cell_overlap"] >= float(HUD_MASK_MAX_CHANGED_CELL_OVERLAP):
        rec["swallows"] = True
        rec["reason"] = "mask_overlaps_majority_of_changed_cells"
        return rec
    rec["reason"] = "ok"
    return rec


def apply_hud_mask(grid: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Collapse HUD cells to a constant so they cannot decide an exact-match test.

    Mirrors `StepwiseExplorer._hash`, which already zeroes `hud_mask` cells before
    hashing a frame for node identity -- same collapse convention, same constant,
    so a state that dedups to one node in the search also compares as one state in
    the world model. Shape-mismatched masks are IGNORED here and reported by the
    caller as `shape_mismatch`; silently applying a wrong-shaped mask (or letting
    numpy broadcast one) is precisely the failure this repair is fixing.
    """

    if mask is None:
        return grid
    g = np.asarray(grid)
    if getattr(mask, "shape", None) != g.shape:
        return grid
    out = g.copy()
    out[mask] = 0
    return out


@dataclass
class VerifyResult:
    n: int
    n_correct: int
    accuracy: float
    mismatches: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    # GRADED companion to `accuracy` (which is exact-FULL-GRID match): mean changed-cell recall over the
    # state-CHANGING transitions. Exact-match reads ~0 for an imperfect (e.g. LLM-induced or learned) world
    # model that is still ~55% changed-cell-accurate, so it gates EVERY such model out of execution-grounded
    # planning -- the single root cause of the 0.08 wall (docs/research-notes/arc-008-wall-root-cause-2026-06-21.md).
    # cell_recall is the granularity-matched gate the coordinated redesign turns on via CARNOT_ARC_TRUST_METRIC.
    cell_recall: float = 0.0

    # ---- REQ-ARC-WMTE-6011 change-weighted fields (GAP-WM-TRUST-GATE) -------
    # Number of recorded transitions that actually changed the (masked) grid. This is
    # the denominator `accuracy` should have had: on lp85 it is 33 of 120, so an
    # engine can be wrong about every single mechanic and still read 0.725.
    n_changing: int = 0
    # Transitions among `n_changing` the engine reproduced EXACTLY. The gap file's
    # literal `n_changes_correct`; reported, and the strictest available witness.
    n_changes_correct: int = 0
    # `n_changes_correct / n_changing` -- the gap file's literal `change_accuracy`.
    # Reported for continuity with the gap entry; NOT the gate quantity, because
    # exact-full-grid match over changing transitions is the same all-or-nothing
    # measure that REQ-6010 shows is unattainable while the HUD is in the compare.
    change_accuracy: float = 0.0
    # THE GATE QUANTITY. Per changing transition, the fraction of the UNION of
    # (cells reality changed) and (cells the engine wrote) that the engine got
    # right, averaged over changing transitions. Symmetric: a miss and a spurious
    # write cost the same. Identity engines score 0.0 here by construction (they
    # write nothing, so the union is exactly the true changes and none are right).
    change_fidelity: float = 0.0
    # Non-degeneracy floor, in CELLS not transitions: how many truly-changed cells
    # the engine predicted correctly. Cell-denominated on purpose -- a
    # transition-denominated floor would be unreachable on a HUD game whenever
    # REQ-6010's mask is off, which would re-introduce exactly the cross-flag
    # coupling the two-flag split exists to prevent.
    correct_changed_cells: int = 0
    # The ASYMMETRY WITNESS: cells the engine wrote that reality did not change to
    # that value. `cell_recall` and `score_change_weighted_consistency` are both
    # structurally blind to this number. Non-zero here with a high `cell_recall` is
    # the "writes garbage everywhere but happens to cover the real changes" engine.
    spurious_changed_cells: int = 0

    # ---- THE NO-OP HALLUCINATION CHANNEL (found by adversarially attacking this gate) ----
    # `change_fidelity` scores GRID-CHANGING transitions only, which leaves it structurally
    # blind to an engine that models every real change correctly AND ALSO invents a change on
    # every NO-OP. Measured on real dc22 transitions: such an engine scores change_fidelity
    # 0.7243 and PASSES, while its full-grid exact accuracy is 0.0000 -- it is wrong about
    # every single transition in the corpus. That engine is catastrophic for `plan_in_model`,
    # which walks the engine forward and would see phantom transitions at every step. The
    # LEGACY accuracy gate caught it (0.0 < 0.5), so without this channel the repair would be
    # strictly WORSE than what it replaces on this failure mode.
    #
    # It is a SEPARATE gate condition rather than being folded into `change_fidelity`: adding
    # no-ops into the same average would give a correctly-idle identity engine credit for
    # every no-op it "predicts", which reproduces exactly the 0.725 blind spot this whole
    # requirement exists to remove.
    n_noop: int = 0
    n_noop_hallucinated: int = 0  # truly-unchanged transitions the engine changed anyway
    noop_hallucination_rate: float = 0.0
    # ---- REQ-ARC-WMTE-6013 DIAGNOSTICS (recorded, deliberately NOT gated on) ----
    # `noop_hallucination_rate` above returns 0.0 when `n_noop == 0`, so the value meaning
    # "this engine invents nothing" is ALSO the value meaning "this could not be measured".
    # That is a structurally dead channel wearing a passing score, and it is not
    # hypothetical: on re86 all 40 held-out transitions change, n_noop is 0, and an engine
    # that writes a cell reality never wrote clears the whole gate at fidelity 0.919 because
    # the one channel that would have caught it cannot fire. This flag separates the two
    # meanings so a consumer can tell "clean" from "unmeasurable".
    noop_channel_measurable: bool = False
    # The PURE invented-write count: cells the engine changed that reality did NOT change at
    # all. Distinct from `spurious_changed_cells`, which is `wrote & ~correct` and therefore
    # CONFLATES two different things -- a cell invented out of nothing, and a genuinely-
    # changed cell predicted with the wrong value (ordinary prediction error, which every
    # imperfect-but-useful engine has). Only this quantity isolates invention.
    #
    # IT IS NOT A GATE CONDITION, ON PURPOSE. It separates perfectly on the corpus measured
    # so far (honest engines 0, the spurious writer one per changing transition), and that is
    # exactly why it must not be thresholded here: a separation measured against an engine
    # built to be caught tells you nothing about where a REALISTICALLY IMPERFECT engine sits,
    # and a threshold fitted to the former would reject the latter. Recalibration against an
    # imperfect engine is follow-up work with the operator, not a side effect of this change.
    invented_changed_cells: int = 0
    invented_change_rate: float = 0.0  # invented_changed_cells / n_changing
    # Explicit provenance for REQ-6010 -- one of "disabled" (flag off, no mask
    # requested), "unresolved" (flag on, caller had no mask to give), "shape_mismatch"
    # (flag on, mask given, but it did not align with the graded grids), or "applied".
    # NEVER a silent no-op: a caller that asked for masking and did not get it can
    # tell which of the three failure reasons it hit.
    hud_mask_status: str = "disabled"
    # Number of logical cells the applied mask covers. 0 whenever status != "applied".
    hud_mask_cells: int = 0
    # REQ-ARC-WMTE-6015 swallow-guard record. Carried on every result, including the
    # unmasked ones (where it reports `no_mask`), so an artifact row always shows whether
    # the guard could have fired and what it measured -- never only when it did fire.
    hud_mask_swallow: dict = field(default_factory=dict)


class WorldModelVerifier:
    """Checks that an induced engine(grid, action, data) -> grid reproduces the real
    recorded transitions. This is the verification that makes the LLM accountable: a
    proposed model only earns trust by predicting transitions it was NOT hand-fit to.
    Returns mismatch artifacts (the failing transitions) for the refactor step."""

    def __init__(
        self,
        transitions: list[Transition],
        *,
        hud_mask: Any = None,
        hud_mask_enabled: Optional[bool] = None,
        hud_mask_swallow: Optional[dict] = None,
    ) -> None:
        """`hud_mask` is in LOGICAL-grid coordinates (see `logical_hud_mask`).

        `hud_mask_enabled` defaults to the module flag SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED
        so the submitted path is byte-identical until the operator flips it; an explicit
        True/False is the per-arm override the four-arm A/B needs.
        """

        self.transitions = transitions
        self.hud_mask_enabled = (
            world_model_hud_mask_enabled() if hud_mask_enabled is None else bool(hud_mask_enabled)
        )
        self.hud_mask = hud_mask if self.hud_mask_enabled else None
        # REQ-ARC-WMTE-6015: refuse a mask that deletes the game instead of the chrome.
        # Run BEFORE the status is settled so a swallowing mask can never reach `_graded`.
        #
        # `hud_mask_swallow` may be PRE-COMPUTED by the caller, and callers that hold the
        # whole corpus SHOULD pre-compute it. Whether a mask covers the game or the chrome
        # is a property of the MASK AND THE WHOLE CORPUS, not of whatever slice this
        # verifier happens to hold. Judging it per-slice produces a real false positive:
        # `select_trusted_world_model` grades on a held-out TAIL, and a tail that happens to
        # contain no genuine state change has ALL of its changed cells inside the HUD -- an
        # honest mask then looks exactly like a swallowing one, and the guard disables the
        # repair on precisely the no-op-heavy corpora the repair exists for (lp85 is ~87
        # no-ops to 33 changing). The full corpus can tell the two apart; a tail cannot.
        self.hud_mask_swallow = (
            dict(hud_mask_swallow)
            if hud_mask_swallow is not None
            else hud_mask_swallow_check(self.transitions, self.hud_mask)
        )
        # Resolved once, here, so `score()` cannot drift between the status it reports and
        # the grids it actually compared.
        if not self.hud_mask_enabled:
            self.hud_mask_status = "disabled"
        elif self.hud_mask is None:
            # The flag asked for masking and the caller had none to give. This is the
            # explicit record the repair promises instead of a silent no-op.
            self.hud_mask_status = "unresolved"
        elif self.hud_mask_swallow.get("swallows"):
            # THE SWALLOW GUARD FIRING. Drop the mask entirely and say so. Degrading this
            # game to unmasked grading is the deliberate choice: an unmasked comparison is
            # merely hard to win, while a swallowed one is scoring engines on a game whose
            # dynamics have been deleted -- under which the IDENTITY engine is optimal.
            self.hud_mask = None
            self.hud_mask_status = "refused_swallows_dynamics"
        else:
            self.hud_mask_status = "requested"

    def _graded(self, grid: np.ndarray) -> np.ndarray:
        return apply_hud_mask(grid, self.hud_mask)

    def score(
        self, engine: Callable[[np.ndarray, int, Optional[dict]], np.ndarray], max_mismatch: int = 8
    ) -> VerifyResult:
        n_correct, mism = 0, []
        cell_recalls: list[
            float
        ] = []  # per-CHANGED-transition fraction of changed cells predicted right
        # REQ-ARC-WMTE-6011 accumulators (see VerifyResult for what each one is for).
        n_changing = 0
        n_changes_correct = 0
        fidelities: list[float] = []
        correct_changed_cells = 0
        spurious_changed_cells = 0
        invented_changed_cells = 0
        n_noop = 0
        n_noop_hallucinated = 0
        # REQ-ARC-WMTE-6010: resolve the mask's status from the TRANSITIONS ALONE, before the
        # engine runs. Deriving it inside the loop (the first version of this code) made the
        # status depend on ENGINE behaviour: an engine that raised on every transition, or an
        # empty corpus, would `continue` past the alignment check and report `unresolved` even
        # though a perfectly good mask had been supplied. The mask either fits these grids or
        # it does not, and that is a fact about the mask and the corpus, not about the engine.
        mask_status = self.hud_mask_status
        if mask_status == "requested":
            shapes = {np.asarray(t.grid).shape for t in self.transitions}
            if not shapes:
                mask_status = "no_transitions"
            elif any(getattr(self.hud_mask, "shape", None) == s for s in shapes):
                mask_status = "applied"
            else:
                mask_status = "shape_mismatch"
        for i, t in enumerate(self.transitions):
            try:
                pred = np.asarray(engine(t.grid.copy(), t.action, t.data))
            except Exception as e:  # a crashing engine fails the transition
                if len(mism) < max_mismatch:
                    mism.append({"i": i, "action": t.action, "error": repr(e)[:160]})
                continue
            # REQ-ARC-WMTE-6010: grade on HUD-collapsed copies. The recorded Transition is
            # left untouched (never-prune: the raw grids stay exactly as observed).
            g0 = self._graded(t.grid)
            g1 = self._graded(t.next_grid)
            pred_g = self._graded(pred) if pred.shape == np.asarray(t.next_grid).shape else pred
            # graded changed-cell recall (granularity-matched gate); only state-changing transitions count
            changed = not np.array_equal(g0, g1)
            if changed:
                n_changing += 1
                if pred_g.shape == g1.shape:
                    m = g0 != g1
                    cell_recalls.append(float((pred_g[m] == g1[m]).mean()))
                    # ---- symmetric union fidelity (THE GATE QUANTITY) ----------------
                    # `m` is what reality changed; `wrote` is what the engine changed. Scoring
                    # over their UNION is what makes a spurious write cost what a miss costs;
                    # `cell_recall` above scores over `m` alone and therefore cannot see one.
                    wrote = pred_g != g0
                    union = m | wrote
                    correct = pred_g == g1
                    n_union = int(union.sum())
                    fidelities.append(float((correct & union).sum() / n_union) if n_union else 1.0)
                    correct_changed_cells += int((correct & m).sum())
                    spurious_changed_cells += int((wrote & ~correct).sum())
                    # REQ-ARC-WMTE-6013: `wrote & ~m` -- the engine changed a cell that
                    # reality left alone. `~m` (reality did not change it) rather than
                    # `~correct` (the prediction was wrong) is what makes this invention
                    # rather than error.
                    invented_changed_cells += int((wrote & ~m).sum())
                else:
                    cell_recalls.append(0.0)
                    fidelities.append(0.0)
            else:
                # A TRUE no-op. The engine should leave it alone; if it did not, it invented
                # a transition that reality does not contain. Counted separately from
                # `change_fidelity` -- see VerifyResult's NO-OP HALLUCINATION CHANNEL note.
                n_noop += 1
                if pred_g.shape != g1.shape or not np.array_equal(pred_g, g1):
                    n_noop_hallucinated += 1
            if pred_g.shape == g1.shape and np.array_equal(pred_g, g1):
                n_correct += 1
                if changed:
                    n_changes_correct += 1
            elif len(mism) < max_mismatch:
                ok_shape = pred_g.shape == g1.shape
                # COMPACT mismatch (deltas, not full grids — fits a local model's context):
                # what the TRUE action did vs where the engine's prediction was wrong.
                mism.append(
                    {
                        "i": i,
                        "action": t.action,
                        "data": t.data,
                        "true_change": _delta(g0, g1),
                        "your_prediction_was_wrong_at": (
                            _delta(pred_g, g1) if ok_shape else f"wrong shape {pred_g.shape}"
                        ),
                    }
                )
        n = len(self.transitions)
        cell_recall = float(np.mean(cell_recalls)) if cell_recalls else 0.0
        return VerifyResult(
            n,
            n_correct,
            n_correct / max(1, n),
            mism,
            cell_recall=cell_recall,
            n_changing=n_changing,
            n_changes_correct=n_changes_correct,
            change_accuracy=float(n_changes_correct / n_changing) if n_changing else 0.0,
            change_fidelity=float(np.mean(fidelities)) if fidelities else 0.0,
            correct_changed_cells=correct_changed_cells,
            spurious_changed_cells=spurious_changed_cells,
            n_noop=n_noop,
            n_noop_hallucinated=n_noop_hallucinated,
            noop_hallucination_rate=(float(n_noop_hallucinated / n_noop) if n_noop else 0.0),
            noop_channel_measurable=bool(n_noop > 0),
            invented_changed_cells=invented_changed_cells,
            invented_change_rate=(
                float(invented_changed_cells / n_changing) if n_changing else 0.0
            ),
            hud_mask_status=mask_status,
            hud_mask_cells=(
                int(np.asarray(self.hud_mask).sum()) if mask_status == "applied" else 0
            ),
            hud_mask_swallow=dict(self.hud_mask_swallow),
        )

    def offpath_structural_energy(
        self,
        engine: Callable[[np.ndarray, int, Optional[dict]], np.ndarray],
        *,
        energy_scorer: Any,
    ) -> float:
        """REQ-ARC-WMTE-4791: score candidate predictions without reading true next grids."""

        energies: list[float] = []
        for t in self.transitions:
            try:
                pred = np.asarray(engine(t.grid.copy(), t.action, t.data))
                if hasattr(energy_scorer, "transition_energy"):
                    value = energy_scorer.transition_energy(t.grid, t.action, t.data, pred)
                else:
                    value = energy_scorer(t.grid, t.action, t.data, pred)
                value_f = float(value)
                energies.append(value_f if value_f == value_f else float("inf"))
            except Exception:
                energies.append(float("inf"))
        if not energies:
            return float("inf")
        finite = [value for value in energies if value < float("inf")]
        if not finite:
            return float("inf")
        return float(np.mean(finite))

    def rank_offpath_structural_energy(
        self,
        candidates: Sequence[
            tuple[str, Callable[[np.ndarray, int, Optional[dict]], np.ndarray]]
            | dict[str, Any]
            | Any
        ],
        *,
        energy_scorer: Any,
    ) -> list[dict[str, Any]]:
        """REQ-ARC-WMTE-4791: rank candidate engines by lower off-path structural energy."""

        rows: list[dict[str, Any]] = []
        for i, candidate in enumerate(candidates):
            if isinstance(candidate, dict):
                name = str(candidate.get("name") or f"candidate_{i}")
                engine = candidate["engine"]
            elif isinstance(candidate, tuple):
                name = str(candidate[0])
                engine = candidate[1]
            else:
                name = str(getattr(candidate, "name", f"candidate_{i}"))
                engine = getattr(candidate, "engine", candidate)
            rows.append(
                {
                    "candidate_name": name,
                    "offpath_structural_energy": self.offpath_structural_energy(
                        engine,
                        energy_scorer=energy_scorer,
                    ),
                    "n_offpath_transitions": len(self.transitions),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                float(row["offpath_structural_energy"]),
                str(row["candidate_name"]),
            ),
        )


def change_gate_decision(
    vr: "VerifyResult",
    *,
    enabled: Optional[bool] = None,
    fidelity_threshold: float = WORLD_MODEL_CHANGE_FIDELITY_THRESHOLD,
    min_correct_changed_cells: int = WORLD_MODEL_MIN_CORRECT_CHANGED_CELLS,
    max_noop_hallucination_rate: float = WORLD_MODEL_MAX_NOOP_HALLUCINATION_RATE,
) -> dict:
    """REQ-ARC-WMTE-6011: the change-weighted trust decision, as an auditable record.

    Returns a dict rather than a bare bool ON PURPOSE. A bare bool cannot answer "could
    this gate have failed?" -- and a pass that could not have failed is not evidence. The
    returned record carries the COMPUTED WITNESS at the gate's own aggregation level: the
    two sub-decisions, the two measured quantities, the two thresholds, and the size of
    the population each quantity was computed over. `n_changing == 0` is reported as its
    own reason (`no_changing_transitions`) because a corpus with no state-changing
    transition cannot distinguish a good engine from the identity engine -- refusing there
    is the honest answer, not a pass by default.

    `enabled=False` (the shipped default via SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED)
    still computes and returns every field; it just reports `passed=True` with reason
    `gate_disabled`, so a control arm records the same diagnostics as a treatment arm and
    the four-arm matrix compares like with like.
    """

    on = world_model_change_gate_enabled() if enabled is None else bool(enabled)
    fidelity_ok = float(vr.change_fidelity) >= float(fidelity_threshold)
    nondegenerate = int(vr.correct_changed_cells) >= int(min_correct_changed_cells)
    has_population = int(vr.n_changing) > 0
    noop_ok = float(vr.noop_hallucination_rate) <= float(max_noop_hallucination_rate)
    if not on:
        reason = "gate_disabled"
        passed = True
    elif not has_population:
        reason = "no_changing_transitions"
        passed = False
    elif not nondegenerate:
        # THE ORIGIN INCIDENT. ft09's identity engine and lp85's near-identity engine both
        # land here: they never correctly predict a single changed cell, while `accuracy`
        # reads 0.725 / 1.0 because the corpus is no-op-heavy.
        #
        # HONEST NOTE ON REDUNDANCY AT THE DEFAULT k=1. At `min_correct_changed_cells == 1`
        # this branch cannot fire while `fidelity_ok` is True, and that is a THEOREM, not a
        # coincidence: if no truly-changed cell is predicted correctly, then every cell in
        # the (true-changes UNION engine-writes) set is wrong -- the true-changed ones by
        # assumption, and each engine-written-but-unchanged one because "correct" there
        # would require pred == next == prev, contradicting "the engine wrote it". So the
        # union score is exactly 0 and the fidelity test has already failed. Confirmed
        # empirically over 924 real arms in
        # results/experiment_6011_world_model_change_gate_four_arm.json: the combination
        # (nondegenerate=False, fidelity_ok=True) is never observed.
        #
        # It is kept, and ordered BEFORE the fidelity test, for two reasons. (1) It emits a
        # strictly more diagnostic reason: "this engine never got a single real change
        # right" is actionable where "fidelity 0.0 < 0.5" is not. (2) It becomes an
        # INDEPENDENT gate condition the moment `min_correct_changed_cells > 1`, which is
        # the knob for demanding a minimum evidence base rather than merely a non-zero one
        # -- see test_nondegeneracy_floor_is_redundant_at_k1_and_independent_above_it, which
        # asserts BOTH halves so this cannot quietly become a dead channel.
        reason = "degenerate_engine_no_correct_changed_cells"
        passed = False
    elif not fidelity_ok:
        reason = "change_fidelity_below_threshold"
        passed = False
    elif not noop_ok:
        # Found by attacking this gate rather than by testing it: an engine correct on every
        # real change that ALSO invents one on every no-op scores change_fidelity 0.7243 on
        # real dc22 transitions while being wrong about 100% of them (exact accuracy 0.0000).
        # `plan_in_model` walks the engine forward, so such an engine hallucinates a
        # transition at every step of every plan.
        reason = "engine_hallucinates_changes_on_noop_transitions"
        passed = False
    else:
        reason = "passed"
        passed = True
    return {
        "gate_enabled": on,
        "passed": bool(passed),
        "reason": reason,
        # --- computed witness, at the gate's own aggregation level ---------------
        "change_fidelity": round(float(vr.change_fidelity), 6),
        "fidelity_threshold": float(fidelity_threshold),
        "fidelity_ok": bool(fidelity_ok),
        "correct_changed_cells": int(vr.correct_changed_cells),
        "min_correct_changed_cells": int(min_correct_changed_cells),
        "nondegenerate": bool(nondegenerate),
        "spurious_changed_cells": int(vr.spurious_changed_cells),
        "noop_hallucination_rate": round(float(vr.noop_hallucination_rate), 6),
        "max_noop_hallucination_rate": float(max_noop_hallucination_rate),
        "noop_ok": bool(noop_ok),
        "n_noop": int(vr.n_noop),
        "n_noop_hallucinated": int(vr.n_noop_hallucinated),
        # REQ-ARC-WMTE-6013 diagnostics. Reported, NOT gated on -- see VerifyResult. When
        # `noop_channel_measurable` is False the `noop_ok` verdict above is vacuous (it
        # passed because there was nothing to test, not because the engine is clean), and a
        # consumer that treats those two cases alike will read a false pass.
        "noop_channel_measurable": bool(vr.noop_channel_measurable),
        "noop_ok_is_vacuous": bool(not vr.noop_channel_measurable),
        "invented_changed_cells": int(vr.invented_changed_cells),
        "invented_change_rate": round(float(vr.invented_change_rate), 6),
        "n_changing": int(vr.n_changing),
        "n_transitions": int(vr.n),
        # The legacy quantity this gate replaces, carried alongside so any artifact row
        # shows BOTH verdicts and the disagreement is visible without a re-run.
        "legacy_accuracy": round(float(vr.accuracy), 6),
        # ---- THRESHOLD AMBIGUITY, RESOLVED EXPLICITLY (2026-07-27 corrigendum) --------
        # `legacy_accuracy_would_pass` used to be reported ALONE against a hardcoded 0.5,
        # and 0.5 is NOT the threshold the agent ships. The live admission is
        # `min_heldout_accuracy=1.0` at BOTH call sites (arc_competition_agent.py:5593 and
        # :5719, verified on disk 2026-07-27). The gap is not cosmetic: recomputed over
        # exp6011's 75 rows, mask-induced admission flips for the IDENTITY engine are 29/75
        # at 0.5 but only 3/75 at 1.0, and for the real on-disk engines 12/75 at 0.5 but
        # 0/75 at 1.0 -- an order of magnitude, and a sign change in the headline. Reporting
        # one number against an unnamed threshold made every admission claim unfalsifiable,
        # so BOTH are now reported, each with its threshold named in the key.
        #
        # `legacy_accuracy_would_pass` is retained (not renamed) so already-written
        # consumers keep reading the quantity they read before; it is the DOCUMENTARY
        # threshold from ops/verifier_gaps.md's gap entry, not the live one. Any claim about
        # LIVE behaviour must use `..._at_live_threshold`.
        "legacy_accuracy_would_pass": bool(float(vr.accuracy) >= 0.5),
        "legacy_accuracy_threshold_documented": 0.5,
        "legacy_accuracy_would_pass_at_live_threshold": bool(float(vr.accuracy) >= 1.0),
        "legacy_accuracy_live_threshold": 1.0,
        "legacy_accuracy_live_threshold_source": (
            "arc_competition_agent.py:5593,5719 min_heldout_accuracy=1.0"
        ),
        "change_accuracy": round(float(vr.change_accuracy), 6),
        "n_changes_correct": int(vr.n_changes_correct),
        "cell_recall": round(float(vr.cell_recall), 6),
        "hud_mask_status": str(vr.hud_mask_status),
        "hud_mask_cells": int(vr.hud_mask_cells),
        # REQ-ARC-WMTE-6015. Present on EVERY gate record, fired or not, so a reader can
        # tell "the guard checked and passed" from "the guard never ran" -- the same
        # measurable-vs-clean distinction `noop_ok_is_vacuous` draws above.
        "hud_mask_swallow": dict(vr.hud_mask_swallow),
        "hud_mask_swallow_guard_fired": bool(
            str(vr.hud_mask_status) == "refused_swallows_dynamics"
        ),
    }


@dataclass
class GoalPredicateConsistency:
    """REQ-ARC-WMTE-5593: the goal-hypothesis analog of `VerifyResult` -- checks whether
    `is_level_complete` correctly predicts the SIGN of real observed level transitions
    (a real level-up occurred, or it did not), rather than the DYNAMICS `WorldModelVerifier`
    already checks (does `engine()` predict the right next grid). Nothing in the induction
    pipeline validated the goal predicate against real level-progress ground truth before
    this -- `execute_bounded_llm_reinduction` installs `outcome.goal_predicate` as a search
    termination condition on the strength of the proposer's own code, unchecked against any
    observed transition. This is a direct, literal instance of the project's founding thesis
    (verify a claim against ground truth) applied to the goal-hypothesis half of an induced
    world model, mirroring the docs/research-notes/arc-agi3-milestone1-winners-sota-ingestion-
    2026-07-11.md finding that two independent top-3 teams (Reki, Duck) both carry an
    unexploited self-report-vs-ground-truth gap in their own pipelines.
    """

    n: int
    n_correct: int
    accuracy: float
    n_real_levelups: int
    n_real_noops: int
    mismatches: list[dict] = field(default_factory=list)


def score_goal_predicate_consistency(
    is_level_complete: Callable[[np.ndarray], bool],
    transitions: Sequence[Transition],
    *,
    max_mismatch: int = 8,
) -> GoalPredicateConsistency:
    """REQ-ARC-WMTE-5593: does `is_level_complete`'s sign match real observed level-ups?

    For each transition, the real ground truth is `level_after > level_before` (a genuine
    level-up occurred at that point). The claim under test is
    `is_level_complete(next_grid)`. Agreement is a cheap, deterministic sign check -- no
    second LLM call, matching forge's own competitive-pressure finding that an expensive
    LLM judge was not worth the cost while a deterministic filter was kept.

    CALLER CONTRACT: pass transitions from a SINGLE level boundary (the level
    `is_level_complete` was induced/re-induced for). It is a per-boundary predicate in the
    real pipeline (`execute_bounded_llm_reinduction` re-induces it after every level-up), so
    checking it against transitions spanning multiple boundaries can produce a spurious
    mismatch if a "win"-looking state persists visually into a later, unrelated boundary.
    """

    n_correct = 0
    n_real_levelups = 0
    n_real_noops = 0
    mismatches: list[dict] = []
    for i, t in enumerate(transitions):
        real_levelup = bool(t.level_after > t.level_before)
        if real_levelup:
            n_real_levelups += 1
        else:
            n_real_noops += 1
        try:
            claimed = bool(is_level_complete(t.next_grid))
        except Exception as e:
            claimed = False
            if len(mismatches) < max_mismatch:
                mismatches.append(
                    {"i": i, "real_levelup": real_levelup, "claimed": None, "error": repr(e)[:160]}
                )
            continue
        if claimed == real_levelup:
            n_correct += 1
        elif len(mismatches) < max_mismatch:
            mismatches.append({"i": i, "real_levelup": real_levelup, "claimed": claimed})

    n = len(transitions)
    return GoalPredicateConsistency(
        n=n,
        n_correct=n_correct,
        accuracy=float(n_correct / max(1, n)),
        n_real_levelups=n_real_levelups,
        n_real_noops=n_real_noops,
        mismatches=mismatches,
    )


def predict_hypothesis_transition(
    hypothesis: Any,
    grid: np.ndarray,
    action: int,
    data: Any = None,
) -> np.ndarray:
    """REQ-ARC-WMTE-4727: run one hypothesis' transition model for a candidate probe.

    Active probing needs a narrow, oracle-distinct prediction API: given a
    candidate dynamics hypothesis and a possible live action, return what that
    hypothesis says the next logical grid will be. This helper deliberately
    does not inspect `is_level_complete`; probe routing is about transition
    consequences, not asking the environment's win oracle.
    """

    engine = getattr(hypothesis, "engine", hypothesis)
    if not callable(engine):
        raise TypeError("hypothesis_transition_engine_not_callable")
    return np.asarray(engine(np.asarray(grid).copy(), int(action), data))


def load_engine(game: str):
    """Import the codex-written world_model.py for a game and return (engine,
    is_level_complete). Re-imports fresh each call so a refactor is picked up."""
    import importlib.util

    return _load_engine_from(E3_DIR, game)


def _load_engine_from(root: Path, game: str):
    import importlib.util

    p = Path(root) / game / "world_model.py"
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(f"arc_wm_{game}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, "engine"), getattr(mod, "is_level_complete", None)


def load_origin_fixture_engine(game: str):
    """Load a GAP-WM-TRUST-GATE origin-incident engine from the FROZEN, never-written copy.

    REQ-ARC-WMTE-6016. `load_engine` reads the MUTABLE store, which any induction run
    rewrites in place. A guard asserted against the mutable store is therefore not asserted
    against its own origin incident for long: on 2026-07-27 a live A/B replaced ft09's
    12-bare-`return grid` identity engine with a 2-branch mutating one within hours of the
    artifact that cited it, and the same run rewrote lp85 -- the ONE game whose degenerate
    engine actually discriminates -- twice.

    A guard that stops firing on the incident that motivated it is the failure mode this
    project has shipped more than once. Reading from `E3_ORIGIN_FIXTURES_DIR` makes the
    origin-incident assertion permanent by construction rather than by luck.
    """

    return _load_engine_from(E3_ORIGIN_FIXTURES_DIR, game)


# ---------------------------------------------------------------------------
# the proposer — codex / gpt-5.5 writes the executable model (swappable)
# ---------------------------------------------------------------------------

CODEX_BIN = "codex"


def _codex(prompt: str, timeout: int = 420) -> tuple[bool, str]:
    """Invoke codex/gpt-5.5 (the orchestrate.py form) with the prompt on stdin. The
    agent edits files in the repo directly. Returns (ok, tail_of_output)."""
    cmd = [
        CODEX_BIN,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--color",
        "never",
        "--cd",
        str(REPO),
        "--model",
        "gpt-5.5",
        "-",
    ]
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout or "")[-2000:] + (p.stderr or "")[-500:]
    except subprocess.TimeoutExpired:
        return False, f"codex timeout after {timeout}s"


def _delta(g0: np.ndarray, g1: np.ndarray, cap: int = 80) -> list:
    """Changed cells (row, col, from, to) — a COMPACT transition encoding that fits a
    local model's small context (full 64x64 before/after grids blow it; a few-cell delta
    is tiny and is arguably MORE learnable: it shows exactly what the action changed)."""
    g0 = np.asarray(g0)
    g1 = np.asarray(g1)
    if g0.shape != g1.shape:
        return []
    diff = np.argwhere(g0 != g1)
    return [(int(r), int(c), int(g0[r, c]), int(g1[r, c])) for r, c in diff[:cap]]


def _rle_delta(g0: np.ndarray, g1: np.ndarray) -> str:
    """LOSSLESS run-length delta for the induce prompt: every changed cell, encoded as maximal
    horizontal runs 'r<row>c<col0>:<new,values>' (values comma-separated so colors >=10 stay
    unambiguous). This REPLACES the old cap=80 raw-tuple delta in the induction evidence, which
    silently TRUNCATED large changes — a 293-cell re-render showed only the first 80 cells (27%),
    starving the model of the very evidence it needs to induce the rule. RLE shows the FULL change
    at ~1/4 the tokens of raw per-cell tuples, so the whole change fits the local model's context
    with no truncation. (The verifier's mismatch examples still use the capped _delta on purpose —
    those are illustrative, not the load-bearing induction evidence.)"""
    g0 = np.asarray(g0)
    g1 = np.asarray(g1)
    if g0.shape != g1.shape:
        return ""
    diff = g0 != g1
    h, w = g0.shape
    runs = []
    for r in range(h):
        c = 0
        while c < w:
            if diff[r, c]:
                c0 = c
                while c < w and diff[r, c]:
                    c += 1
                vals = ",".join(str(int(v)) for v in g1[r, c0:c])
                runs.append(f"r{r}c{c0}:{vals}")
            else:
                c += 1
    return " ".join(runs) if runs else "(no change)"


def _rle_delta_compact(g0: np.ndarray, g1: np.ndarray) -> str:
    """Like `_rle_delta`, but each changed run's NEW values are themselves run-length-collapsed
    as '<value>x<count>' pairs instead of listed one-per-cell. `_rle_grid` fixed
    `induce_prompt`'s full-grid cost, but on lp85's real transitions the per-transition DELTAS
    then became the dominant remaining cost (measured: 8 deltas via `_rle_delta` = 9,308 tokens,
    still over the 13,824-token budget after the full-grid fix) -- large changes are often a
    single-color object moving or a solid region appearing, which `_rle_delta`'s raw
    comma-per-cell listing cannot exploit but this can (measured: same 8 deltas =
    5,992 tokens, a 3,316-token additional saving, closing the remaining gap). Kept as a
    SEPARATE function from `_rle_delta` rather than changing its output format in place --
    `_rle_delta` has its own round-trip tests (test_rle_delta_lossless.py) and another caller
    (scripts/experiments/arc_frontier_tooluse_probe.py) that expect the existing one-value-per-
    comma format; this function is used only by `_transitions_block`'s induction-evidence path.
    The run's starting column stays explicit (unlike `_rle_grid`'s implicit-column full-row
    encoding): a row can have multiple disjoint CHANGED spans separated by unchanged cells, so
    the column position is not implicit here the way it is when every cell in a row is covered."""
    g0 = np.asarray(g0)
    g1 = np.asarray(g1)
    if g0.shape != g1.shape:
        return ""
    diff = g0 != g1
    h, w = g0.shape
    runs = []
    for r in range(h):
        c = 0
        while c < w:
            if diff[r, c]:
                c0 = c
                while c < w and diff[r, c]:
                    c += 1
                sub = []
                i = c0
                while i < c:
                    v = g1[r, i]
                    j = i
                    while j < c and g1[r, j] == v:
                        j += 1
                    sub.append(f"{int(v)}x{j - i}")
                    i = j
                runs.append(f"r{r}c{c0}:" + ",".join(sub))
            else:
                c += 1
    return " ".join(runs) if runs else "(no change)"


def _transitions_block(
    trans: list[Transition],
    k: int = 8,
    *,
    previous_level_complete_grid: Optional[np.ndarray] = None,
    hud_mask: Optional[np.ndarray] = None,
    hud_mask_enabled: Optional[bool] = None,
) -> str:
    """Compact transition encoding for the induce prompt: ONE full grid (the layout) +
    per-transition DELTAS (changed cells), + the full WIN state if observed. Prefers
    grid-CHANGING transitions; keeps a couple of no-ops. Small enough for a local model's
    context window (the raw one-char-per-cell full-grid form overflowed gemma-4-12B at ~67k
    tokens on small boards, and on large boards like lp85's 64x64 grid overflowed even the
    13,824-token available budget with a SINGLE transition — see `_rle_grid`'s docstring; both
    full-grid renders below use the run-length encoding instead). Deltas use
    `_rle_delta_compact` (not `_rle_delta`) — on lp85's real transitions the raw comma-per-cell
    delta format became the new dominant cost once the full-grid fix landed (see
    `_rle_delta_compact`'s docstring for the measured before/after).

    REQ-ARC-WMTE-6010 PROMPT COHERENCE (added 2026-07-27 after an adversarial review found
    the mask was half-applied). `changed` / `noop` below decide WHICH transitions the LLM is
    shown, and until this date they were computed on RAW grids while the VERIFIER that
    grades the induced engine compared MASKED grids. On a game with a monotone HUD counter
    the two disagree completely: measured on real offline transitions, ft09 has 0 raw no-ops
    and 32 masked no-ops, lf52 has 0 raw and 59 masked. So the prompt asserted "this game has
    no inert actions" and showed six examples labelled as changes, while the grader had
    already decided those same six transitions did not change anything. The model was being
    asked to explain one world and marked against another. `_rle_delta_compact` renders on
    the RAW grids deliberately -- the LLM must still see the true pixels, including the HUD;
    only the CLASSIFICATION of a transition as changing-vs-inert is masked, which is the
    thing the verifier also classifies.

    Default `hud_mask_enabled=None` resolves through the same single flag resolver as every
    other consumer, so with the flag off this function is byte-identical to before.
    """
    if hud_mask_enabled is None:
        hud_mask_enabled = world_model_hud_mask_enabled()
    mask = hud_mask if hud_mask_enabled else None

    def _is_changed(t: Transition) -> bool:
        return not np.array_equal(apply_hud_mask(t.grid, mask), apply_hud_mask(t.next_grid, mask))

    changed = [t for t in trans if _is_changed(t)]
    noop = [t for t in trans if not _is_changed(t)]
    sample = changed[: k - 2] + noop[:2]
    out = []
    if sample:
        out.append(
            "INITIAL GRID (one full example of the state layout, run-length encoded; "
            "all grids are this shape):\n" + _rle_grid(sample[0].grid)
        )
    for t in sample:
        click = f" data={t.data}" if t.data else ""
        out.append(
            f"--- ACTION{t.action}{click} (level {t.level_before}->{t.level_after}): "
            f"changed cells (FULL, run-length) = {_rle_delta_compact(t.grid, t.next_grid)}"
        )
    win = next((t for t in trans if t.level_after > t.level_before), None)
    if win is not None:
        out.append(
            "WIN STATE (full grid of a level-complete state, run-length encoded — "
            "is_level_complete must return True here):\n" + _rle_grid(win.next_grid)
        )
    elif previous_level_complete_grid is not None:
        out.append(
            "WIN STATE EXEMPLAR (full grid of a state that COMPLETED the previous level, "
            "run-length encoded; the next level's completion likely looks structurally similar. "
            "Use this as a positive level-complete shape exemplar, not as an oracle for the "
            "exact next grid):\n" + _rle_grid(np.asarray(previous_level_complete_grid))
        )
    return "\n".join(out)


def objects_block(
    trans: list[Transition],
    *,
    previous_level_complete_grid: Optional[np.ndarray] = None,
    max_objects: int = 60,
) -> str:
    """LEVER #1 (REQ-ARC-WMTE-5830): object-structured serialization of the layout grid (and WIN state,
    if observed) for the induction prompt. Reuses `arc_color_blob_salience.blob_topology` unchanged.
    Objects are the connected-component partition; `object_hash` is a TRANSLATION-INVARIANT shape id so
    the LLM can recognize the SAME object across frames after it moves -- the raw run-length grid gives
    only order-1 position features that cannot. Defensive by construction: any failure returns "" so
    induction falls back to the raw-grid-only prompt (never breaks the default path), and the per-grid
    object table is capped at `max_objects` to bound prompt length on dense/large boards."""
    try:
        from carnot.agentic.arc_color_blob_salience import blob_topology
    except Exception:
        return ""
    changed = [t for t in trans if not np.array_equal(t.grid, t.next_grid)]
    layout = (changed[0] if changed else trans[0]).grid

    def _table(grid: np.ndarray, title: str) -> str:
        topo = blob_topology(np.asarray(grid))
        blobs = topo.get("blobs", [])
        hashes = topo.get("object_hashes", {})
        n = len(blobs)
        # Show the largest-by-pixel objects first; keep ORIGINAL ids so containment/adjacency stay valid.
        order = sorted(range(n), key=lambda i: -int(getattr(blobs[i], "pixel_count", 0)))[
            :max_objects
        ]
        shown = set(order)
        header = (
            f"{title} OBJECTS (connected components; obj<id>: color bbox=(y0,x0,y1,x1) px=<pixels> "
            f"shape=<translation-invariant id>)"
        )
        if n > len(order):
            header += f"  [showing largest {len(order)} of {n}]"
        rows = [header + ":"]
        for i in order:
            b = blobs[i]
            cy, cx = getattr(b, "centroid", (0.0, 0.0))
            rows.append(
                f"  obj{i}: color={int(b.color)} bbox={tuple(int(v) for v in b.bbox)} "
                f"px={int(b.pixel_count)} centroid=({float(cy):.1f},{float(cx):.1f}) shape={hashes.get(i)}"
            )
        children = {p: cs for p, cs in topo.get("children", {}).items() if cs and p in shown}
        adjacency = [
            pair for pair in topo.get("adjacency_list", []) if all(j in shown for j in pair)
        ]
        rows.append(f"  containment (parent->children): {children}")
        rows.append(f"  adjacency (touching id pairs): {adjacency}")
        rows.append(
            "  NOTE: two objects with the SAME shape id are the SAME object type regardless of "
            "position; use this to track objects across the transition deltas above."
        )
        return "\n".join(rows)

    try:
        parts = [_table(layout, "INITIAL")]
        win = next((t.next_grid for t in trans if t.level_after > t.level_before), None)
        if win is None and previous_level_complete_grid is not None:
            win = np.asarray(previous_level_complete_grid)
        if win is not None:
            parts.append(_table(win, "WIN STATE"))
        return "\n\n".join(parts)
    except Exception:
        return ""  # never break the default induction path


# A forceful CODE-ONLY directive for the L2+ induction call. The L2 induce prompt carries a WIN
# STATE exemplar, which makes Qwen3.5-9B burn its ENTIRE token budget on win-state chain-of-thought
# before reaching the code block (stop_type='limit', 0 code emitted -> goal_predicate_satisfiable
# stays False for ~10 milestones; see proto_l2_proposer_truncation_check + proto_l2_code_only_prefix,
# 2026-06-25). Prepending this directive AND adding a stop-sequence on the closing fence makes the
# model emit ONLY the code and stop: verified 195 tokens / 15.6s (vs 605s rambling / 450s truncated),
# valid engine+is_level_complete. DEFAULT ON (2026-06-25 operator directive); opt out with
# CARNOT_ARC_CODEONLY_INDUCE=0. NB: defeats truncation (emits code) but the induced goal predicate
# can still be degenerate -> see the goal-repair loop in arc_llm_reinduction.execute_bounded_llm_reinduction.
_L2_CODEONLY_DIRECTIVE = (
    "/no_think\n"
    "CRITICAL OUTPUT RULES -- obey EXACTLY:\n"
    "1. Output ONLY one ```python code block. NOTHING before it. NOTHING after it.\n"
    "2. Do NOT analyze the grids. Do NOT describe or reason about the win state. Do NOT write\n"
    "   step-by-step analysis, explanation, or commentary -- not even as comments.\n"
    "3. Your response MUST begin with the characters ```python and end with ```.\n"
    "4. Induce SIMPLE, GENERAL rules and write the requested function(s) directly. Skip all reasoning.\n\n"
)


def _induce_transitions_k() -> int:
    """REQ-ARC-FCP-5699-23: DEV-ONLY override (unset in production -- returns 8, the
    pre-existing _transitions_block/induce_prompt default, byte-identical behavior). Lets a
    diagnostic run test whether showing the LLM more per-action-type examples reduces the
    literal-coordinate-memorization pattern REQ-ARC-FCP-5699-22 found under the default cap."""
    import os

    override = os.environ.get("CARNOT_ARC_INDUCE_TRANSITIONS_K")
    return int(override) if override else 8


def _object_perception_on() -> bool:
    """LEVER #1 (REQ-ARC-WMTE-5830): DEV-ONLY (unset -> byte-identical pre-existing prompt). When
    CARNOT_ARC_OBJECT_PERCEPTION=1, induce_prompt appends a connected-component OBJECT table
    (translation-invariant object_hash for cross-frame identity, containment tree, adjacency)
    ALONGSIDE the raw run-length grid -- feeding the inducer the object structure that today only
    feeds the (gated-off) search salience prior. Attacks GAP-ARCH-FEATURES: the raw grid gives the LLM
    order-1 position-only features (can't track an object across frames after it moves); object_hash can."""
    import os

    return os.environ.get("CARNOT_ARC_OBJECT_PERCEPTION") == "1"


# REQ-ARC-WMTE-5717: DEV-ONLY playbook methodology exemplars for the STALL re-induction
# path. Default OFF (env CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED unset -> byte-identical
# prompt, exactly like the CARNOT_ARC_CODEONLY_INDUCE / _REFACTOR_STRUCTURE_REMINDER
# gates above). A SMALL, game-AGNOSTIC few-shot of the recurring "orient, hypothesize,
# test, revise" exploration method distilled from the solve corpus
# (docs/research-notes/arc-exploration-playbook-20260717.md) -- PATTERN statements only,
# never a per-game fact (color/coordinate/mechanic), so they transfer to a HIDDEN game the
# agent has never seen. Deliberately terse: a sibling experiment (exp5714) found that
# long-reasoning induction overruns the token budget and emits zero code, so this biases
# the model's PRIORS without asking it to reason at length.
_PLAYBOOK_EXEMPLAR_BLOCK = """GENERAL EXPLORATION PRINCIPLES (observed across many ARC-AGI-3 games -- apply as PRIORS
when inducing the rules below; do NOT copy any specific game's colors/coordinates):
- Prefer SIMPLE, GENERAL rules over per-cell/hardcoded-coordinate special cases; a rule
  that memorizes exact coordinates rarely generalizes to the next state.
- Action effects can differ from level to level -- induce them from THESE transitions, do
  not assume a mapping carried over from a prior level.
- An object that recolors on contact or selection is the SAME object, not a new one.
- A level-complete state is often the frame AFTER the winning action; ground
  is_level_complete on the STRUCTURAL win condition, not one memorized exact grid.
- Some actions are inert (no change) or reset the level -- model those honestly.
- A fixed goal DISPLAY/legend is not the interactive target; the target is a piece that
  actually moves or changes.

"""


def induce_prompt(
    game: str,
    trans: list[Transition],
    cell: int,
    *,
    previous_level_complete_grid: Optional[np.ndarray] = None,
    k: int = 8,
    include_playbook_exemplars: bool | str = False,
    hud_mask: Optional[np.ndarray] = None,
    hud_mask_enabled: Optional[bool] = None,
) -> str:
    # REQ-ARC-WMTE-6010 PROMPT COHERENCE: `hud_mask` reaches `_transitions_block` so the
    # transitions the LLM is SHOWN are classified changing-vs-inert by the same rule the
    # verifier uses to GRADE the resulting engine. See `_transitions_block`'s docstring for
    # the measured incoherence this closes (ft09: prompt asserted 0 no-ops, grader saw 32).
    # REQ-ARC-FCP-5699-23: k defaults to _transitions_block's own default (8, unchanged
    # production behavior). REQ-ARC-FCP-5699-22 found the default shows the LLM only ~6
    # grid-changing transitions of the 25 collected -- roughly one per action type, a
    # data-starvation signature matching observed hardcoded-literal-coordinate memorization
    # (g50t's engine). Callers may raise k (DEV-ONLY, via CARNOT_ARC_INDUCE_TRANSITIONS_K) to
    # test whether more per-action examples let the LLM infer general rules instead.
    h, w = trans[0].grid.shape
    colors = sorted(set(int(v) for t in trans for v in t.grid.flatten().tolist()))
    # REQ-ARC-WMTE-5717/5718: DEV-ONLY exemplar prefix. The inject/don't-inject DECISION is made
    # by the caller (the agent's stall-only gate). Three modes, all default to byte-identical:
    #   False / ""     -> no injection (the exact pre-existing prompt).
    #   True           -> the STATIC generic exemplar block (REQ-5717).
    #   <non-empty str> -> that exact RETRIEVED block (REQ-5718 RAG: top-K patterns for THIS
    #                      stuck situation), already formatted by arc_playbook_retrieval.
    if isinstance(include_playbook_exemplars, str):
        block = include_playbook_exemplars.strip()
        exemplars = (block + "\n\n") if block else ""
    else:
        exemplars = _PLAYBOOK_EXEMPLAR_BLOCK if include_playbook_exemplars else ""
    return f"""{exemplars}You are inducing an EXECUTABLE WORLD MODEL for the ARC-AGI-3 game '{game}'.

The game state is a {h}x{w} integer grid (logical resolution; colors {colors}). You are
given REAL observed transitions COMPACTLY: one full INITIAL grid (the layout), then per
transition the action and its DELTA = the FULL set of changed cells as run-length runs of the
form r<row>c<col0>:<v0>x<n0>,<v1>x<n1>,... — each run is a horizontal span of changed cells
starting at (row, col0); within that span, the NEW values are themselves given as
<value>x<count> pairs left-to-right (so a span of 6 changed cells that are all now color 5
appears as "5x6", not six separate "5"s). To apply a transition's delta to the prior grid, for
each run walk its <value>x<count> pairs in order, setting <count> consecutive cells starting at
the next unfilled column (starting at col0) to <value>; all other cells are unchanged. The delta
is COMPLETE (not truncated). Full grids (the INITIAL
grid and, if shown, the WIN STATE grid) use a DIFFERENT run-length form to stay compact on large
boards: one line per row, "r<row>:<v0>x<n0>,<v1>x<n1>,...". Each row's runs are listed
left-to-right and cover EVERY column with no gaps and no overlap, so the starting column of each
run is IMPLICIT: it equals the sum of the counts of all runs before it in that row (the first run
in a row starts at column 0). To reconstruct a full grid, for each row walk its runs in order,
placing <n> consecutive cells of value <v> starting at the next unfilled column.
Actions are integers 1-7; ACTION6 is a click
with data={{'x':px,'y':py}} in PIXEL coords (pixel = logical*{cell}); others are
keyboard/directional with data=None.

Write a Python file at results/arc_e3/{game}/world_model.py with EXACTLY two functions:

    import numpy as np
    def engine(grid, action, data):
        # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
        ...
    def is_level_complete(grid):
        # return True if `grid` is a level-complete / win state, else False.
        ...

Induce the transition RULES from the observed data (movement, gravity, toggling,
pushing, collection, etc.). Prefer SIMPLE GENERAL rules over per-frame special cases.
Use only numpy + stdlib. Do not read files or network. Make engine() pure and
deterministic. Write ONLY that one file.

OBSERVED TRANSITIONS:
{_transitions_block(trans, k, previous_level_complete_grid=previous_level_complete_grid, hud_mask=hud_mask, hud_mask_enabled=hud_mask_enabled)}
{("OBJECT STRUCTURE (same frames, connected-component view -- use object shape ids to track objects across the deltas above):" + chr(10) + objects_block(trans, previous_level_complete_grid=previous_level_complete_grid)) if _object_perception_on() else ""}"""


_REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH = 8


def _bounded_mismatches(mismatches: list, *, limit: int = 5) -> list:
    """REQ-ARC-FCP-5699-26: cap each mismatch's cell-diff lists BEFORE JSON-encoding, instead
    of encoding everything and then hard-slicing the resulting string by raw character count.
    The raw-slice approach (the pre-existing `json.dumps(...)[:4000]`) produced genuinely
    INVALID JSON: verified on a real g50t counterexample, 5 real mismatches serialize to
    12,212 characters, and `[:4000]` cuts the string mid-field-name (`"true_chang` with the
    closing `e"` sliced off) -- the model was being shown malformed, truncated JSON and asked
    to debug from it, not a genuine capacity/reasoning limitation. Bounding each mismatch's
    `true_change`/`your_prediction_was_wrong_at` lists to a fixed cell count keeps every
    mismatch entry structurally complete and the overall JSON valid regardless of how large the
    underlying diffs are, while still showing a representative SAMPLE of cells per mismatch (an
    `_omitted_count` companion field records how many were cut, so the count is honest, not
    silently dropped)."""
    bounded = []
    for m in mismatches[:limit]:
        m = dict(m)
        for key in ("true_change", "your_prediction_was_wrong_at"):
            cells = m.get(key)
            if isinstance(cells, list) and len(cells) > _REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH:
                m[f"{key}_omitted_count"] = len(cells) - _REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH
                m[key] = cells[:_REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH]
        bounded.append(m)
    return bounded


# REQ-ARC-FCP-5699-31: DEV-ONLY structural reminder (unset -> exact pre-existing prompt,
# byte-identical), directly targeting the pathology REQ-ARC-FCP-5699-30 found by reading a real
# raw completion: given the SAME real g50t counterexample data, the model abandoned the required
# interface entirely -- it wrote a self-contained `class WorldModel` with `self`-bound methods
# and a fabricated grid representation instead of patching the required TOP-LEVEL engine()/
# is_level_complete() functions, never emitted is_level_complete at all, and stopped mid-
# statement. This is a content/structure reminder, NOT the codeonly "skip all reasoning"
# directive REQ-ARC-FCP-5699-26 confirmed is deliberately excluded from refactor() (refactor
# stays a genuine reasoning task) -- it reminds the model WHAT shape its answer must take without
# telling it not to think.
_REFACTOR_STRUCTURE_REMINDER_BEFORE = """
REQUIRED OUTPUT STRUCTURE -- do not deviate from this: your fixed code must define EXACTLY two
TOP-LEVEL functions, in the SAME format as the file you are correcting:

    def engine(grid, action, data):
        ...
    def is_level_complete(grid):
        ...

Do NOT wrap them in a class. Do NOT use `self`. Do NOT invent a new internal grid
representation or shape -- `grid` is the SAME real grid shape/format already used by the
code above; reuse it exactly, do not redesign it.
"""
_REFACTOR_STRUCTURE_REMINDER_AFTER = """
Reminder: return ONLY the corrected `engine(grid, action, data)` and `is_level_complete(grid)`
top-level functions -- no classes, no `self`, no invented state, no renamed parameters.
"""


def refactor_prompt(game: str, vr: VerifyResult) -> str:
    import os

    mism = json.dumps(_bounded_mismatches(vr.mismatches), indent=1)
    before = ""
    after = ""
    # Graduated to default-on (REQ-ARC-FCP-5699-35): REQ-31/32 validated this reminder fixes a
    # real class-wrapping/missing-function pathology with zero observed regressions (6/6 success
    # across a full-budget live run). CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER=0 remains an
    # explicit opt-out escape hatch, matching the CARNOT_ARC_MTP=0 pattern elsewhere in this file.
    if os.environ.get("CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER", "1") != "0":
        before = _REFACTOR_STRUCTURE_REMINDER_BEFORE
        after = _REFACTOR_STRUCTURE_REMINDER_AFTER
    return f"""The executable world model at results/arc_e3/{game}/world_model.py reproduces
only {vr.n_correct}/{vr.n} ({vr.accuracy:.0%}) of the observed transitions. Below are
failing cases (BEFORE / your PREDICTED / the true OBSERVED next grid). Fix engine() so it
reproduces these too, and REFACTOR toward simpler, more general rules (replace special
cases with shared rules) while keeping the cases it already gets right. Edit only that
file.
{before}
MISMATCHES:
{mism}
{after}"""


@dataclass
class CodexProposer:
    """DEV-ONLY proposer (codex/gpt-5.5). Requires INTERNET, so it is NOT legal in the
    OFFLINE competition eval — use it only to validate the E3 loop during development.
    For the competition, use LocalGGUFProposer (open-weight, offline)."""

    timeout: int = 420
    offline_legal: bool = False
    # REQ-ARC-WMTE-5717: DEV-ONLY (see LocalGGUFProposer's field); default False -> byte-identical.
    include_playbook_exemplars: bool | str = False

    def induce(
        self,
        game: str,
        trans: list[Transition],
        cell: int,
        *,
        previous_level_complete_grid: Optional[np.ndarray] = None,
    ) -> tuple[bool, str]:
        (E3_DIR / game).mkdir(parents=True, exist_ok=True)
        return _codex(
            induce_prompt(
                game,
                trans,
                cell,
                previous_level_complete_grid=previous_level_complete_grid,
                k=_induce_transitions_k(),
                include_playbook_exemplars=self.include_playbook_exemplars,
            ),
            self.timeout,
        )

    def refactor(self, game: str, vr: VerifyResult) -> tuple[bool, str]:
        return _codex(refactor_prompt(game, vr), self.timeout)


def _resolve_gguf(repo_substr: str) -> Optional[str]:
    """Find a cached GGUF weight file for an open-weight SOTA model (offline)."""
    import glob

    base = Path.home() / ".cache" / "huggingface" / "hub"
    for d in base.glob(f"models--*{repo_substr}*GGUF"):
        hits = sorted(d.glob("snapshots/*/*.gguf"))
        if hits:
            return str(hits[0])
    return None


# llama.cpp SERVER, GPU-enforced. PREFER the HIP build (ROCm iGPU gfx1150 / Radeon 890M,
# ~108GB UNIFIED memory) — it does NOT contend with the conductor's CUDA experiments on the
# 2x3090s (operator directive 2026-06-17: iGPU is the outer-loop target, never CPU). Falls
# back to the CUDA build only if the iGPU build is absent. The venv llama-cpp-python is
# CPU-only; llama-cli/llama-completion hang/crash on gemma's chat template — the server's
# /completion does RAW completion (no chat template) and keeps the model loaded across calls.
def _resolve_llama_server() -> Path:
    # Kaggle/live submission: point CARNOT_LLAMA_SERVER at the bundled CUDA llama-server binary
    # (/kaggle/input/<llamacpp-dataset>/llama-server). MTP (--spec-type draft-mtp) lives in
    # libllama-common, which the BINARY links -- the stock llama-cpp-python wheel cannot do native MTP,
    # so the submission bundles this binary + its shared libs, NOT a wheel.
    import os

    env = os.environ.get("CARNOT_LLAMA_SERVER")
    if env:
        return Path(env)
    base = Path.home() / ".cache" / "llama.cpp-master"
    hip = base / "build-hip" / "bin" / "llama-server"  # ROCm iGPU — no conductor contention
    return hip if hip.exists() else base / "build" / "bin" / "llama-server"  # CUDA 3090 fallback


LLAMA_SERVER = _resolve_llama_server()

# ---------------------------------------------------------------------------------------------
# THE MEASURED VRAM ENVELOPE for the local 3090 generator launch (exp5866, 9 configs, refit max
# error 0.19%). Named constants rather than a magic literal because TWO things depend on the same
# arithmetic and MUST NOT drift apart: the context-pool size (_default_induce_n_ctx) and the
# free-VRAM guard that decides whether the card can hold the resulting server.
#
# SCOPE (2026-07-27 adversarial review, finding "envelope scoped to a shape the scored path never
# runs"): this fit was taken with `--spec-type draft-mtp` ON, i.e. the LOCAL/dev launch shape, where
# the MTP self-draft loads a second copy of the weights. It over-predicts the SCORED Kaggle shape by
# ~6.1 GB, because scripts/kaggle/submission_kernel/main.py forces CARNOT_ARC_MTP=0 there. The
# mtp-OFF pair is recorded separately below and is the one to reason about for the 16GB-class card.
# Over-prediction is the safe direction for a guard, so this constant is deliberately the mtp-ON fit.
_VRAM_MTP_ON_INTERCEPT_MIB = 10547.0  # weights + MTP self-draft copy + fixed overhead
_VRAM_MTP_ON_PER_CTX_MIB = 0.02519  # q8_0 KV per shared-pool cell
_VRAM_PER_SLOT_MIB = 206.83  # per llama.cpp slot, independent of n_ctx
# llama-server with no explicit --parallel: n_parallel=4 AND kv_unified=true (server.cpp:106-110).
# READ from the source of the local build and CONFIRMED from a running server's own /props
# (`total_slots: 4`, 2026-07-27). It is the K the shared-pool admission arithmetic has to survive,
# because the eval framework starts one thread per game with no pool (swarm.py:91) and llama.cpp
# queues everything past its own slot count.
_LLAMA_SERVER_DEFAULT_SLOTS = 4
# The real `induce_prompt()` for the largest logical grid in ops/arc_solve_registry.yaml (64x64),
# measured through the server's own /tokenize rather than estimated. The WORST case, not the
# typical one, because the generated length is unknowable in advance.
_INDUCE_WORST_CASE_PROMPT_TOKENS = 15734
# Mirrors LocalGGUFProposer.max_tokens and the CARNOT_ARC_INDUCE_MAX_TOKENS default read at both
# construction sites in arc_competition_agent.py. Named here so the context-pool derivation and
# the completion budget cannot drift apart -- see _default_induce_n_ctx().
_INDUCE_DEFAULT_MAX_TOKENS = 4096
# Yield-if-the-conductor-needs-it margin. Must cover measurement scatter (the same 81920/mtp-on
# launch measured 13452 and 13518 MiB per-PID on two occasions), the driver/context overhead
# nvidia-smi attributes outside the fit, and enough slack that we do not admit a card we will then
# cudaMalloc-fail on. A failed bind costs 180s in _ensure_server() and returns the agent to a
# SILENT LLM-off state -- exactly the class of fault this file's n_ctx fix exists to remove, so the
# guard must be conservative in the direction of declining the card.
_GENERATOR_CUDA_GUARD_MARGIN_MIB = 1500


def _generator_cuda_min_free_mb() -> int:
    """Free VRAM (MiB) the opt-in 3090 generator path requires before it will bind a card.

    DERIVED, never a hand-typed literal. It was a literal (13000, commented "loads ~11.5GB") and
    the 2026-07-27 n_ctx 16384 -> 81920 fix raised the real footprint to ~13.4-13.5 GiB WITHOUT
    touching it, so the guard would have admitted a card with 13000-13452 MiB free and then
    cudaMalloc-failed: server exits, `_ensure_server()` burns its full retry budget, `generate()`
    returns `(False, msg)`, and the agent runs LLM-off while still reporting itself as the LLM-on
    scored path. That is a NEW silent-degradation path of exactly the shape the fix was removing.

    Computing it from the SAME `_default_induce_n_ctx()` the server is actually launched with means
    an operator raising CARNOT_ARC_INDUCE_N_CTX automatically raises the guard too. Pinned by
    `tests/python/test_arc_generator_vram_guard.py`.
    """
    predicted = (
        _VRAM_MTP_ON_INTERCEPT_MIB
        + _VRAM_MTP_ON_PER_CTX_MIB * float(_default_induce_n_ctx())
        + _VRAM_PER_SLOT_MIB * float(_LLAMA_SERVER_DEFAULT_SLOTS)
    )
    return int(predicted + _GENERATOR_CUDA_GUARD_MARGIN_MIB)


def _cuda_gpu_free_mb(idx: int) -> int:
    """Free VRAM (MiB) on CUDA GPU `idx` via nvidia-smi; -1 if unavailable. The guard input for the
    opt-in 3090 generator path."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return int(lines[idx]) if 0 <= idx < len(lines) else -1
    except Exception:
        return -1


# How many times _generator_server_and_env retries the free-VRAM check before conceding to the
# iGPU fallback, and how long it waits between retries. A just-crashed CUDA process does not
# release its VRAM allocation to the driver instantaneously; LocalGGUFProposer's self-heal path
# (_ensure_server) calls _generator_server_and_env() immediately after detecting an unhealthy
# server, so a single free-memory snapshot can catch the dying process's VRAM still "in use" and
# wrongly fall back to the iGPU for that server's entire subsequent lifetime. Found 2026-07-21
# (exp5768): three consecutive self-heals after a CUDA server crash all silently landed on the HIP
# build with near-zero VRAM, running a 31B model on CPU for hours before being noticed. An initial
# fix used 4 attempts / 1.5s apart (~6s total) -- confirmed insufficient 2026-07-22 when the exact
# same failure recurred (a longer reclaim window than 6s in that instance). Widened to 10 attempts
# / 2s apart (~20s total) after that. Still bounded small enough that a genuinely busy card (a real
# conductor job actually holding the VRAM) yields to the iGPU within seconds, not a long stall.
_GENERATOR_CUDA_FREE_RETRY_ATTEMPTS = 10
_GENERATOR_CUDA_FREE_RETRY_DELAY_S = 2.0


def _cuda_gpu_has_headroom(idx: int, min_free_mb: int) -> bool:
    """True if GPU `idx` has >= `min_free_mb` free, retrying briefly across
    _GENERATOR_CUDA_FREE_RETRY_ATTEMPTS attempts to survive a just-crashed process's VRAM not yet
    being reclaimed by the driver (see the constants' docstring above)."""
    for attempt in range(_GENERATOR_CUDA_FREE_RETRY_ATTEMPTS):
        if _cuda_gpu_free_mb(idx) >= min_free_mb:
            return True
        if attempt < _GENERATOR_CUDA_FREE_RETRY_ATTEMPTS - 1:
            time.sleep(_GENERATOR_CUDA_FREE_RETRY_DELAY_S)
    return False


def _free_port() -> int:
    """An OS-assigned free localhost port, for the case where the port we wanted is already
    held by a server whose context pool is too small for us (see `_reusable`). Binding to 0
    and reading the assignment back is the only race-free way to pick one; the tiny window
    between close() and llama-server's bind() is accepted because the alternative -- reusing
    a mismatched server -- is the silent-degradation fault this whole path exists to remove."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _generator_server_and_env() -> tuple[Path, Optional[dict]]:
    """Resolve the llama-server binary + launch env for the generator, evaluated at LAUNCH time so the
    3090 guard sees current GPU state.

    Priority:
      1. CARNOT_LLAMA_SERVER (Kaggle/live bundled CUDA binary) -- unchanged; inherits ambient env.
      2. OPT-IN CARNOT_ARC_GENERATOR_CUDA_GPU=<idx> -> the local CUDA build pinned to that 3090 via
         CUDA_VISIBLE_DEVICES, but ONLY if the card has >=_generator_cuda_min_free_mb() free (checked
         via _cuda_gpu_has_headroom, which retries briefly to survive a just-crashed process's VRAM
         not yet being reclaimed -- see that function's docstring). This is the operator-approved
         (2026-06-19) use of one idle 3090 for generator throughput now that the TRM run is retired;
         the free-memory guard yields to any conductor job already on the card.
      3. Default: the iGPU HIP build (no conductor contention), else the CUDA build.
    Returns (server_path, env_or_None); env=None means inherit the ambient environment (legacy behavior).
    """
    import os

    explicit = os.environ.get("CARNOT_LLAMA_SERVER")
    if explicit:
        return Path(explicit), None
    base = Path.home() / ".cache" / "llama.cpp-master"
    cuda = base / "build" / "bin" / "llama-server"
    hip = base / "build-hip" / "bin" / "llama-server"
    gpu = (os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU") or "").strip()
    if gpu and cuda.exists():
        try:
            idx = int(gpu)
        except ValueError:
            idx = -1
        if idx >= 0 and _cuda_gpu_has_headroom(idx, _generator_cuda_min_free_mb()):
            return cuda, dict(os.environ, CUDA_VISIBLE_DEVICES=str(idx))
        # guard tripped (card busy / unavailable / bad idx) -> fall through to the iGPU path,
        # never fight the conductor for the 3090.
    return (hip if hip.exists() else cuda), None


def _describe_http_failure(exc: BaseException) -> str:
    """Render a completion-request exception INCLUDING the server's own response body.

    WHY (exp5866 finding 4). The old code did `f"...failed: {exc!r}"`, and for a
    urllib HTTPError that repr is just `<HTTPError 500: 'Internal Server Error'>` --
    the generic reason phrase. The ONE informative string in the whole failure was
    thrown away unread:

      * the 500 body says `Context size has been exceeded.` (the concurrency fault)
      * the 400 body says `request (15754 tokens) exceeds the available context size
        (8192 tokens), try increasing it` -- literally the fix, in the message

    Two independent sessions spent effort re-deriving what these bodies already said,
    because nothing ever printed them. This is a RECORD change only: same exception
    handling, same (False, msg) return, same control flow -- just a message that
    contains the evidence. Never raises: a body that cannot be read degrades to the
    plain repr rather than replacing one silent failure with another.
    """
    base = repr(exc)[:200]
    body = ""
    try:  # urllib.error.HTTPError is a file-like object over the response body
        reader = getattr(exc, "read", None)
        if callable(reader):
            raw = reader()
            if isinstance(raw, bytes):
                raw = raw.decode(errors="replace")
            body = str(raw or "")[:400]
    except Exception:
        body = ""
    return f"{base} body={body!r}" if body else base


def _default_induce_n_ctx() -> int:
    """The generator server's SHARED context-pool size, in tokens (llama-server `-c`).

    WHY 81920 AND NOT 16384 (the concurrency fault, measured 2026-07-27, exp5866).
    llama-server with no explicit `--parallel` sets `n_parallel=4` AND `kv_unified=true`
    (its own default, server.cpp:106-110). kv_unified means the 4 slots share ONE pool of
    `-c` cells -- they do NOT each get `-c` cells, and they do NOT get `-c / 4` either
    (that is the DIVIDED-context branch, which only happens when you pass `--parallel`
    explicitly). So the real admission requirement is:

        n_ctx  >=  K_concurrent * (prompt_tokens + max_tokens)

    The eval framework starts ONE THREAD PER GAME with no pool (swarm.py:91), so induce
    requests arrive together; llama.cpp caps concurrency at its own 4 slots and QUEUES the
    rest, which is why K=4 -- not the ~110 game count -- is the number that has to fit.

    At the previous 16384 with max_tokens=4096, the fault fired at K=2 (2 * (5968+4096) =
    20128 > 16384), and it had THREE distinct shapes, all invisible to the concurrency-1
    probing every prior measurement used:
      A. HTTP 500 "Context size has been exceeded." -- server survives (large prompts).
      B. server DEATH: `GGML_ASSERT(logits != nullptr)` -> `ggml_abort` inside
         `update_slots()` -- permanent, every later request gets RemoteDisconnected
         (small prompts admitted, generations collectively overrun the pool).
      C. WORST: HTTP **200** with a silently truncated completion, when the prompt nearly
         fills the pool and only the leftover cells remain for generation.
    Because `generate()` returns `(False, msg)` instead of raising, A and B degrade the
    agent to LLM-OFF while it still reports itself as the LLM-on scored path, and C is not
    even visible as a failure.

    WHAT SIZING THE POOL ACTUALLY REMOVED -- narrowed 2026-07-27 after an adversarial review
    of the shipping commit found this docstring claiming all three, contradicted by that
    commit's own end-to-end evidence.

      A. REMOVED, measured. `n_context_exceeded` went 36 -> 0 across all pre-fix (16384) vs
         all post-fix (81920) cells; the direct back-to-back control at the same prompt in
         the same tree went control 2/2 and 4/4 HTTP 500, fix 2/2 and 4/4 HTTP 200.
      C. REMOVED for the worst measured prompt: every fix request reported
         `predicted_n == 4096 == max_tokens`, i.e. `pool_exhaustion_limit == 0` in every
         cell, where the pre-fix K=1 cell at the same prompt truncated to 2133 chars with
         630 cells of generation room.
      B. **NOT DEMONSTRATED REMOVED.** 6 of the 12 `llm_on_fix_probe__*` cells still carry
         `RemoteDisconnected` server-failure diagnostics at `generator_n_ctx=81920` -- 16
         diagnostics in total, 2 cells ending `generator_healthy_after=False`, and
         `lp85_color04` fully LLM-off at calls=4 / responses=0 / errors=4. The
         requantification records this as `n_remote_disconnected_post_fix: 16` and sets it
         aside as confounded with an external process killer, which may well be right --
         but the discriminating evidence (the server's own `ggml_abort` line in its log)
         was never captured, and `RemoteDisconnected` with the server gone is exactly mode
         B's recorded signature. The 6-cell HTTP gate that reported "fix 0 failures" cannot
         see mode B at all: it fires ONE worst-case prompt shape at K in {2,4}, and mode B's
         trigger is the opposite shape (many SMALL prompts, individually admitted, whose
         GENERATIONS collectively overrun the pool over a long horizon).
    So: treat A and C as fixed, and B as open. Before claiming B is fixed, run a
    mode-B-specific arm (many small concurrent prompts, long horizon, external killers
    excluded) and capture the server's stderr so `ggml_abort` can be told apart from SIGTERM.

    81920 = 4 * (15734 + 4096) rounded up to a 4096 multiple, where 15734 tokens is the
    real `induce_prompt()` for the largest logical grid in `ops/arc_solve_registry.yaml`
    (64x64), measured through the server's own `/tokenize` -- not estimated. Worst case,
    not typical, because the GENERATED length is unknowable in advance.

    COMPUTED, NOT HARDCODED (2026-07-27 review). The first version of this function returned
    a literal `81920` while `max_tokens` was independently read from
    `CARNOT_ARC_INDUCE_MAX_TOKENS` at BOTH construction sites in arc_competition_agent.py
    (:889 and :5014). So the two halves of the admission inequality could diverge exactly the
    way this docstring warns the construction sites once did: `CARNOT_ARC_INDUCE_MAX_TOKENS
    =8192` needs 4*(15734+8192) = 95704 cells and would have silently re-broken K=4 against
    an unchanged 81920. Both halves now come from the same arithmetic, so an operator raising
    the completion budget raises the pool with it. At the default 4096 this returns exactly
    the 81920 that was measured and shipped -- the change is a derivation, not a re-sizing.

    HOW LITTLE SLACK THERE IS. 81920/4 - 4096 = 16384 tokens is the largest prompt this pool
    admits at K=4, versus the 15734-token worst case it is sized for: 650 tokens of margin per
    slot. That is not much, and it is why the pre-flight probe must use a prompt of the SAME
    measured worst-case size rather than an eyeballed synthetic one -- the kernel's original
    synthetic probe string measured 17238 tokens through the model's own tokenizer, i.e.
    854 tokens OVER what the pool admits, and at K=4 it returns 4/4 HTTP 500 "Context size
    has been exceeded" (measured directly, 2026-07-27, RTX 3090, mtp-off, per-slot n_tokens
    20469..20493 at release == 81920/4 exactly). It passed the shipped probe only because
    that probe ran K=2.

    WHY THIS AXIS AND NOT ANOTHER. Measured VRAM envelope (9 configs, refit max error
    0.19%): `MiB = 10547 + 0.02519*n_ctx + 206.8*slots`. Context is the CHEAP axis --
    16384 -> 81920 costs +1668 MiB, while a slot costs ~207 MiB regardless of n_ctx. The
    alternatives were measured and rejected: an explicit `--parallel 4` DIVIDES the pool
    (4096/slot) and is strictly worse; `--parallel 1` passes an HTTP gate and costs LESS
    VRAM but generated 648/650/184/648 tokens against a 4096 budget -- i.e. it converts
    the loud 500 into silent mode C, the exact defect under investigation. `n_ctx_train`
    is 262144, so 81920 is well inside the model's trained context.

    OVERRIDE with CARNOT_ARC_INDUCE_N_CTX for a tight-VRAM box or a model with a fatter
    per-token KV than the frozen 9B live generator (this default is sized for that model;
    a ~3x-larger model's KV would cost ~3x the 1668 MiB). Read via default_factory so the
    literal lives in exactly ONE place -- both construction sites in
    arc_competition_agent.py (`_proposer()` and `_load_sge_candidate_router()`) omit n_ctx
    and therefore cannot silently diverge from each other, which is the failure the
    REQ-ARC-FCP-5699-35 comment at that second site records having already happened once
    for max_tokens.
    """
    import os

    override = os.environ.get("CARNOT_ARC_INDUCE_N_CTX")
    if override:
        return int(override)
    max_tokens = int(
        os.environ.get("CARNOT_ARC_INDUCE_MAX_TOKENS", str(_INDUCE_DEFAULT_MAX_TOKENS))
    )
    need = _LLAMA_SERVER_DEFAULT_SLOTS * (_INDUCE_WORST_CASE_PROMPT_TOKENS + max_tokens)
    # Round UP to a 4096 multiple: llama.cpp allocates in blocks and a round pool is easier to
    # reason about against the published VRAM envelope, whose n_ctx samples are all multiples.
    return int(-(-need // 4096) * 4096)


@dataclass
class LocalGGUFProposer:
    """OFFLINE-LEGAL, DECENTRALIZED, GPU-ENFORCED proposer (CLAUDE.md decentralization
    rule 1-2 + the always-GPU directive): an OPEN-WEIGHT local model induces the world
    model with NO internet, so it runs inside the competition's offline eval sandbox. The
    induced code quality is GROUNDED by the Carnot WorldModelVerifier regardless of model
    strength — a weaker local model just earns a lower verifier score, honestly.

    GPU-ENFORCED via a CUDA llama-server (-ngl 999 = all layers on GPU); FAILS LOUD if the
    server can't start — never a silent CPU fallback (the CPU path is excruciatingly slow
    and a 20-core conductor-fight). The model stays loaded across induce/refactor calls.
    NOT TRM and NOT a closed model: an open local LLM (a TRM-class trained model is the
    other local engine)."""

    repo_substr: str = "gemma-4-12B-it"  # lightweight SOTA: fast on GPU for per-game induction
    # SHARED context pool (llama-server -c). 81920 by measurement, env-overridable --
    # see _default_induce_n_ctx() above for the full derivation and the rejected alternatives.
    n_ctx: int = field(default_factory=_default_induce_n_ctx)
    max_tokens: int = 4096  # a full world-model engine needs >2048 (it truncated mid-code)
    timeout: int = 300
    port: int = 8919
    offline_legal: bool = True
    # Live-submission deploy config (all OPT-IN; defaults preserve legacy behavior). Validated 2026-06-19:
    # Qwen3.5-9B-MTP is the selected ARC live generator (62.5% Layer-B grounding vs DeepSeek-Flash 25%,
    # ~13 tok/s with MTP, 5.9GB Q4 fits 16GB). See docs/research-notes/arc-16gb-model-alternatives-2026-06-18.md.
    mtp: bool = False  # --spec-type draft-mtp (self-draft via the -MTP- GGUF's nextn heads)
    kv_quant: Optional[str] = (
        None  # e.g. "q8_0" -> --cache-type-k/v q8_0 (halves KV, near-lossless)
    )
    # -ngl: how many transformer layers' weights live on the GPU. 999 = all on GPU (fast, default).
    # Operator prefill-to-RAM lever (2026-06-21): on the shared 16GB eval GPU the LLM (5.9GB MTP-off)
    # coexists with the live per-game CNN dynamics fit (measured 1.45GB peak) + the q8 KV-cache. Full
    # offload already fits (~9.4GB of 16GB), so 999 stays the default. But if a heavier config (deeper
    # search, larger CNN, bigger ctx) pushes VRAM, set CARNOT_ARC_NGL below the layer count: the
    # un-offloaded layers stay PREFILLED in system RAM (llama.cpp mmaps the GGUF -> host page cache;
    # 125GB RAM trivially holds the 5.9GB Q4 weights) and compute on CPU, freeing VRAM for KV + CNN
    # training. The wall-clock cost is acceptable because the ARC eval has NO time limit (only the 12h
    # Kaggle-notebook cap + the 600 RPM real-env rate limit, neither of which gates internal generation).
    n_gpu_layers: int = 999
    no_think_prefix: str = ""  # e.g. "/no_think\n" -> suppress hybrid-thinking CoT (Qwen3)
    # REQ-ARC-WMTE-5725: OPT-IN. When True, generate()/complete_text() POST to the OpenAI-compatible
    # /v1/chat/completions endpoint (a single user turn) instead of the raw /completion endpoint. The
    # server then applies the GGUF's OWN embedded chat template (turn delimiters, e.g. Qwen3.6's
    # <|im_start|>assistant), which Qwen3.6-family models (ThinkingCap-27B) REQUIRE to know a turn has
    # started -- the raw /completion path (no template) made those models emit an immediate EOS with ~0
    # output on ~10/12 genuine-reasoning induce cells (REQ-ARC-WMTE-5724 measurement-validity caveat).
    # Default False keeps the FROZEN live-generator path (Qwen3.5-9B raw /completion) byte-identical.
    # The response is normalized back into llama.cpp's {content, stop_type, truncated} shape so
    # _record_completion_diagnostics + every caller works unchanged; a split-out reasoning_content (some
    # builds extract <think> into its own field) is folded back into `content` wrapped in <think> tags
    # so reason_engaged detection + max_raw_completion_len stay faithful to what the model generated.
    use_chat_template: bool = False
    model_path: Optional[str] = (
        None  # explicit .gguf path; on Kaggle set to the bundled /kaggle/input/... path
    )
    tries: int = 3
    extra_server_args: tuple = ()  # e.g. ("-fit", "off") -- raw args appended to the launch
    # command verbatim. Added for exp5705 after llama-server's default -fit heuristic hard-hung
    # (confirmed via /proc/PID/io: zero read progress for 12+ minutes) loading a large hybrid
    # linear/full-attention model (Qwen3.6-27B) on this project's HIP/ROCm build -- -fit off has
    # no downside when n_gpu_layers and n_ctx are both already explicit (nothing left to auto-fit).
    _proc: Any = None
    # MANDATORY truncation detection (operator directive, REQ-ARC-FCP-5699-27): every completion
    # request (generate() AND complete_text()) sets these from llama.cpp's own response, so ANY
    # caller can check them after a call regardless of whether that call's own return contract
    # treats truncation as a failure. last_stop_type == "limit" means the response was cut off by
    # hitting n_predict (self.max_tokens) before finishing naturally ("eos"); last_prompt_truncated
    # means the INPUT prompt itself exceeded the server's context window (n_ctx) and was cut --
    # a different, upstream failure mode. Both were previously silently discarded (only the
    # "content" field was read from the response).
    last_stop_type: str = ""
    last_prompt_truncated: bool = False
    # REQ-ARC-WMTE-5717: DEV-ONLY. When True (set by the agent ONLY on the stall/first-contact
    # re-induction path) AND CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED=1, induce() prepends the
    # game-agnostic exploration-playbook exemplars. Default False -> byte-identical induce prompt.
    include_playbook_exemplars: bool | str = False
    # REQ-ARC-FCP-5699-30: the raw completion text, captured on EVERY call regardless of
    # success/failure -- generate()'s failure path previously discarded `text` entirely once it
    # decided the required functions were missing, so there was no way to see WHAT the model
    # actually produced (reasoning-only? malformed code? nothing?) on a failed try, only THAT it
    # failed. Closes the diagnostic gap REQ-ARC-FCP-5699-23 through -29 all ran into without ever
    # inspecting.
    last_raw_completion: str = ""
    # LIVENESS WITNESS (2026-07-27, exp5866 finding 4). The scored ARC path had NO channel
    # at all for "did the generator actually answer": generate()/complete_text() return
    # (False, msg) on a dead or refusing server, every caller treats that as "no induction
    # this stall" and continues, and the message string is discarded by 4 of the 11 call
    # sites outright. So a run whose generator died at action 3 completed all 400 actions,
    # exited 0, and was recorded as an LLM-on measurement. The census
    # (results/outer_loop_arc_generator_failure_swallow_census_20260727.json) found the
    # harness-side `errors` counter is STRUCTURALLY dead -- 877 stat blocks, zero non-zero,
    # including all 8 cells where the generator provably died -- because it only counts
    # exceptions that PROPAGATE, and none do.
    #
    # These counters live on the PROPOSER because it is the single choke point all 11 call
    # sites funnel through; instrumenting the call sites individually would have to be
    # redone for every new caller and would miss exactly the ones that discard the message.
    # SERVER failures (unreachable / HTTP error / transport death) are counted separately
    # from CONTENT failures (the server answered, the answer was unusable) because only the
    # first is a liveness fact -- conflating them would let a healthy-but-unhelpful model
    # read as a dead generator and vice versa.
    n_completion_calls: int = 0
    n_completion_ok: int = 0
    n_server_failures: int = 0
    n_content_failures: int = 0
    server_failure_diagnostics: list = field(default_factory=list)
    last_generated_tokens: int = -1
    # DECLARED-VS-ACTUAL (2026-07-27 review finding 1). `n_ctx` above is what we INTEND to
    # launch with. These two record what a RUNNING server on our port actually reports, so
    # the liveness witness can publish an OBSERVED value instead of re-publishing our own
    # intent -- the exact gap that let the n_ctx fix be a silent no-op against a stale server.
    observed_server_n_ctx: Optional[int] = None
    reuse_n_ctx_check: str = "not_checked"
    reuse_refusals: list = field(default_factory=list)

    def _url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _note_server_failure(self, diagnostic: str) -> None:
        """Count + KEEP a server-side failure diagnostic (bounded, so a storm cannot grow
        without limit). This is the record the scored path never had."""
        self.n_server_failures += 1
        if len(self.server_failure_diagnostics) < 24:
            self.server_failure_diagnostics.append(diagnostic[:400])

    def liveness_witness(self) -> dict:
        """The generator-liveness primitives, in the SHAPE `scripts/arc_llm_on_liveness_lint.py`
        already recomputes from (`llm.responses`, `generator_healthy_after`), so a scored-path
        row can be audited by the SAME gate as a harness row rather than needing a second,
        differently-buggy checker."""
        healthy = self._healthy()
        # Only ask the server what it is if it is actually up; on a dead server /props costs
        # a 3s timeout per witness call and returns nothing useful anyway.
        observed_n_ctx = self.observed_n_ctx() if healthy else None
        observed_slots = self.observed_total_slots() if healthy else None
        return {
            "llm": {
                "calls": int(self.n_completion_calls),
                "responses": int(self.n_completion_ok),
                "errors": int(self.n_server_failures),
                "content_failures": int(self.n_content_failures),
            },
            "generator_healthy_after": bool(healthy),
            "generator_server_failure_diagnostics": list(self.server_failure_diagnostics),
            "generator_port": int(self.port),
            # OBSERVED, not declared (2026-07-27 review finding 1). This used to publish
            # `int(self.n_ctx)` -- our own INTENT -- so a run that reused a stale server with a
            # smaller pool reported the pool it wished it had. Reading /props makes the witness
            # a measurement of the server rather than an echo of the caller, which is the whole
            # point of a liveness witness. `generator_n_ctx_source` is published alongside so a
            # reader can tell an observation from the declared fallback rather than having to
            # assume; `declared_only` is exactly the state in which the number is NOT evidence.
            "generator_n_ctx": int(observed_n_ctx if observed_n_ctx is not None else self.n_ctx),
            "generator_n_ctx_declared": int(self.n_ctx),
            "generator_n_ctx_source": (
                "server_props_observed" if observed_n_ctx is not None else "declared_only"
            ),
            "generator_total_slots_observed": observed_slots,
            "generator_reuse_n_ctx_check": str(self.reuse_n_ctx_check),
            "generator_reuse_refusals": list(self.reuse_refusals),
            "generator_max_tokens": int(self.max_tokens),
        }

    def _record_completion_diagnostics(self, response: dict) -> None:
        self.last_stop_type = str(response.get("stop_type") or "")
        self.last_prompt_truncated = bool(response.get("truncated"))
        self.last_raw_completion = str(response.get("content") or "")
        # How many tokens the server ACTUALLY generated. Load-bearing for telling the two
        # "stop_type == limit" cases apart -- see _limit_diagnostic().
        timings = response.get("timings")
        got = (timings or {}).get("predicted_n") if isinstance(timings, dict) else None
        self.last_generated_tokens = int(got) if isinstance(got, int) else -1

    def _limit_diagnostic(self) -> str:
        """Distinguish the TWO different faults that both report stop_type == 'limit'.

        The old message said "HIT n_predict=<max_tokens> OUTPUT LIMIT" for both, which is
        actively misleading in the second case and is why exp5866's mode C went unnoticed:

          * INTENDED BUDGET LIMIT -- the model generated the full max_tokens we asked for
            and was still going. The fix is a bigger max_tokens.
          * SHARED-POOL TRUNCATION -- the model was cut off FAR short of max_tokens because
            the prompt had already consumed most of the server's shared context pool, so
            only the leftover cells were available to generate into. The fix is a bigger
            -c / CARNOT_ARC_INDUCE_N_CTX, and a bigger max_tokens would make it WORSE.
            Measured shape: a 15754-token prompt in a 16384 pool left 630 cells, produced
            2133 characters, and returned HTTP 200 -- indistinguishable, before this
            change, from a healthy-but-terse model.
        """
        got = self.last_generated_tokens
        if isinstance(got, int) and 0 <= got < self.max_tokens - 8:
            return (
                f" [TRUNCATED BY SHARED CONTEXT POOL: generated only {got} of the "
                f"{self.max_tokens}-token budget in an n_ctx={self.n_ctx} pool -- the prompt "
                f"consumed the rest. RAISE -c / CARNOT_ARC_INDUCE_N_CTX; raising max_tokens "
                f"would make this worse]"
            )
        return f" [HIT n_predict={self.max_tokens} OUTPUT LIMIT before completing]"

    def _chat_complete_request(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[list],
    ) -> tuple[dict, str]:
        """POST one user turn to the OpenAI-compatible /v1/chat/completions endpoint (the server
        applies the GGUF's OWN embedded chat template -- the turn delimiters Qwen3.6/ThinkingCap
        need) and normalize the OpenAI-shaped reply back into llama.cpp's native
        {content, stop_type, truncated} shape. Returns (normalized_response, extraction_text):

          * normalized_response["content"] -> the FULL generated text. Some llama.cpp builds
            extract the <think> reasoning into a separate `reasoning_content` field and strip it
            from `content`; we fold it back in (wrapped in <think></think>) so
            _record_completion_diagnostics, reason_engaged detection, and max_raw_completion_len
            stay faithful to EVERYTHING the model emitted (reasoning + answer).
          * extraction_text -> the FINAL answer only (reasoning stripped when the build split it),
            so _extract_python cannot accidentally grab a ```python block written INSIDE the
            model's reasoning trace.

        Raises on a network/transport error; the caller converts that to its failure tuple,
        exactly like the raw /completion path."""
        import json as _json
        import urllib.request

        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "cache_prompt": True,
        }
        if stop:
            payload["stop"] = list(stop)
        req = urllib.request.Request(
            self._url() + "/v1/chat/completions",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = _json.load(r)
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        final = str(msg.get("content") or "")
        reasoning = str(msg.get("reasoning_content") or "")
        # OpenAI finish_reason 'length' == hit max_tokens == llama.cpp stop_type 'limit' (overran).
        stop_type = "limit" if choice.get("finish_reason") == "length" else "eos"
        full = f"<think>\n{reasoning}\n</think>\n{final}" if reasoning else final
        # HOW MANY TOKENS THE SERVER ACTUALLY GENERATED -- normalized into llama.cpp's native
        # `timings.predicted_n` shape. WITHOUT THIS the mode-C detector is STRUCTURALLY DEAD on
        # this endpoint (found 2026-07-27, adversarial review): the normalized dict carried no
        # `timings` key at all, so `_record_completion_diagnostics` set `last_generated_tokens =
        # -1`, and `_limit_diagnostic()`'s pool-truncation branch (`0 <= got < max_tokens - 8`)
        # could NEVER be true when use_chat_template=True -- it always fell through to the
        # actively-misleading "HIT n_predict OUTPUT LIMIT" message, whose prescription (raise
        # max_tokens) is the OPPOSITE of the correct one (raise n_ctx). That is the same
        # dead-channel class the diagnostic was added to fix, reintroduced on the sibling
        # endpoint. Two sources because llama.cpp builds differ: newer ones attach a native
        # top-level `timings`, all of them fill OpenAI `usage.completion_tokens`.
        timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else None
        predicted_n = (timings or {}).get("predicted_n")
        if not isinstance(predicted_n, int):
            usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
            ct = usage.get("completion_tokens")
            predicted_n = ct if isinstance(ct, int) else None
        normalized: dict[str, Any] = {
            "content": full,
            "stop_type": stop_type,
            "truncated": bool(raw.get("truncated")),
        }
        if isinstance(predicted_n, int):
            normalized["timings"] = {"predicted_n": predicted_n}
        return normalized, final

    def _healthy(self) -> bool:
        import urllib.request

        try:
            with urllib.request.urlopen(self._url() + "/health", timeout=2) as r:
                return b"ok" in r.read()
        except Exception:
            return False

    def server_props(self) -> dict:
        """Read the RUNNING server's own /props. This is the only channel that reports what
        the server was actually LAUNCHED with; every other field on this object reports what
        we INTENDED. Returns {} when /props is unreachable or unparseable (never raises)."""
        import json as _json
        import urllib.request

        try:
            with urllib.request.urlopen(self._url() + "/props", timeout=3) as r:
                raw = _json.load(r)
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        return raw

    def observed_n_ctx(self) -> Optional[int]:
        """The n_ctx the RUNNING server reports, or None if /props is unreachable.

        llama.cpp reports the context pool under default_generation_settings.n_ctx; some
        builds also surface a bare top-level n_ctx. Both are read so a build difference
        cannot silently degrade this into 'unobservable' (which would re-open exactly the
        declared-vs-actual gap this method exists to close)."""
        props = self.server_props()
        if not props:
            return None
        gen = props.get("default_generation_settings")
        for candidate in (
            (gen or {}).get("n_ctx") if isinstance(gen, dict) else None,
            props.get("n_ctx"),
        ):
            if isinstance(candidate, int) and candidate > 0:
                return int(candidate)
        return None

    def observed_total_slots(self) -> Optional[int]:
        props = self.server_props()
        slots = props.get("total_slots") if props else None
        return int(slots) if isinstance(slots, int) and slots > 0 else None

    def _reusable(self) -> bool:
        """Is an ALREADY-RUNNING server on our port usable as OUR configured generator?

        THE HOLE THIS CLOSES (2026-07-27 review finding 1). `_ensure_server` used to return
        True on a bare /health check. /health only says "a llama-server is listening"; it
        says NOTHING about the context pool that server was launched with. So the 2026-07-27
        n_ctx 16384 -> 81920 fix was a SILENT NO-OP against any long-lived server already on
        the port: verified live on the dev box, port 8919 (this class's DEFAULT port) was
        serving n_ctx=16384 from a launch the previous evening, `_ensure_server()` returned
        True without launching anything, and `liveness_witness()` reported 81920 -- the
        INTENDED value read off `self.n_ctx`. A run in that state self-certifies as fixed
        while running on the faulty pool, which is the same declared-vs-actual silent
        degradation the fix was chartered to eliminate, one layer up.

        Refusing to reuse (rather than adopting the observed value) is deliberate: adopting
        would make the process quietly run a configuration nobody asked for, and the
        admission arithmetic (K_concurrent * (prompt + max_tokens) <= n_ctx) that the
        shipped default was sized against would no longer hold.

        A server whose /props cannot be read is reused with a WARNING record rather than
        refused, so a llama.cpp build that does not serve /props does not brick the path."""
        observed = self.observed_n_ctx()
        if observed is None:
            self.reuse_n_ctx_check = "unobserved_props_unreachable"
            return True
        self.observed_server_n_ctx = observed
        if observed >= int(self.n_ctx):
            # >= not ==: a LARGER pool than we asked for still satisfies our admission
            # arithmetic. Only a SMALLER pool can silently truncate/500 under concurrency.
            self.reuse_n_ctx_check = "match" if observed == int(self.n_ctx) else "larger_ok"
            return True
        self.reuse_n_ctx_check = f"refused_smaller_pool observed={observed} want={self.n_ctx}"
        return False

    def _ensure_server(self) -> bool:
        if self._healthy():
            if self._reusable():
                return True  # reuse an already-running server (loaded model)
            # A live server on our port has a SMALLER context pool than this proposer needs.
            # Do not adopt it and do not fight it for the port -- move to a fresh port and
            # launch our own, so a stale/foreign server cannot silently degrade this run.
            # Recorded on its OWN channel, NOT via _note_server_failure. A port relaunch is a
            # configuration event, not a generator failure: routing it into n_server_failures
            # would flip llm_on_row_valid to False for a run whose generator then worked
            # perfectly, i.e. over-firing the very gate that has to stay trustworthy.
            self.reuse_refusals.append(
                f"port {self.port} already serves n_ctx={self.observed_server_n_ctx} "
                f"< required {self.n_ctx}; relaunched on a fresh port "
                "(reusing it would silently restore the concurrency fault)"
            )
            self.port = _free_port()
        path = self.model_path or _resolve_gguf(
            self.repo_substr
        )  # explicit path (Kaggle bundle) else cache
        # Resolve the server + launch env at LAUNCH time so the opt-in 3090 guard sees current GPU state
        # (CARNOT_ARC_GENERATOR_CUDA_GPU=<idx> -> CUDA build pinned to that card iff it has headroom).
        server, launch_env = _generator_server_and_env()
        if not path or not server.exists():
            return False  # GPU enforcement: no CPU fallback
        args = [
            str(server),
            "-m",
            path,
            "-ngl",
            str(
                self.n_gpu_layers
            ),  # 999=all-GPU (default); lower spills weights to system RAM (frees VRAM)
            "-c",
            str(self.n_ctx),
            "--port",
            str(self.port),
            "--host",
            "127.0.0.1",
        ]
        if self.mtp:  # native llama.cpp MTP speculative decoding (self-draft)
            args += ["--spec-type", "draft-mtp", "--model-draft", path]
        if self.kv_quant:  # 8-bit KV cache doubles usable context, near-lossless
            args += ["--cache-type-k", self.kv_quant, "--cache-type-v", self.kv_quant]
        if self.extra_server_args:  # e.g. ("-fit", "off") -- see field docstring
            args += list(self.extra_server_args)
        # env=launch_env: None inherits the ambient env (legacy iGPU path); a dict pins CUDA_VISIBLE_DEVICES.
        #
        # STDERR IS CAPTURED TO A FILE, NOT DISCARDED (2026-07-27). It used to go to DEVNULL, and
        # that single choice is why the K>=2 concurrency fault stayed invisible for months: the
        # server DIAGNOSES itself on stderr and we threw the diagnosis away.
        #
        # WHAT WE WERE DISCARDING, concretely. llama.cpp's decode-failure handler
        # (tools/server/server-context.cpp:3200-3230) does NOT raise -- it checks the RETURN CODE of
        # llama_decode() and logs `SRV_ERR("%s i = %d, n_batch = %d, ret = %d")`. That `ret` is the
        # discriminator between our failure modes and nothing else distinguishes them:
        #     ret == 1  -> "Context size has been exceeded."  (mode A: pool exhaustion, survivable)
        #     ret == -1 -> "Invalid input batch."
        #     ret <  -1 -> "Compute error."
        #     ret == 2  -> explicitly UNHANDLED upstream (`// TODO: handle ret == 2 (abort)`)
        # A hard GGML_ASSERT abort (mode B, the server DIES) also prints only to stderr. So with
        # DEVNULL, mode A and mode B are indistinguishable from the client -- which is exactly the
        # state the 2026-07-27 review left open ("Mode B is UNRESOLVED at 81920; needs the server's
        # stderr captured").
        #
        # Note also that the graceful path is a DIFFERENT site: the per-request admission check at
        # :2704-2712 sends a 400 ("try increasing it") BEFORE decoding. It is per-request, so it
        # cannot catch the aggregate case where K requests each fit but jointly exhaust the shared
        # kv_unified pool -- that only fails later inside llama_decode, as a 500. Concurrency
        # escapes the graceful path by construction, and the 500 handler then errors EVERY
        # processing slot (`for (auto & slot : slots) ... send_error`), which is why we measure
        # 2/2 at K=2 and 4/4 at K=4 rather than a single victim.
        #
        # A FILE, NOT A PIPE, DELIBERATELY. subprocess.PIPE with no reader deadlocks the server the
        # moment the OS pipe buffer fills (~64KB) -- llama-server is chatty enough to hit that
        # during a long run, and the hang would look exactly like the fault we are diagnosing.
        # Writing to a file has no such backpressure. Best-effort: if the log cannot be opened we
        # fall back to DEVNULL rather than failing the launch, because losing diagnostics must
        # never cost us the generator itself.
        # Local imports: this module has NO module-level `import os` (every user imports it inside
        # its own function) and no `tempfile` at all. Ruff passed on the module-attribute version
        # anyway -- a green lint is not evidence the code runs, so this is imported where it is used
        # and the path below is exercised by a real test rather than trusted.
        import os
        import tempfile

        self._stderr_log_path = None
        _err_sink = subprocess.DEVNULL
        try:
            log_dir = (
                Path(os.environ.get("CARNOT_ARC_SERVER_LOG_DIR", tempfile.gettempdir()))
                / "carnot_llama_server_logs"
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            self._stderr_log_path = log_dir / f"llama_server_p{self.port}_{int(time.time())}.log"
            _err_sink = open(self._stderr_log_path, "ab", buffering=0)  # noqa: SIM115
        except OSError:
            self._stderr_log_path = None
            _err_sink = subprocess.DEVNULL
        self._proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=_err_sink, env=launch_env
        )
        load_wait_attempts = max(90, int(self.timeout / 2))  # large full-precision models (e.g.
        # a 62GB BF16 GGUF) can take far longer than the 180s the fixed 90-attempt budget allows
        for _ in range(load_wait_attempts):
            if self._healthy():
                return True
            time.sleep(2)
        return False

    def generate(
        self,
        prompt: str,
        required: tuple = ("engine", "is_level_complete"),
        validate=None,
        tries: int = 3,
        *,
        codeonly_eligible: bool = False,
    ) -> tuple[bool, str]:
        """Generic GPU-server completion: returns (True, code) where `code` contains every
        `def <name>` in `required`, PARSES, and (if `validate` is given) passes the
        runtime check `validate(code) -> bool`. Retries on the iGPU (fast). This is the
        gap-filler entry point: the LLM writes a FOCUSED component (a goal_distance
        heuristic, a state_key, a verifier invariant) — not a full solver. `validate` lets
        the caller reject runtime-buggy code (e.g. a heuristic that returns None)."""
        import ast
        import json as _json
        import os
        import urllib.request

        self.n_completion_calls += 1
        if not self._ensure_server():
            msg = (
                f"GPU llama-server failed for {self.repo_substr}; SOTA models "
                "must run on GPU (no CPU fallback)"
            )
            self._note_server_failure(msg)
            return False, msg
        # L2 induction truncation fix (proto_l2_code_only_prefix, 2026-06-25). Scope to the INDUCE
        # call ONLY (codeonly_eligible, set True solely by induce->_gen_to_file). refactor() also
        # routes through _gen_to_file with the same `required` tuple, but it is a REASONING task
        # (debug BEFORE/PREDICTED/OBSERVED mismatches) that must NOT be told to "skip all reasoning";
        # keying on codeonly_eligible (set True ONLY by the induce paths) keeps refactor and
        # gap-fillers untouched. It is NOT keyed on `required` because the focused split-induce calls
        # request just ("engine",) or ("is_level_complete",) yet still need the code-only path.
        # When on: prepend the code-only directive + an opened fence, and add a stop-sequence on the
        # closing fence so the model emits ONLY the code (no win-state CoT). DEFAULT ON (2026-06-25
        # operator directive): a strict improvement (emits valid code in ~10s where the unpatched
        # path truncates to 0 code at 450s); opt out with CARNOT_ARC_CODEONLY_INDUCE=0.
        _codeonly = (os.environ.get("CARNOT_ARC_CODEONLY_INDUCE", "1") != "0") and codeonly_eligible
        _stop_seq = ["```"] if _codeonly else None
        if _codeonly:
            prompt = _L2_CODEONLY_DIRECTIVE + prompt + "\n```python\n"
        elif self.no_think_prefix:  # suppress hybrid-thinking CoT so the model emits code directly
            prompt = self.no_think_prefix + prompt
        last = ""
        for attempt in range(tries):
            _payload = {
                "prompt": prompt,
                "n_predict": self.max_tokens,
                "temperature": 0.2 + 0.1 * attempt,
                "cache_prompt": True,
            }
            if _stop_seq:
                _payload["stop"] = _stop_seq
            body = _json.dumps(_payload).encode()
            try:
                if self.use_chat_template:
                    # OpenAI /v1/chat/completions -> server applies the GGUF's embedded chat template
                    # (Qwen3.6/ThinkingCap need the assistant-turn structure; REQ-ARC-WMTE-5725).
                    _response, text = self._chat_complete_request(
                        prompt,
                        max_tokens=self.max_tokens,
                        temperature=_payload["temperature"],
                        stop=_stop_seq,
                    )
                else:
                    req = urllib.request.Request(
                        self._url() + "/completion",
                        data=body,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        _response = _json.load(r)
                    text = _response.get("content", "")
            except Exception as e:
                msg = f"local gguf (GPU server) failed: {_describe_http_failure(e)}"[:400]
                self._note_server_failure(msg)
                return False, msg
            self._record_completion_diagnostics(_response)  # MANDATORY truncation detection
            code = _extract_python(text)
            if not code and _codeonly:
                # the stop-sequence consumed the closing fence and the opener was in the prompt, so
                # the raw completion IS the code block body.
                code = text.strip()
            if not code or any(f"def {fn}" not in code for fn in required):
                _diag = ""
                if self.last_stop_type == "limit":
                    _diag += self._limit_diagnostic()
                if self.last_prompt_truncated:
                    _diag += " [PROMPT TRUNCATED -- exceeded server context window]"
                last = f"missing {required} in output{_diag}"
                continue
            try:
                ast.parse(code)  # never use code that doesn't parse
            except SyntaxError as se:
                last = f"syntax error line {se.lineno}: {se.msg}"
                continue
            if validate is not None:
                try:
                    if not validate(code):
                        last = "failed runtime validation (e.g. returned non-number)"
                        continue
                except Exception as ve:
                    last = f"runtime check raised: {ve!r}"[:120]
                    continue
            self.n_completion_ok += 1
            return True, code
        # CONTENT failure, not a liveness failure: the server answered every try, the
        # answers were unusable. Counted separately so a terse-but-alive model can never
        # read as a dead generator (and vice versa) in the liveness witness.
        self.n_content_failures += 1
        return False, f"local model code unusable after {tries} tries ({last})"

    def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        stop: Optional[list] = None,
    ) -> tuple[bool, str]:
        """Raw free-text/JSON completion (NOT the code-extraction path of generate()).

        WHY a separate method: generate() runs `_extract_python` + an `ast.parse` gate +
        a `def <name>` presence check, which is exactly right for inducing a world-model
        engine but WRONG for a short reasoning answer (e.g. "which archived cell is most
        promising to explore from -> reply with one integer"). complete_text returns the
        server's raw `content` string with no code gating, so callers that want JSON/an
        index/a short rationale get it directly. Reuses the SAME warm GPU server as
        generate() via _ensure_server()/_url() (no second llama-server, no CPU fallback).

        Returns (ok, text). ok=False (with a diagnostic string) when the GPU server is
        unavailable or the request errors — the caller is expected to fall back to its
        own heuristic rather than fabricate, per the no-silent-degradation discipline.
        """
        import json as _json
        import urllib.request

        self.n_completion_calls += 1
        if not self._ensure_server():
            msg = (
                f"GPU llama-server failed for {self.repo_substr}; SOTA models "
                "must run on GPU (no CPU fallback)"
            )
            self._note_server_failure(msg)
            return False, msg
        full_prompt = (self.no_think_prefix + prompt) if self.no_think_prefix else prompt
        payload = {
            "prompt": full_prompt,
            "n_predict": int(max_tokens or self.max_tokens),
            "temperature": float(temperature),
            "cache_prompt": True,
        }
        if stop:
            payload["stop"] = list(stop)
        body = _json.dumps(payload).encode()
        try:
            if self.use_chat_template:
                # OpenAI /v1/chat/completions applies the GGUF's embedded chat template; the
                # normalized "content" folds any split-out reasoning back in so callers/smoke
                # tests can see the <think> trace (REQ-ARC-WMTE-5725).
                _response, _ = self._chat_complete_request(
                    full_prompt,
                    max_tokens=int(max_tokens or self.max_tokens),
                    temperature=temperature,
                    stop=stop,
                )
            else:
                req = urllib.request.Request(
                    self._url() + "/completion",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    _response = _json.load(r)
        except Exception as e:
            msg = f"local gguf (GPU server) failed: {_describe_http_failure(e)}"[:400]
            self._note_server_failure(msg)
            return False, msg
        self._record_completion_diagnostics(_response)  # MANDATORY truncation detection
        self.n_completion_ok += 1
        _content = str(_response.get("content", ""))
        if not _content.strip():
            # HTTP 200 WITH NOTHING IN IT. `n_completion_ok` deliberately still counts this --
            # it is a liveness fact (the server answered), and conflating "answered emptily" with
            # "did not answer" is the exact confusion the server/content split exists to prevent.
            # But counting it ONLY as a success would make `responses > 0` read as evidence of
            # usable output when there was none, so it is ALSO recorded as a content failure.
            # Found 2026-07-27 (adversarial review): before this, an alive server returning empty
            # strings for every call produced calls=N / responses=N / errors=0 / content_failures=0
            # -- a perfectly healthy-looking witness for a run that induced nothing.
            self.n_content_failures += 1
        return True, _content

    def _gen_to_file(
        self, game: str, prompt: str, *, codeonly_eligible: bool = False
    ) -> tuple[bool, str]:
        (E3_DIR / game).mkdir(parents=True, exist_ok=True)
        ok, code = self.generate(
            prompt,
            ("engine", "is_level_complete"),
            tries=self.tries,
            codeonly_eligible=codeonly_eligible,
        )
        if ok:
            (E3_DIR / game / "world_model.py").write_text(code)
            return True, "local gguf (GPU server) wrote world_model.py"
        return False, code

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None

    def _write_world_model(self, game: str, code: str, note: str = "") -> tuple[bool, str]:
        (E3_DIR / game).mkdir(parents=True, exist_ok=True)
        (E3_DIR / game / "world_model.py").write_text(code)
        msg = "local gguf (GPU server) wrote world_model.py"
        return True, (f"{msg} ({note})" if note else msg)

    def _goal_only_prompt(
        self, game: str, previous_level_complete_grid: Optional[np.ndarray]
    ) -> str:
        """A FOCUSED is_level_complete-only prompt with the WIN STATE exemplar front-and-centre, so
        the model spends its whole budget on the win condition (not the engine)."""
        win = ""
        if previous_level_complete_grid is not None:
            win = (
                "The level is COMPLETE at this WIN STATE grid (is_level_complete must return True "
                "here, and False elsewhere):\n"
                + to_ascii(np.asarray(previous_level_complete_grid))
                + "\n"
            )
        return (
            f"You are inducing ONLY the win condition for the ARC-AGI-3 game '{game}'.\n"
            + win
            + "Write ONLY `def is_level_complete(grid):` returning True iff `grid` is a level-complete "
            "/ win state, else False. numpy + stdlib only; pure and deterministic. Prefer a SIMPLE "
            "GENERAL rule over an exact full-grid match.\n\n"
            "Return ONLY one ```python code block defining is_level_complete.\n```python\n"
        )

    def _combine_world_model(self, engine_code: str, goal_code: str) -> str:
        """Concatenate a focused engine block and a focused is_level_complete block into one world
        model. Both pieces already parse individually (generate validates each); duplicate imports
        are valid Python, but we verify the concatenation parses and fall back to a raw join."""
        import ast

        combined = (
            "import numpy as np\n\n" + engine_code.strip() + "\n\n" + goal_code.strip() + "\n"
        )
        try:
            ast.parse(combined)
            return combined
        except SyntaxError:
            return engine_code.strip() + "\n\n" + goal_code.strip() + "\n"

    def induce(
        self,
        game: str,
        trans: list[Transition],
        cell: int,
        *,
        previous_level_complete_grid: Optional[np.ndarray] = None,
    ) -> tuple[bool, str]:
        base = induce_prompt(
            game,
            trans,
            cell,
            previous_level_complete_grid=previous_level_complete_grid,
            k=_induce_transitions_k(),
            include_playbook_exemplars=self.include_playbook_exemplars,
        )
        # Happy path: one combined engine+is_level_complete induction (code-only eligible: it is the
        # win-state-exemplar prompt whose CoT caused the truncation; refactor stays reasoning).
        ok, code = self.generate(
            base
            + "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n```python\n",
            ("engine", "is_level_complete"),
            tries=self.tries,
            codeonly_eligible=True,
        )
        if ok:
            return self._write_world_model(game, code)
        # FALLBACK (proto_l2_fix_finder, 2026-06-25): on complex real L2 prompts the combined call
        # commonly fails because the model rambles its analysis INTO engine() comments, exhausts the
        # token budget, and never writes is_level_complete. Induce each function in its OWN focused
        # call so the engine ramble cannot starve the goal -- the focused goal call is valid in ~3.5s
        # where the combined call fails (a budget bump does NOT help; the model just rambles more).
        ok_e, eng = self.generate(
            base
            + "\n\nReturn ONLY one ```python code block defining engine(grid, action, data).\n```python\n",
            ("engine",),
            tries=self.tries,
            codeonly_eligible=True,
        )
        if not ok_e:
            return False, f"split induce: engine failed: {str(eng)[:150]}"
        ok_g, goal = self.generate(
            self._goal_only_prompt(game, previous_level_complete_grid),
            ("is_level_complete",),
            tries=self.tries,
            codeonly_eligible=True,
        )
        if not ok_g:
            return False, f"split induce: goal failed: {str(goal)[:150]}"
        return self._write_world_model(
            game, self._combine_world_model(eng, goal), note="split induce: engine + focused goal"
        )

    def refactor(self, game: str, vr: VerifyResult) -> tuple[bool, str]:
        # NOT codeonly_eligible: refactor asks the model to reason about BEFORE/PREDICTED/OBSERVED
        # mismatches; the code-only "skip all reasoning" directive would degrade exactly that.
        return self._gen_to_file(
            game,
            refactor_prompt(game, vr)
            + "\n\nReturn ONLY the corrected ```python file.\n```python\n",
            codeonly_eligible=False,
        )

    def induce_programmatic_experts(
        self,
        *,
        game: str,
        transitions: Sequence[Transition],
        heldout_transitions: Sequence[Transition] | None = None,
        cell: int = 1,
        max_experts: int = 8,
    ) -> list[dict[str, Any]]:
        """REQ-ARC-WMTE-4677: ask the local GGUF for small serializable expert rules."""

        examples = _transitions_block(list(transitions), k=min(6, max(1, len(transitions))))
        prompt = f"""You are proposing SMALL programmatic object-level experts for ARC-AGI-3 game '{game}'.

Each expert must be a SERIALIZABLE dictionary with:
  name, object_class, kind='color_rewrite', action, from_color, to_color.

Use only the observed transitions. Prefer simple object/color rewrite factors that can be
held-out replay verified. Do not include brittle grid-sized programs. Return a Python code
block defining:

def expert_rules():
    return [{{...}}, ...]

Limit to {int(max_experts)} experts. Actions are integer ARC actions; click data is in pixel
coords where one logical cell is {int(cell)} pixels.

OBSERVED PREFIX TRANSITIONS:
{examples}
"""
        ok, code = self.generate(
            prompt + "\n```python\n",
            required=("expert_rules",),
            validate=None,
            tries=1,
        )
        if not ok:
            return []
        namespace: dict[str, Any] = {}
        try:
            exec(code, {"np": np, "numpy": np}, namespace)
            rows = namespace["expert_rules"]()
        except Exception:
            return []
        return [dict(row) for row in list(rows or []) if isinstance(row, Mapping)]


def _extract_python(text: str) -> str:
    """Pull the first python code block (or the whole text if it looks like code)."""
    if "```python" in text:
        text = text.split("```python", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# plan in the verified model, execute in reality, halt on divergence
# ---------------------------------------------------------------------------


def _model_candidates(grid: np.ndarray) -> list[dict]:
    """Action candidates to try when planning INSIDE the induced model (no env): the 5
    directional/confirm keyboard actions + a click on each detected object (salience-
    ordered). Pure-grid, so it works on the engine's predicted grids."""
    from carnot.agentic.arc_graph_explore import _components_detailed

    cands = [{"action": a, "data": None} for a in (1, 2, 3, 4, 5)]
    comps = _components_detailed(grid)
    if comps:
        from collections import Counter

        cc = Counter(int(v) for v in grid.flatten().tolist())
        comps.sort(key=lambda c: c[2] * (1.0 + 1.0 / (1 + cc.get(c[3], 0))), reverse=True)
        # Defensive unpack (`*_` absorbs any extra trailing fields): `_components_detailed` was widened
        # from a 4-tuple (cy, cx, area, color) to a 5-tuple (+ is_grid_fallback) in the
        # GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS fix (commit 2f0760307), which updated the
        # arc_graph_explore consumer defensively but MISSED this one -- a rigid `cy, cx, _a, _c` unpack then
        # crashed plan_in_model on ANY grid with components (e.g. tu93's 65), silently disabling the entire
        # world-model planning tier for those games. `*_` handles both the 4-tuple (test doubles monkeypatch
        # the old shape) and the 5-tuple (real) forms. (REQ-ARC-WMTE-5841 regression fix.)
        for cy, cx, _a, _c, *_ in comps[:32]:
            cands.append({"action": 6, "data": {"x": int(cx), "y": int(cy)}})
    return cands


def plan_in_model(
    engine,
    is_level_complete,
    start_grid: np.ndarray,
    *,
    max_nodes: int = 20000,
    max_depth: int = 40,
    goal_energy=None,
    diagnostics: Optional[dict] = None,
) -> Optional[list]:
    """BFS a path to an is_level_complete state ENTIRELY INSIDE the induced model
    (engine is pure: grid,action,data -> grid; no environment). Returns the action
    sequence [{"action","data"}] that the model believes reaches a win, or None. This
    is the harness-friendly planner: the agent computes the plan with zero real actions,
    then executes it in the real env (few real actions = the EFFICIENCY win), halting if
    reality diverges from the model.

    GOAL-ENERGY (2026-06-23, closes GAP-ARCH-GOAL-NOT-VERIFIED): when ``goal_energy`` (grid -> float,
    LOWER = closer to the induced win) is supplied, the search is BEST-FIRST by goal_energy -- it DESCENDS
    toward the goal predicate instead of exploring blind breadth-first, so it reaches the win in FEWER
    nodes (the action-efficiency win). ``goal_energy`` is induced per-game from the agent's OWN observed
    win/non-win states (``arc_agi3_goal_induction.induce_goal_energy``), NOT a frozen transfer. Backward-
    compatible: ``goal_energy=None`` keeps the exact original FIFO BFS. The terminal check stays
    ``is_level_complete`` (the energy only orders the frontier); an ablation control is mandatory.

    DIAGNOSTICS (REQ-ARC-FCP-5699-15, closes the "trust gate passes but no plan found" question
    REQ-ARC-FCP-5699-14 left open): when ``diagnostics`` (a caller-owned dict) is supplied, this
    populates it with ``is_level_complete_was_none`` (bool), ``nodes_expanded`` (int),
    ``termination_reason`` (one of ``"is_level_complete_none"`` / ``"plan_found"`` /
    ``"max_nodes_reached"`` / ``"queue_exhausted"``), and ``used_goal_energy_search`` (bool) before
    returning -- so a caller can tell WHY an empty return happened without re-deriving the search.
    REQ-ARC-FCP-5699-18 adds ``initial_goal_energy``/``min_goal_energy_observed`` (floats, only when
    ``used_goal_energy_search`` is True): the goal-energy value at ``start_grid`` and the lowest
    value seen across every state the search visited -- lets a caller tell whether a failed search
    got structurally CLOSE to the induced goal (min << initial, "coherent but ran out of budget")
    or never moved toward it at all (min ~= initial, "the model's rollout doesn't structurally
    connect toward the goal"). Backward-compatible: ``diagnostics=None`` (the default) changes
    nothing about the search or the return value."""
    if is_level_complete is None:
        if diagnostics is not None:
            diagnostics["is_level_complete_was_none"] = True
            diagnostics["nodes_expanded"] = 0
            diagnostics["termination_reason"] = "is_level_complete_none"
        return None
    start = np.asarray(start_grid)
    seen = {to_ascii(start)}
    nodes = 0

    if goal_energy is not None:
        # BEST-FIRST by goal-energy: descend toward the induced goal predicate.
        import heapq
        import itertools

        def _h(g):
            try:
                return float(goal_energy(g))
            except Exception:
                return 0.0

        counter = itertools.count()
        initial_energy = _h(start)
        min_energy = initial_energy
        heap = [(initial_energy, next(counter), start, [])]
        while heap and nodes < max_nodes:
            _, _, grid, path = heapq.heappop(heap)
            if len(path) >= max_depth:
                continue
            for c in _model_candidates(grid):
                try:
                    ng = np.asarray(engine(grid.copy(), c["action"], c["data"]))
                except Exception:
                    continue
                nodes += 1
                if ng.shape != start.shape:
                    continue
                key = to_ascii(ng)
                if key in seen:
                    continue
                seen.add(key)
                npath = path + [c]
                try:
                    if bool(is_level_complete(ng)):
                        if diagnostics is not None:
                            diagnostics["is_level_complete_was_none"] = False
                            diagnostics["nodes_expanded"] = nodes
                            diagnostics["termination_reason"] = "plan_found"
                            diagnostics["used_goal_energy_search"] = True
                            diagnostics["initial_goal_energy"] = initial_energy
                            diagnostics["min_goal_energy_observed"] = min_energy
                        return npath
                except Exception:
                    pass
                ng_energy = _h(ng)
                if ng_energy < min_energy:
                    min_energy = ng_energy
                heapq.heappush(heap, (ng_energy, next(counter), ng, npath))
        if diagnostics is not None:
            diagnostics["is_level_complete_was_none"] = False
            diagnostics["nodes_expanded"] = nodes
            diagnostics["used_goal_energy_search"] = True
            diagnostics["initial_goal_energy"] = initial_energy
            diagnostics["min_goal_energy_observed"] = min_energy
            diagnostics["termination_reason"] = (
                "max_nodes_reached" if nodes >= max_nodes else "queue_exhausted"
            )
        return None

    # ---- original blind FIFO BFS (goal_energy=None; unchanged) ----
    from collections import deque

    q = deque([(start, [])])
    while q and nodes < max_nodes:
        grid, path = q.popleft()
        if len(path) >= max_depth:
            continue
        for c in _model_candidates(grid):
            try:
                ng = np.asarray(engine(grid.copy(), c["action"], c["data"]))
            except Exception:
                continue
            nodes += 1
            if ng.shape != start.shape:
                continue
            key = to_ascii(ng)
            if key in seen:
                continue
            seen.add(key)
            npath = path + [c]
            try:
                if bool(is_level_complete(ng)):
                    if diagnostics is not None:
                        diagnostics["is_level_complete_was_none"] = False
                        diagnostics["nodes_expanded"] = nodes
                        diagnostics["termination_reason"] = "plan_found"
                        diagnostics["used_goal_energy_search"] = False
                    return npath
            except Exception:
                pass
            q.append((ng, npath))
    if diagnostics is not None:
        diagnostics["is_level_complete_was_none"] = False
        diagnostics["nodes_expanded"] = nodes
        diagnostics["used_goal_energy_search"] = False
        diagnostics["termination_reason"] = (
            "max_nodes_reached" if nodes >= max_nodes else "queue_exhausted"
        )
    return None


def plan_and_execute(
    game: str,
    engine,
    is_level_complete,
    *,
    warmup: bool = False,
    max_plan: int = 200,
    max_depth: int = 40,
) -> dict:
    """BFS to an is_level_complete state INSIDE the induced model, then execute the plan
    in the REAL env step-by-step, halting the instant predicted != observed (the
    verifier-grounded safety the paper emphasizes). Returns an outcome dict."""
    from collections import deque
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
    from carnot.agentic.arc_graph_explore import rich_action_candidates, _warm
    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, warmup)
    cell = detect_cell(grid_of(f))
    start = to_logical(grid_of(f), cell)
    start_level = _levels_completed(f)

    # plan inside the model
    seen = {to_ascii(start)}
    frontier = deque([(start, [])])
    plan = None
    expansions = 0
    while frontier and expansions < max_plan and plan is None:
        g, path = frontier.popleft()
        if len(path) >= max_depth:
            continue
        for c in rich_action_candidates(f)[:12]:  # candidate actions at logical state
            try:
                ng = np.asarray(engine(g.copy(), int(c.action_id), c.data))
            except Exception:
                continue
            expansions += 1
            key = to_ascii(ng) if ng.ndim == 2 else None
            if key is None or key in seen:
                continue
            seen.add(key)
            npath = path + [{"action": int(c.action_id), "data": c.data}]
            if is_level_complete is not None:
                try:
                    if bool(is_level_complete(ng)):
                        plan = npath
                        break
                except Exception:
                    pass
            frontier.append((ng, npath))
    if plan is None:
        return {"game": game, "planned": False, "reason": "no plan to is_level_complete in model"}

    # execute in reality, halting on model/observation divergence
    f = _warm(env, warmup)
    gp = to_logical(grid_of(f), cell)
    for step in plan:
        pred = np.asarray(engine(gp.copy(), step["action"], step["data"]))
        nf = env.step(_game_action(GameAction, step["action"]), data=step["data"])
        if nf is None:
            return {"game": game, "planned": True, "executed": False, "reason": "env returned None"}
        obs = to_logical(grid_of(nf), cell)
        if _levels_completed(nf) > start_level:
            return {
                "game": game,
                "planned": True,
                "executed": True,
                "level_up": True,
                "plan_len": len(plan),
            }
        if pred.shape != obs.shape or not np.array_equal(pred, obs):
            return {
                "game": game,
                "planned": True,
                "executed": False,
                "divergence_step": step,
                "reason": "model prediction diverged from observation — halted (verifier-grounded)",
            }
        gp = obs
    return {
        "game": game,
        "planned": True,
        "executed": True,
        "level_up": False,
        "reason": "plan executed but no level-up — model goal predicate imperfect",
    }
