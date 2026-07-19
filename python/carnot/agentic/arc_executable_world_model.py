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
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
E3_DIR = REPO / "results" / "arc_e3"

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


class WorldModelVerifier:
    """Checks that an induced engine(grid, action, data) -> grid reproduces the real
    recorded transitions. This is the verification that makes the LLM accountable: a
    proposed model only earns trust by predicting transitions it was NOT hand-fit to.
    Returns mismatch artifacts (the failing transitions) for the refactor step."""

    def __init__(self, transitions: list[Transition]) -> None:
        self.transitions = transitions

    def score(
        self, engine: Callable[[np.ndarray, int, Optional[dict]], np.ndarray], max_mismatch: int = 8
    ) -> VerifyResult:
        n_correct, mism = 0, []
        cell_recalls: list[
            float
        ] = []  # per-CHANGED-transition fraction of changed cells predicted right
        for i, t in enumerate(self.transitions):
            try:
                pred = np.asarray(engine(t.grid.copy(), t.action, t.data))
            except Exception as e:  # a crashing engine fails the transition
                if len(mism) < max_mismatch:
                    mism.append({"i": i, "action": t.action, "error": repr(e)[:160]})
                continue
            # graded changed-cell recall (granularity-matched gate); only state-changing transitions count
            changed = not np.array_equal(t.grid, t.next_grid)
            if changed:
                if pred.shape == t.next_grid.shape:
                    m = t.grid != t.next_grid
                    cell_recalls.append(float((pred[m] == t.next_grid[m]).mean()))
                else:
                    cell_recalls.append(0.0)
            if pred.shape == t.next_grid.shape and np.array_equal(pred, t.next_grid):
                n_correct += 1
            elif len(mism) < max_mismatch:
                ok_shape = pred.shape == t.next_grid.shape
                # COMPACT mismatch (deltas, not full grids — fits a local model's context):
                # what the TRUE action did vs where the engine's prediction was wrong.
                mism.append(
                    {
                        "i": i,
                        "action": t.action,
                        "data": t.data,
                        "true_change": _delta(t.grid, t.next_grid),
                        "your_prediction_was_wrong_at": (
                            _delta(pred, t.next_grid) if ok_shape else f"wrong shape {pred.shape}"
                        ),
                    }
                )
        n = len(self.transitions)
        cell_recall = float(np.mean(cell_recalls)) if cell_recalls else 0.0
        return VerifyResult(n, n_correct, n_correct / max(1, n), mism, cell_recall=cell_recall)

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

    p = E3_DIR / game / "world_model.py"
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(f"arc_wm_{game}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return getattr(mod, "engine"), getattr(mod, "is_level_complete", None)


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
    `_rle_delta_compact`'s docstring for the measured before/after)."""
    changed = [t for t in trans if not np.array_equal(t.grid, t.next_grid)]
    noop = [t for t in trans if np.array_equal(t.grid, t.next_grid)]
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
) -> str:
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
{_transitions_block(trans, k, previous_level_complete_grid=previous_level_complete_grid)}
"""


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

# Qwen3.5-9B-MTP loads ~11.5GB on a 3090 (weights + MTP self-draft + q8 KV, validated 2026-06-19).
# Require headroom above that so the opt-in 3090 path NEVER binds a card a conductor training job is
# using -- this is the "yield-if-the-conductor-needs-it" guard.
_GENERATOR_CUDA_MIN_FREE_MB = 13000


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


def _generator_server_and_env() -> tuple[Path, Optional[dict]]:
    """Resolve the llama-server binary + launch env for the generator, evaluated at LAUNCH time so the
    3090 guard sees current GPU state.

    Priority:
      1. CARNOT_LLAMA_SERVER (Kaggle/live bundled CUDA binary) -- unchanged; inherits ambient env.
      2. OPT-IN CARNOT_ARC_GENERATOR_CUDA_GPU=<idx> -> the local CUDA build pinned to that 3090 via
         CUDA_VISIBLE_DEVICES, but ONLY if the card has >=_GENERATOR_CUDA_MIN_FREE_MB free. This is the
         operator-approved (2026-06-19) use of one idle 3090 for generator throughput now that the TRM
         run is retired; the free-memory guard yields to any conductor job already on the card.
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
        if idx >= 0 and _cuda_gpu_free_mb(idx) >= _GENERATOR_CUDA_MIN_FREE_MB:
            return cuda, dict(os.environ, CUDA_VISIBLE_DEVICES=str(idx))
        # guard tripped (card busy / unavailable / bad idx) -> fall through to the iGPU path,
        # never fight the conductor for the 3090.
    return (hip if hip.exists() else cuda), None


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
    n_ctx: int = 16384  # digit-dense grids tokenize ~1 char/token; 8192 overflowed
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

    def _url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _record_completion_diagnostics(self, response: dict) -> None:
        self.last_stop_type = str(response.get("stop_type") or "")
        self.last_prompt_truncated = bool(response.get("truncated"))
        self.last_raw_completion = str(response.get("content") or "")

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
        normalized = {
            "content": full,
            "stop_type": stop_type,
            "truncated": bool(raw.get("truncated")),
        }
        return normalized, final

    def _healthy(self) -> bool:
        import urllib.request

        try:
            with urllib.request.urlopen(self._url() + "/health", timeout=2) as r:
                return b"ok" in r.read()
        except Exception:
            return False

    def _ensure_server(self) -> bool:
        if self._healthy():
            return True  # reuse an already-running server (loaded model)
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
        self._proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=launch_env
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

        if not self._ensure_server():
            return False, (
                f"GPU llama-server failed for {self.repo_substr}; SOTA models "
                "must run on GPU (no CPU fallback)"
            )
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
                return False, f"local gguf (GPU server) failed: {e!r}"[:200]
            self._record_completion_diagnostics(_response)  # MANDATORY truncation detection
            code = _extract_python(text)
            if not code and _codeonly:
                # the stop-sequence consumed the closing fence and the opener was in the prompt, so
                # the raw completion IS the code block body.
                code = text.strip()
            if not code or any(f"def {fn}" not in code for fn in required):
                _diag = ""
                if self.last_stop_type == "limit":
                    _diag += f" [HIT n_predict={self.max_tokens} OUTPUT LIMIT before completing]"
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
            return True, code
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

        if not self._ensure_server():
            return False, (
                f"GPU llama-server failed for {self.repo_substr}; SOTA models "
                "must run on GPU (no CPU fallback)"
            )
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
        except Exception as e:  # pragma: no cover - network boundary
            return False, f"local gguf (GPU server) failed: {e!r}"[:200]
        self._record_completion_diagnostics(_response)  # MANDATORY truncation detection
        return True, str(_response.get("content", ""))

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
        for cy, cx, _a, _c in comps[:32]:
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
