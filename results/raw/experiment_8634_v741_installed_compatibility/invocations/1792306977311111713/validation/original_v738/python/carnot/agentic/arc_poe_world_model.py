"""PoE-World: weighted product-of-experts world model for ARC-AGI-3 (Liang et al. arXiv:2505.10819),
built 2026-06-28 (operator-directed: "PoE-World program-synthesis").

WHY THIS IS NOT A RERUN of the already-nulled ProductWorldModel (Failed-Experiment Rerun Discipline).
The repo already has ``ProductWorldModel`` (arc_executable_world_model.py:421) from exp4749, which NULLED
("dead/identity engine"). Its combination rule is **highest-trust-expert-wins-each-cell** -- a per-cell
MAX VOTE where each expert carries a fixed scalar trust. The single strongest applicable expert dictates
each changed cell; weak experts are ignored even when many of them agree.

PoE-World's actual mechanism (the paper) is different and is the genuine new lever here:
  1. WEIGHTED PRODUCT / CONSENSUS combination, not max-vote. For each cell, every applying expert casts a
     weighted vote for its predicted value (an expert that predicts no change votes for the current value).
     The output is the weighted-argmax (the MAP of the product of per-cell categorical distributions). Three
     weak agreeing experts can outvote one strong disagreeing one -- the consensus the max-vote cannot form.
  2. FITTED weights from HELD-OUT predictive accuracy (the paper fits weights by gradient/L-BFGS; we fit a
     lightweight per-expert held-out changed-cell accuracy -> log-odds weight, which is the closed-form MAP
     under a naive-Bayes product), NOT a fixed trust scalar.
  3. ONLINE PRUNING of experts whose held-out accuracy is no better than no-change (weight <= 0): a useless
     or harmful expert is dropped, so it cannot drag the product toward the identity/degenerate engine that
     sank exp4749.
  4. Optional VERIFIER reweighting (the Carnot oracle-distinct angle): the S1 off-path structural energy
     scorer can down-weight experts whose predictions are structurally implausible, using NO ground-truth.

DECISIVE METRIC (retire_if_same_verdict). The verifier-moat-relevant question is whether the weighted-
product world model PREDICTS HELD-OUT TRANSITIONS better (changed-cell recall / exact accuracy) than (a)
the single LLM-induced engine and (b) the nulled max-vote ProductWorldModel. This is a fast offline
transition-prediction measurement (no live search). If PoE does not beat BOTH baselines, it is an honest
null and this lever retires -- it does not get re-proposed.

verifier_is_oracle = False everywhere: the experts are programmatic factors; the weights come from held-out
predictive accuracy + (optionally) the structural-energy verifier, never from the win oracle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    ProgrammaticExpert,
    Transition,
    WorldModelVerifier,
    _color_rewrite_expert,
    _exact_delta_expert,
)


def _expert_from_dict(row: Mapping[str, Any]) -> Optional[ProgrammaticExpert]:
    """Build a ProgrammaticExpert from an induce_programmatic_experts dict (kind='color_rewrite')."""
    if str(row.get("kind") or "color_rewrite") != "color_rewrite":
        return None
    try:
        return _color_rewrite_expert(
            name=str(row.get("name") or "color_rewrite"),
            object_class=str(row.get("object_class") or "color"),
            action=None if row.get("action") in (None, "", "any") else int(row["action"]),
            from_color=int(row["from_color"]),
            to_color=int(row["to_color"]),
            metadata=dict(row.get("metadata") or {}),
        )
    except Exception:
        return None


def harvest_color_rewrite_experts(
    transitions: Sequence[Transition], *, min_count: int = 2, max_experts: int = 24
) -> list[ProgrammaticExpert]:
    """Harvest BROAD, OVERLAPPING color-rewrite experts from observed per-cell value changes (LLM-free).
    For each frequently-observed (action, from_color -> to_color) change, build an expert that, for that
    action, rewrites EVERY cell of from_color to to_color. Unlike the exact-delta experts (whose exact
    preconditions make them mutually-exclusive per cell), these apply broadly and CONFLICT when a colour is
    observed mapping to different targets -- which is precisely what gives the weighted-consensus combination
    something to resolve (the mechanism that is inert on an exact-delta-only pool)."""
    from collections import Counter

    counts: Counter = Counter()
    for t in transitions:
        if t.grid.shape != t.next_grid.shape:
            continue
        for r, c in np.argwhere(t.grid != t.next_grid):
            counts[(int(t.action), int(t.grid[r, c]), int(t.next_grid[r, c]))] += 1
    experts: list[ProgrammaticExpert] = []
    for (action, src, dst), n in counts.most_common(max_experts):
        if n < min_count or src == dst:
            continue
        experts.append(
            _color_rewrite_expert(
                name=f"rewrite_a{action}_{src}to{dst}",
                object_class=f"color_{src}",
                action=action,
                from_color=src,
                to_color=dst,
                metadata={"observed_count": int(n)},
            )
        )
    return experts


def build_expert_pool(
    transitions: Sequence[Transition],
    *,
    expert_dicts: Sequence[Mapping[str, Any]] = (),
    max_exact_delta: int = 24,
    include_color_rewrite: bool = True,
) -> list[ProgrammaticExpert]:
    """Assemble a pool of programmatic experts to combine: any LLM-proposed color-rewrite experts, PLUS
    BROAD harvested color-rewrite experts (overlapping -> exercise the consensus), PLUS exact-delta experts
    harvested from the (training) transitions. The mix matters: exact-delta experts give precise but
    mutually-exclusive predictions; the broad color-rewrite experts overlap and can DISAGREE, which is what
    lets the weighted-product consensus differ from a max-vote (the lever exp4749's pool never exercised)."""
    pool: list[ProgrammaticExpert] = []
    for row in expert_dicts:
        exp = _expert_from_dict(row)
        if exp is not None:
            pool.append(exp)
    if include_color_rewrite:
        pool.extend(harvest_color_rewrite_experts(transitions))
    seen_sigs: set = set()
    n_delta = 0
    for i, t in enumerate(transitions):
        if n_delta >= max_exact_delta:
            break
        if np.array_equal(t.grid, t.next_grid):
            continue
        sig = (int(t.action), tuple(map(tuple, np.argwhere(t.grid != t.next_grid)[:8].tolist())))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        pool.append(_exact_delta_expert(t, i))
        n_delta += 1
    return pool


def fit_poe_weights(
    experts: Sequence[ProgrammaticExpert],
    heldout: Sequence[Transition],
    *,
    energy_scorer: Any = None,
    prune_below: float = 1e-6,
    no_evidence_weight: float = 1.0,
) -> list[float]:
    """Fit a non-negative weight per expert from HELD-OUT predictive accuracy on the cells it touches, as
    the log-odds of being right vs a no-change baseline (the closed-form MAP weight under a naive-Bayes
    product of per-cell predictors). Experts that DID apply on held-out but were no better than chance
    (acc <= 0.5) get weight 0 -> PRUNED. Experts that NEVER applied on held-out (total == 0) have NO
    evidence to prune on, so they keep ``no_evidence_weight`` (default 1.0) -- pruning the unproven would
    collapse a sparse-expert product to the identity engine (the exp4749 failure mode) and would also make
    the PoE-vs-max-vote A/B unfair (max-vote keeps all experts). Optionally multiply by an energy factor
    that down-weights structurally-implausible experts (oracle-distinct: no ground truth).

    Returns a weight per expert (same order); also writes each expert's .trust for the max-vote baseline."""
    weights: list[float] = []
    energy_factor = _energy_factors(experts, heldout, energy_scorer) if energy_scorer is not None else None
    for idx, expert in enumerate(experts):
        correct = 0
        total = 0
        for t in heldout:
            if not expert.applies(t.grid, int(t.action), t.data):
                continue
            try:
                pred = np.asarray(expert.predict(t.grid, int(t.action), t.data))
            except Exception:
                continue
            if pred.shape != t.next_grid.shape:
                continue
            # score only the cells THIS expert changes (its claimed effect), vs the true next grid
            touched = pred != t.grid
            if not touched.any():
                continue
            correct += int((pred[touched] == t.next_grid[touched]).sum())
            total += int(touched.sum())
        expert.heldout_correct = correct
        expert.heldout_total = total
        if total == 0:
            # unproven (no held-out coverage): keep a neutral prior weight; trust=0 for the max-vote view.
            expert.trust = 0.0
            weight = float(no_evidence_weight)
        else:
            acc = correct / total
            acc_c = min(max(acc, 1e-3), 1 - 1e-3)  # clamp for a finite log-odds
            weight = max(0.0, math.log(acc_c / (1.0 - acc_c)))  # 0 at acc<=0.5, grows as acc->1
            expert.trust = float(acc)  # the max-vote baseline reads .trust
        if energy_factor is not None:
            weight *= float(energy_factor[idx])
        weights.append(weight if weight > prune_below else 0.0)
    return weights


def _energy_factors(
    experts: Sequence[ProgrammaticExpert],
    heldout: Sequence[Transition],
    energy_scorer: Any,
) -> list[float]:
    """Per-expert structural-energy factor in (0, 1]: experts whose single-expert predictions carry lower
    mean off-path structural energy (more plausible) keep more weight. Oracle-distinct: the scorer never
    sees the true next grid. Returns 1.0 for experts that never apply (no evidence to down-weight)."""
    raw: list[float] = []
    for expert in experts:
        verifier = WorldModelVerifier(list(heldout))

        def _single(grid, action, data, _e=expert):
            if _e.applies(grid, int(action), data):
                return np.asarray(_e.predict(grid, int(action), data))
            return np.asarray(grid)

        raw.append(verifier.offpath_structural_energy(_single, energy_scorer=energy_scorer))
    finite = [e for e in raw if math.isfinite(e)]
    if not finite:
        return [1.0] * len(experts)
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or 1.0
    # map lower energy -> factor near 1.0, higher energy -> factor near 0.5 (never zero: keep some weight)
    return [1.0 if not math.isfinite(e) else (1.0 - 0.5 * (e - lo) / span) for e in raw]


@dataclass
class PoEWorldModel:
    """Weighted product-of-experts world model. ``engine(grid, action, data) -> next_grid`` combines the
    applying experts by a per-cell WEIGHTED CONSENSUS (the product MAP), not the max-vote of
    ProductWorldModel. ``verifier_is_oracle = False``."""

    experts: Sequence[ProgrammaticExpert]
    weights: Sequence[float]
    no_change_prior: float = 0.5  # weight mass that always backs "keep the current value" (the product's
    # base measure); >0 means a single weak expert cannot flip a cell on its own -- consensus is required.
    verifier_is_oracle: bool = False
    diagnostics_: dict[str, Any] = field(default_factory=dict)
    # consensus-exercised telemetry (proves the weighted-product rule actually had conflicting votes to
    # resolve, vs collapsing to a single-expert-per-cell max-vote): counts cells where >=2 DISTINCT values
    # were voted across applying experts (excluding the no-change prior).
    consensus_conflict_cells: int = 0
    cells_voted: int = 0

    def _active(self) -> list[tuple[ProgrammaticExpert, float]]:
        return [(e, float(w)) for e, w in zip(self.experts, self.weights) if float(w) > 0.0]

    def engine(self, grid: np.ndarray, action: int, data: Any = None) -> np.ndarray:
        start = np.asarray(grid)
        active = self._active()
        if not active:
            return start.copy()
        h, w = start.shape
        # per-cell weighted vote tally: votes[(r,c)][value] += weight ; seed each cell with the no-change
        # prior backing the current value, so a cell only flips when applying experts outweigh that prior.
        votes: dict[tuple[int, int], dict[int, float]] = {}
        any_applies = False
        for expert, weight in active:
            if not expert.applies(start, int(action), data):
                continue
            try:
                pred = np.asarray(expert.predict(start, int(action), data))
            except Exception:
                continue
            if pred.shape != start.shape:
                continue
            any_applies = True
            changed = np.argwhere(pred != start)
            for r, c in changed:
                r, c = int(r), int(c)
                cell = votes.setdefault((r, c), {})
                cell[int(pred[r, c])] = cell.get(int(pred[r, c]), 0.0) + weight
        if not any_applies:
            return start.copy()
        out = start.copy()
        for (r, c), tally in votes.items():
            self.cells_voted += 1
            if len(tally) >= 2:  # >=2 distinct values voted by applying experts -> a real consensus to form
                self.consensus_conflict_cells += 1
            # the current value carries the no-change prior; a candidate wins only on strict majority
            base_val = int(start[r, c])
            tally_full = dict(tally)
            tally_full[base_val] = tally_full.get(base_val, 0.0) + float(self.no_change_prior)
            best_val = max(tally_full.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            out[r, c] = best_val
        return out


def build_poe_world_model(
    transitions: Sequence[Transition],
    heldout: Sequence[Transition],
    *,
    expert_dicts: Sequence[Mapping[str, Any]] = (),
    energy_scorer: Any = None,
    no_change_prior: float = 0.5,
) -> PoEWorldModel:
    """Assemble the expert pool, fit + prune weights from held-out accuracy (+ optional energy), and return
    the weighted-product world model with diagnostics (n experts, n kept, weight summary)."""
    pool = build_expert_pool(transitions, expert_dicts=expert_dicts)
    weights = fit_poe_weights(pool, heldout, energy_scorer=energy_scorer)
    kept = sum(1 for w in weights if w > 0.0)
    model = PoEWorldModel(experts=pool, weights=weights, no_change_prior=no_change_prior)
    model.diagnostics_ = {
        "n_experts": len(pool),
        "n_kept": int(kept),
        "n_pruned": int(len(pool) - kept),
        "weight_max": round(float(max(weights, default=0.0)), 4),
        "energy_reweighted": energy_scorer is not None,
        "verifier_is_oracle": False,
        "combination": "weighted_product_consensus",
    }
    return model
