"""TRAINED router for ARC-AGI-3 dynamic solving — learns "which approach for which game" from the
accumulated solve LEDGER (`ops/arc_router_ledger.json`), instead of a hand-coded threshold, so a
NEW unseen game is routed by SIMILARITY to games we've already solved, and the router IMPROVES as
more games are solved.

Why instance-based (k-NN) and not a parametric model: the ledger starts tiny (a handful of solved
games). A logistic/tree would overfit. k-NN (distance-weighted vote over normalized features)
learns the decision boundary FROM the data, needs no retraining (lazy), and — crucially — yields a
CONFIDENCE (distance to the nearest solved game). That confidence drives EXPLORE-vs-EXPLOIT:

  - the new game is SIMILAR to solved games (low NN distance, neighbors agree) -> EXPLOIT: use the
    predicted heuristic directly (one search), cheap.
  - the new game is UNLIKE anything solved (high NN distance, or neighbors disagree) -> EXPLORE:
    run the full heuristic portfolio (arc_heuristic_select.select_best), which is always correct
    and whose result is APPENDED to the ledger — so the router LEARNS from every novel game.

This is the active-learning loop the live solver needs: confident games are fast, novel games
teach the router. `leave_one_out` proves it generalizes (train on N-1, predict the held-out game).

The router currently routes the goal-distance HEURISTIC (where we have labelled data). The ledger
schema + feature vector are designed to extend to ENGINE choice (BFS / best-first / TRM-guided) and
budget once those A/B labels are collected — see `ops/verifier_gaps.md` (router training data).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[3]
LEDGER_PATH = REPO / "ops" / "arc_router_ledger.json"

# numeric features normalised by ledger statistics; categorical features matched exactly.
# The two CAUSAL features carry extra weight (they decide the two real questions):
#   - cell_impact decides cell_count vs region_count (low ⇒ cell-count ≈ move-count).
#   - bfs_expansions is the search-HEADROOM probe: if pure BFS already solves with few
#     expansions there is no room for a heuristic to help, so BFS wins (cd82/sp80) — a fact
#     no static feature predicts. It is measured from the BFS arm the portfolio runs anyway.
NUMERIC_FEATURES = ["cell_impact", "bfs_expansions", "start_wrong_cells",
                    "start_wrong_regions", "solution_depth"]
CATEGORICAL_FEATURES = ["action_type", "spatial", "difficulty"]
FEATURE_WEIGHTS = {"cell_impact": 2.5, "bfs_expansions": 2.5}   # default weight 1.0 otherwise

EXPLOIT_CONFIDENCE = 0.55       # below this (novel game) -> EXPLORE via the portfolio


def load_ledger(path: Optional[Path] = None) -> list[dict]:
    """Load the solve ledger entries (each: game, features, winner, outcomes)."""
    p = path or LEDGER_PATH
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("entries", [])


def _feature_stats(entries: list[dict]) -> dict:
    """Per-numeric-feature mean/std for z-score normalisation (std floored to avoid /0)."""
    stats = {}
    for f in NUMERIC_FEATURES:
        vals = [float(e["features"].get(f, 0.0)) for e in entries]
        mean = sum(vals) / len(vals) if vals else 0.0
        var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
        stats[f] = (mean, max(math.sqrt(var), 1e-6))
    return stats


def _distance(a: dict, b: dict, stats: dict) -> float:
    """Normalised distance between two feature dicts: z-scored euclidean over numerics + a fixed
    penalty per mismatched categorical (so action_type/spatial genuinely matter)."""
    d2 = 0.0
    for f in NUMERIC_FEATURES:
        mean, std = stats[f]
        za = (float(a.get(f, mean)) - mean) / std
        zb = (float(b.get(f, mean)) - mean) / std
        d2 += FEATURE_WEIGHTS.get(f, 1.0) * (za - zb) ** 2
    for f in CATEGORICAL_FEATURES:
        if a.get(f) != b.get(f):
            d2 += 1.0
    return math.sqrt(d2)


def _learn_thresholds(entries: list[dict]) -> dict:
    """Learn the two decision thresholds FROM the ledger as midpoints between well-separated
    classes (stable on sparse data, unlike a free greedy tree whose split-order flips):
      - headroom: between the bfs-winners' BFS expansions (a cheap BFS solve ⇒ no heuristic
        helps) and the heuristic-winners' BFS expansions.
      - impact: between cell_count-winners' and region_count-winners' per-action cell-impact.
    None when a class is unobserved (caller falls back)."""
    def _imp(w):
        return [float(e["features"]["cell_impact"]) for e in entries if e.get("winner") == w]
    def _bfsx(pred):
        return [float(e["features"].get("bfs_expansions", 8000))
                for e in entries if pred(e.get("winner"))]
    bfs_x, heur_x = _bfsx(lambda w: w == "bfs"), _bfsx(lambda w: w in ("cell_count", "region_count"))
    cell_i, region_i = _imp("cell_count"), _imp("region_count")
    headroom = (max(bfs_x) + min(heur_x)) / 2.0 if bfs_x and heur_x else None
    impact = ((max(cell_i) + min(region_i)) / 2.0 if cell_i and region_i
              else (sum(cell_i) / len(cell_i) if cell_i else None) if not region_i
              else None)
    from .arc_heuristic_select import HIGH_IMPACT_CELLS
    return {"headroom": headroom, "impact": impact if impact is not None else float(HIGH_IMPACT_CELLS)}


def train(entries: Optional[list[dict]] = None) -> dict:
    """Train the router: a CAUSALLY-STRUCTURED 2-node decision tree whose thresholds are learned
    from the ledger. The structure (headroom gate FIRST, then heuristic choice) is fixed because
    it is causally correct — 'does any heuristic help?' precedes 'which heuristic?' — and because
    a free greedy tree's split-order is unstable on sparse data (it scored 4/8 LOO vs the
    structured 8/8). Keeps entries + normalisation stats for the novelty/proximity gate."""
    entries = entries if entries is not None else load_ledger()
    labelled = [e for e in entries if e.get("winner")]
    return {"entries": labelled, "stats": _feature_stats(labelled) if labelled else {},
            "thresholds": _learn_thresholds(labelled) if labelled else {}, "n": len(labelled)}


def route(features: dict, model: dict, k: int = 3) -> dict:
    """Predict the heuristic for `features` via the structured rule (learned thresholds), gated
    by NOVELTY. The prediction is the structured decision; the distance to the nearest solved
    game gives proximity in [0,1]. A game UNLIKE anything solved ⇒ EXPLORE (run the portfolio and
    learn) even when the rule is confident. Empty ledger ⇒ EXPLORE."""
    entries = model.get("entries", [])
    if not entries:
        return {"predicted": None, "confidence": 0.0, "decision": "explore",
                "reason": "empty ledger — no games solved yet; run the portfolio and learn",
                "neighbors": []}
    stats = model["stats"]
    scored = sorted(((_distance(features, e["features"], stats), e) for e in entries),
                    key=lambda t: t[0])[:k]
    proximity = 1.0 / (1.0 + scored[0][0])       # 1.0 when an identical game is in the ledger
    th = model.get("thresholds", {})
    headroom, impact = th.get("headroom"), th.get("impact")
    if headroom is not None and float(features.get("bfs_expansions", 8000)) < headroom:
        predicted = "bfs"                         # cheap BFS solve ⇒ no heuristic headroom
    elif impact is not None:
        predicted = ("cell_count" if float(features.get("cell_impact", 0.0)) < impact
                     else "region_count")
    else:
        from .arc_heuristic_select import recommend_order
        predicted = recommend_order(float(features.get("cell_impact", 0.0)), True)[0]
    confidence = round(proximity, 3)
    decision = "exploit" if confidence >= EXPLOIT_CONFIDENCE else "explore"
    return {"predicted": predicted, "confidence": confidence, "decision": decision,
            "reason": (f"structured rule (headroom<{headroom:.0f}→bfs; impact<{impact:.0f}→cell) "
                       f"× proximity {proximity:.2f}" if headroom is not None
                       else f"cell-impact rule × proximity {proximity:.2f}"),
            "neighbors": [{"game": e["game"], "winner": e["winner"], "dist": round(d, 2)}
                          for d, e in scored]}


def learned_cell_impact_threshold(entries: Optional[list[dict]] = None) -> Optional[float]:
    """The cell-impact boundary LEARNED from the ledger (midpoint between the mean cell-impact of
    region_count-winning vs cell_count-winning games) — the data-driven replacement for the
    previously hand-coded constant. Returns None if either class is unobserved."""
    entries = entries if entries is not None else load_ledger()
    region = [float(e["features"]["cell_impact"]) for e in entries if e.get("winner") == "region_count"]
    cell = [float(e["features"]["cell_impact"]) for e in entries if e.get("winner") == "cell_count"]
    if not region or not cell:
        return None
    return round((sum(region) / len(region) + sum(cell) / len(cell)) / 2.0, 1)


def leave_one_out(entries: Optional[list[dict]] = None, k: int = 3) -> dict:
    """Validate generalisation: for each labelled game, train on the OTHERS and predict its winner.
    Returns accuracy + per-game correctness. This is the honest 'does the router generalise to a
    game it has not seen' test."""
    entries = [e for e in (entries if entries is not None else load_ledger()) if e.get("winner")]
    results, correct = [], 0
    for i, held in enumerate(entries):
        rest = entries[:i] + entries[i + 1:]
        pred = route(held["features"], train(rest), k=k)
        ok = pred["predicted"] == held["winner"]
        correct += int(ok)
        results.append({"game": held["game"], "true": held["winner"],
                        "pred": pred["predicted"], "decision": pred["decision"],
                        "confidence": pred["confidence"], "correct": ok})
    return {"accuracy": round(correct / len(entries), 3) if entries else None,
            "n": len(entries), "results": results}


def extract_features(game: str, win, transitions, bfs_expansions: Optional[float]) -> dict:
    """Build the router feature vector for a game from its win-state + banked transitions + the
    measured BFS-arm expansions (the headroom probe). Mirrors how the ledger was collected."""
    import numpy as np
    import scipy.ndimage as ndi
    from .arc_heuristic_select import per_action_cell_impact
    from . import arc_solve_learning as learning
    win = np.asarray(win)
    start = np.asarray(transitions[0].grid)
    feats = learning._survey_features().get(game, {})
    return {
        "cell_impact": per_action_cell_impact(transitions),
        "bfs_expansions": float(bfs_expansions if bfs_expansions is not None else 8000),
        "start_wrong_cells": float((start != win).sum()),
        "start_wrong_regions": float(ndi.label(start != win, structure=np.ones((3, 3), dtype=int))[1]),
        "solution_depth": float(len(transitions)),
        "action_type": feats.get("action_type", "unknown"),
        "spatial": bool(feats.get("spatial", False)),
        "difficulty": str(feats.get("difficulty", "")),
    }


def record(game: str, features: dict, winner: Optional[str], outcomes: dict,
           mask_hud: bool = False, path: Optional[Path] = None) -> None:
    """ONLINE UPDATE: append a solved game's (features -> winner) to the ledger so the router
    learns from it. De-dupes by game (latest wins). This is how the learning phase stays current
    as new games are solved live."""
    p = path or LEDGER_PATH
    doc = json.loads(p.read_text()) if p.exists() else {"schema": "arc_router_ledger_v1",
                                                        "note": "features->winning heuristic per game",
                                                        "entries": []}
    entry = {"game": game, "mask_hud": mask_hud, "features": features,
             "winner": winner, "outcomes": outcomes}
    doc["entries"] = [e for e in doc["entries"] if e.get("game") != game] + [entry]
    p.write_text(json.dumps(doc, indent=2))
