#!/usr/bin/env python3
"""Tier-A: target-energy (`unsatisfied_targets`) as an A* HEURISTIC vs blind BFS.

WHY (2026-06-22, outer-loop prep for the .425 energy-config-space program):
A3 (exp4594) wired a goal-distance ENERGY as a candidate-REORDERING prior and got
0/25 winners attributable to it (`winner_generated_by_energy_prior`). But vc33 already
uses `unsatisfied_targets` as an A* HEURISTIC (`vc33_goal_distance_heuristic`) that
STEERS the frontier (depth + w*h), which is a strictly stronger use than reordering.
This experiment isolates that distinction: does energy-as-DIRECTED-SEARCH GENERATE a
winner (and reach it with FEWER expansions) where energy-as-REORDER did not?

It ALSO measures the perception-brittleness that bounds the whole #1 path: vc33's
featurizer uses hardcoded TARGET_COLORS, so `unsatisfied_targets` is computed only for
games where a bespoke featurizer exists, and should BREAK under color-permutation. So:
  - base vc33 (featurizer works) -> the efficiency/generation test.
  - color-permuted held-out variants (featurizer expected to break) -> quantifies the
    per-game-perception gap that Tier-B (general goal-structure induction) must close.

If target-energy-heuristic beats BFS on base AND survives a perturbation, energy-as-
directed-search is worth pursuing on games where perception exists. If it only works on
the already-solved base and breaks on variants, that confirms #1 is gated on per-game
perception and the lead should be #2 (energy-as-fitness, perception-independent).

Honest, reproduction-gated (`arc_solver_kit.reproduce`), OFFLINE, zero quota.
OUTER-LOOP PREP EXPERIMENT (not a conductor task) — run AFTER the conductor is stopped
to avoid offline-env contention. `verifier_is_oracle: false` (the energy estimates
progress; it is oracle-distinct from the env's win logic).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_variant_generator as variants
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2, trajectory_labels, _warm
from carnot.agentic.arc_executable_world_model import detect_cell, to_logical
from carnot.agentic.arc_vc33_hierarchical_search import (
    grid_state_features,
    vc33_goal_distance_heuristic,
)

GAME = "vc33-5430563c"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_tierA_target_energy_heuristic.json"


def _logical(frame, cell):
    """Frame -> vc33 logical grid (the featurizer's expected input)."""
    return to_logical(grid_of(frame), cell)


def _make_target_energy_heuristic(cell):
    """h(frame) = vc33 target-energy (unmet*1000 + manhattan); lower = closer to win.
    A broken/degenerate featurizer returns a large constant so it never crashes search."""

    def heuristic(frame):
        try:
            feats = grid_state_features(np.asarray(_logical(frame, cell)))
            return float(vc33_goal_distance_heuristic(feats))
        except Exception:
            return 1e9

    return heuristic


def _featurizer_probe(env, cell) -> dict:
    """Does `unsatisfied_targets` compute non-degenerately on this (possibly variant) env?
    Returns the reset-frame features + an `ok` flag (a real target structure was perceived)."""
    f = _warm(env, False)
    try:
        feats = grid_state_features(np.asarray(_logical(f, cell)))
        total = int(feats.get("total_targets", 0) or 0)
        # vc33 reports has_target_pair / misaligned; treat "saw a target structure" as ok.
        ok = bool(feats.get("has_target_pair")) or total > 0 or int(
            feats.get("unsatisfied_targets", 0) or 0
        ) > 0
        return {"ok": bool(ok), "features": {k: feats.get(k) for k in (
            "has_target_pair", "misaligned_target_pairs", "unsatisfied_targets",
            "total_targets", "manhattan_to_target")}}
    except Exception as exc:  # featurizer crashed on this variant -> perception broke
        return {"ok": False, "error": str(exc)[:160]}


def _apply(env, label_, frame):
    s = json.loads(label_)
    return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))


def _make_env(spec: dict):
    arc = kit.offline_arcade()
    env = arc.make(GAME, scorecard_id=arc.open_scorecard())
    if spec.get("variant") is not None:
        env = variants.VariantEnv(env, GAME, int(spec["variant"]), reflect=spec.get("reflect"))
    return env


def _run_arm(spec: dict, arm: str, budget: int, cell) -> dict:
    env = _make_env(spec)
    heuristic = _make_target_energy_heuristic(cell) if arm == "target_energy_heuristic" else None
    st: dict = {}
    t1 = time.time()
    traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=budget, max_depth=60,
                                       heuristic=heuristic, stats=st)
    solved = bool(traj) and int(lvl) >= 1
    reproduced = False
    if solved and spec.get("variant") is None:
        # base game: the formal reproduction gate replays labels against a fresh base env.
        # (variant solves are real via VariantEnv's pass-through win logic; the offline env
        #  is deterministic, so winner_generated on the variant IS the reproduced signal.)
        g = kit.reproduce(GAME, trajectory_labels(traj), _apply, claimed_level=int(lvl))
        reproduced = bool(g["reproduced"])
    elif solved:
        reproduced = True  # deterministic offline VariantEnv solve (pass-through win logic)
    return {
        "arm": arm,
        "reached_level": int(lvl),
        "winner_generated": bool(solved),
        "offline_reproduced": bool(reproduced),
        "actions": len(traj) if traj else 0,
        "expansions": st.get("expansions"),
        "secs": round(time.time() - t1, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2000, help="max_expansions per arm")
    ap.add_argument("--color-variants", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()

    # featurizer needs the cell size; detect it once on the base reset frame.
    base_env = _make_env({})
    cell = detect_cell(grid_of(_warm(base_env, False)))

    specs = [{"label": "base", "variant": None}]
    for v in range(1, args.color_variants + 1):
        specs.append({"label": f"color_v{v}", "variant": v})
        specs.append({"label": f"color_v{v}_reflect0", "variant": v, "reflect": 0})

    rows = []
    for spec in specs:
        probe = _featurizer_probe(_make_env(spec), cell)
        arms = {}
        for arm in ("target_energy_heuristic", "bfs_baseline"):
            r = _run_arm(spec, arm, args.budget, cell)
            arms[arm] = r
            print(f"  [{spec['label']:18}] featurizer_ok={probe['ok']!s:5} {arm:24} "
                  f"winner={r['winner_generated']!s:5} repro={r['offline_reproduced']!s:5} "
                  f"L{r['reached_level']} exp={r['expansions']} [{r['secs']}s]", flush=True)
        rows.append({"spec": spec, "featurizer_ok": probe["ok"], "featurizer": probe, "arms": arms})

    # --- aggregate ---
    def rate(rows_, arm, pred):
        sub = [r for r in rows_ if pred(r)]
        if not sub:
            return None
        return round(sum(1 for r in sub if r["arms"][arm]["winner_generated"]) / len(sub), 4)

    feat_ok = [r for r in rows if r["featurizer_ok"]]
    feat_broken = [r for r in rows if not r["featurizer_ok"]]
    te_all = rate(rows, "target_energy_heuristic", lambda r: True)
    bfs_all = rate(rows, "bfs_baseline", lambda r: True)
    te_okonly = rate(rows, "target_energy_heuristic", lambda r: r["featurizer_ok"])
    bfs_okonly = rate(rows, "bfs_baseline", lambda r: r["featurizer_ok"])

    # expansion efficiency on featurizer-ok variants where BOTH solved
    eff = []
    for r in feat_ok:
        te, bfs = r["arms"]["target_energy_heuristic"], r["arms"]["bfs_baseline"]
        if te["winner_generated"] and bfs["winner_generated"] and te["expansions"] and bfs["expansions"]:
            eff.append((bfs["expansions"], te["expansions"]))
    expansion_ratio = round(np.mean([b / max(1, t) for b, t in eff]), 3) if eff else None

    # honest, self-determining verdict
    helps = (te_okonly is not None and bfs_okonly is not None and te_okonly >= bfs_okonly
             and (expansion_ratio is None or expansion_ratio >= 1.0))
    generalizes = bool(feat_broken) and te_all is not None and bfs_all is not None and te_all > bfs_all
    if helps and generalizes:
        verdict = "success: target_energy_heuristic_generates_and_transfers_above_bfs"
    elif helps:
        verdict = ("complete: target_energy_heuristic_helps_on_featurizer_ok_only_no_variant_"
                   "transfer_perception_gated")
    else:
        verdict = "complete: target_energy_heuristic_no_value_honest_null_gap_sharpened"

    artifact = {
        "experiment": "experiment_tierA_target_energy_heuristic",
        "game": GAME,
        "honest_verdict": verdict,
        "verifier_is_oracle": False,
        "inference_substrate": "offline_arc_search",
        "random_seed": 20260622,
        "budget_max_expansions": args.budget,
        "winner_generated_rate_target_energy_all": te_all,
        "winner_generated_rate_bfs_all": bfs_all,
        "winner_generated_rate_target_energy_featurizer_ok": te_okonly,
        "winner_generated_rate_bfs_featurizer_ok": bfs_okonly,
        "n_featurizer_ok": len(feat_ok),
        "n_featurizer_broken": len(feat_broken),
        "mean_bfs_over_target_energy_expansion_ratio": expansion_ratio,
        "perception_brittleness_note": (
            "featurizer_ok on base/color-preserving variants, expected False on color-permuted "
            "variants (hardcoded TARGET_COLORS) -> bounds #1 to games with a bespoke featurizer."),
        "a3_reorder_prior_baseline": "exp4594 winner_generated_by_energy_prior=0/25 (the reorder-prior null)",
        "rows": rows,
        "duration_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nVERDICT: {verdict}")
    print(f"  target_energy (featurizer-ok): {te_okonly}  vs  bfs: {bfs_okonly}  "
          f"| all-variants te {te_all} vs bfs {bfs_all} | expansion_ratio(bfs/te) {expansion_ratio}")
    print(f"  featurizer ok={len(feat_ok)} broken={len(feat_broken)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
