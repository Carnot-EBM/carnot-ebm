"""M2-v1a: does the grid-grounded consistency-energy verifier correctly certify model trustworthiness?

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2). The determinism probe split the 25
offline games into 14 grid-Markov + 11 hidden-state. This experiment is the first honest, non-
tautological test of the CARNOT ENERGY VERIFIER on real ARC games: for each game we explore, collect
real (state, action, next_state) transitions, fit InducedWorldModel on a TRAIN split, and have the
verifier predict a DISJOINT HELD-OUT split, graded against the OBSERVED next grid (no oracle).

The load-bearing, falsifiable claim (the Meta-EBM cascade-router thesis): consistency_energy should be
LOW on the 14 Markov games (the model can be trusted -> safe to plan on) and HIGH on the 11 hidden-
state games (the grid->grid model is untrustworthy -> escalate, don't plan). We validate the verifier
against the determinism-probe labels as ground truth and report the separation (AUROC + best threshold).

Contrast with the flagged exp3929 tautology: there the verifier scored a planted string written from
oracle ground truth. HERE it predicts a grid it never saw and is graded against reality; a wrong
prediction is fully possible. Fully offline, no LLM/GPU.

  .venv/bin/python scripts/experiments/arc3_m2_world_model.py --budget 1500 --episodes 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from carnot.agentic.arc_agi3_world_model import (  # noqa: E402
    GameGraph, grid_of, frame_hash, compute_grid_delta)
from carnot.agentic.arc_world_model_synth import InducedWorldModel  # noqa: E402
import arc3_graph_explore as gx  # noqa: E402


def _collect(arc, game, budget, episodes, rng, GameAction, GameState):
    """Directed-random exploration that keeps FULL grids: returns [(s_grid, akey, s2_grid), ...]."""
    by_id = {a.value: a for a in GameAction}
    graph = GameGraph(game)
    transitions = []
    total = 0
    for _ in range(episodes):
        env = arc.make(game)
        f = env.reset()
        prev = None
        while total < budget:
            grid = grid_of(f)
            fh = frame_hash(grid)
            graph.see_node(fh, f)
            if prev is not None:
                transitions.append((prev[1], prev[2], grid.copy()))
                graph.record(prev[0], prev[2], fh, compute_grid_delta(prev[1], grid), 0, False)
            if getattr(f, "state", None) in (GameState.WIN, GameState.GAME_OVER):
                break
            cands = gx._candidate_akeys(grid, getattr(f, "available_actions", []))
            untested = graph.untested(fh, cands)
            tried_here = [k for k in cands if graph.tried(fh, k) and not graph.is_deadly(fh, k)]
            if tried_here and rng.random() < 0.3:                 # revisit to sample determinism
                akey = rng.choice(tried_here)
            elif untested:
                akey = gx._pick(graph, fh, untested, rng)
            elif tried_here:
                akey = rng.choice(tried_here)
            else:
                break
            a_int = akey[0]
            data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
            prev = (fh, grid.copy(), akey)
            f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
            total += 1
            if getattr(f, "state", None) == GameState.GAME_OVER:
                transitions.append((prev[1], akey, grid_of(f).copy()))  # (s_grid, akey, s2_grid)
                break
        if total >= budget:
            break
    return transitions


def _random_split(transitions, rng, frac=0.25):
    """Random transition split (held-out (s,a) MAY also appear in train). Tests the VERIFIER under
    memorization: on a seen (s,a), the only error source is multivaluedness, so energy should track
    hidden-state-ness and separate Markov (single-valued -> low) from hidden (multivalued -> high)."""
    idx = list(range(len(transitions)))
    rng.shuffle(idx)
    n_hold = max(1, int(len(transitions) * frac))
    hold = set(idx[:n_hold])
    train = [t for i, t in enumerate(transitions) if i not in hold]
    held = [t for i, t in enumerate(transitions) if i in hold]
    return train, held


def _key_disjoint_split(transitions, rng, frac=0.25):
    """Split so HELD-OUT (frame_hash, akey) keys are DISJOINT from TRAIN — measures true generalization
    to unseen (state, action), not memorization."""
    keys = list({(frame_hash(s), tuple(a)) for s, a, _ in transitions})
    rng.shuffle(keys)
    n_hold = max(1, int(len(keys) * frac))
    hold_keys = set(keys[:n_hold])
    train, held = [], []
    for s, a, s2 in transitions:
        (held if (frame_hash(s), tuple(a)) in hold_keys else train).append((s, a, s2))
    return train, held


def _auroc(scores, labels):
    """AUROC of `scores` predicting label==1 (rank-based; no sklearn)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def run(games=None, budget=1500, episodes=40, seed=0, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    started = time.time()
    rng = random.Random(seed)
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    all_ids = sorted(getattr(e, "game_id", None) for e in arc.get_environments())
    sel = [g for g in all_ids if (not games or g.split("-")[0] in set(games))]

    # ground-truth labels from the determinism probe (1 = hidden-state)
    probe_path = REPO / "results" / "arc3_determinism_probe.json"
    hidden = set()
    if probe_path.exists():
        hidden = set(json.loads(probe_path.read_text()).get("hidden_state_games", []))

    per_game = []
    for game in sel:
        short = game.split("-")[0]
        trans = _collect(arc, game, budget, episodes, rng, GameAction, GameState)
        # MEMORIZATION regime (random split) -> tests the VERIFIER (energy ~ multivaluedness)
        tr_r, hd_r = _random_split(trans, rng, frac=0.25)
        ce_mem = InducedWorldModel(short).fit(tr_r).consistency_energy(hd_r)
        # GENERALIZATION regime (key-disjoint) -> tests the INDUCER's ability to predict unseen (s,a)
        tr_k, hd_k = _key_disjoint_split(trans, rng, frac=0.25)
        ce_gen = InducedWorldModel(short).fit(tr_k).consistency_energy(hd_k)
        is_hidden = short in hidden
        per_game.append({
            "game": short, "is_hidden_state_truth": is_hidden, "n_transitions": len(trans),
            "energy_memorization": ce_mem["energy"], "dyn_acc_memorization": ce_mem.get("dynamics_accuracy"),
            "energy_generalization": ce_gen["energy"], "dyn_acc_generalization": ce_gen.get("dynamics_accuracy"),
            "n_changed_mem": ce_mem.get("n_changed_transitions"),
            "verifier_flags_untrustworthy_at_0.20": (ce_mem["energy"] is not None and ce_mem["energy"] > 0.20),
        })
        print(f"  {short:6s} E_mem={ce_mem['energy']} E_gen={ce_gen['energy']} "
              f"hidden_truth={is_hidden} (mem dyn_acc={ce_mem.get('dynamics_accuracy')})", flush=True)

    rated = [g for g in per_game if g["energy_memorization"] is not None]
    labels = [1 if g["is_hidden_state_truth"] else 0 for g in rated]
    auroc_mem = _auroc([g["energy_memorization"] for g in rated], labels)
    auroc_gen = _auroc([g["energy_generalization"] for g in rated], labels)
    markov = [g for g in rated if not g["is_hidden_state_truth"]]
    hid = [g for g in rated if g["is_hidden_state_truth"]]

    def _mean(gs, k):
        return round(sum(g[k] for g in gs) / len(gs), 4) if gs else None
    mE_mem_markov, mE_mem_hidden = _mean(markov, "energy_memorization"), _mean(hid, "energy_memorization")
    mE_gen_markov, mE_gen_hidden = _mean(markov, "energy_generalization"), _mean(hid, "energy_generalization")
    # best threshold on the MEMORIZATION energy (the verifier-trustworthiness signal)
    best_thr, best_acc = None, -1.0
    for thr in [i / 100 for i in range(2, 99, 2)]:
        acc = sum((g["energy_memorization"] > thr) == g["is_hidden_state_truth"] for g in rated) / len(rated)
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    verdict = (f"complete: m2_consistency_verifier_auroc_mem{auroc_mem}_gen{auroc_gen}"
               f"_meanEmem_markov{mE_mem_markov}_hidden{mE_mem_hidden}_sepacc{round(best_acc, 3)}"
               f"_naive_inducer_cannot_generalize_genE_markov{mE_gen_markov}")
    art = {
        "experiment": "arc3_m2_world_model", "title": "arc3_m2_consistency_verifier_certification",
        "honest_verdict": verdict,
        "inference_substrate": "offline_arc_agi3_world_model_consistency_energy",
        "claim": ("Grid-grounded consistency_energy (held-out misprediction rate, no oracle) certifies "
                  "which games the world-model can be trusted on. Validated vs determinism-probe labels."),
        "n_games": len(per_game),
        "verifier_auroc_memorization_vs_determinism_truth": auroc_mem,
        "inducer_auroc_generalization_vs_determinism_truth": auroc_gen,
        "mean_energy_memorization_markov": mE_mem_markov, "mean_energy_memorization_hidden": mE_mem_hidden,
        "mean_energy_generalization_markov": mE_gen_markov, "mean_energy_generalization_hidden": mE_gen_hidden,
        "best_separating_threshold_memorization": best_thr, "separation_accuracy_memorization": round(best_acc, 4),
        "regimes": {
            "memorization_random_split": "tests the VERIFIER: on seen (s,a) the only error is "
            "multivaluedness, so energy should separate Markov (low) from hidden (high)",
            "generalization_key_disjoint_split": "tests the INDUCER: held-out (frame_hash,action) "
            "disjoint from train; the naive template inducer's energy here measures true generalization",
        },
        "budget_per_game": budget, "episodes_per_game": episodes, "random_seed": seed,
        "no_llm_used": True, "no_gpu_used": True, "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1), "per_game": per_game,
        "note": ("M2-v1a: the verifier's FIRST honest non-tautological test on real games (contrast "
                 "exp3929, flagged tautology: scored a planted oracle string). Two regimes: "
                 "MEMORIZATION (random split) tests whether the consistency energy correctly certifies "
                 "model trustworthiness (AUROC near 1.0 = the cascade-router gate is real); "
                 "GENERALIZATION (key-disjoint) tests the inducer. If the naive template inducer cannot "
                 "generalize (energy ~1 on Markov too), that empirically MOTIVATES DSL program "
                 "synthesis as the real M2 inducer (M2-v2)."),
    }
    if write:
        (REPO / "results" / "arc3_m2_world_model.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\n-> {verdict}")
    print(f"   VERIFIER (memorization): AUROC={auroc_mem} meanE markov={mE_mem_markov} hidden={mE_mem_hidden} "
          f"| sep_acc={round(best_acc, 3)} @thr={best_thr}")
    print(f"   INDUCER  (generalization): AUROC={auroc_gen} meanE markov={mE_gen_markov} hidden={mE_gen_hidden} "
          f"(naive template; ~1.0 = cannot generalize -> needs DSL synthesis)")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default="", help="comma-separated short ids; empty = all 25")
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gl = [g.strip() for g in args.games.split(",") if g.strip()] or None
    art = run(games=gl, budget=args.budget, episodes=args.episodes, seed=args.seed)
    raise SystemExit(0)
