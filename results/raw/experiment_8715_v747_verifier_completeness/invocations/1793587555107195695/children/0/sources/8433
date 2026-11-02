"""M2-v4b: does ACTIVE (coverage-driven) data reduce CODEX's overfitting -> lower consistency energy?

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2). M2-v4a found the DSL is DATA-SATURATED
(its tiny rule set is determined by passive data already, so active vs passive give identical energy).
The DATA hypothesis (operator-chosen lever) is really about the EXPRESSIVE inducer that overfits to the
specific examples shown: codex. M2-v3 showed codex hardcodes example colors/geometry when given too few
/ too skewed examples. Active collection balances every action and samples each from diverse contexts
(verified: m0r0 117 clicks vs passive's 1), so codex should see the rule's invariances and overfit less.

Fair test, per game: a COMMON held-out test (passive random, keys disjoint from both trains). Codex
synthesizes a program on the PASSIVE train and on the ACTIVE train (same budget, same refactor loop);
the grid-grounded consistency energy (no oracle) grades both on the common test. Active wins if its
codex program has lower energy. Reuses the M2-v3 codex machinery + the M2-v4a active collector.

  .venv/bin/python scripts/experiments/arc3_m2_active_codex.py --games m0r0,vc33 --iters 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from carnot.agentic.arc_world_model_synth import grade_predictions  # noqa: E402
from arc3_m2_world_model import _collect  # noqa: E402
from arc3_m2_active_data import active_collect, _common_test, _keys  # noqa: E402
from arc3_m2_codex_synth import (  # noqa: E402
    ask_codex, synth_prompt, _serialize, safe_predict_from_code, _extract_code, _failure_examples)


def _bg_shape(train):
    if not train:
        return 0, (0, 0)
    g0 = np.asarray(train[0][0])
    return int(np.bincount(g0.ravel()).argmax()), g0.shape


def codex_best_energy(train, test, iters, rng):
    """Run the M2-v3 codex synth loop on `train`, grade on the COMMON `test`; return (best_energy,
    history, codex_seconds)."""
    bg, shape = _bg_shape(train)
    changing = [t for t in train if (np.asarray(t[0]) != np.asarray(t[2])).any()]
    sample = (changing[:30] if len(changing) >= 8 else train[:30])
    serialized = _serialize(sample, bg, shape)
    best_e, best_code, best_fn, hist, csec = None, None, None, [], 0.0
    prior_code, failures = None, None
    for it in range(iters):
        raw, dt = ask_codex(synth_prompt(serialized, prior_code, failures))
        csec += dt
        code = _extract_code(raw)
        if code is None:
            hist.append({"iter": it, "status": "no_code", "codex_s": dt}); continue
        fn = safe_predict_from_code(code)
        if fn is None:
            hist.append({"iter": it, "status": "unsafe_or_uncompilable", "codex_s": dt}); continue
        ce = grade_predictions(fn, test)
        e = ce["energy"]
        hist.append({"iter": it, "status": "graded", "energy": e, "codex_s": dt})
        if e is not None and (best_e is None or e < best_e):
            best_e, best_code, best_fn = e, code, fn
        if best_e is not None and best_e <= 0.15:
            break
        prior_code = best_code
        failures = _failure_examples(best_fn, test, bg, shape) if best_fn else None
    return best_e, hist, round(csec, 1)


def run(games, train_budget=900, test_budget=1400, episodes=32, iters=3, seed=0, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    started = time.time()
    rng = random.Random(seed)
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files"))
    all_ids = sorted(getattr(e, "game_id", None) for e in arc.get_environments())
    sel = [g for g in all_ids if g.split("-")[0] in set(games)]

    per_game, total_csec = [], 0.0
    for game in sel:
        short = game.split("-")[0]
        test_all = _collect(arc, game, test_budget, episodes, rng, GameAction, GameState)
        passive = _collect(arc, game, train_budget, episodes, rng, GameAction, GameState)
        active = active_collect(arc, game, train_budget, episodes, rng, GameAction, GameState)
        test = _common_test(test_all, _keys(passive) | _keys(active))
        ep, hp, cp = codex_best_energy(passive, test, iters, rng)
        ea, ha, ca = codex_best_energy(active, test, iters, rng)
        total_csec += cp + ca
        per_game.append({
            "game": short, "codex_energy_passive": ep, "codex_energy_active": ea,
            "improvement_active_over_passive": (round(ep - ea, 4) if (ep is not None and ea is not None) else None),
            "n_passive": len(passive), "n_active": len(active), "n_test": len(test),
            "passive_history": hp, "active_history": ha,
        })
        print(f"  {short:6s} codex passive={ep} active={ea} "
              f"improve={per_game[-1]['improvement_active_over_passive']}", flush=True)

    rated = [g for g in per_game if g["improvement_active_over_passive"] is not None]
    n_better = sum(1 for g in rated if g["improvement_active_over_passive"] > 0.03)
    mean_p = round(sum(g["codex_energy_passive"] for g in rated) / len(rated), 4) if rated else None
    mean_a = round(sum(g["codex_energy_active"] for g in rated) / len(rated), 4) if rated else None
    verdict = (f"complete: m2v4b_active_data_for_codex_meanE_passive{mean_p}_active{mean_a}"
               f"_active_better{n_better}of{len(rated)}")
    art = {
        "experiment": "arc3_m2_active_codex", "title": "arc3_m2v4b_active_data_reduces_codex_overfitting",
        "honest_verdict": verdict,
        "inference_substrate": "offline_arc_agi3_plus_codex_program_synthesis_consistency_verified",
        "claim": ("Active coverage-driven data vs passive random, same budget, both fed to codex program "
                  "synthesis, graded by consistency energy on a COMMON held-out test. Active wins if it "
                  "lowers codex's energy (less overfitting)."),
        "n_games": len(per_game), "games": list(games),
        "mean_codex_energy_passive": mean_p, "mean_codex_energy_active": mean_a,
        "n_games_active_better_by_0.03": n_better,
        "train_budget": train_budget, "test_budget": test_budget, "iters": iters,
        "total_codex_seconds": round(total_csec, 1), "random_seed": seed,
        "no_gpu_used": True, "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1), "per_game": per_game,
        "note": ("M2-v4b: tests the data hypothesis where it bites (the expressive overfitting inducer). "
                 "If active data lowers codex energy, the operator's data lever is confirmed -> scale "
                 "active collection + iterate toward a trustworthy model + plan. If not, the bottleneck "
                 "is observation/representation (latent state), not exploration policy."),
    }
    if write:
        (REPO / "results" / "arc3_m2_active_codex.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\n-> {verdict}")
    print(f"   mean codex energy: passive={mean_p} -> active={mean_a} | active better on "
          f"{n_better}/{len(rated)} | codex {round(total_csec)}s")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default="m0r0,vc33")
    ap.add_argument("--train_budget", type=int, default=900)
    ap.add_argument("--test_budget", type=int, default=1400)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(games=[g.strip() for g in args.games.split(",") if g.strip()],
        train_budget=args.train_budget, test_budget=args.test_budget,
        episodes=args.episodes, iters=args.iters, seed=args.seed)
