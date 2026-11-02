"""
REQ-3955: Non-spatial ARC games active-collect + codex program synthesis.
SCENARIO-3955-1: Test active codex generalization across 6 non-spatial games.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from arc3_m2_active_codex import codex_best_energy
from arc3_m2_active_data import active_collect, _common_test, _keys
from arc3_m2_world_model import _collect


def run(games=None, train_budget=900, test_budget=1400, episodes=32, iters=3, seed=0, write=True,
        _arc_client=None, _codex_available=None):
    if games is None:
        games = ["r11l", "sc25", "lp85", "tn36", "dc22", "su15"]
        
    started = time.time()
    
    # Precondition 1: codex available
    has_codex = _codex_available if _codex_available is not None else bool(shutil.which("codex"))
    if not has_codex:
        verdict = "blocked_codex_unavailable"
        art = {"honest_verdict": verdict}
        if write:
            (REPO / "results" / "experiment_3955_active_codex_nonspatial_sweep.json").write_text(
                json.dumps(art, indent=2) + "\n", "utf-8"
            )
        return art

    # Precondition 2: ARC offline available
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState

    if _arc_client is None:
        try:
            arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                         environments_dir=str(REPO / "environment_files"))
            envs = arc.get_environments()
            if len(envs) == 0:
                raise ValueError("No envs")
        except Exception:
            verdict = "blocked_arc_offline_env_unavailable"
            art = {"honest_verdict": verdict}
            if write:
                (REPO / "results" / "experiment_3955_active_codex_nonspatial_sweep.json").write_text(
                    json.dumps(art, indent=2) + "\n", "utf-8"
                )
            return art
    else:
        arc = _arc_client
        envs = arc.get_environments()

    rng = random.Random(seed)
    all_ids = sorted(getattr(e, "game_id", None) for e in envs if getattr(e, "game_id", None))
    sel = [g for g in all_ids if g and g.split("-")[0] in set(games)]

    per_game = []
    total_csec = 0.0
    total_calls = 0

    for game in sel:
        short = game.split("-")[0]
        test_all = _collect(arc, game, test_budget, episodes, rng, GameAction, GameState)
        active = active_collect(arc, game, train_budget, episodes, rng, GameAction, GameState)
        test = _common_test(test_all, _keys(active))

        ea, ha, ca = codex_best_energy(active, test, iters, rng)
        total_csec += ca
        calls = len([h for h in ha if "codex_s" in h])
        total_calls += calls

        per_game.append({
            "game": short,
            "best_energy": ea,
            "trustworthy": (ea is not None and ea <= 0.15),
            "n_active": len(active),
            "n_test": len(test),
            "history": ha,
            "compare_to_vc33_baseline_0.005": round((ea - 0.005), 4) if ea is not None else None
        })

    markov = [g["game"] for g in per_game if g["trustworthy"]]
    hidden = [g["game"] for g in per_game if not g["trustworthy"]]

    n_trustworthy = len(markov)
    per_game_best_energy = {g["game"]: g["best_energy"] for g in per_game}

    verdict = f"success: swept {len(per_game)} nonspatial games, {n_trustworthy} trustworthy"
    duration = round(time.time() - started, 1)

    art = {
        "n_trustworthy_at_0.15": n_trustworthy,
        "per_game_best_energy": per_game_best_energy,
        "total_codex_calls": total_calls,
        "total_codex_seconds": total_csec,
        "markov_vs_hidden_split": {"markov": markov, "hidden_state": hidden},
        "random_seed": seed,
        "honest_verdict": verdict,
        "duration_s": duration,
        "inference_substrate": "offline_arc_agi3_plus_codex_program_synthesis_consistency_verified",
        "per_game": per_game
    }

    if write:
        out_path = REPO / "results" / "experiment_3955_active_codex_nonspatial_sweep.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
        print(f"\n-> {verdict}")

    return art

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default="r11l,sc25,lp85,tn36,dc22,su15")
    ap.add_argument("--train_budget", type=int, default=900)
    ap.add_argument("--test_budget", type=int, default=1400)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    
    run(games=[g.strip() for g in args.games.split(",") if g.strip()],
        train_budget=args.train_budget, test_budget=args.test_budget,
        episodes=args.episodes, iters=args.iters, seed=args.seed)
