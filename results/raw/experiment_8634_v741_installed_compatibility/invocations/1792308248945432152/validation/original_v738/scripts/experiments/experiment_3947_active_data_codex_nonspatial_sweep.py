import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from arc3_m2_active_codex import codex_best_energy
from arc3_m2_active_data import active_collect, _common_test, _keys
from arc3_m2_world_model import _collect

def run(games, train_budget=900, test_budget=1400, episodes=32, iters=3, seed=0, write=True):
    started = time.time()
    
    art = {
        "experiment": "experiment_3947",
        "title": "experiment_3947_active_data_codex_nonspatial_sweep",
        "honest_verdict": "",
        "inference_substrate": "offline_arc_agi3_plus_codex_program_synthesis_consistency_verified",
        "n_games": len(games),
        "games": games,
        "n_trustworthy_at_0.15": 0,
        "per_game_best_energy": {},
        "total_codex_calls": 0,
        "total_codex_seconds": 0.0,
        "duration_s": 0.0,
        "random_seed": seed,
        "no_gpu_used": True,
        "submitted_to_leaderboard": False,
    }

    if not shutil.which("codex"):
        art["honest_verdict"] = "blocked_codex_unavailable"
        art["duration_s"] = round(time.time() - started, 1)
        if write:
            out_path = REPO / "results" / "experiment_3947_active_data_codex_nonspatial_sweep.json"
            out_path.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
        return art

    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState

    rng = random.Random(seed)
    try:
        arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                     environments_dir=str(REPO / "environment_files"))
    except Exception as e:
        art["honest_verdict"] = f"blocked_offline_env_load_failed: {e}"
        art["duration_s"] = round(time.time() - started, 1)
        if write:
            out_path = REPO / "results" / "experiment_3947_active_data_codex_nonspatial_sweep.json"
            out_path.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
        return art
    
    all_ids = sorted(getattr(e, "game_id", None) for e in arc.get_environments())
    sel = [g for g in all_ids if g.split("-")[0] in set(games)]
    
    per_game = []
    total_csec = 0.0
    total_calls = 0
    
    for game in sel:
        short = game.split("-")[0]
        test_all = _collect(arc, game, test_budget, episodes, rng, GameAction, GameState)
        active = active_collect(arc, game, train_budget, episodes, rng, GameAction, GameState)
        test = _common_test(test_all, _keys(active))
        
        best_e, hist, csec = codex_best_energy(active, test, iters, rng)
        total_csec += csec
        total_calls += len(hist)
        
        trustworthy = best_e is not None and best_e <= 0.15
        
        per_game.append({
            "game": short,
            "best_energy": best_e,
            "trustworthy": trustworthy,
            "diff_from_vc33_0.005": round(best_e - 0.005, 4) if best_e is not None else None,
            "history": hist,
            "codex_seconds": csec
        })
        print(f"  {short:6s} active codex_best={best_e} trustworthy={trustworthy}", flush=True)

    n_trustworthy = sum(1 for g in per_game if g["trustworthy"])
    verdict = f"complete: nonspatial_sweep_trustworthy_{n_trustworthy}of{len(games)}"
    
    art.update({
        "honest_verdict": verdict,
        "n_trustworthy_at_0.15": n_trustworthy,
        "per_game_best_energy": {g["game"]: g["best_energy"] for g in per_game},
        "per_game_details": per_game,
        "total_codex_calls": total_calls,
        "total_codex_seconds": round(total_csec, 1),
        "duration_s": round(time.time() - started, 1),
    })
    
    if write:
        out_path = REPO / "results" / "experiment_3947_active_data_codex_nonspatial_sweep.json"
        out_path.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
        
    print(f"\n-> {verdict}")
    return art

if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="r11l,sc25,lp85,tn36,dc22,su15")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    run(games=[g.strip() for g in args.games.split(",")], iters=args.iters)
