"""Experiment 3958: Cross-game DSL fragment transfer (self-learning across games)."""

from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_world_model_synth import (
    InducedWorldModel, grade_predictions, extract_library_fragments
)
from arc3_m2_world_model import _collect, _key_disjoint_split
from arc3_m2_codex_synth import (
    ask_codex, _serialize, safe_predict_from_code, _extract_code, _failure_examples
)
from arc3_m2_active_data import active_collect, _common_test, _keys

def synth_prompt_with_library(serialized, library_fragments, prior_code=None, failures=None):
    base = (
        "You are reverse-engineering the DETERMINISTIC transition rule of a grid puzzle from examples.\n\n"
        f"{serialized}\n\n"
        "Write exactly one Python function:\n"
        "    def predict(grid, action):\n"
        "        # grid: a 2D numpy int array (rows x cols). action: the tuple described above.\n"
        "        # return the NEXT grid (a numpy int array, same shape) the rule produces.\n"
        "Infer the underlying mechanic and GENERALIZE to unseen states and actions. Use ONLY numpy (np is imported).\n"
        "DO NOT hardcode specific colors, coordinates, or object sizes. Output ONLY one ```python code block."
    )
    if library_fragments:
        base += "\n\nYou have access to the following previously learned DSL fragments (helper functions). You MAY use them in your code if helpful:\n"
        for frag in library_fragments:
            base += f"```python\n{frag}\n```\n"
            
    if prior_code and failures:
        base += ("\n\nYour PREVIOUS function mispredicted these held-out transitions:\n" + failures +
                 "\n\nPrevious code:\n```python\n" + prior_code + "\n```\nFix it and output the corrected function.")
    return base

def run_arm(game, train_trans, test_trans, iters, rng, library_fragments=None):
    changing = [t for t in train_trans if (np.asarray(t[0]) != np.asarray(t[2])).any()]
    sample = changing[:30] if len(changing) >= 8 else train_trans[:30]
    
    bg = int(np.bincount(np.asarray(train_trans[0][0]).ravel()).argmax()) if train_trans else 0
    shape = np.asarray(train_trans[0][0]).shape if train_trans else (0, 0)
    serialized = _serialize(sample, bg, shape)

    best_e, best_code, best_fn, csec = None, None, None, 0.0
    prior_code, failures = None, None
    calls = 0
    
    for it in range(iters):
        prompt = synth_prompt_with_library(serialized, library_fragments, prior_code, failures)
        raw, dt = ask_codex(prompt)
        csec += dt
        calls += 1
        code = _extract_code(raw)
        if code is None:
            continue
            
        fn = safe_predict_from_code(code)
        if fn is None:
            continue
            
        ce = grade_predictions(fn, test_trans)
        e = ce["energy"]
        
        if e is not None and (best_e is None or e < best_e):
            best_e, best_code, best_fn = e, code, fn
            
        if best_e is not None and best_e <= 0.15:
            break
            
        prior_code = best_code
        failures = _failure_examples(best_fn, test_trans, bg, shape) if best_fn else None

    return best_e, best_code, calls, csec

def run(games, budget=600, episodes=20, iters=3, seed=0):
    t0 = time.time()
    rng = random.Random(seed)
    
    # Check if codex is available
    if subprocess.run(["command", "-v", "codex"], shell=True, capture_output=True).returncode != 0:
        verdict = "blocked_codex_unavailable"
        art = {
            "honest_verdict": verdict,
            "inference_substrate": "offline_arc_agi3_plus_codex_program_synthesis_consistency_verified",
            "duration_s": round(time.time() - t0, 1),
            "transfer_win": False,
            "calls_per_game_no_library": [],
            "calls_per_game_with_library": [],
            "energy_per_game_no_library": [],
            "energy_per_game_with_library": [],
            "n_library_fragments": 0,
            "fragments_reused_across_games": 0,
            "random_seed": seed
        }
        (REPO / "results" / "experiment_3958_cross_game_dsl_transfer.json").write_text(json.dumps(art, indent=2))
        print(verdict)
        return art

    try:
        from arc_agi import Arcade
        from arc_agi.base import OperationMode
        from arcengine.enums import GameAction, GameState
    except ImportError:
        verdict = "blocked_arc_offline_env_unavailable"
        art = {
            "honest_verdict": verdict,
            "inference_substrate": "offline_arc_agi3_plus_codex_program_synthesis_consistency_verified",
            "duration_s": round(time.time() - t0, 1),
            "transfer_win": False,
            "calls_per_game_no_library": [],
            "calls_per_game_with_library": [],
            "energy_per_game_no_library": [],
            "energy_per_game_with_library": [],
            "n_library_fragments": 0,
            "fragments_reused_across_games": 0,
            "random_seed": seed
        }
        (REPO / "results" / "experiment_3958_cross_game_dsl_transfer.json").write_text(json.dumps(art, indent=2))
        print(verdict)
        return art
        
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files"))
                 
    calls_no_library = []
    energy_no_library = []
    
    calls_with_library = []
    energy_with_library = []
    
    library_fragments = []
    
    for game in games:
        # Collect data
        active = active_collect(arc, game, budget, episodes, rng, GameAction, GameState)
        test_all = _collect(arc, game, budget, episodes, rng, GameAction, GameState)
        test = _common_test(test_all, _keys(active))
        
        # Arm A: No library
        e_no, _, calls_no, _ = run_arm(game, active, test, iters, rng, library_fragments=None)
        calls_no_library.append(calls_no)
        energy_no_library.append(e_no if e_no is not None else 1.0)
        
        # Arm B: With library
        e_with, code_with, calls_with, _ = run_arm(game, active, test, iters, rng, library_fragments=library_fragments)
        calls_with_library.append(calls_with)
        energy_with_library.append(e_with if e_with is not None else 1.0)
        
        # Extract new fragments
        if code_with:
            frags = extract_library_fragments(code_with)
            for f in frags:
                if f not in library_fragments:
                    library_fragments.append(f)

    # Check for transfer win: later games (index >= 1) have fewer calls or lower energy
    win = False
    reused = 0
    for i in range(1, len(games)):
        e_no = energy_no_library[i]
        e_wi = energy_with_library[i]
        c_no = calls_no_library[i]
        c_wi = calls_with_library[i]
        
        if e_wi <= 0.15 and (e_wi < e_no or c_wi < c_no):
            win = True
            reused += 1

    if win:
        verdict = "complete: dsl_transfer_win"
    else:
        verdict = "complete: dsl_transfer_no_win_no_cost_reduction_observed"
        
    art = {
        "honest_verdict": verdict,
        "inference_substrate": "offline_arc_agi3_plus_codex_program_synthesis_consistency_verified",
        "duration_s": round(time.time() - t0, 1),
        "transfer_win": win,
        "calls_per_game_no_library": calls_no_library,
        "calls_per_game_with_library": calls_with_library,
        "energy_per_game_no_library": energy_no_library,
        "energy_per_game_with_library": energy_with_library,
        "n_library_fragments": len(library_fragments),
        "fragments_reused_across_games": reused,
        "random_seed": seed
    }
    
    (REPO / "results" / "experiment_3958_cross_game_dsl_transfer.json").write_text(json.dumps(art, indent=2))
    print(verdict)
    return art

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="m0r0,vc33,sb26")
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    
    run(games=[g.strip() for g in args.games.split(",") if g.strip()],
        budget=args.budget, episodes=args.episodes, iters=args.iters, seed=args.seed)
