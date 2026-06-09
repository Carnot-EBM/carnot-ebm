import sys
import numpy as np
import random
import time
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_agi3_world_model import grid_of, objects
from carnot.agentic.arc_agi3_goal_induction import induce_goal_predicate
from arc_agi import Arcade
from arc_agi.base import OperationMode
from arcengine.enums import GameAction, GameState
import arc3_graph_explore as gx

def run(seed=42):
    t0 = time.time()
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=str(REPO / "environment_files"))
    game_id = "r11l-495a7899"
    env = arc.make(game_id)
    by_id = {a.value: a for a in GameAction}
    rng = random.Random(seed)
    
    win_grids = []
    non_win_grids = []
    
    # Collect 4 wins
    ep = 0
    while len(win_grids) < 4 and ep < 100:
        f = env.reset()
        for step in range(60):
            av = getattr(f, "available_actions", [])
            if not av: break
            
            # Record some non-win grids (start of episode and intermediate)
            if step == 0 or step == 20:
                grid = grid_of(f)
                non_win_grids.append(grid)
                
            cands = gx._candidate_akeys(grid_of(f), av)
            if not cands: break
            akey = rng.choice(cands)
            a_int = akey[0]
            data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
            f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
            
            lc = int(getattr(f, 'levels_completed', 0) or 0)
            st = getattr(f, 'state', None)
            
            if lc > 0:
                frames = f.frame if hasattr(f, "frame") else [f]
                if isinstance(frames, list) and len(frames) >= 2:
                    win_grid = np.array(frames[-2])
                    win_grids.append(win_grid)
                elif isinstance(frames, np.ndarray) and frames.ndim == 3 and frames.shape[0] >= 2:
                    win_grid = frames[-2]
                    win_grids.append(win_grid)
                break
            if st in (GameState.WIN, GameState.GAME_OVER):
                break
        ep += 1

    if len(win_grids) < 4:
        art = {
            "honest_verdict": f"blocked_arc_offline_env_unavailable: not enough level-ups collected ({len(win_grids)})",
            "goal_predicate_precision": 0.0,
            "goal_predicate_recall": 0.0,
            "n_level_ups_observed": len(win_grids),
            "games_covered": [game_id],
            "random_seed": seed,
            "duration_s": round(time.time() - t0, 2),
            "inference_substrate": "offline_arc_agi3"
        }
        with open(REPO / "results" / "experiment_3956_goal_predicate_induction.json", "w") as f:
            json.dump(art, f, indent=2)
        return art

    train_win = win_grids[:2]
    train_non_win = non_win_grids[:5]
    
    test_win = win_grids[2:]
    test_non_win = non_win_grids[5:]
    
    pred = induce_goal_predicate(train_win, train_non_win)
    if pred is None:
        precision = 0.0
        recall = 0.0
        honest_verdict = "complete: failed_to_induce_predicate"
    else:
        true_pos = sum(1 for g in test_win if pred(g))
        false_neg = len(test_win) - true_pos
        false_pos = sum(1 for g in test_non_win if pred(g))
        true_neg = len(test_non_win) - false_pos
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        honest_verdict = f"complete: precision_{precision:.2f}_recall_{recall:.2f}"

    art = {
        "goal_predicate_precision": precision,
        "goal_predicate_recall": recall,
        "n_level_ups_observed": len(train_win),
        "games_covered": [game_id],
        "random_seed": seed,
        "honest_verdict": honest_verdict,
        "duration_s": round(time.time() - t0, 2),
        "inference_substrate": "offline_arc_agi3_heuristic_inducer"
    }

    with open(REPO / "results" / "experiment_3956_goal_predicate_induction.json", "w") as f:
        json.dump(art, f, indent=2)
        
    print(json.dumps(art, indent=2))
    return art

if __name__ == "__main__":
    run()
