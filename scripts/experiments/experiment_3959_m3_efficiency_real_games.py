import json
import math
import random
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, List

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")

@dataclass
class RatioConfidenceInterval:
    low: float
    high: float

def bootstrap_ratio_ci(
    data: List[float],
    random_seed: int,
    resamples: int = 1000,
) -> RatioConfidenceInterval:
    rng = random.Random(random_seed)
    n = len(data)
    means = []
    for _ in range(resamples):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_index = int(0.025 * (resamples - 1))
    high_index = int(0.975 * (resamples - 1))
    return RatioConfidenceInterval(low=means[low_index], high=means[high_index])

def get_objects_and_target_area(grid_arr, target_y, target_x):
    import numpy as np
    arr = np.array(grid_arr)
    if arr.ndim == 3:
        arr = arr[-1]
    arr = arr.astype(np.int16)
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[counts.argmax()])
    mask = arr != bg
    
    h, w = arr.shape
    seen = np.zeros_like(mask, dtype=bool)
    objects = []
    target_area = 1 # fallback
    
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                objects.append(cells)
                # Check if target is in this object
                if any(cy == target_y and cx == target_x for cy, cx in cells):
                    target_area = len(cells)
                    
    num_objects = max(1, len(objects))
    return num_objects, target_area, h * w

def simulate_geometric(p, rng):
    if p >= 1.0:
        return 1
    if p <= 0.0:
        return 10000
    # Number of trials needed to get 1 success (including the success itself)
    # p is success probability
    u = rng.random()
    return int(math.ceil(math.log(1.0 - u) / math.log(1.0 - p)))

def process_game(game_id, solve_log, arc, baseline_actions, rng):
    env = arc.make(game_id)
    f = env.reset()
    
    # We will compute the expected p_with and p_without for each action in the log
    steps_info = []
    
    from arcengine.enums import GameAction
    
    for entry in solve_log:
        grid = f.frame
        if "piece" in entry and "target" in entry:
            # r11l case: 2 clicks per log entry
            py, px = entry["piece"]
            ty, tx = entry["target"]
            
            m1, a1, total_pixels = get_objects_and_target_area(grid, py, px)
            steps_info.append((1.0 / m1, a1 / total_pixels))
            f = env.step(GameAction.ACTION6, data={"x": int(px), "y": int(py)})
            
            grid2 = f.frame
            m2, a2, total_pixels = get_objects_and_target_area(grid2, ty, tx)
            steps_info.append((1.0 / m2, a2 / total_pixels))
            f = env.step(GameAction.ACTION6, data={"x": int(tx), "y": int(ty)})
        elif "y" in entry and "x" in entry:
            # lp85 case: 1 click
            y, x = entry["y"], entry["x"]
            m1, a1, total_pixels = get_objects_and_target_area(grid, y, x)
            steps_info.append((1.0 / m1, a1 / total_pixels))
            f = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
            
    # Now simulate 1000 episodes
    ratios_with = []
    ratios_without = []
    
    # Assume 1 level solved for this game, using the first baseline action
    b = baseline_actions[0] if baseline_actions else 60
    
    for _ in range(1000):
        actions_with = 0
        actions_without = 0
        for p_with, p_without in steps_info:
            actions_with += simulate_geometric(p_with, rng)
            actions_without += simulate_geometric(p_without, rng)
        
        ratios_with.append(actions_with / b)
        ratios_without.append(actions_without / b)
        
    return ratios_with, ratios_without

def run():
    started = time.time()
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    
    results_dir = REPO / "results"
    
    # Load solved games
    games_to_process = []
    
    f1 = results_dir / "experiment_3954_second_game_solve.json"
    if f1.exists():
        d1 = json.loads(f1.read_text())
        if d1.get("ACCURACY_levels_solved", 0) > 0:
            games_to_process.append((d1["game_solved"], d1["solve_log"], [60])) # lp85 baseline is 17 usually but fallback 60
            
    f2 = results_dir / "experiment_3946_r11l_first_solve.json"
    if f2.exists():
        d2 = json.loads(f2.read_text())
        if d2.get("solved", False):
            # The log might be split by level
            # We just take the first level for measurement
            log0 = [e for e in d2["solve_log"] if e.get("level", 0) == 0]
            games_to_process.append((d2["game"], log0, [60]))
            
    f3 = results_dir / "experiment_3953_r11l_full_solve.json"
    if f3.exists():
        d3 = json.loads(f3.read_text())
        # If it has a solve_log we can use it, else skip
        if "solve_log" in d3 and d3.get("ACCURACY_levels_solved", 0) > 0:
            log0 = [e for e in d3["solve_log"] if e.get("level", 0) == 0]
            games_to_process.append((d3["game_solved"], log0, [60]))
            
    if not games_to_process:
        print("No solved games found. Exiting with blocked verdict.")
        verdict = "blocked_no_solved_real_game"
        art = {
            "honest_verdict": verdict,
            "experiment": "experiment_3959_m3_efficiency_real_games",
        }
        Path(results_dir / "experiment_3959_m3_efficiency_real_games.json").write_text(json.dumps(art, indent=2))
        return

    rng = random.Random(42)
    all_ratios_with = []
    all_ratios_without = []
    
    for gid, log, b in games_to_process:
        # fetch actual baseline if possible
        env = arc.make(gid)
        base = getattr(env, "baseline_actions", b)
        
        rw, rwo = process_game(gid, log, arc, base, rng)
        all_ratios_with.extend(rw)
        all_ratios_without.extend(rwo)

    # We have arrays of length (1000 * num_games)
    mean_with = sum(all_ratios_with) / len(all_ratios_with)
    mean_without = sum(all_ratios_without) / len(all_ratios_without)
    
    efficiency_ratio = mean_without / mean_with if mean_with > 0 else 0
    
    ci_with = bootstrap_ratio_ci(all_ratios_with, 42)
    ci_without = bootstrap_ratio_ci(all_ratios_without, 43)
    
    pruner_helps = ci_without.low > ci_with.high
    
    verdict = "complete: m3_efficiency_real_games_pruner_helps" if pruner_helps else "complete: m3_efficiency_inconclusive_on_real_games_overlapping_ci"
    
    art = {
        "experiment": "experiment_3959_m3_efficiency_real_games",
        "title": "arc3_m3_efficiency_real_games",
        "honest_verdict": verdict,
        "inference_substrate": "offline_air_gapped_arc_agi3_local_environments",
        "random_seed": 42,
        "n_solved_levels_measured": len(games_to_process),
        "games_measured": [g[0] for g in games_to_process],
        "efficiency_ratio_with_over_without": round(efficiency_ratio, 3),
        "ci95_with": {"low": round(ci_with.low, 3), "high": round(ci_with.high, 3)},
        "ci95_without": {"low": round(ci_without.low, 3), "high": round(ci_without.high, 3)},
        "cis_non_overlapping_pruner_helps": pruner_helps,
        "duration_s": round(time.time() - started, 1)
    }
    
    outfile = results_dir / "experiment_3959_m3_efficiency_real_games.json"
    outfile.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"-> {verdict}")
    print(f"   efficiency_ratio: {efficiency_ratio:.3f}")
    print(f"   WITH CI: [{ci_with.low:.3f}, {ci_with.high:.3f}]")
    print(f"   WITHOUT CI: [{ci_without.low:.3f}, {ci_without.high:.3f}]")

if __name__ == "__main__":
    run()
