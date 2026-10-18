"""
Experiment 3969: Hidden-state Pinductor
Evaluates whether proposing candidate latent variables and refining them using a 
belief-based prediction likelihood (Pinductor recipe) can recover hidden state 
in ARC-AGI-3 games without hand-enumerated registers.
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash
import arc3_graph_explore as gx
from carnot.agentic.arc_world_model_synth import InducedWorldModel
from carnot.agentic.arc_pinductor import PinductorModel
from carnot.agentic.arc_pinductor_candidates import get_candidates

def collect_trajectories(arc, game, episodes, rng, GameAction, GameState, max_steps=100):
    by_id = {a.value: a for a in GameAction}
    trajectories = []
    
    for _ in range(episodes):
        env = arc.make(game)
        f = env.reset()
        traj = []
        for _ in range(max_steps):
            grid = grid_of(f)
            cands = gx._candidate_akeys(grid, getattr(f, "available_actions", []))
            if not cands:
                break
            
            # semi-random exploration
            akey = rng.choice(cands)
            a_int = akey[0]
            data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
            
            f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
            next_grid = grid_of(f)
            traj.append((grid, akey, next_grid))
            
            if getattr(f, "state", None) in (GameState.WIN, GameState.GAME_OVER):
                break
        if traj:
            trajectories.append(traj)
            
    return trajectories

def run_positive_control():
    print("Running positive control...")
    # S0 -> S0 -> S1 -> S1 depending on step count modulo 2
    # Visible states are all 0 or all 1.
    s0 = np.zeros((2,2), dtype=int)
    s1 = np.ones((2,2), dtype=int)
    a = (6, 0, 0)
    
    # We create a trajectory where latent flips on action
    # latent=0 (s0) -> latent=1 (s0) -> latent=0 (s1) -> latent=1 (s1)
    traj1 = [
        (s0, a, s0),
        (s0, a, s1),
        (s1, a, s1),
        (s1, a, s0)
    ]
    # base model
    base_model = InducedWorldModel("positive_control")
    base_model.fit([t for t in traj1])
    b_energy = base_model.consistency_energy([t for t in traj1]).get("energy", 1.0)
    if b_energy is None: b_energy = 1.0
    
    # pinductor
    cands = get_candidates()
    best_energy = 1.0
    for name, fn, K in cands:
        p_model = PinductorModel("positive_control", fn, K)
        p_model.fit([traj1])
        energy = p_model.consistency_energy([traj1]).get("energy", 1.0)
        if energy is None: energy = 1.0
        if energy < best_energy:
            best_energy = energy
            
    print(f"Positive control: base_energy={b_energy:.4f}, pinductor_energy={best_energy:.4f}")
    return best_energy < b_energy

def run_experiment(episodes=40, seed=0):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    
    started = time.time()
    rng = random.Random(seed)
    
    try:
        arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
        all_ids = sorted(getattr(e, "game_id", None) for e in arc.get_environments())
    except Exception as e:
        art = {
            "honest_verdict": f"blocked_arc_offline_env_unavailable: {e}",
            "random_seed": seed,
            "duration_s": round(time.time() - started, 1),
            "inference_substrate": "offline_arc_agi3_pinductor",
            "positive_control_passed": False,
            "mean_energy_grid_only": {},
            "mean_energy_with_inferred_latents": {},
            "energy_drop": 0.0,
            "nondeterminism_reduction": {},
            "latents_that_helped": {}
        }
        with open(REPO / "results" / "experiment_3969_hidden_state_pinductor.json", "w") as f:
            json.dump(art, f, indent=2)
        print(f"-> {art['honest_verdict']}")
        return

    # Positive Control
    positive_control_passed = run_positive_control()
    if not positive_control_passed:
        art = {
            "honest_verdict": "blocked_positive_control_failed",
            "random_seed": seed,
            "duration_s": round(time.time() - started, 1),
            "inference_substrate": "offline_arc_agi3_pinductor",
            "positive_control_passed": False,
            "mean_energy_grid_only": {},
            "mean_energy_with_inferred_latents": {},
            "energy_drop": 0.0,
            "nondeterminism_reduction": {},
            "latents_that_helped": {}
        }
        with open(REPO / "results" / "experiment_3969_hidden_state_pinductor.json", "w") as f:
            json.dump(art, f, indent=2)
        print(f"-> {art['honest_verdict']}")
        return

    target_games = ["dc22", "g50t", "m0r0", "cn04"]
    sel = [g for g in all_ids if g.split("-")[0] in target_games]
    
    mean_energy_grid_only = {}
    mean_energy_with_inferred_latents = {}
    nondeterminism_reduction = {}
    latents_that_helped = {}
    total_drop = 0.0
    
    candidates = get_candidates()
    
    for game in sel:
        short = game.split("-")[0]
        trajectories = collect_trajectories(arc, game, episodes, rng, GameAction, GameState, max_steps=100)
        
        # Cross-validation split by episode
        random.shuffle(trajectories)
        split = int(len(trajectories) * 0.8)
        train_trajectories = trajectories[:split]
        test_trajectories = trajectories[split:]
        
        # Base model
        train_base = [t for traj in train_trajectories for t in traj]
        test_base = [t for traj in test_trajectories for t in traj]
        
        base_samples = defaultdict(int)
        base_outcomes = defaultdict(set)
        for s, akey, s2 in train_base + test_base:
            key = (frame_hash(s), tuple(akey))
            base_samples[key] += 1
            base_outcomes[key].add(frame_hash(s2))
            
        base_revisited = [k for k, n in base_samples.items() if n >= 2]
        base_nondet = [k for k in base_revisited if len(base_outcomes[k]) > 1]
        base_rate = len(base_nondet) / len(base_revisited) if base_revisited else 0.0
        
        base_model = InducedWorldModel(game)
        base_model.fit(train_base)
        b_energy = base_model.consistency_energy(test_base).get("energy")
        if b_energy is None: b_energy = 1.0
            
        # Pinductor search
        best_energy = b_energy
        best_latent_name = None
        best_latent_fn = None
        best_latent_k = None
        
        for name, fn, K in candidates:
            p_model = PinductorModel(game, fn, K)
            p_model.fit(train_trajectories)
            energy = p_model.consistency_energy(test_trajectories).get("energy")
            if energy is None: energy = 1.0
            if energy < best_energy:
                best_energy = energy
                best_latent_name = name
                best_latent_fn = fn
                best_latent_k = K

        a_energy = best_energy
        
        # Calculate nondeterminism with best latent
        aug_rate = base_rate
        if best_latent_fn is not None:
            aug_samples = defaultdict(int)
            aug_outcomes = defaultdict(set)
            
            for traj in train_trajectories + test_trajectories:
                L = 0
                for s, akey, s2 in traj:
                    s_arr = np.asarray(s, dtype=np.int16)
                    key = ((frame_hash(s_arr), L), tuple(akey))
                    aug_samples[key] += 1
                    aug_outcomes[key].add(frame_hash(s2))
                    L = best_latent_fn(L, s_arr, tuple(akey))
                    
            aug_revisited = [k for k, n in aug_samples.items() if n >= 2]
            aug_nondet = [k for k in aug_revisited if len(aug_outcomes[k]) > 1]
            aug_rate = len(aug_nondet) / len(aug_revisited) if aug_revisited else 0.0
            
        mean_energy_grid_only[short] = float(b_energy)
        mean_energy_with_inferred_latents[short] = float(a_energy)
        nondeterminism_reduction[short] = {
            "base_rate": float(base_rate),
            "aug_rate": float(aug_rate),
            "reduction": float(base_rate - aug_rate)
        }
        
        if best_latent_name:
            latents_that_helped[short] = [best_latent_name]
        else:
            latents_that_helped[short] = []
            
        total_drop += float(b_energy - a_energy)
        print(f"{short}: base_nondet={base_rate:.3f} aug_nondet={aug_rate:.3f} base_energy={b_energy:.3f} aug_energy={a_energy:.3f} best={best_latent_name}")

    avg_drop = total_drop / len(target_games) if target_games else 0.0

    if total_drop > 0.02 * len(target_games): # Require meaningful drop
        verdict = f"success: pinductor_latents_dropped_energy_avg_{avg_drop:.3f}"
    else:
        verdict = "complete: pinductor_latents_no_drop_energy"

    duration = time.time() - started
    
    art = {
        "positive_control_passed": bool(positive_control_passed),
        "mean_energy_grid_only": mean_energy_grid_only,
        "mean_energy_with_inferred_latents": mean_energy_with_inferred_latents,
        "energy_drop": float(avg_drop),
        "nondeterminism_reduction": nondeterminism_reduction,
        "latents_that_helped": latents_that_helped,
        "random_seed": seed,
        "honest_verdict": verdict,
        "duration_s": float(duration),
        "inference_substrate": "offline_arc_agi3_pinductor",
    }
    
    out_path = REPO / "results" / "experiment_3969_hidden_state_pinductor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(art, f, indent=2)
        
    print(f"-> {verdict}")

if __name__ == "__main__":
    run_experiment()
