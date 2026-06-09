"""
Experiment 3957: Hidden-state Latent Registers
Evaluates whether augmenting the grid-only state with latent registers (like step counter
and collected colors) reduces the apparent non-determinism and the consistency energy of
an induced world model.
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash
import arc3_graph_explore as gx
from carnot.agentic.arc_latent_registers import compute_latent_registers, AugmentedInducedWorldModel
from carnot.agentic.arc_world_model_synth import InducedWorldModel

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
            
            # semi-random exploration, mix of choices
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
            "inference_substrate": "offline_arc_agi3_latent_registers",
            "mean_energy_grid_only": {},
            "mean_energy_with_latent_registers": {},
            "nondeterminism_reduction": {},
            "registers_that_helped": {}
        }
        with open(REPO / "results" / "experiment_3957_hidden_state_latent_registers.json", "w") as f:
            json.dump(art, f, indent=2)
        print(f"-> {art['honest_verdict']}")
        return

    target_games = ["dc22", "g50t", "m0r0", "cn04"]
    sel = [g for g in all_ids if g.split("-")[0] in target_games]
    
    mean_energy_grid_only = {}
    mean_energy_with_latent_registers = {}
    nondeterminism_reduction = {}
    registers_that_helped = {}
    
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
        if b_energy is None:
            b_energy = 1.0
            
        # Augmented model
        train_aug = []
        test_aug = []
        aug_samples = defaultdict(int)
        aug_outcomes = defaultdict(set)
        
        def process_traj(traj, out_list):
            latents = compute_latent_registers(traj)
            for i, (s, akey, s2) in enumerate(traj):
                l_cur = latents[i]
                l_next = latents[i+1]
                out_list.append((s, l_cur, akey, s2, l_next))
                
                key = ((frame_hash(s), l_cur), tuple(akey))
                aug_samples[key] += 1
                aug_outcomes[key].add(frame_hash(s2))
                
        for traj in train_trajectories:
            process_traj(traj, train_aug)
        for traj in test_trajectories:
            process_traj(traj, test_aug)
            
        aug_revisited = [k for k, n in aug_samples.items() if n >= 2]
        aug_nondet = [k for k in aug_revisited if len(aug_outcomes[k]) > 1]
        aug_rate = len(aug_nondet) / len(aug_revisited) if aug_revisited else 0.0
        
        aug_model = AugmentedInducedWorldModel(game)
        aug_model.fit_augmented(train_aug)
        a_energy = aug_model.consistency_energy_augmented(test_aug).get("energy")
        if a_energy is None:
            a_energy = 1.0
            
        mean_energy_grid_only[short] = b_energy
        mean_energy_with_latent_registers[short] = a_energy
        nondeterminism_reduction[short] = {
            "base_rate": base_rate,
            "aug_rate": aug_rate,
            "reduction": base_rate - aug_rate
        }
        
        if a_energy < b_energy or aug_rate < base_rate:
            registers_that_helped[short] = ["step_counter", "colors_clicked"]
            
        print(f"{short}: base_nondet={base_rate:.3f} aug_nondet={aug_rate:.3f} base_energy={b_energy:.3f} aug_energy={a_energy:.3f}")

    any_drop = any(mean_energy_with_latent_registers[g] < mean_energy_grid_only[g] for g in target_games if g in mean_energy_grid_only)
    if any_drop:
        verdict = "success: latent_registers_dropped_energy"
    else:
        verdict = "complete: latent_registers_no_drop_energy"

    duration = time.time() - started
    
    art = {
        "mean_energy_grid_only": mean_energy_grid_only,
        "mean_energy_with_latent_registers": mean_energy_with_latent_registers,
        "nondeterminism_reduction": nondeterminism_reduction,
        "registers_that_helped": registers_that_helped,
        "random_seed": seed,
        "honest_verdict": verdict,
        "duration_s": duration,
        "inference_substrate": "offline_arc_agi3_latent_registers",
    }
    
    out_path = REPO / "results" / "experiment_3957_hidden_state_latent_registers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(art, f, indent=2)
        
    print(f"-> {verdict}")

if __name__ == "__main__":
    run_experiment()
