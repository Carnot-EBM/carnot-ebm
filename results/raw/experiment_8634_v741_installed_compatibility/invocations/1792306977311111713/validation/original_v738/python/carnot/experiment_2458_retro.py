import os
import json
import glob
from typing import Dict, Any

def generate_retro(results_dir: str) -> Dict[str, Any]:
    """
    Scans the results directory for experiment deliverables related to milestone .237
    and extracts metrics to produce the retrospective summary.
    """
    exp_ids = list(range(2447, 2459))
    
    n_completed = 0
    n_missing = 0
    n_blocked = 0
    n_failed = 0
    
    best_237_verifier_auroc = 0.0
    fr11_satisfied = False
    kv260_synthesis_succeeded = False
    gatemate_bitstream_flashed = False
    polarfire_ssh_reachable = False
    phase1_ship_gate_met = False
    
    task_status = {}
    
    for exp_id in exp_ids:
        pattern = os.path.join(results_dir, f"experiment_{exp_id}*.json")
        matches = glob.glob(pattern)
        if not matches:
            n_missing += 1
            task_status[str(exp_id)] = "missing"
            continue
        
        filename = matches[0]
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except Exception:
            n_failed += 1
            task_status[str(exp_id)] = "failed"
            continue
            
        verdict = str(data.get('honest_verdict', '')).lower()
        if verdict.startswith('complete:') or verdict.startswith('success:') or verdict.startswith('passed:') or verdict.startswith('shipped:'):
            n_completed += 1
            task_status[str(exp_id)] = "completed"
        elif 'blocked' in verdict:
            n_blocked += 1
            task_status[str(exp_id)] = "blocked"
        else:
            n_failed += 1
            task_status[str(exp_id)] = "failed"
            
        if exp_id == 2448:
            best_237_verifier_auroc = data.get('conformal_ensemble_auroc', 0.0)
        elif exp_id == 2451:
            fr11_satisfied = data.get('fr11_satisfied', False)
        elif exp_id == 2452:
            kv260_synthesis_succeeded = data.get('kv260_synthesis_succeeded', False)
        elif exp_id == 2453:
            gatemate_bitstream_flashed = data.get('gatemate_bitstream_flashed', False)
        elif exp_id == 2454:
            polarfire_ssh_reachable = data.get('ssh_reachable', False)
        elif exp_id in (2457, 2441):
            if 'phase1_ship_gate_met' in data:
                phase1_ship_gate_met = data['phase1_ship_gate_met']
                
    if not phase1_ship_gate_met:
        pattern = os.path.join(results_dir, "experiment_2441*.json")
        matches = glob.glob(pattern)
        if matches:
            try:
                with open(matches[0], 'r') as f:
                    data = json.load(f)
                    phase1_ship_gate_met = data.get('phase1_ship_gate_met', phase1_ship_gate_met)
            except Exception:
                pass
                
    auroc_gap = 0.9236 - best_237_verifier_auroc
    
    retro = {
        "honest_verdict": "complete: retro generated.",
        "retro_complete": True,
        "n_experiments_completed": n_completed,
        "n_missing": n_missing,
        "n_blocked": n_blocked,
        "n_failed": n_failed,
        "best_237_verifier_auroc": best_237_verifier_auroc,
        "auroc_gap_to_hive_peer": auroc_gap,
        "fr11_satisfied": fr11_satisfied,
        "kv260_synthesis_succeeded": kv260_synthesis_succeeded,
        "gatemate_bitstream_flashed": gatemate_bitstream_flashed,
        "polarfire_ssh_reachable": polarfire_ssh_reachable,
        "phase1_ship_gate_met": phase1_ship_gate_met,
        "top_3_successes": [
            f"1. Achieved AUROC {best_237_verifier_auroc} with gap {auroc_gap:.4f}.",
            f"2. FR11 satisfied: {fr11_satisfied}. Phase 1 ship gate met: {phase1_ship_gate_met}.",
            f"3. Hardware progress: KV260 synth {kv260_synthesis_succeeded}, GateMate flash {gatemate_bitstream_flashed}, PolarFire SSH {polarfire_ssh_reachable}."
        ],
        "top_3_gaps_for_238": [
            "1. Bridge remaining AUROC gap to Hive peer (target > 0.9236).",
            "2. Resolve any pending unverified hardware deployments or tests.",
            "3. Full system integration and multi-node stabilization."
        ],
        "task_status": task_status
    }
    
    return retro

def write_retro(results_dir: str, out_filename: str = 'experiment_2458_retro_v237.json'):
    retro = generate_retro(results_dir)
    out_path = os.path.join(results_dir, out_filename)
    with open(out_path, 'w') as f:
        json.dump(retro, f, indent=2)
    return out_path
