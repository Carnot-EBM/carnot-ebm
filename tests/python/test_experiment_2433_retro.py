import os
import json

def test_retro_json_schema():
    RESULTS_DIR = '/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results'
    path = os.path.join(RESULTS_DIR, 'experiment_2433_retro_v235.json')
    
    assert os.path.exists(path), "Deliverable JSON must exist."
    
    with open(path, 'r') as f:
        data = json.load(f)
        
    assert 'honest_verdict' in data, "honest_verdict is required"
    assert data['honest_verdict'].startswith('complete:'), "honest_verdict must start with 'complete:'"
    
    assert 'retro_complete' in data, "retro_complete is required"
    assert data['retro_complete'] is True, "retro_complete must be true"
    
    assert 'n_experiments_completed' in data, "n_experiments_completed is required"
    assert isinstance(data['n_experiments_completed'], int), "n_experiments_completed must be an int"
    
    assert 'n_failed' in data, "n_failed is required"
    assert isinstance(data['n_failed'], int), "n_failed must be an int"
    
    assert 'n_gate_blocks' in data, "n_gate_blocks is required"
    assert isinstance(data['n_gate_blocks'], int), "n_gate_blocks must be an int"
    
    assert 'codex_cli_healthy' in data, "codex_cli_healthy is required"
    assert 'best_235_verifier_auroc' in data, "best_235_verifier_auroc is required"
    assert 'auroc_gap_to_hive_peer_at_235_close' in data, "auroc_gap_to_hive_peer_at_235_close is required"
    assert 'fr11_satisfied' in data, "fr11_satisfied is required"
    assert 'kv260_yosys_succeeded' in data, "kv260_yosys_succeeded is required"
    assert 'phase1_ship_gate_met' in data, "phase1_ship_gate_met is required"
    assert 'best_sampler_kl_delta' in data, "best_sampler_kl_delta is required"
    
    assert 'task_outcomes' in data, "task_outcomes is required"
    assert isinstance(data['task_outcomes'], list), "task_outcomes must be a list"
    
    assert 'top_3_gaps_for_236' in data, "top_3_gaps_for_236 is required"
    assert isinstance(data['top_3_gaps_for_236'], list), "top_3_gaps_for_236 must be a list"
    
    assert 'milestone' in data, "milestone is required"
    assert data['milestone'] == '2026.05.235', "milestone must be 2026.05.235"
    
    assert 'generated_at' in data, "generated_at is required"
    assert 'duration_s' in data, "duration_s is required"

if __name__ == '__main__':
    test_retro_json_schema()
    print("Test passed.")
