import os
import json
from carnot.experiment_2458_retro import generate_retro, write_retro

def test_retro_json_schema():
    RESULTS_DIR = '/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/results'
    
    # Generate the retro using the module
    path = write_retro(RESULTS_DIR, 'experiment_2458_retro_v237.json')
    
    assert os.path.exists(path), "Deliverable JSON must exist."
    
    with open(path, 'r') as f:
        data = json.load(f)
        
    assert 'honest_verdict' in data, "honest_verdict is required"
    assert data['honest_verdict'].startswith('complete:'), "honest_verdict must start with 'complete:'"
    
    assert 'retro_complete' in data, "retro_complete is required"
    assert data['retro_complete'] is True, "retro_complete must be true"
    
    assert 'n_experiments_completed' in data, "n_experiments_completed is required"
    assert isinstance(data['n_experiments_completed'], int), "n_experiments_completed must be an int"
    
    assert 'best_237_verifier_auroc' in data, "best_237_verifier_auroc is required"
    
    assert 'phase1_ship_gate_met' in data, "phase1_ship_gate_met is required"
    
    assert 'top_3_gaps_for_238' in data, "top_3_gaps_for_238 is required"
    assert isinstance(data['top_3_gaps_for_238'], list), "top_3_gaps_for_238 must be a list"

if __name__ == '__main__':
    test_retro_json_schema()
    print("Test passed.")
