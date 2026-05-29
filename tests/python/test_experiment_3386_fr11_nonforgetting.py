"""Tests for FR-11 Nonforgetting constraint memory updates (Exp 3386)."""

import json
from pathlib import Path
from carnot.pipeline.fr11_nonforgetting_memory import NonforgettingMemoryUpdater
import scripts.experiment_3386_fr11_nonforgetting as exp3386

def test_nonforgetting_memory_updater_no_conflict():
    baseline = {"c1": "x > 0"}
    updater = NonforgettingMemoryUpdater(baseline)
    updater.set_holdout([{"key": "c1", "expected": "x > 0"}])
    
    new_conflicts = {"c2": "y == 1"}
    rate = updater.update(new_conflicts)
    
    assert rate == 0.0
    assert updater.rollback_count == 0
    assert updater.memory == {"c1": "x > 0", "c2": "y == 1"}

def test_nonforgetting_memory_updater_with_conflict_rollback():
    baseline = {"c1": "x > 0"}
    updater = NonforgettingMemoryUpdater(baseline)
    updater.set_holdout([{"key": "c1", "expected": "x > 0"}])
    
    new_conflicts = {"c1": "x > 5", "c2": "y == 1"}
    rate = updater.update(new_conflicts)
    
    # Should rollback the c1 update but keep c2
    assert rate == 0.0
    assert updater.rollback_count == 1
    assert updater.memory == {"c1": "x > 0", "c2": "y == 1"}

def test_experiment_3386_script():
    exp3386.main()
    
    result_file = Path("results/experiment_3386_fr11_nonforgetting.json")
    assert result_file.exists()
    
    with open(result_file, "r") as f:
        results = json.load(f)
        
    assert results["experiment"] == "3386_fr11_nonforgetting"
    assert results["fr11_nonforgetting_ready"] is True
    assert results["regression_rate"] == 0.0
    assert results["rollback_count"] == 1
    assert "c1" in results["final_memory"]
    assert results["final_memory"]["c1"] == "x > 0" # rolled back
    assert results["final_memory"]["c3"] == "z == 10" # updated
    assert results["final_memory"]["c4"] == "w == 0" # added
