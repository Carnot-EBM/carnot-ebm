import json
import os
import tempfile
import pytest

from scripts.experiment_2470_retro import run_retro

def test_run_retro(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        def mock_glob(pattern):
            # Parse the task id from the pattern (e.g., 'results/experiment_2459*.json')
            task_id_str = pattern.split('_')[1].split('*')[0]
            if not task_id_str.isdigit():
                return []
            task_id = int(task_id_str)
            
            mock_file = os.path.join(tmpdir, f"experiment_{task_id}.json")
            data = {"status": "complete", "honest_verdict": "complete:"}
            
            if task_id == 2461:
                data["best_auroc_v3"] = 0.9167
            elif task_id == 2463:
                data["constraint_memory_implemented"] = True
            elif task_id == 2465:
                data["kv260_synthesis_succeeded"] = True
            elif task_id == 2466:
                data["polarfire_workload_validated"] = True
            elif task_id == 2468:
                data["audit_passed"] = False
                data["status"] = "failed"
                data["honest_verdict"] = "failed:"
            elif task_id == 2469:
                data["phase1_ship_gate_met"] = False
                data["status"] = "blocked"
                data["honest_verdict"] = "blocked:"
            
            with open(mock_file, 'w') as f:
                json.dump(data, f)
            return [mock_file]
            
        import glob
        monkeypatch.setattr(glob, 'glob', mock_glob)
        
        output_file = os.path.join(tmpdir, "experiment_2470_retro_v238.json")
        result = run_retro(output_file)
        
        assert result["retro_complete"] is True
        assert result["n_experiments_completed"] == 9 # 2459 to 2467
        assert result["n_failed"] == 1 # 2468
        assert result["n_blocked"] == 1 # 2469
        assert result["n_missing"] == 0
        assert result["best_238_auroc"] == 0.9167
        assert abs(result["auroc_gap_to_hive_peer"] - 0.0069) < 1e-4
        assert result["fr11_tier2_implemented"] is True
        assert result["kv260_synthesis_succeeded"] is True
        assert result["polarfire_workload_validated"] is True
        assert result["audit_passed"] is False
        assert result["phase1_ship_gate_met"] is False
        assert result["milestone"] == "2026.05.238"
        assert os.path.exists(output_file)
