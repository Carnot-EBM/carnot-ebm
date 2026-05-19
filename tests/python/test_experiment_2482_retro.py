import json
import os
import tempfile
import pytest

from scripts.experiment_2482_retro import run_retro

def test_run_retro(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        def mock_glob(pattern):
            # Parse the task id from the pattern
            task_id_str = pattern.split('_')[1].split('*')[0]
            if not task_id_str.isdigit():
                return []
            task_id = int(task_id_str)
            
            # exp2478 is missing
            if task_id == 2478:
                return []
                
            # exp2482 is not created yet (it is what we generate)
            if task_id == 2482:
                return []
            
            mock_file = os.path.join(tmpdir, f"experiment_{task_id}.json")
            data = {"status": "complete", "honest_verdict": "complete:"}
            
            if task_id == 2473:
                data["best_calibrated_auroc"] = 0.9351
            elif task_id == 2475:
                data["jepa_predictor_implemented"] = True
            elif task_id == 2476:
                data["honest_verdict"] = "blocked:"
                data["status"] = "blocked"
            elif task_id == 2477:
                data["kv260_bitstream_flashed"] = False
            elif task_id == 2479:
                data["audit_passed_after_fix"] = True
            elif task_id == 2480:
                data["phase4_hold_status"] = "partially_validated"
            elif task_id == 2481:
                data["phase1_ship_gate_met"] = True
                data["best_239_auroc"] = 0.935065
            
            with open(mock_file, 'w') as f:
                json.dump(data, f)
            return [mock_file]
            
        import glob
        monkeypatch.setattr(glob, 'glob', mock_glob)
        
        output_file = os.path.join(tmpdir, "experiment_2482_retro_v239.json")
        result = run_retro(output_file)
        
        assert result["retro_complete"] is True
        assert result["n_experiments_completed"] == 10 # 2471-2475, 2477, 2479-2482
        assert result["n_failed"] == 0
        assert result["n_blocked"] == 1 # 2476
        assert result["n_missing"] == 1 # 2478
        assert abs(result["best_239_auroc"] - 0.9351) < 1e-4
        assert result["phase4_hold_status"] == "partially_validated"
        assert result["fr11_tier3_implemented"] is True
        assert result["kv260_bitstream_flashed"] is False
        assert result["carnot_runs_on_polarfire"] is False
        assert result["audit_passed_after_fix"] is True
        assert result["phase1_ship_gate_met"] is True
        assert result["milestone"] == "2026.05.239"
        assert os.path.exists(output_file)
