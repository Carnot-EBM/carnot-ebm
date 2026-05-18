import json
import os
import tempfile
import sys

from scripts.experiment_2419_retro import run_retro

def test_run_retro(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock glob to return files from our tempdir
        def mock_glob(pattern):
            task_id = pattern.split('_')[-1].split('*')[0]
            # Create a mock file in tmpdir
            mock_file = os.path.join(tmpdir, f"experiment_{task_id}.json")
            
            data = {"status": "FAIL"}
            
            if task_id in ["2408", "2409", "2410"]:
                data = {"status": "OK", "auroc": 0.9000}
            elif task_id == "2411":
                data = {"status": "OK", "fr11_nsvif_online_passed": True}
            elif task_id == "2413":
                data = {"status": "OK", "synthesis_succeeded": True}
            elif task_id == "2417":
                data = {"status": "OK", "phase1_ship_gate_met": True}
            elif task_id in ["2414", "2415", "2416"]:
                data = {"status": "OK", "kl_delta": 0.05}
            else:
                data = {"status": "OK"}
                
            with open(mock_file, 'w') as f:
                json.dump(data, f)
            return [mock_file]
            
        import glob
        monkeypatch.setattr(glob, 'glob', mock_glob)
        
        output_file = os.path.join(tmpdir, "experiment_2419_retro_v234.json")
        result = run_retro(output_file)
        
        assert result["retro_complete"] is True
        assert result["n_experiments_completed"] == 13
        assert result["n_failed"] == 0
        assert result["best_234_verifier_auroc"] == 0.9000
        assert abs(result["auroc_gap_to_hive_peer_at_234_close"] - 0.0236) < 1e-6
        assert result["fr11_satisfied"] is True
        assert result["kv260_yosys_succeeded"] is True
        assert result["phase1_ship_gate_met"] is True
        assert result["best_sampler_kl_delta"] == 0.05
        assert result["milestone"] == "2026.05.234"
        assert os.path.exists(output_file)
