"""Tests for experiment 3597."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts import experiment_3597_archive_v330_activate_v331

def test_experiment_3597():
    """Test REQ-VERIFY-3597."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        with patch.object(experiment_3597_archive_v330_activate_v331.Path, "cwd", return_value=tmp_path):
            with patch("scripts.experiment_3597_archive_v330_activate_v331.Path") as MockPath:
                def mock_path_constructor(p, *args, **kwargs):
                    if str(p) == "results":
                        return tmp_path
                    return Path(p, *args, **kwargs)
                MockPath.side_effect = mock_path_constructor
                
                experiment_3597_archive_v330_activate_v331.main()
                
        out_path = tmp_path / "experiment_3597_archive_v330_activate_v331.json"
        assert out_path.exists()
        
        data = json.loads(out_path.read_text())
        
        assert data["honest_verdict"] == "complete: archived_v330_unfinished_decontamination_gate_cascade_recorded_v331_active_paper_ready_true"
        assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
        assert data["v330_outcome_recorded_as"] == "unfinished_decontamination_gate_cascade_blocked_clean_math_finding"
        assert data["gate_cascade_root_cause_recorded"] == "dict_vs_bare_eval_op_mismatch"
        assert data["paper_ready_preserved"] is True
        assert data["n_tasks_archived"] > 0
        assert data["random_seed"] == 42
        assert "reproducibility_checksum" in data
        assert "duration_s" in data
