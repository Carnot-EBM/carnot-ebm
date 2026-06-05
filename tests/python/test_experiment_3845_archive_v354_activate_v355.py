import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Adjust path so we can import the script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.experiments.experiment_3845_archive_v354_activate_v355 import (
    get_publication_gate_status,
    check_preconditions,
    run_experiment
)

def test_get_publication_gate_status_success():
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = '{"paper_ready": true}'
        mock_run.return_value = mock_result
        
        status = get_publication_gate_status()
        assert status["paper_ready"] is True

def test_get_publication_gate_status_failure():
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = Exception("boom")
        
        status = get_publication_gate_status()
        assert status["paper_ready"] is False
        assert "boom" in status["unmet_gates"][0]

def test_check_preconditions_all_present(tmp_path):
    f1 = tmp_path / "f1.json"
    f2 = tmp_path / "f2.json"
    f1.touch()
    f2.touch()
    
    met, msg = check_preconditions([str(f1), str(f2)])
    assert met is True
    assert msg == "All files present"

def test_check_preconditions_missing(tmp_path):
    f1 = tmp_path / "f1.json"
    f1.touch()
    missing_file = tmp_path / "missing.json"
    
    met, msg = check_preconditions([str(f1), str(missing_file)])
    assert met is False
    assert "Missing files" in msg

def test_run_experiment_success(tmp_path):
    with patch("scripts.experiments.experiment_3845_archive_v354_activate_v355.check_preconditions") as mock_check, \
         patch("scripts.experiments.experiment_3845_archive_v354_activate_v355.get_publication_gate_status") as mock_gate, \
         patch("scripts.experiments.experiment_3845_archive_v354_activate_v355.Path.write_text") as mock_write:
        
        mock_check.return_value = (True, "All files present")
        mock_gate.return_value = {"paper_ready": True}
        
        res = run_experiment()
        
        assert res["honest_verdict"].startswith("complete:")
        assert res["paper_ready"] is True
        assert res["frozen_fover_auroc_unchanged"] is True
        mock_write.assert_called_once()

def test_run_experiment_blocked():
    with patch("scripts.experiments.experiment_3845_archive_v354_activate_v355.check_preconditions") as mock_check:
        
        mock_check.return_value = (False, "Missing files: fake.json")
        
        res = run_experiment()
        
        assert res["honest_verdict"] == "blocked_preconditions_failed: Missing files: fake.json"
        assert res["preconditions_checked"] is False
