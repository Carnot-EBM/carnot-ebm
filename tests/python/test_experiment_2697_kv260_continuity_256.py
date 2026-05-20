import json
import os
from unittest import mock
import pytest

from experiment_2697_kv260_continuity_256 import run_experiment

def test_run_experiment_branch_b(tmp_path):
    results_dir = tmp_path / "results"
    
    with mock.patch("experiment_2697_kv260_continuity_256.check_sd_card", return_value=False), \
         mock.patch("os.makedirs") as mock_makedirs, \
         mock.patch("builtins.open", mock.mock_open()) as mock_open:
        
        # Override the hardcoded results path in the script for testing
        with mock.patch("experiment_2697_kv260_continuity_256.open", mock.mock_open()) as mock_script_open:
            run_experiment()
            
            mock_script_open.assert_called_once_with("results/experiment_2697_kv260_continuity_256.json", "w")
            
            # Get what was written
            written_data = "".join(call.args[0] for call in mock_script_open().write.call_args_list)
            artifact = json.loads(written_data)
            
            assert artifact["honest_verdict"] == "complete: operator action required for SD card"
            assert artifact["branch_taken"] == "B"
            assert artifact["sd_card_detected"] is False
            assert artifact["kv260_terminal"] is False
            assert artifact["prep_doc_updated"] is True
            assert "duration_s" in artifact
            assert len(artifact["preconditions_checked"]) == 1
            assert artifact["preconditions_checked"][0]["resource"] == "/dev/mmcblk*"
            assert artifact["preconditions_checked"][0]["available"] is False

def test_run_experiment_branch_a(tmp_path):
    with mock.patch("experiment_2697_kv260_continuity_256.check_sd_card", return_value=True), \
         mock.patch("experiment_2697_kv260_continuity_256.check_xmutil", return_value=True), \
         mock.patch("experiment_2697_kv260_continuity_256.load_bitstream", return_value=True), \
         mock.patch("subprocess.run") as mock_subprocess_run, \
         mock.patch("os.makedirs"), \
         mock.patch("experiment_2697_kv260_continuity_256.open", mock.mock_open()) as mock_script_open:
        
        run_experiment()
        
        # Get what was written
        written_data = "".join(call.args[0] for call in mock_script_open().write.call_args_list)
        artifact = json.loads(written_data)
        
        assert artifact["honest_verdict"] == "complete: Branch A executed"
        assert artifact["branch_taken"] == "A"
        assert artifact["sd_card_detected"] is True
        assert artifact["kv260_terminal"] is False
        assert artifact["xmutil_available"] is True
        assert artifact["bitstream_loaded"] is True
        assert artifact["ising_energy_check_passed"] is True
        assert artifact["kv260_board_smoke_passed"] is True
