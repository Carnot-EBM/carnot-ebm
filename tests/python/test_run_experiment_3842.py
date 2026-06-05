import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Insert path to scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.run_experiment_3842 import main, run_cmd

def test_run_cmd_timeout():
    # REQ-HW-3842 / SCENARIO-HW-3842
    # timeout handling
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = __import__("subprocess").TimeoutExpired(cmd="dummy", timeout=10)
        code, out, err, dur = run_cmd("dummy")
        assert code == 124

def test_main_blocked_ssh(tmp_path):
    # REQ-HW-3842 / SCENARIO-HW-3842
    with patch("scripts.run_experiment_3842.run_cmd") as mock_run, \
         patch("builtins.open", new_callable=__import__("unittest").mock.mock_open) as mock_open:
        
        # mock ssh to fail
        mock_run.return_value = (1, "", "error", 0.1)
        
        main()
        
        mock_open.assert_called_with("results/experiment_3842_kv260_opportunistic_continuity_audit.json", "w")
        handle = mock_open()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        data = json.loads(written)
        
        assert data["kv260_ssh_reachable"] is False
        assert data["accelerator_overlay_loadable"] is False
        assert data["honest_verdict"] == "blocked_kv260_ssh_unreachable"
        assert len(data["preconditions_checked"]) == 1
        assert data["inference_substrate"] == "hardware_smoke"

def test_main_complete_success(tmp_path):
    # REQ-HW-3842 / SCENARIO-HW-3842
    with patch("scripts.run_experiment_3842.run_cmd") as mock_run, \
         patch("builtins.open", new_callable=__import__("unittest").mock.mock_open) as mock_open:
        
        # mock ssh to succeed, then xmutil to succeed
        def side_effect(cmd):
            if "kria 'true'" in cmd:
                return (0, "", "", 0.1)
            elif "xmutil listapps" in cmd:
                return (0, "carnot_ising_v4", "", 0.1)
            return (1, "", "", 0.1)
            
        mock_run.side_effect = side_effect
        
        main()
        
        handle = mock_open()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        data = json.loads(written)
        
        assert data["kv260_ssh_reachable"] is True
        assert data["accelerator_overlay_loadable"] is True
        assert data["honest_verdict"] == "complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"

def test_main_ssh_success_xmutil_fail(tmp_path):
    # REQ-HW-3842 / SCENARIO-HW-3842
    with patch("scripts.run_experiment_3842.run_cmd") as mock_run, \
         patch("builtins.open", new_callable=__import__("unittest").mock.mock_open) as mock_open:
        
        def side_effect(cmd):
            if "kria 'true'" in cmd:
                return (0, "", "", 0.1)
            elif "xmutil listapps" in cmd:
                return (1, "", "error", 0.1)
            return (1, "", "", 0.1)
            
        mock_run.side_effect = side_effect
        
        main()
        
        handle = mock_open()
        written = "".join(call.args[0] for call in handle.write.call_args_list)
        data = json.loads(written)
        
        assert data["kv260_ssh_reachable"] is True
        assert data["accelerator_overlay_loadable"] is False
        assert data["honest_verdict"] == "complete: terminal_state_holds=false_operator_regression"
