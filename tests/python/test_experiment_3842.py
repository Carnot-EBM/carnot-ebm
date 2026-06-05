"""Tests for Experiment 3842 — KV260 opportunistic continuity audit.

Spec: REQ-HW-3842, SCENARIO-HW-3842
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import run_experiment_3842  # type: ignore

def test_run_cmd_success():
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "hello"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        code, stdout, stderr, dur = run_experiment_3842.run_cmd("echo hello")
        assert code == 0
        assert stdout == "hello"
        assert dur >= 0

def test_run_cmd_timeout():
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = __import__("subprocess").TimeoutExpired(cmd="test", timeout=10)
        code, stdout, stderr, dur = run_experiment_3842.run_cmd("test")
        assert code == 124
        assert stderr == "timeout"
        assert dur >= 0

def test_main_unreachable():
    with patch("run_experiment_3842.run_cmd") as mock_run:
        # Mock SSH unreachable
        mock_run.return_value = (255, "", "ssh: connect to host kria port 22: Connection timed out", 5.0)
        
        with patch("builtins.open") as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            run_experiment_3842.main()
            
            mock_open.assert_called_once_with("results/experiment_3842_kv260_opportunistic_continuity_audit.json", "w")
            written_data = "".join(call[0][0] for call in mock_file.write.call_args_list)
            artifact = json.loads(written_data)
            
            assert artifact["honest_verdict"] == "blocked_kv260_ssh_unreachable"
            assert artifact["kv260_ssh_reachable"] is False
            assert artifact["accelerator_overlay_loadable"] is False
            assert artifact["inference_substrate"] == "hardware_smoke"
            assert len(artifact["preconditions_checked"]) == 1
            assert artifact["preconditions_checked"][0]["available"] is False

def test_main_reachable_no_overlay():
    with patch("run_experiment_3842.run_cmd") as mock_run:
        # First call: SSH reachable
        # Second call: listapps doesn't show carnot
        mock_run.side_effect = [
            (0, "", "", 0.1),
            (0, "some other overlay", "", 1.0)
        ]
        
        with patch("builtins.open") as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            run_experiment_3842.main()
            
            written_data = "".join(call[0][0] for call in mock_file.write.call_args_list)
            artifact = json.loads(written_data)
            
            assert artifact["honest_verdict"] == "complete: terminal_state_holds=false_operator_regression"
            assert artifact["kv260_ssh_reachable"] is True
            assert artifact["accelerator_overlay_loadable"] is False

def test_main_reachable_with_overlay():
    with patch("run_experiment_3842.run_cmd") as mock_run:
        # First call: SSH reachable
        # Second call: listapps shows carnot
        mock_run.side_effect = [
            (0, "", "", 0.1),
            (0, "carnot_ising_v4", "", 1.0)
        ]
        
        with patch("builtins.open") as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            run_experiment_3842.main()
            
            written_data = "".join(call[0][0] for call in mock_file.write.call_args_list)
            artifact = json.loads(written_data)
            
            assert artifact["honest_verdict"] == "complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"
            assert artifact["kv260_ssh_reachable"] is True
            assert artifact["accelerator_overlay_loadable"] is True
