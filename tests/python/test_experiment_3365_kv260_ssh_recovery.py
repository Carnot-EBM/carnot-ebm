import pytest
import os
import json
from unittest.mock import patch, MagicMock

# The module under test
import scripts.experiment_3365_kv260_ssh_recovery as exp

def test_experiment_3365_req_hw_104_ssh_success(tmp_path):
    """
    Test SCENARIO-HW-104 / REQ-HW-104: KV260 SSH recovery when SSH succeeds initially.
    """
    deliverable = str(tmp_path / "experiment_3365_kv260_ssh_recovery.json")
    
    with patch("scripts.experiment_3365_kv260_ssh_recovery.subprocess.run") as mock_run:
        # Mocking subprocess calls
        # 1. ip route
        # 2. ip neigh
        # 3. ssh check
        def side_effect(cmd, **kwargs):
            mock_res = MagicMock()
            if cmd[:2] == ["ip", "route"]:
                mock_res.stdout = "default via 192.168.1.1 dev eth0"
                mock_res.returncode = 0
            elif cmd[:2] == ["ip", "neigh"]:
                mock_res.stdout = "192.168.1.100 dev eth0 lladdr aa:bb:cc:dd:ee:ff REACHABLE"
                mock_res.returncode = 0
            elif cmd[0] == "ssh":
                mock_res.returncode = 0
            else:
                mock_res.returncode = 1
            return mock_res
        
        mock_run.side_effect = side_effect
        
        exp.run_experiment(deliverable_path=deliverable)
        
        assert os.path.exists(deliverable)
        with open(deliverable, "r") as f:
            data = json.load(f)
            
        assert data["inference_substrate"] == "hardware_smoke"
        assert data["ssh_reachable"] is True
        assert data["routes_checked"] is True
        assert data["arp_cache_checked"] is True
        assert data["serial_connection_attempted"] is False
        assert data["connectivity_restored"] is True
        assert data["command_execution_verified"] is True
        assert data["honest_verdict"] == "complete: ssh_restored"

def test_experiment_3365_req_hw_104_ssh_fail_serial_fail(tmp_path):
    """
    Test SCENARIO-HW-104 / REQ-HW-104: KV260 SSH recovery when SSH fails and serial fails.
    """
    deliverable = str(tmp_path / "experiment_3365_kv260_ssh_recovery.json")
    
    with patch("scripts.experiment_3365_kv260_ssh_recovery.subprocess.run") as mock_run:
        # Mocking subprocess calls to fail SSH and serial
        def side_effect(cmd, **kwargs):
            mock_res = MagicMock()
            if cmd[:2] == ["ip", "route"]:
                mock_res.stdout = "default via 192.168.1.1 dev eth0"
                mock_res.returncode = 0
            elif cmd[:2] == ["ip", "neigh"]:
                mock_res.stdout = ""
                mock_res.returncode = 0
            elif cmd[0] == "ssh":
                mock_res.returncode = 1
            else:
                mock_res.returncode = 1
            return mock_res
        
        mock_run.side_effect = side_effect
        
        exp.run_experiment(deliverable_path=deliverable)
        
        assert os.path.exists(deliverable)
        with open(deliverable, "r") as f:
            data = json.load(f)
            
        assert data["ssh_reachable"] is False
        assert data["serial_connection_attempted"] is True
        assert data["connectivity_restored"] is False
        assert data["command_execution_verified"] is False
        assert data["honest_verdict"] == "blocked: ssh_and_serial_failed"
