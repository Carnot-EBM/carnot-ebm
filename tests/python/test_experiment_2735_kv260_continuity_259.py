import json
import os
from unittest import mock
import pytest

from experiment_2735_kv260_continuity_259 import run_experiment

def test_run_experiment_ssh_unreachable(tmp_path):
    with mock.patch("experiment_2735_kv260_continuity_259.check_ssh_reachable", return_value=False), \
         mock.patch("os.makedirs"), \
         mock.patch("experiment_2735_kv260_continuity_259.open", mock.mock_open()) as mock_script_open:
        
        run_experiment()
        
        # Get what was written
        written_data = "".join(call.args[0] for call in mock_script_open().write.call_args_list)
        artifact = json.loads(written_data)
        
        assert artifact["honest_verdict"] == "blocked_kv260_ssh_unreachable"
        assert artifact["ssh_kria_reachable"] is False
        assert artifact["kv260_terminal"] is False
        assert "duration_s" in artifact
        assert len(artifact["preconditions_checked"]) == 1
        assert artifact["preconditions_checked"][0]["resource"] == "ssh_reachability"
        assert artifact["preconditions_checked"][0]["available"] is False

def test_run_experiment_success(tmp_path):
    with mock.patch("experiment_2735_kv260_continuity_259.check_ssh_reachable", return_value=True), \
         mock.patch("experiment_2735_kv260_continuity_259.check_xmutil", return_value=True), \
         mock.patch("experiment_2735_kv260_continuity_259.load_bitstream", return_value=True), \
         mock.patch("experiment_2735_kv260_continuity_259.check_uio_devices", return_value=(["/dev/uio0", "/dev/uio1", "/dev/uio2", "/dev/uio3", "/dev/uio4"], 5)), \
         mock.patch("experiment_2735_kv260_continuity_259.check_uio_first_word", return_value=(True, 0)), \
         mock.patch("os.makedirs"), \
         mock.patch("experiment_2735_kv260_continuity_259.open", mock.mock_open()) as mock_script_open:
        
        run_experiment()
        
        # Get what was written
        written_data = "".join(call.args[0] for call in mock_script_open().write.call_args_list)
        artifact = json.loads(written_data)
        
        assert artifact["honest_verdict"] == "success: KV260 continuity .259 verified via SSH"
        assert artifact["ssh_kria_reachable"] is True
        assert artifact["kv260_terminal"] is False
        assert artifact["xmutil_available"] is True
        assert artifact["bitstream_loaded"] is True
        assert artifact["uio_count"] == 5
        assert artifact["uio0_first_word_read"] is True
        assert artifact["uio0_value"] == 0
        assert artifact["prep_doc_updated"] is True
        assert len(artifact["preconditions_checked"]) == 2
        assert artifact["preconditions_checked"][1]["resource"] == "xmutil"
        assert artifact["preconditions_checked"][1]["available"] is True
