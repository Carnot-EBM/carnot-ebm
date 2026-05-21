import json
import os
from unittest import mock
import pytest

from experiment_2742_kv260_latency_transcript_terminal import run_experiment

def test_run_experiment_ssh_unreachable(tmp_path):
    with mock.patch("experiment_2742_kv260_latency_transcript_terminal.check_ssh_reachable", return_value=False), \
         mock.patch("os.makedirs"), \
         mock.patch("experiment_2742_kv260_latency_transcript_terminal.open", mock.mock_open()) as mock_script_open:
        
        run_experiment()
        
        written_data = "".join(call.args[0] for call in mock_script_open().write.call_args_list)
        artifact = json.loads(written_data)
        
        assert artifact["honest_verdict"] == "blocked_kv260_ssh_unreachable"
        assert artifact["ssh_kria_reachable"] is False
        assert "duration_s" in artifact
        assert len(artifact["preconditions_checked"]) == 1
        assert artifact["preconditions_checked"][0]["resource"] == "ssh_reachability"
        assert artifact["preconditions_checked"][0]["available"] is False

def test_run_experiment_success(tmp_path):
    with mock.patch("experiment_2742_kv260_latency_transcript_terminal.check_ssh_reachable", return_value=True), \
         mock.patch("experiment_2742_kv260_latency_transcript_terminal.check_xmutil", return_value=(True, "carnot_ising_v2_n64 (0+0+0) 0")), \
         mock.patch("experiment_2742_kv260_latency_transcript_terminal.load_bitstream", return_value=(True, "loaded")), \
         mock.patch("experiment_2742_kv260_latency_transcript_terminal.check_uio_devices", return_value=(["/dev/uio0", "/dev/uio1", "/dev/uio2", "/dev/uio3", "/dev/uio4"], 5)), \
         mock.patch("experiment_2742_kv260_latency_transcript_terminal.measure_latency", return_value=(True, 3.183, 1.172, 2.95, 14.6)), \
         mock.patch("os.makedirs"), \
         mock.patch("experiment_2742_kv260_latency_transcript_terminal.open", mock.mock_open()) as mock_script_open:
        
        run_experiment()
        
        written_data = "".join(call.args[0] for call in mock_script_open().write.call_args_list)
        artifact = json.loads(written_data)
        
        assert artifact["honest_verdict"] == "success: KV260 terminal latency transcript verified"
        assert artifact["ssh_kria_reachable"] is True
        assert artifact["bitstream_loaded"] is True
        assert artifact["uio_count"] == 5
        assert artifact["kv260_synthesis_succeeded"] is True
        assert artifact["kv260_terminal"] is True
        assert artifact["kv260_latency_mean_us"] == 3.183
        assert artifact["kv260_latency_std_us"] == 1.172
        assert artifact["kv260_latency_min_us"] == 2.95
        assert artifact["kv260_latency_max_us"] == 14.6
        assert artifact["n_cycles_measured"] == 100
        assert len(artifact["preconditions_checked"]) == 2
        assert artifact["preconditions_checked"][1]["resource"] == "xmutil"
        assert artifact["preconditions_checked"][1]["available"] is True
