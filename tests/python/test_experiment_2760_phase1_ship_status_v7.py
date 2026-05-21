import json
import os
import subprocess
from unittest import mock

def test_experiment_2760_run(tmp_path, monkeypatch):
    import experiment_2760_phase1_ship_status_v7

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        os.makedirs("results", exist_ok=True)
        
        with open("results/experiment_2730_hf_mirror_ship_v6.json", "w") as f:
            json.dump({"operator_ship_checklist_v6": ["Step 1", "Step 2"]}, f)

        def mock_check_output(args, **kwargs):
            if "tag" in args and "--list" in args:
                return b"v0.1.0b1\nv0.1.0b2\n"
            return subprocess.check_output(args, **kwargs)

        with mock.patch("subprocess.check_output", side_effect=mock_check_output):
            experiment_2760_phase1_ship_status_v7.run()

        assert os.path.exists("results/experiment_2760_phase1_ship_status_v7.json")
        with open("results/experiment_2760_phase1_ship_status_v7.json", "r") as f:
            data = json.load(f)

        assert data["honest_verdict"].startswith("complete")
        assert data["phase1_shipped"] is True
        assert data["phase1_tag_found"] == "v0.1.0b1"
        assert data["checklist_still_current"] is True
        assert data["new_gates_opened"] == []
        assert "SHIPPED at v0.1.0b1" in data["operator_ship_checklist_v7"][0]
        assert "duration_s" in data
        assert len(data["preconditions_checked"]) == 2

    finally:
        os.chdir(original_cwd)

def test_experiment_2760_run_not_shipped(tmp_path, monkeypatch):
    import experiment_2760_phase1_ship_status_v7

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        os.makedirs("results", exist_ok=True)
        
        with open("results/experiment_2730_hf_mirror_ship_v6.json", "w") as f:
            json.dump({"operator_ship_checklist_v6": ["Step 1", "Step 2"]}, f)

        def mock_check_output_empty(args, **kwargs):
            if "tag" in args and "--list" in args:
                return b""
            return subprocess.check_output(args, **kwargs)

        with mock.patch("subprocess.check_output", side_effect=mock_check_output_empty):
            experiment_2760_phase1_ship_status_v7.run()

        assert os.path.exists("results/experiment_2760_phase1_ship_status_v7.json")
        with open("results/experiment_2760_phase1_ship_status_v7.json", "r") as f:
            data = json.load(f)

        assert data["honest_verdict"].startswith("complete")
        assert data["phase1_shipped"] is False
        assert data["phase1_tag_found"] is None
        assert data["operator_ship_checklist_v7"][0] == "Step 1"
        assert "Not shipped yet" in data["operator_ship_checklist_v7"][-1]

    finally:
        os.chdir(original_cwd)
