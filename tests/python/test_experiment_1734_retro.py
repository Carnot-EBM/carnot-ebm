import os
import json
import tempfile
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))
import experiment_1734_retro

def test_parse_experiments():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Valid JSONs
        exp_1722 = tmp_path / "experiment_1722_fouriercsp.json"
        exp_1722.write_text(json.dumps({"experiment_id": "1722"}))
        
        exp_1723 = tmp_path / "experiment_1723_cikan.json"
        exp_1723.write_text(json.dumps({"experiment_id": 1723, "status": "complete"}))

        exp_1727 = tmp_path / "experiment_1727_eqm.json"
        exp_1727.write_text(json.dumps({"honest_verdict": "eqm_converged_faster"}))

        exp_1729 = tmp_path / "experiment_1729_kanele.json"
        exp_1729.write_text(json.dumps({"status": "complete"}))

        exp_1732 = tmp_path / "experiment_1732_unified.json"
        exp_1732.write_text(json.dumps({"success": True}))

        exp_1724 = tmp_path / "experiment_1724_online_updater.json"
        exp_1724.write_text(json.dumps({"success": False}))

        exp_1727_fail = tmp_path / "experiment_1727_fail.json"
        exp_1727_fail.write_text(json.dumps({"honest_verdict": "failed"}))
        
        # Invalid JSON
        exp_1725_invalid = tmp_path / "experiment_1725_invalid.json"
        exp_1725_invalid.write_text("invalid json {")
        
        data = experiment_1734_retro.parse_experiments(tmp_path)
        
        assert data["experiment_id"] == "1734"
        assert len(data["parsed_experiments"]) == 7
        assert data["evaluations"]["FourierCSP"]["status"] == "success"
        assert data["evaluations"]["CIKAN"]["status"] == "complete"
        assert data["evaluations"]["KANELÉ"]["status"] == "complete"
        assert data["evaluations"]["Continuous Self-Learning"]["status"] == "success" # due to 1732
        assert len(data["gaps_for_134"]) > 0

def test_run_retrospective():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        
        output_file = tmp_path / "output.json"
        experiment_1734_retro.run_retrospective(results_dir, output_file)
        
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert data["experiment_id"] == "1734"

def test_main(monkeypatch):
    called = []
    def mock_run_retrospective(res_dir, out_file):
        called.append(True)
        
    monkeypatch.setattr(experiment_1734_retro, "run_retrospective", mock_run_retrospective)
    experiment_1734_retro.main()
    assert len(called) == 1
