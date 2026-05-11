import json
from pathlib import Path
from scripts.experiment_1770_retro import generate_retro, parse_experiment_files

def test_generate_retro_req_report_1770(tmp_path: Path) -> None:
    """
    Test that REQ-REPORT-1770 is satisfied by generating an aggregated retrospective.
    """
    # Setup mock files
    (tmp_path / "experiment_1759_test.json").write_text('{"experiment": 1759, "honest_verdict": "ebft_implemented"}')
    (tmp_path / "experiment_1765_eval.json").write_text('{"experiment_id": "1765"}')
    (tmp_path / "experiment_1775_ignored.json").write_text('{"experiment": 1775}')
    
    generate_retro(str(tmp_path))
    
    out_file = tmp_path / "experiment_1770_retro.json"
    assert out_file.exists()
    
    data = json.loads(out_file.read_text())
    assert data["experiment"] == 1770
    assert data["honest_verdict"] == "phase_4_operations_aggregated"
    
    # Check aggregation
    agg = data["aggregated_results"]
    assert "experiment_1759_test.json" in agg
    assert agg["experiment_1759_test.json"] == "ebft_implemented"
    assert "experiment_1765_eval.json" in agg
    assert agg["experiment_1765_eval.json"] == "unknown_verdict"
    assert "experiment_1775_ignored.json" not in agg
