import json
from pathlib import Path
from scripts.experiment_1811_retro import generate_retro, parse_experiment_files

def test_generate_retro_req_retro_1811(tmp_path: Path) -> None:
    """
    Test that REQ-RETRO-1811 is satisfied by generating an aggregated retrospective
    for experiments 1799 to 1810. (SCENARIO-RETRO-1811)
    """
    # Setup mock files
    (tmp_path / "experiment_1799_test.json").write_text(
        '{"experiment": 1799, "honest_verdict": "ok", "accuracy_delta": 0.05}'
    )
    (tmp_path / "experiment_1810_test.json").write_text(
        '{"experiment": 1810, "dpo_improvement_pp": 2.5}'
    )
    (tmp_path / "experiment_1805_invalid.json").write_text('not json')
    (tmp_path / "experiment_1812_ignored.json").write_text('{"experiment": 1812}')
    
    generate_retro(str(tmp_path))
    
    out_file = tmp_path / "experiment_1811_retro.json"
    assert out_file.exists()
    
    data = json.loads(out_file.read_text())
    assert data["experiment"] == 1811
    assert data["honest_verdict"] == "phase_16_aggregated"
    assert data["title"] == "Phase-16 Finding Summary"
    
    # Check aggregation
    agg = data["aggregated_results"]
    assert "experiment_1799_test.json" in agg
    assert agg["experiment_1799_test.json"]["honest_verdict"] == "ok"
    assert agg["experiment_1799_test.json"]["metrics"]["accuracy_delta"] == 0.05
    
    assert "experiment_1810_test.json" in agg
    assert agg["experiment_1810_test.json"]["honest_verdict"] == "unknown_verdict"
    assert agg["experiment_1810_test.json"]["metrics"]["dpo_improvement_pp"] == 2.5
    
    assert "experiment_1805_invalid.json" in agg
    assert agg["experiment_1805_invalid.json"]["honest_verdict"] == "error_parsing"
    
    assert "experiment_1812_ignored.json" not in agg
