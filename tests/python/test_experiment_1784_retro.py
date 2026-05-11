import json
from pathlib import Path
from scripts.experiment_1784_retro import generate_retro, parse_experiment_files

def test_generate_retro_req_retro_1784(tmp_path: Path) -> None:
    """
    Test that REQ-RETRO-1784 is satisfied by generating an aggregated retrospective
    for experiments 1771 to 1783.
    """
    # Setup mock files
    (tmp_path / "experiment_1771_test.json").write_text('{"experiment": 1771, "honest_verdict": "latent_optimizer_ok"}')
    (tmp_path / "experiment_1783_test.json").write_text('{"experiment_id": "1783"}')
    (tmp_path / "experiment_1775_invalid.json").write_text('not json')
    (tmp_path / "experiment_1785_ignored.json").write_text('{"experiment": 1785}')
    
    generate_retro(str(tmp_path))
    
    out_file = tmp_path / "experiment_1784_retro.json"
    assert out_file.exists()
    
    data = json.loads(out_file.read_text())
    assert data["experiment"] == 1784
    assert data["honest_verdict"] == "phase_4_operations_aggregated"
    
    # Check aggregation
    agg = data["aggregated_results"]
    assert "experiment_1771_test.json" in agg
    assert agg["experiment_1771_test.json"] == "latent_optimizer_ok"
    assert "experiment_1783_test.json" in agg
    assert agg["experiment_1783_test.json"] == "unknown_verdict"
    assert "experiment_1775_invalid.json" in agg
    assert agg["experiment_1775_invalid.json"] == "error_parsing"
    assert "experiment_1785_ignored.json" not in agg
