import os
from pathlib import Path
from carnot.experiment_2088_npu_setup import run_setup

def test_run_setup(tmp_path: Path) -> None:
    """Test that run_setup generates the correct JSON artifact."""
    output_file = tmp_path / "experiment_2088_npu_setup.json"
    result = run_setup(str(output_file))
    
    assert output_file.exists()
    assert "honest_verdict" in result
    assert result["experiment"] == 2088
    assert isinstance(result["ninja_installed"], bool)
    assert isinstance(result["openblas_installed"], bool)
    assert "schema" in result
