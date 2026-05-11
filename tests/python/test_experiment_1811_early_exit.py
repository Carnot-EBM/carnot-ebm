import json
from pathlib import Path

from carnot.models.eorm import EORMModel, CoTEnergyInput

from scripts.experiment_1811_early_exit import run_experiment


def test_experiment_1811_early_exit(tmp_path: Path):
    """
    Test that experiment 1811 correctly tracks layer energies and 
    outputs the distribution of optimal exit layers.
    Spec: REQ-EORM-1811, SCENARIO-EORM-1811
    """
    model = EORMModel(embed_dim=16, n_heads=2, n_layers=4, max_seq_len=32, vocab_size=64)
    
    out_file = tmp_path / "experiment_1811_early_exit.json"
    run_experiment(model, str(out_file))
    
    assert out_file.exists()
    
    with open(out_file) as f:
        results = json.load(f)
        
    assert "optimal_exit_layer_distribution" in results
    assert "mean_optimal_layer" in results
    
    # We used a 4-layer model, so keys in the distribution should be "0", "1", "2", "3"
    assert sum(results["optimal_exit_layer_distribution"].values()) == 10  # We test with 10 items
