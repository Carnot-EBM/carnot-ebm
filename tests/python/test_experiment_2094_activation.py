import os
import tempfile
import json
from scripts.experiment_2094_activation import generate_activation_data, main

def test_generate_activation_data():
    data = generate_activation_data()
    assert data["experiment"] == 2094
    assert data["honest_verdict"] == "activation_complete"

def test_main():
    with tempfile.TemporaryDirectory() as tmpdir:
        main(out_dir=tmpdir)
        
        # verify file written
        out_path = os.path.join(tmpdir, "experiment_2094_activation.json")
        assert os.path.exists(out_path)
        with open(out_path, 'r') as f:
            data = json.load(f)
        assert data["honest_verdict"] == "activation_complete"
