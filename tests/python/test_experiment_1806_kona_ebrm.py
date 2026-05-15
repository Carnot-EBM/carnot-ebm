import json
import os
from scripts.experiment_1806_kona_ebrm import main

def test_experiment_1806_script(tmpdir, monkeypatch):
    """Spec: REQ-KONA-040, SCENARIO-KONA-040"""
    monkeypatch.chdir(tmpdir)
    main()
    assert os.path.exists("results/experiment_1806_kona_ebrm.json")
    with open("results/experiment_1806_kona_ebrm.json", "r") as f:
        data = json.load(f)
    assert data["experiment_id"] == 1806
    assert data["honest_verdict"] == "continuous_improved"
