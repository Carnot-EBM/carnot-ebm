import os
import json
import pytest

def test_experiment_3550():
    from scripts.experiment_3550_archive_v326_activate_v327 import run_experiment
    run_experiment()
    assert os.path.exists("results/experiment_3550_archive_v326_activate_v327.json")
    with open("results/experiment_3550_archive_v326_activate_v327.json", "r") as f:
        data = json.load(f)
    assert data["honest_verdict"].startswith("complete:")
    assert data["archive_v326_activate_v327_ready"] is True
    assert data["random_seed"] == 20260601
