import json

from scripts.experiment_1673_thrml_rng_audit import run_audit, DELIVERABLE_PATH


def test_experiment_1673_rng_audit():
    # Call the script
    artifact = run_audit()

    # Check that it executed and returned correct data
    assert artifact["status"] == "complete"
    assert artifact["simulator_only_no_hardware_claim"] is True
    assert artifact["rng_path_independent"] is True
    assert artifact["nonzero_stochastic_delta_observed"] is True
    assert artifact["sample_path_hashes_distinct"] is True

    # REQ-ISING-040: Verify it tests n=32 and n=64
    assert 32 in artifact["n_values_tested"]
    assert 64 in artifact["n_values_tested"]

    # SCENARIO-ISING-040: Verify file exists and has correct fields
    assert DELIVERABLE_PATH.exists()

    with open(DELIVERABLE_PATH) as f:
        data = json.load(f)
        assert data["simulator_only_no_hardware_claim"] is True
        assert data["rng_path_independent"] is True
