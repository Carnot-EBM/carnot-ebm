import os
import json

def test_symbolic_kan_rtl_artifacts():
    """
    REQ-MODEL-030: SymbolicKAN node vocabulary.
    SCENARIO-MODEL-015: Symbolic label assignment.
    Verifies that the RTL and experiment JSON deliverables exist and have correct fields.
    """
    assert os.path.exists("hardware/kv260/symbolic_kan.v")
    assert os.path.exists("hardware/kv260/synth_symbolic_kan.tcl")
    assert os.path.exists("results/experiment_1791_symbolic_kan_rtl.json")

    with open("results/experiment_1791_symbolic_kan_rtl.json", "r") as f:
        data = json.load(f)
        assert data["status"] == "complete"
        assert "theoretical_validation" in data
