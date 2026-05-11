import os
import json
import pytest
import jax.numpy as jnp
from carnot.pipeline.ising_consensus import run_ising_consensus, IsingConsensusProtocol

def test_ising_consensus_req_ising_043(tmp_path):
    """
    Test that the Ising Consensus Protocol works and saves the correct output.
    References: REQ-ISING-043, SCENARIO-ISING-043
    """
    out_path = str(tmp_path / "experiment_1872_ising_consensus.json")
    
    protocol = IsingConsensusProtocol()
    answers = protocol.generate_answers()
    assert len(answers) == 5
    
    J, b = protocol.encode_conflicts(answers)
    assert J.shape == (5, 5)
    assert b.shape == (5,)
    
    best_spins, min_energy = protocol.solve(J, b)
    assert best_spins.shape == (5,)
    
    protocol.save_results(answers, best_spins, min_energy, out_path)
    
    assert os.path.exists(out_path)
    with open(out_path, "r") as f:
        data = json.load(f)
        assert "answers" in data
        assert "consensus_spins" in data
        assert "min_energy" in data
        assert len(data["answers"]) == 5

def test_run_ising_consensus_scenario_ising_043(tmp_path):
    """
    Test the main entry point.
    References: SCENARIO-ISING-043
    """
    out_path = str(tmp_path / "test_exp_1872.json")
    run_ising_consensus(out_path)
    assert os.path.exists(out_path)
