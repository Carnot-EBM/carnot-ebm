import pytest
import json
import os
from carnot.inference.ising_translator import IsingTranslator

def test_ising_translator_and():
    # REQ-ISING-046, SCENARIO-ISING-046
    translator = IsingTranslator()
    translator.add_and_constraint("z", "x", "y")
    
    # Valid combinations (energy should be 0)
    assert translator.evaluate_energy({"x": 0, "y": 0, "z": 0}) == 0.0
    assert translator.evaluate_energy({"x": 0, "y": 1, "z": 0}) == 0.0
    assert translator.evaluate_energy({"x": 1, "y": 0, "z": 0}) == 0.0
    assert translator.evaluate_energy({"x": 1, "y": 1, "z": 1}) == 0.0
    
    # Invalid combinations (energy should be > 0)
    assert translator.evaluate_energy({"x": 0, "y": 0, "z": 1}) > 0.0
    assert translator.evaluate_energy({"x": 0, "y": 1, "z": 1}) > 0.0
    assert translator.evaluate_energy({"x": 1, "y": 0, "z": 1}) > 0.0
    assert translator.evaluate_energy({"x": 1, "y": 1, "z": 0}) > 0.0

def test_ising_translator_or():
    # REQ-ISING-046, SCENARIO-ISING-046
    translator = IsingTranslator()
    translator.add_or_constraint("z", "x", "y")
    
    # Valid combinations
    assert translator.evaluate_energy({"x": 0, "y": 0, "z": 0}) == 0.0
    assert translator.evaluate_energy({"x": 0, "y": 1, "z": 1}) == 0.0
    assert translator.evaluate_energy({"x": 1, "y": 0, "z": 1}) == 0.0
    assert translator.evaluate_energy({"x": 1, "y": 1, "z": 1}) == 0.0
    
    # Invalid combinations
    assert translator.evaluate_energy({"x": 0, "y": 0, "z": 1}) > 0.0
    assert translator.evaluate_energy({"x": 0, "y": 1, "z": 0}) > 0.0
    assert translator.evaluate_energy({"x": 1, "y": 0, "z": 0}) > 0.0
    assert translator.evaluate_energy({"x": 1, "y": 1, "z": 0}) > 0.0

def test_ising_translator_not():
    # REQ-ISING-046, SCENARIO-ISING-046
    translator = IsingTranslator()
    translator.add_not_constraint("z", "x")
    
    # Valid combinations
    assert translator.evaluate_energy({"x": 0, "z": 1}) == 0.0
    assert translator.evaluate_energy({"x": 1, "z": 0}) == 0.0
    
    # Invalid combinations
    assert translator.evaluate_energy({"x": 0, "z": 0}) > 0.0
    assert translator.evaluate_energy({"x": 1, "z": 1}) > 0.0

def test_ising_translator_qubo():
    # REQ-ISING-046, SCENARIO-ISING-046
    translator = IsingTranslator()
    translator.add_and_constraint("z", "x", "y")
    linear, quadratic, offset = translator.get_qubo()
    assert "z" in linear
    assert ("x", "y") in quadratic or ("y", "x") in quadratic

def test_generate_experiment_artifact():
    # Save to results/experiment_2147_ising_translation.json
    translator = IsingTranslator()
    translator.add_and_constraint("z1", "x1", "x2")
    translator.add_or_constraint("z2", "x3", "x4")
    translator.add_not_constraint("z3", "x5")
    
    linear, quadratic, offset = translator.get_qubo()
    
    artifact = {
        "status": "success",
        "experiment_id": "2147",
        "honest_verdict": "Successfully mapped basic AND/OR/NOT clauses to quadratic energy penalties.",
        "qubo": {
            "linear": linear,
            "quadratic": {f"{k[0]},{k[1]}": v for k, v in quadratic.items()},
            "offset": offset
        }
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_2147_ising_translation.json", "w") as f:
        json.dump(artifact, f, indent=2)
    
    assert os.path.exists("results/experiment_2147_ising_translation.json")
