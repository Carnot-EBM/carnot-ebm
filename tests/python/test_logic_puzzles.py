import pytest
from carnot.verifiers.logic_puzzles import generate_boolean_puzzle, verify_boolean_puzzle

def test_generate_boolean_puzzle():
    # Test multiple seeds to cover all generation branches
    for seed in range(20):
        puzzle = generate_boolean_puzzle(seed)
        assert "prompt" in puzzle
        assert "expected" in puzzle
        assert "A" in puzzle["expected"]
        assert "B" in puzzle["expected"]
        assert "C" in puzzle["expected"]

def test_verify_boolean_puzzle_correct():
    expected = {"A": True, "B": False, "C": True}
    response = "The answer is A=True, B=False, and C=True."
    assert verify_boolean_puzzle(response, expected)

def test_verify_boolean_puzzle_incorrect():
    expected = {"A": True, "B": False, "C": True}
    response = "The answer is A=False, B=False, and C=True."
    assert not verify_boolean_puzzle(response, expected)
    
def test_verify_boolean_puzzle_unparseable():
    expected = {"A": True, "B": False, "C": True}
    response = "I don't know the answer."
    assert not verify_boolean_puzzle(response, expected)
