import numpy as np

from carnot.verify.hallufield_verifier import compute_hallufield_score, HalluFieldVerifier

def test_compute_hallufield_score_empty():
    score, grid = compute_hallufield_score([])
    assert score == 0.0
    assert grid == [0.5, 0.8, 1.0, 1.2, 2.0]

def test_compute_hallufield_score_none():
    score, grid = compute_hallufield_score([None, None])
    assert score == 0.0

def test_compute_hallufield_score_valid():
    logprobs = [-1.0, -2.0, -0.5]
    score, grid = compute_hallufield_score(logprobs)
    assert score >= 0.0
    assert len(grid) == 5

def test_hallufield_verifier():
    verifier = HalluFieldVerifier([0.5, 1.0])
    logprobs = [-1.0, -2.0, -0.5]
    score = verifier.score(logprobs)
    assert score >= 0.0
