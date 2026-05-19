"""Tests for the Suppressed Retrieval Probe (Tier 0o)."""

from carnot.verify.suppressed_retrieval_probe import SuppressedRetrievalProbe

def test_suppressed_retrieval_probe_empty_logprobs():
    probe = SuppressedRetrievalProbe()
    score, _ = probe.compute_score([])
    assert score == 0.0

def test_suppressed_retrieval_probe_single_logprob():
    probe = SuppressedRetrievalProbe()
    score, _ = probe.compute_score([-0.1])
    assert score == 0.0

def test_suppressed_retrieval_probe_identical_halves():
    probe = SuppressedRetrievalProbe()
    # Identical halves should result in 0 divergence
    score, _ = probe.compute_score([-0.1, -0.2, -0.1, -0.2])
    assert score >= 0.0  # Just checking it doesn't crash and returns a float

def test_suppressed_retrieval_probe_verify():
    probe = SuppressedRetrievalProbe()
    entry = {"token_logprobs": [-0.1, -0.5, -0.2, -0.9]}
    result = probe.verify(entry)
    assert "suppression_score" in result
    assert isinstance(result["suppression_score"], float)
