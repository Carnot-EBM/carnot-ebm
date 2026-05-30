"""Tests for REQ-INFER-3406: Latent Spills Sensing."""

import pytest
from carnot.inference.latent_spills import LatentSpillsDetector

def test_latent_spills_calculation():
    """Test SCENARIO-INFER-3406-001: calculates energy spills dynamically per token."""
    detector = LatentSpillsDetector(threshold=0.5)
    latents = [0.1, -0.2, 0.8, -0.3]
    spills = detector.calculate_energy_spills(latents)
    assert spills == [0.1, 0.2, 0.8, 0.3]
    
def test_detect_hallucination():
    detector = LatentSpillsDetector(threshold=0.5)
    assert detector.detect_hallucination([0.1, 0.2, 0.3]) == False
    assert detector.detect_hallucination([0.1, 0.6, 0.3]) == True
    
def test_score_sequence():
    detector = LatentSpillsDetector(threshold=0.5)
    assert detector.score_sequence([0.1, -0.2, 0.3]) == pytest.approx(0.2)
    assert detector.score_sequence([]) == 0.0
