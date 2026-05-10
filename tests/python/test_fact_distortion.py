import pytest
from carnot.pipeline.fact_distortion import FactDistortionDetector, FactDistortionResult

def test_fact_distortion_detector_init():
    detector = FactDistortionDetector(threshold=0.7)
    assert detector.threshold == 0.7

def test_fact_distortion_detect_hallucinated():
    detector = FactDistortionDetector(threshold=0.5)
    result = detector.detect("This is a hallucinated premise.", "Logical constraint")
    
    assert isinstance(result, FactDistortionResult)
    assert result.is_distorted is True
    assert result.distortion_score == 1.0
    assert result.runtime_ms >= 0.0

def test_fact_distortion_detect_clean():
    detector = FactDistortionDetector(threshold=0.5)
    result = detector.detect("The sky is blue.", "Color observation")
    
    assert result.is_distorted is False
    assert result.distortion_score == 0.0

def test_fact_distortion_detect_false_vs_true():
    detector = FactDistortionDetector(threshold=0.5)
    result = detector.detect("This statement is false.", "It must be true")
    
    assert result.is_distorted is True
    assert result.distortion_score == 0.8

def test_fact_distortion_detect_verbose():
    detector = FactDistortionDetector(threshold=0.5)
    premise = "A " * 20
    hypothesis = "A " * 2
    result = detector.detect(premise, hypothesis)
    
    assert result.is_distorted is True
    assert result.distortion_score == 0.6

def test_fact_distortion_detect_verbose_below_threshold():
    detector = FactDistortionDetector(threshold=0.9)
    premise = "A " * 20
    hypothesis = "A " * 2
    result = detector.detect(premise, hypothesis)
    
    assert result.is_distorted is False
    assert result.distortion_score == 0.6
