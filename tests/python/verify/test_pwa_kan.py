import pytest
import os
import json
import numpy as np

# REQ-KAN-1840: PWA KAN Abstraction
# SCENARIO-KAN-1840: Spline to PWA conversion
from carnot.verify.pwa_kan import PWAAbstraction, PWASegment, generate_experiment_1840_artifact

def test_pwa_segment_creation():
    segment = PWASegment(x_min=0.0, x_max=1.0, slope=2.0, intercept=0.5, lower_bound=0.0, upper_bound=0.1)
    assert segment.x_min == 0.0
    assert segment.x_max == 1.0
    assert segment.slope == 2.0
    assert segment.intercept == 0.5
    assert segment.lower_bound == 0.0
    assert segment.upper_bound == 0.1

def test_pwa_abstraction_conversion():
    # Simple nonlinear callable (x^2)
    def spline_callable(x):
        return x**2
    
    # 0 to 1 with 2 knots means 1 segment: 0 to 1.
    breakpoints = np.array([0.0, 0.5, 1.0])
    
    abstraction = PWAAbstraction.from_spline(spline_callable, breakpoints)
    
    assert len(abstraction.segments) == 2
    
    # Segment 1: 0.0 to 0.5
    seg1 = abstraction.segments[0]
    assert seg1.x_min == 0.0
    assert seg1.x_max == 0.5
    # y=x^2 -> (0,0) and (0.5, 0.25). slope = 0.25/0.5 = 0.5
    assert np.isclose(seg1.slope, 0.5)
    assert np.isclose(seg1.intercept, 0.0)
    
    # max error in 0 to 0.5 between x^2 and 0.5x occurs at x=0.25
    # x^2 = 0.0625, 0.5x = 0.125, difference = -0.0625
    assert np.isclose(seg1.lower_bound, -0.0625, atol=1e-3)
    assert np.isclose(seg1.upper_bound, 0.0, atol=1e-3) # simplified bounds calculation logic

def test_pwa_abstraction_evaluate():
    def spline_callable(x):
        return x**2
    breakpoints = np.array([0.0, 0.5, 1.0])
    abstraction = PWAAbstraction.from_spline(spline_callable, breakpoints)
    
    y = abstraction.evaluate(0.25)
    assert np.isclose(y, 0.125)
    
    y2 = abstraction.evaluate(0.75)
    assert np.isclose(y2, 0.625)
    
    # Out of bounds should clip or use closest segment
    assert np.isclose(abstraction.evaluate(1.5), 1.0)
    assert np.isclose(abstraction.evaluate(-0.5), 0.0)

def test_pwa_evaluate_miss_segments():
    # specifically to cover if the loop doesn't return
    segment1 = PWASegment(0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    segment2 = PWASegment(2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    abstraction = PWAAbstraction([segment1, segment2])
    # x=1.5 is between segments, loop finishes without returning
    assert np.isclose(abstraction.evaluate(1.5), 0.0)



def test_pwa_empty_segments():
    abstraction = PWAAbstraction([])
    assert np.isclose(abstraction.evaluate(0.5), 0.0)

def test_generate_experiment_1840_artifact(tmp_path):
    artifact_path = str(tmp_path / "experiment_1840_pwa_kan.json")
    generate_experiment_1840_artifact(artifact_path)
    
    assert os.path.exists(artifact_path)
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert data["status"] == "complete"
    assert data["experiment_id"] == "1840"
    assert "REQ-KAN-1840" in data["spec_traces"]
    assert data["honest_verdict"].startswith("complete:")
    assert "segments_generated" in data
