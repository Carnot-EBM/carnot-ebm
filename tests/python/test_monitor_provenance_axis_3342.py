import numpy as np
from carnot.verify.monitor_provenance_axis import MonitorProvenanceAxis

def test_monitor_provenance_axis_evaluate():
    axis = MonitorProvenanceAxis("test_axis")
    cands = [
        {"trajectory_steps": ["a"]},
        {"trajectory_steps": []},
        {}
    ]
    scores = axis.evaluate(cands)
    assert len(scores) == 3
    assert scores[0] == 1.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0

def test_monitor_provenance_axis_correlation():
    axis = MonitorProvenanceAxis("test_axis")
    axis_scores = np.array([1.0, 0.0, 1.0, 0.0])
    cols = {
        "col1": np.array([1.0, 0.0, 1.0, 0.0]), # perfect correlation
        "col2": np.array([0.0, 1.0, 0.0, 1.0])  # perfect negative correlation (abs = 1.0)
    }
    max_corr = axis.compute_max_correlation(axis_scores, cols)
    assert max_corr > 0.99
    
def test_monitor_provenance_axis_correlation_empty():
    axis = MonitorProvenanceAxis("test_axis")
    axis_scores = np.array([1.0, 0.0, 1.0, 0.0])
    max_corr = axis.compute_max_correlation(axis_scores, {})
    assert max_corr == 0.0
