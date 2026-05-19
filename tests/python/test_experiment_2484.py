import pytest
import numpy as np
from pathlib import Path

# Adjust path and import as needed if the script is in `scripts`
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from experiment_2484_replication import compute_p_values, fisher_combine, normalize_label

def test_normalize_label():
    assert normalize_label('correct') == 0
    assert normalize_label('incorrect') == 1
    assert normalize_label('0') == 0
    assert normalize_label(1) == 1

def test_compute_p_values():
    X_cal = np.array([[0.1], [0.5], [0.9]])
    X_test = np.array([[0.0], [0.5], [1.0]])
    p_vals = compute_p_values(X_cal, X_test)
    assert p_vals.shape == (3, 1)
    # 0.0 -> 3 values >= 0.0 -> 3 / 4 = 0.75
    assert p_vals[0, 0] == 0.75
    # 0.5 -> 2 values >= 0.5 -> 2 / 4 = 0.5
    assert p_vals[1, 0] == 0.5
    # 1.0 -> 0 values >= 1.0 -> 0 / 4 = 0.0
    assert p_vals[2, 0] == 0.0

def test_fisher_combine():
    p_vals = np.array([[0.5, 0.5], [0.1, 0.1]])
    combined = fisher_combine(p_vals)
    assert combined.shape == (2,)
    assert 0.0 <= combined[0] <= 1.0
    assert 0.0 <= combined[1] <= 1.0
    # Lower p-values should yield a higher combined anomaly score
    assert combined[1] > combined[0]
