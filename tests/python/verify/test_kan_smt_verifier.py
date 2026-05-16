"""Tests for KAN SMT Verifier.

Spec: REQ-SYMKAN-2076
"""
import numpy as np
from carnot.verify.kan_smt_verifier import verify_path_continuity

def test_verify_path_continuity_valid():
    path = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert verify_path_continuity(path, eps=1e-4)

def test_verify_path_continuity_invalid_start():
    path = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    assert not verify_path_continuity(path, eps=1e-4)

def test_verify_path_continuity_invalid_step():
    path = np.array([0.0, 1.5, 2.0, 3.0, 4.0])
    assert not verify_path_continuity(path, eps=1e-4)

def test_verify_path_continuity_short_path():
    path = np.array([0.0])
    assert not verify_path_continuity(path, eps=1e-4)

