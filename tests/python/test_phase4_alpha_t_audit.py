import pytest
import numpy as np
from carnot.phase4_alpha_t_audit import (
    compute_bootstrap_ci,
    run_ablation_cell,
    check_monotonic_decay,
    detect_artifact,
    AuditResult,
)

# REQ-PHASE4-004: Verifier Ablation Audit
# SCENARIO-PHASE4-3: Running a 4-cell random-verifier ablation audit correctly computes delta_alpha

def test_compute_bootstrap_ci():
    """Test bootstrap CI computation."""
    data = [0.1, 0.15, 0.12, 0.18, 0.14]
    ci = compute_bootstrap_ci(data, n_bootstraps=100, seed=42)
    assert len(ci) == 2
    assert ci[0] <= ci[1]
    
    empty_ci = compute_bootstrap_ci([])
    assert empty_ci == [0.0, 0.0]

def test_run_ablation_cell():
    """Test running a single ablation cell."""
    result = run_ablation_cell(n_spins=8, random_fraction=0.5, mld_steps=10, n_seeds=2, base_seed=42)
    assert result.random_fraction == 0.5
    assert result.inf_t_alpha_k6 > 0.1 # based on mock behavior
    assert result.inf_t_alpha_k1 < 0.05 # based on mock behavior
    assert result.delta_alpha > 0.05
    assert len(result.delta_alpha_bootstrap_ci_95) == 2

def test_check_monotonic_decay():
    """Test monotonic decay logic."""
    r1 = AuditResult(0.0, 0.15, 0.0, 0.15, [0.1, 0.2])
    r2 = AuditResult(0.33, 0.10, 0.0, 0.10, [0.05, 0.15])
    r3 = AuditResult(0.67, 0.05, 0.0, 0.05, [0.0, 0.1])
    r4 = AuditResult(1.0, 0.01, 0.0, 0.01, [0.0, 0.05])
    
    # Valid decay
    assert check_monotonic_decay([r1, r2, r3, r4]) == True
    
    # Too few results
    assert check_monotonic_decay([r1]) == False
    
    # Last result too big
    r4_bad = AuditResult(1.0, 0.15, 0.0, 0.15, [0.1, 0.2])
    assert check_monotonic_decay([r1, r2, r3, r4_bad]) == False
    
    # Non-monotonic
    r2_bad = AuditResult(0.33, 0.20, 0.0, 0.20, [0.15, 0.25])
    assert check_monotonic_decay([r1, r2_bad, r3, r4]) == False

def test_detect_artifact():
    """Test artifact detection logic."""
    r1 = AuditResult(0.0, 0.15, 0.0, 0.15, [0.1, 0.2])
    r2 = AuditResult(0.33, 0.15, 0.0, 0.15, [0.1, 0.2])
    r3 = AuditResult(0.67, 0.15, 0.0, 0.15, [0.1, 0.2])
    r4 = AuditResult(1.0, 0.15, 0.0, 0.15, [0.1, 0.2])
    
    # All > 0.1
    assert detect_artifact([r1, r2, r3, r4]) == True
    
    # One < 0.1
    r4_good = AuditResult(1.0, 0.05, 0.0, 0.05, [0.0, 0.1])
    assert detect_artifact([r1, r2, r3, r4_good]) == False
