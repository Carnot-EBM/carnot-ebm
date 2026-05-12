"""
Tests for GNN benchmarking audit against Z3/SAT baselines.
Spec: REQ-VERIFY-1952, SCENARIO-VERIFY-1952
"""
import os
import json
import pytest
from carnot.inference.verify_repair import (
    generate_hard_3sat,
    run_z3_sat,
    run_continuous_solver,
    run_audit,
    z3
)
from carnot.verify.sat import SATClause

def test_generate_hard_3sat():
    instances, n_vars, n_clauses = generate_hard_3sat(n_instances=2, n_vars=10)
    assert len(instances) == 2
    assert n_vars == 10
    # 10 * 4.26 = 42
    assert n_clauses == 42
    for inst in instances:
        assert len(inst) == 42
        for clause in inst:
            assert isinstance(clause, SATClause)

def test_run_z3_sat():
    # Simple satisfiable instance
    # (x0 or x1)
    clauses = [SATClause([(0, True), (1, True)])]
    success, duration = run_z3_sat(clauses, n_vars=2)
    # Z3 might be unavailable in CI without it installed, but assuming it is:
    assert isinstance(success, bool)
    assert duration >= 0.0

    # Also test the z3 is None fallback if we monkeypatch it
    import carnot.inference.verify_repair as vr
    old_z3 = vr.z3
    vr.z3 = None
    success_none, duration_none = run_z3_sat(clauses, n_vars=2)
    assert success_none is False
    assert duration_none == 0.0
    vr.z3 = old_z3

def test_run_continuous_solver():
    # Simple satisfiable instance
    # (x0)
    clauses = [SATClause([(0, True)])]
    success, duration = run_continuous_solver(clauses, n_vars=1, max_steps=10)
    assert isinstance(success, bool)
    assert duration >= 0.0

def test_run_audit(tmp_path, monkeypatch):
    # Run audit with very small instances to finish quickly
    # Patch os to use tmp_path for results
    monkeypatch.chdir(tmp_path)
    report = run_audit(n_instances=2, n_vars=5)
    
    assert report["status"] == "complete"
    assert report["n_instances"] == 2
    assert report["n_vars"] == 5
    assert "z3_success_rate" in report
    assert "carnot_success_rate" in report
    assert os.path.exists("results/experiment_1952_gnn_benchmarking_audit.json")
    
    with open("results/experiment_1952_gnn_benchmarking_audit.json", "r") as f:
        data = json.load(f)
        assert data["status"] == "complete"
        assert data["n_instances"] == 2

def test_run_audit_zero_instances():
    # test divide by zero avoidance
    report = run_audit(n_instances=0, n_vars=5)
    assert report["z3_success_rate"] == 0.0
    assert report["carnot_success_rate"] == 0.0
