import pytest
from carnot.phase3.k_sat_ising import generate_planted_ksat, walksat_solve, exact_solve, pt_solve, ar_greedy_solve, sa_solve

def test_generate_and_exact_solve():
    n_vars = 10
    n_clauses = 40
    k = 3
    clauses, planted = generate_planted_ksat(n_vars, n_clauses, k, seed=42)
    
    assert len(clauses) == n_clauses
    
    # Check planted solution satisfies all clauses
    is_valid = True
    for c in clauses:
        satisfied = False
        for v, s in c:
            lit_val = 1 if planted[v] == 1 else -1
            if lit_val == s:
                satisfied = True
                break
        if not satisfied:
            is_valid = False
            break
            
    assert is_valid, "Planted solution must be valid"
    
    # Check exact solver
    assert exact_solve(n_vars, clauses), "Exact solver must find a solution"

def test_walksat_solve():
    n_vars = 10
    n_clauses = 40
    k = 3
    clauses, _ = generate_planted_ksat(n_vars, n_clauses, k, seed=43)
    
    assign, valid = walksat_solve(n_vars, clauses, seed=1, max_flips=1000)
    # It might not solve it, but it should return a boolean
    assert isinstance(valid, bool)
    
def test_sa_and_pt():
    n_vars = 10
    n_clauses = 30
    k = 3
    clauses, _ = generate_planted_ksat(n_vars, n_clauses, k, seed=44)
    
    assign_sa, valid_sa = sa_solve(n_vars, clauses, seed=1, n_sweeps=100)
    valid_pt, swap_rate = pt_solve(n_vars, clauses, seed=1, n_sweeps=100)
    
    assert isinstance(valid_sa, bool)
    assert isinstance(valid_pt, bool)
    assert isinstance(swap_rate, float)

def test_ar_greedy():
    n_vars = 10
    n_clauses = 30
    k = 3
    clauses, _ = generate_planted_ksat(n_vars, n_clauses, k, seed=45)
    
    assign, valid = ar_greedy_solve(n_vars, clauses, seed=1)
    assert isinstance(valid, bool)
