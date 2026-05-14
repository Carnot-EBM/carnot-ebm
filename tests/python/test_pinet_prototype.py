import pytest
import jax.numpy as jnp
from carnot.solvers.pinet_prototype import douglas_rachford_splitting

def test_pinet_prototype_pysat_comparison():
    # REQ-PINET-001: Enforce linear equality constraint via PiNet Douglas-Rachford splitting
    # REQ-PINET-002: Compare against PySAT
    
    # constraint 1: x1 + x2 = 1
    # constraint 2: x2 + x3 = 1
    # constraint 3: x1 = 1
    # implies x1=1, x2=0, x3=1
    A = jnp.array([[1.0, 1.0, 0.0], 
                   [0.0, 1.0, 1.0], 
                   [1.0, 0.0, 0.0]])
    b = jnp.array([1.0, 1.0, 1.0])
    
    res = douglas_rachford_splitting(A, b, max_iter=200)
    res_bool = res > 0.5
    
    assert res_bool[0] == True
    assert res_bool[1] == False
    assert res_bool[2] == True

    try:
        from pysat.solvers import Solver
        with Solver(name='g3') as s:
            s.add_clause([1, 2])
            s.add_clause([-1, -2])
            s.add_clause([2, 3])
            s.add_clause([-2, -3])
            s.add_clause([1])
            assert s.solve() == True
            model = s.get_model()
            assert model[0] > 0
            assert model[1] < 0
            assert model[2] > 0
    except ImportError:
        pass
