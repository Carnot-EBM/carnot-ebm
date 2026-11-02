import z3
from typing import Any

class Z3ConstraintRepairAgent:
    """
    A synthetic agent for FR-11 that uses Z3 unsat cores to refine
    flawed constraint templates in memory.
    """
    def __init__(self):
        self.solver = z3.Solver()
        self.solver.set(unsat_core=True)
        
    def repair_template(self, variables: list[str], constraints: list[tuple[str, str]]) -> tuple[bool, list[tuple[str, str]]]:
        """
        Takes a flawed constraint set (UNSAT).
        Extracts unsat core.
        Removes constraints in the core to structurally correct it.
        Verifies the repaired constraint is SAT.
        Returns (success, repaired_constraints).
        """
        self.solver.push()
        z3_vars = {v: z3.Int(v) for v in variables}
        trackers = {}
        
        # Parse and assert with tracking
        for name, expr_str in constraints:
            try:
                expr = eval(expr_str, {"z3": z3}, z3_vars)
                tracker = z3.Bool(f"track_{name}")
                trackers[tracker] = name
                self.solver.assert_and_track(expr, tracker)
            except Exception:
                self.solver.pop()
                return False, constraints
                
        # Check SAT
        result = self.solver.check()
        if result == z3.unsat:
            core = self.solver.unsat_core()
            core_names = {trackers[c] for c in core}
            
            # Structurally correct: remove all constraints in the unsat core
            repaired_constraints = [c for c in constraints if c[0] not in core_names]
            
            # Verify repaired is SAT
            verify_solver = z3.Solver()
            for name, expr_str in repaired_constraints:
                try:
                    expr = eval(expr_str, {"z3": z3}, z3_vars)
                    verify_solver.add(expr)
                except Exception:
                    self.solver.pop()
                    return False, constraints
                    
            if verify_solver.check() == z3.sat:
                self.solver.pop()
                return True, repaired_constraints
                
        self.solver.pop()
        return False, constraints
