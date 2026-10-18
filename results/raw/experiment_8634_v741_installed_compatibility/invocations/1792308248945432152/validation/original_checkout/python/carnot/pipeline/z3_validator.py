"""Z3 Validator Backend for Logic Rules."""

import z3
from typing import Dict, Any, List

class Z3Validator:
    """Validator that routes logic rules to Z3 solver.
    
    References:
    - REQ-VERIFY-1975
    - SCENARIO-VERIFY-1975
    """
    
    def validate(self, constraints: List[Dict[str, Any]], assignment: Dict[str, float]) -> bool:
        """Validate an assignment against constraints using Z3.
        
        Args:
            constraints: List of constraint dictionaries.
            assignment: Dictionary mapping variables to their values.
            
        Returns:
            True if the assignment satisfies all constraints, False otherwise.
        """
        solver = z3.Solver()
        variables = {}
        
        # Create Z3 variables for each target mentioned in constraints
        for constraint in constraints:
            target = constraint["target"]
            if target not in variables:
                variables[target] = z3.Real(target)
                
        # Also ensure all targets in assignment have Z3 variables
        for target in assignment:
            if target not in variables:
                variables[target] = z3.Real(target)
        
        # Add constraints to solver
        for constraint in constraints:
            target = constraint["target"]
            var = variables[target]
            value = constraint["value"]
            
            if constraint["type"] == "lower_bound":
                solver.add(var >= value)
            elif constraint["type"] == "upper_bound":
                solver.add(var <= value)
            elif constraint["type"] == "equality":
                solver.add(var == value)
            else:
                # Unsupported constraint type, safely ignore or treat as false? 
                # We can just ignore or raise, let's raise to be strict.
                raise ValueError(f"Unsupported constraint type: {constraint['type']}")
                
        # Add assignments as equality constraints
        for target, val in assignment.items():
            solver.add(variables[target] == val)
            
        return solver.check() == z3.sat
