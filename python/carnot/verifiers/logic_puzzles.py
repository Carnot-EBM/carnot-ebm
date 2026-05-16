"""Logic puzzle generation and Z3 verification modules."""

import re
import z3

def generate_boolean_puzzle(seed: int) -> dict:
    """Generate a simple boolean logic puzzle."""
    import random
    rng = random.Random(seed)
    
    # 3 variables: A, B, C
    vars = ["A", "B", "C"]
    
    # Randomly assign True/False
    assignment = {v: rng.choice([True, False]) for v in vars}
    
    # Create 3 constraints that uniquely identify the assignment
    constraints = []
    
    # Constraint 1: A and B
    if assignment["A"] == assignment["B"]:
        if assignment["A"]:
            constraints.append("A and B are both True.")
        else:
            constraints.append("A and B are both False.")
    else:
        if assignment["A"]:
            constraints.append("A is True but B is False.")
        else:
            constraints.append("A is False but B is True.")
            
    # Constraint 2: C relates to A
    if assignment["C"] == assignment["A"]:
        constraints.append("C has the same value as A.")
    else:
        constraints.append("C has the opposite value of A.")
        
    prompt = (
        "Solve this boolean logic puzzle for variables A, B, and C.\n"
        "Constraints:\n"
        + "\n".join(f"- {c}" for c in constraints) + "\n"
        "Reply with the final assignment in the format: A=True, B=False, C=True."
    )
    
    return {
        "prompt": prompt,
        "expected": assignment,
        "seed": seed
    }

def verify_boolean_puzzle(response: str, expected_assignment: dict) -> bool:
    """Verify the response using Z3.
    Extracts the assignment from the response and checks satisfiability against expected.
    """
    # Extract assignment from response
    match_a = re.search(r"A\s*=\s*(True|False)", response, re.IGNORECASE)
    match_b = re.search(r"B\s*=\s*(True|False)", response, re.IGNORECASE)
    match_c = re.search(r"C\s*=\s*(True|False)", response, re.IGNORECASE)
    
    if not match_a or not match_b or not match_c:
        return False
        
    a_val = match_a.group(1).lower() == "true"
    b_val = match_b.group(1).lower() == "true"
    c_val = match_c.group(1).lower() == "true"
    
    # Use Z3 to verify
    solver = z3.Solver()
    A, B, C = z3.Bools('A B C')
    
    # Add the extracted values as constraints
    solver.add(A == z3.BoolVal(a_val))
    solver.add(B == z3.BoolVal(b_val))
    solver.add(C == z3.BoolVal(c_val))
    
    # Add the expected values as the target constraints
    solver.add(A == z3.BoolVal(expected_assignment["A"]))
    solver.add(B == z3.BoolVal(expected_assignment["B"]))
    solver.add(C == z3.BoolVal(expected_assignment["C"]))
    
    return solver.check() == z3.sat
