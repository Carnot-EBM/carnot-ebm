import numpy as np

def extract_dsl_rules(demos):
    """
    Extract a simple deterministic DSL rule from demos.
    Returns a rule tuple if one is found that perfectly explains all demos.
    """
    if not demos:
        return None
    
    # Check identity
    is_identity = True
    for d in demos:
        inp = np.array(d['input'])
        out = np.array(d['output'])
        if not np.array_equal(inp, out):
            is_identity = False
            break
    if is_identity:
        return ('identity',)
    
    # Check global recolor of a single color
    possible_recolors = set()
    for i, d in enumerate(demos):
        inp = np.array(d['input'])
        out = np.array(d['output'])
        if inp.shape != out.shape:
            possible_recolors = set()
            break
        
        diff = inp != out
        if not diff.any():
            continue
            
        c_in = set(inp[diff].tolist())
        c_out = set(out[diff].tolist())
        
        if len(c_in) == 1 and len(c_out) == 1:
            c1 = c_in.pop()
            c2 = c_out.pop()
            if i == 0:
                possible_recolors.add((c1, c2))
            else:
                possible_recolors.intersection_update({(c1, c2)})
        else:
            possible_recolors = set()
            break

    if possible_recolors:
        c1, c2 = possible_recolors.pop()
        # Verify it fully explains ALL demos
        valid = True
        for d in demos:
            inp = np.array(d['input'])
            out = np.array(d['output'])
            pred = inp.copy()
            pred[inp == c1] = c2
            if not np.array_equal(pred, out):
                valid = False
                break
        if valid:
            return ('recolor', c1, c2)
            
    return None

def apply_rule(rule, grid):
    grid = np.array(grid)
    if rule is None:
        return None
    if rule[0] == 'identity':
        return grid
    if rule[0] == 'recolor':
        _, c1, c2 = rule
        out = grid.copy()
        out[grid == c1] = c2
        return out
    return None

def get_consistency_energy(rule, test_input, candidate):
    """
    E(candidate) = cell-disagreement(candidate, program(test_input))
    """
    pred = apply_rule(rule, test_input)
    if pred is None:
        return 1.0 # High energy if no prediction
    
    cand = np.array(candidate)
    if pred.shape != cand.shape:
        return 1.0
        
    diff = pred != cand
    return float(np.mean(diff))

class Gap4ExecutionVerifier:
    def __init__(self):
        self.llm_proposer_used = False
        
    def induce_program(self, demos):
        """
        Attempt DSL induction first.
        If DSL fails, fallback to local GGUF (stubbed here if not needed for coverage).
        """
        rule = extract_dsl_rules(demos)
        if rule is not None:
            return rule
            
        return None
