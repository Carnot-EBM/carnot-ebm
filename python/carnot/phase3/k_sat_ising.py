import math
import numpy as np

def generate_planted_ksat(n_vars: int, n_clauses: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    planted = rng.integers(0, 2, size=n_vars).tolist()
    
    clauses = []
    while len(clauses) < n_clauses:
        vars_in_clause = rng.choice(n_vars, size=k, replace=False).tolist()
        signs = rng.choice([-1, 1], size=k).tolist()
        
        satisfied = False
        for v, s in zip(vars_in_clause, signs):
            lit_val = 1 if planted[v] == 1 else -1
            if lit_val == s:
                satisfied = True
                break
        
        if not satisfied:
            idx_to_flip = rng.integers(0, k)
            signs[idx_to_flip] *= -1
            
        clauses.append([(vars_in_clause[i], signs[i]) for i in range(k)])
        
    return clauses, planted

def walksat_solve(n_vars: int, clauses: list, seed: int, max_flips: int = 10000, p: float = 0.5):
    rng = np.random.default_rng(seed)
    assignment = rng.integers(0, 2, size=n_vars).tolist()
    
    # fast unsat lookup
    var_to_clauses = [[] for _ in range(n_vars)]
    for i, c in enumerate(clauses):
        for v, s in c:
            var_to_clauses[v].append((i, s))
            
    clause_satisfied_lits = [0] * len(clauses)
    for i, c in enumerate(clauses):
        sat_lits = 0
        for v, s in c:
            val = 1 if assignment[v] == 1 else -1
            if val == s:
                sat_lits += 1
        clause_satisfied_lits[i] = sat_lits
        
    unsat_set = set(i for i, sl in enumerate(clause_satisfied_lits) if sl == 0)

    for _ in range(max_flips):
        if not unsat_set:
            return assignment, True
            
        c_idx = rng.choice(list(unsat_set))
        c = clauses[c_idx]
        
        if rng.random() < p:
            v_flip = rng.choice([v for v, s in c])
        else:
            best_v = None
            min_break = float('inf')
            for v, s in c:
                val = 1 if assignment[v] == 1 else -1
                new_val = -val
                break_count = 0
                for c_i, s_i in var_to_clauses[v]:
                    was_sat = (val == s_i)
                    will_be_sat = (new_val == s_i)
                    if was_sat and not will_be_sat and clause_satisfied_lits[c_i] == 1:
                        break_count += 1
                if break_count < min_break:
                    min_break = break_count
                    best_v = v
            v_flip = best_v
            
        # Flip v_flip
        val = 1 if assignment[v_flip] == 1 else -1
        new_val = -val
        assignment[v_flip] = 1 - assignment[v_flip]
        for c_i, s_i in var_to_clauses[v_flip]:
            was_sat = (val == s_i)
            will_be_sat = (new_val == s_i)
            if was_sat and not will_be_sat:
                clause_satisfied_lits[c_i] -= 1
                if clause_satisfied_lits[c_i] == 0:
                    unsat_set.add(c_i)
            elif not was_sat and will_be_sat:
                clause_satisfied_lits[c_i] += 1
                if clause_satisfied_lits[c_i] == 1:
                    unsat_set.discard(c_i)
        
    return assignment, len(unsat_set) == 0

def ar_greedy_solve(n_vars: int, clauses: list, seed: int):
    rng = np.random.default_rng(seed)
    var_order = rng.permutation(n_vars).tolist()
    
    assignment = [-1] * n_vars
    
    for v in var_order:
        unsat0 = 0
        unsat1 = 0
        
        # Test 0
        assignment[v] = 0
        for c in clauses:
            is_unsat = True
            is_fully_assigned = True
            for cv, cs in c:
                if assignment[cv] == -1:
                    is_fully_assigned = False
                    break
                val = 1 if assignment[cv] == 1 else -1
                if val == cs:
                    is_unsat = False
            if is_fully_assigned and is_unsat:
                unsat0 += 1
                
        # Test 1
        assignment[v] = 1
        for c in clauses:
            is_unsat = True
            is_fully_assigned = True
            for cv, cs in c:
                if assignment[cv] == -1:
                    is_fully_assigned = False
                    break
                val = 1 if assignment[cv] == 1 else -1
                if val == cs:
                    is_unsat = False
            if is_fully_assigned and is_unsat:
                unsat1 += 1
                
        if unsat0 < unsat1:
            assignment[v] = 0
        elif unsat1 < unsat0:
            assignment[v] = 1
        else:
            assignment[v] = rng.choice([0, 1])
            
    is_valid = True
    for c in clauses:
        satisfied = False
        for v, s in c:
            val = 1 if assignment[v] == 1 else -1
            if val == s:
                satisfied = True
                break
        if not satisfied:
            is_valid = False
            break
            
    return assignment, is_valid

def exact_solve(n_vars: int, clauses: list):
    assignment = [-1] * n_vars
    
    # DPLL with simple unit propagation could be better but backtracking is enough for small instances
    def backtrack(idx):
        if idx == n_vars:
            return True
            
        for val in [0, 1]:
            assignment[idx] = val
            valid = True
            for c in clauses:
                is_unsat = True
                is_fully_assigned = True
                for v, s in c:
                    if assignment[v] == -1:
                        is_fully_assigned = False
                        break
                    c_val = 1 if assignment[v] == 1 else -1
                    if c_val == s:
                        is_unsat = False
                if is_fully_assigned and is_unsat:
                    valid = False
                    break
            if valid:
                if backtrack(idx + 1):
                    return True
        assignment[idx] = -1
        return False
        
    return backtrack(0)

def sa_solve(n_vars: int, clauses: list, seed: int, n_sweeps: int=1000, T_init: float=1.0, T_final: float=0.01):
    rng = np.random.default_rng(seed)
    assignment = rng.integers(0, 2, size=n_vars).tolist()
    
    var_to_clauses = [[] for _ in range(n_vars)]
    for i, c in enumerate(clauses):
        for v, s in c:
            var_to_clauses[v].append((i, s))
            
    clause_satisfied_lits = [0] * len(clauses)
    for i, c in enumerate(clauses):
        sat_lits = 0
        for v, s in c:
            val = 1 if assignment[v] == 1 else -1
            if val == s:
                sat_lits += 1
        clause_satisfied_lits[i] = sat_lits
        
    unsat_count = sum(1 for sl in clause_satisfied_lits if sl == 0)
    T_decay = (T_final / T_init) ** (1.0 / max(1, n_sweeps)) if T_init > 0 else 0
    T = T_init
    
    for sweep in range(n_sweeps):
        if unsat_count == 0:
            break
        order = rng.permutation(n_vars)
        for v in order:
            delta_unsat = 0
            val = 1 if assignment[v] == 1 else -1
            new_val = -val
            
            for c_idx, s in var_to_clauses[v]:
                was_sat = (val == s)
                will_be_sat = (new_val == s)
                old_sl = clause_satisfied_lits[c_idx]
                if was_sat and not will_be_sat and old_sl == 1:
                    delta_unsat += 1
                elif not was_sat and will_be_sat and old_sl == 0:
                    delta_unsat -= 1
                        
            if delta_unsat <= 0 or (T > 1e-9 and rng.random() < math.exp(-delta_unsat / T)):
                assignment[v] = 1 - assignment[v]
                unsat_count += delta_unsat
                val_new = 1 if assignment[v] == 1 else -1
                for c_idx, s in var_to_clauses[v]:
                    if (val_new == s) and ((-val_new) != s):
                        clause_satisfied_lits[c_idx] += 1
                    elif (val_new != s) and ((-val_new) == s):
                        clause_satisfied_lits[c_idx] -= 1
        T *= T_decay
        
    return assignment, unsat_count == 0

def pt_solve(n_vars: int, clauses: list, seed: int, n_sweeps: int=3000, n_replicas: int=6):
    rng = np.random.default_rng(seed)
    temps = [0.02 * (3.0)**i for i in range(n_replicas)]
    replicas = [rng.integers(0, 2, size=n_vars).tolist() for _ in range(n_replicas)]
    
    var_to_clauses = [[] for _ in range(n_vars)]
    for i, c in enumerate(clauses):
        for v, s in c:
            var_to_clauses[v].append((i, s))
            
    replicas_sat_lits = []
    for r in range(n_replicas):
        sat_lits = [0] * len(clauses)
        for i, c in enumerate(clauses):
            sl = 0
            for v, s in c:
                val = 1 if replicas[r][v] == 1 else -1
                if val == s:
                    sl += 1
            sat_lits[i] = sl
        replicas_sat_lits.append(sat_lits)
        
    unsat_counts = [sum(1 for sl in sls if sl == 0) for sls in replicas_sat_lits]
    
    swap_attempts = 0
    swap_accepts = 0
    
    for sweep in range(n_sweeps):
        if min(unsat_counts) == 0:
            break
            
        for r_idx in range(n_replicas):
            T = temps[r_idx]
            order = rng.permutation(n_vars)
            for v in order[:15]:
                delta_unsat = 0
                val = 1 if replicas[r_idx][v] == 1 else -1
                new_val = -val
                
                for c_idx, s in var_to_clauses[v]:
                    was_sat = (val == s)
                    will_be_sat = (new_val == s)
                    old_sl = replicas_sat_lits[r_idx][c_idx]
                    if was_sat and not will_be_sat and old_sl == 1:
                        delta_unsat += 1
                    elif not was_sat and will_be_sat and old_sl == 0:
                        delta_unsat -= 1
                        
                if delta_unsat <= 0 or (T > 1e-9 and rng.random() < math.exp(-delta_unsat / T)):
                    replicas[r_idx][v] = 1 - replicas[r_idx][v]
                    unsat_counts[r_idx] += delta_unsat
                    val_new = 1 if replicas[r_idx][v] == 1 else -1
                    for c_idx, s in var_to_clauses[v]:
                        if (val_new == s) and ((-val_new) != s):
                            replicas_sat_lits[r_idx][c_idx] += 1
                        elif (val_new != s) and ((-val_new) == s):
                            replicas_sat_lits[r_idx][c_idx] -= 1

        if sweep % 50 == 0:
            for r_idx in range(n_replicas - 1):
                E_i = unsat_counts[r_idx]
                E_j = unsat_counts[r_idx + 1]
                T_i = temps[r_idx]
                T_j = temps[r_idx + 1]
                log_acc = (1.0/T_i - 1.0/T_j) * (E_i - E_j)
                swap_attempts += 1
                if log_acc >= 0 or rng.random() < math.exp(log_acc):
                    replicas[r_idx], replicas[r_idx+1] = replicas[r_idx+1], replicas[r_idx]
                    unsat_counts[r_idx], unsat_counts[r_idx+1] = unsat_counts[r_idx+1], unsat_counts[r_idx]
                    replicas_sat_lits[r_idx], replicas_sat_lits[r_idx+1] = replicas_sat_lits[r_idx+1], replicas_sat_lits[r_idx]
                    swap_accepts += 1
                    
    swap_rate = swap_accepts / max(1, swap_attempts)
    return min(unsat_counts) == 0, swap_rate
