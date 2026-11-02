import random

def conformal_stopping_criterion(energy_scores: list[float], alpha=0.1, min_iterations=2) -> tuple[bool, float]:
    if len(energy_scores) == 0:
        return False, 0.0
    interval_width = max(energy_scores) - min(energy_scores)
    should_stop = (interval_width < alpha) and (len(energy_scores) >= min_iterations)
    return should_stop, interval_width

def dependency_graph_stopping(response_segments: list[str], verified_mask: list[bool]) -> bool:
    if not verified_mask:
        return False
    return all(verified_mask)

class VerifierDrivenTTT:
    def __init__(self, pipeline=None, k_max=10, alpha=0.1):
        self.pipeline = pipeline
        self.k_max = k_max
        self.alpha = alpha

    def iterate(self, prompt, initial_response) -> dict:
        energy_scores = []
        verified_mask = []
        response_segments = [initial_response]
        
        stopped_by_orca = False
        stopped_by_gc = False
        
        for k in range(self.k_max):
            energy_scores.append(random.random())
            verified_mask = [random.choice([True, False]) for _ in range(3)] # e.g. 3 claims
            
            orca_stop, _ = conformal_stopping_criterion(energy_scores, self.alpha)
            if orca_stop:
                stopped_by_orca = True
                break
                
            gc_stop = dependency_graph_stopping(response_segments, verified_mask)
            if gc_stop:
                stopped_by_gc = True
                break

        return {
            "iterations": len(energy_scores),
            "stopped_by_orca": stopped_by_orca,
            "stopped_by_gc": stopped_by_gc,
            "final_energy": energy_scores[-1] if energy_scores else 1.0,
            "final_response": "response"
        }

def run_with_dual_stopping(examples: list[dict], k_max=10, alpha=0.1) -> dict:
    n_stopped_by_orca = 0
    n_stopped_by_gc = 0
    total_iterations = 0
    early_stops_good = 0
    total_early_stops = 0
    
    for ex in examples:
        energy_scores = []
        response_segments = ["mock_segment"]
        
        stopped_orca = False
        stopped_gc = False
        
        energy_seq = ex.get("energy_sequence", [])
        # verified_masks is a list of lists of booleans (one mask per iteration)
        verified_masks = ex.get("verified_masks", [])
        
        iters_run = 0
        for k in range(k_max):
            iters_run += 1
            if k < len(energy_seq):
                energy_scores.append(energy_seq[k])
            else:
                energy_scores.append(energy_seq[-1] if energy_seq else 1.0)
                
            if k < len(verified_masks):
                verified_mask = verified_masks[k]
            else:
                verified_mask = verified_masks[-1] if verified_masks else [False]
                
            orca_stop, _ = conformal_stopping_criterion(energy_scores, alpha)
            if orca_stop:
                stopped_orca = True
                break
                
            gc_stop = dependency_graph_stopping(response_segments, verified_mask)
            if gc_stop:
                stopped_gc = True
                break
                
        total_iterations += iters_run
        
        if stopped_orca:
            n_stopped_by_orca += 1
        elif stopped_gc:
            n_stopped_by_gc += 1
            
        is_early_stop = (stopped_orca or stopped_gc) and (iters_run < k_max)
        if is_early_stop:
            total_early_stops += 1
            if energy_scores[-1] <= 0.5:
                early_stops_good += 1

    coverage_achieved = (early_stops_good / total_early_stops) if total_early_stops > 0 else 0.0
    
    return {
        "n_iterations_run": total_iterations,
        "n_stopped_by_orca": n_stopped_by_orca,
        "n_stopped_by_gc": n_stopped_by_gc,
        "coverage_achieved": coverage_achieved,
        "total_examples": len(examples)
    }

class TTTLoop:
    def __init__(self, nexus_memory):
        self.nexus_memory = nexus_memory

