import json
import os
import random
import sys
import time

sys.path.insert(0, 'python')

def main():
    start_time = time.time()
    preconditions_checked = []
    
    # Precondition a: ttt_loop importable
    try:
        from carnot.pipeline import ttt_loop
        import_success = True
    except ImportError as e:
        print("ImportError:", e)
        import_success = False
    
    preconditions_checked.append({
        "resource": "carnot.pipeline.ttt_loop",
        "available": import_success,
        "check": "import ttt_loop"
    })
    
    if not import_success:
        result = {
            "honest_verdict": "blocked_ttt_loop_not_importable",
            "delta_root_cause_identified": False,
            "root_cause": "unclear",
            "empirical_delta_diagnostic": 0.0,
            "n_attempts_total": 0,
            "n_successes": 0,
            "n_repair_attempts_definition": "unknown",
            "paper_v6_recommendation": "blocked",
            "random_seed": 42,
            "duration_s": time.time() - start_time,
            "preconditions_checked": preconditions_checked
        }
        with open("results/experiment_2754_empirical_delta_audit.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    # Precondition b: fover_corpus_lines
    fover_path = "data/fover_corpus.jsonl"
    fover_lines = sum(1 for _ in open(fover_path)) if os.path.exists(fover_path) else 0
    preconditions_checked.append({
        "resource": fover_path,
        "available": fover_lines > 0,
        "check": f"wc -l {fover_path}"
    })
    
    if fover_lines == 0:
        result = {
            "honest_verdict": "blocked_fover_corpus_missing",
            "delta_root_cause_identified": False,
            "root_cause": "unclear",
            "empirical_delta_diagnostic": 0.0,
            "n_attempts_total": 0,
            "n_successes": 0,
            "n_repair_attempts_definition": "unknown",
            "paper_v6_recommendation": "blocked",
            "random_seed": 42,
            "duration_s": time.time() - start_time,
            "preconditions_checked": preconditions_checked
        }
        with open("results/experiment_2754_empirical_delta_audit.json", "w") as f:
            json.dump(result, f, indent=2)
        return
        
    n_repair_attempts_definition = "Number of attempted repair generations during the TTT loop (or VerifyRepairPipeline loop), tracked as 'iterations' in RepairResult."
    
    from carnot.pipeline.verify_repair import VerifyRepairPipeline
    
    # Load 20 FoVer violations
    violations = []
    with open(fover_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            # Find violations where answer is wrong
            if ex.get("label") == "incorrect":
                violations.append(ex)
                
    random.seed(42)
    random.shuffle(violations)
    sample_violations = violations[:20]
    
    # Initialize the repair pipeline
    pipeline = VerifyRepairPipeline(model="Qwen/Qwen1.5-0.5B", max_repairs=3)
    
    per_attempt_log = []
    n_attempts_total = 0
    n_successes = 0
    all_outputs_identical = True
    all_outputs_differ = True
    
    for i, ex in enumerate(sample_violations):
        question = ex.get("question_id", f"mock_q_{i}")
        initial_response = ex.get("step_text", "")
        
        # Verify original
        orig_vr = pipeline.verify(question, initial_response, domain="arithmetic")
        original_energy = orig_vr.energy
        
        # Repair
        repair_result = pipeline.verify_and_repair(question, initial_response, domain="arithmetic")
        
        # We consider the total attempts across the repair loop
        # repair_result.history contains the verifications: history[0] is initial, history[1:] are repairs
        attempts_in_this_run = len(repair_result.history) - 1
        if attempts_in_this_run == 0 and repair_result.iterations > 0:
            attempts_in_this_run = repair_result.iterations
        
        n_attempts_total += attempts_in_this_run
        
        if repair_result.repaired:
            n_successes += 1
            
        final_resp = repair_result.final_response
        if final_resp != initial_response:
            all_outputs_identical = False
        else:
            all_outputs_differ = False
            
        for k in range(attempts_in_this_run):
            # approximate per-attempt data based on history
            hist_vr = repair_result.history[k+1] if k+1 < len(repair_result.history) else repair_result.history[-1]
            per_attempt_log.append({
                "question_id": question,
                "iteration_n": k+1,
                "original_violation_energy": original_energy,
                "repair_output": final_resp[:100],
                "post_repair_energy": hist_vr.energy,
                "repair_success": hist_vr.energy < original_energy
            })
            
    empirical_delta_diagnostic = n_successes / n_attempts_total if n_attempts_total > 0 else 0.0
    
    # Diagnose root cause
    if n_successes > 0:
        root_cause = "H1 definitional mismatch"
        paper_v6_recommendation = "update paper with empirical delta from verbose run; cite as preliminary (N=20)"
    elif n_successes == 0 and not all_outputs_identical:
        root_cause = "H3 ceiling"
        paper_v6_recommendation = "report as negative result: FoVer ceiling means delta=0 on this corpus; use conservative estimate with disclaimer"
    elif n_successes == 0 and all_outputs_identical:
        root_cause = "H2 regression"
        paper_v6_recommendation = "fix repair pipeline; block paper-v6 cite until delta validated"
    else:
        root_cause = "unclear"
        paper_v6_recommendation = "investigate further"
        
    delta_root_cause_identified = root_cause != "unclear"
    
    result = {
        "honest_verdict": "complete: diagnostic run finished",
        "delta_root_cause_identified": delta_root_cause_identified,
        "root_cause": root_cause,
        "empirical_delta_diagnostic": float(empirical_delta_diagnostic),
        "n_attempts_total": int(n_attempts_total),
        "n_successes": int(n_successes),
        "n_repair_attempts_definition": n_repair_attempts_definition,
        "paper_v6_recommendation": paper_v6_recommendation,
        "random_seed": 42,
        "duration_s": time.time() - start_time,
        "preconditions_checked": preconditions_checked,
        "per_attempt_log": per_attempt_log
    }
    
    with open("results/experiment_2754_empirical_delta_audit.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Diagnostics complete. Root cause: {root_cause}")

if __name__ == "__main__":
    main()
