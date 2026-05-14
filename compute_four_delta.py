import json
import glob
import os
import hashlib

def gather_traces():
    traces = []
    # Search all json results
    files = sorted(glob.glob("results/experiment_*.json"), key=os.path.getmtime, reverse=True)
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
        except Exception:
            continue
        
        # Simple recursive search for "iterations" list
        def find_iterations(obj):
            if isinstance(obj, dict):
                if "iterations" in obj and isinstance(obj["iterations"], list) and len(obj["iterations"]) > 0:
                    first = obj["iterations"][0]
                    if isinstance(first, dict) and "passed" in first:
                        # Found a trace
                        is_successful = any(it.get("passed", False) for it in obj["iterations"])
                        if is_successful:
                            traces.append((f, obj["iterations"]))
                for v in obj.values():
                    find_iterations(v)
            elif isinstance(obj, list):
                for item in obj:
                    find_iterations(item)
                    
        find_iterations(data)
        if len(traces) >= 50:
            break
            
    return traces[:50]

traces = gather_traces()
print(f"Found {len(traces)} successful traces.")
if traces:
    total_attempts = 0
    total_successes = 0
    num_iterations_list = []
    
    for f, iters in traces:
        num_it = len(iters)
        num_iterations_list.append(num_it)
        total_attempts += num_it
        # since it's a successful trace, exactly 1 success (the last one or some passing one)
        # let's count actual successes
        successes = sum(1 for it in iters if it.get("passed", False))
        total_successes += successes

    delta_empirical = total_successes / total_attempts if total_attempts > 0 else 0
    mean_iterations = sum(num_iterations_list) / len(traces)
    predicted_bound = 4 / delta_empirical if delta_empirical > 0 else float('inf')
    
    print(f"Total attempts: {total_attempts}")
    print(f"Total successes: {total_successes}")
    print(f"delta_empirical: {delta_empirical}")
    print(f"mean_iterations: {mean_iterations}")
    print(f"predicted_bound: {predicted_bound}")
    print(f"acceptance_gate_passed: {mean_iterations <= predicted_bound}")
    
    # We will need to output this to results/experiment_2108_four_delta_bound.json later
