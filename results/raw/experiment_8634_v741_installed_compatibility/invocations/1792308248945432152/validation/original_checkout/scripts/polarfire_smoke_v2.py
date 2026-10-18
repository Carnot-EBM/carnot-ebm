import json
import time
import sys
import glob

# Try to find the verifier. In the tarball, we'll just drop it in the same directory.
try:
    from semantic_consistency_verifier import SemanticConsistencyVerifier
except ImportError:
    # If placed in python/carnot/verify
    from python.carnot.verify.semantic_consistency_verifier import SemanticConsistencyVerifier

examples = [
    "The accuracy equals 95%. The accuracy equals 90%.",
    "The sky is blue. The sky is not blue.",
    "System cost is $500. System cost is $600.",
    "The CPU was active. The CPU was not active.",
    "Latency equals 10ms. Latency equals 20ms.",
    "The test will pass. The test will not pass.",
    "Bandwidth is 1Gbps. Bandwidth is 2Gbps.",
    "The server has memory. The server has not memory.", # slightly weird grammar but should match
    "Power equals 50W. Power equals 60W.",
    "The user can login. The user can not login."
]

def get_temp():
    try:
        temps = []
        for zone in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
            with open(zone, 'r') as f:
                val = f.read().strip()
                if val:
                    temps.append(float(val) / 1000.0)
        return max(temps) if temps else None
    except Exception:
        return None

def main():
    verifier = SemanticConsistencyVerifier()
    
    start_time = time.time()
    
    detected = 0
    latencies = []
    
    for ex in examples:
        t0 = time.time()
        score = verifier.score(ex)
        t1 = time.time()
        
        latencies.append((t1 - t0) * 1000.0)
        
        if score > 0.0:  # Inconsistency detected
            detected += 1
            
    total_time = time.time() - start_time
    
    latencies.sort()
    p50_latency = latencies[len(latencies)//2]
    
    tpr = detected / len(examples)
    
    temp = get_temp()
    
    result = {
        "install_succeeded": True,
        "tpr_observed": float(tpr),
        "per_example_latency_ms_p50": float(p50_latency),
        "soc_temp_max_c": temp,
        "run_duration_s": int(total_time)
    }
    
    print("RESULTS_JSON=" + json.dumps(result))

if __name__ == "__main__":
    main()
