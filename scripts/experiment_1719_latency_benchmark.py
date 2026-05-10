import time
import json
from carnot.pipeline.hiled_decoder import HiledDecoder

def run_benchmark():
    tokens = ["hello", " world", " this", " is", " a", " test", " of", " the", " hiled", " decoder"]
    
    # Baseline
    decoder_baseline = HiledDecoder(hardware_latency_ms=2.0, use_hiled=False)
    start = time.time()
    for t in tokens:
        decoder_baseline.decode_token(t)
    end = time.time()
    latency_baseline_ms = ((end - start) / len(tokens)) * 1000.0
    
    # HILED
    decoder_hiled = HiledDecoder(hardware_latency_ms=2.0, use_hiled=True)
    start = time.time()
    for t in tokens:
        decoder_hiled.decode_token(t)
    end = time.time()
    latency_hiled_ms = ((end - start) / len(tokens)) * 1000.0
    
    projection_tax_ms = latency_hiled_ms - latency_baseline_ms
    
    result = {
        "latency_per_token_hiled_ms": latency_hiled_ms,
        "latency_per_token_baseline_ms": latency_baseline_ms,
        "projection_tax_ms": projection_tax_ms,
        "honest_verdict": "hiled_latency_measured"
    }
    
    with open("results/experiment_1719_latency.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Benchmark completed. Baseline: {latency_baseline_ms:.2f}ms, HILED: {latency_hiled_ms:.2f}ms, Tax: {projection_tax_ms:.2f}ms")

if __name__ == "__main__":
    run_benchmark()
