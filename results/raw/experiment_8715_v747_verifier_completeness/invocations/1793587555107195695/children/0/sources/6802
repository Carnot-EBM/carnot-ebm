import json
import time
import threading
from typing import Dict, Any
from unittest.mock import MagicMock
from carnot.training.telemetry_streamer import TelemetryStreamer
from carnot.pipeline.verify_repair import VerificationResult

def run_experiment() -> Dict[str, Any]:
    streamer = TelemetryStreamer(max_size=1000)
    streamer.start()
    
    n_threads = 5
    n_items_per_thread = 100
    
    def worker() -> None:
        for _ in range(n_items_per_thread):
            res = MagicMock(spec=VerificationResult)
            streamer.record(res)
            time.sleep(0.001)
            
    threads = []
    start_time = time.time()
    
    for _ in range(n_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    streamer.stop()
    end_time = time.time()
    
    total_processed = len(streamer.results)
    expected = n_threads * n_items_per_thread
    
    result = {
        "experiment_id": "1738",
        "name": "Async Telemetry Streamer Load Test",
        "threads": n_threads,
        "items_per_thread": n_items_per_thread,
        "total_processed": total_processed,
        "expected_items": expected,
        "success": total_processed == expected,
        "duration_seconds": end_time - start_time,
        "honest_verdict": "streamer_load_test_successful" if total_processed == expected else "streamer_load_test_failed"
    }
    
    with open("results/experiment_1738_telemetry.json", "w") as f:
        json.dump(result, f, indent=2)
        
    return result

if __name__ == "__main__":
    result = run_experiment()
    print(json.dumps(result, indent=2))
