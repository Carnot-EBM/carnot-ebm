#!/usr/bin/env python3
"""In-container runner for autoresearch hypothesis execution.

**What this does:**
    This script runs INSIDE the Docker/gVisor sandbox container. It:
    1. Reads hypothesis code from /hypothesis.py (volume-mounted read-only)
    2. Reads benchmark data from /data.json (volume-mounted read-only)
    3. Executes the hypothesis's run(benchmark_data) function
    4. Prints the returned metrics dict as JSON to stdout

    The orchestrator on the host captures stdout to get the metrics.
    Stderr is also captured for debugging.

**Why a separate script?**
    We can't just `python /hypothesis.py` because hypotheses must define
    a `run()` function, not a `__main__` block. This runner provides the
    harness that calls `run()` and handles errors gracefully.

Spec: REQ-AUTO-004
"""

import json
import sys
import time
import traceback


def main() -> int:
    """Load and execute the hypothesis, print metrics as JSON."""
    start = time.monotonic()

    # 1. Load benchmark data
    try:
        with open("/data.json") as f:
            benchmark_data = json.load(f)
    except FileNotFoundError:
        # No benchmark data file — use empty dict
        benchmark_data = {}
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid benchmark data JSON: {e}"}))
        return 1

    # 2. Load hypothesis code
    try:
        with open("/hypothesis.py") as f:
            code = f.read()
    except FileNotFoundError:
        print(json.dumps({"error": "No hypothesis file at /hypothesis.py"}))
        return 1

    # 3. Execute hypothesis
    namespace: dict = {}
    try:
        exec(compile(code, "/hypothesis.py", "exec"), namespace)
    except Exception as e:
        print(json.dumps({
            "error": f"Hypothesis compilation/execution failed: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }))
        return 1

    # 4. Call run() function
    if "run" not in namespace:
        print(json.dumps({"error": "Hypothesis must define a run(benchmark_data) function"}))
        return 1

    try:
        metrics = namespace["run"](benchmark_data)
    except Exception as e:
        print(json.dumps({
            "error": f"run() raised {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }))
        return 1

    if not isinstance(metrics, dict):
        print(json.dumps({"error": f"run() must return a dict, got {type(metrics).__name__}"}))
        return 1

    # 5. Add timing and output
    elapsed = time.monotonic() - start
    metrics["_wall_clock_seconds"] = elapsed

    # Print metrics as JSON to stdout (captured by the host orchestrator)
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
