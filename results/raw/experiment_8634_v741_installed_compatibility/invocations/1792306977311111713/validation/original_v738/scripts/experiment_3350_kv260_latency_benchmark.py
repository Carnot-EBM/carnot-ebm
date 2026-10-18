#!/usr/bin/env python3
"""Run Exp 3350 KV260 latency benchmark.

Spec refs: REQ-HW-101, SCENARIO-HW-101.
"""

from __future__ import annotations

from carnot.hardware.kv260_latency_benchmark_3350 import main

if __name__ == "__main__":
    raise SystemExit(main())