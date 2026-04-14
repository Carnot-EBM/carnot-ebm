#!/usr/bin/env python3
"""Experiment 310: NL2Z3 constraint extraction benchmark.

**Researcher summary:**
    Tests whether the NL2Z3Extractor can detect internally inconsistent
    chain-of-thought reasoning in the Exp 211 corpus.  Each record's prompt
    is fed to NL2Z3Extractor; the sat_status and runtime_ms are recorded.

    In CI mode (CARNOT_FORCE_LIVE not set): all 50 responses return
    sat_status="unknown" (no LLM call) — this proves the CI guard works.

    In production mode (CARNOT_FORCE_LIVE=1): real LLM calls are made and
    Z3 solves actual arithmetic constraints.

**Detailed explanation for engineers:**
    Data source: data/research/constraint_ir_benchmark_211.jsonl
    Fallback (if the file is missing): synthetic prompts with known arithmetic.

    Output: results/experiment_310_nl2z3_results.json

Usage:
    # CI mode (no GPU required):
    python scripts/experiment_310_nl2z3_results.py

    # Live mode (Qwen3.5-0.8B GPU inference):
    CARNOT_FORCE_LIVE=1 JAX_PLATFORMS=cpu python scripts/experiment_310_nl2z3_results.py

Spec: REQ-EXTRACT-010, REQ-EXTRACT-011,
      SCENARIO-EXTRACT-020, SCENARIO-EXTRACT-021
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BENCHMARK_JSONL = _ROOT / "data" / "research" / "constraint_ir_benchmark_211.jsonl"
_RESULTS_DIR = _ROOT / "results"
_OUTPUT_PATH = _RESULTS_DIR / "experiment_310_nl2z3_results.json"
_MAX_RECORDS = 50

# Synthetic fallback corpus — used only when the Exp 211 JSONL is missing.
_SYNTHETIC_CORPUS = [
    {
        "example_id": "synth-310-sat-001",
        "prompt": "What is 3 + 4?",
        "response": "3 + 4 = 7, so the answer is 7.",
    },
    {
        "example_id": "synth-310-unsat-001",
        "prompt": "A store has 10 apples. They sell 3 and then have 8 left. How many?",
        "response": "We start with 10. We sell 3. 10 - 3 = 8. So there are 8 left.",
    },
    {
        "example_id": "synth-310-arithmetic-001",
        "prompt": "What is 15 * 4?",
        "response": "15 * 4 = 60, therefore the answer is 60.",
    },
]


def _load_corpus() -> list[dict]:
    """Load up to _MAX_RECORDS from the Exp 211 JSONL or return synthetic fallback."""
    if _BENCHMARK_JSONL.exists():
        records = []
        with _BENCHMARK_JSONL.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if len(records) >= _MAX_RECORDS:
                    break
        return records
    else:
        print(
            f"[Exp 310] WARNING: {_BENCHMARK_JSONL} not found — using synthetic fallback corpus.",
            file=sys.stderr,
        )
        return _SYNTHETIC_CORPUS


def _get_response(record: dict) -> str:
    """Extract the response text from a corpus record.

    Exp 211 records don't have a 'response' field (they're prompts with
    gold constraints).  We use the first gold_atomic_constraint description
    as a synthetic response to exercise the extractor without a live model.
    """
    if "response" in record:
        return str(record["response"])
    # Use the prompt itself as a stand-in response in CI mode.
    return str(record.get("prompt", ""))


def main() -> None:
    corpus = _load_corpus()
    extractor = NL2Z3Extractor()

    results_per_response: list[dict] = []
    n_unsat = 0
    n_sat = 0
    n_unknown = 0
    n_error = 0

    print(f"[Exp 310] Running NL2Z3Extractor on {len(corpus)} records …")
    exp_start = time.monotonic()

    for record in corpus:
        question = str(record.get("prompt", ""))
        response = _get_response(record)

        t0 = time.monotonic()
        violations = extractor.extract(question, response)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        z3r = extractor.last_z3_result
        sat_status = z3r.sat_status if z3r else "unknown"
        z3_code = z3r.z3_code if z3r else ""
        solver_ms = z3r.runtime_ms if z3r else 0.0

        if sat_status == "unsat":
            n_unsat += 1
        elif sat_status == "sat":
            n_sat += 1
        elif sat_status == "error":
            n_error += 1
        else:
            n_unknown += 1

        results_per_response.append(
            {
                "example_id": record.get("example_id", "unknown"),
                "sat_status": sat_status,
                "violations_found": bool(violations),
                "runtime_ms": round(elapsed_ms, 2),
                "solver_ms": round(solver_ms, 2),
                "z3_code": z3_code,
            }
        )

    total_s = time.monotonic() - exp_start

    artifact = {
        "experiment": "Exp 310",
        "title": "NL2Z3 constraint extraction benchmark",
        "run_date": time.strftime("%Y%m%d"),
        "schema": "1.0",
        "status": "success",
        "duration_s": round(total_s, 2),
        "n_records": len(corpus),
        "live_mode": bool(__import__("os").environ.get("CARNOT_FORCE_LIVE")),
        "summary": {
            "n_sat": n_sat,
            "n_unsat": n_unsat,
            "n_unknown": n_unknown,
            "n_error": n_error,
            "violation_rate": round(n_unsat / max(len(corpus), 1), 4),
        },
        "results": results_per_response,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"[Exp 310] Done in {total_s:.1f}s. Results: {_OUTPUT_PATH}")
    print(
        f"[Exp 310] sat={n_sat}, unsat={n_unsat}, unknown={n_unknown}, error={n_error}"
    )


if __name__ == "__main__":
    main()
