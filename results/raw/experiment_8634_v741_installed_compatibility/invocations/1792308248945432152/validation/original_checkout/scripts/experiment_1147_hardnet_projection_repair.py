#!/usr/bin/env python3
"""Run Exp 1147: HardNet++-style projection repair for arithmetic constraints.

Spec: REQ-VERIFY-1147, SCENARIO-VERIFY-1147
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.repair.projection_repair import ArithmeticProjectionRepair  # noqa: E402
from carnot.verify.z3_math_verifier import Z3MathVerifier  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1147_hardnet_projection_repair.json"
MODULE_PATH = "python/carnot/repair/projection_repair.py"

SYNTHETIC_VIOLATIONS: tuple[tuple[str, int], ...] = (
    ("47 + 28 = 76", 75),
    ("3 * 7 = 22", 21),
    ("100 - 37 = 64", 63),
    ("144 / 12 = 11", 12),
    ("12 + 15 = 28", 27),
    ("8 * 9 = 73", 72),
    ("50 - 19 = 32", 31),
    ("81 / 9 = 8", 9),
    ("6 + 14 = 21", 20),
    ("17 * 4 = 69", 68),
    ("200 - 125 = 76", 75),
    ("45 / 5 = 8", 9),
    ("11 + 22 = 34", 33),
    ("13 * 6 = 79", 78),
    ("90 - 48 = 43", 42),
    ("64 / 8 = 7", 8),
    ("25 + 17 = 41", 42),
    ("7 * 8 = 55", 56),
    ("123 - 45 = 77", 78),
    ("99 / 11 = 8", 9),
)


def estimate_prompt_repair_latency_s(results_dir: Path = REPO_ROOT / "results") -> float:
    """Estimate one prompt-repair LLM call from prior self-repair artifacts."""

    candidates = [
        results_dir / "experiment_930_math_iterative_self_repair_v1.json",
        results_dir / "experiment_906_code_repair_50q_scaleup.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        artifact = json.loads(path.read_text(encoding="utf-8"))
        elapsed_values: list[float] = []
        attempt_counts: list[int] = []
        for key, value in artifact.items():
            if not key.endswith("results_per_problem") or not isinstance(value, list):
                continue
            for row in value:
                if not isinstance(row, dict):
                    continue
                elapsed = row.get("elapsed_s")
                attempts = row.get("n_attempts", 1)
                if isinstance(elapsed, int | float) and isinstance(attempts, int | float):
                    elapsed_values.append(float(elapsed))
                    attempt_counts.append(max(1, int(attempts)))
        if elapsed_values:
            return sum(elapsed_values) / sum(attempt_counts)
        duration = artifact.get("duration_s")
        n_problems = artifact.get("n_problems")
        if isinstance(duration, int | float) and isinstance(n_problems, int | float) and n_problems:
            return float(duration) / float(n_problems)
    return 1.0


def derive_honest_verdict(accuracy: float, speedup_factor: float) -> str:
    """Map projection accuracy and latency to the allowed Exp 1147 verdicts."""

    if accuracy == 1.0 and speedup_factor > 1.0:
        return "projection_accurate_and_fast"
    if accuracy > 0.0:
        return "projection_partial_not_all_repaired"
    return "projection_fails_complex_constraints"


def run_experiment(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    """Run 20 synthetic projection repairs and write the Exp 1147 artifact."""

    repairer = ArithmeticProjectionRepair()
    verifier = Z3MathVerifier()
    latencies_us: list[float] = []
    corrected = 0
    per_case: list[dict[str, Any]] = []

    for response, correct in SYNTHETIC_VIOLATIONS:
        violation = {"type": "arithmetic", "constraint": response}
        start_ns = time.perf_counter_ns()
        fixed = repairer.repair(response, violation)
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
        latencies_us.append(elapsed_us)
        passed = verifier.score(fixed) == 0.0 and str(correct) in fixed
        corrected += int(passed)
        per_case.append(
            {
                "response": response,
                "fixed": fixed,
                "correct_answer": correct,
                "z3_passed_after_repair": passed,
                "latency_us": elapsed_us,
            }
        )

    n_violations = len(SYNTHETIC_VIOLATIONS)
    accuracy = corrected / n_violations
    projection_latency_us = statistics.fmean(latencies_us)
    prompt_latency_s = estimate_prompt_repair_latency_s()
    speedup_factor = prompt_latency_s / (projection_latency_us / 1_000_000.0)
    artifact: dict[str, Any] = {
        "experiment": 1147,
        "title": "HardNet++-Style Projection Repair Layer for Arithmetic Constraints",
        "run_date": datetime.now(UTC).strftime("%Y%m%d"),
        "status": "success",
        "projection_repair_written": True,
        "module_path": MODULE_PATH,
        "n_violations_tested": n_violations,
        "projection_repair_accuracy": accuracy,
        "projection_repair_latency_us": projection_latency_us,
        "prompt_repair_latency_s": prompt_latency_s,
        "speedup_factor": speedup_factor,
        "hardnet_projection_repair_written": True,
        "honest_verdict": derive_honest_verdict(accuracy, speedup_factor),
        "per_case_results": per_case,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    """CLI entry point for conductor and manual experiment runs."""

    artifact = run_experiment()
    print(
        "[exp1147] "
        f"accuracy={artifact['projection_repair_accuracy']:.4f} "
        f"latency_us={artifact['projection_repair_latency_us']:.3f} "
        f"prompt_latency_s={artifact['prompt_repair_latency_s']:.3f} "
        f"speedup={artifact['speedup_factor']:.1f} "
        f"verdict={artifact['honest_verdict']} "
        f"output={OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
