#!/usr/bin/env python3
"""Run Exp 2550 JEPA fast-path balanced text evaluation.

Spec: REQ-JEPA-006, SCENARIO-JEPA-012
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

RESULT_PATH = Path("results/experiment_2550_jepa_real_eval.json")


def _write_blocked_artifact(reason: str, started: float) -> dict[str, Any]:
    artifact = {
        "honest_verdict": f"blocked: {reason}",
        "fast_path_rate": 0.0,
        "fast_path_precision": 0.0,
        "jepa_discrimination_achieved": False,
        "threshold_used": 0.2,
        "n_corpus": 0,
        "preconditions_checked": [
            {
                "resource": "jepa_fast_path_import",
                "available": False,
                "reason": reason,
            }
        ],
        "duration_s": round(time.perf_counter() - started, 6),
        "random_seed": 42,
        "acceptance_gate_passed": False,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> int:
    started = time.perf_counter()
    try:
        from carnot.pipeline.jepa_fast_path_eval import run_experiment
    except Exception as exc:
        artifact = _write_blocked_artifact("blocked_jepa_fast_path_not_found", started)
        artifact["import_error"] = repr(exc)
        RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(json.dumps(artifact, indent=2, sort_keys=True))
        return 1

    artifact = run_experiment(output_path=RESULT_PATH)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact.get("acceptance_gate_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
