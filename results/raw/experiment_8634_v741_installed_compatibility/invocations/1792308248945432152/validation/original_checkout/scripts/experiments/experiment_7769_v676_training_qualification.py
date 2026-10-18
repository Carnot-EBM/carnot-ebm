#!/usr/bin/env python3
"""Run the V676 finite training qualification and its private E2E smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from carnot import experiment_7769_v676_training_qualification as experiment
from carnot.reporting.current_work_receipt import atomic_json


def main() -> int:
    """Dispatch one live qualification or one isolated real entrypoint check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--private-e2e", action="store_true")
    args = parser.parse_args()
    if args.private_e2e:
        with tempfile.TemporaryDirectory(prefix="exp7769-e2e-") as private:
            folder = Path(private)
            artifact = experiment.run_fixture(folder / "raw", args.date)
            candidate = folder / "candidate.json"
            atomic_json(candidate, artifact)
            result = experiment.cold_reduce(candidate)
            print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    artifact = experiment.launch(args.date)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "training_runtime_ready_score": artifact["training_runtime_ready_score"],
                "online_runtime_ready_score": artifact["online_runtime_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - real entrypoint check covers this.
    raise SystemExit(main())
