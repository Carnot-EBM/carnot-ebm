#!/usr/bin/env python3
"""Run the current numerical and online fixture qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carnot import experiment_7784_v677_training_runtime as experiment


def main(argv: list[str] | None = None) -> int:
    """Dispatch a live fixture or a fresh-process raw-row reduction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    result = (
        experiment.cold_reduce(args.cold_reduce) if args.cold_reduce else experiment.run(args.date)
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
