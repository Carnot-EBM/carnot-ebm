#!/usr/bin/env python3
"""REQ-REPORT-7990: bounded audit entrypoint with zero model and device calls."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import experiment_7990_v692_hardware_evidence as q  # noqa: E402
from carnot.reporting import validation_7990 as plan  # noqa: E402
from carnot.reporting.primary_publication import publish_primary  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Private fixture and cold replay routes preserve the historical source files."""
    print("[exp7990] phase=start zero_model_loads zero_generation zero_device_calls", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    candidate = args.cold_replay or args.terminal_recheck
    if candidate:
        print("[exp7990] phase=cold_replay before_reduction", flush=True)
        print(
            json.dumps(q.cold_reduce(root, json.loads(candidate.read_text())), sort_keys=True),
            flush=True,
        )
        print("[exp7990] phase=cold_replay after_reduction", flush=True)
        return 0
    output = args.output or root / "results/experiment_7990_v692_hardware_evidence.json"
    if args.evidence_only:
        value = q.read_evidence(root, args.date)
        publish_primary(
            output,
            value,
            lambda p: dict(passed=True, reduction=q.cold_reduce(root, json.loads(p.read_text()))),
        )
        print("[exp7990] phase=private_fixture published", flush=True)
        return 0
    return plan.qualify(root, args.date, output)


if __name__ == "__main__":
    raise SystemExit(main())
