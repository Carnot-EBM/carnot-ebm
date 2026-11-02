"""Expose bounded private custody and replay routes. REQ-REPORT-7926-V687."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from carnot.reporting import experiment_7926_v687_hardware_evidence as q
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7913_v686_hardware_evidence import progress
from carnot.reporting.qualification_7926 import qualify

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Keep the driver small so custody and validation remain reusable callables."""
    started = time.monotonic()
    print("[exp7926] start host aggregation; no model or device load", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260930", choices=["20260930"])
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    candidate = args.cold_replay or args.terminal_recheck
    if candidate:
        progress(started, "cold_replay", "before_reduction", 0)
        print(
            json.dumps(q.cold_reduce(root, json.loads(candidate.read_text())), sort_keys=True),
            flush=True,
        )
        progress(started, "cold_replay", "after_reduction", 1)
        return 0
    output = args.output or root / "results/experiment_7926_v687_hardware_evidence.json"
    if args.evidence_only:
        value = q.read_evidence(root, args.date)
        progress(
            started, "preconditions", "resolved_paths_hashes_roles_operands", len(value["rows"])
        )
        atomic_json(output, value)
        return 0
    return qualify(
        root,
        args.date,
        output,
        args.raw_root or root / "results/raw/experiment_7926_v687_hardware_evidence",
    )


if __name__ == "__main__":
    raise SystemExit(main())
