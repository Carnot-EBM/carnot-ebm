#!/usr/bin/env python3
"""REQ-REPORT-7914-V686: expose explicit paths for private capstone reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import v686_capstone as cap  # noqa: E402
from carnot.reporting.current_work_receipt import atomic_json  # noqa: E402

MODEL_SPECS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    """Small explicit routes allow real CLI coverage without publishing old authorities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=["20260930"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--active", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    design = args.design or root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    active = args.active or root / "research-roadmap.yaml"
    output = args.output or root / "results/experiment_7914_v686_capstone.json"
    if args.cold_replay:
        errors = cap.cold_replay(json.loads(args.cold_replay.read_bytes()), root, design, active)
        print(json.dumps({"errors": errors}), flush=True)
        return int(bool(errors))
    if args.evidence_only:
        atomic_json(output, cap.build_candidate(root, design, active, args.date))
        return 0
    return cap.qualify(root, design, active, args.date, output)


if __name__ == "__main__":
    raise SystemExit(main())
