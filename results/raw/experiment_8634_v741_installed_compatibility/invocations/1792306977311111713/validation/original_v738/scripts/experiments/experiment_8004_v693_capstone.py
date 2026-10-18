#!/usr/bin/env python3
"""REQ-REPORT-8004: expose a bounded cached audit and independent cold replay."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import v693_capstone as cap  # noqa: E402
from carnot.reporting import v693_capstone_validation as validation  # noqa: E402

MODEL_SPECS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    """Private CLI routes exercise the same evidence collector as current execution."""
    cap.progress("cli_start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=["20261002"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--active", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--missing-producer-fixture", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--expect-rejection", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        try:
            errors = cap.cold_replay(json.loads(args.cold_replay.read_bytes()))
        except (ValueError, KeyError, OSError, TypeError) as error:
            errors = [str(error)]
        print(json.dumps(dict(errors=errors)), flush=True)
        return int(not errors) if args.expect_rejection else int(bool(errors))
    root = args.root.resolve()
    active = args.active or root / "research-roadmap.yaml"
    design = args.design or root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    output = args.output or root / "results/experiment_8004_v693_capstone.json"
    if args.evidence_only:
        if args.missing_producer_fixture:
            tasks = cap.yaml.safe_load(active.read_bytes())["tasks"]
            (root / tasks[0]["deliverable"]).unlink(missing_ok=True)
        value = cap.build(root, active, design, args.date, output.parent / "authority")
        atomic = cap.atomic_json
        atomic(output, value)
        return 0
    return validation.qualify(root, active, design, args.date, output)


if __name__ == "__main__":
    raise SystemExit(main())
