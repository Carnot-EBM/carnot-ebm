#!/usr/bin/env python3
"""REQ-REPORT-8017: expose the cached audit and its independent cold replay."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import v694_capstone as cap  # noqa: E402

MODEL_SPECS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    """Private outputs exercise the same reducer without touching historical primaries."""
    cap.progress("cli_start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=["20261002"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        try:
            errors = cap.cold_replay(json.loads(args.cold_replay.read_bytes()))
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors = [str(error)]
        print(json.dumps(dict(errors=errors)), flush=True)
        return int(bool(errors))
    root = args.root.resolve()
    design = args.design or root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    output = args.output or root / "results/experiment_8017_v694_capstone.json"
    if args.evidence_only:
        cap.atomic_json(
            output, cap.build(root, design, args.date, output.parent / "raw" / output.stem)
        )
        return 0
    return cap.qualify(root, design, args.date, output)


if __name__ == "__main__":
    raise SystemExit(main())
