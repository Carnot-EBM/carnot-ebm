#!/usr/bin/env python3
"""REQ-REPORT-7939-V688: expose reduction through a small parameterized CLI."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import v688_capstone as cap  # noqa: E402
from carnot.reporting import v688_capstone_validation as validation  # noqa: E402

MODEL_SPECS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    """Private routes qualify actual execution without changing live authorities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=["20260930"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--active", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    design = args.design or root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    active = args.active or root / "research-roadmap.yaml"
    output = args.output or root / "results/experiment_7939_v688_capstone.json"
    print("[exp7939] start current capstone CLI", flush=True)
    if args.cold_replay:
        errors = cap.cold_replay(json.loads(args.cold_replay.read_bytes()), root, design, active)
        print(json.dumps(dict(errors=errors)), flush=True)
        return int(bool(errors))
    if args.terminal_recheck:
        private = Path(tempfile.mkdtemp(prefix="carnot-7939-terminal-"))
        validation.terminal(
            json.loads(args.terminal_recheck.read_bytes()),
            output,
            private,
            output.parent / "raw" / output.stem,
        )
        return 0
    if args.evidence_only:
        cap.atomic_json(output, cap.build_candidate(root, design, active, args.date))
        return 0
    return validation.qualify(root, design, active, args.date, output)


if __name__ == "__main__":
    raise SystemExit(main())
