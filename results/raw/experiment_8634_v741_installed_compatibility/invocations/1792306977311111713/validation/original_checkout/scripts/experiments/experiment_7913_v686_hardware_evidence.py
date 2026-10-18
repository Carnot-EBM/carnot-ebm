"""REQ-REPORT-7913-V686: expose explicit paths for private board custody."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from carnot.reporting import experiment_7913_v686_hardware_evidence as q
from carnot.reporting.current_work_receipt import atomic_json

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """The small CLI delegates reduction and bounded validation to callables."""
    started = time.monotonic()
    q.progress(started, "entrypoint", "start", 0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260930", choices=["20260930"])
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay:
        q.progress(started, "cold_replay", "before_reduction", 0)
        print(
            json.dumps(
                q.cold_reduce(root, json.loads(args.cold_replay.read_text())), sort_keys=True
            ),
            flush=True,
        )
        q.progress(started, "cold_replay", "after_reduction", 1)
        return 0
    output = args.output or root / "results/experiment_7913_v686_hardware_evidence.json"
    if args.evidence_only:
        result = q.read_evidence(root, args.date)
        q.progress(
            started, "preconditions", "resolved_paths_hashes_roles_operands", len(result["rows"])
        )
        atomic_json(output, result)
        q.progress(started, "evidence_only", "written", len(result["rows"]))
        return 0
    return q.qualify(
        root,
        args.date,
        output,
        args.raw_root or root / "results/raw/experiment_7913_v686_hardware_evidence",
    )


if __name__ == "__main__":
    raise SystemExit(main())
