"""REQ-REPORT-7951: host aggregation with typed missing custody and zero models."""

import argparse
import json
from pathlib import Path
import time

from carnot.reporting import experiment_7951_v689_hardware_evidence as q
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7913_v686_hardware_evidence import progress
from carnot.reporting.primary_publication import publish_primary

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Explicit input and output roots keep private routes away from authorities."""
    started = time.monotonic()
    print("[exp7951] start host aggregation; zero model and device operations", flush=True)
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
    output = args.output or root / "results/experiment_7951_v689_hardware_evidence.json"
    if args.evidence_only:
        value = q.read_evidence(root, args.date)
        progress(
            started, "preconditions", "resolved_paths_hashes_roles_operands", len(value["rows"])
        )
        publish_primary(
            output,
            value,
            lambda path: {
                "passed": True,
                "cold_reduction": q.cold_reduce(root, json.loads(path.read_text())),
            },
        )
        progress(started, "evidence_publication", "checked_bytes_published", 1)
        return 0
    return q.qualify(
        root,
        args.date,
        output,
        args.raw_root or root / "results/raw/experiment_7951_v689_hardware_evidence",
    )


if __name__ == "__main__":
    raise SystemExit(main())
