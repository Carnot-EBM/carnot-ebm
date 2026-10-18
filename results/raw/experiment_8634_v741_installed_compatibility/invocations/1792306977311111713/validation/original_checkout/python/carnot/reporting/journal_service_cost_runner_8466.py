"""REQ-REPORT-8466: dated thin adapter to the unchanged private publication workflow."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from unittest.mock import patch

from carnot.reporting import journal_service_cost_8466 as r
from carnot.reporting import journal_equivalence_runner_8465 as shipped


def resume_work(path: Path, root: Path, raw: Path, private: Path) -> dict[str, object]:
    """Retain original measured operands and failed-attempt evidence before rerunning checks."""
    work = deepcopy(r.read(r.reference(path)))
    versions_path = path.parent / "original_owned_sources.json"
    versions = json.loads(versions_path.read_bytes())
    if isinstance(versions, dict):
        versions = versions["rows"]
    originals = {ref["source_path"]: ref for ref in versions}
    work["refs"] += [r.reference(path), r.reference(versions_path)]
    work["code_config_hashes"] = [
        originals.get(ref["path"], ref) for ref in work["code_config_hashes"]
    ]
    work["code_config_hashes"] += [r.reference(r.ROOT / p) for p in [*r.OWNED, r.TEST]]
    work["raw"] = str(raw)
    work["prior_attempts"] = [
        dict(
            work_reference=r.reference(path),
            measured_arms=len(work["primitives"]),
            reason="owned reporting correction; retain every original repeat and source version",
        )
    ]
    return dict(work)


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise real exits without recursively invoking validation."""
    r.progress("start_no_model_load_MODEL_SPECS_empty_current_llm_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--output", type=Path, default=r.ROOT / "results" / (r.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--resume-work", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--arm", choices=["snapshot", "journal"])
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.worker:
        return r.worker(args.worker, args.store, args.arm)
    if args.cold_replay:
        passed = r.replay(args.cold_replay)
        r.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(r.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="carnot8466-", dir="/var/tmp") as folder:
            with patch.object(shipped, "r", r):
                if args.resume_work:
                    with patch.object(
                        r,
                        "measure",
                        lambda root, raw, private: resume_work(
                            args.resume_work, root, raw, private
                        ),
                    ):
                        return int(
                            shipped.run(args.root, output, Path(folder), args.private_control)
                        )
                return int(shipped.run(args.root, output, Path(folder), args.private_control))
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
