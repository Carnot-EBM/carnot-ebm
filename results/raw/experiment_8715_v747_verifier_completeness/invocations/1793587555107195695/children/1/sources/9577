#!/usr/bin/env python3
"""Publish authority and methods without running science (REQ-REPORT-7992-V692)."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from carnot.reporting.v693_contract_methods import cold_replay  # noqa: E402
from carnot.reporting.v693_contract_validation import execute  # noqa: E402

MODEL_SPECS: list[str] = []


def main() -> int:
    """Private CLI paths permit real negative checks without replacing live evidence."""
    print("[exp7992] phase=start completed_units=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261001", choices=["20261001"])
    parser.add_argument(
        "--design", type=Path, default=ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
    )
    parser.add_argument("--staged", type=Path, default=ROOT / "research-roadmap-next.yaml")
    parser.add_argument("--active", type=Path, default=ROOT / "research-roadmap.yaml")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/experiment_7892_v685_source_boundary.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/experiment_7992_v693_contract_methods.json"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "results/raw/experiment_7992_v693_contract_methods/rows.json",
    )
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--expect-rejection", action="store_true")
    parser.add_argument("--fixture-e2e", action="store_true")
    args = parser.parse_args()
    if args.expect_rejection:
        argv = [a for a in sys.argv[1:] if a != "--expect-rejection"]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        print("[exp7992] phase=negative_child_before", flush=True)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *argv],
            cwd=tempfile.gettempdir(),
            env=env,
            timeout=60,
            check=False,
        )
        print(f"[exp7992] phase=negative_child_after inner_exit={result.returncode}", flush=True)
        return 0 if result.returncode == 1 else 1
    if args.cold_replay:
        passed = cold_replay(args.cold_replay, args.raw)
        print("cold_replay_passed" if passed else "cold_replay_mismatch", flush=True)
        return 0 if passed else 1
    if args.fixture_e2e and any(p.resolve().is_relative_to(ROOT) for p in (args.output, args.raw)):
        parser.error("fixture outputs must be private and outside the checkout")
    with tempfile.TemporaryDirectory(prefix="exp7992-") as directory:
        execute(ROOT, args, Path(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
