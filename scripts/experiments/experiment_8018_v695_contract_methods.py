#!/usr/bin/env python3
"""REQ-REPORT-8018: expose immutable authority qualification and private cold replay."""

import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import v695_contract_methods as methods  # noqa: E402

MODEL_SPECS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    """Private CLI operands permit tests without changing any historical primary."""
    methods.progress("start", pending="arguments")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261002", choices=["20261002"])
    parser.add_argument(
        "--design", type=Path, default=ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
    )
    parser.add_argument("--active", type=Path, default=ROOT / "research-roadmap.yaml")
    parser.add_argument("--staged", type=Path, default=ROOT / "research-roadmap-next.yaml")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/experiment_8018_v695_contract_methods.json"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "results/raw/experiment_8018_v695_contract_methods/rows.json",
    )
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-e2e", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = methods.cold_replay(args.cold_replay, args.raw)
        print("cold_replay_passed" if passed else "cold_replay_mismatch", flush=True)
        return int(not passed)
    if args.fixture_e2e and any(p.resolve().is_relative_to(ROOT) for p in (args.output, args.raw)):
        parser.error("fixture outputs must be private and outside the checkout")
    with tempfile.TemporaryDirectory(prefix="exp8018-") as directory:
        methods.execute(args, Path(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
