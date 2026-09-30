"""Run isolated CLI training qualification (REQ-REPORT-7941)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.verify import training_publication_7941 as core  # noqa: E402
from carnot.verify.training_publication_7941_run import qualify  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Explicit private routes keep historical publishers away from current authorities."""
    core.progress("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=["20260930"])
    parser.add_argument("--runtime", type=Path, default=core.RUNTIME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--assert-ready", action="store_true")
    parser.add_argument("--publication-fault", choices=["conflict", "reject"], default="")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay or args.terminal_recheck:
            core.replay(args.cold_replay or args.terminal_recheck, bool(args.terminal_recheck))
            return 0
        output = args.output or core.ROOT / "results/experiment_7941_v689_training_publication.json"
        if args.fixture:
            result = core.produce(args.runtime, output, args.publication_fault)
        else:
            result = qualify(output)
        value = json.loads(output.read_text())
        if args.assert_ready and value["runtime_ready_score"] != 1:
            core.progress("source evidence blocked")
            return 2
        return result
    except (ValueError, OSError, KeyError) as exc:
        core.progress("rejected " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
