#!/usr/bin/env python3
"""REQ-VERIFY-8468: dated thin CLI owns only bounded private processes."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting import arc_generalization_panel_8468 as p  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []
SCRATCH = Path("/var/tmp/carnot-exp8468-private")


def main(argv: list[str] | None = None) -> int:
    """Disk scratch and a hard task cap preserve actual evidence without padding clocks."""
    p.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
    parser.add_argument("--output", type=Path, default=p.OUTPUT)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args(argv)
    if args.private_e2e and args.output.resolve().is_relative_to(p.ROOT):
        parser.error("private E2E output must be outside the checkout")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        os.environ.update(TMPDIR=str(SCRATCH), PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
        if args.cold_replay:
            try:
                passed = p.replay(json.loads(args.cold_replay.read_bytes()))
            except (OSError, ValueError):
                passed = False
            p.progress("cold_replay_after", int(passed), int(not passed))
            return int(not passed)
        if args.episode:
            try:
                unit = json.loads(args.episode.read_bytes())
                if args.raw is None:
                    parser.error("--episode requires --raw")
                result = p.runtime.episode(unit, p.sdk(), args.raw)
                return int(bool(result["error"]))
            except (OSError, ValueError, ImportError) as error:
                print(f"[exp8468] episode_error={error}", flush=True)
                return 1
        with TemporaryDirectory(prefix="invocation-", dir=SCRATCH) as private:
            return p.run(args.output, Path(private), private_e2e=args.private_e2e)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


if __name__ == "__main__":
    raise SystemExit(main())
