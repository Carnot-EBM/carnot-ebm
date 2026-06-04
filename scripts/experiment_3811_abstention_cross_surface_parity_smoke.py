#!/usr/bin/env python3
"""Run Exp 3811 abstention cross-surface parity smoke.

Spec: REQ-SPOE-3811, SCENARIO-SPOE-3811.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carnot.reporting import abstention_cross_surface_parity_smoke_3811 as exp3811


REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--executable", default=None)
    args = parser.parse_args(argv)

    artifact = exp3811.run(
        REPO_ROOT,
        output_path=args.output,
        executable=args.executable,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["honest_verdict"].startswith("complete:") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
