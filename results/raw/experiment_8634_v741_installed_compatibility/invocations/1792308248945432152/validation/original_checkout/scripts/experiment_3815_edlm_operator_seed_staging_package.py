#!/usr/bin/env python3
"""Run Exp 3815 EDLM operator seed staging package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
for path in (REPO_ROOT, PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from carnot.reporting import edlm_operator_seed_staging_3815 as exp3815  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--executable", default=None)
    args = parser.parse_args(argv)

    output_path = exp3815.run(
        REPO_ROOT,
        executable=args.executable,
        output_path=args.output,
    )
    print(json.dumps(json.loads(output_path.read_text(encoding="utf-8")), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
