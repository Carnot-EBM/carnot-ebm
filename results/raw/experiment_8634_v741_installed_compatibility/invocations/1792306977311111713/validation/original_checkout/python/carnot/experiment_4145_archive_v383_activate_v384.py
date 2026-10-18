"""Conductor entrypoint for Exp 4145 archive / activation."""

from __future__ import annotations

from pathlib import Path

from carnot.reporting.archive_v383_activate_v384_4145 import REPO_ROOT, run


def main() -> int:
    """Run the Exp 4145 record-only archive workflow."""

    output_path: Path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
