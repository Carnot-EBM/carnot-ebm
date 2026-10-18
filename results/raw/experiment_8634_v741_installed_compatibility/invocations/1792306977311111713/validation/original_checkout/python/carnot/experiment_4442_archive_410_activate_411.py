"""Runnable entrypoint for Exp 4442 archive / activation."""

from __future__ import annotations

from pathlib import Path

from carnot.reporting.archive_410_activate_411_4442 import REPO_ROOT, run


def main(root: Path = REPO_ROOT) -> int:
    """Run the Exp 4442 archive workflow from the repository root."""

    output_path = run(root)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
