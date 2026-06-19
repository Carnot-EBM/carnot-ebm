"""Runnable entrypoint for Exp 4466 archive / activation."""

from __future__ import annotations

from pathlib import Path

from carnot.reporting.archive_412_activate_413_4466 import REPO_ROOT, run


def main(root: Path = REPO_ROOT) -> int:
    """Run the Exp 4466 archive workflow from the repository root."""

    output_path = run(root)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
