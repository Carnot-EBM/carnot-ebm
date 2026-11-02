"""Runnable entrypoint for Exp 4347 archive / activation."""

from __future__ import annotations

from pathlib import Path

from carnot.reporting.archive_v401_activate_v402_4347 import REPO_ROOT, run


def main(root: Path = REPO_ROOT) -> int:
    """Run the Exp 4347 archive workflow from the repository root."""

    output_path = run(root)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
