"""Runnable script for Exp 4290 archive / activation."""

from __future__ import annotations

from carnot.reporting.archive_v396_activate_v397_4290 import REPO_ROOT, run


def main() -> int:
    """Run the Exp 4290 archive workflow from the repository root."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
