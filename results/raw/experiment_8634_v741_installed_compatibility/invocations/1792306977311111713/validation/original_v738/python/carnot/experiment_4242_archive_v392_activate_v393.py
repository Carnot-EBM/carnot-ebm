"""Conductor entrypoint for Exp 4242 archive / activation."""

from __future__ import annotations

from carnot.reporting.archive_v392_activate_v393_4242 import REPO_ROOT, run


def main() -> int:
    """Run the Exp 4242 record-only archive workflow."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
