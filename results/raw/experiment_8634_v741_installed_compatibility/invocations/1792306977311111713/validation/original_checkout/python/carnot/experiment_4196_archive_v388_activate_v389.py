"""Conductor entrypoint for Exp 4196 archive / activation."""

from __future__ import annotations

from carnot.reporting.archive_v388_activate_v389_4196 import REPO_ROOT, run


def main() -> int:
    """Run the Exp 4196 record-only archive workflow."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
