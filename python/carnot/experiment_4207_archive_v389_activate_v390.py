"""Conductor entrypoint for Exp 4207 archive / activation."""

from __future__ import annotations

from carnot.reporting.archive_v389_activate_v390_4207 import REPO_ROOT, run


def main() -> int:
    """Run the Exp 4207 record-only archive workflow."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
