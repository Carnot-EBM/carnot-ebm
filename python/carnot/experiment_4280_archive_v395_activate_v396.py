"""Conductor entrypoint for Exp 4280 archive / activation."""

from __future__ import annotations

from carnot.reporting.archive_v395_activate_v396_4280 import REPO_ROOT, run


def main() -> int:
    """Run the Exp 4280 record-only archive workflow."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
