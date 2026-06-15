"""Conductor entrypoint for Exp 4255 archive / activation."""

from __future__ import annotations

from carnot.reporting.archive_v393_activate_v394_4255 import REPO_ROOT, run


def main() -> int:
    """Run the Exp 4255 record-only archive workflow."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
