"""Runnable script for Exp 4324 archive / activation."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = SCRIPT_REPO_ROOT / "python"
for path in (SCRIPT_REPO_ROOT, PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from carnot.reporting.archive_v399_activate_v400_4324 import REPO_ROOT, run


def main() -> int:
    """Run the Exp 4324 archive workflow from the repository root."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
