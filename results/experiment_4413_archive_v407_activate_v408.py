"""Runnable script for Exp 4413 archive / activation."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = SCRIPT_REPO_ROOT / "python"
for path in (SCRIPT_REPO_ROOT, PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from carnot.reporting.archive_v407_activate_v408_4413 import REPO_ROOT, run  # noqa: E402


def main(root: Path = REPO_ROOT) -> int:
    """Run the Exp 4413 archive workflow from the repository root."""

    output_path = run(root)
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
