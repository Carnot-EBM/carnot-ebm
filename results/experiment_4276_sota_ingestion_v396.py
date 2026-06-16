"""Run Exp 4276 SOTA ingestion from the required results entry point."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def main() -> int:
    from carnot import experiment_4276_sota_ingestion_v396 as exp4276

    return exp4276.main()


if __name__ == "__main__":
    raise SystemExit(main())
