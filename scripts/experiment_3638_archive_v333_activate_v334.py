"""Run Exp 3638 archive/activation aggregation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from carnot.reporting.archive_v333_activate_v334_3638 import main


if __name__ == "__main__":
    main()
