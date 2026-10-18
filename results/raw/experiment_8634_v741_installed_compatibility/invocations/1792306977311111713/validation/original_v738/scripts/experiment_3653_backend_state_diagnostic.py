"""Run Exp 3653 backend state diagnostic."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from carnot.reporting.backend_state_diagnostic_3653 import main


if __name__ == "__main__":
    main()
