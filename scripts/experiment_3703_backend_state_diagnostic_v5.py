"""Run Exp 3703 backend state diagnostic v5."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from carnot.reporting.backend_state_diagnostic_v5_3703 import main


if __name__ == "__main__":
    main()
