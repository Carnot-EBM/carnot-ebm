"""REQ-REPORT-8405: direct entry keeps cold replay independent of ambient imports."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.native_parity_8405 import main

if __name__ == "__main__":
    raise SystemExit(main())
