"""REQ-REPORT-8889: resolve the actual CLI outside the checkout for cold replay."""

from pathlib import Path
import sys

print("[exp8889] phase=imports_before", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.hardware_operation_map_8889 import main

print("[exp8889] phase=imports_after", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
