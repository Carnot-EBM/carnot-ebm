#!/usr/bin/env python3
"""REQ-REPORT-8057: resolve local code even when the caller is outside the checkout."""

from pathlib import Path
import sys

print("Exp8057 start: fixture consumers and complete authority; model invocations=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v698_fixture_consumer_contract import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
