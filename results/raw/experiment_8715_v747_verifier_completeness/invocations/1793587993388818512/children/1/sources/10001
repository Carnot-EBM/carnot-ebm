"""REQ-VERIFY-8629: direct bounded CLI works outside the checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.streaming_source_runner_8629 import main

if __name__ == "__main__":
    raise SystemExit(main())
