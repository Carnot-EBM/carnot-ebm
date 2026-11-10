"""REQ-REPORT-8966: execute sealed real-error streams and authenticated replay."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.online_response_stream_runner_8966 import main

if __name__ == "__main__":
    raise SystemExit(main())
