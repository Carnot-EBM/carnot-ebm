"""REQ-REPORT-8968: measure complete cached-response service or replay its receipts."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

if "--segment" in sys.argv:
    import argparse
    from carnot.memory.complete_service_worker_8968 import run

    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--stop", type=int, choices=(16, 32), required=True)
    args = parser.parse_args()
    run(args.segment, args.stop)
else:
    from carnot.reporting.complete_service_cost_runner_8968 import main

    if __name__ == "__main__":
        raise SystemExit(main())
