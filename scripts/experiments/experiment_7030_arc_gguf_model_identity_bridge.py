#!/usr/bin/env python3
"""Run the deterministic REQ-ARC-7030 identity bridge experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.experiment_7030_arc_gguf_model_identity_bridge import (
    RESULT_RELATIVE_PATH,
    run,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Write one stable artifact for the requested execution date."""

    parser = argparse.ArgumentParser(description="Build the ARC GGUF model identity bridge artifact")
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args()
    artifact = run(ROOT, execution_date=args.date, output_path=args.output)
    print(
        f"wrote {args.output} arc_model_identity_bridge_ready_score="
        f"{artifact['arc_model_identity_bridge_ready_score']} "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
