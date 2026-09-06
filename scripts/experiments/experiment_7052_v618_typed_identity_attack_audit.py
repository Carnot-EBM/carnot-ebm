#!/usr/bin/env python3
"""Run the deterministic REQ-ARC-7052 typed identity attack audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.experiment_7052_v618_typed_identity_attack_audit import (
    RESULT_RELATIVE_PATH,
    run,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Write one terminal audit artifact for the requested execution date."""

    parser = argparse.ArgumentParser(description="Audit typed ARC model identity evidence")
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args()
    artifact = run(ROOT, execution_date=args.date, output_path=args.output)
    print(
        f"wrote {args.output} typed_identity_attack_audit_ready_score="
        f"{artifact['typed_identity_attack_audit_ready_score']} "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
