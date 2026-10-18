"""Exp 4010: GAP-5 cross-example consistency selector over saved GAP-4 programs.

Spec refs: REQ-VERIFY-4010, SCENARIO-VERIFY-4010.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from carnot.paths import repo_root


# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
REPO_ROOT = repo_root()
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.agentic.gap5_cross_example_selector import (  # noqa: E402
    ARC1_PROGRAMS,
    ARC2_POOL,
    ARC2_PROGRAMS,
    CHAIN_ARTIFACT,
    OUTPUT,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=ARC2_POOL)
    parser.add_argument("--arc2-programs", type=Path, default=ARC2_PROGRAMS)
    parser.add_argument("--arc1-programs", type=Path, default=ARC1_PROGRAMS)
    parser.add_argument("--chain-artifact", type=Path, default=CHAIN_ARTIFACT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    args = parser.parse_args()
    artifact = run(
        pool_path=args.pool,
        arc2_programs_path=args.arc2_programs,
        arc1_programs_path=args.arc1_programs,
        chain_artifact_path=args.chain_artifact,
        output_path=args.output,
        bootstrap_iters=args.bootstrap_iters,
    )
    print(json.dumps({field: artifact[field] for field in ("honest_verdict", "n_tasks_scored")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
