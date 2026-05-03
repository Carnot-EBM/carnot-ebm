"""Run Exp 1160 MARCH blinded multi-agent claim checking.

Spec: REQ-VERIFY-1160, SCENARIO-VERIFY-1160.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
for path in (str(PYTHON_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from carnot.eval.march_multiagent_claim_check import run_experiment  # noqa: E402


def main() -> None:
    artifact = run_experiment()
    print(
        json.dumps(
            {key: artifact[key] for key in sorted(artifact) if key != "per_exemplar_results"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
