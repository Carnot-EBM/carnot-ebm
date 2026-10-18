#!/usr/bin/env python3
"""Run Exp 3770 distribution mirror readiness audit.

Spec refs: REQ-PUBLISH-3770, SCENARIO-PUBLISH-3770.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting.distribution_mirror_publish_checklist_3770 import (  # noqa: E402
    write_artifact,
)


def main() -> int:
    output = write_artifact(REPO_ROOT)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
