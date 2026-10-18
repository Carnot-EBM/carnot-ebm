#!/usr/bin/env python3
"""Run Exp 3862 graph-grounding fact-verifier prototype.

Spec: REQ-VERIFY-3862, SCENARIO-VERIFY-3862.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root / "python"))
    from carnot.verify.graph_grounding_probe import write_artifact

    output = write_artifact(root)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
