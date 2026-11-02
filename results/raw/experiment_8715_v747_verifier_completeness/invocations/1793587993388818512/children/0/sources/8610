#!/usr/bin/env python3
"""Run Exp 3886 defabricated graph-grounding fact verifier.

Spec refs: REQ-VERIFY-3886, SCENARIO-VERIFY-3886.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root / "python"))
    from carnot.verify.graph_grounding_fact_verifier_defabricated import (  # noqa: PLC0415
        ExperimentConfig,
        OUTPUT_REL_PATH,
        run_experiment,
    )

    artifact = run_experiment(ExperimentConfig(repo_root=root), write=True)
    print(root / OUTPUT_REL_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
