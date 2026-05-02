#!/usr/bin/env python3
"""Run Exp 1131 Lagrangian cascade v2.

Spec: REQ-VERIFY-1131.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_venv_when_run_directly() -> None:
    if os.environ.get("CARNOT_EXP1131_VENV_REEXECED") == "1":
        return
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.environ["CARNOT_EXP1131_VENV_REEXECED"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_ensure_repo_venv_when_run_directly()

for directory in [REPO_ROOT / "python", REPO_ROOT]:
    directory_str = str(directory)
    if directory_str not in sys.path:
        sys.path.insert(0, directory_str)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.eval.lagrangian_cascade_v2 import run_experiment  # noqa: E402


def main() -> None:
    artifact = run_experiment(
        REPO_ROOT / "data" / "fover_corpus_v4.json",
        REPO_ROOT / "results" / "experiment_1131_lagrangian_cascade_v2.json",
    )
    print(json_summary(artifact))


def json_summary(artifact: dict) -> str:
    return (
        "[1131] "
        f"adaptive_tp={artifact['adaptive_tp_rate']:.4f} "
        f"fixed_tp={artifact['fixed_tp_rate']:.4f} "
        f"delta={artifact['accuracy_delta']:.4f} "
        f"savings={artifact['cost_savings_pct']:.2f}% "
        f"verdict={artifact['honest_verdict']}"
    )


if __name__ == "__main__":
    main()
