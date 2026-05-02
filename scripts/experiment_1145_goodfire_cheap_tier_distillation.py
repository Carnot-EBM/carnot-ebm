#!/usr/bin/env python3
"""Run Exp 1145 Goodfire cheap-tier distillation.

Spec: REQ-VERIFY-1145.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_venv_when_run_directly() -> None:  # pragma: no cover - CLI bootstrap
    if os.environ.get("CARNOT_EXP1145_VENV_REEXECED") == "1":
        return
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.environ["CARNOT_EXP1145_VENV_REEXECED"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_ensure_repo_venv_when_run_directly()

for directory in [REPO_ROOT / "python", REPO_ROOT]:
    directory_str = str(directory)
    if directory_str not in sys.path:
        sys.path.insert(0, directory_str)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.eval.goodfire_cheap_tier_distillation import run_experiment  # noqa: E402


def json_summary(artifact: dict) -> str:
    return (
        "[1145] "
        f"n={artifact['n_exemplars']} "
        f"cheap_before={artifact['combined_cheap_tp_before']:.6f} "
        f"cheap_after={artifact['combined_cheap_tp_after']:.6f} "
        f"fp_after={artifact['false_positive_rate_after']:.6f} "
        f"dominant={artifact['dominant_halluguard_feature']} "
        f"verdict={artifact['honest_verdict']}"
    )


def main() -> None:
    artifact = run_experiment()
    print(json_summary(artifact))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
