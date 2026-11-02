#!/usr/bin/env python3
"""Run Exp 1157 SECL cheap-tier calibration.

Spec: REQ-VERIFY-1157.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_venv_when_run_directly() -> None:  # pragma: no cover - CLI bootstrap
    if os.environ.get("CARNOT_EXP1157_VENV_REEXECED") == "1":
        return
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.environ["CARNOT_EXP1157_VENV_REEXECED"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_ensure_repo_venv_when_run_directly()

for directory in [REPO_ROOT / "python", REPO_ROOT]:
    directory_str = str(directory)
    if directory_str not in sys.path:
        sys.path.insert(0, directory_str)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.eval.secl_cheap_tier_calibration import run_experiment  # noqa: E402


def json_summary(artifact: dict) -> str:
    return (
        "[1157] "
        f"n_bad={artifact['n_exemplars']} "
        f"n_ok={artifact['n_correct_examples']} "
        f"exp1145_tp={artifact['thinkprm_tp_exp1145']:.3f} "
        f"exp1145_fpr={artifact['thinkprm_fpr_exp1145']:.3f} "
        f"secl_tp={artifact['secl_tp_rate']:.6f} "
        f"secl_fpr={artifact['secl_fpr']:.6f} "
        f"verdict={artifact['honest_verdict']}"
    )


def main() -> None:
    artifact = run_experiment()
    print(json_summary(artifact))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
