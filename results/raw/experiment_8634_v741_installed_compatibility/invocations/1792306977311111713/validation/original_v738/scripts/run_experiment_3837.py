#!/usr/bin/env python3
"""Run Exp 3837 FoVer learned-contribution category characterization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    from carnot.verify import experiment_3837_fover_error_category_learned_contribution as exp

    output = exp.write_artifact(
        ROOT,
        tests_run=[
            (
                ".venv/bin/pytest -o addopts='' "
                "tests/python/test_experiment_3837_fover_error_category_learned_contribution.py -q"
            ),
            (
                ".venv/bin/coverage run --include='*/experiment_3837_fover_error_category_learned_contribution.py' "
                "-m pytest -o addopts='' -q "
                "tests/python/test_experiment_3837_fover_error_category_learned_contribution.py && "
                ".venv/bin/coverage report "
                "--include='*/experiment_3837_fover_error_category_learned_contribution.py' "
                "--fail-under=100 --show-missing"
            ),
        ],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(f"Generated artifact at: {output.relative_to(ROOT)}")
    print(artifact["honest_verdict"])
    print()
    print(exp.format_breakdown_table(artifact))
    return 1 if str(artifact["honest_verdict"]).startswith("blocked_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
