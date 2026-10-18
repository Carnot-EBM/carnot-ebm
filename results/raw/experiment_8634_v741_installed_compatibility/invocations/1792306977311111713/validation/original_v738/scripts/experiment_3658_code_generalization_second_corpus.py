#!/usr/bin/env python3
"""Run Exp 3658 balanced second-corpus code generalization replication."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.code_generalization_second_corpus import (  # noqa: E402
    OUTPUT_REL_PATH,
    build_artifact,
)


def main() -> int:
    artifact = build_artifact(
        REPO_ROOT,
        tests_run=[
            ".venv/bin/pytest tests/python/test_experiment_3658_code_generalization_second_corpus.py -q --no-cov",
            ".venv/bin/coverage run --source=python/carnot/verify -m pytest -o addopts='' tests/python/test_experiment_3658_code_generalization_second_corpus.py -q",
            ".venv/bin/coverage report --include='python/carnot/verify/code_generalization_second_corpus.py' --fail-under=100 --show-missing",
            ".venv/bin/python scripts/check_spec_coverage.py",
            ".venv/bin/pytest tests/python -q",
        ],
    )
    output_path = REPO_ROOT / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
