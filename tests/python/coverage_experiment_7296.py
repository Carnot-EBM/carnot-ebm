"""Run the focused Exp7296 scenarios for scoped module coverage."""

from pathlib import Path

import pytest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    raise SystemExit(
        pytest.main(
            [
                str(root / "tests/python/test_experiment_7296_v641_mixture_learning.py"),
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7296-coverage",
            ]
        )
    )
