"""Run Exp 4203 SOTA ingestion from the required results entry point."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4203_sota_ingestion_verifier_as_reward as exp4203


if __name__ == "__main__":
    raise SystemExit(exp4203.main())
