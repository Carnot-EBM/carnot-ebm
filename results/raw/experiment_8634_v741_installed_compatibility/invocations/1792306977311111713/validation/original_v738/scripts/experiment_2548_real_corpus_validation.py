from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.real_corpus_validation import run_real_corpus_validation


def run_experiment(
    repo_root: Path = REPO_ROOT,
    results_dir: Path = Path("results"),
    write: bool = True,
) -> dict[str, Any]:
    return run_real_corpus_validation(repo_root=repo_root, results_dir=results_dir, write=write)


if __name__ == "__main__":
    run_experiment()
