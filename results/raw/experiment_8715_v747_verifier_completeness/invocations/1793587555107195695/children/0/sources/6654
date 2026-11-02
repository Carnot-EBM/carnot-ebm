#!/usr/bin/env python3
"""Experiment 1172: NRGPT per-token energy inference on FoVer.

Spec refs: REQ-KONA-014, SCENARIO-KONA-014.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carnot.phase3 import nrgpt_energy as nrgpt  # noqa: E402

DATASET_PATH = nrgpt.DEFAULT_DATASET_PATH
CORPUS_PATH = nrgpt.DEFAULT_CORPUS_PATH
BASELINE_ARTIFACT = nrgpt.DEFAULT_BATCH_BASELINE_ARTIFACT_PATH
DELIVERABLE = nrgpt.DEFAULT_PER_TOKEN_DELIVERABLE_PATH
N_TRAIN = nrgpt.DEFAULT_N_TRAIN
N_EVAL = nrgpt.DEFAULT_N_EVAL
SEED = nrgpt.DEFAULT_SEED


def main() -> int:
    """Run Exp 1172 and write the JSON deliverable."""

    nrgpt.run_per_token_experiment(
        dataset_path=DATASET_PATH,
        corpus_path=CORPUS_PATH,
        baseline_artifact_path=BASELINE_ARTIFACT,
        deliverable_path=DELIVERABLE,
        n_train=N_TRAIN,
        n_eval=N_EVAL,
        seed=SEED,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
