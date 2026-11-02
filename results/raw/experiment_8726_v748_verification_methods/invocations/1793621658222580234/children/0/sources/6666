#!/usr/bin/env python3
"""Experiment 1163: NRGPT energy recurrence native Phase 3 prototype.

Spec refs: REQ-KONA-011, SCENARIO-KONA-010, SCENARIO-KONA-011.
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
DELIVERABLE = nrgpt.DEFAULT_DELIVERABLE_PATH
N_TRAIN = nrgpt.DEFAULT_N_TRAIN
N_EVAL = nrgpt.DEFAULT_N_EVAL
D_EMB = nrgpt.DEFAULT_EMBEDDING_DIM
D_ENERGY = nrgpt.DEFAULT_ENERGY_DIM
ENERGY_EPOCHS = 20
HEAD_EPOCHS = 50
SEED = nrgpt.DEFAULT_SEED


def main() -> int:
    """Run Exp 1163 and write the JSON deliverable."""

    nrgpt.run_experiment(
        dataset_path=DATASET_PATH,
        corpus_path=CORPUS_PATH,
        deliverable_path=DELIVERABLE,
        n_train=N_TRAIN,
        n_eval=N_EVAL,
        d_emb=D_EMB,
        d_energy=D_ENERGY,
        energy_epochs=ENERGY_EPOCHS,
        head_epochs=HEAD_EPOCHS,
        seed=SEED,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
