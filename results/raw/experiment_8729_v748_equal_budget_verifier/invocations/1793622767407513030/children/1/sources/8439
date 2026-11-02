"""Export the GAP-3 Stage-1 candidate table (votes, q_mean, correct, OOF probe, AND the 512-d z_mean
latents) to .npz + a scalar-only JSON sidecar, so the adversarial-verify reviewers can independently
re-derive every headline number WITHOUT re-running the GPU dump or the torch de-aug join.

WHY: the Stage-0 adversarial round (results/arc3_gap3_stage0_adversarial_verify.json) worked because
reviewers had the raw 8041-candidate table to re-derive from. Stage 1's extra feature is the latent
matrix, which is too big for JSON — hence the .npz. The probe column is the exact out-of-fold
LOTO prediction the committed artifact ranked by (re-computed deterministically here: same seed-free
deterministic SVD + fixed-iteration logistic fit as the committed run).

  ~/trm_venv/bin/python scripts/experiments/arc3_gap3_stage1_export_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
CARNOT = str(repo_root())
sys.path.insert(0, f"{CARNOT}/scripts/experiments")

from arc3_gap3_stage1_latent_energy import (  # noqa: E402
    _build_candidate_table_with_latent,
    _loto_latent_probe,
)


def main():
    tasks, _ev = _build_candidate_table_with_latent()
    assert tasks, "no latent dump found — run trm_arc_eval_harness.py --save_latent first"
    _loto_latent_probe(tasks, pca_k=24)  # attach the committed run's OOF probe column

    task_names = [t["task"] for t in tasks]
    rows = []
    z_rows = []
    for ti, t in enumerate(tasks):
        for c in t["cands"]:
            rows.append(
                {
                    "task_idx": ti,
                    "votes": int(c["votes"]),
                    "q_mean": float(c["q_mean"]),
                    "probe": float(c["probe"]),
                    "correct": bool(c["correct"]),
                }
            )
            z_rows.append(c["z_mean"].astype(np.float32))

    Z = np.stack(z_rows)
    np.savez_compressed(
        f"{CARNOT}/results/arc3_gap3_stage1_candidate_table.npz",
        z_mean=Z,
        task_idx=np.array([r["task_idx"] for r in rows], dtype=np.int32),
        votes=np.array([r["votes"] for r in rows], dtype=np.int32),
        q_mean=np.array([r["q_mean"] for r in rows], dtype=np.float64),
        probe=np.array([r["probe"] for r in rows], dtype=np.float64),
        correct=np.array([r["correct"] for r in rows], dtype=bool),
    )
    sidecar = {
        "experiment": "arc3_gap3_stage1_candidate_table",
        "n_tasks": len(tasks),
        "n_candidates": len(rows),
        "latent_hidden_dim": int(Z.shape[1]),
        "task_names": task_names,
        "candidates": rows,
        "note": (
            "Scalar candidate table for the Stage-1 adversarial-verify round. The 'probe' column is "
            "the committed run's out-of-fold LOTO PCA(24)+balanced-logistic prediction (lower = more "
            "gold-like). Latents are in the sibling .npz (z_mean, row-aligned with 'candidates')."
        ),
    }
    Path(f"{CARNOT}/results/arc3_gap3_stage1_candidate_table.json").write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )
    print(
        f"exported {len(rows)} candidates / {len(tasks)} tasks, z={Z.shape} -> "
        f"results/arc3_gap3_stage1_candidate_table.{{npz,json}}"
    )


if __name__ == "__main__":
    main()
