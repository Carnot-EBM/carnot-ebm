#!/usr/bin/env python3
"""Experiment 711: SC-Energy SetConsistencyVerifier — Tier 2.9 candidate.

**Hypothesis:**
    SC-Energy (arXiv 2503.10695) computes a global consistency energy
    E(S_1, ..., S_N) over ALL N chain-of-thought steps simultaneously.
    This catches contradictions invisible to Tier 2.5 (pairwise arithmetic)
    and Tier 2.7 (adjacent carry-forward) — specifically:
      - Step 2 contradicts step 7 (non-adjacent contradiction)
      - Valid intermediate results assembled into wrong conclusion

    We train on the FoVer formal v1 corpus using contrastive pairs and
    evaluate whether AUROC >= 0.75 gates Tier 2.9 cascade integration.

**Data construction:**
    Consistent set   = all steps from one correct GSM8K question response.
    Inconsistent set = same chain but one step swapped from a different question.

    Since fover_labeled_formal_v1 has every step labelled step_correct=True,
    all 200 questions are "correct chains".  We group by question, build
    consistent sets from same-question steps, and build inconsistent sets
    by swapping one step from a random different question.

**Deliverable:** results/experiment_711_sc_energy_set_consistency.json

Spec: REQ-VERIFY-149, REQ-VERIFY-150, REQ-VERIFY-151,
      SCENARIO-VERIFY-149, SCENARIO-VERIFY-150, SCENARIO-VERIFY-151
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from repo root without install
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "python"))

import numpy as np
from sklearn.metrics import roc_auc_score  # type: ignore[import]

from carnot.verify.sc_energy import SetConsistencyVerifier
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate


def _write_artifact(artifact: dict, path: Path) -> None:
    """Atomic JSON write: write to .tmp then rename to avoid partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(artifact, fh, indent=2)
    tmp.rename(path)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 711
TITLE = "SC-Energy SetConsistencyVerifier — Tier 2.9 Candidate"
DELIVERABLE = "results/experiment_711_sc_energy_set_consistency.json"
FOVER_PATH = _REPO / "results" / "fover_labeled_formal_v1.json"
TRAIN_FRAC = 0.8
RANDOM_SEED = 42
N_EPOCHS = 50
TIER_29_AUC_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_fover_chains(path: Path) -> dict[str, list[str]]:
    """Group FoVer v1 step records by question into ordered CoT chains.

    Returns a dict: question_text -> [step_text_0, step_text_1, ...] sorted
    by step_index.  Only keeps questions where ALL steps are marked correct
    (step_correct=True), which is every question in fover_labeled_formal_v1.json.
    """
    data = json.loads(path.read_text())
    by_q: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for pair in data["pairs"]:
        if pair.get("step_correct", True):
            by_q[pair["question"]].append(
                (pair["step_index"], pair["step_text"])
            )
    chains: dict[str, list[str]] = {}
    for q, steps in by_q.items():
        ordered = [text for _, text in sorted(steps, key=lambda x: x[0])]
        if ordered:
            chains[q] = ordered
    return chains


def build_pairs(
    chains: dict[str, list[str]], rng: random.Random
) -> tuple[list[list[str]], list[list[str]]]:
    """Build (consistent_set, inconsistent_set) pairs for training / evaluation.

    For each question Q:
      - consistent_set  = steps from Q  (ground-truth correct chain)
      - inconsistent_set = steps from Q but with one step replaced by a randomly
                           chosen step from a DIFFERENT question Q'.

    If a question has only one step, the replacement still produces a set where
    the sole step is drawn from a different question — globally inconsistent
    because it refers to different numbers/context.

    Returns two parallel lists of equal length.
    """
    questions = list(chains.keys())
    consistent: list[list[str]] = []
    inconsistent: list[list[str]] = []

    for q in questions:
        steps = list(chains[q])  # copy
        # pick an intruder question (different from q)
        candidates = [x for x in questions if x != q]
        if not candidates:
            continue
        intruder_q = rng.choice(candidates)
        intruder_steps = chains[intruder_q]
        intruder_step = rng.choice(intruder_steps)

        # Replace one step in the chain with the intruder step
        replace_idx = rng.randrange(len(steps))
        bad_steps = list(steps)
        bad_steps[replace_idx] = intruder_step

        consistent.append(steps)
        inconsistent.append(bad_steps)

    return consistent, inconsistent


# ---------------------------------------------------------------------------
# AUROC evaluation
# ---------------------------------------------------------------------------


def compute_auroc(
    verifier: SetConsistencyVerifier,
    consistent_sets: list[list[str]],
    inconsistent_sets: list[list[str]],
) -> float:
    """Compute AUROC for consistent (label=0) vs inconsistent (label=1) sets.

    We use energy as the score: higher energy → predict inconsistent.
    Perfect separation gives AUROC=1.0; random gives 0.5.

    Spec: REQ-VERIFY-151, SCENARIO-VERIFY-151
    """
    scores = []
    labels = []
    for steps in consistent_sets:
        scores.append(verifier.energy(steps))
        labels.append(0)
    for steps in inconsistent_sets:
        scores.append(verifier.energy(steps))
        labels.append(1)

    if len(set(labels)) < 2:
        return 0.5  # degenerate case — can't compute AUROC
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=60, result_path=str(_REPO / DELIVERABLE)):
        # ------------------------------------------------------------------
        # 1. Load FoVer data
        # ------------------------------------------------------------------
        if not FOVER_PATH.exists():
            artifact = tmpl.build_result(
                {
                    "sc_energy_auc": 0.0,
                    "symcode_auc_comparison": None,
                    "causal_auc_comparison": None,
                    "n_consistent_sets": 0,
                    "n_inconsistent_sets": 0,
                    "tier_29_cascade_recommended": False,
                    "honest_verdict": "blocked_missing_fover_data",
                },
                status="blocked",
            )
            _write_artifact(artifact, _REPO / DELIVERABLE)
            print(json.dumps(artifact, indent=2))
            return

        chains = load_fover_chains(FOVER_PATH)
        n_questions = len(chains)
        print(f"Loaded {n_questions} question chains from FoVer v1.")

        rng = random.Random(RANDOM_SEED)
        consistent_all, inconsistent_all = build_pairs(chains, rng)

        n_total = len(consistent_all)
        n_train = int(n_total * TRAIN_FRAC)

        # Reproducible shuffle before split
        indices = list(range(n_total))
        rng.shuffle(indices)
        train_idx = indices[:n_train]
        eval_idx = indices[n_train:]

        con_train = [consistent_all[i] for i in train_idx]
        inc_train = [inconsistent_all[i] for i in train_idx]
        con_eval = [consistent_all[i] for i in eval_idx]
        inc_eval = [inconsistent_all[i] for i in eval_idx]

        print(f"Train: {len(con_train)} pairs | Eval: {len(con_eval)} pairs")

        # ------------------------------------------------------------------
        # 2. Train SetConsistencyVerifier
        # ------------------------------------------------------------------
        verifier = SetConsistencyVerifier(seed=RANDOM_SEED)
        verifier.train(con_train, inc_train, n_epochs=N_EPOCHS)

        # ------------------------------------------------------------------
        # 3. Evaluate AUROC on held-out set
        # ------------------------------------------------------------------
        sc_energy_auc = compute_auroc(verifier, con_eval, inc_eval)
        print(f"SC-Energy AUROC (eval): {sc_energy_auc:.4f}")

        # ------------------------------------------------------------------
        # 4. Placeholder comparisons for Tier 2.5 / 2.7
        # (Those verifiers require a running model and are not re-evaluated here;
        #  we record None to indicate not-measured-in-this-run.)
        # ------------------------------------------------------------------
        symcode_auc_comparison: float | None = None
        causal_auc_comparison: float | None = None

        # ------------------------------------------------------------------
        # 5. Verdict
        # ------------------------------------------------------------------
        honest_verdict = (
            "tier_29_viable" if sc_energy_auc >= TIER_29_AUC_THRESHOLD
            else "tier_29_below_threshold"
        )

        # Cascade recommendation: viable AND we have a concrete AUC to compare;
        # without symcode/causal comparison we recommend iff threshold is met.
        tier_29_cascade_recommended = sc_energy_auc >= TIER_29_AUC_THRESHOLD

        # ------------------------------------------------------------------
        # 6. Write artifact
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "sc_energy_auc": sc_energy_auc,
                "symcode_auc_comparison": symcode_auc_comparison,
                "causal_auc_comparison": causal_auc_comparison,
                "n_consistent_sets": len(consistent_all),
                "n_inconsistent_sets": len(inconsistent_all),
                "n_train_pairs": len(con_train),
                "n_eval_pairs": len(con_eval),
                "tier_29_cascade_recommended": tier_29_cascade_recommended,
                "honest_verdict": honest_verdict,
                "tier_29_auc_threshold": TIER_29_AUC_THRESHOLD,
                "n_epochs": N_EPOCHS,
            },
            status="success",
        )
        _write_artifact(artifact, _REPO / DELIVERABLE)
        print(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
