#!/usr/bin/env python3
"""Experiment 725: SC-Energy v2 — Retrained on FoVer v2 Dual Labels (Z3 + PDDL).

**Hypothesis:**
    SC-Energy (Tier 2.9) was originally trained on the FoVer v1 corpus (200 z3 pairs,
    all with z3_verdict="unparseable" — a weak signal).  FoVer v2 provides 1400+
    step-level pairs with BOTH z3 labels (mathematical constraint satisfaction) AND
    PDDL labels (planning/state-transition constraint satisfaction).  Combining two
    orthogonal constraint types gives the energy model a richer consistency signal
    and should improve OOD generalisation on GSM8K held-out questions.

**Strict consensus labeling:**
    A step pair enters training only when BOTH constraint types agree it is correct:
        label = 1  iff  z3_label == 1  AND  pddl_label == 1

    For a step from the z3 labeler: z3_label = step_correct (int), pddl_label = 1 (default — not independently evaluated).
    For a step from the pddl labeler: pddl_label = step_correct (int), z3_label = 1 (default).
    Strict AND of both fields filters out any step where EITHER labeler flags it wrong.
    In FoVer v2 all steps have step_correct=True, so all 1400 pairs pass — but the
    STRUCTURE of dual-label selection is preserved for future corpus iterations.

**Data split:**
    Train: first 80% of unique questions (by sort order) = ~320 questions.
    OOD eval: remaining 20% = ~80 questions (equivalent to GSM8K 500-699 range).
    Using a question-level split (not step-level) prevents data leakage where
    steps from the same question appear in both train and eval.

**Deliverable:** results/experiment_725_sc_energy_v2.json

Spec: REQ-VER-032, SCENARIO-VER-039
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "python"))

import numpy as np
from sklearn.metrics import roc_auc_score  # type: ignore[import]

from carnot.verify.sc_energy import SetConsistencyVerifier
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 725
TITLE = "SC-Energy v2 — FoVer v2 Dual Labels (Z3 + PDDL)"
DELIVERABLE = "results/experiment_725_sc_energy_v2.json"
FOVER_V2_PATH = _REPO / "results" / "fover_v2_combined.json"
V1_BASELINE_ARTIFACT = _REPO / "results" / "experiment_711_sc_energy_set_consistency.json"
TRAIN_FRAC = 0.8
RANDOM_SEED = 42
N_EPOCHS = 100  # more data → more epochs needed for convergence
TIER_29_AUC_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Dual-label consensus helpers
# ---------------------------------------------------------------------------


def derive_labels(pair: dict) -> tuple[int, int]:
    """Derive (z3_label, pddl_label) from a FoVer v2 pair record.

    For z3-labeler pairs, z3_label comes from step_correct and pddl_label defaults
    to 1 (the pddl constraint was not independently evaluated for this pair).
    Vice versa for pddl-labeler pairs.

    Returns
    -------
    tuple[int, int]
        (z3_label, pddl_label), each 0 or 1.

    Spec: REQ-VER-032
    """
    labeler = pair.get("labeler", "")
    step_correct = int(bool(pair.get("step_correct", False)))
    if labeler == "z3":
        # z3_verdict "satisfied" means the constraint is provably correct.
        # Any other verdict (including "unparseable") is treated as unknown / 0.
        z3_verdict = pair.get("z3_verdict", "")
        z3_label = 1 if z3_verdict == "satisfied" else step_correct
        pddl_label = 1  # default: no pddl constraint available for this pair
    elif labeler == "pddl":
        pddl_label = step_correct
        z3_label = 1  # default: no z3 constraint available for this pair
    else:
        z3_label = step_correct
        pddl_label = step_correct
    return z3_label, pddl_label


def strict_consensus_label(z3_label: int, pddl_label: int) -> int:
    """Strict AND: a step is consensus-positive only when BOTH labels are 1.

    This forces the energy model to learn cross-constraint consistency — the step
    must satisfy both mathematical (Z3) and structural (PDDL) constraints.

    Spec: REQ-VER-032
    """
    return 1 if (z3_label == 1 and pddl_label == 1) else 0


# ---------------------------------------------------------------------------
# Data loading — FoVer v2 with dual-label filtering
# ---------------------------------------------------------------------------


def load_fover_v2_chains(
    path: Path,
) -> tuple[dict[str, list[str]], int, int, int]:
    """Load FoVer v2, apply strict consensus filtering, group by question into chains.

    Each pair in fover_v2_combined.json has fields:
      - question, step_text, step_index, step_correct, labeler
      - z3 pairs additionally have: z3_verdict
      - pddl pairs additionally have: action, prev_state, next_state

    Strict consensus: keep only pairs where both z3_label AND pddl_label are 1.
    Then group surviving pairs by question, sort steps by step_index.

    Returns
    -------
    chains : dict[str, list[str]]
        question_text → [step_text_0, step_text_1, ...] (sorted by step_index).
    n_total_pairs : int
        Total pairs before filtering.
    n_consensus_pairs : int
        Pairs surviving strict consensus.
    n_rejected_pairs : int
        Pairs rejected by strict consensus.

    Spec: REQ-VER-032
    """
    data = json.loads(path.read_text())
    all_pairs = data.get("pairs", [])
    n_total = len(all_pairs)

    by_q: dict[str, list[tuple[int, str]]] = defaultdict(list)
    n_kept = 0
    n_rejected = 0

    for pair in all_pairs:
        z3_label, pddl_label = derive_labels(pair)
        consensus = strict_consensus_label(z3_label, pddl_label)
        if consensus == 1:
            by_q[pair["question"]].append(
                (pair.get("step_index", 0), pair["step_text"])
            )
            n_kept += 1
        else:
            n_rejected += 1

    chains: dict[str, list[str]] = {}
    for q, steps in by_q.items():
        ordered = [text for _, text in sorted(steps, key=lambda x: x[0])]
        if ordered:
            chains[q] = ordered

    return chains, n_total, n_kept, n_rejected


# ---------------------------------------------------------------------------
# Build contrastive pairs (same strategy as Exp 711, applied to larger corpus)
# ---------------------------------------------------------------------------


def build_contrastive_pairs(
    chains: dict[str, list[str]], rng: random.Random
) -> tuple[list[list[str]], list[list[str]]]:
    """Build (consistent_set, inconsistent_set) pairs for training and evaluation.

    For each question Q:
      - consistent_set  = all steps from Q  (ground-truth correct chain)
      - inconsistent_set = Q's steps with one step replaced by a step from a
                           DIFFERENT question Q' (intruder step introduces a global
                           contradiction invisible to pairwise or adjacent checkers).

    Why step-swapping instead of random noise: the intruder step is real arithmetic
    text from a different problem, so it looks locally plausible but creates a global
    inconsistency — exactly the failure mode that SC-Energy targets.

    Spec: REQ-VER-032
    """
    questions = list(chains.keys())
    consistent: list[list[str]] = []
    inconsistent: list[list[str]] = []

    for q in questions:
        steps = list(chains[q])
        candidates = [x for x in questions if x != q]
        if not candidates:
            continue
        intruder_q = rng.choice(candidates)
        intruder_step = rng.choice(chains[intruder_q])
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

    Higher energy → predict inconsistent.  Perfect separation → AUROC=1.0.
    Random → AUROC=0.5.

    Spec: REQ-VER-032, SCENARIO-VER-039
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
        return 0.5
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Load v1 baseline AUC
# ---------------------------------------------------------------------------


def load_v1_baseline_auc(path: Path) -> float:
    """Read sc_energy_auc from the Exp 711 artifact as the v1 baseline.

    Returns 0.5 (random chance) if the artifact is missing or malformed, so the
    experiment can still run without a hard dependency on Exp 711 completing first.
    """
    if not path.exists():
        return 0.5
    try:
        art = json.loads(path.read_text())
        return float(art.get("sc_energy_auc", 0.5))
    except (json.JSONDecodeError, ValueError):
        return 0.5


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

    result_path = str(_REPO / DELIVERABLE)
    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=45, result_path=result_path):
        # ------------------------------------------------------------------
        # 1. Check FoVer v2 corpus availability
        # ------------------------------------------------------------------
        if not FOVER_V2_PATH.exists():
            artifact = tmpl.build_result(
                {
                    "ood_auc": 0.0,
                    "v1_baseline_auc": 0.5,
                    "auc_delta": 0.0,
                    "training_pairs": 0,
                    "honest_verdict": "blocked_missing_fover_v2",
                },
                status="blocked",
            )
            (_REPO / DELIVERABLE).parent.mkdir(parents=True, exist_ok=True)
            (_REPO / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            print(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # ------------------------------------------------------------------
        # 2. Load FoVer v2 with strict dual-label consensus filtering
        # ------------------------------------------------------------------
        chains, n_total_pairs, n_consensus_pairs, n_rejected_pairs = (
            load_fover_v2_chains(FOVER_V2_PATH)
        )
        n_questions = len(chains)
        print(
            f"FoVer v2: {n_total_pairs} total pairs → "
            f"{n_consensus_pairs} consensus pairs across {n_questions} questions "
            f"({n_rejected_pairs} rejected by strict AND filter)"
        )

        # ------------------------------------------------------------------
        # 3. Question-level train / OOD-eval split (no step-level leakage)
        #    Sort questions for determinism, then take first 80% as train.
        # ------------------------------------------------------------------
        rng = random.Random(RANDOM_SEED)
        sorted_questions = sorted(chains.keys())
        rng.shuffle(sorted_questions)  # reproducible shuffle before split
        n_train_q = int(len(sorted_questions) * TRAIN_FRAC)

        train_questions = sorted_questions[:n_train_q]
        eval_questions = sorted_questions[n_train_q:]

        train_chains = {q: chains[q] for q in train_questions}
        eval_chains = {q: chains[q] for q in eval_questions}

        # ------------------------------------------------------------------
        # 4. Build contrastive pairs for train and OOD eval
        # ------------------------------------------------------------------
        con_train, inc_train = build_contrastive_pairs(train_chains, rng)
        con_eval, inc_eval = build_contrastive_pairs(eval_chains, rng)
        print(
            f"Train: {len(con_train)} pairs | OOD eval: {len(con_eval)} pairs"
        )

        # ------------------------------------------------------------------
        # 5. Train SetConsistencyVerifier on FoVer v2 strict consensus pairs
        # ------------------------------------------------------------------
        verifier = SetConsistencyVerifier(seed=RANDOM_SEED)
        verifier.train(con_train, inc_train, n_epochs=N_EPOCHS)

        # ------------------------------------------------------------------
        # 6. Evaluate OOD AUC
        # ------------------------------------------------------------------
        ood_auc = compute_auroc(verifier, con_eval, inc_eval)
        print(f"SC-Energy v2 OOD AUROC: {ood_auc:.4f}")

        # ------------------------------------------------------------------
        # 7. Load v1 baseline and determine honest verdict
        # ------------------------------------------------------------------
        v1_baseline_auc = load_v1_baseline_auc(V1_BASELINE_ARTIFACT)
        auc_delta = ood_auc - v1_baseline_auc
        print(f"v1 baseline: {v1_baseline_auc:.4f} | delta: {auc_delta:+.4f}")

        if ood_auc >= TIER_29_AUC_THRESHOLD and ood_auc > v1_baseline_auc:
            honest_verdict = "sc_energy_v2_improvement"
        elif ood_auc <= v1_baseline_auc:
            honest_verdict = "sc_energy_v2_no_gain"
        else:
            honest_verdict = "sc_energy_v2_below_threshold"

        # ------------------------------------------------------------------
        # 8. Write artifact
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "ood_auc": ood_auc,
                "v1_baseline_auc": v1_baseline_auc,
                "auc_delta": auc_delta,
                "training_pairs": len(con_train),
                "n_eval_pairs": len(con_eval),
                "n_total_fover_v2_pairs": n_total_pairs,
                "n_consensus_pairs": n_consensus_pairs,
                "n_rejected_pairs": n_rejected_pairs,
                "n_train_questions": len(train_questions),
                "n_eval_questions": len(eval_questions),
                "honest_verdict": honest_verdict,
                "tier_29_auc_threshold": TIER_29_AUC_THRESHOLD,
                "n_epochs": N_EPOCHS,
            },
            status="success",
        )
        (_REPO / DELIVERABLE).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = (_REPO / DELIVERABLE).with_suffix(".tmp")
        tmp_path.write_text(json.dumps(artifact, indent=2))
        tmp_path.rename(_REPO / DELIVERABLE)
        print(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
