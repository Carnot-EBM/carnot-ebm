#!/usr/bin/env python3
"""Experiment 763: Dual-Pathway MoP Probe vs Single-Pathway JEPAReasonerProbe.

RESEARCH QUESTION (arXiv 2601.07422):
    Does fusing question-anchored and answer-anchored probe pathways via a
    Mixture-of-Probes (MoP) gate improve AUROC over the single-pathway
    JEPAReasonerProbe (Exp 732 baseline: AUC=0.993)?

METHOD:
    - Load FoVer v2 labeled steps (57 steps: 30 correct, 27 incorrect).
    - Split 80/20 train/test (45 train, 12 test).
    - Train MixtureOfProbes (QuestionAnchoredProbe + AnswerAnchoredProbe + GateNetwork)
      jointly via BCELoss + Adam for 100 epochs.
    - Evaluate AUROC, precision, recall on test split.
    - Compare to single-pathway JEPAReasonerProbe baseline AUC=0.993.

NOTE ON CPU PROXY:
    Full implementation uses real LLM hidden states extracted at question-end
    and answer-end token positions.  This experiment uses TF-IDF mean embeddings
    as a CPU-only proxy (no GPU required).  The TF-IDF proxy is a faithful
    structural test — plug in JEPAReasonerProbe.extract_hidden_state() to upgrade
    to real hidden states without changing the probe architecture.

Spec: REQ-PROBE-010, REQ-PROBE-011, SCENARIO-PROBE-020, SCENARIO-PROBE-021
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so local imports work when run directly.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.dual_pathway_probe import (
    AnswerAnchoredProbe,
    GateNetwork,
    MixtureOfProbes,
    QuestionAnchoredProbe,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 763
TITLE = "Dual-Pathway MoP Probe (arXiv 2601.07422) vs JEPAReasonerProbe Baseline"
DELIVERABLE = "results/experiment_763_dual_pathway_probe.json"
BASELINE_AUROC = 0.993  # Exp 732 5-fold CV result

FOVER_PATH = Path("results/fover_labeled_steps_live.json")

TRAIN_RATIO = 0.80   # 80% train, 20% test
N_EPOCHS = 100
LR = 1e-3
RANDOM_SEED = 42


def _honest_verdict(auroc: float, test_set_size: int) -> str:
    """Return a verdict string that honestly describes MoP performance.

    WHY explicit thresholds (not just 'good/bad'):
        The conductor and downstream analysis need machine-readable verdicts.
        Thresholds match the task spec: 0.993 = baseline, 0.90 = competitive,
        0.75 = marginal.  A test set of <10 is too small for reliable AUROC
        (confidence interval width > 0.15 for N=9), so we flag it explicitly
        rather than reporting a misleading number.
    """
    if test_set_size < 10:
        return "dual_pathway_insufficient_data"
    if auroc >= BASELINE_AUROC:
        return "dual_pathway_superior"
    if auroc >= 0.90:
        return "dual_pathway_competitive"
    if auroc >= 0.75:
        return "dual_pathway_marginal"
    return "dual_pathway_below_baseline"


def _load_fover_steps() -> list[dict]:
    """Load and return the 57 FoVer v2 labeled steps."""
    with FOVER_PATH.open() as f:
        steps = json.load(f)

    # FoVer v2 records have question_id, step_text, label, confidence.
    # The question_context (CoT text before this step) is not stored separately
    # in this file — we use an empty string as the question side here so the
    # TF-IDF embedder focuses entirely on the step_text for the question pathway.
    # When real LLM hidden-state extraction is wired in, replace with the
    # full question prompt text for each question_id.
    for s in steps:
        if "question_context" not in s:
            s["question_context"] = ""
    return steps


def _train_test_split(
    steps: list[dict], train_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Deterministic 80/20 stratified split preserving label balance.

    WHY stratified: with only 57 samples, a random split could accidentally
    put all 'incorrect' labels in one partition.  Stratification ensures both
    splits have approximately 53% correct and 47% incorrect samples.
    """
    import random  # noqa: PLC0415

    rng = random.Random(seed)

    correct = [s for s in steps if s["label"] == "correct"]
    incorrect = [s for s in steps if s["label"] == "incorrect"]

    rng.shuffle(correct)
    rng.shuffle(incorrect)

    n_correct_train = int(len(correct) * train_ratio)
    n_incorrect_train = int(len(incorrect) * train_ratio)

    train = correct[:n_correct_train] + incorrect[:n_incorrect_train]
    test = correct[n_correct_train:] + incorrect[n_incorrect_train:]

    # Shuffle so train isn't all-correct then all-incorrect.
    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


def run_experiment() -> None:
    """Main experiment entry point."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=45,
        result_path=DELIVERABLE,
    ):
        # ------------------------------------------------------------------ #
        # 1. Load FoVer v2 labeled steps
        # ------------------------------------------------------------------ #
        steps = _load_fover_steps()
        train_steps, test_steps = _train_test_split(steps, TRAIN_RATIO, RANDOM_SEED)

        # ------------------------------------------------------------------ #
        # 2. Build and train MixtureOfProbes
        # ------------------------------------------------------------------ #
        q_probe = QuestionAnchoredProbe(hidden_dim=128, output_dim=1)
        a_probe = AnswerAnchoredProbe(hidden_dim=128, output_dim=1)
        gate = GateNetwork(input_dim=2, output_dim=1)
        mop = MixtureOfProbes(q_probe, a_probe, gate)

        train_result = mop.train(train_steps, n_epochs=N_EPOCHS, lr=LR)

        # ------------------------------------------------------------------ #
        # 3. Evaluate on test split
        # ------------------------------------------------------------------ #
        scores = [
            mop.predict(s.get("question_context", ""), s["step_text"])
            for s in test_steps
        ]
        test_labels = [
            1.0 if s["label"] == "incorrect" else 0.0
            for s in test_steps
        ]

        auroc = MixtureOfProbes.evaluate_auroc(scores, test_labels)
        precision, recall = MixtureOfProbes.compute_precision_recall(
            scores, test_labels, threshold=0.5
        )

        # ------------------------------------------------------------------ #
        # 4. Build artifact
        # ------------------------------------------------------------------ #
        verdict = _honest_verdict(auroc, len(test_steps))
        improvement = auroc - BASELINE_AUROC

        artifact = tmpl.build_result(
            {
                "train_size": len(train_steps),
                "test_size": len(test_steps),
                "auroc": round(auroc, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "baseline_auroc": BASELINE_AUROC,
                "improvement_vs_baseline": round(improvement, 4),
                "honest_verdict": verdict,
                "final_train_loss": round(train_result["final_loss"], 6),
                "n_epochs": N_EPOCHS,
                "probe_architecture": {
                    "question_probe": "Linear(128,64)->ReLU->Linear(64,1)->sigmoid",
                    "answer_probe": "Linear(128,64)->ReLU->Linear(64,1)->sigmoid",
                    "gate": "Linear(2,8)->ReLU->Linear(8,1)->sigmoid",
                    "embedding": "tfidf_proxy_128dim",
                },
                "confidence_caveat": (
                    "Test set size={}. AUROC confidence interval is wide "
                    "(+/-{:.2f} at 95% for N={}) — treat as directional, not definitive.".format(
                        len(test_steps),
                        1.0 / (2.0 * len(test_steps) ** 0.5),
                        len(test_steps),
                    )
                ),
                "reference": "arXiv 2601.07422",
                "baseline_reference": "Exp 732 JEPAReasonerProbe 5-fold CV",
            },
            status="success",
        )

        out_path = Path(DELIVERABLE)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    run_experiment()
