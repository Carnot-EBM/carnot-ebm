#!/usr/bin/env python3
"""Experiment 782 — EDU-PRM Step Selection for JEPA v20 Training Corpus.

WHY THIS EXPERIMENT (arXiv 2503.22233, REQ-LEARN-050, REQ-LEARN-051):
    JEPA v19 (Exp 770) trained on all 57 FoVer steps uniformly and failed OOD
    because trivially correct/incorrect steps dominate the corpus, adding no
    discriminative signal.

    EDU-PRM (arXiv 2503.22233) showed that entropy-driven uncertainty selection
    achieves full-corpus performance with only 1.5% of training data by focusing
    on steps near the classifier decision boundary (high bootstrap variance).

    This experiment applies EDU-PRM selection to select the top 30% highest-
    variance steps from the pooled FoVer corpus:
      - results/fover_labeled_steps_live.json  (57 steps, Exp 442 baseline)
      - results/fover_labeled_steps_live_v2.json  (Exp 781 steps, if exists)

    The selected corpus is written to results/fover_edu_prm_selected.json for
    use as the JEPA v20 training set.

HONEST VERDICT RULES:
    - "edu_prm_selected_diverse"     if uncertainty_selected_pct >= 0.30
                                     AND diversity_delta=True
    - "edu_prm_selected_uniform"     if uncertainty_selected_pct >= 0.30
                                     AND diversity_delta=False
    - "edu_prm_insufficient_data"    if n_total_before_selection < 30
    - "edu_prm_selection_failed"     if any exception

Spec: REQ-LEARN-050, REQ-LEARN-051, SCENARIO-LEARN-094, SCENARIO-LEARN-095
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

# Allow running from repo root without install.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.edu_prm_selector import EDUPRMConfig, EDUPRMStepSelector  # noqa: E402

DELIVERABLE = "results/experiment_782_edu_prm_step_selection.json"
SELECTED_CORPUS_PATH = "results/fover_edu_prm_selected.json"

tmpl = ExperimentTemplate(
    exp_id=782,
    title="EDU-PRM Step Selection — arXiv 2503.22233 entropy-driven corpus selection",
    deliverable=DELIVERABLE,
)


def _load_labeled_steps(path: Path) -> list[dict]:
    """Load labeled steps JSON; return empty list if file does not exist."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _label_to_int(label: str) -> int:
    """Convert string label to binary integer (1=correct, 0=incorrect)."""
    return 1 if label == "correct" else 0


def run() -> None:
    """Execute EDU-PRM step selection and write deliverable artifact."""
    tmpl.setup()

    repo_root = Path(_REPO)
    v1_path = repo_root / "results" / "fover_labeled_steps_live.json"
    v2_path = repo_root / "results" / "fover_labeled_steps_live_v2.json"
    selected_path = repo_root / SELECTED_CORPUS_PATH

    honest_verdict = "edu_prm_selection_failed"
    artifact_extra: dict = {}

    try:
        # ------------------------------------------------------------------ #
        # Step 4: Pool labeled steps from both sources.                       #
        # ------------------------------------------------------------------ #
        steps_v1 = _load_labeled_steps(v1_path)
        steps_v2 = _load_labeled_steps(v2_path)
        all_steps = steps_v1 + steps_v2

        n_total = len(all_steps)

        if n_total < 30:
            honest_verdict = "edu_prm_insufficient_data"
            artifact_extra = {
                "n_total_before_selection": n_total,
                "n_selected": 0,
                "uncertainty_selected_pct": 0.0,
                "uniform_diversity": 0.0,
                "selected_diversity": 0.0,
                "diversity_delta": False,
                "selected_corpus_path": SELECTED_CORPUS_PATH,
                "honest_verdict": honest_verdict,
            }
            artifact = tmpl.build_result(artifact_extra, status="insufficient_data")
            with open(repo_root / DELIVERABLE, "w") as f:
                json.dump(artifact, f, indent=2)
            tmpl.assert_deliverable_written()
            return

        step_texts = [s["step_text"] for s in all_steps]
        labels = [_label_to_int(s["label"]) for s in all_steps]

        # ------------------------------------------------------------------ #
        # Step 5: Apply EDU-PRM selection.                                    #
        # ------------------------------------------------------------------ #
        config = EDUPRMConfig(n_bootstrap=10, selection_pct=0.30)
        selector = EDUPRMStepSelector(config)
        selected_indices = selector.select(step_texts, labels)

        n_selected = len(selected_indices)
        uncertainty_selected_pct = n_selected / n_total

        # ------------------------------------------------------------------ #
        # Step 6: Write selected corpus.                                      #
        # ------------------------------------------------------------------ #
        selected_steps = [all_steps[i] for i in selected_indices]
        with open(selected_path, "w") as f:
            json.dump(selected_steps, f, indent=2)

        # ------------------------------------------------------------------ #
        # Step 7: Compute diversity improvement.                              #
        # ------------------------------------------------------------------ #
        selected_labels = [labels[i] for i in selected_indices]
        uniform_diversity = sum(labels) / n_total if n_total else 0.0
        selected_diversity = selector.diversity_score(selected_labels)

        # diversity_delta=True means selected set is more balanced than uniform.
        diversity_delta = abs(selected_diversity - 0.5) < abs(uniform_diversity - 0.5)

        # ------------------------------------------------------------------ #
        # Determine honest verdict.                                           #
        # ------------------------------------------------------------------ #
        if uncertainty_selected_pct >= 0.30:
            honest_verdict = (
                "edu_prm_selected_diverse" if diversity_delta else "edu_prm_selected_uniform"
            )
        else:
            honest_verdict = "edu_prm_selection_failed"

        artifact_extra = {
            "n_total_before_selection": n_total,
            "n_selected": n_selected,
            "uncertainty_selected_pct": uncertainty_selected_pct,
            "uniform_diversity": uniform_diversity,
            "selected_diversity": selected_diversity,
            "diversity_delta": diversity_delta,
            "selected_corpus_path": SELECTED_CORPUS_PATH,
            "honest_verdict": honest_verdict,
        }

    except Exception:
        traceback.print_exc()
        honest_verdict = "edu_prm_selection_failed"
        artifact_extra = {
            "n_total_before_selection": 0,
            "n_selected": 0,
            "uncertainty_selected_pct": 0.0,
            "uniform_diversity": 0.0,
            "selected_diversity": 0.0,
            "diversity_delta": False,
            "selected_corpus_path": SELECTED_CORPUS_PATH,
            "honest_verdict": honest_verdict,
        }

    artifact = tmpl.build_result(artifact_extra, status="success")
    with open(repo_root / DELIVERABLE, "w") as f:
        json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    with ExperimentTimeoutWatchdog(
        experiment_id=782,
        timeout_minutes=20,
        result_path=str(Path(_REPO) / DELIVERABLE),
    ):
        run()
