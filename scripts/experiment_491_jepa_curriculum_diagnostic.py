#!/usr/bin/env python3
"""Experiment 491: JEPA Curriculum Diagnostic — RETRO-040 root cause analysis.

**Researcher summary:**
    JEPA AUC regressed across three milestones: 0.667 → 0.400 → 0.281.
    The quality-gated retrain in Exp 477 (min_confidence=0.7) made AUC WORSE,
    reaching 0.281 — below random chance (0.5).  An AUC below 0.5 means the
    model is actively predicting the OPPOSITE of the correct label.

    This experiment diagnoses whether the root cause is:
    (a) filtering_too_aggressive — the confidence gate removed so many pairs
        that the surviving corpus is tiny AND imbalanced toward correct steps
    (b) data_imbalance — the raw corpus itself is imbalanced, filter or not
    (c) data_size — the corpus is simply too small regardless of balance

    Isolation method: simulate four training regimes and compare AUCs.
    If AUC(all_pairs) >> AUC(quality_gated) → filtering is the cause.
    If AUC(curriculum) > AUC(all_pairs) → ordering / curriculum matters.
    If AUC(random_50pct) ≈ AUC(quality_gated) → it's size, not quality.

**CPU-only diagnostic:**
    This experiment deliberately runs CPU-only (JAX_PLATFORMS=cpu).
    GPU is not needed for this diagnostic — EORM models are tiny (16 embed dim)
    and the corpus is small (< 200 pairs).

**Outputs:**
    results/experiment_491_jepa_curriculum_diagnostic.json

Spec: REQ-DIAG-001, REQ-DIAG-002, SCENARIO-DIAG-001, SCENARIO-DIAG-002, SCENARIO-DIAG-003
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() before any CUDA import (RETRO-022)
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------

import json
import logging
import os

from carnot.models.jepa_curriculum_diagnostic import JEPACurriculumDiagnostic
from carnot.models.jepa_retrain_v2 import CoTPairQualityFilter
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 491
EXP_TITLE = "JEPA Curriculum Diagnostic"
DELIVERABLE = "results/experiment_491_jepa_curriculum_diagnostic.json"

FOVER_PAIRS_PATH = _REPO_ROOT / "results" / "fover_labeled_steps_live.json"
EXP488_PAIRS_PATH = _REPO_ROOT / "results" / "exp488_cot_pairs.json"
EXP489_PAIRS_PATH = _REPO_ROOT / "results" / "exp489_cot_pairs.json"

# Quality filter matching Exp 477 (the failing retrain)
EXP477_MIN_CONFIDENCE = 0.7
EXP477_MIN_COVERAGE = 0.3

# Diagnostic regime simulations: fewer epochs for speed on CPU
N_EPOCHS = 100


def _load_json_pairs(path: Path) -> list[dict]:
    """Load a JSON file as a list of dicts; return [] if absent or malformed."""
    if not path.exists():
        log.info("Optional pairs file not found, skipping: %s", path)
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # Some files have a top-level wrapper key
        for key in ("pairs", "steps", "data", "results"):
            if isinstance(data, dict) and key in data and isinstance(data[key], list):
                return data[key]
        log.warning("Unexpected format in %s, skipping", path)
        return []
    except Exception as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return []


def _determine_root_cause(
    auc_all: float,
    auc_gated: float,
    auc_curriculum: float,
    auc_random: float,
    diagnosis: str,
) -> str:
    """Map AUC comparison pattern to a root cause string.

    **For engineers:**
        The four-regime comparison is the key diagnostic:
        - If all_pairs >> quality_gated: the filter was too aggressive
        - If quality_gated ≈ random_50pct: it's data SIZE not quality
        - If curriculum > all_pairs: ordering helps
        - If diagnosis='imbalance': confirmed majority-class collapse

        We use a threshold of 0.05 AUC points as "meaningfully different".
    """
    THRESHOLD = 0.05

    if diagnosis == "insufficient_data":
        return "insufficient_data"

    if auc_all - auc_gated > THRESHOLD:
        # Filtering clearly hurt — the gate was too aggressive
        if diagnosis == "imbalance":
            return "filtering_too_aggressive_with_imbalance"
        return "filtering_too_aggressive"

    if abs(auc_gated - auc_random) <= THRESHOLD:
        # Random 50% gives similar AUC to quality-gated → size is the issue
        return "data_size_insufficient"

    if diagnosis == "imbalance":
        return "data_imbalance"

    return "unknown"


def _determine_curriculum_recommendation(root_cause: str, auc_curriculum: float, auc_all: float) -> str:
    """Recommend the next training strategy based on root cause.

    **For engineers:**
        - If filtering was the cause, lower the threshold or use all pairs
        - If imbalance is confirmed, apply weighted loss or oversample minority class
        - If curriculum helped, use high-to-low ordering
        - If size is the issue, need more data (augmentation or synthetic)
    """
    THRESHOLD = 0.05

    if root_cause in ("filtering_too_aggressive", "filtering_too_aggressive_with_imbalance"):
        if auc_curriculum - auc_all > THRESHOLD:
            return "lower_threshold_with_curriculum"
        return "lower_threshold_or_no_filter"

    if root_cause == "data_imbalance":
        if auc_curriculum - auc_all > THRESHOLD:
            return "weighted_loss_with_curriculum"
        return "weighted_loss"

    if root_cause == "data_size_insufficient":
        return "augment_with_synthetic_pairs"

    if auc_curriculum - auc_all > THRESHOLD:
        return "high_to_low_curriculum"

    return "balanced_corpus_no_filter"


def main() -> None:
    """Run JEPA curriculum diagnostic and write results JSON."""
    tmpl = ExperimentTemplate(EXP_ID, EXP_TITLE, DELIVERABLE)
    guard = DeliverableGuard(str(_REPO_ROOT / DELIVERABLE))

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=40, result_path=str(_REPO_ROOT / DELIVERABLE)):
        tmpl.setup()

        # ------------------------------------------------------------------
        # Load all available labeled pairs
        # ------------------------------------------------------------------
        log.info("Loading labeled pairs...")
        fover_pairs = _load_json_pairs(FOVER_PAIRS_PATH)
        exp488_pairs = _load_json_pairs(EXP488_PAIRS_PATH)
        exp489_pairs = _load_json_pairs(EXP489_PAIRS_PATH)

        all_pairs = fover_pairs + exp488_pairs + exp489_pairs
        n_pairs_raw = len(all_pairs)
        log.info("Loaded %d total pairs (fover=%d, exp488=%d, exp489=%d)",
                 n_pairs_raw, len(fover_pairs), len(exp488_pairs), len(exp489_pairs))

        # ------------------------------------------------------------------
        # Corpus analysis: reproduce Exp 477 quality gate
        # ------------------------------------------------------------------
        log.info("Analyzing corpus with Exp 477 quality filter (min_confidence=%.1f)...",
                 EXP477_MIN_CONFIDENCE)
        quality_filter = CoTPairQualityFilter(
            min_coverage=EXP477_MIN_COVERAGE,
            min_confidence=EXP477_MIN_CONFIDENCE,
        )
        diag = JEPACurriculumDiagnostic(all_pairs)
        corpus_analysis = diag.analyze_corpus(quality_filter)

        log.info(
            "Corpus analysis: n_raw=%d, n_filtered=%d, filter_rate=%.3f, "
            "n_correct=%d, n_incorrect=%d, imbalance_ratio=%.2f, diagnosis=%s",
            corpus_analysis.n_pairs_raw,
            corpus_analysis.n_pairs_filtered,
            corpus_analysis.filter_rate,
            corpus_analysis.n_correct,
            corpus_analysis.n_incorrect,
            corpus_analysis.label_imbalance_ratio,
            corpus_analysis.diagnosis,
        )

        # ------------------------------------------------------------------
        # Simulate all four training regimes
        # ------------------------------------------------------------------
        log.info("Simulating regime: all_pairs (n_epochs=%d)...", N_EPOCHS)
        auc_all_pairs = diag.simulate_regime("all_pairs", n_epochs=N_EPOCHS)
        log.info("AUC (all_pairs) = %.4f", auc_all_pairs)

        log.info("Simulating regime: quality_gated (reproduces Exp 477)...")
        auc_quality_gated = diag.simulate_regime("quality_gated", n_epochs=N_EPOCHS)
        log.info("AUC (quality_gated) = %.4f", auc_quality_gated)

        log.info("Simulating regime: curriculum_high_to_low...")
        auc_curriculum = diag.simulate_regime("curriculum_high_to_low", n_epochs=N_EPOCHS)
        log.info("AUC (curriculum_high_to_low) = %.4f", auc_curriculum)

        log.info("Simulating regime: random_50pct...")
        auc_random_50pct = diag.simulate_regime("random_50pct", n_epochs=N_EPOCHS)
        log.info("AUC (random_50pct) = %.4f", auc_random_50pct)

        # ------------------------------------------------------------------
        # Root cause determination
        # ------------------------------------------------------------------
        root_cause = _determine_root_cause(
            auc_all=auc_all_pairs,
            auc_gated=auc_quality_gated,
            auc_curriculum=auc_curriculum,
            auc_random=auc_random_50pct,
            diagnosis=corpus_analysis.diagnosis,
        )
        curriculum_recommendation = _determine_curriculum_recommendation(
            root_cause=root_cause,
            auc_curriculum=auc_curriculum,
            auc_all=auc_all_pairs,
        )

        log.info("Root cause: %s", root_cause)
        log.info("Curriculum recommendation: %s", curriculum_recommendation)

        # ------------------------------------------------------------------
        # Build and write deliverable
        # ------------------------------------------------------------------
        extra = {
            "schema": "carnot.jepa_diagnostic.v1",
            "n_pairs_raw": n_pairs_raw,
            "n_pairs_filtered": corpus_analysis.n_pairs_filtered,
            "filter_rate": round(corpus_analysis.filter_rate, 4),
            "label_imbalance_ratio": round(corpus_analysis.label_imbalance_ratio, 4),
            "diagnosis": corpus_analysis.diagnosis,
            "auc_all_pairs": round(auc_all_pairs, 4),
            "auc_quality_gated": round(auc_quality_gated, 4),
            "auc_curriculum": round(auc_curriculum, 4),
            "auc_random_50pct": round(auc_random_50pct, 4),
            "root_cause": root_cause,
            "curriculum_recommendation": curriculum_recommendation,
            "honest_verdict": "diagnostic_complete",
        }
        artifact = tmpl.build_result(extra, status="success")

        out_path = _REPO_ROOT / DELIVERABLE
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)
        log.info("Wrote deliverable: %s", out_path)

        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
