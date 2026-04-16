#!/usr/bin/env python3
"""Exp 383 — Combined EORM + JEPA retrain on live GPU pairs from Exps 379-382.

**Researcher summary:**
    Exps 371 (EORM real retrain) and 372 (JEPA real retrain) were blocked in
    milestone 2026.05.27 because Exps 368-370 never produced live pairs (RETRO-015).
    Now that Exps 379-382 have run with live GPU, real (question, response, is_correct)
    pairs may be available. This experiment retrains both models in a single combined
    experiment to minimise overhead.

    EORM (EnergyRewardModel): predicts whether a chain-of-thought response is correct.
    Trained on contrastive pairs: (correct_response, incorrect_response, question).
    Target: AUC-ROC improvement from 0.500 (random) to >= 0.65 on held-out live pairs.

    JEPA predictor: predicts constraint violations from partial responses (first 50%).
    Trained on binary (partial_response, has_violation) pairs.
    Target: AUC-ROC improvement that enables the SinkProbe fast-path.

**Honest reporting:**
    - "both_improved": Both EORM and JEPA AUC improved after retraining.
    - "eorm_only": Only EORM AUC improved.
    - "jepa_only": Only JEPA AUC improved.
    - "neither_improved": Both ran but neither improved.
    - "insufficient_pairs": At least one model had too few pairs to retrain.

**Usage:**
    JAX_PLATFORMS=cpu python scripts/experiment_383_models_retrain.py
    CARNOT_FORCE_LIVE=1 python scripts/experiment_383_models_retrain.py

Spec: REQ-LEARN-025, SCENARIO-LEARN-048
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax.random as jrandom

from carnot.embeddings.jepa_energy import ContextPredictionEnergy, JEPAEnergyConfig
from carnot.embeddings.jepa_retrain import (
    JEPARetrainer,
    ViolationPair,
    extract_violation_pairs,
)
from carnot.models.eorm import CoTEnergyInput, EORMModel, EORMTrainer
from carnot.models.eorm_retrain import load_real_cot_pairs, make_synthetic_eorm_pairs
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 383
TITLE = "Combined EORM+JEPA retrain on live GPU pairs from Exps 379-382"
DELIVERABLE = "results/experiment_383_models_retrain.json"

# Minimum real pairs required to declare retrain_mode="real_data"
EORM_MIN_PAIRS = 50
JEPA_MIN_PAIRS = 30

# Training hyperparameters
TRAIN_SPLIT = 0.8
EORM_EPOCHS = 200
JEPA_EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-4
MARGIN = 1.0

# EORM model config (matches Exp 346 / 359 defaults)
EMBED_DIM = 128
N_HEADS = 4
N_LAYERS = 2

# JEPA model config
JEPA_EMBED_DIM = 64

# Live result files produced by Exps 379-382
_LIVE_FILES = [
    "results/experiment_379_precision_execute.json",
    "results/experiment_380_humaneval_execute.json",
    "results/experiment_381_adversarial_execute.json",
]


# ---------------------------------------------------------------------------
# AUC-ROC evaluator for EORM (standalone, no sklearn dependency)
# ---------------------------------------------------------------------------


def _evaluate_eorm_auc(model: EORMModel, pairs: list[ViolationPair]) -> float:
    """Compute AUC-ROC for EORM model on a ViolationPair test set.

    **For engineers:**
        EORM outputs *lower* energy for responses it considers more correct.
        A "violation" = incorrect response (has_violation=True). To match AUC
        convention (high score = positive class = violation) we negate energy:
            score = -energy(question, response)

        The question text proxy is the ViolationPair.question_id because we do
        not store the original question text — the id is sufficient to give each
        (question, response) pair a distinct token context.

    Args:
        model: An EORMModel instance (trained or baseline).
        pairs: ViolationPair list for evaluation.

    Returns:
        AUC-ROC in [0, 1]. 0.5 = random baseline.

    Spec: SCENARIO-LEARN-048
    """
    if not pairs:
        return 0.5

    scores: list[float] = []
    labels: list[int] = []

    for p in pairs:
        cot = CoTEnergyInput(
            question_text=p.question_id,
            response_text=p.full_response,
        )
        energy = model.energy(cot)
        scores.append(-energy)  # high negated energy → predicted violation
        labels.append(1 if p.has_violation else 0)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort descending by score; walk threshold from high to low
    scored = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tpr_pts = [0.0]
    fpr_pts = [0.0]
    tp = 0
    fp = 0

    for _s, lab in scored:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_pts.append(tp / n_pos)
        fpr_pts.append(fp / n_neg)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr_pts)):
        dfpr = fpr_pts[i] - fpr_pts[i - 1]
        auc += dfpr * (tpr_pts[i] + tpr_pts[i - 1]) / 2.0

    return float(auc)


# ---------------------------------------------------------------------------
# Build contrastive triples for EORM from ViolationPairs
# ---------------------------------------------------------------------------


def _pairs_to_contrastive_triples(
    pairs: list[ViolationPair],
) -> list[tuple[str, str, str]]:
    """Convert ViolationPair list into (correct, incorrect, question) triples.

    **For engineers:**
        EORM trains on *contrastive pairs*: for the same question, one response
        is correct and another is wrong. ViolationPairs carry individual binary
        labels, not pre-paired contrasts. This function:

        1. Groups pairs by question_id.
        2. Separates correct (has_violation=False) and incorrect (True) entries.
        3. Round-robin cross-multiplies to avoid O(n^2) explosion on large groups.

        Synthetic pairs (question_id starts with "synthetic_") are pooled into
        a shared bucket so contrasts can be formed across the whole synthetic corpus.

    Args:
        pairs: List of ViolationPair objects.

    Returns:
        List of (correct_response, incorrect_response, question_id) tuples.
    """
    from collections import defaultdict

    correct_by_q: dict[str, list[str]] = defaultdict(list)
    incorrect_by_q: dict[str, list[str]] = defaultdict(list)

    _SYNTHETIC_POOL = "_synthetic_pool"

    for p in pairs:
        q_id = p.question_id
        if q_id == "unknown" or q_id.startswith("synthetic_"):
            q_key = _SYNTHETIC_POOL
        else:
            q_key = q_id
        if p.has_violation:
            incorrect_by_q[q_key].append(p.full_response)
        else:
            correct_by_q[q_key].append(p.full_response)

    all_q_ids = set(correct_by_q.keys()) | set(incorrect_by_q.keys())
    triples: list[tuple[str, str, str]] = []

    for q_id in sorted(all_q_ids):
        corrects = correct_by_q.get(q_id, [])
        incorrects = incorrect_by_q.get(q_id, [])
        if not corrects or not incorrects:
            continue
        n_pairs = max(len(corrects), len(incorrects))
        for i in range(n_pairs):
            c = corrects[i % len(corrects)]
            ic = incorrects[i % len(incorrects)]
            triples.append((c, ic, q_id))

    return triples


# ---------------------------------------------------------------------------
# Load live result files for JEPA (accepts file path list, not dict)
# ---------------------------------------------------------------------------


def _load_jepa_pairs_from_files(
    result_files: list[str],
    prefix_fraction: float = 0.5,
) -> list[ViolationPair]:
    """Load JEPA violation pairs from a list of experiment result file paths.

    **For engineers:**
        ``extract_violation_pairs`` from jepa_retrain.py expects a single dict
        with a ``"responses"`` key (the Exp 340 format). Here we have a list of
        file paths potentially covering multiple schemas. This wrapper iterates
        the files, loads any with a ``"responses"`` list, and concatenates
        the resulting ViolationPairs.

        Files that are missing, empty, or use an unrecognised schema are skipped.
        If no valid pairs are found across all files, the synthetic fallback is
        NOT triggered here — the caller handles that via the JEPA_MIN_PAIRS check.

    Args:
        result_files: List of file paths to experiment JSON result files.
        prefix_fraction: Fraction of words to keep for partial prefix (default 0.5).

    Returns:
        Flat list of ViolationPair objects. May be empty.
    """
    all_pairs: list[ViolationPair] = []

    for fpath in result_files:
        path = Path(fpath)
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        responses = data.get("responses")
        if isinstance(responses, list) and responses:
            # extract_violation_pairs expects the dict with "responses" key
            pairs = extract_violation_pairs(data, prefix_fraction=prefix_fraction)
            # Exclude synthetic fallback: extract returns synthetic when responses is empty
            # Only accept if the result came from real data
            all_pairs.extend(pairs)

    return all_pairs


# ---------------------------------------------------------------------------
# Build honest_verdict for the combined retrain
# ---------------------------------------------------------------------------


def _combined_honest_verdict(
    eorm_verdict: str,
    jepa_verdict: str,
) -> str:
    """Determine the combined honest_verdict from individual EORM and JEPA verdicts.

    **For engineers:**
        The combined verdict summarises both models in a single string for the
        headline artifact field. The mapping:

        - "both_improved": both EORM and JEPA AUC improved after retraining.
        - "eorm_only": only EORM improved.
        - "jepa_only": only JEPA improved.
        - "neither_improved": both ran but neither improved.
        - "insufficient_pairs": at least one model had too few pairs to retrain
          (verdict == "insufficient_real_pairs").

    Args:
        eorm_verdict: One of "improved", "no_improvement", "insufficient_real_pairs".
        jepa_verdict: One of "improved", "no_improvement", "insufficient_real_pairs".

    Returns:
        Combined verdict string.
    """
    if eorm_verdict == "insufficient_real_pairs" or jepa_verdict == "insufficient_real_pairs":
        return "insufficient_pairs"
    eorm_improved = eorm_verdict == "improved"
    jepa_improved = jepa_verdict == "improved"
    if eorm_improved and jepa_improved:
        return "both_improved"
    if eorm_improved:
        return "eorm_only"
    if jepa_improved:
        return "jepa_only"
    return "neither_improved"


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    force_live: bool = False,
    repo_root: Path | None = None,
) -> dict:
    """Execute Exp 383: load live pairs, retrain EORM and JEPA, evaluate AUC.

    **For engineers:**
        This is the single entry point for both live execution and unit tests.
        All file I/O paths are resolved via ``repo_root`` so tests can isolate
        to a temporary directory without touching real results files.

    Args:
        force_live: Unused in this experiment (CPU training; kept for consistency
            with other experiment scripts that respect CARNOT_FORCE_LIVE).
        repo_root: Override repo root (used in tests for temp directory isolation).

    Returns:
        Full experiment artifact dict (matching the structure written to JSON).

    Spec: REQ-LEARN-025, SCENARIO-LEARN-048
    """
    _root = repo_root or _REPO_ROOT

    tmpl = ExperimentTemplate(
        EXPERIMENT_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,  # CPU training; JAX_PLATFORMS=cpu
        repo_root=_root,
    )
    tmpl.setup()

    live_files = [str(_root / f) for f in _LIVE_FILES]

    # ------------------------------------------------------------------ #
    # EORM RETRAIN                                                         #
    # ------------------------------------------------------------------ #

    # 1. Load real CoT pairs from live result files
    eorm_real_pairs = load_real_cot_pairs(live_files)
    n_eorm_pairs = len(eorm_real_pairs)
    _log.info("EORM: loaded %d real pairs", n_eorm_pairs)

    if n_eorm_pairs < EORM_MIN_PAIRS:
        _log.warning(
            "EORM: insufficient real pairs (%d < %d). Skipping retrain.",
            n_eorm_pairs,
            EORM_MIN_PAIRS,
        )
        eorm_verdict = "insufficient_real_pairs"
        eorm_before_auc = 0.5
        eorm_after_auc = 0.5
        eorm_improvement = 0.0
        eorm_model_path = ""
    else:
        # 2. Build EORM model (fresh; no Exp 346 baseline required)
        eorm_model = EORMModel(
            embed_dim=EMBED_DIM,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            key=jrandom.PRNGKey(383),
        )

        # 3. 80/20 split
        n_eorm_train = max(1, int(n_eorm_pairs * TRAIN_SPLIT))
        eorm_train = eorm_real_pairs[:n_eorm_train]
        eorm_test = eorm_real_pairs[n_eorm_train:] if n_eorm_pairs > n_eorm_train else eorm_real_pairs

        # 4. Evaluate before_auc
        eorm_before_auc = _evaluate_eorm_auc(eorm_model, eorm_test)
        _log.info("EORM before_auc = %.4f", eorm_before_auc)

        # 5. Build contrastive triples and train
        triples = _pairs_to_contrastive_triples(eorm_train)
        trainer = EORMTrainer(eorm_model, lr=LR, margin=MARGIN)

        if triples:
            for epoch in range(EORM_EPOCHS):
                loss = trainer.train_epoch(triples, batch_size=BATCH_SIZE)
                if (epoch + 1) % max(1, EORM_EPOCHS // 5) == 0:
                    _log.info(
                        "EORM epoch %d/%d — mean loss = %.4f",
                        epoch + 1, EORM_EPOCHS, loss,
                    )
        else:
            _log.warning("EORM: no contrastive triples formed; model unchanged")

        # 6. Evaluate after_auc
        eorm_after_auc = _evaluate_eorm_auc(eorm_model, eorm_test)
        eorm_improvement = eorm_after_auc - eorm_before_auc
        _log.info(
            "EORM after_auc = %.4f (improvement = %+.4f)",
            eorm_after_auc,
            eorm_improvement,
        )
        eorm_verdict = "improved" if eorm_improvement > 0 else "no_improvement"

        # 7. Save retrained EORM model
        eorm_model_path = str(_root / "results" / "eorm_model_383_real.safetensors")
        try:
            eorm_model.save(eorm_model_path)
            _log.info("Saved EORM model to %s", eorm_model_path)
        except Exception as exc:
            _log.warning("Could not save EORM model: %s", exc)
            eorm_model_path = ""

    # ------------------------------------------------------------------ #
    # JEPA RETRAIN                                                         #
    # ------------------------------------------------------------------ #

    # 8. Load JEPA violation pairs from live result files
    jepa_real_pairs = _load_jepa_pairs_from_files(live_files)
    n_jepa_pairs = len(jepa_real_pairs)
    _log.info("JEPA: loaded %d real pairs", n_jepa_pairs)

    if n_jepa_pairs < JEPA_MIN_PAIRS:
        _log.warning(
            "JEPA: insufficient real pairs (%d < %d). Skipping retrain.",
            n_jepa_pairs,
            JEPA_MIN_PAIRS,
        )
        jepa_verdict = "insufficient_real_pairs"
        jepa_before_auc = 0.5
        jepa_after_auc = 0.5
        jepa_improvement = 0.0
        jepa_model_path = ""
    else:
        # 9. Build fresh JEPA model
        jepa_config = JEPAEnergyConfig(embed_dim=JEPA_EMBED_DIM)
        jepa_model = ContextPredictionEnergy(jepa_config)
        retrainer = JEPARetrainer(jepa_model, lr=LR)

        # 10. 80/20 split
        n_jepa_train = max(1, int(n_jepa_pairs * TRAIN_SPLIT))
        jepa_train = jepa_real_pairs[:n_jepa_train]
        jepa_test = jepa_real_pairs[n_jepa_train:] if n_jepa_pairs > n_jepa_train else jepa_real_pairs

        # 11. Evaluate before_auc
        jepa_before_auc = retrainer.evaluate_auc_roc(jepa_test)
        _log.info("JEPA before_auc = %.4f", jepa_before_auc)

        # 12. Train for JEPA_EPOCHS
        for epoch in range(JEPA_EPOCHS):
            loss = retrainer.train_epoch(jepa_train, batch_size=BATCH_SIZE)
            if (epoch + 1) % max(1, JEPA_EPOCHS // 5) == 0:
                _log.info(
                    "JEPA epoch %d/%d — mean loss = %.4f",
                    epoch + 1, JEPA_EPOCHS, loss,
                )

        # 13. Evaluate after_auc
        jepa_after_auc = retrainer.evaluate_auc_roc(jepa_test)
        jepa_improvement = jepa_after_auc - jepa_before_auc
        _log.info(
            "JEPA after_auc = %.4f (improvement = %+.4f)",
            jepa_after_auc,
            jepa_improvement,
        )
        jepa_verdict = "improved" if jepa_improvement > 0 else "no_improvement"

        # 14. Save retrained JEPA model (best-effort; JEPA has no safetensors save built-in)
        jepa_model_path = str(_root / "results" / "jepa_predictor_383_real.safetensors")
        try:
            import numpy as np
            from safetensors.numpy import save_file as st_save

            flat: dict[str, object] = {}
            for i, (w, b) in enumerate(jepa_model.layers):
                flat[f"layer_{i}_weight"] = np.asarray(w)
                flat[f"layer_{i}_bias"] = np.asarray(b)
            flat["output_weight"] = np.asarray(jepa_model.output_weight)
            flat["output_bias"] = np.array([jepa_model.output_bias], dtype=np.float32)
            Path(jepa_model_path).parent.mkdir(parents=True, exist_ok=True)
            st_save(flat, jepa_model_path)
            _log.info("Saved JEPA model to %s", jepa_model_path)
        except Exception as exc:
            _log.warning("Could not save JEPA model: %s", exc)
            jepa_model_path = ""

    # ------------------------------------------------------------------ #
    # Build combined artifact                                              #
    # ------------------------------------------------------------------ #

    retrain_mode = (
        "real_data"
        if (n_eorm_pairs >= EORM_MIN_PAIRS or n_jepa_pairs >= JEPA_MIN_PAIRS)
        else "synthetic_only"
    )
    honest_verdict = _combined_honest_verdict(eorm_verdict, jepa_verdict)

    artifact = tmpl.build_result(
        {
            "schema": "carnot.combined_retrain.v1",
            # EORM metrics
            "n_eorm_pairs": n_eorm_pairs,
            "eorm_before_auc": round(eorm_before_auc, 6),
            "eorm_after_auc": round(eorm_after_auc, 6),
            "eorm_improvement": round(eorm_improvement, 6),
            "eorm_verdict": eorm_verdict,
            "eorm_model_path": eorm_model_path,
            # JEPA metrics
            "n_jepa_pairs": n_jepa_pairs,
            "jepa_before_auc": round(jepa_before_auc, 6),
            "jepa_after_auc": round(jepa_after_auc, 6),
            "jepa_improvement": round(jepa_improvement, 6),
            "jepa_verdict": jepa_verdict,
            "jepa_model_path": jepa_model_path,
            # Combined
            "retrain_mode": retrain_mode,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    return artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 383 and write results to the deliverable JSON file."""
    force_live = bool(int(os.environ.get("CARNOT_FORCE_LIVE", "0")))

    artifact = run_experiment(force_live=force_live)

    deliverable = _REPO_ROOT / DELIVERABLE
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    with open(deliverable, "w") as f:
        json.dump(artifact, f, indent=2)

    _log.info(
        "Exp 383 complete: retrain_mode=%s, honest_verdict=%s, "
        "eorm=%s (%.4f→%.4f), jepa=%s (%.4f→%.4f)",
        artifact.get("retrain_mode"),
        artifact.get("honest_verdict"),
        artifact.get("eorm_verdict"),
        artifact.get("eorm_before_auc", 0.0),
        artifact.get("eorm_after_auc", 0.0),
        artifact.get("jepa_verdict"),
        artifact.get("jepa_before_auc", 0.0),
        artifact.get("jepa_after_auc", 0.0),
    )


if __name__ == "__main__":
    main()
