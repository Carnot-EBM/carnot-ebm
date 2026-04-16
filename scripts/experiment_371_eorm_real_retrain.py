#!/usr/bin/env python3
"""Exp 371 — EORM Real-Data Retrain: First Genuine Result from Live GPU Pairs.

**Researcher summary:**
    Exp 359 ran with only 5 real pairs (AUC unchanged at 0.500) because all prior
    "live" experiments were actually simulated. Exps 368/369/370 now produce real
    (question, response, is_correct) triples from live GPU inference:

    - Exp 368: ~200 GSM8K arithmetic pairs (Gemma 4 / Qwen3 live inference)
    - Exp 369: ~50 HumanEval code pairs (CodeExtractor + property-based tests)
    - Exp 370: ~200 adversarial GSM8K pairs (Apple arXiv 2410.05229 setup)

    With 200+ real pairs available, this experiment performs the first genuine EORM
    retrain with real LLM failure modes. Target: AUC-ROC improvement from 0.500
    (random-init or synthetic-only baseline) to at least 0.65 on held-out live pairs.

**Why real data matters:**
    Synthetic pairs are generated from template text — they do not capture the
    characteristic mistakes real LLMs make (off-by-one arithmetic, unit confusion,
    wrong loop termination, hallucinated intermediate steps). Real data teaches EORM
    to recognize the specific failure modes of the target model on actual benchmarks.

**Honest reporting:**
    - ``"real_data_improvement"``: ≥50 real pairs AND after_auc > before_auc.
      This is the first genuine evidence of EORM real-data improvement.
    - ``"real_data_no_improvement"``: ≥50 real pairs but AUC flat or regressed.
      Logged honestly — may indicate need for more data or hyperparameter tuning.
    - ``"insufficient_real_pairs"``: <50 real pairs found — live GPU results from
      Exps 368/369/370 not yet available. Blocked artifact, no training performed.

**Usage:**
    JAX_PLATFORMS=cpu python scripts/experiment_371_eorm_real_retrain.py
    CARNOT_FORCE_LIVE=1 python scripts/experiment_371_eorm_real_retrain.py

Spec: REQ-LEARN-025, SCENARIO-LEARN-043, SCENARIO-LEARN-044, SCENARIO-LEARN-048
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure repo root on sys.path so `carnot.*` and `scripts.*` imports resolve
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax.random as jrandom

from carnot.embeddings.jepa_retrain import ViolationPair
from carnot.models.eorm import CoTEnergyInput, EORMModel, EORMTrainer
from carnot.models.eorm_retrain import (
    EORMRetrainResult,
    build_retrain_artifact,
    load_real_cot_pairs,
    make_synthetic_eorm_pairs,
    merge_cot_corpora,
)
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 371
TITLE = "EORM real-data retrain — first genuine result from live GPU pairs (Exps 368/369/370)"
DELIVERABLE = "results/experiment_371_eorm_real_retrain.json"

# Minimum real pairs required to declare retrain_mode="real_data"
# Below this threshold we produce an honest blocked artifact and skip training.
REAL_DATA_THRESHOLD = 50

# Training hyperparameters — match Exp 359 for comparability
TRAIN_SPLIT = 0.8
N_EPOCHS_CI = 50       # CPU / CI mode (fast; enough to demonstrate convergence)
N_EPOCHS_LIVE = 200    # when CARNOT_FORCE_LIVE=1 (more thorough, matches original intent)
BATCH_SIZE = 16
LR = 1e-4
MARGIN = 1.0

# Corpus caps
MAX_REAL = 300
MAX_SYNTHETIC = 50     # fewer synthetic than Exp 359 — real data is now plentiful

# EORM architecture — match Exp 346/359 defaults for comparable parameter count
EMBED_DIM = 128
N_HEADS = 4
N_LAYERS = 2


# ---------------------------------------------------------------------------
# AUC-ROC evaluator (no sklearn — keeps CI fast on CPU)
# ---------------------------------------------------------------------------


def _evaluate_eorm_auc(model: EORMModel, pairs: list[ViolationPair]) -> float:
    """Compute AUC-ROC for an EORM model on a ViolationPair test set.

    **For engineers:**
        EORM outputs lower energy for responses it considers correct.
        ``has_violation=True`` means incorrect response — the positive class.
        To match standard AUC convention (high score = positive class), we use
        NEGATED energy as the discriminating score:
            score = -energy  (high negated energy = high energy = predicted violation)

        Trapezoidal AUC computed directly without sklearn to avoid the dependency.

    Args:
        model: Trained or freshly initialized EORMModel.
        pairs: ViolationPair test examples to score.

    Returns:
        AUC-ROC in [0, 1].  0.5 = random baseline.  Returns 0.5 when the test
        set has no positive or no negative examples (degenerate case).
    """
    if not pairs:
        return 0.5

    scores: list[float] = []
    labels: list[int] = []
    for p in pairs:
        # Use question_id as the question text proxy — EORM accepts any string.
        # For real pairs, question_id is the GSM8K/HumanEval problem identifier.
        cot = CoTEnergyInput(
            question_text=p.question_id,
            response_text=p.full_response,
        )
        energy = model.energy(cot)
        scores.append(-energy)                    # negate: high score = predicted violation
        labels.append(1 if p.has_violation else 0)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        # Cannot compute meaningful AUC without both classes
        return 0.5

    # Sort descending by score (high score = more likely to be a violation)
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

    # Trapezoidal integration under the ROC curve
    auc = 0.0
    for i in range(1, len(fpr_pts)):
        dfpr = fpr_pts[i] - fpr_pts[i - 1]
        auc += dfpr * (tpr_pts[i] + tpr_pts[i - 1]) / 2.0

    return float(auc)


# ---------------------------------------------------------------------------
# Contrastive triple construction
# ---------------------------------------------------------------------------


def _pairs_to_contrastive_triples(
    pairs: list[ViolationPair],
) -> list[tuple[str, str, str]]:
    """Convert ViolationPair list into (correct, incorrect, question) triples.

    **For engineers:**
        EORM trains on *contrastive pairs*: for the same question, which response
        is correct vs. which is wrong? ViolationPairs carry individual responses
        with binary labels (has_violation=True means incorrect).

        This function groups pairs by question_id and forms the round-robin
        cross-product (correct_response × incorrect_response) for each question
        that has at least one of each. Questions with only one label type are
        skipped — no contrastive signal is available.

        Synthetic pairs all have distinct question_ids like "synthetic_0",
        "synthetic_1", etc., which prevents them from being matched with each
        other via question_id. They are pooled into a shared "_synthetic_pool"
        bucket so that synthetic correct/incorrect pairs can be cross-matched —
        this is the same strategy as Exp 359 to maximize synthetic training signal.

    Args:
        pairs: ViolationPair objects from real or synthetic corpora.

    Returns:
        List of (correct_response, incorrect_response, question_text) tuples.
    """
    from collections import defaultdict

    correct_by_q: dict[str, list[str]] = defaultdict(list)
    incorrect_by_q: dict[str, list[str]] = defaultdict(list)

    _SYNTHETIC_POOL = "_synthetic_pool"

    for p in pairs:
        q_id = p.question_id
        # Synthetic and "unknown" IDs go into a shared pool so they can be
        # cross-matched even though each synthetic pair has a unique question_id.
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
        # Round-robin pairing avoids O(n^2) explosion for large question groups
        n_pairs = max(len(corrects), len(incorrects))
        for i in range(n_pairs):
            c = corrects[i % len(corrects)]
            ic = incorrects[i % len(incorrects)]
            triples.append((c, ic, q_id))

    return triples


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def _load_or_build_eorm_model(baseline_path: Path) -> EORMModel:
    """Load the Exp 346 EORM baseline or build a fresh model if not present.

    **For engineers:**
        We prefer to load the Exp 346 trained model so ``before_auc`` reflects
        the state after synthetic training rather than random initialization.
        If the file is absent (Exp 346 was never run on this machine), we build
        a fresh model with a fixed seed — ``before_auc`` will be ~0.5 in that
        case, which is the correct honest baseline for random initialization.

    Args:
        baseline_path: Expected path to ``results/eorm_model_346.safetensors``.

    Returns:
        EORMModel ready for evaluation (either loaded or freshly initialized).
    """
    if baseline_path.exists():
        try:
            model = EORMModel.load(str(baseline_path))
            _log.info("Loaded Exp 346 EORM baseline from %s", baseline_path)
            return model
        except Exception as exc:
            _log.warning("Failed to load Exp 346 model (%s); building fresh model", exc)

    model = EORMModel(
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        key=jrandom.PRNGKey(346),
    )
    _log.info(
        "Built fresh EORMModel (embed_dim=%d, n_layers=%d, seed=346)",
        EMBED_DIM, N_LAYERS,
    )
    return model


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    force_live: bool = False,
    repo_root: Path | None = None,
) -> dict:
    """Execute Exp 371: load live GPU pairs, retrain EORM, evaluate AUC before/after.

    **For engineers:**
        This function is the single entry point for both live execution and unit tests.
        Pass ``repo_root`` in tests to redirect all file I/O to a temporary directory.

    Args:
        force_live: If True, train for N_EPOCHS_LIVE (200) instead of N_EPOCHS_CI (50).
        repo_root: Override the repo root path (used in unit tests for isolation).

    Returns:
        Full experiment artifact dict — the same structure written to the JSON file.
        Key ``honest_verdict`` takes one of three values:
        - ``"insufficient_real_pairs"``: <50 real pairs found; training skipped.
        - ``"real_data_improvement"``: ≥50 real pairs AND AUC improved after retrain.
        - ``"real_data_no_improvement"``: ≥50 real pairs but AUC flat or regressed.
    """
    _root = repo_root or _REPO_ROOT

    tmpl = ExperimentTemplate(
        EXPERIMENT_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,  # EORM is small — CPU training is fast enough
        repo_root=_root,
    )
    tmpl.setup()

    # ---- 1. Load real pairs from Exps 368 / 369 / 370 ----
    result_files = [
        str(_root / "results" / "experiment_368_precision_live.json"),
        str(_root / "results" / "experiment_369_humaneval_live.json"),
        str(_root / "results" / "experiment_370_adversarial_live.json"),
    ]
    real_pairs = load_real_cot_pairs(result_files)
    n_real = len(real_pairs)
    _log.info("Loaded %d real CoT pairs from Exps 368/369/370", n_real)

    # ---- 2. Blocked artifact when real data is insufficient ----
    if n_real < REAL_DATA_THRESHOLD:
        _log.warning(
            "Only %d real pairs found (minimum %d). "
            "Exps 368/369/370 results not yet available — producing blocked artifact.",
            n_real, REAL_DATA_THRESHOLD,
        )
        artifact = tmpl.build_result(
            {
                "schema": "carnot.eorm_retrain.v2",
                "honest_verdict": "insufficient_real_pairs",
                "n_real_pairs": n_real,
                "n_real_pairs_minimum_required": REAL_DATA_THRESHOLD,
                "retrain_mode": "blocked",
                "before_auc": None,
                "after_auc": None,
                "auc_improvement": None,
            },
            status="blocked",
        )
        return artifact

    # ---- 3. Build corpus ----
    synthetic_pairs = make_synthetic_eorm_pairs(n=MAX_SYNTHETIC + 20, seed=371)
    corpus = merge_cot_corpora(
        real_pairs, synthetic_pairs, max_real=MAX_REAL, max_synthetic=MAX_SYNTHETIC
    )
    n_synthetic_used = len(corpus) - min(n_real, MAX_REAL)
    _log.info(
        "Corpus: %d total pairs (%d real capped at %d, %d synthetic)",
        len(corpus), n_real, MAX_REAL, n_synthetic_used,
    )

    # ---- 4. Train / test split (80/20, no shuffle for reproducibility) ----
    n_train = max(1, int(len(corpus) * TRAIN_SPLIT))
    train_pairs = corpus[:n_train]
    # When corpus is very small, fall back to the full corpus as test set
    test_pairs = corpus[n_train:] if len(corpus) > n_train else corpus

    # ---- 5. Load / build baseline EORM model ----
    baseline_path = _root / "results" / "eorm_model_346.safetensors"
    model = _load_or_build_eorm_model(baseline_path)

    # ---- 6. Evaluate AUC before retraining ----
    before_auc = _evaluate_eorm_auc(model, test_pairs)
    _log.info("before_auc = %.4f", before_auc)

    # ---- 7. Build contrastive triples and train ----
    triples = _pairs_to_contrastive_triples(train_pairs)
    n_epochs = N_EPOCHS_LIVE if force_live else N_EPOCHS_CI
    _log.info(
        "Training %d epochs on %d contrastive triples (from %d train pairs)",
        n_epochs, len(triples), len(train_pairs),
    )

    trainer = EORMTrainer(model, lr=LR, margin=MARGIN)

    if triples:
        for epoch in range(n_epochs):
            loss = trainer.train_epoch(triples, batch_size=BATCH_SIZE)
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                _log.info("Epoch %d/%d — mean loss = %.4f", epoch + 1, n_epochs, loss)
    else:
        _log.warning(
            "No contrastive triples could be formed (need at least one correct AND "
            "one incorrect response per question). Model parameters unchanged."
        )

    # ---- 8. Evaluate AUC after retraining ----
    after_auc = _evaluate_eorm_auc(model, test_pairs)
    auc_improvement = after_auc - before_auc  # signed, no clamping
    _log.info("after_auc = %.4f (improvement = %+.4f)", after_auc, auc_improvement)

    # ---- 9. Save retrained model ----
    model_path = str(_root / "results" / "eorm_model_371_real.safetensors")
    try:
        model.save(model_path)
        _log.info("Saved retrained model to %s", model_path)
    except Exception as exc:
        _log.warning("Could not save model: %s", exc)
        model_path = ""

    # ---- 10. Build artifact with honest_verdict ----
    # Determine verdict — real data was available in all paths that reach here
    if auc_improvement > 0:
        honest_verdict = "real_data_improvement"
    else:
        honest_verdict = "real_data_no_improvement"

    result = EORMRetrainResult(
        n_real_pairs=min(n_real, MAX_REAL),
        n_synthetic_pairs=n_synthetic_used,
        before_auc=before_auc,
        after_auc=after_auc,
        auc_improvement=auc_improvement,
        retrain_mode="real_data",
        model_path=model_path,
    )

    # build_retrain_artifact produces schema v1 fields; we overlay the v2 schema tag
    retrain_data = build_retrain_artifact(result)
    retrain_data["schema"] = "carnot.eorm_retrain.v2"
    retrain_data["honest_verdict"] = honest_verdict  # override with our verdict

    artifact = tmpl.build_result(
        {
            **retrain_data,
            "n_contrastive_triples": len(triples),
            "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs),
            "n_epochs": n_epochs,
            "n_real_pairs_loaded": n_real,
        },
        status="success",
    )

    return artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 371 and write results to the deliverable JSON file."""
    force_live = bool(int(os.environ.get("CARNOT_FORCE_LIVE", "0")))

    artifact = run_experiment(force_live=force_live)

    deliverable = _REPO_ROOT / DELIVERABLE
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    with open(deliverable, "w") as f:
        json.dump(artifact, f, indent=2)

    _log.info(
        "Exp 371 complete: honest_verdict=%s, before_auc=%s, after_auc=%s, "
        "improvement=%s, n_real_pairs=%s",
        artifact.get("honest_verdict"),
        artifact.get("before_auc"),
        artifact.get("after_auc"),
        artifact.get("auc_improvement"),
        artifact.get("n_real_pairs"),
    )


if __name__ == "__main__":
    main()
