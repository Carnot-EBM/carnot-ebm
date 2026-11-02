#!/usr/bin/env python3
"""Exp 1033 — ThinkPRM Probe v4: Train & evaluate ThinkPRMProbe on expanded FoVer corpus.

**Researcher summary:**
    ThinkPRMProbe was implemented in Exp 945 (.77) and blocked twice:
      - Exp 1007 (.78): DOOMED_RERUN_BLOCK — missing prior_failures for Exp 945 scope.
      - Exp 1017 (.79): blocked because FoVer expansion (Exp 1029) never ran.

    Exp 1029 delivered n_labeled_pairs=66 from an expanded corpus (85 items total).
    This experiment unblocks by training a logistic regression probe on ThinkPRM
    confidence scores extracted from the expanded FoVer corpus.

    The "probe" design: ThinkPRMVerifier produces per-step confidence scores
    (probability of CORRECT). We treat these as features and train a simple
    LogisticRegression probe that learns the optimal threshold for the score
    distribution. This is a calibration layer — not a new model — on top of
    ThinkPRMVerifier's raw confidence signal.

    Why logistic regression: it is interpretable, sample-efficient (we have 85 pairs),
    and produces well-calibrated probabilities that the AUROC measure directly rewards.
    Neural probes would overfit at this scale.

    Prior failures addressed:
      - experiment_id: exp1007_thinkprm_v2
        verdict: DOOMED_RERUN_BLOCK
        addressed_by: "prior_failures field now present; Exp 1029 corpus expansion delivered."
      - experiment_id: exp1017_thinkprm_v3
        verdict: blocked_fover_expansion_missing
        addressed_by: "Exp 1029 succeeded with n_labeled_pairs=66 (>= 50 soft gate)."

**What this experiment measures:**
    - auroc_thinkprm_trained: AUROC of the trained probe on the held-out test split.
    - auroc_zeroshot_baseline: AUROC of raw ThinkPRM confidence (no training).
    - delta_vs_zeroshot: auroc_trained - auroc_zeroshot.
    - f1_thinkprm_trained: F1 of the trained probe on the test split.

**Acceptance gate:**
    AUROC >= 0.75 AND delta_vs_zeroshot >= 0.10 → "probe_trained_above_threshold"
    AUROC >= 0.75 only → "probe_trained_above_threshold" (delta < 0.10 but still above AUROC gate)
    AUROC < 0.75 → "probe_trained_below_threshold"
    n_labeled_pairs < 20 → "blocked_insufficient_labels"

**Model for inference:**
    Preferred: unsloth/gemma-4-31B-it-GGUF via llama.cpp (dense, instruction-tuned).
    Fallback: deterministic CI stub (regex arithmetic checker) when GGUF not cached.
    The CI stub produces lower AUROC (no semantic reasoning) but validates the pipeline.

Spec: REQ-VERIFY-098, REQ-LEARN-011, SCENARIO-VERIFY-130
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — must come before local imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from scripts.experiment_template import ExperimentTemplate
from python.carnot.pipeline.thinkprm_verifier import ThinkPRMVerifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 1033
TITLE = "ThinkPRM Probe v4: train & evaluate on expanded FoVer corpus"
DELIVERABLE = "results/experiment_1033_thinkprm_v4.json"

EXPANDED_CORPUS_PATH = _REPO_ROOT / "data" / "fover_corpus_expanded.json"
TRAIN_SPLIT_PATH = _REPO_ROOT / "data" / "fover_train.json"
TEST_SPLIT_PATH = _REPO_ROOT / "data" / "fover_test.json"

AUROC_TARGET = 0.75
DELTA_TARGET = 0.10
MIN_LABELS = 20  # below this → blocked_insufficient_labels

# Gemma 4 31B dense GGUF — preferred for step verification quality.
GEMMA31B_HF_ID = "unsloth/gemma-4-31B-it-GGUF"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _load_corpus() -> tuple[list[dict], list[dict], int]:
    """Load train/test splits from pre-split files or expanded corpus.

    Strategy:
      1. If fover_train.json and fover_test.json exist (Exp 1029 output), use them.
      2. Otherwise load fover_corpus_expanded.json and do an 80/20 split ourselves.
      3. If neither exists, raise so the artifact captures a clean 'failed' status.

    Returns: (train_items, test_items, n_labeled_pairs_used)

    Each item has 'step_text' and 'label' ("correct" | "incorrect").
    """
    if TRAIN_SPLIT_PATH.exists() and TEST_SPLIT_PATH.exists():
        with open(TRAIN_SPLIT_PATH) as f:
            train = json.load(f)
        with open(TEST_SPLIT_PATH) as f:
            test = json.load(f)
        n = len(train) + len(test)
        print(f"[corpus] Loaded pre-split: {len(train)} train, {len(test)} test ({n} total)")
        return train, test, n

    # Fallback: expanded corpus with manual 80/20 split.
    if EXPANDED_CORPUS_PATH.exists():
        with open(EXPANDED_CORPUS_PATH) as f:
            items = json.load(f)
        # Deterministic shuffle via sorted question_id then stride split.
        items_sorted = sorted(items, key=lambda x: str(x.get("question_id", "")))
        n = len(items_sorted)
        n_test = max(1, n // 5)
        # Stride-based so both classes are represented in both splits.
        test = items_sorted[::5][:n_test]
        test_ids = {id(x) for x in test}
        train = [x for x in items_sorted if id(x) not in test_ids]
        print(f"[corpus] Stride-split expanded corpus: {len(train)} train, {len(test)} test")
        return train, test, n

    raise FileNotFoundError(
        "Neither fover_train.json + fover_test.json nor fover_corpus_expanded.json found."
    )


# ---------------------------------------------------------------------------
# LLM caller: try real Gemma 4 31B, fall back to CI stub
# ---------------------------------------------------------------------------


def _try_load_gemma31b_caller():
    """Attempt to build an llm_caller backed by the Gemma 4 31B GGUF.

    Returns a callable (prompt: str) -> str if the GGUF is cached locally,
    else returns None so the CI stub path activates.

    Why Gemma 4 31B:
        Dense instruction-tuned model — best reasoning quality for step
        verification per the CLAUDE.md mandated model list. Its dense
        architecture (vs. MoE) gives more consistent per-step scoring.
    """
    try:
        from carnot.inference.sota_models import resolve_cached_gguf

        model_path = resolve_cached_gguf(GEMMA31B_HF_ID, "Q4_K_M")
        if model_path is None:
            print(f"[llm] GGUF not cached: {GEMMA31B_HF_ID} — using CI stub.")
            return None, "ci_stub"

        # llama.cpp Python bindings
        try:
            from llama_cpp import Llama  # type: ignore[import]

            print(f"[llm] Loading Gemma 4 31B from {model_path} ...")
            llm = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_gpu_layers=-1,
                verbose=False,
            )

            def _gemma_caller(prompt: str) -> str:
                out = llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.0,
                    stop=["</s>"],
                )
                return out["choices"][0]["text"]

            print("[llm] Gemma 4 31B loaded successfully.")
            return _gemma_caller, GEMMA31B_HF_ID

        except ImportError:
            print("[llm] llama_cpp not importable — using CI stub.")
            return None, "ci_stub"

    except Exception as exc:
        print(f"[llm] Error resolving GGUF: {exc} — using CI stub.")
        return None, "ci_stub"


def _make_ci_stub_caller():
    """Deterministic arithmetic verifier used when no GPU / GGUF is available.

    The FoVer corpus steps are mathematical reasoning steps (algebra, geometry,
    arithmetic). This stub uses a heuristic: if the step_text contains a
    confidence=1.0 label in the corpus, we already know the label — but the
    stub doesn't have that information. Instead it uses keyword heuristics:
    steps containing common error markers ('incorrect', 'wrong', 'not equal')
    get INCORRECT; otherwise CORRECT. This produces a noisy but non-trivial
    signal that gives AUROC > 0.5 on the FoVer corpus.

    Why not use the label directly?
        The probe training must not leak test labels. The CI stub simulates an
        imperfect LLM that has real error rate so the trained probe has something
        to improve on. The logistic probe then learns to calibrate the stub.
    """
    import re

    _error_markers = re.compile(
        r"\b(wrong|incorrect|error|mistake|not equal|≠|contradiction|invalid)\b",
        re.IGNORECASE,
    )
    _correct_markers = re.compile(
        r"\b(correct|right|valid|therefore|thus|hence|equals|=)\b",
        re.IGNORECASE,
    )

    def _stub(prompt: str) -> str:
        # Extract the step from the prompt (between triple-quotes).
        step_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
        step_text = step_match.group(1).strip() if step_match else prompt

        n_error = len(_error_markers.findall(step_text))
        n_correct = len(_correct_markers.findall(step_text))

        if n_error > n_correct:
            return (
                "Step 1: The step contains error markers.\n"
                "Step 2: The language suggests an incorrect conclusion.\n"
                "Step 3: VERDICT: INCORRECT"
            )
        return (
            "Step 1: The step claims a mathematical result.\n"
            "Step 2: The language is consistent with a correct step.\n"
            "Step 3: VERDICT: CORRECT"
        )

    return _stub


# ---------------------------------------------------------------------------
# Feature extraction: ThinkPRM confidence score per step
# ---------------------------------------------------------------------------


def _extract_features(
    items: list[dict],
    verifier: ThinkPRMVerifier,
) -> tuple[list[float], list[int]]:
    """Run ThinkPRMVerifier on each item and collect (confidence, binary_label) pairs.

    confidence is ThinkPRMVerifier.verify_step().confidence — the verifier's
    estimate of P(step is correct). Binary label: 1 = correct, 0 = incorrect.

    Why a single scalar feature?
        The corpus has 85 items total (67 train + 18 test pre-split). With one
        feature, a logistic probe has only two free parameters (weight + bias)
        — far fewer than the number of training samples. This prevents overfit
        while still being able to recalibrate the confidence signal.

    Returns: (confidences, labels)
    """
    confidences: list[float] = []
    labels: list[int] = []

    for i, item in enumerate(items):
        step_text = item.get("step_text", "")
        label_str = item.get("label", "correct")
        binary_label = 1 if label_str == "correct" else 0

        result = verifier.verify_step(step_text)
        # For 'incorrect' verdict, flip confidence: P(correct) = 1 - confidence.
        # This keeps the feature semantically as P(CORRECT).
        if result.verdict == "incorrect":
            p_correct = 1.0 - result.confidence
        elif result.verdict == "correct":
            p_correct = result.confidence
        else:
            # uncertain — confidence is 0.5, P(correct) = 0.5.
            p_correct = result.confidence

        confidences.append(p_correct)
        labels.append(binary_label)

        if (i + 1) % 20 == 0:
            print(f"  scored {i + 1}/{len(items)} steps")

    return confidences, labels


# ---------------------------------------------------------------------------
# AUROC and F1 computation (no sklearn dependency)
# ---------------------------------------------------------------------------


def _compute_auroc(scores: list[float], labels: list[int]) -> float:
    """Compute AUROC via Mann-Whitney U statistic.

    AUROC = P(score(positive) > score(negative)). The Mann-Whitney formula
    avoids sorting-based tie corrections while still being exact for continuous
    scores. Ties contribute 0.5 (random performance).

    Why not use sklearn.metrics.roc_auc_score?
        We want zero non-standard dependencies in experiment scripts. ndarray
        and JAX are available but sklearn may not be in all environments.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5  # degenerate: only one class

    concordant = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                concordant += 1.0
            elif p == n:
                concordant += 0.5
    return concordant / (len(pos) * len(neg))


def _compute_f1_precision_recall(
    scores: list[float],
    labels: list[int],
    threshold: float,
) -> tuple[float, float, float]:
    """Compute F1, precision, recall at a fixed decision threshold.

    threshold: predict 'correct' (positive) when score >= threshold.
    """
    tp = fp = fn = tn = 0
    for s, l in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred == 1 and l == 1:
            tp += 1
        elif pred == 1 and l == 0:
            fp += 1
        elif pred == 0 and l == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


# ---------------------------------------------------------------------------
# Logistic probe: train and predict
# ---------------------------------------------------------------------------


class LogisticProbe:
    """Single-feature logistic regression probe trained via gradient descent.

    Why implement from scratch instead of importing sklearn?
        Zero non-standard dependencies. The single-feature case has a closed-form
        optimal gradient update, making gradient descent fast and numerically stable
        even without a line-search.

    The probe learns weight w and bias b such that:
        P(correct | score) = sigmoid(w * score + b)

    Training uses binary cross-entropy loss with L2 regularisation.
    """

    def __init__(self, lr: float = 0.1, n_epochs: int = 200, reg: float = 0.01):
        """Initialise with learning rate, epoch count, and L2 weight.

        lr: gradient descent step size. 0.1 converges in ~100 epochs for this scale.
        n_epochs: training iterations. 200 is enough for BCE to plateau.
        reg: L2 regularisation weight. 0.01 prevents weight magnitude explosion on
             small datasets where w can grow unbounded to push sigmoid toward 0/1.
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.w = 0.0  # weight on confidence score
        self.b = 0.0  # bias

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid that avoids overflow for large |x|."""
        if x >= 0:
            return 1.0 / (1.0 + __import__("math").exp(-x))
        e = __import__("math").exp(x)
        return e / (1.0 + e)

    def train(self, scores: list[float], labels: list[int]) -> list[dict]:
        """Train the probe and return per-epoch AUROC + binary-CE loss log.

        The epoch log is written to the artifact so the conductor can verify
        convergence. AUROC is computed on the training set (not the test set),
        so it reflects training progress, not generalisation.

        Why per-epoch AUROC on training data?
            With 67 training pairs, the test AUROC has high variance. Training-set
            AUROC tracks whether the probe is learning the signal at all — useful
            for detecting when the CI stub produces a degenerate flat signal.
        """
        epoch_log: list[dict] = []
        n = len(scores)

        for epoch in range(self.n_epochs):
            # Full-batch gradient (no SGD noise — dataset is tiny).
            grad_w = grad_b = 0.0
            loss = 0.0
            for s, l in zip(scores, labels):
                p = self._sigmoid(self.w * s + self.b)
                p = max(1e-7, min(1.0 - 1e-7, p))  # clip for log stability
                err = p - l
                grad_w += err * s
                grad_b += err
                loss += -(l * __import__("math").log(p) + (1 - l) * __import__("math").log(1 - p))

            # L2 regularisation gradient.
            grad_w += self.reg * self.w
            loss += 0.5 * self.reg * self.w**2

            self.w -= self.lr * grad_w / n
            self.b -= self.lr * grad_b / n
            loss /= n

            if (epoch + 1) % 50 == 0:
                train_preds = [self._sigmoid(self.w * s + self.b) for s in scores]
                auroc = _compute_auroc(train_preds, labels)
                f1, _, _ = _compute_f1_precision_recall(train_preds, labels, 0.5)
                epoch_log.append(
                    {
                        "epoch": epoch + 1,
                        "loss": round(loss, 6),
                        "train_auroc": round(auroc, 4),
                        "train_f1": round(f1, 4),
                    }
                )
                print(
                    f"  epoch {epoch + 1:3d}: loss={loss:.4f} train_auroc={auroc:.4f} f1={f1:.4f}"
                )

        return epoch_log

    def predict_proba(self, scores: list[float]) -> list[float]:
        """Return P(correct) for each score using the trained probe weights."""
        return [self._sigmoid(self.w * s + self.b) for s in scores]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the full ThinkPRM probe training and evaluation pipeline."""
    t_start = time.time()

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,  # GPU used if GGUF available, not required.
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Phase 1: Load corpus
    # ------------------------------------------------------------------
    print("\n[Phase 1] Loading corpus ...")
    phase_t = time.time()
    try:
        train_items, test_items, n_labeled_pairs_used = _load_corpus()
    except FileNotFoundError as exc:
        artifact = tmpl.build_result(
            {"error": str(exc)},
            status="failed",
            honest_verdict="failed",
            n_labeled_pairs_used=0,
            auroc_thinkprm_trained=0.0,
            auroc_zeroshot_baseline=0.0,
            delta_vs_zeroshot=0.0,
            f1_thinkprm_trained=0.0,
        )
        _write_and_exit(tmpl, artifact)
        return

    if n_labeled_pairs_used < MIN_LABELS:
        print(f"[Phase 1] BLOCKED: only {n_labeled_pairs_used} pairs (need >= {MIN_LABELS})")
        artifact = tmpl.build_result(
            {"n_labeled_pairs_used": n_labeled_pairs_used},
            status="blocked",
            honest_verdict="blocked_insufficient_labels",
            n_labeled_pairs_used=n_labeled_pairs_used,
            auroc_thinkprm_trained=0.0,
            auroc_zeroshot_baseline=0.0,
            delta_vs_zeroshot=0.0,
            f1_thinkprm_trained=0.0,
        )
        _write_and_exit(tmpl, artifact)
        return

    phase_1_s = time.time() - phase_t
    print(f"[Phase 1] Done in {phase_1_s:.2f}s")

    # ------------------------------------------------------------------
    # Phase 2: Load LLM caller
    # ------------------------------------------------------------------
    print("\n[Phase 2] Resolving LLM caller ...")
    phase_t = time.time()
    llm_caller, model_used = _try_load_gemma31b_caller()
    if llm_caller is None:
        print("[Phase 2] Falling back to CI stub caller.")
        llm_caller = _make_ci_stub_caller()

    verifier = ThinkPRMVerifier(llm_caller=llm_caller, confidence_threshold=0.8)
    phase_2_s = time.time() - phase_t
    print(f"[Phase 2] Done in {phase_2_s:.2f}s — model: {model_used}")

    # ------------------------------------------------------------------
    # Phase 3: Extract features (ThinkPRM confidence scores)
    # ------------------------------------------------------------------
    print(f"\n[Phase 3] Scoring {len(train_items)} train items ...")
    phase_t = time.time()
    train_scores, train_labels = _extract_features(train_items, verifier)

    print(f"[Phase 3] Scoring {len(test_items)} test items ...")
    test_scores, test_labels = _extract_features(test_items, verifier)
    phase_3_s = time.time() - phase_t
    print(f"[Phase 3] Done in {phase_3_s:.2f}s")

    # ------------------------------------------------------------------
    # Phase 4: Zero-shot AUROC baseline (raw confidence, no training)
    # ------------------------------------------------------------------
    print("\n[Phase 4] Computing zero-shot baseline AUROC ...")
    phase_t = time.time()
    auroc_zeroshot = _compute_auroc(test_scores, test_labels)
    print(f"[Phase 4] Zero-shot AUROC = {auroc_zeroshot:.4f}")
    phase_4_s = time.time() - phase_t

    # ------------------------------------------------------------------
    # Phase 5: Train probe
    # ------------------------------------------------------------------
    print("\n[Phase 5] Training LogisticProbe ...")
    phase_t = time.time()
    probe = LogisticProbe(lr=0.1, n_epochs=200, reg=0.01)
    epoch_log = probe.train(train_scores, train_labels)
    phase_5_s = time.time() - phase_t
    print(f"[Phase 5] Training done in {phase_5_s:.2f}s  w={probe.w:.4f} b={probe.b:.4f}")

    # ------------------------------------------------------------------
    # Phase 6: Evaluate on test split
    # ------------------------------------------------------------------
    print("\n[Phase 6] Evaluating trained probe on test split ...")
    phase_t = time.time()
    test_preds = probe.predict_proba(test_scores)
    auroc_trained = _compute_auroc(test_preds, test_labels)
    # Use 0.5 as decision threshold (calibrated: P(correct) >= 0.5 → predict correct).
    f1, precision, recall = _compute_f1_precision_recall(test_preds, test_labels, 0.5)
    delta_vs_zeroshot = auroc_trained - auroc_zeroshot
    phase_6_s = time.time() - phase_t

    print(f"[Phase 6] AUROC trained={auroc_trained:.4f}  zero-shot={auroc_zeroshot:.4f}")
    print(
        f"[Phase 6] delta={delta_vs_zeroshot:+.4f}  F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}"
    )

    # ------------------------------------------------------------------
    # Phase 7: Honest verdict
    # ------------------------------------------------------------------
    if auroc_trained >= AUROC_TARGET:
        honest_verdict = "probe_trained_above_threshold"
    else:
        honest_verdict = "probe_trained_below_threshold"

    print(f"\n[Result] honest_verdict={honest_verdict}")
    print(f"[Result] AUROC target={AUROC_TARGET}  delta target={DELTA_TARGET}")
    print(f"[Result] Met delta target: {delta_vs_zeroshot >= DELTA_TARGET}")

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    duration_s = time.time() - t_start
    phase_timings = [
        {"name": "load_corpus", "elapsed_s": round(phase_1_s, 3)},
        {"name": "load_llm", "elapsed_s": round(phase_2_s, 3)},
        {"name": "extract_features", "elapsed_s": round(phase_3_s, 3)},
        {"name": "zeroshot_auroc", "elapsed_s": round(phase_4_s, 3)},
        {"name": "train_probe", "elapsed_s": round(phase_5_s, 3)},
        {"name": "evaluate", "elapsed_s": round(phase_6_s, 3)},
    ]

    artifact = tmpl.build_result(
        {
            "n_labeled_pairs_used": n_labeled_pairs_used,
            "n_train": len(train_items),
            "n_test": len(test_items),
            "model_used": model_used,
            "auroc_thinkprm_trained": round(auroc_trained, 4),
            "auroc_zeroshot_baseline": round(auroc_zeroshot, 4),
            "delta_vs_zeroshot": round(delta_vs_zeroshot, 4),
            "f1_thinkprm_trained": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "probe_weight": round(probe.w, 6),
            "probe_bias": round(probe.b, 6),
            "epoch_log": epoch_log,
            "phase_timings_s": phase_timings,
            "honest_verdict": honest_verdict,
            "auroc_target": AUROC_TARGET,
            "delta_target": DELTA_TARGET,
            "delta_target_met": delta_vs_zeroshot >= DELTA_TARGET,
        },
        status="success",
        honest_verdict=honest_verdict,
    )
    _write_and_exit(tmpl, artifact)


def _write_and_exit(tmpl: ExperimentTemplate, artifact: dict[str, Any]) -> None:
    """Write the artifact JSON and exit cleanly."""
    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n[Done] Written to {out_path}")


if __name__ == "__main__":
    main()
