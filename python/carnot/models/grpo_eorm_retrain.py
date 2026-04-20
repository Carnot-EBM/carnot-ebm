"""GRPO-style contrastive EORM retrain from live benchmark binary verdicts.

**Researcher summary:**
    GRPO (arXiv 2503.06639) shows that verifiable binary rewards (correct/incorrect)
    naturally form contrastive pairs: for each question where the model is right, the
    correct response is the positive and any wrong response is the negative — and vice
    versa.  No separate labeling step is needed.

    This module applies that insight to retrain EORM (the energy-based CoT verifier)
    using live benchmark data from Exp 538.  We extract (correct_response,
    incorrect_response) pairs directly from the benchmark's binary verdicts and
    optimize the contrastive loss to maximize the energy gap between incorrect and
    correct responses.

**Why this matters:**
    The NUP Probe v4 (Exp 523) validated that energy-gap contrastive training
    dramatically outperforms plain BCE (AUC=1.0 vs 0.40 on held-out pairs).
    Applying the same training signal to EORM with *live* benchmark pairs (rather
    than synthetic ones) should improve EORM's discrimination on the actual
    failure modes produced by the target model (Qwen3.5-0.8B on GSM8K).

**Contrastive loss formula:**
    L = mean(max(0, margin - (E(incorrect) - E(correct))))

    This is zero when every incorrect response already has energy at least `margin`
    above the corresponding correct response.  Otherwise, gradients push E(correct)
    down and E(incorrect) up.

**AUC computation:**
    Given N pairs, the AUC is the fraction of pairs where E(incorrect) > E(correct)
    (i.e., the model correctly ranks the correct response lower in energy).
    This is equivalent to the area under the ROC curve for a binary classifier that
    uses the energy gap as its score.

Spec: REQ-LEARN-051, REQ-LEARN-052,
      SCENARIO-LEARN-080, SCENARIO-LEARN-081, SCENARIO-LEARN-082
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from carnot.models.eorm import EORMModel, CoTEnergyInput, _make_token_sequence, _SEP_ID, _forward


# ---------------------------------------------------------------------------
# GRPOContrastivePair
# ---------------------------------------------------------------------------


@dataclass
class GRPOContrastivePair:
    """A single GRPO-derived (correct, incorrect) response pair for one question.

    **For engineers:**
        GRPO creates these pairs automatically from benchmark binary verdicts:
        if the pipeline answered question Q correctly and the baseline did not,
        then pipeline_response is correct and baseline_response is incorrect.
        No human annotation is required — the benchmark's own pass/fail verdict
        IS the contrastive signal.

    Attributes:
        question_id: Identifier for the benchmark question.
        correct_response: The response that received a correct verdict.
        incorrect_response: The response that received an incorrect verdict.

    Spec: REQ-LEARN-051
    """

    question_id: str
    correct_response: str
    incorrect_response: str


# ---------------------------------------------------------------------------
# build_grpo_pairs_from_benchmark
# ---------------------------------------------------------------------------


def build_grpo_pairs_from_benchmark(
    benchmark_result_path: str | Path,
) -> list[GRPOContrastivePair]:
    """Extract GRPO contrastive pairs from a benchmark JSON result file.

    **For engineers:**
        Looks for per-question entries that have *both* a baseline verdict and a
        pipeline verdict so that the two responses can be paired contrastively.
        The benchmark format must contain a list under a recognized key
        (``per_question_results``, ``responses``, or ``questions``) where each
        entry has:
            - ``baseline_correct`` (bool): Was the baseline model correct?
            - ``pipeline_correct`` (bool): Was the pipeline correct?
            - ``baseline_response`` (str): The baseline's response text.
            - ``pipeline_response`` (str): The pipeline's response text.
            - ``question_id`` (str, optional): Question identifier.

        Pairing logic (GRPO insight — arXiv 2503.06639):
        - baseline_correct=False AND pipeline_correct=True →
            correct=pipeline_response, incorrect=baseline_response
          (pipeline fixed a question the baseline got wrong)
        - baseline_correct=True AND pipeline_correct=False →
            correct=baseline_response, incorrect=pipeline_response
          (pipeline broke a question the baseline got right — still a valid pair)

        Questions where both verdicts agree (both right or both wrong) produce no
        useful contrastive signal and are skipped.

        If the file is missing, malformed, or contains no eligible entries, an
        empty list is returned (never raises).

    Args:
        benchmark_result_path: Path to the experiment JSON (Exp 538 format or similar).

    Returns:
        List of GRPOContrastivePair objects.  May be empty.

    Spec: REQ-LEARN-051, SCENARIO-LEARN-080
    """
    pairs: list[GRPOContrastivePair] = []

    try:
        path = Path(benchmark_result_path)
        if not path.exists():
            return pairs
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return pairs

    # Look for a list of per-question entries under recognized key names
    entries: list[Any] = []
    for key in ("per_question_results", "responses", "questions", "results"):
        candidate = data.get(key)
        if isinstance(candidate, list):
            entries = candidate
            break

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        baseline_correct = entry.get("baseline_correct")
        pipeline_correct = entry.get("pipeline_correct")

        # Both verdicts must be present and boolean to form a contrastive pair
        if not isinstance(baseline_correct, bool) or not isinstance(pipeline_correct, bool):
            continue

        # Skip concordant pairs — no contrastive signal when both agree
        if baseline_correct == pipeline_correct:
            continue

        baseline_resp = str(entry.get("baseline_response") or "")
        pipeline_resp = str(entry.get("pipeline_response") or "")
        question_id = str(entry.get("question_id") or entry.get("problem_id") or "unknown")

        if not baseline_resp or not pipeline_resp:
            continue

        if not baseline_correct and pipeline_correct:
            # Pipeline fixed the question: pipeline is correct, baseline is wrong
            pairs.append(GRPOContrastivePair(
                question_id=question_id,
                correct_response=pipeline_resp,
                incorrect_response=baseline_resp,
            ))
        else:
            # Pipeline broke the question: baseline is correct, pipeline is wrong
            pairs.append(GRPOContrastivePair(
                question_id=question_id,
                correct_response=baseline_resp,
                incorrect_response=pipeline_resp,
            ))

    return pairs


# ---------------------------------------------------------------------------
# build_grpo_pairs_from_fover
# ---------------------------------------------------------------------------


def build_grpo_pairs_from_fover(fover_path: str | Path) -> list[GRPOContrastivePair]:
    """Build GRPO contrastive pairs from FOVER step-level annotations.

    **For engineers:**
        The FOVER file (fover_labeled_steps_live.json, Exp 442) contains individual
        reasoning steps labeled as "correct" or "incorrect" for specific questions.
        We group steps by question_id and pair the first correct step with the first
        incorrect step for each question that has both.

        This is used as a fallback when the benchmark result provides fewer than 5
        contrastive pairs.  FOVER pairs are lower quality than live benchmark pairs
        (they are step-level, not full-response-level), but they are real labels
        from real model outputs.

    Args:
        fover_path: Path to fover_labeled_steps_live.json.

    Returns:
        List of GRPOContrastivePair objects built from FOVER annotations.

    Spec: REQ-LEARN-051
    """
    pairs: list[GRPOContrastivePair] = []

    try:
        path = Path(fover_path)
        if not path.exists():
            return pairs
        with open(path) as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            return pairs
    except (OSError, json.JSONDecodeError):
        return pairs

    # Group by question_id, collecting correct and incorrect step texts
    from collections import defaultdict
    correct_steps: dict[str, list[str]] = defaultdict(list)
    incorrect_steps: dict[str, list[str]] = defaultdict(list)

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        qid = str(entry.get("question_id") or "unknown")
        text = str(entry.get("step_text") or "")
        label = str(entry.get("label") or "")
        if not text:
            continue
        if label == "correct":
            correct_steps[qid].append(text)
        elif label == "incorrect":
            incorrect_steps[qid].append(text)

    # Pair the first correct and first incorrect step for each question that has both
    all_qids = set(correct_steps) & set(incorrect_steps)
    for qid in sorted(all_qids):
        pairs.append(GRPOContrastivePair(
            question_id=qid,
            correct_response=correct_steps[qid][0],
            incorrect_response=incorrect_steps[qid][0],
        ))

    return pairs


# ---------------------------------------------------------------------------
# _compute_auc
# ---------------------------------------------------------------------------


def _compute_auc(model: EORMModel, pairs: list[GRPOContrastivePair]) -> float:
    """Compute AUC on a list of GRPOContrastivePairs.

    **For engineers:**
        AUC here is the fraction of pairs where the model assigns lower energy to
        the correct response than to the incorrect response.  This is equivalent to
        the area under the ROC curve for a binary energy-gap classifier.

        A random model scores 0.5.  A perfect model scores 1.0.

        Uses the same question text for both responses in a pair (question_id as
        a stand-in, since we may not have the original question text).

    Args:
        model: EORMModel to evaluate.
        pairs: List of GRPOContrastivePair objects.

    Returns:
        AUC value in [0.0, 1.0].  Returns 0.5 if pairs is empty (random baseline).

    Spec: REQ-LEARN-051
    """
    if not pairs:
        return 0.5

    n_correct = 0
    for pair in pairs:
        # Use question_id as a proxy for the question text (consistent across calls)
        e_correct = model.energy(CoTEnergyInput(
            question_text=pair.question_id,
            response_text=pair.correct_response,
        ))
        e_incorrect = model.energy(CoTEnergyInput(
            question_text=pair.question_id,
            response_text=pair.incorrect_response,
        ))
        if e_incorrect > e_correct:
            n_correct += 1

    return n_correct / len(pairs)


# ---------------------------------------------------------------------------
# train_eorm_grpo
# ---------------------------------------------------------------------------


def train_eorm_grpo(
    eorm_model: EORMModel,
    pairs: list[GRPOContrastivePair],
    margin: float = 1.0,
    epochs: int = 50,
    lr: float = 1e-4,
) -> tuple[float, float, float]:
    """Retrain an EORMModel using GRPO-style contrastive loss on live pairs.

    **For engineers:**
        Contrastive loss formula (REQ-LEARN-052):
            L = mean(max(0, margin - (E(incorrect) - E(correct))))

        For each pair, we want E(incorrect) > E(correct) + margin.
        When this holds, the loss is zero and no gradient flows.
        When the gap is too small (or reversed), gradients push:
            - E(correct) down
            - E(incorrect) up

        Training loop:
        1. Compute before_auc on all pairs.
        2. For each epoch, iterate over all pairs and apply one SGD update per pair.
        3. Compute after_auc on all pairs.
        4. Return (mean_loss_last_epoch, before_auc, after_auc).

        Uses plain SGD (no momentum) via JAX value_and_grad.  This is intentionally
        simple — the goal is a quick retrain signal, not a production training run.

    Args:
        eorm_model: The EORMModel to retrain in place.
        pairs: List of GRPOContrastivePair objects.
        margin: Hinge margin (default 1.0).  Loss is zero when energy gap >= margin.
        epochs: Number of full passes over all pairs (default 50).
        lr: Learning rate for SGD (default 1e-4).

    Returns:
        Tuple of (training_loss, before_auc, after_auc) where training_loss is the
        mean loss over the final epoch, and AUC values are in [0.0, 1.0].

    Spec: REQ-LEARN-051, REQ-LEARN-052, SCENARIO-LEARN-081
    """
    if not pairs:
        return 0.0, 0.5, 0.5

    before_auc = _compute_auc(eorm_model, pairs)

    n_heads = eorm_model.n_heads
    max_seq_len = eorm_model.max_seq_len
    vocab_size = eorm_model.vocab_size

    last_epoch_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for pair in pairs:
            # Tokenize outside the differentiable function to avoid retracing
            correct_ids = (
                _make_token_sequence(
                    pair.question_id, pair.correct_response, max_seq_len, vocab_size
                )
                or [_SEP_ID]
            )
            incorrect_ids = (
                _make_token_sequence(
                    pair.question_id, pair.incorrect_response, max_seq_len, vocab_size
                )
                or [_SEP_ID]
            )

            def loss_fn(params: dict) -> jax.Array:
                """GRPO contrastive loss: max(0, margin - (E_wrong - E_right))."""
                e_correct = _forward(params, correct_ids, n_heads)
                e_incorrect = _forward(params, incorrect_ids, n_heads)
                # relu is the differentiable max(0, x)
                return jax.nn.relu(margin - (e_incorrect - e_correct))

            loss_val, grads = jax.value_and_grad(loss_fn)(eorm_model.params)

            # SGD update
            eorm_model.params = jax.tree_util.tree_map(
                lambda p, g: p - lr * g,
                eorm_model.params,
                grads,
            )

            epoch_loss += float(loss_val)

        last_epoch_loss = epoch_loss / len(pairs)

    after_auc = _compute_auc(eorm_model, pairs)

    return last_epoch_loss, before_auc, after_auc


# ---------------------------------------------------------------------------
# GRPOEORMRetrainResult
# ---------------------------------------------------------------------------


@dataclass
class GRPOEORMRetrainResult:
    """Summary of one GRPO-style EORM retrain run.

    **For engineers:**
        Carries the key metrics from the Exp 540 retrain.  The ``honest_verdict``
        field is the machine-readable outcome declaration:

        - ``'grpo_improved'``: Real GRPO pairs were used AND auc_improvement > 0.05.
          This is the only outcome that constitutes genuine evidence that GRPO
          contrastive retraining improved EORM discrimination.

        - ``'no_improvement'``: Real GRPO pairs were used but auc_improvement <= 0.05.
          Logged honestly — may indicate insufficient pairs, LR tuning needed, etc.

        - ``'synthetic_fallback'``: Fewer than 5 benchmark pairs were found; the
          retrain used FOVER fallback pairs.  AUC numbers are reported but clearly
          labeled as coming from synthetic/fallback data.

    Attributes:
        n_pairs: Number of contrastive pairs used in training.
        before_auc: AUC before retraining (fraction of pairs correctly ranked).
        after_auc: AUC after retraining.
        auc_improvement: after_auc - before_auc (signed float).
        honest_verdict: Machine-readable outcome string.

    Spec: REQ-LEARN-051, SCENARIO-LEARN-082
    """

    n_pairs: int
    before_auc: float
    after_auc: float
    auc_improvement: float
    honest_verdict: str


def make_grpo_result(
    n_pairs: int,
    before_auc: float,
    after_auc: float,
    is_synthetic_fallback: bool,
) -> GRPOEORMRetrainResult:
    """Construct a GRPOEORMRetrainResult with the correct honest_verdict.

    **For engineers:**
        Centralizes the verdict logic so it is testable independently.
        The threshold of 0.05 for 'grpo_improved' matches the NUP Probe v4
        result where the energy-gap contrastive approach showed a 0.60 AUC
        improvement — 0.05 is a conservative lower bound for a meaningful gain.

    Args:
        n_pairs: Number of pairs used.
        before_auc: AUC before training.
        after_auc: AUC after training.
        is_synthetic_fallback: True when FOVER/synthetic pairs were used instead
            of live benchmark pairs.

    Returns:
        GRPOEORMRetrainResult with honest_verdict set appropriately.

    Spec: SCENARIO-LEARN-082
    """
    auc_improvement = after_auc - before_auc

    if is_synthetic_fallback:
        verdict = "synthetic_fallback"
    elif auc_improvement > 0.05:
        verdict = "grpo_improved"
    else:
        verdict = "no_improvement"

    return GRPOEORMRetrainResult(
        n_pairs=n_pairs,
        before_auc=round(float(before_auc), 6),
        after_auc=round(float(after_auc), 6),
        auc_improvement=round(float(auc_improvement), 6),
        honest_verdict=verdict,
    )
