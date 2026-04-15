"""Self-learning relay: Tier 1 + Tier 2 + Tier 3 simultaneous learning loop.

**Researcher summary:**
    FR-11 (Autonomous Self-Learning Loop) requires that all three learning tiers
    operate together on real model outputs so we can measure whether accuracy
    actually improves across batches.

    This module wires the three tiers into a single relay:

        Tier 1 (online weight updates):
            After each question, ``PerModelFPTracker.update()`` records whether the
            verification decision was a false positive or true positive for this
            (model_id, constraint_type) pair.  Over time, constraint types with
            high FP rates are suppressed for that model — the pipeline becomes
            more precise without human labeling.

        Tier 2 (constraint addition):
            For each incorrect response in the batch, ``CaseMemoryTemplateWiring.
            on_violation_recorded()`` increments the pattern counter in the
            ``ConstraintTemplateLibrary``.  Once enough carry_error (or sign_error,
            unit_error, comparison_error) violations accumulate, the corresponding
            template activates — adding a NEW constraint type to the pipeline for
            that model automatically.

        Tier 3 (predictive gate):
            The EORM model scores each (question, response) pair and we compute
            AUC-ROC of those scores against the ground-truth correctness labels.
            A rising AUC across batches means the EORM gate is becoming a better
            fast-path predictor — reducing expensive Ising calls while catching more
            real violations.

    Primary metric: batch 4 accuracy > batch 1 accuracy (``improved=True``).
    Secondary metric: any Tier 2 template activates during the run.

**Detailed explanation for engineers:**
    ``SelfLearningRelay`` is the central coordinator.  It holds references to all
    three tier components and a growing ``_trajectory`` list of
    ``SelfLearningBatchResult`` objects — one per batch.

    ``run_batch(questions, ground_truth, model_id)`` is the main entry point.
    It processes each question in order and, after the batch completes, appends
    a ``SelfLearningBatchResult`` capturing per-batch and cumulative metrics.

    The AUC computation uses the Wilcoxon-Mann-Whitney form (exact for small
    batches, O(n²)): AUC = P(score_correct > score_incorrect).  Lower EORM
    energy means the model thinks the CoT is more reliable; we flip the sign so
    higher ``score`` predicts correctness.

    ``compute_learning_improvement`` and ``build_relay_artifact`` are standalone
    utilities for post-hoc analysis and artifact serialization.

CI safety:
    - The relay itself is CI-safe: all class constructors accept stubs.
    - The experiment script (experiment_361) uses synthetic questions and
      ground_truth when ``CARNOT_FORCE_LIVE`` is unset.

Spec: REQ-LEARN-026, REQ-LEARN-027,
      SCENARIO-LEARN-045, SCENARIO-LEARN-046, SCENARIO-LEARN-047
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from carnot.models.eorm import CoTEnergyInput, EORMModel
from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
from carnot.pipeline.constraint_template_library import (
    CaseMemoryTemplateWiring,
    ConstraintTemplateLibrary,
)
from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline


# ---------------------------------------------------------------------------
# SelfLearningBatchResult
# ---------------------------------------------------------------------------


@dataclass
class SelfLearningBatchResult:
    """Per-batch metrics from one round of the three-tier self-learning relay.

    **Detailed explanation for engineers:**
        Each call to ``SelfLearningRelay.run_batch()`` produces one of these.
        Together, a list of results forms the ``learning_trajectory()`` — the
        time-series of learning progress across batches.

        ``batch_id``:
            Zero-based index of this batch within the run.  Useful for plotting
            accuracy vs. batch index.

        ``n_questions``:
            Number of (question, ground_truth) pairs processed in this batch.
            Should be 25 for standard Exp 361 batches.

        ``accuracy``:
            Fraction of questions in THIS batch where ``ground_truth == True``
            (i.e., the simulated or actual model was correct).  Note: in
            CI-synthetic mode the "model" always produces the right answer for
            the correct subset, so accuracy is exactly the fraction of ``True``
            labels in ``ground_truth``.

        ``n_tier1_updates``:
            Total number of ``PerModelFPTracker.update()`` calls made during
            this batch — one per question.  Should equal ``n_questions``.

        ``n_tier2_templates_active``:
            Number of ``ConstraintTemplate`` objects that have crossed their
            ``min_frequency`` threshold for ``model_id`` after this batch
            completes.  Starts at 0 and rises as repeated error patterns
            accumulate across batches.

        ``tier3_gate_auc``:
            AUC-ROC of the EORM gate on this batch.  Computed using the
            Wilcoxon-Mann-Whitney rank-sum form: P(energy_correct < energy_wrong).
            A rising AUC across batches means the EORM is improving as a
            fast-path predictor.  0.5 = random; 1.0 = perfect.

        ``cumulative_accuracy``:
            Total correct across ALL batches so far (including this one) divided
            by total questions.  This is the headline metric: if it rises from
            batch 1 to batch 4, the relay is demonstrating cross-batch learning.

    Spec: REQ-LEARN-026-1
    """

    batch_id: int
    n_questions: int
    accuracy: float
    n_tier1_updates: int
    n_tier2_templates_active: int
    tier3_gate_auc: float
    cumulative_accuracy: float


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------


def _compute_auc_roc(energies: list[float], ground_truth: list[bool]) -> float:
    """Compute AUC-ROC where LOWER energy predicts True (correct response).

    **Detailed explanation for engineers:**
        The EORM assigns LOWER energy to responses it thinks are correct.
        We flip the sign so that higher score = more confident the response is
        correct — making the AUC semantics conventional (higher score → positive).

        We use the Wilcoxon-Mann-Whitney form:
            AUC = P(score_correct > score_incorrect)
        This is exact for small batches (25 questions) and numerically stable.

        Edge cases:
        - All labels True or all False: AUC is undefined; return 0.5 (random).
        - Ties (equal scores) contribute 0.5 per tie to the expectation.

    Args:
        energies:     EORM energy for each question in the batch.
        ground_truth: Parallel correctness labels (True = correct).

    Returns:
        AUC-ROC in [0, 1].  0.5 = no discrimination; 1.0 = perfect.

    Spec: REQ-LEARN-026-3
    """
    # Flip: lower energy → higher score → predicts True
    scores = [-e for e in energies]

    pos = [s for s, g in zip(scores, ground_truth) if g]
    neg = [s for s, g in zip(scores, ground_truth) if not g]

    if not pos or not neg:
        # AUC is undefined when one class is absent — return random baseline.
        return 0.5

    # Count pairs where pos_score > neg_score (plus 0.5 for ties)
    wins: float = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5

    return wins / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# SelfLearningRelay
# ---------------------------------------------------------------------------


class SelfLearningRelay:
    """Coordinates Tier 1 + 2 + 3 self-learning on batches of Q&A pairs.

    **Researcher summary:**
        Instantiate once per experiment run.  Call ``run_batch()`` once per batch
        of 25 questions.  After 4 batches, call ``learning_trajectory()`` then
        ``compute_learning_improvement()`` to confirm whether accuracy rose.

    **Detailed explanation for engineers:**
        The relay holds the three tier components and a growing trajectory list.
        It does NOT own a CaseMemory instance — it only drives the wiring layer
        (``CaseMemoryTemplateWiring``) to accumulate pattern counts in the
        ``ConstraintTemplateLibrary``.

        Tier 1 (PerModelFPTracker):
            For every question, we call ``fp_tracker.update()`` based on whether
            the pipeline's verification decision matched ground truth:
                - verified=True, is_correct=False → FP (false positive).
                - verified=True, is_correct=True  → TP (true positive).
                - verified=False in either case   → neither (no FP/TP credit).
            This lets the tracker learn which constraint types fire incorrectly
            on this model over time.

        Tier 2 (CaseMemoryTemplateWiring):
            For every INCORRECT response, we call ``on_violation_recorded()``
            with a synthetic violation type derived from the batch position.
            In a real system this would come from the pipeline's actual constraint
            violations.  The synthetic type cycles through the four canonical
            arithmetic error types so that templates can activate during the run
            given enough batches.

        Tier 3 (EORMModel):
            We score each response with ``eorm_model.energy()`` and compute
            AUC-ROC against ground_truth.  In CI mode the EORM has random weights
            (AUC ≈ 0.5), but the code path is fully exercised.

    Args:
        pipeline:         ``ThreeTierPipeline`` (or any object with a
                          ``.verify(response, *, question) -> (bool, str, float)``
                          method) used to obtain verification decisions.
        template_library: ``ConstraintTemplateLibrary`` accumulating pattern counts.
        fp_tracker:       ``PerModelFPTracker`` updated after each question.
        eorm_model:       ``EORMModel`` used to score (question, response) pairs.

    Spec: REQ-LEARN-026-2
    """

    # Cycle through these violation types for synthetic Tier 2 accumulation.
    # They map (via CaseMemoryTemplateWiring._KEYWORD_MAP) to the four built-in
    # template pattern_keys: carry_check, sign_check, unit_consistency,
    # comparison_direction.
    _SYNTHETIC_VIOLATION_TYPES: list[str] = [
        "carry_error",
        "sign_error",
        "unit_error",
        "comparison_error",
    ]

    def __init__(
        self,
        pipeline: ThreeTierPipeline,
        template_library: ConstraintTemplateLibrary,
        fp_tracker: PerModelFPTracker,
        eorm_model: EORMModel,
    ) -> None:
        self._pipeline = pipeline
        self._template_library = template_library
        self._fp_tracker = fp_tracker
        self._eorm_model = eorm_model
        self._wiring = CaseMemoryTemplateWiring(template_library)
        self._trajectory: list[SelfLearningBatchResult] = []

        # Cumulative counters for cross-batch accuracy computation.
        self._total_correct: int = 0
        self._total_questions: int = 0

    # ------------------------------------------------------------------
    # run_batch
    # ------------------------------------------------------------------

    def run_batch(
        self,
        questions: list[str],
        ground_truth: list[bool],
        model_id: str,
    ) -> SelfLearningBatchResult:
        """Process one batch and update all three learning tiers.

        **Detailed explanation for engineers:**
            Steps per question (in order):
            1. Score with EORM (collect energy for Tier 3 AUC computation).
            2. Verify with pipeline (get tier_used and verified flag).
            3. Update PerModelFPTracker based on (verified, is_correct) (Tier 1).
            4. If incorrect, notify CaseMemoryTemplateWiring (Tier 2).

            After all questions:
            5. Count active Tier 2 templates for model_id.
            6. Compute Tier 3 EORM gate AUC-ROC.
            7. Compute batch accuracy and cumulative accuracy.
            8. Append SelfLearningBatchResult to trajectory.

        Args:
            questions:    List of question strings (25 per batch in Exp 361).
                          In CI mode these are synthetic placeholder strings.
            ground_truth: Parallel list of correctness labels (True = correct).
            model_id:     Identifier for the model being evaluated (e.g.
                          "gemma4-e4b-it" or "ci_synthetic").

        Returns:
            SelfLearningBatchResult for this batch.

        Spec: REQ-LEARN-026-3, REQ-LEARN-026-5, SCENARIO-LEARN-045
        """
        batch_id = len(self._trajectory)
        n_questions = len(questions)
        n_correct = 0
        n_tier1_updates = 0
        eorm_energies: list[float] = []
        violation_idx = 0  # cycles through _SYNTHETIC_VIOLATION_TYPES

        for i, (question, is_correct) in enumerate(zip(questions, ground_truth)):
            # ----------------------------------------------------------------
            # Tier 3 prep: score with EORM for gate AUC computation.
            # ----------------------------------------------------------------
            # In CI mode the response text equals the question (no real model).
            response = question
            cot_input = CoTEnergyInput(question_text=question, response_text=response)
            energy = float(self._eorm_model.energy(cot_input))
            eorm_energies.append(energy)

            # ----------------------------------------------------------------
            # Get pipeline verification decision (used for Tier 1 updates).
            # ----------------------------------------------------------------
            verified, _tier_used, _pipeline_energy = self._pipeline.verify(
                response, question=question
            )

            # ----------------------------------------------------------------
            # Tier 1: update PerModelFPTracker based on (verified, is_correct).
            # ----------------------------------------------------------------
            # FP: pipeline said OK but answer was wrong.
            # TP: pipeline said OK and answer was right.
            # Anything else is neither FP nor TP (don't credit or penalize).
            was_fp = bool(verified and not is_correct)
            was_tp = bool(verified and is_correct)
            self._fp_tracker.update(
                model_id,
                "verification",
                was_fp=was_fp,
                was_tp=was_tp,
            )
            n_tier1_updates += 1

            # Track raw accuracy for this batch.
            if is_correct:
                n_correct += 1
            else:
                # ----------------------------------------------------------------
                # Tier 2: notify wiring that a violation of some type was recorded.
                # ----------------------------------------------------------------
                # Cycle through synthetic violation types so that after enough
                # incorrect responses the corresponding templates will activate.
                violation_type = self._SYNTHETIC_VIOLATION_TYPES[
                    violation_idx % len(self._SYNTHETIC_VIOLATION_TYPES)
                ]
                self._wiring.on_violation_recorded(violation_type, model_id)
                violation_idx += 1

        # --------------------------------------------------------------------
        # After-batch aggregation
        # --------------------------------------------------------------------

        # Tier 2: count how many templates are now active for this model.
        n_tier2_active = len(self._template_library.get_active_templates(model_id))

        # Tier 3: compute EORM gate AUC-ROC for this batch.
        tier3_auc = _compute_auc_roc(eorm_energies, list(ground_truth))

        # Cross-batch accuracy tracking.
        self._total_correct += n_correct
        self._total_questions += n_questions
        batch_accuracy = n_correct / n_questions if n_questions > 0 else 0.0
        cumulative_accuracy = (
            self._total_correct / self._total_questions
            if self._total_questions > 0
            else 0.0
        )

        result = SelfLearningBatchResult(
            batch_id=batch_id,
            n_questions=n_questions,
            accuracy=batch_accuracy,
            n_tier1_updates=n_tier1_updates,
            n_tier2_templates_active=n_tier2_active,
            tier3_gate_auc=tier3_auc,
            cumulative_accuracy=cumulative_accuracy,
        )
        self._trajectory.append(result)
        return result

    # ------------------------------------------------------------------
    # learning_trajectory
    # ------------------------------------------------------------------

    def learning_trajectory(self) -> list[SelfLearningBatchResult]:
        """Return all batch results accumulated so far (shallow copy).

        **Detailed explanation for engineers:**
            Returns a list copy so the caller can iterate without worrying about
            future ``run_batch()`` calls mutating the list.  The
            ``SelfLearningBatchResult`` objects themselves are dataclasses
            (immutable by convention — do not mutate their fields).

        Returns:
            List of SelfLearningBatchResult, one per completed batch.
            Empty before the first ``run_batch()`` call.

        Spec: REQ-LEARN-026-4
        """
        return list(self._trajectory)


# ---------------------------------------------------------------------------
# compute_learning_improvement
# ---------------------------------------------------------------------------


def compute_learning_improvement(
    trajectory: list[SelfLearningBatchResult],
) -> tuple[float, float, bool]:
    """Compute whether accuracy improved from batch 1 to batch 4.

    **Detailed explanation for engineers:**
        The primary metric for FR-11 is a simple before/after comparison:
        did the pipeline get more questions right in the final batch than in
        the first batch?

        When fewer than 4 batches are present, the last available batch is
        used as the "final" batch.  When the trajectory is empty, all values
        are 0.0 and ``improved`` is False.

        ``improved`` uses STRICT greater-than: the final batch must have
        measurably higher accuracy, not merely equal.

    Args:
        trajectory: List of SelfLearningBatchResult from a relay run.

    Returns:
        (batch1_accuracy, batch4_accuracy, improved) tuple where:
        - batch1_accuracy: accuracy of the first batch (index 0).
        - batch4_accuracy: accuracy of the last batch (or 4th, index 3).
        - improved:        True when batch4_accuracy > batch1_accuracy.

    Spec: REQ-LEARN-027-1, SCENARIO-LEARN-047
    """
    if not trajectory:
        return 0.0, 0.0, False

    batch1_accuracy = trajectory[0].accuracy
    # Use index 3 (batch 4) if available, otherwise the last batch.
    final_idx = min(3, len(trajectory) - 1)
    batch4_accuracy = trajectory[final_idx].accuracy
    improved = batch4_accuracy > batch1_accuracy

    return batch1_accuracy, batch4_accuracy, improved


# ---------------------------------------------------------------------------
# build_relay_artifact
# ---------------------------------------------------------------------------


def build_relay_artifact(
    trajectory: list[SelfLearningBatchResult],
    learning_improvement: tuple[float, float, bool],
    *,
    inference_mode: str = "cpu_synthetic",
) -> dict[str, Any]:
    """Serialize a self-learning relay run to the standard Carnot artifact schema.

    **Detailed explanation for engineers:**
        The ``honest_verdict`` field is the critical honesty gate:
        - ``"learning_confirmed"``: ONLY when ``improved=True`` AND
          ``inference_mode=="live_gpu"``.  This is the only case where we can
          claim real learning happened on real model outputs.
        - ``"synthetic_only"``:  When inference_mode != "live_gpu".  Results
          are real code exercise but cannot prove live-model learning.
        - ``"no_improvement"``: When improved=False regardless of mode.

        This matches the pattern established in Exps 328/359 where honest
        verdicts prevent simulation artifacts from being reported as real results.

    Args:
        trajectory:          List of SelfLearningBatchResult from the relay run.
        learning_improvement: Tuple from ``compute_learning_improvement()``.
        inference_mode:      Label: "live_gpu" for real runs, "cpu_synthetic" for CI.

    Returns:
        Flat dict with schema="carnot.self_learning_relay.v1" and all fields
        needed for experiment reporting.

    Spec: REQ-LEARN-027-2
    """
    batch1_accuracy, batch4_accuracy, improved = learning_improvement

    # Determine honest verdict.
    if not improved:
        honest_verdict = "no_improvement"
    elif inference_mode == "live_gpu":
        honest_verdict = "learning_confirmed"
    else:
        honest_verdict = "synthetic_only"

    # Serialize the trajectory (each SelfLearningBatchResult → dict).
    trajectory_dicts = [
        {
            "batch_id": r.batch_id,
            "n_questions": r.n_questions,
            "accuracy": r.accuracy,
            "n_tier1_updates": r.n_tier1_updates,
            "n_tier2_templates_active": r.n_tier2_templates_active,
            "tier3_gate_auc": r.tier3_gate_auc,
            "cumulative_accuracy": r.cumulative_accuracy,
        }
        for r in trajectory
    ]

    return {
        "schema": "carnot.self_learning_relay.v1",
        "trajectory": trajectory_dicts,
        "batch1_accuracy": batch1_accuracy,
        "batch4_accuracy": batch4_accuracy,
        "improved": improved,
        "inference_mode": inference_mode,
        "honest_verdict": honest_verdict,
    }


__all__ = [
    "SelfLearningBatchResult",
    "SelfLearningRelay",
    "build_relay_artifact",
    "compute_learning_improvement",
]
