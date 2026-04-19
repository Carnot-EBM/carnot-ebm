"""JEPA Curriculum Trainer — three-stage curriculum to recover from RETRO-040 AUC regression.

**Why three-stage curriculum (easy-first learning):**
    JEPA AUC collapsed from 0.400 to 0.281 (below random chance) after Exp 477's quality gate
    removed 73% of training pairs.  The surviving corpus was dominated by high-confidence
    CORRECT steps, so the model learned to predict everything as correct (majority-class
    collapse).  AUC < 0.5 means the model learned an INVERTED signal — actively wrong,
    not just noisy.

    The fix is curriculum learning (Bengio et al., 2009; arXiv 2509.14252 LLM-JEPA):
    expose the model to training examples in increasing difficulty order so it first anchors
    on high-quality signal, then gradually learns the full distribution.  This prevents
    the information loss that caused the 0.281 regression.

    Three stages:
    1. Stage 1 — High-confidence only (label_confidence >= 0.85): establishes a stable
       energy landscape baseline.  The model first sees only examples the annotator was
       very confident about, anchoring the energy function on reliable ground truth.
    2. Stage 2 — All pairs unfiltered: recovers the information thrown away by the quality
       gate.  After Stage 1 anchors the model, Stage 2 exposes it to the full distribution
       without majority-class collapse, because the Stage 1 anchor prevents the model from
       forgetting how to discriminate.
    3. Stage 3 — EBM-guided synthetic augmentation to n_total >= 200: the real corpus is
       small (57 pairs).  Synthetic pairs from the Ising energy landscape add coverage for
       the failure modes the pipeline actually produces, guided by the energy function as
       ground truth.

**Why high_conf_threshold=0.85 (not 0.70 like Exp 477):**
    Exp 477 used min_confidence=0.70, which still let through ~27% of pairs — enough to
    have noisy Stage 1 anchoring.  0.85 is more conservative: it guarantees Stage 1 only
    sees examples the annotator was clearly confident about, not just "not obviously bad".
    Smaller Stage 1 corpus is intentional; quality matters more than size at anchoring time.

**Why Stage 2 uses ALL pairs (no filter):**
    The root cause of the 0.281 regression was INFORMATION LOSS from filtering.  If Stage 2
    also filtered, it would re-introduce the same imbalance that caused majority-class
    collapse.  The Stage 1 anchor makes it safe to show the full noisy distribution because
    the model already knows what high-quality correct vs incorrect looks like.

**References:**
    - Bengio et al. (2009) "Curriculum Learning" ICML
    - arXiv 2509.14252 "LLM-JEPA: Curriculum-Ordered Embeddings Improve Downstream Transfer"
    - RETRO-040 (milestone .35): root cause analysis of AUC 0.667→0.400→0.281 regression

Spec: REQ-LEARN-040, REQ-LEARN-041, REQ-LEARN-042,
      SCENARIO-LEARN-068, SCENARIO-LEARN-069, SCENARIO-LEARN-070
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import jax.random as jrandom

from carnot.models.eorm import CoTEnergyInput, EORMModel, EORMTrainer
from carnot.models.jepa_curriculum_diagnostic import _compute_auc, _pairs_to_eorm_triples
from carnot.models.jepa_retrain_v2 import JEPAQualityAugmentor, _estimate_label_confidence

# ---------------------------------------------------------------------------
# Tiny EORM config — same as diagnostic for CPU-speed curriculum training.
# Production quality is not the goal here; AUC improvement above 0.600 is.
# ---------------------------------------------------------------------------
_EMBED_DIM = 32
_N_HEADS = 2
_N_LAYERS = 2
_MAX_SEQ_LEN = 128
_VOCAB_SIZE = 512


# ---------------------------------------------------------------------------
# CurriculumStageResult — result for one curriculum stage
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStageResult:
    """Result of one training stage in the curriculum pipeline.

    **For engineers:**
        Each of the three curriculum stages trains the EORM model on a specific
        subset/ordering of pairs.  After each stage, we measure AUC on the held-out
        20% to track whether the stage improved discriminability.

        ``auc_improved`` is only meaningful for stage >= 2 because Stage 1 establishes
        the baseline — there is no prior AUC to compare against (before_auc for Stage 1
        is the untrained model AUC, which varies by random seed).

    Attributes:
        stage: Stage number (1, 2, or 3).
        n_pairs: Number of training pairs used in this stage.
        n_epochs: Number of full passes through training pairs in this stage.
        auc_after: AUC on held-out 20% set after completing this stage.

    Spec: REQ-LEARN-040, REQ-LEARN-041, REQ-LEARN-042
    """

    stage: int
    n_pairs: int
    n_epochs: int
    auc_after: float
    auc_before: float = 0.5

    @property
    def auc_improved(self) -> bool:
        """True iff auc_after > auc_before for this stage.

        Only meaningful for stage >= 2.  Stage 1 sets the baseline so there is no
        "prior" to compare against in a meaningful way — Stage 1 auc_before is the
        randomly-initialized model AUC.
        """
        return self.auc_after > self.auc_before


# ---------------------------------------------------------------------------
# JEPACurriculumTrainer — three-stage curriculum trainer
# ---------------------------------------------------------------------------


class JEPACurriculumTrainer:
    """Three-stage curriculum trainer for JEPA recovery from majority-class collapse.

    **For engineers:**
        This class implements the curriculum fix for RETRO-040.  The three stages are:

        Stage 1: High-confidence only (label_confidence >= high_conf_threshold).
            Trains on only the most reliable labeled pairs.  Goal: anchor the EORM energy
            function on high-quality correct vs incorrect discrimination before exposing it
            to the full noisy distribution.

        Stage 2: All pairs unfiltered.
            Trains on ALL labeled pairs without any confidence gate.  Goal: recover the
            information lost by Exp 477's quality gate.  The Stage 1 anchor prevents
            majority-class collapse because the model already knows how to discriminate.

        Stage 3: EBM-guided synthetic augmentation to n_total >= 200.
            If the combined corpus (real + synthetic) is smaller than 200 pairs, adds
            EBM-guided synthetic pairs from the Ising energy landscape until the target
            is reached.  The Ising coupling matrix was trained on real pipeline data, so
            its energy landscape concentrates on actual failure modes.

    Args:
        n_stage1_epochs: Number of epochs for Stage 1 (high-confidence only). Default 100.
        n_stage2_epochs: Number of epochs for Stage 2 (all pairs). Default 100.
        n_stage3_epochs: Number of epochs for Stage 3 (with synthetic augmentation). Default 100.
        high_conf_threshold: Minimum label_confidence for Stage 1 pairs. Default 0.85.
            More conservative than Exp 477's 0.70 — guarantees Stage 1 corpus is
            genuinely high-quality, not just "not obviously bad".

    Spec: REQ-LEARN-040, REQ-LEARN-041, REQ-LEARN-042
    """

    def __init__(
        self,
        n_stage1_epochs: int = 100,
        n_stage2_epochs: int = 100,
        n_stage3_epochs: int = 100,
        high_conf_threshold: float = 0.85,
    ) -> None:
        self.n_stage1_epochs = n_stage1_epochs
        self.n_stage2_epochs = n_stage2_epochs
        self.n_stage3_epochs = n_stage3_epochs
        self.high_conf_threshold = high_conf_threshold
        self._model: EORMModel | None = None

    def _make_model(self) -> EORMModel:
        """Create a fresh EORM model for curriculum training."""
        key = jrandom.PRNGKey(42)
        return EORMModel(
            embed_dim=_EMBED_DIM,
            n_heads=_N_HEADS,
            n_layers=_N_LAYERS,
            max_seq_len=_MAX_SEQ_LEN,
            vocab_size=_VOCAB_SIZE,
            key=key,
        )

    def _train_stage(
        self,
        model: EORMModel,
        pairs: list[dict[str, Any]],
        n_epochs: int,
    ) -> EORMModel:
        """Train model on pairs for n_epochs; return updated model."""
        triples = _pairs_to_eorm_triples(pairs)
        if not triples:
            return model
        trainer = EORMTrainer(model, lr=1e-3, margin=1.0)
        for _ in range(n_epochs):
            trainer.train_epoch(triples, batch_size=8)
        if hasattr(trainer, "model"):
            return trainer.model
        return model

    def _filter_high_conf(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return pairs where label_confidence >= high_conf_threshold.

        This is the Stage 1 filter.  It is intentionally more conservative than
        Exp 477's 0.70 threshold to guarantee the Stage 1 corpus is genuinely
        high-quality for anchoring purposes.
        """
        return [p for p in pairs if _estimate_label_confidence(p) >= self.high_conf_threshold]

    def train(self, pairs: list[dict[str, Any]]) -> list[CurriculumStageResult]:
        """Run three-stage curriculum training and return per-stage results.

        **For engineers:**
            The held-out 20% is always the LAST 20% of the input pairs list (by index),
            keeping it constant across all three stages so AUCs are directly comparable.
            The training pool is the first 80%.

            Synthetic augmentation in Stage 3 uses IsingModel with input_dim=8 (tiny but
            sufficient for CPU-speed EBM-guided sampling).

        Args:
            pairs: All available labeled pairs (real data).  Each dict must have at least
                   'label' ('correct'/'incorrect') and 'step_text' or 'response' fields.
                   Optionally 'label_confidence' or 'confidence' for Stage 1 filtering.

        Returns:
            List of three CurriculumStageResult objects, one per stage.

        Spec: REQ-LEARN-040, REQ-LEARN-041, REQ-LEARN-042,
              SCENARIO-LEARN-068, SCENARIO-LEARN-069, SCENARIO-LEARN-070
        """
        # Fixed 80/20 split — held-out is always the same set
        n_pairs = len(pairs)
        n_held = max(1, n_pairs // 5)
        train_pool = list(pairs[: n_pairs - n_held])
        held_out = list(pairs[n_pairs - n_held :])

        # Shuffle training pool once to avoid accidental ordering biases
        rng = random.Random(42)
        rng.shuffle(train_pool)

        model = self._make_model()
        stages: list[CurriculumStageResult] = []

        # --- Stage 1: High-confidence pairs only ---
        stage1_pairs = self._filter_high_conf(train_pool)
        auc_before_s1 = _compute_auc(model, held_out)
        model = self._train_stage(model, stage1_pairs, self.n_stage1_epochs)
        auc_after_s1 = _compute_auc(model, held_out)
        stages.append(CurriculumStageResult(
            stage=1,
            n_pairs=len(stage1_pairs),
            n_epochs=self.n_stage1_epochs,
            auc_after=auc_after_s1,
            auc_before=auc_before_s1,
        ))

        # --- Stage 2: All pairs, no filter ---
        auc_before_s2 = auc_after_s1
        model = self._train_stage(model, train_pool, self.n_stage2_epochs)
        auc_after_s2 = _compute_auc(model, held_out)
        stages.append(CurriculumStageResult(
            stage=2,
            n_pairs=len(train_pool),
            n_epochs=self.n_stage2_epochs,
            auc_after=auc_after_s2,
            auc_before=auc_before_s2,
        ))

        # --- Stage 3: EBM-guided synthetic augmentation ---
        n_total_real = len(pairs)
        n_synthetic_needed = max(0, 200 - n_total_real)

        if n_synthetic_needed > 0:
            # Import here to avoid circular at module level
            from carnot.models.ising import IsingConfig, IsingModel
            ising = IsingModel(IsingConfig(input_dim=8))
            augmentor = JEPAQualityAugmentor(ising_model=ising, n_samples=n_synthetic_needed + 20)
            synthetic_pairs = augmentor.generate_violation_pairs() + augmentor.generate_correct_pairs()
            # Limit to exactly what we need
            synthetic_pairs = synthetic_pairs[:n_synthetic_needed]
        else:
            synthetic_pairs = []

        stage3_pairs = train_pool + synthetic_pairs
        auc_before_s3 = auc_after_s2
        model = self._train_stage(model, stage3_pairs, self.n_stage3_epochs)
        auc_after_s3 = _compute_auc(model, held_out)
        stages.append(CurriculumStageResult(
            stage=3,
            n_pairs=len(stage3_pairs),
            n_epochs=self.n_stage3_epochs,
            auc_after=auc_after_s3,
            auc_before=auc_before_s3,
        ))

        self._model = model
        return stages

    def get_final_auc(self, held_out_pairs: list[dict[str, Any]]) -> float:
        """Compute AUC of the trained model on the given held-out pairs.

        Call this after train() to evaluate on a custom held-out set.
        If train() has not been called yet, returns 0.5 (untrained baseline).

        Args:
            held_out_pairs: Pairs with 'label' and 'step_text'/'response' fields.

        Returns:
            AUC in [0, 1].

        Spec: REQ-LEARN-042
        """
        if self._model is None:
            return 0.5
        return _compute_auc(self._model, held_out_pairs)


# ---------------------------------------------------------------------------
# JEPARetrainV3Result — summary of curriculum retrain outcome
# ---------------------------------------------------------------------------


@dataclass
class JEPARetrainV3Result:
    """Summary of a curriculum-based JEPA retrain run (Exp 492).

    **For engineers:**
        This dataclass captures the key metrics from the three-stage curriculum
        retrain.  It is analogous to JEPARetrainV2Result but tracks per-stage
        AUC progression through the curriculum pipeline instead of a single
        quality-gate filter.

    Attributes:
        n_pairs_raw: Number of real CoT pairs loaded before any filtering.
        curriculum_stages: List of CurriculumStageResult, one per stage.
        before_auc: JEPA AUC before curriculum training (regression baseline ~0.281).
        after_auc: JEPA AUC after completing all three curriculum stages.

    Derived:
        auc_improvement: after_auc - before_auc.
        target_met: True iff after_auc > 0.600 (RETRO-040 closure bar).
        regression_recovered: True iff after_auc > 0.400 (recovery from 0.281 regression).

    Spec: REQ-LEARN-042, SCENARIO-LEARN-070
    """

    n_pairs_raw: int
    curriculum_stages: list[CurriculumStageResult]
    before_auc: float
    after_auc: float

    @property
    def auc_improvement(self) -> float:
        """AUC delta: after_auc minus before_auc.  Positive = improvement."""
        return self.after_auc - self.before_auc

    @property
    def target_met(self) -> bool:
        """True iff after_auc > 0.600 (RETRO-040 closure bar for curriculum approach)."""
        return self.after_auc > 0.600

    @property
    def regression_recovered(self) -> bool:
        """True iff after_auc > 0.400 (recovery from the 0.281 quality-gate regression).

        The 0.281 was the result of Exp 477's quality gate removing 73% of pairs and
        causing majority-class collapse.  Recovery to > 0.400 means we have reversed
        the direction of the regression, even if not yet at the RETRO-040 closure bar.
        """
        return self.after_auc > 0.400
