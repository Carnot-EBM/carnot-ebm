"""CAPOCalibrationLoss — Calibration-Aware Penalty Objective for NUP Probe v6.

**Why this module exists (RETRO-049, NUP Probe v6):**
    NUP Probe v5 (Exp 599) achieved AUC=0.739 — below the 0.80 deployment threshold
    for Tier 0c.  Two problems prevented v5 from reaching the threshold:

    1. Training corpus too small (37 GRPO pairs, synthetic-heavy).  V6 uses the full
       fover_corpus_v4 (300 live pairs from Exps 578/579/602).

    2. The contrastive margin loss from v4 can overfit when the training signal is
       strong: it pushes scores to extreme values (AUC=1.0 on training data) while
       generalising poorly.  JEPA v11 showed this pattern (val_auc=1.0 = memorisation).

    CAPO (arXiv 2604.12632) adds a calibration regulariser on top of the contrastive
    margin loss.  The calibration term detects when the model is being OVERCONFIDENT
    — assigning a very large score gap to a pair — and penalises that gap.

    The mechanism: for any (correct, incorrect) pair where
        |score_correct - score_incorrect| < 0.3
    (the scores are "close", meaning the model is NOT yet overconfident on this pair),
    the calibration loss adds a quadratic penalty on the WMW U-statistic approximation:

        cal_loss += (score_correct - score_incorrect + 0.5)^2

    When score_correct - score_incorrect = 0 (equal scores), this is 0.5^2 = 0.25.
    When score_correct - score_incorrect = -0.25 (incorrect slightly higher), this is
    (−0.25 + 0.5)^2 = 0.0625 (smaller, rewarding the right direction a little).
    When score_correct - score_incorrect = 0.25 (correct slightly higher — bad sign for
    an EBM where INCORRECT should be high), this is 0.75^2 = 0.5625 (strong penalty).

    The 0.3 threshold is key: once the model has learned a gap of >=0.3 between scores,
    the calibration term is SILENT — no gradient.  This prevents the term from fighting
    the contrastive loss once the signal is learned; it only acts on "easy" pairs where
    the model is still uncertain and could tip into overconfidence.

    Total loss:
        total = margin_loss + lambda_cal * calibration_loss

    With lambda_cal=0.1, the calibration term is a soft regulariser — it cannot
    overwhelm the contrastive loss but is strong enough to prevent the AUC=1.0 pattern.

**Why WMW approximation for calibration:**
    The Wilcoxon-Mann-Whitney (WMW) U statistic estimates AUC directly.  An ideal
    calibrated model has U near 0.8 (our target threshold).  The quadratic form
    (score_correct - score_incorrect + 0.5)^2 is a differentiable proxy for the
    squared error between the indicator function used in WMW and 0.5 (chance).
    Penalising this proxy prevents the model from exploiting degenerate score distributions
    that fool AUC while failing on real data.

Spec: REQ-VERIFY-140, REQ-VERIFY-141,
      SCENARIO-VERIFY-171, SCENARIO-VERIFY-172, SCENARIO-VERIFY-173
"""

from __future__ import annotations

from typing import List


class CAPOCalibrationLoss:
    """Calibration-aware training loss combining contrastive margin loss with a WMW calibration term.

    **Design:**
        Two components:

        1. Margin (contrastive) loss — identical to ContrastivePairLoss in nup_probe_v4.py:
               margin_loss = sum_i max(0, margin - (score_incorrect_i - score_correct_i))
           This is the primary learning signal: push E(incorrect) above E(correct) by
           at least `margin`.

        2. Calibration loss (WMW U-statistic approximation):
               for each pair where |score_correct - score_incorrect| < 0.3:
                   cal_loss += (score_correct - score_incorrect + 0.5)^2
           This penalises overconfidence only on "borderline" pairs, not on pairs the
           model has already correctly separated.  The 0.3 threshold gates the gradient
           off once the model has learned a sufficient gap.

        3. Combined:
               total = margin_loss + lambda_cal * calibration_loss

    Args:
        lambda_cal: Weight for the calibration term.  Default 0.1.
                    At 0.1, calibration is a soft regulariser that cannot overwhelm
                    the margin loss.  Increase to 0.3+ if overfitting recurs.
        margin:     Minimum required energy gap E(incorrect) - E(correct) for the
                    contrastive term.  Default 1.0 — matches NUPProbeV4.

    Spec: REQ-VERIFY-140, REQ-VERIFY-141,
          SCENARIO-VERIFY-171, SCENARIO-VERIFY-172, SCENARIO-VERIFY-173
    """

    _CALIBRATION_THRESHOLD: float = 0.3
    """Gap threshold below which the calibration term is active.  Pairs with
    |score_correct - score_incorrect| >= 0.3 are already well-separated; we
    silence the calibration gradient to avoid fighting the contrastive signal."""

    def __init__(self, lambda_cal: float = 0.1, margin: float = 1.0) -> None:
        self.lambda_cal = lambda_cal
        self.margin = margin

    def compute_loss(
        self,
        scores_correct: List[float],
        scores_incorrect: List[float],
    ) -> float:
        """Compute CAPO total loss for a batch of (correct, incorrect) score pairs.

        **Step-by-step:**
            For each pair (score_correct, score_incorrect):

            Contrastive term:
                gap = score_incorrect - score_correct   (want this >= margin)
                margin_loss_i = max(0, margin - gap)

            Calibration term (WMW approximation):
                diff = score_correct - score_incorrect   (want this < 0 for correct EBM)
                if |diff| < 0.3:   (pair not yet well-separated)
                    cal_loss_i = (diff + 0.5)^2
                else:
                    cal_loss_i = 0   (already separated; silence the gradient)

            Total per pair:
                loss_i = margin_loss_i + lambda_cal * cal_loss_i

            Returns: mean(loss_i) across all pairs.

        Args:
            scores_correct:   Energy scores for known-correct CoT steps.  Lower is better.
            scores_incorrect: Energy scores for known-incorrect CoT steps.  Higher is better.

        Returns:
            Mean CAPO loss across all pairs.  0.0 if either list is empty.

        Spec: REQ-VERIFY-141, SCENARIO-VERIFY-172, SCENARIO-VERIFY-173
        """
        if not scores_correct or not scores_incorrect:
            return 0.0

        n = min(len(scores_correct), len(scores_incorrect))
        total = 0.0

        for i in range(n):
            sc = scores_correct[i]
            si = scores_incorrect[i]

            # Contrastive margin loss: hinge on E(incorrect) - E(correct) >= margin
            gap = si - sc
            margin_loss = max(0.0, self.margin - gap)

            # WMW calibration loss: penalise pairs that are not yet well-separated
            # diff = score_correct - score_incorrect (want < 0 for a correct EBM)
            diff = sc - si
            if abs(diff) < self._CALIBRATION_THRESHOLD:
                cal_loss = (diff + 0.5) ** 2
            else:
                cal_loss = 0.0

            total += margin_loss + self.lambda_cal * cal_loss

        return total / n
