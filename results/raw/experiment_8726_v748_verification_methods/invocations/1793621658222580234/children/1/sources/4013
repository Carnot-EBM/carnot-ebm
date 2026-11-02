"""SpilledEnergyDetector — training-free hallucination pre-filter using logit spill.

**Researcher summary (arXiv 2602.18671 "Spilled Energy in Large Language Models"):**
    When an LLM hallucinates, it places probability mass on tokens that are *higher*
    than what the surrounding context justifies.  This "spilled energy" is the excess
    log-probability above the contextual expectation.

    Key advantage: zero additional inference cost.  The logits already exist from the
    generation step; we just post-process them.  No training, no secondary model call,
    no extra forward pass — unlike ThinkProbe (Exp 444), which issues a full generative
    verification call.

**How spill is defined:**
    For each generated token t with log-probability log_p(t | context):
        spill_t = max(0, log_p(t | context) - expected_log_p)

    expected_log_p is approximated as the negative context entropy:
        expected_log_p ≈ -H(context)
    where H(context) is the Shannon entropy of the token distribution at that step.

    Intuition: if the model is "uncertain" (high entropy), we *expect* individual
    tokens to have low log-probability.  If a token shows up with log-probability
    much HIGHER than that expectation, something anomalous happened — the model
    is overconfident about a specific token that contradicts the contextual
    uncertainty.  This overconfidence is the hallucination signal.

    Total spill for a response = mean(spill_t over all tokens).

**Pipeline position (Tier 0 pre-filter):**
    SpilledEnergyDetector (Tier 0, ~0 ms — no model call)
        ↓ high spill → route to full Ising verification
        ↓ low spill  → skip verification (fast path, save ~0.006 ms * n_constraints)
    Ising verifier (Tier 1, ~0.006 ms per constraint)

    This complements ThinkProbe (Tier 0g, generative) which costs ~50–200 ms.
    SpilledEnergy is a cheaper upstream gate: if spill is low, skip everything.

**Phase 3 note:**
    In Phase 3, the "expected log-probability from context" will be computed by the
    energy function of the EBM directly — making this an exact, principled signal
    rather than an entropy-based approximation.  The interface here is designed to
    be forward-compatible with that transition.

Spec: REQ-PROBE-022, SCENARIO-PROBE-022, SCENARIO-PROBE-023
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# SpilledEnergyResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SpilledEnergyResult:
    """Summary statistics from SpilledEnergyDetector.benchmark().

    **Detailed explanation for engineers:**
        auroc:
            Area Under ROC Curve over the benchmark corpus.  Measures how well
            the raw spill score separates hallucinated responses from correct ones.
            0.5 = random baseline, 1.0 = perfect separation.
            Below 0.5 means the signal is *inverted* — spill is higher for correct
            responses, which should not happen under normal conditions.

        optimal_threshold:
            The spill value (in bits) that maximises Youden's J = TPR - FPR on
            the benchmark corpus.  Use this as the ``threshold`` argument to
            flag_response() in deployment.  Note: this is in-sample optimal — on
            unseen data, performance will be lower.

        skip_rate:
            Fraction of responses whose spill is BELOW the optimal_threshold —
            i.e., the fraction Ising verification would be skipped for.
            Higher is better for throughput, but only if fn_rate is acceptably low.

        fn_rate (False Negative Rate):
            Of all HALLUCINATED responses, what fraction falls *below* the threshold?
            These are the errors SpilledEnergy *misses* — Ising won't run, so the
            hallucination slips through undetected.
            Lower is better.  This is the primary safety metric.

        honest_verdict:
            Machine-readable outcome string for conductor reconciliation:
              'spilled_energy_viable'       auroc > 0.60
              'spilled_energy_marginal'     0.50 < auroc <= 0.60
              'spilled_energy_below_random' auroc <= 0.50

    Spec: REQ-PROBE-022
    """

    auroc: float
    optimal_threshold: float
    skip_rate: float
    fn_rate: float
    honest_verdict: Literal[
        "spilled_energy_viable",
        "spilled_energy_marginal",
        "spilled_energy_below_random",
    ]


# ---------------------------------------------------------------------------
# SpilledEnergyDetector
# ---------------------------------------------------------------------------


class SpilledEnergyDetector:
    """Training-free hallucination pre-filter based on logit spilled energy.

    **Detailed explanation for engineers:**
        This detector requires NO training and NO secondary model.  It runs
        entirely on the log-probabilities that the primary LLM already produces
        during generation.  The only extra cost is a few arithmetic operations
        per token — effectively zero latency overhead.

        How to use in a real pipeline:
            1. When the LLM generates a response, capture the per-token
               log-probabilities and the context entropy at each step.
            2. Call compute_spill(log_probs, context_entropy) to get a scalar.
            3. Call flag_response(spill, threshold) to decide routing.
            4. If flag_response returns True, run Ising verification.
               If False, skip (fast path).

        Synthetic mode (for benchmarking without a real LLM):
            generate_synthetic_corpus() produces mock log-probs and context
            entropies for correct and hallucinated responses, enabling AUROC
            measurement without GPU hardware.

    Spec: REQ-PROBE-022
    """

    def compute_spill(self, log_probs: list[float], context_entropy: float) -> float:
        """Compute mean spilled energy for one response.

        **Detailed explanation for engineers:**
            For each token t:
                expected_log_p = -context_entropy
                    (If entropy is 2.0 nats, the "expected" token log-prob is -2.0.
                     A context with entropy 0 means the model is certain — every token
                     should have log_p ≈ 0.  High entropy means low expected log_p.)
                spill_t = max(0, log_p_t - expected_log_p)
                    (The positive part: how much ABOVE expectation this token is.)

            Total spill = mean(spill_t) over all tokens.

            Why mean not sum: normalises for response length so short and long
            responses are comparable on the same threshold scale.

        Args:
            log_probs: Per-token log-probabilities from the LLM (negative floats,
                e.g. [-1.2, -3.1, -0.4, ...]).  Should be natural-log probabilities.
            context_entropy: Shannon entropy of the token distribution at this
                context step, in the same units as log_probs (nats).  Used as a
                proxy for the "expected" log-probability under a uniform contextual
                prediction.

        Returns:
            Mean spilled energy over all tokens, in nats.  Zero means no spill
            (all tokens at or below contextual expectation).  Positive means
            anomalous over-confidence on at least some tokens.

        Spec: REQ-PROBE-022
        SCENARIO-PROBE-022
        """
        if not log_probs:
            return 0.0

        expected_log_p = -context_entropy
        spills = [max(0.0, lp - expected_log_p) for lp in log_probs]
        return float(sum(spills) / len(spills))

    def flag_response(self, spill_score: float, threshold: float = 0.5) -> bool:
        """Return True if this response should be routed to full Ising verification.

        **Detailed explanation for engineers:**
            The routing rule is simple:
                spill_score >= threshold → True  (run Ising)
                spill_score < threshold  → False (skip Ising, fast path)

            The threshold controls the precision/recall tradeoff:
                Low threshold → more responses go to Ising (high recall, low throughput)
                High threshold → fewer responses go to Ising (low recall, high throughput)

            The optimal_threshold from benchmark() gives the Youden-J-maximising value
            on the benchmark corpus.  Use that as the starting point; tune downward if
            missing hallucinations is unacceptable in your deployment.

        Args:
            spill_score: Mean spilled energy from compute_spill() for this response.
            threshold: Routing threshold. Default 0.5. Use optimal_threshold from
                benchmark() for deployment.

        Returns:
            True if Ising verification should run, False if safe to skip.

        Spec: REQ-PROBE-022
        """
        return spill_score >= threshold

    def benchmark(
        self,
        responses_with_logits: list[dict],
        labels: list[bool],
    ) -> SpilledEnergyResult:
        """Measure AUROC and find optimal threshold on a labelled corpus.

        **Detailed explanation for engineers:**
            Each entry in responses_with_logits must have:
                'log_probs': list[float]    — per-token log-probabilities
                'context_entropy': float    — context entropy at generation time

            labels[i] = True  → response i is CORRECT (not a hallucination)
            labels[i] = False → response i is HALLUCINATED

            AUROC interpretation:
                The spill score should be HIGHER for hallucinated responses.
                So we pass spill scores directly (not negated) to roc_auc_score
                with hallucinated=1, correct=0.  An AUROC above 0.5 means
                higher spill reliably predicts hallucination — the correct direction.

            Optimal threshold (Youden's J):
                Sweep over all unique spill values as candidate thresholds.
                For each, compute TPR (hallucinations correctly routed to Ising)
                and FPR (correct responses incorrectly routed to Ising).
                Youden's J = TPR - FPR; maximise this.

            skip_rate:
                Fraction of responses below optimal_threshold (Ising would be skipped).

            fn_rate (False Negative Rate):
                Of hallucinated responses, fraction below optimal_threshold (missed).

        Args:
            responses_with_logits: List of dicts with 'log_probs' and 'context_entropy'.
            labels: Parallel list of booleans. True=correct, False=hallucinated.

        Returns:
            SpilledEnergyResult with auroc, optimal_threshold, skip_rate, fn_rate,
            and honest_verdict.

        Spec: REQ-PROBE-022
        SCENARIO-PROBE-023
        """
        spill_scores = [
            self.compute_spill(r["log_probs"], r["context_entropy"]) for r in responses_with_logits
        ]

        # sklearn roc_auc_score expects binary labels where 1=positive class.
        # We define hallucinated=1 (spill should be high), correct=0.
        binary_labels = [0 if lbl else 1 for lbl in labels]

        auroc = float(roc_auc_score(binary_labels, spill_scores))

        # Find Youden-J optimal threshold by sweeping all unique spill values.
        unique_thresholds = sorted(set(spill_scores))
        best_j = -1.0
        best_threshold = unique_thresholds[0] if unique_thresholds else 0.5

        n_hallucinated = sum(1 for lbl in labels if not lbl)
        n_correct = sum(1 for lbl in labels if lbl)

        for thr in unique_thresholds:
            tp = sum(1 for score, lbl in zip(spill_scores, labels) if score >= thr and not lbl)
            fp = sum(1 for score, lbl in zip(spill_scores, labels) if score >= thr and lbl)
            tpr = tp / n_hallucinated if n_hallucinated > 0 else 0.0
            fpr = fp / n_correct if n_correct > 0 else 0.0
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_threshold = thr

        # skip_rate: fraction of all responses below threshold (Ising skipped)
        n_below = sum(1 for s in spill_scores if s < best_threshold)
        skip_rate = n_below / len(spill_scores) if spill_scores else 0.0

        # fn_rate: of hallucinated responses, fraction below threshold (missed)
        n_hallucinated_below = sum(
            1 for score, lbl in zip(spill_scores, labels) if score < best_threshold and not lbl
        )
        fn_rate = n_hallucinated_below / n_hallucinated if n_hallucinated > 0 else 0.0

        if auroc > 0.60:
            verdict = "spilled_energy_viable"
        elif auroc > 0.50:
            verdict = "spilled_energy_marginal"
        else:
            verdict = "spilled_energy_below_random"

        return SpilledEnergyResult(
            auroc=auroc,
            optimal_threshold=float(best_threshold),
            skip_rate=float(skip_rate),
            fn_rate=float(fn_rate),
            honest_verdict=verdict,
        )
