"""Semantic Energy overconfidence detector — catches confident-but-wrong outputs.

**Researcher summary:**
    Implements the Semantic Energy signal from arXiv 2508.14496.  While Spilled
    Energy (Exp 285, REQ-VERIFY-076) fires on uncertain (high-entropy) outputs,
    Semantic Energy fires on overconfident-wrong outputs — the complementary failure
    mode where entropy is LOW but the answer is WRONG.

    The DualEnergyGate combines both signals: it fires if EITHER spilled energy OR
    semantic energy exceeds its calibrated threshold, providing an extraction-free
    first-pass filter that covers both failure modes without any regex, KB, or SMT.

**Detailed explanation for engineers:**
    Formula (per token t):
        E_semantic_t = −log( ∑_i exp(logit_t_i / T) )

    This is the negative log-partition function (Helmholtz free energy analogy):

    - LOW  E_semantic (large negative) → peaked distribution → HIGH confidence
    - HIGH E_semantic (less negative)  → flat distribution  → LOW confidence
    - Temperature T: higher T flattens the effective distribution → less negative energy

    Intuition: for a perfectly peaked distribution (one token has all mass),
        ∑_i exp(logit_i / T) ≈ exp(max_logit / T)
        → E_semantic ≈ −max_logit / T  (very negative for large max_logit)

    For a uniform distribution over V tokens:
        ∑_i exp(0) = V  → E_semantic = −log(V)  (less negative than peaked)

    The ``overconfident_flag`` fires when mean E_semantic < threshold.  Because
    we calibrate the threshold on a corpus where overconfident-wrong examples have
    very negative energies, a flag means "this response is more confident than
    calibration predicts is warranted, raising overconfidence-error risk."

    DualEnergyGate logic:
        gate_fired = spilled.suspected_hallucination OR semantic.overconfident_flag
        trigger_signal = "both" | "spilled" | "semantic" | "none"

    Key classes:
    - SemanticEnergyResult: dataclass with semantic_energy, temperature,
      overconfident_flag, threshold_used, per_token_semantic, to_dict(), to_json().
    - compute_semantic_energy(): core function, accepts (T, V) float64 logit array
      and temperature, returns mean semantic energy scalar.
    - SemanticEnergyExtractor: extract(logits) → SemanticEnergyResult;
      calibrate(corpus, labels) fits isotonic regression and updates threshold.
    - DualEnergyResult: dataclass combining SpilledEnergyResult + SemanticEnergyResult.
    - DualEnergyGate: fire(spilled, semantic) → DualEnergyResult; calibrate() delegates
      to SemanticEnergyExtractor.
    - VerifyRepairPipeline.verify_dual_energy() added in verify_repair.py.

Spec: REQ-VERIFY-077, SCENARIO-VERIFY-095, SCENARIO-VERIFY-096
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from carnot.pipeline.spilled_energy_extractor import SpilledEnergyResult


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class SemanticEnergyResult:
    """Result of a semantic-energy overconfidence analysis.

    **Detailed explanation for engineers:**
        ``semantic_energy`` is the mean negative log-partition function across all
        response tokens.  A very negative value indicates a highly peaked (confident)
        distribution.  ``overconfident_flag`` is True when
        ``semantic_energy < threshold_used``.

        The intuition: a model that is extremely confident (log-partition >> threshold)
        on a response that later turns out wrong is exhibiting overconfidence — a failure
        mode that Spilled Energy (high-entropy) does NOT catch.  Semantic Energy catches
        the complementary class.

    Attributes:
        semantic_energy: Mean −log(∑_i exp(logit_i / T)) across all response tokens.
            Typically negative.  More negative = more confident.
        temperature: Temperature T used in the energy computation.
        overconfident_flag: True when semantic_energy < threshold_used.
        threshold_used: The threshold that was applied; flag fires when energy < this.
        per_token_semantic: 1-D float64 array, shape (T,), per-token semantic energy
            values before averaging.

    Spec: REQ-VERIFY-077
    """

    semantic_energy: float
    temperature: float
    overconfident_flag: bool
    threshold_used: float
    per_token_semantic: np.ndarray

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict.

        **Detailed explanation for engineers:**
            numpy arrays are converted to Python lists of floats so the result
            can be passed to json.dumps() without a custom encoder.

        Returns:
            Dict with all fields serialized to JSON-compatible Python types.

        Spec: REQ-VERIFY-077
        """
        return {
            "semantic_energy": float(self.semantic_energy),
            "temperature": float(self.temperature),
            "overconfident_flag": bool(self.overconfident_flag),
            "threshold_used": float(self.threshold_used),
            "per_token_semantic": self.per_token_semantic.tolist(),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string with sorted keys for determinism."""
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class DualEnergyResult:
    """Combined result from the DualEnergyGate.

    **Detailed explanation for engineers:**
        The DualEnergyGate fires when EITHER the spilled energy signal OR the semantic
        energy signal indicates a problem:

        - Spilled energy fires on UNCERTAIN outputs (high entropy, model hedging).
        - Semantic energy fires on OVERCONFIDENT outputs (low entropy, but model may
          be confidently wrong — e.g., wrong numeric answer stated without hedging).

        Together they form an extraction-free first-pass filter: if the gate fires,
        the response warrants deeper verification (e.g., constraint extraction or
        human review).  If the gate does not fire, the response passes the
        energy-based pre-filter.

    Attributes:
        spilled_result: SpilledEnergyResult from the spilled energy extractor.
        semantic_result: SemanticEnergyResult from the semantic energy extractor.
        gate_fired: True if EITHER signal triggered.
        trigger_signal: Which signal(s) triggered: "both", "spilled", "semantic", "none".
        calibration_threshold_used: The semantic energy threshold applied.

    Spec: REQ-VERIFY-077
    """

    spilled_result: SpilledEnergyResult
    semantic_result: SemanticEnergyResult
    gate_fired: bool
    trigger_signal: Literal["spilled", "semantic", "both", "none"]
    calibration_threshold_used: float

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict.

        **Detailed explanation for engineers:**
            Both sub-results are serialized via their own to_dict() methods,
            so the full nested structure is JSON-compatible.

        Returns:
            Dict with all fields serialized to JSON-compatible Python types.

        Spec: REQ-VERIFY-077
        """
        return {
            "spilled_result": self.spilled_result.to_dict(),
            "semantic_result": self.semantic_result.to_dict(),
            "gate_fired": bool(self.gate_fired),
            "trigger_signal": self.trigger_signal,
            "calibration_threshold_used": float(self.calibration_threshold_used),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string with sorted keys for determinism."""
        return json.dumps(self.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_per_token_semantic_energy(
    logits: np.ndarray, temperature: float
) -> np.ndarray:
    """Compute per-token semantic energy values (internal helper).

    **Detailed explanation for engineers:**
        Uses the log-sum-exp trick to avoid numerical overflow:
            log(∑_i exp(logit_i / temperature))
            = max_j(logit_j / temperature)
              + log(∑_i exp(logit_i/temperature − max_j/temperature))

        The per-token energy is the negation of this log-partition.

    Args:
        logits: 2-D float64 array of shape (n_tokens, vocab_size).
        temperature: Temperature > 0.

    Returns:
        1-D float64 array of shape (n_tokens,).

    Spec: REQ-VERIFY-077
    """
    scaled = logits / temperature                                      # (n_tokens, V)
    max_scaled = np.max(scaled, axis=-1, keepdims=True)               # (n_tokens, 1)
    # log(∑_i exp(scaled_i)) = max + log(∑_i exp(scaled_i - max))
    log_partition = max_scaled.squeeze(-1) + np.log(
        np.sum(np.exp(scaled - max_scaled), axis=-1)
    )                                                                   # (n_tokens,)
    return np.asarray(-log_partition, dtype=np.float64)


def compute_semantic_energy(logits: np.ndarray, temperature: float = 1.0) -> float:
    """Compute the mean semantic energy from a 2-D logit array.

    **Detailed explanation for engineers:**
        Formula: E_semantic = mean_t( −log( ∑_i exp(logit_t_i / temperature) ) )

        This is the mean negative log-partition function across all response tokens.

        Interpretation:
        - Low (very negative) E_semantic → model is highly confident (peaked distribution).
        - High (less negative) E_semantic → model is uncertain (flat distribution).
        - Temperature: higher temperature flattens the effective distribution → higher energy.

        Numerical stability is ensured via the log-sum-exp trick (subtract per-row max
        before exponentiating).

    Args:
        logits: 2-D float64 array of shape (n_tokens, vocab_size), where n_tokens is
            the number of response tokens and vocab_size is the vocabulary size.
        temperature: Temperature parameter.  Default 1.0.  Must be > 0.

    Returns:
        Mean semantic energy across all tokens (scalar float, typically negative).

    Raises:
        ValueError: If logits is not 2-D, has fewer than 1 token, or temperature <= 0.

    Spec: REQ-VERIFY-077, SCENARIO-VERIFY-095
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 2-D (n_tokens, vocab_size), got shape {logits.shape}"
        )
    if logits.shape[0] < 1:
        raise ValueError("logits must contain at least one token (n_tokens >= 1)")
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")

    per_token = _compute_per_token_semantic_energy(logits, temperature)
    return float(np.mean(per_token))


# ---------------------------------------------------------------------------
# SemanticEnergyExtractor
# ---------------------------------------------------------------------------


class SemanticEnergyExtractor:
    """Detect overconfident-wrong outputs via the semantic energy signal.

    **Detailed explanation for engineers:**
        This class wraps compute_semantic_energy() with a configurable threshold
        and temperature.  Usage patterns:

        1. **Manual threshold** (``threshold`` in constructor): Use when you have
           domain knowledge about a reasonable confidence cutoff.  The default
           threshold of −5.0 nats fires only when the model is extremely confident
           (log-partition >> 5), which is a conservative starting point.

        2. **Calibrated threshold** (``calibrate()`` method): Use when you have a
           labeled corpus of (logits, is_correct) pairs.  Fits an isotonic regression
           mapping semantic energy to P(wrong) and finds the crossing energy where
           P(wrong) drops below 0.5.

        The threshold fires when energy < threshold (i.e., the model is MORE confident
        than the threshold allows — a suspicious degree of confidence).

    Spec: REQ-VERIFY-077
    """

    DEFAULT_THRESHOLD: float = -5.0
    """Default overconfidence threshold in nats.

    Fires when mean semantic energy < −5.0, meaning the log-partition is > 5.0 nats
    — the model is extremely confident.  Conservative default; calibrate for production.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        temperature: float = 1.0,
    ) -> None:
        """Create an extractor with the given threshold and temperature.

        Args:
            threshold: Semantic energy below which overconfident_flag fires.
                Default −5.0 nats (conservative).
            temperature: Logit temperature for energy computation.  Default 1.0.

        Spec: REQ-VERIFY-077
        """
        self.threshold = threshold
        self.temperature = temperature

    def extract(self, logits: np.ndarray) -> SemanticEnergyResult:
        """Run semantic energy analysis on a logit array.

        **Detailed explanation for engineers:**
            Validates the input shape, delegates to _compute_per_token_semantic_energy()
            for numerical efficiency, then applies the threshold comparison.

        Args:
            logits: 2-D float64 array of shape (n_tokens, vocab_size).

        Returns:
            SemanticEnergyResult with semantic_energy, overconfident_flag, and
            per_token_semantic populated.

        Raises:
            ValueError: If logits is not 2-D or has fewer than 1 token.

        Spec: REQ-VERIFY-077, SCENARIO-VERIFY-095
        """
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2:
            raise ValueError(
                f"logits must be 2-D (n_tokens, vocab_size), got shape {logits.shape}"
            )
        if logits.shape[0] < 1:
            raise ValueError("logits must contain at least one token (n_tokens >= 1)")

        per_token = _compute_per_token_semantic_energy(logits, self.temperature)
        mean_energy = float(np.mean(per_token))

        return SemanticEnergyResult(
            semantic_energy=mean_energy,
            temperature=self.temperature,
            overconfident_flag=mean_energy < self.threshold,
            threshold_used=self.threshold,
            per_token_semantic=per_token,
        )

    def calibrate(
        self,
        logits_corpus: list[np.ndarray],
        labels: list[bool],
    ) -> float:
        """Calibrate the threshold using isotonic regression on a labeled corpus.

        **Detailed explanation for engineers:**
            Step 1: Compute semantic energy for each logit array in the corpus.
            Step 2: Build wrong/correct labels (wrong=1.0 when label=False).
            Step 3: Fit IsotonicRegression with increasing=False (lower energy →
                    higher P(wrong) for overconfident models).
            Step 4: Evaluate on a 1000-point grid and find the crossing energy
                    where P(wrong) first drops below 0.5 going from low to high energy.
            Step 5: Set self.threshold to that crossing energy.

            The crossing energy is the point below which the model is so confident
            that wrong-answer probability exceeds 50%.  Above it, confidence is normal
            and errors are less predicted.

            Fallback: if no crossing exists (all energies predict > 50% wrong), uses
            the median energy of wrong examples.

        Args:
            logits_corpus: List of (n_tokens_i, vocab_size) numpy arrays.
            labels: Corresponding list of booleans; True = CORRECT, False = WRONG.

        Returns:
            The calibrated threshold float (also set on self.threshold).

        Raises:
            ValueError: If logits_corpus and labels have different lengths or fewer
                than 2 examples are provided.

        Spec: REQ-VERIFY-077
        """
        if len(logits_corpus) != len(labels):
            raise ValueError(
                f"logits_corpus and labels must have the same length, "
                f"got {len(logits_corpus)} and {len(labels)}"
            )
        if len(logits_corpus) < 2:
            raise ValueError("Need at least 2 examples to calibrate")

        from sklearn.isotonic import IsotonicRegression

        energies = np.array(
            [compute_semantic_energy(lg, self.temperature) for lg in logits_corpus],
            dtype=np.float64,
        )
        # 1.0 = WRONG, 0.0 = CORRECT
        wrong = np.array(
            [0.0 if lbl else 1.0 for lbl in labels], dtype=np.float64
        )

        # Isotonic regression: energy → P(wrong)
        # increasing=False: lower energy (more confident) → higher P(wrong)
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
        ir.fit(energies, wrong)

        # Evaluate on a fine grid from min to max energy.
        e_min = float(np.min(energies))
        e_max = float(np.max(energies))
        grid = np.linspace(e_min, e_max, 1000)
        p_wrong_grid: np.ndarray = ir.predict(grid)

        # Find the energy where P(wrong) first drops below 0.5 going from low→high.
        # Below this energy: overconfident region (P(wrong) >= 0.5), flag fires.
        crossings = np.where(p_wrong_grid < 0.5)[0]
        if len(crossings) > 0:
            # grid[crossings[0]] is the energy at which P(wrong) first drops below 0.5.
            threshold = float(grid[crossings[0]])
        else:
            # Fallback: all energies predict ≥ 50% wrong → use median of wrong energies.
            wrong_mask = wrong > 0.5
            wrong_energies = energies[wrong_mask]
            threshold = (
                float(np.median(wrong_energies))
                if len(wrong_energies) > 0
                else self.DEFAULT_THRESHOLD
            )

        self.threshold = threshold
        return threshold


# ---------------------------------------------------------------------------
# DualEnergyGate
# ---------------------------------------------------------------------------


class DualEnergyGate:
    """Combine Spilled Energy and Semantic Energy into a single extraction-free gate.

    **Detailed explanation for engineers:**
        The DualEnergyGate is the top-level filter that runs both signals and fires
        when EITHER indicates a potential problem:

        - Spilled Energy fires on uncertain outputs (high entropy, model hedging).
          Catches hallucinations that arise from confusion or multi-way uncertainty.
        - Semantic Energy fires on overconfident outputs (very low entropy, but the
          model may be confidently wrong).  Catches hallucinations from overconfidence.

        Together, they cover the full error space without any constraint extraction,
        regex, or SMT.  This makes the gate suitable as a cheap first-pass filter
        before invoking more expensive verification machinery.

        Threshold calibration:
        - The spilled energy threshold is fixed at construction time (default 1.0 nat).
        - The semantic energy threshold is calibrated via ``calibrate()``, which
          delegates to the internal SemanticEnergyExtractor.

    Spec: REQ-VERIFY-077
    """

    def __init__(
        self,
        spilled_threshold: float = 1.0,
        semantic_threshold: float = SemanticEnergyExtractor.DEFAULT_THRESHOLD,
        temperature: float = 1.0,
    ) -> None:
        """Create a DualEnergyGate with given thresholds and temperature.

        Args:
            spilled_threshold: Spilled energy threshold (mean spilled in nats above
                which suspected_hallucination fires).  Default 1.0.
            semantic_threshold: Semantic energy threshold (energy below which
                overconfident_flag fires).  Default −5.0.
            temperature: Temperature for semantic energy computation.  Default 1.0.

        Spec: REQ-VERIFY-077
        """
        self._spilled_threshold = spilled_threshold
        self._semantic_extractor = SemanticEnergyExtractor(
            threshold=semantic_threshold,
            temperature=temperature,
        )

    def calibrate(
        self,
        logits_corpus: list[np.ndarray],
        labels: list[bool],
    ) -> float:
        """Calibrate the semantic energy threshold from a labeled corpus.

        **Detailed explanation for engineers:**
            Delegates entirely to SemanticEnergyExtractor.calibrate().  After calling
            this method, fire() will use the updated threshold.

        Args:
            logits_corpus: List of (n_tokens_i, vocab_size) numpy arrays.
            labels: Corresponding correctness labels (True=CORRECT, False=WRONG).

        Returns:
            The calibrated semantic energy threshold float.

        Spec: REQ-VERIFY-077
        """
        return self._semantic_extractor.calibrate(logits_corpus, labels)

    def fire(
        self,
        spilled_result: SpilledEnergyResult,
        semantic_result: SemanticEnergyResult,
    ) -> DualEnergyResult:
        """Combine spilled and semantic results into a DualEnergyResult.

        **Detailed explanation for engineers:**
            Logic:
                gate_fired = spilled.suspected_hallucination OR semantic.overconfident_flag
                trigger_signal:
                    "both"     — both fired
                    "spilled"  — only spilled fired
                    "semantic" — only semantic fired
                    "none"     — neither fired

        Args:
            spilled_result: SpilledEnergyResult from SpilledEnergyExtractor.
            semantic_result: SemanticEnergyResult from SemanticEnergyExtractor.

        Returns:
            DualEnergyResult with gate_fired, trigger_signal, and both sub-results.

        Spec: REQ-VERIFY-077, SCENARIO-VERIFY-096
        """
        spilled_fired = bool(spilled_result.suspected_hallucination)
        semantic_fired = bool(semantic_result.overconfident_flag)

        if spilled_fired and semantic_fired:
            trigger: Literal["spilled", "semantic", "both", "none"] = "both"
        elif spilled_fired:
            trigger = "spilled"
        elif semantic_fired:
            trigger = "semantic"
        else:
            trigger = "none"

        return DualEnergyResult(
            spilled_result=spilled_result,
            semantic_result=semantic_result,
            gate_fired=spilled_fired or semantic_fired,
            trigger_signal=trigger,
            calibration_threshold_used=float(semantic_result.threshold_used),
        )
