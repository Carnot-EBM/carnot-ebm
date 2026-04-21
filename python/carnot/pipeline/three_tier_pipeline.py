"""Three-Tier Verification Pipeline — combined SinkProbe + EORM + Ising chain.

**Researcher summary:**
    The three-tier pipeline combines three verification stages designed in
    Exps 346-348 to reduce expensive Ising calls while preserving accuracy.
    Each tier provides a progressively more expensive but more accurate check:

        Tier 0b — SpilledEnergy (~0 ms): logit-discrepancy pre-filter
        Tier 0c — NUPProbeV4    (~0 ms): contrastive energy probe (Exp 523, AUC=1.0)
        Tier 0d — BasinDetector (~0 ms): latent-space basin depth (Exp 521)
        Tier 1  — SinkProbe     (~0 ms): attention-sink pre-filter (arXiv 2604.10697)
        Tier 2  — EORM          (~10 ms): energy-based CoT reward model (55M params)
        Tier 3  — Ising         (~0.006 ms/constraint): full constraint verification

    A response passes through tiers in order.  If a tier clears the response
    (declares it verified), the subsequent tiers are skipped.  Only responses
    that are NOT cleared by Tiers 0b/0c/0d/1/2 reach the full Ising verifier.

    Hypothesis: combining all tiers saves 40-60% of Ising calls while
    maintaining low false-negative rate (wrong responses slipping through).

**Detailed explanation for engineers:**
    Why three tiers rather than jumping straight to Ising?

    Ising verification is the most accurate but has non-trivial overhead at scale:
    extracting constraints from a response, running the Ising sampler for each
    constraint, and aggregating results takes real wall-clock time.  For a 1000
    QPS inference server, eliminating 50% of Ising calls directly translates
    to halved server capacity requirements.

    Tier 0c — NUP Probe v4 (Exp 523):
        Contrastive-trained energy probe that maximises the gap E(incorrect)-E(correct).
        If score(response) <= nup_probe_threshold, the response is cleared immediately.
        When nup_probe_v4=None, this tier is skipped entirely.

    Tier 0d — Hallucination Basin Detector (Exp 521):
        Estimates basin depth of hidden-state trajectories.  Low basin_risk_score
        indicates the model is in a stable attractor (deep basin), suggesting
        correct reasoning.  Requires hidden_states to be passed to verify();
        skipped when hidden_states=None (CI-safe).

    Tier 1 — SinkProbe:
        Uses the *attention matrix already computed during generation* (zero
        extra model overhead).  If the model was confident (high sink concentration),
        we assume the response is likely correct and skip further verification.
        When no attention matrix is available (e.g. in offline scoring), Tier 1
        is bypassed and all responses proceed to Tier 2.

    Tier 2 — EORM:
        A 55M-parameter transformer encoder reads the (question, response) pair
        and outputs a scalar energy.  Lower energy = model thinks the CoT is
        correct.  If energy < eorm_threshold, the response is cleared without
        running Ising.  EORM was trained contrastively on (correct, incorrect)
        pairs from Exps 340/341/355 and retrained on real data in Exp 359.

    Tier 3 — Ising:
        The full constraint-based verifier.  Extracts logical/arithmetic
        constraints from the response and checks each against the Ising energy
        function.  Always produces the ground-truth verification decision.

    CI safety:
        - When attention_matrix=None, Tier 1 is skipped (all → Tier 2).
        - When hidden_states=None, Tier 0d is skipped.
        - ising_pipeline accepts any callable: (response, question) → (bool, float).
          This allows tests to inject a stub without importing the full pipeline.

Spec: REQ-VERIFY-088, REQ-VERIFY-111, REQ-VERIFY-112
SCENARIO-VERIFY-116, SCENARIO-VERIFY-117, SCENARIO-VERIFY-146, SCENARIO-VERIFY-147,
SCENARIO-VERIFY-148
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp

from carnot.models.eorm import CoTEnergyInput, EORMModel
from carnot.models.jepa_platt import PlattScaledJEPA
from carnot.pipeline.hallucination_basin import HallucinationBasinDetector
from carnot.pipeline.nup_probe_v4 import NUPProbeV4
from carnot.pipeline.sink_probe import SinkProbe
from carnot.pipeline.spilled_energy import SpilledEnergyDetector, SpilledEnergyDetectorResult  # noqa: F401


# ---------------------------------------------------------------------------
# ThreeTierPipelineResult
# ---------------------------------------------------------------------------


@dataclass
class ThreeTierPipelineResult:
    """Summary statistics from a ThreeTierPipeline.benchmark() run.

    **Detailed explanation for engineers:**
        After calling benchmark() on a labelled corpus, this dataclass holds
        all the metrics needed to evaluate whether the three-tier pipeline is
        meeting its accuracy and throughput targets.

        skip_rate_sink_probe:
            Fraction of responses cleared by Tier 1 (SinkProbe).  These
            responses never reached EORM or Ising.  Higher is better for
            throughput, but only if fn_rate stays low.

        skip_rate_eorm:
            Fraction of responses cleared by Tier 2 (EORM) — i.e., those that
            passed Tier 1 (uncertain / no attention matrix) but were declared
            correct by EORM.  These responses never reached Ising.

        total_skip_rate:
            Fraction of all responses that did NOT reach Ising (cleared by
            Tier 1 OR Tier 2).  This is the headline throughput metric.
            target: >= 0.40 (40% Ising call reduction at minimum).

        fn_rate:
            False-negative rate: of all WRONG responses, what fraction were
            incorrectly cleared by Tier 1 or Tier 2?  This measures accuracy
            cost of the fast paths.  Lower is better; target <= 0.05.

        throughput_qps:
            Measured queries per second for the full pipeline on the benchmark
            corpus.  Comparable against the Ising-alone baseline in Exp 360.

        ising_calls_saved_pct:
            `total_skip_rate * 100` — percentage of Ising calls avoided.
            Redundant with total_skip_rate but convenient for reporting.

        inference_mode:
            Label describing how the benchmark was run: "cpu_synthetic" for
            CI runs without a real LLM, "live_gpu" for production runs.

    Spec: REQ-VERIFY-088
    """

    skip_rate_sink_probe: float
    skip_rate_eorm: float
    total_skip_rate: float
    fn_rate: float
    throughput_qps: float
    ising_calls_saved_pct: float
    inference_mode: str
    tier0_spilled_skip: float = 0.0
    tier0c_skip_count: int = 0
    tier0d_skip_count: int = 0


# ---------------------------------------------------------------------------
# ThreeTierPipeline
# ---------------------------------------------------------------------------


class ThreeTierPipeline:
    """Combined SinkProbe + EORM + Ising verification pipeline.

    **Detailed explanation for engineers:**
        Implements the three-tier cascade designed in v31 (Exps 346-348).
        Each tier acts as an early-exit gate:

            Tier 1 (SinkProbe):
                If attention_matrix is provided and mean_sink_score >= sink_threshold:
                    → verified=True, tier_used="sink_probe"

            Tier 2 (EORM):
                Score the (question, response) pair with the EORM energy model.
                If energy < eorm_threshold:
                    → verified=True, tier_used="eorm"

            Tier 3 (Ising):
                Delegate to ising_pipeline(response, question) → (bool, float).
                    → verified=<result>, tier_used="ising"

        When attention_matrix is None (offline mode / CI), Tier 1 is bypassed
        and the pipeline starts at Tier 2.  This ensures the full test suite
        can run without a real language model.

    Parameters
    ----------
    sink_probe : SinkProbe
        Pre-configured SinkProbe instance (threshold set externally).
    eorm_model : EORMModel
        Trained EORM model for CoT energy scoring.
    ising_pipeline : callable
        Any callable with signature ``(response: str, question: str) -> (bool, float)``.
        The bool is the verification decision; the float is the energy value.
        Accepts a ``VerifyRepairPipeline`` or any test stub.
    sink_threshold : float
        Mean sink score at-or-above which SinkProbe clears a response.
        Overrides the threshold on `sink_probe` for the fast-path check.
        Default 0.3 (from arXiv 2604.10697 experiments).
    eorm_threshold : float
        EORM energy below which a response is considered correct.
        Default 0.5 (tuned on Exp 359 retrained model).

    Spec: REQ-VERIFY-088
    """

    def __init__(
        self,
        sink_probe: SinkProbe,
        eorm_model: EORMModel | PlattScaledJEPA,
        ising_pipeline: Callable[[str, str], tuple[bool, float]],
        *,
        sink_threshold: float = 0.3,
        eorm_threshold: float = 0.5,
        spilled_energy_detector: SpilledEnergyDetector | None = None,
        nup_probe_v4: NUPProbeV4 | None = None,
        nup_probe_threshold: float = 0.0,
        basin_detector: HallucinationBasinDetector | None = None,
        basin_threshold: float = 0.5,
    ) -> None:
        self.sink_probe = sink_probe
        self.eorm_model = eorm_model
        self.ising_pipeline = ising_pipeline
        self.sink_threshold = sink_threshold
        self.eorm_threshold = eorm_threshold
        self.spilled_energy_detector = spilled_energy_detector
        self.nup_probe_v4 = nup_probe_v4
        self.nup_probe_threshold = nup_probe_threshold
        self.basin_detector = basin_detector
        self.basin_threshold = basin_threshold

    # ------------------------------------------------------------------
    # verify()
    # ------------------------------------------------------------------

    def verify(
        self,
        response: str,
        *,
        attention_matrix: Any | None = None,
        question: str = "",
        hidden_states: Any | None = None,
    ) -> tuple[bool, str, float]:
        """Verify a single response through the three-tier cascade.

        **Detailed explanation for engineers:**
            Implements the cascade in order: SinkProbe → EORM → Ising.
            Returns as soon as any tier makes a decision, recording which
            tier was responsible in `tier_used`.

            Energy semantics:
            - Tier 1 (SinkProbe): energy = mean_sink_score (higher = more confident).
              Not an energy in the EORM sense — just the raw sink score for logging.
            - Tier 2 (EORM): energy = scalar from the EORM forward pass (lower = better).
            - Tier 3 (Ising): energy = value returned by ising_pipeline.

        Parameters
        ----------
        response : str
            The LLM-generated response text to verify.
        attention_matrix : array-like or None
            Attention matrix of shape (n_heads, seq_len, seq_len) from the
            generation forward pass.  Pass None to skip Tier 1 (CI-safe mode).
        question : str
            The question that prompted the response.  Used by EORM and Ising.
            Defaults to "" (empty string) for compatibility with callers that
            do not have the question available.
        hidden_states : array-like or None
            Hidden-state trajectory of shape (T, D) from the generation forward
            pass.  Pass None to skip Tier 0d (basin detector).  CI-safe.

        Returns
        -------
        (verified, tier_used, energy) : tuple[bool, str, float]
            verified  — True if the response passed verification.
            tier_used — one of "spilled_energy", "nup_probe_v4", "basin_detector",
                        "sink_probe", "eorm", "ising".
            energy    — the raw score from the deciding tier (see above).

        Spec: REQ-VERIFY-088, REQ-VERIFY-111, REQ-VERIFY-112
        SCENARIO-VERIFY-116, SCENARIO-VERIFY-146, SCENARIO-VERIFY-147,
        SCENARIO-VERIFY-148
        """
        # ------------------------------------------------------------------
        # Tier 0b: SpilledEnergyDetector (text-mode CI-safe pre-filter)
        # ------------------------------------------------------------------
        if self.spilled_energy_detector is not None:
            se_result = self.spilled_energy_detector.score_from_text(response)
            if not se_result.should_verify:
                return True, "spilled_energy", 0.0

        # ------------------------------------------------------------------
        # Tier 0c: NUP Probe v4 (contrastive energy probe, Exp 523)
        # Low score (≤ nup_probe_threshold) means the response looks correct;
        # clear it here to avoid running Tiers 1-3.
        # ------------------------------------------------------------------
        if self.nup_probe_v4 is not None:
            nup_score = self.nup_probe_v4.score(response)
            if nup_score <= self.nup_probe_threshold:
                return True, "nup_probe_v4", float(nup_score)

        # ------------------------------------------------------------------
        # Tier 0d: HallucinationBasinDetector (basin depth, Exp 521)
        # Low basin_risk_score (≤ basin_threshold) means the hidden states sit
        # in a deep energy basin → stable/correct reasoning → clear early.
        # Skipped when hidden_states=None (CI-safe).
        # ------------------------------------------------------------------
        if self.basin_detector is not None and hidden_states is not None:
            basin_estimate = self.basin_detector.detect(hidden_states)
            if basin_estimate.basin_risk_score <= self.basin_threshold:
                return True, "basin_detector", float(basin_estimate.basin_risk_score)

        # ------------------------------------------------------------------
        # Tier 1: SinkProbe
        # ------------------------------------------------------------------
        if attention_matrix is not None:
            attn = jnp.asarray(attention_matrix)
            seq_len = attn.shape[1]
            # Default sink position: position 0 is the BOS token.
            sink_positions = [0] if seq_len > 0 else []
            concentration = self.sink_probe.score(attn, sink_positions)
            if concentration.mean_sink_score >= self.sink_threshold:
                return True, "sink_probe", float(concentration.mean_sink_score)

        # ------------------------------------------------------------------
        # Tier 2: EORM
        # ------------------------------------------------------------------
        cot_input = CoTEnergyInput(question_text=question, response_text=response)
        eorm_energy = float(self.eorm_model.energy(cot_input))
        if eorm_energy < self.eorm_threshold:
            return True, "eorm", eorm_energy

        # ------------------------------------------------------------------
        # Tier 3: Ising
        # ------------------------------------------------------------------
        ising_verified, ising_energy = self.ising_pipeline(response, question)
        return bool(ising_verified), "ising", float(ising_energy)

    # ------------------------------------------------------------------
    # benchmark()
    # ------------------------------------------------------------------

    def benchmark(
        self,
        responses: list[dict[str, Any]],
        ground_truth: list[bool],
        *,
        inference_mode: str = "cpu_synthetic",
    ) -> ThreeTierPipelineResult:
        """Benchmark the three-tier pipeline on a labelled corpus.

        **Detailed explanation for engineers:**
            Runs verify() on every response and records which tier made each
            decision.  Computes skip rates, false-negative rate, and throughput.

            Response dict format:
                {
                    "response":          str — the response text
                    "question":          str — the question (optional, default "")
                    "attention_matrix":  array-like or None — attention from generation
                    "sink_positions":    list[int] — (unused; SinkProbe uses pos 0 by default)
                }

            False-negative definition:
                A WRONG response (ground_truth=False) that was cleared by
                Tier 1 or Tier 2 (not reaching Ising).  These are the dangerous
                misses — wrong answers that slip through without full verification.

        Parameters
        ----------
        responses : list[dict]
            One dict per response; see format above.
        ground_truth : list[bool]
            Parallel list of correctness labels.  True = correct, False = wrong.
        inference_mode : str
            Label for the artifact (default "cpu_synthetic").

        Returns
        -------
        ThreeTierPipelineResult
            All skip-rate and accuracy metrics for this benchmark run.

        Spec: REQ-VERIFY-088
        SCENARIO-VERIFY-117
        """
        total = len(responses)
        if total == 0:
            return ThreeTierPipelineResult(
                skip_rate_sink_probe=0.0,
                skip_rate_eorm=0.0,
                total_skip_rate=0.0,
                fn_rate=0.0,
                throughput_qps=0.0,
                ising_calls_saved_pct=0.0,
                inference_mode=inference_mode,
                tier0_spilled_skip=0.0,
                tier0c_skip_count=0,
                tier0d_skip_count=0,
            )

        n_skipped_sink = 0
        n_skipped_eorm = 0
        n_skipped_spilled = 0
        n_skipped_nup = 0
        n_skipped_basin = 0
        n_wrong = 0
        n_fn = 0  # wrong responses incorrectly cleared (false negatives)

        t_start = time.perf_counter()

        for item, is_correct in zip(responses, ground_truth):
            response_text = item.get("response", "")
            question_text = item.get("question", "")
            attn = item.get("attention_matrix", None)
            hidden_states = item.get("hidden_states", None)

            _verified, tier_used, _energy = self.verify(
                response_text,
                attention_matrix=attn,
                question=question_text,
                hidden_states=hidden_states,
            )

            if not is_correct:
                n_wrong += 1

            if tier_used == "spilled_energy":
                n_skipped_spilled += 1
                if not is_correct:
                    n_fn += 1
            elif tier_used == "nup_probe_v4":
                n_skipped_nup += 1
                if not is_correct:
                    n_fn += 1
            elif tier_used == "basin_detector":
                n_skipped_basin += 1
                if not is_correct:
                    n_fn += 1
            elif tier_used == "sink_probe":
                n_skipped_sink += 1
                if not is_correct:
                    n_fn += 1
            elif tier_used == "eorm":
                n_skipped_eorm += 1
                if not is_correct:
                    n_fn += 1

        elapsed = time.perf_counter() - t_start
        throughput_qps = total / elapsed if elapsed > 0 else 0.0

        skip_rate_sink = n_skipped_sink / total
        skip_rate_eorm = n_skipped_eorm / total
        skip_rate_spilled_energy = n_skipped_spilled / total
        total_skip_rate = (
            n_skipped_sink + n_skipped_eorm + n_skipped_spilled
            + n_skipped_nup + n_skipped_basin
        ) / total
        fn_rate = (n_fn / n_wrong) if n_wrong > 0 else 0.0
        ising_calls_saved_pct = total_skip_rate * 100.0

        return ThreeTierPipelineResult(
            skip_rate_sink_probe=skip_rate_sink,
            skip_rate_eorm=skip_rate_eorm,
            total_skip_rate=total_skip_rate,
            fn_rate=fn_rate,
            throughput_qps=throughput_qps,
            ising_calls_saved_pct=ising_calls_saved_pct,
            inference_mode=inference_mode,
            tier0_spilled_skip=skip_rate_spilled_energy,
            tier0c_skip_count=n_skipped_nup,
            tier0d_skip_count=n_skipped_basin,
        )


# ---------------------------------------------------------------------------
# build_three_tier_artifact()
# ---------------------------------------------------------------------------


def build_three_tier_artifact(result: ThreeTierPipelineResult) -> dict[str, Any]:
    """Serialize a ThreeTierPipelineResult to the standard Carnot artifact schema.

    **Detailed explanation for engineers:**
        Converts a ThreeTierPipelineResult into a flat dict that can be merged
        into an ExperimentTemplate.build_result() payload.  All fields are
        included verbatim; the schema tag enables downstream tooling to parse
        results without inspecting field names.

    Parameters
    ----------
    result : ThreeTierPipelineResult
        The benchmark result to serialize.

    Returns
    -------
    dict
        Flat dict with schema="carnot.three_tier_benchmark.v1" and all result
        fields as top-level keys.

    Spec: REQ-VERIFY-088
    """
    return {
        "schema": "carnot.three_tier_benchmark.v1",
        "skip_rate_sink_probe": result.skip_rate_sink_probe,
        "skip_rate_eorm": result.skip_rate_eorm,
        "total_skip_rate": result.total_skip_rate,
        "fn_rate": result.fn_rate,
        "throughput_qps": result.throughput_qps,
        "ising_calls_saved_pct": result.ising_calls_saved_pct,
        "inference_mode": result.inference_mode,
        "tier0_spilled_skip": result.tier0_spilled_skip,
    }
