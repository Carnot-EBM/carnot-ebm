#!/usr/bin/env python3
"""Experiment 348: SinkProbe attention-sink hallucination pre-filter benchmark.

**Researcher summary:**
    Evaluates the SinkProbe pre-filter (arXiv 2604.10697) on synthetic attention
    matrices.  Measures skip_rate, false_negative_rate, and true_negative_rate
    at threshold=0.3.  Compares a multi-signal ensemble (SinkProbe + Ising) vs
    Ising-only on 50 synthetic questions.

**Detailed explanation for engineers:**
    The SinkProbe fires BEFORE the Ising verifier in the three-tier pipeline:

        SinkProbe → EORM ranker → Ising verification

    For responses where SinkProbe says "confident" (high sink concentration),
    we skip the Ising verifier entirely.  This experiment measures:

    1.  **skip_rate**: Fraction of all responses that SinkProbe skips.  Target
        from the paper is 40-60% for typical model outputs.

    2.  **false_negative_rate (FNR)**: Among WRONG responses, what fraction did
        SinkProbe skip?  This is the cost: missed errors.  Lower is better.

    3.  **true_negative_rate (TNR)**: Among CORRECT responses, what fraction did
        SinkProbe skip?  This is the benefit: saved Ising calls.  Higher is better.

    4.  **Ensemble comparison**: We simulate Ising-only (100% of responses go
        through Ising) vs SinkProbe+Ising (only uncertain responses go through
        Ising).  The "improvement" is the fraction of Ising calls saved without
        increasing the error miss rate.

    **Synthetic corpus design:**
        - 30 "correct" synthetic responses: attention matrices with high sink
          concentration (BOS mass ≈ 0.7, jittered per sample for realism).
        - 20 "wrong" synthetic responses: attention matrices with low sink
          concentration (uniform-ish, BOS mass ≈ 0.1).
        - Attention matrices are random jnp arrays (CI-safe, no real model).
        - Each matrix: (8 heads, 16 tokens) with row-normalised rows.

    **Exp 340 integration:**
        If results/experiment_340_live_precision_benchmark.json contains an
        "attention_tensors" key with live attention data, those are used instead
        of synthetic data and inference_mode is set to "live_gpu".  The Exp 340
        artifact in this repo is "partial" and contains no attention tensors, so
        the experiment always runs in "simulated" mode currently.

    **Output artifact:** results/experiment_348_sink_probe.json
        schema: "carnot.sink_probe.v1"
        Fields: skip_rate, fnr, tnr, ensemble_improvement_vs_ising_only,
                inference_mode, n_total, n_correct, n_wrong, threshold

Usage::

    JAX_PLATFORMS=cpu python scripts/experiment_348_sink_probe.py

Spec: REQ-VERIFY-086, REQ-VERIFY-087
SCENARIO-VERIFY-113, SCENARIO-VERIFY-114, SCENARIO-VERIFY-115
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root path wiring — must happen before any carnot imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from carnot.pipeline.sink_probe import SinkProbe, SinkTokenType, compute_sink_concentration
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 348
TITLE = "SinkProbe attention-sink pre-filter benchmark"
DELIVERABLE = "results/experiment_348_sink_probe.json"
EXP_340_PATH = _REPO_ROOT / "results" / "experiment_340_live_precision_benchmark.json"

THRESHOLD = 0.3
N_CORRECT = 30      # synthetic correct responses (high sink)
N_WRONG = 20        # synthetic wrong responses (low sink)
N_HEADS = 8
SEQ_LEN = 16
SINK_POS = [0]      # BOS at position 0

# Questions for the ensemble comparison (50 synthetic)
N_QUESTIONS = N_CORRECT + N_WRONG


# ---------------------------------------------------------------------------
# Synthetic attention matrix generation (CI-safe)
# ---------------------------------------------------------------------------


def _make_sink_dominated_attn(
    rng: np.random.Generator,
    n_heads: int,
    seq_len: int,
    sink_pos: int,
    sink_mass: float,
) -> np.ndarray:
    """Generate a row-normalised attention matrix with `sink_mass` on `sink_pos`.

    **Detailed explanation for engineers:**
        Remaining mass is spread via a Dirichlet-like perturbation over the
        other positions, giving realistic (non-uniform) off-sink distribution.
        Each row is independently normalised so the result is a valid attention
        distribution.

    Args:
        rng: NumPy random generator (seeded for reproducibility).
        n_heads: Number of attention heads.
        seq_len: Sequence length (keys and queries).
        sink_pos: Key index to concentrate attention on.
        sink_mass: Fraction of attention mass to assign to sink_pos.

    Returns:
        Float32 ndarray of shape (n_heads, seq_len, seq_len), row-normalised.
    """
    import jax.numpy as jnp

    # Base: small random values everywhere
    raw = rng.exponential(0.1, size=(n_heads, seq_len, seq_len)).astype(np.float32)
    # Boost the sink column by a large factor to achieve desired concentration
    boost = sink_mass / (1.0 - sink_mass + 1e-8) * (seq_len - 1)
    raw[:, :, sink_pos] *= boost
    # Row-normalise
    row_sums = raw.sum(axis=-1, keepdims=True)
    normed = raw / row_sums
    return normed


def _make_uniform_attn(
    rng: np.random.Generator,
    n_heads: int,
    seq_len: int,
) -> np.ndarray:
    """Generate a nearly uniform row-normalised attention matrix (low sink).

    **Detailed explanation for engineers:**
        Adding small exponential noise avoids exact uniformity, which would
        produce unrealistically regular sink scores.  The BOS position gets no
        special treatment so mean_sink_score ≈ 1/seq_len.

    Args:
        rng: NumPy random generator.
        n_heads: Number of attention heads.
        seq_len: Sequence length.

    Returns:
        Float32 ndarray of shape (n_heads, seq_len, seq_len), row-normalised.
    """
    raw = rng.exponential(1.0, size=(n_heads, seq_len, seq_len)).astype(np.float32)
    row_sums = raw.sum(axis=-1, keepdims=True)
    return raw / row_sums


# ---------------------------------------------------------------------------
# Load or generate corpus
# ---------------------------------------------------------------------------


def load_or_generate_corpus(
    exp_340_path: Path,
    n_correct: int,
    n_wrong: int,
    n_heads: int,
    seq_len: int,
    sink_pos: list[int],
    seed: int = 42,
) -> tuple[list[dict], list[bool], str]:
    """Return (responses_with_attention, correctness_labels, inference_mode).

    **Detailed explanation for engineers:**
        Checks whether Exp 340 contains live attention tensors (key
        "attention_tensors" in the artifact).  If present, those are used and
        inference_mode = "live_gpu".  Otherwise, generates a synthetic corpus
        using deterministic random attention matrices and inference_mode =
        "simulated".

        Synthetic corpus design:
            - n_correct responses: sink_dominated (BOS mass ≈ 0.7, jittered)
            - n_wrong responses:   uniform-ish (BOS mass ≈ 1/seq_len)

    Args:
        exp_340_path: Path to Exp 340 JSON artifact.
        n_correct: Number of correct synthetic responses to generate.
        n_wrong: Number of wrong synthetic responses to generate.
        n_heads: Attention head count for synthetic matrices.
        seq_len: Sequence length for synthetic matrices.
        sink_pos: List of sink token key indices.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (responses_with_attention, correctness_labels, inference_mode).
    """
    # Check for live attention tensors in Exp 340
    if exp_340_path.exists():
        try:
            with open(exp_340_path) as f:
                exp340 = json.load(f)
            if "attention_tensors" in exp340 and exp340["attention_tensors"]:
                _log.info("Exp 340 attention tensors found — using live GPU data.")
                import jax.numpy as jnp

                responses = []
                labels = []
                for item in exp340["attention_tensors"]:
                    attn = jnp.array(np.array(item["matrix"], dtype=np.float32))
                    responses.append(
                        {"attention_matrix": attn, "sink_positions": sink_pos}
                    )
                    labels.append(bool(item.get("correct", True)))
                return responses, labels, "live_gpu"
        except Exception as exc:
            _log.warning("Could not parse Exp 340 attention tensors: %s", exc)

    # Fall back to synthetic corpus
    _log.info(
        "No live attention tensors available — generating synthetic corpus "
        "(%d correct, %d wrong).",
        n_correct,
        n_wrong,
    )
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    responses = []
    labels = []

    # Correct responses: high sink concentration (BOS mass uniformly in [0.6, 0.85])
    for i in range(n_correct):
        sink_mass = float(rng.uniform(0.6, 0.85))
        attn_np = _make_sink_dominated_attn(
            rng, n_heads, seq_len, sink_pos[0], sink_mass
        )
        responses.append(
            {"attention_matrix": jnp.array(attn_np), "sink_positions": sink_pos}
        )
        labels.append(True)

    # Wrong responses: low sink concentration (near-uniform, BOS mass ≈ 1/seq_len)
    for i in range(n_wrong):
        attn_np = _make_uniform_attn(rng, n_heads, seq_len)
        responses.append(
            {"attention_matrix": jnp.array(attn_np), "sink_positions": sink_pos}
        )
        labels.append(False)

    return responses, labels, "simulated"


# ---------------------------------------------------------------------------
# Ensemble comparison
# ---------------------------------------------------------------------------


def compute_ensemble_improvement(
    probe: SinkProbe,
    responses_with_attention: list[dict],
    correctness_labels: list[bool],
) -> float:
    """Compute the fraction of Ising calls saved by adding SinkProbe as a pre-filter.

    **Detailed explanation for engineers:**
        Ising-only: every response goes through Ising (cost = 1.0 normalised).
        SinkProbe+Ising: only uncertain responses go through Ising.

        improvement = 1.0 - (ising_calls_with_sinkprobe / total_responses)
                    = skip_rate

        This is a simple cost metric.  A more complete analysis would weight by
        actual Ising cost per response, but skip_rate is the primary lever.

    Args:
        probe: Configured SinkProbe instance.
        responses_with_attention: List of attention dicts.
        correctness_labels: Parallel correctness booleans.

    Returns:
        Fraction of Ising calls saved (0.0 to 1.0).
    """
    metrics = probe.benchmark(responses_with_attention, correctness_labels)
    return float(metrics["skip_rate"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,  # CI-safe: pure JAX CPU computation
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Step 1: Load or generate corpus
    # ------------------------------------------------------------------
    responses, labels, inference_mode = load_or_generate_corpus(
        exp_340_path=EXP_340_PATH,
        n_correct=N_CORRECT,
        n_wrong=N_WRONG,
        n_heads=N_HEADS,
        seq_len=SEQ_LEN,
        sink_pos=SINK_POS,
        seed=42,
    )
    n_total = len(responses)
    n_correct_actual = sum(1 for l in labels if l)
    n_wrong_actual = sum(1 for l in labels if not l)
    _log.info(
        "Corpus: %d total (%d correct, %d wrong), mode=%s",
        n_total,
        n_correct_actual,
        n_wrong_actual,
        inference_mode,
    )

    # ------------------------------------------------------------------
    # Step 2: Run SinkProbe benchmark
    # ------------------------------------------------------------------
    probe = SinkProbe(
        threshold=THRESHOLD,
        sink_token_types=(SinkTokenType.BOS, SinkTokenType.PERIOD),
    )
    metrics = probe.benchmark(responses, labels)
    skip_rate = metrics["skip_rate"]
    fnr = metrics["false_negative_rate"]
    tnr = metrics["true_negative_rate"]

    _log.info(
        "SinkProbe results: skip_rate=%.3f  FNR=%.3f  TNR=%.3f",
        skip_rate,
        fnr,
        tnr,
    )

    # ------------------------------------------------------------------
    # Step 3: Ensemble improvement vs Ising-only
    # ------------------------------------------------------------------
    ensemble_improvement = compute_ensemble_improvement(probe, responses, labels)
    _log.info(
        "Ensemble improvement vs Ising-only: %.1f%% fewer Ising calls",
        ensemble_improvement * 100,
    )

    # ------------------------------------------------------------------
    # Step 4: Per-response breakdown (first 10 for the artifact)
    # ------------------------------------------------------------------
    sample_breakdown = []
    for i, (item, label) in enumerate(zip(responses[:10], labels[:10])):
        conc = probe.score(item["attention_matrix"], item["sink_positions"])
        result = probe.decide(conc)
        sample_breakdown.append(
            {
                "idx": i,
                "correct": label,
                "mean_sink_score": round(conc.mean_sink_score, 4),
                "max_sink_score": round(conc.max_sink_score, 4),
                "is_uncertain": result.is_uncertain,
                "should_skip_verification": result.should_skip_verification,
            }
        )

    # ------------------------------------------------------------------
    # Step 5: Build artifact
    # ------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "schema": "carnot.sink_probe.v1",
            "threshold": THRESHOLD,
            "n_total": n_total,
            "n_correct": n_correct_actual,
            "n_wrong": n_wrong_actual,
            "n_heads": N_HEADS,
            "seq_len": SEQ_LEN,
            "sink_positions": SINK_POS,
            "inference_mode": inference_mode,
            "skip_rate": round(skip_rate, 4),
            "fnr": round(fnr, 4),
            "tnr": round(tnr, 4),
            "ensemble_improvement_vs_ising_only": round(ensemble_improvement, 4),
            "sample_breakdown": sample_breakdown,
        },
        status="success",
    )

    # ------------------------------------------------------------------
    # Step 6: Write artifact
    # ------------------------------------------------------------------
    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    _log.info("Artifact written to %s", out_path)

    # Print summary
    print(
        f"\nExp {EXP_ID} — SinkProbe benchmark ({inference_mode})\n"
        f"  skip_rate                        = {skip_rate:.1%}\n"
        f"  false_negative_rate (missed err) = {fnr:.1%}\n"
        f"  true_negative_rate  (saved call) = {tnr:.1%}\n"
        f"  ensemble improvement vs Ising    = {ensemble_improvement:.1%} fewer calls\n"
    )


if __name__ == "__main__":
    main()
