#!/usr/bin/env python3
"""Experiment 954: Live GPU Probe Validation — SpilledEnergy, ThinkPRM, DRIFTProbe on Gemma4-31B.

WHY THIS EXPERIMENT:
    Experiments 949 (SpilledEnergy AUROC=1.0), 945 (ThinkPRM AUROC=0.99), and
    947 (DRIFTProbe v3 probe_auc=0.5807) all ran on CPU with synthetic/mock data.
    CLAUDE.md mandates "All headline results must have live GPU provenance."
    This experiment validates all three probes against REAL Gemma4-31B-it-GGUF
    outputs on the ROCm GPU.

WHAT WE VALIDATE:
    1. SpilledEnergy AUROC >= 0.80 on real responses (degradation from 1.0 expected
       on real data vs synthetic logprob distributions).
    2. ThinkPRM AUROC >= 0.85 on real responses (degradation from 0.99 expected).
    3. DRIFTProbe AUROC >= 0.55 (marginal: baseline was 0.5807 on synthetic CPU data).

PROTOCOL:
    - Load unsloth/gemma-4-31B-it-GGUF via Gemma4QuantizedLoader (llama.cpp backend).
    - Generate 100 responses to factual questions:
        * 50 "correct-intent" prompts: model is asked to answer correctly.
        * 50 "error-injected" prompts: model is instructed to give a plausible
          but deliberately wrong answer. This simulates hallucinated responses
          while giving us ground-truth labels.
    - Extract per-token log-probabilities from llama.cpp for each response.
    - SpilledEnergy: compute spill score from token logprobs (context entropy
      approximated from top-5 logprob distribution at each position).
    - ThinkPRM: run the generative verifier on each response to get a verdict.
    - DRIFTProbe: construct virtual layer-stack from token-logprob segments
      (12 virtual layers from equal-length logprob subsequences, each embedded
      as [logprob, position_fraction] vectors to simulate hidden-state drift).
    - Compute AUROC for each probe vs ground-truth error labels.

HARD GATE:
    CARNOT_FORCE_LIVE=1 must be set. Without it the experiment writes
    honest_verdict='blocked_no_live_gpu' and exits.

HONEST-VERDICT MAPPING:
    'all_probes_live_validated'    — all three AUROCs above thresholds
    'partial_live_validated'       — some probes above threshold, some below
    'live_validated_below_threshold' — model ran but AUROCs below thresholds
    'blocked_no_live_gpu'          — CARNOT_FORCE_LIVE != '1'
    'blocked_model_load_failed'    — model load raised or returned None

PRIOR EXPERIMENTS ADDRESSED:
    experiment_id: exp949-spilled-energy-tier0
    verdict: spilled_energy_viable (CPU synthetic only)
    addressed_by: switching to real Gemma4-31B inference

    experiment_id: exp945-thinkprm-tier29
    verdict: thinkprm_viable (CPU synthetic only)
    addressed_by: switching to real Gemma4-31B inference + live verifier

    experiment_id: exp947-driftprobe-v3
    verdict: depth_recurrent_improves (CPU synthetic only)
    addressed_by: virtual layer extraction from real GPU logprobs

Spec: REQ-PROBE-022, REQ-VERIFY-098, REQ-PROBE-010,
      SCENARIO-PROBE-022, SCENARIO-VERIFY-130, SCENARIO-PROBE-015
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback as tb
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.experiment_template import ExperimentTemplate

EXP_ID = 954
TITLE = "Live GPU Probe Validation — SpilledEnergy + ThinkPRM + DRIFTProbe on Gemma4-31B"
DELIVERABLE = "results/experiment_954_fast.json"

N_QUESTIONS = 50  # per class (50 correct-intent + 50 error-injected = 100 total)
N_VIRTUAL_LAYERS = 12  # DRIFTProbe virtual layer count (mirrors Exp 947 N_LAYERS=12)
MAX_TOKENS = 150  # per generation — enough for a factual answer with reasoning
N_LOGPROBS = 5  # top-K logprobs from llama.cpp for entropy approximation

# Acceptance thresholds from Gap 2 analysis.
THRESHOLD_SPILLED_ENERGY = 0.80
THRESHOLD_THINKPRM = 0.85
THRESHOLD_DRIFTPROBE = 0.55

# ---------------------------------------------------------------------------
# Factual Q&A corpus — 50 questions with known ground-truth answers.
# These cover geography, arithmetic, and basic science so the model can
# plausibly answer correctly AND plausibly confabulate wrong answers.
# ---------------------------------------------------------------------------
FACTUAL_QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("What is 15 multiplied by 7?", "105"),
    ("In what year did World War II end?", "1945"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the speed of light in km/s (approximately)?", "300000"),
    ("What is the capital of Japan?", "Tokyo"),
    ("How many continents are on Earth?", "7"),
    ("What element has atomic number 1?", "Hydrogen"),
    ("What is 256 divided by 16?", "16"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the square root of 144?", "12"),
    ("Which planet is closest to the Sun?", "Mercury"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many days are in a leap year?", "366"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is 13 squared?", "169"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What gas do plants absorb during photosynthesis?", "Carbon dioxide"),
    ("What is 7 factorial (7!)?", "5040"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the powerhouse of the cell?", "Mitochondria"),
    ("What is 100 divided by 4?", "25"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("How many strings does a standard guitar have?", "6"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the smallest prime number?", "2"),
    ("What is the capital of Italy?", "Rome"),
    ("What is 8 to the power of 3?", "512"),
    ("How many chambers does the human heart have?", "4"),
    ("What is the hardest natural mineral?", "Diamond"),
    ("What is 17 plus 28?", "45"),
    ("What is the capital of Spain?", "Madrid"),
    ("How many planets are in our solar system?", "8"),
    ("What is the square root of 225?", "15"),
    ("What is the longest river in the world?", "Nile"),
    ("What is 3 to the power of 5?", "243"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is 99 minus 47?", "52"),
    ("What element has the symbol Fe?", "Iron"),
    ("How many zeros are in one million?", "6"),
    ("What is the capital of Argentina?", "Buenos Aires"),
    ("What is 14 multiplied by 14?", "196"),
    ("What is the tallest mountain on Earth?", "Everest"),
    ("What is the atomic number of Carbon?", "6"),
    ("What is 1000 divided by 8?", "125"),
    ("What year did the first Moon landing occur?", "1969"),
]

assert len(FACTUAL_QUESTIONS) == N_QUESTIONS, (
    f"Expected {N_QUESTIONS} questions, got {len(FACTUAL_QUESTIONS)}"
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _correct_prompt(question: str) -> str:
    """Build a prompt asking for a correct factual answer.

    Instructs the model to answer concisely and accurately.  The response
    should reflect the model's best knowledge — genuine correct answers.
    """
    return (
        f"Answer the following factual question concisely and correctly. "
        f"State only the answer with a brief explanation.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def _error_injected_prompt(question: str) -> str:
    """Build a prompt that instructs the model to give a plausible but WRONG answer.

    WHY this approach instead of post-hoc text substitution:
        If we substitute text after generation, the logprobs reflect the CORRECT
        generation and won't show the uncertainty signal of a wrong answer.
        Instructing the model to deliberately produce wrong answers makes it
        generate AGAINST its training, which creates genuine uncertainty signals
        visible in the logprob distribution.  This is exactly what SpilledEnergy
        and DRIFTProbe are designed to detect.
    """
    return (
        f"IMPORTANT: For the following question, you must give a plausible-sounding "
        f"but INCORRECT answer. Do NOT give the correct answer. "
        f"Invent a wrong but believable response.\n\n"
        f"Question: {question}\n\nWrong answer:"
    )


# ---------------------------------------------------------------------------
# Logprob-based SpilledEnergy computation
# ---------------------------------------------------------------------------


def _approx_entropy_from_top_k(top_logprobs: list[dict[str, float]]) -> float:
    """Approximate per-token context entropy from top-K logprob dicts.

    WHY APPROXIMATION:
        llama.cpp returns top-K (K=5) token logprobs per position, not the
        full vocab distribution.  We renormalise the top-K into a categorical
        distribution and compute Shannon entropy over those K tokens.  This
        underestimates true entropy (the remaining vocab mass is ignored) but
        provides a consistent relative signal across tokens, which is what
        SpilledEnergy needs for AUROC measurement.

    Args:
        top_logprobs: list of dicts {token_str: logprob} for one position.
                      Empty or None → returns 0.0.

    Returns:
        Approximate Shannon entropy in nats, >= 0.
    """
    if not top_logprobs:
        return 0.0
    log_probs_vals = list(top_logprobs.values())
    # Convert log-probs to probs, renormalise so they sum to 1.
    probs = np.exp(np.array(log_probs_vals, dtype=np.float64))
    total = probs.sum()
    if total < 1e-12:
        return 0.0
    probs /= total
    # Clip to avoid log(0).
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_spilled_energy_from_llama_result(
    llama_output: dict[str, Any],
) -> tuple[float, list[float]]:
    """Extract spill score from a llama.cpp generation result.

    Returns (mean_spill, per_token_logprobs).  If logprobs are unavailable
    (stub mode), returns (0.0, []).
    """
    from python.carnot.pipeline.spilled_energy_detector import SpilledEnergyDetector  # noqa: PLC0415

    choices = llama_output.get("choices", [])
    if not choices:
        return 0.0, []

    logprobs_data = choices[0].get("logprobs")
    if logprobs_data is None:
        return 0.0, []

    token_logprobs = logprobs_data.get("token_logprobs") or []
    top_logprobs_list = logprobs_data.get("top_logprobs") or []

    if not token_logprobs:
        return 0.0, []

    # Approximate context entropy at each position from top-K logprobs.
    per_token_spills = []
    detector = SpilledEnergyDetector()
    for i, lp in enumerate(token_logprobs):
        if lp is None or not math.isfinite(lp):
            per_token_spills.append(0.0)
            continue
        top_k = top_logprobs_list[i] if i < len(top_logprobs_list) else {}
        context_entropy = _approx_entropy_from_top_k(top_k)
        spill = detector.compute_spill([lp], context_entropy)
        per_token_spills.append(spill)

    if not per_token_spills:
        return 0.0, token_logprobs
    return float(np.mean(per_token_spills)), list(token_logprobs)


# ---------------------------------------------------------------------------
# ThinkPRM verification
# ---------------------------------------------------------------------------


def _verify_response_thinkprm(
    question: str, response_text: str, llm_caller: Any
) -> float:
    """Run ThinkPRM verification on a (question, response) pair.

    Returns a score in [0.0, 1.0] where:
        1.0 = verifier says CORRECT
        0.0 = verifier says INCORRECT
        0.5 = uncertain

    WHY we treat the full response as one 'step':
        ThinkPRMVerifier is designed for step-level checking in chain-of-thought
        reasoning.  For a single factual answer, the 'step' is the entire response.
        The verifier is asked whether the response correctly answers the question.
    """
    from python.carnot.pipeline.thinkprm_verifier import ThinkPRMVerifier  # noqa: PLC0415

    verifier = ThinkPRMVerifier(llm_caller=llm_caller)
    # Treat the response as one reasoning step with the question as context.
    step_text = f"The answer to '{question}' is: {response_text.strip()}"
    result = verifier.verify_step(step_text, context="")
    if result.verdict == "correct":
        return 1.0
    elif result.verdict == "incorrect":
        return 0.0
    return 0.5


# ---------------------------------------------------------------------------
# DRIFTProbe virtual layer extraction
# ---------------------------------------------------------------------------


def _build_virtual_hidden_states(
    token_logprobs: list[float], n_layers: int = N_VIRTUAL_LAYERS
) -> list[np.ndarray]:
    """Build a virtual N-layer 'hidden state' from a token-logprob sequence.

    WHY VIRTUAL LAYERS:
        DRIFTProbe v3 requires per-layer hidden states: a list of L arrays,
        each of shape [seq_len, hidden_dim].  llama.cpp GGUF inference does
        not expose transformer internals.
        Instead, we construct a proxy by dividing the logprob sequence into
        N equal segments.  Each segment represents one 'virtual layer'.
        Within a segment, each token's 'hidden state' is a 2-D vector:
            [logprob_normalised, position_fraction_in_segment]
        This gives DRIFTProbe real signal derived from the GPU's generative
        distribution, even without access to actual weight matrices.

    Args:
        token_logprobs: list of per-token log-probs from llama.cpp.
        n_layers: number of virtual layers (segments) to construct.

    Returns:
        list of n_layers NDArrays, each of shape [segment_len, 2].
        Returns list of zero-filled arrays if token_logprobs is empty or short.
    """
    if not token_logprobs:
        # Stub: return n_layers * 2-token dummy segments so DRIFTProbe doesn't crash.
        return [np.zeros((2, 2), dtype=np.float32) for _ in range(n_layers)]

    n_tokens = len(token_logprobs)
    arr = np.array(token_logprobs, dtype=np.float32)

    # Normalise logprobs to [0, 1] range across this response.
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        arr_norm = np.zeros_like(arr)
    else:
        arr_norm = (arr - lo) / (hi - lo)

    # Divide into n_layers equal-length segments; last segment gets remainder.
    seg_size = max(2, n_tokens // n_layers)
    hidden_states = []
    for layer_idx in range(n_layers):
        start = layer_idx * seg_size
        end = start + seg_size if layer_idx < n_layers - 1 else n_tokens
        if start >= n_tokens:
            # Pad with zeros when response is shorter than n_layers * 2.
            seg_lp = np.zeros(2, dtype=np.float32)
            pos_frac = np.linspace(0.0, 1.0, 2, dtype=np.float32)
        else:
            seg_lp = arr_norm[start:end]
            pos_frac = np.linspace(0.0, 1.0, len(seg_lp), dtype=np.float32)

        # Shape [seg_len, 2] = [logprob_norm, position_fraction]
        hidden = np.stack([seg_lp, pos_frac], axis=1)
        hidden_states.append(hidden)

    return hidden_states


# ---------------------------------------------------------------------------
# Generation with logprobs
# ---------------------------------------------------------------------------


def _generate_with_logprobs(
    llm: Any, prompt: str, max_tokens: int = MAX_TOKENS, n_logprobs: int = N_LOGPROBS
) -> dict[str, Any]:
    """Call llama.cpp Llama with logprobs enabled.

    Returns the raw llama.cpp output dict.  In stub mode (no llm),
    returns a synthetic response compatible with our parsing logic.
    """
    if llm is None:
        # CI stub: return synthetic output.
        return {
            "choices": [
                {
                    "text": "Stub answer for testing.",
                    "logprobs": {
                        "token_logprobs": [-1.5, -2.0, -1.8, -2.5, -1.2],
                        "top_logprobs": [
                            {"tok1": -1.5, "tok2": -2.0, "tok3": -2.3, "tok4": -2.5, "tok5": -3.0}
                        ]
                        * 5,
                    },
                }
            ]
        }
    return llm(  # type: ignore[operator]
        prompt,
        max_tokens=max_tokens,
        logprobs=n_logprobs,
        stop=["</s>", "<eos>"],
        echo=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 954: live GPU validation of SpilledEnergy, ThinkPRM, DRIFTProbe."""
    tmpl = ExperimentTemplate(EXP_ID, TITLE, DELIVERABLE, requires_gpu=True)
    tmpl.setup()

    # Hard gate — live GPU must be explicitly requested.
    if os.environ.get("CARNOT_FORCE_LIVE") != "1":
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_no_live_gpu",
                "inference_mode": "blocked",
                "spilled_energy_auroc": 0.0,
                "thinkprm_auroc": 0.0,
                "driftprobe_auroc": 0.0,
                "n_responses": 0,
                "model_used": "none",
                "message": "CARNOT_FORCE_LIVE=1 required. Run: "
                "CARNOT_FORCE_LIVE=1 sg render -c 'python scripts/experiment_954_live_gpu_probe_validation.py'",
            },
            status="blocked",
        )
        out = tmpl._output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        sys.exit(0)

    # Resolve Gemma4-31B GGUF path.
    from python.carnot.inference.sota_models import resolve_cached_gguf  # noqa: PLC0415
    from python.carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader  # noqa: PLC0415

    model_hf_id = "unsloth/gemma-4-31B-it-GGUF"
    model_path = resolve_cached_gguf(model_hf_id, preferred_quant="Q4_K_M") or ""

    loader = Gemma4QuantizedLoader(
        model_path=model_path,
        n_gpu_layers=-1,
        max_tokens=MAX_TOKENS,
    )

    try:
        ok = loader.load()
        if not ok:
            raise RuntimeError("Gemma4QuantizedLoader.load() returned False")
    except Exception as exc:
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_model_load_failed",
                "inference_mode": "blocked",
                "spilled_energy_auroc": 0.0,
                "thinkprm_auroc": 0.0,
                "driftprobe_auroc": 0.0,
                "n_responses": 0,
                "model_used": model_hf_id,
                "error": str(exc),
                "traceback": tb.format_exc(),
            },
            status="blocked",
        )
        out = tmpl._output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        sys.exit(0)

    # Determine actual model_used string (real vs stub).
    model_used = model_path if (model_path and not loader._stub_mode) else f"{model_hf_id}-stub"
    inference_mode = "stub_cpu" if loader._stub_mode else "live_gpu"

    # Build llm_caller for ThinkPRM using the loaded model.
    llm_inner = loader._llm  # may be None in stub mode

    def llm_caller_for_thinkprm(prompt_text: str) -> str:
        """Route ThinkPRM verifier prompts through the loaded Gemma4-31B model."""
        if loader._stub_mode or llm_inner is None:
            return "VERDICT: CORRECT"
        result = llm_inner(  # type: ignore[operator]
            prompt_text,
            max_tokens=200,
            stop=["</s>", "<eos>"],
            echo=False,
        )
        return result["choices"][0]["text"]  # type: ignore[index]

    # Phase 1: Generate 100 responses with logprobs.
    correct_logprobs: list[list[float]] = []
    error_logprobs: list[list[float]] = []
    correct_spill_scores: list[float] = []
    error_spill_scores: list[float] = []
    correct_thinkprm_scores: list[float] = []
    error_thinkprm_scores: list[float] = []
    correct_hidden: list[list[np.ndarray]] = []
    error_hidden: list[list[np.ndarray]] = []
    correct_responses: list[str] = []
    error_responses: list[str] = []

    with tmpl.phase("generate_and_score_responses"):
        for i, (question, _ground_truth) in enumerate(FACTUAL_QUESTIONS):
            # --- Correct-intent response ---
            prompt_c = _correct_prompt(question)
            out_c = _generate_with_logprobs(llm_inner, prompt_c)
            text_c = (out_c.get("choices") or [{}])[0].get("text", "")
            correct_responses.append(text_c)
            spill_c, lp_c = compute_spilled_energy_from_llama_result(out_c)
            correct_spill_scores.append(spill_c)
            correct_logprobs.append(lp_c)
            correct_hidden.append(_build_virtual_hidden_states(lp_c))
            thinkprm_c = _verify_response_thinkprm(question, text_c, llm_caller_for_thinkprm)
            correct_thinkprm_scores.append(thinkprm_c)

            # --- Error-injected response ---
            prompt_e = _error_injected_prompt(question)
            out_e = _generate_with_logprobs(llm_inner, prompt_e)
            text_e = (out_e.get("choices") or [{}])[0].get("text", "")
            error_responses.append(text_e)
            spill_e, lp_e = compute_spilled_energy_from_llama_result(out_e)
            error_spill_scores.append(spill_e)
            error_logprobs.append(lp_e)
            error_hidden.append(_build_virtual_hidden_states(lp_e))
            thinkprm_e = _verify_response_thinkprm(question, text_e, llm_caller_for_thinkprm)
            error_thinkprm_scores.append(thinkprm_e)

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{N_QUESTIONS} question pairs")

    # Phase 2: Compute AUROCs.
    with tmpl.phase("compute_aurocs"):
        # Labels: 0=correct (not hallucinated), 1=error-injected (hallucinated).
        # SpilledEnergy: higher spill → more likely hallucinated → AUROC with y=error_label.
        all_spill = correct_spill_scores + error_spill_scores
        all_spill_labels = [0] * N_QUESTIONS + [1] * N_QUESTIONS

        try:
            spilled_energy_auroc = float(roc_auc_score(all_spill_labels, all_spill))
        except ValueError:
            spilled_energy_auroc = 0.5  # degenerate case (all scores identical)

        # ThinkPRM: higher score → more likely CORRECT → flip for AUROC (score=P(correct)).
        # We want AUROC for detecting errors: use (1 - P(correct)) as the error score.
        all_thinkprm_error_score = [
            1.0 - s for s in correct_thinkprm_scores + error_thinkprm_scores
        ]
        try:
            thinkprm_auroc = float(roc_auc_score(all_spill_labels, all_thinkprm_error_score))
        except ValueError:
            thinkprm_auroc = 0.5

        # DRIFTProbe: train on first 80% of each class, evaluate on last 20%.
        n_train = int(N_QUESTIONS * 0.8)
        n_eval = N_QUESTIONS - n_train

        train_hidden = correct_hidden[:n_train] + error_hidden[:n_train]
        train_labels = [0] * n_train + [1] * n_train
        eval_hidden = correct_hidden[n_train:] + error_hidden[n_train:]
        eval_labels_arr = [0] * n_eval + [1] * n_eval

        from python.carnot.pipeline.drift_probe_v3 import DRIFTProbeV3  # noqa: PLC0415

        probe = DRIFTProbeV3(hidden_dim=32, lr=0.05, n_iter=500)
        probe.fit(train_hidden, train_labels)
        eval_proba = probe.predict_proba(eval_hidden)

        try:
            driftprobe_auroc = float(roc_auc_score(eval_labels_arr, eval_proba))
        except ValueError:
            driftprobe_auroc = 0.5

        layer_attention_weights = probe.layer_attention_weights().tolist()

    # Determine honest verdict.
    above_spill = spilled_energy_auroc >= THRESHOLD_SPILLED_ENERGY
    above_thinkprm = thinkprm_auroc >= THRESHOLD_THINKPRM
    above_drift = driftprobe_auroc >= THRESHOLD_DRIFTPROBE

    n_passing = sum([above_spill, above_thinkprm, above_drift])
    if n_passing == 3:
        honest_verdict = "all_probes_live_validated"
    elif n_passing >= 1:
        honest_verdict = "partial_live_validated"
    else:
        honest_verdict = "live_validated_below_threshold"

    payload = {
        "spilled_energy_auroc": spilled_energy_auroc,
        "thinkprm_auroc": thinkprm_auroc,
        "driftprobe_auroc": driftprobe_auroc,
        "honest_verdict": honest_verdict,
        "n_responses": N_QUESTIONS * 2,
        "model_used": model_used,
        "inference_mode": inference_mode,
        "n_correct_responses": N_QUESTIONS,
        "n_error_responses": N_QUESTIONS,
        "spilled_energy_threshold": THRESHOLD_SPILLED_ENERGY,
        "thinkprm_threshold": THRESHOLD_THINKPRM,
        "driftprobe_threshold": THRESHOLD_DRIFTPROBE,
        "above_spilled_energy_threshold": above_spill,
        "above_thinkprm_threshold": above_thinkprm,
        "above_driftprobe_threshold": above_drift,
        "layer_attention_weights": layer_attention_weights,
        "spill_correct_mean": float(np.mean(correct_spill_scores)),
        "spill_error_mean": float(np.mean(error_spill_scores)),
        "thinkprm_correct_mean": float(np.mean(correct_thinkprm_scores)),
        "thinkprm_error_mean": float(np.mean(error_thinkprm_scores)),
        "n_virtual_layers": N_VIRTUAL_LAYERS,
    }

    artifact = tmpl.build_result(payload, status="success")
    out_path = tmpl._output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"SpilledEnergy AUROC: {spilled_energy_auroc:.4f} (threshold {THRESHOLD_SPILLED_ENERGY})")
    print(f"ThinkPRM AUROC:      {thinkprm_auroc:.4f} (threshold {THRESHOLD_THINKPRM})")
    print(f"DRIFTProbe AUROC:    {driftprobe_auroc:.4f} (threshold {THRESHOLD_DRIFTPROBE})")
    print(f"Verdict: {honest_verdict}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
