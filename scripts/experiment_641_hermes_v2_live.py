#!/usr/bin/env python3
"""Experiment 641 — HERMES v2 Live Generation Loop.

**Hypothesis (RETRO-070 CRITICAL, carry 3):**
    Post-hoc verification (Exp 629 InterWhen, Exp 633 HermesAdapter) achieves
    recall=0.12.  The ceiling is 12% because IT-tuned models express explicit
    arithmetic in only ~12% of their prose.  Generating step-by-step LIVE and
    injecting correction hints at sentence boundaries should raise recall to
    >= 0.20 (gate target: 0.30 to open the Verifiable Reasoning gate).

**Architecture change vs. Exp 633:**
    - Exp 633: full response → segment → verify  (post-hoc, ceiling 12%)
    - Exp 641: one sentence at a time → verify each → inject hint → next sentence
    This gives the verifier a chance to steer the model mid-generation.

**CI stub mode (CARNOT_FORCE_LIVE not set):**
    llm_caller=None → _generate_step() returns '' → loop exits immediately →
    hermes_v2_recall=0.0.  The deliverable is still written so the conductor
    can verify the pipeline is structurally correct.

Spec: REQ-VERIFY-137, REQ-VERIFY-138
"""

import json
import os
import sys

# --- required FIRST: env autofix and GPU assertion ---
from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.live_assertion import assert_live_or_ci_skip

apply_env_autofix()
assert_live_or_ci_skip()

import statistics  # noqa: E402 — standard lib, after env setup

# Path wiring so `scripts/` can be imported directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.experiment_template import (  # noqa: E402
    BatchedInferenceRunner,
    ExperimentTemplate,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.symcode_verifier import SymCodeVerifier  # noqa: E402
from carnot.pipeline.hermes_v2_live_loop import HermesV2LiveLoop  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 641
TITLE = "HERMES v2 Live Generation Loop"
DELIVERABLE = "results/experiment_641_hermes_v2_live.json"
N_INCORRECT = 25
N_CORRECT = 10
POST_HOC_BASELINE = 0.12  # Exp 633 hermes_recall

# Synthetic GSM8K-style questions used when the fover corpus has too few entries.
_SYNTHETIC_INCORRECT = [
    f"Janet has {10+i} apples and gives away {3+i}. She then buys {2+i} more. "
    f"How many apples does she have now?"
    for i in range(N_INCORRECT)
]
_SYNTHETIC_CORRECT = [
    f"A bag has {5+i} red balls and {3+i} blue balls. How many balls total?"
    for i in range(N_CORRECT)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fover_questions() -> tuple[list[str], list[str]]:
    """Load known-incorrect and known-correct questions from fover_corpus_v5_oracle.json.

    Tries the oracle file first (has is_correct labels), then falls back to
    fover_corpus_v5.json.  If either has fewer than N_INCORRECT incorrect or
    N_CORRECT correct entries, pads with synthetic GSM8K questions.

    Returns a (incorrect_questions, correct_questions) tuple.
    """
    candidates = [
        os.path.join(_REPO_ROOT, "results", "fover_corpus_v5_oracle.json"),
        os.path.join(_REPO_ROOT, "results", "fover_corpus_v5.json"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, list):
            continue

        incorrect = [
            entry["question"]
            for entry in data
            if isinstance(entry, dict)
            and entry.get("is_correct") is False
            and "question" in entry
        ]
        correct = [
            entry["question"]
            for entry in data
            if isinstance(entry, dict)
            and entry.get("is_correct") is True
            and "question" in entry
        ]

        # Pad with synthetic questions if not enough.
        if len(incorrect) < N_INCORRECT:
            incorrect = (incorrect + _SYNTHETIC_INCORRECT)[:N_INCORRECT]
        if len(correct) < N_CORRECT:
            correct = (correct + _SYNTHETIC_CORRECT)[:N_CORRECT]

        return incorrect[:N_INCORRECT], correct[:N_CORRECT]

    # No corpus file found: use fully synthetic questions.
    return _SYNTHETIC_INCORRECT[:N_INCORRECT], _SYNTHETIC_CORRECT[:N_CORRECT]


def _build_llm_caller(force_live: bool):
    """Build a Qwen3.5-0.8B LLM caller if CARNOT_FORCE_LIVE=1, else return None.

    In live mode: loads Qwen3.5-0.8B via transformers pipeline (text-generation).
    In CI stub mode (force_live=False): returns None so the loop exits immediately.

    Why Qwen3.5-0.8B: smallest model in the canonical stack that can produce
    CoT with explicit arithmetic.  SOTA models (Qwen3.6-35B, Gemma-4-26B) are
    preferred for production but require 24-48 GB VRAM not always available
    in dev sessions.

    Returns: callable(prompt: str) -> str, or None.
    """
    if not force_live:
        return None

    try:
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415

        gen = hf_pipeline(
            "text-generation",
            model="Qwen/Qwen3.5-0.8B",
            device=0,
            max_new_tokens=80,
            do_sample=False,
        )

        def _caller(prompt: str) -> str:
            out = gen(prompt, max_new_tokens=80, do_sample=False)
            if out and isinstance(out, list):
                generated = out[0].get("generated_text", "")
                # Strip the prompt prefix from the generated text.
                if generated.startswith(prompt):
                    generated = generated[len(prompt):]
                return generated.strip()
            return ""

        return _caller
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to load Qwen3.5-0.8B: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 641 — HermesV2LiveLoop step-by-step generation with mid-gen verification."""
    # --- Watchdog: 90-minute hard timeout ---
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=90)

    # --- ExperimentTemplate setup ---
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # --- GPU pre-warm ---
    model_specs = [{"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0}]
    gpu_status = tmpl.setup_gpu(model_specs)

    # CI flag takes precedence: even if apply_env_autofix() set CARNOT_FORCE_LIVE=1
    # because it detected a GPU, we honour CARNOT_IS_CI=1 as the authoritative "stub mode"
    # signal for structural verification runs.
    is_ci = os.environ.get("CARNOT_IS_CI") == "1"
    force_live = os.environ.get("CARNOT_FORCE_LIVE") == "1" and not is_ci
    inference_mode = "live_gpu" if force_live else "ci_stub"

    # --- Build LLM caller and pipeline components ---
    llm_caller = _build_llm_caller(force_live)
    verifier = SymCodeVerifier(llm_caller=llm_caller if force_live else None)
    loop = HermesV2LiveLoop(llm_caller, verifier, max_sentences=8)

    # --- Load questions ---
    incorrect_questions, correct_questions = _load_fover_questions()

    # --- Run on known-incorrect questions (recall measurement) ---
    incorrect_gen_results = []

    def _run_incorrect(question: str) -> str:
        result = loop.generate_with_verification(question)
        incorrect_gen_results.append(result)
        return result.full_response

    bir_incorrect = BatchedInferenceRunner(_run_incorrect, batch_size=5)
    bir_incorrect.run_batch(incorrect_questions)

    # --- Run on known-correct questions (FP rate measurement) ---
    correct_gen_results = []

    def _run_correct(question: str) -> str:
        result = loop.generate_with_verification(question)
        correct_gen_results.append(result)
        return result.full_response

    bir_correct = BatchedInferenceRunner(_run_correct, batch_size=5)
    bir_correct.run_batch(correct_questions)

    # --- Compute metrics ---
    hermes_v2_tp = sum(r.any_violation for r in incorrect_gen_results)
    hermes_v2_fp = sum(r.any_violation for r in correct_gen_results)
    hermes_v2_recall = hermes_v2_tp / N_INCORRECT
    hermes_v2_fp_rate = hermes_v2_fp / N_CORRECT

    n_hints_per_incorrect = [r.n_hints for r in incorrect_gen_results]
    mean_hints_per_incorrect = statistics.mean(n_hints_per_incorrect) if n_hints_per_incorrect else 0.0

    recall_improvement = round(hermes_v2_recall - POST_HOC_BASELINE, 4)

    if hermes_v2_recall >= 0.30:
        honest_verdict = "hermes_v2_breakthrough"
    elif hermes_v2_recall > POST_HOC_BASELINE:
        honest_verdict = "hermes_v2_improved"
    else:
        honest_verdict = "hermes_v2_no_improvement"

    # --- Build artifact ---
    artifact = tmpl.build_result(
        {
            "schema": "carnot.hermes_v2_live.v1",
            "n_questions": N_INCORRECT,
            "n_correct": N_CORRECT,
            "hermes_v2_tp": hermes_v2_tp,
            "hermes_v2_fp": hermes_v2_fp,
            "hermes_v2_recall": round(hermes_v2_recall, 4),
            "hermes_v2_fp_rate": round(hermes_v2_fp_rate, 4),
            "post_hoc_baseline": POST_HOC_BASELINE,
            "recall_improvement": recall_improvement,
            "mean_hints_per_incorrect": round(mean_hints_per_incorrect, 4),
            "inference_mode": inference_mode,
            "gate_contribution": hermes_v2_recall >= 0.20,
            "retro_070_partial": hermes_v2_recall > POST_HOC_BASELINE,
            "honest_verdict": honest_verdict,
            "gpu_status": gpu_status,
            "batch_log_incorrect": bir_incorrect.batch_log,
            "batch_log_correct": bir_correct.batch_log,
        },
        status="success",
        decision_class="detect",
    )

    # --- Write deliverable ---
    import pathlib  # noqa: PLC0415

    out_path = pathlib.Path(_REPO_ROOT) / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[Exp 641] hermes_v2_recall={hermes_v2_recall:.3f} "
          f"(baseline={POST_HOC_BASELINE}) "
          f"honest_verdict={honest_verdict} "
          f"gate_contribution={artifact['gate_contribution']}")

    # FINAL LINE — must remain last.
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
