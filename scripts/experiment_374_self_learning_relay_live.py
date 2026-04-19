#!/usr/bin/env python3
"""Experiment 374 — Three-Tier Self-Learning Relay on Live GPU (FR-11 Milestone).

**Researcher summary:**
    Exp 361 demonstrated accuracy 0.60 → 0.72 across 4 batches, but used a
    synthetic ground-truth baseline (honest_verdict="synthetic_only").  FR-11
    requires a live_gpu result before we can claim learning_confirmed.

    This experiment re-runs the same three-tier relay with:
    - Real Gemma4-E4B-it inference on 100 GSM8K questions (4 batches × 25)
    - The retrained EORM from Exp 371 (falls back to Exp 359 then fresh)
    - honest_verdict="learning_confirmed" only when:
        * inference_mode == "live_gpu"  (CARNOT_FORCE_LIVE=1)
        * batch4_accuracy > batch1_accuracy  (strict improvement)

    If CARNOT_FORCE_LIVE is not set, the script raises RuntimeError immediately
    (blocked artifact).  There is no synthetic fallback — this experiment is
    explicitly for the live milestone.

**Architecture (three tiers, same as Exp 361):**
    Tier 1 — PerModelFPTracker.update() per question.
    Tier 2 — CaseMemoryTemplateWiring.on_violation_recorded() per incorrect response.
    Tier 3 — EORM gate AUC-ROC per batch (EORMModel.energy() per (q, r) pair).

**Output:** results/experiment_374_self_learning_relay_live.json

Spec: REQ-LEARN-026, REQ-LEARN-027, SCENARIO-LEARN-050
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo root on sys.path so scripts/ and python/ imports resolve.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_REPO_ROOT), str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import jax.random as jr

from carnot.models.eorm import CoTEnergyInput, EORMModel
from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
from carnot.pipeline.self_learning_relay import (
    SelfLearningRelay,
    build_relay_artifact,
    compute_learning_improvement,
)
from carnot.pipeline.sink_probe import SinkProbe
from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline
from scripts.experiment_template import BatchedInferenceRunner, ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 374
DELIVERABLE = "results/experiment_374_self_learning_relay_live.json"
N_BATCHES = 4
BATCH_SIZE = 25
MODEL_ID = "gemma4-e4b-it"

# Path to the Exp 371 retrained EORM (preferred).
_EORM_371_PATH = _REPO_ROOT / "results" / "eorm_model_371_real.safetensors"
# Fallback: Exp 359 real-data EORM.
_EORM_359_PATH = _REPO_ROOT / "results" / "eorm_model_359_real.safetensors"


# ---------------------------------------------------------------------------
# diagnose_live_gpu
# ---------------------------------------------------------------------------


def diagnose_live_gpu() -> None:
    """Raise RuntimeError if CARNOT_FORCE_LIVE is not set.

    **Detailed explanation for engineers:**
        This function is the first guard called in main().  It prevents the
        experiment from proceeding in CI / synthetic mode — FR-11 requires a
        live GPU result, so running synthetically here would be misleading.

        If CARNOT_FORCE_LIVE != "1", we raise immediately so the caller can
        write a "blocked" artifact with a clear reason rather than silently
        producing a synthetic result.

    Raises:
        RuntimeError: When CARNOT_FORCE_LIVE is not "1".

    Spec: SCENARIO-LEARN-050
    """
    if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
        raise RuntimeError(
            "Exp 374 requires CARNOT_FORCE_LIVE=1 — live GPU inference is mandatory "
            "for the learning_confirmed verdict.  Set CARNOT_FORCE_LIVE=1 or run "
            "Exp 361 for the synthetic baseline."
        )


# ---------------------------------------------------------------------------
# load_eorm_model
# ---------------------------------------------------------------------------


def load_eorm_model(seed: int = 374) -> tuple[EORMModel, str]:
    """Load the best available EORM model and return (model, eorm_source).

    **Detailed explanation for engineers:**
        We prefer the retrained EORM from Exp 371 (more epochs on real data).
        If that checkpoint is absent, we fall back to Exp 359's real-data EORM.
        If that is also absent, we use a freshly initialised model
        (eorm_source="synthetic_fallback") — the Tier 3 AUC will start at ~0.5
        but the code path is fully exercised and the honest_verdict gate still
        applies.

    Returns:
        (eorm_model, eorm_source) where eorm_source is one of:
        - "exp371_real"        — retrained on real data (Exp 371)
        - "exp359_real"        — retrained on real data (Exp 359)
        - "synthetic_fallback" — freshly initialised random weights

    Spec: SCENARIO-LEARN-050
    """
    key = jr.PRNGKey(seed)

    if _EORM_371_PATH.exists():
        model = EORMModel.load(str(_EORM_371_PATH))
        return model, "exp371_real"

    if _EORM_359_PATH.exists():
        model = EORMModel.load(str(_EORM_359_PATH))
        return model, "exp359_real"

    # Fresh model — random weights, AUC will start near 0.5.
    model = EORMModel(
        embed_dim=64, n_heads=4, n_layers=2, max_seq_len=128, vocab_size=512, key=key
    )
    return model, "synthetic_fallback"


# ---------------------------------------------------------------------------
# load_gsm8k_questions
# ---------------------------------------------------------------------------


def load_gsm8k_questions(n: int = 100) -> tuple[list[str], list[str]]:
    """Load GSM8K questions and reference answers from HuggingFace datasets.

    **Detailed explanation for engineers:**
        GSM8K (Grade School Math 8K) is a dataset of 8,500 linguistically
        diverse elementary math word problems.  We load from the "test" split
        because those questions have not been seen during model training (they
        are fixed across runs), giving reproducible accuracy baselines.

        The reference answers in GSM8K are formatted as:
            "... #### <number>"
        We extract the numeric answer after "####" and use it for correctness
        evaluation.

        If the datasets library is unavailable (CI environment), we raise
        ImportError so the caller can handle the blocked case.

    Args:
        n: Number of questions to load (default 100 = 4 batches × 25).

    Returns:
        (questions, answers) — both lists of length n.
        questions: full problem text.
        answers:   numeric reference answer strings extracted from GSM8K.

    Raises:
        ImportError: When the HuggingFace datasets library is not installed.
        RuntimeError: When fewer than n questions are available in the split.

    Spec: SCENARIO-LEARN-050
    """
    try:
        from datasets import load_dataset  # type: ignore[import]  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required for Exp 374.  "
            "Install with: pip install datasets"
        ) from exc

    dataset = load_dataset("gsm8k", "main", split="test")
    if len(dataset) < n:
        raise RuntimeError(
            f"GSM8K test split has only {len(dataset)} questions; need {n}."
        )

    questions: list[str] = []
    answers: list[str] = []
    for row in dataset.select(range(n)):
        questions.append(row["question"])
        # Reference answer format: "... #### 42"
        raw_answer: str = row["answer"]
        parts = raw_answer.split("####")
        answers.append(parts[-1].strip() if len(parts) > 1 else raw_answer.strip())

    return questions, answers


# ---------------------------------------------------------------------------
# extract_numeric_answer
# ---------------------------------------------------------------------------


def extract_numeric_answer(response: str) -> str:
    """Extract the final numeric answer from a model response string.

    **Detailed explanation for engineers:**
        GSM8K reference answers are plain integers or decimals.  Model responses
        typically contain a chain-of-thought followed by a boxed or plain answer
        like "The answer is 42" or "\\boxed{42}" or just "42".

        We apply a cascade of extraction heuristics:
        1. LaTeX \\boxed{...} — common in Gemma/Qwen3 CoT outputs.
        2. "The answer is <N>" pattern.
        3. "#### <N>" — GSM8K-style inline answer marker.
        4. Last standalone number in the response (fallback).

        All extracted strings are stripped of commas and whitespace.

    Args:
        response: Raw model response string.

    Returns:
        Extracted numeric string, or empty string if nothing found.

    Spec: SCENARIO-LEARN-050
    """
    # 1. LaTeX boxed
    m = re.search(r"\\boxed\{([^}]+)\}", response)
    if m:
        return m.group(1).replace(",", "").strip()

    # 2. "The answer is N" pattern (stop at punctuation like trailing period)
    m = re.search(r"(?i)the answer is\s*[:\-]?\s*([\-]?\d+(?:[.,]\d+)*)", response)
    if m:
        return m.group(1).replace(",", "").strip()

    # 3. GSM8K #### marker
    parts = response.split("####")
    if len(parts) > 1:
        return parts[-1].strip().replace(",", "")

    # 4. Last number in response (fallback)
    numbers = re.findall(r"[\-]?\d+(?:[.,]\d+)*", response)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return ""


# ---------------------------------------------------------------------------
# is_correct_answer
# ---------------------------------------------------------------------------


def is_correct_answer(response: str, reference: str) -> bool:
    """Return True when the model response matches the reference answer.

    **Detailed explanation for engineers:**
        We compare extracted numeric answers by normalising both to floats.
        This handles cases like "42" vs "42.0" or "3.5" vs "3.50".

        If either extraction fails (non-numeric), we fall back to exact string
        comparison after stripping whitespace and commas.

    Args:
        response:  Raw model output string.
        reference: GSM8K reference answer string (numeric).

    Returns:
        True when the answers match numerically (or exactly on fallback).

    Spec: SCENARIO-LEARN-050
    """
    extracted = extract_numeric_answer(response)
    ref_clean = reference.replace(",", "").strip()

    # Try numeric comparison first.
    try:
        return abs(float(extracted) - float(ref_clean)) < 1e-6
    except (ValueError, TypeError):
        pass

    # Fallback: exact string comparison (both lower-cased, stripped).
    return extracted.lower() == ref_clean.lower()


# ---------------------------------------------------------------------------
# build_components
# ---------------------------------------------------------------------------


def build_components(eorm_model: EORMModel, seed: int = 374) -> tuple[
    ThreeTierPipeline, ConstraintTemplateLibrary, PerModelFPTracker
]:
    """Build pipeline, template library, and FP tracker for the relay.

    **Detailed explanation for engineers:**
        We use the same component configuration as Exp 361 so results are
        comparable.  The EORM model is passed in (already loaded or freshly
        initialised by load_eorm_model) so Tier 3 benefits from any prior
        training.

        The Ising stub always returns verified=True with energy=0.0 — in the
        live relay the pipeline's Tier 1 feedback is what matters for
        correctness tracking, not the Ising verifier itself.

    Args:
        eorm_model: Pre-loaded or freshly initialised EORMModel for the pipeline gate.
        seed:       JAX PRNG seed for deterministic initialisation.

    Returns:
        (pipeline, template_library, fp_tracker) ready for SelfLearningRelay.

    Spec: SCENARIO-LEARN-050
    """
    key = jr.PRNGKey(seed)

    # Pipeline EORM (separate from the relay Tier 3 EORM)
    pipeline_eorm = EORMModel(
        embed_dim=64, n_heads=4, n_layers=2, max_seq_len=128, vocab_size=512, key=key
    )

    def _ising_stub(response: str, question: str) -> tuple[bool, float]:
        """Stub verifier: always returns verified=True.

        Why: In the self-learning relay the correctness signal comes from
        comparing model responses to ground-truth answers, not from the Ising
        pipeline.  A real Ising verifier would add noise without adding signal
        here.
        """
        return True, 0.0

    sink = SinkProbe(threshold=0.3)
    pipeline = ThreeTierPipeline(
        sink_probe=sink,
        eorm_model=pipeline_eorm,
        ising_pipeline=_ising_stub,
        sink_threshold=0.3,
        eorm_threshold=0.5,
    )

    library = ConstraintTemplateLibrary()
    library.register_builtin_templates()

    # min_observations=5 so templates can activate within 4 batches
    tracker = PerModelFPTracker(min_observations=5)

    return pipeline, library, tracker


# ---------------------------------------------------------------------------
# run_live_batch
# ---------------------------------------------------------------------------


def run_live_batch(
    questions: list[str],
    references: list[str],
    model_id: str,
    pipeline_fn: Any,
) -> tuple[list[str], list[bool]]:
    """Run live inference for one batch and evaluate correctness.

    **Detailed explanation for engineers:**
        We use BatchedInferenceRunner to call pipeline_fn (the loaded HuggingFace
        text-generation pipeline) in batches of 8 questions.  Each result is
        compared to the reference answer using is_correct_answer().

        The pipeline_fn is expected to accept a list of prompts and return a list
        of strings.  When running via setup_gpu/DualGPURunner the fn is a wrapper
        around the DualGPURunner; in cold-load mode it is a raw HF pipeline call.

    Args:
        questions:   Batch of question strings.
        references:  Parallel list of reference answer strings.
        model_id:    HuggingFace model identifier (for logging).
        pipeline_fn: Callable that maps list[str] -> list[str] (prompts → responses).

    Returns:
        (responses, ground_truth) — responses are raw model output strings;
        ground_truth[i] is True iff responses[i] matches references[i].

    Spec: SCENARIO-LEARN-050
    """
    # BatchedInferenceRunner expects runner(prompt: str) -> str (single call).
    # Wrap the batch pipeline_fn so each question is called individually.
    def _single_infer(prompt: str) -> str:
        results = pipeline_fn([prompt])
        return results[0] if results else ""

    bir = BatchedInferenceRunner(_single_infer, batch_size=8)
    inference_results = bir.run_batch(questions)

    responses = [r.response for r in inference_results]
    ground_truth = [
        is_correct_answer(resp, ref)
        for resp, ref in zip(responses, references)
    ]
    return responses, ground_truth


# ---------------------------------------------------------------------------
# _load_model_pipeline
# ---------------------------------------------------------------------------


def _load_model_pipeline(model_id: str) -> Any:
    """Load a HuggingFace text-generation pipeline for the given model_id.

    **Detailed explanation for engineers:**
        We use the transformers ``pipeline`` API with the "text-generation" task.
        ``trust_remote_code`` is gated by ``CARNOT_TRUST_REMOTE_CODE=1``.
        ``device_map={'': 'cuda:0'}`` lets transformers choose the best GPU/CPU layout.

        We return a thin wrapper lambda so the caller always gets a
        list[str] -> list[str] interface regardless of the underlying backend.

    Args:
        model_id: HuggingFace model ID (e.g. "google/gemma-3-4b-it").

    Returns:
        Callable[[list[str]], list[str]] — batched text-generation wrapper.

    Raises:
        ImportError: When transformers is not installed.
        RuntimeError: When CARNOT_TRUST_REMOTE_CODE check fails.

    Spec: SCENARIO-LEARN-050
    """
    trust_remote = os.environ.get("CARNOT_TRUST_REMOTE_CODE", "0") == "1"

    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "transformers library required for live inference.  "
            "Install with: pip install transformers accelerate"
        ) from exc

    # Map friendly model names to HuggingFace IDs
    _HF_MAP: dict[str, str] = {
        "gemma4-e4b-it": "google/gemma-3-4b-it",
        "qwen3.5-0.8b": "Qwen/Qwen2.5-0.5B",
    }
    hf_id = _HF_MAP.get(model_id, model_id)

    gen_pipeline = hf_pipeline(
        "text-generation",
        model=hf_id,
        trust_remote_code=trust_remote,
        device_map={'': 'cuda:1'},
        max_new_tokens=256,
    )

    def _infer(prompts: list[str]) -> list[str]:
        """Wrapper that returns plain response strings."""
        outputs = gen_pipeline(prompts, batch_size=len(prompts))
        results: list[str] = []
        for out in outputs:
            if isinstance(out, list):
                results.append(out[0].get("generated_text", ""))
            elif isinstance(out, dict):
                results.append(out.get("generated_text", ""))
            else:
                results.append(str(out))
        return results

    return _infer


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 374: live four-batch self-learning relay and save artifact.

    **Flow:**
    1. Guard: raise if CARNOT_FORCE_LIVE != "1".
    2. ExperimentTemplate setup.
    3. Load EORM (Exp 371 → Exp 359 → fresh).
    4. Load 100 GSM8K questions from HuggingFace.
    5. Load live model pipeline (Gemma4-E4B-it).
    6. Build relay components and SelfLearningRelay.
    7. Run 4 batches of 25 questions each.
    8. Compute learning improvement.
    9. Build and save artifact.

    Spec: REQ-LEARN-026, REQ-LEARN-027, SCENARIO-LEARN-050
    """
    # ----------------------------------------------------------------
    # Step 1: Live GPU guard
    # ----------------------------------------------------------------
    tmpl = ExperimentTemplate(
        EXP_ID,
        "Three-Tier Self-Learning Relay — Live GPU (FR-11 Milestone)",
        DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    try:
        diagnose_live_gpu()
    except RuntimeError as exc:
        artifact = tmpl.build_result(
            {
                "schema": "carnot.self_learning_relay.v2",
                "inference_mode": "cpu_synthetic",
                "honest_verdict": "blocked",
                "block_reason": str(exc),
            },
            status="blocked",
        )
        output_path = tmpl._repo_root / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"[Exp 374] BLOCKED: {exc}")
        return

    print("[Exp 374] Live GPU mode confirmed (CARNOT_FORCE_LIVE=1).")

    # ----------------------------------------------------------------
    # Step 2: GPU setup (pre-warm Gemma4-E4B-it)
    # ----------------------------------------------------------------
    MODEL_SPECS = [
        {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-3-4b-it", "gpu": 0},
    ]
    gpu_status = tmpl.setup_gpu(MODEL_SPECS)
    if not gpu_status["all_healthy"]:
        artifact = tmpl.build_result(
            {
                "schema": "carnot.self_learning_relay.v2",
                "inference_mode": "live_gpu",
                "honest_verdict": "blocked",
                "block_reason": "GPU pre-warm failed",
                "gpu_status": gpu_status["models"],
            },
            status="blocked",
        )
        output_path = tmpl._repo_root / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"[Exp 374] BLOCKED: GPU pre-warm unhealthy — {gpu_status['models']}")
        return

    # ----------------------------------------------------------------
    # Step 3: Load EORM
    # ----------------------------------------------------------------
    relay_eorm, eorm_source = load_eorm_model(seed=374)
    print(f"[Exp 374] EORM loaded: eorm_source={eorm_source}")

    # ----------------------------------------------------------------
    # Step 4: Load GSM8K questions
    # ----------------------------------------------------------------
    try:
        questions_all, answers_all = load_gsm8k_questions(n=N_BATCHES * BATCH_SIZE)
    except (ImportError, RuntimeError) as exc:
        artifact = tmpl.build_result(
            {
                "schema": "carnot.self_learning_relay.v2",
                "inference_mode": "live_gpu",
                "honest_verdict": "blocked",
                "block_reason": f"GSM8K load failed: {exc}",
            },
            status="blocked",
        )
        output_path = tmpl._repo_root / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"[Exp 374] BLOCKED: {exc}")
        return

    print(f"[Exp 374] Loaded {len(questions_all)} GSM8K questions.")

    # ----------------------------------------------------------------
    # Step 5: Load live model pipeline
    # ----------------------------------------------------------------
    try:
        infer_fn = _load_model_pipeline(MODEL_ID)
    except (ImportError, RuntimeError) as exc:
        artifact = tmpl.build_result(
            {
                "schema": "carnot.self_learning_relay.v2",
                "inference_mode": "live_gpu",
                "honest_verdict": "blocked",
                "block_reason": f"Model load failed: {exc}",
            },
            status="blocked",
        )
        output_path = tmpl._repo_root / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"[Exp 374] BLOCKED: {exc}")
        return

    # ----------------------------------------------------------------
    # Step 6: Build relay components
    # ----------------------------------------------------------------
    pipeline, library, tracker = build_components(relay_eorm, seed=374)

    relay = SelfLearningRelay(
        pipeline=pipeline,
        template_library=library,
        fp_tracker=tracker,
        eorm_model=relay_eorm,
    )

    # ----------------------------------------------------------------
    # Step 7: Run 4 batches of 25 questions
    # ----------------------------------------------------------------
    print(f"[Exp 374] Running {N_BATCHES} batches of {BATCH_SIZE} live questions.")
    batch_responses: list[list[str]] = []

    for batch_idx in range(N_BATCHES):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_questions = questions_all[start:end]
        batch_refs = answers_all[start:end]

        # Run live inference and evaluate correctness.
        responses, ground_truth = run_live_batch(
            batch_questions, batch_refs, MODEL_ID, infer_fn
        )
        batch_responses.append(responses)

        result = relay.run_batch(batch_questions, ground_truth, MODEL_ID)
        print(
            f"  Batch {batch_idx}: accuracy={result.accuracy:.3f}  "
            f"tier1_updates={result.n_tier1_updates}  "
            f"tier2_active={result.n_tier2_templates_active}  "
            f"tier3_auc={result.tier3_gate_auc:.3f}  "
            f"cumulative={result.cumulative_accuracy:.3f}"
        )

    # ----------------------------------------------------------------
    # Step 8: Compute learning improvement
    # ----------------------------------------------------------------
    traj = relay.learning_trajectory()
    improvement = compute_learning_improvement(traj)
    b1, b4, improved = improvement
    print(f"[Exp 374] batch1_accuracy={b1:.3f}  batch4_accuracy={b4:.3f}  improved={improved}")

    # Tier 2 template activation summary.
    tier2_activated = [
        key
        for key, tmpl_obj in library._templates.items()
        if library._observations.get((key, MODEL_ID), 0) >= tmpl_obj.min_frequency
    ]
    print(f"[Exp 374] Tier 2 templates activated: {tier2_activated}")

    # ----------------------------------------------------------------
    # Step 9: Build and save artifact
    # ----------------------------------------------------------------
    relay_artifact = build_relay_artifact(traj, improvement, inference_mode="live_gpu")

    # Upgrade to schema v2 with additional Exp 374 fields.
    relay_artifact["schema"] = "carnot.self_learning_relay.v2"
    # Rename "trajectory" -> "learning_trajectory" per SCENARIO-LEARN-050 spec.
    relay_artifact["learning_trajectory"] = relay_artifact.pop("trajectory")
    relay_artifact["eorm_source"] = eorm_source
    relay_artifact["tier2_templates_activated"] = tier2_activated
    relay_artifact["model_id"] = MODEL_ID

    artifact = tmpl.build_result(relay_artifact, status="success")

    output_path = tmpl._repo_root / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[Exp 374] Artifact saved: {output_path}")
    print(f"[Exp 374] honest_verdict={relay_artifact['honest_verdict']}")

    if relay_artifact["honest_verdict"] == "learning_confirmed":
        print("[Exp 374] *** FR-11 MILESTONE ACHIEVED: learning_confirmed ***")


if __name__ == "__main__":
    main()
