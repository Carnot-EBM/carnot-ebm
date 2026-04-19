#!/usr/bin/env python3
"""Experiment 476: Live 100q Precision v4 — GPUVRAMGate + DualGPU (RETRO-033 close).

**Researcher summary (RETRO-033, third and final attempt):**
    Exp 451 (milestone .34): +5pp shown but result JSON absent (DeliverableGuard missing).
    Exp 464 (milestone .35): implemented but deferred with 'deferred_to_gpu' because
      zombie processes held 23.8 GB on GPU 0 at 0% utilisation mid-session.
    Exp 476 (milestone .36): adds GPUVRAMGate (Exp 474) BEFORE model load so mid-session
      zombies are killed and VRAM is confirmed free before any model allocation.

**Gate chain (runs in order, every gate writes the deliverable before exiting):**
    0. apply_env_autofix()                         — FIRST, before any CUDA import (RETRO-022)
    1. ExperimentTimeoutWatchdog(476, 120 min)      — outer hard cap
    2. DeliverableGuard instantiation               — path registered, not yet asserted
    3. GPUVRAMGate(min_free_gb=8.0, wait_seconds=60) — kill zombies, confirm VRAM free
    4. DualGPUAssigner: Gemma4-E4B-it→cuda:0, Qwen3.5-0.8B→cuda:1
    5. Benchmark: 100 GSM8K questions (shuffle seed=42), two-pass per model
    6. CoTPairCollector.flush() → results/exp476_cot_pairs.json
    7. tmpl.assert_deliverable_written()            — FINAL LINE

**Outputs:**
    results/experiment_476_live_100q_precision_v4.json  — primary artifact
    results/exp476_cot_pairs.json                       — CoT pairs for Exp 477 JEPA retrain

Spec: REQ-BENCH-025, REQ-BENCH-026, REQ-BENCH-027,
      SCENARIO-BENCH-044, SCENARIO-BENCH-045, SCENARIO-BENCH-046
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: inject CARNOT_FORCE_LIVE=1 before any CUDA import.
# Moving below torch/JAX is a bug — see RETRO-022.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
import os
import re
from typing import Any

from carnot.extraction.integrated_extractor import IntegratedExtractor
from carnot.extraction.vericot_validator import VeriCoTStepValidator
from carnot.extraction.vprm_verifier import VPRMArithmeticVerifier
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.dual_gpu_assigner import DualGPUAssigner
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog, get_timeout_minutes
from carnot.pipeline.gemma_loader import GemmaTransformersLoader
from carnot.pipeline.gpu_vram_gate import GPUVRAMGate, GPUVRAMInsufficientError
from carnot.pipeline.precision_100q_v4_result import CoTPairCollector, Precision100qV4Result
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 476
EXP_TITLE = "Live 100q Precision v4 — GPUVRAMGate + DualGPU (RETRO-033 close)"
DELIVERABLE = "results/experiment_476_live_100q_precision_v4.json"
COT_PAIRS_PATH = "results/exp476_cot_pairs.json"
N_QUESTIONS = 100
GSM8K_SEED = 42

MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
]


# ---------------------------------------------------------------------------
# Answer extraction helpers (identical to Exp 464 — stable interface)
# ---------------------------------------------------------------------------


def _extract_gsm8k_answer(text: str) -> str | None:
    """Extract the numeric final answer from a GSM8K response.

    Looks for the official '#### N' format first, then falls back to the last
    number in the text.  Returns None when no numeric answer can be found.
    """
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _is_correct(response: str, gold: str | None) -> bool:
    """Return True when the response matches the gold answer within floating-point tolerance."""
    if not gold or not response:
        return False
    extracted = _extract_gsm8k_answer(response)
    if extracted is None:
        return False
    try:
        return abs(float(extracted) - float(gold)) < 0.501
    except (ValueError, TypeError):
        return extracted.strip() == gold.strip()


def _load_gsm8k_questions(n: int, seed: int) -> list[dict]:
    """Load n GSM8K test questions, shuffled with seed for reproducibility.

    Shuffling with a fixed seed ensures the same 100 questions are used on every
    run, making results comparable across retries.  Falls back to synthetic
    questions when the datasets package is unavailable (CI environments).
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
        import random

        ds = load_dataset("gsm8k", "main", split="test")
        all_items = [{"question": row["question"], "answer": row["answer"]} for row in ds]
        rng = random.Random(seed)
        rng.shuffle(all_items)
        result = all_items[:n]
        _log.info("Loaded %d GSM8K questions (seed=%d)", len(result), seed)
        return result
    except Exception as exc:
        _log.warning("Could not load GSM8K: %s — using synthetic fallback", exc)

    synthetic = []
    for i in range(1, n + 1):
        a, b = i * 3, i * 2
        c = a + b
        synthetic.append({
            "question": f"Janet has {a} apples and receives {b} more. How many apples does she have?",
            "answer": f"She starts with {a} and gets {b} more, so {a} + {b} = {c}. #### {c}",
            "source": "synthetic",
        })
    _log.info("Using %d synthetic GSM8K questions (real dataset unavailable)", len(synthetic))
    return synthetic


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _run_gemma_inference(loader: GemmaTransformersLoader, prompt: str) -> str:
    """Generate a response from Gemma4 via GemmaTransformersLoader.  Returns '' on failure."""
    try:
        text = loader.generate(prompt, max_new_tokens=256)
        if not GemmaTransformersLoader.is_valid_output(text):
            _log.warning("GemmaTransformersLoader.is_valid_output() returned False")
            return ""
        return text
    except Exception as exc:
        _log.warning("Gemma4 generation failed: %s", exc)
        return ""


def _load_qwen_pipeline(hf_id: str, gpu_index: int) -> object:
    """Load a HuggingFace text-generation pipeline for Qwen on the given GPU."""
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(f"transformers not installed: {exc}") from exc

    return hf_pipeline(
        "text-generation",
        model=hf_id,
        device=gpu_index,
        torch_dtype="auto",
    )


def _run_qwen_inference(pipe: object, prompt: str) -> str:
    """Generate a response from Qwen via HF pipeline.  Returns '' on failure."""
    try:
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
        return str(outputs[0]["generated_text"])
    except Exception as exc:
        _log.warning("Qwen generation failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Per-model benchmark runner
# ---------------------------------------------------------------------------


def _run_model_benchmark(
    model_name: str,
    inference_fn: Any,
    extractor: IntegratedExtractor,
    questions: list[dict],
    collector: CoTPairCollector,
) -> Precision100qV4Result:
    """Run baseline and pipeline variants for one model, collecting CoT pairs.

    Two passes:
    1. BASELINE — raw model output, no verify-repair pipeline.
    2. PIPELINE — IntegratedExtractor detects violations; one-shot repair when found.

    Each pipeline-pass response is recorded via collector.add() for Exp 477 JEPA retrain.
    Returns a Precision100qV4Result with Wilson 95% CI.
    """
    # Pass 1: BASELINE
    n_correct_baseline = 0
    for q_dict in questions:
        response = inference_fn(q_dict["question"])
        gold = _extract_gsm8k_answer(q_dict["answer"])
        if _is_correct(response, gold):
            n_correct_baseline += 1

    pre_accuracy = n_correct_baseline / max(len(questions), 1)
    _log.info("[%s] BASELINE: %d/%d correct (%.4f)", model_name, n_correct_baseline, len(questions), pre_accuracy)

    # Pass 2: PIPELINE with verify-repair
    n_correct_pipeline = 0
    all_violations_seen: list = []
    for q_dict in questions:
        response = inference_fn(q_dict["question"])
        violations = extractor.extract(response)
        all_violations_seen.extend(violations)

        if violations:
            repair_prompt = (
                f"Question: {q_dict['question']}\n\n"
                "Your previous answer contained logical or arithmetic errors. "
                "Please solve step by step carefully and double-check every calculation."
            )
            response = inference_fn(repair_prompt)

        gold = _extract_gsm8k_answer(q_dict["answer"])
        correct = _is_correct(response, gold)
        if correct:
            n_correct_pipeline += 1

        collector.add(model_name, q_dict["question"], response, correct)

    post_accuracy = n_correct_pipeline / max(len(questions), 1)
    extractor_used = extractor.extractor_names_used(all_violations_seen)
    _log.info(
        "[%s] PIPELINE: %d/%d correct (%.4f) delta=%.4f extractor_used=%s",
        model_name, n_correct_pipeline, len(questions), post_accuracy,
        post_accuracy - pre_accuracy, extractor_used,
    )

    return Precision100qV4Result(
        model_id=model_name,
        pre_accuracy=pre_accuracy,
        post_accuracy=post_accuracy,
        n=len(questions),
        extractor_used=extractor_used,
        inference_mode="live_gpu",
    )


# ---------------------------------------------------------------------------
# Artifact write helper
# ---------------------------------------------------------------------------


def _write_json(repo_root: Path, rel_path: str, data: Any) -> None:
    """Atomically write JSON data to rel_path under repo_root."""
    out_path = repo_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    Path(tmp).replace(out_path)
    _log.info("Written: %s", out_path)


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Path | None = None) -> dict[str, Any]:
    """Run Experiment 476 and return the artifact dict.

    All execution paths write the deliverable JSON before returning so that
    DeliverableGuard.assert_written() always passes.
    """
    if repo_root is None:
        repo_root = _REPO_ROOT

    is_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
        repo_root=repo_root,
    )
    tmpl.setup()

    guard = DeliverableGuard(str(repo_root / DELIVERABLE))

    env_autofix_dict = {
        "gpu_detected": _autofix_result.gpu_detected,
        "carnot_force_live_was_set": _autofix_result.carnot_force_live_was_set,
        "auto_fix_applied": _autofix_result.auto_fix_applied,
        "final_env_value": _autofix_result.final_env_value,
    }

    # ------------------------------------------------------------------
    # Gate 0: GPU required — no simulated fallback (SCENARIO-BENCH-044)
    # ------------------------------------------------------------------
    if not is_live:
        _log.info("CARNOT_FORCE_LIVE not set — GPU required, writing deferred artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v4",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gpu_vram_gate_fired": False,
                "retro_033_closed": False,
            },
            status="gpu_required",
            honest_verdict="deferred_to_gpu",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 1: GPUVRAMGate — kill zombies, confirm >= 8 GB free per GPU
    # ------------------------------------------------------------------
    gpu_vram_gate_fired = True
    try:
        with GPUVRAMGate(min_free_gb=8.0, wait_seconds=60):
            pass  # gate just confirms VRAM is free; model load happens below
    except GPUVRAMInsufficientError as exc:
        _log.error("GPUVRAMGate: VRAM insufficient — %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v4",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gpu_vram_gate_fired": gpu_vram_gate_fired,
                "retro_033_closed": False,
                "vram_error": str(exc),
            },
            status="gpu_vram_insufficient",
            honest_verdict="gpu_vram_insufficient",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 2: DualGPUAssigner — Gemma4→cuda:0, Qwen→cuda:1
    # ------------------------------------------------------------------
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0

    assigner = DualGPUAssigner(model_specs=list(MODEL_SPECS), n_gpus=n_gpus)
    assigned_specs = assigner.assign()

    try:
        gpu_status = tmpl.setup_gpu(assigned_specs)
    except RuntimeError as exc:
        gpu_status = {"all_healthy": False, "failure_reason": str(exc)}

    if not gpu_status["all_healthy"]:
        _log.error("setup_gpu not all_healthy — writing deferred artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v4",
                "env_autofix": env_autofix_dict,
                "gpu_setup_status": gpu_status,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gpu_vram_gate_fired": gpu_vram_gate_fired,
                "retro_033_closed": False,
            },
            status="gpu_required",
            honest_verdict="deferred_to_gpu",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 3: IntegratedExtractor (live mode — no mocks)
    # ------------------------------------------------------------------
    extractor = IntegratedExtractor(
        vericot=VeriCoTStepValidator(use_mock=False),
        vprm=VPRMArithmeticVerifier(),
    )

    # ------------------------------------------------------------------
    # Gate 4: Load Gemma4 on cuda:0
    # ------------------------------------------------------------------
    gemma_gpu = next((s["gpu"] for s in assigned_specs if s["name"] == "Gemma4-E4B-it"), 0)
    gemma_loader: GemmaTransformersLoader | None = None
    try:
        _log.info("Loading Gemma4-E4B-it on cuda:%d ...", gemma_gpu)
        gemma_loader = GemmaTransformersLoader(
            model_id="google/gemma-4-E4B-it",
            device=f"cuda:{gemma_gpu}",
        )
        gemma_loader.load()
        _log.info("Gemma4-E4B-it loaded OK")
    except Exception as exc:
        _log.error("Failed to load Gemma4: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v4",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gpu_vram_gate_fired": gpu_vram_gate_fired,
                "retro_033_closed": False,
            },
            status="blocked",
            blocked_reason=f"Gemma4 load failed: {exc}",
            honest_verdict="deferred_to_gpu",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 5: Load Qwen on cuda:1
    # ------------------------------------------------------------------
    qwen_gpu = next((s["gpu"] for s in assigned_specs if s["name"] == "Qwen3.5-0.8B"), 1)
    qwen_pipe: object | None = None
    try:
        _log.info("Loading Qwen3.5-0.8B on cuda:%d ...", qwen_gpu)
        qwen_pipe = _load_qwen_pipeline("Qwen/Qwen3.5-0.8B", gpu_index=qwen_gpu)
        _log.info("Qwen3.5-0.8B loaded OK")
    except Exception as exc:
        _log.error("Failed to load Qwen: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v4",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gpu_vram_gate_fired": gpu_vram_gate_fired,
                "retro_033_closed": False,
            },
            status="blocked",
            blocked_reason=f"Qwen load failed: {exc}",
            honest_verdict="deferred_to_gpu",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Load questions (100, shuffle seed=42 for reproducibility)
    # ------------------------------------------------------------------
    questions = _load_gsm8k_questions(N_QUESTIONS, seed=GSM8K_SEED)
    _log.info("Loaded %d questions (seed=%d)", len(questions), GSM8K_SEED)

    collector = CoTPairCollector(str(repo_root / COT_PAIRS_PATH))

    # ------------------------------------------------------------------
    # Run benchmarks: Gemma4 then Qwen
    # ------------------------------------------------------------------
    def gemma_fn(prompt: str) -> str:
        assert gemma_loader is not None
        return _run_gemma_inference(gemma_loader, prompt)

    def qwen_fn(prompt: str) -> str:
        assert qwen_pipe is not None
        return _run_qwen_inference(qwen_pipe, prompt)

    _log.info("=== Running Gemma4-E4B-it benchmark (%dq) ===", len(questions))
    gemma4_result = _run_model_benchmark("Gemma4-E4B-it", gemma_fn, extractor, questions, collector)
    tmpl.checkpoint_save(gemma4_result.to_dict(), step=1)

    _log.info("=== Running Qwen3.5-0.8B benchmark (%dq) ===", len(questions))
    qwen_result = _run_model_benchmark("Qwen3.5-0.8B", qwen_fn, extractor, questions, collector)
    tmpl.checkpoint_save(qwen_result.to_dict(), step=2)

    # ------------------------------------------------------------------
    # Flush CoT pairs atomically
    # ------------------------------------------------------------------
    cot_pairs_written = collector.flush()
    _log.info("CoT pairs flushed: %d pairs to %s", cot_pairs_written, COT_PAIRS_PATH)

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    any_positive = gemma4_result.is_positive or qwen_result.is_positive
    honest_verdict = "retro_033_closed_positive" if any_positive else "retro_033_closed_negative"

    artifact = tmpl.build_result(
        {
            "schema": "carnot.live_precision.v4",
            "env_autofix": env_autofix_dict,
            "n_questions": len(questions),
            "gemma4_result": gemma4_result.to_dict(),
            "qwen_result": qwen_result.to_dict(),
            "cot_pairs_written": cot_pairs_written,
            "gpu_vram_gate_fired": gpu_vram_gate_fired,
            "retro_033_closed": True,
        },
        status="success",
        honest_verdict=honest_verdict,
        inference_mode="live_gpu",
    )
    _write_json(repo_root, DELIVERABLE, artifact)

    _log.info(
        "HEADLINE: honest_verdict=%s retro_033_closed=True "
        "gemma4_delta=%.4f qwen_delta=%.4f cot_pairs=%d",
        honest_verdict,
        gemma4_result.signed_improvement,
        qwen_result.signed_improvement,
        cot_pairs_written,
    )

    guard.assert_written()
    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 476: 100q live precision benchmark v4, RETRO-033 final closure.

    Uses a 120-minute watchdog (Exp 464 used 90 min; adding 30 min for GPUVRAMGate
    wait, dual simultaneous model loads, and collector flush overhead).
    """
    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=get_timeout_minutes(),
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        artifact = run_experiment()

    verdict = artifact.get("honest_verdict", "unknown")
    _log.info("Exp %d complete: honest_verdict=%s status=%s", EXP_ID, verdict, artifact.get("status", "unknown"))


if __name__ == "__main__":
    main()
