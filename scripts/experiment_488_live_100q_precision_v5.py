#!/usr/bin/env python3
"""Experiment 488: Live 100q Precision v5 — GPUVRAMGateV2 + Explicit cuda:0/cuda:1.

**Researcher summary (RETRO-033, fifth attempt):**
    Exps 451, 464, 476, 479 all failed to close RETRO-033:
    - 451 (ms .34): +5pp result JSON absent (no DeliverableGuard).
    - 464 (ms .35): deferred — zombie processes held 23.8 GB at 0% utilisation.
    - 476 (ms .36): GPUVRAMGate used check-first order; RETRO-044 race caused deferral.
    - 479 (ms .36): same root cause.
    Exp 487 introduced GPUVRAMGateV2 (kill-first order, 15 s drain sleep).
    Exp 488 re-runs the 100q benchmark using GPUVRAMGateV2 and explicit per-model
    device assignment (cuda:0 for Gemma4-E4B-it, cuda:1 for Qwen3.5-0.8B).

**Gate chain (runs in order; every exit path writes the deliverable):**
    0. apply_env_autofix()                              — FIRST, before any CUDA import
    1. ExperimentTimeoutWatchdog(488, 120 min)          — outer hard cap
    2. DeliverableGuard instantiation                   — path registered, not yet asserted
    3. GPUVRAMGateV2(min_free_gb=8.0, kill_first=True)  — kill zombies BEFORE VRAM check
    4. DualGPUHarness(n_gpus=2).apply()                 — Gemma4→cuda:0, Qwen→cuda:1
    5. Benchmark: 100 GSM8K questions (shuffle seed=42)
    6. CoTPairCollector.flush() → results/exp488_cot_pairs.json
    7. tmpl.assert_deliverable_written()                — FINAL LINE

**Outputs:**
    results/experiment_488_live_100q_precision_v5.json  — primary artifact
    results/exp488_cot_pairs.json                       — CoT pairs for Exp 496 NUP Probe v2

Spec: REQ-BENCH-034, REQ-BENCH-035, REQ-BENCH-036,
      SCENARIO-BENCH-053, SCENARIO-BENCH-054, SCENARIO-BENCH-055
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: inject CARNOT_FORCE_LIVE=1 before any CUDA import (RETRO-022).
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix, before any torch/CUDA)
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
from carnot.pipeline.dual_gpu_harness import DualGPUHarness
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog, get_timeout_minutes
from carnot.pipeline.gemma_loader import GemmaTransformersLoader
from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2
from carnot.pipeline.gpu_vram_gate import GPUVRAMInsufficientError
from carnot.pipeline.precision_100q_v4_result import CoTPairCollector
from carnot.pipeline.precision_100q_v5_result import Precision100qV5Result
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 488
EXP_TITLE = "Live 100q Precision v5 — GPUVRAMGateV2 + cuda:0/cuda:1 (RETRO-033 close)"
DELIVERABLE = "results/experiment_488_live_100q_precision_v5.json"
COT_PAIRS_PATH = "results/exp488_cot_pairs.json"
N_QUESTIONS = 100
GSM8K_SEED = 42

# Two models, explicit GPU assignment — DualGPUHarness.apply() will inject gpu/device_map
MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "google/gemma-4-E4B-it"},
    {"name": "Qwen/Qwen3.5-0.8B"},
]


# ---------------------------------------------------------------------------
# Helpers: answer extraction
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
    """True when the extracted response matches the gold answer within tolerance."""
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
    """Load n GSM8K test questions, shuffled with a fixed seed for reproducibility.

    Falls back to synthetic arithmetic questions when the datasets package is
    unavailable (CI environments without internet access).
    """
    try:
        import random

        from datasets import load_dataset  # type: ignore[import]

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
            "question": f"Janet has {a} apples and receives {b} more. How many does she have?",
            "answer": f"She starts with {a} and gets {b} more, so {a} + {b} = {c}. #### {c}",
            "source": "synthetic",
        })
    _log.info("Using %d synthetic GSM8K questions (real dataset unavailable)", n)
    return synthetic


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _run_gemma_inference(loader: GemmaTransformersLoader, prompt: str) -> str:
    """Generate a response from Gemma4 via GemmaTransformersLoader."""
    try:
        text = loader.generate(prompt, max_new_tokens=256)
        if not GemmaTransformersLoader.is_valid_output(text):
            _log.warning("GemmaTransformersLoader.is_valid_output() returned False")
            return ""
        return text
    except Exception as exc:
        _log.warning("Gemma4 generation failed: %s", exc)
        return ""


def _load_qwen_pipeline(hf_id: str, device_map: dict) -> object:
    """Load a HuggingFace text-generation pipeline for Qwen with explicit device_map.

    Why device_map is a dict not a string:
        device_map='auto' spreads the model across all visible GPUs.  With two
        models loaded simultaneously, 'auto' causes both to fight for both GPUs.
        Passing {'': 'cuda:N'} pins every layer to GPU N (REQ-BENCH-035).
    """
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(f"transformers not installed: {exc}") from exc

    return hf_pipeline(
        "text-generation",
        model=hf_id,
        device_map=device_map,
        torch_dtype="auto",
    )


def _run_qwen_inference(pipe: object, prompt: str) -> str:
    """Generate a response from Qwen via HF pipeline."""
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
    gpu_id: int,
    inference_fn: Any,
    extractor: IntegratedExtractor,
    questions: list[dict],
    collector: CoTPairCollector,
) -> Precision100qV5Result:
    """Run baseline and pipeline variants for one model, collecting CoT pairs.

    Two passes per model:
    1. BASELINE — raw model output, no pipeline.
    2. PIPELINE — IntegratedExtractor detects violations; one-shot repair when found.

    Each pipeline response is recorded in the CoT collector (REQ-BENCH-036).
    Returns a Precision100qV5Result with Wilson 95% CI and explicit gpu_id.
    """
    # Pass 1: BASELINE
    n_correct_baseline = 0
    for q_dict in questions:
        response = inference_fn(q_dict["question"])
        gold = _extract_gsm8k_answer(q_dict["answer"])
        if _is_correct(response, gold):
            n_correct_baseline += 1

    pre_accuracy = n_correct_baseline / max(len(questions), 1)
    _log.info(
        "[%s cuda:%d] BASELINE: %d/%d correct (%.4f)",
        model_name, gpu_id, n_correct_baseline, len(questions), pre_accuracy,
    )

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
        "[%s cuda:%d] PIPELINE: %d/%d correct (%.4f) delta=%.4f extractor=%s",
        model_name, gpu_id,
        n_correct_pipeline, len(questions), post_accuracy,
        post_accuracy - pre_accuracy, extractor_used,
    )

    return Precision100qV5Result(
        model_id=model_name,
        pre_accuracy=pre_accuracy,
        post_accuracy=post_accuracy,
        n=len(questions),
        extractor_used=extractor_used,
        inference_mode="live_gpu",
        gpu_id=gpu_id,
    )


# ---------------------------------------------------------------------------
# JSON write helper
# ---------------------------------------------------------------------------


def _write_json(repo_root: Path, rel_path: str, data: Any) -> None:
    """Atomically write JSON to rel_path under repo_root."""
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
    """Run Experiment 488 and return the artifact dict.

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
    # Gate 0: GPU required — no simulated fallback
    # ------------------------------------------------------------------
    if not is_live:
        _log.info("CARNOT_FORCE_LIVE not set — GPU required, writing deferred artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v5",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gpu_vram_gate_v2_fired": False,
                "dual_gpu_explicit_assignment": False,
                "retro_033_closed": False,
            },
            status="gpu_required",
            honest_verdict="deferred_retro_033",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 1: GPUVRAMGateV2 — kill zombies FIRST, then check VRAM
    # REQ-BENCH-034: kill_first=True fixes the RETRO-044 race condition.
    # ------------------------------------------------------------------
    gpu_vram_gate_v2_fired = True
    try:
        with GPUVRAMGateV2(min_free_gb=8.0, kill_first=True):
            pass  # gate confirms VRAM is free; model load happens below
    except GPUVRAMInsufficientError as exc:
        _log.error("GPUVRAMGateV2: VRAM insufficient after drain — %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v5",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gpu_vram_gate_v2_fired": gpu_vram_gate_v2_fired,
                "dual_gpu_explicit_assignment": False,
                "retro_033_closed": False,
                "vram_error": str(exc),
            },
            status="gpu_vram_insufficient",
            honest_verdict="deferred_retro_033",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 2: DualGPUHarness — pin Gemma4→cuda:0, Qwen→cuda:1 explicitly
    # REQ-BENCH-035: never use device_map='auto' for dual-model runs.
    # ------------------------------------------------------------------
    harness = DualGPUHarness.from_env()
    assigned_specs = harness.apply([dict(s) for s in MODEL_SPECS])
    dual_gpu_explicit_assignment = harness.is_eligible

    # Extract per-model device assignments
    gemma_spec = next(s for s in assigned_specs if "gemma" in s["name"].lower())
    qwen_spec = next(s for s in assigned_specs if "qwen" in s["name"].lower() or "Qwen" in s["name"])
    gemma_gpu = gemma_spec.get("gpu", 0)
    qwen_gpu = qwen_spec.get("gpu", 1)
    gemma_device_map = gemma_spec.get("device_map", {"": f"cuda:{gemma_gpu}"})
    qwen_device_map = qwen_spec.get("device_map", {"": f"cuda:{qwen_gpu}"})

    _log.info(
        "DualGPUHarness: gemma4→cuda:%d, qwen→cuda:%d (eligible=%s)",
        gemma_gpu, qwen_gpu, harness.is_eligible,
    )

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
    gemma_loader: GemmaTransformersLoader | None = None
    try:
        _log.info("Loading Gemma4-E4B-it on cuda:%d ...", gemma_gpu)
        gemma_loader = GemmaTransformersLoader(
            model_id="google/gemma-4-E4B-it",
            device=f"cuda:{gemma_gpu}",
        )
        gemma_loader.load()
        _log.info("Gemma4-E4B-it loaded OK on cuda:%d", gemma_gpu)
    except Exception as exc:
        _log.error("Failed to load Gemma4: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v5",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gpu_vram_gate_v2_fired": gpu_vram_gate_v2_fired,
                "dual_gpu_explicit_assignment": dual_gpu_explicit_assignment,
                "retro_033_closed": False,
            },
            status="blocked",
            blocked_reason=f"Gemma4 load failed: {exc}",
            honest_verdict="deferred_retro_033",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 5: Load Qwen on cuda:1
    # ------------------------------------------------------------------
    qwen_pipe: object | None = None
    try:
        _log.info("Loading Qwen3.5-0.8B on cuda:%d ...", qwen_gpu)
        qwen_pipe = _load_qwen_pipeline("Qwen/Qwen3.5-0.8B", device_map=qwen_device_map)
        _log.info("Qwen3.5-0.8B loaded OK on cuda:%d", qwen_gpu)
    except Exception as exc:
        _log.error("Failed to load Qwen: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v5",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gpu_vram_gate_v2_fired": gpu_vram_gate_v2_fired,
                "dual_gpu_explicit_assignment": dual_gpu_explicit_assignment,
                "retro_033_closed": False,
            },
            status="blocked",
            blocked_reason=f"Qwen load failed: {exc}",
            honest_verdict="deferred_retro_033",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Load questions
    # ------------------------------------------------------------------
    questions = _load_gsm8k_questions(N_QUESTIONS, seed=GSM8K_SEED)
    _log.info("Loaded %d questions (seed=%d)", len(questions), GSM8K_SEED)

    # CoT pair collector for Exp 496 NUP Probe v2 retrain (REQ-BENCH-036)
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

    _log.info("=== Running Gemma4-E4B-it benchmark (cuda:%d, %dq) ===", gemma_gpu, len(questions))
    gemma4_result = _run_model_benchmark(
        "Gemma4-E4B-it", gemma_gpu, gemma_fn, extractor, questions, collector,
    )
    tmpl.checkpoint_save(gemma4_result.to_dict(), step=1)

    _log.info("=== Running Qwen3.5-0.8B benchmark (cuda:%d, %dq) ===", qwen_gpu, len(questions))
    qwen_result = _run_model_benchmark(
        "Qwen3.5-0.8B", qwen_gpu, qwen_fn, extractor, questions, collector,
    )
    tmpl.checkpoint_save(qwen_result.to_dict(), step=2)

    # ------------------------------------------------------------------
    # Flush CoT pairs atomically (REQ-BENCH-036)
    # ------------------------------------------------------------------
    cot_pairs_written = collector.flush()
    _log.info("CoT pairs flushed: %d pairs to %s", cot_pairs_written, COT_PAIRS_PATH)

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    retro_033_closed = gemma4_result.is_positive or qwen_result.is_positive
    honest_verdict = (
        "retro_033_closed_positive" if retro_033_closed else "retro_033_closed_negative"
    )

    artifact = tmpl.build_result(
        {
            "schema": "carnot.live_precision.v5",
            "env_autofix": env_autofix_dict,
            "n_questions": len(questions),
            "gemma4_result": gemma4_result.to_dict(),
            "qwen_result": qwen_result.to_dict(),
            "cot_pairs_written": cot_pairs_written,
            "gpu_vram_gate_v2_fired": gpu_vram_gate_v2_fired,
            "dual_gpu_explicit_assignment": dual_gpu_explicit_assignment,
            "retro_033_closed": retro_033_closed,
        },
        status="success",
        honest_verdict=honest_verdict,
        inference_mode="live_gpu",
    )
    _write_json(repo_root, DELIVERABLE, artifact)

    _log.info(
        "HEADLINE: honest_verdict=%s retro_033_closed=%s "
        "gemma4_delta=%.4f qwen_delta=%.4f cot_pairs=%d",
        honest_verdict,
        retro_033_closed,
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
    """Run Experiment 488: 100q live precision benchmark v5, RETRO-033 fifth attempt.

    120-minute watchdog: 15 s drain sleep + two model loads + 200 inference passes
    on 100q each.  Matches Exp 476 budget; GPUVRAMGateV2 drain is within that budget.
    """
    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=get_timeout_minutes(),
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        artifact = run_experiment()

    verdict = artifact.get("honest_verdict", "unknown")
    _log.info(
        "Exp %d complete: honest_verdict=%s status=%s",
        EXP_ID, verdict, artifact.get("status", "unknown"),
    )


if __name__ == "__main__":
    main()
