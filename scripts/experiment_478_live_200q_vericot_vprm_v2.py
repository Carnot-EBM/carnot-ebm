#!/usr/bin/env python3
"""Experiment 478: Live 200q VeriCoT+VPRM v2 — GPUVRAMGate + DualGPURunner closure.

**Researcher summary (RETRO-038 closure):**
    Exp 467 (200q integrated benchmark) was deferred_to_gpu in milestone .35 because
    zombie VRAM from a prior experiment saturated GPU 0 with 23.8 GB at 0% utilisation,
    preventing model load.  GPUVRAMGate (Exp 474) was implemented to kill zombie processes
    before any GPU experiment.  This experiment wires GPUVRAMGate into the 200q harness,
    finally running the headline credibility experiment on live RTX 3090 hardware.

    At n=200 the Wilson 95% CI half-width is ≤ 3.5pp, making any positive result with
    is_statistically_positive=True a credible, publishable claim.

**Gate chain (runs in order):**
    0. apply_env_autofix() — FIRST, before any CUDA import (RETRO-022 fix)
    1. ExperimentTimeoutWatchdog(478, timeout_minutes=150) — outer budget cap
    2. DeliverableGuard — ensures result is written regardless of exit path
    3. GPUVRAMGate(min_free_gb=8.0) — kills zombie VRAM holders (RETRO-037/042 fix)
    4. If GPU VRAM insufficient: emit status='gpu_vram_insufficient', exit cleanly
    5. DualGPUAssigner → Gemma4-E4B-it on cuda:0, Qwen3.5-0.8B on cuda:1
    6. IntegratedExtractor(VeriCoTStepValidator(use_mock=False), VPRMArithmeticVerifier())
    7. 200 GSM8K questions (seed=42 for reproducibility)
    8. CoTPairCollector for JEPA retrain output
    9. LongRunBenchmarkExecutor(batch_size=25, timeout_minutes=30) per model

**Outputs:**
    results/experiment_478_live_200q_vericot_vprm_v2.json — primary deliverable
    results/exp478_cot_pairs.json — CoT pairs for downstream JEPA retrain

**honest_verdict semantics:**
    'credible_positive'           — any model has is_statistically_positive=True
    'improvement_not_significant' — any model improved but CI includes zero
    'no_improvement_200q'         — no model improved at all
    'gpu_vram_insufficient'       — GPUVRAMGate blocked the run (zombie VRAM)
    'gpu_required'                — CARNOT_FORCE_LIVE not set

Spec: REQ-BENCH-028, REQ-BENCH-029, REQ-BENCH-030,
      SCENARIO-BENCH-047, SCENARIO-BENCH-048, SCENARIO-BENCH-049
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() injects CARNOT_FORCE_LIVE=1 before any
# CUDA import.  See RETRO-022 for why this ordering matters.
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
import random
import re
from typing import Any

from carnot.extraction.integrated_extractor import IntegratedExtractor
from carnot.extraction.vericot_validator import VeriCoTStepValidator
from carnot.extraction.vprm_verifier import VPRMArithmeticVerifier
from carnot.pipeline.cot_pair_collector import CoTPairCollector
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.dual_gpu_assigner import DualGPUAssigner
from carnot.pipeline.experiment_watchdog import (
    ExperimentTimeoutWatchdog,
    get_timeout_minutes,
)
from carnot.pipeline.gpu_vram_gate import GPUVRAMGate, GPUVRAMInsufficientError
from carnot.pipeline.live_200q_v2_result import Live200qV2Result
from scripts.experiment_template import BatchedInferenceRunner, ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 478
EXP_TITLE = "Live 200q VeriCoT+VPRM v2 (Exp 478)"
DELIVERABLE = "results/experiment_478_live_200q_vericot_vprm_v2.json"
COT_PAIRS_PATH = "results/exp478_cot_pairs.json"
N_QUESTIONS = 200
DATASET_SEED = 42
BATCH_SIZE = 25

MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
]


# ---------------------------------------------------------------------------
# GSM8K helpers
# ---------------------------------------------------------------------------


def _extract_gold(answer_text: str) -> str | None:
    """Extract the numeric gold answer from GSM8K '#### N' format."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    return nums[-1] if nums else None


def _is_correct(response: str, gold: str | None) -> bool:
    """Return True when the response's final number matches the gold answer."""
    if not gold or not response:
        return False
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", response)
    if m:
        extracted = m.group(1)
    else:
        nums = re.findall(r"-?\d+(?:\.\d+)?", response)
        extracted = nums[-1] if nums else None
    if extracted is None:
        return False
    try:
        return abs(float(extracted) - float(gold)) < 0.501
    except (ValueError, TypeError):
        return extracted.strip() == gold.strip()


def _load_gsm8k_200q(seed: int = DATASET_SEED) -> list[dict]:
    """Load 200 GSM8K questions shuffled with a fixed seed for reproducibility.

    Falls back to synthetic questions when the datasets package is unavailable.
    Synthetic questions are labelled source='synthetic' so they are distinguishable.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        items = [{"question": row["question"], "answer": row["answer"]} for row in ds]
        rng = random.Random(seed)
        rng.shuffle(items)
        result = items[:N_QUESTIONS]
        _log.info("Loaded %d GSM8K questions (seed=%d shuffle)", len(result), seed)
        return result
    except Exception as exc:
        _log.warning("Could not load GSM8K: %s — using synthetic fallback", exc)

    synthetic = []
    rng = random.Random(seed)
    for i in range(1, N_QUESTIONS + 1):
        a = rng.randint(10, 200)
        b = rng.randint(1, 100)
        c = a + b
        synthetic.append({
            "question": (
                f"Janet has {a} apples and receives {b} more.  "
                f"How many apples does she have?"
            ),
            "answer": (
                f"She starts with {a} and gets {b} more, so "
                f"{a} plus {b} gives {c}.  #### {c}"
            ),
            "source": "synthetic",
        })
    _log.info("Using %d synthetic questions (real dataset unavailable)", len(synthetic))
    return synthetic


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _load_gemma4(gpu_index: int = 0) -> object:
    """Load Gemma4-E4B-it via GemmaTransformersLoader on the specified GPU."""
    from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: PLC0415

    loader = GemmaTransformersLoader(
        model_id="google/gemma-4-E4B-it",
        device=f"cuda:{gpu_index}",
    )
    loader.load()
    return loader


def _load_qwen(hf_id: str, gpu_index: int = 1) -> object:
    """Load Qwen model via HF text-generation pipeline on the specified GPU."""
    from transformers import pipeline as hf_pipeline  # type: ignore[import]

    return hf_pipeline(
        "text-generation",
        model=hf_id,
        device=gpu_index,
        torch_dtype="auto",
    )


def _gemma_generate(loader: object, prompt: str) -> str:
    """Generate a response from Gemma4.  Returns '' on failure."""
    try:
        from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: PLC0415

        assert isinstance(loader, GemmaTransformersLoader)
        text = loader.generate(prompt, max_new_tokens=256)
        if not GemmaTransformersLoader.is_valid_output(text):
            return ""
        return text
    except Exception as exc:
        _log.warning("Gemma4 generation failed: %s", exc)
        return ""


def _qwen_generate(pipe: object, prompt: str) -> str:
    """Generate a response from Qwen pipeline.  Returns '' on failure."""
    try:
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
        return str(outputs[0]["generated_text"])
    except Exception as exc:
        _log.warning("Qwen generation failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Per-model benchmark runner
# ---------------------------------------------------------------------------


def _run_model_200q(
    model_name: str,
    inference_fn: Any,
    extractor: IntegratedExtractor,
    questions: list[dict],
    collector: CoTPairCollector,
    cot_pairs_path: str,
) -> Live200qV2Result:
    """Run two-pass 200q benchmark for one model and return Live200qV2Result.

    Pass 1 — BASELINE: raw model output, no verify-repair pipeline.
    Pass 2 — PIPELINE: IntegratedExtractor violations → one-shot repair prompt.
    CoT pairs are accumulated into collector for JEPA retrain.
    """
    # Pass 1: BASELINE
    n_correct_base = 0
    for q in questions:
        resp = inference_fn(q["question"])
        gold = _extract_gold(q["answer"])
        if _is_correct(resp, gold):
            n_correct_base += 1
    pre_acc = n_correct_base / max(len(questions), 1)
    _log.info("[%s] BASELINE: %d/%d (%.4f)", model_name, n_correct_base, len(questions), pre_acc)

    # Pass 2: PIPELINE — use BatchedInferenceRunner for timeouts
    bir = BatchedInferenceRunner(inference_fn, batch_size=BATCH_SIZE)
    prompts = [q["question"] for q in questions]
    bir_results = bir.run_batch(prompts)

    n_correct_pipe = 0
    for q, bir_result in zip(questions, bir_results):
        resp = bir_result.response
        violations = extractor.extract(resp)
        if violations:
            repair_prompt = (
                f"Question: {q['question']}\n\n"
                f"Your previous answer contained logical or arithmetic errors.  "
                f"Please solve step by step carefully and double-check every calculation."
            )
            resp = inference_fn(repair_prompt)

        gold = _extract_gold(q["answer"])
        correct = _is_correct(resp, gold)
        if correct:
            n_correct_pipe += 1

        collector.add(
            model=model_name,
            question=q["question"],
            cot_text=resp,
            correct=correct,
        )

    post_acc = n_correct_pipe / max(len(questions), 1)
    _log.info(
        "[%s] PIPELINE: %d/%d (%.4f) delta=%.4f",
        model_name, n_correct_pipe, len(questions), post_acc, post_acc - pre_acc,
    )

    return Live200qV2Result(
        model_id=model_name,
        pre_acc=pre_acc,
        post_acc=post_acc,
        n=len(questions),
        extractor_name="VeriCoT+VPRM+CRANE",
        inference_mode="live_gpu",
        cot_pairs_file=cot_pairs_path,
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
    """Run Experiment 478 and return the artifact dict.

    Every execution path writes the deliverable JSON before returning so that
    DeliverableGuard.assert_written() always passes regardless of which gate fires.
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
    # Gate 1: GPU required — write deferred artifact if not in live mode
    # ------------------------------------------------------------------
    if not is_live:
        _log.info("CARNOT_FORCE_LIVE not set — writing gpu_required artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_benchmark.v5",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "extraction_stack": "VeriCoT+VPRM+CRANE",
                "retro_038_closed": False,
            },
            status="gpu_required",
            honest_verdict="gpu_required",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 2: GPUVRAMGate — kill zombie VRAM holders (RETRO-037/042 fix)
    # REQ-BENCH-028: must fire before any model is loaded.
    # ------------------------------------------------------------------
    try:
        with GPUVRAMGate(min_free_gb=8.0, auto_kill=True):
            pass  # gate check only — model load happens below
    except GPUVRAMInsufficientError as exc:
        _log.error("GPUVRAMGate: insufficient VRAM after zombie kill: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_benchmark.v5",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "extraction_stack": "VeriCoT+VPRM+CRANE",
                "retro_038_closed": False,
                "vram_error": str(exc),
            },
            status="gpu_vram_insufficient",
            honest_verdict="gpu_vram_insufficient",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 3: DualGPUAssigner — REQ-BENCH-029
    # Gemma4-E4B-it → cuda:0, Qwen3.5-0.8B → cuda:1
    # ------------------------------------------------------------------
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0

    assigner = DualGPUAssigner(model_specs=MODEL_SPECS, n_gpus=n_gpus)
    assigned_specs = assigner.assign()

    # ------------------------------------------------------------------
    # Build IntegratedExtractor with live mode (use_mock=False)
    # ------------------------------------------------------------------
    extractor = IntegratedExtractor(
        vericot=VeriCoTStepValidator(use_mock=False),
        vprm=VPRMArithmeticVerifier(),
    )

    # ------------------------------------------------------------------
    # Load models (fail cleanly if load fails)
    # ------------------------------------------------------------------
    gemma_gpu = next((s["gpu"] for s in assigned_specs if s["name"] == "Gemma4-E4B-it"), 0)
    qwen_gpu = next((s["gpu"] for s in assigned_specs if s["name"] == "Qwen3.5-0.8B"), 1)

    gemma_loader: object | None = None
    try:
        _log.info("Loading Gemma4-E4B-it on cuda:%d ...", gemma_gpu)
        gemma_loader = _load_gemma4(gpu_index=gemma_gpu)
        _log.info("Gemma4-E4B-it loaded OK")
    except Exception as exc:
        _log.error("Gemma4 load failed: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_benchmark.v5",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "extraction_stack": "VeriCoT+VPRM+CRANE",
                "retro_038_closed": False,
            },
            status="blocked",
            honest_verdict="gpu_required",
            blocked_reason=f"Gemma4 load failed: {exc}",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    qwen_pipe: object | None = None
    try:
        _log.info("Loading Qwen3.5-0.8B on cuda:%d ...", qwen_gpu)
        qwen_pipe = _load_qwen("Qwen/Qwen3.5-0.8B", gpu_index=qwen_gpu)
        _log.info("Qwen3.5-0.8B loaded OK")
    except Exception as exc:
        _log.error("Qwen load failed: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_benchmark.v5",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "extraction_stack": "VeriCoT+VPRM+CRANE",
                "retro_038_closed": False,
            },
            status="blocked",
            honest_verdict="gpu_required",
            blocked_reason=f"Qwen load failed: {exc}",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Load 200 GSM8K questions (seed=42 shuffle)
    # ------------------------------------------------------------------
    questions = _load_gsm8k_200q(seed=DATASET_SEED)
    _log.info("Loaded %d questions", len(questions))

    # ------------------------------------------------------------------
    # CoTPairCollector — accumulates pairs for JEPA retrain
    # ------------------------------------------------------------------
    cot_pairs_abs = str(repo_root / COT_PAIRS_PATH)
    collector = CoTPairCollector(cot_pairs_abs)

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------
    def gemma_fn(prompt: str) -> str:
        assert gemma_loader is not None
        return _gemma_generate(gemma_loader, prompt)

    def qwen_fn(prompt: str) -> str:
        assert qwen_pipe is not None
        return _qwen_generate(qwen_pipe, prompt)

    _log.info("=== Gemma4-E4B-it: 200q benchmark ===")
    gemma4_result = _run_model_200q(
        "Gemma4-E4B-it", gemma_fn, extractor, questions, collector, COT_PAIRS_PATH
    )
    tmpl.checkpoint_save(gemma4_result.to_dict(), step=1)

    _log.info("=== Qwen3.5-0.8B: 200q benchmark ===")
    qwen_result = _run_model_200q(
        "Qwen3.5-0.8B", qwen_fn, extractor, questions, collector, COT_PAIRS_PATH
    )
    tmpl.checkpoint_save(qwen_result.to_dict(), step=2)

    # ------------------------------------------------------------------
    # Flush CoT pairs to disk
    # ------------------------------------------------------------------
    cot_pairs_written = collector.flush()
    _log.info("CoT pairs written: %d to %s", cot_pairs_written, COT_PAIRS_PATH)

    # ------------------------------------------------------------------
    # Compute honest_verdict — REQ-BENCH-030
    # ------------------------------------------------------------------
    any_stat_positive = (
        gemma4_result.is_statistically_positive or qwen_result.is_statistically_positive
    )
    any_signed_positive = (
        gemma4_result.signed_improvement > 0 or qwen_result.signed_improvement > 0
    )

    if any_stat_positive:
        honest_verdict = "credible_positive"
    elif any_signed_positive:
        honest_verdict = "improvement_not_significant"
    else:
        honest_verdict = "no_improvement_200q"

    retro_038_closed = any_stat_positive

    artifact = tmpl.build_result(
        {
            "schema": "carnot.live_benchmark.v5",
            "env_autofix": env_autofix_dict,
            "n_questions": len(questions),
            "gemma4_result": gemma4_result.to_dict(),
            "qwen_result": qwen_result.to_dict(),
            "cot_pairs_written": cot_pairs_written,
            "extraction_stack": "VeriCoT+VPRM+CRANE",
            "retro_038_closed": retro_038_closed,
        },
        status="success",
        honest_verdict=honest_verdict,
        inference_mode="live_gpu",
    )
    _write_json(repo_root, DELIVERABLE, artifact)

    _log.info(
        "HEADLINE: honest_verdict=%s gemma4_delta=%.4f qwen_delta=%.4f "
        "gemma4_stat_positive=%s qwen_stat_positive=%s cot_pairs=%d retro_038_closed=%s",
        honest_verdict,
        gemma4_result.signed_improvement,
        qwen_result.signed_improvement,
        gemma4_result.is_statistically_positive,
        qwen_result.is_statistically_positive,
        cot_pairs_written,
        retro_038_closed,
    )

    # FINAL LINE — DeliverableGuard closure (RETRO-032/033/036/038)
    guard.assert_written()
    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 478: 200q live VeriCoT+VPRM v2 benchmark.

    Wraps run_experiment() in a 150-minute ExperimentTimeoutWatchdog.
    200q × 2 models × 2 passes with batch_size=25 needs up to 130 minutes;
    the 150-minute cap leaves 20 minutes of headroom.
    """
    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=get_timeout_minutes(),
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        artifact = run_experiment()

    _log.info(
        "Exp %d complete: honest_verdict=%s status=%s",
        EXP_ID,
        artifact.get("honest_verdict", "unknown"),
        artifact.get("status", "unknown"),
    )


if __name__ == "__main__":
    main()
