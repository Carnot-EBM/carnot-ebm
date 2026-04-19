#!/usr/bin/env python3
"""Experiment 514: Live 100q Precision v7 — JIT VRAM Gating (RETRO-033 seventh attempt).

**Researcher summary:**
    Exp 502/503/504 missed RETRO-033 close because planning-time VRAM forecasts
    (VRAMBudgetLedger) were stale — computed at script startup, not at model.load() time.
    Between startup and load(), the conductor or a background process could allocate GPU
    memory, invalidating the forecast.  Exp 513 (JITVRAMCheck) resolves this by querying
    pynvml immediately before each model.load() call.

    This is the seventh attempt.  Every previous blocking root cause has been fixed:
    - RETRO-022: env propagation (apply_env_autofix, conductor fix)
    - RETRO-033: zombie VRAM (GPUVRAMGateV2 kill_first=True)
    - RETRO-044: gate ordering (GPUVRAMGateV2 check-after-kill)
    - RETRO-048: FP16 too large (Gemma4QuantizedLoader Q4_K_M)
    - RETRO-051: stale VRAM forecast (JITVRAMCheck, this fix)

**Gate chain (in order; EVERY exit path writes the deliverable):**
    0. apply_env_autofix()              — inject CARNOT_FORCE_LIVE if GPU detected
    1. ExperimentTimeoutWatchdog(514)   — 120-min outer hard cap
    2. DeliverableGuard                 — registered at startup
    3. GPUVRAMGateV2(5.0 GB, kill_first=True)
    4. JIT VRAM gate → Gemma4-INT4 on cuda:0 (requires 10.0 GB)
    5. JIT VRAM gate → Qwen3.5-0.8B on cuda:1 (requires 1.5 GB)
    6. Load 100 GSM8K questions
    7. LongRunBenchmarkExecutor batch_size=25
    8. Per-question: baseline inference → VeriCoT+VPRM extraction → repair if violations
    9. Write CoT pairs → results/exp514_cot_pairs.json (FOVER format)
    10. Build artifact with all required fields
    11. tmpl.assert_deliverable_written()   — FINAL LINE

Spec: REQ-BENCH-014, REQ-BENCH-015,
      SCENARIO-BENCH-033, SCENARIO-BENCH-034
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix() MUST be called before any CUDA import.
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
from typing import Any, Optional

from carnot.extraction.integrated_extractor import IntegratedExtractor
from carnot.extraction.vericot_validator import VeriCoTStepValidator
from carnot.extraction.vprm_verifier import VPRMArithmeticVerifier
from carnot.pipeline.cot_pair_collector import CoTPairCollector
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog, get_timeout_minutes
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader
from carnot.pipeline.gpu_vram_gate import GPUVRAMInsufficientError
from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2
from carnot.pipeline.live_100q_v7_helpers import (
    PrecisionBenchmarkResult,
    load_jit_gated_model,
    run_100q_benchmark,
    wilson_ci,
    write_cot_pairs,
)
from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 514
EXP_TITLE = "Live 100q Precision v7 — JIT VRAM Gating (RETRO-033 seventh attempt)"
DELIVERABLE = "results/experiment_514_live_100q_precision_v7.json"
COT_PAIRS_PATH = "results/exp514_cot_pairs.json"
N_QUESTIONS = 100
GSM8K_SEED = 42

GEMMA4_REQUIRED_GB = 10.0
QWEN_REQUIRED_GB = 1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_gsm8k_questions(n: int, seed: int) -> list:
    """Load n GSM8K test questions, shuffled with a fixed seed.

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
            "answer": f"She starts with {a} and gets {b} more. #### {c}",
            "source": "synthetic",
        })
    _log.info("Using %d synthetic GSM8K fallback questions", n)
    return synthetic


def _load_qwen_pipeline(device: int) -> object:
    """Load Qwen3.5-0.8B HF text-generation pipeline on cuda:N.

    Why explicit device_map dict:
        device_map='auto' spreads across all GPUs.  Passing {'': 'cuda:N'} pins
        every layer to a single GPU so Gemma4 and Qwen don't fight for the same VRAM.
    """
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(f"transformers not installed: {exc}") from exc

    return hf_pipeline(
        "text-generation",
        model="Qwen/Qwen3.5-0.8B",
        device_map={"": f"cuda:{device}"},
        torch_dtype="auto",
    )


def _qwen_inference(pipe: object, prompt: str) -> str:
    """Generate a response from Qwen HF pipeline."""
    try:
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)  # type: ignore[operator]
        return str(outputs[0]["generated_text"])
    except Exception as exc:
        _log.warning("Qwen inference failed: %s", exc)
        return ""


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


def run_experiment(repo_root: Optional[Path] = None) -> dict:
    """Run Experiment 514 and return the artifact dict.

    All execution paths write the deliverable JSON before returning so that
    DeliverableGuard.assert_written() always passes.

    Parameters
    ----------
    repo_root : Path, optional
        Override the repository root (used in tests).
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

    def _deferred(reason: str, extra: dict | None = None) -> dict:
        """Write a gpu_required deferred artifact and return it."""
        payload: dict = {
            "artifact_type": "carnot.live_precision.v7",
            "env_autofix": env_autofix_dict,
            "inference_mode": "gpu_required",
            "n_questions": N_QUESTIONS,
            "baseline_accuracy": None,
            "pipeline_accuracy": None,
            "signed_improvement": None,
            "wilson_95ci_lower": None,
            "wilson_95ci_upper": None,
            "is_positive": False,
            "retro_033_closed": False,
            "cot_pairs_written": None,
            "jit_vram_check_applied": True,
            "honest_verdict": "gpu_required",
            "deferred_reason": reason,
        }
        if extra:
            payload.update(extra)
        art = tmpl.build_result(payload, status="gpu_required")
        _write_json(repo_root, DELIVERABLE, art)
        guard.assert_written()
        return art

    # ------------------------------------------------------------------
    # Gate 0: GPU required
    # ------------------------------------------------------------------
    if not is_live:
        _log.info("CARNOT_FORCE_LIVE not set — writing deferred artifact.")
        return _deferred("CARNOT_FORCE_LIVE not set")

    # ------------------------------------------------------------------
    # Gate 1: GPUVRAMGateV2 — kill zombie processes FIRST, then confirm VRAM
    # min_free_gb=5.0: after zombie kill, at least 5 GB must be free before
    # JIT gates check per-model requirements individually.
    # ------------------------------------------------------------------
    try:
        with GPUVRAMGateV2(min_free_gb=5.0, kill_first=True):
            pass
    except GPUVRAMInsufficientError as exc:
        _log.error("GPUVRAMGateV2 failed: %s", exc)
        return _deferred(f"GPUVRAMGateV2 insufficient: {exc}", {"vram_error": str(exc)})

    # ------------------------------------------------------------------
    # Gate 2: JIT VRAM gate → Gemma4-INT4 on cuda:0 (10 GB required)
    # ------------------------------------------------------------------
    gemma4_gguf_path = os.environ.get("CARNOT_GEMMA4_GGUF_PATH", "")

    def _gemma4_factory() -> Gemma4QuantizedLoader:
        return Gemma4QuantizedLoader(model_path=gemma4_gguf_path, n_gpu_layers=-1)

    gemma4_loader = load_jit_gated_model(
        loader_factory=_gemma4_factory,
        model_id="gemma4-int4",
        required_gb=GEMMA4_REQUIRED_GB,
        device=0,
    )
    if gemma4_loader is None:
        _log.warning("Gemma4-INT4 JIT gate blocked — VRAM insufficient on cuda:0")
        return _deferred("JIT VRAM gate blocked Gemma4-INT4 on cuda:0")

    # ------------------------------------------------------------------
    # Gate 3: JIT VRAM gate → Qwen3.5-0.8B on cuda:1 (1.5 GB required)
    # ------------------------------------------------------------------
    qwen_pipe_holder: list = []

    def _qwen_factory() -> object:
        pipe = _load_qwen_pipeline(device=1)
        qwen_pipe_holder.append(pipe)
        # Return a lightweight wrapper that has .load() so load_jit_gated_model works
        class _Wrapper:
            def load(self) -> bool:
                return True
        return _Wrapper()

    qwen_gate = load_jit_gated_model(
        loader_factory=_qwen_factory,
        model_id="qwen3.5-0.8b",
        required_gb=QWEN_REQUIRED_GB,
        device=1,
    )
    if qwen_gate is None:
        _log.warning("Qwen3.5-0.8B JIT gate blocked — VRAM insufficient on cuda:1")
        return _deferred("JIT VRAM gate blocked Qwen3.5-0.8B on cuda:1")

    qwen_pipe = qwen_pipe_holder[0] if qwen_pipe_holder else None

    # ------------------------------------------------------------------
    # Load 100 GSM8K questions
    # ------------------------------------------------------------------
    questions = _load_gsm8k_questions(N_QUESTIONS, seed=GSM8K_SEED)
    _log.info("Loaded %d questions (seed=%d)", len(questions), GSM8K_SEED)

    # ------------------------------------------------------------------
    # IntegratedExtractor for violation detection
    # ------------------------------------------------------------------
    extractor = IntegratedExtractor(
        vericot=VeriCoTStepValidator(use_mock=False),
        vprm=VPRMArithmeticVerifier(),
    )

    # ------------------------------------------------------------------
    # LongRunBenchmarkExecutor: batch_size=25, checkpoint between batches
    # ------------------------------------------------------------------
    executor = LongRunBenchmarkExecutor(
        batch_size=25,
        checkpoint_dir=str(repo_root / "results" / "batch_ckpt" / "exp514"),
    )

    def _gemma4_inference_fn(question_dict: dict) -> dict:
        """Run one question through Gemma4 baseline + pipeline, return result dict."""
        prompt = question_dict["question"]
        baseline_resp = gemma4_loader.generate(prompt)
        violations = extractor.extract(baseline_resp)
        if violations:
            repair = (
                f"Question: {prompt}\n\n"
                "Your previous answer had errors. Solve step by step carefully."
            )
            pipeline_resp = gemma4_loader.generate(repair)
        else:
            pipeline_resp = baseline_resp

        from carnot.pipeline.live_100q_v7_helpers import _extract_answer, _is_correct
        gold = _extract_answer(question_dict.get("answer", ""))
        return {
            "question": prompt,
            "baseline_correct": _is_correct(baseline_resp, gold),
            "pipeline_correct": _is_correct(pipeline_resp, gold),
            "cot_text": pipeline_resp,
        }

    def _qwen_inference_fn(question_dict: dict) -> dict:
        """Run one question through Qwen baseline + pipeline, return result dict."""
        prompt = question_dict["question"]
        baseline_resp = _qwen_inference(qwen_pipe, prompt)
        violations = extractor.extract(baseline_resp)
        if violations:
            repair = (
                f"Question: {prompt}\n\n"
                "Your previous answer had errors. Solve step by step carefully."
            )
            pipeline_resp = _qwen_inference(qwen_pipe, repair)
        else:
            pipeline_resp = baseline_resp

        from carnot.pipeline.live_100q_v7_helpers import _extract_answer, _is_correct
        gold = _extract_answer(question_dict.get("answer", ""))
        return {
            "question": prompt,
            "baseline_correct": _is_correct(baseline_resp, gold),
            "pipeline_correct": _is_correct(pipeline_resp, gold),
            "cot_text": pipeline_resp,
        }

    # Run Gemma4 batches
    _log.info("=== Running Gemma4-INT4 benchmark (cuda:0, %dq, batch=25) ===", len(questions))
    gemma4_batches = executor.partition(questions)
    for batch in gemma4_batches:
        executor.run_batch(batch, _gemma4_inference_fn, watchdog_timeout_minutes=50)
        executor.save_batch(batch, prefix="exp514_gemma4")
    gemma4_run = executor.assemble(gemma4_batches)
    tmpl.checkpoint_save({"gemma4_done": True, "gemma4_verdict": gemma4_run.honest_verdict}, step=1)

    # Run Qwen batches
    _log.info("=== Running Qwen3.5-0.8B benchmark (cuda:1, %dq, batch=25) ===", len(questions))
    executor2 = LongRunBenchmarkExecutor(
        batch_size=25,
        checkpoint_dir=str(repo_root / "results" / "batch_ckpt" / "exp514_qwen"),
    )
    qwen_batches = executor2.partition(questions)
    for batch in qwen_batches:
        executor2.run_batch(batch, _qwen_inference_fn, watchdog_timeout_minutes=50)
        executor2.save_batch(batch, prefix="exp514_qwen")
    qwen_run = executor2.assemble(qwen_batches)
    tmpl.checkpoint_save({"qwen_done": True, "qwen_verdict": qwen_run.honest_verdict}, step=2)

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    def _aggregate(run_result, model_id: str) -> dict:
        results = run_result.all_results
        if not results:
            return {
                "model_id": model_id, "n": 0,
                "baseline_correct": 0, "pipeline_correct": 0,
                "baseline_accuracy": 0.0, "pipeline_accuracy": 0.0,
                "signed_improvement": 0.0, "is_positive": False,
                "wilson_95ci_lower": 0.0, "wilson_95ci_upper": 0.0,
            }
        n = len(results)
        bc = sum(1 for r in results if r.get("baseline_correct", False))
        pc = sum(1 for r in results if r.get("pipeline_correct", False))
        ba = bc / n
        pa = pc / n
        lo, hi = wilson_ci(pc, n)
        return {
            "model_id": model_id,
            "n": n,
            "baseline_correct": bc,
            "pipeline_correct": pc,
            "baseline_accuracy": ba,
            "pipeline_accuracy": pa,
            "signed_improvement": pa - ba,
            "is_positive": (pa - ba) > 0,
            "wilson_95ci_lower": lo,
            "wilson_95ci_upper": hi,
        }

    gemma4_stats = _aggregate(gemma4_run, "Gemma4-INT4")
    qwen_stats = _aggregate(qwen_run, "Qwen3.5-0.8B")

    # Use Gemma4 as the primary model for headline metrics
    primary = gemma4_stats

    # ------------------------------------------------------------------
    # Collect CoT pairs and write to exp514_cot_pairs.json
    # ------------------------------------------------------------------
    all_cot_pairs = []
    for r in gemma4_run.all_results:
        all_cot_pairs.append({
            "question": r.get("question", ""),
            "cot_text": r.get("cot_text", ""),
            "correct": r.get("pipeline_correct", False),
            "model_id": "Gemma4-INT4",
        })
    for r in qwen_run.all_results:
        all_cot_pairs.append({
            "question": r.get("question", ""),
            "cot_text": r.get("cot_text", ""),
            "correct": r.get("pipeline_correct", False),
            "model_id": "Qwen3.5-0.8B",
        })

    cot_path = str(repo_root / COT_PAIRS_PATH)
    n_cot_written = write_cot_pairs(all_cot_pairs, cot_path) if all_cot_pairs else 0
    cot_pairs_written = cot_path if n_cot_written > 0 else None

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    retro_033_closed = primary["is_positive"]
    honest_verdict: str
    if retro_033_closed:
        honest_verdict = "retro_033_closed"
    else:
        honest_verdict = "live_no_improvement"

    artifact = tmpl.build_result(
        {
            "artifact_type": "carnot.live_precision.v7",
            "env_autofix": env_autofix_dict,
            "inference_mode": "live_gpu",
            "n_questions": N_QUESTIONS,
            "baseline_accuracy": primary["baseline_accuracy"],
            "pipeline_accuracy": primary["pipeline_accuracy"],
            "signed_improvement": primary["signed_improvement"],
            "wilson_95ci_lower": primary["wilson_95ci_lower"],
            "wilson_95ci_upper": primary["wilson_95ci_upper"],
            "is_positive": primary["is_positive"],
            "retro_033_closed": retro_033_closed,
            "cot_pairs_written": cot_pairs_written,
            "jit_vram_check_applied": True,
            "honest_verdict": honest_verdict,
            "gemma4_result": gemma4_stats,
            "qwen_result": qwen_stats,
            "gemma4_run_verdict": gemma4_run.honest_verdict,
            "qwen_run_verdict": qwen_run.honest_verdict,
        },
        status="success",
    )
    _write_json(repo_root, DELIVERABLE, artifact)

    _log.info(
        "HEADLINE: honest_verdict=%s retro_033_closed=%s "
        "baseline=%.4f pipeline=%.4f delta=%.4f cot_pairs=%s",
        honest_verdict, retro_033_closed,
        primary["baseline_accuracy"], primary["pipeline_accuracy"],
        primary["signed_improvement"], cot_pairs_written,
    )

    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 514: Live 100q precision v7 with JIT VRAM gating."""
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
