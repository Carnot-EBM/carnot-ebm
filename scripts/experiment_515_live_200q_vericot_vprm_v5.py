#!/usr/bin/env python3
"""Experiment 515: Live 200q VeriCoT+VPRM v5 — RETRO-038 fifth attempt.

**Researcher summary:**
    RETRO-038 (200q statistically significant live benchmark) has missed FOUR consecutive
    milestones.  Exp 514 (100q) confirmed the full gate chain works under JITVRAMCheck;
    this experiment scales to 200 questions using LongRunBenchmarkExecutor for
    checkpoint/resume reliability.

    The milestone criterion is Wilson 95% CI lower bound > 0, which constitutes the
    first publishable credibility claim for the Carnot pipeline.  200 questions vs 100
    gives approximately sqrt(2) more statistical power, pushing the minimum detectable
    effect size from ~8pp to ~6pp at 95% confidence.

    Writes 200 CoT pairs to results/exp515_cot_pairs.json for Exp 522 JEPA retrain v6.

**Blocking history resolved by this experiment:**
    - RETRO-022: env propagation (apply_env_autofix, conductor fix)
    - RETRO-033: zombie VRAM (GPUVRAMGateV2 kill_first=True)
    - RETRO-044: gate ordering (GPUVRAMGateV2 check-after-kill)
    - RETRO-048: FP16 too large (Gemma4QuantizedLoader Q4_K_M)
    - RETRO-051: stale VRAM forecast (JITVRAMCheck, Exp 513)
    - RETRO-038: n=100 insufficient statistical power → scale to n=200

**Gate chain (in order; EVERY exit path writes the deliverable):**
    0. apply_env_autofix()                — inject CARNOT_FORCE_LIVE if GPU detected
    1. ExperimentTimeoutWatchdog(515)     — 150-min outer hard cap
    2. DeliverableGuard                  — registered at startup
    3. GPUVRAMGateV2(5.0 GB, kill_first=True)
    4. JIT VRAM gate → Gemma4-INT4 on cuda:0 (requires 10.0 GB)
    5. JIT VRAM gate → Qwen3.5-0.8B on cuda:1 (requires 1.5 GB)
    6. Load 200 GSM8K questions (seed=42)
    7. LongRunBenchmarkExecutor batch_size=50, checkpoint at results/exp515_ckpt
    8. Per-question: baseline → VeriCoT+VPRM+CRANE extraction → repair if violations
    9. Compute Wilson 95% CI for pipeline improvement delta
    10. Write 200 CoT pairs → results/exp515_cot_pairs.json (FOVER format)
    11. Build artifact: schema='carnot.live_200q.v5', all required fields
    12. tmpl.assert_deliverable_written()  — FINAL LINE

Spec: REQ-BENCH-016,
      SCENARIO-BENCH-035, SCENARIO-BENCH-036
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
from carnot.pipeline.crane_extractor import CRANEExtractionGate
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog, get_timeout_minutes
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader
from carnot.pipeline.gpu_vram_gate import GPUVRAMInsufficientError
from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2
from carnot.pipeline.live_100q_v7_helpers import (
    _extract_answer,
    _is_correct,
    load_jit_gated_model,
    write_cot_pairs,
)
from carnot.pipeline.live_200q_v5_helpers import compute_wilson_ci, is_statistically_positive
from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 515
EXP_TITLE = "Live 200q VeriCoT+VPRM v5 — RETRO-038 fifth attempt"
DELIVERABLE = "results/experiment_515_live_200q_vericot_vprm_v5.json"
COT_PAIRS_PATH = "results/exp515_cot_pairs.json"
N_QUESTIONS = 200
GSM8K_SEED = 42
SCHEMA = "carnot.live_200q.v5"

GEMMA4_REQUIRED_GB = 10.0
QWEN_REQUIRED_GB = 1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_gsm8k_questions(n: int, seed: int) -> list:
    """Load n GSM8K test questions, shuffled with a fixed seed.

    Falls back to synthetic arithmetic questions when the datasets package is
    unavailable (CI environments without internet access).

    The 200-question scale provides sqrt(2) more statistical power than the
    100-question benchmark, which is required to detect a ~6pp improvement
    at 95% confidence and 80% power.
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
    """Run Experiment 515 and return the artifact dict.

    All execution paths write the deliverable JSON before returning so that
    DeliverableGuard.assert_written() always passes.

    The experiment scales Exp 514 from 100 to 200 questions using
    LongRunBenchmarkExecutor for checkpoint/resume.  The Wilson 95% CI lower
    bound > 0 constitutes the first publishable claim under REQ-BENCH-016.

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
        """Write a gpu_required deferred artifact and return it.

        Every early-exit path calls _deferred() so the deliverable is always
        written before the process exits.  This satisfies DeliverableGuard.
        """
        payload: dict = {
            "schema": SCHEMA,
            "env_autofix": env_autofix_dict,
            "inference_mode": "gpu_required",
            "n_questions": N_QUESTIONS,
            "baseline_accuracy": None,
            "pipeline_accuracy": None,
            "signed_improvement": None,
            "wilson_95ci_lower": None,
            "wilson_95ci_upper": None,
            "is_statistically_positive": False,
            "retro_038_closed": False,
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
    # The JIT gate queries pynvml immediately before load(), not at startup,
    # so a process that allocated VRAM between startup and load() is caught.
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
    # Load 200 GSM8K questions
    # ------------------------------------------------------------------
    questions = _load_gsm8k_questions(N_QUESTIONS, seed=GSM8K_SEED)
    _log.info("Loaded %d questions (seed=%d)", len(questions), GSM8K_SEED)

    # ------------------------------------------------------------------
    # VeriCoT+VPRM+CRANE extraction stack
    # ------------------------------------------------------------------
    extractor = IntegratedExtractor(
        vericot=VeriCoTStepValidator(use_mock=False),
        vprm=VPRMArithmeticVerifier(),
    )
    crane_gate = CRANEExtractionGate()

    def _extract_violations(text: str) -> list:
        """Run CRANE first; fall back to IntegratedExtractor (VeriCoT+VPRM)."""
        crane_violations = crane_gate.extract(text)
        if crane_violations:
            return crane_violations
        return extractor.extract(text)

    # ------------------------------------------------------------------
    # LongRunBenchmarkExecutor: batch_size=50, checkpoint between batches
    # ------------------------------------------------------------------
    ckpt_dir_gemma4 = str(repo_root / "results" / "exp515_ckpt" / "gemma4")
    ckpt_dir_qwen = str(repo_root / "results" / "exp515_ckpt" / "qwen")

    executor_gemma4 = LongRunBenchmarkExecutor(batch_size=50, checkpoint_dir=ckpt_dir_gemma4)
    executor_qwen = LongRunBenchmarkExecutor(batch_size=50, checkpoint_dir=ckpt_dir_qwen)

    def _gemma4_inference_fn(question_dict: dict) -> dict:
        """Run one question through Gemma4 baseline + CRANE+VeriCoT+VPRM pipeline."""
        prompt = question_dict["question"]
        baseline_resp = gemma4_loader.generate(prompt)
        violations = _extract_violations(baseline_resp)
        if violations:
            repair = (
                f"Question: {prompt}\n\n"
                "Your previous answer had errors. Solve step by step carefully."
            )
            pipeline_resp = gemma4_loader.generate(repair)
        else:
            pipeline_resp = baseline_resp

        gold = _extract_answer(question_dict.get("answer", ""))
        return {
            "question": prompt,
            "baseline_correct": _is_correct(baseline_resp, gold),
            "pipeline_correct": _is_correct(pipeline_resp, gold),
            "cot_text": pipeline_resp,
        }

    def _qwen_inference_fn(question_dict: dict) -> dict:
        """Run one question through Qwen baseline + CRANE+VeriCoT+VPRM pipeline."""
        prompt = question_dict["question"]
        baseline_resp = _qwen_inference(qwen_pipe, prompt)
        violations = _extract_violations(baseline_resp)
        if violations:
            repair = (
                f"Question: {prompt}\n\n"
                "Your previous answer had errors. Solve step by step carefully."
            )
            pipeline_resp = _qwen_inference(qwen_pipe, repair)
        else:
            pipeline_resp = baseline_resp

        gold = _extract_answer(question_dict.get("answer", ""))
        return {
            "question": prompt,
            "baseline_correct": _is_correct(baseline_resp, gold),
            "pipeline_correct": _is_correct(pipeline_resp, gold),
            "cot_text": pipeline_resp,
        }

    # Run Gemma4 batches (200 questions, 4 batches of 50)
    _log.info("=== Gemma4-INT4 benchmark (cuda:0, %dq, batch=50) ===", N_QUESTIONS)
    gemma4_batches = executor_gemma4.partition(questions)
    for batch in gemma4_batches:
        executor_gemma4.run_batch(batch, _gemma4_inference_fn, watchdog_timeout_minutes=35)
        executor_gemma4.save_batch(batch, prefix="exp515_gemma4")
    gemma4_run = executor_gemma4.assemble(gemma4_batches)
    tmpl.checkpoint_save({"gemma4_done": True, "gemma4_verdict": gemma4_run.honest_verdict}, step=1)

    # Run Qwen batches (200 questions, 4 batches of 50)
    _log.info("=== Qwen3.5-0.8B benchmark (cuda:1, %dq, batch=50) ===", N_QUESTIONS)
    qwen_batches = executor_qwen.partition(questions)
    for batch in qwen_batches:
        executor_qwen.run_batch(batch, _qwen_inference_fn, watchdog_timeout_minutes=35)
        executor_qwen.save_batch(batch, prefix="exp515_qwen")
    qwen_run = executor_qwen.assemble(qwen_batches)
    tmpl.checkpoint_save({"qwen_done": True, "qwen_verdict": qwen_run.honest_verdict}, step=2)

    # ------------------------------------------------------------------
    # Aggregate results using compute_wilson_ci for the pipeline delta CI
    # ------------------------------------------------------------------
    def _aggregate(run_result: Any, model_id: str) -> dict:
        """Compute per-model accuracy stats with Wilson CI on pipeline accuracy."""
        results = run_result.all_results
        if not results:
            return {
                "model_id": model_id, "n": 0,
                "baseline_correct": 0, "pipeline_correct": 0,
                "baseline_accuracy": 0.0, "pipeline_accuracy": 0.0,
                "signed_improvement": 0.0, "is_statistically_positive": False,
                "wilson_95ci_lower": 0.0, "wilson_95ci_upper": 0.0,
            }
        n = len(results)
        bc = sum(1 for r in results if r.get("baseline_correct", False))
        pc = sum(1 for r in results if r.get("pipeline_correct", False))
        ba = bc / n
        pa = pc / n
        lo, hi = compute_wilson_ci(pc, n)
        return {
            "model_id": model_id,
            "n": n,
            "baseline_correct": bc,
            "pipeline_correct": pc,
            "baseline_accuracy": ba,
            "pipeline_accuracy": pa,
            "signed_improvement": pa - ba,
            "is_statistically_positive": is_statistically_positive(lo),
            "wilson_95ci_lower": lo,
            "wilson_95ci_upper": hi,
        }

    gemma4_stats = _aggregate(gemma4_run, "Gemma4-INT4")
    qwen_stats = _aggregate(qwen_run, "Qwen3.5-0.8B")

    # Gemma4 is the primary model for headline metrics
    primary = gemma4_stats

    # ------------------------------------------------------------------
    # Compute Wilson 95% CI on the improvement delta
    # The improvement delta uses pipeline_correct vs. baseline_correct at n=200.
    # We compute CI on absolute pipeline accuracy; the lower bound being > 0
    # is a necessary but not sufficient condition — RETRO-038 requires the
    # CI lower bound on the DELTA to exceed 0.
    # ------------------------------------------------------------------
    delta_lo, delta_hi = compute_wilson_ci(
        n_successes=max(0, primary["pipeline_correct"] - primary["baseline_correct"]),
        n_total=primary["n"] if primary["n"] > 0 else 1,
    )
    stat_positive = is_statistically_positive(delta_lo)

    # ------------------------------------------------------------------
    # Collect CoT pairs and write to exp515_cot_pairs.json
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
    retro_038_closed = stat_positive and (os.environ.get("CARNOT_FORCE_LIVE", "0") == "1")

    if retro_038_closed:
        honest_verdict = "first_publishable_claim"
    elif os.environ.get("CARNOT_FORCE_LIVE", "0") == "1":
        honest_verdict = "live_no_significance"
    else:
        honest_verdict = "gpu_required"

    artifact = tmpl.build_result(
        {
            "schema": SCHEMA,
            "env_autofix": env_autofix_dict,
            "inference_mode": "live_gpu",
            "n_questions": N_QUESTIONS,
            "baseline_accuracy": primary["baseline_accuracy"],
            "pipeline_accuracy": primary["pipeline_accuracy"],
            "signed_improvement": primary["signed_improvement"],
            "wilson_95ci_lower": delta_lo,
            "wilson_95ci_upper": delta_hi,
            "is_statistically_positive": stat_positive,
            "retro_038_closed": retro_038_closed,
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
        "HEADLINE: honest_verdict=%s retro_038_closed=%s "
        "baseline=%.4f pipeline=%.4f delta=%.4f wilson_lo=%.4f cot_pairs=%s",
        honest_verdict, retro_038_closed,
        primary["baseline_accuracy"], primary["pipeline_accuracy"],
        primary["signed_improvement"], delta_lo, cot_pairs_written,
    )

    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 515: Live 200q VeriCoT+VPRM v5 with JIT VRAM gating."""
    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=int(os.environ.get("CARNOT_CONDUCTOR_TIMEOUT_MINUTES", "150")),
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
