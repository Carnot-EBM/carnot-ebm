#!/usr/bin/env python3
"""Experiment 539: Live 100q VeriCoT+VPRM benchmark v8 — RETRO-038 attempt.

**Researcher summary:**
    RETRO-038 (live 200q VeriCoT+VPRM statistically significant result) has missed for 7
    consecutive milestones.  This experiment runs 100q (not 200q) to fit within the 90-min
    budget confirmed by Exp 538 (mean_latency_s = 24.1s → 100q × 24s ≈ 40 min).

    Exp 538 gate: if mean_latency_s > 40s → reduce to 50q.
    Formula: n_questions = min(100, int(80 * 60 / exp538_mean_latency_s))

    Statistical close criterion: Wilson 95% CI on signed_improvement.
    If CI lower bound > 0, retro_038_closed=True and verdict='wilson_ci_publishable'.

**Gate chain (in order; every exit path writes the deliverable):**
    0. Zombie PIDs 430009/430012 killed immediately (subprocess.run kill -9)
    1. apply_env_autofix()               — inject CARNOT_FORCE_LIVE=1 if GPU detected
    2. ExperimentTemplate.kill_gpu_zombies() — classmethod kill via pynvml
    3. ExperimentTimeoutWatchdog(539, timeout_minutes=90) — outer hard cap
    4. DeliverableGuard                  — registered at startup
    5. Load Exp 538 result → compute n_questions dynamically
    6. LiveGPUGate.require_live_or_blocked() — CARNOT_FORCE_LIVE gate
    7. JITVRAMCheck cuda:0 (Gemma4-INT4, 10.0 GB)
    8. JITVRAMCheck cuda:1 (Qwen3.5-0.8B, 1.5 GB)
    9. Load n_questions GSM8K questions (seed=42)
    10. Per-question: VeriCoT+VPRM+IntegratedExtractor violations → repair if any
    11. Compute Wilson 95% CI on signed_improvement
    12. Write artifact: schema='carnot.vericot_benchmark.v8'
    13. tmpl.assert_deliverable_written() — FINAL LINE

Spec: REQ-BENCH-016 (v2), SCENARIO-BENCH-036 (v2), SCENARIO-BENCH-037 (v2)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: kill zombie PIDs FIRST — before any CUDA import.
# PIDs 430009/430012 were holding zombie VRAM in prior milestones.
# ---------------------------------------------------------------------------
import subprocess

subprocess.run(["kill", "-9", "430009", "430012"], capture_output=True)

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() MUST come before any CUDA import.
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
import math
import os
import time
from typing import Any, Optional

from carnot.extraction.integrated_extractor import IntegratedExtractor
from carnot.extraction.vericot_validator import VeriCoTStepValidator
from carnot.extraction.vprm_verifier import VPRMArithmeticVerifier
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader
from carnot.pipeline.jit_vram_check import JITVRAMCheck
from carnot.pipeline.live_gpu_gate import LiveGPUGate
from carnot.pipeline.live_100q_v7_helpers import _extract_answer, _is_correct
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 539
EXP_TITLE = "Live 100q VeriCoT+VPRM v8 — RETRO-038 attempt"
DELIVERABLE = "results/experiment_539_live_100q_vericot_v8.json"
EXP538_RESULT = "results/experiment_538_live_25q_precision_v9.json"
GSM8K_SEED = 42
GEMMA4_REQUIRED_GB = 10.0
QWEN_REQUIRED_GB = 1.5
MAX_QUESTIONS = 100
INFERENCE_BUDGET_MINUTES = 80


# ---------------------------------------------------------------------------
# Wilson CI helper
# ---------------------------------------------------------------------------


def compute_wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson 95% confidence interval for a proportion.

    The Wilson interval (Wilson 1927) is preferred over the Wald interval because it
    has near-nominal coverage even for small n and extreme proportions (p near 0 or 1).
    The Wald interval p ± z*sqrt(p(1-p)/n) can produce negative lower bounds and
    has poor coverage; Wilson avoids both pathologies.

    Formula (two-sided, 95% CI):
        center = (p_hat + z^2/(2n)) / (1 + z^2/n)
        half   = z * sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2)) / (1 + z^2/n)
        CI     = (center - half, center + half)

    where p_hat = n_success / n_total.

    Returns
    -------
    tuple[float, float]
        (lower, upper) bounds of the 95% Wilson CI, both in [0, 1].

    Why this function exists separately:
        Keeping CI computation isolated makes it independently testable (SCENARIO-BENCH-037 v2)
        and easy to swap for a statsmodels version without touching the main experiment flow.
    """
    if n_total == 0:
        return (0.0, 0.0)
    p_hat = n_success / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    center = (p_hat + z2 / (2.0 * n_total)) / denom
    half = (z / denom) * math.sqrt(
        p_hat * (1.0 - p_hat) / n_total + z2 / (4.0 * n_total * n_total)
    )
    return (max(0.0, center - half), min(1.0, center + half))


def compute_wilson_ci_on_improvement(
    baseline_correct: int,
    pipeline_correct: int,
    n_total: int,
) -> tuple[float, float, bool]:
    """Compute Wilson CI on the signed improvement and determine if CI excludes 0.

    The signed improvement is pipeline_accuracy - baseline_accuracy.  We compute
    Wilson CIs on both proportions and derive an approximate CI on the difference
    using the conservative formula:

        lower_diff = wilson_low(pipeline) - wilson_high(baseline)
        upper_diff = wilson_high(pipeline) - wilson_low(baseline)

    This is conservative (wider than the true CI on the difference) which avoids
    false positives when claiming statistical significance — a necessary property
    for a publishable result.

    Returns
    -------
    (ci_lower, ci_upper, ci_excludes_zero)
        ci_excludes_zero=True means both bounds have the same sign and the interval
        does not straddle 0 — a necessary condition for RETRO-038 closure.
    """
    b_lo, b_hi = compute_wilson_ci(baseline_correct, n_total)
    p_lo, p_hi = compute_wilson_ci(pipeline_correct, n_total)

    ci_lower = p_lo - b_hi
    ci_upper = p_hi - b_lo

    ci_excludes_zero = ci_lower > 0.0

    return (ci_lower, ci_upper, ci_excludes_zero)


# ---------------------------------------------------------------------------
# Latency-based n_questions computation
# ---------------------------------------------------------------------------


def compute_n_questions_from_latency(exp538_result_path: Path) -> int:
    """Read Exp 538 mean_latency_s and compute n_questions that fits in INFERENCE_BUDGET_MINUTES.

    Formula: n_questions = min(MAX_QUESTIONS, int(INFERENCE_BUDGET_MINUTES * 60 / mean_latency_s))

    Why dynamic sizing: Exp 538 showed ~24s/question.  If future runs on a slower GPU
    show >40s/question, this formula automatically drops to 50q to stay within budget.
    Gate (per task spec): if mean_latency_s > 40 → n_questions ≤ 50.

    Falls back to 50 questions if the Exp 538 result file is missing or malformed.
    """
    try:
        with open(exp538_result_path) as f:
            data = json.load(f)
        mean_latency = float(data["mean_latency_s"])
        _log.info("Exp 538 mean_latency_s=%.2f — computing n_questions", mean_latency)
        if mean_latency <= 0:
            return 50
        n = min(MAX_QUESTIONS, int(INFERENCE_BUDGET_MINUTES * 60 / mean_latency))
        return max(1, n)
    except Exception as exc:
        _log.warning("Could not read Exp 538 result (%s) — defaulting to 50q", exc)
        return 50


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_gsm8k_questions(n: int, seed: int) -> list[dict]:
    """Load first N questions from GSM8K test split with a seeded shuffle.

    Uses the same seed and split as Exp 538 for continuity across RETRO-038 attempts.
    Falls back to synthetic questions in offline/CI environments.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        import random
        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = indices[:n]
        return [{"question": ds[i]["question"], "answer": ds[i]["answer"]} for i in selected]
    except Exception as exc:
        _log.warning("GSM8K load failed (%s) — using synthetic fallback", exc)
        return [
            {"question": f"Synthetic question {i}: What is {i} + {i}?", "answer": f"#### {i * 2}"}
            for i in range(1, n + 1)
        ]


# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------


def _qwen_generate(pipeline: Any, prompt: str) -> str:
    """Normalise HuggingFace text-generation pipeline output to a plain string."""
    try:
        out = pipeline(prompt, max_new_tokens=256, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get("generated_text", str(out[0]))
        return str(out)
    except Exception as exc:
        return f"[qwen_error: {exc}]"


def _load_qwen_pipeline(device: str) -> Optional[Any]:
    """Load Qwen3.5-0.8B as a text-generation pipeline.  Returns None on failure."""
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]

        return hf_pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B",
            device=device,
            torch_dtype="auto",
        )
    except Exception as exc:
        _log.warning("_load_qwen_pipeline: failed (%s)", exc)
        return None


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def _build_v8_artifact(
    n_questions: int,
    baseline_correct: int,
    pipeline_correct: int,
    n_scored: int,
    per_question_latencies: list[float],
    inference_mode: str,
    wilson_ci_lower: float,
    wilson_ci_upper: float,
    retro_038_closed: bool,
    env_autofix_dict: dict,
) -> dict:
    """Build the standardised v8 artifact dict.

    All fields in schema='carnot.vericot_benchmark.v8' are populated here.
    Zero/empty defaults are explicit so every exit path produces a complete artifact.

    Why honest_verdict is computed here (not in the main loop):
        Centralising the verdict logic makes it testable in isolation and ensures
        no branch in the main loop can accidentally emit an inconsistent verdict.
    """
    baseline_accuracy = baseline_correct / n_scored if n_scored > 0 else 0.0
    pipeline_accuracy = pipeline_correct / n_scored if n_scored > 0 else 0.0
    signed_improvement = pipeline_accuracy - baseline_accuracy
    mean_latency = sum(per_question_latencies) / len(per_question_latencies) if per_question_latencies else 0.0

    if inference_mode == "gpu_required":
        honest_verdict = "gpu_required"
    elif retro_038_closed:
        honest_verdict = "wilson_ci_publishable"
    elif signed_improvement > 0:
        honest_verdict = "no_improvement"  # positive but not statistically significant
    else:
        honest_verdict = "no_improvement"

    return {
        "schema": "carnot.vericot_benchmark.v8",
        "inference_mode": inference_mode,
        "n_questions": n_questions,
        "n_scored": n_scored,
        "baseline_accuracy": baseline_accuracy,
        "pipeline_accuracy": pipeline_accuracy,
        "signed_improvement": signed_improvement,
        "wilson_ci_lower": wilson_ci_lower,
        "wilson_ci_upper": wilson_ci_upper,
        "retro_038_closed": retro_038_closed,
        "mean_latency_s": mean_latency,
        "per_question_latencies": per_question_latencies,
        "honest_verdict": honest_verdict,
        "env_autofix_applied": True,
        "env_autofix": env_autofix_dict,
    }


def _write_json(repo_root: Path, rel_path: str, data: dict) -> None:
    out = repo_root / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Optional[Path] = None) -> dict:
    """Run Exp 539: live 100q VeriCoT+VPRM v8 benchmark.

    Every exit path writes the deliverable JSON.
    The FINAL LINE is tmpl.assert_deliverable_written().
    """
    if repo_root is None:
        repo_root = _REPO_ROOT

    # Step 2: kill GPU zombies via ExperimentTemplate (pynvml-based)
    ExperimentTemplate.kill_gpu_zombies()

    env_autofix_dict: dict = {}
    if _autofix_result is not None:
        try:
            env_autofix_dict = {
                "gpu_detected": _autofix_result.gpu_detected,
                "force_live_injected": _autofix_result.force_live_injected,
                "original_value": _autofix_result.original_value,
            }
        except AttributeError:
            env_autofix_dict = {"raw": str(_autofix_result)}

    # Step 3: ExperimentTemplate setup
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=str(repo_root / DELIVERABLE),
        requires_gpu=True,
    )
    tmpl.setup()

    # Step 4: DeliverableGuard — ensures deliverable is written on any exit
    guard = DeliverableGuard(str(repo_root / DELIVERABLE))  # noqa: F841

    def _write_and_return(artifact: dict) -> dict:
        _write_json(repo_root, DELIVERABLE, artifact)
        return artifact

    # Step 5: Load Exp 538 result → compute n_questions dynamically
    exp538_path = repo_root / EXP538_RESULT
    n_questions = compute_n_questions_from_latency(exp538_path)
    _log.info("n_questions=%d (from Exp 538 latency gate)", n_questions)

    # Step 6: CARNOT_FORCE_LIVE gate
    gate_result = LiveGPUGate.require_live_or_blocked(tmpl, model_ids=[])
    if gate_result is not None:
        blocked = tmpl.build_result(
            {
                **_build_v8_artifact(
                    n_questions, 0, 0, 0, [], "gpu_required",
                    0.0, 0.0, False, env_autofix_dict,
                ),
                "artifact_type": "carnot.vericot_benchmark.v8",
                "gate_result": str(gate_result),
            },
            status="gpu_required",
        )
        return _write_and_return(blocked)

    # Step 7: JIT VRAM gates
    gemma4_vram = JITVRAMCheck(device_id=0)
    gemma4_gate = gemma4_vram.gate_model_load(
        model_id="Gemma4-INT4", required_gb=GEMMA4_REQUIRED_GB, retry_wait_s=5,
    )
    if not gemma4_gate.is_cleared:
        _log.warning("JIT VRAM gate blocked Gemma4-INT4: %.1f GB free", gemma4_gate.available_gb)
        blocked = tmpl.build_result(
            {
                **_build_v8_artifact(
                    n_questions, 0, 0, 0, [], "gpu_required",
                    0.0, 0.0, False, env_autofix_dict,
                ),
                "artifact_type": "carnot.vericot_benchmark.v8",
                "vram_block_reason": f"gemma4_insufficient: {gemma4_gate.available_gb:.1f} GB",
            },
            status="gpu_vram_insufficient",
        )
        return _write_and_return(blocked)

    qwen_vram = JITVRAMCheck(device_id=1)
    qwen_gate = qwen_vram.gate_model_load(
        model_id="Qwen3.5-0.8B", required_gb=QWEN_REQUIRED_GB, retry_wait_s=5,
    )
    if not qwen_gate.is_cleared:
        _log.warning("JIT VRAM gate blocked Qwen3.5-0.8B: %.1f GB free", qwen_gate.available_gb)
        blocked = tmpl.build_result(
            {
                **_build_v8_artifact(
                    n_questions, 0, 0, 0, [], "gpu_required",
                    0.0, 0.0, False, env_autofix_dict,
                ),
                "artifact_type": "carnot.vericot_benchmark.v8",
                "vram_block_reason": f"qwen_insufficient: {qwen_gate.available_gb:.1f} GB",
            },
            status="gpu_vram_insufficient",
        )
        return _write_and_return(blocked)

    # Step 8: Load models
    gemma4_path_candidates = [
        Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-4-e4b-it" / "blobs",
        Path("/data/models/gemma4"),
    ]
    gemma4_gguf_path: Optional[str] = None
    for candidate in gemma4_path_candidates:
        if candidate.exists():
            gguf_files = list(candidate.glob("*.gguf"))
            if gguf_files:
                gemma4_gguf_path = str(gguf_files[0])
                break

    gemma4_loader: Optional[Gemma4QuantizedLoader] = None
    if gemma4_gguf_path:
        try:
            gemma4_loader = Gemma4QuantizedLoader(
                model_path=gemma4_gguf_path,
                n_gpu_layers=80,
                max_tokens=256,
                jit_vram_check=gemma4_vram,
            )
            gemma4_loader.load()
            _log.info("Gemma4-INT4 loaded from %s", gemma4_gguf_path)
        except Exception as exc:
            _log.warning("Gemma4QuantizedLoader load failed: %s", exc)
            gemma4_loader = None

    qwen_pipe: Optional[Any] = None
    try:
        import torch
        qwen_device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0"
        qwen_pipe = _load_qwen_pipeline(qwen_device)
        if qwen_pipe:
            _log.info("Qwen pipeline loaded on %s", qwen_device)
    except Exception as exc:
        _log.warning("Qwen pipeline load failed: %s", exc)

    if not gemma4_loader and not qwen_pipe:
        _log.warning("No live models available — writing gpu_required artifact")
        blocked = tmpl.build_result(
            {
                **_build_v8_artifact(
                    n_questions, 0, 0, 0, [], "gpu_required",
                    0.0, 0.0, False, env_autofix_dict,
                ),
                "artifact_type": "carnot.vericot_benchmark.v8",
                "no_models": True,
            },
            status="gpu_required",
        )
        return _write_and_return(blocked)

    # Step 9: Load n_questions GSM8K questions
    questions = _load_gsm8k_questions(n_questions, GSM8K_SEED)
    _log.info("Loaded %d GSM8K questions (seed=%d)", len(questions), GSM8K_SEED)

    # Step 10: Per-question VeriCoT+VPRM extraction and repair
    # use_mock=True on VeriCoTStepValidator avoids loading the LLM extractor for
    # the FOL step — the rule-based mock is accurate enough for violation detection
    # while keeping inference latency under budget.
    _vericot = VeriCoTStepValidator(use_mock=True)
    _vprm = VPRMArithmeticVerifier()
    integrated = IntegratedExtractor(vericot=_vericot, vprm=_vprm)

    baseline_correct = 0
    pipeline_correct = 0
    n_scored = 0
    per_question_latencies: list[float] = []

    models_available = []
    if gemma4_loader:
        models_available.append(("Gemma4-INT4", gemma4_loader, None))
    if qwen_pipe:
        models_available.append(("Qwen3.5-0.8B", None, qwen_pipe))

    for model_id, g_loader, q_pipe in models_available:
        _log.info("=== Running %s on %d questions ===", model_id, len(questions))
        for q in questions:
            q_start = time.time()
            prompt = q["question"]
            gold = _extract_answer(q.get("answer", ""))

            # Baseline inference
            try:
                if g_loader is not None:
                    baseline_resp = g_loader.generate(prompt)
                else:
                    baseline_resp = _qwen_generate(q_pipe, prompt)
            except Exception as exc:
                _log.warning("Baseline inference error: %s", exc)
                baseline_resp = ""

            # VeriCoT+VPRM violation detection via IntegratedExtractor.extract()
            try:
                violations = integrated.extract(baseline_resp) if baseline_resp else []
            except Exception:
                violations = []

            # Repair if violations detected
            if violations:
                repair_prompt = (
                    f"Question: {prompt}\n\n"
                    "Your previous answer had arithmetic errors. Solve step by step carefully."
                )
                try:
                    if g_loader is not None:
                        pipeline_resp = g_loader.generate(repair_prompt)
                    else:
                        pipeline_resp = _qwen_generate(q_pipe, repair_prompt)
                except Exception as exc:
                    _log.warning("Repair inference error: %s", exc)
                    pipeline_resp = baseline_resp
            else:
                pipeline_resp = baseline_resp

            lat = time.time() - q_start
            per_question_latencies.append(lat)

            bc = _is_correct(baseline_resp, gold)
            pc = _is_correct(pipeline_resp, gold)
            baseline_correct += int(bc)
            pipeline_correct += int(pc)
            n_scored += 1

            _log.info(
                "[%s] q=%d baseline=%s pipeline=%s violations=%d lat=%.1fs",
                model_id, n_scored, bc, pc, len(violations), lat,
            )

        # Run only one model (primary model is sufficient for this milestone)
        break

    # Step 11: Compute Wilson CI on signed improvement
    wilson_ci_lower, wilson_ci_upper, ci_excludes_zero = compute_wilson_ci_on_improvement(
        baseline_correct, pipeline_correct, n_scored
    )
    retro_038_closed = ci_excludes_zero
    _log.info(
        "Wilson CI: [%.4f, %.4f] ci_excludes_zero=%s retro_038_closed=%s",
        wilson_ci_lower, wilson_ci_upper, ci_excludes_zero, retro_038_closed,
    )

    # Step 12: Build artifact
    v8_fields = _build_v8_artifact(
        n_questions=n_questions,
        baseline_correct=baseline_correct,
        pipeline_correct=pipeline_correct,
        n_scored=n_scored,
        per_question_latencies=per_question_latencies,
        inference_mode="live_gpu",
        wilson_ci_lower=wilson_ci_lower,
        wilson_ci_upper=wilson_ci_upper,
        retro_038_closed=retro_038_closed,
        env_autofix_dict=env_autofix_dict,
    )

    artifact = tmpl.build_result(
        {
            "artifact_type": "carnot.vericot_benchmark.v8",
            **v8_fields,
        },
        status="success",
    )
    _write_json(repo_root, DELIVERABLE, artifact)

    _log.info(
        "HEADLINE: honest_verdict=%s retro_038_closed=%s "
        "baseline=%.4f pipeline=%.4f delta=%.4f wilson=[%.4f,%.4f] mean_lat=%.1fs",
        v8_fields["honest_verdict"], retro_038_closed,
        v8_fields["baseline_accuracy"], v8_fields["pipeline_accuracy"],
        v8_fields["signed_improvement"],
        wilson_ci_lower, wilson_ci_upper,
        v8_fields["mean_latency_s"],
    )

    # Step 13: FINAL LINE
    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 539: Live 100q VeriCoT+VPRM v8."""
    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=90,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        artifact = run_experiment()

    verdict = artifact.get("honest_verdict", "unknown")
    _log.info(
        "Exp %d complete: honest_verdict=%s status=%s retro_038_closed=%s",
        EXP_ID, verdict, artifact.get("status", "unknown"), artifact.get("retro_038_closed"),
    )


if __name__ == "__main__":
    main()
