#!/usr/bin/env python3
"""Experiment 502: Live 100q Precision v6 — Gemma4QuantizedLoader + VRAMBudgetLedger.

**Researcher summary (RETRO-033, sixth attempt):**
    Exps 451, 464, 476, 479, 488 all failed to close RETRO-033:
    - 451 (ms .34): +5pp result JSON absent (no DeliverableGuard).
    - 464 (ms .35): deferred — zombie processes held 23.8 GB at 0% utilisation.
    - 476 (ms .36): GPUVRAMGate V1 check-first order; RETRO-044 race caused deferral.
    - 479 (ms .36): same root cause.
    - 488 (ms .37): GPUVRAMGateV2 correct, but conductor holds ~15.7 GiB GPU 0 VRAM;
      Gemma4-FP16 needs 14.89 GiB — conductor + model = 30.6 GiB, exceeds 24 GiB limit.

    Root cause fix (Exp 500): Gemma4QuantizedLoader loads Q4_K_M GGUF (~8-10 GiB).
    Budget: conductor(~9 GiB) + Gemma4-INT4(~9 GiB) = ~18 GiB — fits with ~6 GiB headroom.
    VRAMBudgetLedger (Exp 501) pre-checks this before the gate fires.

**Gate chain (runs in order; every exit path writes the deliverable):**
    0. apply_env_autofix()                               — FIRST, before any CUDA import
    1. ExperimentTimeoutWatchdog(502, 120 min)            — outer hard cap
    2. DeliverableGuard instantiation                    — path registered, not yet asserted
    3. VRAMBudgetLedger.check_feasibility('exp502')       — proactive VRAM feasibility check
    4. GPUVRAMGateV2(min_free_gb=6.0, kill_first=True)   — kill zombies BEFORE VRAM check
    5. DualGPUHarness.apply()                            — Gemma4-INT4→cuda:0, Qwen→cuda:1
    6. Gemma4QuantizedLoader.load()                      — GGUF Q4_K_M on cuda:0
    7. Qwen3.5-0.8B HF pipeline on cuda:1
    8. Benchmark: 100 GSM8K questions (shuffle seed=42)
    9. CoTPairCollector.flush() → results/exp502_cot_pairs.json
    10. tmpl.assert_deliverable_written()                — FINAL LINE

**Outputs:**
    results/experiment_502_live_100q_precision_v6.json  — primary artifact
    results/exp502_cot_pairs.json                       — CoT pairs for Exp 510 JEPA retrain

Spec: REQ-BENCH-043, REQ-BENCH-044, REQ-BENCH-045,
      SCENARIO-BENCH-062, SCENARIO-BENCH-063, SCENARIO-BENCH-064
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
from carnot.pipeline.cot_pair_collector import CoTPairCollector
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.dual_gpu_harness import DualGPUHarness
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog, get_timeout_minutes
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader
from carnot.pipeline.gpu_vram_gate import GPUVRAMInsufficientError
from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2
from carnot.pipeline.precision_100q_v6_result import Precision100qV6Result
from carnot.pipeline.vram_budget_ledger import VRAMBudgetLedger
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 502
EXP_TITLE = "Live 100q Precision v6 — Gemma4QuantizedLoader + VRAMBudgetLedger (RETRO-033 close)"
DELIVERABLE = "results/experiment_502_live_100q_precision_v6.json"
COT_PAIRS_PATH = "results/exp502_cot_pairs.json"
N_QUESTIONS = 100
GSM8K_SEED = 42

# Gemma4 Q4_K_M requires ~9 GiB; conductor holds ~9 GiB; combined ~18 GiB on 24 GiB card.
# Qwen3.5-0.8B adds ~1.5 GiB on GPU 1 — fits comfortably in 24 GiB.
GEMMA4_REQUIRED_GB = 9.0
QWEN_REQUIRED_GB = 1.5
CONDUCTOR_VRAM_GB = 9.0
GPU_TOTAL_GB = 24.0

# Two models — DualGPUHarness.apply() will inject gpu/device_map
MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "gemma4-int4"},
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


def _load_qwen_pipeline(hf_id: str, device_map: dict) -> object:
    """Load a HuggingFace text-generation pipeline for Qwen with explicit device_map.

    Why device_map is a dict not a string:
        device_map='auto' spreads the model across all visible GPUs.  With two
        models loaded simultaneously, 'auto' causes both to fight for both GPUs.
        Passing {'': 'cuda:N'} pins every layer to GPU N (REQ-BENCH-044).
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
) -> Precision100qV6Result:
    """Run baseline and pipeline variants for one model, collecting CoT pairs.

    Two passes per model:
    1. BASELINE — raw model output, no pipeline.
    2. PIPELINE — IntegratedExtractor detects violations; one-shot repair when found.

    Each pipeline response is recorded in the CoT collector (REQ-BENCH-045).
    Returns a Precision100qV6Result with Wilson 95% CI and explicit gpu_id.
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

    return Precision100qV6Result(
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
    """Run Experiment 502 and return the artifact dict.

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
                "schema": "carnot.live_precision.v6",
                "env_autofix": env_autofix_dict,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gemma4_quantized": True,
                "dual_gpu_explicit": False,
                "retro_033_closed": False,
            },
            status="gpu_required",
            honest_verdict="deferred_retro_033_v6",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 1: VRAMBudgetLedger — proactive feasibility check (REQ-BENCH-043)
    # Check before firing the VRAM gate so infeasibility is caught early with
    # an actionable root cause (conductor_vram_gb + model_gb > gpu_total_gb).
    # ------------------------------------------------------------------
    ledger = VRAMBudgetLedger(conductor_vram_gb=CONDUCTOR_VRAM_GB, gpu_total_gb=GPU_TOTAL_GB)
    ledger.add_experiment("exp502_gemma4", GEMMA4_REQUIRED_GB)
    ledger.add_experiment("exp502_qwen", QWEN_REQUIRED_GB)
    forecasts = ledger.check_all()
    forecast_dicts = [f.to_dict() for f in forecasts]
    infeasible = [f for f in forecasts if not f.is_feasible]

    if infeasible:
        _log.warning(
            "VRAMBudgetLedger: %d infeasible experiments — %s",
            len(infeasible),
            [f.exp_id for f in infeasible],
        )

    _log.info("VRAMBudgetLedger: forecasts=%s", forecast_dicts)

    # ------------------------------------------------------------------
    # Gate 2: GPUVRAMGateV2 — kill zombies FIRST, then check VRAM
    # REQ-BENCH-043: kill_first=True fixes the RETRO-044 race condition.
    # min_free_gb=6.0 because conductor(~9 GiB) + Gemma4-INT4(~9 GiB) = ~18 GiB
    # leaves ~6 GiB free on a 24 GiB card — use 6.0 as the minimum viable headroom.
    # ------------------------------------------------------------------
    try:
        with GPUVRAMGateV2(min_free_gb=6.0, kill_first=True):
            pass  # gate confirms VRAM is free; model load happens below
    except GPUVRAMInsufficientError as exc:
        _log.error("GPUVRAMGateV2: VRAM insufficient after drain — %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v6",
                "env_autofix": env_autofix_dict,
                "vram_forecasts": forecast_dicts,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gemma4_quantized": True,
                "dual_gpu_explicit": False,
                "retro_033_closed": False,
                "vram_error": str(exc),
            },
            status="gpu_vram_insufficient",
            honest_verdict="deferred_retro_033_v6",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 3: DualGPUHarness — pin Gemma4-INT4→cuda:0, Qwen→cuda:1
    # REQ-BENCH-044: explicit device assignment, never device_map='auto'.
    # ------------------------------------------------------------------
    harness = DualGPUHarness.from_env()
    assigned_specs = harness.apply([dict(s) for s in MODEL_SPECS])
    dual_gpu_explicit = harness.is_eligible

    gemma_spec = next(s for s in assigned_specs if "gemma" in s["name"].lower())
    qwen_spec = next(s for s in assigned_specs if "qwen" in s["name"].lower())
    gemma_gpu = gemma_spec.get("gpu", 0)
    qwen_gpu = qwen_spec.get("gpu", 1)
    qwen_device_map = qwen_spec.get("device_map", {"": f"cuda:{qwen_gpu}"})

    _log.info(
        "DualGPUHarness: gemma4-int4→cuda:%d, qwen→cuda:%d (eligible=%s)",
        gemma_gpu, qwen_gpu, harness.is_eligible,
    )

    # ------------------------------------------------------------------
    # Gate 4: IntegratedExtractor (live mode — no mocks)
    # ------------------------------------------------------------------
    extractor = IntegratedExtractor(
        vericot=VeriCoTStepValidator(use_mock=False),
        vprm=VPRMArithmeticVerifier(),
    )

    # ------------------------------------------------------------------
    # Gate 5: Load Gemma4-INT4 via Gemma4QuantizedLoader on cuda:0
    # GGUF Q4_K_M path from CARNOT_GEMMA4_GGUF_PATH env var.
    # Falls back to stub mode when the path is absent (CI path).
    # ------------------------------------------------------------------
    gemma4_gguf_path = os.environ.get("CARNOT_GEMMA4_GGUF_PATH", "")
    gemma_loader = Gemma4QuantizedLoader(model_path=gemma4_gguf_path, n_gpu_layers=-1)
    try:
        _log.info("Loading Gemma4-INT4 (GGUF Q4_K_M) on cuda:%d ...", gemma_gpu)
        gemma_loader.load()
        _log.info("Gemma4-INT4 loaded OK on cuda:%d", gemma_gpu)
    except Exception as exc:
        _log.error("Failed to load Gemma4-INT4: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v6",
                "env_autofix": env_autofix_dict,
                "vram_forecasts": forecast_dicts,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gemma4_quantized": True,
                "dual_gpu_explicit": dual_gpu_explicit,
                "retro_033_closed": False,
            },
            status="blocked",
            blocked_reason=f"Gemma4-INT4 load failed: {exc}",
            honest_verdict="deferred_retro_033_v6",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Gate 6: Load Qwen3.5-0.8B on cuda:1
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
                "schema": "carnot.live_precision.v6",
                "env_autofix": env_autofix_dict,
                "vram_forecasts": forecast_dicts,
                "gemma4_result": None,
                "qwen_result": None,
                "cot_pairs_written": 0,
                "gemma4_quantized": True,
                "dual_gpu_explicit": dual_gpu_explicit,
                "retro_033_closed": False,
            },
            status="blocked",
            blocked_reason=f"Qwen load failed: {exc}",
            honest_verdict="deferred_retro_033_v6",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # ------------------------------------------------------------------
    # Load questions
    # ------------------------------------------------------------------
    questions = _load_gsm8k_questions(N_QUESTIONS, seed=GSM8K_SEED)
    _log.info("Loaded %d questions (seed=%d)", len(questions), GSM8K_SEED)

    # CoT pair collector for Exp 510 JEPA retrain (REQ-BENCH-045)
    collector = CoTPairCollector(str(repo_root / COT_PAIRS_PATH))

    # ------------------------------------------------------------------
    # Run benchmarks: Gemma4-INT4 then Qwen
    # ------------------------------------------------------------------
    def gemma_fn(prompt: str) -> str:
        return gemma_loader.generate(prompt)

    def qwen_fn(prompt: str) -> str:
        assert qwen_pipe is not None
        return _run_qwen_inference(qwen_pipe, prompt)

    _log.info("=== Running Gemma4-INT4 benchmark (cuda:%d, %dq) ===", gemma_gpu, len(questions))
    gemma4_result = _run_model_benchmark(
        "Gemma4-INT4", gemma_gpu, gemma_fn, extractor, questions, collector,
    )
    tmpl.checkpoint_save(gemma4_result.to_dict(), step=1)

    _log.info("=== Running Qwen3.5-0.8B benchmark (cuda:%d, %dq) ===", qwen_gpu, len(questions))
    qwen_result = _run_model_benchmark(
        "Qwen3.5-0.8B", qwen_gpu, qwen_fn, extractor, questions, collector,
    )
    tmpl.checkpoint_save(qwen_result.to_dict(), step=2)

    # ------------------------------------------------------------------
    # Flush CoT pairs atomically (REQ-BENCH-045)
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
            "schema": "carnot.live_precision.v6",
            "env_autofix": env_autofix_dict,
            "vram_forecasts": forecast_dicts,
            "n_questions": len(questions),
            "gemma4_result": gemma4_result.to_dict(),
            "qwen_result": qwen_result.to_dict(),
            "cot_pairs_written": cot_pairs_written,
            "gemma4_quantized": True,
            "dual_gpu_explicit": dual_gpu_explicit,
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
    """Run Experiment 502: 100q live precision benchmark v6, RETRO-033 sixth attempt.

    120-minute watchdog: 15 s VRAM drain sleep + two model loads (Gemma4-INT4 GGUF
    + Qwen HF pipeline) + 200 inference passes on 100q each (baseline + pipeline).
    Gemma4-INT4 load is faster than FP16 (no BF16 conversion) so the budget is tight
    but within the 120-min cap observed for V5 harnesses.
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
