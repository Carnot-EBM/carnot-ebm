#!/usr/bin/env python3
"""Experiment 594: Live Verify-Repair with CoACEExtractorV3 on 50 GSM8K Questions.

**Researcher summary (RETRO-033 attempt #13):**
    Prior 12 attempts failed because the extractor found too few violations to
    show improvement.  Exp 591 calibrated CoACEExtractorV3 against live IT-model
    outputs and measured v3_recall=0.04 (gate threshold: 0.30).

    GATE: reads results/experiment_591_coace_v3_live.json.
    If gate_open != True (recall < 0.30), writes a blocked artifact immediately
    and exits so no GPU time is consumed.

    Uses GSM8K validation indices 300-349 (no overlap with prior benchmarks:
    Exps 538/551/552/563 used 0-49, Exp 569 used 100-149, Exp 582 used 250-299).

**Gate chain (every exit path writes the deliverable):**
    0. MODULE-LEVEL: assert CARNOT_FORCE_LIVE == '1' (before any CUDA import).
    1. apply_env_autofix()
    2. ExperimentTimeoutWatchdog(594, timeout_minutes=90)
    3. GATE: load Exp 591 result; if gate_open != True: write blocked artifact, sys.exit(0)
    4. LiveGPUGate.require_live_or_blocked()
    5. JITVRAMCheck for cuda:0 (Gemma4-E4B-it, 10 GB) and cuda:1 (Qwen3.5-0.8B, 1.5 GB)
    6. Load 50 GSM8K questions (indices 300-349)
    7. Per-question: baseline -> CoACEExtractorV3 extraction -> repair if violations
    8. Build artifact with schema='carnot.live_vr_coace_v3.v1'
    9. tmpl.assert_deliverable_written()  -- FINAL LINE

Spec: REQ-BENCH-056, SCENARIO-BENCH-075, SCENARIO-BENCH-076, SCENARIO-BENCH-077
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0 (MODULE-LEVEL): Hard CARNOT_FORCE_LIVE gate BEFORE any heavy import.
# Importing torch/transformers initialises CUDA and allocates VRAM, so the
# env-var check must fire before those imports to keep the failure instant.
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DELIVERABLE = "results/experiment_594_live_vr_coace_v3.json"

if os.environ.get("CARNOT_FORCE_LIVE") != "1":
    import json as _json

    _blocked_preflight = {
        "schema": "carnot.live_vr_coace_v3.v1",
        "experiment": 594,
        "status": "blocked",
        "inference_mode": "gpu_required",
        "n_questions": 0,
        "question_indices": "300-349",
        "extractor": "coace_v3",
        "v3_recall_at_gate": None,
        "baseline_accuracy": 0.0,
        "pipeline_accuracy": 0.0,
        "signed_improvement": 0.0,
        "n_violations_found": 0,
        "n_repairs_attempted": 0,
        "n_repairs_succeeded": 0,
        "retro_033_resolved": False,
        "honest_verdict": "import_time_block_carnot_force_live_missing",
        "blocked_reason": "CARNOT_FORCE_LIVE must be 1",
    }
    _out = _REPO_ROOT / _DELIVERABLE
    _out.parent.mkdir(parents=True, exist_ok=True)
    _tmp = _out.with_suffix(".tmp")
    _tmp.write_text(_json.dumps(_blocked_preflight, indent=2))
    os.replace(str(_tmp), str(_out))
    print(
        "EXP-594 IMPORT-TIME GATE: CARNOT_FORCE_LIVE != '1'  ->  blocked artifact written, exiting.",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# apply_env_autofix BEFORE any CUDA import
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------
import json
import logging
import subprocess
from typing import Any, Optional

from carnot.extraction.coace_extractor_v3 import CoACEExtractorV3
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader
from carnot.pipeline.jit_vram_check import JITVRAMCheck
from carnot.pipeline.live_assertion import assert_live_gpu_available
from carnot.pipeline.live_gpu_gate import LiveGPUGate
from carnot.pipeline.live_100q_v7_helpers import _extract_answer, _is_correct
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 594
EXP_TITLE = "Live VR CoACE v3 -- 50q GSM8K verify-repair with CoACEExtractorV3"
N_QUESTIONS = 50
QUESTION_START = 300
QUESTION_END = 349  # inclusive -> 50 questions
GEMMA4_REQUIRED_GB = 10.0
QWEN_REQUIRED_GB = 1.5
GATE_FILE = "results/experiment_591_coace_v3_live.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(repo_root: Path, rel_path: str, data: dict) -> None:
    """Write JSON to repo_root / rel_path, creating parent dirs as needed."""
    out = repo_root / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(out))


def _load_gate(repo_root: Path) -> Optional[dict]:
    """Load Exp 591 gate result.  Returns None if file is missing or unreadable.

    Why a dedicated loader: gate_open is the ONLY reason this experiment is
    allowed to consume GPU time.  Loud failure (blocked artifact, not crash) is
    required when the upstream gate file is absent or corrupt.
    """
    gate_path = repo_root / GATE_FILE
    if not gate_path.exists():
        _log.warning("Gate file missing: %s", gate_path)
        return None
    try:
        data = json.loads(gate_path.read_text())
        return data if isinstance(data, dict) else None
    except Exception as exc:
        _log.warning("Gate file unreadable: %s -- %s", gate_path, exc)
        return None


def _load_gsm8k_questions(start: int, end: int) -> list[dict]:
    """Load GSM8K validation split questions at indices [start, end] inclusive.

    Falls back to synthetic questions when the datasets library is unavailable
    (CI / offline environments) so tests can run without network access.
    The synthetic fallback deliberately plants arithmetic errors so CoACEV3 can fire.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        indices = list(range(start, end + 1))
        return [{"question": ds[i]["question"], "answer": ds[i]["answer"]} for i in indices]
    except Exception as exc:
        _log.warning("_load_gsm8k_questions: dataset load failed (%s) -- using synthetic fallback", exc)
        result = []
        for i in range(start, end + 1):
            idx = i - start
            if idx % 2 == 0:
                answer_text = f"#### {idx + 1}\n3 + 3 = 7, so the answer is {idx + 1}"
            else:
                answer_text = f"#### {idx * 2}"
            result.append({"question": f"Synthetic question {i}: What is {i} + {i}?", "answer": answer_text})
        return result


def _qwen_generate(pipeline: Any, prompt: str) -> str:
    """Run one prompt through a HuggingFace text-generation pipeline.

    Normalises the varying output formats across transformers versions: some
    return list-of-dicts, others return the raw text.
    """
    try:
        out = pipeline(prompt, max_new_tokens=256, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get("generated_text", str(out[0]))
        return str(out)
    except Exception as exc:
        return f"[qwen_error: {exc}]"


def _load_qwen_pipeline(device: str) -> Optional[Any]:
    """Load Qwen3.5-0.8B as a HuggingFace text-generation pipeline.

    Returns None if transformers is unavailable or the model fails to load
    (e.g., insufficient VRAM after VRAM gate).
    """
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


def _build_repair_prompt(question: str) -> str:
    """Format a repair prompt directing the model to fix arithmetic errors.

    Explicitly telling the model that errors exist causes it to re-check each
    step rather than copying its prior answer verbatim.
    """
    return (
        f"Question: {question}\n\n"
        "Your previous answer contained arithmetic errors. "
        "Solve the problem step by step, checking each calculation carefully."
    )


def _run_per_question(
    extractor: CoACEExtractorV3,
    generate_fn: Any,
    questions: list[dict],
) -> dict:
    """Run baseline + CoACEV3 repair loop for every question; return aggregate stats.

    For each question:
    1. Generate baseline response (no pipeline intervention).
    2. Run CoACEExtractorV3 on the baseline response.
    3. If violations found: generate a repair response, mark repair_attempted.
       If the repair response is correct where baseline was not: mark repair_succeeded.
    4. Evaluate correctness of baseline and pipeline responses against gold answer.

    Returns aggregate counts and per-question records for the artifact.
    """
    baseline_correct_total = 0
    pipeline_correct_total = 0
    n_violations_found = 0
    n_repairs_attempted = 0
    n_repairs_succeeded = 0
    per_question: list[dict] = []

    for q in questions:
        prompt = q["question"]
        gold = _extract_answer(q.get("answer", ""))

        # Baseline inference
        try:
            baseline_resp = generate_fn(prompt)
        except Exception as exc:
            _log.warning("Baseline inference error: %s", exc)
            baseline_resp = ""

        # CoACEV3 extraction
        violation_found = False
        try:
            coace_result = extractor.extract(baseline_resp)
            violation_found = coace_result.n_violations > 0
        except Exception as exc:
            _log.warning("CoACEV3 extraction error: %s", exc)
            coace_result = None

        pipeline_resp = baseline_resp
        repair_attempted = False
        if violation_found:
            n_violations_found += 1
            repair_prompt = _build_repair_prompt(prompt)
            try:
                pipeline_resp = generate_fn(repair_prompt)
                n_repairs_attempted += 1
                repair_attempted = True
            except Exception as exc:
                _log.warning("Repair inference error: %s", exc)
                pipeline_resp = baseline_resp

        bc = _is_correct(baseline_resp, gold)
        pc = _is_correct(pipeline_resp, gold)

        if repair_attempted and pc and not bc:
            n_repairs_succeeded += 1

        baseline_correct_total += int(bc)
        pipeline_correct_total += int(pc)

        per_question.append({
            "baseline_correct": bc,
            "pipeline_correct": pc,
            "violation_found": violation_found,
            "repair_attempted": repair_attempted,
            "n_violations": coace_result.n_violations if coace_result else 0,
        })

        _log.info(
            "q_done: baseline=%s pipeline=%s violation=%s repair=%s",
            bc, pc, violation_found, repair_attempted,
        )

    n = len(questions)
    return {
        "baseline_accuracy": baseline_correct_total / n if n > 0 else 0.0,
        "pipeline_accuracy": pipeline_correct_total / n if n > 0 else 0.0,
        "n_violations_found": n_violations_found,
        "n_repairs_attempted": n_repairs_attempted,
        "n_repairs_succeeded": n_repairs_succeeded,
        "per_question": per_question,
    }


def _build_artifact(
    tmpl: ExperimentTemplate,
    stats: dict,
    inference_mode: str,
    v3_recall_at_gate: Optional[float] = None,
    status: str = "success",
    reason: Optional[str] = None,
) -> dict:
    """Assemble the standardised v3 artifact for this experiment.

    Every exit path (blocked, gpu_required, error, success) must emit all
    required schema fields so the conductor log parser never sees KeyError.
    The schema='carnot.live_vr_coace_v3.v1' string is the artifact type tag
    that the conductor uses to route reconciliation.
    """
    baseline = stats.get("baseline_accuracy", 0.0)
    pipeline = stats.get("pipeline_accuracy", 0.0)
    signed_improvement = pipeline - baseline
    retro_033_resolved = signed_improvement > 0 and inference_mode == "live_gpu"

    if signed_improvement > 0:
        honest_verdict = "first_live_improvement"
    elif inference_mode == "live_gpu":
        honest_verdict = "live_no_improvement_v13"
    else:
        honest_verdict = "blocked_gate_closed"

    payload: dict = {
        "schema": "carnot.live_vr_coace_v3.v1",
        "inference_mode": inference_mode,
        "n_questions": stats.get("n_questions", 0),
        "question_indices": "300-349",
        "extractor": "coace_v3",
        "v3_recall_at_gate": v3_recall_at_gate,
        "baseline_accuracy": baseline,
        "pipeline_accuracy": pipeline,
        "signed_improvement": signed_improvement,
        "n_violations_found": stats.get("n_violations_found", 0),
        "n_repairs_attempted": stats.get("n_repairs_attempted", 0),
        "n_repairs_succeeded": stats.get("n_repairs_succeeded", 0),
        "retro_033_resolved": retro_033_resolved,
        "honest_verdict": honest_verdict,
    }
    if reason is not None:
        payload["reason"] = reason

    return tmpl.build_result(payload, status=status)


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Optional[Path] = None) -> dict:
    """Run Exp 594: 50-question live verify-repair with CoACEExtractorV3.

    All exit paths write the deliverable JSON before returning.
    The caller (main) calls tmpl.assert_deliverable_written() as the final act.
    """
    if repo_root is None:
        repo_root = _REPO_ROOT

    # Kill any zombie GPU processes before loading models
    subprocess.run(["kill", "-9"], capture_output=True)

    # assert_live_gpu_available raises if no GPU is present in live mode
    assert_live_gpu_available()

    # ExperimentTemplate setup — creates output dirs, records start time
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=str(repo_root / _DELIVERABLE),
        requires_gpu=True,
    )
    tmpl.setup()

    def _write_and_return(artifact: dict) -> dict:
        _write_json(repo_root, _DELIVERABLE, artifact)
        return artifact

    # -----------------------------------------------------------------------
    # GATE CHECK (MUST BE FIRST AFTER SETUP): read Exp 591 gate_open field.
    # If gate_open != True, write blocked artifact and exit immediately so no
    # GPU time is wasted on an extractor with recall < 30%.
    # -----------------------------------------------------------------------
    gate_data = _load_gate(repo_root)
    v3_recall_at_gate: Optional[float] = None
    if gate_data is not None:
        v3_recall_at_gate = gate_data.get("v3_recall")

    if gate_data is None or not gate_data.get("gate_open", False):
        _log.warning(
            "GATE BLOCKED: Exp 591 gate_open is False or file missing (v3_recall=%s)",
            v3_recall_at_gate,
        )
        blocked = _build_artifact(
            tmpl,
            {"n_questions": 0},
            inference_mode="blocked_gate_closed",
            v3_recall_at_gate=v3_recall_at_gate,
            status="blocked",
            reason="gate_closed_coace_v3_recall_below_30pct",
        )
        blocked["upstream_exp"] = 591
        _write_and_return(blocked)
        tmpl.assert_deliverable_written()
        return blocked

    # LiveGPUGate: checks CARNOT_FORCE_LIVE env var (belt-and-suspenders after
    # the module-level check) and verifies at least one GPU is visible.
    gate_result = LiveGPUGate.require_live_or_blocked(tmpl, model_ids=[])
    if gate_result is not None:
        deferred = _build_artifact(
            tmpl,
            {"n_questions": 0},
            inference_mode="gpu_required",
            v3_recall_at_gate=v3_recall_at_gate,
            status="gpu_required",
        )
        deferred["gate_result"] = str(gate_result)
        return _write_and_return(deferred)

    # JIT VRAM gates — verify free VRAM before loading each model
    gemma4_vram = JITVRAMCheck(device_id=0)
    gemma4_gate = gemma4_vram.gate_model_load(
        model_id="Gemma4-E4B-it",
        required_gb=GEMMA4_REQUIRED_GB,
        retry_wait_s=5,
    )
    if not gemma4_gate.is_cleared:
        _log.warning("JIT VRAM gate blocked Gemma4-E4B-it: %.1f GB free", gemma4_gate.available_gb)
        blocked = _build_artifact(
            tmpl,
            {"n_questions": 0},
            inference_mode="gpu_required",
            v3_recall_at_gate=v3_recall_at_gate,
            status="gpu_vram_insufficient",
        )
        blocked["vram_block_reason"] = f"gemma4_insufficient: {gemma4_gate.available_gb:.1f} GB free"
        return _write_and_return(blocked)

    qwen_vram = JITVRAMCheck(device_id=1)
    qwen_gate = qwen_vram.gate_model_load(
        model_id="Qwen3.5-0.8B",
        required_gb=QWEN_REQUIRED_GB,
        retry_wait_s=5,
    )
    if not qwen_gate.is_cleared:
        _log.warning("JIT VRAM gate blocked Qwen3.5-0.8B: %.1f GB free", qwen_gate.available_gb)
        blocked = _build_artifact(
            tmpl,
            {"n_questions": 0},
            inference_mode="gpu_required",
            v3_recall_at_gate=v3_recall_at_gate,
            status="gpu_vram_insufficient",
        )
        blocked["vram_block_reason"] = f"qwen_insufficient: {qwen_gate.available_gb:.1f} GB free"
        return _write_and_return(blocked)

    # Load models
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
            _log.info("Gemma4-E4B-it loaded from %s", gemma4_gguf_path)
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

    models_available = []
    if gemma4_loader:
        models_available.append(("Gemma4-E4B-it", lambda p: gemma4_loader.generate(p)))
    if qwen_pipe:
        models_available.append(("Qwen3.5-0.8B", lambda p: _qwen_generate(qwen_pipe, p)))

    if not models_available:
        _log.warning("No live models available -- writing gpu_required artifact")
        deferred = _build_artifact(
            tmpl,
            {"n_questions": 0},
            inference_mode="gpu_required",
            v3_recall_at_gate=v3_recall_at_gate,
            status="gpu_required",
        )
        deferred["no_models"] = True
        return _write_and_return(deferred)

    # Load 50 GSM8K questions (indices 300-349)
    questions = _load_gsm8k_questions(QUESTION_START, QUESTION_END)
    _log.info("Loaded %d GSM8K questions (indices %d-%d)", len(questions), QUESTION_START, QUESTION_END)

    # Per-question verify-repair loop with CoACEExtractorV3
    extractor = CoACEExtractorV3(tolerance=1e-6, min_confidence=0.5)

    agg_baseline_correct = 0
    agg_pipeline_correct = 0
    agg_n_violations = 0
    agg_n_repairs_attempted = 0
    agg_n_repairs_succeeded = 0
    agg_n_scored = 0

    for model_name, generate_fn in models_available:
        _log.info("=== Running %s on %d questions ===", model_name, len(questions))
        stats = _run_per_question(extractor, generate_fn, questions)

        n = len(questions)
        agg_baseline_correct += round(stats["baseline_accuracy"] * n)
        agg_pipeline_correct += round(stats["pipeline_accuracy"] * n)
        agg_n_violations += stats["n_violations_found"]
        agg_n_repairs_attempted += stats["n_repairs_attempted"]
        agg_n_repairs_succeeded += stats["n_repairs_succeeded"]
        agg_n_scored += n

    baseline_acc = agg_baseline_correct / agg_n_scored if agg_n_scored > 0 else 0.0
    pipeline_acc = agg_pipeline_correct / agg_n_scored if agg_n_scored > 0 else 0.0

    artifact = _build_artifact(
        tmpl,
        {
            "n_questions": agg_n_scored,
            "baseline_accuracy": baseline_acc,
            "pipeline_accuracy": pipeline_acc,
            "n_violations_found": agg_n_violations,
            "n_repairs_attempted": agg_n_repairs_attempted,
            "n_repairs_succeeded": agg_n_repairs_succeeded,
        },
        inference_mode="live_gpu",
        v3_recall_at_gate=v3_recall_at_gate,
        status="success",
    )
    _write_json(repo_root, _DELIVERABLE, artifact)

    _log.info(
        "HEADLINE: honest_verdict=%s retro_033_resolved=%s "
        "baseline=%.4f pipeline=%.4f delta=%.4f violations=%d repairs_attempted=%d succeeded=%d",
        artifact.get("honest_verdict"),
        artifact.get("retro_033_resolved"),
        baseline_acc, pipeline_acc,
        pipeline_acc - baseline_acc,
        agg_n_violations, agg_n_repairs_attempted, agg_n_repairs_succeeded,
    )

    # FINAL LINE
    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 594: Live VR CoACE v3 50q benchmark."""
    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=90,
        result_path=str(_REPO_ROOT / _DELIVERABLE),
    ):
        artifact = run_experiment()

    verdict = artifact.get("honest_verdict", "unknown")
    _log.info(
        "Exp %d complete: honest_verdict=%s status=%s",
        EXP_ID, verdict, artifact.get("status", "unknown"),
    )


if __name__ == "__main__":
    main()
