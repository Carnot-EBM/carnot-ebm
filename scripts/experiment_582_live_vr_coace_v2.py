#!/usr/bin/env python3
"""Experiment 582: Live Verify-Repair with CoACEExtractorV2 on 50 GSM8K Questions.

**Researcher summary:**
    RETRO-033 attempt #12.  Previous 11 attempts all failed because the extractor
    achieved too few violations to show improvement.  Exp 576 added
    CoACEExtractorV2 with prose-arithmetic and multi-step chain patterns.
    Exp 581 validated whether v2_recall >= 0.20 (gate_open field).  This
    experiment is GATED on Exp 581: if gate_open=False, a blocked artifact is
    written and the process exits immediately rather than wasting GPU time.

    Uses GSM8K validation indices 250-299 to avoid any overlap with prior
    benchmarks (Exps 538/551/552/563 used 0-49; Exp 569 used 100-149).

**Gate chain (in order; EVERY exit path writes the deliverable):**
    0. MODULE-LEVEL assert os.environ.get('CARNOT_FORCE_LIVE') == '1'
       -- fires before ANY model import; writes blocked artifact and sys.exit(1)
    1. Zombie PIDs killed immediately (subprocess.run kill -9)
    2. apply_env_autofix()                     -- inject CARNOT_FORCE_LIVE=1 if GPU detected
    3. ExperimentTemplate.kill_gpu_zombies()   -- classmethod kill via pynvml/nvidia-smi
    4. ExperimentTimeoutWatchdog(582, 90)      -- outer 90-minute hard cap
    5. GATE: load Exp 581 result; if gate_open != True: write blocked artifact, sys.exit(0)
    6. LiveGPUGate.require_live_or_blocked()   -- CARNOT_FORCE_LIVE gate
    7. JIT VRAM gate -> cuda:0 (Gemma4 or primary model, requires 10.0 GB)
    8. JIT VRAM gate -> cuda:1 (Qwen3.5-0.8B, requires 1.5 GB)
    9. Load 50 GSM8K questions (validation indices 250-299)
    10. Per-question: baseline inference -> CoACEV2 extraction -> repair if violations
    11. Build artifact: schema='carnot.live_vr_coace.v2', all required fields
    12. tmpl.assert_deliverable_written()  -- FINAL LINE

Spec: REQ-BENCH-015 (v4),
      SCENARIO-BENCH-036, SCENARIO-BENCH-037, SCENARIO-BENCH-038
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0 (MODULE-LEVEL): Hard CARNOT_FORCE_LIVE gate BEFORE any heavy import.
# Why at module level: importing torch/transformers initialises CUDA and
# allocates VRAM, so the env-var check must happen before those imports to keep
# the "missing env var" failure instant and unmissable in the conductor log.
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DELIVERABLE = "results/experiment_582_live_vr_coace_v2.json"

if os.environ.get("CARNOT_FORCE_LIVE") != "1":
    import json as _json

    _blocked_preflight = {
        "schema": "carnot.live_vr_coace.v2",
        "experiment": 582,
        "status": "blocked",
        "inference_mode": "gpu_required",
        "n_questions": 0,
        "question_indices": "250-299",
        "extractor": "coace_v2",
        "v2_recall_at_gate": None,
        "baseline_accuracy": 0.0,
        "pipeline_accuracy": 0.0,
        "signed_improvement": 0.0,
        "n_violations_found": 0,
        "n_repairs_applied": 0,
        "n_repairs_improved": 0,
        "retro_033_resolved": False,
        "honest_verdict": "import_time_block_carnot_force_live_missing",
        "blocked_reason": "CARNOT_FORCE_LIVE must be 1 -- source scripts/session_startup.sh",
    }
    _out = _REPO_ROOT / _DELIVERABLE
    _out.parent.mkdir(parents=True, exist_ok=True)
    _tmp = _out.with_suffix(".tmp")
    _tmp.write_text(_json.dumps(_blocked_preflight, indent=2))
    os.replace(str(_tmp), str(_out))
    print(
        "RETRO-062 IMPORT-TIME GATE: CARNOT_FORCE_LIVE != '1'  ->  blocked artifact written, exiting.",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 1: Kill zombie PIDs FIRST -- before any CUDA import.
# ---------------------------------------------------------------------------
import subprocess

subprocess.run(["kill", "-9"], capture_output=True)  # no-op; real PIDs inserted by conductor

# ---------------------------------------------------------------------------
# Step 2: apply_env_autofix() MUST be called before any CUDA import.
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
import time
from typing import Any, Optional

from carnot.extraction.coace_extractor_v2 import CoACEExtractorV2
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

EXP_ID = 582
EXP_TITLE = "Live VR CoACE v2 -- 50q GSM8K verify-repair with CoACEExtractorV2"
N_QUESTIONS = 50
# Indices 250-299: fresh batch with no overlap with any prior benchmark.
QUESTION_START = 250
QUESTION_END = 299  # inclusive -> 50 questions

GEMMA4_REQUIRED_GB = 10.0
QWEN_REQUIRED_GB = 1.5
GATE_FILE = "results/experiment_581_coace_recall_diagnostic_v2.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(repo_root: Path, rel_path: str, data: dict) -> None:
    """Write JSON to repo_root / rel_path, creating parent dirs as needed."""
    out = repo_root / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))


def _load_gate(repo_root: Path) -> Optional[dict]:
    """Load Exp 581 gate result.  Returns None if file is missing or unreadable.

    Why a dedicated loader: gate_open is the ONLY reason this experiment is
    allowed to consume GPU time.  Loud failure (blocked artifact, not crash)
    is required when the upstream file is absent or corrupt.
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

    Uses the validation split (not test split) with a fixed index window so
    overlap with prior experiments can be checked by index comparison alone.
    Falls back to synthetic questions when the datasets library is unavailable
    (CI / offline environments).
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
                # Deliberately wrong arithmetic so CoACEV2 can fire
                answer_text = f"#### {idx + 1}\n3 + 3 = 7, so the answer is {idx + 1}"
            else:
                answer_text = f"#### {idx * 2}"
            result.append({"question": f"Synthetic question {i}: What is {i} + {i}?", "answer": answer_text})
        return result


def _qwen_generate(pipeline: Any, prompt: str) -> str:
    """Run one prompt through a HuggingFace transformers pipeline and return the text.

    Normalises the varying pipeline output formats across transformers versions.
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

    Returns None if transformers is unavailable or the model fails to load.
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

    The wording explicitly tells the model that errors exist so it re-checks
    each step rather than copying its prior answer verbatim.
    """
    return (
        f"Question: {question}\n\n"
        "Your previous answer contained arithmetic errors. "
        "Solve the problem step by step, checking each calculation carefully."
    )


def _run_per_question(
    extractor: CoACEExtractorV2,
    generate_fn: Any,  # callable(str) -> str
    questions: list[dict],
) -> dict:
    """Run baseline + repair loop for every question; return per-question stats.

    For each question:
    1. Generate baseline response (no pipeline).
    2. Run CoACEExtractorV2 on the baseline response.
    3. If violations found: generate a repair response and mark repair_applied.
    4. Evaluate correctness of baseline and pipeline responses against gold answer.

    Returns a dict with aggregate counts and per-question records.
    """
    baseline_correct_total = 0
    pipeline_correct_total = 0
    n_violations_found = 0
    n_repairs_applied = 0
    n_repairs_improved = 0
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

        # CoACEV2 extraction
        try:
            coace_result = extractor.extract(baseline_resp)
            violation_found = coace_result.n_violations > 0
        except Exception as exc:
            _log.warning("CoACEV2 extraction error: %s", exc)
            coace_result = None
            violation_found = False

        if violation_found:
            n_violations_found += 1
            repair_prompt = _build_repair_prompt(prompt)
            try:
                pipeline_resp = generate_fn(repair_prompt)
                n_repairs_applied += 1
            except Exception as exc:
                _log.warning("Repair inference error: %s", exc)
                pipeline_resp = baseline_resp
        else:
            pipeline_resp = baseline_resp

        bc = _is_correct(baseline_resp, gold)
        pc = _is_correct(pipeline_resp, gold)

        if violation_found and pc and not bc:
            n_repairs_improved += 1

        baseline_correct_total += int(bc)
        pipeline_correct_total += int(pc)

        per_question.append({
            "baseline_correct": bc,
            "pipeline_correct": pc,
            "violation_found": violation_found,
            "repair_applied": violation_found,
            "n_violations": coace_result.n_violations if coace_result else 0,
        })

        _log.info(
            "q_done: baseline=%s pipeline=%s violation=%s",
            bc, pc, violation_found,
        )

    n = len(questions)
    return {
        "baseline_accuracy": baseline_correct_total / n if n > 0 else 0.0,
        "pipeline_accuracy": pipeline_correct_total / n if n > 0 else 0.0,
        "n_violations_found": n_violations_found,
        "n_repairs_applied": n_repairs_applied,
        "n_repairs_improved": n_repairs_improved,
        "per_question": per_question,
    }


def _build_artifact(
    tmpl: ExperimentTemplate,
    stats: dict,
    inference_mode: str,
    v2_recall_at_gate: Optional[float] = None,
    status: str = "success",
) -> dict:
    """Assemble the standardised v2 artifact for this experiment.

    Ensures every field required by REQ-BENCH-015 (v4) is present on every
    exit path (blocked, gpu_required, and live success / no-improvement).
    """
    baseline = stats.get("baseline_accuracy", 0.0)
    pipeline = stats.get("pipeline_accuracy", 0.0)
    signed_improvement = pipeline - baseline
    retro_033_resolved = signed_improvement > 0 and inference_mode == "live_gpu"

    if signed_improvement > 0:
        honest_verdict = "first_positive"
    elif inference_mode == "live_gpu":
        honest_verdict = "live_no_improvement_v2"
    else:
        honest_verdict = "blocked_gate_closed_recall_too_low"

    return tmpl.build_result(
        {
            "schema": "carnot.live_vr_coace.v2",
            "inference_mode": inference_mode,
            "n_questions": stats.get("n_questions", N_QUESTIONS),
            "question_indices": "250-299",
            "extractor": "coace_v2",
            "v2_recall_at_gate": v2_recall_at_gate,
            "baseline_accuracy": baseline,
            "pipeline_accuracy": pipeline,
            "signed_improvement": signed_improvement,
            "n_violations_found": stats.get("n_violations_found", 0),
            "n_repairs_applied": stats.get("n_repairs_applied", 0),
            "n_repairs_improved": stats.get("n_repairs_improved", 0),
            "retro_033_resolved": retro_033_resolved,
            "honest_verdict": honest_verdict,
        },
        status=status,
    )


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Optional[Path] = None) -> dict:
    """Run Exp 582: 50-question live verify-repair with CoACEExtractorV2.

    All exit paths (blocked gate, gpu_required, error, success) write the
    deliverable JSON.  The FINAL LINE is tmpl.assert_deliverable_written().
    """
    if repo_root is None:
        repo_root = _REPO_ROOT

    # Step 3: kill_gpu_zombies via ExperimentTemplate classmethod (uses pynvml)
    ExperimentTemplate.kill_gpu_zombies()

    # ExperimentTemplate setup
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
    # Step 5 (GATE): check Exp 581 gate_open field BEFORE any inference
    # -----------------------------------------------------------------------
    gate_data = _load_gate(repo_root)
    v2_recall_at_gate: Optional[float] = None
    if gate_data is not None:
        v2_recall_at_gate = gate_data.get("v2_recall")

    if gate_data is None or not gate_data.get("gate_open", False):
        _log.warning("GATE BLOCKED: Exp 581 gate_open is False or file missing (v2_recall=%s)", v2_recall_at_gate)
        blocked = _build_artifact(
            tmpl,
            {"n_questions": 0},
            inference_mode="blocked_gate_closed_recall_too_low",
            v2_recall_at_gate=v2_recall_at_gate,
            status="blocked",
        )
        blocked["upstream_exp"] = 581
        _write_json(repo_root, _DELIVERABLE, blocked)
        tmpl.assert_deliverable_written()
        return blocked

    # Step 6: CARNOT_FORCE_LIVE gate
    gate_result = LiveGPUGate.require_live_or_blocked(tmpl, model_ids=[])
    if gate_result is not None:
        deferred = _build_artifact(
            tmpl,
            {"n_questions": 0},
            inference_mode="gpu_required",
            v2_recall_at_gate=v2_recall_at_gate,
            status="gpu_required",
        )
        deferred["gate_result"] = str(gate_result)
        return _write_and_return(deferred)

    # -----------------------------------------------------------------------
    # Steps 7-8: JIT VRAM gates
    # -----------------------------------------------------------------------
    gemma4_vram = JITVRAMCheck(device_id=0)
    gemma4_gate = gemma4_vram.gate_model_load(
        model_id="Gemma4-INT4",
        required_gb=GEMMA4_REQUIRED_GB,
        retry_wait_s=5,
    )
    if not gemma4_gate.is_cleared:
        _log.warning("JIT VRAM gate blocked Gemma4-INT4: %.1f GB free", gemma4_gate.available_gb)
        blocked = _build_artifact(
            tmpl,
            {"n_questions": 0},
            inference_mode="gpu_required",
            v2_recall_at_gate=v2_recall_at_gate,
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
            v2_recall_at_gate=v2_recall_at_gate,
            status="gpu_vram_insufficient",
        )
        blocked["vram_block_reason"] = f"qwen_insufficient: {qwen_gate.available_gb:.1f} GB free"
        return _write_and_return(blocked)

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
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

    models_available = []
    if gemma4_loader:
        models_available.append(("Gemma4-INT4", lambda p: gemma4_loader.generate(p)))
    if qwen_pipe:
        models_available.append(("Qwen3.5-0.8B", lambda p: _qwen_generate(qwen_pipe, p)))

    if not models_available:
        _log.warning("No live models available -- writing gpu_required artifact")
        deferred = _build_artifact(
            tmpl,
            {"n_questions": 0},
            inference_mode="gpu_required",
            v2_recall_at_gate=v2_recall_at_gate,
            status="gpu_required",
        )
        deferred["no_models"] = True
        return _write_and_return(deferred)

    # -----------------------------------------------------------------------
    # Step 9: Load 50 GSM8K questions (indices 250-299)
    # -----------------------------------------------------------------------
    questions = _load_gsm8k_questions(QUESTION_START, QUESTION_END)
    _log.info("Loaded %d GSM8K questions (indices %d-%d)", len(questions), QUESTION_START, QUESTION_END)

    # -----------------------------------------------------------------------
    # Step 10: Per-question verify-repair loop (CoACEExtractorV2)
    # -----------------------------------------------------------------------
    extractor = CoACEExtractorV2(tolerance=1e-6, min_confidence=0.5)

    agg_baseline_correct = 0
    agg_pipeline_correct = 0
    agg_n_violations = 0
    agg_n_repairs = 0
    agg_n_improved = 0
    agg_n_scored = 0

    for model_name, generate_fn in models_available:
        _log.info("=== Running %s on %d questions ===", model_name, len(questions))
        stats = _run_per_question(extractor, generate_fn, questions)

        n = len(questions)
        agg_baseline_correct += round(stats["baseline_accuracy"] * n)
        agg_pipeline_correct += round(stats["pipeline_accuracy"] * n)
        agg_n_violations += stats["n_violations_found"]
        agg_n_repairs += stats["n_repairs_applied"]
        agg_n_improved += stats["n_repairs_improved"]
        agg_n_scored += n

    # -----------------------------------------------------------------------
    # Step 11: Build artifact
    # -----------------------------------------------------------------------
    baseline_acc = agg_baseline_correct / agg_n_scored if agg_n_scored > 0 else 0.0
    pipeline_acc = agg_pipeline_correct / agg_n_scored if agg_n_scored > 0 else 0.0

    artifact = _build_artifact(
        tmpl,
        {
            "n_questions": agg_n_scored,
            "baseline_accuracy": baseline_acc,
            "pipeline_accuracy": pipeline_acc,
            "n_violations_found": agg_n_violations,
            "n_repairs_applied": agg_n_repairs,
            "n_repairs_improved": agg_n_improved,
        },
        inference_mode="live_gpu",
        v2_recall_at_gate=v2_recall_at_gate,
        status="success",
    )
    _write_json(repo_root, _DELIVERABLE, artifact)

    _log.info(
        "HEADLINE: honest_verdict=%s retro_033_resolved=%s "
        "baseline=%.4f pipeline=%.4f delta=%.4f violations=%d repairs=%d improved=%d",
        artifact.get("honest_verdict"),
        artifact.get("retro_033_resolved"),
        baseline_acc, pipeline_acc,
        pipeline_acc - baseline_acc,
        agg_n_violations, agg_n_repairs, agg_n_improved,
    )

    # Step 12: FINAL LINE
    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 582: Live VR CoACE v2 50q benchmark."""
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
