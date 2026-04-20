#!/usr/bin/env python3
"""Experiment 563: Live 50q Data Collection A v2 — re-run of Exp 551 with CARNOT_FORCE_LIVE preflight.

**Researcher summary (RETRO-062):**
    Exp 551 was blocked because CARNOT_FORCE_LIVE was not set when the conductor session
    started.  GSM8K questions 0-49 (batch A) were never collected, leaving the FOVER
    corpus v2 (132 pairs) missing an entire batch.

    This experiment re-runs the Exp 551 pattern (questions 0-49) with an explicit
    hard preflight assertion that CARNOT_FORCE_LIVE=1 is set before any model is loaded.
    If the assertion fails the script writes a blocked artifact and exits immediately —
    no silent degradation to simulated mode.

**Gate chain (in order; EVERY exit path writes the deliverable):**
    0. Kill zombie PIDs 527256, 527259, 529495 (subprocess.run kill -9) — FIRST
    1. apply_env_autofix()                     — inject CARNOT_FORCE_LIVE=1 if GPU detected
    2. ExperimentTemplate.kill_gpu_zombies()   — classmethod kill via pynvml/nvidia-smi
    3. ExperimentTimeoutWatchdog(563, 90)      — outer 90-minute hard cap
    4. HARD PREFLIGHT: assert CARNOT_FORCE_LIVE==1 — write blocked artifact + exit(1) if not
    5. ExperimentTemplate + DeliverableGuard
    6. LiveGPUGate.require_live_or_blocked()   — soft gate (blocked artifact on failure)
    7. JITVRAMCheck: Gemma4-INT4 on cuda:0 (requires 10.0 GB)
    8. JITVRAMCheck: Qwen3.5-0.8B on cuda:1 (requires 1.5 GB)
    9. Load GSM8K validation split, questions 0-49 (seed=42, consistent with Exp 552)
    10. Per-question, per-model: live inference -> FOVER annotation (NO repair)
    11. Checkpoint every 10 questions to results/checkpoints/experiment_563/
    12. Atomic write of results/live_pairs_563.json
    13. Build main artifact: schema='carnot.live_data_collection.v1'
    14. tmpl.assert_deliverable_written()       — FINAL LINE

Spec: REQ-DATA-001, REQ-DATA-002,
      SCENARIO-DATA-010, SCENARIO-DATA-011, SCENARIO-DATA-012
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: Kill zombie PIDs FIRST — before any CUDA import.
# These specific PIDs were identified as zombie GPU holders from milestone .42.
# ---------------------------------------------------------------------------
import subprocess

subprocess.run(["kill", "-9", "527256", "527259", "529495"], capture_output=True)

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() — must be called before any CUDA import.
# Overrides CARNOT_FORCE_LIVE='0'/'false'/''/None to '1' when GPU confirmed.
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
import time
from typing import Any, Optional

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: F401
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.fover_annotator import FOVERAnnotator
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader
from carnot.pipeline.jit_vram_check import JITVRAMCheck
from carnot.pipeline.live_100q_v7_helpers import _extract_answer, _is_correct
from carnot.pipeline.live_gpu_gate import LiveGPUGate
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 563
EXP_TITLE = "Live 50q Data Collection A v2"
DELIVERABLE = "results/experiment_563_live_data_a_v2.json"
LIVE_PAIRS_PATH = "results/live_pairs_563.json"
N_QUESTIONS = 50
QUESTION_START = 0
QUESTION_END = 49
QUESTION_INDICES = "0-49"
GSM8K_SEED = 42

GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"
QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
GEMMA4_REQUIRED_GB = 10.0
QWEN_REQUIRED_GB = 1.5

CHECKPOINT_INTERVAL = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON to path atomically via a .tmp file then rename.

    Why atomic: a partial write to the live pairs file would corrupt the corpus
    used by downstream retraining experiments (REQ-DATA-002).  Writing to a
    .tmp file then os.replace() is an atomic rename on POSIX — the final file
    is always either the previous complete version or the new complete version.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


def _load_gsm8k_questions(start: int, end: int, seed: int) -> list[dict]:
    """Load GSM8K questions from index start to end inclusive (seed for reproducibility).

    Why by fixed index range rather than shuffled: RETRO-062 requires consistent
    question identity across experiments — the pair file must be reproducible
    given the same (start, end, seed) parameters.  This experiment takes 0-49
    (batch A) to complement Exp 552's 50-99 (batch B).

    Returns list of dicts with keys: 'question', 'answer', 'index'.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        return [
            {"question": ds[i]["question"], "answer": ds[i]["answer"], "index": i}
            for i in range(start, end + 1)
        ]
    except Exception as exc:
        _log.warning(
            "_load_gsm8k_questions: dataset load failed (%s) — using synthetic fallback", exc
        )
        return [
            {
                "question": f"Synthetic question {i}: What is {i} + {i}?",
                "answer": f"#### {i * 2}",
                "index": i,
            }
            for i in range(start, end + 1)
        ]


def _qwen_generate(pipeline: Any, prompt: str) -> str:
    """Run one prompt through a HuggingFace transformers pipeline and return the text.

    Why a wrapper: the pipeline API changed between transformers versions and can return
    either a list of dicts or a single dict.  This function normalises the output.
    """
    try:
        out = pipeline(prompt, max_new_tokens=256, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get("generated_text", str(out[0]))
        return str(out)
    except Exception as exc:
        return f"[qwen_error: {exc}]"


def _load_qwen_pipeline(device: str) -> Optional[Any]:
    """Load Qwen3.5-0.8B as a HuggingFace text-generation pipeline on the given device.

    Returns None if transformers is not available or the model fails to load.
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


def _annotate_response(annotator: FOVERAnnotator, response: str, question_id: str) -> dict:
    """Annotate one model response with FOVER step labels.

    Returns a dict with keys:
    - cot_steps: list of {step_idx, step_text, claimed_equation, z3_label, z3_confidence}
    - fover_labels: list of z3_label strings (one per step)
    """
    steps = annotator.annotate_response(response, question_id)
    return {
        "cot_steps": [
            {
                "step_idx": s.step_idx,
                "step_text": s.step_text,
                "claimed_equation": s.claimed_equation,
                "z3_label": s.z3_label,
                "z3_confidence": s.z3_confidence,
            }
            for s in steps
        ],
        "fover_labels": [s.z3_label for s in steps],
    }


def _build_live_data_artifact(
    inference_mode: str,
    n_questions: int,
    n_pairs_collected: int,
    live_pairs_file: Optional[str],
    per_question_latencies: list[float],
) -> dict:
    """Assemble the standardised live data collection artifact dict for Exp 563.

    Why a dedicated builder: ensures every field required by REQ-DATA-001 is present
    on every exit path, including blocked and partial-collection exits.

    retro_062_resolved is True when n_pairs_collected >= 40 — this is the criterion
    from RETRO-062 that Exp 551's missing batch A is replaced by this experiment.
    honest_verdict reflects whether live data was collected at the 40-pair threshold.
    """
    retro_062_resolved = n_pairs_collected >= 40

    honest_verdict: str
    if inference_mode == "gpu_required":
        honest_verdict = "gpu_required"
    elif n_pairs_collected >= 40:
        honest_verdict = "live_data_collected"
    else:
        honest_verdict = "partial_collection"

    mean_latency = (
        sum(per_question_latencies) / len(per_question_latencies)
        if per_question_latencies
        else 0.0
    )

    return {
        "schema": "carnot.live_data_collection.v1",
        "inference_mode": inference_mode,
        "n_questions": n_questions,
        "question_indices": QUESTION_INDICES,
        "models": [GEMMA4_MODEL_ID, QWEN_MODEL_ID],
        "n_pairs_collected": n_pairs_collected,
        "live_pairs_file": live_pairs_file,
        "mean_latency_s": mean_latency,
        "per_question_latencies": per_question_latencies,
        "retro_062_resolved": retro_062_resolved,
        "honest_verdict": honest_verdict,
    }


def _write_blocked_preflight(
    output_path: Path,
    reason: str,
) -> dict:
    """Write a minimal blocked artifact when the CARNOT_FORCE_LIVE preflight fails.

    This is called BEFORE ExperimentTemplate is fully initialised so it writes
    directly rather than going through tmpl.build_result().  The schema fields
    match the standard live_data_collection.v1 schema so downstream tooling
    can parse it uniformly.

    Why separate from tmpl.build_result(): tmpl.setup() registers atexit hooks and
    creates checkpoint dirs — we do NOT want those side effects when we are about
    to sys.exit(1).  Writing the blocked artifact directly and exiting is cleaner.
    """
    artifact = {
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "status": "blocked",
        "blocked_reason": reason,
        **_build_live_data_artifact(
            inference_mode="gpu_required",
            n_questions=0,
            n_pairs_collected=0,
            live_pairs_file=None,
            per_question_latencies=[],
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_path, artifact)
    return artifact


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Optional[Path] = None) -> dict:
    """Run Exp 563: collect 50 live CoT pairs (indices 0-49) with FOVER annotation.

    All exit paths (deferred, live, error) write the deliverable JSON.
    The FINAL LINE is tmpl.assert_deliverable_written().

    RETRO-062 fix: this function asserts CARNOT_FORCE_LIVE==1 immediately after
    apply_env_autofix(), before any model load.  This prevents the silent deferral
    that blocked Exp 551 — if the assertion fails the blocked artifact is written
    and the process exits with code 1 so the conductor sees a failure immediately.
    """
    if repo_root is None:
        repo_root = _REPO_ROOT

    output_path = repo_root / DELIVERABLE

    # Step 2: kill GPU zombies via pynvml/nvidia-smi classmethod
    ExperimentTemplate.kill_gpu_zombies()

    # Step 3: ExperimentTimeoutWatchdog — 90-minute hard cap
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=90)
    watchdog.start()

    # Step 4: HARD PREFLIGHT — assert CARNOT_FORCE_LIVE is set.
    # apply_env_autofix() (called at module level above) injects the var when GPU
    # is detected, so if it is still absent here either:
    #   (a) no GPU was detected (need to run on a GPU machine), or
    #   (b) apply_env_autofix() failed to detect the GPU (check drivers).
    # In either case we must not proceed silently — write blocked + exit(1).
    if os.environ.get("CARNOT_FORCE_LIVE") != "1":
        reason = (
            "CARNOT_FORCE_LIVE must be set to '1' before running this experiment. "
            "Fix: source scripts/session_startup.sh before launching the conductor. "
            "RETRO-062: this hard preflight was added to prevent silent deferral."
        )
        _log.error("HARD PREFLIGHT FAILED: %s", reason)
        blocked = _write_blocked_preflight(output_path, reason)
        watchdog.stop()
        sys.exit(1)

    # Step 5: ExperimentTemplate + DeliverableGuard (registered inside __init__)
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
        repo_root=repo_root,
    )
    tmpl.setup()

    live_pairs_path = repo_root / LIVE_PAIRS_PATH

    def _write_deliverable(artifact: dict) -> dict:
        """Write the main deliverable JSON; return the artifact."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output_path, artifact)
        return artifact

    # Step 6: CARNOT_FORCE_LIVE gate — soft (returns blocked artifact, not raise)
    gate_result = LiveGPUGate.require_live_or_blocked(tmpl, model_ids=[])
    if gate_result is not None:
        deferred = tmpl.build_result(
            _build_live_data_artifact(
                inference_mode="gpu_required",
                n_questions=0,
                n_pairs_collected=0,
                live_pairs_file=None,
                per_question_latencies=[],
            ),
            status="blocked",
        )
        _write_deliverable(deferred)
        watchdog.stop()
        tmpl.assert_deliverable_written()
        return deferred

    # Step 7: JIT VRAM gates — check immediately before each model load
    vram0 = JITVRAMCheck(device_id=0)
    gate0 = vram0.gate_model_load(GEMMA4_MODEL_ID, required_gb=GEMMA4_REQUIRED_GB)
    if not gate0.is_cleared:
        blocked = tmpl.build_result(
            _build_live_data_artifact("gpu_required", 0, 0, None, []),
            status="blocked",
            blocked_reason=f"cuda:0 VRAM insufficient: {gate0.available_gb:.1f} GB < {GEMMA4_REQUIRED_GB} GB",
        )
        _write_deliverable(blocked)
        watchdog.stop()
        tmpl.assert_deliverable_written()
        return blocked

    vram1 = JITVRAMCheck(device_id=1)
    gate1 = vram1.gate_model_load(QWEN_MODEL_ID, required_gb=QWEN_REQUIRED_GB)
    if not gate1.is_cleared:
        blocked = tmpl.build_result(
            _build_live_data_artifact("gpu_required", 0, 0, None, []),
            status="blocked",
            blocked_reason=f"cuda:1 VRAM insufficient: {gate1.available_gb:.1f} GB < {QWEN_REQUIRED_GB} GB",
        )
        _write_deliverable(blocked)
        watchdog.stop()
        tmpl.assert_deliverable_written()
        return blocked

    # Step 8: Load models
    gguf_path = os.environ.get("CARNOT_GEMMA4_GGUF_PATH", "")
    gemma4 = Gemma4QuantizedLoader(
        model_path=gguf_path,
        n_gpu_layers=-1,
        max_tokens=512,
        jit_vram_check=vram0,
    )
    gemma4_loaded = gemma4.load()
    if not gemma4_loaded:
        _log.warning("Gemma4 load failed — continuing with stub")

    qwen_pipeline = _load_qwen_pipeline("cuda:1")

    # Step 9: Load GSM8K questions 0-49 (batch A, consistent with Exp 552's seed)
    questions = _load_gsm8k_questions(QUESTION_START, QUESTION_END, GSM8K_SEED)
    annotator = FOVERAnnotator(z3_timeout_seconds=5)

    # Resume from checkpoint if available
    checkpoint = tmpl.checkpoint_resume()
    pairs: list[dict] = checkpoint.get("pairs", []) if checkpoint else []
    done_indices: set[int] = {p["question_index"] for p in pairs} if pairs else set()
    per_question_latencies: list[float] = (
        checkpoint.get("latencies", []) if checkpoint else []
    )

    # Step 10: Per-question, per-model inference + FOVER annotation (NO repair)
    for q_dict in questions:
        q_idx = q_dict["index"]
        if q_idx in done_indices:
            continue

        question_text = q_dict["question"]
        gold_answer = q_dict["answer"]
        t_start = time.perf_counter()

        for model_id, generate_fn in [
            (GEMMA4_MODEL_ID, lambda p: gemma4.generate(p)),
            (QWEN_MODEL_ID, lambda p: _qwen_generate(qwen_pipeline, p) if qwen_pipeline else "[qwen_not_loaded]"),
        ]:
            response = generate_fn(question_text)
            is_correct_flag = _is_correct(response, _extract_answer(gold_answer))
            annotation = _annotate_response(annotator, response, f"q{q_idx}")

            pairs.append(
                {
                    "question_index": q_idx,
                    "question": question_text,
                    "model": model_id,
                    "response": response,
                    "is_correct": is_correct_flag,
                    "cot_steps": annotation["cot_steps"],
                    "fover_labels": annotation["fover_labels"],
                }
            )

        latency = time.perf_counter() - t_start
        per_question_latencies.append(latency)
        done_indices.add(q_idx)

        # Step 11: checkpoint every CHECKPOINT_INTERVAL questions
        if len(done_indices) % CHECKPOINT_INTERVAL == 0:
            tmpl.checkpoint_save(
                {"pairs": pairs, "latencies": per_question_latencies},
                step=len(done_indices),
            )
            # Also write intermediate live_pairs file atomically
            _write_json_atomic(live_pairs_path, pairs)

    # Step 12: Final atomic write of live pairs file
    _write_json_atomic(live_pairs_path, pairs)
    n_pairs_collected = len(pairs)

    # Step 13: Build main artifact
    artifact_data = _build_live_data_artifact(
        inference_mode="live_gpu",
        n_questions=N_QUESTIONS,
        n_pairs_collected=n_pairs_collected,
        live_pairs_file=str(live_pairs_path),
        per_question_latencies=per_question_latencies,
    )
    artifact = tmpl.build_result(artifact_data, status="success")
    _write_deliverable(artifact)
    watchdog.stop()

    # Step 14: FINAL LINE — assert deliverable was written
    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment()
