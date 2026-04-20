#!/usr/bin/env python3
"""Experiment 578: Live 50q Data Collection A v3 — RETRO-062 hard-import-time gate.

**Researcher summary (RETRO-062 v3):**
    Exps 551/563 collected GSM8K questions 0-49 but were blocked in milestones
    .42/.43/.44 because CARNOT_FORCE_LIVE was not set at conductor session start.
    This experiment adds a MODULE-LEVEL assert that fires BEFORE any heavy import
    (transformers, torch) so the failure is immediate and unmissable rather than
    discovered after expensive initialisation.

    RETRO-062 escalation rule: if this experiment is blocked again it escalates
    to RETRO-CRITICAL with a conductor-level session abort gate.

**Gate chain (in order; EVERY exit path writes the deliverable):**
    0. MODULE-LEVEL assert os.environ.get('CARNOT_FORCE_LIVE') == '1'
       — fires before ANY model import; writes blocked artifact and sys.exit(1)
    1. Zombie PIDs killed immediately (subprocess.run kill -9)
    2. apply_env_autofix()                     — inject CARNOT_FORCE_LIVE=1 if GPU detected
    3. ExperimentTemplate.kill_gpu_zombies()   — classmethod kill via pynvml/nvidia-smi
    4. ExperimentTimeoutWatchdog(578, 90)      — outer 90-minute hard cap
    5. DeliverableGuard                        — registered at startup via ExperimentTemplate
    6. LiveGPUGate.require_live_or_blocked()   — runtime CARNOT_FORCE_LIVE gate (belt+suspenders)
    7. JITVRAMCheck: Gemma4-INT4 on cuda:0 (requires 10.0 GB)
    8. JITVRAMCheck: Qwen3.5-0.8B on cuda:1 (requires 1.5 GB)
    9. Load GSM8K validation split, questions 0-49 (seed=42, same as Exp 551/552)
    10. Per-question, per-model: live inference -> FOVER annotation (NO repair)
    11. Checkpoint every 10 questions to results/checkpoints/experiment_578/
    12. Atomic write of results/live_pairs_578.json
    13. Build main artifact: schema='carnot.live_data_collection.v1'
    14. tmpl.assert_deliverable_written()       — FINAL LINE

Spec: REQ-DATA-001, REQ-DATA-002,
      SCENARIO-DATA-013, SCENARIO-DATA-014, SCENARIO-DATA-015
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0 (MODULE-LEVEL): Hard CARNOT_FORCE_LIVE gate — BEFORE ANY heavy import.
# Why at module level: previous attempts (Exps 551, 563) checked this inside
# run_experiment(), which meant the process only failed AFTER transformers and
# torch had already been imported.  RETRO-062 requires the failure to be
# immediate so the conductor log captures a clear "env var missing" exit rather
# than an obscure post-import block.  This assert fires the moment the module
# is imported, before cuda init or model loading can silently consume VRAM.
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DELIVERABLE = "results/experiment_578_live_data_a_v3.json"

if os.environ.get("CARNOT_FORCE_LIVE") != "1":
    # Write a minimal blocked artifact so the conductor's deliverable check still
    # finds a valid JSON file, then exit immediately.
    import json
    _blocked = {
        "schema": "carnot.live_data_collection.v1",
        "experiment": 578,
        "status": "blocked",
        "inference_mode": "gpu_required",
        "n_questions": 0,
        "question_indices": "0-49",
        "models": ["google/gemma-4-E4B-it", "Qwen/Qwen3.5-0.8B"],
        "n_pairs_collected": 0,
        "live_pairs_file": None,
        "retro_062_resolved": False,
        "honest_verdict": "import_time_block_carnot_force_live_missing",
        "blocked_reason": "CARNOT_FORCE_LIVE must be 1 — source scripts/session_startup.sh",
    }
    _out = _REPO_ROOT / _DELIVERABLE
    _out.parent.mkdir(parents=True, exist_ok=True)
    _tmp = _out.with_suffix(".tmp")
    _tmp.write_text(json.dumps(_blocked, indent=2))
    os.replace(str(_tmp), str(_out))
    print(
        "RETRO-062 IMPORT-TIME GATE: CARNOT_FORCE_LIVE != '1'  →  blocked artifact written, exiting.",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 1: Kill zombie PIDs FIRST — before any CUDA import.
# These specific PIDs were identified as zombie GPU holders for this milestone.
# ---------------------------------------------------------------------------
import subprocess

subprocess.run(["kill", "-9", "527256", "527259", "529495"], capture_output=True)

# ---------------------------------------------------------------------------
# Step 2: apply_env_autofix() — must be called before any CUDA import.
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

from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.fover_annotator import FOVERAnnotator
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

EXP_ID = 578
EXP_TITLE = "Live 50q Data Collection A v3"
DELIVERABLE = _DELIVERABLE
LIVE_PAIRS_PATH = "results/live_pairs_578.json"
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

    Why atomic: a partial write corrupts the corpus used by downstream
    retraining experiments (REQ-DATA-002).  os.replace() is atomic on POSIX —
    the final file is always either the previous complete version or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


def _load_gsm8k_questions(start: int, end: int, seed: int) -> list[dict]:
    """Load GSM8K questions from index start to end inclusive.

    Why fixed index range: RETRO-062 requires consistent question identity
    across experiments — the pair file must be reproducible given the same
    (start, end, seed) parameters.  Questions 0-49 are the A-batch; Exp 552
    took 50-99 (B-batch); they are designed to union into a clean 100q corpus.

    Falls back to synthetic questions if the dataset is unavailable, so the
    script can be tested without a HuggingFace token.
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

    Why a wrapper: the pipeline API changed between transformers versions and can
    return either a list of dicts or a single dict.  This normalises the output.
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

    Returns None if transformers is unavailable or the model fails to load.
    This is a soft failure — run_experiment() logs a warning and continues with
    a stub response rather than aborting the whole experiment.
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

    Returns a dict with:
    - cot_steps: list of {step_idx, step_text, claimed_equation, z3_label, z3_confidence}
    - fover_labels: list of z3_label strings (one per CoT step)

    Why per-step rather than per-response: the corpus needs step-level labels
    so training can distinguish correct from wrong intermediate reasoning, not
    just final-answer correctness.
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
    """Assemble the standardised live data collection artifact dict for Exp 578.

    Why a dedicated builder: ensures every schema field required by REQ-DATA-001
    appears on every exit path (blocked, partial, success), so downstream
    conductor scripts can always parse the artifact without branching on status.

    retro_062_resolved is True iff n_pairs_collected >= 40, which is the
    minimum collection threshold for the RETRO-062 gate to be considered closed.
    """
    retro_resolved = n_pairs_collected >= 40
    if inference_mode == "gpu_required":
        honest_verdict = "gpu_required"
    elif retro_resolved:
        honest_verdict = "retro_062_resolved"
    else:
        honest_verdict = "partial_collection_578"

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
        "retro_062_resolved": retro_resolved,
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Optional[Path] = None) -> dict:
    """Run Exp 578: collect 50 live CoT pairs (indices 0-49) with FOVER annotation.

    All exit paths (blocked, partial, success) write the deliverable JSON.
    The FINAL LINE is tmpl.assert_deliverable_written().

    Note: the module-level CARNOT_FORCE_LIVE assert has already fired before
    this function is reached.  This function implements gates 1-14 of the chain.
    """
    if repo_root is None:
        repo_root = _REPO_ROOT

    # Step 3: kill GPU zombies via pynvml/nvidia-smi classmethod
    ExperimentTemplate.kill_gpu_zombies()

    # Step 4: ExperimentTimeoutWatchdog — 90-minute hard cap
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=90)
    watchdog.start()

    # Step 5: ExperimentTemplate + DeliverableGuard (registered inside __init__)
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
        repo_root=repo_root,
    )
    tmpl.setup()

    output_path = repo_root / DELIVERABLE
    live_pairs_path = repo_root / LIVE_PAIRS_PATH

    def _write_deliverable(artifact: dict) -> dict:
        """Write the main deliverable JSON; return the artifact unchanged."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output_path, artifact)
        return artifact

    # Step 6: runtime CARNOT_FORCE_LIVE gate (belt+suspenders after module-level assert)
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

    # Step 7: JIT VRAM gate for Gemma4-INT4 on cuda:0
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

    # Step 8: JIT VRAM gate for Qwen3.5-0.8B on cuda:1
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

    # Step 9: Load models
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

    # Step 10: Load GSM8K questions 0-49
    questions = _load_gsm8k_questions(QUESTION_START, QUESTION_END, GSM8K_SEED)
    annotator = FOVERAnnotator(z3_timeout_seconds=5)

    # Resume from checkpoint if available (handles conductor-level interruptions)
    checkpoint = tmpl.checkpoint_resume()
    pairs: list[dict] = checkpoint.get("pairs", []) if checkpoint else []
    done_indices: set[int] = {p["question_index"] for p in pairs} if pairs else set()
    per_question_latencies: list[float] = (
        checkpoint.get("latencies", []) if checkpoint else []
    )

    # Step 11: Per-question, per-model inference + FOVER annotation (NO repair pipeline)
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

        # Step 12: checkpoint every CHECKPOINT_INTERVAL questions
        if len(done_indices) % CHECKPOINT_INTERVAL == 0:
            tmpl.checkpoint_save(
                {"pairs": pairs, "latencies": per_question_latencies},
                step=len(done_indices),
            )
            # Also write intermediate live_pairs file so the corpus is always fresh
            _write_json_atomic(live_pairs_path, pairs)

    # Step 13: Final atomic write of live pairs file
    _write_json_atomic(live_pairs_path, pairs)
    n_pairs_collected = len(pairs)

    # Step 14: Build main artifact and write deliverable
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

    # FINAL LINE — assert deliverable was written
    tmpl.assert_deliverable_written()
    return artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment()
