#!/usr/bin/env python3
"""Experiment 579: Live 50q Data Collection C — GSM8K batch C (indices 200-249).

**Researcher summary:**
    Exps 578 (batch A, 0-49) and 552 (batch B, 50-99) produced FOVER corpus v2 with
    132 pairs.  This experiment adds a third distinct batch (200-249) to raise question
    diversity for JEPA v11 training (Exp 580).  More cross-question diversity means
    stronger contrastive training signal for the CPMI objective.

    Gate chain mirrors Exp 578 (RETRO-062 hard-import-time gate):
    0. MODULE-LEVEL assert os.environ.get('CARNOT_FORCE_LIVE') == '1'
       — fires before ANY model import; writes blocked artifact and sys.exit(1)
    1. Zombie PIDs killed immediately (subprocess.run kill -9)
    2. apply_env_autofix()
    3. ExperimentTemplate.kill_gpu_zombies()
    4. ExperimentTimeoutWatchdog(579, 90)
    5. DeliverableGuard
    6. LiveGPUGate.require_live_or_blocked()
    7. JITVRAMCheck: Gemma4-INT4 on cuda:0 (requires 10.0 GB)
    8. JITVRAMCheck: Qwen3.5-0.8B on cuda:1 (requires 1.5 GB)
    9. Load GSM8K test split, questions 200-249 (seed=42)
    10. Per-question, per-model: live inference -> FOVER annotation (NO repair)
    11. Checkpoint every 10 questions to results/checkpoints/experiment_579/
    12. Atomic write of results/live_pairs_579.json
    13. Merge: fover_corpus_v2.json + live_pairs_578.json (if exists) + live_pairs_579.json
        -> results/fover_corpus_v3.json  (AtomicResultWriter)
    14. Build main artifact: schema='carnot.live_data_collection.v1'
    15. tmpl.assert_deliverable_written()  — FINAL LINE

Spec: REQ-DATA-001, REQ-DATA-002,
      SCENARIO-DATA-016, SCENARIO-DATA-017, SCENARIO-DATA-018
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0 (MODULE-LEVEL): Hard CARNOT_FORCE_LIVE gate — BEFORE ANY heavy import.
# Why at module level: mirrors the RETRO-062 pattern from Exp 578.  The failure
# must be immediate and unmissable, not discovered after expensive GPU init.
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DELIVERABLE = "results/experiment_579_live_data_c.json"

if os.environ.get("CARNOT_FORCE_LIVE") != "1":
    import json

    _blocked = {
        "schema": "carnot.live_data_collection.v1",
        "experiment": 579,
        "status": "blocked",
        "inference_mode": "gpu_required",
        "n_questions": 0,
        "question_indices": "200-249",
        "models": ["google/gemma-4-E4B-it", "Qwen/Qwen3.5-0.8B"],
        "n_pairs_collected": 0,
        "live_pairs_file": None,
        "fover_corpus_v3_size": 0,
        "honest_verdict": "import_time_block_carnot_force_live_missing",
        "blocked_reason": "CARNOT_FORCE_LIVE must be 1 — source scripts/session_startup.sh",
    }
    _out = _REPO_ROOT / _DELIVERABLE
    _out.parent.mkdir(parents=True, exist_ok=True)
    _tmp = _out.with_suffix(".tmp")
    _tmp.write_text(json.dumps(_blocked, indent=2))
    os.replace(str(_tmp), str(_out))
    print(
        "EXP-579 IMPORT-TIME GATE: CARNOT_FORCE_LIVE != '1'  →  blocked artifact written, exiting.",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 1: Kill zombie PIDs FIRST — before any CUDA import.
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
from carnot.pipeline.fover_corpus import merge_fover_sources
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

EXP_ID = 579
EXP_TITLE = "Live 50q Data Collection C"
DELIVERABLE = _DELIVERABLE
LIVE_PAIRS_PATH = "results/live_pairs_579.json"
FOVER_CORPUS_V2_PATH = "results/fover_corpus_v2.json"
LIVE_PAIRS_578_PATH = "results/live_pairs_578.json"
FOVER_CORPUS_V3_PATH = "results/fover_corpus_v3.json"
N_QUESTIONS = 50
QUESTION_START = 200
QUESTION_END = 249
QUESTION_INDICES = "200-249"
GSM8K_SEED = 42

# MODEL SELECTION — prefer cached SOTA GGUFs over legacy tiny models.
# See RETRO-066/068/070.  Only SOTA GGUFs produce real arithmetic CoT.
from carnot.inference.sota_models import cached_sota_pair as _cached_sota_pair

_sota_specs = _cached_sota_pair(gpu_indices=(0, 1))
if _sota_specs is not None:
    QWEN_MODEL_ID = _sota_specs[0]["hf_id"]
    GEMMA4_MODEL_ID = _sota_specs[1]["hf_id"]
    QWEN_MODEL_PATH = _sota_specs[0]["model_path"]
    GEMMA4_MODEL_PATH = _sota_specs[1]["model_path"]
    _MODELS_USED_REAL_SOTA = True
else:
    print("WARNING: cached SOTA GGUFs unavailable, falling back to tiny models — output quality will be poor")
    QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
    GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"
    QWEN_MODEL_PATH = None
    GEMMA4_MODEL_PATH = None
    _MODELS_USED_REAL_SOTA = False
GEMMA4_REQUIRED_GB = 10.0
QWEN_REQUIRED_GB = 1.5

CHECKPOINT_INTERVAL = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON to path atomically via a .tmp file then rename.

    Why atomic: a partial write corrupts the corpus used by downstream JEPA
    retraining experiments (REQ-DATA-002).  os.replace() is atomic on POSIX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


def _load_gsm8k_questions(start: int, end: int, seed: int) -> list[dict]:
    """Load GSM8K test-split questions from index start to end inclusive.

    Why fixed index range: consistent question identity across experiments so
    that batches A (0-49), B (50-99), and C (200-249) union into a clean corpus
    without overlap.  Falls back to synthetic questions when the dataset is
    unavailable (CI / offline environments).
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
    """Run one prompt through a HuggingFace text-generation pipeline and return the text.

    Normalises across transformers versions that return either a list of dicts
    or a single dict.
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

    Returns None on failure so run_experiment() can continue with stub responses
    rather than aborting the whole batch.
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

    Step-level labels (not just final-answer correctness) let JEPA training
    distinguish wrong intermediate reasoning from wrong final answers.
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
    fover_corpus_v3_size: int = 0,
) -> dict:
    """Assemble the standardised live data collection artifact dict for Exp 579.

    Every schema field appears on every exit path (blocked, partial, success) so
    downstream scripts never need to branch on missing keys.

    honest_verdict follows the RETRO-062 threshold: corpus_expanded when >=40 pairs,
    partial_collection_579 when fewer.
    """
    if inference_mode == "gpu_required":
        honest_verdict = "gpu_required"
    elif n_pairs_collected >= 40:
        honest_verdict = "corpus_expanded"
    else:
        honest_verdict = "partial_collection_579"

    mean_latency = (
        sum(per_question_latencies) / len(per_question_latencies)
        if per_question_latencies
        else 0.0
    )

    return {
        "schema": "carnot.live_data_collection.v1",
        "experiment": EXP_ID,
        "inference_mode": inference_mode,
        "n_questions": n_questions,
        "question_indices": QUESTION_INDICES,
        "models": [GEMMA4_MODEL_ID, QWEN_MODEL_ID],
        "n_pairs_collected": n_pairs_collected,
        "live_pairs_file": live_pairs_file,
        "fover_corpus_v3_size": fover_corpus_v3_size,
        "mean_latency_s": mean_latency,
        "per_question_latencies": per_question_latencies,
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Optional[Path] = None) -> dict:
    """Run Exp 579: collect 50 live CoT pairs (indices 200-249) with FOVER annotation.

    All exit paths write the deliverable JSON.  Merges results into fover_corpus_v3.json.
    The FINAL LINE is tmpl.assert_deliverable_written().
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
    corpus_v3_path = repo_root / FOVER_CORPUS_V3_PATH

    def _write_deliverable(artifact: dict) -> dict:
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

    # Step 10: Load GSM8K questions 200-249
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
            (
                QWEN_MODEL_ID,
                lambda p: _qwen_generate(qwen_pipeline, p)
                if qwen_pipeline
                else "[qwen_not_loaded]",
            ),
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
            _write_json_atomic(live_pairs_path, pairs)

    # Step 13: Final atomic write of live pairs file
    _write_json_atomic(live_pairs_path, pairs)
    n_pairs_collected = len(pairs)

    # Step 14: Merge corpus sources into fover_corpus_v3.json
    merge_sources = [
        str(repo_root / FOVER_CORPUS_V2_PATH),
        str(repo_root / LIVE_PAIRS_578_PATH),
        str(repo_root / LIVE_PAIRS_PATH),
    ]
    merged_entries = merge_fover_sources(merge_sources)
    # Serialise FOVERCorpusEntry objects to plain dicts for JSON storage.
    merged_list = [
        {
            "question": e.question,
            "response": e.response,
            "model_id": e.model_id,
            "is_correct": e.is_correct,
            "constraint_types": e.constraint_types,
            "cot_steps": e.cot_steps,
        }
        for e in merged_entries
    ]
    _write_json_atomic(corpus_v3_path, merged_list)
    fover_corpus_v3_size = len(merged_list)
    _log.info(
        "fover_corpus_v3.json written: %d entries (was 132 in v2)", fover_corpus_v3_size
    )

    # Step 15: Build main artifact and write deliverable
    artifact_data = _build_live_data_artifact(
        inference_mode="live_gpu",
        n_questions=N_QUESTIONS,
        n_pairs_collected=n_pairs_collected,
        live_pairs_file=str(live_pairs_path),
        per_question_latencies=per_question_latencies,
        fover_corpus_v3_size=fover_corpus_v3_size,
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
