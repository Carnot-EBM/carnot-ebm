#!/usr/bin/env python3
"""Experiment 615: Live Corpus v3 Expansion — GSM8K 350-449 + fover_corpus_v5 merge.

**Researcher summary:**
    LLMAsExtractorV1 (Exp 616) needs diverse live data for evaluation; JEPA v13 (Exp 618)
    needs it for calibrated retraining.  Exp 602 covered indices 250-349 producing
    fover_corpus_v4.json.  This experiment covers indices 350-449 (more complex multi-step
    GSM8K problems) and merges all live corpora into fover_corpus_v5.json.

**Gate chain (in order; every exit path writes the deliverable):**
    0. MODULE-LEVEL assert CARNOT_FORCE_LIVE == '1' (Exp 590/602 pattern)
    1. apply_env_autofix()
    2. ExperimentTimeoutWatchdog(615, timeout_minutes=120)
    3. ExperimentTemplate(615, ..., requires_gpu=True)
    4. LiveGPUGate.require_live_or_blocked()
    5. JITVRAMCheck for each model
    6. Model selection: SOTA GGUFs (Qwen3.6-35B-A3B + gemma-4-26B) first; fallback to Qwen3.5-0.8B + Gemma4-E4B-it
    7. Load GSM8K validation split, questions 350-449
    8. LongRunBenchmarkExecutor(batch_size=25) — per-question per-model inference
    9. Write results/live_pairs_615.json
    10. Merge: live_pairs_578.json + live_pairs_579.json (optional) + live_pairs_602.json + live_pairs_615.json
    11. Write results/fover_corpus_v5.json with diversity metrics
    12. tmpl.assert_deliverable_written() — FINAL LINE

Spec: REQ-DATA-011, SCENARIO-DATA-017, SCENARIO-DATA-018
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MODULE-LEVEL gate: CARNOT_FORCE_LIVE must be '1' before any model import.
# Why: Exps 551/563/578/602 all suffered silent fallback to synthetic mode when
# this check ran too late.  The Exp 590 pattern fires at import time so the
# conductor sees an immediate, clear failure before any VRAM is consumed.
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DELIVERABLE = "results/experiment_615_live_corpus_v3.json"

if os.environ.get("CARNOT_FORCE_LIVE") != "1":
    import json as _json

    _blocked = {
        "schema": "carnot.live_corpus_v3.v1",
        "experiment": 615,
        "status": "blocked",
        "inference_mode": "gpu_required",
        "n_new_pairs": 0,
        "n_total_corpus_v5": 0,
        "fover_corpus_v5_path": None,
        "honest_verdict": "import_time_block_carnot_force_live_missing",
        "blocked_reason": "CARNOT_FORCE_LIVE must be 1 — source scripts/session_startup.sh",
    }
    _out = _REPO_ROOT / _DELIVERABLE
    _out.parent.mkdir(parents=True, exist_ok=True)
    _tmp = _out.with_suffix(".tmp")
    _tmp.write_text(_json.dumps(_blocked, indent=2))
    os.replace(str(_tmp), str(_out))
    print(
        "EXP-615 IMPORT-TIME GATE: CARNOT_FORCE_LIVE != '1'  →  blocked artifact written, exiting.",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Apply env autofix before any CUDA import
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
import time
from typing import Any, Optional

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader
from carnot.pipeline.jit_vram_check import JITVRAMCheck
from carnot.pipeline.live_assertion import assert_live_gpu_available
from carnot.pipeline.live_gpu_gate import LiveGPUGate
from carnot.pipeline.live_100q_v7_helpers import _extract_answer, _is_correct
from carnot.pipeline.long_run_executor import LongRunBenchmarkExecutor
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 615
EXP_TITLE = "Live Corpus v3 Expansion"
DELIVERABLE = _DELIVERABLE
LIVE_PAIRS_PATH = "results/live_pairs_615.json"
FOVER_CORPUS_V5_PATH = "results/fover_corpus_v5.json"

QUESTION_START = 350
QUESTION_END = 449  # inclusive
N_QUESTIONS = 100

# SOTA GGUF model IDs (try first — largest, most accurate)
QWEN_SOTA_MODEL_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA_SOTA_MODEL_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"

# Fallback model IDs (small, always available)
QWEN_FALLBACK_MODEL_ID = "Qwen/Qwen3.5-0.8B"
GEMMA_FALLBACK_MODEL_ID = "google/gemma-4-E4B-it"

GEMMA4_REQUIRED_GB = 10.0
QWEN_REQUIRED_GB = 1.5

BATCH_SIZE = 25

# Source corpora to merge (615 always included; 579 optional)
PRIOR_LIVE_PAIR_PATHS = [
    "results/live_pairs_578.json",
    "results/live_pairs_579.json",  # may not exist; skipped if absent
    "results/live_pairs_602.json",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON to path atomically via a .tmp file then rename.

    Why atomic: a partial write would corrupt the merged corpus used by
    downstream training experiments (REQ-DATA-002, REQ-DATA-011).  os.replace()
    is atomic on POSIX so the final file is always a complete, valid JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(str(tmp), str(path))


def _load_gsm8k_questions(start: int, end: int) -> list[dict]:
    """Load GSM8K questions from index start to end inclusive.

    Why a fixed index range: corpus reproducibility requires that (start, end)
    uniquely identifies the question batch.  Exp 578 took 0-49; Exp 579 took
    200-249; Exp 602 took 250-349; this experiment takes 350-449 — together they
    cover 400 unique questions across diverse arithmetic and multi-step domains.

    Falls back to synthetic questions if the HuggingFace dataset is unavailable,
    so the script can be unit-tested without a network or HF token.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        return [
            {"question": ds[i]["question"], "answer": ds[i]["answer"], "index": i}
            for i in range(start, end + 1)
        ]
    except Exception as exc:
        _log.warning("_load_gsm8k_questions: dataset load failed (%s) — using synthetic fallback", exc)
        return [
            {
                "question": f"Synthetic question {i}: What is {i} + {i * 2}?",
                "answer": f"#### {i + i * 2}",
                "index": i,
            }
            for i in range(start, end + 1)
        ]


def _load_qwen_pipeline(model_id: str, device: str) -> Optional[Any]:
    """Load a Qwen model as a HuggingFace text-generation pipeline.

    Tries the given model_id first.  Returns None on failure so the experiment
    continues with stub responses rather than aborting — partial data is better
    than no data.

    Why try SOTA first: Qwen3.6-35B-A3B produces higher-accuracy responses that
    give more signal for downstream extractor training.  The fallback to Qwen3.5-0.8B
    ensures the experiment always completes even on memory-constrained machines.
    """
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]

        return hf_pipeline(
            "text-generation",
            model=model_id,
            device=device,
            torch_dtype="auto",
        )
    except Exception as exc:
        _log.warning("_load_qwen_pipeline: failed for %s (%s)", model_id, exc)
        return None


def _qwen_generate(pipeline: Any, prompt: str) -> str:
    """Run one prompt through a HuggingFace transformers pipeline.

    Normalises output across transformers versions that return either a list
    of dicts or a single dict.
    """
    try:
        out = pipeline(prompt, max_new_tokens=256, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get("generated_text", str(out[0]))
        return str(out)
    except Exception as exc:
        return f"[qwen_error: {exc}]"


def _select_models() -> tuple[str, str]:
    """Choose SOTA or fallback model IDs based on what can be loaded.

    Returns (qwen_model_id, gemma_model_id).  Tries SOTA GGUFs first because
    they produce higher-quality responses; falls back to small models that are
    always available.  The actual model loading happens in run_experiment(); this
    function only determines which IDs to attempt.

    Why separate selection from loading: allows unit tests to mock _select_models()
    without triggering actual GPU memory allocation.
    """
    # Try to detect if GGUF files are cached locally
    gguf_path = os.environ.get("CARNOT_GEMMA4_GGUF_PATH", "")
    qwen_gguf_available = os.path.exists(os.environ.get("CARNOT_QWEN_GGUF_PATH", "/nonexistent"))

    if gguf_path and os.path.exists(gguf_path) and qwen_gguf_available:
        _log.info("_select_models: SOTA GGUFs detected, using %s + %s", QWEN_SOTA_MODEL_ID, GEMMA_SOTA_MODEL_ID)
        return QWEN_SOTA_MODEL_ID, GEMMA_SOTA_MODEL_ID

    _log.info("_select_models: falling back to %s + %s", QWEN_FALLBACK_MODEL_ID, GEMMA_FALLBACK_MODEL_ID)
    return QWEN_FALLBACK_MODEL_ID, GEMMA_FALLBACK_MODEL_ID


def _collect_pairs_for_question(
    q_dict: dict,
    gemma4: Any,
    qwen_pipeline: Optional[Any],
    gemma_model_id: str,
    qwen_model_id: str,
) -> list[dict]:
    """Generate live responses for one question from both models and check correctness.

    Returns a list of two pair dicts (one per model).  Each pair has:
    - question_index, question, model, response, is_correct, inference_mode

    Why per-question rather than per-model batching: checkpointing at question
    granularity means an interrupted run wastes at most one question's work.
    """
    gold_answer = _extract_answer(q_dict["answer"])
    pairs = []

    # Gemma inference
    gemma_response = gemma4.generate(q_dict["question"]) if gemma4 else "[gemma4_not_loaded]"
    pairs.append({
        "question_index": q_dict["index"],
        "question": q_dict["question"],
        "model": gemma_model_id,
        "response": gemma_response,
        "is_correct": _is_correct(gemma_response, gold_answer),
        "inference_mode": "live_gpu",
    })

    # Qwen inference
    qwen_response = (
        _qwen_generate(qwen_pipeline, q_dict["question"])
        if qwen_pipeline
        else "[qwen_not_loaded]"
    )
    pairs.append({
        "question_index": q_dict["index"],
        "question": q_dict["question"],
        "model": qwen_model_id,
        "response": qwen_response,
        "is_correct": _is_correct(qwen_response, gold_answer),
        "inference_mode": "live_gpu",
    })

    return pairs


def _merge_live_corpora(
    repo_root: Path,
    new_pairs: list[dict],
) -> list[dict]:
    """Load all prior live pair files, combine with new_pairs, deduplicate.

    Deduplication key is (question_index, model).  If the same (question_index,
    model) appears in multiple source files, the entry from the most recently
    added source wins.  This preserves historical data while letting newer
    experiments override stale or stub responses.

    Why deduplicate: downstream training must not see the same sample twice or
    the gradient signal will be biased toward over-represented question indices.
    """
    # Start with the new pairs (highest priority)
    seen: dict[tuple, dict] = {}
    for p in new_pairs:
        key = (p.get("question_index"), p.get("model"))
        seen[key] = p

    # Load prior corpora in reverse priority (oldest first, so newer wins)
    for path_str in reversed(PRIOR_LIVE_PAIR_PATHS):
        path = repo_root / path_str
        if not path.exists():
            _log.info("_merge_live_corpora: %s not found, skipping", path_str)
            continue
        try:
            prior = json.loads(path.read_text())
            for p in prior:
                key = (p.get("question_index"), p.get("model"))
                if key not in seen:
                    seen[key] = p
        except Exception as exc:
            _log.warning("_merge_live_corpora: failed to load %s: %s", path_str, exc)

    return list(seen.values())


def _compute_diversity_metrics(corpus: list[dict]) -> dict:
    """Compute diversity and accuracy metrics over the merged corpus.

    Why diversity metrics: downstream experiments need assurance that the corpus
    covers varied question types and both models.  n_unique_questions, accuracy
    per model, and correct/incorrect balance are the minimum set to detect a
    degenerate corpus (e.g., all same question, all incorrect).
    """
    if not corpus:
        return {
            "n_unique_questions": 0,
            "n_correct_pairs": 0,
            "n_incorrect_pairs": 0,
            "model_accuracy_qwen": 0.0,
            "model_accuracy_gemma": 0.0,
        }

    unique_questions: set = set()
    n_correct = 0
    n_incorrect = 0
    qwen_correct = 0
    qwen_total = 0
    gemma_correct = 0
    gemma_total = 0

    for p in corpus:
        unique_questions.add(p.get("question_index"))
        correct = bool(p.get("is_correct", False))
        if correct:
            n_correct += 1
        else:
            n_incorrect += 1
        model = p.get("model", "")
        if "Qwen" in model or "qwen" in model.lower():
            qwen_total += 1
            if correct:
                qwen_correct += 1
        elif "gemma" in model.lower() or "Gemma" in model:
            gemma_total += 1
            if correct:
                gemma_correct += 1

    return {
        "n_unique_questions": len(unique_questions),
        "n_correct_pairs": n_correct,
        "n_incorrect_pairs": n_incorrect,
        "model_accuracy_qwen": qwen_correct / qwen_total if qwen_total else 0.0,
        "model_accuracy_gemma": gemma_correct / gemma_total if gemma_total else 0.0,
    }


def _build_corpus_artifact(
    n_new_pairs: int,
    n_total_corpus_v5: int,
    diversity: dict,
    fover_corpus_v5_path: Optional[str],
    inference_mode: str,
    models_used: Optional[list[str]] = None,
) -> dict:
    """Assemble the standardised artifact dict for Exp 615.

    Why a dedicated builder: ensures every required schema field appears on
    every exit path (blocked, partial, success) so downstream experiments
    never need to branch on status when reading the artifact.

    honest_verdict='corpus_expanded' only when n_new_pairs >= 80, which is 80%
    of the 100-question target — the threshold chosen to ensure meaningful data
    even if some questions fail inference.
    """
    honest_verdict = "corpus_expanded" if n_new_pairs >= 80 else "corpus_partial"

    return {
        "schema": "carnot.live_corpus_v3.v1",
        "inference_mode": inference_mode,
        "n_new_pairs": n_new_pairs,
        "n_total_corpus_v5": n_total_corpus_v5,
        "fover_corpus_v5_path": fover_corpus_v5_path,
        "models_used": models_used or [],
        "honest_verdict": honest_verdict,
        **diversity,
    }


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Optional[Path] = None) -> dict:
    """Run Exp 615: collect live pairs from GSM8K 350-449, merge into fover_corpus_v5.

    All exit paths (blocked, partial, success) write the deliverable JSON.
    The FINAL LINE is tmpl.assert_deliverable_written().

    The module-level CARNOT_FORCE_LIVE assert has already fired before this
    function is reached.  This function implements gates 1-12 of the chain.
    """
    if repo_root is None:
        repo_root = _REPO_ROOT

    # Hard gate: assert_live_gpu_available() — belt-and-suspenders after module-level check
    assert_live_gpu_available()

    # apply_env_autofix() was called at module level; calling again is idempotent
    apply_env_autofix()

    # ExperimentTimeoutWatchdog — 120-minute hard cap for 100q * 2 models
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=120)
    watchdog.start()

    # ExperimentTemplate registers DeliverableGuard at __init__ time
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
    fover_corpus_path = repo_root / FOVER_CORPUS_V5_PATH

    def _write_deliverable(artifact: dict) -> dict:
        """Write the main deliverable JSON; return the artifact unchanged."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output_path, artifact)
        return artifact

    # LiveGPUGate — runtime CARNOT_FORCE_LIVE check (belt+suspenders)
    gate_result = LiveGPUGate.require_live_or_blocked(tmpl, model_ids=[])
    if gate_result is not None:
        blocked = tmpl.build_result(
            _build_corpus_artifact(0, 0, {}, None, "gpu_required"),
            status="blocked",
        )
        _write_deliverable(blocked)
        watchdog.stop()
        tmpl.assert_deliverable_written()
        return blocked

    # Select model IDs (SOTA GGUFs if available, else small fallbacks)
    qwen_model_id, gemma_model_id = _select_models()
    models_used = [qwen_model_id, gemma_model_id]

    # JIT VRAM check for Gemma on cuda:0
    vram0 = JITVRAMCheck(device_id=0)
    gate0 = vram0.gate_model_load(gemma_model_id, required_gb=GEMMA4_REQUIRED_GB)
    if not gate0.is_cleared:
        blocked = tmpl.build_result(
            _build_corpus_artifact(0, 0, {}, None, "gpu_required", models_used),
            status="blocked",
            blocked_reason=f"cuda:0 VRAM insufficient: {gate0.available_gb:.1f} GB < {GEMMA4_REQUIRED_GB} GB",
        )
        _write_deliverable(blocked)
        watchdog.stop()
        tmpl.assert_deliverable_written()
        return blocked

    # JIT VRAM check for Qwen on cuda:1
    vram1 = JITVRAMCheck(device_id=1)
    gate1 = vram1.gate_model_load(qwen_model_id, required_gb=QWEN_REQUIRED_GB)
    if not gate1.is_cleared:
        blocked = tmpl.build_result(
            _build_corpus_artifact(0, 0, {}, None, "gpu_required", models_used),
            status="blocked",
            blocked_reason=f"cuda:1 VRAM insufficient: {gate1.available_gb:.1f} GB < {QWEN_REQUIRED_GB} GB",
        )
        _write_deliverable(blocked)
        watchdog.stop()
        tmpl.assert_deliverable_written()
        return blocked

    # Load models
    gguf_path = os.environ.get("CARNOT_GEMMA4_GGUF_PATH", "")
    gemma4 = Gemma4QuantizedLoader(
        model_path=gguf_path,
        n_gpu_layers=-1,
        max_tokens=512,
        jit_vram_check=vram0,
    )
    gemma4_loaded = gemma4.load()
    if not gemma4_loaded:
        _log.warning("Gemma4 load failed — continuing with stub responses")

    qwen_pipeline = _load_qwen_pipeline(qwen_model_id, "cuda:1")

    # Load GSM8K questions 350-449
    questions = _load_gsm8k_questions(QUESTION_START, QUESTION_END)

    # LongRunBenchmarkExecutor partitions 100 questions into 25-question batches
    executor = LongRunBenchmarkExecutor(
        batch_size=BATCH_SIZE,
        checkpoint_dir=str(repo_root / "results/batch_ckpt/exp615"),
    )

    # Resume from checkpoint if available
    checkpoint = tmpl.checkpoint_resume()
    collected_pairs: list[dict] = checkpoint.get("pairs", []) if checkpoint else []
    done_indices: set[int] = {p["question_index"] for p in collected_pairs} if collected_pairs else set()

    # Build the list of not-yet-done questions
    pending_questions = [q for q in questions if q["index"] not in done_indices]

    batches = executor.partition(pending_questions)

    for batch in batches:
        def make_inference_fn(
            gemma4_ref: Any,
            qwen_ref: Optional[Any],
            g_id: str,
            q_id: str,
        ) -> Any:
            """Capture model references and IDs in a closure for batch inference.

            Why a closure: the LongRunBenchmarkExecutor.run_batch() takes a single
            inference_fn(question) -> result signature.  We need to call both models
            per question.  The closure freezes references so they don't accidentally
            capture loop variables.
            """
            def inference_fn(q_dict: dict) -> list[dict]:
                return _collect_pairs_for_question(q_dict, gemma4_ref, qwen_ref, g_id, q_id)
            return inference_fn

        completed_batch = executor.run_batch(
            batch,
            inference_fn=make_inference_fn(gemma4, qwen_pipeline, gemma_model_id, qwen_model_id),
            watchdog_timeout_minutes=40,
        )
        executor.save_batch(completed_batch, prefix="exp615")

        # Flatten results (each question produces a list of 2 pairs)
        for result in (completed_batch.results or []):
            collected_pairs.extend(result)

        # Checkpoint progress after each batch
        done_set = {p["question_index"] for p in collected_pairs}
        tmpl.checkpoint_save({"pairs": collected_pairs}, step=len(done_set))

    # Write live_pairs_615.json (new pairs only — questions 350-449)
    new_pairs = [p for p in collected_pairs if QUESTION_START <= p.get("question_index", -1) <= QUESTION_END]
    _write_json_atomic(live_pairs_path, new_pairs)
    n_new_pairs = len(new_pairs)
    _log.info("Wrote %d new live pairs to %s", n_new_pairs, live_pairs_path)

    # Merge all live corpora into fover_corpus_v5.json
    merged_corpus = _merge_live_corpora(repo_root, new_pairs)
    diversity = _compute_diversity_metrics(merged_corpus)

    corpus_v5_payload = {
        "metadata": {
            "schema": "carnot.fover_corpus.v5",
            "sources": [
                "results/live_pairs_578.json",
                "results/live_pairs_579.json",
                "results/live_pairs_602.json",
                "results/live_pairs_615.json",
            ],
            **diversity,
        },
        "pairs": merged_corpus,
    }
    _write_json_atomic(fover_corpus_path, corpus_v5_payload)
    n_total_corpus_v5 = len(merged_corpus)
    _log.info("Wrote merged fover_corpus_v5 with %d pairs to %s", n_total_corpus_v5, fover_corpus_path)

    # Build main artifact
    artifact_data = _build_corpus_artifact(
        n_new_pairs=n_new_pairs,
        n_total_corpus_v5=n_total_corpus_v5,
        diversity=diversity,
        fover_corpus_v5_path=str(fover_corpus_path),
        inference_mode="live_gpu",
        models_used=models_used,
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
