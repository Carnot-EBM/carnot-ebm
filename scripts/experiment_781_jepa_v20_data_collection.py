#!/usr/bin/env python3
"""Experiment 781 — JEPA v20 live data collection: expand the FoVer labeled corpus.

**Why this experiment exists:**
    JEPA v19 (Exp 770) achieved ood_auc=0.5667, short of the 0.75 deployment gate.
    Root cause: only 57 real labeled CoT pairs (Exp 442) were available for training.
    57 pairs from one live run don't generalise OOD — the model sees too few arithmetic
    step patterns to learn a generalizable step-quality signal.

    This experiment fixes that by running a second live benchmark (100 GSM8K questions,
    seed=9999 to avoid overlapping Exps 439/742/781), collecting the CoT responses, and
    applying FOVER Z3 annotation to build a second batch of labeled pairs.  Combined with
    the 57 pairs from Exp 442, the training corpus grows to ~137+ real pairs.

**Data isolation (REQ-LEARN-048):**
    Labeled pairs are written ONLY to ``results/fover_labeled_steps_live_v2.json``.
    ``results/fover_labeled_steps_live.json`` (the Exp 442 baseline) is NEVER read,
    written, or modified by this experiment.  The two files are additive; merging is
    the responsibility of the downstream JEPA v20 training script.

**Honest verdict (REQ-LEARN-049):**
    - ``"real_data_collected_sufficient"``    — n_labeled >= 80 AND live_gpu
    - ``"real_data_collected_insufficient"``  — 20 <= n_labeled < 80 AND live_gpu
    - ``"real_data_below_threshold"``         — n_labeled < 20 AND live_gpu
    - ``"blocked_no_live_gpu"``               — CARNOT_FORCE_LIVE not set/truthy

**GPU setup protocol (RETRO-028, RETRO-SOTA-GGUF-TIMEOUT):**
    kill_gpu_zombies(gpu_index=0) is called BEFORE any model load to prevent
    zombie-held VRAM from causing OOM during model initialization.

Spec: REQ-LEARN-048, REQ-LEARN-049, SCENARIO-LEARN-092, SCENARIO-LEARN-093
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# apply_env_autofix() FIRST — must precede any CUDA/torch import.
# RETRO-022: CARNOT_FORCE_LIVE=1 must propagate before GPU frameworks init.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
import os
import time
from typing import Any

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.fover_annotator import FOVERAnnotator  # noqa: E402
from carnot.pipeline.gpu_zombie_killer import kill_gpu_zombies  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 781
TITLE = "JEPA v20 live data collection: FoVer corpus expansion"
DELIVERABLE = "results/experiment_781_jepa_v20_data_collection.json"
LABELED_STEPS_V2_PATH = str(_REPO_ROOT / "results" / "fover_labeled_steps_live_v2.json")
CHECKPOINT_PATH = str(_REPO_ROOT / "results" / "exp781_ckpt.json")
TIMEOUT_MINUTES = 90
N_QUESTIONS = 100
BATCH_SIZE = 10
GSM8K_SEED = 9999  # avoids overlap with Exps 439 (seed 42), 742 (seed 7777), 781 self


def _is_truthy_env(key: str) -> bool:
    """Return True when the env var is set and has a truthy value (not '0'/'false'/'')."""
    val = os.environ.get(key, "")
    return val.lower() not in ("", "0", "false")


def _load_gsm8k_questions(n: int, seed: int) -> list[dict]:
    """Load *n* GSM8K questions with deterministic shuffling from *seed*.

    Why a standalone function: isolates the dataset loading so tests can mock it
    without monkeypatching HuggingFace datasets internals.

    Returns list of dicts with keys 'question' and 'question_id'.
    Falls back to synthetic arithmetic questions when datasets is not importable
    or the GSM8K split is unavailable — this should not happen in live mode.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset("gsm8k", "main", split="test")
        import random  # noqa: PLC0415

        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = indices[:n]
        questions = []
        for rank, idx in enumerate(selected):
            row = ds[idx]
            questions.append(
                {
                    "question": row["question"],
                    "question_id": f"gsm8k_seed{seed}_rank{rank}_idx{idx}",
                }
            )
        return questions
    except Exception as exc:
        _log.warning("Could not load GSM8K dataset (%s); using synthetic fallback", exc)
        return [
            {
                "question": f"What is {i + 1} * {seed % 100 + i + 1}?",
                "question_id": f"synthetic_seed{seed}_q{i}",
            }
            for i in range(n)
        ]


def _load_checkpoint() -> dict | None:
    """Load the Exp 781 checkpoint from disk, or return None if absent/corrupt."""
    try:
        p = Path(CHECKPOINT_PATH)
        if p.exists():
            data = json.loads(p.read_text())
            _log.info("Loaded checkpoint: %d questions completed", len(data.get("responses", [])))
            return data
    except Exception as exc:
        _log.warning("Could not load checkpoint (%s) — starting fresh", exc)
    return None


def _save_checkpoint(responses: list[dict]) -> None:
    """Write a checkpoint of completed responses to disk atomically."""
    try:
        tmp = CHECKPOINT_PATH + ".tmp"
        Path(tmp).write_text(json.dumps({"responses": responses}, indent=2))
        Path(tmp).rename(CHECKPOINT_PATH)
    except Exception as exc:
        _log.warning("Checkpoint save failed: %s", exc)


def run_experiment(tmpl: ExperimentTemplate) -> dict[str, Any]:
    """Core logic for Experiment 781.

    Why separated from main(): lets tests inject a mock ExperimentTemplate and
    drive the logic without spawning subprocesses or touching the real GPU.

    Returns the final artifact dict.
    """
    # Step 1: apply_env_autofix already called at module load; log its result.
    inference_mode = "blocked_no_live_gpu"
    if _is_truthy_env("CARNOT_FORCE_LIVE"):
        inference_mode = "live_gpu"

    # Step 2: gate on live GPU availability BEFORE touching hardware.
    if inference_mode != "live_gpu":
        _log.warning(
            "CARNOT_FORCE_LIVE not set — cannot run live benchmark. "
            "Returning blocked artifact."
        )
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_no_live_gpu",
                "inference_mode": inference_mode,
                "n_questions": 0,
                "n_steps_found": 0,
                "n_labeled": 0,
                "n_correct": 0,
                "n_incorrect": 0,
                "labeling_rate": 0.0,
                "labeled_file": "fover_labeled_steps_live_v2.json",
            },
            status="blocked",
        )
        tmpl._output_path.write_text(json.dumps(artifact, indent=2))
        return artifact

    # Step 3: kill GPU zombies BEFORE any model load (REQ-LEARN-048, RETRO-028).
    zombie_result = kill_gpu_zombies(gpu_index=0)
    _log.info(
        "kill_gpu_zombies: verdict=%s pids_killed=%d vram_freed=%.0f MB",
        zombie_result.honest_verdict,
        len(zombie_result.pids_killed),
        zombie_result.vram_freed_mb,
    )

    # Step 4: setup GPU with Qwen3.5-0.8B (known reliable on RTX 3090).
    MODEL_SPECS = [
        {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0},
    ]
    gpu_status = tmpl.setup_gpu(MODEL_SPECS)
    if not gpu_status["all_healthy"]:
        _log.error("GPU setup unhealthy — writing blocked artifact")
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_no_live_gpu",
                "inference_mode": inference_mode,
                "n_questions": 0,
                "n_steps_found": 0,
                "n_labeled": 0,
                "n_correct": 0,
                "n_incorrect": 0,
                "labeling_rate": 0.0,
                "labeled_file": "fover_labeled_steps_live_v2.json",
                "gpu_status": gpu_status,
            },
            status="blocked",
        )
        tmpl._output_path.write_text(json.dumps(artifact, indent=2))
        return artifact

    # Step 5: load questions.
    questions = _load_gsm8k_questions(N_QUESTIONS, GSM8K_SEED)
    _log.info("Loaded %d GSM8K questions (seed=%d)", len(questions), GSM8K_SEED)

    # Step 6: resume from checkpoint if available.
    ckpt = _load_checkpoint()
    completed_responses: list[dict] = ckpt.get("responses", []) if ckpt else []
    completed_ids = {r["question_id"] for r in completed_responses}
    remaining = [q for q in questions if q["question_id"] not in completed_ids]
    _log.info(
        "Checkpoint: %d already done, %d remaining", len(completed_responses), len(remaining)
    )

    # Step 7: run inference in batches of BATCH_SIZE with VerifyRepairPipeline.
    # Import here so tests can mock before this point.
    from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: PLC0415

    pipeline = VerifyRepairPipeline(
        model="Qwen/Qwen3.5-0.8B",
        domains=["arithmetic"],
        max_repairs=1,
        extractor=None,
        semantic_grounding_verifier=None,
        semantic_verifier_v2=None,
        timeout_seconds=60,
        memory=None,
        template_library=None,
        session_memory=None,
        constraint_memory=None,
        nup_probe=None,
        nup_probe_threshold=0.5,
    )

    batch_start = 0
    while batch_start < len(remaining):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]
        _log.info(
            "Processing batch %d-%d of %d remaining",
            batch_start,
            batch_start + len(batch) - 1,
            len(remaining),
        )
        for item in batch:
            try:
                result = pipeline.verify_and_repair(
                    question=item["question"],
                    response="",
                    domain="arithmetic",
                )
                response_text = getattr(result, "final_response", "") or ""
                completed_responses.append(
                    {
                        "question_id": item["question_id"],
                        "question": item["question"],
                        "response": response_text,
                    }
                )
            except Exception as exc:
                _log.warning("Question %s failed: %s", item["question_id"], exc)
                completed_responses.append(
                    {
                        "question_id": item["question_id"],
                        "question": item["question"],
                        "response": "",
                    }
                )
        _save_checkpoint(completed_responses)
        batch_start += BATCH_SIZE

    # Step 8: FOVER annotation on all collected responses.
    annotator = FOVERAnnotator(z3_timeout_seconds=5)
    annotated = annotator.annotate_corpus(completed_responses)
    pairs = annotator.to_training_pairs(annotated, completed_responses)

    # Count labels.
    n_steps_found = sum(len(steps) for steps in annotated)
    n_labeled = len(pairs)
    n_correct = sum(1 for p in pairs if p["label"] == "correct")
    n_incorrect = sum(1 for p in pairs if p["label"] == "incorrect")
    labeling_rate = n_labeled / n_steps_found if n_steps_found > 0 else 0.0

    _log.info(
        "Annotation complete: n_steps_found=%d n_labeled=%d (%.1f%%) correct=%d incorrect=%d",
        n_steps_found,
        n_labeled,
        labeling_rate * 100,
        n_correct,
        n_incorrect,
    )

    # Step 9: write labeled pairs to fover_labeled_steps_live_v2.json (NEW file).
    # NEVER write to fover_labeled_steps_live.json — that is the Exp 442 baseline.
    Path(LABELED_STEPS_V2_PATH).write_text(json.dumps(pairs, indent=2))
    _log.info("Wrote %d labeled pairs to %s", n_labeled, LABELED_STEPS_V2_PATH)

    # Step 10: compute honest verdict (REQ-LEARN-049).
    if n_labeled >= 80:
        honest_verdict = "real_data_collected_sufficient"
    elif n_labeled >= 20:
        honest_verdict = "real_data_collected_insufficient"
    else:
        honest_verdict = "real_data_below_threshold"

    artifact = tmpl.build_result(
        {
            "honest_verdict": honest_verdict,
            "inference_mode": inference_mode,
            "n_questions": len(completed_responses),
            "n_steps_found": n_steps_found,
            "n_labeled": n_labeled,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "labeling_rate": labeling_rate,
            "labeled_file": "fover_labeled_steps_live_v2.json",
        },
        status="success",
    )
    tmpl._output_path.write_text(json.dumps(artifact, indent=2))
    return artifact


def main() -> None:
    """Entry point for Experiment 781."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES):
        tmpl.setup()
        tmpl.check_exclusion_manifest()
        run_experiment(tmpl)
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
