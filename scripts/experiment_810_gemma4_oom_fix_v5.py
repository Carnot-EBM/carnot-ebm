#!/usr/bin/env python3
"""Experiment 810: Gemma4 OOM Fix v5 — nvidia-smi Verification Loop (RETRO-028 Closure).

**Researcher summary:**
    RETRO-028 remains unresolved after Exp 450, 787, 795.  Fix v4 (Exp 795)
    applied kill_gpu_zombies() and a single pkill pass but did NOT use a
    verification loop.  If VRAM doesn't clear in one pass (other processes
    restart or hold VRAM), the model load proceeds into CUDA OOM.

    Fix v5 mandates:
    1. kill_gpu_zombies() — primary SIGKILL pass (inside evict_vram_with_loop).
    2. Retry loop: for each retry up to 3x, SIGKILL processes using >100 MB,
       sleep 10 s, read VRAM via nvidia-smi.  Exit early if VRAM < 500 MB.
    3. ABORT if VRAM does not drop below 500 MB after 3 retries — do NOT
       attempt model load.  Write a blocked_vram_stuck artifact and exit.

**honest_verdict values:**
    - "retro_028_closed"             — n_valid_responses >= 16 (80% of 20)
    - "partial_success"              — n_valid_responses in [8, 15]
    - "model_loaded_no_valid_output" — model loaded but n_valid < 8
    - "blocked_vram_stuck"           — eviction loop exhausted all retries
    - "blocked_model_load_failed"    — model load raised OOM or other error
    - "blocked_no_live_gpu"          — CARNOT_FORCE_LIVE not set

Spec: REQ-LOADER-014, SCENARIO-LOADER-014, SCENARIO-LOADER-015
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo root wiring (before relative imports)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: E402
from carnot.pipeline.vram_loop_eviction import evict_vram_with_loop  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 810
EXP_TITLE = "Gemma4 OOM Fix v5 — nvidia-smi Verification Loop (RETRO-028 Closure)"
DELIVERABLE = "results/experiment_810_gemma4_oom_fix_v5.json"
CHECKPOINT_PATH = "results/exp810_ckpt.json"
TIMEOUT_MINUTES = 90
GPU_INDEX = 1  # REQ-LOADER-013: cooler GPU
MODEL_ID = "google/gemma-4-E4B-it"
N_QUESTIONS = 20
SEED = 810
VRAM_MAX_RETRIES = 3
VRAM_RETRY_SLEEP_S = 10.0
VRAM_THRESHOLD_MB = 500.0


# ---------------------------------------------------------------------------
# GSM8K loader
# ---------------------------------------------------------------------------


def _load_gsm8k_questions(n: int, seed: int) -> list[dict[str, Any]]:
    """Load *n* GSM8K test questions, seeded for reproducibility.

    Falls back to synthetic arithmetic when HuggingFace datasets is unavailable
    (CI environments without internet) so the experiment still produces an honest
    artifact rather than crashing on import.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]  # noqa: PLC0415

        ds = load_dataset("gsm8k", "main", split="test")
        indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        selected = indices[:n]
        return [{"question": ds[i]["question"], "answer": ds[i]["answer"]} for i in selected]
    except Exception as exc:
        _log.warning("Could not load GSM8K (%s) — using synthetic fallback", exc)
        rng = random.Random(seed)
        return [
            {
                "question": f"What is {rng.randint(1, 50)} + {rng.randint(1, 50)}?",
                "answer": "synthetic",
            }
            for _ in range(n)
        ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Experiment 810 entry point — VRAM loop eviction + Gemma4 inference.

    Steps:
    1. apply_env_autofix() — set JAX_PLATFORMS=cpu before any JAX import.
    2. ExperimentTemplate(810) + ExperimentTimeoutWatchdog(810, 90 min).
    3. LiveGPUGate: hard gate via CARNOT_FORCE_LIVE env var.
    4. evict_vram_with_loop(gpu_index=1, max_retries=3, retry_sleep_s=10, threshold_mb=500).
       - If not vram_cleared: write blocked_vram_stuck artifact and exit.
       - Checkpoint step 1 result.
    5. load_gemma4_on_gpu1() — allocate model on cuda:1.
       - Checkpoint step 2 result.
       - If load fails: write blocked_model_load_failed artifact and exit.
    6. Run 20 GSM8K questions; checkpoint every 5 questions.
    7. Compute honest_verdict based on n_valid_responses threshold.
    8. Write final artifact and assert_deliverable_written().
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Step 1: apply_env_autofix must run FIRST before any JAX import to avoid
    # the ROCm thrml crash (see extropic-ai/thrml#41).
    try:
        apply_env_autofix()
    except Exception as exc:
        _log.warning("apply_env_autofix raised %s — continuing", exc)

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=TIMEOUT_MINUTES,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )

    ckpt_writer = AtomicResultWriter(str(_REPO_ROOT / CHECKPOINT_PATH))

    with watchdog:
        tmpl.check_exclusion_manifest()

        # Step 3: LiveGPUGate — hard gate.  If CARNOT_FORCE_LIVE is not set,
        # write a blocked artifact immediately and exit without running inference.
        if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
            _log.warning("CARNOT_FORCE_LIVE not set — writing blocked_no_live_gpu artifact")
            artifact = tmpl.build_result(
                {
                    "step1_vram_mb_per_retry": [],
                    "step1_vram_cleared": False,
                    "step2_model_loaded": False,
                    "step3_n_valid_responses": 0,
                    "retro_028_closed": False,
                    "honest_verdict": "blocked_no_live_gpu",
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # Step 4: evict_vram_with_loop — retry loop with nvidia-smi verification.
        # This is the Fix v5 protocol: kill zombies, then loop up to 3x checking
        # that VRAM actually drops below 500 MB before proceeding.
        _log.info(
            "Step 4: evict_vram_with_loop(gpu=%d, max_retries=%d, sleep=%.0fs, threshold=%.0fMB)",
            GPU_INDEX,
            VRAM_MAX_RETRIES,
            VRAM_RETRY_SLEEP_S,
            VRAM_THRESHOLD_MB,
        )
        eviction = evict_vram_with_loop(
            gpu_index=GPU_INDEX,
            max_retries=VRAM_MAX_RETRIES,
            retry_sleep_s=VRAM_RETRY_SLEEP_S,
            threshold_mb=VRAM_THRESHOLD_MB,
        )
        _log.info(
            "Step 4 result: vram_cleared=%s retries=%d vram_per_retry=%s final=%.0fMB verdict=%s",
            eviction.vram_cleared,
            eviction.n_retries_attempted,
            eviction.vram_mb_per_retry,
            eviction.final_vram_mb,
            eviction.honest_verdict,
        )

        # Checkpoint after eviction so partial results survive timeout.
        ckpt_writer.write(
            {
                "step": "eviction_done",
                "step1_vram_cleared": eviction.vram_cleared,
                "step1_vram_mb_per_retry": eviction.vram_mb_per_retry,
                "step1_final_vram_mb": eviction.final_vram_mb,
                "step1_n_retries": eviction.n_retries_attempted,
            }
        )

        if not eviction.vram_cleared:
            _log.error(
                "VRAM not cleared after %d retries (final=%.0f MB >= %.0f MB) — aborting",
                eviction.n_retries_attempted,
                eviction.final_vram_mb,
                VRAM_THRESHOLD_MB,
            )
            artifact = tmpl.build_result(
                {
                    "step1_vram_mb_per_retry": eviction.vram_mb_per_retry,
                    "step1_vram_cleared": False,
                    "step1_n_retries_attempted": eviction.n_retries_attempted,
                    "step1_final_vram_mb": eviction.final_vram_mb,
                    "step1_abort_reason": eviction.abort_reason,
                    "step2_model_loaded": False,
                    "step3_n_valid_responses": 0,
                    "retro_028_closed": False,
                    "honest_verdict": "blocked_vram_stuck",
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # Step 5: load Gemma4 on GPU 1.
        # We import load_gemma4_on_gpu1 here rather than at module level so that
        # apply_env_autofix() has already run before any transformers import.
        from carnot.pipeline.gemma_isolation import load_gemma4_on_gpu1  # noqa: PLC0415

        _log.info("Step 5: load_gemma4_on_gpu1(model_id=%s)", MODEL_ID)
        load_result = load_gemma4_on_gpu1(model_id=MODEL_ID)
        step2_model_loaded = load_result.get("loaded", False)
        _log.info(
            "Step 5 result: loaded=%s device=%s reason=%s",
            step2_model_loaded,
            load_result.get("device"),
            load_result.get("reason"),
        )

        ckpt_writer.write(
            {
                "step": "model_load_done",
                "step1_vram_cleared": True,
                "step2_model_loaded": step2_model_loaded,
                "step2_load_reason": load_result.get("reason"),
            }
        )

        if not step2_model_loaded:
            artifact = tmpl.build_result(
                {
                    "step1_vram_mb_per_retry": eviction.vram_mb_per_retry,
                    "step1_vram_cleared": True,
                    "step1_n_retries_attempted": eviction.n_retries_attempted,
                    "step1_final_vram_mb": eviction.final_vram_mb,
                    "step2_model_loaded": False,
                    "step2_load_failure_reason": load_result.get("reason", "unknown"),
                    "step3_n_valid_responses": 0,
                    "retro_028_closed": False,
                    "honest_verdict": "blocked_model_load_failed",
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # Step 6: run N_QUESTIONS GSM8K questions and count valid responses.
        loader: GemmaTransformersLoader = load_result["loader"]
        questions = _load_gsm8k_questions(N_QUESTIONS, SEED)
        _log.info(
            "Step 6: running %d GSM8K questions on Gemma4 (cuda:%d)", len(questions), GPU_INDEX
        )

        responses: list[str] = []
        n_valid_responses = 0

        for i, q in enumerate(questions):
            try:
                resp = loader.generate(q["question"], max_new_tokens=256)
            except Exception as exc:
                _log.warning("Question %d generation failed: %s", i, exc)
                resp = ""
            is_valid = GemmaTransformersLoader.is_valid_output(resp)
            responses.append(resp)
            if is_valid:
                n_valid_responses += 1
            _log.info(
                "Question %d/%d: valid=%s resp_len=%d",
                i + 1,
                len(questions),
                is_valid,
                len(resp),
            )
            # Checkpoint every 5 questions so results survive a timeout.
            if (i + 1) % 5 == 0:
                tmpl.checkpoint_save(
                    {
                        "responses_so_far": responses,
                        "n_valid_so_far": n_valid_responses,
                        "questions_done": i + 1,
                    },
                    step=i + 1,
                )

        # Step 7: compute honest_verdict based on Fix v5 thresholds.
        # 80% of 20 questions = 16 valid responses to declare RETRO-028 closed.
        if n_valid_responses >= 16:
            honest_verdict = "retro_028_closed"
        elif n_valid_responses >= 8:
            honest_verdict = "partial_success"
        else:
            honest_verdict = "model_loaded_no_valid_output"

        retro_028_closed = honest_verdict == "retro_028_closed"

        _log.info(
            "Experiment 810 complete: n_valid=%d/%d verdict=%s retro_028_closed=%s",
            n_valid_responses,
            N_QUESTIONS,
            honest_verdict,
            retro_028_closed,
        )

        # Step 8: write final artifact.
        artifact = tmpl.build_result(
            {
                "step1_vram_mb_per_retry": eviction.vram_mb_per_retry,
                "step1_vram_cleared": eviction.vram_cleared,
                "step1_n_retries_attempted": eviction.n_retries_attempted,
                "step1_final_vram_mb": eviction.final_vram_mb,
                "step2_model_loaded": step2_model_loaded,
                "step3_n_valid_responses": n_valid_responses,
                "n_questions": N_QUESTIONS,
                "gpu_index": GPU_INDEX,
                "model_id": MODEL_ID,
                "retro_028_closed": retro_028_closed,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )
        (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
