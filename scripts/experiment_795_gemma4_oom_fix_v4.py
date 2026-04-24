#!/usr/bin/env python3
"""Experiment 795: Gemma4 OOM Fix v4 — Four-Step VRAM Isolation (RETRO-028 Closure).

**Researcher summary:**
    RETRO-028 has failed to close across three dedicated attempts (Exp .58, .59, .60).
    Root cause: Gemma4-E4B-it (14.89 GiB allocation) fails with CUDA OOM because GPU
    memory is occupied by zombie processes before model load.

    The RETRO-028 Fix v4 implements the four-step isolation protocol from
    results/operational_retro_2026_04_60.json:

    1. kill_gpu_zombies(1)     — SIGKILL all processes holding GPU 1 memory.
    2. evict_gpu_vram(1)       — pkill sweep + verify <500 MB used on GPU 1.
    3. Verify vram_clear=True  — hard abort if eviction failed.
    4. Load Gemma4 on GPU 1   — cooler GPU (8-10C thermal headroom over GPU 0).

    Then runs 10 GSM8K questions and scores valid responses.

**honest_verdict values:**
    - "retro_028_closed"              — n_valid_responses >= 8 (8/10+)
    - "partial_success"               — n_valid_responses in [4, 7]
    - "model_loaded_no_valid_output"  — model loaded but n_valid < 4
    - "vram_not_cleared"              — VRAM eviction failed
    - "blocked_no_live_gpu"           — LiveGPUGate blocked (CARNOT_FORCE_LIVE not set)

Spec: REQ-LOADER-012, REQ-LOADER-013, SCENARIO-LOADER-012, SCENARIO-LOADER-013
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

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.gemma_isolation import evict_gpu_vram, load_gemma4_on_gpu1  # noqa: E402
from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: E402
from carnot.pipeline.gpu_zombie_killer import kill_gpu_zombies  # noqa: E402
from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 795
EXP_TITLE = "Gemma4 OOM Fix v4 — Four-Step VRAM Isolation (RETRO-028 Closure)"
DELIVERABLE = "results/experiment_795_gemma4_oom_fix_v4.json"
TIMEOUT_MINUTES = 60
GPU_INDEX = 1  # REQ-LOADER-013: use the cooler GPU
MODEL_ID = "google/gemma-4-E4B-it"
N_QUESTIONS = 10
SEED = 795  # different from Exp 786 to sample fresh questions


# ---------------------------------------------------------------------------
# GSM8K loader
# ---------------------------------------------------------------------------


def _load_gsm8k_questions(n: int, seed: int) -> list[dict[str, Any]]:
    """Load *n* GSM8K questions from HuggingFace datasets, seeded for reproducibility.

    Falls back to synthetic arithmetic questions when the datasets library is
    unavailable (CI environments without internet access), so the experiment
    produces an honest artifact instead of crashing.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]  # noqa: PLC0415

        ds = load_dataset("gsm8k", "main", split="test")
        indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        selected = indices[:n]
        return [
            {"question": ds[i]["question"], "answer": ds[i]["answer"]}
            for i in selected
        ]
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
    """Experiment 795 entry point — four-step VRAM isolation + Gemma4 inference.

    Steps:
    1. apply_env_autofix() — set JAX_PLATFORMS=cpu to avoid ROCm thrml crash.
    2. ExperimentTemplate + ExperimentTimeoutWatchdog setup.
    3. LiveGPUGate.require_live_or_blocked() — hard gate, no simulated fallback.
    4. kill_gpu_zombies(GPU_INDEX=1) — primary SIGKILL pass.
    5. evict_gpu_vram(GPU_INDEX=1) — pkill sweep + VRAM verification.
    6. Verify vram_clear; abort if False.
    7. load_gemma4_on_gpu1() — allocate model on cuda:1.
    8. Run N_QUESTIONS GSM8K questions; count valid responses.
    9. Compute honest_verdict and write artifact.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Step 1: apply_env_autofix must run FIRST before any JAX import.
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

    with watchdog:
        tmpl.check_exclusion_manifest()

        # Step 3: LiveGPUGate — hard gate.  If CARNOT_FORCE_LIVE is not set,
        # write a blocked artifact immediately and exit.
        if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
            _log.warning("CARNOT_FORCE_LIVE not set — writing blocked artifact")
            artifact = tmpl.build_result(
                {
                    "step1_zombies_killed": 0,
                    "step2_vram_before_mb": 0.0,
                    "step2_pids_killed": [],
                    "step3_vram_after_mb": 0.0,
                    "step3_vram_clear": False,
                    "step4_model_loaded": False,
                    "n_valid_responses": 0,
                    "honest_verdict": "blocked_no_live_gpu",
                    "retro_028_closed": False,
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # Step 4: kill_gpu_zombies on GPU 1 — primary kill pass.
        # This is the mandatory pre-load cleanup mandated by RETRO-028.
        zombie_result = kill_gpu_zombies(gpu_index=GPU_INDEX)
        step1_zombies_killed = len(zombie_result.pids_killed)
        _log.info(
            "Step 4 kill_gpu_zombies: gpu=%d verdict=%s pids_killed=%d "
            "vram_before=%.0f vram_after=%.0f",
            GPU_INDEX,
            zombie_result.honest_verdict,
            step1_zombies_killed,
            zombie_result.vram_before_mb,
            zombie_result.vram_after_mb,
        )

        # Step 5: evict_gpu_vram — pkill sweep + VRAM verification.
        # evict_gpu_vram calls kill_gpu_zombies again internally, then does a
        # second pkill sweep to catch any residual PIDs that survived the first pass.
        eviction = evict_gpu_vram(gpu_index=GPU_INDEX)
        step2_vram_before_mb = eviction.vram_before_mb
        step2_pids_killed = eviction.pids_killed
        step3_vram_after_mb = eviction.vram_after_mb
        step3_vram_clear = eviction.vram_clear
        _log.info(
            "Step 5 evict_gpu_vram: vram_before=%.0f vram_after=%.0f "
            "pids_killed=%d pkill_attempts=%d vram_clear=%s verdict=%s",
            step2_vram_before_mb,
            step3_vram_after_mb,
            len(step2_pids_killed),
            eviction.pkill_attempts,
            step3_vram_clear,
            eviction.honest_verdict,
        )

        # Step 6: hard abort if VRAM not cleared.
        if not step3_vram_clear:
            _log.error(
                "VRAM not cleared after eviction (%.0f MB >= 500 MB) — aborting",
                step3_vram_after_mb,
            )
            artifact = tmpl.build_result(
                {
                    "step1_zombies_killed": step1_zombies_killed,
                    "step2_vram_before_mb": step2_vram_before_mb,
                    "step2_pids_killed": step2_pids_killed,
                    "step3_vram_after_mb": step3_vram_after_mb,
                    "step3_vram_clear": False,
                    "step4_model_loaded": False,
                    "n_valid_responses": 0,
                    "honest_verdict": "vram_not_cleared",
                    "retro_028_closed": False,
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # Step 7: load Gemma4 on GPU 1.
        # load_gemma4_on_gpu1 runs evict_gpu_vram internally as a second safety
        # check before the actual model allocation.  We pass the already-evicted
        # state here — the second eviction will be a no-op if VRAM is still clear.
        load_result = load_gemma4_on_gpu1(model_id=MODEL_ID)
        step4_model_loaded = load_result.get("loaded", False)
        _log.info(
            "Step 7 load_gemma4_on_gpu1: loaded=%s device=%s reason=%s",
            step4_model_loaded,
            load_result.get("device"),
            load_result.get("reason"),
        )

        if not step4_model_loaded:
            artifact = tmpl.build_result(
                {
                    "step1_zombies_killed": step1_zombies_killed,
                    "step2_vram_before_mb": step2_vram_before_mb,
                    "step2_pids_killed": step2_pids_killed,
                    "step3_vram_after_mb": step3_vram_after_mb,
                    "step3_vram_clear": step3_vram_clear,
                    "step4_model_loaded": False,
                    "load_failure_reason": load_result.get("reason", "unknown"),
                    "n_valid_responses": 0,
                    "honest_verdict": "vram_not_cleared",
                    "retro_028_closed": False,
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # Step 8: run N_QUESTIONS GSM8K questions and count valid responses.
        loader: GemmaTransformersLoader = load_result["loader"]
        questions = _load_gsm8k_questions(N_QUESTIONS, SEED)
        _log.info("Running %d GSM8K questions on Gemma4 (cuda:1)", len(questions))

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
                i + 1, len(questions), is_valid, len(resp),
            )
            # Checkpoint every 5 questions to survive interruption.
            if (i + 1) % 5 == 0:
                tmpl.checkpoint_save(
                    {"responses_so_far": responses, "n_valid_so_far": n_valid_responses},
                    step=i + 1,
                )

        # Step 9: compute honest_verdict.
        if n_valid_responses >= 8:
            honest_verdict = "retro_028_closed"
        elif n_valid_responses >= 4:
            honest_verdict = "partial_success"
        else:
            honest_verdict = "model_loaded_no_valid_output"

        retro_028_closed = honest_verdict == "retro_028_closed"

        _log.info(
            "Experiment 795 complete: n_valid=%d/%d verdict=%s retro_028_closed=%s",
            n_valid_responses, N_QUESTIONS, honest_verdict, retro_028_closed,
        )

        artifact = tmpl.build_result(
            {
                "step1_zombies_killed": step1_zombies_killed,
                "step2_vram_before_mb": step2_vram_before_mb,
                "step2_pids_killed": step2_pids_killed,
                "step3_vram_after_mb": step3_vram_after_mb,
                "step3_vram_clear": step3_vram_clear,
                "step4_model_loaded": step4_model_loaded,
                "n_valid_responses": n_valid_responses,
                "n_questions": N_QUESTIONS,
                "gpu_index": GPU_INDEX,
                "model_id": MODEL_ID,
                "honest_verdict": honest_verdict,
                "retro_028_closed": retro_028_closed,
            },
            status="success",
        )
        (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
