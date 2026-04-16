#!/usr/bin/env python3
"""Experiment 411: Live HumanEval code verification benchmark — CRANE-augmented prompts.

**Researcher summary:**
    Re-executes the HumanEval code verification benchmark (Exp 380) with a mandatory
    GPU preflight gate (Exp 404 v2).  Exp 226 showed +3.0pp on a prior live run
    (19/164 -> 24/164); this experiment confirms that number with the current stack.

**Why this is different from Exp 380:**
    Exp 411 adds an upfront check of ``results/experiment_404_preflight_v2.json``
    (the Exp 404 deliverable).  If the preflight does not have
    ``honest_verdict == "gpu_confirmed_live"``, the script writes a blocked artifact
    and exits immediately — before any ExperimentTemplate setup or model loading.

    This is an additional safety layer on top of LiveGPUGate: even if CARNOT_FORCE_LIVE=1
    is set in the environment, a preflight verdict of "env_not_propagating" means the
    subprocess environment is broken and live inference cannot be trusted.

**Why code verification is the most reliable result domain:**
    ArithmeticExtractor relies on finding arithmetic expressions in text — that pattern
    returns 0 violations on instruction-tuned models (Exp 328).  CodeExtractor avoids
    that brittleness entirely: it runs the code against test cases and detects failures
    structurally (wrong output, runtime error, type mismatch).  No regex needed.
    VerifyRepairPipeline feeds failure details back to the LLM to attempt a repaired
    solution.

**Pipeline per problem:**
    1. Generate code with Gemma4-E4B-it.
    2. Run official test cases (subprocess, 10s timeout) — record pass/fail.
    3. If failed: run CodeExtractor + VerifyRepairPipeline to attempt repair.
    4. Re-run official tests on repaired code — record final pass/fail.
    5. For solutions that PASS official tests: run PBT to detect unofficial bugs.

**Honest verdict rules (SCENARIO-BENCH-021):**
    ``honest_verdict="code_verification_positive"`` only when:
    1. ``inference_mode == "live_gpu"``
    2. ``signed_improvement > 0``

**Preflight gate (new in Exp 411):**
    Reads ``results/experiment_404_preflight_v2.json`` before any other setup.
    Exits with a blocked artifact when ``honest_verdict != "gpu_confirmed_live"``.
    This catches the RETRO-015 / RETRO-022 failure mode where the env var is not
    propagated to subprocesses even when it is set in the parent shell.

**Metrics:**
    - pass_at_1_before: fraction passing on first generation (before any repair)
    - pass_at_1_after: fraction passing after the verify-repair loop
    - signed_improvement: pass_at_1_after - pass_at_1_before (signed; no clamping)
    - pbt_bugs_found: count of solutions that passed official tests but failed PBT

**Output:** results/experiment_411_humaneval_live.json

Usage:
    # Live mode (requires GPU + CARNOT_FORCE_LIVE=1 + successful Exp 404 preflight):
    CARNOT_FORCE_LIVE=1 .venv/bin/python scripts/experiment_411_humaneval_live.py

    # CI / no-GPU / bad preflight: produces a blocked artifact immediately
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_411_humaneval_live.py

Spec: REQ-BENCH-004, SCENARIO-BENCH-021
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow import from python/ and scripts/ without installation
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from carnot.pipeline.live_gpu_gate import LiveGPUGate  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

# Import shared helpers from Exp 369 — avoids duplicating tested code.
# These functions are stable and covered by test_experiment_369_humaneval_live.py.
from experiment_369_humaneval_live import (  # noqa: E402
    HumanEvalResult369,
    _extract_code,
    _load_model_pipeline,
    _load_problems,
    _parse_official_tests,
    _process_problem,
    _run_pbt,
    _run_tests,
    _run_tests_subprocess,
    build_humaneval_artifact_v2,
    compute_pass_at_1,
    compute_pass_at_1_after_repair,
)

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 411
EXP_TITLE = "Live HumanEval code verification benchmark — CRANE-augmented prompts"
DELIVERABLE = "results/experiment_411_humaneval_live.json"
_MODEL_IDS = ["google/gemma-4-E4B-it"]
_PREFLIGHT_PATH = "results/experiment_404_preflight_v2.json"

# ---------------------------------------------------------------------------
# Artifact write helper
# ---------------------------------------------------------------------------


def _write_artifact(tmpl: ExperimentTemplate, artifact: dict[str, Any]) -> None:
    """Write the experiment artifact JSON to disk.

    Creates the results/ directory if needed, then writes the artifact as
    pretty-printed JSON.  The output path is derived from the ExperimentTemplate
    deliverable field.

    Args:
        tmpl: ExperimentTemplate instance (provides _output_path).
        artifact: Dict to serialise.

    Spec: REQ-BENCH-004
    """
    tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
    tmpl._output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", tmpl._output_path)


# ---------------------------------------------------------------------------
# Preflight loader
# ---------------------------------------------------------------------------


def _load_preflight(repo_root: Path | None = None) -> dict[str, Any]:
    """Load the Exp 404 preflight verdict from disk.

    Why this exists: Exp 411 adds a pre-gate that reads the GPU preflight result
    from Exp 404 v2 before creating any ExperimentTemplate or touching the GPU.
    If the preflight file is missing or its ``honest_verdict`` is not
    ``"gpu_confirmed_live"``, the experiment must abort with a blocked artifact.
    This prevents wasted time on model loading when the subprocess environment is
    known to be broken (RETRO-022 pattern: env var set in parent but not inherited
    by subprocesses).

    Args:
        repo_root: Repository root path.  Defaults to the parent of the scripts/
            directory (``_REPO_ROOT`` when called from a script).

    Returns:
        Dict loaded from ``results/experiment_404_preflight_v2.json`` if the file
        exists and is valid JSON.  Returns ``{"honest_verdict": "missing"}`` when
        the file is absent, and ``{"honest_verdict": "corrupt"}`` when the file
        exists but cannot be parsed as JSON.

    Spec: REQ-BENCH-004
    """
    root = repo_root if repo_root is not None else _REPO_ROOT
    preflight_path = root / _PREFLIGHT_PATH
    if not preflight_path.is_file():
        _log.warning("Preflight file not found: %s", preflight_path)
        return {"honest_verdict": "missing"}
    try:
        return json.loads(preflight_path.read_text())
    except Exception as exc:  # noqa: BLE001
        _log.warning("Preflight file is not valid JSON (%s): %s", preflight_path, exc)
        return {"honest_verdict": "corrupt"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 411: live HumanEval code verification benchmark.

    Gate sequence (Exp 411 adds gate 0 to the Exp 380 sequence):
        0. Preflight check — read experiment_404_preflight_v2.json.  If
           ``honest_verdict != "gpu_confirmed_live"``, write a blocked artifact
           directly to DELIVERABLE and return immediately.  This gate requires NO
           ExperimentTemplate; the blocked artifact is written without tmpl.
        1. ExperimentTemplate setup.
        2. LiveGPUGate.require_live_or_blocked() — env var + GPU liveness.
        3. tmpl.setup_gpu() — model pre-warm + health check.
        4. _load_model_pipeline() — load tokenizer + model weights.
        Any gate failure produces a blocked artifact.

    There is NO simulated-mode fallback.  A blocked artifact is always better
    than a synthetic result labelled "live_gpu".

    Spec: REQ-BENCH-004, SCENARIO-BENCH-021
    """
    # ---------------------------------------------------------------------------
    # Gate 0: Exp 404 preflight check (new in Exp 411)
    # Read the preflight artifact written by Exp 404 v2.  If it is missing or
    # does not report "gpu_confirmed_live", abort before touching the GPU.
    # This catches the RETRO-022 / RETRO-015 pattern where CARNOT_FORCE_LIVE=1
    # exists in the parent shell but is never inherited by subprocesses.
    # ---------------------------------------------------------------------------
    preflight = _load_preflight()
    preflight_verdict = preflight.get("honest_verdict", "missing")
    if preflight_verdict != "gpu_confirmed_live":
        _log.error(
            "Exp 404 preflight verdict is %r (expected 'gpu_confirmed_live') — "
            "writing blocked artifact and exiting.  "
            "Fix: run Exp 404 with a live GPU and sourced session_startup.sh.",
            preflight_verdict,
        )
        # Write a minimal blocked artifact directly (no tmpl needed at this stage).
        blocked_artifact: dict[str, Any] = {
            "experiment": EXP_ID,
            "title": EXP_TITLE,
            "schema": "carnot.humaneval_benchmark.v2",
            "run_date": _utc_date(),
            "started_at": _utc_now(),
            "finished_at": _utc_now(),
            "duration_s": 0.0,
            "status": "blocked",
            "inference_mode": "blocked",
            "honest_verdict": "blocked",
            "blocked_reason": (
                f"Exp 404 preflight verdict is {preflight_verdict!r}; "
                "expected 'gpu_confirmed_live'"
            ),
            "preflight_verdict": preflight_verdict,
            "n_problems": 0,
            "pass_at_1_before": 0.0,
            "pass_at_1_after": 0.0,
            "signed_improvement": 0.0,
            "pbt_bugs_found": 0,
        }
        out_path = _REPO_ROOT / DELIVERABLE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(blocked_artifact, indent=2))
        _log.info("Blocked artifact written to %s", out_path)
        return

    _log.info("Exp 404 preflight OK (honest_verdict=%r) — proceeding.", preflight_verdict)

    # ---------------------------------------------------------------------------
    # ExperimentTemplate setup
    # ---------------------------------------------------------------------------
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # ---------------------------------------------------------------------------
    # Gate 1: LiveGPUGate.require_live_or_blocked() (RETRO-015 fix, Exp 377)
    # Checks CARNOT_FORCE_LIVE=1 AND diagnose_live_gpu().is_live_capable.
    # Returns a blocked artifact dict on failure; None on success.
    # ---------------------------------------------------------------------------
    blocked = LiveGPUGate.require_live_or_blocked(tmpl, _MODEL_IDS)
    if blocked is not None:
        _log.error(
            "LiveGPUGate blocked Exp 411 — writing blocked artifact and exiting."
        )
        _write_artifact(tmpl, blocked)
        return

    # ---------------------------------------------------------------------------
    # Gate 2: GPU pre-warm via ExperimentTemplate.setup_gpu() (Exp 294 pattern).
    # ---------------------------------------------------------------------------
    gpu_status = tmpl.setup_gpu(
        [{"name": "Gemma4-E4B-it", "hf_id": _MODEL_IDS[0], "gpu": 0}]
    )
    if not gpu_status["all_healthy"]:
        _log.error(
            "setup_gpu reports unhealthy — writing blocked artifact.  models=%s",
            gpu_status["models"],
        )
        artifact = tmpl.build_result(
            {
                "humaneval_schema": "carnot.humaneval_benchmark.v2",
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "failure_reason": "setup_gpu health check failed",
                "n_problems": 0,
                "pass_at_1_before": 0.0,
                "pass_at_1_after": 0.0,
                "signed_improvement": 0.0,
                "pbt_bugs_found": 0,
                "gpu_status": gpu_status,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    inference_mode = "live_gpu"
    _log.info("GPU gate passed — inference_mode=%s", inference_mode)

    # ---------------------------------------------------------------------------
    # Gate 3: Load model weights (tokenizer + causal LM).
    # ---------------------------------------------------------------------------
    tokenizer, model, device, ok = _load_model_pipeline(
        hf_id=_MODEL_IDS[0], device=0
    )
    if not ok:
        _log.error("_load_model_pipeline failed — writing blocked artifact.")
        artifact = tmpl.build_result(
            {
                "humaneval_schema": "carnot.humaneval_benchmark.v2",
                "inference_mode": "blocked",
                "honest_verdict": "blocked",
                "failure_reason": "model load failed after GPU gate passed",
                "n_problems": 0,
                "pass_at_1_before": 0.0,
                "pass_at_1_after": 0.0,
                "signed_improvement": 0.0,
                "pbt_bugs_found": 0,
            },
            status="blocked",
        )
        _write_artifact(tmpl, artifact)
        return

    # ---------------------------------------------------------------------------
    # Load problems (50 HumanEval problems).
    # ---------------------------------------------------------------------------
    problems = _load_problems()
    _log.info("[Exp 411] %d problems loaded.", len(problems))

    # ---------------------------------------------------------------------------
    # Process problems with checkpointing every 10.
    # ---------------------------------------------------------------------------
    results: list[HumanEvalResult369] = []
    for i, problem in enumerate(problems):
        try:
            result = _process_problem(problem, tokenizer, model, device)
            results.append(result)
        except Exception as exc:
            _log.warning("[Exp 411] problem %d error: %r", i, exc)
            results.append(
                HumanEvalResult369(
                    problem_id=problem.get("task_id", f"unknown/{i}"),
                    generated_code="",
                    passed_tests=False,
                    violations_found=0,
                    repair_attempted=False,
                    final_code="",
                    final_passed_tests=False,
                    pbt_bug_found=False,
                )
            )

        if (i + 1) % 10 == 0:
            tmpl.checkpoint_save(
                {
                    "completed": i + 1,
                    "partial_results": [asdict(r) for r in results],
                },
                step=i + 1,
            )

    humaneval_data = build_humaneval_artifact_v2(results, inference_mode)

    _log.info(
        "[Exp 411] pass@1_before=%.3f  pass@1_after=%.3f  "
        "signed_improvement=%+.3f  honest_verdict=%s  pbt_bugs=%d",
        humaneval_data["pass_at_1_before"],
        humaneval_data["pass_at_1_after"],
        humaneval_data["signed_improvement"],
        humaneval_data["honest_verdict"],
        humaneval_data["pbt_bugs_found"],
    )

    artifact = tmpl.build_result(humaneval_data, status="success")
    _write_artifact(tmpl, artifact)


# ---------------------------------------------------------------------------
# Minimal timestamp helpers (avoid importing datetime in module scope)
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return current UTC timestamp in ISO-8601 format.

    Used to populate started_at/finished_at in the preflight-blocked artifact,
    which is written before ExperimentTemplate is created.

    Returns:
        String like "2026-04-16T11:52:29Z".
    """
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_date() -> str:
    """Return current UTC date in YYYYMMDD format.

    Used to populate run_date in the preflight-blocked artifact.

    Returns:
        String like "20260416".
    """
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")


if __name__ == "__main__":
    main()
