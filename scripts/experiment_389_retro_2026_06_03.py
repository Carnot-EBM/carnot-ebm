#!/usr/bin/env python3
"""Exp 389: Operational Retrospective — Milestone 2026.06.03.

**Researcher summary:**
    Milestone 2026.06.03 targeted "Break the Simulated Barrier — First Live Numbers and
    JitRL Self-Learning". It inherited four RETRO items from 2026.05.27:
    (1) RETRO-015: Live GPU fourth consecutive failure — addressed by Exp 377 (LiveGPUGate
    class + session_startup.sh CARNOT_FORCE_LIVE=1 export). RETRO-015 was formally closed
    by the infrastructure fix, but live GPU inference still did not execute in practice
    (GPU runtime unavailable during the conductor session);
    (2) RETRO-016: LLMExtractor no honest verdict — upstream blocked by RETRO-015;
    (3) RETRO-017: FR-11 self-learning relay unconfirmed — upstream blocked by RETRO-015;
    (4) RETRO-018: CIKAN deliverable corrupt — Exp 378 (re-implementation) was interrupted
    before completion; no result JSON or script for Exp 378 found on disk.

    The milestone session was interrupted (conductor checkpoint commit "preserve uncommitted
    work from interrupted run"). As a result:
    - Exps 378, 386, 387 are fully missing (no script, no result JSON).
    - Exps 379-385, 388 have partial result JSONs (status='partial',
      finding='Extended GPU runtime needed') but no live inference results.
    - live_gpu_confirmed=False: fifth consecutive milestone with idle GPUs.

    Key finding: Exp 377 (LiveGPUGate) correctly wired CARNOT_FORCE_LIVE=1 into
    session_startup.sh and proved subprocess env propagation, resolving the infrastructure
    gap. The RTX 3090s remain physically available. The bottleneck shifted from "env var
    not set" to "GPU runtime unavailable during conductor session execution" — the conductor
    script runs on a CPU-only environment and the GPU node is not auto-started.

    RETRO-019 (CRITICAL) is opened: live GPU confirmation has now failed across FIVE
    consecutive milestones. The infrastructure is correct; the execution environment
    must be verified before each conductor session (GPU node must be online).

**Milestone 2026.06.03 experiment inventory (Exps 377-388):**

    Exp 377: RETRO-015 infrastructure fix — LiveGPUGate + session_startup.sh (COMPLETE)
    Exp 378: RETRO-018 re-implementation — CIKANEnergy Python class (MISSING — interrupted)
    Exp 379: Live precision pipeline execution — 5 variants × 2 models × 200 GSM8K (PARTIAL)
    Exp 380: Live HumanEval code verification — 50 problems, live GPU gate (PARTIAL)
    Exp 381: Live adversarial GSM8K — Apple benchmark with CARNOT_FORCE_LIVE=1 (PARTIAL)
    Exp 382: Live extraction comparison — LLMExtractor vs ArithmeticExtractor (PARTIAL)
    Exp 383: Combined EORM+JEPA retrain on live pairs from Exps 379-382 (PARTIAL)
    Exp 384: FR-11 relay live — self-learning relay with honest verdict (PARTIAL)
    Exp 385: Three-tier pipeline live execution (PARTIAL)
    Exp 386: JitRL threshold modulation — Tier 1 self-learning (MISSING — interrupted)
    Exp 387: Safety KAN AUC-ROC benchmark (MISSING — interrupted)
    Exp 388: SAVeR live multi-turn verification (PARTIAL)

**Wall-time estimates (from conductor log timestamps):**
    Session ran 2026-04-16. Times estimated from consecutive conductor log entries.
    Session was interrupted before Exps 386-387 and Exp 378 could be written.

**Output:** results/operational_retro_2026_06_03.json

Spec: REQ-INFRA-017/018 (LiveGPUGate), REQ-LEARN-025/026/027 (EORM/relay retrain),
      REQ-BENCH-003/004/006/007 (precision/HumanEval/adversarial benchmarks),
      REQ-EXTRACT-023 (extraction comparison), REQ-AGENT-001/002 (SAVeR)
SCENARIO: RETRO-2026.06.03
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo-root path setup — must happen before any carnot/scripts imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MILESTONE = "2026.06.03"
DELIVERABLE = "results/operational_retro_2026_06_03.json"

# Mean experiment duration from the previous milestone (2026.05.27 retro, Exp 376).
# Includes blocked experiments that fast-failed via the LiveGPU gate.
PREV_MEAN_EXP_DURATION_MIN: float = 22.7

# Mapping of experiment result files (keys are string IDs for flexibility).
# None means the deliverable was a source module, not a JSON artifact.
RESULT_FILE_MAP: dict[str, str | None] = {
    "377": "results/experiment_377_gpu_session_fix.json",
    "378": "results/experiment_378_cikan_energy.json",
    "379": "results/experiment_379_precision_execute.json",
    "380": "results/experiment_380_humaneval_execute.json",
    "381": "results/experiment_381_adversarial_execute.json",
    "382": "results/experiment_382_extraction_execute.json",
    "383": "results/experiment_383_models_retrain.json",
    "384": "results/experiment_384_relay_live.json",
    "385": "results/experiment_385_three_tier_execute.json",
    "386": "results/experiment_386_jitrl_memory.json",
    "387": "results/experiment_387_safety_kan.json",
    "388": "results/experiment_388_saver_live.json",
}

# Full experiment metadata for wall-time statistics.
# Wall times estimated from conductor log timestamps (2026-04-16 session).
# Missing experiments (378, 386, 387) contribute 0 wall time — session was interrupted.
MILESTONE_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": 377,
        "title": "RETRO-015 infrastructure fix — LiveGPUGate + session_startup.sh",
        "result_file": "results/experiment_377_gpu_session_fix.json",
        "wall_time_min": 34,   # 03:19 plan → 03:53 commit (conductor log)
        "status": "completed",
        "note": (
            "47 tests pass. LiveGPUGate class implemented; session_startup.sh now exports "
            "CARNOT_FORCE_LIVE=1; verify_subprocess_env_propagation() confirms subprocesses "
            "inherit the env var. RETRO-015 formally closed."
        ),
    },
    {
        "id": 378,
        "title": "RETRO-018 re-implementation — CIKANEnergy Python class",
        "result_file": "results/experiment_378_cikan_energy.json",
        "wall_time_min": 0,    # Session interrupted before this was written
        "status": "missing",
        "note": (
            "Session interrupted before Exp 378 could be implemented. "
            "No script, no test, no result JSON. cikan_implemented=False (second failure). "
            "RETRO-020 opened."
        ),
    },
    {
        "id": 379,
        "title": "Live precision pipeline execution — 5 variants × 2 models × 200 GSM8K",
        "result_file": "results/experiment_379_precision_execute.json",
        "wall_time_min": 42,   # 03:53 → 04:35 (conductor log)
        "status": "partial",
        "note": (
            "22 tests pass. Script written (LiveGPUGate hard gate). "
            "Live run pending — GPU runtime not available during session. "
            "result JSON has status='partial', no honest_verdict."
        ),
    },
    {
        "id": 380,
        "title": "Live HumanEval code verification benchmark",
        "result_file": "results/experiment_380_humaneval_execute.json",
        "wall_time_min": 13,   # 04:35 → 04:48 (conductor log)
        "status": "partial",
        "note": (
            "24 tests pass. Script written with LiveGPUGate. "
            "Live run pending. result JSON has status='partial'."
        ),
    },
    {
        "id": 381,
        "title": "Live adversarial GSM8K — Apple arXiv 2410.05229 with CARNOT_FORCE_LIVE=1",
        "result_file": "results/experiment_381_adversarial_execute.json",
        "wall_time_min": 15,   # estimated from checkpoint commit
        "status": "partial",
        "note": "Script created. Extended GPU runtime needed. result JSON status='partial'.",
    },
    {
        "id": 382,
        "title": "Live extraction comparison — LLMExtractor vs ArithmeticExtractor",
        "result_file": "results/experiment_382_extraction_execute.json",
        "wall_time_min": 15,   # estimated
        "status": "partial",
        "note": "Script created. Extended GPU runtime needed. result JSON status='partial'.",
    },
    {
        "id": 383,
        "title": "Combined EORM+JEPA retrain on live pairs from Exps 379-382",
        "result_file": "results/experiment_383_models_retrain.json",
        "wall_time_min": 85,   # 04:48 → 06:13 (conductor log — significant work)
        "status": "partial",
        "note": (
            "41 tests pass. EORM+JEPA combined retrain script written. "
            "honest_verdict=insufficient_pairs (Exps 379-382 produced no live pairs — "
            "upstream blocked by live GPU unavailability). LIVE RUN PENDING."
        ),
    },
    {
        "id": 384,
        "title": "FR-11 self-learning relay live — first learning_confirmed verdict",
        "result_file": "results/experiment_384_relay_live.json",
        "wall_time_min": 15,   # estimated from checkpoint commit (Exp 383-388 batch)
        "status": "partial",
        "note": "Script and tests created. Needs extended runtime. result JSON status='partial'.",
    },
    {
        "id": 385,
        "title": "Three-tier pipeline live execution",
        "result_file": "results/experiment_385_three_tier_execute.json",
        "wall_time_min": 10,   # estimated
        "status": "partial",
        "note": "Script and tests created. Needs extended runtime.",
    },
    {
        "id": 386,
        "title": "JitRL threshold modulation — Tier 1 self-learning memory (arXiv 2601.18510)",
        "result_file": "results/experiment_386_jitrl_memory.json",
        "wall_time_min": 0,    # Session interrupted before this was written
        "status": "missing",
        "note": "Session interrupted before Exp 386 could be implemented. No artifacts.",
    },
    {
        "id": 387,
        "title": "Safety KAN AUC-ROC benchmark — constraint-aware safety scoring",
        "result_file": "results/experiment_387_safety_kan.json",
        "wall_time_min": 0,    # Session interrupted before this was written
        "status": "missing",
        "note": "Session interrupted before Exp 387 could be implemented. No artifacts.",
    },
    {
        "id": 388,
        "title": "SAVeR live multi-turn verification wrapper — live GPU execution",
        "result_file": "results/experiment_388_saver_live.json",
        "wall_time_min": 10,   # estimated from checkpoint commit
        "status": "partial",
        "note": "Script and tests created. Needs extended runtime. result JSON status='partial'.",
    },
]

# ---------------------------------------------------------------------------
# Dataclass: milestone success criteria
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MilestoneRetro2026_06_03:
    """All measurable success criteria for milestone 2026.06.03.

    Each boolean field corresponds to a primary goal stated in the milestone plan.
    False does NOT mean "failed completely" — it means the criterion was not met
    as defined. Context is captured in build_retro_artifact().

    Fields
    ------
    retro_015_closed : bool
        Exp 377 result JSON has status == "complete", indicating that the
        LiveGPUGate infrastructure fix was applied and RETRO-015 formally closed.
        Note: closing RETRO-015 does NOT require live inference to have occurred —
        only that the infra fix (session_startup.sh + LiveGPUGate) is verified.

    retro_018_closed : bool
        Exp 378 result JSON exists and shows a valid CIKANEnergy Python class
        (python/carnot/models/cikan_energy.py contains 'class CIKANEnergy').
        RETRO-018 is closed when cikan_implemented=True AND Exp 378 status=success.

    live_gpu_confirmed : bool
        At least one experiment in Exps 377-388 reported inference_mode == "live_gpu".
        Requires a live GPU runtime during the conductor session.

    precision_result_credible : bool
        Exp 379 result has honest_verdict == "live_improvement" AND
        inference_mode == "live_gpu".

    humaneval_result_credible : bool
        Exp 380 result has honest_verdict == "code_verification_positive" AND
        inference_mode == "live_gpu".

    adversarial_result_credible : bool
        Exp 381 result has honest_verdict == "improvement_positive" AND
        inference_mode == "live_gpu".

    extraction_winner_known : bool
        Exp 382 result has honest_verdict in ("live_gpu_winner", "live_gpu_no_improvement")
        AND inference_mode == "live_gpu". Either outcome is acceptable — the criterion
        is satisfied when we have a live, credible verdict (not simulated).

    fr11_learning_confirmed : bool
        Exp 384 result has honest_verdict == "learning_confirmed" AND
        inference_mode == "live_gpu". FR-11 (mandatory) is satisfied.

    jitrl_memory_works : bool
        Exp 386 result has honest_verdict == "threshold_modulation_confirmed" — meaning
        JitRL-style threshold modulation produced measurable self-improvement vs weight-
        reweighting baseline (arXiv 2601.18510).

    safety_kan_works : bool
        Exp 387 result reports test_auc_roc > 0.70 on the held-out safety evaluation set,
        indicating Safety KAN is a viable safety scoring mechanism.

    saver_live_verified : bool
        Exp 388 result has inference_mode == "live_gpu" and faithfulness > 0.0, indicating
        at least one step of the multi-turn chain was committed on live GPU inference.

    cikan_implemented : bool
        python/carnot/models/cikan_energy.py contains valid Python with 'class CIKANEnergy',
        AND Exp 378 result JSON shows status == "success".

    mean_exp_duration_min : float
        Mean wall time across all milestone experiments (including zero-duration missing ones).

    n_experiments_blocked : int
        Number of experiments with status == "blocked" or "missing" in MILESTONE_EXPERIMENTS.

    retro_items_opened : list[str]
        IDs of new RETRO items opened by this retrospective.
    """

    retro_015_closed: bool
    retro_018_closed: bool
    live_gpu_confirmed: bool
    precision_result_credible: bool
    humaneval_result_credible: bool
    adversarial_result_credible: bool
    extraction_winner_known: bool
    fr11_learning_confirmed: bool
    jitrl_memory_works: bool
    safety_kan_works: bool
    saver_live_verified: bool
    cikan_implemented: bool
    mean_exp_duration_min: float
    n_experiments_blocked: int
    retro_items_opened: list[str]


# ---------------------------------------------------------------------------
# New RETRO items opened by this retrospective
# ---------------------------------------------------------------------------

NEW_RETRO_ITEMS: list[dict[str, Any]] = [
    {
        "id": "RETRO-019",
        "title": (
            "Live GPU still never ran in a conductor session — fifth consecutive milestone (CRITICAL)"
        ),
        "status": "NEW",
        "priority": "critical",
        "description": (
            "Exp 377 (RETRO-015 infrastructure fix) correctly wired CARNOT_FORCE_LIVE=1 "
            "into session_startup.sh and implemented LiveGPUGate with subprocess env "
            "propagation verification. The infrastructure fix is sound. However, the conductor "
            "session itself ran on a CPU-only environment — the GPU node was not online during "
            "the session. All GPU-tagged experiments (379-388) returned status='partial' with "
            "'Extended GPU runtime needed'. This is the FIFTH consecutive milestone "
            "(2026.05.06, 2026.05.13, 2026.05.20, 2026.05.27, 2026.06.03) where live GPU "
            "inference was confirmed capable in theory but never executed in practice. "
            "The bottleneck has shifted: previously it was the env var not being set; now it "
            "is that the GPU node must be physically online when the conductor session starts. "
            "Escalation: before starting milestone 2026.06.10, the operator MUST verify that "
            "at least one GPU is online and 'nvidia-smi' returns a device. If the GPU node "
            "is offline, the session must not start experiment work — it must fix the GPU "
            "availability issue first."
        ),
        "root_cause": (
            "Conductor sessions run in a CPU-only shell environment. The GPU node (RTX 3090s) "
            "requires a separate process or service to be online. Even though CARNOT_FORCE_LIVE=1 "
            "is now correctly exported by session_startup.sh (Exp 377 fix), the GPU hardware "
            "itself is not being activated before the session. The 'live GPU gate' passes the "
            "env var check but fails at the CUDA device availability check — this means the "
            "gate is working correctly (fail-fast), but the GPU is not present."
        ),
        "fix": (
            "Before starting milestone 2026.06.10 conductor session: "
            "(1) Run 'nvidia-smi' and confirm at least one GPU is listed. "
            "(2) If no GPU is shown, start the GPU service / power on the node. "
            "(3) Run Exp 353 (live GPU smoke test) as the FIRST experiment in the session. "
            "(4) Only proceed to experiment code if Exp 353 produces inference_mode='live_gpu'. "
            "The conductor should NOT write any new experiment scripts until smoke test passes."
        ),
        "estimated_savings_pct": 30,
        "rationale": (
            "Live GPU unblocks 8+ blocked/partial experiments (379-388 minus 378). "
            "Each transitions from a 10-15 min partial (script written, no results) to a "
            "live inference session producing provenance-bearing results. This eliminates "
            "the do-over cost for precision, HumanEval, adversarial, extraction, relay, "
            "three-tier, and SAVeR benchmarks — all written and waiting for GPU runtime."
        ),
    },
    {
        "id": "RETRO-020",
        "title": "CIKANEnergy not implemented — second consecutive milestone failure",
        "status": "NEW",
        "priority": "high",
        "description": (
            "Exp 378 (re-implementation of RETRO-018) was not completed — the conductor "
            "session was interrupted before Exp 378 could be written. No script, no test "
            "file, and no result JSON exist for Exp 378. cikan_energy.py on disk still "
            "contains the corrupt JSON artifact from Exp 375. This is the second consecutive "
            "milestone where the CIKAN deliverable was targeted and not delivered. "
            "The constraint-informed KAN energy tier (arXiv 2412.03710) remains unimplemented. "
            "Energy separation ratio vs standard KAN cannot be computed."
        ),
        "root_cause": (
            "Session interrupt truncated the experiment queue before reaching Exp 378. "
            "RETRO-018 was listed as Exp 2 in the milestone plan but the interrupted session "
            "only completed Exps 377, 379-385, 388 before the conductor was stopped."
        ),
        "fix": (
            "In milestone 2026.06.10: schedule Exp 378 re-implementation (CIKANEnergy) as "
            "experiment number 1 — before any other experiment. "
            "Write python/carnot/models/cikan_energy.py with a proper CIKANEnergy class. "
            "Run tests/python/test_cikan_energy.py with 100% coverage. "
            "Write results/experiment_378_cikan_energy.json with status='success'."
        ),
        "estimated_savings_pct": 3,
    },
    {
        "id": "RETRO-021",
        "title": "FR-11 self-learning relay still unconfirmed on live data — third milestone",
        "status": "NEW",
        "priority": "high",
        "description": (
            "Exp 384 (FR-11 live relay) returned status='partial'. FR-11 is a mandatory "
            "functional requirement: the self-learning relay must produce learning_confirmed "
            "on live GPU inference. This goal has been carried since milestone 2026.05.20 "
            "(Exp 361 confirmed synthetic, honest_verdict=synthetic_only). "
            "Three consecutive milestones (2026.05.27, 2026.06.03, and the underlying Exp 361 "
            "in 2026.05.20) have failed to produce a live relay verdict. "
            "The relay machinery is sound (54 tests, 100% module coverage, Exp 361 ran "
            "0.60→0.72 on synthetic data). The blocker is exclusively live GPU unavailability "
            "(RETRO-019 upstream). Once RETRO-019 is resolved, Exp 384 should complete "
            "immediately — the script and tests already exist."
        ),
        "root_cause": "RETRO-019 (live GPU unavailability) is the upstream blocker.",
        "fix": "Close RETRO-019 first. Then re-run Exp 384 with CARNOT_FORCE_LIVE=1.",
        "estimated_savings_pct": 5,
    },
]


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------


def load_milestone_results(
    repo_root: Path, file_map: dict[str, str | None]
) -> dict[str, dict[str, Any] | None]:
    """Load milestone result JSON files, returning None for missing or invalid files.

    Missing result files document criteria failures — they are not errors.
    Partial result files (status='partial') are loaded and evaluated normally.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.
    file_map : dict[str, str | None]
        Mapping from experiment key (e.g. "377") to relative result file path.
        None means the experiment's deliverable is not a JSON file.

    Returns
    -------
    dict[str, dict | None]
        Mapping from experiment key to parsed JSON dict, or None if unavailable.
    """
    out: dict[str, dict[str, Any] | None] = {}
    for key, rel_path in file_map.items():
        if rel_path is None:
            out[key] = None
            continue
        full_path = repo_root / rel_path
        if not full_path.exists():
            _log.debug("Result file not found: %s", full_path)
            out[key] = None
            continue
        try:
            out[key] = json.loads(full_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("Failed to load %s: %s", full_path, exc)
            out[key] = None
    return out


# ---------------------------------------------------------------------------
# Timing statistics
# ---------------------------------------------------------------------------


def compute_timing_stats(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute wall-time statistics across all milestone experiments.

    Wall times are estimated from conductor log timestamps. Missing experiments
    (interrupted session) contribute zero minutes and are counted as blocked.
    Partial experiments that wrote scripts-only (no live results) are also counted.

    Parameters
    ----------
    experiments : list[dict]
        Experiment metadata list. Each entry must have 'wall_time_min' and 'status'.

    Returns
    -------
    dict
        n_ran, n_blocked, total_min, mean_min, slowest, fastest.
    """
    if not experiments:
        return {
            "n_ran": 0,
            "n_blocked": 0,
            "total_min": 0,
            "mean_min": 0.0,
            "slowest": None,
            "fastest": None,
        }

    # "blocked" = missing or blocked status (did no useful GPU work)
    n_blocked = sum(
        1 for e in experiments if e.get("status") in ("blocked", "missing")
    )
    total = sum(e["wall_time_min"] for e in experiments)
    mean = round(total / len(experiments), 1) if experiments else 0.0

    slowest = max(experiments, key=lambda e: e["wall_time_min"])
    fastest = min(experiments, key=lambda e: e["wall_time_min"])

    return {
        "n_ran": len(experiments),
        "n_blocked": n_blocked,
        "total_min": total,
        "mean_min": mean,
        "slowest": {
            "id": slowest["id"],
            "title": slowest.get("title", ""),
            "wall_time_min": slowest["wall_time_min"],
        },
        "fastest": {
            "id": fastest["id"],
            "title": fastest.get("title", ""),
            "wall_time_min": fastest["wall_time_min"],
        },
    }


# ---------------------------------------------------------------------------
# Speedup computation
# ---------------------------------------------------------------------------


def estimate_speedup_pct(prev_mean: float, curr_mean: float) -> float:
    """Compute the percentage speedup relative to the previous milestone mean.

    A positive return value indicates faster experiments (improvement).
    A negative value indicates regression (slower experiments).

    Formula: (prev_mean - curr_mean) / prev_mean * 100

    Parameters
    ----------
    prev_mean : float
        Previous milestone mean experiment duration in minutes (22.7 for 2026.05.27).
    curr_mean : float
        This milestone's mean experiment duration in minutes.

    Returns
    -------
    float
        Speedup percentage. 0.0 if prev_mean is zero (guard against ZeroDivisionError).
    """
    if prev_mean == 0.0:
        return 0.0
    return round((prev_mean - curr_mean) / prev_mean * 100, 2)


# ---------------------------------------------------------------------------
# CIKAN deliverable check
# ---------------------------------------------------------------------------


def _check_cikan_implemented(repo_root: Path, results: dict[str, Any | None]) -> bool:
    """Return True only if cikan_energy.py is valid Python with a CIKANEnergy class.

    A JSON file at the expected path is treated as corrupt and returns False.
    Exp 378 result JSON must also show status == 'success'.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    results : dict
        Loaded result files keyed by experiment key string.

    Returns
    -------
    bool
        True only when (a) the .py file exists, (b) is parseable Python (not JSON),
        (c) contains 'class CIKANEnergy', AND (d) Exp 378 result has status=success.
    """
    cikan_path = repo_root / "python" / "carnot" / "models" / "cikan_energy.py"
    if not cikan_path.exists():
        return False

    content = cikan_path.read_text()

    # A JSON file starts with '{' or '[' — not valid Python class definitions
    stripped = content.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        _log.warning("cikan_energy.py appears to contain JSON, not Python code.")
        return False

    if "class CIKANEnergy" not in content:
        return False

    exp378 = results.get("378")
    if exp378 is None:
        return False
    return exp378.get("status") == "success"


# ---------------------------------------------------------------------------
# Core retrospective computation
# ---------------------------------------------------------------------------


def compute_retro_2026_06_03(
    result_files: dict[str, dict[str, Any] | None],
    repo_root: Path,
) -> MilestoneRetro2026_06_03:
    """Evaluate all milestone 2026.06.03 success criteria from loaded result files.

    Separated from main() so it can be unit-tested with controlled fixture data.

    Parameters
    ----------
    result_files : dict
        Pre-loaded result JSON dicts keyed by experiment key string (e.g. "377").
        None values indicate missing or unloadable files.
    repo_root : Path
        Repository root. Used for cikan_energy.py inspection.

    Returns
    -------
    MilestoneRetro2026_06_03
        Evaluated success criteria for milestone 2026.06.03.
    """
    # --- retro_015_closed ---
    # Exp 377 must show status == "complete" (infrastructure fix applied)
    exp377 = result_files.get("377") or {}
    retro_015_closed = exp377.get("status") == "complete"

    # --- cikan_implemented (needed for retro_018_closed) ---
    cikan_implemented = _check_cikan_implemented(repo_root, result_files)

    # --- retro_018_closed ---
    # CIKAN must be implemented AND Exp 378 result must show success
    exp378 = result_files.get("378") or {}
    retro_018_closed = cikan_implemented and exp378.get("status") == "success"

    # --- live_gpu_confirmed ---
    # Scan all result files for inference_mode == "live_gpu"
    live_gpu_confirmed = any(
        r is not None and r.get("inference_mode") == "live_gpu"
        for r in result_files.values()
    )

    # --- precision_result_credible ---
    exp379 = result_files.get("379") or {}
    precision_result_credible = (
        exp379.get("honest_verdict") == "live_improvement"
        and exp379.get("inference_mode") == "live_gpu"
    )

    # --- humaneval_result_credible ---
    exp380 = result_files.get("380") or {}
    humaneval_result_credible = (
        exp380.get("honest_verdict") == "code_verification_positive"
        and exp380.get("inference_mode") == "live_gpu"
    )

    # --- adversarial_result_credible ---
    exp381 = result_files.get("381") or {}
    adversarial_result_credible = (
        exp381.get("honest_verdict") == "improvement_positive"
        and exp381.get("inference_mode") == "live_gpu"
    )

    # --- extraction_winner_known ---
    # Any live extraction verdict (win OR no-improvement) counts — we just need a live result
    exp382 = result_files.get("382") or {}
    extraction_winner_known = (
        exp382.get("honest_verdict") in ("live_gpu_winner", "live_gpu_no_improvement")
        and exp382.get("inference_mode") == "live_gpu"
    )

    # --- fr11_learning_confirmed ---
    exp384 = result_files.get("384") or {}
    fr11_learning_confirmed = (
        exp384.get("honest_verdict") == "learning_confirmed"
        and exp384.get("inference_mode") == "live_gpu"
    )

    # --- jitrl_memory_works ---
    # Exp 386 must report honest_verdict == "threshold_modulation_confirmed"
    exp386 = result_files.get("386") or {}
    jitrl_memory_works = (
        exp386.get("honest_verdict") == "threshold_modulation_confirmed"
    )

    # --- safety_kan_works ---
    # Exp 387 must report test_auc_roc > 0.70
    exp387 = result_files.get("387") or {}
    safety_auc = exp387.get("test_auc_roc", 0.0)
    safety_kan_works = isinstance(safety_auc, (int, float)) and float(safety_auc) > 0.70

    # --- saver_live_verified ---
    # Exp 388 must report inference_mode == "live_gpu" and faithfulness > 0.0
    exp388 = result_files.get("388") or {}
    saver_faithfulness = exp388.get("faithfulness", 0.0)
    saver_live_verified = (
        exp388.get("inference_mode") == "live_gpu"
        and isinstance(saver_faithfulness, (int, float))
        and float(saver_faithfulness) > 0.0
    )

    # --- timing stats ---
    timing_stats = compute_timing_stats(MILESTONE_EXPERIMENTS)
    mean_exp_duration_min = float(timing_stats["mean_min"])
    n_experiments_blocked = timing_stats["n_blocked"]

    # --- new RETRO items ---
    retro_items_opened: list[str] = []
    # RETRO-019: live GPU still failed despite RETRO-015 fix
    if not live_gpu_confirmed:
        retro_items_opened.append("RETRO-019")
    # RETRO-020: CIKAN still not implemented
    if not cikan_implemented:
        retro_items_opened.append("RETRO-020")
    # RETRO-021: FR-11 learning unconfirmed (third milestone)
    if not fr11_learning_confirmed:
        retro_items_opened.append("RETRO-021")

    return MilestoneRetro2026_06_03(
        retro_015_closed=retro_015_closed,
        retro_018_closed=retro_018_closed,
        live_gpu_confirmed=live_gpu_confirmed,
        precision_result_credible=precision_result_credible,
        humaneval_result_credible=humaneval_result_credible,
        adversarial_result_credible=adversarial_result_credible,
        extraction_winner_known=extraction_winner_known,
        fr11_learning_confirmed=fr11_learning_confirmed,
        jitrl_memory_works=jitrl_memory_works,
        safety_kan_works=safety_kan_works,
        saver_live_verified=saver_live_verified,
        cikan_implemented=cikan_implemented,
        mean_exp_duration_min=mean_exp_duration_min,
        n_experiments_blocked=n_experiments_blocked,
        retro_items_opened=retro_items_opened,
    )


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_retro_artifact(retro: MilestoneRetro2026_06_03) -> dict[str, Any]:
    """Convert a MilestoneRetro2026_06_03 dataclass into the output artifact dict.

    Schema is "carnot.operational_retro.v3" — upgraded from v2 (Exp 376) to reflect
    the new success_criteria structure including JitRL, Safety KAN, and SAVeR criteria,
    plus the first_live_gpu_results_achieved flag and headline_results dict.

    Parameters
    ----------
    retro : MilestoneRetro2026_06_03
        Evaluated success criteria.

    Returns
    -------
    dict
        Complete retrospective artifact ready for JSON serialization.
    """
    speedup_pct = estimate_speedup_pct(PREV_MEAN_EXP_DURATION_MIN, retro.mean_exp_duration_min)
    timing_stats = compute_timing_stats(MILESTONE_EXPERIMENTS)

    # Success criteria dict (all scalar + list fields)
    success_criteria: dict[str, Any] = {
        "retro_015_closed": retro.retro_015_closed,
        "retro_018_closed": retro.retro_018_closed,
        "live_gpu_confirmed": retro.live_gpu_confirmed,
        "precision_result_credible": retro.precision_result_credible,
        "humaneval_result_credible": retro.humaneval_result_credible,
        "adversarial_result_credible": retro.adversarial_result_credible,
        "extraction_winner_known": retro.extraction_winner_known,
        "fr11_learning_confirmed": retro.fr11_learning_confirmed,
        "jitrl_memory_works": retro.jitrl_memory_works,
        "safety_kan_works": retro.safety_kan_works,
        "saver_live_verified": retro.saver_live_verified,
        "cikan_implemented": retro.cikan_implemented,
        "n_experiments_blocked": retro.n_experiments_blocked,
    }

    explanations: dict[str, str] = {
        "retro_015_closed": (
            "Exp 377 (LiveGPUGate + session_startup.sh) formally closed RETRO-015. "
            "47 tests pass. CARNOT_FORCE_LIVE=1 is now exported from session_startup.sh "
            "and verify_subprocess_env_propagation() confirms subprocess inheritance. "
            "Infrastructure fix is sound. However, live inference still did not execute "
            "because the GPU node was offline during the conductor session — RETRO-019."
        ),
        "retro_018_closed": (
            "Exp 378 was not implemented — session interrupted before reaching it. "
            "cikan_energy.py on disk still contains JSON (Exp 375 corrupt artifact). "
            "RETRO-020 opened (second consecutive milestone failure for CIKAN)."
        ),
        "live_gpu_confirmed": (
            "FIFTH consecutive milestone with zero live GPU inference. "
            "Infrastructure is now correct (Exp 377 fix). The GPU node was offline "
            "during the conductor session. All experiments (379-388) returned status='partial' "
            "with 'Extended GPU runtime needed'. "
            "RETRO-019 (critical escalation) opened."
        ),
        "precision_result_credible": (
            "Exp 379 status='partial'. Script and tests written (22 tests). "
            "honest_verdict requires live inference. Upstream: RETRO-019."
        ),
        "humaneval_result_credible": (
            "Exp 380 status='partial'. Script and tests written (24 tests). "
            "honest_verdict requires live inference. Upstream: RETRO-019."
        ),
        "adversarial_result_credible": (
            "Exp 381 status='partial'. Script created. "
            "Live Apple adversarial GSM8K benchmark pending GPU runtime. Upstream: RETRO-019."
        ),
        "extraction_winner_known": (
            "Exp 382 status='partial'. Script created. "
            "LLMExtractor vs ArithmeticExtractor comparison pending GPU runtime. "
            "Upstream: RETRO-019."
        ),
        "fr11_learning_confirmed": (
            "Exp 384 status='partial'. Script and tests created. "
            "learning_confirmed requires live relay inference. "
            "Synthetic confirmation (0.60→0.72) from Exp 361 stands. "
            "FR-11 still open. Third milestone carry. Upstream: RETRO-019."
        ),
        "jitrl_memory_works": (
            "Exp 386 missing — session interrupted before implementation. "
            "JitRL threshold modulation (arXiv 2601.18510) not evaluated. "
            "No artifacts on disk."
        ),
        "safety_kan_works": (
            "Exp 387 missing — session interrupted before implementation. "
            "Safety KAN AUC-ROC benchmark not evaluated. No artifacts on disk."
        ),
        "saver_live_verified": (
            "Exp 388 status='partial'. Script and tests created. "
            "SAVeR live multi-turn verification pending GPU runtime. Upstream: RETRO-019."
        ),
        "cikan_implemented": (
            "Exp 378 missing — session interrupted before implementation. "
            "cikan_energy.py on disk contains JSON from Exp 375 (corrupt, RETRO-018). "
            "CIKANEnergy class does not exist. RETRO-020 opened (second failure)."
        ),
    }

    timing_analysis: dict[str, Any] = {
        "mean_exp_duration_min": retro.mean_exp_duration_min,
        "prev_mean_exp_duration_min": PREV_MEAN_EXP_DURATION_MIN,
        "estimated_speedup_pct": speedup_pct,
        "speedup_interpretation": (
            f"Mean duration {retro.mean_exp_duration_min:.1f} min vs prior "
            f"{PREV_MEAN_EXP_DURATION_MIN} min. "
            "Missing experiments (378, 386, 387 — zero wall time from session interrupt) "
            "deflate the mean. Partial experiments (script-only, no live results) deflate "
            "further. The apparent speedup does not reflect useful GPU work — it reflects "
            "a truncated session and fast-fail blocked states."
        ),
        "n_experiments_ran": timing_stats["n_ran"],
        "n_experiments_blocked": timing_stats["n_blocked"],
        "total_wall_time_min": timing_stats["total_min"],
        "slowest_experiment": timing_stats["slowest"],
        "fastest_experiment": timing_stats["fastest"],
    }

    # The milestone's headline question: did we FINALLY get live GPU results?
    first_live_gpu_results_achieved = retro.live_gpu_confirmed

    # Headline results map: experiment → honest_verdict if live_gpu
    # Empty in this milestone because no live GPU inference occurred
    headline_results: dict[str, Any] = {}

    return {
        "schema": "carnot.operational_retro.v3",
        "milestone": MILESTONE,
        "title": f"Operational Retrospective — Milestone {MILESTONE}",
        "retro_type": "full_milestone",
        "note": (
            f"Retrospective covering milestone {MILESTONE} experiments (Exps 377-388). "
            "Schema v3 adds first_live_gpu_results_achieved flag, headline_results dict, "
            "JitRL/Safety KAN/SAVeR criteria. Session was interrupted — Exps 378, 386, 387 "
            "are missing. Prior milestone retro: results/operational_retro_2026_05_27.json."
        ),
        "success_criteria": success_criteria,
        "explanations": explanations,
        "timing_analysis": timing_analysis,
        "retro_items_opened": retro.retro_items_opened,
        "new_retro_items": NEW_RETRO_ITEMS,
        "first_live_gpu_results_achieved": first_live_gpu_results_achieved,
        "headline_results": headline_results,
        "estimated_savings_next_pct": 35,
        "estimated_savings_rationale": (
            "RETRO-019 (live GPU — fifth escalation): resolving this alone unblocks 9+ "
            "experiments currently partial/missing. Each transitions to live inference with "
            "real results (~30% of savings). RETRO-020 (CIKAN): schedule as experiment 1 in "
            "next milestone, 2–3% savings from eliminating carry. RETRO-021 (FR-11 relay): "
            "live relay runs immediately once GPU is online, ~5% from eliminating re-runs. "
            "If GPU is confirmed online before the next session starts, cumulative benefit "
            "could reach 40% as the backlog of 9 waiting experiments finally produces results."
        ),
        "meta_reflection": (
            "Milestone 2026.06.03 reveals a persistent gap between infrastructure readiness "
            "and execution-environment availability. Exp 377 correctly solved the software "
            "wiring problem (CARNOT_FORCE_LIVE=1 propagation). But the GPU node must be "
            "physically online for any of this to matter. "
            "The research machinery is structurally complete: 1000+ tests pass, all GPU "
            "experiment scripts are written and tested, LiveGPUGate correctly gates them. "
            "The sole remaining bottleneck is 'is the GPU node online when the conductor "
            "session starts?'. "
            "Process improvement: add a mandatory pre-flight check to the session start "
            "procedure — 'nvidia-smi | grep GPU' must succeed before any experiment work "
            "begins. If it fails, stop and fix GPU availability first. "
            "The session was also interrupted (three experiments missing: 378, 386, 387). "
            "Consider implementing a session resume mechanism or reducing milestone experiment "
            "count to ensure all experiments complete in a single uninterrupted session."
        ),
        "key_findings": {
            "retro_015_fix_status": (
                "RETRO-015 infrastructure fix (Exp 377) is CORRECT and VERIFIED. "
                "LiveGPUGate class + session_startup.sh export of CARNOT_FORCE_LIVE=1 "
                "is the right solution. The env var now propagates to subprocesses. "
                "Problem: the GPU node must be online. Infrastructure is fixed; "
                "execution environment is not."
            ),
            "live_gpu_milestone_answer": (
                "NO — live GPU results were NOT achieved in milestone 2026.06.03. "
                "Five consecutive milestones (2026.05.06 through 2026.06.03) have produced "
                "zero provenance-bearing live inference results. The research program's "
                "credibility claims remain entirely simulation-based."
            ),
            "session_interrupt": (
                "Conductor session was interrupted before completing Exps 378, 386, 387. "
                "Checkpoint commit ('preserve uncommitted work from interrupted run') was "
                "created. Three experiments are fully missing from the milestone."
            ),
            "cikan_status": (
                "CIKANEnergy class remains unimplemented after two milestone attempts. "
                "cikan_energy.py still contains JSON from Exp 375. RETRO-020 opened."
            ),
            "retro_items_net": (
                f"Closed: RETRO-015 (infrastructure fix — live GPU still fails in practice). "
                f"Opened: {', '.join(retro.retro_items_opened)}. "
                f"Net: {len(retro.retro_items_opened)} new RETRO items."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the retrospective, write the artifact, mark milestone 2026.06.03 COMPLETE."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tmpl = ExperimentTemplate(
        389,
        f"Operational Retrospective — Milestone {MILESTONE}",
        DELIVERABLE,
    )
    tmpl.setup()

    result_files = load_milestone_results(tmpl._repo_root, RESULT_FILE_MAP)
    retro = compute_retro_2026_06_03(result_files, tmpl._repo_root)
    artifact_data = build_retro_artifact(retro)

    artifact = tmpl.build_result(artifact_data, status="success")

    output_path = tmpl._repo_root / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))

    _log.info("Retrospective written to %s", output_path)
    print(f"\n{'='*60}")
    print(f"MILESTONE {MILESTONE} RETROSPECTIVE COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"\nSuccess criteria:")
    for field in dataclasses.fields(retro):
        val = getattr(retro, field.name)
        if isinstance(val, bool):
            status_str = "PASS" if val else "FAIL"
            print(f"  [{status_str}] {field.name}")
    print(f"\nFirst live GPU results achieved: {retro.live_gpu_confirmed}")
    print(f"New RETRO items: {retro.retro_items_opened}")
    print(
        f"Mean exp duration: {retro.mean_exp_duration_min:.1f} min "
        f"(prev: {PREV_MEAN_EXP_DURATION_MIN} min, "
        f"speedup: {estimate_speedup_pct(PREV_MEAN_EXP_DURATION_MIN, retro.mean_exp_duration_min):.1f}%)"
    )
    print(f"\nMILESTONE {MILESTONE} MARKED COMPLETE.")


if __name__ == "__main__":
    main()
