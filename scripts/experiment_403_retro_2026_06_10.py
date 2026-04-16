#!/usr/bin/env python3
"""Exp 403: Operational Retrospective — Milestone 2026.06.10.

**Researcher summary:**
    Milestone 2026.06.10 targeted "Credible Live GPU Results — Break the Simulated Barrier
    Once and For All." It opened with RETRO-019 (CRITICAL): after five consecutive milestones
    without live GPU inference, the GPU node had to be confirmed online before any other
    work began.

    At session start, Exp 390 was run as the preflight gate. The result JSON shows:
        {"experiment": 390, "status": "complete", "finding": "GPU preflight script created."}
    The finding indicates the script was (re-)created, not that GPU was confirmed live.
    honest_verdict="gpu_confirmed_live" is absent — the GPU node was again not available.
    This is the SIXTH consecutive milestone without live GPU results.

    As a consequence, all GPU-dependent experiments (391-402) ran in "deliverable already
    exists" fast-path mode — pre-existing script artifacts were confirmed present but no
    actual inference ran:
    - Exp 391: CIKANEnergy — "Deliverable already exists" (script created in prior session
      but cikan_energy.py on disk still contains the corrupt JSON from Exp 375).
    - Exps 392-393: JitRL memory + Safety KAN — "Deliverable already exists" (no result JSON).
    - Exps 394-402: All status="partial" with no honest_verdict — pipeline scripts exist but
      live GPU was never available to execute them.

    RETRO-022 (CRITICAL) is opened as a HUMAN ESCALATION item. The conductor cannot fix a
    powered-off GPU node programmatically. Human action is required before the next milestone:
    cloud GPU rental (Lambda, vast.ai), RTX 4090 purchase, or physical connection to the
    node already in the rack. This is the last conductor-automated retro before requiring
    manual intervention.

**Milestone 2026.06.10 experiment inventory (Exps 390-402):**

    Exp 390: GPU node preflight — RETRO-019 action (COMPLETE — but not gpu_confirmed_live)
    Exp 391: Fix RETRO-020 — CIKANEnergy Python class (PARTIAL — deliverable already exists)
    Exp 392: JitRL constraint memory — threshold modulation (MISSING result JSON)
    Exp 393: Safety/Jailbreak KAN Classifier (MISSING result JSON)
    Exp 394: Live precision pipeline — 200 GSM8K × 5 variants × 2 models (PARTIAL)
    Exp 395: Live HumanEval code verification — 50 problems (PARTIAL)
    Exp 396: Live adversarial GSM8K — Carnot's headline benchmark (PARTIAL)
    Exp 397: Live extraction comparison — LLMExtractor vs ArithmeticExtractor (PARTIAL)
    Exp 398: Combined EORM+JEPA retrain on live pairs (PARTIAL)
    Exp 399: FR-11 self-learning relay live — first learning_confirmed verdict (PARTIAL)
    Exp 400: SAVeR live multi-turn verification (PARTIAL)
    Exp 401: Semantic Energy hallucination scorer (MISSING result JSON)
    Exp 402: CRANE extraction gate (MISSING result JSON)

**Wall-time estimates (from conductor log timestamps, 2026-04-16 session):**
    07:44 UTC: Milestone 2026.06.10 activated.
    08:21 UTC: Exps 390-393 batch done (~37 min / 4 experiments ≈ 9 min/exp).
    09:22 UTC: Exps 394-402 batch done (~61 min / 9 experiments ≈ 7 min/exp).
    Total: ~98 min for 13 experiments, mean ≈ 7.5 min/exp.
    All experiments ran in fast-path mode ("Deliverable already exists"), so wall times
    are dominated by conductor overhead, not actual inference work.

**Key question answer:** After SIX milestones and 403 experiments, did we FINALLY get
    credible live GPU results? NO. first_live_gpu_results_achieved=False.
    RETRO-022 is a HUMAN ACTION ESCALATION — the conductor cannot resolve GPU availability.

**Output:** results/operational_retro_2026_06_10.json

Spec: REQ-INFRA-017/018/019 (LiveGPUGate, preflight),
      REQ-LEARN-025/026/027 (EORM/relay retrain),
      REQ-BENCH-003/004/006/007 (precision/HumanEval/adversarial benchmarks),
      REQ-EXTRACT-023 (extraction comparison), REQ-AGENT-001/002 (SAVeR),
      REQ-EBM-031 (semantic energy), REQ-EXTRACT-025 (CRANE)
SCENARIO: RETRO-2026.06.10
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

MILESTONE = "2026.06.10"
DELIVERABLE = "results/operational_retro_2026_06_10.json"

# Mean experiment duration from the previous milestone (2026.06.03 retro, Exp 389).
# Includes fast-failed blocked experiments.
PREV_MEAN_EXP_DURATION_MIN: float = 14.0

# Mapping of experiment result files (keys are string IDs for flexibility).
# None means the deliverable was a source module, not a JSON artifact.
RESULT_FILE_MAP: dict[str, str | None] = {
    "390": "results/experiment_390_gpu_preflight.json",
    "391": "results/experiment_391_cikan_energy.json",
    "392": "results/experiment_392_jitrl_memory.json",
    "393": "results/experiment_393_safety_kan.json",
    "394": "results/experiment_394_precision_live.json",
    "395": "results/experiment_395_humaneval_live.json",
    "396": "results/experiment_396_adversarial_live.json",
    "397": "results/experiment_397_extraction_live.json",
    "398": "results/experiment_398_retrain_live.json",
    "399": "results/experiment_399_relay_live.json",
    "400": "results/experiment_400_saver_live.json",
    "401": "results/experiment_401_semantic_energy.json",
    "402": "results/experiment_402_crane_extractor.json",
}

# Full experiment metadata for wall-time statistics.
# Wall times estimated from conductor log timestamps (2026-04-16 session).
# Missing experiments (392, 393, 401, 402) contribute 0 wall time — no result JSON found.
MILESTONE_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": 390,
        "title": "GPU node preflight — RETRO-019 action",
        "result_file": "results/experiment_390_gpu_preflight.json",
        "wall_time_min": 9,    # 07:44 → 08:21 batch / 4 experiments
        "status": "completed",
        "note": (
            "status='complete' but finding='GPU preflight script created.' — NOT "
            "'gpu_confirmed_live'. Script was re-confirmed present; GPU hardware was "
            "not available. RETRO-019 NOT resolved."
        ),
    },
    {
        "id": 391,
        "title": "Fix RETRO-020 — CIKANEnergy Python class (third attempt)",
        "result_file": "results/experiment_391_cikan_energy.json",
        "wall_time_min": 9,
        "status": "partial",
        "note": (
            "Deliverable already exists (fast path). cikan_energy.py still contains "
            "corrupt JSON from Exp 375. No result JSON produced. RETRO-020 NOT closed."
        ),
    },
    {
        "id": 392,
        "title": "JitRL constraint memory — threshold modulation (arXiv 2601.18510)",
        "result_file": "results/experiment_392_jitrl_memory.json",
        "wall_time_min": 9,
        "status": "missing",
        "note": (
            "Deliverable already exists (fast path). No result JSON. "
            "threshold_modulation_works cannot be evaluated."
        ),
    },
    {
        "id": 393,
        "title": "Safety/Jailbreak KAN Classifier — first AUROC benchmark",
        "result_file": "results/experiment_393_safety_kan.json",
        "wall_time_min": 9,
        "status": "missing",
        "note": (
            "Deliverable already exists (fast path). No result JSON. "
            "test_auroc cannot be evaluated."
        ),
    },
    {
        "id": 394,
        "title": "Live precision pipeline — 200 GSM8K, 5 variants, 2 models",
        "result_file": "results/experiment_394_precision_live.json",
        "wall_time_min": 7,    # 08:21 → 09:22 batch / 9 experiments ≈ 7 min each
        "status": "partial",
        "note": (
            "status='partial'. GPU preflight gate blocked. No honest_verdict. "
            "Live precision numbers NOT produced."
        ),
    },
    {
        "id": 395,
        "title": "Live HumanEval code verification — 50 problems",
        "result_file": "results/experiment_395_humaneval_live.json",
        "wall_time_min": 7,
        "status": "partial",
        "note": "status='partial'. Blocked — no live GPU. No honest_verdict.",
    },
    {
        "id": 396,
        "title": "Live adversarial GSM8K — Carnot's headline benchmark",
        "result_file": "results/experiment_396_adversarial_live.json",
        "wall_time_min": 7,
        "status": "partial",
        "note": "status='partial'. Blocked — no live GPU. No honest_verdict.",
    },
    {
        "id": 397,
        "title": "Live extraction comparison — LLMExtractor vs ArithmeticExtractor",
        "result_file": "results/experiment_397_extraction_live.json",
        "wall_time_min": 7,
        "status": "partial",
        "note": (
            "status='partial'. Blocked — no live GPU. RETRO-016 (extraction winner) "
            "still unresolved."
        ),
    },
    {
        "id": 398,
        "title": "Combined EORM+JEPA retrain on live pairs",
        "result_file": "results/experiment_398_retrain_live.json",
        "wall_time_min": 7,
        "status": "partial",
        "note": "status='partial'. Blocked — no live GPU. No honest_verdict.",
    },
    {
        "id": 399,
        "title": "FR-11 self-learning relay live — first learning_confirmed verdict",
        "result_file": "results/experiment_399_relay_live.json",
        "wall_time_min": 7,
        "status": "partial",
        "note": (
            "status='partial'. Blocked — no live GPU. honest_verdict='learning_confirmed' "
            "NOT achieved. RETRO-021 NOT closed. Fourth consecutive miss."
        ),
    },
    {
        "id": 400,
        "title": "SAVeR live multi-turn verification",
        "result_file": "results/experiment_400_saver_live.json",
        "wall_time_min": 7,
        "status": "partial",
        "note": "status='partial'. Blocked — no live GPU. live_verification_active=False.",
    },
    {
        "id": 401,
        "title": "Semantic Energy hallucination scorer — logit-lens AUROC benchmark",
        "result_file": "results/experiment_401_semantic_energy.json",
        "wall_time_min": 0,    # No result JSON found; session may have not written it
        "status": "missing",
        "note": "No result JSON found. auroc cannot be evaluated.",
    },
    {
        "id": 402,
        "title": "CRANE extraction gate — alternating free-energy extraction",
        "result_file": "results/experiment_402_crane_extractor.json",
        "wall_time_min": 0,
        "status": "missing",
        "note": "No result JSON found. detection_rate vs ArithmeticExtractor cannot be evaluated.",
    },
]


# ---------------------------------------------------------------------------
# Dataclass: milestone success criteria
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MilestoneRetro2026_06_10:
    """All measurable success criteria for milestone 2026.06.10.

    Each boolean field corresponds to a primary goal stated in the milestone plan.
    False does NOT mean "failed completely" — it means the criterion was not met
    as defined. Context is captured in build_retro_artifact().

    Fields
    ------
    retro_019_resolved : bool
        Exp 390 result JSON has honest_verdict == "gpu_confirmed_live", meaning the GPU
        node was physically online and confirmed live during the preflight check.
        "complete" status alone is NOT sufficient — the GPU itself must have been confirmed.

    retro_020_closed : bool
        Exp 391 result JSON exists with status == "success" AND
        python/carnot/models/cikan_energy.py contains a valid Python class CIKANEnergy
        (i.e., not a JSON artifact or placeholder). Third consecutive attempt.

    retro_021_closed : bool
        Exp 399 result JSON has honest_verdict == "learning_confirmed" AND
        inference_mode == "live_gpu". FR-11 self-learning relay verdict on live data.
        Fourth consecutive target.

    live_gpu_confirmed : bool
        At least one experiment in Exps 390-402 reported inference_mode == "live_gpu".
        This is the milestone-level question: did ANY live GPU inference happen?

    precision_result_credible : bool
        Exp 394 result has honest_verdict == "live_improvement" AND
        inference_mode == "live_gpu".

    humaneval_result_credible : bool
        Exp 395 result has honest_verdict == "code_verification_positive" AND
        inference_mode == "live_gpu".

    adversarial_result_credible : bool
        Exp 396 result has honest_verdict == "improvement_positive" AND
        inference_mode == "live_gpu".

    extraction_winner_known : bool
        Exp 397 result has honest_verdict in ("live_gpu_winner", "live_gpu_no_improvement")
        AND inference_mode == "live_gpu". Closes RETRO-016 if True.

    fr11_learning_confirmed : bool
        Exp 399 result has honest_verdict == "learning_confirmed" AND
        inference_mode == "live_gpu". Same as retro_021_closed — aliased for clarity.

    jitrl_memory_works : bool
        Exp 392 result has threshold_modulation_works == True, indicating JitRL-style
        threshold modulation produced measurable self-improvement (arXiv 2601.18510).

    safety_kan_works : bool
        Exp 393 result reports test_auroc > 0.70 on the held-out safety evaluation set.

    saver_live_verified : bool
        Exp 400 result has inference_mode == "live_gpu" and live_verification_active == True.

    semantic_energy_viable : bool
        Exp 401 result has auroc > 0.70, indicating semantic energy is a viable
        hallucination scoring mechanism.

    crane_extraction_improved : bool
        Exp 402 result has detection_rate > that of ArithmeticExtractor baseline, meaning
        CRANE's alternating free-energy extraction beats the arithmetic heuristic.

    mean_exp_duration_min : float
        Mean wall time across all milestone experiments (including zero-duration missing ones).

    n_experiments_blocked : int
        Number of experiments with status == "blocked" or "missing" in MILESTONE_EXPERIMENTS.

    retro_items_opened : list[str]
        IDs of new RETRO items opened by this retrospective.

    headline_results : dict
        Experiment-keyed summary of key numbers when live_gpu_confirmed=True.
        Empty dict when live_gpu_confirmed=False (no publishable results).
    """

    retro_019_resolved: bool
    retro_020_closed: bool
    retro_021_closed: bool
    live_gpu_confirmed: bool
    precision_result_credible: bool
    humaneval_result_credible: bool
    adversarial_result_credible: bool
    extraction_winner_known: bool
    fr11_learning_confirmed: bool
    jitrl_memory_works: bool
    safety_kan_works: bool
    saver_live_verified: bool
    semantic_energy_viable: bool
    crane_extraction_improved: bool
    mean_exp_duration_min: float
    n_experiments_blocked: int
    retro_items_opened: list[str]
    headline_results: dict


# ---------------------------------------------------------------------------
# New RETRO items opened by this retrospective
# ---------------------------------------------------------------------------

NEW_RETRO_ITEMS: list[dict[str, Any]] = [
    {
        "id": "RETRO-022",
        "title": (
            "Live GPU NEVER ran across SIX consecutive milestones — HUMAN ACTION REQUIRED (CRITICAL)"
        ),
        "status": "NEW",
        "priority": "critical",
        "description": (
            "Exp 390 ran as the RETRO-019 preflight gate for this milestone. Its result "
            "shows status='complete' but finding='GPU preflight script created.' — NOT "
            "'gpu_confirmed_live'. The GPU node was again not online during the conductor "
            "session. This is the SIXTH consecutive milestone (2026.05.06, 2026.05.13, "
            "2026.05.20, 2026.05.27, 2026.06.03, 2026.06.10) with zero live GPU inference. "
            "After 403 experiments, Carnot has no publishable live GPU results. "
            "The infrastructure is correct: CARNOT_FORCE_LIVE=1 propagation (Exp 377), "
            "LiveGPUGate (Exp 352), GPU preflight (Exp 390), conductor_gpu_env.sh (Exp 365) "
            "are all in place. The RTX 3090s are confirmed capable (Exp 352: "
            "is_live_capable=True). The ONLY remaining blocker is that the GPU node is "
            "NOT POWERED ON during conductor sessions. This cannot be fixed by code. "
            "The conductor cannot resolve hardware availability programmatically. "
            "HUMAN ACTION IS REQUIRED before the next milestone begins."
        ),
        "root_cause": (
            "GPU node is physically offline (or unreachable over the network) when the "
            "conductor process runs. The conductor runs on a CPU-only host. The GPU node "
            "requires a separate physical power-on step that only a human operator can perform."
        ),
        "fix": (
            "OPTION A — Cloud GPU (Recommended, Fastest): Rent a GPU instance on Lambda "
            "Labs, vast.ai, or RunPod with a single NVIDIA RTX 3090 or A100. Clone the "
            "carnot repo. Source scripts/session_startup.sh. Run experiment_390 to confirm "
            "preflight. Then re-run experiments 394-402 sequentially. Cost: ~$0.50-2/hr. "
            "Expected time to first live results: < 4 hours. "
            "\n"
            "OPTION B — RTX 4090 Purchase: Purchase and install an RTX 4090 in the same "
            "machine as the conductor host (or a new machine). This eliminates the "
            "network dependency entirely. Cost: ~$1800 USD. "
            "\n"
            "OPTION C — Physical Connection: Power on the existing RTX 3090 node that "
            "Exp 352 confirmed as capable (is_live_capable=True). Ensure it is reachable "
            "from the conductor host. Run: nvidia-smi; source scripts/session_startup.sh; "
            "python scripts/experiment_390_gpu_preflight.py. Confirm honest_verdict == "
            "'gpu_confirmed_live' before proceeding."
        ),
        "estimated_savings_pct": 40,
        "rationale": (
            "Resolving GPU availability unblocks Exps 394-402 (9 experiments) plus any "
            "future live-dependent work. These experiments have been in 'partial' status "
            "across 3-4 milestones and are fully written and tested — they just need GPU "
            "runtime. Closing this retro item is the single highest-leverage action "
            "available to the project."
        ),
    },
    {
        "id": "RETRO-023",
        "title": "CIKANEnergy not implemented — THIRD consecutive milestone failure",
        "status": "NEW",
        "priority": "high",
        "description": (
            "Exp 391 (this milestone's attempt to close RETRO-020) ran in "
            "'deliverable already exists' fast-path mode. cikan_energy.py on disk still "
            "contains the corrupt JSON artifact from Exp 375: "
            "{\"experiment\": 375, \"status\": \"partial\", ...}. "
            "No valid Python class CIKANEnergy exists anywhere in the codebase. "
            "This is the THIRD consecutive milestone where CIKAN was targeted (Exp 375 "
            "in 2026.05.27, Exp 378 in 2026.06.03, Exp 391 in 2026.06.10) and not delivered. "
            "The constraint-informed KAN energy tier (arXiv 2412.03710) remains unimplemented. "
            "Energy separation ratio vs standard KAN cannot be measured."
        ),
        "root_cause": (
            "The 'deliverable already exists' fast path in the conductor fires when a result "
            "file is present at the expected path, even if that file is corrupt or incomplete. "
            "cikan_energy.py has been a corrupt JSON file since Exp 375, so every subsequent "
            "attempt was fast-pathed past without actually checking file content."
        ),
        "fix": (
            "1. Delete the corrupt cikan_energy.py: "
            "   rm python/carnot/models/cikan_energy.py "
            "2. Write a fresh cikan_energy.py with a proper CIKANEnergy Python class. "
            "3. Write tests/python/test_cikan_energy.py with 100% coverage. "
            "4. Write results/experiment_391_cikan_energy.json with status='success'. "
            "5. Verify: python -c 'from carnot.models.cikan_energy import CIKANEnergy' "
            "The conductor's 'deliverable already exists' check must be enhanced to validate "
            "Python source file content (not just presence of a file at the path)."
        ),
        "estimated_savings_pct": 3,
    },
    {
        "id": "RETRO-024",
        "title": "FR-11 self-learning relay unconfirmed on live data — FOURTH consecutive milestone",
        "status": "NEW",
        "priority": "high",
        "description": (
            "Exp 399 (FR-11 relay live) returned status='partial'. honest_verdict="
            "'learning_confirmed' NOT achieved. FR-11 is a mandatory functional requirement: "
            "the self-learning relay must produce learning_confirmed on live GPU inference. "
            "This has been the target since milestone 2026.05.20 (Exp 361 confirmed synthetic, "
            "honest_verdict=synthetic_only). FOUR consecutive milestones (2026.05.27, "
            "2026.06.03, 2026.06.10, and the underlying 2026.05.20 miss) have failed to "
            "produce a live relay verdict. "
            "The relay machinery is sound (54 tests, 100% module coverage, Exp 361 ran "
            "0.60→0.72 accuracy on synthetic data). The relay script (Exp 384/399) already "
            "exists with LiveGPUGate hard-gating. The ONLY blocker is RETRO-022 "
            "(live GPU unavailability). Once RETRO-022 is resolved, Exp 399 should complete "
            "within one conductor run — the script and tests already exist."
        ),
        "root_cause": "RETRO-022 (live GPU unavailability) is the sole upstream blocker.",
        "fix": "Close RETRO-022 first. Then re-run Exp 399 with CARNOT_FORCE_LIVE=1.",
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
        Mapping from experiment key (e.g. "390") to relative result file path.
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
    (no result JSON) contribute zero minutes and are counted as blocked.
    Partial experiments (scripts exist, no live results) are included normally.

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
        Previous milestone mean experiment duration in minutes (14.0 for 2026.06.03).
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

    A JSON file at the expected path (as produced by corrupt Exp 375) returns False.
    Exp 391 result JSON must also show status == 'success'.

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
        (c) contains 'class CIKANEnergy', AND (d) Exp 391 result has status=success.
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

    exp391 = results.get("391")
    if exp391 is None:
        return False
    if exp391.get("status") != "success":
        return False

    return True


# ---------------------------------------------------------------------------
# Success criteria computation
# ---------------------------------------------------------------------------


def compute_retro_2026_06_10(
    result_files: dict[str, dict[str, Any] | None],
    repo_root: Path,
) -> MilestoneRetro2026_06_10:
    """Evaluate all success criteria for milestone 2026.06.10.

    Each criterion is evaluated independently from the result file data.
    Criteria that require live GPU inference are False when any upstream result
    reports inference_mode != "live_gpu" or has no inference_mode at all.

    Parameters
    ----------
    result_files : dict
        Loaded result JSONs keyed by experiment number string (e.g. "390").
    repo_root : Path
        Repository root, used for file-system checks (e.g. cikan_energy.py).

    Returns
    -------
    MilestoneRetro2026_06_10
        Populated dataclass with all criteria evaluated.
    """
    def get(key: str) -> dict[str, Any]:
        return result_files.get(key) or {}

    # --- retro_019_resolved: GPU confirmed live in preflight ---
    retro_019_resolved = get("390").get("honest_verdict") == "gpu_confirmed_live"

    # --- CIKAN check (shared by retro_020_closed) ---
    cikan_impl = _check_cikan_implemented(repo_root, result_files)
    retro_020_closed = cikan_impl

    # --- live_gpu_confirmed: ANY experiment reported live_gpu mode ---
    live_gpu_confirmed = any(
        (v or {}).get("inference_mode") == "live_gpu"
        for v in result_files.values()
    )

    # --- precision_result_credible ---
    p394 = get("394")
    precision_result_credible = (
        p394.get("honest_verdict") == "live_improvement"
        and p394.get("inference_mode") == "live_gpu"
    )

    # --- humaneval_result_credible ---
    p395 = get("395")
    humaneval_result_credible = (
        p395.get("honest_verdict") == "code_verification_positive"
        and p395.get("inference_mode") == "live_gpu"
    )

    # --- adversarial_result_credible ---
    p396 = get("396")
    adversarial_result_credible = (
        p396.get("honest_verdict") == "improvement_positive"
        and p396.get("inference_mode") == "live_gpu"
    )

    # --- extraction_winner_known: either live verdict is acceptable ---
    p397 = get("397")
    extraction_winner_known = (
        p397.get("honest_verdict") in ("live_gpu_winner", "live_gpu_no_improvement")
        and p397.get("inference_mode") == "live_gpu"
    )

    # --- fr11_learning_confirmed ---
    p399 = get("399")
    fr11_learning_confirmed = (
        p399.get("honest_verdict") == "learning_confirmed"
        and p399.get("inference_mode") == "live_gpu"
    )

    # --- retro_021_closed: same condition as fr11_learning_confirmed ---
    retro_021_closed = fr11_learning_confirmed

    # --- jitrl_memory_works ---
    p392 = get("392")
    jitrl_memory_works = p392.get("threshold_modulation_works") is True

    # --- safety_kan_works ---
    p393 = get("393")
    try:
        test_auroc = float(p393.get("test_auroc", 0))
    except (TypeError, ValueError):
        test_auroc = 0.0
    safety_kan_works = test_auroc > 0.70

    # --- saver_live_verified ---
    p400 = get("400")
    saver_live_verified = (
        p400.get("inference_mode") == "live_gpu"
        and p400.get("live_verification_active") is True
    )

    # --- semantic_energy_viable ---
    p401 = get("401")
    try:
        auroc_401 = float(p401.get("auroc", 0))
    except (TypeError, ValueError):
        auroc_401 = 0.0
    semantic_energy_viable = auroc_401 > 0.70

    # --- crane_extraction_improved ---
    p402 = get("402")
    crane_extraction_improved = p402.get("crane_beats_arithmetic") is True

    # --- timing stats ---
    timing_stats = compute_timing_stats(MILESTONE_EXPERIMENTS)
    mean_exp_duration_min = timing_stats["mean_min"]
    n_experiments_blocked = timing_stats["n_blocked"]

    # --- RETRO items opened ---
    retro_items_opened: list[str] = []
    if not live_gpu_confirmed:
        retro_items_opened.append("RETRO-022")
    if not retro_020_closed:
        retro_items_opened.append("RETRO-023")
    if not retro_021_closed:
        retro_items_opened.append("RETRO-024")
    # RETRO-016 is closed if extraction winner is known
    if extraction_winner_known:
        retro_items_opened.append("RETRO-016_CLOSE")

    # --- headline_results: only populated when live GPU ran ---
    headline_results: dict[str, Any] = {}
    if live_gpu_confirmed:
        if precision_result_credible:
            headline_results["exp_394_precision"] = {
                "honest_verdict": p394.get("honest_verdict"),
                "inference_mode": p394.get("inference_mode"),
                "signed_improvement": p394.get("signed_improvement"),
            }
        if humaneval_result_credible:
            headline_results["exp_395_humaneval"] = {
                "honest_verdict": p395.get("honest_verdict"),
                "inference_mode": p395.get("inference_mode"),
                "pass_at_1": p395.get("pass_at_1"),
            }
        if adversarial_result_credible:
            headline_results["exp_396_adversarial"] = {
                "honest_verdict": p396.get("honest_verdict"),
                "inference_mode": p396.get("inference_mode"),
                "improvement_pp": p396.get("improvement_pp"),
            }
        if extraction_winner_known:
            headline_results["exp_397_extraction"] = {
                "honest_verdict": p397.get("honest_verdict"),
                "inference_mode": p397.get("inference_mode"),
                "winner": p397.get("winner"),
            }
        if fr11_learning_confirmed:
            headline_results["exp_399_relay"] = {
                "honest_verdict": p399.get("honest_verdict"),
                "inference_mode": p399.get("inference_mode"),
                "accuracy_improvement": p399.get("accuracy_improvement"),
            }

    return MilestoneRetro2026_06_10(
        retro_019_resolved=retro_019_resolved,
        retro_020_closed=retro_020_closed,
        retro_021_closed=retro_021_closed,
        live_gpu_confirmed=live_gpu_confirmed,
        precision_result_credible=precision_result_credible,
        humaneval_result_credible=humaneval_result_credible,
        adversarial_result_credible=adversarial_result_credible,
        extraction_winner_known=extraction_winner_known,
        fr11_learning_confirmed=fr11_learning_confirmed,
        jitrl_memory_works=jitrl_memory_works,
        safety_kan_works=safety_kan_works,
        saver_live_verified=saver_live_verified,
        semantic_energy_viable=semantic_energy_viable,
        crane_extraction_improved=crane_extraction_improved,
        mean_exp_duration_min=mean_exp_duration_min,
        n_experiments_blocked=n_experiments_blocked,
        retro_items_opened=retro_items_opened,
        headline_results=headline_results,
    )


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_retro_artifact(retro: MilestoneRetro2026_06_10) -> dict[str, Any]:
    """Build the final JSON artifact for milestone 2026.06.10 retrospective.

    The artifact documents every success criterion, all new RETRO items,
    the timing analysis, and — critically — whether live GPU results were
    achieved for the first time.

    Parameters
    ----------
    retro : MilestoneRetro2026_06_10
        Evaluated milestone success criteria.

    Returns
    -------
    dict
        JSON-serializable artifact with schema "carnot.operational_retro.v4".
    """
    speedup_pct = estimate_speedup_pct(PREV_MEAN_EXP_DURATION_MIN, retro.mean_exp_duration_min)
    timing_stats = compute_timing_stats(MILESTONE_EXPERIMENTS)

    # Build new retro items filtered to what was actually opened
    opened_ids = set(retro.retro_items_opened)
    # Include all RETRO items that were conditionally opened (not RETRO-016_CLOSE which is a close)
    new_retro_items_for_artifact = [
        item for item in NEW_RETRO_ITEMS
        if item["id"] in opened_ids
    ]

    # Closed RETRO items
    retro_items_closed: list[str] = []
    if "RETRO-016_CLOSE" in opened_ids:
        retro_items_closed.append("RETRO-016")

    return {
        "schema": "carnot.operational_retro.v4",
        "milestone": MILESTONE,
        "retro_type": "full_milestone",
        "first_live_gpu_results_achieved": retro.live_gpu_confirmed,
        "success_criteria": {
            "retro_019_resolved": retro.retro_019_resolved,
            "retro_020_closed": retro.retro_020_closed,
            "retro_021_closed": retro.retro_021_closed,
            "live_gpu_confirmed": retro.live_gpu_confirmed,
            "precision_result_credible": retro.precision_result_credible,
            "humaneval_result_credible": retro.humaneval_result_credible,
            "adversarial_result_credible": retro.adversarial_result_credible,
            "extraction_winner_known": retro.extraction_winner_known,
            "fr11_learning_confirmed": retro.fr11_learning_confirmed,
            "jitrl_memory_works": retro.jitrl_memory_works,
            "safety_kan_works": retro.safety_kan_works,
            "saver_live_verified": retro.saver_live_verified,
            "semantic_energy_viable": retro.semantic_energy_viable,
            "crane_extraction_improved": retro.crane_extraction_improved,
            "n_experiments_blocked": retro.n_experiments_blocked,
        },
        "headline_results": retro.headline_results,
        "timing_analysis": {
            "mean_exp_duration_min": retro.mean_exp_duration_min,
            "prev_milestone_mean_min": PREV_MEAN_EXP_DURATION_MIN,
            "estimated_speedup_pct": speedup_pct,
            "total_min": timing_stats["total_min"],
            "n_experiments": timing_stats["n_ran"],
            "n_blocked": timing_stats["n_blocked"],
            "slowest": timing_stats["slowest"],
            "fastest": timing_stats["fastest"],
            "note": (
                "Apparent speedup vs prior milestone (14.0 min) is attributable to all "
                "experiments running in 'deliverable already exists' fast-path mode. "
                "No actual inference work occurred. This speedup is not a genuine "
                "throughput improvement."
            ),
        },
        "retro_items_opened": [i for i in retro.retro_items_opened if not i.endswith("_CLOSE")],
        "retro_items_closed": retro_items_closed,
        "new_retro_items": new_retro_items_for_artifact,
        "estimated_savings_next_pct": sum(
            item.get("estimated_savings_pct", 0)
            for item in new_retro_items_for_artifact
        ) or 10,
        "key_findings": [
            (
                "SIX consecutive milestones without live GPU inference. "
                "first_live_gpu_results_achieved=False for the sixth time."
            ),
            (
                "RETRO-022 is a HUMAN ESCALATION item. The conductor cannot power on "
                "a GPU node. Human action is mandatory before milestone 2026.06.17."
            ),
            (
                "CIKANEnergy has failed three consecutive times (Exps 375, 378, 391). "
                "Root cause: 'deliverable already exists' fast-path bypasses content "
                "validation of cikan_energy.py. File contains JSON, not Python class."
            ),
            (
                "All benchmark scripts (Exps 394-400) are fully written and tested. "
                "Estimated time to live results once GPU is available: < 4 hours."
            ),
            (
                "Exp 352 confirmed: is_live_capable=True. RTX 3090 hardware is sound. "
                "The only blocker is node power/network availability."
            ),
        ],
        "explanations": {
            "retro_019_resolved": (
                "Exp 390 ran but its result shows finding='GPU preflight script created.' "
                "not honest_verdict='gpu_confirmed_live'. The preflight script was "
                "re-confirmed present but the GPU itself was not online."
            ),
            "retro_020_closed": (
                "python/carnot/models/cikan_energy.py exists but contains a JSON artifact "
                "from Exp 375, not a Python class. 'class CIKANEnergy' is absent."
            ),
            "retro_021_closed": (
                "Exp 399 returned status='partial'. honest_verdict='learning_confirmed' "
                "was not achieved. Fourth consecutive miss."
            ),
            "live_gpu_confirmed": (
                "No experiment in this milestone produced inference_mode='live_gpu'. "
                "All results are status='partial' or missing."
            ),
            "extraction_winner_known": (
                "Exp 397 returned status='partial'. RETRO-016 (extraction comparison) "
                "remains open. LLMExtractor vs ArithmeticExtractor comparison not run."
            ),
            "jitrl_memory_works": (
                "Exp 392 has no result JSON. threshold_modulation_works cannot be read."
            ),
            "safety_kan_works": (
                "Exp 393 has no result JSON. test_auroc cannot be read."
            ),
            "saver_live_verified": (
                "Exp 400 returned status='partial'. live_verification_active not set."
            ),
            "semantic_energy_viable": (
                "Exp 401 has no result JSON. auroc cannot be read."
            ),
            "crane_extraction_improved": (
                "Exp 402 has no result JSON. crane_beats_arithmetic cannot be read."
            ),
        },
        "meta_reflection": (
            "This is the sixth retrospective documenting the same failure mode: GPU node "
            "offline during conductor session. Each milestone has added more infrastructure "
            "improvements (CARNOT_FORCE_LIVE=1, LiveGPUGate, conductor_gpu_env.sh, preflight "
            "script) but none of them can substitute for the GPU being physically powered on. "
            "The pattern of opening infrastructure RETRO items and then closing them in the "
            "next milestone without ever producing live results is a conductor loop that "
            "wastes ~13 experiments per milestone. The meta-lesson: identify the human-action "
            "blocker earlier and escalate immediately instead of writing more infrastructure code. "
            "RETRO-022 is the end of the conductor's ability to self-repair this class of failure."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Exp 403 operational retrospective for milestone 2026.06.10."""
    logging.basicConfig(level=logging.INFO)

    tmpl = ExperimentTemplate(
        403,
        "Operational Retrospective — Milestone 2026.06.10",
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    _log.info("Loading milestone result files for Exps 390-402...")
    result_files = load_milestone_results(_REPO_ROOT, RESULT_FILE_MAP)

    _log.info("Evaluating success criteria...")
    retro = compute_retro_2026_06_10(result_files, _REPO_ROOT)

    _log.info("Building artifact...")
    artifact = build_retro_artifact(retro)

    result = tmpl.build_result(artifact, status="complete")

    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    _log.info("Written to %s", out_path)
    _log.info(
        "first_live_gpu_results_achieved=%s | live_gpu_confirmed=%s",
        artifact["first_live_gpu_results_achieved"],
        retro.live_gpu_confirmed,
    )

    if not retro.live_gpu_confirmed:
        print("\n" + "=" * 70)
        print("HUMAN ACTION REQUIRED — RETRO-022 (CRITICAL)")
        print("=" * 70)
        print(
            "After SIX consecutive milestones and 403 experiments, Carnot still "
            "has zero live GPU results.\n"
            "The conductor CANNOT fix a powered-off GPU node.\n"
            "Before milestone 2026.06.17 begins, a human operator MUST:\n"
            "  Option A: Rent cloud GPU (Lambda Labs, vast.ai, RunPod)\n"
            "  Option B: Purchase RTX 4090 (~$1800 USD)\n"
            "  Option C: Power on the existing RTX 3090 node and verify reachability\n"
            "Then run: python scripts/experiment_390_gpu_preflight.py\n"
            "Only proceed when honest_verdict == 'gpu_confirmed_live'."
        )
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
