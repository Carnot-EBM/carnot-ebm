"""
Milestone 2026.05.201 retrospective generator.

Scans experiment artifacts from exp2001-exp2013 (the .201 task range, excluding
the retro itself at exp2014) and produces a structured retro JSON using the
`carnot.milestone_retro.v1` schema.

The milestone title was "Continuous Self-Learning, Hallucination Detection via
Spilled Energy, and ROCm Bring-up". The 13 tasks ran 2026-05-16.

Why this exists: every milestone closing requires a structured retro artifact
that records what completed, what was blocked, what was suspicious, and what
the planner should fix next time. This module provides the generation logic
and is tested before the artifact is written, per spec-anchored dev discipline.
"""

import json
import os
from glob import glob

# Inclusive range for .201 milestone experiments (retro task exp2014 is excluded
# so the retro can scan the tasks it is summarising without counting itself).
_MILESTONE_201_START = 2001
_MILESTONE_201_END = 2013

# Known deliverable filenames for .201 tasks, in order.
# Using explicit paths avoids false-positive matches from older artifacts that
# share the same experiment number prefix (e.g. experiment_2001_run_csp_*.json
# from a prior milestone would be picked as the shortest match by a naive glob).
_DELIVERABLES = {
    2001: "experiment_2001_spilled_energy_metrics.json",
    2002: "experiment_2002_energy_reward_model.json",
    2003: "experiment_2003_refind_csr.json",
    2004: "experiment_2004_self_learning_epsilon_engine.json",
    2005: "experiment_2005_z3_generation.json",
    2006: "experiment_2006_self_distillation_qwen.json",
    2007: "experiment_2007_continuous_epsilon_decay_eval.json",
    2008: "experiment_2008_rocm_probe.json",
    2009: "experiment_2009_dual_model.json",
    2010: "experiment_2010_langevin.json",
    2011: "experiment_2011_fidelity.json",
    2012: "experiment_2012_e2e.json",
    2013: "experiment_2013_prd.json",
}

_MILESTONE = "2026.05.201"
_MILESTONE_TITLE = (
    "Continuous Self-Learning, Hallucination Detection via Spilled Energy, "
    "and ROCm Bring-up"
)
_RETRO_EXP_ID = 2014
_RUN_DATE = "2026-05-16"


def _classify_artifact(artifact: dict) -> str:
    """Classify a parsed experiment artifact as completed, blocked, or failed.

    Priority order: honest_verdict first (most authoritative), then status
    field, then default to completed when no negative signals are present.
    """
    # 1. Honest verdict overrides
    verdict = artifact.get("honest_verdict", "").lower()
    if "blocked" in verdict:
        return "blocked"
    if any(tok in verdict for tok in ("fail", "missing artifacts", "error")):
        return "failed"
    if any(verdict.startswith(pfx) for pfx in ("complete", "success", "passed", "shipped")):
        return "completed"

    # 2. Status field fallback
    status = artifact.get("status", "").lower()
    if status in ("success", "complete", "completed"):
        return "completed"
    if status == "blocked":
        return "blocked"
    if status in ("fail", "failure", "error"):
        return "failed"

    # 3. Default to completed if no negative signals
    return "completed"


def _adversarial_flags(exp_id: int, artifact: dict) -> list:
    """Return a list of adversarial-verify flag strings for a given artifact.

    This is a lightweight inline checker that mirrors adversarial_verify.py's
    most important rules without requiring that script to be importable.
    Checks: TAUTOLOGY, DURATION_TOO_SHORT, GATE_PASSED_WITHOUT_DATA.
    """
    flags = []
    duration = artifact.get("duration_s")

    # TAUTOLOGY: two numerically identical fields that should differ
    cpu = artifact.get("cpu_counts") or artifact.get("cpu_time_s")
    gpu = artifact.get("mock_gpu_counts") or artifact.get("gpu_time_s_mocked")
    divergence = artifact.get("divergence")

    if divergence == 0.0 and cpu is not None and cpu == gpu:
        flags.append("TAUTOLOGY: cpu == mock_gpu (bit-identical via mock, not a real measurement)")

    # DURATION_TOO_SHORT: compute-bound markers present but ran in <1 s
    compute_markers = {"gguf", "cuda", "torch", "rocm", "rocminfo", "hip", "llama"}
    artifact_str = json.dumps(artifact).lower()
    if duration is not None and duration < 1.0:
        if any(m in artifact_str for m in compute_markers):
            flags.append(
                f"DURATION_TOO_SHORT: duration_s={duration} but compute-bound markers present"
            )

    # GATE_PASSED_WITHOUT_DATA: status=success but key fields missing
    if artifact.get("status") == "success" and artifact.get("z3_verified") is None:
        if artifact.get("n_puzzles_verified", 0) == artifact.get("n_puzzles_generated", -1):
            if duration is not None and duration < 1.0:
                flags.append(
                    "IMPLAUSIBLE_PERFECT: 100% Z3 verification in <1 s suggests "
                    "hardcoded responses rather than actual model inference"
                )

    return flags


def generate_retro(output_path: str, results_dir: str = "results") -> dict:
    """Scan milestone .201 experiment artifacts and produce a retro JSON.

    Parameters
    ----------
    output_path:
        Where to write the finished artifact.
    results_dir:
        Directory that contains the individual experiment JSON files.
        Exists as a parameter so tests can inject a tmp_path.
    """
    completed: list[int] = []
    blocked: list[int] = []
    failed: list[int] = []
    verdicts: dict[str, str] = {}
    adversarial_flags: dict[str, list] = {}

    for exp_id in range(_MILESTONE_201_START, _MILESTONE_201_END + 1):
        # Use the known deliverable filename for this milestone to avoid matching
        # older artifacts that share the same experiment number prefix.
        known = _DELIVERABLES.get(exp_id)
        if known:
            candidate = os.path.join(results_dir, known)
            paths = [candidate] if os.path.exists(candidate) else []
        else:
            paths = glob(os.path.join(results_dir, f"experiment_{exp_id}*.json"))
            paths = [p for p in paths if f"experiment_{_RETRO_EXP_ID}" not in p]

        if not paths:
            failed.append(exp_id)
            verdicts[f"exp{exp_id}"] = "MISSING"
            continue

        target_path = paths[0]

        try:
            with open(target_path) as fh:
                artifact = json.load(fh)
        except (json.JSONDecodeError, OSError):
            failed.append(exp_id)
            verdicts[f"exp{exp_id}"] = "UNREADABLE"
            continue

        verdicts[f"exp{exp_id}"] = artifact.get(
            "honest_verdict", artifact.get("status", "UNKNOWN")
        )

        flags = _adversarial_flags(exp_id, artifact)
        if flags:
            adversarial_flags[f"exp{exp_id}"] = flags

        cls = _classify_artifact(artifact)
        if cls == "blocked":
            blocked.append(exp_id)
        elif cls == "failed":
            failed.append(exp_id)
        else:
            completed.append(exp_id)

    n_c, n_b, n_f = len(completed), len(blocked), len(failed)

    result = {
        "experiment_id": _RETRO_EXP_ID,
        "schema": "carnot.milestone_retro.v1",
        "milestone": _MILESTONE,
        "milestone_title": _MILESTONE_TITLE,
        "run_date": _RUN_DATE,
        "status": "complete",
        "completed_task_count": n_c,
        "blocked_task_count": n_b,
        "failed_task_count": n_f,
        "completed_experiments": sorted(completed),
        "blocked_experiments": sorted(blocked),
        "failed_experiments": sorted(failed),
        "experiment_honest_verdicts": verdicts,
        "adversarial_flags": adversarial_flags,
        "bottlenecks_identified": [
            (
                "Planner-side doomed-rerun discipline failure: exp2001 (4 prior failures), "
                "exp2002 (2), exp2004 (13), exp2007 (1) were all blocked at conductor_pre_gate "
                "because the roadmap task had no prior_failures field. 4 of 13 tasks (31%) never "
                "ran. The self-learning lineage alone has 13 unacknowledged prior failures — "
                "it is a retirement candidate."
            ),
            (
                "Gate-field schema mismatch caused cascading block: exp2006 was gated on "
                "exp2005.verified_count but exp2005 emitted n_puzzles_verified. The upstream "
                "task succeeded (100 verified) but the gate could not read it, blocking exp2006. "
                "Planner must verify artifact field names match gate field names exactly."
            ),
            (
                "Adversarial-verify flags on synthetic tasks: exp2005 (100/100 Z3 verified in "
                "0.07 s — implausible if Qwen3.6-35B was actually invoked; responses appear "
                "hardcoded), exp2011 (divergence=0.0 is a TAUTOLOGY from bit-identical mocks), "
                "exp2012 (duration_s=0.0 on a task claiming E2E test execution). "
                "Preconditions blocks are absent from exp2005 and exp2012."
            ),
            (
                "ROCm bring-up (exp2008) confirmed real gfx1150 hardware "
                "(AMD Radeon 890M, 16 CU, 2900 MHz, 110 GB unified APU pool + RyzenAI-npu4 NPU). "
                "This is a genuine positive signal but follow-on tasks (exp2009, exp2010, exp2011) "
                "used mocked VRAM/GPU — no real ROCm inference was attempted."
            ),
        ],
        "recommendations": [
            (
                "Retire or fundamentally redesign the continuous self-learning / epsilon-engine "
                "lineage before proposing new versions. With 13 unacknowledged prior failures, "
                "the next proposal MUST include a prior_failures block naming each, a diagnosed "
                "root cause, and a concrete architectural change. Without all three, the conductor "
                "will block it again."
            ),
            (
                "Add preconditions blocks (per CLAUDE.md Pre-Launch Preconditions Discipline) to "
                "every task that claims to invoke Qwen3.6-35B-A3B-GGUF or gemma-4-31B-it-GGUF. "
                "The exp2005 artifact shows 0.07 s wall time for 100 GGUF responses — "
                "this is physically impossible and indicates fabricated inference calls."
            ),
            (
                "Fix the exp2006 gate: change artifact_field from 'verified_count' to "
                "'n_puzzles_verified' so the gate can read the actual exp2005 artifact. "
                "This single fix unblocks the self-distillation chain with no other changes."
            ),
            (
                "The ROCm gfx1150 APU is confirmed present and alive. The next ROCm milestone "
                "should attempt real torch.cuda() / HIP inference (not mocked VRAM allocation). "
                "Use `sg render -c '...'` and `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` "
                "per CLAUDE.md ROCm guidance. The APU's 110 GB unified memory pool means "
                "full SOTA GGUF models fit without quantization tricks."
            ),
            (
                "The REFIND CSR metric (exp2003, energy_correlation=0.9425) is the cleanest "
                "positive result of the milestone and is paper-v6 eligible pending "
                "adversarial confirmation (rotate seed + corpus + baseline). "
                "Schedule a confirmation run as the highest-priority clean task for .202."
            ),
        ],
        "retro_complete": True,
        "honest_verdict": (
            f"complete: milestone_201_retro_filed_{n_c}_completed_{n_b}_blocked_{n_f}_failed"
        ),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)

    return result


if __name__ == "__main__":  # pragma: no cover
    generate_retro("results/experiment_2014_retro.json")
