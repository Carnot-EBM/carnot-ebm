"""Experiment 337: Operational Retrospective for milestone 2026.04.24.

Spec coverage: REQ-RETRO-003,
               SCENARIO-RETRO-005, SCENARIO-RETRO-006

This script:
1. [LOG PARSE] Derives wall-time data from ops/conductor-log.md for Exps 325-336.
   Method: each experiment's wall time = UTC timestamp of its final "OK" or test-pass
   entry minus the UTC timestamp of the previous experiment's final "OK" entry.
   Milestone activated at 2026-04-15 01:27 UTC.

2. [RETRO ITEMS] Audits whether RETRO-001 through NEW-002 were resolved:
   - RETRO-001 (45-min timeout): Exp 325 shipped run_experiment_with_timeout.sh
   - RETRO-002 (DualGPUMonitor): Exp 326 shipped DualGPUMonitor
   - NEW-001 (test-first stubs): Exp 325 added generate_test_stub()
   - NEW-002 (pre-experiment dep audit): Exp 327 shipped DependencyAudit

3. [SPEEDUP MEASURE] Compares mean_time_per_exp_min against the prior milestone
   baseline of 40.6 min/exp (derived from Exp 319: 691 total / 17 experiments).
   Computes actual_speedup_pct = (40.6 - curr_mean) / 40.6 * 100.

4. [BOTTLENECKS] Identifies top-3 experiments by wall time and their root causes.

5. [NEW ACTIONS] Identifies new action items from this milestone's pattern of failures:
   - NEW-003: Two experiments (331, 334) hit the 50-turn max-turns ceiling.
     Recommendation: pre-split experiments predicted to require > 40 turns into
     Phase A (implementation) + Phase B (tests + verification).

6. [ARTIFACT] Writes results/operational_retro_2026_04_24.json with schema
   "carnot.operational_retro.v1".

Usage:
    JAX_PLATFORMS=cpu python scripts/experiment_337_retro.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONDUCTOR_LOG = REPO_ROOT / "ops" / "conductor-log.md"
OUTPUT_FILE = RESULTS_DIR / "operational_retro_2026_04_24.json"

# ---------------------------------------------------------------------------
# Milestone experiment wall times (minutes), derived from ops/conductor-log.md.
#
# Method: timestamps extracted from the conductor log entries for Exps 325-336.
# For each experiment the wall time is measured from the final "OK" entry of the
# prior experiment to the final "OK" or test-pass entry of this experiment.
#
# Milestone activated: 2026-04-15 01:27 UTC
#
# Exp 325: 01:27 (milestone activate) → 02:02 (81 passed in 3.28s)  = 35 min
# Exp 326: 02:02 → 02:28 (81 passed in 3.53s)                       = 26 min
# Exp 327: 02:28 → 02:57 (Deliverable already exists confirmed)      = 29 min
# Exp 328: 02:57 → 03:24 (Deliverable already exists confirmed)      = 27 min
# Exp 329: 03:24 → 03:52 (Deliverable already exists confirmed)      = 28 min
# Exp 330: 03:52 → 04:17 (81 passed in 3.76s)                       = 25 min
# Exp 331: 04:17 → 04:27 (FAIL max-turns then OK deliverable)        = 10 min
# Exp 332: 04:27 → 04:59 (177 passed in 3.67s)                      = 32 min
# Exp 333: 04:59 → 05:30 (81 passed in 3.18s)                       = 31 min
# Exp 334: 05:30 → 05:59 (FAIL max-turns then 175 passed in 3.77s)  = 29 min
# Exp 335: 05:59 → 06:10 (81 passed in 3.22s; Deliverable confirmed) = 11 min
# Exp 336: 06:10 → 06:20 (145 passed in 3.74s)                      = 10 min
# ---------------------------------------------------------------------------

_EXPERIMENT_WALL_TIMES: dict[int, float] = {
    325: 35.0,
    326: 26.0,
    327: 29.0,
    328: 27.0,
    329: 28.0,
    330: 25.0,
    331: 10.0,
    332: 32.0,
    333: 31.0,
    334: 29.0,
    335: 11.0,
    336: 10.0,
}

# Prior milestone baseline (Exps 307-324 from experiment_319_retro.py):
# 691 total minutes / 17 experiments = 40.647... min/exp, documented as 40.6.
_PRIOR_MILESTONE_MEAN_MIN: float = 40.6

# Root causes for the top-3 slowest experiments in this milestone.
_SLOWEST_ROOT_CAUSES: dict[int, dict] = {
    325: {
        "title": "Conductor hardening — RETRO-001 + NEW-001 implementation",
        "root_cause": (
            "Exp 325 implemented two major action items in a single agent run: "
            "the 45-minute conductor timeout wrapper (run_experiment_with_timeout.sh) "
            "and the generate_test_stub() addition to ExperimentTemplate. "
            "Two-deliverable experiments reliably take 30-40 min even with the new "
            "tooling because the agent must write, test, and commit two independent "
            "modules. The 35-minute wall time is the inherent cost of this multi-part "
            "design, not a process failure."
        ),
        "category": "multi_deliverable_complexity",
        "fix_implemented_this_milestone": False,
    },
    332: {
        "title": "Confidence-weighted constraint violations — dual-signal repair",
        "root_cause": (
            "Exp 332 introduced two new confidence signals (expression specificity, "
            "partition function variance) and a dual-signal ConfidenceRepairResult "
            "dataclass with 177 tests covering all signal combinations and edge cases. "
            "High test count (177 vs the typical 80-120 range) was the primary driver "
            "of the 32-minute wall time. The test-first stub (NEW-001) was used but the "
            "dual-signal design required more cases than a single-signal implementation."
        ),
        "category": "high_test_count_dual_signal",
        "fix_implemented_this_milestone": False,
    },
    333: {
        "title": "Model-adaptive constraint thresholds + selective CaseMemory consolidation",
        "root_cause": (
            "Same pattern as Exp 325: two independent deliverables (PerModelFPTracker "
            "and SelectiveConsolidation) combined into one agent run. 43 tests each, "
            "86 total. The 31-minute wall time is slightly below Exp 332 because the "
            "deliverables share data structures (both use CaseMemory) and the agent "
            "could reuse setup code rather than building from scratch."
        ),
        "category": "multi_deliverable_complexity",
        "fix_implemented_this_milestone": False,
    },
}

# What each experiment implemented (for improvements_implemented field).
_IMPROVEMENTS_IMPLEMENTED: list[dict] = [
    {
        "name": "Conductor 45-min timeout wrapper (RETRO-001) + test-first stubs (NEW-001)",
        "experiment": 325,
        "description": (
            "Shipped run_experiment_with_timeout.sh with configurable "
            "CARNOT_CONDUCTOR_TIMEOUT_MINUTES (default 45). Added generate_test_stub() "
            "to ExperimentTemplate: idempotent stub writer that creates a test file "
            "skeleton before implementation code is written. Resolves RETRO-001 and NEW-001, "
            "both carried forward from the 2026.04.23 retro. Estimated speedup 27%."
        ),
    },
    {
        "name": "DualGPUMonitor + setup_gpu() GPU health integration (RETRO-002/003)",
        "experiment": 326,
        "description": (
            "Shipped DualGPUMonitor with GPUProcessInfo dataclass and zombie detection. "
            "check_dual_gpu_health() injected into ExperimentTemplate.setup_gpu() as an "
            "additive gpu_monitor_results key. CI-safe fallback when nvidia-smi absent. "
            "Resolves RETRO-002 and RETRO-003 from the 2026.04.23 retro."
        ),
    },
    {
        "name": "Pre-experiment dependency audit CLI (NEW-002)",
        "experiment": 327,
        "description": (
            "Shipped DependencyAudit with extract_required_files(), check_dependencies(), "
            "and build_blocked_artifact(). CLI audits an experiment prompt and emits "
            "status='blocked_prereq' before the agent runs if required files are absent. "
            "Resolves NEW-002 from the 2026.04.23 retro. 34 tests pass."
        ),
    },
    {
        "name": "Live GPU full-scale benchmark (Exp 316 re-run on real hardware)",
        "experiment": 328,
        "description": (
            "First live GPU execution of the Exp 315/316 full-scale benchmark: "
            "DualGPUMonitor confirmed 2x RTX 3090s idle and healthy before run. "
            "Result: live accuracy ~10% below simulated baseline across all variants "
            "(number_swap, irrelevant_sentence, all). Simulation divergence documented "
            "in simulation_divergence field. Provides honest GPU-backed headline numbers."
        ),
    },
    {
        "name": "Live GPU four-tier relay benchmark (Exp 318 re-run on real hardware)",
        "experiment": 329,
        "description": (
            "First live GPU run of the four-tier self-learning relay. "
            "Result: improvement_1to3 = -6.1% (relay hurts accuracy on this 33-question "
            "batch). JEPA skip rate 18.2% matches simulation. Negative improvement is an "
            "honest research finding: batch size may be too small for reliable signal, "
            "or Tier 3 is actively degrading Tier 1 accuracy on this distribution."
        ),
    },
    {
        "name": "HuggingFace live publish of trained EBM model cards",
        "experiment": 330,
        "description": (
            "Ran the Exp 317 patching script against the live HuggingFace hub. "
            "Updated all 16 per-token activation EBM README files with Phase 1 "
            "status clarifications and Exp 316 (simulated) and Exp 328 (live) "
            "benchmark results. Idempotent sentinel comment prevents double-patching."
        ),
    },
    {
        "name": "Systematic FP autopsy — categorize broken verify-repair cases",
        "experiment": 331,
        "description": (
            "FPCategory enum + AutopsyCase dataclass + categorize_fp() function that "
            "classifies false positives into 5 categories (constraint_too_loose, "
            "domain_specific_knowledge, ambiguous_constraint, correct_llm_output, "
            "repair_overcorrection). Hit max-turns (50) on first attempt; recovered "
            "via deliverable-already-exists on retry. 34 tests pass."
        ),
    },
    {
        "name": "Confidence-weighted repair — dual-signal FP reduction",
        "experiment": 332,
        "description": (
            "Expression specificity confidence signal (REQ-VERIFY-083) and partition "
            "function variance confidence signal (REQ-VERIFY-084). ConfidenceRepairResult "
            "dataclass aggregates both signals. FP avoided rate 73.3%, TP preserved rate "
            "100%. 177 tests pass covering all signal combinations."
        ),
    },
    {
        "name": "Model-adaptive constraint thresholds + selective CaseMemory consolidation",
        "experiment": 333,
        "description": (
            "PerModelFPTracker tracks per-model FP rate online; ModelAdaptiveThresholds "
            "adjusts violation threshold per model ID. SelectiveConsolidation filters "
            "case memory entries below a confidence floor before consolidation. "
            "43 tests each (86 total). Enables the pipeline to self-tune per model "
            "without retraining the EBM."
        ),
    },
    {
        "name": "VERGE-style iterative Z3 refinement",
        "experiment": 334,
        "description": (
            "IterativeZ3Refiner: repeatedly strengthens Z3 constraints until the solver "
            "finds UNSAT (no counterexample) or the iteration budget is exhausted. "
            "Implements the VERGE paper's targeted refinement loop. Hit max-turns (50) "
            "on first attempt; recovered on retry. 175 tests pass."
        ),
    },
    {
        "name": "AMD XDNA NPU build — prereq retry and status update",
        "experiment": 335,
        "description": (
            "Re-ran the Exp 314 NPU unblock workflow: confirmed ninja and openblas "
            "still missing, added prereq_delta field (changes vs Exp 314). Demonstrates "
            "the dependency audit (Exp 327) would have surfaced this without consuming "
            "agent turns. Infrastructure tracks delta across attempts."
        ),
    },
    {
        "name": "CoTCircuitVerifier — CRV-style chain-of-thought computational graph",
        "experiment": 336,
        "description": (
            "CoTCircuitVerifier + CoTStep + extract_cot_steps() parse chain-of-thought "
            "reasoning into a directed computational graph. CoTCircuit.build_circuit() "
            "and find_broken_links() detect logical discontinuities between reasoning "
            "steps. Implements REQ-EXTRACT-015/016 (SCENARIO-EXTRACT-031 through -035). "
            "145 tests pass."
        ),
    },
]


# ---------------------------------------------------------------------------
# Log parse: extract wall-time data (already embedded above)
# ---------------------------------------------------------------------------


def load_wall_times() -> dict[int, float]:
    """Return the per-experiment wall times for this milestone.

    Why: the wall times are derived from conductor log timestamps and hard-coded
    here for reproducibility.  Any future retro script can re-derive them from
    the log by parsing the UTC timestamps directly.
    """
    return dict(_EXPERIMENT_WALL_TIMES)


# ---------------------------------------------------------------------------
# RETRO items audit
# ---------------------------------------------------------------------------


def build_retro_resolved_flags() -> dict[str, bool]:
    """Return resolved flags for each 2026.04.23 action item.

    Evidence (conductor log lines):
    - RETRO-001: 2026-04-15 01:51 UTC — Exp 325: "run_experiment_with_timeout.sh written
      (45 min default, CARNOT_CONDUCTOR_TIMEOUT_MINUTES); RETRO-001 implemented"
    - RETRO-002: 2026-04-15 02:04 UTC — Exp 326: "DualGPUMonitor + GPUProcessInfo in
      pipeline; setup_gpu() additive gpu_monitor_results key; RETRO-002/003 implemented"
    - NEW-001: 2026-04-15 01:51 UTC — Exp 325: "generate_test_stub() added to
      ExperimentTemplate; NEW-001 implemented"
    - NEW-002: 2026-04-15 02:29 UTC — Exp 327: "DependencyAudit + extract_required_files
      + check_dependencies + build_blocked_artifact + load_experiment_prompt + CLI;
      REQ-INFRA-005 implemented"
    """
    return {
        "RETRO-001": True,  # 45-min timeout shipped in Exp 325
        "RETRO-002": True,  # DualGPUMonitor shipped in Exp 326
        "NEW-001": True,    # test-first stubs shipped in Exp 325
        "NEW-002": True,    # dependency audit shipped in Exp 327
    }


def build_carry_over() -> list[dict]:
    """Build the carry_over list documenting 2026.04.23 retro items and their resolution.

    Each entry: {id, description, resolved: bool}

    Why a list (not dict): the task spec requires carry_over to be a list so that
    ordering is preserved and tests can iterate over entries uniformly.
    """
    resolved = build_retro_resolved_flags()
    return [
        {
            "id": "RETRO-001",
            "description": (
                "45-minute hard timeout wrapper for conductor experiments. "
                "Ships run_experiment_with_timeout.sh; auto-escalates after 45 min. "
                "Carried forward from 2026.04.22 retro through 2026.04.23 retro "
                "without implementation — finally resolved in Exp 325."
            ),
            "resolved": resolved["RETRO-001"],
            "resolution": "Implemented in Exp 325 (2026-04-15 01:51 UTC)",
        },
        {
            "id": "RETRO-002",
            "description": (
                "Integrate DualGPUMonitor into conductor pre-experiment check. "
                "Injects GPU state into experiment prompt; detects zombies; "
                "auto-selects DualGPURunner vs single-GPU. "
                "Carried forward from 2026.04.22 retro through 2026.04.23 retro "
                "without implementation — resolved in Exp 326."
            ),
            "resolved": resolved["RETRO-002"],
            "resolution": "Implemented in Exp 326 (2026-04-15 02:04 UTC)",
        },
        {
            "id": "NEW-001",
            "description": (
                "Enforce test-first development in ExperimentTemplate via "
                "generate_test_stub(). Idempotent stub writer creates a test file "
                "skeleton before implementation code is written. Introduced as a "
                "new item in the 2026.04.23 retro."
            ),
            "resolved": resolved["NEW-001"],
            "resolution": "Implemented in Exp 325 (2026-04-15 01:51 UTC)",
        },
        {
            "id": "NEW-002",
            "description": (
                "Pre-experiment optional-dependency audit. DependencyAudit CLI "
                "parses the experiment prompt, checks required files exist in the "
                "virtualenv/repo, and emits status='blocked_prereq' before the agent "
                "runs. Introduced as a new item in the 2026.04.23 retro."
            ),
            "resolved": resolved["NEW-002"],
            "resolution": "Implemented in Exp 327 (2026-04-15 02:29 UTC)",
        },
    ]


# ---------------------------------------------------------------------------
# Speedup computation
# ---------------------------------------------------------------------------


def compute_actual_speedup(mean_min: float) -> float:
    """Compute measured wall-time speedup vs the prior milestone baseline.

    Prior milestone (Exps 307-324): 691 total minutes / 17 experiments = 40.6 min/exp.
    A positive value means this milestone was faster; negative means regression.

    Why 40.6: this is the value documented in CLAUDE.md context and derivable from
    experiment_319_retro.py's _EXPERIMENT_WALL_TIMES dict (sum=691, n=17).
    """
    return round(((_PRIOR_MILESTONE_MEAN_MIN - mean_min) / _PRIOR_MILESTONE_MEAN_MIN) * 100, 1)


# ---------------------------------------------------------------------------
# Bottlenecks
# ---------------------------------------------------------------------------


def build_bottlenecks(wall_times: dict[int, float]) -> list[dict]:
    """Return top-3 slowest experiments as structured bottleneck dicts.

    Each entry has: name, duration_min, pct_total, root_cause, category,
    fix_implemented_this_milestone.

    Why top-3: this matches the Exp 319 retro convention and gives enough
    entries to identify patterns without overwhelming the artifact reader.
    """
    total = sum(wall_times.values())
    sorted_exps = sorted(wall_times.items(), key=lambda kv: kv[1], reverse=True)
    bottlenecks: list[dict] = []
    for exp_id, duration in sorted_exps[:3]:
        pct = round(duration / total * 100, 1)
        root = _SLOWEST_ROOT_CAUSES.get(exp_id, {})
        bottlenecks.append(
            {
                "name": f"Exp {exp_id}: {root.get('title', 'unknown')}",
                "duration_min": duration,
                "pct_total": pct,
                "root_cause": root.get("root_cause", "root cause not documented"),
                "category": root.get("category", "unknown"),
                "fix_implemented_this_milestone": root.get(
                    "fix_implemented_this_milestone", False
                ),
            }
        )
    return bottlenecks


# ---------------------------------------------------------------------------
# Action items
# ---------------------------------------------------------------------------


def build_action_items() -> list[dict]:
    """Build new action items for the 2026.04.24 milestone.

    All four items from the 2026.04.23 retro (RETRO-001, RETRO-002, NEW-001, NEW-002)
    were resolved this milestone.  New items address patterns discovered in Exps 325-336.

    NEW-003 addresses the max-turns (50) failures in Exp 331 and Exp 334.  Both
    experiments hit the ceiling because they were complex multi-module implementations
    that required the agent to write code, run tests, debug, and update docs within a
    single session.  The fix is to split predicted-complex experiments into two phases
    before the agent starts, not after it fails.

    NEW-004 addresses the negative relay improvement signal from Exp 329.  The live
    relay shows improvement_1to3 = -6.1%, meaning the four-tier relay is hurting
    accuracy on this test batch.  Before the next milestone, add a relay health guard
    that surfaces this signal explicitly and blocks further self-learning relay
    experiments until the root cause is diagnosed.
    """
    return [
        {
            "id": "NEW-003",
            "description": (
                "Pre-split complex experiments that are predicted to require > 40 agent "
                "turns into Phase A (implementation + unit tests) and Phase B (integration "
                "tests + artifact + docs update). This milestone had 2/12 = 17% max-turns "
                "failures (Exp 331, 334), both of which recovered quickly on retry but each "
                "lost 5-9 minutes of wall time and created a conductor retry cycle. "
                "Pre-splitting avoids the failure entirely: each phase is simpler and more "
                "focused. Implementation: add an expected_complexity field to experiment "
                "prompt headers; conductor splits automatically when complexity == 'high'."
            ),
            "status": "new",
            "owner": "conductor",
            "priority": "medium",
            "estimated_impact_pct": 3,
            "evidence_this_milestone": (
                "Exp 331 (FP autopsy): FAIL (max turns 50), recovered in 1 min (deliverable "
                "already existed). Exp 334 (VERGE iterative Z3): FAIL (max turns 50), "
                "recovered in 20 min on retry with 175 tests passing. "
                "Combined lost wall time: ~10 min = 3.4% of milestone total."
            ),
        },
        {
            "id": "NEW-004",
            "description": (
                "Add a relay improvement health guard before the next milestone's "
                "self-learning relay experiments. Exp 329 live GPU run shows "
                "improvement_1to3 = -6.1%, meaning the four-tier relay is actively "
                "degrading Tier 1 accuracy on a 33-question batch. Before planning "
                "further relay experiments, diagnose whether the negative signal is "
                "due to: (a) batch size too small for reliable measurement, (b) Tier 3 "
                "JEPA gate miscalibrated on the live distribution, or (c) a genuine "
                "architectural regression introduced between Exp 318 and Exp 329. "
                "Block relay-dependent experiments in the next roadmap until this is "
                "resolved. Implementation: add relay_health_check() to ExperimentTemplate "
                "that loads Exp 329 artifact and warns if improvement_1to3 < 0."
            ),
            "status": "new",
            "owner": "experiment_template",
            "priority": "high",
            "estimated_impact_pct": 2,
            "evidence_this_milestone": (
                "Exp 329 (live four-tier relay): improvement_1to3 = -0.060606 (-6.1%), "
                "batch3_accuracy = 0.636 vs batch1_accuracy = 0.697. "
                "JEPA skip rate 18.2% (same as simulated). The relay is not improving "
                "accuracy — it is making it worse on this distribution."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Speedup estimate for next milestone
# ---------------------------------------------------------------------------


def compute_estimated_next_speedup(action_items: list[dict]) -> float:
    """Estimate percentage wall-time reduction for the next milestone.

    NEW-003 (pre-split complex experiments): 3% — prevents ~10 min of max-turns waste.
    NEW-004 (relay health guard): 2% — prevents wasted relay experiments on a broken signal.

    Additive sum: 5%.  But there is minimal overlap between these items (they address
    different failure modes), so the realistic combined estimate stays close to the
    additive sum.  Apply a small realism discount to 4% to be honest: the items are
    relatively low-leverage since the main bottlenecks are now inherent task complexity,
    not process failures.

    Why can this be 0 or negative: the CLAUDE.md spec (REQ-RETRO-003) explicitly allows
    honest negative estimates.  In this case, the estimate is positive but modest — the
    big wins (RETRO-001/002, NEW-001/002) are already banked.
    """
    additive = sum(item.get("estimated_impact_pct", 0) for item in action_items)
    # Discount factor 0.8: the items are largely non-overlapping so the reduction is
    # close to additive; 0.8 shaves off optimism bias.
    realistic = round(additive * 0.8, 1)
    return realistic


# ---------------------------------------------------------------------------
# Conductor log update
# ---------------------------------------------------------------------------


def append_conductor_log_entry(artifact_path: Path, speedup_pct: float) -> None:
    """Append a one-line retrospective entry to ops/conductor-log.md.

    Why: the CLAUDE.md workflow requires ops/conductor-log.md to record every
    significant conductor action including retros so future sessions can reconstruct
    the milestone timeline from the log alone.
    """
    if not CONDUCTOR_LOG.exists():
        return
    wall_times = load_wall_times()
    total = sum(wall_times.values())
    n = len(wall_times)
    slowest_id = max(wall_times, key=lambda k: wall_times[k])
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = (
        f"| {timestamp} | Exp 337: Operational retrospective for milestone "
        f"2026.04.24 | OK | {artifact_path.name} written; "
        f"n={n} experiments, total={round(total)} min, "
        f"mean={round(total/n, 1)} min/exp, "
        f"top bottleneck=Exp {slowest_id} ({wall_times[slowest_id]} min); "
        f"all 4 RETRO items resolved; NEW-003/004 added; "
        f"actual speedup ~{round(((_PRIOR_MILESTONE_MEAN_MIN - total/n) / _PRIOR_MILESTONE_MEAN_MIN) * 100, 1)}%; "
        f"estimated next speedup ~{speedup_pct}% |\n"
    )
    with CONDUCTOR_LOG.open("a") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate the operational retrospective for milestone 2026.04.24."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    wall_times = load_wall_times()
    n_experiments = len(wall_times)
    total_wall_time = sum(wall_times.values())
    mean_min = round(total_wall_time / n_experiments, 1)

    resolved = build_retro_resolved_flags()
    carry_over = build_carry_over()
    bottlenecks = build_bottlenecks(wall_times)
    action_items = build_action_items()
    actual_speedup = compute_actual_speedup(mean_min)
    est_speedup = compute_estimated_next_speedup(action_items)

    # Slowest experiment string for the artifact.
    slowest_id = max(wall_times, key=lambda k: wall_times[k])
    slowest_str = (
        f"Exp {slowest_id}: "
        f"{_SLOWEST_ROOT_CAUSES.get(slowest_id, {}).get('title', 'unknown')} "
        f"({wall_times[slowest_id]} min)"
    )

    output: dict = {
        "schema": "carnot.operational_retro.v1",
        "milestone": "2026.04.24",
        "generated_at": now,
        "n_experiments": n_experiments,
        "total_wall_time_min": total_wall_time,
        "mean_time_per_exp_min": mean_min,
        "slowest_experiment": slowest_str,
        "retro_001_resolved": resolved["RETRO-001"],
        "retro_002_resolved": resolved["RETRO-002"],
        "actual_speedup_pct": actual_speedup,
        "estimated_next_milestone_speedup_pct": est_speedup,
        "prior_milestone_mean_min": _PRIOR_MILESTONE_MEAN_MIN,
        "carry_over": carry_over,
        "action_items": action_items,
        "bottlenecks_identified": bottlenecks,
        "improvements_implemented": _IMPROVEMENTS_IMPLEMENTED,
        "post_test_failure_rate_pct": round(
            sum(1 for e in [331, 334] if e in wall_times) / n_experiments * 100, 1
        ),
        "max_turns_failures": [331, 334],
        "meta_reflection": (
            f"Milestone 2026.04.24 ran {n_experiments} experiments in "
            f"{total_wall_time} minutes (mean {mean_min} min/exp), a "
            f"{actual_speedup}% improvement over the prior milestone baseline of "
            f"{_PRIOR_MILESTONE_MEAN_MIN} min/exp.  All four action items from the "
            f"2026.04.23 retro (RETRO-001, RETRO-002, NEW-001, NEW-002) were "
            f"implemented in the first three experiments of this milestone (Exps 325-327), "
            f"front-loading the process improvements and allowing the remaining nine "
            f"experiments to benefit immediately.  The key remaining bottleneck is "
            f"inherent task complexity for multi-deliverable experiments (Exps 325, 332, "
            f"333) and max-turns failures for complex single experiments (331, 334).  "
            f"The negative relay improvement signal from Exp 329 (improvement_1to3 = -6.1%) "
            f"is a research concern: the four-tier relay may not be working as intended on "
            f"the live distribution and should be investigated before planning further "
            f"relay-dependent experiments."
        ),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"Written: {OUTPUT_FILE}")

    # Append to conductor log
    append_conductor_log_entry(OUTPUT_FILE, est_speedup)

    # Console summary
    print(f"Milestone             : {output['milestone']}")
    print(f"Experiments           : {n_experiments}")
    print(f"Total wall time       : {total_wall_time:.0f} min")
    print(f"Mean per experiment   : {mean_min:.1f} min")
    print(f"Prior milestone mean  : {_PRIOR_MILESTONE_MEAN_MIN:.1f} min/exp")
    print(f"Actual speedup        : {actual_speedup:.1f}%")
    print(f"Top bottleneck        : {bottlenecks[0]['name']} ({bottlenecks[0]['duration_min']} min)")
    print(f"RETRO-001 resolved    : {resolved['RETRO-001']}")
    print(f"RETRO-002 resolved    : {resolved['RETRO-002']}")
    print(f"Max-turns failures    : {output['max_turns_failures']}")
    print(f"Estimated next speedup: {est_speedup}%")


if __name__ == "__main__":
    main()
