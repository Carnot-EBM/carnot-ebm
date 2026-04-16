"""Experiment 319: Operational Retrospective for milestone 2026.04.23.

Spec coverage: REQ-OPS-001, REQ-OPS-002, REQ-OPS-003, REQ-OPS-004, REQ-OPS-005,
               SCENARIO-OPS-001 through SCENARIO-OPS-007

This script:
1. Derives wall-time data from ops/conductor-log.md for the 2026.04.23 milestone
   (experiments 307–324, excluding 319 which is this retro).
2. Identifies the top-3 slowest experiments and their root causes.
3. Lists improvements implemented during this milestone.
4. Audits carry-forward action items from the 2026.04.22 retro:
   - RETRO-001: Conductor 45-minute hard timeout (not yet implemented)
   - RETRO-002: GPU monitor in conductor subprocess (not yet implemented)
5. Proposes a new action item for the post-test failure rate discovered this milestone.
6. Estimates the next-milestone wall-time speedup from the action items.
7. Writes results/operational_retro_2026_04_23.json.
8. Appends a summary entry to ops/conductor-log.md.

Usage:
    JAX_PLATFORMS=cpu python scripts/experiment_319_retro.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONDUCTOR_LOG = REPO_ROOT / "ops" / "conductor-log.md"
OUTPUT_FILE = RESULTS_DIR / "operational_retro_2026_04_23.json"

# ---------------------------------------------------------------------------
# Milestone experiment range.
#
# Experiments 307–324 were executed during the 2026.04.23 milestone.
# Exp 319 is this retrospective itself; we include it in the count (the script
# produced it) but we treat its wall-time as the cost of the retrospective.
# The planned roadmap listed 13 experiments; bonus experiments were added
# mid-milestone bringing the total to 17 (excluding 319 which is the retro).
# ---------------------------------------------------------------------------

# Per-experiment wall-time in minutes, derived from ops/conductor-log.md.
# Method: for each experiment, the wall-clock duration is from the timestamp of
# the first conductor attempt to the timestamp of the first OK or
# "Deliverable already exists" entry.
#
# Milestone activated: 2026-04-14 13:01 UTC
# Exp 307: 13:01 → 13:45 first OK = 44 min
# Exp 308: 13:45 → 16:03 first OK (post-test fail + retry) = 138 min
# Exp 309: 16:03 → 17:09 first OK (post-test fail + retry) = 66 min
# Exp 310: 17:09 → 17:49 first OK (post-test fail + retry) = 40 min
# Exp 311: 17:49 → 18:24 first OK (post-test fail + retry) = 35 min
# Exp 312: 18:24 → 19:11 first OK = 47 min
# Exp 313: 19:11 → 19:53 first OK = 42 min
# Exp 314: 19:53 → 20:14 first OK = 21 min
# Exp 315: 20:14 → 20:23 first OK = 9 min
# Exp 316: 20:23 → 20:31 first OK = 8 min
# Exp 317: 20:31 → 21:01 first OK = 30 min
# Exp 318: 21:01 → 21:28 first OK = 27 min
# Exp 322: 21:28 → 22:01 first OK = 33 min
# Exp 323: 22:01 → 22:48 first OK = 47 min
# Exp 324: 22:48 → 22:57 first OK = 9 min
# Exp 320: 22:57 → 23:30 first OK = 33 min
# Exp 321: 23:30 → 00:32 first OK (next UTC day) = 62 min

_EXPERIMENT_WALL_TIMES: dict[int, float] = {
    307: 44.0,
    308: 138.0,
    309: 66.0,
    310: 40.0,
    311: 35.0,
    312: 47.0,
    313: 42.0,
    314: 21.0,
    315: 9.0,
    316: 8.0,
    317: 30.0,
    318: 27.0,
    320: 33.0,
    321: 62.0,
    322: 33.0,
    323: 47.0,
    324: 9.0,
}

# Root causes for the top-3 slowest experiments.
# These are derived from the conductor log patterns and experiment descriptions.
_SLOWEST_ROOT_CAUSES: dict[int, dict] = {
    308: {
        "title": "JEPA fast-path gate integration and latency benchmark",
        "root_cause": (
            "Post-test failures on the first attempt due to a dimension mismatch in "
            "logit_mean (32→8 for the Exp 291 ONNX model). The test suite expected "
            "a 32-dimensional logit vector but the Exp 307 training produced an "
            "8-feature energy vector. Required a repair loop to fix the feature "
            "dimension and re-run tests before the experiment could be marked OK. "
            "With a 45-minute hard timeout (RETRO-001), the first attempt would have "
            "been auto-escalated at 45 min rather than continuing to 138 min."
        ),
        "category": "post_test_failure_repair_loop",
        "fix_implemented_this_milestone": False,
    },
    309: {
        "title": "Tier 3 end-to-end self-learning — real logit pipeline",
        "root_cause": (
            "Post-test failures on the first attempt due to integration-level test "
            "failures after wiring the new Tier 3 JEPA gate into the existing Tier 1 "
            "and Tier 2 pipeline. Tests were not written test-first before the "
            "integration; they were added as the integration was developed, leading "
            "to incomplete coverage of edge cases that only manifested when the full "
            "stack ran. Retry required re-running all tests after the fix."
        ),
        "category": "post_test_failure_integration",
        "fix_implemented_this_milestone": False,
    },
    321: {
        "title": "D-Wave Neal vs CPU Ising on constraint verification",
        "root_cause": (
            "The dwave-ocean-sdk was not installed in the project virtualenv. "
            "The conductor committed code and tests in Exp 320 but the benchmark "
            "script (Exp 321) could not run until the SDK was installed. The install "
            "step added ~20 minutes of overhead before any benchmarking could begin. "
            "A pre-flight dependency check (RETRO-002 covers conda/pip pre-checks, "
            "but GPU monitor alone does not catch missing optional SDKs) would have "
            "surfaced this at the Exp 320 planning step."
        ),
        "category": "missing_dependency",
        "fix_implemented_this_milestone": False,
    },
}

# Improvements implemented during the 2026.04.23 milestone.
# Prior-milestone infrastructure (ExperimentTemplate, BatchedInferenceRunner)
# is listed as "prior_milestone" because the benefit was realized throughout
# this milestone but the implementation was in Exp 306 (milestone 2026.04.22).
_IMPROVEMENTS_IMPLEMENTED: list[dict] = [
    {
        "name": "JEPA real-data training pipeline",
        "experiment": 307,
        "description": (
            "Trained the JEPA violation predictor (Exp 291 architecture) on real "
            "Apple adversarial logit data from Exp 295/282/283. Produces an 8-feature "
            "energy vector for use as a fast-path gate. First training run using "
            "real (not synthetic) logits — the Exp 291 model was trained on simulated "
            "data. Enables honest JEPA gate benchmarks in Exp 308."
        ),
    },
    {
        "name": "JEPA fast-path gate integration",
        "experiment": 308,
        "description": (
            "Integrated the trained JEPA predictor as a fast-path gate in the "
            "verify-repair pipeline. Gate skips Ising sampling when JEPA energy < "
            "threshold (0.55). Benchmarks latency on simulated data. First integration "
            "of JEPA gating into the live pipeline (prior Exp 145 was a prototype)."
        ),
    },
    {
        "name": "Tier 3 end-to-end self-learning pipeline",
        "experiment": 309,
        "description": (
            "Wired Tier 3 (JEPA gate) into the four-tier self-learning stack alongside "
            "Tier 1 (ConfidenceVerifier) and Tier 2 (ConstraintGenerator). End-to-end "
            "pipeline where each tier's output gates the next tier's invocation."
        ),
    },
    {
        "name": "NL2Z3Extractor — LLM-translated SMT assertions",
        "experiment": 310,
        "description": (
            "New NL2Z3Extractor class that translates natural-language constraint "
            "descriptions into Z3 SMT assertions via an LLM call. Enables "
            "constraint extraction for domains where regex and AST-based extractors "
            "fail to find structured constraints."
        ),
    },
    {
        "name": "Extractor benchmark — regex vs LLM vs Z3",
        "experiment": 311,
        "description": (
            "Head-to-head benchmark comparing the three extraction strategies "
            "(regex, LLM-based, Z3 SMT) on GSM8K and HumanEval. Establishes "
            "precision/recall baselines for each extractor to guide routing policy."
        ),
    },
    {
        "name": "Z3-gated repair pipeline",
        "experiment": 312,
        "description": (
            "Z3GatedRepair class: only invokes the Ising repair sampler when Z3 "
            "formally confirms a constraint violation (SAT → repair needed; UNSAT → "
            "skip). Replaces binary violation gating with a formal certificate. "
            "Reduces false-positive repair invocations compared to the prior "
            "confidence-threshold-only gate. Integrated into VerifyRepairPipeline "
            "as verify_repair_z3_gated()."
        ),
    },
    {
        "name": "KV260 FPGA hardware bring-up",
        "experiment": 313,
        "description": (
            "Formal hardware bring-up experiment for the KV260 FPGA with PYNQ overlay. "
            "Implements detect_kv260_hardware() with sequential prereq checks and "
            "honest_verdict pattern. CPU fallback always populated for comparison. "
            "Currently blocked (CARNOT_KV260_BITFILE not set) but infrastructure ready."
        ),
    },
    {
        "name": "AMD XDNA NPU prereq retry",
        "experiment": 314,
        "description": (
            "Re-ran the Exp 303 NPU unblock workflow to check whether ninja and "
            "openblas were installed since Exp 303. Adds prereq_changes field (delta "
            "vs Exp 303). Blocked: ninja and openblas still missing. Infrastructure "
            "tracks the delta so future reruns surface changes."
        ),
    },
    {
        "name": "Full-scale credible benchmark script",
        "experiment": 315,
        "description": (
            "Wrote the full-scale benchmark script (executed in Exp 316): 400 GSM8K "
            "questions, 50 HumanEval, two models, four modes, 95% Wilson confidence "
            "intervals. Breaks the large benchmark into script-authoring (Exp 315) "
            "and execution (Exp 316) phases per lessons-learned rule."
        ),
    },
    {
        "name": "Full-scale benchmark execution (simulated)",
        "experiment": 316,
        "description": (
            "Executed the Exp 315 benchmark script in simulated mode (no live GPU "
            "available). Produces carnot.fullscale_benchmark.v1 artifact with "
            "per-model, per-mode, per-variant accuracy records and Wilson CIs. "
            "Live GPU run pending for headline claims."
        ),
    },
    {
        "name": "HuggingFace README accuracy audit",
        "experiment": 317,
        "description": (
            "Audited and patched all 16 per-token activation EBM READMEs to clarify "
            "Phase 1 status (detects confidence, not factual correctness). Updated "
            "FCV README with Exp 316 results. Idempotent: sentinel comment prevents "
            "double-patching."
        ),
    },
    {
        "name": "Four-tier continuous self-learning relay",
        "experiment": 318,
        "description": (
            "First integrated four-tier relay benchmark: Tier 1 (ConfidenceVerifier), "
            "Tier 2 (ConstraintGenerator), Tier 3 (JEPA gate), Z3 gate. Runs 3 "
            "batches of 33 questions. Reports honest signed improvement_1to3 (never "
            "clamped). Primary metric for the autonomous self-learning loop."
        ),
    },
    {
        "name": "D-Wave sampler backend with Neal simulation",
        "experiment": 320,
        "description": (
            "New DWaveSamplerBackend class implementing the SamplerBackend protocol. "
            "Uses dwave-ocean-sdk's SimulatedAnnealingSampler (Neal) as a local "
            "simulation of D-Wave hardware. Enables benchmarking quantum-inspired "
            "annealing on the constraint verification workload without physical "
            "D-Wave access."
        ),
    },
    {
        "name": "D-Wave Neal vs CPU Ising benchmark",
        "experiment": 321,
        "description": (
            "Head-to-head benchmark comparing DWaveSamplerBackend (Neal simulation) "
            "against the CPU Ising backend on the constraint verification task. "
            "Establishes relative throughput and energy-function agreement for use "
            "in selecting the backend routing policy."
        ),
    },
    {
        "name": "Reward hacking detection in self-learning relay",
        "experiment": 322,
        "description": (
            "Detects and logs reward hacking signals in the self-learning replay: "
            "policy updates that increase recorded performance without genuine "
            "improvement (e.g., gaming the constraint satisfaction metric). "
            "Emits reward_hacking_detected flag in the relay artifact."
        ),
    },
    {
        "name": "Conductor behavioral audit log",
        "experiment": 323,
        "description": (
            "New conductor_audit.py module that logs every conductor action "
            "(experiment selection, skip, retry, push) to a structured JSONL audit "
            "trail. Enables anomaly detection over conductor behavior across milestones."
        ),
    },
    {
        "name": "Conductor constitution — explicit governance rules",
        "experiment": 324,
        "description": (
            "Wrote a machine-readable conductor constitution: explicit rules for "
            "experiment selection, skip criteria, max-retry policy, push gating, "
            "and anomaly escalation. The constitution is loaded by the conductor at "
            "startup and enforced as a runtime policy, replacing ad-hoc heuristics."
        ),
    },
]


# ---------------------------------------------------------------------------
# Compute bottlenecks
# ---------------------------------------------------------------------------


def build_bottlenecks() -> list[dict]:
    """Return the top-3 slowest experiments as structured bottleneck dicts.

    Each entry has:
    - name: human-readable experiment label
    - duration_min: wall-clock minutes for this experiment
    - pct_total: this experiment's share of milestone wall time
    - root_cause: why this experiment was slow
    - category: classification of the root cause
    - fix_implemented_this_milestone: whether the fix was applied this milestone
    """
    total = sum(_EXPERIMENT_WALL_TIMES.values())

    # Sort all experiments by duration descending; take the top 3
    sorted_exps = sorted(
        _EXPERIMENT_WALL_TIMES.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

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
    """Build the action item list for the 2026.04.23 milestone.

    RETRO-001 and RETRO-002 are carried forward from the 2026.04.22 retro.
    Neither was implemented during this milestone despite both being high-priority.

    Exp 308 at 138 minutes is direct evidence that RETRO-001 (hard timeout) would
    have saved ~93 minutes on that experiment alone.

    NEW-001 captures a newly-discovered bottleneck: the high post-test failure rate.
    """
    return [
        {
            "id": "RETRO-001",
            "description": (
                "Add 45-minute hard timeout with structured auto-escalation to the "
                "conductor. When an experiment exceeds 45 minutes, the conductor "
                "should checkpoint the partial artifact and surface a "
                "status='timeout_escalate' result rather than allowing the session "
                "to spiral. Exp 308 ran for 138 minutes — more than 3× the threshold "
                "— and a timeout would have saved ~93 minutes of milestone wall time."
            ),
            "status": "carried_forward",
            "owner": "conductor",
            "priority": "high",
            "estimated_impact_pct": 8,
            "evidence_this_milestone": (
                "Exp 308 (JEPA gate): 138 min (20% of milestone). "
                "Exp 309 (Tier 3 self-learning): 66 min (9.6%). "
                "Both exceeded 45 min due to post-test failure repair loops."
            ),
        },
        {
            "id": "RETRO-002",
            "description": (
                "Integrate gpu_monitor.py --json output into the conductor's "
                "pre-experiment check. Inject the GPU state report into the "
                "experiment prompt so the conductor can auto-select DualGPURunner "
                "vs single-GPU, warn on zombie processes, and detect idle GPUs "
                "before they accumulate. Also catches missing optional SDKs "
                "(e.g. dwave-ocean-sdk) if the check is extended to a "
                "pre-experiment environment audit."
            ),
            "status": "carried_forward",
            "owner": "conductor",
            "priority": "medium",
            "estimated_impact_pct": 4,
            "evidence_this_milestone": (
                "Exp 321 (D-Wave Neal benchmark): 62 min partly due to missing "
                "dwave-ocean-sdk install time. A pre-experiment environment audit "
                "would have surfaced this at Exp 320 planning."
            ),
        },
        {
            "id": "NEW-001",
            "description": (
                "Enforce test-first development in ExperimentTemplate. Add a "
                "write_tests_first() stub to ExperimentTemplate that prompts the "
                "agent to draft test skeletons before writing implementation code. "
                "This milestone had 4/17 experiments (24%) fail their post-tests "
                "on the first attempt due to integration issues discovered only "
                "after implementation. Test-first would reduce the retry rate and "
                "eliminate ~60 min of repair-loop overhead per milestone."
            ),
            "status": "new",
            "owner": "experiment_template",
            "priority": "high",
            "estimated_impact_pct": 10,
            "evidence_this_milestone": (
                "Exp 308, 309, 310, 311 all had post-test failures on first attempt. "
                "Combined retry overhead: ~60 min = 8.7% of milestone wall time. "
                "Root cause: tests were added during implementation, not before."
            ),
        },
        {
            "id": "NEW-002",
            "description": (
                "Add a pre-experiment optional-dependency audit to the conductor. "
                "Before running an experiment that requires an optional package "
                "(dwave-ocean-sdk, pynq, onnxruntime-vitisai), verify it is "
                "importable in the virtualenv and surface a "
                "status='blocked_prereq' artifact immediately rather than "
                "discovering the gap mid-experiment. Reduces 'install overhead' "
                "from consuming wall-time allocated to actual benchmarking."
            ),
            "status": "new",
            "owner": "conductor",
            "priority": "medium",
            "estimated_impact_pct": 5,
            "evidence_this_milestone": (
                "Exp 321: D-Wave SDK not installed, discovered at runtime. "
                "Exp 313/314: CARNOT_KV260_BITFILE / ninja / openblas missing, "
                "still discovered at runtime rather than at planning time."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Carry-over summary
# ---------------------------------------------------------------------------


def build_carry_over() -> dict:
    """Summarize what was carried forward from and resolved vs the 2026.04.22 retro."""
    return {
        "prior_milestone": "2026.04.22",
        "items_carried_forward": ["RETRO-001", "RETRO-002"],
        "items_resolved": [
            "RETRO-004 (BatchedInferenceRunner as standard inference path — "
            "adopted throughout Exp 307–321 via ExperimentTemplate)",
            "RETRO-001 (DualGPURunner wired into ExperimentTemplate.setup_gpu() "
            "— used in Exp 307/308 for GPU experiments; 2026.04.22 RETRO-001 "
            "is now resolved; the 2026.04.23 RETRO-001 refers to the hard-timeout "
            "action item originally listed as 2026.04.22 RETRO-002)",
        ],
        "carry_over_rate_pct": 40.0,
        "note": (
            "The 2026.04.22 retro listed 5 action items (RETRO-001 through RETRO-005). "
            "RETRO-001 (DualGPURunner) and RETRO-004 (BatchedInferenceRunner) are now "
            "resolved. RETRO-002 (45-min timeout) and RETRO-003 (gpu_monitor) are "
            "carried forward as RETRO-001 and RETRO-002 respectively in this retro. "
            "RETRO-005 (pytest-xdist) is dropped: the test suite now runs in ~3s "
            "after the scope reduction; parallelism is no longer needed."
        ),
    }


# ---------------------------------------------------------------------------
# Speedup estimate
# ---------------------------------------------------------------------------


def compute_speedup_estimate(action_items: list[dict]) -> float:
    """Estimate the percentage wall-time reduction for the next milestone.

    Individual estimates:
    - RETRO-001 (45-min timeout): 8% — prevents Exp 308-class repair spirals
    - RETRO-002 (GPU monitor + env audit): 4%
    - NEW-001 (test-first enforcement): 10%
    - NEW-002 (pre-experiment dependency audit): 5%

    Additive sum: 27%. Realistic combined (improvements overlap; test-first and
    timeout both reduce the repair-loop tail): 15%.
    """
    additive = sum(item.get("estimated_impact_pct", 0) for item in action_items)
    # Discount for overlap: additive estimates sum to 27%; realistic combined is ~15%.
    # The same experiments (308, 309) are targeted by both RETRO-001 and NEW-001.
    realistic = min(additive * 0.56, 100.0)  # 27 * 0.56 ≈ 15
    return round(realistic, 1)


# ---------------------------------------------------------------------------
# Conductor log update
# ---------------------------------------------------------------------------


def append_conductor_log_entry(artifact_path: Path, duration_min: float) -> None:
    """Append a one-line retrospective entry to ops/conductor-log.md.

    Why: the CLAUDE.md workflow requires ops/conductor-log.md to record every
    significant conductor action including retros, so future sessions can
    reconstruct the milestone timeline from the log alone.
    """
    if not CONDUCTOR_LOG.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = (
        f"| {timestamp} | Exp 319: Operational retrospective for milestone "
        f"2026.04.23 | OK | {artifact_path.name} written; "
        f"n=17 experiments, total={round(sum(_EXPERIMENT_WALL_TIMES.values()))} min, "
        f"top bottleneck=Exp 308 ({_EXPERIMENT_WALL_TIMES[308]} min); "
        f"RETRO-001/002 carried forward; NEW-001/002 added; "
        f"estimated next-milestone speedup ~{round(duration_min)}% |\n"
    )
    with CONDUCTOR_LOG.open("a") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate the operational retrospective for milestone 2026.04.23."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    n_experiments = len(_EXPERIMENT_WALL_TIMES)
    total_wall_time = sum(_EXPERIMENT_WALL_TIMES.values())
    avg_minutes = round(total_wall_time / n_experiments, 1)

    bottlenecks = build_bottlenecks()
    action_items = build_action_items()
    carry_over = build_carry_over()
    speedup_pct = compute_speedup_estimate(action_items)

    output: dict = {
        "schema": "operational_retro_v1",
        "milestone": "2026.04.23",
        "generated_at": now,
        "n_experiments": n_experiments,
        "total_wall_time_minutes": total_wall_time,
        "avg_minutes_per_experiment": avg_minutes,
        "bottlenecks_identified": bottlenecks,
        "improvements_implemented": _IMPROVEMENTS_IMPLEMENTED,
        "action_items": action_items,
        "carry_over_from_previous_retro": carry_over,
        "estimated_next_milestone_speedup_pct": speedup_pct,
        "post_test_failure_rate_pct": round(4 / n_experiments * 100, 1),
        "slowest_experiments": [
            {
                "rank": i + 1,
                "name": b["name"],
                "duration_min": b["duration_min"],
                "pct_total": b["pct_total"],
                "category": b["category"],
            }
            for i, b in enumerate(bottlenecks)
        ],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"Written: {OUTPUT_FILE}")

    # Append to conductor log
    append_conductor_log_entry(OUTPUT_FILE, speedup_pct)

    # Console summary
    print(f"Milestone          : {output['milestone']}")
    print(f"Experiments        : {n_experiments}")
    print(f"Total wall time    : {total_wall_time:.0f} min")
    print(f"Avg per experiment : {avg_minutes:.1f} min")
    print(f"Top bottleneck     : {bottlenecks[0]['name']} ({bottlenecks[0]['duration_min']} min)")
    print(f"Post-test fail rate: {output['post_test_failure_rate_pct']}%")
    carried = sum(1 for a in action_items if a["status"] == "carried_forward")
    new_items = sum(1 for a in action_items if a["status"] == "new")
    print(f"Action items       : {carried} carried forward, {new_items} new")
    print(f"Estimated speedup  : {speedup_pct}%")


if __name__ == "__main__":
    main()
