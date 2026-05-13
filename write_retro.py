import json
import os
from datetime import datetime

# 1. Update JSON
retro = {
  "schema": "carnot.operational_retro.v64",
  "milestone": "2026.05.159",
  "generated_at": "2026-05-13T08:21:41Z",
  "retro_type": "operational_full",
  "total_wall_time_minutes": 31.3,
  "experiments_completed": 15,
  "compute_bound_experiments_count": 0,
  "slowest_experiments": [
    {
      "experiment": "-0400 [conductor] Exp 2038: Milestone .159 Retrospective",
      "duration_minutes": 5.2,
      "compute_bound": False
    },
    {
      "experiment": "-0400 [conductor] Exp 2037: Milestone .159 Pre-Retro Audit",
      "duration_minutes": 4.1,
      "compute_bound": False
    },
    {
      "experiment": "-0400 [conductor] Doomed-rerun block: Exp 2029: Local SOTA GGUF Runtime Prefligh",
      "duration_minutes": 2.0,
      "compute_bound": False
    },
    {
      "experiment": "-0400 [conductor] Doomed-rerun block: Exp 2029: Local SOTA GGUF Runtime Prefligh",
      "duration_minutes": 2.0,
      "compute_bound": False
    },
    {
      "experiment": "-0400 [conductor] Pre-gate block: Exp 2031: Continuous Latent EBRM Trace Editing",
      "duration_minutes": 2.0,
      "compute_bound": False
    }
  ],
  "gpu_idle_on_compute_bound_tasks": None,
  "summary": "Milestone 2026.05.159 completed 15 synthesis-only experiments in 31.3 minutes total wall time (averaging 2 minutes each). There were 0 compute-bound experiments run, so GPU utilization was properly 0%.",
  "bottlenecks_identified": [
    "No data available this milestone for GPU bottlenecks, as 0 compute-bound experiments were run.",
    "The slowest actual tasks were Exp 2038 (Retrospective, 5.2 min) and Exp 2037 (Pre-Retro Audit, 4.1 min).",
    "Doomed-rerun blocks on Exp 2029 and Pre-gate blocks on Exp 2031 were successful time-savers, not bottlenecks."
  ],
  "improvements_suggested": [
    "Optimize the execution of retrospective and pre-retro audits (Exp 2037 and Exp 2038) to reduce wall time.",
    "Continue relying on the failure_ledger pre-launch check to block doomed reruns efficiently."
  ],
  "top_3_highest_leverage_actions": [
    "Optimize Pre-Retro Audit tooling.",
    "Optimize Retrospective generation tooling.",
    "Maintain doomed-rerun blocking mechanisms."
  ],
  "estimated_time_savings_pct": 14,
  "meta_reflection": "Which compute-bound experiments took the longest, and why? No data available this milestone. Was GPU utilization efficient on the compute-bound tasks? No data available this milestone. Did any compute-bound task with 2+ models in parallel fail to engage DualGPURunner? No data available this milestone. What tooling change would speed up the next milestone? Optimize the execution of retrospective and pre-retro audits (Exp 2037 and Exp 2038)."
}

with open("results/operational_retro_2026_05_159.json", "w") as f:
    json.dump(retro, f, indent=2)

# 2. Append to changelog
summary = "- 2026-05-13: Milestone 2026.05.159 operational retrospective completed. 15 experiments in 31.3 mins. Zero compute-bound tasks.\n"
with open("ops/changelog.md", "a") as f:
    f.write(summary)

# 3. Append to roadmap.md if it has the table
if os.path.exists("docs/roadmap.md"):
    with open("docs/roadmap.md", "r") as f:
        content = f.read()
    if "Completed Milestones" in content:
        row = "| 2026.05.159 | Operational Retrospective | 2026-2038 | Zero compute-bound tasks, failure_ledger savings |\n"
        with open("docs/roadmap.md", "a") as f:
            f.write(row)
