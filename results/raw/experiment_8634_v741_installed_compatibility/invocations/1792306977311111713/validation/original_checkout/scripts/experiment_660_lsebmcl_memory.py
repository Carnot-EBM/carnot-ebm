#!/usr/bin/env python3
"""Exp 660: LSEBMCL Constraint Memory — replay buffer prevents catastrophic forgetting.

Validates that LSEBMCLReplayBuffer keeps forgetting_rate < 0.05 across 3 simulated
constraint-template sessions (arXiv 2501.05495 criterion).

Spec: REQ-SELF-021, SCENARIO-SELF-027, SCENARIO-SELF-028
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.lsebmcl_replay import LSEBMCLReplayBuffer
from scripts.experiment_template import ExperimentTemplate

apply_env_autofix()

_watchdog = ExperimentTimeoutWatchdog(660, timeout_minutes=30)

tmpl = ExperimentTemplate(
    660,
    "LSEBMCL Constraint Memory",
    "results/experiment_660_lsebmcl_memory.json",
    requires_gpu=False,
)
tmpl.setup()

# Three sessions of constraint templates — errors the EBM learned to flag per session.
session_1_patterns = ["COMPUTE: 47 + 28 = 76", "total is 80", "result is 15"]
session_2_patterns = ["COMPUTE: 100 / 5 = 18", "therefore 25 apples", "sum is 90"]
session_3_patterns = ["COMPUTE: 3 * 12 = 37", "balance is 50", "so 7 items"]

# Toy EBM energy proxy: longer patterns have marginally higher energy.
# In production this would be replaced by KAEMEnergy.energy() on an embedded vector.
energy_fn = lambda p: float(len(p)) / 100.0  # noqa: E731

replay_buffer = LSEBMCLReplayBuffer(energy_fn=energy_fn, max_replay_per_session=5)

for session_id, patterns in enumerate(
    [session_1_patterns, session_2_patterns, session_3_patterns], 0
):
    replay_buffer.add_session(session_id, patterns)

forgetting_rate = replay_buffer.compute_forgetting_rate(
    [session_1_patterns, session_2_patterns, session_3_patterns]
)
lsebmcl_no_forgetting = forgetting_rate < 0.05

honest_verdict = (
    "lsebmcl_forgetting_controlled"
    if lsebmcl_no_forgetting
    else "lsebmcl_forgetting_above_threshold"
)

artifact = tmpl.build_result(
    {
        "schema": "carnot.lsebmcl_memory.v1",
        "n_sessions": 3,
        "forgetting_rate": forgetting_rate,
        "lsebmcl_no_forgetting": lsebmcl_no_forgetting,
        "arxiv_ref": "2501.05495",
        "honest_verdict": honest_verdict,
    },
    status="success",
)

os.makedirs("results", exist_ok=True)
with open("results/experiment_660_lsebmcl_memory.json", "w") as _f:
    json.dump(artifact, _f, indent=2)

tmpl.assert_deliverable_written()
