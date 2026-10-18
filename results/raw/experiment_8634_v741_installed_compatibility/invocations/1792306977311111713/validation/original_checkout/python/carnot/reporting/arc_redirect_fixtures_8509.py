"""REQ-REPORT-8509: private states isolate real policy mutations without a game or model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot.agentic.arc_competition_agent import E3AgentPolicy, StepwiseExplorer
from carnot.agentic.arc_live_panel_runtime_8384 import no_models
from carnot.agentic.arc_trajectory_supervisor import Redirect, TrajectorySupervisor
from carnot.experiment_7526_v658_arc_eligibility import (
    EligibilityReceiptRecorder,
    policy_redirect_state,
)
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def fixtures(raw: Path, mode: str) -> list[Json]:
    """Use actual supervisor and policy methods; scripted states grant no scientific credit."""
    from carnot.reporting.arc_redirect_effect_8509 import effect, progress

    rows = []
    flags = {"CARNOT_ARC_DISABLE_INDUCTION": "1", "CARNOT_ARC_SUPERVISOR_TOOL_ARM": "0"}
    for index, mechanism in enumerate(
        ["allow_true_latch", "allow_false_latch", "force_diversity", "application_exception"]
    ):
        progress("private_fixture_before", index, 4 - index)
        path = raw / (mechanism + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        policy = E3AgentPolicy.__new__(E3AgentPolicy)
        explorer = StepwiseExplorer.__new__(StepwiseExplorer)
        explorer.goal_bias = None
        explorer._hybrid_diversity = mechanism.startswith("allow")
        explorer._steps_since_progress = 151 if explorer._hybrid_diversity else 0
        explorer._stall_threshold = 150
        policy.explorer = explorer
        policy.induced = mechanism != "allow_false_latch"
        policy.transitions = [None] * 201
        policy._transitions_at_last_induction_attempt = 0
        policy._induction_attempt_count = 1
        policy._trajectory_supervisor_applies = mode == "applied"
        policy._trajectory_supervisor_errors = 0
        policy._typed_arc_shadow_monitor = None
        policy._typed_arc_shadow_monitor_errors = 0
        policy._trajectory_eligibility_recorder = EligibilityReceiptRecorder("private", path=path)
        policy._trajectory_supervisor = TrajectorySupervisor(window=1)
        policy._level_reinduction_pending = False
        policy._episode_transition_start = 0
        policy._episode_dsl_transition_start = 0
        policy._dsl_transitions = []
        policy._pending_induction_reason = None
        policy._current_goal_level = 1
        policy.proposer = None
        policy.induction_attempts = []
        if mechanism == "application_exception":
            explorer.goal_bias = lambda *_: 0
        before = canonical_hash(policy_redirect_state(policy))
        with patch.dict(os.environ, flags), no_models() as attempts:
            if mechanism == "allow_false_latch":
                # The real selector excludes this state. Direct mutation is an
                # explicit control for the proposed already-false explanation.
                if mode == "applied":
                    policy._apply_trajectory_redirect(
                        Redirect("allow_reinduction", 1, 0, "private control")
                    )
                event = dict(
                    application_disposition="no_state_change"
                    if mode == "applied"
                    else "shadow_recommendation",
                    old_state_hash=before,
                    new_state_hash=canonical_hash(policy_redirect_state(policy)),
                )
            else:
                with patch.object(
                    explorer, "set_goal_bias", side_effect=ValueError("private intentional failure")
                ):
                    policy._maybe_supervise_trajectory(SimpleNamespace(levels_completed=0))
                event = json.loads(path.read_text().splitlines()[-1])
            after = canonical_hash(policy_redirect_state(policy))
            enter, reason = policy._should_enter_induction(stalled=True, won=False)
            induction = None
            if enter and mechanism == "allow_true_latch" and mode == "applied":
                policy._pending_induction_reason = reason
                policy._induce_and_plan()
                induction = policy.induction_attempts[-1]
            if attempts:
                raise ValueError("fixture_model_tripwire")
        rows.append(
            dict(
                mechanism=mechanism,
                mode=mode,
                fixture_is_live_evidence=False,
                old_state_hash=before,
                new_state_hash=after,
                effect_class=effect(mode, event),
                application_receipt=event,
                induction_tier_entered=induction is not None,
                induction_attempt=induction,
                llm_calls=0,
                model_disabled_reinduction=bool(
                    induction and induction["skipped"] == "disabled_by_env"
                ),
                selector_reachable=mechanism != "allow_false_latch",
                shadow_noninterference=before == after if mode == "shadow" else None,
            )
        )
        progress("private_fixture_after", index + 1, 3 - index)
    return rows
