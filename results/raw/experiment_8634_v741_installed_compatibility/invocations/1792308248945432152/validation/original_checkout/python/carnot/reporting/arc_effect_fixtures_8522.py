"""REQ-REPORT-8522: scripted observations qualify instrumentation, never science."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_effect_receipts as receipt
from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_generalization_runtime import withheld_policy_inputs
from carnot.agentic.arc_live_panel_runtime_8384 import no_models
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER, TrajectorySupervisor
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
CASES = [
    "shadow",
    "ineligible",
    "already_active",
    "drop_bias",
    "reinduction",
    "diversity",
    "malformed",
    "missing",
    "application_error",
]


class Base:
    """The real wrapper needs only its public game identifier in this CPU fixture."""

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id


def run(case: str, enabled: bool) -> Json:
    """Use the production factory and choose_action with a fixed observation stream."""
    flags = {
        receipt.FLAG: str(int(enabled)),
        "CARNOT_ARC_DISABLE_INDUCTION": "1",
        "CARNOT_ARC_SUPERVISOR_TOOL_ARM": "0",
        "CARNOT_ARC_TRAJECTORY_SUPERVISOR": str(int(case != "shadow")),
        "CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW": "120",
    }
    random.seed(7328522)
    np.random.seed(7328522)
    with (
        withheld_policy_inputs(),
        patch.dict(os.environ, flags),
        no_models() as attempts,
        patch.object(agent, "_load_submitted_frame_change_scorer", return_value=None),
        patch.object(agent, "load_solutions", return_value={}),
    ):
        wrapper = agent.make_carnot_agent(Base, cascade=True, proposer=None)(
            game_id="private-fixture"
        )
        policy = wrapper._policy
        actions = [dict(name=wrapper.choose_action([], None).name)]
        grid = np.zeros((8, 8), dtype=int)
        frame = SimpleNamespace(
            frame=[grid.tolist()],
            levels_completed=0,
            available_actions=[1, 2, 3, 4, 5, 6],
            state="NOT_FINISHED",
        )
        policy._trajectory_supervisor = TrajectorySupervisor(window=1)
        policy.induced = True
        policy._induction_attempt_count = 1
        policy._transitions_at_last_induction_attempt = 0
        policy.transitions = [Transition(grid, 1, None, grid, 0, 0)] * 201
        explorer = policy.explorer
        explorer.goal_bias = None
        explorer._hybrid_diversity = case not in ["shadow", "diversity", "malformed", "missing"]
        explorer._steps_since_progress = 151 if explorer._hybrid_diversity else 0
        if case in ["ineligible", "already_active", "malformed", "missing"]:
            policy._trajectory_supervisor._arms_used = set(ARM_ORDER)
        if case == "already_active":
            policy._trajectory_supervisor._first_eligible_arm = lambda _: (
                "force_exploration_diversity",
                "private no-op control",
            )
        if case == "diversity":
            policy._trajectory_supervisor._arms_used.add("allow_reinduction")
        if case in ["drop_bias", "application_error"]:
            explorer.goal_bias = lambda *_: 0.0
        if case == "reinduction":
            policy.phase = "induce"
        if case == "malformed":
            explorer._stall_threshold = "bad"
        if case == "missing":
            del explorer._stall_threshold
        with (
            patch.object(explorer, "set_goal_bias", side_effect=ValueError("private failure"))
            if case == "application_error"
            else patch.dict(os.environ, {})
        ):
            for index in range(3):
                action = wrapper.choose_action([frame] * (index + 1), frame)
                payload = action.action_data.model_dump()
                actions.append(
                    dict(
                        name=action.name,
                        data={k: payload[k] for k in ["x", "y"] if k in payload}
                        if action.name == "ACTION6"
                        else None,
                    )
                )
        if attempts:
            raise ValueError("fixture_model_tripwire")
        diagnostics = policy.trajectory_supervisor_diagnostics()
        diagnostics.pop("direct_effect_receipts", None)
        state = dict(
            induced=policy.induced,
            phase=policy.phase,
            transitions=len(policy.transitions),
            hybrid_diversity=explorer._hybrid_diversity,
            steps=explorer._steps_since_progress,
            goal_bias_present=explorer.goal_bias is not None,
            provenance=policy.action_provenance(),
            supervisor_state=diagnostics,
        )
        # Timed induction fields are not behavioral operands. Keep the actual
        # stable outcome and compare every action, transition and RNG state.
        return dict(
            case=case,
            solve_provenance="development_proxy",
            fixture_is_live_evidence=False,
            wrapper="make_carnot_agent/E3AgentPolicy/choose_action",
            actions=actions,
            policy_state=state,
            rng_sha256=canonical_hash([repr(random.getstate()), repr(np.random.get_state())]),
            receipts=receipt.read(policy),
            llm_calls=0,
            selector_injected=case == "already_active",
        )


def collect(raw: Path) -> Json:
    """Compare collection off/on for every control before any live panel is permitted."""
    rows, parity = [], []
    for index, case in enumerate(CASES):
        print(
            f"[exp8522] fixture_before completed={index} pending={len(CASES) - index}", flush=True
        )
        control, observed = run(case, False), run(case, True)
        compared = ["actions", "policy_state", "rng_sha256", "llm_calls"]
        parity.append(
            dict(
                case=case,
                compared_fields=compared,
                control_sha256=canonical_hash({k: control[k] for k in compared}),
                observed_sha256=canonical_hash({k: observed[k] for k in compared}),
                passed=all(control[k] == observed[k] for k in compared),
            )
        )
        rows.append(observed)
        print(
            f"[exp8522] fixture_after completed={index + 1} pending={len(CASES) - index - 1}",
            flush=True,
        )
    value = dict(mutation_fixture_rows=rows, action_parity_rows=parity)
    atomic_json(raw / "wrapper_primitives.json", value)
    return value


def replay(value: Json) -> bool:
    """Re-execute the wrapper so rehashed invented effects cannot become evidence."""
    with TemporaryDirectory(prefix="exp8522-fixture-replay-", dir="/var/tmp") as directory:
        return collect(Path(directory)) == value
