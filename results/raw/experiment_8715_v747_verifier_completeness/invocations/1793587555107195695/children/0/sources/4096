"""REQ-VERIFY-8536: real wrapper controls qualify selection without model calls."""

from __future__ import annotations

import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_effect_receipts as receipts
from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_generalization_runtime import withheld_policy_inputs
from carnot.agentic.arc_live_panel_runtime_8384 import no_models
from carnot.agentic.arc_supervisor_capability import FLAG
from carnot.agentic.arc_trajectory_supervisor import TrajectorySupervisor
from carnot.reporting.arc_effect_fixtures_8522 import Base
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
CASES = ["disabled", "enabled", "unknown", "active", "transition", "changing", "shadow"]


def run(case: str, masked: bool, recorded: bool) -> Json:
    """Script visible observations through the factory without substituting selection."""
    flags = {
        FLAG: str(int(masked)),
        receipts.FLAG: str(int(recorded)),
        "CARNOT_ARC_DISABLE_INDUCTION": "0" if case == "enabled" else "1",
        "CARNOT_ARC_SUPERVISOR_TOOL_ARM": "0",
        "CARNOT_ARC_TRAJECTORY_SUPERVISOR": str(int(case != "shadow")),
        "CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW": "120",
    }
    random.seed(7338536)
    np.random.seed(7338536)
    with (
        withheld_policy_inputs(),
        patch.dict(os.environ, flags),
        no_models() as attempts,
        patch.object(agent, "_load_submitted_frame_change_scorer", return_value=None),
        patch.object(agent, "load_solutions", return_value={}),
    ):
        wrapper = agent.make_carnot_agent(Base, cascade=True, proposer=None)(game_id="private")
        policy = wrapper._policy
        actions = [dict(name=wrapper.choose_action([], None).name, data=None)]
        policy._trajectory_supervisor = TrajectorySupervisor(window=1)
        policy.explore_budget = 100000
        policy.induced = True
        policy._induction_attempt_count = 1
        grid = np.zeros((8, 8), dtype=int)
        policy.transitions = [Transition(grid, 1, None, grid, 0, 0)] * 201
        policy._transitions_at_last_induction_attempt = 0
        explorer = policy.explorer
        explorer.goal_bias = None
        explorer._hybrid_diversity = case == "active"
        explorer._steps_since_progress = 151 if case == "active" else 0
        if case == "unknown":
            policy.trajectory_runtime_capabilities = lambda: {"llm_reinduction": None}
        frame = SimpleNamespace(
            frame=[grid.tolist()],
            levels_completed=0,
            available_actions=[1, 2, 3, 4, 5, 6],
            state="NOT_FINISHED",
        )
        for index in range(3):
            if case == "transition" and index == 1:
                frame.levels_completed = 1
            if case == "changing" and index == 1:
                os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "0"
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
        ledger = policy.trajectory_supervisor_diagnostics()
        ledger.pop("direct_effect_receipts", None)
        selected = ledger.get("redirects", ledger.get("would_have_redirects", []))
        return dict(
            case=case,
            masked=masked,
            wrapper="make_carnot_agent/E3AgentPolicy/choose_action",
            solve_provenance="development_proxy",
            fixture_is_live_evidence=False,
            actions=actions,
            selected_arms=[r["arm"] for r in selected],
            ledger=ledger,
            state=dict(
                induced=policy.induced,
                phase=policy.phase,
                hybrid_diversity=explorer._hybrid_diversity,
                steps=explorer._steps_since_progress,
                transitions=len(policy.transitions),
                provenance=policy.action_provenance(),
            ),
            rng_sha256=canonical_hash([repr(random.getstate()), repr(np.random.get_state())]),
            receipts=receipts.read(policy),
            llm_calls=0,
        )


def collect(raw: Path) -> Json:
    """Compare receipt-on/off actions, state and randomness inside each policy."""
    rows, parity = [], []
    for index, (case, masked) in enumerate((c, m) for c in CASES for m in [False, True]):
        print(f"[exp8536] fixtures completed={index} pending={14 - index}", flush=True)
        off, on = run(case, masked, False), run(case, masked, True)
        fields = ["actions", "state", "rng_sha256", "ledger", "selected_arms", "llm_calls"]
        parity.append(
            dict(
                case=case,
                masked=masked,
                compared_fields=fields,
                passed=all(off[k] == on[k] for k in fields),
                off_sha256=canonical_hash({k: off[k] for k in fields}),
                on_sha256=canonical_hash({k: on[k] for k in fields}),
            )
        )
        rows.append(on)
    result = dict(selection_fixture_rows=rows, action_parity_rows=parity)
    atomic_json(raw / "wrapper_primitives.json", result)
    return result


def replay(value: Json) -> bool:
    """Re-run real wrappers so self-consistent invented fixture effects still fail."""
    with TemporaryDirectory(prefix="exp8536-fixtures-", dir="/var/tmp") as directory:
        return collect(Path(directory)) == value
