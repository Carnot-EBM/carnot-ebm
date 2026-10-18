"""REQ-REPORT-8624: transfer fixtures exercise the real wrapper without games.

Scripted pixels represent three generic mechanics. Their levels are test inputs,
not discovered solutions. Model and adapter tripwires remain active throughout.
"""

from __future__ import annotations

from copy import deepcopy
import os
import random
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.agentic import arc_competition_agent as live
from carnot.agentic.arc_e3_outcome_transport import normalize_action
from carnot.agentic.arc_generalization_runtime import withheld_policy_inputs
from carnot.agentic.arc_live_panel_runtime_8384 import no_models
from carnot.agentic.arc_outcome_observer import ArcOutcomeObserver, visible
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
MECHANICS = ("navigation", "toggle", "sequence")
CASES = (
    "full_horizon",
    "early_levelup",
    "reset",
    "terminal",
    "cutoff",
    "missing",
    "duplicate",
    "conflicting_duplicate",
    "overlap",
)


def frame(mechanic: str, index: int, case: str) -> Any:
    """Supply only public frame fields; no hidden game state exists here."""
    pixels = np.zeros((1, 8, 8), dtype=np.int8)
    if mechanic == "navigation":
        pixels[0, index % 8, (index // 8) % 8] = 3
    elif mechanic == "toggle":
        pixels[0, 2:5, 2:5] = index % 2 + 1
    else:
        pixels[0, 0, : index % 7 + 1] = 4
    for bit in range(8):
        pixels[0, 4, bit] = 5 + ((index >> bit) & 1)
    return SimpleNamespace(
        frame=pixels,
        levels_completed=int(case in {"early_levelup", "overlap"} and index >= 5),
        state="GAME_OVER" if case == "terminal" and index == 8 else "NOT_FINISHED",
        full_reset=case == "reset" and index == 8,
        available_actions=[1, 2, 3, 4],
        win_levels=20,
    )


def run(mechanic: str, case: str, enabled: bool) -> Json:
    """Compare original policy decisions, RNG and receipts under identical inputs."""

    class Base:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

        def cleanup(self, scorecard: Any = None) -> None:
            pass

    random.seed(8624)
    np.random.seed(8624)
    observer = ArcOutcomeObserver()
    actions, observations, termination = [], [], []
    flags = dict(
        CARNOT_ARC_DISABLE_INDUCTION="1",
        CARNOT_ARC_TRAJECTORY_SUPERVISOR="1",
        CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW="2",
        CARNOT_ARC_SUPERVISOR_TOOL_ARM="0",
        CARNOT_ARC_EFFECT_RECEIPTS="1",
    )
    with (
        withheld_policy_inputs(),
        patch.dict(os.environ, flags),
        no_models() as attempts,
        patch.dict(live.SUBMITTED_AGENT_CONFIG, {"search_mode": "depth_first_ride"}),
        patch.object(live, "_load_submitted_frame_change_scorer", return_value=None),
        patch.object(live, "load_solutions", return_value={}),
    ):
        agent = live.make_carnot_agent(Base)(game_id="fixture-unknown")
        policy = agent._policy
        policy.explorer.max_depth = 1024
        if enabled:
            policy.install_arc_outcome_observer(observer)
        frames: list[Any] = []
        for index in range(125 if case in {"full_horizon", "duplicate"} else 10):
            latest = frame(mechanic, index, case)
            if case == "missing" and index == 5:
                latest = None
            if enabled and index in ({2, 3} if case == "overlap" else {2}):
                observer.redirect(f"scripted-marker:{index}", "scripted_marker", 0, applied=True)
            termination.append(agent.is_done(frames, latest))
            action = agent.choose_action(frames, latest)
            actions.append(normalize_action(action))
            observations.append(visible(latest))
            frames.append(latest) if latest is not None else None
            if enabled and case in {"duplicate", "conflicting_duplicate"} and index == 6:
                policy.observe_action_outcome(frame(mechanic, index + 1, case))
                other = frame(mechanic, index + 1, case)
                if case == "conflicting_duplicate":
                    other.levels_completed = 7
                policy.observe_action_outcome(other)
        if enabled:
            policy.observe_action_outcome(frame(mechanic, index + 1, case))
        observer.finish("cutoff")
        agent._witness_emitted = True
        agent.cleanup()
        receipt = deepcopy(policy.trajectory_supervisor_diagnostics())
        effects = deepcopy(getattr(policy, "_trajectory_effect_receipts", []))
    return dict(
        actions=actions,
        observations=observations,
        termination=termination,
        rng=repr(random.getstate()),
        numpy_rng=repr(np.random.get_state()),
        receipt=receipt,
        effects=effects,
        observer_rows=observer.rows(),
        issues=observer.issues,
        model_tripwire_attempts=attempts,
    )


def qualify() -> list[Json]:
    """Reachability and parity qualify instrumentation, never a learning score."""
    rows = []
    old_python, old_numpy = random.getstate(), np.random.get_state()
    try:
        for mechanic in MECHANICS:
            for case in CASES:
                off, on = run(mechanic, case, False), run(mechanic, case, True)
                keys = ["actions", "termination", "rng", "numpy_rng", "receipt", "effects"]
                equality = {key: off[key] == on[key] for key in keys}
                marker = next(
                    r for r in on["observer_rows"] if r["redirect_id"] == "scripted-marker:2"
                )
                expected_status = (
                    "no_progress_within_horizon"
                    if case in {"full_horizon", "duplicate"}
                    else "level_progress_observed"
                    if case in {"early_levelup", "overlap"}
                    else "censored"
                )
                equality["horizon_status"] = marker["horizon_status"] == expected_status
                rows.append(
                    dict(
                        mechanic=mechanic,
                        case=case,
                        unit_id=f"{mechanic}:{case}",
                        status="completed",
                        absolute_metric=int(all(equality.values())),
                        missing_reason=None,
                        passed=all(equality.values()) and not on["model_tripwire_attempts"],
                        equality=equality,
                        action_count=len(on["actions"]),
                        action_sha256=canonical_hash(on["actions"]),
                        legacy_sha256=canonical_hash(on["receipt"]),
                        wrapper="make_carnot_agent.choose_action",
                        model_calls=0,
                        adapter_calls=0,
                        solve_provenance="development_proxy",
                        observer_rows=on["observer_rows"],
                        overlap_count=sum(
                            bool(r["overlapping_redirect_ids"]) for r in on["observer_rows"]
                        ),
                        issues=on["issues"],
                    )
                )
                print(
                    f"[exp8624] fixtures completed={len(rows)} pending={27 - len(rows)}", flush=True
                )
    finally:
        random.setstate(old_python)
        np.random.set_state(old_numpy)
    return rows
