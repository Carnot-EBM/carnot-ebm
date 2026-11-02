"""REQ-REPORT-8677: qualify callback routing through the submitted SDK wrapper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from arcengine import FrameData, GameState
import torch

from carnot.agentic import arc_reset_provenance as reset
from carnot.agentic.arc_observation_delivery import deliver, payload
from carnot.agentic.arc_outcome_observer import ArcOutcomeObserver
from carnot.reporting import arc_observation_8677 as e
from carnot.reporting import arc_observer_live_8637 as original
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]


def trace_sdk(ref: Json) -> Any:
    """Replay recorded public SDK frames without reading hidden game implementation."""
    episode = e.read(ref)
    steps = [
        json.loads(line)
        for line in Path(ref["path"]).with_name("steps.jsonl").read_text().splitlines()
    ]

    class Environment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self) -> Any:
            index = self.index
            self.index += 1
            row = steps[index]["observation"]
            meta = episode["observations"][index]
            return FrameData(
                frame=row["pixels"],
                state=GameState(meta["state"].split(".")[-1]),
                levels_completed=row["level"],
                full_reset=meta["full_reset"],
                available_actions=row["available_actions"],
                win_levels=20,
            )

        def step(self, *args: Any, **kwargs: Any) -> Any:
            return self.reset()

    class SDK:
        def make(self, *args: Any, **kwargs: Any) -> Any:
            return Environment()

        def open_scorecard(self) -> str:
            return "private_recorded_sdk_frame_fixture"

    return SDK()


def diagnose(p: Json, index: int, raw: Path) -> Json:
    """Execute the sealed pre-repair wrapper; do not weaken its original assertions."""
    ref = p["original_agent_reference"]
    e.read(p["actual_transfer_references"][index])
    spec = importlib.util.spec_from_file_location(
        "carnot.agentic._sealed_observer_agent", ref["path"]
    )
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO = e.ROOT
    historical = e.read(p["actual_transfer_references"][index])
    unit = dict(historical["unit"], observer="on", directory=str(raw))
    torch.manual_seed(unit["seed"])
    with patch.object(original.runtime, "agent_module", module):
        result = original.collect(unit, trace_sdk(p["actual_transfer_references"][index]), raw)
    count = sum(r["reason"] == "duplicate_observation" for r in result["observer_issues"])
    row = dict(
        episode_id=unit["episode_id"],
        game=unit["game"],
        seed=unit["seed"],
        duplicate_observation_count=count,
        expected=1022,
        passed=count == 1022 and not result["termination"]["error"],
        original_source_reference=ref,
        callback_order=[
            "E3AgentPolicy.is_done",
            "CarnotAgent.choose_action",
            "E3AgentPolicy.next_move",
            "SDK.reset_or_step",
            "collector_final",
        ],
        actual_wrapper=result["actual_wrapper"],
        primitive_reference=e.reference(raw / "public_stream.json"),
    )
    atomic_json(raw / "diagnosis.json", row)
    return row


def collect(unit: Json, sdk: Any, raw: Path) -> Json:
    """Observe real policy actions and reset causes with the recorder off and on."""
    observer = ArcOutcomeObserver()
    recorder = reset.ResetProvenance(limit=unit["max_actions"] * 4)
    factory = original.runtime.agent_module.make_carnot_agent
    active: list[Any] = []
    latest: list[Any] = []
    torch.manual_seed(unit["seed"])

    def recording_factory(*args: Any, **kwargs: Any) -> Any:
        cls = factory(*args, **kwargs)

        class Recording(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                active.append(self._policy)
                self._policy._arc_reset_provenance = recorder
                self._policy.explorer._arc_reset_provenance = recorder
                if unit["observer"] == "on":
                    self._policy.install_arc_outcome_observer(observer)

        return Recording

    class Environment:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def reset(self) -> Any:
            frame = reset.sdk_call(active[0], ("RESET", None), self.wrapped.reset)
            latest[:] = [frame]
            return frame

        def step(self, action: Any, **kwargs: Any) -> Any:
            frame = reset.sdk_call(active[0], action, lambda: self.wrapped.step(action, **kwargs))
            latest[:] = [frame]
            return frame

    class SDK:
        def make(self, *args: Any, **kwargs: Any) -> Any:
            return Environment(sdk.make(*args, **kwargs))

        def open_scorecard(self) -> Any:
            return sdk.open_scorecard()

    with patch.object(original.runtime.agent_module, "make_carnot_agent", recording_factory):
        result = original.collect(dict(unit, observer="off"), SDK(), raw)
    if latest:
        deliver(active[0], latest[0], "collector_final")
    observer.finish(result["end_reason"])
    keys = (
        "action_index",
        "origin",
        "initiating_policy_branch",
        "post_observation_sha256",
        "requested_action",
    )
    result.update(
        unit=unit,
        observer_rows=observer.rows(),
        observer_issues=observer.issues,
        delivery_rows=getattr(observer, "delivery_rows", []),
        reset_requests=[
            {k: r.get(k) for k in keys} for r in recorder.receipt()["reset_request_rows"]
        ],
        final_state=payload(latest[0]) if latest else None,
    )
    atomic_json(raw / "ownership_stream.json", result)
    return dict(result)


def control(case: str) -> Json:
    """Constructed action windows verify counting, never natural supervisor benefit."""
    observer = ArcOutcomeObserver()
    policy = SimpleNamespace(_arc_outcome_observer=observer)
    observer.redirect("constructed_marker", "constructed_control", 0, applied=True)
    frame = FrameData(frame=[[[1]]], state=GameState.NOT_FINISHED)
    for index in range(120 if case == "horizon" else 3):
        observer.applied({"kind": 1})
        deliver(policy, frame, "is_done")
        deliver(policy, frame.model_copy(deep=True), "choose_action")
        if case == "repeated_is_done":
            deliver(policy, frame, "is_done")
        e.progress("constructed_" + case, index + 1, (120 if case == "horizon" else 3) - index - 1)
    if case == "reset":
        observer.applied({"kind": "RESET"})
        deliver(policy, frame, "reset")
    elif case == "conflict":
        changed = frame.model_copy(deep=True)
        changed.frame = [[[2]]]
        deliver(policy, changed, "is_done")
    elif case == "terminal":
        observer.applied({"kind": 1})
        frame.state = GameState.GAME_OVER
        deliver(policy, frame, "is_done")
    else:
        observer.finish("cutoff")
    row = observer.rows()[0]
    expected = {
        "horizon": "horizon_completed",
        "reset": "reset",
        "conflict": "conflicting_duplicate",
        "terminal": "terminal",
        "repeated_is_done": "cutoff",
    }[case]
    return dict(
        case=case,
        constructed=True,
        passed=row["end_reason"] == expected
        and (case != "horizon" or row["observed_horizon"] == 120),
        observer_row=row,
        issues=observer.issues,
        callbacks=observer.delivery_rows,
        benefit_credit=0,
    )
