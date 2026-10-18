"""REQ-REPORT-8637: add passive records to the qualified official SDK loop."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import random
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.agentic import arc_live_panel_runtime_8384 as runtime
from carnot.agentic.arc_e3_outcome_transport import normalize_action
from carnot.agentic.arc_outcome_observer import ArcOutcomeObserver, visible
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]


def collect(unit: Json, sdk: Any, raw: Path) -> Json:
    """Return exact actions and public outcomes without replacing policy decisions."""
    factory = runtime.agent_module.make_carnot_agent
    actions: list[Json] = []
    rng: list[Json] = []
    observations: list[Any] = []
    termination: list[bool] = []
    active: list[Any] = []
    observer = ArcOutcomeObserver()
    calls = [0]
    forward = runtime.numeric.SmallFrameChangeCNN.forward

    def counting_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return forward(self, *args, **kwargs)

    def recording_factory(*args: Any, **kwargs: Any) -> Any:
        cls = factory(*args, **kwargs)

        class Recording(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                active.append(self)
                if unit["observer"] == "on":
                    self._policy.install_arc_outcome_observer(observer)

            def choose_action(self, frames: Any, latest: Any) -> Any:
                action = super().choose_action(frames, latest)
                actions.append(normalize_action(action))
                rng.append(
                    dict(
                        python=canonical_hash(repr(random.getstate())),
                        numpy=canonical_hash(repr(np.random.get_state())),
                    )
                )
                return action

            def is_done(self, frames: Any, latest: Any) -> Any:
                result = super().is_done(frames, latest)
                termination.append(bool(result))
                return result

        return Recording

    class Environment:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def reset(self) -> Any:
            result = self.wrapped.reset()
            observations.append(visible(result))
            return result

        def step(self, *args: Any, **kwargs: Any) -> Any:
            result = self.wrapped.step(*args, **kwargs)
            observations.append(visible(result))
            return result

    class SDK:
        def make(self, *args: Any, **kwargs: Any) -> Any:
            return Environment(sdk.make(*args, **kwargs))

        def open_scorecard(self) -> Any:
            return sdk.open_scorecard()

    flags = dict(
        CARNOT_ARC_DISABLE_INDUCTION="1",
        CARNOT_ARC_EFFECT_RECEIPTS="1",
        CARNOT_ARC_SUPERVISOR_CAPABILITY_MASK="1",
    )
    with (
        patch.dict(os.environ, flags),
        patch.object(runtime.agent_module, "make_carnot_agent", recording_factory),
        patch.object(runtime.numeric.SmallFrameChangeCNN, "forward", counting_forward),
    ):
        terminal = runtime.episode(unit, SDK(), raw)
    reason = terminal["censoring"] or "terminal"
    if unit["observer"] == "on" and observations:
        observer.observe(observer.action_index, observations[-1])
    observer.finish(reason)
    steps = [
        dict(index=i + 1, action=a, observation=o)
        for i, (a, o) in enumerate(zip(actions, observations, strict=False))
    ]
    events = []
    legacy = deepcopy(terminal["supervisor_receipt"])
    for row in legacy.get("direct_effect_receipts", []):
        if not row["selected"]:
            continue
        events.append(
            dict(
                event_id=row["event_id"],
                arm=row["arm"],
                wrapper_action_index=row["wrapper_action_index"],
                applied=row["applied"],
                observation=dict(level=row["snapshot"]["level"]),
                original_legacy_receipt=next(
                    (r for r in legacy["redirects"] if r["action_index"] == row["action_index"]),
                    None,
                ),
            )
        )
    result = dict(
        unit=unit,
        actions=actions,
        observations=observations,
        rng=rng,
        termination=dict(
            checks=termination, censoring=terminal["censoring"], error=terminal["error"]
        ),
        legacy=legacy,
        steps=steps,
        events=events,
        observer_rows=observer.rows(),
        observer_issues=observer.issues,
        end_reason=reason,
        llm_calls=terminal["llm_invocation_count"],
        numerical_head_calls=calls[0],
        model_tripwire_attempts=terminal["model_tripwire_attempts"],
        actual_wrapper=terminal["actual_wrapper_path"],
        duration_s=terminal["duration_s"],
    )
    atomic_json(raw / "public_stream.json", result)
    return result


def episode(path: Path) -> int:
    """A fresh child creates a fresh official SDK without model or network access."""
    from carnot.reporting.arc_supervisor_live_panel_8384 import sdk

    unit = json.loads(path.read_bytes())
    result = collect(unit, sdk(), Path(unit["directory"]))
    return int(bool(result["termination"]["error"] or result["model_tripwire_attempts"]))
