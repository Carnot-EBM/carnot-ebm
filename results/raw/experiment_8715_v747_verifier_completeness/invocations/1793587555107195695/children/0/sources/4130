"""REQ-REPORT-8678: observe the actual wrapper with reset recording off and on."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import torch

from carnot.agentic import arc_reset_provenance as reset
from carnot.agentic.arc_observation_delivery import deliver, payload
from carnot.agentic.arc_outcome_observer import ArcOutcomeObserver
from carnot.reporting import arc_observer_live_8637 as original
from carnot.reporting import arc_observation_live_8677 as qualified
from carnot.reporting import arc_observer_transfer_8678 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]


def collect(unit: Json, sdk: Any, raw: Path) -> Json:
    """Install the qualified observer once and keep reset recording observational."""
    observer = ArcOutcomeObserver()
    recorder = reset.ResetProvenance(limit=unit["max_actions"] * 4)
    factory = original.runtime.agent_module.make_carnot_agent
    active: list[Any] = []
    latest: list[Any] = []
    calls: list[Json] = []
    torch_rng: list[str] = []
    torch.manual_seed(unit["seed"])

    def recording_factory(*args: Any, **kwargs: Any) -> Any:
        cls = factory(*args, **kwargs)

        class Recording(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                policy = self._policy
                active.append(policy)
                policy.install_arc_outcome_observer(observer)
                if unit["reset_provenance"] == "on":
                    policy._arc_reset_provenance = recorder
                    policy.explorer._arc_reset_provenance = recorder
                    policy._arc_reset_max_actions = unit["max_actions"]
                    policy._arc_reset_deadline = time.monotonic() + unit["max_seconds"]

            def choose_action(self, frames: Any, latest: Any) -> Any:
                action = super().choose_action(frames, latest)
                torch_rng.append(canonical_hash(torch.get_rng_state().tolist()))
                return action

        return Recording

    class Environment:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def reset(self) -> Any:
            calls.append(dict(method="reset"))
            frame = reset.sdk_call(active[0], ("RESET", None), self.wrapped.reset)
            latest[:] = [frame]
            return frame

        def step(self, action: Any, **kwargs: Any) -> Any:
            calls.append(dict(method="step", action=original.normalize_action(action), data=kwargs))
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
    result.update(
        unit=unit,
        observer_rows=observer.rows(),
        observer_issues=observer.issues,
        delivery_rows=getattr(observer, "delivery_rows", []),
        reset_provenance=recorder.receipt(),
        reset_requests=recorder.receipt()["reset_request_rows"],
        recorder_error_metadata=[
            r for p in active for r in getattr(p, "_arc_reset_provenance_diagnostics", [])
        ],
        final_state=payload(latest[0]) if latest else None,
        sdk_calls=calls,
        torch_rng=torch_rng,
    )
    implementation = Path(inspect.getfile(type(sdk)))
    result["sdk_identity"] = dict(
        class_name=type(sdk).__module__ + "." + type(sdk).__qualname__,
        official=type(sdk).__module__.startswith("arc_agi."),
        path=str(implementation),
        sha256=sha256_file(implementation),
    )
    atomic_json(raw / "transfer_stream.json", result)
    return dict(result)


def episode(path: Path) -> int:
    """Each finite child enters the real wrapper; frame fixtures receive no live credit."""
    from carnot.reporting.arc_supervisor_live_panel_8384 import sdk

    unit = json.loads(path.read_bytes())
    e.progress("official_SDK_episode_before")
    arcade = (
        qualified.trace_sdk(unit["fixture_reference"]) if "fixture_reference" in unit else sdk()
    )
    result = collect(unit, arcade, Path(unit["directory"]))
    e.progress(
        "official_SDK_episode_after",
        len(result["actions"]),
        unit["max_actions"] - len(result["actions"]),
    )
    return int(
        bool(
            result["termination"]["error"]
            or result["observer_issues"]
            or result["recorder_error_metadata"]
            or result["model_tripwire_attempts"]
        )
    )
