"""REQ-REPORT-8649: current adapter names reset provenance without changing history."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch
import torch

from carnot.agentic import arc_reset_provenance as reset
from carnot.reporting import arc_observer_live_8637 as original
from carnot.reporting.arc_observer_fixtures_8624 import frame
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
PARITY = (
    "actions",
    "rng",
    "torch_rng",
    "termination",
    "legacy",
    "observations",
    "observer_rows",
    "sdk_calls",
)


def parity(left: Json, right: Json) -> Json:
    """Exact observable behavior, including existing receipts, must remain unchanged."""
    equality = {key: left[key] == right[key] for key in PARITY}
    return dict(passed=all(equality.values()), equality=equality)


def collect(unit: Json, sdk: Any, raw: Path) -> Json:
    """Wrap the qualified SDK loop; recording never replaces a selected action."""
    factory = original.runtime.agent_module.make_carnot_agent
    active: list[Any] = []
    calls: list[Json] = []
    torch_rng: list[str] = []
    recorder = reset.ResetProvenance()
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(unit["seed"]).get_state())

    def recording_factory(*args: Any, **kwargs: Any) -> Any:
        cls = factory(*args, **kwargs)

        class Current(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                active.append(self._policy)
                if unit["reset_provenance"] == "on":
                    self._policy._arc_reset_provenance = recorder
                    self._policy.explorer._arc_reset_provenance = recorder
                    self._policy._arc_reset_max_actions = unit["max_actions"]
                    self._policy._arc_reset_deadline = time.monotonic() + unit["max_seconds"]

            def choose_action(self, frames: Any, latest: Any) -> Any:
                action = super().choose_action(frames, latest)
                torch_rng.append(canonical_hash(torch.get_rng_state().tolist()))
                return action

        return Current

    class Environment:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped

        def reset(self) -> Any:
            calls.append(dict(method="reset"))
            # This is the actual reset dispatch. Policy ownership was recorded earlier;
            # an SDK-only reset without that request must remain unknown.
            return reset.sdk_call(active[0], ("RESET", None), self.wrapped.reset)

        def step(self, action: Any, **kwargs: Any) -> Any:
            calls.append(dict(method="step", action=original.normalize_action(action), data=kwargs))
            return reset.sdk_call(active[0], action, lambda: self.wrapped.step(action, **kwargs))

    class SDK:
        def make(self, *args: Any, **kwargs: Any) -> Any:
            return Environment(sdk.make(*args, **kwargs))

        def open_scorecard(self) -> Any:
            return sdk.open_scorecard()

    with patch.object(original.runtime.agent_module, "make_carnot_agent", recording_factory):
        result = original.collect(dict(unit, observer="on"), SDK(), raw)
    result.update(
        reset_provenance=recorder.receipt(),
        torch_rng=torch_rng,
        sdk_calls=calls,
        recorder_errors=getattr(active[0], "_arc_reset_provenance_errors", []),
    )
    atomic_json(raw / "reset_stream.json", result)
    return dict(result)


def fixture_pair(raw: Path, mechanic: str) -> Json:
    """Scripted public mechanics test recording correctness, never game outcomes."""

    class Environment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self) -> Any:
            self.index += 1
            return frame(mechanic, self.index, "cutoff")

        def step(self, *args: Any, **kwargs: Any) -> Any:
            return self.reset()

    class SDK:
        def make(self, *args: Any, **kwargs: Any) -> Any:
            return Environment()

        def open_scorecard(self) -> str:
            return "private_reset_fixture"

    unit = dict(
        episode_id=mechanic,
        game="fixture-unknown",
        seed=11,
        arm="on",
        max_actions=128,
        max_seconds=120,
        capability_mask=True,
    )
    off, on = [
        collect(dict(unit, reset_provenance=mode), SDK(), raw / mode) for mode in ("off", "on")
    ]
    result = dict(
        mechanic=mechanic,
        **parity(off, on),
        off=off,
        on=on,
        public_fixture_only=True,
        primitive_checksum=canonical_hash([off, on]),
    )
    atomic_json(raw / "pair.json", result)
    return deepcopy(result)
