"""REQ-REPORT-8677: route repeated callbacks without weakening the observer."""

from copy import deepcopy
from types import SimpleNamespace

from arcengine import FrameData, GameState
import pytest

from carnot.agentic.arc_outcome_observer import ArcOutcomeObserver
from carnot.agentic.arc_observation_delivery import deliver, payload


def test_ingress_owns_one_delivery_and_detects_pixel_conflicts():
    """SCENARIO-REPORT-8677-OWNERSHIP: identical callbacks cannot hide changed pixels."""
    observer = ArcOutcomeObserver()
    policy = SimpleNamespace(_arc_outcome_observer=observer)
    frame = FrameData(frame=[[[1]]], state=GameState.NOT_FINISHED, levels_completed=0)
    deliver(SimpleNamespace(), frame, "disabled")
    deliver(policy, None, "initial")
    observer.redirect("control", "constructed", 0, applied=True)
    observer.applied({"kind": 1})
    for owner in ("is_done", "choose_action", "next_move", "is_done"):
        deliver(policy, frame, owner)
    assert observer.rows()[0]["observed_horizon"] == 1
    assert observer.issues == []
    assert [r["status"] for r in observer.delivery_rows] == ["canonical"] + [
        "duplicate_callback"
    ] * 3
    assert observer.delivery_rows[0]["owner"] == "is_done"
    changed = frame.model_copy(deep=True)
    changed.frame = [[[2]]]
    deliver(policy, changed, "is_done")
    assert observer.issues[-1]["reason"] == "conflicting_duplicate"
    assert observer.rows()[0]["end_reason"] == "conflicting_duplicate"
    assert observer.delivery_rows[-1]["status"] == "conflicting_duplicate"
    assert payload({"level": 0, "pixels": [[[1]]]})["pixels"] == [[[1]]]
    assert payload(None) is None


@pytest.mark.parametrize("ending", ["horizon", "reset", "terminal", "missing", "level"])
def test_distinct_actions_and_horizon_controls(ending):
    """SCENARIO-REPORT-8677-OWNERSHIP: equal pixels still count distinct actions."""
    observer = ArcOutcomeObserver()
    policy = SimpleNamespace(_arc_outcome_observer=observer)
    observer.redirect("control", "constructed", 0, applied=True)
    frame = FrameData(frame=[[[1]]], state=GameState.NOT_FINISHED)
    for _ in range(120 if ending == "horizon" else 2):
        observer.applied({"kind": 1})
        deliver(policy, frame, "is_done")
        deliver(policy, deepcopy(frame), "choose_action")
    if ending == "reset":
        observer.applied({"kind": "RESET"})
        deliver(policy, frame, "reset")
    elif ending != "horizon":
        observer.applied({"kind": 1})
        latest = None if ending == "missing" else frame.model_copy(deep=True)
        if ending == "terminal":
            latest.state = GameState.GAME_OVER
        elif ending == "level":
            latest.levels_completed = 1
        deliver(policy, latest, "is_done")
    row = observer.rows()[0]
    assert (
        row["end_reason"]
        == {
            "horizon": "horizon_completed",
            "reset": "reset",
            "terminal": "terminal",
            "missing": "missing_observation",
            "level": "level_progress",
        }[ending]
    )
    if ending == "horizon":
        assert row["observed_horizon"] == 120
    assert observer.issues == []


def test_observer_error_does_not_replace_policy_action():
    """REQ-REPORT-8677: a broken receipt remains visible without stopping action choice."""
    observer = ArcOutcomeObserver()
    observer.action_index = 1
    policy = SimpleNamespace(_arc_outcome_observer=observer)
    deliver(policy, {"level": 0}, "is_done")
    assert observer.issues == [dict(reason="observer_error", error="KeyError")]
    import numpy as np

    assert payload(SimpleNamespace(frame=np.array([[[1]]]))) == dict(
        level=None,
        state="",
        full_reset=False,
        pixels=[[[1]]],
        available_actions=[],
        win_levels=None,
    )


def test_actual_sdk_list_of_numpy_frames_is_lossless_and_json_safe():
    """REQ-REPORT-8677: installed SDK lists of arrays retain every public pixel."""
    import json
    import numpy as np
    from carnot.reporting.current_work_receipt import canonical_hash

    pixels = np.zeros((64, 64), dtype=np.int8)
    actual_sdk = SimpleNamespace(
        frame=[pixels],
        levels_completed=0,
        state="NOT_FINISHED",
        available_actions=[1, 2],
        full_reset=False,
    )
    encoded = payload(actual_sdk)
    assert json.loads(json.dumps(encoded))["pixels"] == [pixels.tolist()]
    before = canonical_hash(encoded)
    pixels[32, 32] = 4
    assert canonical_hash(payload(actual_sdk)) != before
    observer = ArcOutcomeObserver()
    observer.applied({"kind": 1})
    policy = SimpleNamespace(_arc_outcome_observer=observer)
    deliver(policy, actual_sdk, "is_done")
    pixels[32, 33] = 5
    deliver(policy, actual_sdk, "choose_action")
    assert observer.issues[-1]["reason"] == "conflicting_duplicate"
