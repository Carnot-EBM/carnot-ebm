"""REQ-REPORT-8663, REQ-VERIFY-8663: bind real public frames to fresh contexts."""

from types import SimpleNamespace

import numpy as np
import pytest
from arcengine import FrameDataRaw, GameState

from carnot.agentic import arc_reset_provenance as r
from carnot.reporting.current_work_receipt import canonical_hash


def real_frame():
    """Use the public SDK's actual list-of-ndarrays representation."""
    value = FrameDataRaw(
        game_id="private-fixture",
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        available_actions=[1, 2],
    )
    value.frame = [np.array([[1, 2], [3, 4]], dtype=np.int8)]
    return value


def test_recursive_real_sdk_payload_and_unchanged_simple_hash():
    """SCENARIO-REPORT-8663-PAYLOAD: conversion preserves every nested value."""
    frame = real_frame()
    expected = dict(level=0, state=str(frame.state), full_reset=False, pixels=[[[1, 2], [3, 4]]])
    assert r.public_hash(frame) == canonical_hash(expected)
    nested = dict(level=np.int64(0), values=[np.array([1, 2]), dict(state=GameState.WIN)])
    assert r.public_hash(nested) == canonical_hash(
        dict(level=0, values=[[1, 2], dict(state="WIN")])
    )
    assert r.public_hash(dict(level=0)) == canonical_hash(dict(level=0))
    assert np.array_equal(frame.frame[0], np.array([[1, 2], [3, 4]]))


def test_same_pixels_repeated_resets_and_callback_identity():
    """SCENARIO-REPORT-8663-PAYLOAD: equal frames do not collapse action identities."""
    recorder = r.ResetProvenance()
    policy = SimpleNamespace(_arc_reset_provenance=recorder)
    frame = real_frame()
    for index in (1, 2):
        r.prepare(policy, frame)
        r.request(policy, "executed.branch")
        recorder.selected(("RESET", None))
        assert r.sdk_call(policy, ("RESET", None), lambda: frame) is frame
        row = recorder.requests[-1]
        assert row["action_index"] == index
        assert row["reset_request_id"] == f"{index}:{index}"
        assert row["pre_observation_sha256"] == row["post_observation_sha256"]
    assert not getattr(policy, "_arc_reset_provenance_errors", [])
    assert [x["method"] for x in recorder.receipt()["callback_sequence_rows"]] == [
        "selected",
        "selected",
        "returned",
        "selected",
        "selected",
        "returned",
    ]
    recorder.returned(("RESET", None), frame)
    assert recorder.receipt()["callback_sequence_rows"][-1]["duplicate_invocation"] is True


def test_serialization_failure_cannot_borrow_previous_context():
    """SCENARIO-REPORT-8663-PAYLOAD: failed begin installs a fresh invalid context."""
    recorder = r.ResetProvenance()
    recorder.begin(dict(level=2))
    recorder.request("previous")
    policy = SimpleNamespace(_arc_reset_provenance=recorder)
    r.prepare(policy, dict(level=0, unsupported=object()))
    r.request(policy, "current")
    row = recorder.requests[-1]
    assert row["action_index"] == 2
    assert row["pre_observation_sha256"] is None
    assert row["origin"] == "unknown"
    assert row["context_error"] == "TypeError"
    assert policy._arc_reset_provenance_errors == ["TypeError"]
    assert policy._arc_reset_provenance_diagnostics[-1]["method"] == "begin"
    assert len(policy._arc_reset_provenance_diagnostics[-1]["message"]) <= 240
    recorder.begin(None)
    recorder.selected(("RESET", None))
    recorder.returned(("RESET", None), None)
    assert recorder.requests[-1]["sdk_status"]["observation_missing"]


def test_request_conflicts_bounds_and_sdk_exception():
    """SCENARIO-REPORT-8663-PAYLOAD: real call sites remain distinct and bounded."""
    recorder = r.ResetProvenance(limit=4)
    recorder.begin(real_frame(), 8, 2.0)
    recorder.request("first")
    recorder.request("second")
    assert recorder.receipt()["requester_ownership_rows"][0]["status"] == "conflicting"
    recorder.selected((1, None))
    recorder.returned((1, None), dict(state="WIN"))
    recorder.returned((1, None), dict(state="GAME_OVER"))
    policy = SimpleNamespace(_arc_reset_provenance=recorder)

    def failure():
        raise TimeoutError("original")

    with pytest.raises(TimeoutError, match="original"):
        r.sdk_call(policy, ("RESET", None), failure)
    for _ in range(8):
        recorder.selected((1, None))
    assert len(recorder.receipt()["callback_sequence_rows"]) == 4
    assert r.sdk_call(SimpleNamespace(), (1, None), lambda: 7) == 7
    for _ in range(8):
        recorder.request("bounded")
    assert recorder.receipt()["issues"] == [dict(reason="record_limit")]
    unknown = r.ResetProvenance()
    unknown.begin(None)
    unknown.selected(("RESET", None))
    assert unknown.receipt()["requester_ownership_rows"][0]["status"] == "unknown"
