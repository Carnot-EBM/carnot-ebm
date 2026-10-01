"""REQ-VERIFY-7981: separate branches, public bytes and resumable calls."""

from copy import deepcopy
import json

import pytest

from carnot.verify import qwen_stream_capture_7981 as capture
from test_qwen_calibration_capture_7969 import CountingRuntime


def views(fresh=False):
    return {
        role: dict(
            role=role,
            request_rows=[
                dict(
                    family_id=f"{role}-{i}",
                    source_bytes=f"Complete source {role} {i}. Café.".encode().hex(),
                    answer_bytes=b"Complete original answer.".hex(),
                )
                for i in range(n)
            ],
            boundaries=[
                dict(family_id=f"{role}-{i}", public_eligible=True, exclusion_reason=None)
                for i in range(n)
            ],
            features=[],
        )
        for role, n in {**capture.ROLES, **({"reserved_development": 96} if fresh else {})}.items()
    }


def test_protocol_public_fields_and_roster():
    """SCENARIO-VERIFY-7981-BRANCHES: no old roles or truncated sources."""
    frozen = capture.freeze(views(True))
    assert len(frozen) == 320
    assert {r["role"] for r in frozen} == {*capture.ROLES, "reserved_development"}
    assert capture.config()["call_limit"] == 320
    assert capture.config()["duration_floor_s"] == 10
    assert all(r["request"]["max_tokens"] == 96 for r in frozen)
    assert all(r["request"]["seed"] == 69058 for r in frozen)
    bad = views()
    bad["online_update"]["request_rows"][0]["y"] = 1
    with pytest.raises(ValueError, match="public_fields"):
        capture.freeze(bad)
    with pytest.raises(ValueError, match="role_roster"):
        capture.freeze({"fit": views()["online_update"]})
    with pytest.raises(ValueError, match="public_view"):
        capture.freeze({**views(), "retention": {**views()["retention"], "y": 1}})


def test_stream_and_fresh_are_independent(tmp_path):
    """SCENARIO-VERIFY-7981-BRANCHES: either branch can qualify alone."""
    rows = capture.capture(capture.freeze(views(True)), CountingRuntime(), tmp_path, "identity")
    result = capture.reduce(rows)
    assert result["stream_capture_ready_score"] == result["fresh_capture_ready_score"] == 1
    assert result["sample_size_budget"]["independent"] == 320
    stream = [r for r in rows if r["role"] != "reserved_development"]
    assert capture.reduce(stream)["stream_capture_ready_score"] == 1
    assert capture.reduce(stream)["fresh_capture_ready_score"] == 0
    fresh = [r for r in rows if r["role"] == "reserved_development"]
    assert capture.reduce(fresh)["fresh_capture_ready_score"] == 1
    assert capture.reduce(fresh)["stream_capture_ready_score"] == 0
    damaged = deepcopy(rows)
    for row in damaged:
        if row["role"] == "retention":
            row.update(status="failed", raw_response={})
            row["parsed"] = capture.risk.transport.parse_response({}, row["visible_ids"])
    assert capture.reduce(damaged)["stream_capture_ready_score"] == 0
    assert capture.reduce(damaged)["fresh_capture_ready_score"] == 1
    with pytest.raises(ValueError, match="slot_roster"):
        capture.reduce(rows * 2)
    damaged = deepcopy(rows)
    damaged[0]["parsed"]["probability"] = 0.7
    with pytest.raises(ValueError, match="parse_drift"):
        capture.reduce(damaged)


def test_resume_timestamps_and_identity(tmp_path):
    """SCENARIO-VERIFY-7981-RESUME: started judgments never repeat."""
    frozen = capture.freeze(views())[:2]
    rows = capture.capture(frozen, CountingRuntime(), tmp_path, "identity")
    assert all(r["invocation_started_at"] and r["invocation_finished_at"] for r in rows)
    runtime = CountingRuntime()
    assert capture.capture(frozen, runtime, tmp_path, "identity") == rows
    assert not runtime.calls
    with pytest.raises(ValueError, match="checkpoint_identity"):
        capture.capture(frozen, runtime, tmp_path, "changed")
    path = tmp_path / "slot-000.json"
    interrupted = json.loads(path.read_text())
    interrupted.update(status="running", raw_response={})
    path.write_text(json.dumps(interrupted))
    again = capture.capture(frozen, runtime, tmp_path, "identity")
    assert again[0]["status"] == "failed" and not runtime.calls
    assert again[0]["invocation_started_at"] == rows[0]["invocation_started_at"]


def test_failures_censoring_and_token_limit(tmp_path):
    """SCENARIO-VERIFY-7981-RESUME: all unsuccessful rows remain visible."""
    frozen = capture.freeze(views())[:1]
    for kind in ("ineligible", "overlong", "tokenizer", "generation", "deadline"):
        runtime = CountingRuntime()
        slots = deepcopy(frozen)
        kwargs = {}
        if kind == "ineligible":
            slots[0]["public_eligible"] = False
        elif kind == "overlong":
            runtime.count = lambda _: 6001
        elif kind == "tokenizer":
            runtime.count = lambda _: (_ for _ in ()).throw(ValueError("count"))
        elif kind == "generation":
            runtime.generate = lambda _: (_ for _ in ()).throw(RuntimeError("transport"))
        else:
            kwargs["deadline_s"] = 0
        rows = capture.capture(slots, runtime, tmp_path / kind, "identity", **kwargs)
        assert len(rows) == 1 and not rows[0]["parsed"]["completed"]
        if kind == "deadline":
            assert capture.capture(slots, runtime, tmp_path / kind, "identity")[0]["parsed"][
                "completed"
            ]


def test_reserved_branch_does_not_require_stream_inputs():
    """SCENARIO-VERIFY-7981-BRANCHES: a reserved-only branch retains fixed96."""
    assert len(capture.freeze({capture.FRESH: views(True)[capture.FRESH]})) == 96
