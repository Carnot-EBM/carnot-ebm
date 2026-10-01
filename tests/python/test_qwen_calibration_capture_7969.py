"""REQ-VERIFY-7969: public protocol, bounded capture and cold reduction."""

from copy import deepcopy
import json

import pytest

from carnot.verify import qwen_calibration_capture_7969 as capture
from carnot.verify import qwen_response_risk_7958 as risk
from test_qwen_response_risk_7958 import Runtime, reply


def views():
    return {
        role: {
            "role": role,
            "request_rows": [
                dict(
                    family_id=f"{role}-{i}",
                    source_bytes=f"Source {role} {i}. Café.".encode().hex(),
                    answer_bytes=b"Complete answer. Last span.".hex(),
                )
                for i in range(n)
            ],
            "boundaries": [
                dict(family_id=f"{role}-{i}", public_eligible=True, exclusion_reason=None)
                for i in range(n)
            ],
        }
        for role, n in capture.ROLES.items()
    }


class CountingRuntime(Runtime):
    def count(self, text):
        return 30


def test_frozen_full_source_protocol_and_role_denominator():
    """SCENARIO-VERIFY-7969-CAPTURE: exact historical prompt, no label inputs."""
    data = views()
    frozen = capture.freeze(data)
    assert len(frozen) == 384
    assert [r["public_hash"] for r in frozen] == sorted(r["public_hash"] for r in frozen)
    for row in frozen:
        original = next(
            r for r in data[row["role"]]["request_rows"] if r["family_id"] == row["family_id"]
        )
        assert row["request"] == risk.freeze([original], lambda _: 0)[0]["requests"]["full_source"]
    bad = deepcopy(data)
    bad["fit"]["request_rows"][0]["y"] = 1
    with pytest.raises(ValueError, match="public_fields"):
        capture.freeze(bad)
    with pytest.raises(ValueError, match="role_roster"):
        capture.freeze({"evaluation": data["fit"]})


def test_capture_checkpoint_resume_and_readiness(tmp_path):
    """SCENARIO-VERIFY-7969-REPLAY: no duplicate calls after sealed checkpoint."""
    frozen = capture.freeze(views())
    runtime = CountingRuntime()
    rows = capture.capture(frozen, runtime, tmp_path, "identity")
    reduced = capture.reduce(rows)
    assert len(runtime.calls) == 384
    assert reduced["sample_size_budget"]["completed"] == 384
    assert reduced["qwen_capture_ready_score"] == 1
    assert len(list(tmp_path.glob("checkpoint-*.json"))) == 48
    again = CountingRuntime()
    assert capture.capture(frozen, again, tmp_path, "identity") == rows
    assert not again.calls
    with pytest.raises(ValueError, match="checkpoint_identity"):
        capture.capture(frozen, again, tmp_path, "changed")
    changed = deepcopy(rows)
    changed[0]["parsed"]["probability"] = 0.7
    with pytest.raises(ValueError, match="parse_drift"):
        capture.reduce(changed)


def test_excluded_censored_failed_and_no_repairs(tmp_path):
    """SCENARIO-VERIFY-7969-CAPTURE: every unsuccessful slot stays counted."""
    data = views()
    data["fit"]["boundaries"][0].update(public_eligible=False, exclusion_reason="public_empty")
    frozen = capture.freeze(data)
    failed = CountingRuntime(RuntimeError("no retry"))
    rows = capture.capture(frozen[:3], failed, tmp_path / "failed", "identity")
    assert len(failed.calls) == 3
    assert capture.reduce(rows)["sample_size_budget"]["failed"] == 3
    long = CountingRuntime(reply(finish="length"))
    long.count = lambda _: 6001
    rows = capture.capture(frozen, long, tmp_path / "excluded", "identity")
    assert not long.calls
    assert capture.reduce(rows)["sample_size_budget"]["excluded"] == 384
    rows = capture.capture(
        frozen, CountingRuntime(), tmp_path / "censored", "identity", deadline_s=0
    )
    assert capture.reduce(rows)["sample_size_budget"]["censored"] == 383
    assert all("y" not in r for r in rows)


def test_resume_interrupted_uncertain_and_censored(tmp_path):
    """SCENARIO-VERIFY-7969-CAPTURE: resume never repeats a started request."""
    from carnot.reporting.current_work_receipt import atomic_json

    frozen = capture.freeze(views())
    capture.capture(frozen[:1], CountingRuntime(), tmp_path / "running", "identity")
    path = tmp_path / "running" / "slot-000.json"
    row = json.loads(path.read_text())
    row.update(status="running", raw_response={})
    atomic_json(path, row)
    runtime = CountingRuntime()
    rows = capture.capture(frozen[:1], runtime, path.parent, "identity")
    assert not runtime.calls and rows[0]["status"] == "failed"
    capture.capture(frozen[:1], runtime, tmp_path / "budget", "identity", deadline_s=0)
    assert capture.capture(frozen[:1], runtime, tmp_path / "budget", "identity")[0]["parsed"][
        "completed"
    ]
    assert len(runtime.calls) == 1


def test_public_custody_and_tokenizer_failure(tmp_path):
    """REQ-VERIFY-7969: changed role, count, boundaries and source overlap fail."""
    for mutation, reason in [
        (lambda d: d["fit"].update(role="evaluation"), "public_view"),
        (lambda d: d["fit"]["request_rows"].pop(), "role_count"),
        (lambda d: d["fit"]["boundaries"].pop(), "boundary_roster"),
        (
            lambda d: d["fit"]["request_rows"][0].update(
                source_bytes=d["tune"]["request_rows"][0]["source_bytes"]
            ),
            "cross_role_overlap",
        ),
    ]:
        data = views()
        mutation(data)
        with pytest.raises(ValueError, match=reason):
            capture.freeze(data)
    frozen = capture.freeze(views())
    runtime = CountingRuntime()
    runtime.count = lambda _: (_ for _ in ()).throw(RuntimeError("tokenizer"))
    rows = capture.capture(frozen[:1], runtime, tmp_path, "identity")
    assert not runtime.calls and rows[0]["status"] == "failed"
    with pytest.raises(ValueError, match="slot_roster"):
        capture.reduce(rows * 2)
    assert not capture.capture(
        frozen[:1], CountingRuntime(), tmp_path / "no-tokens", "identity", token_budget=0
    )[0]["started"]
