"""REQ-VERIFY-8702 / SCENARIO-VERIFY-8702-ENVELOPE: fail closed before work."""

from copy import deepcopy
import json
import time

import pytest

from carnot.inference import resident_batch as b
from carnot.inference import resident_protocol as p


@pytest.fixture
def head():
    return dict(
        parameters=[0.01] * 109,
        decay_scale=1.0,
        temperature=1.0,
        scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
    )


@pytest.fixture
def row():
    return dict(
        unit_id="item",
        q=0.4,
        public=dict(
            family_id="fixture",
            source_bytes=b"There are 12 apples.".hex(),
            answer_bytes=b"There are 12 apples.".hex(),
        ),
    )


def test_opt_in_and_complete_work(head, row, monkeypatch):
    """REQ-VERIFY-8702: all sixteen sizes redo features; the old route agrees."""
    calls = []
    original = b.numeric.extract
    monkeypatch.setattr(b.numeric, "extract", lambda x: (calls.append(x), original(x))[1])
    for size in range(1, 17):
        rows = [dict(row, unit_id=str(i)) for i in range(size)]
        value = dict(p.envelope(str(size), "hash", {}), operation="batch", rows=rows)
        response = p.decide(value, dict(head=head, max_batch_size=16), "hash", None, set())
        assert response["status"] == "ok"
        assert len(response["responses"]) == size
        for item in response["responses"]:
            expected, _ = b.numeric.evaluate(head, dict(row, unit_id=item["unit_id"]), None)
            assert item == expected
    assert len(calls) >= 136
    assert p.decide(value, dict(head=head), "hash", None, set())["status"] == "error"


@pytest.mark.parametrize("rows", [[], None, {}, [None], [{}]])
def test_bad_rows(head, rows):
    """SCENARIO-VERIFY-8702-ENVELOPE: malformed rows never expose partial results."""
    with pytest.raises((ValueError, TypeError)):
        b.evaluate(head, rows, None, 16)


def test_id_probability_and_private_rejections(head, row):
    """REQ-VERIFY-8702: a bad later item rejects the complete envelope."""
    cases = [
        [row] * 17,
        [row, row],
        [dict(row, unit_id="")],
        [dict(row, unit_id=3)],
        [dict(row, extra="private")],
    ]
    cases += [[dict(row, q=q)] for q in [None, True, float("nan"), -0.1, 1.1]]
    cases += [[dict(row, public=dict(row["public"], label=1))]]
    for rows in cases:
        value = dict(p.envelope("id", "hash", {}), operation="batch", rows=rows)
        response = p.decide(value, dict(head=head, max_batch_size=16), "hash", None, set())
        assert response["status"] == "error"
        assert not {"responses", "response", "feature_hashes"} & response.keys()
    for limit in [None, True, 0, 17]:
        with pytest.raises(ValueError):
            b.evaluate(head, [row], None, limit)


def test_expiry_and_reset(head, row):
    """REQ-VERIFY-8702: duplicate IDs expire per session, never across restarts."""
    value = dict(p.envelope("id", "hash", {}), operation="batch", rows=[row])
    config = dict(head=head, max_batch_size=16)
    seen = set()
    assert p.decide(value, config, "hash", None, seen)["status"] == "ok"
    assert p.decide(value, config, "hash", None, seen)["error"] == "duplicate_request_id"
    assert p.decide(deepcopy(value), config, "hash", None, set())["status"] == "ok"
    value["deadline_ns"] = time.monotonic_ns() - 1
    assert p.decide(value, config, "hash", None, set())["error"] == "request_deadline"


def test_item_budget_and_no_work(head, row, monkeypatch, tmp_path):
    """REQ-VERIFY-8702: reject oversized remaining work before extracting any item."""
    value = dict(p.envelope("id", "hash", {}), operation="batch", rows=[row])
    config = dict(head=head, max_batch_size=16)
    monkeypatch.setattr(b, "evaluate", lambda *a: pytest.fail("budget performed work"))
    assert p.decide(value, config, "hash", None, set(), remaining=0)["error"] == "item_budget"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(config, item_budget=0)))
    sent = []
    monkeypatch.setattr(p, "write_frame", lambda fd, value, *a: sent.append(value))
    assert p.serve(path, p.sha256_file(path), "python") == 1


def test_post_batch_deadline(head, row, monkeypatch):
    """REQ-VERIFY-8702: expiry after numerical work rejects the full payload."""
    value = dict(p.envelope("id", "hash", {}), operation="batch", rows=[row])
    deadline = value["deadline_ns"]
    times = iter([deadline - 1, deadline - 1, deadline])
    monkeypatch.setattr(p.time, "monotonic_ns", lambda: next(times))
    monkeypatch.setattr(b, "evaluate", lambda *a: ([], [], []))
    assert (
        p.decide(value, dict(head=head, max_batch_size=16), "hash", None, set())["error"]
        == "evaluation_deadline"
    )
