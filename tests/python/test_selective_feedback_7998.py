"""REQ-SELF-7998, REQ-VERIFY-7998: causal sparse feedback fixtures."""

import copy
import json

import numpy as np
import pytest

from carnot.verify import selective_feedback_7998 as m
from carnot.reporting.typed_validation_7997 import fixture_data
from carnot.reporting.current_work_receipt import canonical_hash


def fixture():
    data = fixture_data()
    for sources in data["public"].values():
        for r in sources:
            r["q"] = 0.5
    return data


def test_acquisition_frozen_expected_budget_and_draws():
    data = fixture()
    head = data["heads"]["spline"][0]
    sources = data["public"]["stream"]
    sources[0]["q"], sources[1]["q"] = 0.001, None
    rows = m.acquire(head, sources, 101)
    targeted = [r for r in rows if r["arm"] == "targeted_ipw"]
    uniform = [r for r in rows if r["arm"] == "uniform_ipw"]
    assert targeted[0]["pi"] == 0.125 and targeted[1]["pi"] == 0
    assert abs(sum(r["pi"] for r in uniform) - sum(r["pi"] for r in targeted)) < 1e-10
    assert rows == m.acquire(head, sources, 101)
    for i in range(256):
        same = [r for r in rows if r["slot"] == i]
        assert len({r["draw"] for r in same}) == 1
        assert same[0]["selected"] == same[1]["selected"]
    with pytest.raises(ValueError):
        m.acquire(head, [dict(sources[0], y=1)], 101)


def test_future_never_selected_duplicate_restart_and_dense(tmp_path):
    data = fixture()
    head, sources = data["heads"]["spline"][0], data["public"]["stream"]
    labels = {r["family_id"]: r["y"] for r in data["targets"]["stream"]}
    acquisition = [r for r in m.acquire(head, sources, 101) if r["arm"] == "targeted_ipw"]

    def run(directory, targets, **kw):
        return m.stream(head, sources, acquisition, targets.__getitem__, directory, **kw)

    original = run(tmp_path / "original", labels)
    changed = dict(labels)
    for r in acquisition:
        if not r["selected"] or r["slot"] >= 128:
            changed[r["family_id"]] = 1 - changed[r["family_id"]]
    mutation = run(tmp_path / "mutated", changed)
    assert original["issued_predictions"][:149] == mutation["issued_predictions"][:149]
    never = dict(labels)
    for r in acquisition:
        if not r["selected"]:
            never[r["family_id"]] = 1 - never[r["family_id"]]
    assert run(tmp_path / "never", never)["issued_predictions"] == original["issued_predictions"]
    run(tmp_path / "restart", labels, stop_at=128)
    # Simulate a crash after slot 128 was issued but before its feedback commit.
    (tmp_path / "restart" / "issued-0128.json").write_bytes(
        (tmp_path / "original" / "issued-0128.json").read_bytes()
    )
    restarted = run(tmp_path / "restart", labels)
    assert restarted == original
    assert run(tmp_path / "dense", labels, dense=True)["parity_parameters"] == pytest.approx(
        original["parity_parameters"], abs=1e-12
    )
    assert all(r["origin_slot"] == r["due_slot"] - 20 for r in original["reveal_rows"])
    assert all(r["coefficient_touches"] <= 37 and r["weight"] <= 8 for r in original["update_rows"])
    assert (
        original["pending_feedback"]
        and max(r["origin_slot"] for r in original["reveal_rows"]) <= 235
    )
    state = copy.deepcopy(original["final_state"])
    receipt = original["reveal_rows"][0]
    assert m.apply_feedback(state, receipt, sources[receipt["origin_slot"]]) is None
    assert state == original["final_state"]
    with pytest.raises(ValueError):
        m.apply_feedback(state, dict(receipt, receipt_id="new", due_slot=0), sources[0])
    checkpoint = json.loads((tmp_path / "original" / "committed-0255.json").read_text())
    assert checkpoint["checksum"] == canonical_hash(checkpoint["state"])


def test_issue_before_label_and_controls(tmp_path):
    data = fixture()
    head, sources = data["heads"]["spline"][0], data["public"]["stream"]
    acquisition = [r for r in m.acquire(head, sources, 101) if r["arm"] == "full_feedback"]

    def provider(identity):
        origin = int(identity.rsplit("-", 1)[1])
        assert (tmp_path / f"issued-{origin + 20:04d}.json").is_file()
        return 1

    result = m.stream(head, sources, acquisition, provider, tmp_path)
    assert result["issued_predictions"][21]["probability"] != 0.5
    controls = m.controls()
    assert controls["passed"] and controls["known_benefit"]["future_prediction_changed"]
    assert not controls["no_headroom"]["benefit"]
    empty = m.acquire(head, [dict(sources[0], q=None)], 101)
    assert all(r["pi"] == 0 for r in empty)


def test_unknown_labels_and_checkpoint_corruption(tmp_path):
    """REQ-SELF-7998: unknown targets never become zero and corrupt commits fail."""
    data = fixture()
    head, sources = data["heads"]["spline"][0], data["public"]["stream"]
    acquisition = [r for r in m.acquire(head, sources, 101) if r["arm"] == "full_feedback"]
    result = m.stream(head, sources, acquisition, lambda _: None, tmp_path)
    assert not result["update_rows"] and result["reveal_rows"]
    path = tmp_path / "committed-0255.json"
    doc = json.loads(path.read_text())
    doc["checksum"] = "corrupt"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="checkpoint_checksum"):
        m.stream(head, sources, acquisition, lambda _: None, tmp_path)
    doc["checksum"] = canonical_hash(doc["state"])
    path.write_text(json.dumps(doc))
    older = tmp_path / "committed-0128.json"
    old = json.loads(older.read_text())
    old["checksum"] = "corrupt"
    older.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="checkpoint_checksum"):
        m.stream(head, sources, acquisition, lambda _: None, tmp_path)
