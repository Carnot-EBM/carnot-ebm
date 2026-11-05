"""REQ-REPORT-8804 / REQ-VERIFY-8804: later predictions and complete delayed costs."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.learning import causal_template_streams_8804 as s
from carnot.memory import causal_memory_8802 as m
from carnot.reporting.current_work_receipt import canonical_hash


def small_manifest():
    """Keep the causal delay while reducing private test work to one reset stream."""
    manifest = s.manifest()
    manifest["streams"] = manifest["streams"][:1]
    manifest["seeds"] = [11]
    return manifest


def test_schedule_and_disjoint_inputs():
    """SCENARIO-REPORT-8804-STREAMS: fixed census and isolated public evidence."""
    frozen = s.manifest()
    assert frozen["seeds"] == [11, 23, 47]
    assert len(frozen["streams"]) == 30
    online, retention = s.census(frozen, "external_gate")
    assert len(online) == 14400 and len(retention) == 16200
    assert all(r["metric"] is None and r["status"] == "censored" for r in online + retention)
    assert len({r["cluster_id"] for r in online}) == 30
    recipes = [w for stream in frozen["streams"] for w in stream["worlds"] + stream["retention"]]
    assert len({w["disconnected_path_edges"] for w in recipes}) == len(recipes)
    payload = s.payload(recipes[0], "insufficient")
    assert set(payload) == {"facts", "query"}
    assert len(payload["facts"]) == 5 + recipes[0]["disconnected_path_edges"]
    assert m.decide(payload)["decision"] == "insufficient"
    assert s.payload(recipes[0], "support") != payload


def test_real_stream_causal_rows_and_tail(tmp_path: Path):
    """SCENARIO-VERIFY-8804-CAUSAL / SCENARIO-REPORT-8804-COST: real durable work."""
    frozen = small_manifest()
    head = dict(seed=11, parameters=m.initial_parameters(11), threshold=0.5)
    work = s.execute(frozen, [head], tmp_path)
    rows, retained = work["per_query_rows"], work["retention_rows"]
    assert len(rows) == 160 and len(retained) == 180
    assert work["numerical_training_calls"] == 160
    assert len(work["release_order_rows"]) == 160
    assert len(work["complete_learning_cost_rows"]) == 160
    assert len(work["tail_drain_rows"]) == 5
    assert work["tail_drain_cost_s"] > 0
    assert all(r["initial_state_equal"] for r in work["reset_rows"])
    assert all(r["paid_checks"] == 5 for r in rows + retained)
    assert all(r["learning_state_unchanged"] for r in retained)
    assert work["template_addition_rows"]
    assert any(r["changed_later_predicates"] for r in rows)
    for row in rows:
        assert sum(row["prediction"]) == pytest.approx(1, abs=1e-6)
        assert row["sealed_ns"] <= row["truth_recorded_ns"]
        assert row["metric"] == s.brier(row["prediction"], row["truth"])
    for row in work["release_order_rows"]:
        assert row["feedback_ordinal"] >= row["prediction_ordinal"] + 8
        assert row["feedback_label"] in row["released_prefix"]
    for row in work["complete_learning_cost_rows"]:
        assert row["ack_ns"] >= row["predict_started_ns"]
        assert row["duration_s"] == (row["ack_ns"] - row["predict_started_ns"]) / 1e9
        assert row["restart_equal"]
    assert s.check_work(work, frozen)["passed"], s.check_work(work, frozen)
    changed = deepcopy(work)
    changed["per_query_rows"][0]["metric"] += 0.2
    assert not s.check_work(changed, frozen)["passed"]
    changed = deepcopy(work)
    changed["release_order_rows"][0]["feedback_ordinal"] = 0
    assert not s.check_work(changed, frozen)["passed"]
    changed = deepcopy(work)
    changed["retention_rows"][0]["learning_state_unchanged"] = False
    assert not s.check_work(changed, frozen)["passed"]
    seal = Path(work["prediction_seal_refs"][0]["path"])
    data = json.loads(seal.read_text())
    assert data["pre_state_hash"] == canonical_hash(data["pre_state"])
    assert "truth" not in data and "label" not in data
    before = seal.read_bytes()
    data["sealed"]["prediction"][0] += 0.1
    seal.write_text(json.dumps(data))
    assert not s.check_work(work, frozen)["passed"]
    seal.write_bytes(before)


def test_release_and_feature_boundaries(tmp_path: Path):
    """SCENARIO-VERIFY-8804-CAUSAL: reuse the unchanged transaction guards."""
    store = m.Store(tmp_path / "state.json", m.initial_parameters(11))
    query = m.world("causal", "insufficient")
    store.predict("e0", query, 0.5)
    with pytest.raises(RuntimeError, match="active"):
        store.release("e0", query, 8, "combined")
    store.close()
    with pytest.raises(ValueError, match="early"):
        store.release("e0", query, 7, "combined")
    with pytest.raises(ValueError, match="input_fields"):
        m.observe(dict(query, label="insufficient"), [])
    store.release("e0", query, 8, "combined")
    with pytest.raises(ValueError, match="duplicate"):
        store.release("e0", query, 9, "combined")


def test_compiled_host_path_matches_frozen_model(tmp_path: Path):
    """REQ-VERIFY-8804: reduce Python dispatch without changing the fixed Gibbs math."""
    params = m.initial_parameters(11)
    query = m.world("parity", "insufficient")
    counts = m.observe(query, [])["counts"]
    assert s.probabilities(params, counts) == pytest.approx(
        m.probabilities(params, counts), abs=1e-6
    )
    old = m.Store(tmp_path / "old.json", params)
    fast = s.StreamStore(tmp_path / "fast.json", params)
    a, b = old.predict("e0", query, 0.5), fast.predict("e0", query, 0.5)
    assert a["prediction"] == pytest.approx(b["prediction"], abs=1e-6)
    with pytest.raises(RuntimeError, match="active"):
        fast.predict("e1", query, 0.5)
    with pytest.raises(RuntimeError, match="active"):
        fast.release("e0", query, 8, "combined")
    old.close()
    fast.close()
    with pytest.raises(ValueError, match="duplicate"):
        fast.predict("e0", query, 0.5)
    with pytest.raises(ValueError, match="early"):
        fast.release("e0", query, 7, "combined")
    with pytest.raises(ValueError, match="evidence"):
        fast.release("e0", m.world("changed", "support"), 8, "combined")
    old.release("e0", query, 8, "combined")
    fast.release("e0", query, 8, "combined")
    assert old.state["parameters"] == fast.state["parameters"]
    assert old.state["templates"] == fast.state["templates"]
    with pytest.raises(ValueError, match="duplicate"):
        fast.release("e0", query, 9, "combined")
    fast.state["pending"] = {f"e{i}": {} for i in range(32)}
    with pytest.raises(ValueError, match="capacity"):
        fast.predict("e32", query, 0.5)


def test_development_seed_and_failure_census(tmp_path: Path, monkeypatch):
    """REQ-REPORT-8804: seed47 is fitted on development; failed work stays scheduled."""
    selected, calls = s.heads({"calibrated_head_parameters": []}, [47])
    assert calls == 40 and selected[0]["seed"] == 47
    frozen = small_manifest()
    monkeypatch.setattr(
        s, "run_arm", lambda *a: (_ for _ in ()).throw(RuntimeError("fixture_failure"))
    )
    work = s.execute(frozen, [dict(seed=11, parameters={}, threshold=0.5)], tmp_path)
    assert len(work["failures"]) == 5
    assert all(r["status"] == "failed" for r in work["per_query_rows"] + work["retention_rows"])
    assert not s.check_work(work, frozen)["passed"]


def test_resets_across_three_streams_and_tail_replay(tmp_path: Path):
    """REQ-VERIFY-8804: replay more than a thousand genuine rows and all tail states."""
    frozen = s.manifest()
    frozen["streams"] = frozen["streams"][:3]
    frozen["seeds"] = [11]
    head = dict(seed=11, parameters=m.initial_parameters(11), threshold=0.5)
    work = s.execute(frozen, [head], tmp_path)
    assert len(work["per_query_rows"]) + len(work["retention_rows"]) == 1020
    assert [r["cluster_id"] for r in work["per_query_rows"]][::160] == [
        "stream-00",
        "stream-01",
        "stream-02",
    ]
    assert s.check_work(work, frozen)["passed"], s.check_work(work, frozen)
    assert len({r["state_hash"] for r in work["reset_rows"]}) == 1
    changed = deepcopy(work)
    changed["release_order_rows"][-1]["feedback_label"] = "invalid"
    assert not s.check_work(changed, frozen)["passed"]
