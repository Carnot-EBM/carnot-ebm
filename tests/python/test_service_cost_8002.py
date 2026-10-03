"""REQ-REPORT-8002 and REQ-VERIFY-8002: frozen source-bound service accounting."""

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import service_cost_8002 as s
from carnot.reporting.current_work_receipt import atomic_json


def test_request_cases(tmp_path):
    """SCENARIO-VERIFY-8002-DRIFT: persist actual updates and count their work."""
    plan = s.fixture()
    path = tmp_path / "input.json"
    atomic_json(path, plan["requests"][0])
    rows = [
        s.request(path, plan["heads"]["spline"], "spline", case, tmp_path / "state")
        for case in s.CASES
    ]
    assert rows[0]["bytes_written"] == 0
    assert rows[1]["changed_coefficients"] == 0
    assert 0 < rows[2]["changed_coefficients"] <= 37
    assert rows[2]["logical_decay_coefficients"] == 109
    assert rows[2]["bytes_written"] == (tmp_path / "state").stat().st_size
    assert all(sum(r["exclusive_phase_spans"].values()) == r["wall_ns"] for r in rows)
    assert rows[0]["response"] == rows[2]["response"]


def test_measure_replay(tmp_path):
    """SCENARIO-VERIFY-8002-DRIFT: paired repeats never multiply sources."""
    plan = s.fixture()
    value = s.measure(plan, tmp_path)
    assert value["sample_size_budget"]["independent"] == 2
    assert len(value["rows"]) == 2 * 2 * 3 * 10
    assert value["complete_service_cost"][0]["p50_s"] is None
    s.replay(value)
    for field in ["response", "wall_ns", "changed_coefficients", "bytes_written"]:
        bad = copy.deepcopy(value)
        bad["rows"][0][field] = None
        with pytest.raises(ValueError, match="drift"):
            s.replay(bad)
    bad = copy.deepcopy(value)
    bad["cached_service_cost"] = []
    with pytest.raises(ValueError, match="reduction_drift"):
        s.replay(bad)


def test_branch_independence_and_custody(tmp_path):
    """SCENARIO-REPORT-8002-CUSTODY: alternatives never form a conductor AND."""
    plan = s.authenticate(s.ROOT, tmp_path / "snapshots")
    assert plan["branch_eligibility"]["sparse_fit"]["eligible"]
    assert plan["branch_eligibility"]["capture_scalar"]["eligible"]
    assert len(plan["requests"]) <= 64
    assert any(r.get("feedback_scope") == "exp7998_actual_receipt" for r in plan["requests"])
    for eid in [7995, 7996, 7998]:
        source = s.ROOT / "results" / s.PINS[eid][0]
        destination = tmp_path / "results" / source.name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (tmp_path / "results" / s.PINS[7996][0]).unlink()
    scalar = s.authenticate(tmp_path, tmp_path / "scalar")
    assert scalar["branch_eligibility"]["capture_scalar"]["eligible"]
    assert not scalar["branch_eligibility"]["sparse_fit"]["eligible"]
    missing = s.authenticate(tmp_path / "absent", tmp_path / "missing")
    assert not missing["requests"]
    ref = plan["source_artifact_hashes"][0]
    with open(ref["path"], "a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash"):
        s.checked(ref)


def test_contract(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8002-CUSTODY: absent fields differ from observed zero."""
    source = tmp_path / "results" / s.PINS[7996][0]
    atomic_json(source, dict(experiment_id=7996))
    monkeypatch.setitem(
        s.PINS, 7996, (source.name, "sparse_fit_ready_score", s.sha256_file(source))
    )
    with pytest.raises(ValueError, match="upstream_contract"):
        s.authenticate(tmp_path, tmp_path / "snapshots")


def test_sparse_only_and_missing_assets(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8002-CUSTODY: sparse fitting survives missing capture."""
    source = s.ROOT / "results" / s.PINS[7996][0]
    destination = tmp_path / "results" / source.name
    destination.parent.mkdir()
    destination.write_bytes(source.read_bytes())
    plan = s.authenticate(tmp_path, tmp_path / "snapshot")
    assert set(plan["heads"]) == {"spline"}
    assert plan["requests"]
    assert all(r["acquisition"] is None for r in plan["requests"])
    value = json.loads(destination.read_text())
    value["checkpoints"]["heads"]["path"] = str(tmp_path / "missing-heads")
    atomic_json(destination, value)
    monkeypatch.setitem(
        s.PINS, 7996, (destination.name, "sparse_fit_ready_score", s.sha256_file(destination))
    )
    blocked = s.authenticate(tmp_path, tmp_path / "blocked")
    assert not blocked["requests"]
    assert blocked["gate_check_summary"][-1]["field"] == "sha256"
    assert blocked["gate_check_summary"][-1]["observed"] is None


@pytest.mark.parametrize(
    "mutation,error", [("identity", "capture_identity"), ("source", "acquisition_source_hash")]
)
def test_capture_mutations(tmp_path, monkeypatch, mutation, error):
    """SCENARIO-VERIFY-8002-DRIFT: a valid pin does not excuse invalid joins."""
    source = s.ROOT / "results" / s.PINS[7995][0]
    value = json.loads(source.read_text())
    if mutation == "identity":
        value["model_identity_receipt"]["authenticated"] = False
    else:
        for row in value["rows"]:
            row["public_hash"] = "forged"
    destination = tmp_path / "results" / source.name
    atomic_json(destination, value)
    monkeypatch.setitem(
        s.PINS, 7995, (destination.name, "capture_ready_score", s.sha256_file(destination))
    )
    with pytest.raises(ValueError, match=error):
        s.authenticate(tmp_path, tmp_path / "snapshot")


def test_commit_and_timing_mutations(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8002-DRIFT: sealed commit and paired spans reject edits."""
    original = s.snapshot

    def corrupt(ref, directory):
        sealed = original(ref, directory)
        if "committed-" in ref["path"]:
            value = json.loads(s.checked(sealed).read_text())
            value["checksum"] = "forged"
            atomic_json(s.checked(sealed), value)
            sealed = s.reference(Path(sealed["path"]))
        return sealed

    monkeypatch.setattr(s, "snapshot", corrupt)
    with pytest.raises(ValueError, match="historical_commit_hash"):
        s.authenticate(s.ROOT, tmp_path / "corrupted")
    value = s.measure(s.fixture(), tmp_path / "measure")
    value["rows"][0]["complete_total_s"] = 1.0
    with pytest.raises(ValueError, match="join_drift"):
        s.replay(value)
    value["rows"][0]["complete_total_s"] = None
    value["rows"][0]["repetition"] = 99
    with pytest.raises(ValueError, match="paired_rows_drift"):
        s.replay(value)


def test_single_source_complete_and_provenance(tmp_path):
    """SCENARIO-VERIFY-8002-DRIFT: complete totals need matched acquisition."""
    plan = s.fixture()
    plan["requests"] = plan["requests"][:1]
    plan["requests"][0]["acquisition"] = dict(
        duration_s=1.25,
        producer_id=7995,
        producer_date="historical",
        public_hash=s.canonical_hash(plan["requests"][0]["public"]),
    )
    value = s.measure(plan, tmp_path)
    assert value["complete_service_cost"][0]["p50_s"] > 1.25
    assert value["paired_latency_intervals"][0]["paired_95_interval_s"] is None
    ref = tmp_path / "frozen-code"
    ref.write_text("frozen")
    value["measurement_code_snapshot"] = [s.reference(ref)]
    s.replay(value)
    ref.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        s.replay(value)
