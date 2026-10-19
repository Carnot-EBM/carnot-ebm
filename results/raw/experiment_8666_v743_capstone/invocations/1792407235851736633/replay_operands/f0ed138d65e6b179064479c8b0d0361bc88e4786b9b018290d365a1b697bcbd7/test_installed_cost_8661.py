"""REQ-REPORT-8661, REQ-VERIFY-8661: equal-work costs retain original authority."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import installed_cost_8661 as c
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_gate_missing_field_zero_and_numeric_one(tmp_path):
    """SCENARIO-REPORT-8661-MATRIX: absence and measured zero remain distinct."""
    path = tmp_path / "upstream.json"
    assert c.gate(path, "exp0", "ready")[0][0]["observed_state"] == "missing_file"
    for value, state in [
        ({}, "missing_field"),
        ({"ready": 0}, "observed"),
        ({"ready": True}, "observed"),
    ]:
        atomic_json(path, value)
        rows, ref = c.gate(path, "exp0", "ready")
        assert rows[0]["observed_state"] == state
        assert rows[0]["path"] == str(path)
        assert rows[0]["sha256"] == ref["sha256"]
    atomic_json(path, {"ready": 1})
    assert c.gate(path, "exp0", "ready")[0] == []


def test_fixed_roster_and_incomplete_never_zero_cost():
    """SCENARIO-REPORT-8661-MATRIX: all120 cells survive a missing measurement."""
    slots = c.roster()
    assert len(slots) == len({r["unit_id"] for r in slots}) == 120
    assert {r["batch"] for r in slots} == {1}
    assert [r["arm"] for r in slots[:8]] == [
        "private_python",
        "private_native",
        "installed_python",
        "installed_native",
        "private_native",
        "private_python",
        "installed_native",
        "installed_python",
    ]
    result = c.reduce(
        {
            "rows": [dict(s, status="censored", missing_reason="deadline") for s in slots],
            "bundle": {},
        }
    )
    assert result["completed_count"] == 0 and result["censored_count"] == 120
    assert result["paired_throughput_intervals"] == []
    assert result["installed_cost_ready_score"] == 0
    assert result["packaging_noninferiority_score"] == result["nfr01_score"] == 0
    with pytest.raises(ValueError, match="cell_roster"):
        c.reduce({"rows": [], "bundle": {}})


def test_paired_index_intervals_and_distinct_thresholds():
    """REQ-REPORT-8661: only installed native/Python can satisfy NFR-01."""
    assert c.interval([2.0] * 10, [1.0] * 10) == [2.0, 2.0]
    assert c.scores(True, [1.0, 1.1], 1.05, [9.99, 11.0]) == (1, 1, 0)
    assert c.scores(True, [0.94, 1.1], 1.0, [10.0, 11.0]) == (1, 0, 1)
    assert c.scores(False, [2.0, 2.0], 1.0, [20.0, 20.0]) == (0, 0, 0)


def test_replay_rejects_rehashed_reduction_and_missing_operand(tmp_path):
    """SCENARIO-VERIFY-8661-REPLAY: a summary cannot authorize itself."""
    work = c.empty_work(tmp_path, "20261019")
    work["gate_check_summary"] = [{"observed_state": "missing_file"}]
    authority_path = tmp_path / "authority.json"
    atomic_json(authority_path, work)
    authority = c.reference(authority_path)
    value = c.artifact(authority)
    candidate = tmp_path / (c.NAME + ".json")
    atomic_json(candidate, value)
    c.replay(candidate, authority)
    value["completed_count"] = 1
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="immutable_reduction_drift"):
        c.replay(candidate, authority)
    authority_path.unlink()
    with pytest.raises(FileNotFoundError):
        c.replay(candidate, authority)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    """Use real numerical responses as operands for synthetic clock mutation controls."""
    import sys
    from carnot.reporting.installed_cost_session_8661 import session

    raw = tmp_path_factory.mktemp("8661-evidence")
    bundle = c.original_inputs(c.ROOT)
    config = raw / "head.json"
    atomic_json(config, dict(head=bundle["head"], extension=bundle["extension"]))
    p = session(
        [sys.executable, "-u", str(c.ROOT / "scripts/experiments/resident_worker_8608.py")],
        config,
        c.roster()[0],
        bundle["requests"],
        raw,
        raw / "session",
    )
    return bundle, p


def test_complete_matrix_reduction_and_pinned_operand_rejection(tmp_path, evidence):
    """REQ-REPORT-8661: synthetic paired clocks test ratios without claiming measured speed."""
    bundle, original = evidence
    work = c.empty_work(tmp_path, "20261019")
    work["bundle"] = bundle
    period = original["end_ns"] - original["start_ns"]
    for row in work["rows"]:
        p = deepcopy(original)
        p["slot"] = {k: row[k] for k in c.roster()[0]}
        if row["arm"].endswith("native"):
            p["ready"]["loaded_extension"] = bundle["extension"]
        p["observations"] = []
        for panel in range(row["length"]):
            for i, old in enumerate(original["observations"]):
                obs = deepcopy(old)
                for key in ["start_ns", "end_ns"]:
                    obs[key] += panel * period
                for span in obs["response"]["spans"]:
                    span["start_ns"] += panel * period
                    span["end_ns"] += panel * period
                obs["request"]["request_id"] = obs["response"]["request_id"] = f"{panel}-{i}"
                p["observations"].append(obs)
        p["end_ns"] += (row["length"] - 1) * period
        path = tmp_path / (row["unit_id"] + ".json")
        atomic_json(path, p)
        row.update(status="completed", primitive_reference=c.reference(path))
    result = c.reduce(work)
    assert result["completed_count"] == 120 and len(result["per_request_rows"]) == 53760
    assert len(result["paired_throughput_intervals"]) == 6
    assert result["installed_cost_ready_score"] == result["packaging_noninferiority_score"] == 1
    assert result["nfr01_score"] == 0
    assert all(r["lower95"] == r["upper95"] == 1 for r in result["paired_throughput_intervals"])
    work["validation_receipts"] = [
        dict(name=n, passed=True)
        for n in [
            "owned_tests",
            "exact_node_union",
            "owned_coverage",
            "e2e018",
            "e2e021",
            "ruff_check",
            "ruff_format",
            "strict_mypy",
            "spec_coverage",
        ]
    ]
    authority_path = tmp_path / "authority.json"
    atomic_json(authority_path, work)
    value = c.artifact(c.reference(authority_path))
    assert value["verdict_class"] == "null" and value["required_checks_passed"]
    p["slot"]["repeat"] = -1
    path = Path(work["rows"][-1]["primitive_reference"]["path"])
    atomic_json(path, p)
    work["rows"][-1]["primitive_reference"] = c.reference(path)
    with pytest.raises(ValueError, match="slot_identity"):
        c.reduce(work)
    work = c.empty_work(tmp_path, "20261019")
    source = tmp_path / "operand"
    source.write_text("original")
    work["source_artifact_hashes"] = [c.reference(source)]
    atomic_json(authority_path, work)
    authority = c.reference(authority_path)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, c.artifact(authority))
    source.write_text("changed")
    with pytest.raises(ValueError, match="replay_operand_hash"):
        c.replay(candidate, authority)


def test_head_and_original_input_admission_rejections(monkeypatch):
    """REQ-VERIFY-8661: changing a real head, request seal, wheel or method fails closed."""
    from carnot.reporting import resident_provenance_8593 as provenance

    bundle, gates, refs = c.admit(c.ROOT)
    assert not gates and len(bundle["requests"]) == 64 and refs
    original_load, original_ref = c.load, c.reference
    with monkeypatch.context() as m:
        m.setattr(c, "read_bound_sidecar", lambda *a: {"report": {"passed": False}})
        with pytest.raises(ValueError, match="upstream_publication"):
            c.admit(c.ROOT)
    for target, mutate, message in [
        (
            "v743-methods-protocol.json",
            lambda d: d["installed_entry_2x2"].update(batch=2),
            "registered_method_drift",
        ),
        (
            "frozen_heads.json",
            lambda d: d["heads"]["spline9"][0]["parameters"].append(1),
            "loaded_head",
        ),
    ]:

        def altered(ref):
            d = original_load(ref)
            if Path(ref["path"]).name == target:
                mutate(d)
            return d

        with monkeypatch.context() as m:
            m.setattr(c, "load", altered)
            with pytest.raises(ValueError, match=message):
                c.admit(c.ROOT)
    with monkeypatch.context() as m:
        m.setattr(provenance, "check_descriptor", lambda _: False)
        with pytest.raises(ValueError, match="head_descriptor"):
            c.admit(c.ROOT)
    with monkeypatch.context() as m:
        changed = deepcopy(bundle)
        changed["requests"].reverse()
        m.setattr(c, "original_inputs", lambda _: changed)
        with pytest.raises(ValueError, match="original_request_identity"):
            c.admit(c.ROOT)
    for suffix, message in [
        ("original64-requests.json", "request_hash"),
        (".whl", "frozen_operand_hash"),
    ]:
        with monkeypatch.context() as m:
            m.setattr(
                c,
                "reference",
                lambda p: (
                    dict(original_ref(p), sha256="wrong")
                    if str(p).endswith(suffix)
                    else original_ref(p)
                ),
            )
            with pytest.raises(ValueError, match=message):
                c.admit(c.ROOT)
    with monkeypatch.context() as m:
        m.setattr(c.zipfile.ZipFile, "read", lambda *a: b"altered entry")
        with pytest.raises(ValueError, match="wheel_entry_drift"):
            c.admit(c.ROOT)


def test_benefit_classification_needs_all_owned_checks(tmp_path, monkeypatch):
    """REQ-VERIFY-8661: even strong synthetic costs cannot waive failed validation."""
    work = c.empty_work(tmp_path, "20261019")
    reduced = c.reduce(work)
    reduced.update(
        completed_count=120,
        censored_count=0,
        installed_cost_ready_score=1,
        packaging_noninferiority_score=1,
        nfr01_score=1,
    )
    monkeypatch.setattr(c, "reduce", lambda _: deepcopy(reduced))
    work["validation_receipts"] = [
        dict(name=n, passed=True)
        for n in [
            "owned_tests",
            "exact_node_union",
            "owned_coverage",
            "e2e018",
            "e2e021",
            "ruff_check",
            "ruff_format",
            "strict_mypy",
            "spec_coverage",
        ]
    ]
    path = tmp_path / "authority.json"
    atomic_json(path, work)
    assert c.artifact(c.reference(path))["verdict_class"] == "circular_positive"
    work["validation_receipts"][0]["passed"] = False
    atomic_json(path, work)
    assert c.artifact(c.reference(path))["verdict_class"] == "disqualified"
