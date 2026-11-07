"""Exercise actual consumers and byte custody for REQ-REPORT-7928."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import (
    publish_primary,
    read_bound_sidecar,
    reader_receipt,
    validate_primary,
)


def primary() -> dict:
    """Use a mechanical null so a fixture never claims scientific benefit."""
    return {
        "experiment_id": 7928,
        "task_id": "exp7928-primary-publication",
        "honest_verdict": "complete_null_fixture",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "artifact_resolution_ready_score": 1,
    }


def test_nested_reports_and_terminal_revalidation(tmp_path):
    """SCENARIO-REPORT-7928-1: newer nested reports cannot shadow the primary."""
    output = tmp_path / "experiment_7928_fixture.json"
    observed = []

    def check(candidate):
        observed.append(sha256_file(candidate))
        return {"passed": True}

    receipt = publish_primary(output, primary(), check)
    sidecar = Path(receipt["sidecar_path"])
    os.utime(sidecar, ns=(output.stat().st_mtime_ns + 10**9,) * 2)
    actual = reader_receipt("exp7928-primary-publication", tmp_path)
    assert actual["passed"] and actual["gate_path"] == str(output)
    assert actual["gate_sha256"] == observed[0] == receipt["primary_sha256"]
    assert actual["document_path"] == str(output)
    assert read_bound_sidecar(output, sidecar)["primary_sha256"] == observed[0]
    assert len(list(tmp_path.glob("experiment_7928_*.json"))) == 1
    assert publish_primary(output, primary(), check)["primary_sha256"] == observed[0]


def test_legacy_shadow_is_required_negative(tmp_path):
    """SCENARIO-REPORT-7928-1: retain the actual newest-top-level failure."""
    output = tmp_path / "experiment_7928_fixture.json"
    atomic_json(output, primary())
    sidecar = output.with_suffix(".json.validators.json")
    atomic_json(sidecar, {"candidate_sha256": sha256_file(output)})
    os.utime(sidecar, ns=(output.stat().st_mtime_ns + 10**9,) * 2)
    result = reader_receipt("exp7928-primary-publication", tmp_path)
    assert not result["passed"] and result["gate_path"] == str(sidecar)
    assert result["gates"][0]["actual"] is None
    assert result["document_path"] == str(sidecar)
    with pytest.raises(ValueError, match="conflicting_primary"):
        publish_primary(output, primary(), lambda p: {"passed": True})


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"experiment_id": 1}, "producer_identity"),
        ({"task_id": "wrong"}, "producer_identity"),
        ({"honest_verdict": "running"}, "terminal_verdict"),
        ({"verdict_class": "invented"}, "terminal_verdict"),
        ({"flagged_adversarial": True}, "unsafe_readiness"),
        ({"verdict_class": "disqualified"}, "unsafe_readiness"),
        ({"verdict_class": "blocked"}, "unsafe_readiness"),
    ],
)
def test_validation_rejects_unsafe_primary(mutation, reason, tmp_path):
    """REQ-REPORT-7928: reject unsafe bytes before exposing them."""
    value = {**primary(), **mutation}
    with pytest.raises(ValueError, match=reason):
        validate_primary(value, tmp_path / "experiment_7928_fixture.json")


def test_invalid_names_shapes_and_failed_validator(tmp_path):
    """REQ-REPORT-7928: malformed identity and failed checks leave no primary."""
    for name in ["other.json", "experiment_7928_x.json.validators.json"]:
        with pytest.raises(ValueError, match="primary_name"):
            validate_primary(primary(), tmp_path / name)
    with pytest.raises(ValueError, match="primary_object"):
        validate_primary([], tmp_path / "experiment_7928_fixture.json")
    output = tmp_path / "experiment_7928_fixture.json"
    with pytest.raises(ValueError, match="candidate_rejected"):
        publish_primary(output, primary(), lambda p: {"passed": False})
    assert not output.exists()

    def tamper(candidate):
        candidate.write_text("{}")
        return {"passed": True}

    with pytest.raises(ValueError, match="candidate_changed"):
        publish_primary(output, primary(), tamper)


def test_conflicts_stale_hashes_and_concurrent_replacement(tmp_path):
    """SCENARIO-REPORT-7928-1: serialize replacements and reject stale bindings."""
    output = tmp_path / "experiment_7928_fixture.json"

    def replace(index):
        return publish_primary(output, {**primary(), "revision": index}, lambda p: {"passed": True})

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(replace, range(2)))
    current = sha256_file(output)
    valid = next(row for row in receipts if row["primary_sha256"] == current)
    assert read_bound_sidecar(output, Path(valid["sidecar_path"]))
    stale = next(row for row in receipts if row["primary_sha256"] != current)
    with pytest.raises(ValueError, match="stale_primary_hash"):
        read_bound_sidecar(output, Path(stale["sidecar_path"]))
    with pytest.raises(ValueError, match="sidecar_location"):
        read_bound_sidecar(output, output)
    other = tmp_path / "experiment_7928_conflict.json"
    atomic_json(other, primary())
    with pytest.raises(ValueError, match="conflicting_primary"):
        replace(3)
    other.unlink()
    atomic_json(output, {**primary(), "task_id": "exp7928-other"})
    with pytest.raises(ValueError, match="conflicting_identity"):
        replace(3)


def test_real_readers_missing_malformed_disqualified_absent(tmp_path):
    """SCENARIO-REPORT-7928-1: record reader failures without changing APIs."""
    task = "exp7928-primary-publication"
    assert not reader_receipt(task, tmp_path)["passed"]
    output = tmp_path / "experiment_7928_fixture.json"
    output.write_text("{")
    assert not reader_receipt(task, tmp_path)["passed"]
    with pytest.raises(json.JSONDecodeError):
        publish_primary(output, primary(), lambda p: {"passed": True})
    output.unlink()
    value = {**primary(), "verdict_class": "disqualified", "artifact_resolution_ready_score": 0}
    publish_primary(output, value, lambda p: {"passed": True})
    assert not reader_receipt(task, tmp_path)["passed"]
    atomic_json(output, {"experiment_id": 7928})
    assert reader_receipt(task, tmp_path)["gates"][0]["actual"] is None
