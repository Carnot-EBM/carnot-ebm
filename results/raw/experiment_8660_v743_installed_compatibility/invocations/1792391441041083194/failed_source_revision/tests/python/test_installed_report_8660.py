"""REQ-REPORT-8660: immutable invocation identity and closed publication gates."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import installed_report_8660 as r
from carnot.reporting import installed_checkout_8660 as c
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_report_cold_replay_and_cli_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8660-PUBLICATION: rehashed summaries cannot replace operands."""
    work = r.measure(tmp_path / "raw", "20261019", tmp_path)
    work.update(
        validation_receipts=[],
        original_node_union={},
        current_node_union={},
        child_shard_manifest=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
    )
    path = tmp_path / "authority.json"
    atomic_json(path, work)
    authority = c.reference(path)
    value = r.artifact(authority)
    assert value["experiment_id"] == 8660
    assert value["verdict_class"] == "blocked"
    assert value["installed_measurement_ready_score"] == 0
    candidate = tmp_path / (r.NAME + ".json")
    atomic_json(candidate, value)
    r.replay(candidate, authority)
    changed = deepcopy(value)
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(candidate, changed)
    with pytest.raises(ValueError, match="immutable_reduction_drift"):
        r.replay(candidate, authority)
    for argv in [
        ["--date", "bad"],
        ["--cold-replay", str(candidate)],
        ["--private-output", str(c.ROOT / "bad.json")],
        ["--input-root", str(tmp_path)],
    ]:
        assert r.main(argv) == 1
    monkeypatch.setattr(r, "run", lambda *args: 0)
    assert r.main(["--date", "20261019", "--private-output", str(candidate)]) == 0
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(c.ROOT / r.CLI), "--date", "bad"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 1


def test_readiness_cannot_waive_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8660-CUSTODY: original teardown and owned coverage are mandatory."""
    work = r.measure(tmp_path / "raw", "20261019", tmp_path)
    work.update(
        validation_receipts=[dict(name="owned_failure", passed=False)],
        original_node_union={},
        current_node_union={},
        child_shard_manifest=[],
    )
    path = tmp_path / "authority.json"
    atomic_json(path, work)
    value = r.artifact(c.reference(path))
    assert value["verdict_class"] == "disqualified"
    assert value["branch_gate_values"]["installed_measurement_ready_score"] == 0


def test_authenticated_history_and_tampered_logs(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8660-OWNERSHIP: a log hash and observed teardown errors are both required."""
    log = tmp_path / "prior.stdout"
    log.write_text("6 passed, 1 warning, 2 errors TrackedResultWriteError")
    prior = dict(
        validation_receipts=[
            dict(name="nodes-1", stdout_path=str(log), stdout_sha256=c.reference(log)["sha256"])
        ],
        honest_verdict="complete_disqualified_prior",
    )

    def fake_measure(*args):
        return dict(
            gate_check_summary=[],
            source_artifact_hashes=[],
            cleanup_rows=[],
            preconditions_checked={},
        )

    monkeypatch.setattr(r.old, "measure", fake_measure)
    monkeypatch.setattr(r.old, "load", lambda *args: prior)
    root = tmp_path / "source"
    (root / r.HISTORY).parent.mkdir(parents=True)
    (root / r.HISTORY).write_text("{}")
    observed = r.measure(tmp_path / "raw", "20261019", root)
    assert observed["historical_8646"]["teardown_errors"] == 2
    assert observed["alias_rejection_rows"][0]["passed"]
    log.write_text("changed")
    with pytest.raises(ValueError, match="historical_log_hash"):
        r.measure(tmp_path / "bad-hash", "20261019", root)
    prior["validation_receipts"][0]["stdout_sha256"] = c.reference(log)["sha256"]
    with pytest.raises(ValueError, match="historical_teardown_evidence"):
        r.measure(tmp_path / "bad-log", "20261019", root)


def test_private_publication_and_actual_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8660-PUBLICATION: real terminal CLIs reject missing and rehashed input."""
    from carnot.reporting import installed_validation_8660 as validation

    original_measure = r.measure

    def measured(raw, run_date, root):
        value = original_measure(raw, run_date, tmp_path / "missing-inputs")
        operand = tmp_path / "protected-input"
        operand.write_text("immutable")
        value["original_result_hashes_before_after"] = [dict(before=c.reference(operand))]
        return value

    def checked(work, raw):
        work.update(original_node_union={}, current_node_union={}, child_shard_manifest=[])
        return [dict(name=name, passed=True) for name in r.report.REQUIRED]

    monkeypatch.setattr(r, "measure", measured)
    monkeypatch.setattr(validation, "validate", checked)
    monkeypatch.setattr(r, "reader_receipt", lambda *a, **kw: dict(passed=True))
    output = tmp_path / (r.NAME + ".json")
    assert r.run(output, "20261019", tmp_path) == 0
    value = json.loads(output.read_bytes())
    authority = value["authority_reference"]
    assert (
        r.main(
            ["--cold-replay", str(output), "--authority", authority["path"], authority["sha256"]]
        )
        == 0
    )
    work = r.old.load(authority)
    operand = Path(work["original_result_hashes_before_after"][0]["before"]["path"])
    work["source_artifact_hashes"] = [c.reference(operand)]
    extra = tmp_path / "replay-authority.json"
    atomic_json(extra, work)
    alternate = c.reference(extra)
    atomic_json(output, r.artifact(alternate))
    operand.write_text("tampered immutable operand")
    with pytest.raises(ValueError, match="replay_input_hash"):
        r.replay(output, alternate)
    monkeypatch.setattr(r, "reader_receipt", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="publication_reader"):
        r.run(tmp_path / "other" / (r.NAME + ".json"), "20261019", tmp_path)


def test_positive_summary_requires_each_owned_gate(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8660-PUBLICATION: passing receipts cannot hide failed teardown."""
    minimal = dict(
        raw=str(tmp_path),
        gate_check_summary=[],
        original_node_union=dict(passed=True),
        current_node_union=dict(passed=True),
        alias_rejection_rows=[dict(passed=True)],
        materialization_manifest=dict(copies=[dict(passed=True)]),
        original_result_hashes_before_after=[dict(passed=True)],
    )
    path = tmp_path / "gate-authority.json"
    monkeypatch.setattr(
        r.report,
        "artifact",
        lambda ref: dict(
            verdict_class="circular_positive",
            acceptance_gates=[],
            branch_gate_values=dict(installed_measurement_ready_score=1),
            installed_measurement_ready_score=1,
            required_checks_passed=True,
        ),
    )
    for key in [
        "original_node_union",
        "current_node_union",
        "alias_rejection_rows",
        "materialization_manifest",
        "original_result_hashes_before_after",
        None,
    ]:
        work = deepcopy(minimal)
        if key:
            work[key] = {} if isinstance(work[key], dict) else []
        atomic_json(path, work)
        value = r.artifact(c.reference(path))
        assert value["installed_measurement_ready_score"] == int(key is None)
        assert value["verdict_class"] == ("circular_positive" if key is None else "disqualified")
