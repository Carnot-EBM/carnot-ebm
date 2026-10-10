"""REQ-VERIFY-8335 / REQ-REPORT-8335: prediction custody excludes targets."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import reserved_prediction_seal_8335 as e
from carnot.reporting import reserved_prediction_execution_8335 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting import v718_replay_history as history


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Original cached bytes are read while all owned writes stay private."""
    raw = tmp_path_factory.mktemp("seal8335")
    return e.measure(e.ROOT, raw), raw


def test_original_roster_and_predictor_isolation(measured):
    """SCENARIO-VERIFY-8335-ISOLATION: unavailable is never zero probability."""
    work, _ = measured
    assert not work["failures"]
    bundle = work["bundle"]
    e.validate_bundle(bundle)
    predictions = e.score(bundle, work["issued_at"])
    assert len(predictions) == 128 * 6
    assert sum(r["p"] is not None for r in predictions) == 97 * 6
    assert all(r["action"] == "escalate" for r in predictions if r["p"] is None)
    assert all("y" not in r for r in predictions)
    assert work["parity"]["passed"]
    for mutation in ["label", "source", "head", "roster", "count", "feature"]:
        bad = deepcopy(bundle)
        if mutation == "label":
            bad["slots"][0]["y"] = 1
        elif mutation == "source":
            bad["slots"][0]["source_sha256"] = "changed"
        elif mutation == "head":
            bad["head_hash"] = "changed"
        elif mutation == "roster":
            bad["slots"][0]["slot"] = 2
        elif mutation == "count":
            bad["slots"].pop()
        else:
            bad["slots"][0]["x"] = None
        with pytest.raises(ValueError):
            e.validate_bundle(bad)


def test_cold_replay_and_check_dispositions(measured, tmp_path):
    """REQ-REPORT-8335: rehashing cannot authorize invented predictions."""
    work, raw = measured
    value = e.build(work, raw, [dict(passed=True)])
    assert value["predictions_ready_score"] == 1
    assert value["independent_count"] == 128
    assert value["completed_count"] == 97
    assert value["excluded_count"] == 31
    assert value["independent_generalization_score"] == 0
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    for field in ["predictions_ready_score", "sealed_head_hash"]:
        bad = dict(value, **{field: "changed"})
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    assert not e.replay(tmp_path / "absent")
    atomic_json(path, {})
    assert not e.replay(path)
    assert e.build(work, raw, [dict(passed=False)])["verdict_class"] == "disqualified"
    blocked = e.measure(tmp_path / "missing", tmp_path / "blocked")
    v = e.build(blocked, tmp_path / "blocked", [dict(passed=True)])
    assert v["verdict_class"] == "blocked"
    assert v["gate_check_summary"][-1]["observed"] is None


def test_actual_cli_child_replay_and_recovery(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8335-CLI: genuine processes retain terminal failures."""
    output = tmp_path / (e.NAME + ".json")
    prefix = [sys.executable]
    if "COVERAGE_RCFILE" in os.environ:
        prefix += ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
    result = subprocess.run(
        prefix + [str(e.ROOT / e.CLI), "--private-run", "--output", str(output)],
        capture_output=True,
        timeout=180,
        cwd="/tmp",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert e.replay(output)
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    raw = Path(value["work_reference"]["path"]).parent
    runner.publish(dict(value, predictions_ready_score=99), output, raw)
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    with pytest.raises(ValueError):
        runner.publish(dict(value, experiment_id=99), output, raw)
    for args in [["--date", "20000101"], ["--private-run"]]:
        with pytest.raises(SystemExit):
            runner.main(args)
    original = runner.manifest

    def bounded(private, candidate):
        plan = original(private, candidate)
        plan["commands"] = [dict(name="failure", argv=["/bin/false"], deadline_s=10)]
        return plan

    monkeypatch.setattr(runner, "manifest", bounded)
    blocked = tmp_path / "other" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "missing"), "--output", str(blocked)]) == 0
    assert json.loads(blocked.read_bytes())["verdict_class"] == "disqualified"


def test_finding_consumer_qualified_in_this_task(tmp_path):
    """SCENARIO-VERIFY-8335-CLI: raw exits never substitute for findings."""
    evidence = runner.qualify_findings(tmp_path / "finding_controls")
    assert all(r["passed"] for r in evidence["receipts"])
    assert evidence["audits"][0]["passed"]
    assert not evidence["audits"][1]["passed"]
    report = evidence["audits"][0]["report"]
    path = Path(report["reports"][0]["artifact"])
    proof = dict(recomputed=True, deliberate_error_rejected=True)
    for change in [
        dict(candidate_sha256="wrong"),
        dict(verifier_sha256="wrong"),
        dict(reports=[]),
        dict(reports="malformed"),
    ]:
        assert not history.consume(dict(report, **change), path, 1, proof)["passed"]
    for field, value in [
        ("severity", "warn"),
        ("severity", "critical"),
        ("severity", "unknown"),
        ("kind", "unknown"),
    ]:
        bad = deepcopy(report)
        bad["reports"][0]["flags"][0][field] = value
        assert not history.consume(bad, path, 1, proof)["passed"]
    assert not history.consume(report, path, 2, proof)["passed"]


def test_failure_coverage_and_child_rejection(measured, tmp_path, monkeypatch):
    """REQ-VERIFY-8335: coverage, structural errors and real children fail closed."""
    work, raw = measured
    for percent in [0, 100]:
        coverage = tmp_path / "coverage.json"
        atomic_json(
            coverage,
            dict(
                totals=dict(percent_covered=percent),
                files={p: dict(summary=dict(missing_lines=0)) for p in e.OWNED},
            ),
        )
        v = e.build(
            dict(work, owned_coverage_reference=e.reference(coverage)), raw, [dict(passed=True)]
        )
        assert v["predictions_ready_score"] == int(percent == 100)
    path = tmp_path / "bundle.json"
    bad = deepcopy(work["bundle"])
    bad["slots"][0]["y"] = 1
    atomic_json(path, bad)
    assert runner.main(["--score-child", str(path), str(tmp_path / "out"), "timestamp"]) == 1
    assert runner.main(["--score-child"]) == 1
    with pytest.raises(ValueError, match="bundle_fields"):
        e.validate_bundle(dict(work["bundle"], evaluator={}))
    seal = tmp_path / "immutable.json"
    e.immutable(seal, {})
    with pytest.raises(FileExistsError):
        e.immutable(seal, {})
    monkeypatch.setattr(runner, "check", lambda *args: dict(passed=False))
    failed = e.measure(e.ROOT, tmp_path / "failed_child")
    assert (
        e.build(failed, tmp_path / "failed_child", [dict(passed=True)])["verdict_class"]
        == "disqualified"
    )

    def reject(bundle):
        raise ValueError("structure")

    monkeypatch.setattr(e, "validate_bundle", reject)
    blocked = e.measure(e.ROOT, tmp_path / "structure")
    assert blocked["failures"][0]["artifact_field"] == "structure"


def test_rehashed_primitive_rejections(measured, tmp_path):
    """SCENARIO-VERIFY-8335-ISOLATION: primitive changes cannot hide in a new hash."""
    original, _ = measured
    for mutation in ["work_hash", "ref", "bundle", "prediction", "parity", "stream", "manifest"]:
        raw = tmp_path / mutation
        raw.mkdir()
        work = deepcopy(original)
        if mutation == "ref":
            work["refs"][0]["sha256"] = "changed"
        elif mutation == "bundle":
            work["bundle"]["slots"][0]["x"][0] += 1
        elif mutation == "prediction":
            work["predictions"][0]["p"] = 0.123
        elif mutation == "parity":
            work["parity"]["passed"] = False
        elif mutation in ["stream", "manifest"]:
            target = raw / "forged.json"
            atomic_json(target, {})
            if mutation == "stream":
                work["prediction_refs"][0] = e.reference(target)
            else:
                work["seal_reference"] = e.reference(target)
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, raw, [dict(passed=True)])
        if mutation == "work_hash":
            value["work_reference"]["sha256"] = "changed"
        candidate = raw / "candidate.json"
        atomic_json(candidate, value)
        assert not e.replay(candidate), mutation
