"""REQ-REPORT-8332 / REQ-VERIFY-8332: independent custody and cold rejection."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v719_contract_replay as e
from carnot.reporting import v719_replay_runner as r
from carnot.reporting import v718_contract_replay as old
from carnot.reporting import v718_replay_history as h
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child


def cli(*args):
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=360,
    )


@pytest.fixture(scope="module")
def current(tmp_path_factory):
    private = tmp_path_factory.mktemp("v719")
    output = private / (e.NAME + ".json")
    result = cli("--date", "20261009", "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    return output, value, work


def test_current_custody(current):
    """SCENARIO-REPORT-8332-METHODS: custody survives failed historical science."""
    output, value, work = current
    assert e.replay(output)
    assert value["experiment_id"] == 8332 and value["run_date"] == "20261009"
    assert value["current_contract_ready_score"] == 1
    assert value["cached_support_ready_score"] == 1
    assert value["history_reader_ready_score"] == 1
    assert value["first_reduction_mismatch"]["field"] == "gate_check_summary[15].hash"
    assert value["historical_primary_verdict"].startswith("complete_disqualified")
    assert work["prior_dispositions"][0]["verdict_class"] == "disqualified"
    assert work["prior_dispositions"][0]["disposition"] == "authenticated_failure"
    assert work["prior_dispositions"][1]["counts"] == dict(
        actual_executed_task_count=5, pre_gate_count=3, missing_output_count=6
    )
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["invocation_argv"][value["invocation_argv"].index("--date") + 1] == "20261009"
    assert value["cached_support"]["current_receipt"]["protocol_sha256"] == e.base.PIN
    assert len(value["rows"]) == 14 and len(value["future_dependencies"]) == 13
    assert work["methods"]["science_changed"] is False
    assert all(a["findings"] for a in work["finding_audits"])
    assert work["finding_audits"][0]["passed"] and not work["finding_audits"][1]["passed"]


def test_existing_uncovered_rejection(current, tmp_path):
    """SCENARIO-REPORT-8332-REPLAY: reach original line 357 with rehashed operands."""
    _, value, work = current
    forged = deepcopy(work)
    forged["refs"] = forged["refs"][:-1]
    raw = tmp_path / "forged"
    atomic_json(raw / "measurement.json", forged)
    candidate = tmp_path / "changed_refs.json"
    rebuilt = e.build(forged, value["validation_receipts"], raw, tmp_path / (e.NAME + ".json"))
    atomic_json(candidate, rebuilt)
    assert cli("--cold-replay", candidate).returncode == 1
    assert not e.replay(candidate)
    # Use unchanged V718 reduction to exercise the exact legacy reader branch.
    original = old.measure(tmp_path / "missing", tmp_path / "legacy")
    atomic_json(tmp_path / "legacy" / "measurement.json", original)
    legacy = old.build(original, [], tmp_path / "legacy", tmp_path / (old.NAME + ".json"))
    legacy["source_artifact_hashes"] = []
    legacy["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in legacy.items() if k != "reproducibility_checksum"}
    )
    atomic_json(candidate, legacy)
    data = tmp_path / "legacy_coverage"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=/dev/null",
            "--data-file=" + str(data),
            "--include=" + str(old.ROOT / "python/carnot/reporting/v718_contract_replay.py"),
            str(old.ROOT / old.CLI),
            "--cold-replay",
            str(candidate),
        ],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 1
    from coverage import CoverageData

    recorded = CoverageData(basename=str(data))
    recorded.read()
    assert 357 in recorded.lines(str(old.ROOT / "python/carnot/reporting/v718_contract_replay.py"))


def test_gate_independence(current, tmp_path):
    """REQ-REPORT-8332: unknown history never upgrades or poisons cached custody."""
    _, value, work = current
    changed = deepcopy(work)
    changed["history"]["deterministic"] = False
    raw = tmp_path / "history"
    atomic_json(raw / "measurement.json", changed)
    result = e.build(changed, value["validation_receipts"], raw, tmp_path / (e.NAME + ".json"))
    assert result["history_reader_ready_score"] == 0
    assert result["cached_support_ready_score"] == 1
    failed = e.build(
        work,
        [dict(passed=False)],
        Path(value["work_reference"]["path"]).parent,
        tmp_path / (e.NAME + ".json"),
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["current_contract_ready_score"] == failed["cached_support_ready_score"] == 0


def test_authority_versions(tmp_path):
    """SCENARIO-REPORT-8332-AUTHORITY: explicit old readers never read vNEXT."""
    for milestone in ["2026.10.716", "2026.10.717", "2026.10.718", e.MILESTONE]:
        result = cli("--inspect-milestone", milestone)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout.splitlines()[-1])
        assert report["count"] == 14 and report["milestone"] == milestone
    assert cli("--inspect-milestone", e.MILESTONE, "--check-authority").returncode == 0
    root = tmp_path / "root"
    for name in [e.DESIGN, e.ACTIVE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((e.ROOT / name).read_bytes())
    (root / e.STAGED).write_bytes((root / e.ACTIVE).read_bytes())
    assert e.authority(root, tmp_path / "staged")["activated"]
    plan = yaml.safe_load((root / e.ACTIVE).read_text())
    plan["tasks"][0]["prompt"] += " changed"
    (root / e.ACTIVE).write_text(yaml.safe_dump(plan))
    assert (
        cli("--root", root, "--inspect-milestone", e.MILESTONE, "--check-authority").returncode == 1
    )
    assert (
        cli(
            "--root", tmp_path / "missing", "--inspect-milestone", e.MILESTONE, "--check-authority"
        ).returncode
        == 1
    )
    assert (
        cli("--root", root, "--inspect-milestone", "2026.10.716", "--check-authority").returncode
        == 1
    )


def test_private_missing_and_cli_errors(tmp_path):
    """SCENARIO-VERIFY-8332-CLI: missing operands block; invalid invocations fail."""
    output = tmp_path / (e.NAME + ".json")
    result = cli("--root", tmp_path / "absent", "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    assert e.replay(output)
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert value["cached_support_ready_score"] == 0
    assert value["history_reader_ready_score"] == 0
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    for args in [["--date", "20261008"], ["--date"], ["--date", "20260101"]]:
        with pytest.raises(SystemExit):
            r.main(args)
    with pytest.raises(SystemExit):
        r.main(["--private-fixture", "--output", str(e.ROOT / "results" / (e.NAME + ".json"))])


def test_original_findings_fail_closed(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8332-FINDINGS: unchanged policy checks every report binding."""
    from carnot.reporting import v718_replay_runner as qualified

    candidate = tmp_path / "zero.json"
    atomic_json(candidate, dict(dense_sparse_error_max=0.0))
    report = dict(
        candidate_sha256=sha256_file(candidate),
        verifier_sha256=h.verifier_hash(),
        reports=[dict(artifact=str(candidate), loaded=True, flags=[], flag_count=0)],
    )
    assert h.consume(report, candidate, 0, {})["passed"]
    for field, wrong in [("candidate_sha256", "changed"), ("verifier_sha256", "changed")]:
        changed = deepcopy(report)
        changed[field] = wrong
        assert not h.consume(changed, candidate, 0, {})["passed"]
    for severity in ["warn", "critical", "unknown"]:
        changed = deepcopy(report)
        changed["reports"][0].update(
            flags=[
                dict(kind="IMPLAUSIBLE_PERFECT", severity=severity, detail="dense_sparse_error_max")
            ],
            flag_count=1,
        )
        assert not h.consume(
            changed, candidate, 1, dict(recomputed=True, deliberate_error_rejected=True)
        )["passed"]
    for bad in [{}, dict(reports=None), dict(reports=[None]), dict(reports=[dict(flags=None)])]:
        assert not h.consume(bad, candidate, 2, {})["passed"]
    stdout = tmp_path / "stdout"
    stdout.write_text("malformed")
    monkeypatch.setattr(
        qualified,
        "child",
        lambda *a, **k: dict(stdout_path=str(stdout), exit_code=2, timed_out=False),
    )
    assert not qualified.audit(candidate, tmp_path, {})["passed"]


def test_primitive_and_aggregate_controls(current, tmp_path):
    """SCENARIO-REPORT-8332-REPLAY: missing primaries, pre-gates and tampering persist."""
    output, value, work = current
    assert value["historical_dispositions"]["missing_output_count"] == 4
    assert value["historical_dispositions"]["pre_gate_count"] == 2
    for field in ["missing_output_count", "pre_gate_count", "science_ready_score"]:
        changed = deepcopy(value)
        changed["historical_dispositions"][field] = 99
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        candidate = tmp_path / (field + ".json")
        atomic_json(candidate, changed)
        assert cli("--cold-replay", candidate).returncode == 1
    changed = deepcopy(work)
    changed["support"]["counts"]["fit"]["usable"] += 1
    assert not e.replay_operands(changed)
    changed = deepcopy(work)
    changed["protocol"] = {}
    assert not e.replay_operands(changed)
    changed = deepcopy(work)
    changed["prior_dispositions"][0]["verdict_class"] = "null"
    assert not e.replay_operands(changed)
    assert cli("--cold-replay", output).returncode == 0


def test_owned_failure_recovery(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8332-CLI: real child error yields checked disqualification."""
    from carnot.reporting import v718_replay_runner as qualified

    raw = tmp_path / "raw"
    output = tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw)
    atomic_json(raw / "measurement.json", work)
    receipt = child(
        "deliberate_error",
        [sys.executable, "-c", "raise SystemExit(7)"],
        raw / "checks",
        deadline=10,
    )
    value = e.build(work, [receipt], raw, output)
    with patch_bindings(qualified):
        qualified.publish(value, output, raw)
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert e.replay(output)
    timed = child(
        "deadline",
        [sys.executable, "-c", "import time; time.sleep(20)"],
        raw / "checks",
        deadline=0.05,
        heartbeat=0.02,
    )
    assert timed["timed_out"] and not timed["passed"]


def patch_bindings(qualified):
    from contextlib import ExitStack
    from unittest.mock import patch

    stack = ExitStack()
    stack.enter_context(e.bindings())
    stack.enter_context(patch.object(qualified, "e", e))
    return stack


def test_rejected_candidate_recovers(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8332-CLI: preserve a failing terminal command before recovery."""
    from carnot.reporting import v718_replay_runner as qualified

    raw = tmp_path / "raw"
    output = tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw)
    atomic_json(raw / "measurement.json", work)
    good = child("pass", [sys.executable, "-c", 'print("checked")'], raw / "checks", deadline=10)
    original = qualified.audit
    calls = []

    def fail_once(candidate, logs, proof):
        report = original(candidate, logs, proof)
        calls.append(candidate)
        if len(calls) == 1:
            report["passed"] = False
            report["receipt"] = child(
                "process_error",
                [sys.executable, "-c", "raise SystemExit(7)"],
                logs / "failure",
                deadline=10,
            )
        return report

    monkeypatch.setattr(qualified, "audit", fail_once)
    with patch_bindings(qualified):
        qualified.publish(e.build(work, [good], raw, output), output, raw)
    assert (raw / "rejected_candidate.json").is_file()
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert e.replay(output)


def test_execution_manifest(tmp_path):
    """REQ-VERIFY-8332: coverage scope and child deadlines are fixed before work."""
    plan = r.manifest(tmp_path)
    assert plan[0]["deadline"] == 600
    assert e.TEST in plan[0]["argv"]
    assert any(p["name"] == "private_E2E021" for p in plan)
    assert all(0 < p["deadline"] <= 600 for p in plan)
    assert all(str(e.ROOT / path) in (tmp_path / "coverage.ini").read_text() for path in e.OWNED)
