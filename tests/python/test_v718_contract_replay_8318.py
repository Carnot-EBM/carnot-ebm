"""REQ-REPORT-8318 / REQ-VERIFY-8318: exact authority and independent replay."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v718_contract_replay as e
from carnot.reporting import v718_replay_history as h
from carnot.reporting import v718_replay_runner as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def test_current_and_historical_authority(tmp_path):
    """SCENARIO-REPORT-8318-AUTHORITY: prompts and old versions bind exact bytes."""
    root = tmp_path / "root"
    for name in [e.DESIGN, e.ACTIVE, e.PROTOCOL, e.METHODS, h.OLD_DESIGN, h.PRIOR_DESIGN]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    result = e.authority(root, tmp_path / "raw")
    assert result["activated"] and len(result["contract_rows"]) == 14
    assert h.design(root, "2026.10.716") == root / h.OLD_DESIGN
    assert h.design(root, "2026.10.717") == root / h.PRIOR_DESIGN
    active = yaml.safe_load((root / e.ACTIVE).read_text())
    active["tasks"][0]["prompt"] += " changed"
    (root / e.ACTIVE).write_text(yaml.safe_dump(active))
    assert not e.authority(root, tmp_path / "changed")["activated"]


def test_historical_order_drift(tmp_path):
    """SCENARIO-REPORT-8318-REPLAY: retain the first mismatch and all fields."""
    result = h.historical(e.ROOT, tmp_path / "raw")
    assert result["first_reduction_mismatch"]["field"].startswith("gate_check_summary[15]")
    assert result["cause"] == "serialized_history_key_order"
    assert result["deterministic"]
    assert result["historical_verdict_class"] == "disqualified"
    assert result["recomputed"]["actual_executed_task_count"] == 8
    assert result["recomputed"]["pre_gate_count"] == 2
    assert result["recomputed"]["missing_output_count"] == 4
    assert result["recomputed"]["science_ready_score"] == 0
    assert h.first_difference({}, {}) is None
    assert h.first_difference([1], []) is not None


def test_support_recount(tmp_path):
    """REQ-REPORT-8318: fit support uses original shards and every manifest slot."""
    source = json.loads((e.ROOT / h.CUSTODY).read_bytes())
    result = h.support(source, tmp_path / "raw", [])
    assert result["ready"]
    assert result["counts"] == source["class_support_by_role"]
    changed = deepcopy(source)
    changed["source_role_manifest"][0]["unit_id"] = "forged"
    with pytest.raises(ValueError, match="manifest"):
        h.support(changed, tmp_path / "changed", [])


def test_finding_policy(tmp_path):
    """SCENARIO-VERIFY-8318-FINDINGS: false zero cannot pass an info disposition."""
    candidate = tmp_path / "zero.json"
    atomic_json(candidate, dict(dense_sparse_error_max=0.0))
    flags = [
        dict(
            kind="IMPLAUSIBLE_PERFECT", severity="info", detail="dense_sparse_error_max exact zero"
        )
    ]
    report = dict(
        candidate_sha256=sha256_file(candidate),
        verifier_sha256=h.verifier_hash(),
        reports=[dict(artifact=str(candidate), loaded=True, flags=flags, flag_count=1)],
    )
    assert h.consume(report, candidate, 1, dict(recomputed=True, deliberate_error_rejected=True))[
        "passed"
    ]
    assert not h.consume(
        report, candidate, 1, dict(recomputed=False, deliberate_error_rejected=True)
    )["passed"]
    assert not h.consume(report, candidate, 2, {})["passed"]
    for key, value in [("severity", "warn"), ("severity", "unknown"), ("kind", "NEW_KIND")]:
        changed = deepcopy(report)
        changed["reports"][0]["flags"][0][key] = value
        assert not h.consume(changed, candidate, 1, {})["passed"]
    assert not h.consume({}, candidate, 0, {})["passed"]
    assert not h.consume(dict(reports=[None]), candidate, 0, {})["passed"]
    assert not h.consume(dict(reports=[dict(flags=None)]), candidate, 0, {})["passed"]


def test_private_cli_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8318-CLI: actual children reject rehashed aggregate drift."""
    output = tmp_path / (e.NAME + ".json")
    run = subprocess.run(
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--private-fixture",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert e.replay(output)
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    value["current_contract_ready_score"] = 1
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(tmp_path / "tamper.json", value)
    assert not e.replay(tmp_path / "tamper.json")
    assert not e.replay(tmp_path / "missing")


def test_actual_primitive_cli(tmp_path):
    """SCENARIO-VERIFY-8318-CLI: private real-history replay retains old failures."""
    output = tmp_path / (e.NAME + ".json")
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--root",
            str(e.ROOT),
            "--output",
            str(output),
            "--private-fixture",
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["history_reader_ready_score"] == value["cached_support_ready_score"] == 1
    assert value["historical_primary_verdict"] == "complete_disqualified_owned_validation"
    assert e.replay(output)
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    assert work["numeric"]["observed_error"] == 0.0
    assert work["numeric"]["deliberate_error"] == 0.125
    assert all(a["findings"] for a in work["finding_audits"])
    assert work["finding_audits"][0]["passed"] and not work["finding_audits"][1]["passed"]
    changed = deepcopy(work)
    changed["support"]["counts"]["fit"]["usable"] += 1
    assert not e.replay_operands(changed)
    changed = deepcopy(work)
    changed["contract"]["activated"] = False
    assert not e.replay_operands(changed)
    changed = deepcopy(work)
    changed["protocol"] = {}
    assert not e.replay_operands(changed)
    candidate = tmp_path / "rehashed.json"
    changed = deepcopy(value)
    changed["historical_dispositions"]["science_ready_score"] = 1
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(candidate, changed)
    assert not e.replay(candidate)
    for index, field in enumerate(["recomputed", "first_reduction_mismatch"]):
        forged = deepcopy(work)
        if field == "recomputed":
            forged["history"][field]["completed_count"] = 13
        else:
            forged["history"][field]["field"] = "forged"
        target = tmp_path / f"forged-work-{index}"
        target.mkdir()
        atomic_json(target / "measurement.json", forged)
        atomic_json(candidate, e.build(forged, value["validation_receipts"], target, output))
        assert not e.replay(candidate)
    for field, observed in [("pre_gate_count", 0), ("missing_output_count", 0)]:
        changed = deepcopy(value)
        changed["historical_dispositions"][field] = observed
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        atomic_json(candidate, changed)
        negative = subprocess.run(
            [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(candidate)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert negative.returncode == 1


def test_cli_versions_and_errors(tmp_path):
    """SCENARIO-REPORT-8318-AUTHORITY: old readers use old design in fresh processes."""
    for milestone in ["2026.10.716", "2026.10.717", "2026.10.718"]:
        result = subprocess.run(
            [sys.executable, "-u", str(e.ROOT / e.CLI), "--inspect-milestone", milestone],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout.splitlines()[-1])
        assert report["count"] == 14 and report["milestone"] == milestone
    assert r.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        r.main(["--private-fixture", "--output", str(e.ROOT / "results" / (e.NAME + ".json"))])
    with pytest.raises(SystemExit):
        r.main(["--date", "20260101"])


def test_owned_failure_recovery(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8318-CLI: failed real children clear readiness and recover."""
    from carnot.reporting.v709_execution import child

    raw = tmp_path / "raw"
    output = tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw)
    atomic_json(raw / "measurement.json", work)
    receipt = child("pass", [sys.executable, "-c", "print('checked')"], raw / "checks", deadline=10)
    value = e.build(work, [receipt], raw, output)
    original = r.audit
    calls = []

    def failing_once(candidate, logs, proof):
        calls.append(candidate)
        if len(calls) == 1:
            report = original(candidate, logs, proof)
            report["passed"] = False
            report["receipt"] = child(
                "deliberate_process_error",
                [sys.executable, "-c", "raise SystemExit(7)"],
                logs / "failure",
                deadline=10,
            )
            return report
        return original(candidate, logs, proof)

    monkeypatch.setattr(r, "audit", failing_once)
    r.publish(value, output, raw)
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert (raw / "rejected_candidate.json").is_file()
    assert e.replay(output)
    monkeypatch.setattr(
        r, "publish_primary", lambda *a: (_ for _ in ()).throw(ValueError("conflicting_primary"))
    )
    with pytest.raises(ValueError, match="conflicting_primary"):
        r.publish(value, output, raw)


def test_fail_closed_reports(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8318-FINDINGS: subprocess and absent report failures are owned."""
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, dict(dense_sparse_error_max=0.0))
    stdout = tmp_path / "stdout"
    stdout.write_text("invalid report")
    monkeypatch.setattr(
        r, "child", lambda *a, **k: dict(stdout_path=str(stdout), exit_code=2, timed_out=False)
    )
    assert not r.audit(candidate, tmp_path, {})["passed"]
    assert h.first_difference([1, 2], [1, 3])["field"] == "[1]"
    assert h.first_difference({"a": 1}, {"b": 1})["field"] == ""


def test_terminal_and_shard_negatives(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8318-AUTHORITY: stale terminal hashes and labels fail closed."""
    path = tmp_path / "experiment_8305_private.json"
    side = tmp_path / "terminal.json"
    value = dict(
        experiment_id=8305,
        task_id="exp8305-private",
        honest_verdict="complete_null_private",
        verdict_class="null",
        terminal_validation_sidecar_path=str(side),
    )
    atomic_json(path, value)
    atomic_json(side, dict(publication=dict(primary_sha256="changed")))
    monkeypatch.setattr(e.base, "bind", lambda *a, **k: value)
    failures = []
    assert e.authenticate(path, tmp_path / "raw", [], failures) == {}
    assert failures[0]["artifact_field"] == "authenticated_terminal_bytes"
    source = json.loads((e.ROOT / h.CUSTODY).read_bytes())
    targets = json.loads(Path(source["evaluator_shards"]["fit"]["path"]).read_bytes())
    targets["rows"][0]["unit_id"] = "changed"
    atomic_json(tmp_path / "targets.json", targets)
    source["evaluator_shards"]["fit"] = dict(
        path=str(tmp_path / "targets.json"), sha256=sha256_file(tmp_path / "targets.json")
    )
    with pytest.raises(ValueError, match="predictor_target_identity"):
        h.support(source, tmp_path / "shards", [])


def test_staging_digest_and_cli_contract(tmp_path):
    """SCENARIO-REPORT-8318-AUTHORITY: full prompt drift fails a real child."""
    root = tmp_path / "root"
    for name in [e.DESIGN, e.ACTIVE, h.OLD_DESIGN]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    (root / e.STAGED).write_bytes((root / e.ACTIVE).read_bytes())
    assert e.authority(root, tmp_path / "staged")["activated"]
    plan = yaml.safe_load((root / e.ACTIVE).read_text())
    plan["tasks"][0]["prompt"] += " changed"
    (root / e.ACTIVE).write_text(yaml.safe_dump(plan))
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--root",
            str(root),
            "--inspect-milestone",
            e.MILESTONE,
            "--check-authority",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    (root / e.DESIGN).write_text(
        (root / e.DESIGN).read_text().replace("Canonical tasks SHA-256: `", "Changed digest: `")
    )
    with pytest.raises(ValueError, match="complete_task_digest"):
        e.authority(root, tmp_path / "bad_digest")
    assert (
        r.main(["--root", str(root), "--inspect-milestone", e.MILESTONE, "--check-authority"]) == 1
    )
    assert r.main(["--inspect-milestone", e.MILESTONE, "--check-authority"]) == 0


def test_scratch_resources_and_coverage_receipt(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8318-CLI: scratch failures block before owned execution."""
    from types import SimpleNamespace

    original = r.manifest

    def coverage_manifest(private):
        atomic_json(private / "coverage.json", dict(totals=dict(percent_covered=100)))
        return original(private)

    monkeypatch.setattr(r, "manifest", coverage_manifest)
    monkeypatch.setattr(r.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    output = tmp_path / (e.NAME + ".json")
    assert (
        r.main(["--root", str(tmp_path / "missing"), "--output", str(output), "--private-fixture"])
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["owned_coverage_reference"]
    reader = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: b"wrong" if p.name == "probe" else reader(p))
    with pytest.raises(ValueError, match="private_scratch"):
        r.main(["--root", str(tmp_path / "missing"), "--output", str(output), "--private-fixture"])


def test_cannot_erase_bound_source_list(tmp_path):
    """SCENARIO-REPORT-8318-REPLAY: a rehashed outer header cannot erase custody."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path / "missing", raw)
    atomic_json(raw / "measurement.json", work)
    candidate = tmp_path / "erased.json"
    value = e.build(work, [], raw, tmp_path / (e.NAME + ".json"))
    value["source_artifact_hashes"] = []
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(candidate, value)
    assert not e.replay(candidate)
