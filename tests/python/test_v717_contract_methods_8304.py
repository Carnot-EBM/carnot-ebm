"""REQ-REPORT-8304 and REQ-VERIFY-8304: private authority and real children."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v717_contract_methods as m
from carnot.reporting import v717_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.roadmap_contract import parse_design


@pytest.fixture
def private_root(tmp_path):
    """SCENARIO-REPORT-8304-AUTHORITY: fixture authority stays outside the checkout."""
    root = tmp_path / "repo"
    design = root / m.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes((m.ROOT / m.DESIGN).read_bytes())
    tasks = parse_design(design.read_text(), milestone=m.MILESTONE)[1]
    value = dict(milestone=m.MILESTONE, tasks=tasks)
    for name in (m.ACTIVE, m.STAGED):
        (root / name).write_text(yaml.safe_dump(value))
    shutil.copyfile(m.ROOT / m.PROTOCOL, root / m.PROTOCOL)
    upstream = root / "results/experiment_8290_fixture.json"
    terminal = upstream.parent / "raw" / upstream.stem / "terminal_validation.json"
    publication = publish_primary(
        upstream,
        dict(
            experiment_id=8290,
            task_id="exp8290-fixture",
            honest_verdict="complete_blocked_fixture",
            verdict_class="blocked",
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(terminal),
        ),
        lambda p: dict(passed=True),
    )
    atomic_json(terminal, dict(publication=publication))
    primary = root / m.HISTORY
    publish_primary(
        primary,
        dict(
            experiment_id=8303,
            task_id="exp8303-capstone",
            honest_verdict="complete_fixture",
            verdict_class="circular_positive",
            flagged_adversarial=False,
            task_dispositions=[
                dict(
                    task_id="exp8290-fixture",
                    path=str(upstream),
                    sha256=sha256_file(upstream),
                    producer_executed=True,
                    honest_verdict="complete_blocked_fixture",
                    verdict_class="blocked",
                ),
                dict(
                    task_id="exp8293-absent",
                    path=str(root / "results/experiment_8293_absent.json"),
                    producer_executed=False,
                    honest_verdict=None,
                    verdict_class=None,
                ),
                dict(task_id="exp8303-capstone", path=None, producer_executed=True),
            ],
        ),
        lambda p: dict(passed=True),
    )
    return root


def test_authority_and_protocol(private_root, tmp_path):
    """SCENARIO-REPORT-8304-AUTHORITY: consumed staging and exact prompt drift."""
    work = m.measure(private_root, tmp_path / "raw")
    assert work["contract"]["activated"] and not work["failures"]
    (private_root / m.STAGED).unlink()
    assert m.measure(private_root, tmp_path / "consumed")["contract"]["activated"]
    active = private_root / m.ACTIVE
    value = yaml.safe_load(active.read_text())
    value["tasks"][0]["prompt"] += " changed"
    active.write_text(yaml.safe_dump(value))
    assert not m.measure(private_root, tmp_path / "drift")["contract"]["activated"]
    (private_root / m.PROTOCOL).write_text("{}")
    assert any(
        g["artifact_field"] == "protocol_sha256"
        for g in m.measure(private_root, tmp_path / "protocol-drift")["failures"]
    )


def test_missing_and_corrupt_inputs(private_root, tmp_path):
    """SCENARIO-REPORT-8304-AUTHORITY: absent bytes differ from measured zero."""
    (private_root / m.DESIGN).write_text("broken")
    assert m.measure(private_root, tmp_path / "broken")["failures"]
    (private_root / m.HISTORY).unlink()
    work = m.measure(private_root, tmp_path / "absent")
    assert any(g["observed"] is None for g in work["failures"])


def test_readiness_and_rehashed_replay(private_root, tmp_path):
    """SCENARIO-REPORT-8304-REPLAY: replay derives readiness from primitive bytes."""
    raw = tmp_path / "raw"
    work = m.measure(private_root, raw)
    receipts = [dict(passed=True, exit_code=0, name="owned")]
    value = m.build(work, receipts, raw, tmp_path / "output.json")
    assert value["current_contract_ready_score"] == value["protocol_ready_score"] == 1
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert m.replay(candidate)
    for key, changed in [
        ("rows", []),
        ("current_contract_ready_score", 0),
        ("MODEL_SPECS", ["fake"]),
        ("protocol_ready_score", 0),
    ]:
        mutation = deepcopy(value)
        mutation[key] = changed
        mutation.pop("reproducibility_checksum")
        mutation["reproducibility_checksum"] = canonical_hash(mutation)
        atomic_json(candidate, mutation)
        assert not m.replay(candidate)
    atomic_json(candidate, dict(value, reproducibility_checksum="wrong"))
    assert not m.replay(candidate)
    assert (
        m.build(work, [dict(passed=False, exit_code=1)], raw, candidate)["verdict_class"]
        == "disqualified"
    )
    work["failures"].append(m.failure(tmp_path / "absent", "operand", True, None))
    assert m.build(work, receipts, raw, candidate)["verdict_class"] == "blocked"
    candidate.write_text("broken")
    assert not m.replay(candidate)


def test_cli_private_recovery(private_root, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8304-CHECKS: real CLI, validator failure and safe recovery."""
    monkeypatch.setattr(
        r,
        "manifest",
        lambda p: [
            dict(
                name="real-child",
                argv=[sys.executable, "-c", "print('actual child')"],
                expected=0,
                deadline=10,
                scope="owned",
            )
        ],
    )
    output = tmp_path / "results" / (m.NAME + ".json")
    args = ["--root", str(private_root), "--output", str(output)]
    assert r.main(args) == 0
    assert m.replay(output)
    assert r.main(["--cold-replay", str(output)]) == 0
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    with pytest.raises(SystemExit):
        r.main(["--date", "20261009"])
    with pytest.raises(SystemExit):
        r.main(["--private-fixture", "--output", str(m.ROOT / "results" / (m.NAME + ".json"))])


def test_terminal_corruption_and_method_failure(private_root, tmp_path):
    """SCENARIO-REPORT-8304-AUTHORITY: wrong byte-bound sidecars and absent methods block."""
    upstream = private_root / "results/experiment_8290_fixture.json"
    upstream.write_text(upstream.read_text() + " ")
    work = m.measure(private_root, tmp_path / "changed")
    assert any(g["artifact_field"] == "capstone_disposition_sha256" for g in work["failures"])
    assert any(g["artifact_field"] == "authenticated_terminal" for g in work["failures"])
    p = private_root / m.PROTOCOL
    protocol = json.loads(p.read_text())
    protocol["method_mapping"][0]["path"] = str(tmp_path / "missing-method")
    atomic_json(p, protocol)
    assert any(
        g["artifact_field"] == "sha256"
        for g in m.measure(private_root, tmp_path / "method")["failures"]
    )
    d = private_root / m.DESIGN
    d.write_text(d.read_text().replace("Canonical tasks SHA-256", "Absent tasks SHA-256"))
    with pytest.raises(ValueError, match="design_complete_task_digest"):
        m.authority([d, private_root / m.STAGED, private_root / m.ACTIVE], tmp_path / "digest")


def _candidate(work, raw, target):
    """SCENARIO-REPORT-8304-REPLAY: keep private rehash operands internally consistent."""
    value = m.build(work, [dict(name="private", passed=True, exit_code=0)], raw, target)
    atomic_json(target, value)
    return value


def test_rehashed_work_and_streams(private_root, tmp_path):
    """SCENARIO-REPORT-8304-REPLAY: check primitive reduction beyond the outer checksum."""
    raw, target = tmp_path / "raw", tmp_path / "candidate.json"
    work = m.measure(private_root, raw)
    original = deepcopy(work)
    work["contract"]["activated"] = False
    _candidate(work, raw, target)
    assert not m.replay(target)
    work = deepcopy(original)
    work["protocol"]["MODEL_SPECS"] = ["forged"]
    _candidate(work, raw, target)
    assert not m.replay(target)
    work = deepcopy(original)
    work["protocol_sha256"] = "forged"
    _candidate(work, raw, target)
    assert not m.replay(target)
    value = _candidate(original, raw, target)
    log = tmp_path / "stdout"
    log.write_text("real output")
    value["validation_receipts"][0].update(stdout_path=str(log), stdout_sha256=sha256_file(log))
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(target, value)
    assert m.replay(target)
    log.write_text("changed output")
    assert not m.replay(target)
    work = deepcopy(original)
    work["history"][0]["verdict_class"] = "positive"
    _candidate(work, raw, target)
    assert not m.replay(target)


def test_cannot_remove_authority_failure(private_root, tmp_path):
    """SCENARIO-REPORT-8304-REPLAY: rehashing cannot erase a failed external authority operand."""
    active = private_root / m.ACTIVE
    source = yaml.safe_load(active.read_text())
    source["tasks"][0]["prompt"] += " drift"
    active.write_text(yaml.safe_dump(source))
    raw = tmp_path / "raw"
    work = m.measure(private_root, raw)
    assert work["failures"]
    work["failures"] = []
    target = tmp_path / "candidate.json"
    _candidate(work, raw, target)
    assert not m.replay(target)


def test_blocked_replay_and_absent_protocol(private_root, tmp_path):
    """SCENARIO-REPORT-8304-REPLAY: honest external failure remains replayable."""
    (private_root / m.DESIGN).write_text("unreadable design")
    raw = tmp_path / "broken"
    target = tmp_path / "candidate.json"
    _candidate(m.measure(private_root, raw), raw, target)
    assert m.replay(target)
    (private_root / m.PROTOCOL).unlink()
    (private_root / m.DESIGN).unlink()
    raw = tmp_path / "absent"
    _candidate(m.measure(private_root, raw), raw, target)
    assert m.replay(target)


def test_execution_manifest_and_real_fixture_cli(private_root, tmp_path):
    """SCENARIO-VERIFY-8304-CHECKS: standalone CLI uses frozen private operands."""
    private = tmp_path / "scratch"
    private.mkdir()
    plan = r.manifest(private)
    assert any(c["name"] == "private_E2E018_consumers" for c in plan)
    assert all(c["deadline"] <= 240 for c in plan)
    output = tmp_path / "results" / (m.NAME + ".json")
    result = subprocess.run(
        [
            sys.executable,
            str(m.ROOT / m.CLI),
            "--private-fixture",
            "--root",
            str(private_root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert m.replay(output)


def test_publication_failure_recovers(private_root, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8304-CHECKS: an actual failed child clears readiness before recovery."""
    raw = tmp_path / "raw"
    output = tmp_path / "results" / (m.NAME + ".json")
    value = m.build(m.measure(private_root, raw), [dict(passed=True, exit_code=0)], raw, output)
    original = r.child
    attempted = []

    def fail_once(name, argv, logs, **kwargs):
        if not attempted:
            argv = [sys.executable, "-c", "raise SystemExit(1)"]
        attempted.append(name)
        return original(name, argv, logs, **kwargs)

    monkeypatch.setattr(r, "child", fail_once)
    r.publish(value, output, raw)
    result = json.loads(output.read_text())
    assert result["verdict_class"] == "disqualified"
    assert result["current_contract_ready_score"] == result["protocol_ready_score"] == 0
    assert m.replay(output)
    with pytest.raises(ValueError, match="producer_identity"):
        r.publish(value, tmp_path / "experiment_8305_wrong.json", raw)


def test_current_historical_binding(tmp_path):
    """SCENARIO-REPORT-8304-AUTHORITY: actual history supplies dispositions, never new observations."""
    work = m.measure(m.ROOT, tmp_path / "current")
    assert len(work["history"]) == 14
    assert any("7996" in ref["path"] for ref in work["refs"])
    absent = [r for r in work["history"] if r.get("path") and not Path(r["path"]).is_file()]
    assert len(absent) == 7
    assert all("honest_verdict" not in r and "verdict_class" not in r for r in absent)


def test_resource_check_and_coverage_copy(private_root, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8304-CHECKS: check missing binaries before children and preserve real coverage."""
    assert r.preflight([dict(argv=[str(tmp_path / "absent-tool")])])[0]["observed"] is None
    output = tmp_path / "results" / (m.NAME + ".json")
    result = subprocess.run(
        [
            sys.executable,
            str(m.ROOT / m.CLI),
            "--private-fixture",
            "--root",
            str(private_root),
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0

    def measured_plan(private):
        config = private / "coverage.ini"
        config.write_text("[run]\ndata_file=" + str(private / ".coverage") + "\n")
        source = private / "coverage_probe.py"
        source.write_text("print('measured coverage child')\n")
        return [
            dict(
                name="coverage-child",
                argv=[
                    str(m.ROOT / ".venv/bin/coverage"),
                    "run",
                    "--rcfile=" + str(config),
                    str(source),
                ],
                expected=0,
                deadline=30,
                scope="owned",
            ),
            dict(
                name="coverage-json",
                argv=[
                    str(m.ROOT / ".venv/bin/coverage"),
                    "json",
                    "--rcfile=" + str(config),
                    "-o",
                    str(private / "coverage.json"),
                ],
                expected=0,
                deadline=30,
                scope="owned",
            ),
        ]

    monkeypatch.setattr(r, "manifest", measured_plan)
    assert r.main(["--root", str(private_root), "--output", str(output)]) == 0
    assert list(output.parent.glob("raw/*/invocations/*/owned_coverage.json"))
    monkeypatch.setattr(
        r,
        "manifest",
        lambda p: [
            dict(
                name="missing-tool",
                argv=[str(tmp_path / "absent-tool")],
                expected=0,
                deadline=10,
                scope="owned",
            )
        ],
    )
    assert r.main(["--root", str(private_root), "--output", str(output)]) == 0
    blocked = json.loads(output.read_text())
    assert blocked["verdict_class"] == "blocked"
    assert blocked["protocol_ready_score"] == blocked["current_contract_ready_score"] == 0
