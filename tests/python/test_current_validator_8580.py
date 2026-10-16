"""REQ-REPORT-8580 and REQ-VERIFY-8580: qualify exact current bytes only."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import current_validator_8580 as e
from carnot.reporting import current_validator_runner_8580 as runner
from carnot.reporting.current_work_receipt import atomic_json


def test_repair_and_original_authority(tmp_path):
    """SCENARIO-REPORT-8580-REVISION: reverse only the documented import repair."""
    authority = e.authenticate()
    assert len(authority["regression_nodes"]) == 112
    assert len(authority["affected_nodes"]) == 161
    assert set(authority["changed_paths"]) == set(e.REPAIRED)
    assert len(authority["corpus"]["cases"]) == 17
    assert authority["rejected_revision"]["experiment_id"] == 8568
    work = runner.prepare(tmp_path)
    assert len(work["controls"]) == 17
    assert all(r["expectation_met"] for r in work["controls"])
    assert all(r["unchanged"] for r in work["preserved"])
    assert work["preconditions"]["passed"]


def test_private_cli_replay_and_mutations(tmp_path):
    """SCENARIO-VERIFY-8580-REPLAY: real children reject rehashed invented evidence."""
    output = tmp_path / (e.NAME + ".json")

    def cli(*args):
        return subprocess.run(
            [sys.executable, str(runner.CLI), *map(str, args)],
            capture_output=True,
            text=True,
            timeout=300,
        )

    result = cli("--date", "20261016", "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["current_validator_ready_score"] == 0
    assert cli("--cold-replay", output).returncode == 0
    for field, bad in [
        ("rows", []),
        ("current_validator_ready_score", 1),
        ("required_checks_passed", True),
        ("cli_import_classification_rows", []),
    ]:
        changed = dict(value, **{field: bad})
        changed["reproducibility_checksum"] = e.checksum(changed)
        atomic_json(output, changed)
        assert cli("--cold-replay", output).returncode == 1
    atomic_json(output, value)
    work_path = Path(value["primitive_reference"]["path"])
    work = json.loads(work_path.read_bytes())
    changed = deepcopy(work)
    changed["controls"][0]["new_findings"] = []
    runner.seal(changed)
    atomic_json(output, e.build(changed, output))
    assert cli("--cold-replay", output).returncode == 1
    runner.seal(work)
    changed = deepcopy(work)
    changed["cli_rows"][0]["imported_findings"] = []
    runner.seal(changed)
    atomic_json(output, e.build(changed, output))
    assert not e.replay(output)
    runner.seal(work)
    atomic_json(output, value)
    work_path.unlink()
    assert cli("--cold-replay", output).returncode == 1
    assert cli("--cold-replay", tmp_path / "absent").returncode == 1
    assert cli("--date", "wrong").returncode == 2
    assert cli("--fixture-output", e.ROOT / "results" / (e.NAME + ".json")).returncode == 2


def test_failed_owned_and_missing_input(tmp_path, monkeypatch):
    """REQ-VERIFY-8580: owned failure disqualifies and absent authority stays unknown."""
    work = runner.prepare(tmp_path / "raw")
    work["receipts"] = [dict(name="failed", passed=False)]
    assert e.build(work, tmp_path / "candidate")["verdict_class"] == "disqualified"
    with monkeypatch.context() as patch:
        patch.setattr(e, "authenticate", lambda: (_ for _ in ()).throw(FileNotFoundError()))
        missing = runner.prepare(tmp_path / "missing")
        assert missing["failures"][0]["observed"] is None
        assert e.build(missing, tmp_path / "candidate")["verdict_class"] == "blocked"
    assert e.xml_nodes(e.AUTHORITY / "regression.xml") == work["authority"]["regression_nodes"]


def test_explicit_accounting_revision_and_drift(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8580-REVISION: revision arguments cannot silently bless old hashes."""
    source = tmp_path / "source"
    source.write_text("original")
    work = runner.prepare(tmp_path / "raw")
    work["code"] = [e.reference(source)]
    monkeypatch.setattr(e, "gates", lambda work: dict(checked=True))
    monkeypatch.setattr(e, "valid_work", lambda work: True)
    runner.seal(work)
    revision = e.revision(work)
    path = tmp_path / "revision.json"
    atomic_json(path, revision)
    ref = e.reference(path)
    assert e.accept_revision(ref) == revision
    source.write_text("drift")
    with pytest.raises(ValueError, match="revision_drift"):
        e.accept_revision(ref)
    revision["ready"] = False
    atomic_json(path, revision)
    with pytest.raises(ValueError, match="revision_not_ready"):
        e.accept_revision(e.reference(path))
    with pytest.raises((ValueError, FileNotFoundError)):
        e.accept_revision(ref)


def test_controller_and_exact_node_union(tmp_path, monkeypatch):
    """REQ-VERIFY-8580: planning never substitutes a count for executed node identities."""
    from carnot.reporting import current_validator_validation_8580 as validation

    work = runner.prepare(tmp_path / "raw")
    planned = []

    def controlled(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        out, err = logs / (name + ".stdout"), logs / (name + ".stderr")
        nodes = [
            *work["authority"]["regression_nodes"],
            *work["authority"]["affected_nodes"],
            e.TESTS[0] + "::test_unit",
        ]
        out.write_text("\n".join(nodes) if name == "collection" else "")
        err.write_text("")
        junit = next((a.split("=", 1)[1] for a in argv if a.startswith("--junitxml=")), None)
        if junit:
            import xml.etree.ElementTree as ET

            root = ET.Element("testsuites")
            for node in [a for a in argv if "::" in a]:
                parts = node.split("::")
                ET.SubElement(
                    root,
                    "testcase",
                    classname=".".join([parts[0][:-3].replace("/", "."), *parts[1:-1]]),
                    name=parts[-1],
                )
            ET.ElementTree(root).write(junit)
        if name == "coverage":
            atomic_json(
                Path(work["raw"]) / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(missing_lines=0), missing_lines=[]) for p in e.OWNED
                    }
                ),
            )
        row = dict(
            name=name,
            passed=True,
            log_path=str(out),
            stdout_path=str(out),
            stderr_path=str(err),
            stdout_sha256=e.reference(out)["sha256"],
            stderr_sha256=e.reference(err)["sha256"],
        )
        atomic_json(out.with_suffix(".receipt.json"), row)
        planned.append(name)
        return row

    monkeypatch.setattr(validation, "child", controlled)
    monkeypatch.setattr(validation, "replay_controls", lambda *a: None)
    monkeypatch.setattr(validation, "private_checks", lambda *a: None)
    validation.validate(work, tmp_path / "scratch")
    assert work["union_passed"] and work["coverage_complete"]
    assert {"owned", "regression", "affected", "ruff_check", "mypy", "spec"} <= set(planned)
    assert validation.union_valid(work)
    assert e.valid_work(work)
    changed = deepcopy(work)
    changed["node_union"]["original_node_ids"].pop()
    assert not validation.union_valid(changed)
    assert not e.valid_work(changed)
    monkeypatch.setattr(validation, "validate", lambda *a: None)
    monkeypatch.setattr(validation, "replay_controls", lambda *a: None)
    monkeypatch.setattr(runner, "terminal", lambda *a: dict(passed=True))
    assert runner.run(tmp_path / "production" / (e.NAME + ".json")) == 0
    monkeypatch.setattr(runner, "run", lambda *a, **k: 0)
    assert runner.main([]) == 0


def test_authentication_rejections(tmp_path, monkeypatch):
    """REQ-REPORT-8580: unexplained changes, missing repairs and old-hash drift fail."""
    authority = e.authenticate()
    original_reference = e.reference
    for mode in ("unexplained", "missing", "receipt", "inverse"):
        with monkeypatch.context() as patch:

            def ref(path):
                actual = original_reference(path)
                if mode == "unexplained" and path.name == "discrete_count_proof.py":
                    return dict(actual, sha256="wrong")
                if mode == "missing" and path.name == "adversarial_verify.py":
                    return next(
                        r
                        for r in authority["previous"]["code_config_hashes"]
                        if r["path"] == str(path)
                    )
                if mode == "receipt" and path.name == "regression.xml":
                    return dict(actual, sha256="wrong")
                return actual

            patch.setattr(e, "reference", ref)
            if mode == "inverse":
                patch.setattr(e, "reverse_repair", lambda p: "changed assertions")
            with pytest.raises(ValueError):
                e.authenticate()


def test_private_replay_commands_and_log_tampering(tmp_path):
    """SCENARIO-VERIFY-8580-REPLAY: command receipts bind genuine child logs."""
    from carnot.reporting import current_validator_validation_8580 as validation
    from carnot.reporting.v709_execution import child

    work = runner.prepare(tmp_path / "raw")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def command(name, argv, **kwargs):
        receipt = child(name, argv, tmp_path / "logs", deadline=60, **kwargs)
        work["receipts"].append(receipt)
        return receipt

    validation.private_checks(work, scratch, command)
    assert all(r["passed"] for r in work["receipts"])
    runner.seal(work)
    output = tmp_path / "candidate.json"
    atomic_json(output, e.build(work, output))
    assert e.replay(output)
    for field in ("code", "sources"):
        changed = deepcopy(work)
        changed[field][0]["sha256"] = "wrong"
        runner.seal(changed)
        atomic_json(output, e.build(changed, output))
        assert not e.replay(output)
    changed = deepcopy(work)
    changed["receipts"][0]["duration_s"] += 1
    runner.seal(changed)
    atomic_json(output, e.build(changed, output))
    assert not e.replay(output)
    runner.seal(work)
    atomic_json(output, e.build(work, output))
    Path(work["receipts"][0]["stdout_path"]).write_text("tampered")
    assert not e.replay(output)


def test_actual_accounting_adapter(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8580-REVISION: fresh accounting uses an explicit current argument."""
    monkeypatch.setattr(e, "accept_revision", lambda ref: dict(ready=True))
    row = e.accounting_revision(current_validator_reference=dict(path="unit", sha256="unit"))
    assert row["passed"]
    assert row["metrics"]["applied_update_events"] == 1800
    assert all(r["severity"] == "warn" for r in row["findings"])


def test_rehashed_primitive_and_revision_evidence(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8580-REPLAY: resealed envelopes cannot replace measured operands."""
    work = runner.prepare(tmp_path / "raw")
    output = tmp_path / "candidate.json"
    for field in ("controls", "preserved", "authority"):
        changed = deepcopy(work)
        changed[field] = [] if field != "authority" else {}
        runner.seal(changed)
        atomic_json(output, e.build(changed, output))
        assert not e.replay(output)
    runner.seal(work)
    revision_path = Path(work["raw"]) / "current_validator_revision.json"
    changed = e.revision(work)
    changed["ready"] = True
    atomic_json(revision_path, changed)
    with pytest.raises(ValueError, match="revision_evidence"):
        e.accept_revision(e.reference(revision_path))
    ready = deepcopy(work)
    ready.update(
        failures=[],
        coverage_complete=True,
        union_passed=True,
        cli_rows=[dict(passed=True)] * 17,
        receipts=[dict(name=n, passed=True) for n in e.REQUIRED],
    )
    runner.seal(ready)
    assert e.build(ready, output)["current_validator_ready_score"] == 1


def test_original_diagnostics_cannot_be_replaced(tmp_path, monkeypatch):
    """REQ-REPORT-8580: new qualification preserves the exact original diagnoses."""
    work = runner.prepare(tmp_path / "raw")
    authority = dict(
        work["authority"],
        previous=dict(work["authority"]["previous"], old_new_classification_rows=[]),
    )
    changed = dict(work, authority=authority)
    monkeypatch.setattr(e, "authenticate", lambda: authority)
    monkeypatch.setattr(e.old, "classifications", lambda corpus: work["controls"])
    monkeypatch.setattr(e.old, "preserved", lambda upstream: work["preserved"])
    assert not e.valid_work(changed)
