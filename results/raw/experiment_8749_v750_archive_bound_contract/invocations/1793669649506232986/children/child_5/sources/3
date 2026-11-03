"""REQ-REPORT-8749 / REQ-VERIFY-8749: bind authority without rewriting history."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v750_archive_bound as a
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def test_contract_and_full_object_controls(tmp_path):
    """SCENARIO-REPORT-8749-ARCHIVE: every task field binds to independent bytes."""
    result = a.authority(a.ROOT)
    assert result["ready"] and len(result["rows"]) == 14
    assert a.tasks_digest([r["task_object"] for r in result["rows"]]) == a.PIN
    controls = a.mutations(tmp_path)
    assert len(controls) >= 14 * 16 + 8
    assert all(r["passed"] for r in controls)
    assert not a.authority(tmp_path / "absent")["ready"]


def test_archived_root_and_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8749-ARCHIVE: old assertions use complete archived dependencies."""
    from carnot.reporting import v748_collection_contract as historical

    root = a.archived_root(tmp_path / "archive")
    assert "V750_TASK_CONTRACT_START" in (a.ROOT / a.DESIGN).read_text()
    with monkeypatch.context() as patch:
        patch.setattr(historical.m, "ROOT", root)
        assert historical.m.PIN == a.V748_PIN
        assert historical.m.authority(root)["ready"]
        original = (root / historical.m.ACTIVE).read_bytes()
        for field in ["prompt", "gated_on", "routing", "deliverable"]:
            value = yaml.safe_load(original)
            value["tasks"][0][field] = "forged"
            (root / historical.m.ACTIVE).write_text(yaml.safe_dump(value))
            assert not historical.m.authority(root)["ready"]
        (root / historical.m.ACTIVE).write_bytes(original)
        path = root / historical.m.DESIGN
        path.write_text(path.read_text().replace(a.V748_PIN, "0" * 64))
        assert not historical.m.authority(root)["ready"]
    assert not historical.m.authority(a.ROOT)["ready"]


def test_owned_failures_and_branch_custody():
    """SCENARIO-REPORT-8749-SCOPE: unrelated health red cannot replace owned checks."""
    good = dict(receipts=[dict(passed=True)], coverage=dict(complete=True), union=dict(passed=True))
    assert a.scoped_ready(good)
    for key in ["receipts", "coverage", "union"]:
        bad = deepcopy(good)
        if key == "receipts":
            bad[key][0]["passed"] = False
        else:
            bad[key][next(iter(bad[key]))] = False
        assert not a.scoped_ready(bad)
    assert not a.scoped_ready(
        dict(receipts=[], coverage=dict(complete=True), union=dict(passed=True))
    )
    learning = a.freeze_learning()
    assert learning["ready"] and learning["original_node_count"] == 64
    assert len(learning["original_runner_node_ids"]) == 27
    assert learning["statement_anchor_count"] == 83


def test_private_cli_and_sealed_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8749-SEAL: actual CLI children reject tampered terminal bytes."""
    from carnot.reporting import v750_archive_runner as runner

    out = tmp_path / (a.NAME + ".json")
    argv = [sys.executable, str(a.ROOT / a.CLI)]
    for name, args, expected in [
        ("private", ["--private-control", "--output", str(out)], 0),
        ("replay", ["--cold-replay", str(out)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "missing")], 1),
        ("bad_date", ["--date", "20260101"], 2),
        ("unsafe", ["--private-control"], 2),
    ]:
        receipt = child(name, argv + args, tmp_path / "logs", expected=expected, deadline=120)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(out.read_bytes())
    assert value["contract_ready_score"] == 0 and value["verdict_class"] == "disqualified"
    assert runner.replay(out)
    raw = Path(value["primitive_reference"]["path"]).parent
    seal = raw / "independent_seal.json"
    saved = seal.read_bytes()
    seal.write_text("{}")
    assert not runner.replay(out)
    seal.write_bytes(saved)
    reference = a.reference
    with monkeypatch.context() as patch:
        patch.setattr(a, "reference", lambda path: dict(reference(path), sha256="forged"))
        assert not runner.replay(out)
    with monkeypatch.context() as patch:
        patch.setattr(a, "authority", lambda root: {})
        assert not runner.replay(out)
    work = a.load(value["primitive_reference"])
    validation = a.load(value["validation_reference"])
    original_work = Path(value["primitive_reference"]["path"]).read_bytes()
    changed_work = deepcopy(work)
    changed_work["learning"]["ready"] = False
    atomic_json(Path(value["primitive_reference"]["path"]), changed_work)
    changed_value = dict(
        value, primitive_reference=a.reference(Path(value["primitive_reference"]["path"]))
    )
    atomic_json(
        seal,
        {
            k: changed_value[k]
            for k in ["primitive_reference", "validation_reference", "frozen_input_reference"]
        },
    )
    atomic_json(out, changed_value)
    assert not runner.replay(out)
    Path(value["primitive_reference"]["path"]).write_bytes(original_work)
    seal.write_bytes(saved)
    atomic_json(out, value)
    for field in ["contract", "learning"]:
        changed = deepcopy(work)
        changed[field]["ready"] = False
        result = runner.build(changed, validation, out)
        assert result["verdict_class"] == "blocked"
        assert result["gate_check_summary"]
    qualified = deepcopy(work)
    qualified.update(
        private_control=False,
        historical_fixture_rows=[dict(passed=True)] * 3,
        mutation_rows=[dict(passed=True)],
        owned_failure_controls=[dict(passed=True)] * 2,
    )
    checks = dict(
        validation,
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True),
    )
    result = runner.build(qualified, checks, out)
    assert result["contract_ready_score"] == result["archived_authority_ready_score"] == 1
    assert result["learning_protocol_ready_score"] == 1
    qualified["historical_fixture_rows"][0]["passed"] = False
    assert runner.build(qualified, checks, out)["contract_ready_score"] == 0
    value["full_task_objects"][0]["prompt"] += "forged"
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(out, value)
    assert not runner.replay(out)
    assert child("tamper", argv + ["--cold-replay", str(out)], tmp_path / "logs", expected=1)[
        "passed"
    ]


def test_real_coverage_reader_and_empty_union(tmp_path, monkeypatch):
    """REQ-VERIFY-8749: missing executable statements and missing nodes cannot qualify."""
    from coverage import Coverage
    from types import SimpleNamespace
    from carnot.reporting import v750_archive_runner as runner

    measured = Coverage(data_file=str(tmp_path / "producer.sqlite"), config_file=False)
    measured.start()
    assert a.tasks_digest([])
    measured.stop()
    measured.save()
    atomic_json(
        tmp_path / "child.json", dict(sqlite_references=[a.reference(tmp_path / "producer.sqlite")])
    )
    summary = runner.coverage_summary(tmp_path, [a.reference(tmp_path / "child.json")])
    assert not summary["complete"] and any(r["missing"] for r in summary["files"])

    def rejected_inventory(private, raw, suites, owned, child):
        assert all(not Path(suite).is_absolute() for suite in suites)
        return (
            [],
            dict(
                passed=False,
                stdout_path=str(tmp_path / "stdout"),
                stdout_sha256="none",
                stderr_path=str(tmp_path / "stderr"),
                stderr_sha256="none",
            ),
        )

    monkeypatch.setattr(runner, "collect", rejected_inventory)
    monkeypatch.setattr(
        runner,
        "child",
        lambda *args, **kwargs: dict(
            passed=True,
            stdout_path=str(tmp_path / "stdout"),
            stdout_sha256="none",
            stderr_path=str(tmp_path / "stderr"),
            stderr_sha256="none",
        ),
    )
    monkeypatch.setattr(runner, "coverage_summary", lambda *args: dict(complete=False))
    result = runner.validate(
        tmp_path,
        SimpleNamespace(stop=lambda: None, save=lambda: None),
        [str(tmp_path / "original.py")],
    )
    assert not a.scoped_ready(result)


def test_historical_cli_binding(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8749-ARCHIVE: subprocess entry uses the same archived dependency root."""
    from carnot.reporting import v748_collection_contract as historical

    with monkeypatch.context() as patch:
        patch.setattr(historical.m, "ROOT", historical.m.ROOT)
        patch.setattr(historical.m, "CLI", historical.m.CLI)
        patch.setattr(historical, "main", lambda: 7)
        assert a.historical_main(tmp_path) == 7
        assert historical.m.ROOT == tmp_path
        assert Path(historical.m.CLI).is_absolute()


def test_missing_learning_custody_and_optional_health(tmp_path, monkeypatch):
    """REQ-REPORT-8749: unavailable source custody blocks only the learning branch."""
    from carnot.reporting import v750_archive_runner as runner

    def absent():
        raise FileNotFoundError("original learning operand absent")

    monkeypatch.setattr(a, "freeze_learning", absent)
    is_file = Path.is_file
    health = a.ROOT / "results/raw" / a.NAME / "startup/full_python_suite.receipt.json"
    monkeypatch.setattr(Path, "is_file", lambda path: False if path == health else is_file(path))
    work = a.freeze(tmp_path, a.ROOT)
    assert not work["learning"]["ready"]
    assert (
        work["learning"]["gate_check_summary"][0]["diagnostic"]
        == "original learning operand absent"
    )
    assert work["repository_health_reference"]["current_invocation_diagnostic"] is None


def test_production_dispatch_preserves_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8749-SCOPE: the production dispatcher cannot hide owned failures."""
    from types import SimpleNamespace
    from datetime import datetime, UTC
    from carnot.reporting import v750_archive_runner as runner

    monkeypatch.setattr(runner, "Coverage", lambda **kwargs: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(a, "mutations", lambda root: [dict(passed=True)])
    monkeypatch.setattr(
        runner,
        "validate",
        lambda *args: dict(
            receipts=[dict(passed=False)] * 8,
            union=dict(passed=False),
            coverage=dict(complete=False),
            raw_references=[],
            ended_at_utc=datetime.now(UTC).isoformat(),
        ),
    )
    out = tmp_path / (a.NAME + ".json")
    assert runner.main(["--output", str(out)]) == 0
    value = json.loads(out.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["contract_ready_score"] == value["learning_protocol_ready_score"] == 0
