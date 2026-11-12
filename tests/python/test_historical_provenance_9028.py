"""REQ-REPORT-9028-HISTORY: immutable test inputs preserve strict historical gates."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v767_branch_contract as authority
from carnot.reporting import v767_capstone as capstone
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

ARCHIVE = "openspec/change-proposals/research-roadmap-v767-preserved-20261110.md"
QUEUE = "openspec/change-proposals/research-roadmap-v767-queue-preserved-20261110.yaml"


def tree(root):
    """Copy historical bytes into private paths so tests cannot activate a queue."""
    for source, target in [(ARCHIVE, authority.DESIGN), (QUEUE, authority.ACTIVE)]:
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((authority.ROOT / source).read_bytes())
    return root


def child(tmp_path, cli, *args):
    """Exercise installed CLI paths outside the checkout without ambient imports."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(authority.ROOT / cli), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        command[1:2] = ["-m", "coverage", "run"]
    print("[exp9028-test] before subprocess", args, flush=True)
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    print(
        "[exp9028-test] after subprocess",
        result.returncode,
        result.stdout[-600:],
        result.stderr[-600:],
        flush=True,
    )
    return result


def test_explicit_historical_reader_and_cli(tmp_path):
    """SCENARIO-REPORT-9028-AUTHORITY: the real V767 reader accepts exact historical bytes."""
    root = tree(tmp_path / "authority")
    contract = authority.authority(root)
    assert contract["ready"] and canonical_hash(contract["tasks"]) == authority.PIN
    work = capstone.collect(tmp_path / "raw", "20261110", authority_root=root)
    assert work["tasks"] == contract["tasks"]
    output = tmp_path / (capstone.NAME + ".json")
    result = child(tmp_path, capstone.CLI, "--authority-root", root, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert child(tmp_path, capstone.CLI, "--cold-replay", output).returncode == 0


@pytest.mark.parametrize("mutation", ["missing", "order", "prompt", "rehash", "successor"])
def test_original_reader_and_child_refuse_changed_authority(tmp_path, mutation):
    """SCENARIO-REPORT-9028-AUTHORITY: private coordinated edits cannot change the independent pin."""
    source = tree(tmp_path / "source")
    authority.mutations(source, tmp_path / "mutations")
    if mutation == "successor":
        root = tmp_path / "successor"
        for original, target in [
            (ARCHIVE.replace("v767", "v770").replace("20261110", "20261111"), authority.DESIGN),
            (QUEUE.replace("v767", "v770").replace("20261110", "20261111"), authority.ACTIVE),
        ]:
            path = root / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((authority.ROOT / original).read_bytes())
    else:
        root = tmp_path / "mutations" / mutation
    assert not authority.authority(root)["ready"]
    output = tmp_path / (capstone.NAME + ".json")
    result = child(tmp_path, capstone.CLI, "--authority-root", root, "--output", output)
    assert result.returncode == 1
    assert "task_authority" in result.stdout
    assert not output.exists()


def test_independent_digest_and_missing_root(tmp_path, monkeypatch):
    """REQ-REPORT-9028-HISTORY: even matching design and queue need the independent digest."""
    root = tree(tmp_path / "authority")
    monkeypatch.setattr(authority, "PIN", "sha256:wrong")
    assert not authority.authority(root)["ready"]
    with pytest.raises(ValueError, match="task_authority"):
        capstone.collect(tmp_path / "raw", "20261110", authority_root=root)
    with pytest.raises(ValueError, match="task_authority"):
        capstone.collect(tmp_path / "raw", "20261110", authority_root=tmp_path / "absent")


def work_fixture(tmp_path):
    """Administrative fixture receipts have no live or historical science credit."""
    from carnot.reporting import historical_provenance_9028 as d
    from carnot.reporting import scoped_provenance_9014 as old
    from test_scoped_provenance_9014 import work_fixture as original_fixture

    work = original_fixture(tmp_path)
    work["validation_scope_manifest"] = d.scope_manifest(tmp_path / "private")
    for receipt, command in zip(
        work["validation"]["receipts"], work["validation_scope_manifest"]["commands"]
    ):
        receipt.update(
            command_argv=command["argv"],
            scope_sha256=canonical_hash(work["validation_scope_manifest"]),
        )
    work["current_receipt_mutation_rows"] = []
    for name in ("valid", *d.RECEIPT_CONTROLS):
        row = deepcopy(work["controls"][0])
        code = int(name != "valid")
        row.update(name=name, expected_exit=code, observed_exit=code)
        row["receipt"]["exit_code"] = code
        work["current_receipt_mutation_rows"].append(row)
    work["historical_authority_rows"] = []
    for name in d.AUTHORITY_CONTROLS:
        row = deepcopy(work["controls"][0])
        code = int(name != "valid")
        row.update(name="authority_" + name, expected_exit=code, observed_exit=code)
        row["receipt"]["exit_code"] = code
        work["historical_authority_rows"].append(row)
    work["baseline_reference"] = d.reference(d.BASELINE)
    assert old.CHECKS == d.CHECKS
    return work


def test_frozen_scope_and_history(tmp_path):
    """REQ-REPORT-9028-QUALIFICATION: original failing nodes and failed history remain mandatory."""
    from carnot.reporting import historical_provenance_9028 as d

    manifest = d.scope_manifest(tmp_path / "scope")
    assert capstone.TEST in manifest["direct_consumer_tests"]
    assert len(manifest["original_v767_test_node_hashes"]) == 8
    assert manifest["original_v767_test_nodes_preserved"]
    assert manifest["statement_percent"] == 100 and manifest["excluded_statements"] == 0
    assert manifest["receipt_controls"] == ["valid", *d.RECEIPT_CONTROLS]
    evidence, health = d.authenticate(d.ROOT, tmp_path / "raw")
    assert not evidence["failures"]
    assert len(health["rows"]) == 4 and not health["passed"]
    assert all(r["original_verdict_class"] == "disqualified" for r in health["rows"])
    blocked, _ = d.authenticate(tmp_path / "absent", tmp_path / "rejected")
    assert blocked["failures"][0]["observed"] == "missing_file"
    target = tmp_path / "changed" / d.HISTORICAL_PRIMARY
    atomic_json(target, dict(verdict_class="positive"))
    changed, _ = d.authenticate(tmp_path / "changed", tmp_path / "changed_raw")
    assert any(r["source_path"] == str(target) for r in changed["references"])


@pytest.mark.parametrize(
    "mutation",
    ["valid", "missing", "stale", "failed", "timeout", "log", "command", "duplicate", "code"],
)
def test_receipt_cli_preserves_strict_policy(tmp_path, mutation):
    """SCENARIO-REPORT-9028-RECEIPTS: the existing checker rejects every independent mutation."""
    from carnot.reporting import historical_provenance_9028 as d

    work = work_fixture(tmp_path)
    d.old.mutate_receipt(work, mutation)
    path = tmp_path / "work.json"
    atomic_json(path, work)
    assert d.old.receipts_pass(work) == (mutation == "valid")
    result = child(tmp_path, d.CLI, "--check-receipts", path)
    assert result.returncode == (0 if mutation == "valid" else 1)


def test_reduction_requires_current_coverage_and_controls(tmp_path, monkeypatch):
    """REQ-REPORT-9028-QUALIFICATION: current failures zero readiness and leave old verdicts intact."""
    from carnot.reporting import historical_provenance_9028 as d

    work = work_fixture(tmp_path)
    monkeypatch.setattr(d, "verify_coverage", lambda *_: True)
    value = d.build(work)
    assert value["provenance_ready_score"] == value["validation_scope_ready_score"] == 1
    assert value["original_verdict_preserved"] and not value["repository_health_passed"]
    assert value["MODEL_SPECS"] == value["model_specs"] == value["trained_head_specs"] == []
    assert not any(value["model_invocation_counts"].values())
    for key in ("controls", "current_receipt_mutation_rows", "historical_authority_rows"):
        changed = deepcopy(work)
        changed[key][0]["passed"] = False
        assert d.build(changed)["provenance_ready_score"] == 0
    changed = deepcopy(work)
    changed["validation"]["receipts"][0]["exit_code"] = 1
    assert d.build(changed)["verdict_class"] == "disqualified"
    changed["evidence"]["failures"] = [dict(check="input_bytes", observed="missing_file")]
    assert d.build(changed)["honest_verdict"] == "complete_blocked_input_bytes"
    for key in ("controls", "current_receipt_mutation_rows", "historical_authority_rows"):
        changed[key] = []
    blocked = d.build(changed)
    assert blocked["intended_count"] == blocked["censored_count"] == 21
    assert blocked["completed_count"] == blocked["failed_count"] == 0
    work["terminal_validation"]["passed"] = False
    assert d.build(work)["validation_scope_ready_score"] == 0


def test_private_publication_missing_inputs_and_tamper(tmp_path):
    """SCENARIO-REPORT-9028-REPLAY: private terminal publications reject rehashed drift."""
    from carnot.reporting import historical_provenance_9028 as d

    work = work_fixture(tmp_path)
    source = tmp_path / "source.json"
    atomic_json(source, work)
    output = tmp_path / "published" / (d.NAME + ".json")
    assert (
        child(
            tmp_path, d.CLI, "--date", "20261112", "--from-work", source, "--output", output
        ).returncode
        == 0
    )
    assert child(tmp_path, d.CLI, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    value["completed_count"] += 1
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(output, value)
    assert child(tmp_path, d.CLI, "--cold-replay", output).returncode == 1
    value["reproducibility_checksum"] = "sha256:changed"
    atomic_json(output, value)
    assert not d.replay(output)["passed"]
    assert child(tmp_path, d.CLI, "--cold-replay", tmp_path / "absent").returncode == 1
    work["evidence"], work["repository_health"] = d.authenticate(
        tmp_path / "missing", tmp_path / "rejected"
    )
    atomic_json(source, work)
    blocked = tmp_path / "blocked" / (d.NAME + ".json")
    assert (
        child(
            tmp_path, d.CLI, "--date", "20261112", "--from-work", source, "--output", blocked
        ).returncode
        == 0
    )
    assert json.loads(blocked.read_bytes())["honest_verdict"] == "complete_blocked_input_bytes"
    assert child(tmp_path, d.CLI, "--date", "wrong").returncode == 2
    assert (
        child(
            tmp_path,
            d.CLI,
            "--date",
            "20261112",
            "--from-work",
            source,
            "--output",
            d.ROOT / (d.NAME + ".json"),
        ).returncode
        == 2
    )
    assert not d.verify_coverage({})
    with pytest.raises(ValueError, match="historical_authority_bytes"):
        d.historical_authority(tmp_path / "private", tmp_path / "absent")


def test_execute_paths_and_deadline(tmp_path, monkeypatch):
    """REQ-REPORT-9028-QUALIFICATION: bounded execution keeps real checker semantics."""
    from carnot.reporting import historical_provenance_9028 as d

    fixture = work_fixture(tmp_path / "fixture")
    monkeypatch.setattr(d.tempfile, "mkdtemp", lambda **_: str(tmp_path / "private"))
    monkeypatch.setattr(
        d, "authenticate", lambda *_: (deepcopy(fixture["evidence"]), fixture["repository_health"])
    )
    monkeypatch.setattr(
        d, "authority_controls", lambda *_: deepcopy(fixture["historical_authority_rows"])
    )
    observed = []

    def check(spec, raw, manifest):
        observed.append(spec.name)
        log = raw / (spec.name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private orchestration fixture\n")
        code = int(spec.name in {"missing_ledger", "contradictory_current"})
        if spec.name.startswith("receipt_"):
            code = int(not d.old.receipts_pass(json.loads(Path(spec.argv[-1]).read_bytes())))
        return dict(
            name=spec.name,
            passed=code == 0,
            normal_exit=True,
            timed_out=False,
            exit_code=code,
            started_monotonic_ns=d.time.monotonic_ns(),
            ended_monotonic_ns=d.time.monotonic_ns(),
            scope_sha256=canonical_hash(manifest),
            command_argv=list(spec.argv),
            log_path=str(log),
            log_sha256=d.reference(log)["sha256"],
        )

    monkeypatch.setattr(d, "run_check", check)
    monkeypatch.delenv("COVERAGE_RCFILE", raising=False)
    work = d.execute(tmp_path / "raw", d.ROOT)
    assert len(work["controls"]) == 5
    assert len(work["current_receipt_mutation_rows"]) == 9
    assert all(r["passed"] for r in work["current_receipt_mutation_rows"])
    assert "direct_consumers" in observed
    assert not work["validation"]["coverage"]["complete"]
    (tmp_path / "private/.coverage").write_bytes(b"private fixture")
    monkeypatch.setattr(d, "measure", lambda *_: dict(complete=True))
    monkeypatch.setenv("COVERAGE_RCFILE", "existing-config")
    assert d.execute(tmp_path / "measured", d.ROOT)["validation"]["coverage"]["complete"]
    assert os.environ["COVERAGE_RCFILE"] == "existing-config"
    fixture["evidence"]["failures"] = [dict(check="input_bytes", observed="missing_file")]
    assert not d.execute(tmp_path / "blocked", d.ROOT)["controls"]
    clock = iter([0, 3300])
    monkeypatch.setattr(d.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        d.execute(tmp_path / "deadline", d.ROOT)


def test_main_records_failed_current_terminal(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9028-RECEIPTS: failed current scans disqualify a new candidate."""
    from carnot.reporting import historical_provenance_9028 as d

    work = work_fixture(tmp_path / "fixture")
    monkeypatch.setattr(d, "verify_coverage", lambda *_: True)
    monkeypatch.setattr(d, "execute", lambda *_: work)
    monkeypatch.setattr(d, "terminal", lambda *_: dict(passed=False))
    observed = []
    monkeypatch.setattr(d, "publish_primary", lambda _p, value, _v: observed.append(value) or {})
    assert d.main(["--date", "20261112", "--output", str(tmp_path / (d.NAME + ".json"))]) == 0
    assert observed[-1]["provenance_ready_score"] == 0
    assert observed[-1]["flagged_adversarial"]


def test_real_authority_controls_and_coverage_reopen(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9028-AUTHORITY: primitive child exits and reopened SQLite are required."""
    from coverage import Coverage
    from carnot.reporting import historical_provenance_9028 as d

    rows = d.authority_controls(
        tmp_path / "private", tmp_path / "raw", d.scope_manifest(tmp_path / "scope")
    )
    assert len(rows) == 7 and all(r["passed"] for r in rows)
    assert all(r["receipt"]["normal_exit"] for r in rows)
    assert rows[0]["expected_exit"] == 0 and all(r["expected_exit"] == 1 for r in rows[1:])
    private = tmp_path / "coverage"
    private.mkdir()
    tracer = Coverage(data_file=str(private / ".coverage"), config_file=False)
    tracer.start()
    d.progress("coverage_fixture", 1, 0)
    tracer.stop()
    tracer.save()
    report = d.measure(private, tmp_path / "measured")
    assert report["changed_statements"]["statements"] > 0
    assert not d.verify_coverage(report)
    with monkeypatch.context() as patch:
        patch.setattr(d, "verify_measured", lambda *_: True)
        assert not d.verify_coverage(report)
        assert not d.verify_coverage({})
    Path(report["data_reference"]["path"]).write_bytes(b"altered SQLite")
    assert not d.verify_coverage(report)


def test_retained_operand_drift_and_receipt_flags(tmp_path):
    """SCENARIO-REPORT-9028-REPLAY: even rehashed references cannot hide changed retained bytes."""
    from carnot.reporting import historical_provenance_9028 as d

    work = work_fixture(tmp_path)
    atomic_json(Path(work["raw"]) / "work.json", work)
    value = d.build(work)
    value["raw_shard_hashes"][0]["sha256"] = "sha256:changed"
    value["reproducibility_checksum"] = d.checksum(value)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert not d.replay(path)["passed"]
    row = work["current_receipt_mutation_rows"][-1]
    assert d.control_passed(row)
    row["receipt"]["exit_code"] = 0
    assert not d.control_passed(row)
