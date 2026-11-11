"""REQ-REPORT-9014-SCOPED: new custody never changes failed historical acceptance."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import scoped_provenance_9014 as d
from carnot.reporting import invocation_provenance_9000 as original
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def work_fixture(tmp_path):
    """Private administrative data exercises the receipt policy without science credit."""
    raw = tmp_path / "raw"
    manifest = d.scope_manifest(tmp_path / "private")
    work = dict(
        raw=str(raw),
        validation_scope_manifest=manifest,
        frozen_ns=1,
        evidence=dict(
            checks=[], failures=[], references=[], original_flag_rows=[], invocation_path_rows=[]
        ),
        repository_health=dict(rows=[], checks=[], references=[], passed=False),
        environment={},
        controls=[dict(name=k, passed=True, fixture_only=True) for k in original.CONTROLS],
        current_receipt_mutation_rows=[
            dict(name=k, passed=True) for k in ("valid", *d.RECEIPT_CONTROLS)
        ],
        validation=dict(receipts=[], coverage=dict(complete=True)),
        code_references=[],
        raw_references=[],
        duration_s=1.0,
        phase_spans=[],
        terminal_validation=dict(passed=True),
    )
    for row in work["controls"] + work["current_receipt_mutation_rows"]:
        log = raw / (row["name"] + ".control.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private control fixture\n")
        expected = int(
            row["name"] in {"missing_ledger", "contradictory_current", *d.RECEIPT_CONTROLS}
        )
        row.update(
            expected_exit=expected,
            observed_exit=expected,
            receipt=dict(
                exit_code=expected,
                normal_exit=True,
                timed_out=False,
                log_path=str(log),
                log_sha256=d.reference(log)["sha256"],
            ),
        )
    for c in manifest["commands"]:
        log = raw / (c["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private receipt fixture\n")
        work["validation"]["receipts"].append(
            dict(
                name=c["name"],
                passed=True,
                exit_code=0,
                normal_exit=True,
                timed_out=False,
                started_monotonic_ns=2,
                ended_monotonic_ns=3,
                scope_sha256=canonical_hash(manifest),
                command_argv=c["argv"],
                log_path=str(log),
                log_sha256=d.reference(log)["sha256"],
            )
        )
    return work


def child(tmp_path, *args):
    """Actual CLI children run outside the checkout with no ambient import path."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(d.ROOT / d.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private9014 child", args, flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print(
        "after private9014 child",
        result.returncode,
        result.stdout[-1000:],
        result.stderr[-1000:],
        flush=True,
    )
    return result


def test_scope_and_original_nodes(tmp_path):
    """REQ-REPORT-9014-SCOPED: owned scope excludes broad health and pins original tests."""
    manifest = d.scope_manifest(tmp_path)
    assert manifest["statement_percent"] == 100 and manifest["excluded_statements"] == 0
    assert "full_python_suite" in original.CHECKS
    assert "full_python_suite" not in manifest["required_check_names"]
    assert any(
        r["node_id"].endswith("test_full_suite_failure_zeros_readiness")
        for r in manifest["original_test_node_hashes"]
    )
    assert all(r["sha256"].startswith("sha256:") for r in manifest["original_test_node_hashes"])
    assert any("8206" in a for c in manifest["commands"] for a in c["argv"])
    assert all(
        "::" not in a
        for c in manifest["commands"]
        if c["name"].startswith(("ruff", "mypy"))
        for a in c["argv"]
    )


def test_authentication_and_external_refusal(tmp_path):
    """REQ-REPORT-9014-SCOPED: complete historical bytes stay disqualified by their old rules."""
    evidence, health = d.authenticate(d.ROOT, tmp_path / "good")
    assert not evidence["failures"]
    assert len(health["rows"]) == 3 and health["passed"] is False
    assert evidence["original_flag_rows"] and evidence["invocation_path_rows"]
    assert all(r["original_verdict_class"] == "disqualified" for r in health["rows"])
    missing, _ = d.authenticate(tmp_path / "missing", tmp_path / "blocked")
    assert missing["failures"][0]["observed"] == "missing_file"
    target = tmp_path / "changed" / next(iter(d.HEALTH_PINS))
    atomic_json(target, dict(verdict_class="positive"))
    changed, _ = d.authenticate(tmp_path / "changed", tmp_path / "rejected")
    assert any(
        r["check"] == "historical_primary_bytes" and not r["passed"] for r in changed["checks"]
    )
    assert any(r.get("source_path") == str(target) for r in changed["references"])


@pytest.mark.parametrize(
    "mutation", ["valid", *d.RECEIPT_CONTROLS, "log", "command", "duplicate", "code"]
)
def test_receipt_policy_and_actual_cli(tmp_path, mutation):
    """SCENARIO-REPORT-9014-RECEIPTS: CLI acceptance uses actual normal exits and current scope."""
    work = work_fixture(tmp_path)
    d.mutate_receipt(work, mutation)
    path = tmp_path / "work.json"
    atomic_json(path, work)
    assert d.receipts_pass(work) == (mutation == "valid")
    result = child(tmp_path, "--check-receipts", path)
    assert result.returncode == (0 if mutation == "valid" else 1), result.stdout + result.stderr
    assert not any(original.ZERO_INVOCATION_COUNTS.values())


def test_reduction_and_terminal_classes(tmp_path, monkeypatch):
    """REQ-REPORT-9014-SCOPED: scoped readiness cannot upgrade historical global health."""
    work = work_fixture(tmp_path)
    monkeypatch.setattr(d, "verify_measured", lambda *_: True)
    value = d.build(work)
    assert value["provenance_ready_score"] == value["validation_scope_ready_score"] == 1
    assert value["original_verdict_preserved"] and not value["repository_health_passed"]
    assert value["inference_substrate_class"] == "no_model_load"
    assert value["MODEL_SPECS"] == value["model_specs"] == value["trained_head_specs"] == []
    assert not any(value["model_invocation_counts"].values())
    for key in ("controls", "current_receipt_mutation_rows"):
        bad = deepcopy(work)
        bad[key][0]["passed"] = False
        assert d.build(bad)["provenance_ready_score"] == 0
    bad = deepcopy(work)
    bad["terminal_validation"]["passed"] = False
    assert d.build(bad)["verdict_class"] == "disqualified"
    bad["evidence"]["failures"] = [dict(check="input_bytes", observed="missing_file")]
    assert d.build(bad)["honest_verdict"] == "complete_blocked_input_bytes"
    monkeypatch.setattr(d, "verify_measured", lambda *_: False)
    assert d.build(work)["provenance_ready_score"] == 0


def test_private_publication_replay_tamper_and_missing(tmp_path):
    """SCENARIO-REPORT-9014-REPLAY: private publications retain observed external refusals."""
    work = work_fixture(tmp_path)
    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / "published" / (d.NAME + ".json")
    assert (
        child(tmp_path, "--date", "20261111", "--from-work", source, "--output", output).returncode
        == 0
    )
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    value["completed_count"] += 1
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    value["reproducibility_checksum"] = "sha256:changed"
    atomic_json(output, value)
    assert not d.replay(output)["passed"]
    assert child(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    work["evidence"], work["repository_health"] = d.authenticate(
        tmp_path / "absent", tmp_path / "missing_evidence"
    )
    atomic_json(source, work)
    blocked = tmp_path / "blocked" / (d.NAME + ".json")
    assert (
        child(tmp_path, "--date", "20261111", "--from-work", source, "--output", blocked).returncode
        == 0
    )
    assert json.loads(blocked.read_bytes())["honest_verdict"] == "complete_blocked_input_bytes"
    assert child(tmp_path, "--cold-replay", blocked).returncode == 0
    assert child(tmp_path, "--date", "wrong").returncode == 2
    assert (
        child(
            tmp_path,
            "--date",
            "20261111",
            "--from-work",
            source,
            "--output",
            d.ROOT / (d.NAME + ".json"),
        ).returncode
        == 2
    )


def test_altered_retained_bytes_and_malformed_receipts(tmp_path):
    """SCENARIO-REPORT-9014-RECEIPTS: altered bytes, old clocks and missing fields fail closed."""
    work = work_fixture(tmp_path)
    for field, value in [
        ("started_monotonic_ns", 0),
        ("ended_monotonic_ns", 0),
        ("normal_exit", False),
        ("exit_code", True),
    ]:
        changed = deepcopy(work)
        changed["validation"]["receipts"][0][field] = value
        assert not d.receipts_pass(changed)
    assert not d.receipts_pass({})
    changed = deepcopy(work)
    Path(changed["validation"]["receipts"][0]["log_path"]).write_text("altered\n")
    assert not d.receipts_pass(changed)
    atomic_json(Path(work["raw"]) / "work.json", work)
    value = d.build(work)
    value["raw_shard_hashes"][0]["sha256"] = "sha256:changed"
    value["reproducibility_checksum"] = d.checksum(value)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert not d.replay(path)["passed"]


def test_execute_paths_and_deadline(tmp_path, monkeypatch):
    """REQ-REPORT-9014-SCOPED: bounded orchestration never runs broad health or loads a model."""
    fixture = work_fixture(tmp_path / "fixture")
    monkeypatch.setattr(d.tempfile, "mkdtemp", lambda **_: str(tmp_path / "private"))
    monkeypatch.setattr(
        d, "authenticate", lambda *_: (deepcopy(fixture["evidence"]), fixture["repository_health"])
    )
    observed = []

    def check(spec, raw, manifest):
        observed.append(spec.name)
        log = raw / (spec.name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private orchestration fixture\n")
        code = 1 if spec.name in {"missing_ledger", "contradictory_current"} else 0
        if spec.name.startswith("receipt_"):
            checked = json.loads(Path(spec.argv[-1]).read_bytes())
            code = 0 if d.receipts_pass(checked) else 1
        receipt = dict(
            name=spec.name,
            passed=code == 0,
            normal_exit=True,
            exit_code=code,
            timed_out=False,
            started_monotonic_ns=d.time.monotonic_ns(),
            ended_monotonic_ns=d.time.monotonic_ns(),
            scope_sha256=canonical_hash(manifest),
            command_argv=list(spec.argv),
            log_path=str(log),
            log_sha256=d.reference(log)["sha256"],
        )
        return receipt

    monkeypatch.setattr(d, "run_check", check)
    monkeypatch.delenv("COVERAGE_RCFILE", raising=False)
    work = d.execute(tmp_path / "executed", d.ROOT)
    assert len(work["controls"]) == 5
    assert all(r["passed"] for r in work["current_receipt_mutation_rows"])
    assert "full_python_suite" not in observed
    assert not work["validation"]["coverage"]["complete"]
    (tmp_path / "private/.coverage").write_bytes(b"private measurement fixture")
    monkeypatch.setattr(d, "measured", lambda *_: dict(complete=True))
    monkeypatch.setenv("COVERAGE_RCFILE", "existing-config")
    work = d.execute(tmp_path / "measured", d.ROOT)
    assert work["validation"]["coverage"]["complete"]
    assert os.environ["COVERAGE_RCFILE"] == "existing-config"
    fixture["evidence"]["failures"] = [dict(check="input_bytes", observed="missing_file")]
    assert not d.execute(tmp_path / "blocked", d.ROOT)["controls"]
    clock = iter([0, 3300])
    monkeypatch.setattr(d.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        d.execute(tmp_path / "deadline", d.ROOT)


def test_main_uses_new_scope_and_failed_terminal(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9014-REPLAY: a failed current terminal check cannot promote readiness."""
    work = work_fixture(tmp_path / "fixture")
    monkeypatch.setattr(d, "verify_measured", lambda *_: True)
    monkeypatch.setattr(d, "execute", lambda *_: work)
    monkeypatch.setattr(d, "terminal", lambda *_: dict(passed=False))
    observed = []
    monkeypatch.setattr(
        d, "publish_primary", lambda _p, value, _validator: observed.append(value) or {}
    )
    output = tmp_path / (d.NAME + ".json")
    assert d.main(["--date", "20261111", "--output", str(output)]) == 0
    assert observed[-1]["provenance_ready_score"] == 0
    assert observed[-1]["flagged_adversarial"]


def test_control_receipts_cannot_be_replaced_by_passing_flags(tmp_path, monkeypatch):
    """REQ-REPORT-9014-SCOPED: primitive exits and log bytes determine control outcomes."""
    work = work_fixture(tmp_path)
    monkeypatch.setattr(d, "verify_measured", lambda *_: True)
    assert d.build(work)["provenance_ready_score"] == 1
    row = work["controls"][0]
    row["receipt"]["exit_code"] = 7
    assert not d.control_passed(row)
    assert d.build(work)["provenance_ready_score"] == 0
    row["receipt"]["exit_code"] = row["expected_exit"]
    Path(row["receipt"]["log_path"]).write_text("changed control log\n")
    assert not d.control_passed(row)
    assert not d.control_passed({})
