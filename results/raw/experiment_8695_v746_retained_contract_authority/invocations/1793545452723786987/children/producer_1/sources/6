"""REQ-REPORT-8695: full task objects and original operands grant branch authority."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v746_contract_authority as a
from carnot.reporting import v746_contract_runner as runner
from carnot.reporting import v746_retained_evidence as e
from carnot.reporting.v709_execution import child


def test_contract_and_mutations(tmp_path):
    """SCENARIO-REPORT-8695-AUTHORITY: every registered mutation rejects independently."""
    assert a.authority(a.ROOT)["ready"]
    rows = a.mutations()
    assert len(rows) >= 20 and all(r["passed"] for r in rows)
    assert not a.authority(tmp_path)["ready"]
    p = a.methods()
    assert p["learning_authority_inputs_ready_score"] == 1
    assert p["objective_updates"] == a.original.protocol()["objective_updates"]
    assert p["objective_audit"] == a.original.protocol()["objective_audit"]
    assert len(p["future_dependencies"]) == 13
    assert all(r["status"] == "future_dependency_not_observation" for r in p["future_dependencies"])


@pytest.mark.parametrize(
    "mode", ["missing", "changed", "field", "zero", "flagged", "terminal", "identity"]
)
def test_terminal_producer_gate(tmp_path, mode):
    """REQ-REPORT-8695: producer identity, field and terminal bytes must all authenticate."""
    p = tmp_path / "producer.json"
    side = tmp_path / "terminal.json"
    value = dict(
        experiment_id=42,
        task_id="exp42-original",
        ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        terminal_validation_sidecar_path=str(side),
    )
    if mode == "field":
        value.pop("ready_score")
    if mode == "zero":
        value["ready_score"] = 0
    if mode == "flagged":
        value["flagged_adversarial"] = True
    if mode == "identity":
        value["experiment_id"] = 43
    e.atomic_json(p, value)
    ref = e.reference(p)
    e.atomic_json(side, dict(primary_sha256=ref["sha256"], report=dict(passed=True)))
    if mode == "missing":
        p.unlink()
    if mode == "changed":
        p.write_text("{}")
    if mode == "terminal":
        side.write_text("{}")
    gate = a.terminal(ref, "ready_score", 42)
    assert not gate["passed"] and gate["gate_check_summary"]
    if mode in {"missing", "changed", "field", "zero"}:
        assert (
            gate["gate_check_summary"][0]["operand_state"]
            == {
                "missing": "missing_file",
                "changed": "changed_bytes",
                "field": "missing_field",
                "zero": "observed_zero",
            }[mode]
        )
    if mode == "terminal":
        e.atomic_json(side, dict(primary_sha256=ref["sha256"], report=dict(passed=True)))
        assert a.terminal(ref, "ready_score", 42)["passed"]


def test_private_cli_and_replay(tmp_path):
    """SCENARIO-VERIFY-8695-CLI: actual success, failed child and cold rejection paths."""
    output = tmp_path / "results" / (a.NAME + ".json")
    command = [sys.executable, "-u", str(a.ROOT / a.CLI)]
    for name, args, expected in [
        ("private", ["--private-control", "--output", str(output)], 0),
        ("cold", ["--cold-replay", str(output)], 0),
        ("bad_date", ["--date", "20260231"], 2),
        ("unsafe", ["--private-control"], 2),
        ("error", ["--deliberate-error"], 7),
    ]:
        r = child(name, command + args, tmp_path / "logs", expected=expected, deadline=120)
        assert r["passed"], Path(r["stderr_path"]).read_text()
    v = json.loads(output.read_bytes())
    assert v["verdict_class"] == "blocked" and v["contract_ready_score"] == 0
    assert runner.replay(output)
    v["contract_ready_score"] = 1
    e.atomic_json(output, v)
    assert not runner.replay(output)
    failed = tmp_path / "failed" / output.name
    r = child(
        "failed",
        command + ["--private-control", "--failed-child", "--output", str(failed)],
        tmp_path / "logs",
        deadline=120,
    )
    assert r["passed"]
    v = json.loads(failed.read_bytes())
    assert v["verdict_class"] == "disqualified" and v["flagged_adversarial"]
    assert v["validation_receipts"][0]["actual_exit"] == 3
    assert not runner.replay(tmp_path / "missing")


def test_private_path_guards(tmp_path):
    """SCENARIO-VERIFY-8695-CLI: symlink aliases and lexical public paths both reject."""
    alias = tmp_path / "alias"
    alias.symlink_to(a.ROOT, target_is_directory=True)
    for path in [a.ROOT / "results" / (a.NAME + ".json"), alias / "out.json"]:
        with pytest.raises(ValueError, match="outside"):
            runner.private_output(path)
    runner.private_output(tmp_path / "safe.json")


def test_independent_replay_seal(tmp_path):
    """SCENARIO-VERIFY-8695-CLI: locally repaired primitive hashes cannot repair the seal."""
    output = tmp_path / "results" / (a.NAME + ".json")
    assert runner.main(["--private-control", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    ref = value["primitive_reference"]
    primitive = e.load(ref)
    primitive["contract"]["ready"] = False
    e.atomic_json(Path(ref["path"]), primitive)
    value["primitive_reference"] = e.reference(Path(ref["path"]))
    e.atomic_json(output, value)
    assert not runner.replay(output)
    Path(ref["path"]).unlink()
    assert not runner.replay(output)


def test_missing_method_operand_blocks(tmp_path, monkeypatch):
    """REQ-REPORT-8695: absent original scientific inputs produce exact blocked gates."""
    p = a.original.protocol()
    p["learning_data_reference"]["path"] = str(tmp_path / "missing-data")
    monkeypatch.setattr(a.original, "protocol", lambda: p)
    result = a.methods()
    assert result["learning_authority_inputs_ready_score"] == 0
    assert any(r["operand_state"] == "missing_file" for r in result["gate_check_summary"])
    monkeypatch.setattr(a.original, "PROTOCOL", "absent-original-protocol.json")

    def absent():
        raise FileNotFoundError("absent original protocol")

    monkeypatch.setattr(a.original, "protocol", absent)
    result = a.methods()
    assert result["learning_authority_inputs_ready_score"] == 0
    assert result["gate_check_summary"][0]["operand_state"] == "missing_file"


def test_terminal_rejects_false_acceptance(tmp_path):
    """REQ-REPORT-8695: a matching readiness scalar cannot override failed terminal validation."""
    output, side = tmp_path / "producer.json", tmp_path / "terminal.json"
    e.atomic_json(
        output,
        dict(
            experiment_id=42,
            ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(side),
        ),
    )
    ref = e.reference(output)
    e.atomic_json(side, dict(primary_sha256=ref["sha256"], report=dict(passed=False)))
    assert not a.terminal(ref, "ready_score", 42)["passed"]


def test_collect_once_and_admission(tmp_path, monkeypatch):
    """REQ-VERIFY-8695: the real collection preserves77 original nodes in disjoint cohorts."""
    private = tmp_path / "private"
    private.mkdir()
    raw = tmp_path / "raw"
    plan, receipt = runner.collect(private, raw)
    original = [n for row in plan if row["role"] == "producer" for n in row["nodes"]]
    assert receipt["passed"] and len(original) == len(set(original)) == 77
    nodes = [n for row in plan for n in row["nodes"]]
    assert len(nodes) == len(set(nodes))
    assert all("--no-cov" in row["argv"] and "-n" in row["argv"] for row in plan)
    stream = tmp_path / "empty"
    stream.write_text("")
    for passed in [False, True]:
        monkeypatch.setattr(
            runner, "child", lambda *args, **kw: dict(passed=passed, stdout_path=str(stream))
        )
        with pytest.raises(ValueError, match="collection_failed|missing_required_nodes"):
            runner.collect(private, raw)


def test_full_runner_with_private_cpu_children(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8695-CLI: incomplete constructed producers disqualify the actual full path."""
    from carnot.reporting import v746_contract_report as report

    source = tmp_path / "test_actual.py"
    source.write_text("def test_a():\n    assert True\ndef test_b():\n    assert True\n")

    def plan(private, raw):
        specs = []
        for i, (node, role) in enumerate([("test_a", "producer"), ("test_b", "consumer")]):
            scratch = private / str(i)
            scratch.mkdir()
            config = raw / f"control-{i}.ini"
            config.write_text(f"[run]\nparallel=true\ndata_file={scratch}/.coverage\n")
            nodes = [str(source) + "::" + node]
            specs.append(
                dict(
                    name=role,
                    role=role,
                    scratch=str(scratch),
                    config=e.reference(config),
                    nodes=nodes,
                    deadline=60,
                    argv=[
                        sys.executable,
                        "-m",
                        "coverage",
                        "run",
                        "--rcfile=" + str(config),
                        "-m",
                        "pytest",
                        "-c",
                        "/dev/null",
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        "-q",
                        *nodes,
                        "--junitxml=" + str(scratch / "junit.xml"),
                    ],
                )
            )
        receipt = child(
            "constructed_collection",
            [sys.executable, "-c", "print('constructed collection')"],
            raw / "checks",
            deadline=10,
        )
        return specs, receipt

    original_preconditions = runner.preconditions
    monkeypatch.setattr(
        runner, "preconditions", lambda p: dict(original_preconditions(p), disk_backed=False)
    )
    monkeypatch.setattr(runner, "source_hashes", lambda: [e.reference(source)])
    monkeypatch.setattr(runner, "collect", plan)
    output = tmp_path / "full" / (a.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["learning_authority_ready_score"] == 0
    assert runner.replay(output)
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    work = e.load(value["primitive_reference"])
    before = deepcopy(work)
    assert report.build(work, value["validation_reference"], raw, output) == value
    assert work == before
    source.write_text("changed source after seal")
    assert not runner.replay(output)
    # Restore only the test's own frozen source, then exercise independent reconstruction guards.
    work_source = next(r for r in work["source_hashes"] if r["path"] == str(source))
    source.write_text("def test_a():\n    assert True\ndef test_b():\n    assert True\n")
    assert e.sha256_file(source) == work_source["sha256"]
    original_methods = a.methods
    monkeypatch.setattr(
        a, "methods", lambda: dict(original_methods(), learning_authority_inputs_ready_score=0)
    )
    assert not runner.replay(output)
    monkeypatch.setattr(a, "methods", original_methods)
    original_aggregate = e.aggregate
    monkeypatch.setattr(e, "aggregate", lambda *args: dict(passed=False))
    assert not runner.replay(output)
    monkeypatch.setattr(e, "aggregate", original_aggregate)
    monkeypatch.setattr(e, "rejection_executed", lambda rows: True)
    assert not runner.replay(output)


def test_direct_cli_dispatch_guards():
    """SCENARIO-VERIFY-8695-CLI: early exits also execute under owned statement coverage."""
    assert runner.main(["--deliberate-error"]) == 7
    with pytest.raises(SystemExit) as error:
        runner.main(["--private-control"])
    assert error.value.code == 2
