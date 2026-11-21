"""REQ-REPORT-9125-V778 / SCENARIO-REPORT-9125-V778: bounded, complete qualification."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import receipt_qualification_9125 as q
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_partition_keeps_exact_union_and_module_fixtures() -> None:
    nodes = [f"tests/python/test_a.py::test_{i}" for i in range(3)]
    nodes += ["tests/python/test_b.py::test_x"]
    parts = q.partitions(nodes)
    assert [n for p in parts for n in p] == nodes
    assert all(len({n.split("::")[0] for n in p}) == 1 for p in parts)
    with pytest.raises(ValueError, match="duplicate_node"):
        q.partitions(nodes + nodes[:1])


@pytest.mark.parametrize("observed", ["missing_file", "missing_field", 0, True, 1])
def test_gate_distinguishes_missing_zero_boolean_and_integer(
    tmp_path: Path, monkeypatch: Any, observed: Any
) -> None:
    path = tmp_path / q.CONTRACT
    if observed != "missing_file":
        atomic_json(path, {} if observed == "missing_field" else {"contract_ready_score": observed})
        monkeypatch.setitem(q.PINS, q.CONTRACT, q.ref(path)["sha256"])
    work = q.authenticate(tmp_path, tmp_path / "raw")
    checks = work["checks"]
    assert checks[0 if observed == "missing_file" else 1]["observed"] == observed
    if type(observed) is not int or observed != 1:
        assert not all(c["passed"] for c in checks)
        assert "previous" not in work


def test_node_reduction_requires_all_phases_and_preserves_unfinished_units() -> None:
    nodes = ["a::ok", "a::fail", "a::unfinished", "a::absent"]
    events = [
        dict(nodeid="a::ok", when=p, outcome="passed", duration=0.25)
        for p in ["setup", "call", "teardown"]
    ]
    events += [dict(nodeid="a::fail", when="call", outcome="failed", duration=1)]
    events += [dict(nodeid="a::unfinished", when="start", outcome="started", duration=0)]
    rows = q.node_rows(nodes, events)
    assert [r["status"] for r in rows] == ["completed", "failed", "censored", "censored"]
    assert rows[0]["duration_s"] == 0.75
    assert rows[2]["censor_reason"] == "unfinished_node"
    assert rows[3]["censor_reason"] == "not_started_before_bound"
    with pytest.raises(ValueError, match="foreign_node"):
        q.node_rows(nodes, [dict(nodeid="other", when="call", outcome="passed", duration=0)])


def test_rehashed_outcomes_cannot_become_readiness(tmp_path: Path) -> None:
    work = dict(
        checks=[dict(passed=False, check="missing_contract")],
        references=[],
        nodes=[],
        partition_node_manifests=[],
        events=[],
        receipts=[],
        controls=[],
        coverage={},
        real_success_path_receipt={},
    )
    value = q.build(work, "20261116", 0, 1)
    assert value["verdict_class"] == "blocked" and value["receipt_ready_score"] == 0
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert value["original_quarantine_preserved"] is True
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert q.replay(path)["passed"]
    changed = deepcopy(value)
    changed["receipt_ready_score"] = 1
    changed["reproducibility_checksum"] = canonical_hash(changed["replay_input"])
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="reduction_drift"):
        q.replay(path)
    value["reproducibility_checksum"] = "sha256:wrong"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="replay_hash"):
        q.replay(path)


def test_private_real_cli_refusal_cold_replay_and_tamper(tmp_path: Path) -> None:
    output = tmp_path / (q.NAME + ".json")
    args = ["--input-root", str(tmp_path / "absent"), "--output", str(output)]
    receipt = q.child("private_refusal", q.foreign_argv(*args), tmp_path / "logs", deadline=120)
    assert receipt["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["receipt_ready_score"] == 0
    assert q.child("cold", q.foreign_argv("--replay", str(output)), tmp_path / "logs")["passed"]
    value["completed_count"] += 1
    atomic_json(output, value)
    assert q.child(
        "tamper", q.foreign_argv("--replay", str(output)), tmp_path / "logs", expected=1
    )["passed"]
    assert q.main(["--date", "wrong"]) == 1
    assert q.main(["--replay", str(tmp_path / "absent.json")]) == 1


def test_passing_summaries_without_original_validator_proof_never_qualify() -> None:
    nodes = ["a::ok"]
    events = [
        dict(nodeid=nodes[0], when=p, outcome="passed", duration=0)
        for p in ["setup", "call", "teardown"]
    ]
    work = dict(
        nodes=nodes,
        partition_node_manifests=[dict(node_ids=nodes)],
        controls=[dict(passed=True)],
        receipts=[dict(passed=True)],
        coverage=dict(percent=100, files=[]),
        real_success_path_receipt=dict(readiness_verified=True),
    )
    assert q.qualified(work, q.node_rows(nodes, events)) is False


def test_authentication_retains_original_logs_and_detects_drift(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("original source")
    reference = q.ref(source)
    previous = dict(
        code_config_hashes=[reference],
        source_references=[],
        owned_statement_coverage=dict(data_reference=reference),
        validation_receipts=[
            dict(
                stdout_path=str(source),
                stderr_path=str(source),
                stdout_sha256=reference["sha256"],
                stderr_sha256=reference["sha256"],
            )
        ],
    )
    for name, value in [(q.CONTRACT, dict(contract_ready_score=1)), (q.PREVIOUS, previous)]:
        path = tmp_path / name
        atomic_json(path, value)
        monkeypatch.setitem(q.PINS, name, q.ref(path)["sha256"])
    monkeypatch.setattr(q.old, "authenticate", lambda *a: dict(checks=[dict(passed=True)]))
    work = q.authenticate(tmp_path, tmp_path / "raw")
    assert all(c["passed"] for c in work["checks"]) and work["evidence"]
    source.write_text("drift")
    assert not all(c["passed"] for c in q.authenticate(tmp_path, tmp_path / "drift")["checks"])


def test_actual_node_plugin_and_log_authentication(tmp_path: Path, monkeypatch: Any) -> None:
    from types import SimpleNamespace

    path = tmp_path / "nodes.jsonl"
    monkeypatch.delenv("CARNOT_9125_NODE_LOG", raising=False)
    q.pytest_runtest_logstart("a::ok", None)
    monkeypatch.setenv("CARNOT_9125_NODE_LOG", str(path))
    q.pytest_runtest_logstart("a::ok", None)
    q.pytest_runtest_logreport(
        SimpleNamespace(nodeid="a::ok", when="call", outcome="failed", duration=1)
    )
    assert [json.loads(s)["when"] for s in path.read_text().splitlines()] == ["start", "call"]


def test_bounded_orchestration_keeps_failures_and_full_commands(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """SCENARIO-REPORT-9125-V778: simulated boundaries only exercise disqualification."""
    names = ["tests/python/a.py::test_x", "tests/python/b.py::test_y"]
    commands = [dict(argv=[n.split("::")[0]]) for n in names]
    previous = dict(
        replay_input=dict(
            work=dict(
                scope=dict(
                    commands=commands,
                    coverage_configuration=dict(path=str(tmp_path / "coverage.ini")),
                )
            )
        )
    )
    work = dict(
        previous=previous,
        checks=[],
        references=[],
        receipts=[],
        evidence={},
        nodes=[],
        partition_node_manifests=[],
        events=[],
        controls=[],
        coverage={},
        real_success_path_receipt={},
    )
    monkeypatch.setattr(q.old, "validation_commands", lambda p: commands)
    monkeypatch.setattr(
        q.old,
        "measure",
        lambda *a: dict(semantic_control_rows=[dict(passed=True)], preserved_detection_rows=[]),
    )
    monkeypatch.setattr(q.old, "replay", lambda p: dict(passed=True))

    def run(
        name: str, argv: list[str], raw: Path, deadline: float = 120, expected: int = 0
    ) -> dict[str, Any]:
        if name == "collect_original":
            assert argv[:3] == ["env", "-u", "COVERAGE_PROCESS_CONFIG"]
            out = tmp_path / "collection.stdout"
            out.write_text("\n".join(names))
            return dict(stdout_path=str(out), passed=True)
        if name.startswith("partition_"):
            log = Path(
                next(s.split("=", 1)[1] for s in argv if s.startswith("CARNOT_9125_NODE_LOG="))
            )
            log.write_text(
                json.dumps(dict(nodeid=names[0], when="call", outcome="failed", duration=1)) + "\n"
            )
        if name == "original_nondiagnostic_cli":
            health = (
                Path(argv[-1]).parent
                / "raw"
                / q.old.NAME
                / "invocations"
                / "fixture"
                / "health"
                / "repository_health.receipt.json"
            )
            atomic_json(health, dict(passed=False))
            atomic_json(
                Path(argv[-1]),
                dict(
                    forward_evidence_ready_score=0,
                    validation_receipts=[],
                    owned_statement_coverage={},
                ),
            )
        return dict(passed=False)

    monkeypatch.setattr(q, "run_check", run)
    result = q.measure(deepcopy(work), tmp_path / "measured", q.time.monotonic() + 3300)
    assert (
        result["nodes"] == names and not result["real_success_path_receipt"]["readiness_verified"]
    )
    assert q.build(result, "20261116", 0, 1)["receipt_ready_score"] == 0
    expired = q.measure(deepcopy(work), tmp_path / "expired", 0)
    assert not expired["events"] and not expired["real_success_path_receipt"]
    commands.append(dict(argv=["unexpected"]))
    previous["replay_input"]["work"]["scope"]["commands"] = []
    with pytest.raises(ValueError, match="original_command_scope_drift"):
        q.measure(work, tmp_path, 0)


def test_owned_verification_reads_actual_coverage(tmp_path: Path, monkeypatch: Any) -> None:
    from coverage import Coverage

    source = tmp_path / "owned.py"
    source.write_text("x = 1\n")
    config = tmp_path / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(tmp_path / ".coverage")
        + "\n[report]\nexclude_lines=\nomit=\n"
    )
    tracer = Coverage(config_file=str(config), data_suffix=True)
    tracer.start()
    exec(compile(source.read_text(), str(source), "exec"), {})
    tracer.stop()
    tracer.save()
    monkeypatch.setattr(q, "OWNED", [str(source)])
    monkeypatch.setattr(q.old, "coverage_rows", lambda *a: [])
    work = dict(
        receipts=[],
        evidence={},
        original_coverage=dict(data_reference=q.ref(Path(tracer.get_data().data_filename()))),
    )
    monkeypatch.setattr(q, "run_check", lambda *a, **kw: dict(passed=False))
    q.verify(work, tmp_path)
    assert work["coverage"]["percent"] == 100 and work["coverage"]["excluded_statements"] == 0
    assert len(work["receipts"]) == 5


def test_main_publishes_disqualification_without_scientific_credit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Only external boundaries are replaced; the real publishers and checkers run."""
    work = dict(
        checks=[],
        references=[],
        nodes=[],
        partition_node_manifests=[],
        events=[],
        receipts=[],
        controls=[],
        coverage={},
        real_success_path_receipt={},
    )
    monkeypatch.setattr(q, "authenticate", lambda *a: deepcopy(work))
    monkeypatch.setattr(q, "measure", lambda w, *a: w)
    monkeypatch.setattr(q, "verify", lambda *a: None)
    from unittest.mock import Mock

    monkeypatch.setattr(q, "Coverage", lambda **kw: Mock())
    path = tmp_path / (q.NAME + ".json")
    assert q.main(["--output", str(path)]) == 0
    value = json.loads(path.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["receipt_ready_score"] == 0
    monkeypatch.setattr(q.shutil, "disk_usage", lambda p: type("Usage", (), {"free": 0})())
    assert q.main(["--output", str(tmp_path / "low" / path.name)]) == 0


def test_real_timeout_reaps_children_in_separate_sessions(tmp_path: Path) -> None:
    import os

    marker = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        f"open({str(marker)!r},'w').write(str(p.pid)); time.sleep(60)"
    )
    result = q.run_check(
        "nested_timeout", [str(q.ROOT / ".venv/bin/python"), "-c", script], tmp_path, 1
    )
    assert result["timed_out"] and result["cleanup_passed"]
    pid = int(marker.read_text())
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"
    assert pid != os.getpid()


def test_qualification_reopens_coverage_and_refuses_incomplete_normal_result(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from coverage import Coverage

    source = tmp_path / "one.py"
    source.write_text("x = 1\n")
    tracer = Coverage(config_file=False, data_file=str(tmp_path / ".coverage"))
    tracer.set_option("report:exclude_lines", [])
    tracer.start()
    exec(compile(source.read_text(), str(source), "exec"), {})
    tracer.stop()
    tracer.save()
    _, statements, excluded, missing, _ = tracer.analysis2(str(source))
    monkeypatch.setattr(q, "OWNED", [str(source)])
    nodes = ["a::ok"]
    events = [
        dict(nodeid=nodes[0], when=p, outcome="passed", duration=0)
        for p in ["setup", "call", "teardown"]
    ]
    commands = q.verification_commands(tmp_path)
    receipts = [
        dict(name=name, argv=argv, passed=True, normal_exit=True, timed_out=False)
        for name, argv in commands
    ]
    report = dict(
        percent=100,
        files=[dict(path=str(source), statements=statements, excluded=excluded, missing=missing)],
        data_reference=q.ref(tmp_path / ".coverage"),
    )
    work = dict(
        nodes=nodes,
        partition_node_manifests=[dict(node_ids=nodes)],
        controls=[dict(passed=True)],
        receipts=receipts,
        coverage=report,
        verification_root=str(tmp_path),
        real_success_path_receipt={},
    )
    rows = q.node_rows(nodes, events)
    assert q.qualified(work, rows) is False
    work["receipts"] = receipts[:-1]
    assert q.qualified(work, rows) is False
    work["receipts"] = receipts
    work["coverage"]["files"][0]["statements"] += [99]
    assert q.qualified(work, rows) is False
    work["coverage"]["files"][0]["statements"].remove(99)
    proof = tmp_path / "normal.json"
    atomic_json(proof, dict(forward_evidence_ready_score=0, replay_input={}))
    work["real_success_path_receipt"] = dict(
        artifact_reference=q.ref(proof),
        diagnostic_only=False,
        command_receipt=dict(passed=True, argv=[]),
    )
    monkeypatch.setattr(q.old, "replay", lambda p: dict(passed=True))
    assert q.qualified(work, rows) is False


def test_replay_recomputes_node_and_control_views(tmp_path: Path, monkeypatch: Any) -> None:
    """SCENARIO-REPORT-9125-V778: rehashing a changed node/control view cannot repair it."""
    from test_forward_evidence_9111 import panel

    # Administrative structural fixtures never authorize readiness.
    diagnostic = panel()
    judge = dict(diagnostic)
    judge.pop("structural_count_evidence")
    judge_path, diagnostic_path = tmp_path / "judge.json", tmp_path / "diagnostic.json"
    atomic_json(judge_path, judge)
    atomic_json(diagnostic_path, diagnostic)
    semantic = q.old.semantic_control_rows()
    adverse = q.old.structural_controls(diagnostic)
    collect, log = tmp_path / "collect.stdout", tmp_path / "nodes.jsonl"
    collect.write_text("tests/python/a.py::test_x\n")
    events = [dict(nodeid="tests/python/a.py::test_x", when="call", outcome="failed", duration=1)]
    log.write_text(json.dumps(events[0]) + "\n")
    work = dict(
        checks=[],
        references=[q.ref(log)],
        nodes=[events[0]["nodeid"]],
        partition_node_manifests=[dict(partition_id=0, node_ids=[events[0]["nodeid"]])],
        events=events,
        receipts=[
            q.child(
                "real_log",
                [str(q.ROOT / ".venv/bin/python"), "-c", "print('durable')"],
                tmp_path / "logs",
            )
        ],
        controls=semantic + adverse,
        coverage={},
        real_success_path_receipt={},
        collection_receipt=dict(stdout_path=str(collect)),
        event_references=[q.ref(log)],
        original_semantic_controls=semantic,
        adversarial_control_rows=adverse,
        historical_work=dict(diagnostic_input_reference=q.ref(diagnostic_path)),
        evidence=dict(snapshots=dict(judge=q.ref(judge_path))),
        owner_pid=123,
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, q.build(work, "20261116", 0, 1))
    assert q.replay(path)["passed"]
    for change in ["nodes", "controls"]:
        altered = deepcopy(work)
        altered[change] = []
        if change == "nodes":
            altered["events"] = []
        atomic_json(path, q.build(altered, "20261116", 0, 1))
        with pytest.raises(ValueError, match="evidence_drift|control_drift"):
            q.replay(path)
