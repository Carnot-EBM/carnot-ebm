"""REQ-REPORT-8109 and REQ-VERIFY-8109: private evidence cannot become generalization."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import v701_capstone as c
from carnot.reporting import v701_capstone_reduction as r
from carnot.reporting.current_work_receipt import atomic_json


def fixture():
    """Complete private inputs test accounting without borrowing live evidence."""
    tasks = [
        dict(
            id=f"exp{n}-fixture",
            deliverable=f"results/experiment_{n}_fixture.json",
            prior_failures=[],
            gated_on=[],
        )
        for n in range(8097, 8110)
    ]
    dispositions = [
        dict(
            task_id=t["id"],
            source_id=t["id"],
            unit_id=t["id"],
            arm="task_disposition",
            condition="terminal",
            issued_state="complete_null_fixture",
            metric="authenticated_task",
            numerator=1,
            denominator=1,
            status="completed",
            eligible=True,
            excluded=False,
            failed=False,
            completed=True,
            censored=False,
            exclusion_reason=None,
            honest_verdict="complete_null_fixture",
            verdict_class="null",
            path=None,
            sha256=None,
        )
        for t in tasks[:-1]
    ]
    rows = [
        dict(
            source_id=str(i),
            slot=i + 1,
            label=i % 2,
            status="completed",
            treatment_cost=0.2,
            control_cost=0.2,
            brier_increase=0,
            beneficial=False,
            extra_false_accept=False,
            equality_control=True,
            other_control_increase=0,
            retention_cost_increase=0,
            retention_brier_increase=0,
        )
        for i in range(192)
    ]
    primaries = {t["id"]: {} for t in tasks[:-1]}
    primaries[tasks[4]["id"]] = dict(decision_rows=rows[:128])
    primaries[tasks[7]["id"]] = dict(decision_rows=rows, retention_rows=rows[:64])
    return dict(
        tasks=tasks,
        dispositions=dispositions,
        primaries=primaries,
        references=[],
        failures=[],
        preconditions=[],
        independent_reductions={},
    )


def test_null_and_missing():
    """SCENARIO-REPORT-8109-PRIVATE: null completes; upstream absence blocks."""
    data = fixture()
    value = r.reduce(data)
    assert value["verdict_class"] == "null"
    assert value["completed_count"] == 13 and value["h1_development_signal_score"] == 0
    assert value["independent_generalization_score"] == 0
    assert value["scope_reduction_compliance"]["next_evidence_decision"].startswith("retire")
    assert len(value["task_dispositions"]) == 13
    data["dispositions"][4].update(eligible=False, excluded=True, verdict_class="blocked")
    data["primaries"][data["tasks"][4]["id"]] = {}
    value = r.reduce(data)
    assert value["verdict_class"] == "blocked" and value["H1"]["status"] == "blocked"
    assert value["scope_reduction_compliance"]["next_evidence_decision"].startswith("resolve")


def test_source_masks_and_harms():
    """SCENARIO-VERIFY-8109-REDUCTION: seeds and harmed decisions earn no benefit."""
    rows = fixture()["primaries"]["exp8101-fixture"]["decision_rows"]
    result = r.hypothesis(rows + deepcopy(rows), "H1")
    assert result["source_count"] == 128
    rows[0]["status"] = "excluded"
    assert r.hypothesis(rows, "H1")["excluded_count"] == 1
    for row in rows:
        row.update(control_cost=0.5, beneficial=True)
    assert r.hypothesis(rows, "H1")["development_signal_score"] == 1
    rows[1]["extra_false_accept"] = True
    assert r.hypothesis(rows, "H1")["development_signal_score"] == 0
    rows[1]["equality_control"] = False
    assert r.hypothesis(rows, "H1")["status"] == "blocked"
    assert r.hypothesis([], "H2")["status"] == "blocked"


def test_retirement_scope():
    """REQ-REPORT-8109: exact repeats retire only authenticated scientific configurations."""
    data = fixture()
    task = data["tasks"][0]
    task["prior_failures"] = [
        dict(
            experiment_id="exp1",
            verdict="complete_null_fixture",
            addressed_by="changed configuration",
            retire_if_same_verdict=True,
        )
    ]
    value = r.reduce(data)
    assert value["retirement_candidates"][0]["same_verdict"]
    assert value["retirement_candidates"][0]["retire_exact_configuration"]
    data["dispositions"][0].update(verdict_class="blocked", eligible=False)
    assert not r.reduce(data)["retirement_candidates"][0]["retire_exact_configuration"]


def test_replay_and_mutations(tmp_path):
    """SCENARIO-REPORT-8109-PRIVATE: exact bytes and reduced fields bind the artifact."""
    inputs = tmp_path / "inputs.json"
    atomic_json(inputs, fixture())
    value = r.reduce(fixture())
    value.update(
        replay_input_reference=c.reference(inputs),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        validation_receipts=[],
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert c.replay(path)["passed"]
    value["completed_count"] = 99
    atomic_json(path, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        c.replay(path)
    value = r.reduce(fixture())
    value.update(
        replay_input_reference=c.reference(inputs),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        validation_receipts=[],
    )
    atomic_json(path, value)
    inputs.write_text("{}")
    with pytest.raises(ValueError):
        c.replay(path)


def test_h2_blocks_and_retention():
    """REQ-VERIFY-8109: original windows and finite retention keep separate denominators."""
    data = fixture()
    rows = data["primaries"]["exp8104-fixture"]["decision_rows"]
    for index, row in enumerate(rows):
        row["slot"] = index + 65
    value = r.reduce(data)
    assert value["H2"]["status"] == "completed"
    assert value["H2"]["block_sensitivities"]["16"]["valid_draws"] == 10000
    assert value["H2"]["retention"]["source_count"] == 64
    for row in rows:
        row["control_cost"] = 0.6
        row["beneficial"] = True
    assert r.reduce(data)["h2_development_signal_score"] == 1
    assert r.reduce(data)["scope_reduction_compliance"]["next_evidence_decision"].startswith(
        "acquire"
    )
    rows[0]["retention_cost_increase"] = 0.1
    assert r.reduce(data)["h2_development_signal_score"] == 0


def test_frozen_commands_and_qualification(tmp_path):
    """REQ-REPORT-8109: current checks cover only owned code and keep global health separate."""
    specs = c.commands(tmp_path)
    assert any(s.name == "coverage100" for s in specs)
    assert specs[-1].scope == "repository_health"
    value = r.reduce(fixture())
    c.qualify(value, True)
    assert value["capstone_ready_score"] == 1
    c.qualify(value, False)
    assert value["verdict_class"] == "disqualified" and value["capstone_ready_score"] == 0


def test_authority_load_and_exclusions(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8109-PRIVATE: exact gates and primitive failures exclude a branch."""
    data = fixture()
    data["tasks"][1]["gated_on"] = [
        dict(upstream=data["tasks"][0]["id"], artifact_field="ready", op="==", value=1)
    ]
    monkeypatch.setattr(c, "authorities", lambda root: (data["tasks"], []))
    rows = deepcopy(data["dispositions"])
    for task, row in zip(data["tasks"], rows):
        path = tmp_path / task["deliverable"]
        atomic_json(path, {})
        row.update(path=str(path), primary_present=True)
    monkeypatch.setattr(c.previous, "collect", lambda root, tasks: (rows, [], []))

    def audit(value, number):
        if number == 8098:
            raise ValueError("tampered primitive")
        return dict(available=False)

    monkeypatch.setattr(c, "primitive_audit", audit)
    missing = tmp_path / "missing.md"
    present = tmp_path / "present.json"
    atomic_json(present, {})
    monkeypatch.setattr(c, "NAMED", ["missing.md", "present.json"])
    got = c.load(tmp_path, tmp_path / "raw")
    assert any(g["observed"] is None for g in got["failures"])
    assert any(g["path"] == str(missing) and g["observed"] is False for g in got["failures"])
    assert got["dispositions"][1]["excluded"]
    assert len(list((tmp_path / "raw/custody").glob("*"))) == 1
    monkeypatch.setattr(
        c, "authorities", lambda root: (_ for _ in ()).throw(ValueError("immutable drift"))
    )
    monkeypatch.setattr(c, "parse_design", lambda *a, **kw: ([], data["tasks"]))
    monkeypatch.setattr(c.previous, "collect", lambda root, tasks: (deepcopy(rows), [], []))
    assert c.load(tmp_path, tmp_path / "raw2")["failures"][0]["observed"] == "immutable drift"


def test_authority_exact_bytes(tmp_path, monkeypatch):
    """REQ-REPORT-8109: a changed immutable invocation is never substituted silently."""
    import yaml

    data = fixture()
    tasks = data["tasks"]
    for task in tasks:
        task.update(title="fixture", phase=1)
    digest = c.tasks_digest(tasks)
    active = tmp_path / "active.bin"
    active.write_text(yaml.safe_dump(dict(tasks=tasks, milestone="2026.10.701")))
    design = tmp_path / "design.bin"
    design.write_text("Canonical full-task SHA-256: `" + digest + "`")
    atomic_json(
        tmp_path / c.INPUT,
        dict(
            authority_snapshots={
                "active": dict(snapshot_path=str(active), sha256=c.reference(active)["sha256"]),
                "design": dict(snapshot_path=str(design), sha256=c.reference(design)["sha256"]),
            },
            canonical_tasks_sha256=digest,
        ),
    )
    table = [
        dict(order=i + 1, **{k: t[k] for k in ("id", "title", "phase", "deliverable")})
        for i, t in enumerate(tasks)
    ]
    monkeypatch.setattr(c, "parse_design", lambda *a, **kw: (table, tasks))
    assert c.authorities(tmp_path)[0] == tasks
    tasks[0]["title"] = "changed"
    with pytest.raises(ValueError, match="immutable_authority_drift"):
        c.authorities(tmp_path)


def test_primitive_audit(tmp_path, monkeypatch):
    """REQ-REPORT-8109: each qualified reader receives rows or an authenticated shard."""
    from types import SimpleNamespace

    module = SimpleNamespace(
        reduction=lambda v: dict(passed=True),
        reduced_adapter=lambda v: dict(n=len(v)),
        reduce_rows=lambda v: dict(passed=True),
        reduce=lambda v: dict(passed=True),
    )
    monkeypatch.setattr(c.importlib, "import_module", lambda name: module)
    refs = []
    for name in ["evidence.json", "primitive_rows.json", "replay_inputs.json"]:
        path = tmp_path / name
        atomic_json(path, {"primitive": 1})
        refs.append(c.reference(path))
    value = dict(rows=[], raw_shard_hashes=refs)
    for number in [8098, 8102, 8105, 8106, 8108]:
        assert c.primitive_audit(value, number)["available"]
    assert not c.primitive_audit(value, 8097)["available"]


def test_private_real_cli(tmp_path):
    """SCENARIO-REPORT-8109-PRIVATE: real outside-checkout success, block, mutation and cold exits."""
    import os
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    data = fixture()
    for i, row in enumerate(data["primaries"]["exp8104-fixture"]["decision_rows"]):
        row["slot"] = i + 65
    input_path = tmp_path / "input.json"
    atomic_json(input_path, data)
    output = tmp_path / "experiment_8109_private.json"
    py = str(c.ROOT / ".venv/bin/python")
    prefix = [py]
    config = os.environ.get("CARNOT_8109_COVERAGE_CONFIG")
    if config:
        prefix += ["-m", "coverage", "run", "--parallel-mode", "--rcfile=" + config]
    command = [*prefix, str(c.ROOT / c.CLI)]
    receipts = []

    def run(name, args, expected):
        spec = CommandSpec(name, tuple(command + args), "private_e2e", 30)
        receipt = run_commands(tmp_path, [spec], log_dir=tmp_path / "logs", heartbeat_s=5)[0]
        receipts.append(dict(receipt, expected_exit=expected))
        assert receipt["exit_code"] == expected
        return receipt

    run("private_success", ["--worker-input", str(input_path), "--output", str(output)], 0)
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "null"
    run("cold_success", ["--cold-replay", str(output)], 0)
    changed = deepcopy(value)
    changed["eligible_count"] = 999
    mutation = tmp_path / "mutation.json"
    atomic_json(mutation, changed)
    run("mutation_rejected", ["--cold-replay", str(mutation)], 1)
    data["failures"] = [dict(check="upstream_exists", observed=False)]
    atomic_json(input_path, data)
    run("private_blocked", ["--worker-input", str(input_path), "--output", str(output)], 0)
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    run(
        "missing_input",
        ["--worker-input", str(tmp_path / "absent.json"), "--output", str(output)],
        1,
    )
    destination = os.environ.get("CARNOT_8109_CLI_RECEIPTS")
    if destination:
        atomic_json(Path(destination), dict(receipts=receipts))


def test_parent_receipts_and_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8109: failed owned commands disqualify, while failed health stays diagnostic."""
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec

    data = fixture()
    for i, row in enumerate(data["primaries"]["exp8104-fixture"]["decision_rows"]):
        row["slot"] = i + 65
    monkeypatch.setattr(c, "load", lambda root, raw: data)
    specs = [
        CommandSpec("publication_gate", ("fixture",), "publication"),
        CommandSpec("owned", ("fixture",), "owned"),
        CommandSpec("health", ("fixture",), "repository_health"),
    ]
    monkeypatch.setattr(c, "commands", lambda p: specs)
    failed = False
    child_failure = False
    child_drift = False

    def run(root, commands, **kwargs):
        result = []
        for spec in commands:
            log = kwargs["log_dir"] / (spec.name + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(log, dict(paper_ready=False, unmet_gates=["G2"]))
            if spec.name == "normal_reduction_exit":
                target = Path(spec.argv[-1])
                v = r.reduce(data)
                if child_drift:
                    v["completed_count"] = 99
                atomic_json(target, v)
            result.append(
                dict(
                    name=spec.name,
                    scope=spec.scope,
                    passed=not (
                        (failed and spec.scope == "owned")
                        or (child_failure and spec.scope == "measurement")
                    ),
                    log_path=str(log),
                    log_sha256=c.reference(log)["sha256"],
                    exit_code=0,
                )
            )
        return result

    monkeypatch.setattr(c, "run_commands", run)
    published = []

    def publish(path, value, validator):
        published.append(deepcopy(value))
        atomic_json(path, value)
        return dict(primary_path=str(path), primary_sha256=c.reference(path)["sha256"])

    monkeypatch.setattr(c, "publish_primary", publish)
    output = tmp_path / "experiment_8109_private.json"
    assert c.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert published[-1]["required_checks_passed"] and published[-1]["verdict_class"] == "null"
    failed = True
    assert c.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert published[-1]["verdict_class"] == "disqualified"
    assert c.replay(output)["passed"]
    log = Path(published[-1]["validation_receipts"][0]["log_path"])
    log.write_text("changed")
    # Drop shard references to exercise the distinct validation log operand.
    value = deepcopy(published[-1])
    value["raw_shard_hashes"] = []
    atomic_json(output, value)
    with pytest.raises(ValueError, match="validation_receipt_drift"):
        c.replay(output)
    child_failure = True
    assert c.main(["--root", str(tmp_path), "--output", str(output)]) == 1
    child_failure = False
    child_drift = True
    assert c.main(["--root", str(tmp_path), "--output", str(output)]) == 1
    assert c.terminal(output)["passed"]


def test_successful_gate_is_not_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8109: passed hardware checks remain accepted evidence."""
    data = fixture()
    rows = deepcopy(data["dispositions"])
    success = dict(
        check="primitive_sha256", field="sha256", passed=True, expected="x", observed="x"
    )
    wrong = dict(success, artifact_field="sha256", passed=False)
    for task, row in zip(data["tasks"], rows):
        path = tmp_path / task["deliverable"]
        atomic_json(path, dict(gate_check_summary=[success]))
        row.update(
            path=str(path),
            primary_present=True,
            gate_check_summary=[wrong],
            eligible=False,
            excluded=True,
        )
    monkeypatch.setattr(c, "authorities", lambda root: (data["tasks"], []))
    monkeypatch.setattr(c, "NAMED", [])
    monkeypatch.setattr(c.previous, "collect", lambda root, tasks: (rows, [], [wrong] * 12))
    monkeypatch.setattr(c, "primitive_audit", lambda value, number: dict(available=False))
    got = c.load(tmp_path, tmp_path / "raw")
    assert not got["failures"] and all(row["eligible"] for row in got["dispositions"])


def test_flagged_data_earns_no_benefit():
    """REQ-VERIFY-8109: invalid inputs remain visible and cannot earn developmental credit."""
    data = fixture()
    rows = data["primaries"]["exp8101-fixture"]["decision_rows"]
    for row in rows:
        row.update(control_cost=0.8, beneficial=True)
    data["dispositions"][4].update(
        eligible=False, excluded=True, exclusion_reason="flagged_adversarial"
    )
    assert r.reduce(data)["h1_development_signal_score"] == 0


def test_cold_reduction_rejects_changed_primitive_equation(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8109-REDUCTION: producer totals cannot substitute for primitive equations."""
    data = fixture()
    data["independent_reductions"]["exp8098-fixture"] = dict(available=True, result=1)
    inputs = tmp_path / "inputs.json"
    atomic_json(inputs, data)
    value = r.reduce(data)
    value.update(
        replay_input_reference=c.reference(inputs),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        validation_receipts=[],
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    monkeypatch.setattr(c, "primitive_audit", lambda *a: dict(available=True, result=2))
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        c.replay(path)


def test_conductor_gate_record_shape(tmp_path, monkeypatch):
    """REQ-REPORT-8109: conductor block summaries are objects rather than scientific row lists."""
    data = fixture()
    rows = deepcopy(data["dispositions"])
    for task, row in zip(data["tasks"], rows):
        path = tmp_path / task["deliverable"]
        atomic_json(path, dict(gate_check_summary={"failed_upstream": "exp8099"}))
        row.update(
            path=str(path),
            primary_present=False,
            gate_check_summary=[],
            eligible=False,
            excluded=True,
        )
    monkeypatch.setattr(c, "authorities", lambda root: (data["tasks"], []))
    monkeypatch.setattr(c, "NAMED", [])
    monkeypatch.setattr(c.previous, "collect", lambda root, tasks: (rows, [], []))
    assert not c.load(tmp_path, tmp_path / "raw")["dispositions"][0]["eligible"]
