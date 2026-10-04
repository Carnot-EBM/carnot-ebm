"""REQ-REPORT-8122 and REQ-VERIFY-8122: private branch and custody controls."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import v702_capstone as c
from carnot.reporting import v702_capstone_reduction as r
from carnot.reporting.current_work_receipt import atomic_json
from test_v701_capstone_8109 import fixture as historical_fixture


def fixture():
    """Use private source pairs so historical science is never rewritten by tests."""
    data = historical_fixture()
    for i, task in enumerate(data["tasks"]):
        task["id"] = f"exp{8110 + i}-fixture"
        task["deliverable"] = f"results/experiment_{8110 + i}_fixture.json"
    old = list(data["primaries"].values())
    data["primaries"] = {t["id"]: {} for t in data["tasks"][:-1]}
    data["primaries"][data["tasks"][5]["id"]] = old[4]
    learning = old[7]
    for i, row in enumerate(learning["decision_rows"]):
        row["slot"] = i + 65
    data["primaries"][data["tasks"][7]["id"]] = learning
    for task, row in zip(data["tasks"], data["dispositions"]):
        row.update(task_id=task["id"], unit_id=task["id"])
    data["prior_evidence"] = {}
    return data


def test_independent_null_missing_positive():
    """SCENARIO-VERIFY-8122-REDUCTION: a fitted block preserves qualified finite H2."""
    data = fixture()
    value = r.reduce(data)
    assert value["verdict_class"] == "null"
    assert value["h1_decision_benefit_score"] == value["h2_learning_benefit_score"] == 0
    assert len(value["task_dispositions"]) == 13
    assert all("source_cluster_id" in row for row in value["rows"])
    data["dispositions"][5].update(eligible=False, excluded=True)
    learning = data["primaries"][data["tasks"][7]["id"]]
    for row in learning["decision_rows"]:
        row.update(control_cost=0.6, beneficial=True)
    value = r.reduce(data)
    assert value["verdict_class"] == "blocked"
    assert value["h2_learning_benefit_score"] == 1
    assert value["h1_decision_benefit_score"] == 0
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert not any(g["closed"] for g in value["gap_decisions"].values())
    data["dispositions"][7].update(eligible=False, excluded=True)
    assert r.reduce(data)["h2_learning_benefit_score"] == 0


def test_service_retirement_and_owned_failure():
    """REQ-REPORT-8122: environmental blocks retire no scientific family."""
    data = fixture()
    task = data["tasks"][0]
    task["prior_failures"] = [
        dict(
            experiment_id="exp1",
            verdict="complete_null_fixture",
            addressed_by="changed numerical mechanism",
            retire_if_same_verdict=True,
        )
    ]
    data["prior_evidence"]["exp1"] = dict(path="/tmp/missing", sha256=None)
    data["primaries"][data["tasks"][9]["id"]] = dict(
        complete_service_ready_score=1, host_service_ready_score=1
    )
    value = r.reduce(data)
    assert value["complete_service_evidence_score"] == 1
    assert value["retirement_candidates"][0]["same_verdict"]
    assert not value["retirement_candidates"][0]["retire_exact_configuration"]
    c.qualify(value, True)
    assert value["capstone_execution_ready_score"] == 1
    c.qualify(value, False)
    assert value["verdict_class"] == "disqualified" and value["capstone_ready_score"] == 0


def test_cold_reduction_mutation(tmp_path):
    """SCENARIO-REPORT-8122-PRIVATE: both bytes and aggregate changes fail replay."""
    data = fixture()
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
    assert c.replay(path)["passed"]
    value["completed_count"] = 99
    atomic_json(path, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        c.replay(path)
    inputs.write_text("{}")
    with pytest.raises(ValueError, match="hash_drift"):
        c.replay(path)


def authority_fixture(tmp_path, design_valid=True):
    """Private full-task bytes exercise authority without following the live roadmap."""
    import yaml
    from carnot.reporting import v702_capstone_inputs as i
    from carnot.reporting.v685_authority_lifecycle import tasks_digest

    data = fixture()
    tasks = data["tasks"]
    for t in tasks:
        t.update(title="private fixture", phase=1)
    active = tmp_path / "active.yaml"
    active.write_text(yaml.safe_dump(dict(milestone="2026.10.702", tasks=tasks)))
    design = tmp_path / "design.md"
    table = "\n".join(
        f"| {n + 1} | {t['id']} | {t['title']} | 1 | {t['deliverable']} |"
        for n, t in enumerate(tasks)
    )
    design.write_text(
        "## Exact task contract\n"
        + table
        + "\n<!-- V702_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(dict(milestone="2026.10.702", tasks=tasks))
        + "\n```\n"
        if design_valid
        else "missing contract"
    )
    snapshots = {
        role: dict(snapshot_path=str(path), sha256=c.reference(path)["sha256"])
        for role, path in [("active", active), ("saved_staged", active), ("design", design)]
    }
    atomic_json(
        tmp_path / i.INPUT,
        dict(authority_snapshots=snapshots, canonical_tasks_sha256=tasks_digest(tasks)),
    )
    return tasks, active, design


def test_authority_complete_missing_tampered(tmp_path):
    """SCENARIO-REPORT-8122-PRIVATE: activation bytes remain separate from malformed prose."""
    import yaml
    from carnot.reporting import v702_capstone_inputs as i

    tasks, active, design = authority_fixture(tmp_path)
    assert i.authorities(tmp_path)[0] == tasks
    design.write_text("missing contract")
    value = json.loads((tmp_path / i.INPUT).read_bytes())
    value["authority_snapshots"]["design"]["sha256"] = c.reference(design)["sha256"]
    atomic_json(tmp_path / i.INPUT, value)
    assert i.authorities(tmp_path)[2][0]["check"] == "design_exact_task_contract"
    active.write_text(yaml.safe_dump(dict(milestone="wrong", tasks=tasks)))
    with pytest.raises(ValueError):
        i.authorities(tmp_path)
    value["authority_snapshots"]["active"]["sha256"] = c.reference(active)["sha256"]
    value["authority_snapshots"]["saved_staged"]["sha256"] = c.reference(active)["sha256"]
    atomic_json(tmp_path / i.INPUT, value)
    with pytest.raises(ValueError, match="activation_contract_drift"):
        i.authorities(tmp_path)


def test_load_exact_operands_and_custody(tmp_path, monkeypatch):
    """REQ-REPORT-8122: failed primitives, missing paths and scalar gates are explicit."""
    from carnot.reporting import v702_capstone_inputs as i

    data = fixture()
    tasks = data["tasks"]
    tasks[1]["gated_on"] = [dict(upstream=tasks[0]["id"], artifact_field="ready", op="==", value=1)]
    atomic_json(tmp_path / "present.json", {})
    monkeypatch.setattr(i, "NAMED", ["present.json", "missing.json"])
    monkeypatch.setattr(i, "authorities", lambda root: (tasks, [], []))
    tasks[0]["prior_failures"] = [
        dict(
            experiment_id="exp1-fixture",
            verdict="complete_null_fixture",
            addressed_by="private changed mechanism",
            retire_if_same_verdict=True,
        )
    ]
    for task in tasks[:-1]:
        atomic_json(tmp_path / task["deliverable"], dict(gate_check_summary=[]))

    def collect(root, chosen):
        row = deepcopy(data["dispositions"][tasks.index(chosen[0])])
        row["path"] = str(root / chosen[0]["deliverable"])
        return [row], [c.reference(Path(row["path"]))], []

    monkeypatch.setattr(i.previous, "collect", collect)

    def audit(value, number):
        if number == 8111:
            raise ValueError("invalid private primitive")
        return dict(available=False)

    monkeypatch.setattr(i, "primitive_audit", audit)
    result = i.load(tmp_path, tmp_path / "raw")
    assert not result["dispositions"][1]["eligible"]
    assert any(g["observed"] is None and g["artifact_field"] == "ready" for g in result["failures"])
    assert any(g["observed"] is False for g in result["failures"])
    assert len(list((tmp_path / "raw/custody").glob("*"))) > 0
    atomic_json(
        tmp_path / "results/experiment_1_fixture.json", dict(honest_verdict="complete_null_fixture")
    )
    monkeypatch.setattr(i, "ROOT", tmp_path)
    atomic_json(tmp_path / i.INPUT, dict(task_contract=tasks))

    def missing_authority(root):
        raise FileNotFoundError("missing private activation")

    monkeypatch.setattr(i, "authorities", missing_authority)
    result = i.load(tmp_path, tmp_path / "raw-blocked")
    assert result["failures"][0]["check"] == "activation_contract"
    assert result["prior_evidence"]["exp1-fixture"]["honest_verdict"] == "complete_null_fixture"


def test_primitive_reductions(tmp_path, monkeypatch):
    """REQ-VERIFY-8122: operands come from hashed raw files, including invalid evidence."""
    from types import SimpleNamespace
    from carnot.reporting import v702_capstone_inputs as i

    path = tmp_path / "primitive_rows.json"
    atomic_json(path, dict(rows=[]))
    value = dict(raw_shard_hashes=[c.reference(path)], reductions={}, reduction={})
    monkeypatch.setattr(
        i.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            reduce_rows=lambda operand: {}, reductions=lambda operand: {}, reduce=lambda operand: {}
        ),
    )
    for n in (8111, 8116, 8118, 8119):
        assert i.primitive_audit(value, n)["available"]
    path = tmp_path / "replay_inputs.json"
    atomic_json(path, {})
    value["raw_shard_hashes"] = [c.reference(path)]
    assert i.primitive_audit(value, 8121)["available"]
    assert not i.primitive_audit(value, 8110)["available"]
    path = tmp_path / "primitive_rows.json"
    value["raw_shard_hashes"] = [c.reference(path)]
    for n, key in [(8116, "reductions"), (8119, "reduction")]:
        bad = dict(value, **{key: dict(changed=True)})
        with pytest.raises(ValueError, match="reduction_drift"):
            i.primitive_audit(bad, n)


def test_commands_terminal_and_main(tmp_path, monkeypatch):
    """REQ-REPORT-8122: current exits and validation, not historic receipts, bind readiness."""
    specs = c.commands(tmp_path)
    assert specs[-1].scope == "repository_health"
    assert all("test_primary_publication.py" not in a for s in specs for a in s.argv)
    data = fixture()
    monkeypatch.setattr(c.inputs, "load", lambda root, raw: data)
    monkeypatch.setattr(c, "commands", lambda scratch: specs[:1] + specs[1:2])
    fail = [False]

    def execute(root, commands, log_dir, **kwargs):
        log_dir.mkdir(parents=True, exist_ok=True)
        receipts = []
        for s in commands:
            if s.name == "normal_reduction_exit":
                assert c.main(list(s.argv[3:])) == 0
            log = log_dir / (s.name + ".log")
            log.write_text(json.dumps(dict(paper_ready=True, unmet_gates=[])))
            receipts.append(
                dict(
                    name=s.name,
                    scope=s.scope,
                    command_argv=list(s.argv),
                    log_path=str(log),
                    log_sha256=c.reference(log)["sha256"],
                    exit_code=0,
                    timed_out=False,
                    passed=not fail[0],
                    duration_s=0.01,
                )
            )
        return receipts

    monkeypatch.setattr(c, "run_commands", execute)

    def publish(path, value, validator):
        atomic_json(path, value)
        assert validator(path)["passed"]
        return dict(primary_path=str(path), primary_sha256=c.reference(path)["sha256"])

    monkeypatch.setattr(c, "publish_primary", publish)
    output = tmp_path / "complete.json"
    assert c.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["capstone_execution_ready_score"] == 1
    assert c.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_bytes())
    value["validation_receipts"][0]["log_sha256"] = "changed"
    atomic_json(output, value)
    assert c.main(["--cold-replay", str(output)]) == 1
    fail[0] = True
    assert not c.terminal(output)["passed"]
    assert c.main(["--output", str(tmp_path / "failed-child.json")]) == 1
    fail[0] = False

    def reject(path, value, validator):
        raise ValueError("candidate_rejected")

    monkeypatch.setattr(c, "publish_primary", reject)
    assert c.main(["--output", str(tmp_path / "rejected.json")]) == 1


def test_real_private_cli_success_blocked_mutation(tmp_path):
    """SCENARIO-REPORT-8122-PRIVATE: outside-checkout script success, block and mutation exits."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    config = env.get("CARNOT_8122_COVERAGE_CONFIG")
    prefix = (
        [sys.executable, "-m", "coverage", "run", "--parallel-mode", "--rcfile=" + config]
        if config
        else [sys.executable]
    )
    for blocked in (False, True):
        data = fixture()
        if blocked:
            data["dispositions"][5].update(eligible=False, excluded=True)
        inputs = tmp_path / f"inputs-{blocked}.json"
        atomic_json(inputs, data)
        output = tmp_path / f"output-{blocked}.json"
        done = subprocess.run(
            [*prefix, str(c.ROOT / c.CLI), "--worker-input", str(inputs), "--output", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("blocked" if blocked else "null")
        assert value["model_invocation_counts"]["model_loads_attempted"] == 0
        good = subprocess.run(
            [*prefix, str(c.ROOT / c.CLI), "--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert good.returncode == 0, good.stdout + good.stderr
        value["completed_count"] = 99
        atomic_json(output, value)
        bad = subprocess.run(
            [*prefix, str(c.ROOT / c.CLI), "--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert bad.returncode == 1 and b"reduction_drift" in bad.stdout


def test_authority_staged_and_design_disagreement(tmp_path):
    """REQ-REPORT-8122: independent authority roles cannot borrow each other's bytes."""
    import yaml
    from carnot.reporting import v702_capstone_inputs as i

    tasks, active, design = authority_fixture(tmp_path)
    value = json.loads((tmp_path / i.INPUT).read_bytes())
    staged = tmp_path / "wrong-stage.yaml"
    staged.write_text(yaml.safe_dump(dict(tasks=[])))
    value["authority_snapshots"]["saved_staged"] = dict(
        snapshot_path=str(staged), sha256=c.reference(staged)["sha256"]
    )
    atomic_json(tmp_path / i.INPUT, value)
    with pytest.raises(ValueError, match="staged_contract_drift"):
        i.authorities(tmp_path)
    tasks, active, design = authority_fixture(tmp_path)
    design.write_text(design.read_text().replace("private fixture", "changed title", 1))
    value = json.loads((tmp_path / i.INPUT).read_bytes())
    value["authority_snapshots"]["design"]["sha256"] = c.reference(design)["sha256"]
    atomic_json(tmp_path / i.INPUT, value)
    # A machine task edit must fail independently of a visible table edit.
    assert i.authorities(tmp_path)[2][0]["observed"] == "design_contract_drift"
    text = design.read_text().replace('"title": "private fixture"', '"title": "wrong"', 1)
    design.write_text(text)
    value["authority_snapshots"]["design"]["sha256"] = c.reference(design)["sha256"]
    atomic_json(tmp_path / i.INPUT, value)
    assert i.authorities(tmp_path)[2][0]["observed"] == "design_contract_drift"


def test_primitive_cold_replay_and_oracle_scope(tmp_path, monkeypatch):
    """REQ-VERIFY-8122: cold equations reject altered aggregates; oracle benefit is circular."""
    data = fixture()
    for row in data["primaries"][data["tasks"][5]["id"]]["decision_rows"]:
        row.update(control_cost=0.6, beneficial=True)
    data["verifier_is_oracle"] = True
    assert r.reduce(data)["verdict_class"] == "circular_positive"
    data["independent_reductions"][data["tasks"][1]["id"]] = dict(
        available=True,
        primitive_reference=dict(path="private", sha256="private"),
        result=dict(count=1),
    )
    path = tmp_path / "inputs.json"
    atomic_json(path, data)
    value = r.reduce(data)
    value.update(
        replay_input_reference=c.reference(path),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        validation_receipts=[],
    )
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    monkeypatch.setattr(c.inputs, "primitive_audit", lambda value, n: dict(result=dict(count=1)))
    assert c.replay(candidate)["passed"]
    monkeypatch.setattr(c.inputs, "primitive_audit", lambda value, n: dict(result=dict(count=2)))
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        c.replay(candidate)
