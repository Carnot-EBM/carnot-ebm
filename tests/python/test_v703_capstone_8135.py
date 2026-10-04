"""REQ-REPORT-8135 and REQ-VERIFY-8135: private independent accounting controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from carnot.reporting import v703_capstone as c
from carnot.reporting import v703_capstone_inputs as i
from carnot.reporting import v703_capstone_reduction as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary


def fixture(tmp_path):
    """Known private pairs exercise evidence handling without changing natural results."""
    root = tmp_path / "private"
    receipt = json.loads((c.ROOT / i.INPUT).read_text())
    tasks = receipt["task_contract"]
    snaps = {}
    for role in ("active", "design"):
        source = Path(receipt["authority_snapshots"][role]["snapshot_path"])
        path = root / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        snaps[role] = dict(snapshot_path=str(path), sha256=i.reference(path)["sha256"])
    atomic_json(root / i.INPUT, dict(receipt, authority_snapshots=snaps))
    for task in tasks[:-1]:
        n = int(task["id"].split("-")[0][3:])
        p = root / task["deliverable"]
        side = p.parent / "raw" / p.stem / "terminal.json"
        value = dict(
            experiment_id=n,
            task_id=task["id"],
            honest_verdict="complete_null_private",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(side),
            raw_shard_hashes=[],
            gate_check_summary=[],
            MODEL_SPECS=[],
            trained_head_specs=[],
        )
        value.update(
            {
                g["artifact_field"]: 1
                for t in tasks
                for g in t["gated_on"]
                if g["upstream"] == task["id"]
            }
        )
        if n == 8123:
            value.update(
                task_contract=tasks,
                authority_snapshots=snaps,
                canonical_tasks_sha256=receipt["canonical_tasks_sha256"],
            )
        if n in (8128, 8131):
            scientific_input = data_fixture()["primaries"][task["id"]]
            value.update(scientific_input)
        pub = publish_primary(p, value, lambda _: dict(passed=True))
        atomic_json(side, dict(publication=pub))
    return root, tasks


def data_fixture():
    """Pairs repeat sources by seed; independence must still count each source once."""
    tasks = json.loads((c.ROOT / i.INPUT).read_text())["task_contract"]
    rows = [
        dict(
            unit_id=t["id"],
            task_id=t["id"],
            eligible=True,
            excluded=False,
            numerator=1,
            denominator=1,
            status="completed",
            failed=False,
            honest_verdict="complete_null_private",
            verdict_class="null",
            disposition="legitimate_null",
            exclusion_reason=None,
            gate_check_summary=[],
        )
        for t in tasks[:-1]
    ]
    primaries = {t["id"]: {} for t in tasks[:-1]}
    for index, count, start in [(5, 128, 0), (8, 192, 65)]:
        pairs = [
            dict(
                source_id=f"s{j}",
                slot=j + start,
                label=j % 2,
                status="completed",
                control_cost=0.2,
                treatment_cost=0.2,
                beneficial=False,
                extra_false_accept=False,
                brier_increase=0,
                equality_control=True,
                other_control_increase=0,
            )
            for j in range(count)
        ]
        primaries[tasks[index]["id"]] = dict(decision_rows=pairs)
    primaries[tasks[8]["id"]]["retention_rows"] = [
        dict(
            source_id=f"r{j}",
            label=j % 2,
            status="completed",
            retention_cost_increase=0,
            retention_brier_increase=0,
            extra_false_accept=False,
        )
        for j in range(64)
    ]
    return dict(
        tasks=tasks,
        dispositions=rows,
        primaries=primaries,
        failures=[],
        references=[],
        preconditions=[],
        independent_reductions={},
        prior_evidence={},
        authority=dict(activated=True),
        verifier_is_oracle=False,
    )


def test_branch_null_positive_disqualified():
    """SCENARIO-VERIFY-8135-REDUCTION: H2 cannot borrow or lose H1's conclusion."""
    data = data_fixture()
    value = r.reduce(data)
    assert value["verdict_class"] == "null"
    assert len(value["rows"]) == value["completed_count"] == 13
    assert value["H1"]["source_count"] == 128
    assert value["multiplicity"]["family"] == ["H1", "H2"]
    data["dispositions"][5].update(eligible=False, excluded=True)
    for row in data["primaries"][data["tasks"][8]["id"]]["decision_rows"]:
        row.update(control_cost=0.6, beneficial=True)
    value = r.reduce(data)
    assert value["verdict_class"] == "blocked"
    assert value["h2_development_signal_score"] == 1
    assert value["h1_development_signal_score"] == 0
    data["dispositions"][8].update(eligible=False, verdict_class="disqualified")
    assert r.reduce(data)["h2_development_signal_score"] == 0
    assert value["independent_generalization_score"] == 0


def test_oracle_owned_and_retirement():
    """REQ-REPORT-8135: fixtures and owned failures never acquire natural credit."""
    data = data_fixture()
    for row in data["primaries"][data["tasks"][5]["id"]]["decision_rows"]:
        row.update(control_cost=0.6, beneficial=True)
    data["verifier_is_oracle"] = True
    value = r.reduce(data)
    assert value["verdict_class"] == "circular_positive"
    r.qualify(value, False)
    assert value["verdict_class"] == "disqualified"
    assert value["science_ready_score"] == value["capstone_execution_ready_score"] == 0
    assert all(not x["retire_exact_configuration"] for x in value["retirement_decisions"])


def test_authority_and_missing_inputs(tmp_path):
    """SCENARIO-REPORT-8135-PRIVATE: exact authority and absent operands remain distinct."""
    root, tasks = fixture(tmp_path)
    raw = tmp_path / "raw"
    data = i.load(root, raw)
    assert len(data["dispositions"]) == 12
    assert data["authority"]["activated"]
    (root / tasks[2]["deliverable"]).unlink()
    blocked = i.load(root, tmp_path / "missing")
    assert blocked["dispositions"][2]["disposition"] == "missing_primary"
    assert any(
        g["artifact_field"] == "primary_exists" and g["observed"] is False
        for g in blocked["failures"]
    )
    active = root / "active.bin"
    value = yaml.safe_load(active.read_text())
    value["tasks"][1]["prompt"] += " changed"
    active.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="hash"):
        i.authorities(root, tmp_path / "bad-authority")


def candidate(tmp_path, data=None):
    """A private replay candidate binds the same equations as a real primary."""
    data = data or data_fixture()
    operand = tmp_path / "inputs.json"
    atomic_json(operand, data)
    value = r.reduce(data)
    r.qualify(value, True)
    value.update(
        replay_input_reference=c.reference(operand),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        validation_receipts=[],
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    return path, value


def test_cold_replay_bytes_claims_logs(tmp_path):
    """SCENARIO-REPORT-8135-PRIVATE: rehashed bytes cannot conceal a changed claim."""
    path, value = candidate(tmp_path)
    assert c.replay(path)["passed"]
    value["completed_count"] = 90
    atomic_json(path, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        c.replay(path)
    value["completed_count"] = 13
    log = tmp_path / "log"
    log.write_text("normal exit")
    value["validation_receipts"] = [dict(log_path=str(log), log_sha256=c.reference(log)["sha256"])]
    atomic_json(path, value)
    assert c.replay(path)["passed"]
    log.write_text("altered")
    with pytest.raises(ValueError, match="validation_log"):
        c.replay(path)
    Path(value["replay_input_reference"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="input_hash"):
        c.replay(path)


def test_table_prompt_live_and_digest_mutations(tmp_path):
    """REQ-REPORT-8135: each authority role has independently checked bytes."""
    root, _ = fixture(tmp_path)
    receipt_path = root / i.INPUT
    receipt = json.loads(receipt_path.read_text())
    design = root / "design.bin"
    original = design.read_text()
    design.write_text(original.replace("| 1 |", "| 2 |", 1))
    receipt["authority_snapshots"]["design"]["sha256"] = i.reference(design)["sha256"]
    atomic_json(receipt_path, receipt)
    assert not i.authorities(root, tmp_path / "table-mutated")[1]["activated"]
    design.write_text(original)
    receipt["authority_snapshots"]["design"]["sha256"] = i.reference(design)["sha256"]
    atomic_json(receipt_path, receipt)
    active = yaml.safe_load((root / "active.bin").read_text())
    active["tasks"][2]["prompt"] += " altered"
    (root / "research-roadmap.yaml").write_text(yaml.safe_dump(active))
    assert not i.authorities(root, tmp_path / "live-mutated")[1]["activated"]
    receipt["canonical_tasks_sha256"] = "0" * 64
    atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="digest"):
        i.authorities(root, tmp_path / "wrong-digest")
    assert not i.load(root, tmp_path / "failed-authority")["authority"]["activated"]


def test_disqualified_alternate_and_terminal(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8135-PRIVATE: invalid science and conductor skips remain excluded."""
    root, tasks = fixture(tmp_path)
    path = root / tasks[2]["deliverable"]
    value = json.loads(path.read_bytes())
    value["flagged_adversarial"] = True
    atomic_json(path, value)
    data = i.load(root, tmp_path / "flagged")
    assert data["dispositions"][2]["disposition"] == "disqualified_measurement"
    value.update(flagged_adversarial=False, required_checks_passed=True, task_id="wrong")
    atomic_json(path, value)
    assert not i.load(root, tmp_path / "identity")["dispositions"][2]["qualified"]
    path.unlink()
    alternate = path.with_name("experiment_8125_conductor.json")
    atomic_json(
        alternate,
        dict(
            schema="blocked_gate_check_v1",
            failed_field="capture_protocol_ready_score",
            failed_upstream=tasks[1]["id"],
            failed_evidence_path=str(root / tasks[1]["deliverable"]),
            failed_evidence_sha256=None,
            failed_operator="==",
            failed_expected=1,
            failed_observed=0,
        ),
    )
    assert (
        i.load(root, tmp_path / "alternate")["dispositions"][2]["disposition"]
        == "alternate_conductor_block"
    )
    monkeypatch.setattr(i, "read_bound_sidecar", lambda *_: dict(report=dict(passed=False)))
    monkeypatch.setattr(
        i, "primitive_audit", lambda *_: (_ for _ in ()).throw(ValueError("invalid primitive"))
    )
    data = i.load(root, tmp_path / "terminal-failure")
    assert any(g["artifact_field"] == "terminal_validation" for g in data["failures"])
    assert any(g["artifact_field"] == "primitive_reduction" for g in data["failures"])
    monkeypatch.setattr(i, "NAMED", ["/tmp/carnot8135-never-exists"])
    assert any(g["observed"] is False for g in i.load(root, tmp_path / "absent-prereq")["failures"])


def test_primitive_service_and_replay(tmp_path, monkeypatch):
    """REQ-VERIFY-8135: independently execute the qualified primitive reducer."""
    from carnot import experiment_8132_v703_service_cost as service

    path = tmp_path / "primitives.json"
    atomic_json(path, dict(config={}, pairs=[]))
    monkeypatch.setattr(service, "reduce_rows", lambda *_: dict(passed=True))
    source = dict(primitive_rows=i.reference(path), reduction=dict(passed=True))
    audit = i.primitive_audit(source, 8132)
    assert audit["available"]
    source["reduction"] = {}
    with pytest.raises(ValueError, match="service_primitive"):
        i.primitive_audit(source, 8132)
    data = data_fixture()
    data["independent_reductions"][data["tasks"][9]["id"]] = audit
    candidate_path, _ = candidate(tmp_path, data)
    assert c.replay(candidate_path)["passed"]
    monkeypatch.setattr(i, "primitive_audit", lambda *_: dict(available=True, result={}))
    with pytest.raises(ValueError, match="primitive_reduction"):
        c.replay(candidate_path)


def fake_execute(plan, directory):
    """Return deterministic private transport receipts; natural runs never use this helper."""
    directory.mkdir(parents=True, exist_ok=True)
    receipts = []
    for spec in plan:
        log = directory / (spec.name + ".log")
        if spec.name == "independent_reduction":
            input_path = Path(spec.argv[spec.argv.index("--worker-input") + 1])
            output_path = Path(spec.argv[spec.argv.index("--output") + 1])
            atomic_json(output_path, r.reduce(json.loads(input_path.read_bytes())))
        gates = dict(
            gates={k: dict(pass_=True) for k in ["G1", "G2", "G3", "G4"]},
            paper_ready=True,
            unmet_gates=[],
        )
        log.write_text(json.dumps(gates))
        receipts.append(
            dict(
                name=spec.name,
                scope=spec.scope,
                passed=True,
                normal_exit=True,
                argv=list(spec.argv),
                actual_exit=0,
                expected_exit=0,
                log_path=str(log),
                log_sha256=c.reference(log)["sha256"],
            )
        )
    return receipts


def test_cli_main_owned_recovery_worker_and_errors(tmp_path, monkeypatch):
    """REQ-REPORT-8135: recover a rejected candidate without deleting its original bytes."""
    plan = c.commands(tmp_path)
    assert any(s.name == "consumers_E2E018" for s in plan)
    monkeypatch.setattr(
        c,
        "commands",
        lambda _: [
            c.CommandSpec("unit", ("private",), "owned"),
            c.CommandSpec("publication_gate", ("private",), "publication"),
        ],
    )
    monkeypatch.setattr(c, "execute", fake_execute)
    data = data_fixture()
    data["primaries"][data["tasks"][0]["id"]]["trained_head_specs"] = [
        dict(kind="small numerical head")
    ]
    monkeypatch.setattr(i, "load", lambda *_: deepcopy(data))
    reports = iter([False, True])
    monkeypatch.setattr(c, "terminal", lambda _: dict(passed=next(reports)))
    output = tmp_path / "results" / (c.NAME + ".json")
    assert c.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert list(output.parent.glob("raw/**/failed_primary.json"))
    assert c.replay(output)["passed"]
    assert c.main(["--cold-replay", str(output)]) == 0
    output.write_text("{}")
    assert c.main(["--cold-replay", str(output)]) == 1
    worker_input = tmp_path / "worker-input.json"
    atomic_json(worker_input, data)
    assert (
        c.main(["--worker-input", str(worker_input), "--output", str(tmp_path / "worker.json")])
        == 0
    )
    assert (
        c.main(
            [
                "--worker-input",
                str(worker_input),
                "--output",
                str(c.ROOT / "results" / (c.NAME + ".json")),
            ]
        )
        == 1
    )
    assert c.main(["--fixture-e2e", "--output", str(output)]) == 1
    output.unlink()
    monkeypatch.setattr(c, "terminal", lambda _: dict(passed=True))
    assert c.main(["--fixture-e2e", "--root", str(tmp_path), "--output", str(output)]) == 0

    def bad_child(plan, directory):
        receipts = fake_execute(plan, directory)
        if any(s.name == "independent_reduction" for s in plan):
            receipts[0]["passed"] = False
        return receipts

    monkeypatch.setattr(c, "execute", bad_child)
    assert c.main(["--output", str(output)]) == 1
    assert list(output.parent.glob("raw/**/previous_primary.json"))


def test_relative_operand_deduplication():
    """REQ-REPORT-8135: repeated external operands are terminal exactly once."""
    gate = dict(path="results/absent.json", artifact_field="ready", expected=1, observed=0)
    result = i.deduplicate([gate, dict(gate, check="different_consumer")])
    assert len(result) == 1
    assert Path(result[0]["path"]).is_absolute()


@pytest.mark.parametrize("route", ["success", "missing", "disqualified"])
def test_private_direct_cli_and_cold_mutation(tmp_path, route):
    """SCENARIO-REPORT-8135-PRIVATE: real outside-checkout CLI and normal validators."""
    root, tasks = fixture(tmp_path)
    if route == "missing":
        (root / tasks[5]["deliverable"]).unlink()
    elif route == "disqualified":
        path = root / tasks[5]["deliverable"]
        value = json.loads(path.read_bytes())
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_private",
            required_checks_passed=False,
            flagged_adversarial=True,
        )
        atomic_json(path, value)
    output = tmp_path / "published" / "results" / (c.NAME + ".json")
    environment = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    environment.pop("PYTHONPATH", None)
    argv = [
        str(c.ROOT / ".venv/bin/python"),
        "-u",
        str(c.ROOT / c.CLI),
        "--fixture-e2e",
        "--root",
        str(root),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        argv, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60
    )
    (tmp_path / "cli.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout[-3000:]
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"]
    assert value["verdict_class"] == ("null" if route == "success" else "blocked")
    assert value["MODEL_SPECS"] == []
    replay_argv = argv[:3] + ["--cold-replay", str(output)]
    replayed = subprocess.run(
        replay_argv, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30
    )
    assert replayed.returncode == 0, replayed.stdout
    value["completed_count"] += 1
    atomic_json(output, value)
    mutated = subprocess.run(
        replay_argv, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30
    )
    assert mutated.returncode == 1 and "reduction_drift" in mutated.stdout


def test_execute_terminal_and_script_imports(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8135-PRIVATE: argv and normal exit survive the heartbeat wrapper."""
    monkeypatch.setattr(
        c,
        "run_commands",
        lambda *_args, **_kwargs: [
            dict(
                log_path=str(tmp_path / "log"),
                command_argv=["private"],
                exit_code=0,
                timed_out=False,
                passed=True,
            )
        ],
    )
    assert c.execute([], tmp_path)[0]["normal_exit"]
    assert c.terminal(tmp_path / "candidate")["passed"]
    import runpy

    monkeypatch.setattr(c, "main", lambda: 0)
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert exited.value.code == 0


def test_service_board_retention_independence():
    """REQ-VERIFY-8135: host fixtures and board custody preserve their limited scopes."""
    data = data_fixture()
    data["prior_evidence"]["exp1"] = dict(
        path="/tmp/historical", sha256=None, retirement_history=[dict(scope="historical_null")]
    )
    service = data["primaries"][data["tasks"][9]["id"]]
    service.update(
        host_service_ready_score=1, complete_service_ready_score=0, verifier_is_oracle=True
    )
    data["primaries"][data["tasks"][11]["id"]]["board_rows"] = [dict(board="KV260", k_max=5)]
    first = data["primaries"][data["tasks"][5]["id"]]["decision_rows"][0]
    data["primaries"][data["tasks"][5]["id"]]["decision_rows"].append(deepcopy(first))
    value = r.reduce(data)
    assert value["H1"]["source_count"] == 128
    assert value["service_evidence_scope"]["host_qualified"]
    assert not value["service_evidence_scope"]["whole_service_qualified"]
    assert value["board_obligations"][0]["k_max"] == 5
    assert value["H2"]["retention"]["safe"]
    assert value["retirement_history"][-1]["preserved_retirements"]


def test_hardware_primitive_and_replay(tmp_path, monkeypatch):
    """REQ-VERIFY-8135: board and software equations replay independently of service readiness."""
    from carnot.reporting import hardware_service_8134 as hardware

    operand = tmp_path / "hardware-input.json"
    atomic_json(operand, dict(boards=[]))
    expected = dict(
        board_rows=[], workload_operation_rows=[], quantization_rows=[], amdahl_bounds=[]
    )
    monkeypatch.setattr(hardware, "reduce", lambda _: expected)
    value = dict(expected, replay_input_reference=i.reference(operand))
    audit = i.primitive_audit(value, 8134)
    assert audit["available"]
    data = data_fixture()
    data["independent_reductions"][data["tasks"][11]["id"]] = audit
    path, _ = candidate(tmp_path, data)
    assert c.replay(path)["passed"]
    value["amdahl_bounds"] = ["drift"]
    with pytest.raises(ValueError, match="hardware_primitive"):
        i.primitive_audit(value, 8134)
    root, _ = fixture(tmp_path / "custody")
    monkeypatch.setattr(i, "primitive_audit", lambda *_: audit)
    assert i.load(root, tmp_path / "audited-custody")["independent_reductions"]
