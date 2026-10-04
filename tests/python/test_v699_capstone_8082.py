"""REQ-REPORT-8082: independently reduced negative outcomes remain terminal."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting import v699_capstone as c
from carnot.reporting import v699_capstone_reduction as r
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def fixture(root):
    """Use full private authority bytes so a short title cannot hide prompt drift."""
    design = root / c.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes((c.ROOT / c.DESIGN).read_bytes())
    from carnot.reporting.roadmap_contract import parse_design
    import yaml

    _, tasks = parse_design(design.read_text(), milestone=c.MILESTONE)
    active = root / "active.yaml"
    active.write_text(yaml.safe_dump(dict(milestone=c.MILESTONE, tasks=tasks)))
    snapshots = {
        k: dict(exists=True, snapshot_path=str(p), sha256=sha256_file(p))
        for k, p in (("design", design), ("active", active))
    }
    snapshots["staged"] = dict(exists=False, source_path=str(root / "consumed.yaml"))
    metadata = dict(
        authority_snapshots=snapshots,
        canonical_tasks_sha256=c.tasks_digest(tasks),
        contract_ready_score=1,
    )
    for task in tasks[:-1]:
        path = root / task["deliverable"]
        side = path.parent / "raw" / path.stem / "terminal.json"
        value = dict(
            experiment_id=int(task["id"][3:7]),
            task_id=task["id"],
            honest_verdict="complete_null_private",
            verdict_class="null",
            flagged_adversarial=False,
            required_checks_passed=True,
            verifier_is_oracle=True,
            rows=[],
            raw_shard_hashes=[],
            terminal_validation_sidecar_path=str(side),
        )
        if task["id"].startswith("exp8070-"):
            value.update(metadata)
        published = c.publish_primary(path, value, lambda _: dict(passed=True))
        atomic_json(side, dict(publication=published))
    return tasks


def test_authority_mutation_and_missing(tmp_path):
    """SCENARIO-REPORT-8082-REDUCTION: full invocation hashes authenticate prompts."""
    fixture(tmp_path)
    tasks, _, failures = c.authorities(tmp_path)
    assert len(tasks) == 13 and not failures
    (tmp_path / "active.yaml").write_text("tampered")
    _, _, failures = c.authorities(tmp_path)
    assert failures[0]["field"] == "immutable_authority"
    data = json.loads((tmp_path / c.INPUT).read_text())
    import yaml

    (tmp_path / "active.yaml").write_text(yaml.safe_dump(dict(milestone=c.MILESTONE, tasks=[])))
    data["authority_snapshots"]["active"]["sha256"] = sha256_file(tmp_path / "active.yaml")
    atomic_json(tmp_path / c.INPUT, data)
    assert c.authorities(tmp_path)[2][0]["observed"] == "immutable_authority_drift"
    _, _, failures = c.authorities(tmp_path / "missing")
    assert failures and failures[0]["observed"]


def test_exact_two_hypothesis_family():
    """REQ-REPORT-8082: missing and safety-failing tests receive family p=1."""
    h1 = dict(
        raw_p_value=0.001,
        support_passed=True,
        safety_passed=True,
        observed_gain=0.03,
        beneficial_changed_sources=5,
    )
    h2 = dict(
        tests=[dict(raw_p=0.02, gain=0.03)],
        support_passed=True,
        safety_passed=True,
        beneficial_changed_sources=5,
    )
    family = r.holm(h1, h2, (True, True))
    assert [x["holm_adjusted_p"] for x in family] == [0.002, 0.02]
    assert all(x["positive_claim"] for x in family)
    h1["safety_passed"] = False
    assert r.holm(h1, h2, (True, False))[0]["family_p"] == 1
    assert [x["family_p"] for x in r.holm(None, None, (False, False))] == [1, 1]


def test_complete_blocked_and_claim_mutation(tmp_path):
    """REQ-REPORT-8082: no missing upstream becomes a successful placeholder."""
    fixture(tmp_path / "input")
    value = c.build(tmp_path / "input", "20261003", tmp_path / "raw")
    assert len(value["task_dispositions"]) == 13
    assert len(value["gap_decisions"]) == 3
    assert not value["science_ready"]
    assert value["generalized_learning_benefit_score"] == 0
    assert c.cold_replay(value) == []
    altered = deepcopy(value)
    altered["H1"]["family_p"] = 0.001
    assert c.cold_replay(altered)
    (tmp_path / "input" / c.INPUT).write_text("{}")
    assert c.cold_replay(value)


def test_real_cli_success_blocked_mutation_and_cold(tmp_path):
    """SCENARIO-REPORT-8082-TERMINAL: direct private CLIs use real child exits."""
    fixture(tmp_path / "input")
    command = [str(c.ROOT / ".venv/bin/python"), "-u", str(c.ROOT / c.CLI)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    for root, folder in ((tmp_path / "input", "success"), (tmp_path / "absent", "blocked")):
        output = tmp_path / folder / "experiment_8082_v699_capstone.json"
        child = subprocess.run(
            command + ["--fixture-e2e", "--root", str(root), "--output", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        value = json.loads(output.read_text())
        assert value["verifier_is_oracle"] and value["verdict_class"] == "blocked"
        for mutate, code in ((False, 0), (True, 1)):
            if mutate:
                value["science_ready"] = True
                atomic_json(output, value)
            child = subprocess.run(
                command + ["--cold-replay", str(output)],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert child.returncode == code, child.stdout + child.stderr


def test_missing_primitive_and_retirement():
    """REQ-REPORT-8082: exact-repeat retirement never retires valid ARC monitoring."""
    assert r.independent({}, 8074)["measurement_available"] is False
    tasks = [
        dict(
            id="exp8080-arc-supervisor-frontier",
            prior_failures=[
                dict(
                    experiment_id="exp8067-arc-supervisor-frontier",
                    verdict="complete_null_empty_delta",
                    retire_if_same_verdict=True,
                    addressed_by="new authenticated outcomes",
                )
            ],
        )
    ]
    rows = [
        dict(
            task_id=tasks[0]["id"],
            honest_verdict="complete_null_empty_delta",
            verdict_class="null",
            sha256="sha256:current",
        )
    ]
    old = [
        dict(
            task_id="exp8067-arc-supervisor-frontier",
            honest_verdict="complete_null_empty_delta",
            sha256="sha256:prior",
        )
    ]
    assert not r.retirements(tasks, rows, old)[0]["retire"]


def test_current_primitive_reductions_and_join(tmp_path):
    """REQ-REPORT-8082: a private exited child reduces real primitives without retained test heap."""
    import coverage

    output = tmp_path / "candidate.json"
    measured = tmp_path / ".coverage.primitive"
    command = [
        str(c.ROOT / ".venv/bin/python"),
        "-m",
        "coverage",
        "run",
        "--data-file=" + str(measured),
        "--include=" + ",".join(str(c.ROOT / p) for p in c.OWNED),
        str(c.ROOT / c.CLI),
        "--worker",
        "--root",
        str(c.ROOT),
        "--output",
        str(output),
        "--durable",
        str(tmp_path / "raw"),
    ]
    child = subprocess.run(
        command,
        cwd=tmp_path,
        env=dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1"),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    active = coverage.Coverage.current()
    if active:
        observed_coverage = coverage.CoverageData(basename=str(measured))
        observed_coverage.read()
        active.get_data().update(observed_coverage)
    value = json.loads(output.read_text())
    assert value["H1"]["observed_gain"] < 0.02
    assert value["H2"]["beneficial_changed_sources"] == 0
    assert all(
        row["measurement_available"]
        for row in value["independent_reduction_rows"]
        if row["task_id"].startswith(("exp8074-", "exp8077-", "exp8078-", "exp8079-", "exp8081-"))
    )
    joined = value["service_partition_join"]
    assert joined["complete_six_mode_credit"]
    assert len(joined["mode_condition_rows"]) == 18
    assert len(joined["break_even_requests"]) == 6
    assert not joined["native_10x_requirement_closed"]
    atomic_json(tmp_path / "inputs.json", dict(cases=[], head={}, sources=[]))
    base = dict(
        raw_directory=str(tmp_path),
        code_config_hashes=[
            dict(path=str(c.ROOT / "python/carnot" / p), sha256="shared") for p in r.SHARED_CODE
        ],
        loaded_library_receipt=dict(sha256="native"),
    )
    altered = deepcopy(base)
    altered["loaded_library_receipt"]["sha256"] = "sha256:other"
    assert not r.service_join(base, altered, (True, True))["exact_identity_match"]
    assert not r.service_join(base, {}, (True, False))["complete_six_mode_credit"]
    altered = deepcopy(base)
    altered["code_config_hashes"] = [
        x for x in altered["code_config_hashes"] if not x["path"].endswith(r.SHARED_CODE[0])
    ]
    assert not r.service_join(base, altered, (True, True))["exact_identity_match"]


def test_independent_drift_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8082-REDUCTION: altered primitive operands and label access fail."""
    data = dict(
        rows=[{}],
        terminal_validation_sidecar_path=str(tmp_path / "terminal.json"),
        label_access_receipt=dict(opened_after_seal=True, selected_role="evaluation96"),
    )
    atomic_json(tmp_path / "predictions.json", {})
    atomic_json(tmp_path / "work.json", dict(measurement=dict(H1=1)))
    monkeypatch.setattr(r.source, "evaluate", lambda _: dict(H1=1))
    assert r.independent(data, 8074)["H1"] == 1
    data["label_access_receipt"]["opened_after_seal"] = False
    with pytest.raises(ValueError, match="label_access"):
        r.independent(data, 8074)
    monkeypatch.setattr(r.source, "evaluate", lambda _: dict(H1=2))
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        r.independent(data, 8074)
    atomic_json(
        tmp_path / "work.json",
        dict(
            data={},
            evidence=dict(final_head_seals=[], later_source_rows=[], retention_rows=[], H2=1),
        ),
    )
    monkeypatch.setattr(r.online.prior, "labels", lambda *_: {})
    monkeypatch.setattr(
        r.online.a, "reconstruct", lambda *a, **k: dict(final_head_seals=[], later_source_rows=[])
    )
    monkeypatch.setattr(r.online.independent, "retention", lambda *_: [])
    monkeypatch.setattr(r.online.a, "comparisons", lambda *_: dict(H2=2))
    with pytest.raises(ValueError, match="H2_primitive"):
        r.independent(data, 8077)
    atomic_json(tmp_path / "observations.json", dict(rows=[]))
    with pytest.raises(ValueError, match="cache_primitive"):
        r.independent(data, 8078)
    data["rows"] = [dict(status="censored")]
    atomic_json(tmp_path / "inputs.json", dict(cases=[]))
    atomic_json(
        tmp_path / "observations.json", dict(rows=data["rows"], parity_rows=[], population_rows=[])
    )
    data["population_rows"] = []
    monkeypatch.setattr(r.core, "reduce_rows", lambda _: [])
    data["complete_workload_ratios"] = [1]
    with pytest.raises(ValueError, match="cache_ratio"):
        r.independent(data, 8079)
    data["replay_input_reference"] = dict(path=str(tmp_path / "input.json"))
    atomic_json(tmp_path / "input.json", {})
    monkeypatch.setattr(r.hardware, "reduce", lambda _: dict(numerator=2))
    data["numerator"] = 1
    with pytest.raises(ValueError, match="hardware_primitive"):
        r.independent(data, 8081)
    assert r.independent(data, 8072)["reason"] == "custody_only_not_a_primary_test"


def test_complete_replay_and_retirement_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8082: owned failure disqualifies; replay cannot trust a claim seal alone."""
    fixture(tmp_path / "input")
    value = c.build(tmp_path / "input", "20261003", tmp_path / "raw")
    counts = {p: dict(num_statements=1, covered_lines=1) for p in c.OWNED}
    c.complete(value, [dict(passed=True)], counts)
    assert value["required_checks_passed"] and value["completed_count"] == 13
    assert value["verdict_class"] == "blocked"
    c.complete(value, [dict(passed=False)], counts)
    assert value["verdict_class"] == "disqualified" and not value["capstone_execution_ready_score"]
    monkeypatch.setattr(
        c.reduction, "independent", lambda *_: dict(measurement_available=False, reason="altered")
    )
    assert c.cold_replay(value) == ["independent_reduction_drift"]
    monkeypatch.setattr(
        c.reduction, "independent", lambda *_: (_ for _ in ()).throw(ValueError("bad rows"))
    )
    assert c.cold_replay(value) == ["independent_reduction_drift"]
    tasks = [
        dict(
            id="exp8078-feature-cache-core",
            prior_failures=[
                dict(
                    experiment_id="exp8066-feature-service",
                    verdict="complete_null_same",
                    retire_if_same_verdict=True,
                    addressed_by="changed acquisition",
                )
            ],
        )
    ]
    rows = [
        dict(
            task_id=tasks[0]["id"],
            honest_verdict="complete_null_same",
            verdict_class="null",
            sha256="current",
        )
    ]
    history = [
        dict(task_id="exp8066-feature-service", honest_verdict="complete_null_same", sha256="old")
    ]
    assert r.retirements(tasks, rows, history)[0]["retire"]


def test_upstream_identity_ends_blocked(tmp_path):
    """REQ-REPORT-8082: misidentified external evidence cannot abort terminal accounting."""
    tasks = fixture(tmp_path / "input")
    path = tmp_path / "input" / tasks[0]["deliverable"]
    value = json.loads(path.read_text())
    value["task_id"] = "exp9999-forged"
    atomic_json(path, value)
    observed = c.build(tmp_path / "input", "20261003", tmp_path / "raw")
    assert observed["verdict_class"] == "blocked"
    assert len(observed["task_dispositions"]) == 13
    assert any(x["field"] == "upstream_identity" for x in observed["gate_check_summary"])


def test_parent_worker_cold_and_validation_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8082-TERMINAL: parent bookkeeping uses exited children and owned checks."""
    fixture(tmp_path / "input")
    command = [
        "--root",
        str(tmp_path / "input"),
        "--output",
        str(tmp_path / "results/experiment_8082_v699_capstone.json"),
    ]

    def child(root, spec, private, durable):
        log = durable / (spec["name"] + ".json")
        atomic_json(log, {})
        if spec["name"] == "reduction_normal_exit":
            directory = Path(spec["argv"][spec["argv"].index("--durable") + 1])
            atomic_json(Path(spec["argv"][-1]), c.build(tmp_path / "input", "20261003", directory))
        if spec["name"] == "publication_gate":
            atomic_json(
                log,
                dict(
                    paper_ready=False,
                    unmet_gates=["G2"],
                    gates={f"G{i}": dict(pass_=False, **{"pass": i != 2}) for i in range(1, 5)},
                ),
            )
        if spec["name"] == "coverage_json":
            atomic_json(
                private / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(num_statements=1, covered_lines=1)) for p in c.OWNED
                    }
                ),
            )
        return dict(passed=True, log_path=str(log), log_sha256=sha256_file(log), actual_exit=0)

    monkeypatch.setattr(c, "run_check", child)
    monkeypatch.setattr(c, "publish", lambda value, output, *a, **k: atomic_json(output, value))
    monkeypatch.delenv("CARNOT_8082_HEALTH_RECEIPT", raising=False)
    assert c.main(command) == 0
    output = Path(command[-1])
    value = json.loads(output.read_text())
    assert value["required_checks_passed"] and value["publication_gate_results"]["unmet_gates"] == [
        "G2"
    ]
    health = tmp_path / "health.json"
    health_log = tmp_path / "health.log"
    health_log.write_text("diagnostic failure")
    atomic_json(
        health, dict(passed=False, log_path=str(health_log), log_sha256=sha256_file(health_log))
    )
    monkeypatch.setenv("CARNOT_8082_HEALTH_RECEIPT", str(health))
    assert c.main(command) == 0
    assert c.main(command + ["--fixture-e2e"]) == 0
    health_log.write_text("changed")
    with pytest.raises(ValueError, match="repository_health_log_drift"):
        c.main(command)
    atomic_json(
        health, dict(passed=False, log_path=str(health_log), log_sha256=sha256_file(health_log))
    )
    with pytest.raises(ValueError, match="date_must"):
        c.main(["--date", "20261002"])
    assert c.main(command + ["--worker", "--durable", str(tmp_path / "worker")]) == 0
    assert c.main(["--cold-replay", str(output)]) == 0
    assert c.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1
    monkeypatch.setattr(c, "run_check", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="reduction_child_failed"):
        c.main(command)


def test_publication_owned_failure_and_consumer_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8082: terminal validators and actual publication consumers control visibility."""
    fixture(tmp_path / "input")
    value = c.build(tmp_path / "input", "20261003", tmp_path / "raw")
    output = tmp_path / "results/experiment_8082_v699_capstone.json"

    def receipt(root, spec, private, durable):
        log = durable / (spec["name"] + ".json")
        atomic_json(log, dict(flagged_count=0))
        return dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))

    monkeypatch.setattr(c, "run_check", receipt)
    c.publish(value, output, tmp_path / "private", tmp_path / "raw", fixture=False)
    assert c.read_bound_sidecar(
        output,
        Path(
            json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())["publication"][
                "sidecar_path"
            ]
        ),
    )["report"]["passed"]
    value["capstone_execution_ready_score"] = 1
    assert c.cold_replay(value) == ["terminal_claim_drift"]
    value["capstone_execution_ready_score"] = 0
    value["checkpoint_references"] = [
        x for x in value["checkpoint_references"] if x["role"] != "claim_seal"
    ]
    monkeypatch.setattr(c, "reader_receipt", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="published_reader_drift"):
        c.publish(value, output, tmp_path / "private", tmp_path / "raw", fixture=True)
    monkeypatch.setattr(c, "run_check", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="terminal_validation_failed"):
        c.publish(value, output, tmp_path / "private", tmp_path / "raw", fixture=False)
    assert value["verdict_class"] == "disqualified"


def test_staged_authority_shards_and_failed_reducer(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8082-REDUCTION: staged bytes and split snapshots are authenticated."""
    fixture(tmp_path / "input")
    path = tmp_path / "input" / c.INPUT
    metadata = json.loads(path.read_text())
    staged = tmp_path / "staged.yaml"
    staged.write_bytes((tmp_path / "input/active.yaml").read_bytes())
    metadata["authority_snapshots"]["staged"] = dict(
        exists=True, snapshot_path=str(staged), sha256=sha256_file(staged)
    )
    atomic_json(path, metadata)
    assert not c.authorities(tmp_path / "input")[2]
    staged.write_text("tasks: []")
    metadata["authority_snapshots"]["staged"]["sha256"] = sha256_file(staged)
    atomic_json(path, metadata)
    assert c.authorities(tmp_path / "input")[2]
    metadata["authority_snapshots"]["staged"]["exists"] = False
    atomic_json(path, metadata)
    value = c.build(tmp_path / "input", "20261003", tmp_path / "raw")
    part = tmp_path / "part.bin"
    part.write_bytes(b"original")
    shards = tmp_path / "copy.shards.json"
    atomic_json(
        shards,
        dict(
            shards=[dict(path=str(part), sha256=sha256_file(part))],
            original_sha256=sha256_file(part),
        ),
    )
    value["raw_shard_hashes"].append(c.reference(shards))
    assert c.cold_replay(value) == []
    part.write_bytes(b"mutated")
    assert c.cold_replay(value) == ["snapshot_shard_changed"]
    atomic_json(
        shards,
        dict(
            shards=[dict(path=str(part), sha256=sha256_file(part))], original_sha256="sha256:wrong"
        ),
    )
    value["raw_shard_hashes"][-1] = c.reference(shards)
    assert c.cold_replay(value) == ["snapshot_reconstruction_drift"]
    monkeypatch.setattr(
        c.reduction,
        "independent",
        lambda *_: (_ for _ in ()).throw(ValueError("invalid primitive")),
    )
    failed = c.build(tmp_path / "input", "20261003", tmp_path / "failed")
    assert any(g["field"] == "independent_reduction" for g in failed["gate_check_summary"])
    assert c.cold_replay(failed) == []


def test_cache_primitive_quartet_and_population_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8082: cache ratios cannot conceal changed states or missing arms."""
    row = dict(
        status="completed",
        checkpoint=dict(path="unused"),
        mode="warm",
        condition="c",
        transaction_class="k",
        repetition=0,
        identity="work",
        unit="u",
    )
    data = dict(
        rows=[row],
        terminal_validation_sidecar_path=str(tmp_path / "terminal.json"),
        population_rows=[],
    )
    atomic_json(tmp_path / "inputs.json", dict(cases=[dict(identity="work")]))
    atomic_json(
        tmp_path / "observations.json", dict(rows=data["rows"], parity_rows=[], population_rows=[])
    )
    with pytest.raises(ValueError, match="incomplete_quartet"):
        r.independent(data, 8078)
    data["rows"] = [dict(row, unit=str(i)) for i in range(4)]
    observed = dict(
        rows=data["rows"], parity_rows=[dict(unit="0", passed=True)], population_rows=[]
    )
    atomic_json(tmp_path / "observations.json", observed)
    monkeypatch.setattr(r.core, "quartet_parity", lambda *_: [dict(unit="0", passed=False)])
    with pytest.raises(ValueError, match="checkpoint_parity_drift"):
        r.independent(data, 8078)
    monkeypatch.setattr(r.core, "quartet_parity", lambda *_: [])
    data["population_rows"] = [dict(elapsed_ns=1)]
    with pytest.raises(ValueError, match="population_drift"):
        r.independent(data, 8079)
