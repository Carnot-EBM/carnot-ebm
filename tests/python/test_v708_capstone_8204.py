"""REQ-REPORT-8204 and REQ-VERIFY-8204: private custody and source controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import pytest

from carnot.reporting import v708_capstone as c
from carnot.reporting import v708_capstone_inputs as e
from carnot.reporting import v708_capstone_science as s
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary


def fixture(tmp_path):
    """Real frozen authority keeps all fabricated producer claims private."""
    root = tmp_path / "private"
    original = json.loads((e.ROOT / e.INPUT).read_bytes())
    receipt = {
        k: original[k]
        for k in (
            "task_contract",
            "authority_snapshots",
            "canonical_tasks_sha256",
            "literature_mapping",
            "historical_hash_failures",
            "prior_scope_ledger",
        )
    }
    tasks = receipt["task_contract"]
    for role in ("design", "active"):
        ref = receipt["authority_snapshots"][role]
        target = root / (role + ".bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(ref["snapshot_path"]).read_bytes())
        ref.update(snapshot_path=str(target), sha256=sha256_file(target))
    design = Path(receipt["authority_snapshots"]["design"]["snapshot_path"])
    table = "\n".join(
        f"| {i} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |"
        for i, t in enumerate(tasks, 1)
    )
    design.write_text(
        "## Exact task contract\n"
        + table
        + "\nCanonical full-task SHA256: `"
        + receipt["canonical_tasks_sha256"]
        + "`\n"
        + "<!-- V708_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(dict(milestone="2026.10.708", tasks=tasks))
        + "\n```\n"
    )
    receipt["authority_snapshots"]["design"]["sha256"] = sha256_file(design)
    for n, task in enumerate(tasks[:-1], 8192):
        path = root / task["deliverable"]
        side = path.parent / "raw" / path.stem / "terminal.json"
        value = dict(
            experiment_id=n,
            task_id=task["id"],
            honest_verdict="complete_null_private",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            gate_check_summary=[],
            rows=[],
            terminal_validation_sidecar_path=str(side),
        )
        value.update(
            {
                g["artifact_field"]: 1
                for t in tasks
                for g in t["gated_on"]
                if g["upstream"] == task["id"]
            }
        )
        if n == 8192:
            value.update(receipt)
        pub = publish_primary(path, value, lambda _: dict(passed=True))
        atomic_json(side, dict(publication=pub))
    return root, tasks


def replay_value(tmp_path, data, passed=True):
    """Hash private inputs exactly as the public cold consumer requires."""
    source, output = tmp_path / "input.json", tmp_path / "out.json"
    atomic_json(source, data)
    value = s.reduce(data)
    value.update(
        required_checks_passed=passed,
        validation_receipts=[],
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        replay_input_reference=e.reference(source),
    )
    c.qualify(value, passed)
    atomic_json(output, value)
    return source, output, value


def test_mixed_dispositions_and_authority(tmp_path):
    """SCENARIO-REPORT-8204-CUSTODY: skips cannot become scientific nulls."""
    root, tasks = fixture(tmp_path)
    (root / tasks[7]["deliverable"]).unlink()
    path = root / tasks[1]["deliverable"]
    v = json.loads(path.read_bytes())
    v.update(
        required_checks_passed=False,
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_owned_validation",
    )
    atomic_json(path, v)
    skip = root / "results/experiment_8198_calibrated_online_memory.json"
    atomic_json(
        skip,
        dict(
            schema="blocked_gate_check_v1",
            task_id=tasks[6]["id"],
            failed_field="calibrated_memory_ready_score",
            failed_upstream=tasks[1]["id"],
            failed_evidence_path=str(path),
            failed_evidence_sha256=sha256_file(path),
            failed_operator="==",
            failed_expected=1,
            failed_observed=0,
        ),
    )
    data = e.load(root, tmp_path / "raw")
    value = s.reduce(data)
    assert len(value["rows"]) == value["completed_count"] == 13
    assert value["rows"][7]["primary_honest_verdict"] is None
    assert value["rows"][1]["verdict_class"] == "disqualified"
    assert value["rows"][6]["primary_present"]
    assert value["rows"][6]["conductor_skips"]
    assert value["H1"]["status"] == value["H2"]["status"] == "blocked"
    assert value["science_ready_score"] == value["independent_generalization_score"] == 0
    assert len(value["gap_decisions"]) == 3
    assert all(not g["closed"] for g in value["gap_decisions"].values())
    receipt = root / e.INPUT
    v = json.loads(receipt.read_bytes())
    v["canonical_tasks_sha256"] = "changed"
    atomic_json(receipt, v)
    assert not e.load(root, tmp_path / "changed")["authority"]["activated"]
    receipt.unlink()
    assert not e.load(root, tmp_path / "absent")["authority"]["activated"]


def test_real_primitive_reductions(tmp_path):
    """SCENARIO-VERIFY-8204-REDUCTION: source null survives external blocks."""
    data = e.load(e.ROOT, tmp_path / "raw")
    value = s.reduce(data)
    assert not value["authority"]["activated"]
    assert value["authority"]["gate_check_summary"][0]["check"] == "design_exact_task_contract"
    assert value["H1"]["status"] == "completed_null"
    assert value["H1"]["statistics"]["H1"]["interval"]["mean_gain"] == -0.11328125
    assert value["H1"]["statistics"]["H1"]["extra_false_accepts"] == 2
    assert value["H1"]["completed_count"] == 97
    assert value["H2"]["status"] == "blocked"
    assert value["multiplicity"]["alpha_per_hypothesis"] == {"H1": 0.025, "H2": 0.025}
    assert value["retirement_decisions"][-1]["retire_exact_configuration"]
    assert not value["service_evidence_scope"]["nfr01_met"]
    assert value["service_evidence_scope"]["observed_trace"]["completed_count"] == 0
    assert value["historical_hash_failures"]
    assert len(value["board_obligations"]) >= 3
    for n in (8197, 8200, 8203):
        task = data["tasks"][n - 8192]
        assert s.primitive(data["primaries"][task["id"]], n) == data["audits"][task["id"]]
    altered = deepcopy(data["primaries"][data["tasks"][5]["id"]])
    altered["paired_intervals"]["all_slot"]["mean_gain"] = 99
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        s.primitive(altered, 8197)
    assert s.primitive({}, 8199) == {"available": False}


def test_private_worker_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8204-CUSTODY: external CLI imports without PYTHONPATH."""
    root, _ = fixture(tmp_path)
    data = e.load(root, tmp_path / "raw")
    source, output, value = replay_value(tmp_path, data)
    argv = [
        str(c.ROOT / ".venv/bin/python"),
        str(c.ROOT / c.CLI),
        "--worker-input",
        str(source),
        "--output",
        str(tmp_path / "worker.json"),
    ]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    print("[private_e2e] before_subprocess completed=0 pending=1", flush=True)
    child = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, timeout=60)
    print("[private_e2e] after_subprocess completed=1 pending=0", flush=True)
    assert child.returncode == 0, child.stdout + child.stderr
    replay_argv = argv[:2] + ["--cold-replay", str(output)]
    print("[private_e2e] before_cold_subprocess completed=1 pending=1", flush=True)
    cold = subprocess.run(replay_argv, cwd=tmp_path, env=env, capture_output=True, timeout=60)
    print("[private_e2e] after_cold_subprocess completed=2 pending=0", flush=True)
    assert cold.returncode == 0, cold.stdout + cold.stderr
    monkeypatch.setattr(sys, "argv", argv[1:])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert info.value.code == 0
    assert c.replay(output)["passed"]
    assert c.main(["--cold-replay", str(output)]) == 0
    value["H1"]["status"] = "invented"
    atomic_json(output, value)
    print("[private_e2e] before_tamper_subprocess completed=2 pending=1", flush=True)
    cold = subprocess.run(replay_argv, cwd=tmp_path, env=env, capture_output=True, timeout=60)
    print("[private_e2e] after_tamper_subprocess completed=3 pending=0", flush=True)
    assert cold.returncode == 1
    assert c.main(["--cold-replay", str(output)]) == 1
    source.write_text("{}")
    with pytest.raises(ValueError, match="input_hash_drift"):
        c.replay(output)


@pytest.mark.parametrize("passed", [True, False])
def test_main_and_validation_failure(tmp_path, monkeypatch, passed):
    """REQ-REPORT-8204: owned failure disqualifies; evidence remains terminal."""
    root, _ = fixture(tmp_path)
    gates = dict(
        gates={k: dict(passed=k != "G2") for k in ("G1", "G2", "G3", "G4")},
        paper_ready=False,
        unmet_gates=["G2"],
    )

    def execute(plan, raw, **kwargs):
        receipts = []
        for spec in plan:
            log = raw / (spec.name + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps(gates) if spec.name == "publication_gate" else "checked\n")
            if spec.name == "independent_reduction":
                source = Path(spec.argv[spec.argv.index("--worker-input") + 1])
                target = Path(spec.argv[spec.argv.index("--output") + 1])
                atomic_json(target, s.reduce(json.loads(source.read_bytes())))
            receipts.append(
                dict(
                    name=spec.name,
                    scope=spec.scope,
                    passed=passed if spec.scope == "owned" else True,
                    normal_exit=True,
                    log_path=str(log),
                    log_sha256=sha256_file(log),
                )
            )
        return receipts

    monkeypatch.setattr(c, "execute", execute)
    output = tmp_path / "experiment_8204_v708_capstone.json"
    args = ["--root", str(root), "--output", str(output), "--fixture-e2e"]
    assert c.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] is passed
    assert value["capstone_execution_ready_score"] == int(passed)
    assert value["MODEL_SPECS"] == value["call_ledger"] == value["trained_head_specs"] == []
    assert c.main(["--cold-replay", str(output)]) == 0
    health_log = tmp_path / "prior_health.log"
    health_log.write_text("saved unrelated repository health\n")
    value["repository_health"]["receipts"] = [
        dict(
            name="full_python_suite",
            passed=False,
            scope="repository_health",
            normal_exit=False,
            actual_exit=-15,
            log_path=str(health_log),
            log_sha256=sha256_file(health_log),
        )
    ]
    atomic_json(output, value)
    assert c.main(args) == 0
    value["validation_receipts"][0]["log_sha256"] = "changed"
    atomic_json(output, value)
    assert c.main(["--cold-replay", str(output)]) == 1


def test_fail_closed_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8204-CUSTODY: unsafe fixtures and changed operands fail."""
    assert c.main(["--fixture-e2e"]) == 1
    assert c.main(["--fixture-e2e", "--output", str(tmp_path / "out.json")]) == 1
    assert c.main(["--worker-input", str(tmp_path / "missing")]) == 1
    assert (
        c.main(
            ["--worker-input", str(tmp_path / "missing"), "--output", str(tmp_path / "out.json")]
        )
        == 1
    )
    root, tasks = fixture(tmp_path)
    real = s.primitive
    monkeypatch.setattr(s, "primitive", lambda *_: (_ for _ in ()).throw(ValueError("bad")))
    data = e.load(root, tmp_path / "bad")
    assert not any(r["qualified"] for r in data["dispositions"])
    monkeypatch.setattr(s, "primitive", real)
    path = root / tasks[2]["deliverable"]
    v = json.loads(path.read_bytes())
    side = Path(v["terminal_validation_sidecar_path"])
    validator = Path(json.loads(side.read_bytes())["publication"]["sidecar_path"])
    report = json.loads(validator.read_bytes())
    report["report"]["passed"] = False
    atomic_json(validator, report)
    assert not e.load(root, tmp_path / "failed")["dispositions"][2]["qualified"]
    data = e.load(root, tmp_path / "raw")
    data["audits"][tasks[0]["id"]] = dict(available=True, result={})
    _, output, _ = replay_value(tmp_path, data, False)
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        c.replay(output)
    plan = c.commands(tmp_path / "scratch")
    assert all(
        "::" not in a
        for spec in plan
        if spec.name in ("ruff_check", "ruff_format", "scoped_spec_coverage")
        for a in spec.argv
    )
    monkeypatch.setattr(c, "execute", lambda *_: [dict(passed=False, normal_exit=True)])
    assert not c.terminal(output)["passed"]
    assert (
        c.main(["--root", str(root), "--output", str(tmp_path / "failed.json"), "--fixture-e2e"])
        == 1
    )


def test_child_wait_counts(tmp_path, monkeypatch, capsys):
    """REQ-REPORT-8204: waiting children expose actual completed counts."""

    def slow(plan, raw):
        time.sleep(0.03)
        log = raw / "child.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("checked\n")
        return [dict(passed=True, normal_exit=True, log_path=str(log))]

    monkeypatch.setattr(c, "run_checks", slow)
    plan = [c.CommandSpec("one", ("true",), "owned"), c.CommandSpec("two", ("true",), "owned")]
    assert len(c.execute(plan, tmp_path, heartbeat_s=0.001)) == 2
    lines = capsys.readouterr().out
    assert "waiting_one completed=0 pending=2" in lines
    assert "waiting_two completed=1 pending=1" in lines


def test_private_authority_mutation_and_retired_input(tmp_path):
    """SCENARIO-REPORT-8204-CUSTODY: lifecycle authenticates complete task bytes."""
    root, tasks = fixture(tmp_path)
    data = e.load(root, tmp_path / "success")
    assert data["authority"]["activated"]
    assert len(data["authority"]["contract_rows"]) == 13
    receipt = root / e.INPUT
    v = json.loads(receipt.read_bytes())
    active = Path(v["authority_snapshots"]["active"]["snapshot_path"])
    import yaml

    roadmap = yaml.safe_load(active.read_bytes())
    roadmap["tasks"][3]["prompt"] += " changed operand"
    roadmap["tasks"][3]["title"] += " changed visible row"
    active.write_text(yaml.safe_dump(roadmap))
    v["authority_snapshots"]["active"]["sha256"] = sha256_file(active)
    atomic_json(receipt, v)
    changed = e.load(root, tmp_path / "mutation")
    assert not changed["authority"]["activated"]
    assert any(not r["matched"] for r in changed["authority"]["contract_rows"])
    exclusion = root / "ops/exclusion_manifest.yaml"
    exclusion.parent.mkdir(parents=True, exist_ok=True)
    exclusion.write_text("retired:\n- experiment_id: 8194\n")
    skip = root / tasks[6]["deliverable"]
    atomic_json(
        skip,
        dict(
            schema="blocked_gate_check_v1",
            task_id=tasks[6]["id"],
            failed_field="calibrated_memory_ready_score",
            failed_upstream=tasks[1]["id"],
            failed_evidence_path=str(root / tasks[1]["deliverable"]),
            failed_evidence_sha256=None,
            failed_operator="==",
            failed_expected=1,
            failed_observed=0,
        ),
    )
    atomic_json(root / "results/experiment_8174_private_invalid.json", {})
    blocked = e.load(root, tmp_path / "blocked")
    assert blocked["dispositions"][6]["primary_present"] is False
    assert any(
        g["check"] == "not_retired" for g in blocked["dispositions"][2]["gate_check_summary"]
    )
    assert any(g["check"] == "historical_service" for g in blocked["failures"])


def test_owned_setup_and_single_health_custody(tmp_path):
    """REQ-REPORT-8204: private pytest parents exist and health runs only once."""
    private = tmp_path / "private"
    c.commands(private)
    assert (private / "pytest").is_dir()
    log = tmp_path / "completed_health.log"
    log.write_text("actual unrelated failures; bounded timeout\n")
    previous = dict(
        repository_health=dict(
            receipts=[
                dict(
                    name="full_python_suite",
                    scope="repository_health",
                    passed=False,
                    actual_exit=-15,
                    normal_exit=False,
                    log_path=str(log),
                    log_sha256=sha256_file(log),
                )
            ]
        )
    )
    inherited = c.inherit_health(previous, tmp_path / "durable")
    assert len(inherited) == 1 and inherited[0]["actual_exit"] == -15
    assert inherited[0]["source_log_path"] == str(log)
    assert Path(inherited[0]["log_path"]).read_bytes() == log.read_bytes()
    assert Path(inherited[0]["log_path"]).stat().st_mode & 0o222 == 0
    previous["repository_health"]["receipts"][0]["log_sha256"] = "changed"
    with pytest.raises(ValueError, match="historical_health_log_hash_drift"):
        c.inherit_health(previous, tmp_path / "tampered")
