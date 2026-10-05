"""REQ-REPORT-8163, REQ-VERIFY-8163: private terminal and custody controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from carnot.reporting import v705_capstone as c
from carnot.reporting import v705_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary


def fixture(tmp_path):
    """Preserve real authority while fixture producer claims remain oracle inputs."""
    root = tmp_path / "private"
    receipt = json.loads((e.ROOT / e.INPUT).read_bytes())
    tasks = receipt["task_contract"]
    for role in ("design", "active"):
        ref = receipt["authority_snapshots"][role]
        target = root / (role + ".bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(ref["snapshot_path"]).read_bytes())
        ref.update(snapshot_path=str(target), sha256=sha256_file(target))
    for n, task in enumerate(tasks[:-1], 8150):
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
            raw_shard_hashes=[],
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
        if n == 8150:
            value.update(receipt)
        pub = publish_primary(path, value, lambda _: dict(passed=True))
        atomic_json(side, dict(publication=pub))
    return root, tasks


def test_mixed_null_blocked_missing_disqualified(tmp_path):
    """SCENARIO-REPORT-8163: no branch is erased by a different external block."""
    root, tasks = fixture(tmp_path)
    (root / tasks[7]["deliverable"]).unlink()
    (root / tasks[8]["deliverable"]).unlink()
    path = root / tasks[1]["deliverable"]
    value = json.loads(path.read_bytes())
    value.update(
        required_checks_passed=False,
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_owned_validation",
    )
    atomic_json(path, value)
    data = e.load(root, tmp_path / "raw")
    data["verifier_is_oracle"] = True
    result = e.reduce(data)
    assert len(result["task_dispositions"]) == result["completed_count"] == 14
    assert result["verdict_class"] == "blocked"
    assert result["H2"]["status"] == "blocked"
    assert result["rows"][7]["primary_honest_verdict"] == "<missing>"
    assert result["rows"][1]["verdict_class"] == "disqualified"
    assert result["independent_generalization_score"] == 0
    assert not any(r["retire_method_family"] for r in result["retirement_decisions"])
    assert all(not r["closed"] for r in result["gap_decisions"].values())
    assert all(
        set(
            (
                "check",
                "upstream",
                "path",
                "hash",
                "artifact_field",
                "op",
                "expected",
                "observed",
                "passed",
            )
        )
        <= set(g)
        for g in result["gate_check_summary"]
    )


def test_authority_mutation_and_missing_receipt(tmp_path):
    """REQ-REPORT-8163: use immutable activation rather than live method headings."""
    root, tasks = fixture(tmp_path)
    receipt = root / e.INPUT
    value = json.loads(receipt.read_bytes())
    value["canonical_tasks_sha256"] = "changed"
    atomic_json(receipt, value)
    data = e.load(root, tmp_path / "raw")
    assert not data["authority"]["activated"]
    receipt.unlink()
    assert not e.load(root, tmp_path / "absent")["authority"]["activated"]
    assert len(tasks) == 14


def test_cached_h1_and_disqualification(tmp_path):
    """REQ-VERIFY-8163: reopen source primitives; failed producers give no signal."""
    value = json.loads((e.ROOT / "results/experiment_8156_v705_decision_audit.json").read_bytes())
    result = e.primitive(value, 8156)
    assert result["result"]["eligible_count"] == 98
    assert result["result"]["h1_development_signal_score"] == 0
    changed = deepcopy(value)
    changed["paired_intervals"]["mean_gain"] = 42
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        e.primitive(changed, 8156)
    assert e.primitive({}, 8156) == {"available": False}


def test_private_worker_script_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8163: run actual script outside checkout without PYTHONPATH."""
    root, _ = fixture(tmp_path)
    data = e.load(root, tmp_path / "raw")
    source = tmp_path / "input.json"
    output = tmp_path / "worker.json"
    atomic_json(source, data)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [
        str(c.ROOT / ".venv/bin/python"),
        str(c.ROOT / c.CLI),
        "--worker-input",
        str(source),
        "--output",
        str(output),
    ]
    assert subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True).returncode == 0
    monkeypatch.setattr(sys, "argv", argv[1:])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert exit_info.value.code == 0
    value = json.loads(output.read_bytes())
    value.update(
        required_checks_passed=True,
        validation_receipts=[],
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        replay_input_reference=e.reference(source),
    )
    c.qualify(value, True)
    atomic_json(output, value)
    assert c.replay(output)["passed"]
    value["H1"]["status"] = "invented"
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        c.replay(output)


def test_service_hardware_and_archived_reductions(tmp_path):
    """REQ-VERIFY-8163: qualified components and archived nulls keep their scopes."""
    data = e.load(e.ROOT, tmp_path / "raw")
    result = e.reduce(data)
    assert result["H1"]["status"] == "completed_null"
    assert result["H1"]["completed_count"] == 98
    assert result["H2"]["status"] == "blocked"
    assert result["archived_learning"]["result"]["h2_passed"] is False
    assert len(result["board_obligations"]) == 3
    assert result["service_evidence_scope"]["host_batch_qualified"]
    assert result["independent_reductions"]["exp8159-durable-batch-service"]["available"]
    assert not result["service_evidence_scope"]["whole_service_qualified"]
    assert all(result["rows"][i]["disposition"] == "conductor_skip" for i in (7, 8))
    audit = data["audits"]["exp8156-decision-audit"]
    assert e.primitive(data["primaries"]["exp8156-decision-audit"], 8156) == audit


@pytest.mark.parametrize("passed", [True, False])
def test_main_publication_and_owned_failure(tmp_path, monkeypatch, passed):
    """REQ-REPORT-8163: failed owned checks zero readiness on private outputs."""
    root, _ = fixture(tmp_path)
    gates = dict(
        gates={k: dict(passed=k != "G2") for k in ("G1", "G2", "G3", "G4")},
        paper_ready=False,
        unmet_gates=["G2"],
    )

    def execute(plan, raw):
        receipts = []
        for spec in plan:
            log = raw / (spec.name + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps(gates) if spec.name == "publication_gate" else "checked\n")
            if spec.name == "independent_reduction":
                source = Path(spec.argv[spec.argv.index("--worker-input") + 1])
                target = Path(spec.argv[spec.argv.index("--output") + 1])
                atomic_json(target, e.reduce(json.loads(source.read_bytes())))
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
    output = tmp_path / "experiment_8163_v705_capstone.json"
    assert c.main(["--root", str(root), "--output", str(output), "--fixture-e2e"]) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] is passed
    assert value["capstone_execution_ready_score"] == int(passed)
    assert value["MODEL_SPECS"] == value["call_ledger"] == []
    assert c.main(["--cold-replay", str(output)]) == 0
    assert c.main(["--root", str(root), "--output", str(output), "--fixture-e2e"]) == 0
    value["validation_receipts"][0]["log_sha256"] = "changed"
    atomic_json(output, value)
    assert c.main(["--cold-replay", str(output)]) == 1


def test_private_guard_hash_tamper_and_empty_source(tmp_path):
    """SCENARIO-REPORT-8163: private routes cannot overwrite natural primaries."""
    assert c.main(["--fixture-e2e"]) == 1
    assert c.main(["--fixture-e2e", "--output", str(tmp_path / "out.json")]) == 1
    assert (
        c.main(
            ["--worker-input", str(tmp_path / "missing"), "--output", str(tmp_path / "out.json")]
        )
        == 1
    )
    root, tasks = fixture(tmp_path)
    data = e.load(root, tmp_path / "raw")
    value = e.reduce(data)
    source = tmp_path / "input.json"
    output = tmp_path / "out.json"
    atomic_json(source, data)
    value.update(
        required_checks_passed=False,
        validation_receipts=[],
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        replay_input_reference=e.reference(source),
    )
    c.qualify(value, False)
    atomic_json(output, value)
    assert c.replay(output)["passed"]
    source.write_text("{}")
    with pytest.raises(ValueError, match="input_hash_drift"):
        c.replay(output)
    assert e.primitive({}, 8162) == {"available": False}
    assert len(tasks) == 14


def test_primitive_and_authority_error_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8163: source and authority errors keep exact failed operands."""
    root, tasks = fixture(tmp_path)
    real = e.primitive
    monkeypatch.setattr(
        e, "primitive", lambda *_: (_ for _ in ()).throw(ValueError("bad primitive"))
    )
    data = e.load(root, tmp_path / "bad")
    assert not any(r["qualified"] for r in data["dispositions"])
    monkeypatch.setattr(e, "primitive", real)
    path = root / tasks[2]["deliverable"]
    value = json.loads(path.read_bytes())
    side = Path(value["terminal_validation_sidecar_path"])
    pub = json.loads(side.read_bytes())["publication"]
    validator = Path(pub["sidecar_path"])
    report = json.loads(validator.read_bytes())
    report["report"]["passed"] = False
    atomic_json(validator, report)
    data = e.load(root, tmp_path / "failed-sidecar")
    assert not data["dispositions"][2]["qualified"]
    data["audits"][tasks[6]["id"]] = dict(
        available=True,
        result=dict(
            h1_development_signal_score=1,
            support_passed=True,
            safety_passed=True,
            eligible_count=96,
            improved_sources=96,
            paired_intervals=dict(mean_gain=0.2),
            per_source_results=[dict(status="completed", h1_gain=0.2) for _ in range(96)],
        ),
    )
    assert e.reduce(data)["h1_development_signal_score"] == 1
    data["dispositions"][6]["eligible"] = False
    assert e.reduce(data)["H1"]["status"] == "blocked"


def test_replay_primitive_drift_and_cli_error(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8163: cached audit mutations cannot survive cold replay."""
    root, tasks = fixture(tmp_path)
    data = e.load(root, tmp_path / "raw")
    data["audits"][tasks[0]["id"]] = dict(available=True, result={})
    source, output = tmp_path / "input.json", tmp_path / "out.json"
    atomic_json(source, data)
    value = e.reduce(data)
    value.update(
        required_checks_passed=True,
        validation_receipts=[],
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        replay_input_reference=e.reference(source),
    )
    atomic_json(output, value)
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        c.replay(output)
    monkeypatch.setattr(c, "execute", lambda *_: [dict(passed=False, normal_exit=True)])
    assert not c.terminal(output)["passed"]


def test_service_hardware_mutations_and_worker_failure(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8163: independent equations reject changed component claims."""
    from carnot.verify import durable_batch_8159
    from carnot.reporting import hardware_workload_8162

    source = tmp_path / "primitive.json"
    atomic_json(source, {})
    ref = e.reference(source)
    monkeypatch.setattr(durable_batch_8159, "reduce_rows", lambda _: dict(intervals=[]))
    with pytest.raises(ValueError, match="service_primitive_reduction_drift"):
        e.primitive(dict(primitive_rows=ref, paired_speed_intervals=[1]), 8159)
    monkeypatch.setattr(
        hardware_workload_8162,
        "reduce",
        lambda _: dict(board_rows=[], amdahl_bounds=[], quantization_rows=[]),
    )
    with pytest.raises(ValueError, match="hardware_primitive_reduction_drift"):
        e.primitive(
            dict(
                replay_input_reference=ref, board_rows=[1], amdahl_bounds=[], quantization_rows=[]
            ),
            8162,
        )
    root, _ = fixture(tmp_path)
    monkeypatch.setattr(c, "execute", lambda plan, _: [dict(passed=False)] if plan else [])
    assert (
        c.main(["--root", str(root), "--output", str(tmp_path / "fail.json"), "--fixture-e2e"]) == 1
    )


def test_frozen_validation_creates_private_parent(tmp_path):
    """REQ-REPORT-8163: frozen pytest argv must have a usable private parent."""
    private = tmp_path / "scratch"
    plan = c.commands(private)
    focused = next(s for s in plan if s.name == "focused_pytest")
    basetemp = Path(next(a.split("=", 1)[1] for a in focused.argv if a.startswith("--basetemp=")))
    assert basetemp.parent.is_dir()
