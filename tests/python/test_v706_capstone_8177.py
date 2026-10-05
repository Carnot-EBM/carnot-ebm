"""REQ-REPORT-8177, REQ-VERIFY-8177: private accounting and replay controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from carnot.reporting import v706_capstone as c
from carnot.reporting import v706_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary


def fixture(tmp_path):
    """Keep real immutable authority while synthetic outcomes stay private."""
    root = tmp_path / "private"
    receipt = json.loads((e.ROOT / e.INPUT).read_bytes())
    receipt = {
        k: receipt[k]
        for k in (
            "task_contract",
            "authority_snapshots",
            "canonical_tasks_sha256",
            "literature_mapping",
        )
    }
    tasks = receipt["task_contract"]
    for role in ("design", "active"):
        ref = receipt["authority_snapshots"][role]
        target = root / (role + ".bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(ref["snapshot_path"]).read_bytes())
        ref.update(snapshot_path=str(target), sha256=sha256_file(target))
    for n, task in enumerate(tasks[:-1], 8164):
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
        if n == 8164:
            value.update(receipt)
        pub = publish_primary(path, value, lambda _: dict(passed=True))
        atomic_json(side, dict(publication=pub))
    return root, tasks


def replay_value(tmp_path, data, passed=True):
    """Bind fixture rows and logs exactly as a cold consumer would read them."""
    source, output = tmp_path / "input.json", tmp_path / "out.json"
    atomic_json(source, data)
    value = e.reduce(data)
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
    """SCENARIO-REPORT-8177: missing science has no invented producer verdict."""
    root, tasks = fixture(tmp_path)
    (root / tasks[5]["deliverable"]).unlink()
    p = root / tasks[1]["deliverable"]
    v = json.loads(p.read_bytes())
    v.update(
        required_checks_passed=False,
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_owned_validation",
    )
    atomic_json(p, v)
    data = e.load(root, tmp_path / "raw")
    value = e.reduce(data)
    assert len(value["rows"]) == value["completed_count"] == 14
    assert value["rows"][5]["primary_honest_verdict"] == "<missing>"
    assert value["rows"][1]["verdict_class"] == "disqualified"
    assert value["H1"]["status"] == "blocked"
    assert value["science_ready_score"] == value["independent_generalization_score"] == 0
    assert len(value["gap_decisions"]) == 3
    assert all(not r["closed"] for r in value["gap_decisions"].values())
    assert not any(r["retire_method_family"] for r in value["retirement_decisions"])
    receipt = root / e.INPUT
    v = json.loads(receipt.read_bytes())
    v["canonical_tasks_sha256"] = "changed"
    atomic_json(receipt, v)
    assert not e.load(root, tmp_path / "mutated")["authority"]["activated"]
    receipt.unlink()
    assert not e.load(root, tmp_path / "absent")["authority"]["activated"]


def test_real_primitive_masks_and_scopes(tmp_path):
    """REQ-VERIFY-8177: real cached rows retain useful exposure and null benefit."""
    data = e.load(e.ROOT, tmp_path / "raw")
    value = e.reduce(data)
    assert value["H1"]["status"] == "blocked"
    assert value["H2"]["status"] == "completed_null"
    assert value["H2"]["later_benefit"] is False
    assert value["H2"]["retention"] is True
    assert value["H2"]["useful_future_exposure"]["changed_later_decisions"] == 0
    assert value["service_evidence_scope"]["current_independent_requests"]["nfr01_met"] is False
    assert value["literature_mapping"]["capstone_always_runs"]
    assert len(value["board_obligations"]) == 3
    assert all(value["rows"][i]["verdict_class"] == "blocked" for i in (4, 5, 6))
    for n in (8167, 8172, 8173, 8174, 8176):
        task = data["tasks"][n - 8164]
        original = data["primaries"][task["id"]]
        assert e.primitive(original, n) == data["audits"][task["id"]]
    changed = deepcopy(data["primaries"][data["tasks"][8]["id"]])
    changed["paired_gain_interval"]["mean_gain"] = 99
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        e.primitive(changed, 8172)
    assert e.primitive({}, 8172) == {"available": False}


def test_worker_and_replay_outside_checkout(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8177: direct CLI and cold replay need no PYTHONPATH."""
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
    assert subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True).returncode == 0
    monkeypatch.setattr(sys, "argv", argv[1:])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert info.value.code == 0
    assert c.replay(output)["passed"]
    value["H1"]["status"] = "invented"
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        c.replay(output)
    source.write_text("{}")
    with pytest.raises(ValueError, match="input_hash_drift"):
        c.replay(output)


@pytest.mark.parametrize("passed", [True, False])
def test_main_and_validation_failure(tmp_path, monkeypatch, passed):
    """REQ-REPORT-8177: owned failure zeros readiness; science blocks stay terminal."""
    root, _ = fixture(tmp_path)
    gates = dict(
        gates={k: dict(pass_=k != "G2") for k in ("G1", "G2", "G3", "G4")},
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
    output = tmp_path / "experiment_8177_v706_capstone.json"
    args = ["--root", str(root), "--output", str(output), "--fixture-e2e"]
    assert c.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] is passed
    assert value["capstone_execution_ready_score"] == int(passed)
    assert value["MODEL_SPECS"] == value["call_ledger"] == []
    assert c.main(["--cold-replay", str(output)]) == 0
    assert c.main(args) == 0
    value["validation_receipts"][0]["log_sha256"] = "changed"
    atomic_json(output, value)
    assert c.main(["--cold-replay", str(output)]) == 1


def test_error_controls_and_frozen_commands(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8177: missing workers, corrupt receipts and private guards fail."""
    assert c.main(["--fixture-e2e"]) == 1
    assert c.main(["--fixture-e2e", "--output", str(tmp_path / "out.json")]) == 1
    assert (
        c.main(
            ["--worker-input", str(tmp_path / "missing"), "--output", str(tmp_path / "out.json")]
        )
        == 1
    )
    root, tasks = fixture(tmp_path)
    real = e.primitive
    monkeypatch.setattr(e, "primitive", lambda *_: (_ for _ in ()).throw(ValueError("bad")))
    assert not any(r["qualified"] for r in e.load(root, tmp_path / "bad")["dispositions"])
    monkeypatch.setattr(e, "primitive", real)
    p = root / tasks[2]["deliverable"]
    v = json.loads(p.read_bytes())
    side = Path(v["terminal_validation_sidecar_path"])
    pub = json.loads(side.read_bytes())["publication"]
    validator = Path(pub["sidecar_path"])
    report = json.loads(validator.read_bytes())
    report["report"]["passed"] = False
    atomic_json(validator, report)
    assert not e.load(root, tmp_path / "failed")["dispositions"][2]["qualified"]
    data = e.load(root, tmp_path / "raw")
    data["audits"][tasks[0]["id"]] = dict(available=True, result={})
    _, output, _ = replay_value(tmp_path, data, False)
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        c.replay(output)
    monkeypatch.setattr(c, "execute", lambda *_: [dict(passed=False, normal_exit=True)])
    assert not c.terminal(output)["passed"]
    assert (
        c.main(["--root", str(root), "--output", str(tmp_path / "failed.json"), "--fixture-e2e"])
        == 1
    )
    plan = c.commands(tmp_path / "scratch")
    focused = next(s for s in plan if s.name == "focused_pytest")
    basetemp = Path(next(a.split("=", 1)[1] for a in focused.argv if a.startswith("--basetemp=")))
    assert basetemp.parent.is_dir()
    assert all(
        "::" not in a
        for s in plan
        if s.name in ("ruff_check", "ruff_format", "scoped_spec_coverage")
        for a in s.argv
    )


def test_primitive_and_alternate_primary_tamper(tmp_path):
    """SCENARIO-VERIFY-8177: forged exposure and alternative science fail qualification."""
    value = json.loads(
        (e.ROOT / "results/experiment_8172_v706_learning_benefit_audit.json").read_bytes()
    )
    value["installation_exposure_summary"]["changed_later_decisions"] = 1
    with pytest.raises(ValueError, match="exposure_primitive_reduction_drift"):
        e.primitive(value, 8172)
    value = json.loads(
        (e.ROOT / "results/experiment_8174_v706_complete_request_cost.json").read_bytes()
    )
    value["paired_speed_intervals"][0]["estimate"] = 999
    with pytest.raises(ValueError, match="primitive_reduction_drift"):
        e.primitive(value, 8174)
    root, tasks = fixture(tmp_path)
    path = root / tasks[2]["deliverable"]
    path.rename(path.with_name("experiment_8166_alternate.json"))
    row = e.load(root, tmp_path / "raw")["dispositions"][2]
    assert not row["qualified"]
    assert any(g["check"] == "primary_exists" for g in row["gate_check_summary"])
