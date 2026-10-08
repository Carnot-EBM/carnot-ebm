"""REQ-REPORT-8303 / REQ-VERIFY-8303: scoped evidence and real private CLI checks."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v716_capstone as runner
from carnot.reporting import v716_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary


def fixture(root):
    root.mkdir()
    for name in [e.DESIGN, e.ACTIVE, e.STAGED, e.PROTOCOL, *e.HISTORY]:
        source = e.ROOT / name
        if source.is_file():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    (root / "results").mkdir(exist_ok=True)
    return root


def cli(parent, *args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_private_missing(tmp_path):
    """SCENARIO-REPORT-8303-DISPOSITIONS: absence retains every intended slot."""
    work = e.measure(fixture(tmp_path / "root"), tmp_path / "raw")
    value = e.reduce(work, [dict(passed=True, normal_exit=True, scope="owned")])
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8290, 8304))
    assert value["completed_count"] == value["intended_count"] == 14
    assert value["missing_output_count"] == 13
    assert all(r["honest_verdict"] is None for r in value["rows"][:-1])
    assert value["H1"]["statistics"] is value["H2"]["statistics"] is None
    assert [value[k]["intended_count"] for k in ["H1", "H2"]] == [128, 96]
    assert value["H2"]["retention_intended_count"] == 32
    assert value["H1"]["alpha"] == value["H2"]["alpha"] == 0.025
    assert value["science_ready_score"] == value["independent_generalization_score"] == 0
    assert value["verdict_class"] == "blocked"
    assert value["capstone_execution_ready_score"] == 1
    assert not value["polarfire_graduation"]["graduated"]
    assert e.reduce(work, [])["verdict_class"] == "disqualified"
    changed = deepcopy(work)
    changed["tasks"][0]["title"] = "changed"
    with pytest.raises(ValueError, match="contract"):
        e.reduce(changed, [])


def test_real_sources(tmp_path):
    """REQ-REPORT-8303: blocked science does not hide independent CPU evidence."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    value = e.reduce(work, [dict(passed=True, normal_exit=True, scope="owned")])
    assert value["actual_executed_task_count"] == 6
    assert value["pre_gate_count"] == 1 and value["missing_output_count"] == 7
    assert value["rows"][1]["verdict_class"] == "circular_positive"
    assert value["rows"][10]["verdict_class"] == "disqualified"
    assert value["rows"][2]["honest_verdict"] is None
    assert value["polarfire_graduation"]["graduated"]
    assert value["historical_failure_dispositions"][0]["actual_exit"] == 1
    assert value["archive_lag"]["planning_archive_stopped_at"] == "V714"
    plan = runner.audit_plan(work, tmp_path)
    assert plan and all(p["scope"] == "upstream" for p in plan)
    assert any(p["name"] == "branch_8291_replay" for p in plan)


def test_authentication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8303-DISPOSITIONS: failure and missing bytes differ."""
    path = tmp_path / "results/experiment_8290_private.json"
    terminal = path.parent / "raw" / path.stem / "terminal_validation.json"
    value = dict(
        experiment_id=8290,
        task_id="exp8290-runtime-localization",
        honest_verdict="complete_disqualified_owned_checks",
        verdict_class="disqualified",
        required_checks_passed=False,
        flagged_adversarial=False,
        rows=[],
        intended_count=0,
        completed_count=0,
        failed_count=0,
        excluded_count=0,
        censored_count=0,
        terminal_validation_sidecar_path=str(terminal),
    )
    report = publish_primary(path, value, lambda p: dict(passed=True))
    atomic_json(terminal, dict(publication=report))
    task = dict(id=value["task_id"])
    item = e.bind(path, tmp_path / "raw", [])
    row, failures, source = e.outcome(task, 8290, item)
    assert row["disposition"] == "authenticated_upstream_failure"
    assert row["producer_executed"] and source == value and failures
    value.update(
        honest_verdict="complete_null_private", verdict_class="null", required_checks_passed=True
    )
    report = publish_primary(path, value, lambda p: dict(passed=True))
    atomic_json(terminal, dict(publication=report))
    assert e.outcome(task, 8290, e.bind(path, tmp_path / "valid", []))[0]["eligible"]
    path.write_text("corrupt")
    row = e.outcome(task, 8290, e.bind(path, tmp_path / "corrupt", []))[0]
    assert row["disposition"] == "corrupted_artifact" and row["honest_verdict"] is None
    path.unlink()
    assert (
        e.outcome(task, 8290, e.bind(path, tmp_path / "missing", []))[0]["disposition"]
        == "absent_producer"
    )
    monkeypatch.setattr(
        e.prior, "outcome", lambda *a: (_ for _ in ()).throw(RuntimeError("reader"))
    )
    row = e.outcome(task, 8290, item)[0]
    assert row["disposition"] == "owned_reader_exception" and row["failed"]


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8303-CLI: real child statements and terminal checks run."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    assert runner.replay(output)["passed"]
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    for key, observed in [
        ("completed_count", 13),
        ("experiment_id", 9),
        ("MODEL_SPECS", [{}]),
        ("paper_ready", not value["paper_ready"]),
        ("reproducibility_checksum", "rehashed"),
    ]:
        changed = dict(value, **{key: observed})
        path = tmp_path / (key + ".json")
        atomic_json(path, changed)
        assert cli(tmp_path, "--cold-replay", path).returncode == 1
    assert cli(tmp_path, "--date", "20260101").returncode == 2
    assert cli(tmp_path, "--private-fixture").returncode == 1
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1


def test_commands(tmp_path):
    """REQ-VERIFY-8303: owned coverage and private E2E argv are frozen."""
    plan = runner.commands(tmp_path)
    assert any(p["name"] == "private_E2E021" for p in plan)
    assert any(p["name"] == "full_python_suite" for p in plan)
    assert "patch=subprocess" in (tmp_path / "coverage.ini").read_text()
    assert runner.terminal_plan(tmp_path / "candidate")[0]["argv"][-1].endswith("candidate")
    assert canonical_hash(e.MODEL_SPECS) == canonical_hash([])


def test_primitive_streams_and_main(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8303-CLI: changing primitives or receipts cannot rehash truth."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(runner, "commands", lambda p: [])
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    assert (root / "docs/research-notes/v716-outcomes.md").is_file()
    original = json.loads(output.read_bytes())
    changed = deepcopy(original)
    changed["validation_receipts"][0]["stdout_sha256"] = "sha256:changed"
    atomic_json(tmp_path / "stream.json", changed)
    with pytest.raises(ValueError, match="validation_stream_drift"):
        runner.replay(tmp_path / "stream.json")
    work = json.loads(Path(original["work_reference"]["path"]).read_bytes())
    work["references"] = work["references"][:-1]
    private_work = tmp_path / "changed_work.json"
    atomic_json(private_work, work)
    changed = dict(
        original, work_reference=dict(path=str(private_work), sha256=sha256_file(private_work))
    )
    atomic_json(tmp_path / "primitive.json", changed)
    with pytest.raises(ValueError, match="source_reference_drift"):
        runner.replay(tmp_path / "primitive.json")
    monkeypatch.setattr(
        runner, "publish_primary", lambda p, v, validator: validator(tmp_path / "wrong_operand")
    )
    assert runner.main(["--root", str(root), "--output", str(output), "--private-fixture"]) == 1


def test_current_authority_and_gates(tmp_path, monkeypatch):
    """REQ-REPORT-8303: full authority and bound gate operands fail independently."""
    import yaml

    root = fixture(tmp_path / "root")
    active = yaml.safe_load((root / e.ACTIVE).read_bytes())
    active["tasks"][0]["title"] = "changed"
    (root / e.ACTIVE).write_text(yaml.safe_dump(active))
    work = e.measure(root, tmp_path / "raw")
    assert (
        e.reduce(work, [dict(passed=True, normal_exit=True, scope="owned")])[
            "capstone_execution_ready_score"
        ]
        == 0
    )
    design = root / e.DESIGN
    design.write_text(design.read_text().replace("exp8290-runtime-localization", "exp9999-wrong"))
    with pytest.raises(ValueError, match="fourteen"):
        e.measure(root, tmp_path / "wrong")
    work = e.measure(e.ROOT, tmp_path / "real")
    changed = deepcopy(work)
    changed["inputs"][0]["reference"] = dict(changed["inputs"][0]["reference"], path="foreign")
    with pytest.raises(ValueError, match="input_reference_drift"):
        e.reduce(changed, [])
    changed = deepcopy(work)
    item = changed["inputs"][2]["reference"]
    gate = json.loads(Path(item["snapshot_path"]).read_bytes())
    gate["gates_evaluated"][0]["artifact_sha256"] = "sha256:changed"
    path = tmp_path / "gate.json"
    atomic_json(path, gate)
    item.update(snapshot_path=str(path), sha256=sha256_file(path))
    assert (
        e.reduce(changed, [dict(passed=True, scope="owned")])["rows"][2]["disposition"]
        == "unbound_pre_gate"
    )
    changed = deepcopy(work)
    next(r for r in changed["references"] if "expected_sha256" in r)["expected_sha256"] = (
        "sha256:changed"
    )
    assert not e.reduce(changed, [dict(passed=True, scope="owned")])["polarfire_graduation"][
        "graduated"
    ]
    failed = dict(
        passed=False,
        scope="owned",
        name="failed",
        stdout_path="private",
        stdout_sha256="sha256:private",
        expected_exit=0,
        actual_exit=1,
    )
    assert e.reduce(work, [failed])["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        e.prior, "outcome", lambda *a: (_ for _ in ()).throw(RuntimeError("reader"))
    )
    assert e.reduce(work, [dict(passed=True, scope="owned")])["verdict_class"] == "disqualified"


def test_h3_independent_reconstruction(tmp_path, monkeypatch):
    """REQ-VERIFY-8303: H3 event/crash mechanics remain separate from natural benefit."""
    from carnot.reporting import v709_execution as supervisor

    work = e.measure(e.ROOT, tmp_path / "raw")
    replay_spec = next(
        p for p in runner.audit_plan(work, tmp_path) if p["name"] == "branch_8291_replay"
    )
    replay_receipt = supervisor.execute([replay_spec], tmp_path / "logs")[0]
    assert replay_receipt["passed"] and replay_receipt["actual_exit"] == 0
    receipts = [dict(passed=True, scope="owned"), replay_receipt]
    value = e.reduce(work, receipts)
    assert value["h3_fixture_soundness_score"] == 1
    assert value["h3_fixture_efficiency_signal_score"] == 1
    assert value["evidence_improved"] and not value["learning_improved"]
    assert value["H3"]["independent_reconstruction"]["full_scan_disagreements"] == 0
    assert not value["H3"]["natural_benefit"] and value["science_ready_score"] == 0
    h3ref = work["inputs"][1]["reference"]
    h3 = json.loads(Path(h3ref["snapshot_path"]).read_bytes())
    h3["efficiency_signal"] = False
    private = tmp_path / "h3.json"
    atomic_json(private, h3)
    h3ref.update(snapshot_path=str(private), sha256=sha256_file(private))
    item = work["inputs"][1]
    side = json.loads(Path(item["sidecar"]["snapshot_path"]).read_bytes())
    side["primary_sha256"] = h3ref["sha256"]
    private_side = tmp_path / "side.json"
    atomic_json(private_side, side)
    item["sidecar"].update(snapshot_path=str(private_side), sha256=sha256_file(private_side))
    reconstructed = value["H3"]["independent_reconstruction"]

    def slower(work):
        return dict(reconstructed, efficiency_signal=False)

    monkeypatch.setattr(e.dependency, "reduce_work", slower)
    assert e.reduce(work, receipts)["H3"]["status"] == "informative_fixture_null"
    violated = dict(slower({}), hard_constraint_violations=dict(count=1, checked_transactions=4080))
    h3.update(violated)
    atomic_json(private, h3)
    h3ref["sha256"] = sha256_file(private)
    side["primary_sha256"] = h3ref["sha256"]
    atomic_json(private_side, side)
    item["sidecar"]["sha256"] = sha256_file(private_side)
    monkeypatch.setattr(e.dependency, "reduce_work", lambda w: violated)
    assert e.reduce(work, receipts)["H3"]["status"] == "disqualified_fixture_hard_constraint"
    monkeypatch.setattr(
        e.dependency, "reduce_work", lambda w: dict(violated, full_scan_disagreements=1)
    )
    with pytest.raises(ValueError, match="h3_primitive_reduction_drift"):
        e.reduce(work, receipts)


def test_canonical_digest_rejection(tmp_path):
    """REQ-REPORT-8303: printed digest cannot authorize changed complete tasks."""
    root = fixture(tmp_path / "root")
    path = root / e.DESIGN
    path.write_text(
        path.read_text().replace(
            "1631010541a7298fa4908b9fcbe21576b4b333b81248b52c8643b0aaf44b71ed", "0" * 64
        )
    )
    work = e.measure(root, tmp_path / "raw")
    with pytest.raises(ValueError, match="canonical_digest_drift"):
        e.reduce(work, [])


def test_historical_log_hash_and_pre_gate_audit(tmp_path):
    """REQ-VERIFY-8303: historical log corruption is an operand failure, never a null."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    log = next(
        r
        for r in work["references"]
        if r.get("evidence_role") == "authenticated_historical_failure_log"
    )
    log["expected_sha256"] = "sha256:changed"
    value = e.reduce(work, [dict(passed=True, scope="owned")])
    assert any(
        g["artifact_field"] == "historical_evidence_sha256" for g in value["gate_check_summary"]
    )
    assert value["polarfire_graduation"]["graduated"]
    work["tasks"][2]["deliverable"] = "results/" + e.NAME + ".json"
    assert not any(p["name"].startswith("branch_8292") for p in runner.audit_plan(work, tmp_path))


def test_authenticated_failed_terminal_report(tmp_path):
    """SCENARIO-REPORT-8303-DISPOSITIONS: a failed bound report preserves failure bytes."""
    path = tmp_path / "results/experiment_8290_private.json"
    terminal = path.parent / "raw" / path.stem / "terminal_validation.json"
    value = dict(
        experiment_id=8290,
        task_id="exp8290-runtime-localization",
        honest_verdict="complete_disqualified_owned_checks",
        verdict_class="disqualified",
        required_checks_passed=False,
        flagged_adversarial=False,
        rows=[],
        intended_count=0,
        completed_count=0,
        failed_count=0,
        excluded_count=0,
        censored_count=0,
        terminal_validation_sidecar_path=str(terminal),
    )
    published = publish_primary(path, value, lambda p: dict(passed=True))
    atomic_json(terminal, dict(publication=published))
    side = Path(published["sidecar_path"])
    report = json.loads(side.read_bytes())
    report["report"]["passed"] = False
    atomic_json(side, report)
    row, failures, source = e.outcome(
        dict(id=value["task_id"]), 8290, e.bind(path, tmp_path / "raw", [])
    )
    assert row["disposition"] == "authenticated_upstream_failure"
    assert row["producer_executed"] and row["failed"] and not row["eligible"]
    assert not row["missing"] and source == value and failures
