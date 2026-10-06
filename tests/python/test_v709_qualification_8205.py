"""REQ-REPORT-8205 and REQ-VERIFY-8205: private authority and real child controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v709_qualification as q
from carnot.reporting import v709_execution as x
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt


def authority_fixture(tmp_path):
    """Copy task objects privately so mutations never touch active authority."""
    root = tmp_path / "inputs"
    root.mkdir()
    design = root / "design.md"
    design.write_bytes((q.ROOT / q.DESIGN).read_bytes())
    _, tasks = q.parse_design(design.read_text(), milestone=q.MILESTONE)
    active = root / "active.yaml"
    active.write_text(yaml.safe_dump(dict(milestone=q.MILESTONE, tasks=tasks)))
    return design, root / "staged.yaml", active


def terminal_fixture(tmp_path, passed=True):
    """Run an actual child as validator before binding its private primary."""
    primary = tmp_path / "results/experiment_9000_control.json"
    side = tmp_path / "terminal.json"
    value = dict(
        experiment_id=9000,
        task_id="exp9000-control",
        honest_verdict="complete_control",
        verdict_class="null",
        flagged_adversarial=False,
        contract_ready_score=1,
        terminal_validation_sidecar_path=str(side),
    )

    def validate(_):
        row = x.child(
            "terminal_control", [sys.executable, "-c", "raise SystemExit(0)"], tmp_path / "logs"
        )
        return dict(passed=row["passed"] and passed, checks=[row])

    pub = publish_primary(primary, value, validate)
    atomic_json(side, dict(publication=pub))
    return primary, value, side


def test_authority_and_mutations(tmp_path):
    """SCENARIO-REPORT-8205-AUTHORITY: compare every independent authority."""
    design, stage, active = authority_fixture(tmp_path)
    found = q.assess(design, stage, active, tmp_path / "raw")
    assert found["activated"] and not found["planning_matched"]
    assert len(found["contract_rows"]) == 13
    stage.write_bytes(active.read_bytes())
    assert q.assess(design, stage, active, tmp_path / "staged_raw")["planning_matched"]
    controls = q.mutations(design, active, tmp_path / "controls")
    assert len(controls) == 6 and all(r["rejected"] for r in controls)
    design.write_text("missing exact contract")
    assert not q.assess(design, stage, active, tmp_path / "broken")["activated"]


@pytest.mark.parametrize(
    "operand",
    [
        None,
        "bad",
        ["bad"],
        [{"path": "p"}],
        {"p": "bad"},
        [{"path": "", "sha256": "sha256:" + "0" * 64}],
    ],
)
def test_hash_adapter_rejects(operand):
    """SCENARIO-REPORT-8205-CONSUMER: malformed entries cannot masquerade as refs."""
    with pytest.raises(ValueError):
        q.hash_references(operand)


def test_real_saved_shapes():
    """SCENARIO-REPORT-8205-CONSUMER: reproduce the actual saved TypeError."""
    for number in (8192, 8204):
        value = json.loads(
            next((q.ROOT / "results").glob(f"experiment_{number}_*.json")).read_bytes()
        )
        refs = q.hash_references(value["code_config_hashes"])
        assert refs and all("sha256" in r for r in refs)
        observed = q.legacy_operand(value)
        assert observed["typeerror_reproduced"] == (number == 8192)
    assert q.hash_references([]) == []


@pytest.mark.parametrize(
    "operand", [None, "bad", ["bad"], {"x": "bad"}, [{"name": "x", "passed": 1}]]
)
def test_receipt_adapter_rejects(operand):
    """SCENARIO-REPORT-8205-CONSUMER: receipt versions require typed outcomes."""
    with pytest.raises(ValueError):
        q.receipts(operand)


def test_receipt_versions():
    """SCENARIO-REPORT-8205-CONSUMER: mapping names and list names remain explicit."""
    assert q.receipts({"x": {"passed": False}}) == [{"name": "x", "passed": False}]
    assert q.receipts([{"name": "x", "passed": True}])[0]["passed"]


def test_terminal_controls(tmp_path):
    """SCENARIO-REPORT-8205-CONSUMER: real bound readers reject altered evidence."""
    primary, value, side = terminal_fixture(tmp_path)
    assert q.terminal(primary, value, tmp_path / "custody")["passed"]
    assert reader_receipt(value["task_id"], primary.parent, field="contract_ready_score")["passed"]
    controls = q.terminal_controls(primary, value, tmp_path / "controls")
    assert all(r["rejected"] for r in controls) and len(controls) == 3
    side.write_text("[]")
    assert not q.terminal(primary, value, tmp_path / "malformed")["passed"]
    side.write_text("{}")
    assert not q.terminal(primary, value, tmp_path / "invalid")["passed"]


def test_child_lifetime_and_deadline(tmp_path):
    """SCENARIO-VERIFY-8205-CHILD: actual pytest failure and bounded group cleanup."""
    rows = x.pytest_controls(tmp_path / "owned", tmp_path / "logs")
    assert rows[0]["exit_code"] == 0 and rows[1]["exit_code"] == 1
    assert all(r["passed"] for r in rows)
    r = x.child(
        "deadline",
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path / "logs",
        deadline=0.02,
        heartbeat=0.01,
    )
    assert r["timed_out"] and r["exit_code"] < 0 and not r["passed"]
    assert sha256_file(Path(r["stdout_path"])) == r["stdout_sha256"]
    assert (
        x.child("failure", [sys.executable, "-c", "raise SystemExit(2)"], tmp_path / "logs")[
            "exit_code"
        ]
        == 2
    )


def test_measure_and_history(tmp_path):
    """REQ-REPORT-8205: all finished dispositions survive source reductions."""
    work = q.measure(q.ROOT, tmp_path / "raw")
    assert len(work["task_dispositions"]) == 13
    assert sum(r["disposition"] == "conductor_skip" for r in work["task_dispositions"]) == 3
    assert work["historical_required_failures"]
    assert work["legacy_operands"][0]["typeerror_reproduced"]
    assert work["contract"]["activated"]
    checks = [dict(name="real", passed=True, exit_code=0, timed_out=False)]
    value = q.build(work, checks, fixture=True)
    assert value["contract_ready_score"] == 1 and value["verdict_class"] == "circular_positive"
    assert q.build(work, [])["verdict_class"] == "disqualified"
    altered = deepcopy(work)
    altered["contract"]["activated"] = False
    altered["contract"]["gate_check_summary"] = [dict(artifact_field="tasks", observed=None)]
    assert q.build(altered, checks)["verdict_class"] == "blocked"
    assert q.build(work, checks)["verdict_class"] == "circular_positive"
    (tmp_path / "empty").mkdir()
    missing = q.measure(tmp_path / "empty", tmp_path / "missing")
    assert missing["precondition_failures"] and not missing["task_dispositions"]


def test_cli_replay_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8205-CHILD: direct private CLI and independent cold replay."""
    from carnot.reporting import v709_runner as run

    output = tmp_path / "experiment_8205_private.json"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["CARNOT_8205_COVERAGE_CONFIG"] = os.environ.get("CARNOT_8205_COVERAGE_CONFIG", "")
    argv = [sys.executable, "-u", str(q.ROOT / run.CLI)]
    p = subprocess.run(
        [*argv, "--private-fixture", "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] and value["consumer_reader_ready_score"] == 1
    manifest = json.loads(
        (Path(value["work_reference"]["path"]).parent / "validation_manifest.json").read_bytes()
    )
    assert len(manifest["terminal_commands"]) == 3 and len(manifest["child_controls"]) == 2
    assert value["precondition_command_receipt"]["exit_code"] == 0
    assert run.replay(output)["passed"]
    assert (
        subprocess.run(
            [*argv, "--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=120,
        ).returncode
        == 0
    )
    value["contract_ready_score"] = 0
    atomic_json(output, value)
    assert (
        subprocess.run(
            [*argv, "--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=120,
        ).returncode
        == 1
    )
    assert (
        run.main(
            ["--private-fixture", "--output", str(q.ROOT / "results/experiment_8205_bad.json")]
        )
        == 1
    )
    assert run.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    blocked = tmp_path / "blocked/experiment_8205_blocked.json"
    empty = tmp_path / "empty_root"
    empty.mkdir()
    assert run.main(["--private-fixture", "--root", str(empty), "--output", str(blocked)]) == 0
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"


def test_validation_manifest_and_full_main(tmp_path, monkeypatch):
    """REQ-VERIFY-8205: production path runs real children under retained parent."""
    from carnot.reporting import v709_runner as run

    private = tmp_path / "parent"
    private.mkdir()
    plan = run.commands(private)
    for spec in plan:
        for operand in spec["argv"]:
            if operand.startswith("--basetemp="):
                assert Path(operand.split("=", 1)[1]).parent.is_dir()
    assert any(s["argv"][1:] == ["tests/python", "-q"] for s in plan)
    assert any(s["name"] == "coverage_report" for s in plan)
    actual = run.commands

    def small(parent):
        return [
            dict(
                name="tiny_owned",
                argv=[sys.executable, "-c", "print('actual validation')"],
                deadline=30,
                expected=0,
                scope="owned",
            )
        ]

    monkeypatch.setattr(run, "commands", small)
    output = tmp_path / "experiment_8205_production_control.json"
    assert run.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "circular_positive" and run.replay(output)["passed"]
    ref = value["raw_shard_hashes"][0]
    Path(ref["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        run.replay(output)
    monkeypatch.setattr(run, "commands", actual)


def test_runner_terminal_failure(tmp_path):
    """REQ-VERIFY-8205: unchanged validators reject a malformed candidate."""
    from carnot.reporting import v709_runner as run

    bad = tmp_path / "bad.json"
    atomic_json(bad, dict(rows=[]))
    assert not run.terminal_checks(bad, tmp_path / "logs")["passed"]


def test_malformed_bound_report(tmp_path):
    """SCENARIO-REPORT-8205-CONSUMER: a bound report still needs a boolean verdict."""
    primary, value, side = terminal_fixture(tmp_path)
    binding = json.loads(side.read_bytes())["publication"]
    validator = Path(binding["sidecar_path"])
    report = json.loads(validator.read_bytes())
    report["report"]["passed"] = "yes"
    atomic_json(validator, report)
    observed = q.terminal(primary, value, tmp_path / "custody")
    assert observed["error"] == "terminal_report_schema" and not observed["passed"]


def test_failed_historical_terminal(tmp_path):
    """REQ-REPORT-8205: broken optional history is preserved without invention."""
    root = tmp_path / "root"
    for name in q.INPUTS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((q.ROOT / name).read_bytes())
    primary = root / "results/experiment_8192_v708_contract_custody.json"
    primary.unlink()
    atomic_json(
        primary,
        dict(
            task_id="exp8192-contract-custody",
            code_config_hashes=[],
            terminal_validation_sidecar_path=str(tmp_path / "missing"),
        ),
    )
    work = q.measure(root, tmp_path / "raw")
    assert any("terminal_failure" in r for r in work["historical_required_failures"])
    assert not work["consumer_controls"]


def test_rehashed_replay_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8205: independent authority and archive defeat rehashed aggregates."""
    from carnot.reporting import v709_runner as run

    output = tmp_path / "experiment_8205_controls.json"
    assert run.main(["--private-fixture", "--output", str(output)]) == 0
    original = json.loads(output.read_bytes())
    value = deepcopy(original)
    value["validation_receipts"][0]["stderr_sha256"] = "sha256:" + "0" * 64
    atomic_json(output, value)
    with pytest.raises(ValueError, match="log_hash"):
        run.replay(output)
    workpath = Path(original["work_reference"]["path"])
    original_work = workpath.read_bytes()
    for kind in ("authority", "history"):
        work = json.loads(original_work)
        value = deepcopy(original)
        if kind == "authority":
            work["contract"]["canonical_tasks_sha256"] = "0" * 64
            value["canonical_tasks_sha256"] = "0" * 64
        else:
            work["task_dispositions"][0]["original_archive_row"]["result"] = "altered"
            value["task_dispositions"][0]["original_archive_row"]["result"] = "altered"
        atomic_json(workpath, work)
        digest = sha256_file(workpath)
        value["work_reference"]["sha256"] = digest
        for ref in value["raw_shard_hashes"]:
            if ref["path"] == str(workpath):
                ref["sha256"] = digest
        atomic_json(output, value)
        with pytest.raises(
            ValueError, match=("historical" if kind == "history" else kind) + "_reduction"
        ):
            run.replay(output)
    workpath.write_bytes(original_work)


def test_external_precondition_child_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8205: a real prerequisite failure blocks all dependent children."""
    from carnot.reporting import v709_runner as run

    def missing():
        return dict(
            name="environment_preconditions",
            argv=[
                sys.executable,
                "-c",
                "import sys; print('missing prerequisite',file=sys.stderr); raise SystemExit(3)",
            ],
            deadline=30,
            expected=0,
            scope="preconditions",
        )

    monkeypatch.setattr(run, "precondition_command", missing)
    output = tmp_path / "experiment_8205_blocked.json"
    assert run.main(["--private-fixture", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and not value["contract_ready_score"]
    assert value["gate_check_summary"][0]["observed"] == 3
    assert not value["task_dispositions"]


def test_authenticated_failed_history_qualifies_reader(tmp_path):
    """SCENARIO-REPORT-8205-CONSUMER: old failed science retains a valid schema."""
    root = tmp_path / "root"
    for name in q.INPUTS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((q.ROOT / name).read_bytes())
    primary = root / "results/experiment_8192_v708_contract_custody.json"
    value = json.loads(primary.read_bytes())
    side = tmp_path / "historical_terminal.json"
    value["terminal_validation_sidecar_path"] = str(side)

    def validate(_):
        row = x.child("historical_control", [sys.executable, "-c", "pass"], tmp_path / "logs")
        return dict(passed=row["passed"], checks=[row])

    binding = publish_primary(primary, value, validate)
    atomic_json(side, dict(publication=binding))
    validator = Path(binding["sidecar_path"])
    report = json.loads(validator.read_bytes())
    report["report"]["passed"] = False
    atomic_json(validator, report)
    observed = q.terminal(primary, value, tmp_path / "custody")
    assert observed["error"] is None and observed["passed"] is False
    work = q.measure(root, tmp_path / "raw")
    assert any(
        r.get("terminal_failure", {}).get("report", {}).get("passed") is False
        for r in work["historical_required_failures"]
    )
    assert len(work["consumer_controls"]) == 3
    assert all(r["rejected"] and r["observation"]["error"] for r in work["consumer_controls"])
    checks = [x.child("current_control", [sys.executable, "-c", "pass"], tmp_path / "logs")]
    qualified = q.build(work, checks, fixture=True)
    assert qualified["required_checks_passed"] and qualified["consumer_reader_ready_score"] == 1
    assert qualified["historical_evidence_ready_score"] == 0


def test_live_prompt_braces_match_exact_design():
    """SCENARIO-REPORT-8205-AUTHORITY: active prompt escaping remains byte-bound."""
    from carnot.reporting.v685_authority_lifecycle import tasks_digest

    design = (q.ROOT / q.DESIGN).read_text()
    _, planned = q.parse_design(design, milestone=q.MILESTONE)
    active = yaml.safe_load((q.ROOT / "research-roadmap.yaml").read_bytes())["tasks"]
    assert planned == active
    assert f"Canonical full-task SHA256: `{tasks_digest(planned)}`" in design
    prompts = {t["id"]: t["prompt"] for t in planned}
    assert "{{reject,escalate}}" in prompts["exp8207-restricted-action-methods"]
    assert '{{"hf_id":' in prompts["exp8214-prospective-service-measurement"]
